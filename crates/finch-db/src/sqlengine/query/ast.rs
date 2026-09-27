use finch_types::{Status, ZResult};

use super::parser::Parser;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SqlSelect {
    pub table: String,
    pub selected: Vec<SelectItem>,
    pub where_expr: Option<LogicExpr>,
    pub order_by: Vec<OrderByItem>,
    /// LIMIT accepts a signed integer; values <= 0 behave as "unset"
    /// (QueryAnalyzer defaults topN when limit <= 0).
    pub limit: Option<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectItem {
    Asterisk,
    Field { name: String, alias: Option<String> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderByItem {
    pub field: String,
    pub desc: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogicExpr {
    Rel(RelExpr),
    And(Box<LogicExpr>, Box<LogicExpr>),
    Or(Box<LogicExpr>, Box<LogicExpr>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelExpr {
    Compare {
        lhs: ValueExpr,
        op: CmpOp,
        rhs: ValueExpr,
    },
    Like {
        field: ValueExpr,
        pattern: ValueExpr,
    },
    InList {
        field: ValueExpr,
        negated: bool,
        values: Vec<ValueExpr>,
    },
    Contain {
        field: ValueExpr,
        op: ContainOp,
        negated: bool,
        values: Vec<ValueExpr>,
    },
    IsNull {
        field: ValueExpr,
        negated: bool,
    },
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ContainOp {
    All,
    Any,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueExpr {
    Ident(String),
    FunctionCall { name: String, args: Vec<ValueExpr> },
    Int(String),
    Float(String),
    StringLiteral(String),
    BoolLiteral(bool),
    VectorLiteral(String),
    MatrixLiteral(Vec<String>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorCond {
    pub field: String,
    pub literal: VectorLiteral,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VectorLiteral {
    Vector(String),
    Matrix(Vec<String>),
}

impl SqlSelect {
    pub fn parse(sql: &str) -> ZResult<Self> {
        Parser::new(sql).parse_compilation_unit_first_select()
    }
}

const VECTOR_UNDER_OR: &str = "vector condition must NOT be OR ancestor.";
const ONLY_ONE_VECTOR: &str = "only one vector condition is supported";

impl LogicExpr {
    /// Split out an optional vector condition and return the remaining (scalar) logic expr.
    ///
    /// a vector condition must not have an `OR` ancestor.
    pub fn split_vector_condition(self) -> ZResult<(Option<VectorCond>, Option<LogicExpr>)> {
        match self {
            LogicExpr::Or(a, b) => {
                // Any vector cond under OR is invalid.
                if a.contains_vector_cond() || b.contains_vector_cond() {
                    return Err(Status::invalid_argument(VECTOR_UNDER_OR));
                }
                Ok((None, Some(LogicExpr::Or(a, b))))
            }
            LogicExpr::And(a, b) => {
                let (va, ra) = a.split_vector_condition()?;
                let (vb, rb) = b.split_vector_condition()?;
                if va.is_some() && vb.is_some() {
                    return Err(Status::invalid_argument(ONLY_ONE_VECTOR));
                }
                Ok((va.or(vb), LogicExpr::join_remaining(ra, rb, LogicExpr::And)))
            }
            LogicExpr::Rel(rel) => {
                if let Some(vc) = rel.is_vector_condition() {
                    Ok((Some(vc), None))
                } else {
                    Ok((None, Some(LogicExpr::Rel(rel))))
                }
            }
        }
    }

    /// Joins what remains of two sides after a vector condition was split out.
    pub(crate) fn join_remaining(
        left: Option<LogicExpr>,
        right: Option<LogicExpr>,
        join: fn(Box<LogicExpr>, Box<LogicExpr>) -> LogicExpr,
    ) -> Option<LogicExpr> {
        match (left, right) {
            (None, None) => None,
            (Some(x), None) | (None, Some(x)) => Some(x),
            (Some(x), Some(y)) => Some(join(Box::new(x), Box::new(y))),
        }
    }

    fn contains_vector_cond(&self) -> bool {
        match self {
            LogicExpr::Rel(r) => r.is_vector_condition().is_some(),
            LogicExpr::And(a, b) | LogicExpr::Or(a, b) => {
                a.contains_vector_cond() || b.contains_vector_cond()
            }
        }
    }

    pub fn to_filter_string(&self) -> String {
        let mut s = String::new();
        self.write_filter_string(0, &mut s);
        s
    }

    fn precedence(&self) -> u8 {
        match self {
            LogicExpr::Or(_, _) => 1,
            LogicExpr::And(_, _) => 2,
            LogicExpr::Rel(_) => 3,
        }
    }

    fn write_filter_string(&self, parent_prec: u8, out: &mut String) {
        let my_prec = self.precedence();
        let need_paren = my_prec < parent_prec;
        if need_paren {
            out.push('(');
        }
        match self {
            LogicExpr::Rel(r) => out.push_str(r.to_filter_string().as_str()),
            LogicExpr::And(a, b) => {
                a.write_filter_string(my_prec, out);
                out.push_str(" AND ");
                b.write_filter_string(my_prec, out);
            }
            LogicExpr::Or(a, b) => {
                a.write_filter_string(my_prec, out);
                out.push_str(" OR ");
                b.write_filter_string(my_prec, out);
            }
        }
        if need_paren {
            out.push(')');
        }
    }
}

/// Renders values as a `, `-separated filter list.
fn join_filter_strings(values: &[ValueExpr]) -> String {
    values
        .iter()
        .map(ValueExpr::to_filter_string)
        .collect::<Vec<_>>()
        .join(", ")
}

impl RelExpr {
    fn is_vector_condition(&self) -> Option<VectorCond> {
        let RelExpr::Compare {
            lhs: ValueExpr::Ident(field),
            op: CmpOp::Eq,
            rhs,
        } = self
        else {
            return None;
        };
        let literal = match rhs {
            ValueExpr::VectorLiteral(v) => VectorLiteral::Vector(v.clone()),
            ValueExpr::MatrixLiteral(vs) => VectorLiteral::Matrix(vs.clone()),
            _ => return None,
        };
        Some(VectorCond {
            field: field.clone(),
            literal,
        })
    }

    pub fn to_filter_string(&self) -> String {
        match self {
            RelExpr::Compare { lhs, op, rhs } => format!(
                "{} {} {}",
                lhs.to_filter_string(),
                op.to_filter_string(),
                rhs.to_filter_string()
            ),
            RelExpr::Like { field, pattern } => format!(
                "{} LIKE {}",
                field.to_filter_string(),
                pattern.to_filter_string()
            ),
            RelExpr::InList {
                field,
                negated,
                values,
            } => format!(
                "{}{}({})",
                field.to_filter_string(),
                if *negated { " NOT IN " } else { " IN " },
                join_filter_strings(values)
            ),
            RelExpr::Contain {
                field,
                op,
                negated,
                values,
            } => format!(
                "{}{}{} ({})",
                field.to_filter_string(),
                if *negated { " NOT " } else { " " },
                match op {
                    ContainOp::All => "CONTAIN_ALL",
                    ContainOp::Any => "CONTAIN_ANY",
                },
                join_filter_strings(values)
            ),
            RelExpr::IsNull { field, negated } => {
                if *negated {
                    format!("{} IS NOT NULL", field.to_filter_string())
                } else {
                    format!("{} IS NULL", field.to_filter_string())
                }
            }
        }
    }
}

impl CmpOp {
    fn to_filter_string(self) -> &'static str {
        match self {
            CmpOp::Eq => "=",
            CmpOp::Ne => "!=",
            CmpOp::Lt => "<",
            CmpOp::Gt => ">",
            CmpOp::Le => "<=",
            CmpOp::Ge => ">=",
        }
    }
}

impl ValueExpr {
    pub fn to_filter_string(&self) -> String {
        match self {
            ValueExpr::Ident(s) => s.clone(),
            ValueExpr::FunctionCall { name, args } => {
                format!("{}({})", name, join_filter_strings(args))
            }
            ValueExpr::Int(s) => s.clone(),
            ValueExpr::Float(s) => s.clone(),
            ValueExpr::StringLiteral(s) => s.clone(),
            ValueExpr::BoolLiteral(b) => {
                if *b {
                    "true".to_string()
                } else {
                    "false".to_string()
                }
            }
            ValueExpr::VectorLiteral(s) => s.clone(),
            ValueExpr::MatrixLiteral(vs) => format!("[{}]", vs.join(", ")),
        }
    }
}
