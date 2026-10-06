use finch_types::ZResult;

use super::parser::Parser;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SqlSelect {
    pub table: String,
    pub selected: Vec<SelectItem>,
    pub where_expr: Option<LogicExpr>,
    pub order_by: Vec<OrderByItem>,
    /// Parsed as a signed integer; the planner rejects a negative LIMIT and only a missing one
    /// gets the default.
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

impl LogicExpr {
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
