use super::*;

/// One end of a range scan, already in index key order.
struct RangeBound {
    bytes: Vec<u8>,
    inclusive: bool,
}

impl InvertIndex {
    /// Equality lookup: return all doc_ids where field == value
    pub fn lookup_eq(&self, value: &Value) -> ZResult<RoaringTreemap> {
        let mut prefix = encode_header(&self.field_name, NS_TERMS);
        encode_value_bytes_ordered(self.index_value_type, value, &mut prefix)
            .ok_or_else(|| Status::invalid_argument("invert index: unsupported value for field"))?;
        prefix.push(VALUE_DELIM);
        self.scan_prefix_doc_ids(&prefix)
    }

    /// Inequality lookup: return all doc_ids where field != value.
    ///
    /// NULL does not match `!=` (SQL NULL semantics), so this
    /// lookup is anchored on the non-null marker set.
    pub fn lookup_ne(&self, value: &Value) -> ZResult<RoaringTreemap> {
        let mut bm = self.lookup_is_not_null()?;
        let eq = self.lookup_eq(value)?;
        bm -= &eq;
        Ok(bm)
    }

    pub fn lookup_is_null(&self) -> ZResult<RoaringTreemap> {
        self.lookup_marker(NS_NULL)
    }

    pub fn lookup_is_not_null(&self) -> ZResult<RoaringTreemap> {
        self.lookup_marker(NS_NONNULL)
    }

    fn lookup_marker(&self, namespace: u8) -> ZResult<RoaringTreemap> {
        let mut prefix = encode_header(&self.field_name, namespace);
        prefix.push(VALUE_DELIM);
        self.scan_prefix_doc_ids(&prefix)
    }

    pub fn lookup_lt(&self, value: &Value, include_eq: bool) -> ZResult<RoaringTreemap> {
        self.lookup_range_impl_bounded(None, Some((value, include_eq)), None)?
            .ok_or_else(|| Status::internal("lookup_lt: bounded lookup returned None"))
    }

    pub fn lookup_gt(&self, value: &Value, include_eq: bool) -> ZResult<RoaringTreemap> {
        self.lookup_range_impl_bounded(Some((value, include_eq)), None, None)?
            .ok_or_else(|| Status::internal("lookup_gt: bounded lookup returned None"))
    }

    /// Range lookup with an optional early-exit bound.
    ///
    /// When `max_hits` is `Some(n)`, the lookup returns `Ok(None)` as soon as it
    /// observes more than `n` matching doc IDs. This is used to implement
    /// Ratio heuristics (invert → forward scan).
    pub fn lookup_lt_bounded(
        &self,
        value: &Value,
        include_eq: bool,
        max_hits: Option<u64>,
    ) -> ZResult<Option<RoaringTreemap>> {
        self.lookup_range_impl_bounded(None, Some((value, include_eq)), max_hits)
    }

    pub fn lookup_gt_bounded(
        &self,
        value: &Value,
        include_eq: bool,
        max_hits: Option<u64>,
    ) -> ZResult<Option<RoaringTreemap>> {
        self.lookup_range_impl_bounded(Some((value, include_eq)), None, max_hits)
    }

    fn lookup_range_impl_bounded(
        &self,
        min: Option<(&Value, bool)>,
        max: Option<(&Value, bool)>,
        max_hits: Option<u64>,
    ) -> ZResult<Option<RoaringTreemap>> {
        if !self.enable_range_optimization_terms {
            return Err(Status::permission_denied(
                "invert index range optimization is disabled",
            ));
        }
        let bound = |(v, inclusive): (&Value, bool)| -> ZResult<RangeBound> {
            let mut bytes = Vec::new();
            encode_value_bytes_ordered(self.index_value_type, v, &mut bytes).ok_or_else(|| {
                Status::invalid_argument("invert index: unsupported range literal")
            })?;
            Ok(RangeBound { bytes, inclusive })
        };
        let min = min.map(bound).transpose()?;
        let max = max.map(bound).transpose()?;
        self.scan_range_bounded(NS_TERMS, min, max, max_hits)
    }

    /// Scan one namespace between encoded value bounds; Ok(None) once more than `max_hits` match.
    fn scan_range_bounded(
        &self,
        namespace: u8,
        min: Option<RangeBound>,
        max: Option<RangeBound>,
        max_hits: Option<u64>,
    ) -> ZResult<Option<RoaringTreemap>> {
        let ns_prefix = namespace_prefix(&self.field_name, namespace);
        let header_len = ns_prefix.len();

        let start_key = if let Some(b) = &min {
            let mut k = ns_prefix.clone();
            k.extend_from_slice(&b.bytes);
            k.push(VALUE_DELIM);
            k.extend_from_slice(&0u64.to_be_bytes());
            k
        } else {
            ns_prefix.clone()
        };
        let exclusive_min: Option<&[u8]> = match &min {
            Some(b) if !b.inclusive => Some(b.bytes.as_slice()),
            _ => None,
        };

        let mut bitmap = RoaringTreemap::new();
        let mut seen: u64 = 0;
        for item in self.items.range(start_key..) {
            let k = item.key().map_err(|e| Status::io_error(e.to_string()))?;
            if !k.starts_with(&ns_prefix) {
                break;
            }
            if k.len() < header_len + 1 + 8 {
                continue;
            }
            let doc_id_offset = k.len().saturating_sub(8);
            if doc_id_offset == 0 {
                continue;
            }
            let delim_pos = doc_id_offset.saturating_sub(1);
            if delim_pos < header_len || k[delim_pos] != VALUE_DELIM {
                continue;
            }
            let val_bytes = &k[header_len..delim_pos];

            if exclusive_min == Some(val_bytes) {
                continue;
            }
            if let Some(b) = &max {
                match val_bytes.cmp(b.bytes.as_slice()) {
                    std::cmp::Ordering::Greater => break,
                    std::cmp::Ordering::Equal if !b.inclusive => break,
                    _ => {}
                }
            }

            let Ok(doc_id_bytes): Result<[u8; 8], _> = k[doc_id_offset..].try_into() else {
                continue;
            };
            let doc_id = u64::from_be_bytes(doc_id_bytes);
            bitmap.insert(doc_id);
            seen = seen.saturating_add(1);
            if let Some(max_hits) = max_hits {
                if seen > max_hits {
                    return Ok(None);
                }
            }
        }

        Ok(Some(bitmap))
    }

    /// Prefix lookup for string fields
    pub fn lookup_prefix(&self, prefix_str: &str) -> ZResult<RoaringTreemap> {
        if self.index_value_type != DataType::String || self.data_type != DataType::String {
            return Err(Status::invalid_argument(
                "prefix lookup only valid for string fields",
            ));
        }
        if prefix_str.as_bytes().contains(&VALUE_DELIM) {
            return Ok(RoaringTreemap::new());
        }

        let mut prefix = encode_header(&self.field_name, NS_TERMS);
        prefix.extend_from_slice(prefix_str.as_bytes());

        let mut bitmap = RoaringTreemap::new();
        for item in self.items.prefix(&prefix) {
            let k = item.key().map_err(|e| Status::io_error(e.to_string()))?;
            if k.len() < prefix.len() + 1 + 8 {
                continue;
            }
            let Ok(doc_id_bytes): Result<[u8; 8], _> = k[k.len() - 8..].try_into() else {
                continue;
            };
            let doc_id = u64::from_be_bytes(doc_id_bytes);
            bitmap.insert(doc_id);
        }
        Ok(bitmap)
    }

    pub fn lookup_suffix(&self, suffix_str: &str) -> ZResult<RoaringTreemap> {
        if !self.enable_extended_wildcard {
            return Err(Status::permission_denied(
                "invert index extended wildcard is disabled",
            ));
        }
        if self.index_value_type != DataType::String || self.data_type != DataType::String {
            return Err(Status::invalid_argument(
                "suffix lookup only valid for string fields",
            ));
        }
        if suffix_str.as_bytes().contains(&VALUE_DELIM) {
            return Ok(RoaringTreemap::new());
        }

        let mut prefix = encode_header(&self.field_name, NS_REVERSED_TERMS);
        let rev: String = suffix_str.chars().rev().collect();
        prefix.extend_from_slice(rev.as_bytes());

        let mut bitmap = RoaringTreemap::new();
        for item in self.items.prefix(&prefix) {
            let k = item.key().map_err(|e| Status::io_error(e.to_string()))?;
            if k.len() < prefix.len() + 1 + 8 {
                continue;
            }
            let Ok(doc_id_bytes): Result<[u8; 8], _> = k[k.len() - 8..].try_into() else {
                continue;
            };
            let doc_id = u64::from_be_bytes(doc_id_bytes);
            bitmap.insert(doc_id);
        }
        Ok(bitmap)
    }

    /// Array membership lookup: `CONTAIN_ANY` / `CONTAIN_ALL` / `NOT_CONTAIN_*`.
    ///
    /// Empty-list semantics (see .contain_empty_list. behaviors):
    /// - `contain_any ()` => false (empty result)
    /// - `contain_all ()` => `is not null`
    /// - `not contain_all ()` => false (empty result)
    /// - `not contain_any ()` => `is not null`
    pub fn lookup_contain_any(&self, values: &[Value]) -> ZResult<RoaringTreemap> {
        if !is_array_type(self.data_type) {
            return Err(Status::invalid_argument(
                "contain lookup only valid for array fields",
            ));
        }
        if values.is_empty() {
            return Ok(RoaringTreemap::new());
        }
        let mut out = RoaringTreemap::new();
        for v in values {
            out |= self.lookup_eq(v)?;
        }
        Ok(out)
    }

    pub fn lookup_contain_all(&self, values: &[Value]) -> ZResult<RoaringTreemap> {
        if !is_array_type(self.data_type) {
            return Err(Status::invalid_argument(
                "contain lookup only valid for array fields",
            ));
        }
        if values.is_empty() {
            return self.lookup_is_not_null();
        }
        let mut it = values.iter();
        let first = it.next().expect("values is non-empty");
        let mut out = self.lookup_eq(first)?;
        for v in it {
            out &= self.lookup_eq(v)?;
        }
        Ok(out)
    }

    pub fn lookup_not_contain_any(&self, values: &[Value]) -> ZResult<RoaringTreemap> {
        if !is_array_type(self.data_type) {
            return Err(Status::invalid_argument(
                "contain lookup only valid for array fields",
            ));
        }
        if values.is_empty() {
            return self.lookup_is_not_null();
        }
        let mut out = self.lookup_is_not_null()?;
        let any = self.lookup_contain_any(values)?;
        out -= &any;
        Ok(out)
    }

    pub fn lookup_not_contain_all(&self, values: &[Value]) -> ZResult<RoaringTreemap> {
        if !is_array_type(self.data_type) {
            return Err(Status::invalid_argument(
                "contain lookup only valid for array fields",
            ));
        }
        if values.is_empty() {
            return Ok(RoaringTreemap::new());
        }
        let mut out = self.lookup_is_not_null()?;
        let all = self.lookup_contain_all(values)?;
        out -= &all;
        Ok(out)
    }

    /// Contains substring lookup (linear scan - expensive)
    pub fn lookup_contains(&self, substr: &str) -> ZResult<RoaringTreemap> {
        if self.index_value_type != DataType::String || self.data_type != DataType::String {
            return Err(Status::invalid_argument(
                "contains lookup only valid for string fields",
            ));
        }

        let ns_prefix = namespace_prefix(&self.field_name, NS_TERMS);
        let header_len = ns_prefix.len();

        let mut bitmap = RoaringTreemap::new();
        for item in self.items.prefix(&ns_prefix) {
            let k = item.key().map_err(|e| Status::io_error(e.to_string()))?;
            if k.len() < header_len + 1 + 8 {
                continue;
            }
            let doc_id_offset = k.len() - 8;
            let delim_pos = doc_id_offset - 1;
            if k[delim_pos] != VALUE_DELIM {
                continue;
            }
            let val_bytes = &k[header_len..delim_pos];
            let Ok(s) = std::str::from_utf8(val_bytes) else {
                continue;
            };
            if !s.contains(substr) {
                continue;
            }
            let Ok(doc_id_bytes): Result<[u8; 8], _> = k[doc_id_offset..].try_into() else {
                continue;
            };
            let doc_id = u64::from_be_bytes(doc_id_bytes);
            bitmap.insert(doc_id);
        }
        Ok(bitmap)
    }

    pub fn lookup_array_len_eq(&self, len: u32) -> ZResult<RoaringTreemap> {
        if !is_array_type(self.data_type) {
            return Err(Status::invalid_argument(
                "array_len lookup only valid for array fields",
            ));
        }
        let mut prefix = encode_header(&self.field_name, NS_ARRAY_LEN);
        let v = Value::U32(len);
        encode_value_bytes_ordered(DataType::Uint32, &v, &mut prefix).ok_or_else(|| {
            Status::invalid_argument("invert index: unsupported array_len literal")
        })?;
        prefix.push(VALUE_DELIM);
        self.scan_prefix_doc_ids(&prefix)
    }

    pub fn lookup_array_len_ne(&self, len: u32) -> ZResult<RoaringTreemap> {
        if !is_array_type(self.data_type) {
            return Err(Status::invalid_argument(
                "array_len lookup only valid for array fields",
            ));
        }
        let mut bm = self.lookup_is_not_null()?;
        let eq = self.lookup_array_len_eq(len)?;
        bm -= &eq;
        Ok(bm)
    }

    pub fn lookup_array_len_lt(&self, len: u32, include_eq: bool) -> ZResult<RoaringTreemap> {
        if !is_array_type(self.data_type) {
            return Err(Status::invalid_argument(
                "array_len lookup only valid for array fields",
            ));
        }
        self.lookup_array_len_range_bounded(None, Some((len, include_eq)), None)?
            .ok_or_else(|| Status::internal("lookup_array_len_lt: bounded lookup returned None"))
    }

    pub fn lookup_array_len_gt(&self, len: u32, include_eq: bool) -> ZResult<RoaringTreemap> {
        if !is_array_type(self.data_type) {
            return Err(Status::invalid_argument(
                "array_len lookup only valid for array fields",
            ));
        }
        self.lookup_array_len_range_bounded(Some((len, include_eq)), None, None)?
            .ok_or_else(|| Status::internal("lookup_array_len_gt: bounded lookup returned None"))
    }

    pub fn lookup_array_len_lt_bounded(
        &self,
        len: u32,
        include_eq: bool,
        max_hits: Option<u64>,
    ) -> ZResult<Option<RoaringTreemap>> {
        if !is_array_type(self.data_type) {
            return Err(Status::invalid_argument(
                "array_len lookup only valid for array fields",
            ));
        }
        self.lookup_array_len_range_bounded(None, Some((len, include_eq)), max_hits)
    }

    pub fn lookup_array_len_gt_bounded(
        &self,
        len: u32,
        include_eq: bool,
        max_hits: Option<u64>,
    ) -> ZResult<Option<RoaringTreemap>> {
        if !is_array_type(self.data_type) {
            return Err(Status::invalid_argument(
                "array_len lookup only valid for array fields",
            ));
        }
        self.lookup_array_len_range_bounded(Some((len, include_eq)), None, max_hits)
    }

    fn lookup_array_len_range_bounded(
        &self,
        min: Option<(u32, bool)>,
        max: Option<(u32, bool)>,
        max_hits: Option<u64>,
    ) -> ZResult<Option<RoaringTreemap>> {
        if !self.enable_range_optimization_array_len {
            return Err(Status::permission_denied(
                "invert index range optimization is disabled",
            ));
        }
        let bound = |(v, inclusive): (u32, bool)| -> ZResult<RangeBound> {
            let mut bytes = Vec::new();
            let vv = Value::U32(v);
            encode_value_bytes_ordered(DataType::Uint32, &vv, &mut bytes).ok_or_else(|| {
                Status::invalid_argument("invert index: unsupported array_len literal")
            })?;
            Ok(RangeBound { bytes, inclusive })
        };
        let min = min.map(bound).transpose()?;
        let max = max.map(bound).transpose()?;
        self.scan_range_bounded(NS_ARRAY_LEN, min, max, max_hits)
    }
}
