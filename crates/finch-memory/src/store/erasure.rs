use super::lifecycle::correction_targets_claim;
use super::*;
use crate::state::StateMutationResult;
use finch_types::Value;

const SCOPE_FIELDS: &[&str] = &[
    "id",
    "space_id",
    "tenant_id",
    "user_id",
    "agent_id",
    "project_id",
    "thread_id",
];

/// Rows that one erase deleted.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErasureReport {
    /// Number of deleted rows by table name. A table with no deleted rows is absent.
    pub deleted_rows: BTreeMap<String, usize>,
}

/// The claims that forget corrections target, and the markers that replace them.
#[derive(Debug, Default)]
pub(crate) struct ForgetPlan {
    claim_ids: BTreeSet<MemoryId>,
    markers: Vec<ClaimRecord>,
}

/// Ids of the rows to delete, by table name.
#[derive(Debug, Default)]
pub(crate) struct ErasedIds(BTreeMap<&'static str, BTreeSet<MemoryId>>);

impl ErasedIds {
    fn of(&self, table: &str) -> &BTreeSet<MemoryId> {
        static EMPTY: BTreeSet<MemoryId> = BTreeSet::new();
        self.0.get(table).unwrap_or(&EMPTY)
    }

    fn insert(&mut self, table: &'static str, id: MemoryId) -> bool {
        self.0.entry(table).or_default().insert(id)
    }

    fn is_empty(&self) -> bool {
        self.0.values().all(BTreeSet::is_empty)
    }

    fn report(&self) -> ErasureReport {
        ErasureReport {
            deleted_rows: self
                .0
                .iter()
                .filter(|(_, ids)| !ids.is_empty())
                .map(|(table, ids)| (table.to_string(), ids.len()))
                .collect(),
        }
    }
}

impl MemoryStore {
    /// Deletes every row of `scope` and of each scope inside it, in every table, as one write.
    /// Rows of broader and sibling scopes stay unchanged.
    pub fn erase_scope(&self, scope: &MemoryScope) -> ZResult<ErasureReport> {
        self.with_state_mutation(scope, || {
            let filter = scope_filter(scope);
            let mut report = ErasureReport::default();
            for (table, rows) in self.tables() {
                let deleted = rows.delete(&filter)?;
                if deleted > 0 {
                    report.deleted_rows.insert(table.to_string(), deleted);
                }
            }
            Ok(report)
        })
    }

    /// Deletes the episodes of `scope` or a scope inside it, their spans, the claims, rules, and
    /// corrections that cite them, and every record derived from those. Then projects the state
    /// of each affected scope again.
    pub fn erase_episodes(
        &self,
        scope: &MemoryScope,
        episode_ids: &[MemoryId],
    ) -> ZResult<ErasureReport> {
        self.with_state_mutation(scope, || {
            let ids = episode_ids.iter().map(String::as_str).collect::<Vec<_>>();
            let episodes =
                scan_in_chunks(&self.episodes, &ids, usize::MAX, SCOPE_FIELDS, |chunk| {
                    scoped_field_in(scope, "id", chunk)
                })?;
            let mut erased = ErasedIds::default();
            let mut scopes = BTreeSet::new();
            for doc in &episodes {
                erased.insert(EPISODES_COLLECTION, doc.pk.clone());
                scopes.insert(doc_scope(doc));
            }
            let found = erased.of(EPISODES_COLLECTION).clone();
            let found = found.iter().map(String::as_str).collect::<Vec<_>>();
            let spans = scan_in_chunks(&self.spans, &found, usize::MAX, &["id"], |chunk| {
                format!(
                    "{} AND source_type = 'episode'",
                    scoped_field_in(scope, "source_id", chunk)
                )
            })?;
            for doc in spans {
                erased.insert(SPANS_COLLECTION, doc.pk.clone());
            }
            let jobs = scan_in_chunks(
                &self.extraction_jobs,
                &found,
                usize::MAX,
                &["episode_id"],
                |chunk| scoped_field_in(scope, "episode_id", chunk),
            )?;
            for doc in jobs {
                erased.insert(EXTRACTION_JOBS_COLLECTION, doc.pk.clone());
            }
            let erased = self.erase_with_dependents(scope, erased)?;
            for scope in &scopes {
                self.rebuild_state_projection(scope, None)?;
            }
            Ok(erased.report())
        })
    }

    /// The correction without any text of the claims it forgets, and the plan that deletes
    /// them. The record drops its selector and new value, and any reason or metadata that names
    /// a forgotten value. An id that names one becomes a hash of the scope, slot, and times. The
    /// target ids then name only forget markers, so the record reaches no other claim. Any other
    /// correction returns unchanged.
    pub(crate) fn plan_forget(
        &self,
        mut correction: CorrectionRecord,
        plan: &mut ForgetPlan,
    ) -> ZResult<CorrectionRecord> {
        if !forgets_claims(&correction) {
            return Ok(correction);
        }
        let targets = self.forget_targets(&correction)?;
        if targets.is_empty() {
            return Ok(correction);
        }
        let values = targets
            .iter()
            .filter_map(|claim| claim.object_value.as_deref())
            .map(str::to_lowercase)
            .filter(|value| !value.is_empty())
            .collect::<Vec<_>>();
        let names_value = |text: &str| {
            let text = text.to_lowercase();
            values.iter().any(|value| text.contains(value.as_str()))
        };
        if correction.reason.as_deref().is_some_and(names_value) {
            correction.reason = None;
        }
        if correction.metadata_json.as_deref().is_some_and(names_value) {
            correction.metadata_json = None;
        }
        if names_value(&correction.id) {
            correction.id = format!(
                "forget_{}",
                crate::ingest::stable_hash_hex(&[
                    &format!("{:?}", correction.scope),
                    correction.target_slot_id.as_deref().unwrap_or_default(),
                    &correction.effective_at_ms.to_string(),
                    &correction.created_at_ms.to_string(),
                ])
            );
        }
        let (marker_ids, markers) = forget_markers(&correction, &targets);
        correction.target_ids = marker_ids;
        correction.target_selector = None;
        correction.new_value = None;
        plan.claim_ids
            .extend(targets.into_iter().map(|claim| claim.id));
        plan.markers.extend(markers);
        Ok(correction)
    }

    /// Deletes the planned claims with every record derived from them, writes the forget markers
    /// that keep each single-valued slot a tombstone, and projects the scope's state again.
    pub(crate) fn apply_forget(&self, scope: &MemoryScope, plan: ForgetPlan) -> ZResult<ErasedIds> {
        let mut seed = ErasedIds::default();
        for id in plan.claim_ids {
            seed.insert(CLAIMS_COLLECTION, id);
        }
        if seed.is_empty() {
            return Ok(seed);
        }
        let erased = self.erase_with_dependents(scope, seed)?;
        let marker_docs = plan
            .markers
            .iter()
            .map(|claim| claim_doc(claim, None).map_err(json_error))
            .collect::<ZResult<Vec<_>>>()?;
        upsert_many(&self.claims, marker_docs)?;
        self.rebuild_state_projection(scope, None)?;
        Ok(erased)
    }

    /// Every stored claim version the correction targets, whatever its status or validity.
    fn forget_targets(&self, correction: &CorrectionRecord) -> ZResult<Vec<ClaimRecord>> {
        self.scope_rows(&self.claims, &correction.scope)?
            .iter()
            .map(|doc| claim_from_doc(doc))
            .filter(|claim| {
                claim
                    .as_ref()
                    .map_or(true, |claim| correction_targets_claim(correction, claim))
            })
            .collect()
    }

    /// Adds to `erased` every row that cites an erased row, and deletes them all. Entities,
    /// aliases, and slots that keep other evidence stay, without the erased ids.
    fn erase_with_dependents(
        &self,
        scope: &MemoryScope,
        mut erased: ErasedIds,
    ) -> ZResult<ErasedIds> {
        let evidence = erased
            .of(SPANS_COLLECTION)
            .union(erased.of(EPISODES_COLLECTION))
            .cloned()
            .collect::<BTreeSet<_>>();
        let cites_evidence = |doc: &Doc, fields: &[&str]| cites_any(doc, fields, &evidence);
        for (table, rows, fields) in [
            (
                CLAIMS_COLLECTION,
                &self.claims,
                &["source_spans_json", "source_eps_json"][..],
            ),
            (
                RULES_COLLECTION,
                &self.rules,
                &["source_span_ids_json", "source_eps_json"][..],
            ),
            (
                CORRECTIONS_COLLECTION,
                &self.corrections,
                &["source_spans_json", "source_eps_json"][..],
            ),
        ] {
            if evidence.is_empty() {
                break;
            }
            for doc in self.scope_rows(rows, scope)?.iter() {
                if cites_evidence(doc, fields)? {
                    erased.insert(table, doc.pk.clone());
                }
            }
        }

        let traces = self.scope_rows(&self.dependency_traces, scope)?;
        self.add_derived_claims(&traces, &mut erased)?;
        for doc in traces.iter() {
            let cites_claim = [
                "trigger_claim_id",
                "prior_trigger_claim_id",
                "derived_claim_id",
            ]
            .iter()
            .any(|field| string_in(doc, field, erased.of(CLAIMS_COLLECTION)));
            if cites_claim
                || string_in(doc, "rule_id", erased.of(RULES_COLLECTION))
                || cites_evidence(doc, &["source_spans_json", "source_eps_json"])?
            {
                erased.insert(DEPENDENCY_TRACES_COLLECTION, doc.pk.clone());
            }
        }
        for doc in self.scope_rows(&self.state_records, scope)?.iter() {
            if cites_any(doc, &["claim_ids_json"], erased.of(CLAIMS_COLLECTION))?
                || cites_any(
                    doc,
                    &["correction_ids_json"],
                    erased.of(CORRECTIONS_COLLECTION),
                )?
                || cites_any(doc, &["rule_ids_json"], erased.of(RULES_COLLECTION))?
                || cites_evidence(doc, &["source_spans_json", "source_eps_json"])?
            {
                erased.insert(STATE_RECORDS_COLLECTION, doc.pk.clone());
            }
        }
        for doc in self.scope_rows(&self.profiles, scope)?.iter() {
            if cites_any(doc, &["evidence_claims_json"], erased.of(CLAIMS_COLLECTION))?
                || cites_evidence(doc, &["source_spans_json"])?
            {
                erased.insert(PROFILES_COLLECTION, doc.pk.clone());
            }
        }

        let claims = erased.of(CLAIMS_COLLECTION).clone();
        let mut rewrites = Vec::new();
        for (table, rows, field) in [
            (ENTITIES_COLLECTION, &self.entities, "source_claims_json"),
            (
                SLOT_ALIASES_COLLECTION,
                &self.slot_aliases,
                "source_claims_json",
            ),
        ] {
            let kept = self.strip_ids(table, rows, scope, &[(field, &claims)], &mut erased)?;
            rewrites.push((rows, kept));
        }
        let entities = erased.of(ENTITIES_COLLECTION).clone();
        for doc in self.scope_rows(&self.entity_aliases, scope)?.iter() {
            if string_in(doc, "entity_id", &entities) {
                erased.insert(ENTITY_ALIASES_COLLECTION, doc.pk.clone());
            }
        }
        let kept = self.strip_ids(
            ENTITY_ALIASES_COLLECTION,
            &self.entity_aliases,
            scope,
            &[("source_claims_json", &claims)],
            &mut erased,
        )?;
        rewrites.push((&self.entity_aliases, kept));
        for doc in self.scope_rows(&self.edges, scope)?.iter() {
            if string_in(doc, "claim_id", &claims)
                || string_in(doc, "src_entity_id", &entities)
                || string_in(doc, "dst_entity_id", &entities)
                || cites_evidence(doc, &["source_spans_json"])?
            {
                erased.insert(EDGES_COLLECTION, doc.pk.clone());
            }
        }
        let rules = erased.of(RULES_COLLECTION).clone();
        let kept = self.strip_ids(
            SLOTS_COLLECTION,
            &self.slots,
            scope,
            &[
                ("source_claim_ids_json", &claims),
                ("source_rule_ids_json", &rules),
                ("source_entity_ids_json", &entities),
            ],
            &mut erased,
        )?;
        rewrites.push((&self.slots, kept));

        for (table, rows) in self.tables() {
            let ids = erased
                .of(table)
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>();
            for chunk in ids.chunks(MAX_CONTAINS_FILTER_VALUES) {
                rows.delete(&scoped_field_in(scope, "id", chunk))?;
            }
        }
        for (rows, docs) in rewrites {
            upsert_many(rows.as_ref(), docs)?;
        }
        Ok(erased)
    }

    /// Adds each claim that a rule derived from an erased claim or an erased rule, transitively.
    fn add_derived_claims(&self, traces: &[Arc<Doc>], erased: &mut ErasedIds) -> ZResult<()> {
        loop {
            let mut grew = false;
            for doc in traces {
                let from_erased = string_in(doc, "trigger_claim_id", erased.of(CLAIMS_COLLECTION))
                    || string_in(doc, "rule_id", erased.of(RULES_COLLECTION));
                if let Some(derived) = from_erased
                    .then(|| doc.fields.get("derived_claim_id").and_then(Value::as_str))
                    .flatten()
                {
                    grew |= erased.insert(CLAIMS_COLLECTION, derived.to_string());
                }
            }
            if !grew {
                return Ok(());
            }
        }
    }

    /// Removes the erased ids from the id lists of each row of `rows` that cites one. A row left
    /// with no ids in any of the lists goes into `erased`; the others return rewritten.
    fn strip_ids(
        &self,
        table: &'static str,
        rows: &Arc<dyn MemoryTable>,
        scope: &MemoryScope,
        lists: &[(&str, &BTreeSet<MemoryId>)],
        erased: &mut ErasedIds,
    ) -> ZResult<Vec<Doc>> {
        let mut citing = Vec::new();
        for doc in self.scope_rows(rows, scope)?.iter() {
            let mut cites = false;
            for (field, ids) in lists {
                cites |= cites_any(doc, &[field], ids)?;
            }
            if cites && !erased.of(table).contains(&doc.pk) {
                citing.push(doc.pk.clone());
            }
        }
        // A fetch returns the vector fields too, so the rewrite keeps them.
        let full = rows.fetch(citing.clone())?;
        let mut kept = Vec::new();
        for pk in citing {
            let Some(doc) = full.get(&pk) else {
                continue;
            };
            let mut doc = Doc::clone(doc);
            let mut any_left = false;
            for (field, ids) in lists {
                let left = id_list(&doc, field)?
                    .into_iter()
                    .filter(|id| !ids.contains(id))
                    .collect::<Vec<_>>();
                any_left |= !left.is_empty();
                let json = serde_json::to_string(&left).map_err(json_error)?;
                doc.fields.insert(field.to_string(), Value::String(json));
            }
            if any_left {
                kept.push(doc);
            } else {
                erased.insert(table, pk);
            }
        }
        Ok(kept)
    }

    /// Every row of `scope` and the scopes inside it, without vector fields.
    fn scope_rows(
        &self,
        rows: &Arc<dyn MemoryTable>,
        scope: &MemoryScope,
    ) -> ZResult<Vec<Arc<Doc>>> {
        rows.scan_filter_only(
            VectorQuery::new("", Vec::new(), usize::MAX).with_filter(scope_filter(scope)),
        )
    }

    fn tables(&self) -> [(&'static str, &Arc<dyn MemoryTable>); 15] {
        [
            (EPISODES_COLLECTION, &self.episodes),
            (SPANS_COLLECTION, &self.spans),
            (ARTIFACTS_COLLECTION, &self.artifacts),
            (CORRECTIONS_COLLECTION, &self.corrections),
            (CLAIMS_COLLECTION, &self.claims),
            (PROFILES_COLLECTION, &self.profiles),
            (ENTITIES_COLLECTION, &self.entities),
            (ENTITY_ALIASES_COLLECTION, &self.entity_aliases),
            (EDGES_COLLECTION, &self.edges),
            (RULES_COLLECTION, &self.rules),
            (STATE_RECORDS_COLLECTION, &self.state_records),
            (DEPENDENCY_TRACES_COLLECTION, &self.dependency_traces),
            (SLOTS_COLLECTION, &self.slots),
            (SLOT_ALIASES_COLLECTION, &self.slot_aliases),
            (EXTRACTION_JOBS_COLLECTION, &self.extraction_jobs),
        ]
    }
}

/// Removes from a batch result the records that a forget deleted.
pub(crate) fn retain_unerased(result: &mut StateMutationResult, erased: &ErasedIds) {
    let claims = erased.of(CLAIMS_COLLECTION);
    result.claims.retain(|claim| !claims.contains(&claim.id));
    result
        .derived_claims
        .retain(|claim| !claims.contains(&claim.id));
    result
        .rule_applications
        .retain(|application| !claims.contains(&application.claim.id));
    let entities = erased.of(ENTITIES_COLLECTION);
    result
        .entities
        .retain(|entity| !entities.contains(&entity.id));
    let aliases = erased.of(SLOT_ALIASES_COLLECTION);
    result
        .slot_aliases
        .retain(|alias| !aliases.contains(&alias.id));
    let states = erased.of(STATE_RECORDS_COLLECTION);
    result
        .state_records
        .retain(|record| !states.contains(&record.id));
}

/// One forget id per slot of the forgotten claims, and a marker claim with that id for each
/// single-valued slot. A marker has no value and holds from the forget, and the forget targets
/// it, so the slot projects a tombstone and an older claim does not become current. A set
/// member is forgotten alone, so a set-valued slot gets no marker.
fn forget_markers(
    correction: &CorrectionRecord,
    targets: &[ClaimRecord],
) -> (Vec<MemoryId>, Vec<ClaimRecord>) {
    let mut newest = BTreeMap::<String, Option<&ClaimRecord>>::new();
    for claim in targets {
        // A claim with no slot gets its own id, hashed so the id names no value.
        let key = claim.slot_id.clone().unwrap_or_else(|| claim.id.clone());
        let entry = newest.entry(key).or_default();
        if !is_set_valued(claim) && entry.is_none_or(|current| claim_is_newer(claim, current)) {
            *entry = Some(claim);
        }
    }
    let mut ids = Vec::new();
    let mut markers = Vec::new();
    for (key, claim) in newest {
        let id = format!(
            "forget_{}",
            crate::ingest::stable_hash_hex(&[&correction.id, &key])
        );
        if let Some(claim) = claim {
            markers.push(ClaimRecord {
                id: id.clone(),
                status: MemoryStatus::Active,
                claim_text: String::new(),
                object_value: None,
                source_span_ids: Vec::new(),
                source_episode_ids: Vec::new(),
                source_sequence_no: None,
                confidence: None,
                observed_at_ms: correction.effective_at_ms,
                valid_from_ms: Some(correction.effective_at_ms),
                valid_to_ms: None,
                correction_ids: Vec::new(),
                ..claim.clone()
            });
        }
        ids.push(id);
    }
    (ids, markers)
}

/// Claims of these kinds are members of a set, one lifecycle per value.
fn is_set_valued(claim: &ClaimRecord) -> bool {
    matches!(
        claim.claim_kind,
        ClaimKind::Relationship | ClaimKind::Event | ClaimKind::Preference
    )
}

fn forgets_claims(correction: &CorrectionRecord) -> bool {
    correction.operation == CorrectionOperation::Forget && correction.target_type == "claim"
}

fn doc_scope(doc: &Doc) -> MemoryScope {
    let field = |name: &str| {
        doc.fields
            .get(name)
            .and_then(Value::as_str)
            .map(str::to_string)
    };
    MemoryScope {
        space_id: field("space_id").unwrap_or_default(),
        tenant_id: field("tenant_id"),
        user_id: field("user_id"),
        agent_id: field("agent_id"),
        project_id: field("project_id"),
        thread_id: field("thread_id"),
    }
}

fn string_in(doc: &Doc, field: &str, ids: &BTreeSet<MemoryId>) -> bool {
    doc.fields
        .get(field)
        .and_then(Value::as_str)
        .is_some_and(|id| ids.contains(id))
}

/// The ids in the JSON list `field`; an absent or null field holds none.
fn id_list(doc: &Doc, field: &str) -> ZResult<Vec<MemoryId>> {
    match doc.fields.get(field).and_then(Value::as_str) {
        Some(json) => serde_json::from_str(json).map_err(|error| {
            Status::internal(format!(
                "{field} of row {} is not an id list: {error}",
                doc.pk
            ))
        }),
        None => Ok(Vec::new()),
    }
}

fn cites_any(doc: &Doc, fields: &[&str], ids: &BTreeSet<MemoryId>) -> ZResult<bool> {
    if ids.is_empty() {
        return Ok(false);
    }
    for field in fields {
        if id_list(doc, field)?.iter().any(|id| ids.contains(id)) {
            return Ok(true);
        }
    }
    Ok(false)
}
