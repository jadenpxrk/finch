const native = require('./index.node');
const crypto = require('crypto');

function nowMs() {
  return Date.now();
}

function toMs(value) {
  if (value == null) return undefined;
  if (value instanceof Date) {
    const time = value.getTime();
    if (Number.isNaN(time)) throw new TypeError('timestamp Date is invalid');
    return time;
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new TypeError('timestamp number must be finite epoch ms');
    return value;
  }
  if (typeof value === 'string') {
    const parsed = Date.parse(value);
    if (Number.isNaN(parsed)) throw new TypeError('timestamp string must parse as RFC3339/ISO date');
    return parsed;
  }
  throw new TypeError('timestamp must be Date, RFC3339 string, epoch ms, or undefined');
}

function scope(input) {
  const space = input.space ?? input.spaceId ?? input.space_id;
  if (!space) throw new Error('scope.space is required');
  return {
    space_id: space,
    tenant_id: input.tenantId ?? input.tenant_id ?? null,
    user_id: input.userId ?? input.user_id ?? null,
    agent_id: input.agentId ?? input.agent_id ?? null,
    project_id: input.projectId ?? input.project_id ?? null,
    thread_id: input.threadId ?? input.thread_id ?? null,
  };
}

function parse(raw) {
  return JSON.parse(raw);
}

function stableHash(text) {
  return crypto.createHash('sha256').update(text).digest('hex');
}

function stableId(prefix, text) {
  return `${prefix}_${stableHash(text).slice(0, 16)}`;
}

function normalizeStateSelections(selections = []) {
  return selections.map((selection) => {
    const slotId = selection.slotId ?? selection.slot_id;
    if (!slotId) throw new TypeError('state selection slotId is required');
    return {
      slot_id: slotId,
      view: selection.view ?? 'current',
    };
  });
}

class RawMemoryApi {
  constructor(store) {
    this.store = store;
  }

  async call(operation, payload = {}) {
    const result = this.store[operation](...(payload.args || []));
    return typeof result === 'string' ? parse(result) : result;
  }
}

class MemoryEvalApi {
  constructor(memory) {
    this.memory = memory;
  }

  async runRetrieval(input) {
    return this.compareSearchModes(input);
  }

  async compareSearchModes(input) {
    return this.memory.compareSearchModes(input);
  }
}

class FinchMemory {
  constructor(store) {
    this.store = store;
    this.raw = new RawMemoryApi(store);
    this.experimental = this.raw;
    this.evals = new MemoryEvalApi(this);
  }

  static async create({ path, embeddingDim, options }) {
    return new FinchMemory(native.MemoryStore.create(path, embeddingDim, options));
  }

  static async open({ path, options }) {
    return new FinchMemory(native.MemoryStore.open(path, options));
  }

  ingestEpisode(input) {
    const at = toMs(input.eventTime) ?? nowMs();
    const payload = {
      id: input.id ?? null,
      scope: scope(input.scope),
      visibility: input.visibility ?? 'private',
      policy_tags: input.policyTags ?? [],
      source_kind: input.sourceKind ?? 'user_message',
      actor: input.actor ?? 'user',
      sequence_no: input.sequenceNo ?? at,
      event_time_ms: at,
      valid_from_ms: toMs(input.validFrom),
      valid_to_ms: toMs(input.validTo),
      raw_text: input.text,
      blob_ref: input.blobRef ?? null,
      mime_type: input.mimeType ?? 'text/plain',
      causal_parent_ids: input.causalParentIds ?? [],
      metadata_json: JSON.stringify(input.metadata ?? {}),
    };
    return Promise.resolve(parse(this.store.ingestEpisodeJson(JSON.stringify(payload), at)));
  }

  ingestArtifact(input) {
    const text = input.text ?? '';
    const at = toMs(input.eventTime) ?? nowMs();
    const title = input.title ?? input.uri ?? 'artifact';
    const payload = {
      id: input.id ?? stableId('artifact', `${title}:${text}`),
      scope: scope(input.scope),
      status: 'active',
      visibility: input.visibility ?? 'private',
      policy_tags: input.policyTags ?? [],
      artifact_kind: input.artifactKind ?? (input.uri ? 'url' : 'document'),
      title,
      uri: input.uri ?? null,
      blob_ref: input.blobRef ?? null,
      mime_type: input.mimeType ?? null,
      content_hash: stableHash(text),
      source_created_at_ms: toMs(input.sourceCreatedAt),
      source_modified_at_ms: toMs(input.sourceModifiedAt),
      created_at_ms: at,
      ingested_at_ms: at,
      valid_from_ms: toMs(input.validFrom),
      valid_to_ms: toMs(input.validTo),
      extracted_text_ref: input.extractedTextRef ?? null,
      metadata_json: JSON.stringify(input.metadata ?? {}),
    };
    return Promise.resolve(parse(this.store.ingestArtifactTextJson(JSON.stringify(payload), text)));
  }

  addClaim(input) {
    const now = nowMs();
    const payload = {
      id: input.id ?? null,
      scope: scope(input.scope),
      visibility: input.visibility ?? 'private',
      policy_tags: input.policyTags ?? [],
      claim_text: input.text,
      subject: input.subject ?? null,
      predicate: input.predicate ?? null,
      object_value: input.objectValue ?? null,
      claim_kind: input.claimKind ?? 'fact',
      polarity: input.polarity ?? 'affirmative',
      source_span_ids: input.sourceSpanIds ?? input.sourceIds ?? [],
      source_episode_ids: input.sourceEpisodeIds ?? [],
      asserted_by: input.assertedBy ?? 'user',
      confidence: input.confidence ?? null,
      observed_at_ms: toMs(input.observedAt) ?? now,
      valid_from_ms: toMs(input.validFrom),
      valid_to_ms: toMs(input.validTo),
    };
    return Promise.resolve(parse(this.store.addManualClaimJson(JSON.stringify(payload), input.embedding)));
  }

  async listClaims(input) {
    const claims = parse(this.store.scanCurrentClaimsJson(JSON.stringify(scope(input.scope)), input.limit ?? 1000, toMs(input.at)));
    return { claims };
  }

  async getClaim(id, input) {
    const claims = parse(this.store.scanClaimsJson(JSON.stringify(scope(input.scope)), input.limit ?? 1000, toMs(input.at)));
    return claims.find((claim) => claim.id === id) ?? null;
  }

  addCorrection(input) {
    const target = input.target ?? {};
    const selector = input.selector == null || typeof input.selector === 'string'
      ? input.selector
      : JSON.stringify(input.selector);
    const payload = {
      id: input.id ?? null,
      scope: scope(input.scope),
      visibility: input.visibility ?? 'private',
      policy_tags: input.policyTags ?? [],
      operation: input.operation,
      target_type: target.type ?? input.targetType ?? 'span',
      target_ids: target.id ? [target.id] : (input.targetIds ?? []),
      target_selector: selector ?? null,
      new_value: input.text ?? null,
      reason: input.reason ?? null,
      actor: input.actor ?? 'user',
      authority: input.authority ?? 'user',
      effective_at_ms: toMs(input.effectiveAt),
      applies_valid_from_ms: toMs(input.appliesValidFrom),
      applies_valid_to_ms: toMs(input.appliesValidTo),
      cascade_policy: input.cascadePolicy ?? null,
      metadata_json: JSON.stringify(input.metadata ?? {}),
    };
    return Promise.resolve(parse(this.store.addCorrectionJson(JSON.stringify(payload), nowMs())));
  }

  tombstone(input) {
    return this.addCorrection({ ...input, operation: 'tombstone' });
  }

  restore(input) {
    return this.addCorrection({ ...input, operation: 'restore' });
  }

  addSlotAlias(input) {
    const payload = {
      id: input.id ?? null,
      scope: scope(input.scope),
      visibility: input.visibility ?? 'private',
      policy_tags: input.policyTags ?? [],
      alias_subject: input.aliasSubject,
      alias_predicate: input.aliasPredicate,
      canonical_slot_id: input.canonicalSlotId ?? null,
      target_claim_id: input.targetClaimId ?? null,
      source_claim_ids: input.sourceClaimIds,
      valid_from_ms: toMs(input.validFrom),
      valid_to_ms: toMs(input.validTo),
    };
    return Promise.resolve(parse(this.store.addSlotAliasJson(JSON.stringify(payload), nowMs())));
  }

  search(input) {
    const scopeJson = JSON.stringify(scope(input.scope));
    const k = input.topK ?? 10;
    const at = toMs(input.at);
    const embedding = input.queryEmbedding ?? [];
    let raw;
    if (input.mode === 'keyword') {
      raw = this.store.keywordSearchSpansJson(scopeJson, input.query, k, 1000, at);
    } else if (input.mode === 'vector') {
      raw = this.store.querySpansJson(scopeJson, embedding, k, at);
    } else {
      raw = this.store.hybridSearchSpansJson(scopeJson, embedding, input.query, k, 1000, at);
    }
    let hits = parse(raw);
    if (input.includeNeighbors) {
      hits = parse(this.store.completeSpanEvidenceJson(scopeJson, JSON.stringify(hits), 1, 1, 1000, at));
    }
    return Promise.resolve({ hits });
  }

  async packContext(input) {
    const result = await this.search({
      scope: input.scope,
      query: input.query,
      topK: input.topK ?? 10,
      mode: input.mode ?? 'hybrid',
      at: input.at,
      queryEmbedding: input.queryEmbedding ?? [],
      includeNeighbors: input.completeEvidence ?? true,
    });
    let selections = normalizeStateSelections(input.stateSelections);
    if (selections.length && input.preferredSlotIds?.length) {
      throw new TypeError('stateSelections and preferredSlotIds are mutually exclusive');
    }
    if (input.preferredSlotIds?.length) {
      selections = input.preferredSlotIds.map((slotId) => ({ slot_id: String(slotId) }));
    }
    return parse(this.store.buildAnswerReadyContextJson(JSON.stringify({
      scope: scope(input.scope),
      query_text: input.query,
      hits: result.hits,
      selections,
      profile_limit: input.profileLimit ?? 10,
      claim_limit: input.claimLimit ?? 20,
      artifact_limit: input.artifactLimit ?? 10,
      at_ms: toMs(input.at),
      token_budget: input.maxTokens,
      include_provenance: input.includeProvenance ?? true,
    })));
  }

  async projectAnswerReadyState(input) {
    const selections = normalizeStateSelections(input.stateSelections);
    const temporal = {
      valid_at_ms: toMs(input.validAt),
      transaction_at_ms: toMs(input.transactionAt),
    };
    return parse(this.store.projectAnswerReadyStateJson(JSON.stringify({
      projection_seed: input.projectionSeed,
      scope: scope(input.scope),
      query_text: input.query,
      evidence_hits: input.evidenceHits ?? [],
      selections,
      claim_limit: input.claimLimit ?? 256,
      entity_limit: input.entityLimit ?? 256,
      temporal,
    })));
  }

  async validateAnswer(input) {
    const scopeJson = JSON.stringify(scope(input.scope));
    if (!input.context) {
      throw new TypeError('validateAnswer requires the packed context returned by packContext');
    }
    return parse(this.store.validateAnswerContextJson(
      scopeJson,
      JSON.stringify(input.context),
      JSON.stringify(input.emission),
    ));
  }

  async finalizeAnswer(input) {
    const scopeJson = JSON.stringify(scope(input.scope));
    if (!input.context) {
      throw new TypeError('finalizeAnswer requires the packed context returned by packContext');
    }
    return parse(this.store.finalizeAnswerContextJson(
      scopeJson,
      JSON.stringify(input.context),
      JSON.stringify(input.emission),
    ));
  }

  compareSearchModes(input) {
    const scopeJson = JSON.stringify(scope(input.scope));
    const k = input.topK ?? 10;
    const at = toMs(input.at);
    const embedding = input.queryEmbedding ?? [];
    const vector = this.store.querySpansJson(scopeJson, embedding, k, at);
    const keyword = this.store.keywordSearchSpansJson(scopeJson, input.query, k, 1000, at);
    const hybrid = this.store.hybridSearchSpansJson(scopeJson, embedding, input.query, k, 1000, at);
    return Promise.resolve(parse(this.store.evaluateRetrievalBaselineJson(JSON.stringify({
      vector_hits: parse(vector),
      keyword_hits: parse(keyword),
      hybrid_hits: parse(hybrid),
      relevant_span_ids: input.relevantSpanIds ?? [],
      required_exact_tokens: input.requiredExactTokens ?? [],
      stale_or_inadmissible_span_ids: input.staleOrInadmissibleSpanIds ?? [],
      k,
    }))));
  }
}

module.exports = {
  ...native,
  FinchMemory,
};
