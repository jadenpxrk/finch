/** Errors from finch operations carry the numeric finch status code. */
export interface FinchError extends Error {
  code: number
}

export type Json =
  | null
  | boolean
  | number
  | string
  | Json[]
  | { [key: string]: Json }

export interface MemoryScope {
  space: string
  tenantId?: string
  userId?: string
  agentId?: string
  projectId?: string
  threadId?: string
}

export interface EpisodeInput {
  scope: MemoryScope
  text: string
  sourceKind?: string
  actor?: string
  eventTime?: string | number | Date
  metadata?: Record<string, Json>
}

export interface ArtifactInput {
  scope: MemoryScope
  text?: string
  uri?: string
  title?: string
  mimeType?: string
  eventTime?: string | number | Date
  metadata?: Record<string, Json>
}

export interface ClaimInput {
  scope: MemoryScope
  text: string
  sourceIds?: string[]
  validFrom?: string | number | Date
  validTo?: string | number | Date | null
  metadata?: Record<string, Json>
}

export interface TargetRef {
  type: string
  id: string
}

export interface CorrectionInput {
  scope: MemoryScope
  operation: string
  target?: TargetRef
  selector?: string | Record<string, Json>
  text?: string
  reason?: string
  effectiveAt?: string | number | Date
  metadata?: Record<string, Json>
}

export interface SearchRequest {
  scope: MemoryScope
  query: string
  topK?: number
  mode?: 'vector' | 'keyword' | 'hybrid'
  at?: string | number | Date
  queryEmbedding?: number[]
  includeNeighbors?: boolean
}

export type StateReadView = 'current' | 'timeline' | 'set'

export interface StateReadSelection {
  slotId: string
  view?: StateReadView
}

export interface ProjectAnswerReadyStateRequest {
  projectionSeed: string
  scope: MemoryScope
  query: string
  evidenceHits?: Record<string, unknown>[]
  stateSelections?: StateReadSelection[]
  claimLimit?: number
  entityLimit?: number
  validAt?: string | number | Date
  transactionAt?: string | number | Date
}

export interface PackContextRequest {
  scope: MemoryScope
  query: string
  maxTokens: number
  mode?: 'vector' | 'keyword' | 'hybrid'
  at?: string | number | Date
  queryEmbedding?: number[]
  includeProvenance?: boolean
  completeEvidence?: boolean
  preferredSlotIds?: string[]
  stateSelections?: StateReadSelection[]
}

export type AnswerSupportState = 'supported' | 'unsupported' | 'deleted' | 'no_evidence'

export interface AnswerSlotSupport {
  slot_id: string
  subject: string | null
  predicate: string | null
  view: StateReadView
  state: AnswerSupportState
  source_claim_ids: string[]
  source_span_ids: string[]
}

export interface AnswerSupportContract {
  state: AnswerSupportState
  source_claim_ids: string[]
  source_span_ids: string[]
  slots: AnswerSlotSupport[]
}

export interface PackedContextItem {
  id: string
  kind: string
  estimated_tokens: number
  slot_ids?: string[]
  source_text?: string
}

export interface PackedMemoryContext {
  body: string
  estimated_tokens: number
  budget: number
  included: PackedContextItem[]
  truncated: boolean
  support: AnswerSupportContract
  rule_outcomes: Record<string, unknown>[]
}

export interface AnswerEvidence {
  span_id: string
  quote: string
}

export interface GroundedAnswerClaim {
  text: string
  slot_id?: string
  evidence: AnswerEvidence[]
}

export interface AnswerEmission {
  disposition: 'answer' | 'refuse'
  claims: GroundedAnswerClaim[]
}

export interface ValidatedAnswerEmission extends AnswerEmission {
  source_span_ids: string[]
}

export type AnswerClaimRejectionReason =
  | 'refusal_contains_claims'
  | 'missing_text'
  | 'missing_evidence'
  | 'ambiguous_slot'
  | 'unknown_slot'
  | 'unsupported_slot'
  | 'deleted_slot'
  | 'no_evidence'
  | 'evidence_outside_support'
  | 'evidence_unavailable'
  | 'quote_mismatch'

export interface AnswerClaimValidation {
  input_index: number
  decision: 'emitted' | 'rejected'
  slot_id: string | null
  rejection_reason: AnswerClaimRejectionReason | null
}

export interface GroundedAnswerResult extends AnswerEmission {
  source_span_ids: string[]
  refusal_reason: 'requested' | 'unsupported_state' | 'deleted_state' | 'no_evidence' | 'no_valid_claims' | null
  claim_validations: AnswerClaimValidation[]
}

export interface ValidateAnswerRequest {
  scope: MemoryScope
  context: PackedMemoryContext
  emission: AnswerEmission
}

export interface SlotAliasInput {
  id?: string
  scope: MemoryScope
  visibility?: string
  policyTags?: string[]
  aliasSubject: string
  aliasPredicate: string
  canonicalSlotId?: string
  targetClaimId?: string
  sourceClaimIds: string[]
  validFrom?: string | number | Date
  validTo?: string | number | Date
}

export interface ExtractionLeaseRequest {
  workerId: string
  limit?: number
  leaseMs?: number
  maxAttempts?: number
  now?: string | number | Date
}

export interface ExtractionFailure {
  jobId: string
  workerId: string
  error: string
  maxAttempts?: number
}

export interface ExtractionJob {
  id: string
  scope: Record<string, string | null>
  episode_id: string
  sequence_no: number
  event_time_ms: number | null
  status: 'pending' | 'leased' | 'done' | 'failed' | 'skipped'
  attempt_count: number
  lease_owner: string | null
  lease_expires_at_ms: number | null
  last_error: string | null
  created_at_ms: number
  updated_at_ms: number
}

export interface ExtractionBacklog {
  jobs: number
  failed_jobs: number
  oldest_created_at_ms: number | null
}

export interface SearchResponse {
  hits: Record<string, unknown>[]
}

export interface ClaimListResult {
  claims: Record<string, unknown>[]
}

export interface ErasureReport {
  /** Number of deleted rows by table name. */
  deleted_rows: Record<string, number>
}

export interface PurgeReport {
  /** Number of deleted rows by table name. */
  deleted_rows: Record<string, number>
  /** Spans whose text lost a forgotten value. Embed them again. */
  redacted_span_ids: string[]
}

export interface MemoryEvalApi {
  runRetrieval(input: Record<string, unknown>): Promise<Record<string, unknown>>
  compareSearchModes(input: Record<string, unknown>): Promise<Record<string, unknown>>
}

export interface MemoryRawApi {
  call(operation: string, payload?: { args?: unknown[] }): Promise<unknown>
}

export declare class FinchMemory {
  readonly raw: MemoryRawApi
  readonly experimental: MemoryRawApi
  readonly evals: MemoryEvalApi
  static create(input: { url: string; name: string; embeddingDim: number; vectorIndex?: boolean; maxConnections?: number; tenantIsolation?: boolean }): Promise<FinchMemory>
  static open(input: { url: string; name: string; maxConnections?: number; tenantIsolation?: boolean }): Promise<FinchMemory>
  ingestEpisode(input: EpisodeInput): Promise<Record<string, unknown>>
  ingestArtifact(input: ArtifactInput): Promise<Record<string, unknown>>
  addClaim(input: ClaimInput): Promise<Record<string, unknown>>
  listClaims(input: { scope: MemoryScope; limit?: number; at?: string | number | Date }): Promise<ClaimListResult>
  getClaim(id: string, input: { scope: MemoryScope; limit?: number; at?: string | number | Date }): Promise<Record<string, unknown> | null>
  addCorrection(input: CorrectionInput): Promise<Record<string, unknown>>
  tombstone(input: Omit<CorrectionInput, 'operation'>): Promise<Record<string, unknown>>
  restore(input: Omit<CorrectionInput, 'operation'>): Promise<Record<string, unknown>>
  eraseScope(input: { scope: MemoryScope }): Promise<ErasureReport>
  eraseEpisodes(input: { scope: MemoryScope; episodeIds: string[] }): Promise<ErasureReport>
  purgeForgotten(input: { scope: MemoryScope; before: Date | number }): Promise<PurgeReport>
  addSlotAlias(input: SlotAliasInput): Promise<Record<string, unknown>>
  leaseExtractionJobs(input: ExtractionLeaseRequest): Promise<{ jobs: ExtractionJob[] }>
  completeExtractionJob(jobId: string, workerId: string): Promise<ExtractionJob>
  failExtractionJob(input: ExtractionFailure): Promise<ExtractionJob>
  retryExtractionJob(jobId: string): Promise<ExtractionJob>
  skipExtractionJob(jobId: string): Promise<ExtractionJob>
  extractionBacklog(scope: MemoryScope): Promise<ExtractionBacklog>
  search(input: SearchRequest): Promise<SearchResponse>
  projectAnswerReadyState(input: ProjectAnswerReadyStateRequest): Promise<Record<string, unknown>>
  packContext(input: PackContextRequest): Promise<PackedMemoryContext>
  validateAnswer(input: ValidateAnswerRequest): Promise<ValidatedAnswerEmission>
  finalizeAnswer(input: ValidateAnswerRequest): Promise<GroundedAnswerResult>
  compareSearchModes(input: Record<string, unknown>): Promise<Record<string, unknown>>
}

export declare class MemoryStore {
  /** Creates a store in the new Postgres schema `name` at `url`. */
  static create(url: string, name: string, embeddingDim: number, vectorIndex?: boolean, maxConnections?: number, tenantIsolation?: boolean): Promise<MemoryStore>
  /** Opens the store in the Postgres schema `name` at `url`. */
  static open(url: string, name: string, maxConnections?: number, tenantIsolation?: boolean): Promise<MemoryStore>
  applyStateMutationBatchJson(batchJson: string): Promise<string>
  /** JSON of `AnswerReadyStateJsonRequest`. */
  projectAnswerReadyStateJson(requestJson: string): string
  ingestEpisodeJson(inputJson: string, ingestedAtMs: number, chunkOptionsJson?: string): Promise<string>
  appendArtifactJson(recordJson: string): Promise<string>
  ingestArtifactTextJson(recordJson: string, text: string, chunkOptionsJson?: string): Promise<string>
  addCorrectionJson(inputJson: string, createdAtMs: number): Promise<string>
  eraseScopeJson(scopeJson: string): Promise<string>
  eraseEpisodesJson(scopeJson: string, episodeIds: string[]): Promise<string>
  purgeForgottenJson(scopeJson: string, beforeMs: number): Promise<string>
  addManualClaimJson(inputJson: string, embedding?: number[]): Promise<string>
  addProfileJson(inputJson: string): Promise<string>
  addEntityJson(inputJson: string, embedding?: number[]): Promise<string>
  addEdgeJson(inputJson: string): Promise<string>
  addSlotAliasJson(inputJson: string, recordedAtMs: number): Promise<string>
  /** JSON of `ExtractionLeaseRequest`. */
  leaseExtractionJobsJson(requestJson: string): Promise<string>
  completeExtractionJobJson(jobId: string, workerId: string): Promise<string>
  /** JSON of `ExtractionFailure`. */
  failExtractionJobJson(failureJson: string): Promise<string>
  retryExtractionJobJson(jobId: string): Promise<string>
  skipExtractionJobJson(jobId: string): Promise<string>
  extractionBacklogJson(scopeJson: string): string
  scanSlotAliasesJson(scopeJson: string, limit: number, atMs?: number): string
  querySpansJson(
    scopeJson: string,
    queryEmbedding: number[],
    k: number,
    atMs?: number
  ): string
  keywordSearchSpansJson(
    scopeJson: string,
    queryText: string,
    k: number,
    scanLimit: number,
    atMs?: number
  ): string
  hybridSearchSpansJson(
    scopeJson: string,
    queryEmbedding: number[],
    queryText: string,
    k: number,
    scanLimit: number,
    atMs?: number
  ): string
  hybridSearchSpansDebugJson(
    scopeJson: string,
    queryEmbedding: number[],
    queryText: string,
    k: number,
    scanLimit: number,
    atMs?: number
  ): string
  completeSpanEvidenceJson(
    scopeJson: string,
    hitsJson: string,
    before: number,
    after: number,
    scanLimit: number,
    atMs?: number
  ): string
  evaluateSpanHitsJson(
    hitsJson: string,
    relevantSpanIdsJson: string,
    requiredExactTokensJson: string,
    staleOrInadmissibleSpanIdsJson: string,
    k: number
  ): string
  /** JSON of `RetrievalBaselineJsonRequest`. */
  evaluateRetrievalBaselineJson(requestJson: string): string
  scanCorrectionsJson(scopeJson: string, limit: number): string
  scanClaimsJson(scopeJson: string, limit: number, atMs?: number): string
  scanCurrentClaimsJson(scopeJson: string, limit: number, atMs?: number): string
  scanArtifactsJson(scopeJson: string, limit: number, atMs?: number): string
  scanProfilesJson(scopeJson: string, limit: number, atMs?: number): string
  scanEntitiesJson(scopeJson: string, limit: number): string
  expandEdgesJson(
    scopeJson: string,
    seedEntityIdsJson: string,
    maxDepth: number,
    limit: number,
    atMs?: number
  ): string
  buildContextJson(hitsJson: string, tokenBudget: number, includeProvenance: boolean): string
  /** JSON of `MemoryContextJsonRequest`. */
  buildMemoryContextJson(requestJson: string): string
  /** JSON of `AnswerContextJsonRequest`; set `search` to retrieve evidence and targets. */
  buildAnswerReadyContextJson(requestJson: string): string
  validateAnswerEmissionJson(scopeJson: string, supportJson: string, emissionJson: string): string
  validateAnswerContextJson(scopeJson: string, contextJson: string, emissionJson: string): string
  finalizeAnswerContextJson(scopeJson: string, contextJson: string, emissionJson: string): string
  close(): void
}
