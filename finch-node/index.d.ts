export interface HnswIndexParams {
  metric: number
  m: number
  efConstruction: number
  quantize: number
}

export interface IvfIndexParams {
  metric: number
  nList: number
  nIters: number
  useSoar?: boolean
  quantize: number
}

export interface FlatIndexParams {
  metric: number
  quantize: number
}

export interface InvertIndexParams {
  enableRangeOptimization?: boolean
  enableExtendedWildcard?: boolean
}

export interface FieldSchemaOptions {
  name: string
  dataType: number
  nullable?: boolean
  dimension?: number
  hnswIndex?: HnswIndexParams
  ivfIndex?: IvfIndexParams
  flatIndex?: FlatIndexParams
  invertIndex?: InvertIndexParams
}

export interface CollectionSchemaOptions {
  name: string
  fields: FieldSchemaOptions[]
  maxDocCountPerSegment?: number
}

export interface DocObject {
  pk: string
  score?: number
  fields: Record<string, any>
}

export interface VectorQueryOptions {
  fieldName: string
  queryVector: number[]
  /**
   * Binary32 query vector (Hamming). When provided (non-empty), it takes
   * precedence over `queryVector`.
   */
  queryVectorU32?: number[]
  /**
   * Binary64 query vector (Hamming), encoded as decimal strings to avoid JS
   * number precision loss. When provided (non-empty), it takes precedence over
   * `queryVector`.
   */
  queryVectorU64?: string[]
  topk: number
  filter?: string
  ef?: number
  nProbe?: number
  concurrency?: number
  bfPks?: string[]
  radius?: number
  isLinear?: boolean
  useRefiner?: boolean
  refinerK?: number
  refinerScaleFactor?: number
  sparseIndices?: number[]
  sparseValues?: number[]
  includeVector?: boolean
  includeDocId?: boolean
  outputFields?: string[]
}

export interface GroupResultObject {
  groupValue: any
  docs: DocObject[]
}

export interface StatusObject {
  ok: boolean
  code: number
  message: string
}

export interface MemoryStoreOptions {
  readOnly?: boolean
  enableMmap?: boolean
  maxBufferSize?: number
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

export interface SearchResponse {
  hits: Record<string, unknown>[]
}

export interface ClaimListResult {
  claims: Record<string, unknown>[]
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
  static create(input: { path: string; embeddingDim: number; options?: MemoryStoreOptions }): Promise<FinchMemory>
  static open(input: { path: string; options?: MemoryStoreOptions }): Promise<FinchMemory>
  ingestEpisode(input: EpisodeInput): Promise<Record<string, unknown>>
  ingestArtifact(input: ArtifactInput): Promise<Record<string, unknown>>
  addClaim(input: ClaimInput): Promise<Record<string, unknown>>
  listClaims(input: { scope: MemoryScope; limit?: number; at?: string | number | Date }): Promise<ClaimListResult>
  getClaim(id: string, input: { scope: MemoryScope; limit?: number; at?: string | number | Date }): Promise<Record<string, unknown> | null>
  addCorrection(input: CorrectionInput): Promise<Record<string, unknown>>
  tombstone(input: Omit<CorrectionInput, 'operation'>): Promise<Record<string, unknown>>
  restore(input: Omit<CorrectionInput, 'operation'>): Promise<Record<string, unknown>>
  addSlotAlias(input: SlotAliasInput): Promise<Record<string, unknown>>
  search(input: SearchRequest): Promise<SearchResponse>
  projectAnswerReadyState(input: ProjectAnswerReadyStateRequest): Promise<Record<string, unknown>>
  packContext(input: PackContextRequest): Promise<PackedMemoryContext>
  validateAnswer(input: ValidateAnswerRequest): Promise<ValidatedAnswerEmission>
  finalizeAnswer(input: ValidateAnswerRequest): Promise<GroundedAnswerResult>
  compareSearchModes(input: Record<string, unknown>): Promise<Record<string, unknown>>
}

export declare class MemoryStore {
  static create(path: string, embeddingDim: number, options?: MemoryStoreOptions): MemoryStore
  static open(path: string, options?: MemoryStoreOptions): MemoryStore
  /** JSON of `AnswerReadyStateJsonRequest`. */
  projectAnswerReadyStateJson(requestJson: string): string
  ingestEpisodeJson(inputJson: string, ingestedAtMs: number, chunkOptionsJson?: string): string
  appendArtifactJson(recordJson: string): string
  ingestArtifactTextJson(recordJson: string, text: string, chunkOptionsJson?: string): string
  addCorrectionJson(inputJson: string, createdAtMs: number): string
  addManualClaimJson(inputJson: string, embedding?: number[]): string
  addProfileJson(inputJson: string): string
  addEntityJson(inputJson: string, embedding?: number[]): string
  addEdgeJson(inputJson: string): string
  addSlotAliasJson(inputJson: string, recordedAtMs: number): string
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

export declare class Collection {
  insert(docs: DocObject[]): StatusObject[]
  upsert(docs: DocObject[]): StatusObject[]
  update(docs: DocObject[]): StatusObject[]
  delete(pks: string[]): StatusObject[]
  deleteByFilter(filter: string): StatusObject
	  query(opts: VectorQueryOptions): DocObject[]
	  /**
	   * Execute a SQL SELECT query (Finch extension).
	   *
	   * Note: 64-bit unsigned integers (e.g. `_finch_g_doc_id_`, `_finch_row_id_`) are
	   * returned as decimal strings to avoid JavaScript precision loss.
	   */
	  querySql(sql: string): DocObject[]
  fetch(pks: string[]): Record<string, DocObject>
  statsInfo(): any
  groupByQuery(
    opts: VectorQueryOptions,
    groupByField: string,
    groupCount: number,
    groupTopk: number
  ): GroupResultObject[]
  createIndex(
    field: string,
    fieldOpts: FieldSchemaOptions,
    rebuild?: boolean,
    concurrency?: number
  ): void
  dropIndex(field: string): void
  addColumn(
    fieldOpts: FieldSchemaOptions,
    rebuildIndex?: boolean,
    concurrency?: number,
    expression?: string
  ): void
  dropColumn(field: string): void
  alterColumn(
    field: string,
    renameTo?: string,
    rebuildIndex?: boolean,
    concurrency?: number
  ): void
  optimize(maxSegments?: number, concurrency?: number): void
	flush(): void
	stats(): Record<string, number>
	schemaInfo(): any
	path(): string
	options(): any
	destroy(): void
}

export declare function createAndOpen(
  path: string,
  schema: CollectionSchemaOptions,
  readOnly?: boolean,
  enableMmap?: boolean,
  maxBufferSize?: number
): Collection

export declare function open(
  path: string,
  readOnly?: boolean,
  enableMmap?: boolean,
  maxBufferSize?: number
): Collection

export interface GlobalConfigOptions {
  memoryLimitBytes?: number
  /** 0=Debug, 1=Info, 2=Warn, 3=Error, 4=Fatal */
  logLevel?: number
  logToFile?: boolean
  logDir?: string
  logBasename?: string
  logFileSizeMb?: number
  logOverdueDays?: number
  queryThreadCount?: number
  walFlushEveryDocs?: number
  walFsyncEveryDocs?: number
  invertToForwardScanRatio?: number
  bruteForceByKeysRatio?: number
  optimizeThreadCount?: number
}

export declare function initGlobalConfig(opts?: GlobalConfigOptions): void

export declare enum MetricType {
  Undefined = 0,
  L2 = 1,
  InnerProduct = 2,
  Cosine = 3,
  MipsL2 = 4,
  Hamming = 5,
}

export declare enum DataType {
  Undefined = 0,
  Binary = 1,
  String = 2,
  Bool = 3,
  Int32 = 4,
  Int64 = 5,
  Uint32 = 6,
  Uint64 = 7,
  Float32 = 8,
  Float64 = 9,
  Int8 = 100,
  Int16 = 101,
  Uint8 = 102,
  Uint16 = 103,
  Float16 = 104,
  Bytes = 105,
  VectorBinary32 = 20,
  VectorBinary64 = 21,
  VectorFp16 = 22,
  VectorFp32 = 23,
  VectorFp64 = 24,
  VectorInt4 = 25,
  VectorInt8 = 26,
  VectorInt16 = 27,
  VectorBool = 120,
  VectorInt32 = 121,
  VectorInt64 = 122,
  VectorUint32 = 123,
  VectorUint64 = 124,
  SparseFp16 = 30,
  SparseFp32 = 31,
  ArrayBinary = 40,
  ArrayString = 41,
  ArrayBool = 42,
  ArrayInt32 = 43,
  ArrayInt64 = 44,
  ArrayUint32 = 45,
  ArrayUint64 = 46,
  ArrayFp32 = 47,
  ArrayFp64 = 48,
}

export declare enum QuantizeType {
  Undefined = 0,
  Fp16 = 1,
  Int8 = 2,
  Int4 = 3,
}
