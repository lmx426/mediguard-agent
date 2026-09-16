/** 前端类型定义——与后端 Pydantic 模型对齐。 */

/** 规则命中结果 */
export interface RuleCheckItem {
  label: string;
  current_value: string;
  threshold: string;
  hit: boolean;
  layer: 'data_quality_or_applicability' | 'risk_signal' | 'strong_review_signal';
  severity: 'info' | 'low' | 'medium' | 'high' | 'critical';
  explanation: string;
}

export interface RuleHit {
  rule_id: string;
  rule_name: string;
  hit: boolean;
  severity: 'info' | 'low' | 'medium' | 'high' | 'critical';
  reason: string;
  evidence_ref: string;
  version: string;
  layer: 'data_quality_or_applicability' | 'risk_signal' | 'strong_review_signal';
  action: 'MANUAL_REVIEW' | 'REQUEST_SUPPLEMENT' | 'SPECIAL_AUDIT' | 'DATA_QUALITY_CHECK';
  current_value: string;
  threshold: string;
  business_explanation: string;
  check_items: RuleCheckItem[];
}

/** 案件摘要——列表展示 */
export interface CaseSummary {
  case_id: string;
  case_title: string;
  case_type: string;
  risk_level: 'low' | 'medium' | 'high' | 'insufficient';
  risk_score: number;
  review_status: 'pending' | 'reviewed';
  rule_signal_count: number;
  claim_amount: number | null;
}

export interface FraudScreeningSignal {
  result: 'suspected' | 'not_suspected' | 'not_available';
  label: '有预警' | '无预警' | '未接入' | string;
  source: string;
  evidence_ref: string;
  reason?: string | null;
  probability?: number | null;
}

export interface PCAFeatureScore {
  name: string;
  score: number;
}

export interface RiskScoreComponent {
  key: 'model_warning' | 'rule_check' | 'peer_deviation' | 'data_flow';
  label: string;
  score: number;
  max_score: number;
  summary: string;
  details: string[];
  source_detail?: string | null;
}

export interface RiskScoreBreakdown {
  total_score: number;
  max_score: number;
  display_score: string;
  level: CaseSummary['risk_level'];
  level_label: string;
  components: RiskScoreComponent[];
  cap_note?: string | null;
}

/** 案件详情 */
export interface CaseDetail extends CaseSummary {
  claim_summary: string;
  model_evidence_ref: string;
  rule_hits: RuleHit[];
  expected_recommendation: string;
  model_signal_source: string;
  model_signal_reasons: string[];
  review_priority: 'routine' | 'manual_review' | 'high_priority' | 'special_audit';
  evidence_consistency: string;
  subject_ref: string | null;
  rule_pool_version: string;
  rule_baseline_version: string;
  input_features: Record<string, number>;
  source_record: Record<string, number | string>;
  fraud_screening?: FraudScreeningSignal;
  pca_feature_scores: PCAFeatureScore[];
  risk_score_breakdown?: RiskScoreBreakdown | null;
}

/** 单条统计特征输入 */
export interface FeatureRecordInput {
  case_id?: string | null;
  case_title?: string | null;
  case_type?: string | null;
  features: Record<string, number | string>;
}

/** 输入字段校验定义 */
export interface FeatureSchemaResponse {
  required_fields: string[];
  supported_fields: string[];
  forbidden_fields: string[];
  example_json: FeatureRecordInput;
}

/** 完整脱敏宽表输入 */
export interface IngestRecordInput {
  case_title?: string | null;
  case_type?: string | null;
  source_system?: string | null;
  record_version?: string | null;
  case_context?: Record<string, unknown>;
  record: Record<string, number | string>;
}

/** 脱敏业务样本记录摘要 */
export interface IngestRecordSummary {
  record_id: string;
  subject_ref: string;
  risk_level: CaseSummary['risk_level'];
  risk_score: number;
  sample_result?: 'normal' | 'abnormal' | string | null;
  sample_category?: string | null;
  sample_label?: string | null;
  access_mode?: string | null;
  case_type?: string | null;
  expected_rule_ids?: string[] | null;
}

/** 脱敏业务样本记录详情 */
export interface IngestRecordResponse {
  record_id: string;
  record: Record<string, number | string>;
  metadata?: Record<string, unknown>;
  case_context?: Record<string, unknown>;
}

/** 批量完整脱敏宽表接入 */
export interface BatchIngestInput {
  records: IngestRecordInput[];
}

/** 批量完整脱敏宽表接入响应 */
export interface BatchIngestResponse {
  record_count: number;
  cases: CaseFullResponse[];
}

/** 证据引用 */
export interface Citation {
  label: string;
  source: 'model_evidence' | 'rule_evidence' | 'fraud_screening';
  ref: string;
}

export interface RuleEvidenceCard {
  rule_id: string;
  rule_name: string;
  severity: RuleHit['severity'];
  hit_fact: string;
  review_impact: string;
  suggested_action: string;
  supplement_materials: string[];
  source_ref: string;
}

export interface ReviewAction {
  title: string;
  action: string;
  rationale: string;
  source_refs: string[];
}

/** 确定性证据包生成器输出 */
export interface EvidencePackage {
  case_id: string;
  risk_summary: string;
  model_evidence: string;
  rule_evidence?: string[];
  rule_cards?: RuleEvidenceCard[];
  review_actions?: ReviewAction[];
  recommendation: string;
  missing_information?: string[];
  citations?: Citation[];
  forbidden_actions?: string[];
}

export type EvidenceAgentAnalysisType = 'comprehensive';
export type EvidenceAgentRunStatus = 'queued' | 'running' | 'complete' | 'partial' | 'failed';

export interface EvidenceAgentStatus {
  enabled: boolean;
  available: boolean;
  provider: string;
  model: string;
  persistence: string;
  thinking_enabled: boolean;
  reason?: string | null;
  error_type?: string;
  showcase?: boolean;
  read_only?: boolean;
}

export interface EvidenceAgentCitation {
  citation_id: string;
  source_type: string;
  source_ref: string;
  label: string;
  version?: string | null;
  current_value?: string | null;
  threshold?: string | null;
  metadata: Record<string, unknown>;
}

export interface EvidenceAgentStatement {
  statement: string;
  source_refs: string[];
}

export interface ReviewAdvisorSystemRiskPrompt {
  generated_by: 'backend_risk_engine';
  risk_level: 'high' | 'medium' | 'low' | 'insufficient';
  risk_level_label: string;
  risk_score: number;
  score_text: string;
  source_summary: string[];
  source_refs: string[];
}

export interface ReviewAdvisorEvidenceReview {
  relation: 'supports' | 'partially_supports' | 'weakly_supports' | 'insufficient_evidence' | 'inconsistent';
  label: string;
  support_level: '高' | '中' | '低' | '证据不足';
  summary: string;
  source_refs: string[];
}

export interface ReviewAdvisorClueReview {
  clue_id: string;
  title: string;
  status: 'supported' | 'needs_review' | 'unconfirmed';
  explanation: string;
  source_refs: string[];
}

export interface EvidenceAgentVerificationItem {
  title: string;
  action: string;
  rationale: string;
  relation_type?: 'risk_score_related' | 'scan_discovered';
  priority?: 'high' | 'medium' | 'low';
  source_refs: string[];
}

export interface EvidenceAgentSignalReview {
  case_review_hint?: string;
  case_review_hint_source_refs?: string[];
  supported_clues: EvidenceAgentStatement[];
  needs_review: EvidenceAgentStatement[];
  unconfirmed_items: string[];
  supplementary_review_hints: EvidenceAgentStatement[];
}

export interface EvidenceAgentAnalysis {
  analysis_id: string;
  run_id: string;
  case_id: string;
  analysis_type: EvidenceAgentAnalysisType;
  status: 'complete' | 'partial';
  system_risk_prompt?: ReviewAdvisorSystemRiskPrompt | null;
  evidence_review?: ReviewAdvisorEvidenceReview | null;
  clue_reviews?: ReviewAdvisorClueReview[];
  ai_risk_label: '存在疑似欺诈风险线索' | '未发现明确疑似欺诈风险线索' | '当前证据不足';
  ai_risk_label_source_refs: string[];
  risk_judgement: string;
  risk_judgement_source_refs: string[];
  evidence_strength: string;
  evidence_strength_source_refs: string[];
  key_risk_signals: EvidenceAgentStatement[];
  human_review_focus: EvidenceAgentVerificationItem[];
  risk_overview: string;
  risk_overview_source_refs: string[];
  supporting_evidence: EvidenceAgentStatement[];
  conflicts: EvidenceAgentStatement[];
  missing_information: string[];
  missing_information_details?: EvidenceAgentStatement[];
  signal_review?: EvidenceAgentSignalReview | null;
  verification_checklist: EvidenceAgentVerificationItem[];
  citations: EvidenceAgentCitation[];
  boundary_notice: string;
  generated_notice: string;
  input_fingerprint: string;
  model_name: string;
  prompt_version: string;
  tool_version: string;
  created_at: string;
}

export interface EvidenceAgentRun {
  run_id: string;
  case_id: string;
  actor_id: string;
  analysis_type: EvidenceAgentAnalysisType;
  status: EvidenceAgentRunStatus;
  current_node: string;
  input_fingerprint: string;
  reused: boolean;
  error_code?: string | null;
  error_message?: string | null;
  model_call_count: number;
  tool_call_count: number;
  created_at: string;
  updated_at: string;
  completed_at?: string | null;
  analysis?: EvidenceAgentAnalysis | null;
}

export interface EvidenceAgentEvent {
  run_id: string;
  sequence: number;
  event_type: string;
  message: string;
  payload: Record<string, unknown>;
  created_at: string;
}

export type ReviewAdvisorStatus = EvidenceAgentStatus;
export type ReviewAdvisorCitation = EvidenceAgentCitation;
export type ReviewAdvisorStatement = EvidenceAgentStatement;
export type ReviewAdvisorVerificationItem = EvidenceAgentVerificationItem;
export type ReviewAdvisorSignalReview = EvidenceAgentSignalReview;
export type ReviewAdvisorAnalysis = EvidenceAgentAnalysis;
export type ReviewAdvisorRun = EvidenceAgentRun;
export type ReviewAdvisorEvent = EvidenceAgentEvent;

export interface StatisticalMaterialDocument {
  document_id: string;
  title: string;
  document_type: string;
  visit_date: string;
  institution: string;
  department: string;
  status: string;
  content: string;
  check_points: string[];
  category_id?: string;
  category_title?: string;
  occurred_at?: string;
  material_source?: string;
  occurrence_scene?: string;
  material_shape?: string;
  summary_items?: string[];
  tables?: BusinessMaterialTable[];
  assets?: BusinessMaterialAsset[];
  metadata?: Record<string, unknown>;
}

export interface BusinessMaterialAsset {
  asset_id: string;
  material_id: string;
  title: string;
  asset_type: 'prescription_png' | 'pharmacy_receipt_png';
  file_type: 'image/png';
  preview_url: string;
  sha256: string;
  size_bytes: number;
}

export interface BusinessMaterialTable {
  title: string;
  columns: string[];
  rows: Array<Record<string, unknown>>;
}

export interface BusinessMaterialCategory {
  category_id: string;
  title: string;
  documents: StatisticalMaterialDocument[];
}

export interface MaterialSubjectProfile {
  subject_ref: string;
  gender: string;
  age_group: string;
  insurance_type: string;
  patient_group_tags: string[];
  chronic_condition_tags: string[];
  allergy_history: string;
  primary_visit_type: string;
  registration_status: string;
}

export interface StatisticalMaterialResponse {
  case_id: string;
  disclaimer: string;
  notice?: string;
  generation_status?: 'ready' | 'failed';
  template_version?: string;
  subject_profile?: MaterialSubjectProfile | null;
  case_context?: Record<string, unknown>;
  categories?: BusinessMaterialCategory[];
  documents: StatisticalMaterialDocument[];
}

/** Trace 时间线节点 */
export interface TraceNode {
  node_id: string;
  node_name: string;
  status: 'completed' | 'pending';
  summary: string;
  order: number;
  metadata: Record<string, unknown>;
}

export type WorkflowStepKey =
  | 'case_intake'
  | 'fact_base'
  | 'risk_screening'
  | 'rule_check'
  | 'evidence_package'
  | 'initial_review'
  | 'secondary_review'
  | 'appeal_handling'
  | 'case_result';

export type WorkflowContentKey = WorkflowStepKey;

export interface WorkflowStep {
  key: WorkflowStepKey;
  title: string;
  status: 'completed' | 'current' | 'pending' | 'recorded' | 'conditional';
  summary: string;
  content_key: WorkflowContentKey;
  metadata: Record<string, unknown>;
}

export interface WorkflowResponse {
  case_id: string;
  current_step: WorkflowStepKey;
  steps: WorkflowStep[];
}

export interface AuthenticatedUser {
  id: string;
  username: string;
  display_name: string;
  department?: string | null;
  roles: string[];
}

export interface LoginInput {
  username: string;
  password: string;
}

export interface AuthResponse {
  user: AuthenticatedUser;
}

/** 人工审核意见 */
export interface ReviewDecision {
  reviewer: string;
  decision: string;
  reason: string;
  attachments: ReviewAttachment[];
  submitted_at: string;
}

export type AuditNoteSource = 'manual' | 'assistant' | 'external' | 'case_agent_adopted';

export interface AuditNoteInput {
  author?: string;
  source: AuditNoteSource;
  content: string;
  source_refs?: string[];
  materials?: ReviewAttachment[];
}

export interface AuditNote extends AuditNoteInput {
  note_id: string;
  created_at: string;
  status: 'active' | 'voided';
  voided_at?: string | null;
  voided_by?: string | null;
}

/** API 请求——提交审核意见 */
export interface ReviewAttachment {
  name: string;
  material_type: string;
  source: string;
  verification_status: string;
  remark: string;
  file_name?: string;
  file_size?: number;
  file_type?: string;
  file_last_modified?: string;
}

export interface ReviewInput {
  reviewer?: string;
  decision: string;
  reason: string;
  attachments?: ReviewAttachment[];
}

/** GET /api/v1/cases/{id} 完整响应 */
export interface CaseFullResponse {
  case: CaseDetail;
  evidence: EvidencePackage;
  review: ReviewDecision | null;
  notes: AuditNote[];
}

/** POST /api/v1/cases/{id}/review 响应 */
export interface ReviewResponse {
  review: ReviewDecision;
  trace_node: TraceNode;
}

export interface CaseAgentStatus {
  enabled: boolean;
  available: boolean;
  provider: string;
  model: string;
  classifier_model?: string;
  generator_model?: string;
  persistence: string;
  reason?: string | null;
  error_type?: string;
  showcase?: boolean;
  read_only?: boolean;
}

export interface ShowcaseCaseAgentSource {
  source_ref: string;
  label: string;
  summary: string;
}

export interface ShowcaseCaseAgentAnswer {
  key: string;
  question: string;
  answer: string;
  checks: string[];
  source_refs: string[];
}

export interface ShowcaseCaseAgentProjection {
  case_id: string;
  title: string;
  generated_notice: string;
  boundary_notice: string;
  answers: ShowcaseCaseAgentAnswer[];
  sources: ShowcaseCaseAgentSource[];
}

export interface HealthResponse {
  status: string;
  service: string;
  cases_loaded: number;
  persistence: string;
  showcase: {
    enabled: boolean;
    read_only: boolean;
    seed_case_count: number;
  };
  fraud_model: Record<string, unknown>;
  review_advisor?: EvidenceAgentStatus;
  case_agent?: CaseAgentStatus;
}

export type CaseAgentAnswerDisplayMode = 'plain' | 'grounded' | 'unavailable' | 'error';

export interface CaseAgentSourceDetailField {
  label: string;
  value: string;
}

export interface CaseAgentSourceDetail {
  fields: CaseAgentSourceDetailField[];
}

export interface CaseAgentSource {
  source_ref: string;
  source_type: string;
  title: string;
  version?: string | null;
  detail: CaseAgentSourceDetail;
  metadata: Record<string, unknown>;
}

export interface CaseAgentContentBlock {
  text: string;
  source_refs: string[];
}

export interface CaseAgentClaim {
  claim_id: string;
  need_id: string;
  need_text: string;
  text: string;
  fact_refs: string[];
  source_refs: string[];
  citation_ids: string[];
  support_status: string;
}

export interface CaseAgentCitation {
  citation_id: string;
  label: number;
  claim_id?: string | null;
  fact_refs: string[];
  evidence_refs: string[];
  source_refs: string[];
}

export interface CaseAgentAnswer {
  display_mode: CaseAgentAnswerDisplayMode;
  content_blocks: CaseAgentContentBlock[];
  sources: CaseAgentSource[];
  answer_markdown?: string | null;
  claims?: CaseAgentClaim[];
  citations?: CaseAgentCitation[];
  fallback_notice: string;
  metadata: Record<string, unknown>;
}

export interface CaseAgentMessage {
  message_id: string;
  session_id: string;
  role: 'user' | 'assistant';
  content: string;
  active_stage?: string | null;
  answer_payload?: CaseAgentAnswer | null;
  source_refs: string[];
  created_at: string;
}

export interface CaseAgentSession {
  session_id: string;
  case_id: string;
  actor_id: string;
  title: string;
  status: 'active' | 'archived';
  session_summary: string;
  current_topic?: string | null;
  pending_tool?: string | null;
  referenced_source_refs: string[];
  latest_run?: CaseAgentRunSummary | null;
  attention_type: 'none' | 'new_result' | 'needs_input' | 'degraded' | 'failed';
  has_unread_activity: boolean;
  last_read_at?: string | null;
  created_at: string;
  updated_at: string;
  messages: CaseAgentMessage[];
}

export interface CaseAgentRunSummary {
  run_id: string;
  status: CaseAgentRun['status'];
  effective_status: CaseAgentRun['status'] | 'cancelling';
  current_node: string;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  cancel_requested_at?: string | null;
}

export interface CaseAgentRun {
  run_id: string;
  session_id: string;
  case_id: string;
  actor_id: string;
  user_message_id: string;
  assistant_message_id?: string | null;
  parent_run_id?: string | null;
  resumed_by_run_id?: string | null;
  status:
    | 'created'
    | 'running'
    | 'waiting_for_user'
    | 'resuming'
    | 'completed'
    | 'failed'
    | 'cancelled'
    | 'timed_out'
    | 'degraded';
  current_node: string;
  error_code?: string | null;
  error_message?: string | null;
  degraded_reason?: string | null;
  pending_clarification?: Record<string, unknown>;
  resume_context?: Record<string, unknown>;
  model_name: string;
  model_call_count: number;
  tool_call_count: number;
  timeout_at?: string | null;
  started_at?: string | null;
  cancel_requested_at?: string | null;
  cancelled_at?: string | null;
  cancel_reason?: string | null;
  client_request_id?: string | null;
  case_context_fingerprint?: string | null;
  created_at: string;
  updated_at: string;
  completed_at?: string | null;
}

export interface CaseAgentEvent {
  run_id: string;
  sequence: number;
  event_type: string;
  message: string;
  payload: Record<string, unknown>;
  created_at: string;
}

export type MemoryType =
  | 'intent_route_hint'
  | 'policy_search_hint'
  | 'failure_hint'
  | 'answer_style_hint'
  | 'decision_plan_hint';
export type MemoryLevel =
  | 'L0_source_event'
  | 'L1_atomic_memory'
  | 'L2_scenario_memory'
  | 'L3_stable_profile_or_playbook';
export type MemoryStatus =
  | 'candidate'
  | 'active'
  | 'shadow'
  | 'rejected'
  | 'snoozed'
  | 'archived'
  | 'superseded'
  | 'revoked'
  | 'tombstoned'
  | 'conflict_review';
export type MemoryCreationMode = 'system_extracted' | 'system_consolidated';
export type MemoryActivationMode =
  | 'human_confirmed'
  | 'auto_active'
  | 'pending_review'
  | 'shadow';

export interface MemoryScope {
  scope_type: 'auditor' | 'team' | 'department' | 'global';
  scope_id: string;
}

export interface MemoryPresentationItem {
  memory_id: string;
  memory_type: MemoryType;
  memory_level: MemoryLevel;
  status: MemoryStatus;
  summary: string;
  source: string[];
  created_at: string;
  last_verified_at?: string | null;
  confidence: number;
  observation_count: number;
  scope: MemoryScope;
  allowed_consumers: string[];
  freshness_score: number;
  importance_score: number;
  projection_sync_status: string;
  creation_mode: MemoryCreationMode;
  activation_mode: MemoryActivationMode;
  activated_at?: string | null;
  activated_by?: string | null;
  archived_at?: string | null;
  archived_by?: string | null;
}

export interface MemorySafeDetail extends MemoryPresentationItem {
  structured_action: Record<string, unknown>;
  detail: string;
  source_refs: string[];
  status_events: Array<Record<string, unknown>>;
}

export interface MemoryHintPack {
  request_id: string;
  consumer: string;
  memory_type: MemoryType;
  control_hints: Array<Record<string, unknown>>;
  prompt_contexts: Array<Record<string, unknown>>;
  tool_param_hints: Array<Record<string, unknown>>;
  trace_refs: Array<Record<string, unknown>>;
  memory_ids: string[];
  retrieved_levels: MemoryLevel[];
}

export interface MemoryPreview {
  memory_id: string;
  consumer: string;
  node?: string | null;
  available: boolean;
  memory_level?: MemoryLevel | null;
  allowed_fields: string[];
  delivery_locations: string[];
  hint_pack?: MemoryHintPack | null;
  reason?: string | null;
}
