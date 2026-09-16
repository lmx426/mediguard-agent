/** API 调用封装——使用原生 fetch，不引入 TanStack Query。 */

import type {
  AuditNote,
  AuditNoteInput,
  AuthenticatedUser,
  AuthResponse,
  BatchIngestInput,
  BatchIngestResponse,
  CaseSummary,
  CaseFullResponse,
  CaseAgentEvent,
  CaseAgentRun,
  CaseAgentSession,
  CaseAgentStatus,
  FeatureRecordInput,
  FeatureSchemaResponse,
  IngestRecordInput,
  IngestRecordResponse,
  IngestRecordSummary,
  StatisticalMaterialResponse,
  ReviewInput,
  ReviewResponse,
  LoginInput,
  WorkflowResponse,
  EvidenceAgentAnalysis,
  EvidenceAgentEvent,
  EvidenceAgentRun,
  EvidenceAgentStatus,
  MemoryPreview,
  MemoryPresentationItem,
  MemorySafeDetail,
  HealthResponse,
  ShowcaseCaseAgentProjection,
} from '../types';

const BASE = '/api';

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const headers = new Headers(options?.headers);
  if (!headers.has('Content-Type')) headers.set('Content-Type', 'application/json');
  const res = await fetch(`${BASE}${url}`, {
    ...options,
    credentials: 'include',
    headers,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    const message = Array.isArray(detail.detail)
      ? detail.detail.join('；')
      : typeof detail.detail === 'object'
        ? detail.detail.message || `HTTP ${res.status}`
        : detail.detail || `HTTP ${res.status}`;
    throw new ApiError(message, res.status);
  }
  return res.json();
}

export function login(data: LoginInput): Promise<AuthResponse> {
  return request<AuthResponse>('/auth/login', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export function fetchCurrentUser(): Promise<AuthenticatedUser> {
  return request<AuthenticatedUser>('/auth/me');
}

export function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health');
}

export function logout(): Promise<{ status: string }> {
  return request<{ status: string }>('/auth/logout', { method: 'POST' });
}

/** 获取案件列表 */
export function fetchCases(): Promise<CaseSummary[]> {
  return request<CaseSummary[]>('/cases');
}

/** 获取案件完整详情（含证据包、审核意见） */
export function fetchCaseDetail(caseId: string): Promise<CaseFullResponse> {
  return request<CaseFullResponse>(`/cases/${caseId}`);
}

/** 获取案件业务流程状态 */
export function fetchCaseWorkflow(caseId: string): Promise<WorkflowResponse> {
  return request<WorkflowResponse>(`/cases/${caseId}/workflow`);
}

export function fetchMemoryPendingCount(): Promise<{ count: number }> {
  return request<{ count: number }>('/memory/pending-count');
}

export function fetchPersonalMemoryPreference(): Promise<{ enabled: boolean }> {
  return request<{ enabled: boolean }>('/memory/preference');
}

export function updatePersonalMemoryPreference(enabled: boolean): Promise<{ enabled: boolean }> {
  return request<{ enabled: boolean }>('/memory/preference', {
    method: 'PUT',
    body: JSON.stringify({ enabled }),
  });
}

export function fetchMemoryCandidates(): Promise<MemoryPresentationItem[]> {
  return request<MemoryPresentationItem[]>('/memory/candidates');
}

export function fetchActiveMemories(memoryType?: string): Promise<MemoryPresentationItem[]> {
  const query = memoryType ? `?memory_type=${encodeURIComponent(memoryType)}` : '';
  return request<MemoryPresentationItem[]>(`/memory/active${query}`);
}

export function fetchArchivedMemories(memoryType?: string): Promise<MemoryPresentationItem[]> {
  const query = memoryType ? `?memory_type=${encodeURIComponent(memoryType)}` : '';
  return request<MemoryPresentationItem[]>(`/memory/archived${query}`);
}

export function fetchMemoryDetail(memoryId: string): Promise<MemorySafeDetail> {
  return request<MemorySafeDetail>(`/memory/${encodeURIComponent(memoryId)}`);
}

export function updateMemoryStatus(
  memoryId: string,
  action: 'confirm' | 'reject' | 'snooze' | 'archive' | 'revoke' | 'restore',
  snoozedUntil?: string,
): Promise<{ memory_id: string; status: string }> {
  return request<{ memory_id: string; status: string }>(`/memory/${encodeURIComponent(memoryId)}/actions`, {
    method: 'POST',
    body: JSON.stringify({ action, ...(snoozedUntil ? { snoozed_until: snoozedUntil } : {}) }),
  });
}

export function previewMemory(
  memoryId: string,
  consumer: string,
  taskContext: Record<string, unknown>,
): Promise<MemoryPreview> {
  return request<MemoryPreview>(`/memory/${encodeURIComponent(memoryId)}/preview`, {
    method: 'POST',
    body: JSON.stringify({ consumer, task_context: taskContext }),
  });
}

/** 获取由 81 字段生成的统计材料视图 */
export function fetchCaseStatisticalMaterials(
  caseId: string,
): Promise<StatisticalMaterialResponse> {
  return request<StatisticalMaterialResponse>(
    `/cases/${caseId}/statistical-materials`,
  );
}

export function fetchReviewAdvisorStatus(): Promise<EvidenceAgentStatus> {
  return request<EvidenceAgentStatus>('/agent/status');
}

export const fetchEvidenceAgentStatus = fetchReviewAdvisorStatus;

export function fetchCaseAgentStatus(): Promise<CaseAgentStatus> {
  return request<CaseAgentStatus>('/case-agent/status');
}

export function fetchShowcaseCaseAgent(
  caseId: string,
): Promise<ShowcaseCaseAgentProjection> {
  return request<ShowcaseCaseAgentProjection>(
    `/cases/${caseId}/case-agent/showcase`,
  );
}

export function fetchCaseAgentSessions(
  caseId: string,
): Promise<CaseAgentSession[]> {
  return request<CaseAgentSession[]>(`/cases/${caseId}/case-agent/sessions`);
}

export function createCaseAgentSession(
  caseId: string,
  input: { title?: string | null } = {},
): Promise<CaseAgentSession> {
  return request<CaseAgentSession>(`/cases/${caseId}/case-agent/sessions`, {
    method: 'POST',
    body: JSON.stringify(input),
  });
}

export function fetchCaseAgentSession(
  sessionId: string,
): Promise<CaseAgentSession> {
  return request<CaseAgentSession>(`/case-agent/sessions/${sessionId}`);
}

export function archiveCaseAgentSession(sessionId: string): Promise<{ archived: boolean }> {
  return request<{ archived: boolean }>(`/case-agent/sessions/${sessionId}`, {
    method: 'DELETE',
  });
}

export function restoreCaseAgentSession(
  sessionId: string,
): Promise<CaseAgentSession> {
  return request<CaseAgentSession>(`/case-agent/sessions/${sessionId}/restore`, {
    method: 'POST',
    body: JSON.stringify({}),
  });
}

export function renameCaseAgentSession(
  sessionId: string,
  input: { title: string },
): Promise<CaseAgentSession> {
  return request<CaseAgentSession>(`/case-agent/sessions/${sessionId}`, {
    method: 'PATCH',
    body: JSON.stringify(input),
  });
}

export function sendCaseAgentMessage(
  sessionId: string,
  input: { content: string; active_stage?: string | null; client_request_id?: string },
): Promise<CaseAgentRun> {
  return request<CaseAgentRun>(`/case-agent/sessions/${sessionId}/messages`, {
    method: 'POST',
    body: JSON.stringify(input),
  });
}

export function cancelCaseAgentRun(runId: string): Promise<CaseAgentRun> {
  return request<CaseAgentRun>(`/case-agent/runs/${runId}/cancel`, {
    method: 'POST',
    body: JSON.stringify({}),
  });
}

export function markCaseAgentSessionRead(
  sessionId: string,
  throughRunId: string,
): Promise<CaseAgentSession> {
  return request<CaseAgentSession>(`/case-agent/sessions/${sessionId}/read`, {
    method: 'POST',
    body: JSON.stringify({ through_run_id: throughRunId }),
  });
}

export function fetchCaseAgentRun(runId: string): Promise<CaseAgentRun> {
  return request<CaseAgentRun>(`/case-agent/runs/${runId}`);
}

export function subscribeCaseAgentEvents(
  runId: string,
  onEvent: (event: CaseAgentEvent) => void,
  onError: () => void,
  after = 0,
): () => void {
  const source = new EventSource(`${BASE}/case-agent/runs/${runId}/events?after=${after}`);
  const eventTypes = [
    'message_received',
    'resuming',
    'resumed',
    'classifying',
    'intent_classified',
    'clarification_requested',
    'resolving_context',
    'business_semantic_planning',
    'input_ingestion',
    'early_entity_extracted',
    'semantic_intent_perception',
    'rule_precheck_hit',
    'rule_precheck_miss',
    'answer_rewrite_followup_detected',
    'intent_example_biencoder_matching',
    'intent_example_biencoder_accepted',
    'intent_example_biencoder_fallback',
    'light_semantic_context_assembled',
    'llm_semantic_parsing',
    'llm_semantic_parsed',
    'llm_semantic_parser_failed',
    'slot_merged',
    'context_need_resolved',
    'minimal_planning_context_built',
    'perception_gssc_complete',
    'perception_state_normalized',
    'perception_confidence_calibrated',
    'readiness_signal_built',
    'perceptual_state_ready',
    'perceptual_state_loaded',
    'decision_readiness_checked',
    'planner_strategy_selected',
    'previous_answer_reuse_planned',
    'decision_context_slice_loaded',
    'rule_based_plan_created',
    'heuristic_plan_created',
    'planner_llm_context_built',
    'llm_planning',
    'llm_plan_created',
    'execution_plan_normalized',
    'execution_plan_validating',
    'execution_dag_built',
    'dag_step_ready',
    'l2_derivation_running',
    'l2_derivation_complete',
    'expert_task_materialized',
    'expert_analysis_calling',
    'expert_analysis_started',
    'expert_task_understood',
    'expert_case_context_observed',
    'expert_policy_rag_calling',
    'expert_policy_rag_called',
    'expert_evidence_normalized',
    'expert_structured_facts_extracted',
    'expert_sentence_windows_built',
    'expert_candidate_windows_selected',
    'expert_slot_window_judged',
    'expert_quote_validation_completed',
    'expert_answerability_checking',
    'expert_answerability_checked',
    'expert_query_rewriting',
    'expert_query_rewritten',
    'expert_claims_composed',
    'expert_markdown_rendered',
    'expert_synthesizing',
    'expert_analysis_completed',
    'expert_analysis_unavailable',
    'expert_model_call_failed',
    'answer_context_building',
    'answer_policy_resolved',
    'answer_style_resolved',
    'capability_running',
    'capability_complete',
    'capability_unavailable',
    'capability_failed',
    'tool_running',
    'tool_complete',
    'tool_unavailable',
    'tool_failed',
    'generating',
    'validating',
    'repairing_answer',
    'waiting_for_user',
    'complete',
    'degraded',
    'failed',
    'cancel_requested',
    'cancelled',
  ];
  const handler = (raw: Event) => {
    const message = raw as MessageEvent<string>;
    onEvent(JSON.parse(message.data) as CaseAgentEvent);
  };
  eventTypes.forEach((type) => source.addEventListener(type, handler));
  source.onerror = onError;
  return () => source.close();
}

export function adoptCaseAgentNote(
  sessionId: string,
  messageId: string,
  input: { content: string; source_refs: string[] },
): Promise<AuditNote> {
  return request<AuditNote>(
    `/case-agent/sessions/${sessionId}/messages/${messageId}/adopt-note`,
    {
      method: 'POST',
      body: JSON.stringify(input),
    },
  );
}

export function fetchCurrentReviewAdvisorAnalysis(
  caseId: string,
): Promise<EvidenceAgentAnalysis | null> {
  return request<EvidenceAgentAnalysis | null>(
    `/cases/${caseId}/evidence-agent/current`,
  );
}

export function fetchLatestReviewAdvisorRun(
  caseId: string,
): Promise<EvidenceAgentRun | null> {
  return request<EvidenceAgentRun | null>(`/cases/${caseId}/evidence-agent/run`);
}

export function startReviewAdvisorRun(
  caseId: string,
  input: { analysis_type: 'comprehensive' },
): Promise<EvidenceAgentRun> {
  return request<EvidenceAgentRun>(`/cases/${caseId}/evidence-agent/runs`, {
    method: 'POST',
    body: JSON.stringify(input),
  });
}

export function fetchReviewAdvisorRun(runId: string): Promise<EvidenceAgentRun> {
  return request<EvidenceAgentRun>(`/evidence-agent/runs/${runId}`);
}

export function subscribeReviewAdvisorEvents(
  runId: string,
  onEvent: (event: EvidenceAgentEvent) => void,
  onError: () => void,
): () => void {
  const source = new EventSource(`${BASE}/evidence-agent/runs/${runId}/events`);
  const eventTypes = [
    'run_created',
    'base_loaded',
    'planning',
    'tool_running',
    'ledger_updated',
    'evidence_saturated',
    'generating',
    'repairing',
    'repairing_structure',
    'validating',
    'complete',
    'partial',
    'failed',
  ];
  const handler = (raw: Event) => {
    const message = raw as MessageEvent<string>;
    onEvent(JSON.parse(message.data) as EvidenceAgentEvent);
  };
  eventTypes.forEach((type) => source.addEventListener(type, handler));
  source.onerror = onError;
  return () => source.close();
}

export const fetchCurrentEvidenceAgentAnalysis = fetchCurrentReviewAdvisorAnalysis;
export const fetchLatestEvidenceAgentRun = fetchLatestReviewAdvisorRun;
export const startEvidenceAgentRun = startReviewAdvisorRun;
export const fetchEvidenceAgentRun = fetchReviewAdvisorRun;
export const subscribeEvidenceAgentEvents = subscribeReviewAdvisorEvents;

/** 获取单条统计记录输入校验定义 */
export function fetchFeatureSchema(): Promise<FeatureSchemaResponse> {
  return request<FeatureSchemaResponse>('/feature-schema');
}

/** 提交单条统计记录并生成案件 */
export function submitAuditRecord(
  data: FeatureRecordInput,
): Promise<CaseFullResponse> {
  return request<CaseFullResponse>('/audit-record', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

/** 获取脱敏业务样本记录摘要 */
export function fetchIngestRecords(): Promise<IngestRecordSummary[]> {
  return request<IngestRecordSummary[]>('/ingest-records');
}

/** 获取游客仿真样本摘要 */
export function fetchVisitorIngestRecords(): Promise<IngestRecordSummary[]> {
  return request<IngestRecordSummary[]>('/visitor-ingest-records');
}

/** 获取单条脱敏业务样本完整宽表记录 */
export function fetchIngestRecord(recordId: string): Promise<IngestRecordResponse> {
  return request<IngestRecordResponse>(`/ingest-records/${recordId}`);
}

/** 从脱敏业务样本记录生成案件 */
export function pushIngestRecord(recordId: string): Promise<CaseFullResponse> {
  return request<CaseFullResponse>(`/ingest-records/${recordId}/push`, {
    method: 'POST',
  });
}

/** 提交一条完整脱敏宽表记录生成案件 */
export function submitIngestRecord(
  data: IngestRecordInput,
): Promise<CaseFullResponse> {
  return request<CaseFullResponse>('/ingest-record', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

/** 提交多条完整脱敏宽表记录并批量生成案件 */
export function submitIngestBatch(
  data: BatchIngestInput,
): Promise<BatchIngestResponse> {
  return request<BatchIngestResponse>('/ingest-records/batch', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

/** 提交人工审核意见 */
export function submitReview(
  caseId: string,
  data: ReviewInput,
): Promise<ReviewResponse> {
  return request<ReviewResponse>(`/cases/${caseId}/review`, {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

/** 记录案件级审核工作笔记，不改变审核状态或 workflow。 */
export function submitAuditNote(
  caseId: string,
  data: AuditNoteInput,
): Promise<AuditNote> {
  return request<AuditNote>(`/cases/${caseId}/notes`, {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

/** 删除案件级审核工作笔记。 */
export function deleteAuditNote(
  caseId: string,
  noteId: string,
): Promise<AuditNote> {
  return request<AuditNote>(`/cases/${caseId}/notes/${noteId}`, {
    method: 'DELETE',
  }).catch((err) => {
    if (err instanceof ApiError && err.status === 404) {
      return request<AuditNote>(`/cases/${caseId}/notes/${noteId}/void`, {
        method: 'POST',
      });
    }
    throw err;
  });
}
