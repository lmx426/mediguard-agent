import {
  ArrowUpOutlined,
  CloseOutlined,
  HistoryOutlined,
  PlusOutlined,
  StopOutlined,
} from '@ant-design/icons';
import {
  Actions,
  Bubble,
  Prompts,
} from '@ant-design/x';
import type { BubbleItemType } from '@ant-design/x';
import {
  Alert,
  Button,
  Empty,
  Input,
  Modal,
  Popover,
  Tooltip,
  Typography,
  message as antdMessage,
} from 'antd';
import type { InputRef } from 'antd';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  archiveCaseAgentSession,
  cancelCaseAgentRun,
  createCaseAgentSession,
  fetchCaseAgentSession,
  fetchCaseAgentSessions,
  fetchCaseAgentRun,
  markCaseAgentSessionRead,
  renameCaseAgentSession,
  restoreCaseAgentSession,
  sendCaseAgentMessage,
  subscribeCaseAgentEvents,
} from '../services/api';
import type {
  CaseAgentAnswer,
  CaseAgentEvent,
  CaseAgentRun,
  CaseAgentSource,
  CaseAgentSession,
  CaseFullResponse,
  WorkflowStepKey,
} from '../types';
import { businessText } from '../utils/businessLabels';
import AuditAssistantIcon from './AuditAssistantIcon';
import MemoryGovernancePanel from './MemoryGovernancePanel';
import { buildPresetAnswers } from './case-agent/caseAgentPresets';
import type { PresetAnswer } from './case-agent/caseAgentPresets';

interface CaseAgentPanelProps {
  data: CaseFullResponse | null;
  activeStage?: WorkflowStepKey;
  onClose?: () => void;
  onStageChange?: (stage: WorkflowStepKey) => void;
  onNoteSaved?: (note: CaseFullResponse['notes'][number]) => void;
}

type AgentProgressStepKey =
  | 'receive'
  | 'perception'
  | 'context'
  | 'perceptual_state'
  | 'decision'
  | 'execution'
  | 'compose'
  | 'validate';
type AgentProgressLayerKey = 'l1' | 'l2' | 'l3';
type AgentProgressStatus = 'active' | 'completed' | 'failed';

type AgentProgressSubstepDefinition = {
  key: string;
  group: AgentProgressStepKey;
  label: string;
  eventTypes?: string[];
  match?: (event: CaseAgentEvent) => boolean;
};

type AgentProgressStepDefinition = {
  key: AgentProgressStepKey;
  label: string;
};

type AgentProgressSubstepView = {
  key: string;
  label: string;
  status: AgentProgressStatus;
  time: number;
};

type AgentProgressStepView = {
  key: AgentProgressStepKey;
  label: string;
  status: AgentProgressStatus;
  elapsedMs?: number;
  substeps: AgentProgressSubstepView[];
};

type AgentProgressNodeVisit = {
  definition: AgentProgressSubstepDefinition;
  time: number;
};

type SessionRunState = {
  runId: string;
  status: CaseAgentRun['status'] | 'cancelling';
  events: CaseAgentEvent[];
  pendingMessage?: string;
  startedAt: number;
  stopping: boolean;
  lastSequence: number;
};

const ACTIVE_RUN_STATUSES = new Set<CaseAgentRun['status']>([
  'created',
  'running',
  'resuming',
]);

function isActiveRunStatus(status: CaseAgentRun['status'] | 'cancelling'): boolean {
  return status === 'cancelling' || ACTIVE_RUN_STATUSES.has(status as CaseAgentRun['status']);
}

const TOOL_EVENT_TYPES = new Set([
  'tool_running',
  'tool_complete',
  'tool_unavailable',
  'tool_failed',
]);

const L2_CAPABILITY_NAMES = new Set([
  'derive_aggregation',
  'derive_filter_sort_topn',
  'derive_peer_comparison',
  'explain_rule_metric',
]);

const L3_CAPABILITY_NAMES = new Set([
  'ask_policy_expert',
  'ask_drug_clinical_expert',
  'ask_precedent_expert',
  'ask_complex_material_expert',
]);

const AGENT_PROGRESS_GROUPS: AgentProgressStepDefinition[] = [
  { key: 'receive', label: '接收问题' },
  { key: 'perception', label: '感知问题' },
  { key: 'context', label: '装配上下文' },
  { key: 'perceptual_state', label: '生成感知状态' },
  { key: 'decision', label: '决策规划' },
  { key: 'execution', label: '执行能力' },
  { key: 'compose', label: '组织最终回答' },
  { key: 'validate', label: '校验回答' },
];

const AGENT_PROGRESS_NODES: AgentProgressSubstepDefinition[] = [
  { key: 'receive_message', group: 'receive', label: '接收用户问题', eventTypes: ['message_received', 'resuming', 'resumed'] },
  { key: 'load_scope', group: 'receive', label: '加载 run / session / case binding', eventTypes: ['input_ingestion'] },

  { key: 'input_ingestion', group: 'perception', label: '输入适配', eventTypes: ['input_ingestion'] },
  { key: 'early_entity_extractor', group: 'perception', label: '基础实体抽取', eventTypes: ['early_entity_extracted'] },
  { key: 'semantic_intent', group: 'perception', label: '语义与意图感知', eventTypes: ['semantic_intent_perception'] },
  { key: 'rule_precheck', group: 'perception', label: '规则预检', eventTypes: ['rule_precheck_hit', 'rule_precheck_miss', 'answer_rewrite_followup_detected'] },
  {
    key: 'intent_example_biencoder',
    group: 'perception',
    label: '意图样例检索',
    eventTypes: [
      'intent_example_biencoder_matching',
      'intent_example_biencoder_accepted',
      'intent_example_biencoder_fallback',
    ],
  },
  {
    key: 'light_semantic_context',
    group: 'perception',
    label: '轻量上下文装配',
    eventTypes: ['light_semantic_context_assembled'],
  },
  {
    key: 'llm_semantic_parser',
    group: 'perception',
    label: 'LLM 语义解析',
    eventTypes: ['llm_semantic_parsing', 'llm_semantic_parsed', 'llm_semantic_parser_failed'],
  },
  { key: 'slot_merger', group: 'perception', label: '合并槽位与语义', eventTypes: ['slot_merged'] },

  { key: 'context_need', group: 'context', label: '判断需要哪些 Context Pack', eventTypes: ['context_need_resolved'] },
  { key: 'minimal_context', group: 'context', label: '生成 MinimalPlanningContext', eventTypes: ['minimal_planning_context_built'] },
  { key: 'perception_gssc', group: 'context', label: '执行 Perception GSSC', eventTypes: ['perception_gssc_complete'] },
  { key: 'decision_context_ref', group: 'context', label: '生成 DecisionContextRef', eventTypes: ['perception_gssc_complete'] },

  { key: 'state_normalizer', group: 'perceptual_state', label: '统一 schema / 默认值', eventTypes: ['perception_state_normalized'] },
  { key: 'confidence_calibrator', group: 'perceptual_state', label: '校准置信度', eventTypes: ['perception_confidence_calibrated'] },
  { key: 'readiness_signal', group: 'perceptual_state', label: '生成 readiness_signal', eventTypes: ['readiness_signal_built'] },
  { key: 'perceptual_state', group: 'perceptual_state', label: '输出 PerceptualState', eventTypes: ['perceptual_state_ready'] },

  { key: 'load_perceptual_state', group: 'decision', label: '读取 PerceptualState', eventTypes: ['perceptual_state_loaded'] },
  { key: 'decision_readiness', group: 'decision', label: '判断是否可规划', eventTypes: ['decision_readiness_checked'] },
  { key: 'planner_strategy', group: 'decision', label: '选择规划策略', eventTypes: ['planner_strategy_selected'] },
  { key: 'reuse_previous_answer', group: 'decision', label: '复用上一轮回答', eventTypes: ['previous_answer_reuse_planned'] },
  { key: 'rule_planner', group: 'decision', label: '规则规划', eventTypes: ['rule_based_plan_created'] },
  { key: 'heuristic_planner', group: 'decision', label: '启发式规划', eventTypes: ['heuristic_plan_created'] },
  { key: 'llm_planner', group: 'decision', label: 'LLM 规划', eventTypes: ['llm_planning', 'llm_plan_created'] },
  { key: 'plan_normalizer', group: 'decision', label: '归一化 ExecutionPlan', eventTypes: ['execution_plan_normalized'] },
  { key: 'validate_plan', group: 'decision', label: '校验计划', eventTypes: ['execution_plan_validating'] },
  { key: 'build_dag', group: 'decision', label: '生成 Execution DAG', eventTypes: ['execution_dag_built'] },

  { key: 'dispatch_l1', group: 'execution', label: '调度 L1 案件事实查询', match: (event) => eventIsLayerStep(event, 'l1') || eventIsToolForLayer(event, 'l1') },
  { key: 'dispatch_l2', group: 'execution', label: '调度 L2 业务事实整理', match: (event) => eventIsLayerStep(event, 'l2') || ['l2_derivation_running', 'l2_derivation_complete'].includes(event.event_type) },
  { key: 'dispatch_l3', group: 'execution', label: '调度 L3 Expert Analysis', match: (event) => eventIsLayerStep(event, 'l3') || ['expert_task_materialized', 'expert_analysis_calling'].includes(event.event_type) },
  { key: 'expert_task', group: 'execution', label: '生成 ExpertAnalysisTask', eventTypes: ['expert_task_materialized'] },
  { key: 'expert_understand', group: 'execution', label: '识别政策场景 / 地区 / 时间 / 政策域', eventTypes: ['expert_analysis_started', 'expert_task_understood'] },
  { key: 'expert_query', group: 'execution', label: '生成政策检索问题', eventTypes: ['expert_task_understood'] },
  { key: 'expert_rag', group: 'execution', label: '调用 Policy RAG MCP 查询依据', eventTypes: ['expert_policy_rag_calling', 'expert_policy_rag_called'] },
  { key: 'expert_evidence_normalize', group: 'execution', label: '统一政策证据格式', eventTypes: ['expert_evidence_normalized'] },
  { key: 'expert_structured_extract', group: 'execution', label: '表格证据规则抽取', eventTypes: ['expert_structured_facts_extracted'] },
  { key: 'expert_sentence_windows', group: 'execution', label: '正文切句并构建窗口', eventTypes: ['expert_sentence_windows_built'] },
  { key: 'expert_candidate_windows', group: 'execution', label: '规则粗筛候选窗口', eventTypes: ['expert_candidate_windows_selected'] },
  { key: 'expert_slot_window_judge', group: 'execution', label: 'Slot-Window Judge & Extract', eventTypes: ['expert_slot_window_judged'] },
  { key: 'expert_quote_validator', group: 'execution', label: '校验证据原文引用', eventTypes: ['expert_quote_validation_completed'] },
  { key: 'expert_slot_coverage', group: 'execution', label: '按 slot 判断证据覆盖', eventTypes: ['expert_answerability_checking'] },
  { key: 'expert_answerability', group: 'execution', label: '决定可回答 / 部分回答 / 证据不足', eventTypes: ['expert_answerability_checked'] },
  { key: 'expert_rewrite', group: 'execution', label: 'LLM 重写问题后重查', eventTypes: ['expert_query_rewriting', 'expert_query_rewritten'] },
  { key: 'expert_claims', group: 'execution', label: '生成 claim 并绑定引用', eventTypes: ['expert_claims_composed'] },
  { key: 'expert_markdown', group: 'execution', label: '生成句内引用 Markdown', eventTypes: ['expert_markdown_rendered'] },
  { key: 'expert_result', group: 'execution', label: '生成 ExpertAnalysisResult', eventTypes: ['expert_synthesizing', 'expert_analysis_completed', 'expert_analysis_unavailable'] },
  { key: 'l3_artifact', group: 'execution', label: '写入 l3_result_ref', match: (event) => eventIsExpertCapabilityFinished(event) },

  { key: 'answer_context', group: 'compose', label: '读取 artifacts / sources / refs', eventTypes: ['answer_context_building'] },
  { key: 'answer_policy', group: 'compose', label: '判断回答策略', eventTypes: ['answer_policy_resolved'] },
  { key: 'answer_merge', group: 'compose', label: '融合执行结果', eventTypes: ['answer_style_resolved'] },
  { key: 'answer_generate', group: 'compose', label: '生成最终回复', eventTypes: ['generating'] },

  { key: 'validate_citation', group: 'validate', label: '校验引用闭包', eventTypes: ['validating'] },
  { key: 'validate_decision', group: 'validate', label: '校验禁止裁决', eventTypes: ['validating'] },
  { key: 'validate_sensitive', group: 'validate', label: '校验敏感字段', eventTypes: ['validating'] },
  { key: 'validate_repair', group: 'validate', label: '必要时修复回答', eventTypes: ['repairing_answer'] },
  { key: 'persist_trace', group: 'validate', label: '保存回答与 trace', eventTypes: ['complete', 'degraded', 'failed'] },
];

const DAY_MS = 24 * 60 * 60 * 1000;

const CASE_AGENT_FIELD_VALUE_LABELS: Record<string, string> = {
  critical: '重点风险',
  high: '高风险',
  medium: '中风险',
  low: '低风险',
  pending: '待核验',
  unknown: '待确认',
  present_but_needs_verification: '已提供，需核验',
  missing: '缺失',
  ready: '已就绪',
  hit: '已命中',
  not_hit: '未命中',
  not_ready: '未就绪',
  not_available: '暂不可用',
  true: '是',
  false: '否',
  special_audit: '专项核验',
  standard_audit: '常规核验',
  routine_audit: '常规核验',
  manual_review: '人工核验',
};

type SourceDetailDisplayField = {
  label: string;
  value: string;
  tone?: 'evidence' | 'value';
};

const CASER_SOURCE_SECTION_TITLES: Record<string, string> = {
  case_basic_info: '案件基础信息',
  claimant_profile: '申报人基础信息',
  material_overview: '业务材料总览',
  medical_materials: '就诊诊疗材料',
  prescription_materials: '处方购药材料',
  settlement_materials: '费用结算材料',
  statistics_report: '统计报表',
  risk_score: '综合风险评分',
  evidence_package: '基础证据包',
  case_judgement: '案件研判',
  verification_items: '待人工核验事项',
  rule_verification: '业务规则核验清单',
};

const CASER_SOURCE_SECTION_BY_TITLE = Object.fromEntries(
  Object.entries(CASER_SOURCE_SECTION_TITLES).map(([key, title]) => [title, key]),
) as Record<string, string>;

const SOURCE_STAGE_BY_SECTION: Record<string, WorkflowStepKey> = {
  case_basic_info: 'fact_base',
  claimant_profile: 'fact_base',
  medical_materials: 'fact_base',
  prescription_materials: 'fact_base',
  settlement_materials: 'fact_base',
  material_overview: 'fact_base',
  statistics_report: 'fact_base',
  risk_score: 'risk_screening',
  rule_verification: 'rule_check',
  evidence_package: 'evidence_package',
  case_judgement: 'evidence_package',
  verification_items: 'evidence_package',
};

const MATERIAL_SECTION_KEYS = new Set([
  'material_overview',
  'medical_materials',
  'prescription_materials',
  'settlement_materials',
]);

function formatSourceFieldValue(value: string): string {
  return CASE_AGENT_FIELD_VALUE_LABELS[value] ?? businessText(value);
}

function getSourceDetailField(source: CaseAgentSource, labels: string[]): string {
  const normalizedLabels = new Set(labels);
  const field = source.detail.fields.find((item) => normalizedLabels.has(item.label));
  return field?.value?.trim() ?? '';
}

function getSourceDetailFields(source: CaseAgentSource, labels: string[]): string[] {
  const normalizedLabels = new Set(labels);
  return source.detail.fields
    .filter((item) => normalizedLabels.has(item.label))
    .map((item) => item.value?.trim() ?? '')
    .filter(Boolean);
}

function parseJsonLikeValue(value: unknown): unknown {
  if (typeof value !== 'string') return value;
  const text = value.trim();
  if (!text || (!text.startsWith('{') && !text.startsWith('['))) return value;
  try {
    return JSON.parse(text);
  } catch {
    return value;
  }
}

function sourceStructuredValueCandidates(source: CaseAgentSource): unknown[] {
  const preview = parseJsonLikeValue(source.metadata?.payload_preview);
  const previewData = isPlainRecord(preview) ? preview.data : undefined;
  const detailValues = getSourceDetailFields(source, ['数据快照', '具体数值'])
    .map(parseJsonLikeValue);
  return [
    previewData,
    ...detailValues,
  ].filter((item) => item !== undefined && item !== null && item !== '');
}

function isJsonLikeText(value: string): boolean {
  const text = value.trim();
  return text.startsWith('{') || text.startsWith('[');
}

function metadataString(source: CaseAgentSource, key: string): string {
  const value = source.metadata?.[key];
  if (typeof value === 'string') return value.trim();
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  return '';
}

function sourceSectionKey(source: CaseAgentSource): string {
  const fromMetadata = metadataString(source, 'section_key') || metadataString(source, 'section');
  if (fromMetadata) return fromMetadata;
  const sectionTitle = getSourceDetailField(source, ['业务分区']);
  return CASER_SOURCE_SECTION_BY_TITLE[sectionTitle] ?? '';
}

function sourceSectionTitle(source: CaseAgentSource): string {
  const sectionKey = sourceSectionKey(source);
  return (
    CASER_SOURCE_SECTION_TITLES[sectionKey] ||
    getSourceDetailField(source, ['业务分区']) ||
    getSourceDetailField(source, ['依据类型']) ||
    source.title ||
    '案件业务事实'
  );
}

function isInternalVersion(value: string): boolean {
  const normalized = value.trim().toLowerCase();
  return (
    !normalized ||
    normalized.startsWith('rag_') ||
    normalized.startsWith('reference_') ||
    normalized.startsWith('policy_evidence') ||
    normalized.startsWith('case_') ||
    normalized.includes('payload') ||
    normalized.includes('hash')
  );
}

function formatDateParts(year: string, month?: string, day?: string): string {
  if (!month || !day) return year;
  return `${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}`;
}

function extractPolicyTimeLabel(value: string): string {
  const text = value.trim();
  if (!text) return '';
  const cutoffMatch = text.match(/截至\s*(20\d{2})[-年./]?(\d{1,2})[-月./]?(\d{1,2})日?/);
  if (cutoffMatch) {
    return `截至 ${formatDateParts(cutoffMatch[1], cutoffMatch[2], cutoffMatch[3])}`;
  }
  const dateMatch = text.match(/(20\d{2})[-年./](\d{1,2})[-月./](\d{1,2})日?/);
  if (dateMatch) return formatDateParts(dateMatch[1], dateMatch[2], dateMatch[3]);
  const compactDateMatch = text.match(/(20\d{2})(\d{2})(\d{2})/);
  if (compactDateMatch) {
    return formatDateParts(compactDateMatch[1], compactDateMatch[2], compactDateMatch[3]);
  }
  const yearMatch = text.match(/\b(20\d{2})\b/);
  return yearMatch?.[1] ?? '';
}

function isPolicySource(source: CaseAgentSource): boolean {
  return (
    source.source_type === 'policy_rag' ||
    Boolean(metadataString(source, 'evidence_ref')) ||
    Boolean(getSourceDetailField(source, ['证据片段', '政策域', '发布日期', '生效日期']))
  );
}

function formatPolicySourceTime(source: CaseAgentSource): string {
  const candidates = [
    getSourceDetailField(source, ['发布日期']),
    getSourceDetailField(source, ['生效日期']),
    metadataString(source, 'publish_date'),
    metadataString(source, 'effective_date'),
    getSourceDetailField(source, ['时间']),
    getSourceDetailField(source, ['版本']),
    source.version ?? '',
  ];
  const value = candidates.find((item) => item && !isInternalVersion(item));
  if (value) return formatSourceFieldValue(value);
  const inferred = [source.title, ...candidates]
    .map(extractPolicyTimeLabel)
    .find(Boolean);
  return inferred || '未标注';
}

function formatPolicyRegion(source: CaseAgentSource): string {
  return (
    getSourceDetailField(source, ['地区']) ||
    metadataString(source, 'jurisdiction') ||
    metadataString(source, 'region') ||
    '未标注'
  );
}

function policySourceDetailFields(source: CaseAgentSource): SourceDetailDisplayField[] {
  const excerpt =
    getSourceDetailField(source, ['证据片段']) ||
    metadataString(source, 'excerpt') ||
    getSourceDetailField(source, ['摘要']) ||
    '未提供可展示片段';
  return [
    { label: '政策原文', value: source.title || '政策文件', tone: 'value' },
    { label: '证据片段', value: formatPolicyEvidenceSnippet(excerpt), tone: 'evidence' },
    { label: '时间', value: formatPolicySourceTime(source), tone: 'value' },
    { label: '地区', value: formatPolicyRegion(source), tone: 'value' },
  ];
}

function formatPolicyEvidenceSnippet(value: string): string {
  const normalized = value.replace(/\r\n?/g, '\n').trim();
  const evidenceText = normalized.match(
    /(?:^|\n)\s*(?:evidence_text|证据文本)[:：]\s*([^\n]+)/i,
  )?.[1]?.trim();
  const metadataLine = /^\s*(?:资料标题|地区|政策领域|来源ID|source_id|source_url|node_id|field_key|row_id|case_relevance|content_type|dataset|fetched_at|官方来源|page|extraction_method)\s*[:：]/i;
  const businessTextOnly = evidenceText || normalized
    .split('\n')
    .filter((line) => line.trim() && !metadataLine.test(line))
    .join(' ')
    .trim();
  const text = clipSourceDetailText(businessTextOnly, 900);
  if (text.length < 120) return text;
  return text
    .replace(/；\s*/g, '；\n')
    .replace(/。\s*(?=\S)/g, '。\n')
    .trim();
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function sourceScalarText(value: unknown): string {
  if (value === null || value === undefined || value === '') return '';
  if (typeof value === 'string') return formatSourceFieldValue(value);
  if (typeof value === 'number' || typeof value === 'boolean') return formatSourceFieldValue(String(value));
  return '';
}

function nestedRecord(record: Record<string, unknown>, key: string): Record<string, unknown> {
  const value = record[key];
  return isPlainRecord(value) ? value : {};
}

function recordText(record: Record<string, unknown>, keys: string[]): string {
  for (const key of keys) {
    const text = sourceScalarText(record[key]);
    if (text) return text;
  }
  return '';
}

function listText(value: unknown, limit = 3): string {
  if (Array.isArray(value)) {
    return value
      .map((item) => sourceScalarText(item))
      .filter(Boolean)
      .slice(0, limit)
      .join('；');
  }
  return sourceScalarText(value);
}

function sourcePayloadRecord(source: CaseAgentSource): Record<string, unknown> | null {
  for (const candidate of sourceStructuredValueCandidates(source)) {
    const normalized = parseJsonLikeValue(candidate);
    if (isPlainRecord(normalized)) {
      const data = normalized.data;
      return isPlainRecord(data) ? data : normalized;
    }
  }
  return null;
}

function recordArray(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.filter(isPlainRecord) : [];
}

function rowsToDetailText(rows: Array<[string, string]>): string {
  return rows
    .map(([label, value]) => [label, value.trim()] as [string, string])
    .filter(([, value]) => Boolean(value))
    .map(([label, value]) => `${label}：${value}`)
    .join('\n');
}

function compactRecordFields(
  record: Record<string, unknown>,
  keys: string[],
  separator = '，',
): string {
  return keys
    .map((key) => recordText(record, [key]))
    .filter(Boolean)
    .join(separator);
}

function recordListSummary(
  value: unknown,
  render: (record: Record<string, unknown>, index: number) => string,
  limit = 4,
): string {
  return recordArray(value)
    .slice(0, limit)
    .map(render)
    .map((item) => item.trim())
    .filter(Boolean)
    .join('；');
}

function numberPercentText(value: unknown): string {
  if (typeof value !== 'number') return '';
  if (value <= 1) return `${(value * 100).toFixed(1)}%`;
  return String(value);
}

function materialRecordMatchesSource(record: Record<string, unknown>, source: CaseAgentSource): boolean {
  const basicInfo = nestedRecord(record, 'basic_info');
  const refs = [
    recordText(record, ['source_ref', 'material_id', 'document_id']),
    recordText(basicInfo, ['source_ref', 'material_id', 'document_id']),
  ].filter(Boolean);
  return refs.some((ref) => ref === source.source_ref || source.source_ref.includes(ref));
}

function isMaterialSource(source: CaseAgentSource): boolean {
  const sectionKey = sourceSectionKey(source);
  const module = businessSourceModule(source);
  return (
    MATERIAL_SECTION_KEYS.has(sectionKey) ||
    source.source_ref.startsWith('material:') ||
    source.source_type.includes('material') ||
    module.includes('材料') ||
    source.title.includes('材料')
  );
}

function isMaterialRecord(record: Record<string, unknown>): boolean {
  const basicInfo = nestedRecord(record, 'basic_info');
  const content = nestedRecord(record, 'content');
  return Boolean(
    recordText(record, [
      'document_type',
      'material_type',
      'material_shape',
      'material_source',
      'institution',
      'occurred_at',
      'visit_date',
    ]) ||
    recordText(basicInfo, ['material_type', 'material_shape', 'material_source', 'material_status', 'occurred_at']) ||
    recordText(content, ['raw_material_summary', 'summary']) ||
    Array.isArray(record.summary_items) ||
    Array.isArray(record.check_points),
  );
}

function firstMaterialRecord(
  value: unknown,
  source: CaseAgentSource,
  depth = 0,
): Record<string, unknown> | null {
  if (depth > 4) return null;
  const normalized = parseJsonLikeValue(value);
  if (Array.isArray(normalized)) {
    const records = normalized
      .map((item) => firstMaterialRecord(item, source, depth + 1))
      .filter(isPlainRecord);
    return records.find((record) => materialRecordMatchesSource(record, source)) ?? records[0] ?? null;
  }
  if (!isPlainRecord(normalized)) return null;
  for (const key of ['materials', 'documents', 'payload', 'data']) {
    const record = firstMaterialRecord(normalized[key], source, depth + 1);
    if (record) return record;
  }
  return isMaterialRecord(normalized) ? normalized : null;
}

function materialRecordFromSource(source: CaseAgentSource): Record<string, unknown> | null {
  if (!isMaterialSource(source)) return null;
  const candidates = sourceStructuredValueCandidates(source);
  for (const candidate of candidates) {
    const record = firstMaterialRecord(candidate, source);
    if (record) return record;
  }
  return null;
}

function materialDisplayRows(source: CaseAgentSource): Array<[string, string]> {
  const material = materialRecordFromSource(source);
  if (!material) return [];
  const basicInfo = nestedRecord(material, 'basic_info');
  const content = nestedRecord(material, 'content');
  const rows: Array<[string, string]> = [
    ['材料名称', recordText(material, ['title', 'name', 'document_name']) || source.title],
    ['材料类型', recordText(material, ['document_type', 'material_type']) || recordText(basicInfo, ['material_type'])],
    ['材料形态', recordText(material, ['material_shape']) || recordText(basicInfo, ['material_shape'])],
    ['材料来源', recordText(material, ['material_source', 'institution']) || recordText(basicInfo, ['material_source'])],
    ['整理状态', recordText(material, ['status']) || recordText(basicInfo, ['material_status'])],
    ['时间范围', recordText(material, ['occurred_at', 'visit_date']) || recordText(basicInfo, ['occurred_at'])],
    [
      '摘要',
      listText(material.summary_items) ||
        recordText(content, ['raw_material_summary', 'summary']) ||
        sourceScalarText(material.content),
    ],
    ['核验要点', listText(material.check_points) || listText(basicInfo.check_points)],
  ];
  return rows.filter(([, value]) => Boolean(value));
}

function manualReviewTaskRecords(source: CaseAgentSource): Record<string, unknown>[] {
  const records: Record<string, unknown>[] = [];
  for (const candidate of sourceStructuredValueCandidates(source)) {
    const normalized = parseJsonLikeValue(candidate);
    if (Array.isArray(normalized)) {
      records.push(...normalized.filter(isManualReviewTaskRecord));
      continue;
    }
    if (!isPlainRecord(normalized)) continue;
    const tasks = normalized.risk_related_tasks ?? normalized.review_tasks ?? normalized.tasks;
    if (Array.isArray(tasks)) {
      records.push(...tasks.filter(isManualReviewTaskRecord));
      continue;
    }
    if (isManualReviewTaskRecord(normalized)) records.push(normalized);
  }
  return records;
}

function isManualReviewTaskRecord(value: unknown): value is Record<string, unknown> {
  if (!isPlainRecord(value)) return false;
  return Boolean(recordText(value, ['task_name', 'action', 'reason', 'task_status', 'status']));
}

function manualReviewTaskDisplayRows(source: CaseAgentSource): Array<[string, string]> {
  const tasks = manualReviewTaskRecords(source).slice(0, 3);
  if (!tasks.length) return [];
  return tasks.flatMap((task, index) => {
    const suffix = tasks.length > 1 ? ` ${index + 1}` : '';
    return [
      [`核验事项${suffix}`, recordText(task, ['task_name', 'name', 'title'])],
      [`处理建议${suffix}`, recordText(task, ['action', 'suggestion'])],
      [`触发原因${suffix}`, recordText(task, ['reason', 'summary'])],
      [`当前状态${suffix}`, recordText(task, ['task_status', 'status'])],
    ].filter(([, value]) => Boolean(value)) as Array<[string, string]>;
  });
}

function ruleIdFromSourceRef(sourceRef: string): string {
  const [, ruleId] = sourceRef.split(':');
  return ruleId ?? '';
}

function ruleDisplayRows(source: CaseAgentSource): Array<[string, string]> {
  if (source.source_type !== 'rule' && !source.source_ref.startsWith('rule:')) return [];
  const rows: Array<[string, string]> = [
    ['规则编号', ruleIdFromSourceRef(source.source_ref)],
    ['规则名称', getSourceDetailField(source, ['规则名称'])],
    ['核验状态', getSourceDetailField(source, ['核验状态', '规则状态'])],
    ['关注等级', getSourceDetailField(source, ['关注等级'])],
    ['核验说明', getSourceDetailField(source, ['核验说明', '判断口径'])],
  ];
  return rows.filter(([, value]) => Boolean(value));
}

const HIDDEN_BUSINESS_DETAIL_LABELS = new Set([
  '依据类型',
  '业务分区',
  '来源引用',
  '数据版本',
  '请求字段',
  '原始元数据',
  '数据快照',
  '具体数值',
]);

function detailFieldDisplayRows(source: CaseAgentSource): Array<[string, string]> {
  return source.detail.fields
    .filter((field) => !HIDDEN_BUSINESS_DETAIL_LABELS.has(field.label))
    .map((field) => [field.label, formatSourceFieldValue(field.value)] as [string, string])
    .filter(([, value]) => Boolean(value));
}

const GENERIC_SOURCE_FIELD_LABELS: Record<string, string> = {
  action: '核验动作',
  reason: '触发原因',
  task_name: '核验事项',
  task_status: '状态',
  status: '状态',
  count: '数量',
  total: '合计',
  amount: '金额',
  level: '等级',
  score: '评分',
  summary: '摘要',
};

const INTERNAL_SOURCE_FIELD_KEYS = new Set([
  'id',
  'source_ref',
  'source_refs',
  'material_id',
  'document_id',
  'artifact_ref',
  'payload_hash',
  'metadata',
  'meta',
]);

function readableFieldLabel(key: string): string {
  const lastSegment = key.split('.').at(-1) ?? key;
  return GENERIC_SOURCE_FIELD_LABELS[lastSegment] ?? businessText(lastSegment);
}

function collectReadableBusinessValues(
  value: unknown,
  prefix = '',
  output: string[] = [],
): string[] {
  if (output.length >= 10) return output;
  if (Array.isArray(value)) {
    value.slice(0, 4).forEach((item, index) => {
      if (output.length >= 10) return;
      const text = sourceScalarText(item);
      if (text && prefix) output.push(`${readableFieldLabel(prefix)} ${index + 1}：${text}`);
      else if (isPlainRecord(item)) collectReadableBusinessValues(item, prefix, output);
    });
    return output;
  }
  if (!isPlainRecord(value)) return output;
  for (const [key, item] of Object.entries(value)) {
    if (output.length >= 10) break;
    if (INTERNAL_SOURCE_FIELD_KEYS.has(key)) continue;
    const label = prefix ? `${prefix}.${key}` : key;
    const text = sourceScalarText(item);
    if (text) {
      output.push(`${readableFieldLabel(label)}：${text}`);
      continue;
    }
    if (isPlainRecord(item) || Array.isArray(item)) {
      collectReadableBusinessValues(item, label, output);
    }
  }
  return output;
}

function clipSourceDetailText(value: string, limit = 360): string {
  const text = value.replace(/\s+/g, ' ').trim();
  if (text.length <= limit) return text;
  return `${text.slice(0, limit)}...`;
}

function caseBasicDetailRows(source: CaseAgentSource): Array<[string, string]> {
  const data = sourcePayloadRecord(source) ?? {};
  const rows: Array<[string, string]> = [
    ['案件编号', recordText(data, ['case_number'])],
    ['案件类型', recordText(data, ['case_type'])],
    ['审核状态', recordText(data, ['review_status'])],
    ['脱敏申报人编号', recordText(data, ['claimant_ref'])],
    ['参保地', recordText(data, ['insured_region'])],
    ['就医地', recordText(data, ['treatment_region'])],
    ['就诊类型', recordText(data, ['visit_type'])],
    ['报销方式', recordText(data, ['claim_mode'])],
    ['是否直接结算', sourceScalarText(data.direct_settlement)],
    ['备案状态', recordText(data, ['filing_status'])],
    ['诊断', recordText(data, ['diagnosis'])],
    ['申报金额', sourceScalarText(data.claim_amount)],
    ['案件摘要', clipSourceDetailText(recordText(data, ['claim_summary']), 420)],
  ];
  return rows.filter(([, value]) => Boolean(value));
}

function claimantProfileDetailRows(source: CaseAgentSource): Array<[string, string]> {
  const data = sourcePayloadRecord(source) ?? {};
  const rows: Array<[string, string]> = [
    ['脱敏申报人编号', recordText(data, ['claimant_code'])],
    ['性别', recordText(data, ['gender'])],
    ['年龄段', recordText(data, ['age_group'])],
    ['参保险种', recordText(data, ['insurance_type'])],
    ['慢病标签', listText(data.chronic_condition_tags, 6)],
    ['过敏史', recordText(data, ['allergy_history'])],
    ['人群标签', listText(data.patient_group_tags, 6)],
  ];
  return rows.filter(([, value]) => Boolean(value));
}

function materialOverviewDetailRows(source: CaseAgentSource): Array<[string, string]> {
  const data = sourcePayloadRecord(source) ?? {};
  const groupSummary = recordListSummary(data.groups, (group) => {
    const title = recordText(group, ['title', 'category_id']);
    const count = sourceScalarText(group.count);
    return title && count ? `${title} ${count} 份` : title;
  });
  const materialSummary = recordArray(data.groups).flatMap((group) =>
    recordArray(group.materials).map((material) => {
      const name = recordText(material, ['name']);
      const occurredAt = recordText(material, ['occurred_at']);
      const status = recordText(material, ['status']);
      return [name, occurredAt, status].filter(Boolean).join('，');
    }),
  ).filter(Boolean).slice(0, 5).join('；');
  const rows: Array<[string, string]> = [
    ['材料总数', sourceScalarText(data.material_total)],
    ['材料分组', groupSummary],
    ['材料清单', materialSummary],
  ];
  return rows.filter(([, value]) => Boolean(value));
}

function statisticsReportDetailRows(source: CaseAgentSource): Array<[string, string]> {
  const data = sourcePayloadRecord(source) ?? {};
  const groups = nestedRecord(data, 'groups');
  const groupSummary = Object.entries(groups)
    .map(([key, value]) => {
      const count = Array.isArray(value) ? value.length : 0;
      return count ? `${businessText(key)}：${count} 项` : '';
    })
    .filter(Boolean)
    .slice(0, 6)
    .join('；');
  const rows: Array<[string, string]> = [
    ['统计指标数', sourceScalarText(data.metric_count)],
    ['基线状态', recordText(data, ['metric_baseline_status'])],
    ['统计分组', groupSummary],
  ];
  return rows.filter(([, value]) => Boolean(value));
}

function riskComponentSummary(value: unknown): string {
  return recordListSummary(value, (component) => {
    const label = recordText(component, ['label', 'key']);
    const score = sourceScalarText(component.score);
    const maxScore = sourceScalarText(component.max_score);
    const summary = recordText(component, ['summary']);
    const scoreText = score && maxScore ? `${score}/${maxScore}` : score;
    return [label, scoreText, summary].filter(Boolean).join('，');
  });
}

function riskScoreDetailRows(source: CaseAgentSource): Array<[string, string]> {
  const data = sourcePayloadRecord(source) ?? {};
  const modelWarning = nestedRecord(data, 'model_warning');
  const signal = nestedRecord(modelWarning, 'signal');
  const probability = numberPercentText(signal.probability);
  const rows: Array<[string, string]> = [
    ['综合风险提示强度', recordText(data, ['overall_strength'])],
    ['风险等级', recordText(data, ['risk_level'])],
    [
      '模型识别预警',
      [
        recordText(signal, ['label', 'result']),
        probability ? `预警概率 ${probability}` : '',
        recordText(signal, ['reason']),
      ].filter(Boolean).join('，'),
    ],
    ['主要风险构成', riskComponentSummary(data.components)],
  ];
  return rows.filter(([, value]) => Boolean(value));
}

function ruleRecordMatchesSource(rule: Record<string, unknown>, source: CaseAgentSource): boolean {
  const sourceRuleId = ruleIdFromSourceRef(source.source_ref);
  const ruleId = recordText(rule, ['rule_id']);
  if (sourceRuleId && sourceRuleId === ruleId) return true;
  const basisRefs = recordArray(rule.basis)
    .map((item) => recordText(item, ['source_ref']))
    .filter(Boolean);
  return basisRefs.some((ref) => ref === source.source_ref || source.source_ref.includes(ref));
}

function ruleRecordFromSource(source: CaseAgentSource): Record<string, unknown> | null {
  const data = sourcePayloadRecord(source) ?? {};
  const rules = recordArray(data.rules);
  if (!rules.length) return null;
  return rules.find((rule) => ruleRecordMatchesSource(rule, source)) ?? rules[0] ?? null;
}

function ruleVerificationDetailRows(source: CaseAgentSource): Array<[string, string]> {
  const rule = ruleRecordFromSource(source);
  if (!rule) return ruleDisplayRows(source);
  const result = nestedRecord(rule, 'verification_result');
  const rows: Array<[string, string]> = [
    ['规则编号', recordText(rule, ['rule_id']) || ruleIdFromSourceRef(source.source_ref)],
    ['规则名称', recordText(rule, ['rule_name'])],
    ['命中状态', recordText(rule, ['rule_status'])],
    ['关注等级', recordText(rule, ['attention_level'])],
    ['核验说明', recordText(rule, ['verification_description'])],
    ['当前值', recordText(result, ['current_value'])],
    ['阈值', recordText(result, ['threshold'])],
    ['建议动作', recordText(result, ['action'])],
  ];
  return rows.filter(([, value]) => Boolean(value));
}

function evidencePackageDetailRows(source: CaseAgentSource): Array<[string, string]> {
  const data = sourcePayloadRecord(source) ?? {};
  const summary = nestedRecord(data, 'base_summary');
  const clueSummary = recordListSummary(data.discovered_clues, (clue) => {
    const name = recordText(clue, ['name']);
    const status = recordText(clue, ['status']);
    return [name, status].filter(Boolean).join('，');
  });
  const rows: Array<[string, string]> = [
    ['风险摘要', recordText(summary, ['risk_summary'])],
    ['模型提示', recordText(summary, ['model_warning'])],
    ['命中规则数', sourceScalarText(summary.rule_hit_count)],
    ['材料提示', recordText(summary, ['material_prompt'])],
    ['发现线索', clueSummary],
  ];
  return rows.filter(([, value]) => Boolean(value));
}

function caseJudgementDetailRows(source: CaseAgentSource): Array<[string, string]> {
  const data = sourcePayloadRecord(source) ?? {};
  const clueReview = nestedRecord(data, 'clue_review');
  const needsReview = recordListSummary(clueReview.needs_review, (item) =>
    compactRecordFields(item, ['statement', 'title', 'reason', 'status']),
  );
  const supported = recordListSummary(clueReview.supported, (item) =>
    compactRecordFields(item, ['statement', 'title', 'status']),
  );
  const rows: Array<[string, string]> = [
    ['研判摘要', clipSourceDetailText(recordText(data, ['review_note']), 420)],
    ['需关注线索', needsReview],
    ['已支撑线索', supported],
  ];
  return rows.filter(([, value]) => Boolean(value));
}

function verificationItemsDetailRows(source: CaseAgentSource): Array<[string, string]> {
  return manualReviewTaskDisplayRows(source);
}

function sectionSpecificBusinessRows(source: CaseAgentSource): Array<[string, string]> {
  switch (sourceSectionKey(source)) {
    case 'case_basic_info':
      return caseBasicDetailRows(source);
    case 'claimant_profile':
      return claimantProfileDetailRows(source);
    case 'material_overview':
      return materialOverviewDetailRows(source);
    case 'medical_materials':
    case 'prescription_materials':
    case 'settlement_materials':
      return materialDisplayRows(source);
    case 'statistics_report':
      return statisticsReportDetailRows(source);
    case 'risk_score':
      return riskScoreDetailRows(source);
    case 'rule_verification':
      return ruleVerificationDetailRows(source);
    case 'evidence_package':
      return evidencePackageDetailRows(source);
    case 'case_judgement':
      return caseJudgementDetailRows(source);
    case 'verification_items':
      return verificationItemsDetailRows(source);
    default:
      return [];
  }
}

function businessSourceValues(source: CaseAgentSource): string {
  const sectionRows = sectionSpecificBusinessRows(source);
  if (sectionRows.length) return rowsToDetailText(sectionRows);
  const detailRows = detailFieldDisplayRows(source);
  if (detailRows.length) {
    return rowsToDetailText(detailRows);
  }
  for (const candidate of sourceStructuredValueCandidates(source)) {
    const values = collectReadableBusinessValues(candidate);
    if (values.length) return values.join('\n');
  }
  const snapshot = getSourceDetailField(source, ['数据快照', '具体数值']);
  if (snapshot && !isJsonLikeText(snapshot)) return clipSourceDetailText(snapshot);
  return '当前来源已结构化，但没有可展示的业务字段';
}

function businessSourceModule(source: CaseAgentSource): string {
  return sourceSectionTitle(source);
}

function businessSourceDetailFields(source: CaseAgentSource): SourceDetailDisplayField[] {
  return [
    { label: '业务模块', value: businessSourceModule(source), tone: 'value' },
    { label: '具体数值', value: businessSourceValues(source), tone: 'value' },
  ];
}

function getSourceDetailDisplayFields(source: CaseAgentSource): SourceDetailDisplayField[] {
  if (isPolicySource(source)) return policySourceDetailFields(source);
  return businessSourceDetailFields(source);
}

function sourceDetailTargetStage(source: CaseAgentSource): WorkflowStepKey | null {
  if (isPolicySource(source)) return null;
  const sectionKey = sourceSectionKey(source);
  if (sectionKey && SOURCE_STAGE_BY_SECTION[sectionKey]) {
    return SOURCE_STAGE_BY_SECTION[sectionKey];
  }
  const module = businessSourceModule(source);
  if (
    source.source_type === 'rule' ||
    source.source_ref.startsWith('rule:') ||
    module.includes('规则')
  ) {
    return 'rule_check';
  }
  if (
    source.source_type.includes('risk') ||
    source.source_type.includes('model') ||
    source.source_type.includes('fraud') ||
    source.source_ref.includes('risk') ||
    source.source_ref.includes('model')
  ) {
    return 'risk_screening';
  }
  if (
    module.includes('核验事项') ||
    source.title.includes('核验事项')
  ) {
    return 'evidence_package';
  }
  return 'fact_base';
}

function formatSessionTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  const now = new Date();
  const dayStart = new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const dayDiff = Math.round((todayStart - dayStart) / DAY_MS);
  const time = date.toLocaleTimeString('zh-CN', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });
  if (dayDiff === 0) return `今天 ${time}`;
  if (dayDiff === 1) return `昨天 ${time}`;
  if (date.getFullYear() === now.getFullYear()) {
    return `${date.getMonth() + 1}月${date.getDate()}日 ${time}`;
  }
  return `${date.getFullYear()}年${date.getMonth() + 1}月${date.getDate()}日`;
}

function formatAnswerTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  const now = new Date();
  const dayStart = new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const dayDiff = Math.round((todayStart - dayStart) / DAY_MS);
  const time = date.toLocaleTimeString('zh-CN', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });
  if (dayDiff === 0) return time;
  if (dayDiff === 1) return `昨天 ${time}`;
  if (date.getFullYear() === now.getFullYear()) {
    return `${date.getMonth() + 1}月${date.getDate()}日 ${time}`;
  }
  return `${date.getFullYear()}年${date.getMonth() + 1}月${date.getDate()}日 ${time}`;
}

function sortCaseAgentSessions(items: CaseAgentSession[]): CaseAgentSession[] {
  return [...items].sort((left, right) => {
    const updatedDiff = new Date(right.updated_at).getTime() - new Date(left.updated_at).getTime();
    return updatedDiff || right.session_id.localeCompare(left.session_id);
  });
}

function NewSessionIcon() {
  return (
    <svg
      aria-hidden="true"
      focusable="false"
      viewBox="0 0 1024 1024"
      className="case-agent-session-svg"
    >
      <path
        d="M512 0c280.448 0 512 218.24 512 483.392a46.08 46.08 0 0 1-91.904 0c0-213.888-190.208-391.488-420.096-391.488-225.664 0-411.52 164.544-419.776 369.92l-0.32 14.208 0.32 16.896c3.584 81.728 34.752 157.248 92.352 224.32l12.672 14.208 7.488 7.68a46.08 46.08 0 0 1 12.608 36.864l-1.344 7.04-27.008 103.936c-5.248 22.4 11.456 45.824 30.144 45.632l4.48-0.512 3.2-0.896 164.352-69.76a46.08 46.08 0 0 1 19.2-3.584l6.848 0.704 15.36 2.624c33.28 5.568 53.12 7.488 79.424 7.488a45.952 45.952 0 1 1-0.064 91.904 492.8 492.8 0 0 1-79.104-6.208l-17.216-2.88-155.52 65.344-8.64 2.88c-86.656 23.68-167.808-54.464-153.6-145.28l1.856-9.408 20.544-79.168-4.48-7.68C48.96 700.672 10.24 612.416 1.92 516.736l-1.472-20.928L0 477.056C0 219.52 219.712 9.6 493.248 0.32L512 0z m249.6 515.968c25.472 0 46.08 20.544 46.08 45.952v149.76l153.664 4.032a45.952 45.952 0 1 1 0 91.904h-149.76l-3.968 153.728a46.08 46.08 0 0 1-91.904 0v-149.76l-153.792-3.968a46.08 46.08 0 0 1 0-91.904h149.76l4.032-153.792a46.08 46.08 0 0 1 38.464-45.44l7.488-0.512z"
        fill="currentColor"
      />
    </svg>
  );
}

function CopySoftIcon() {
  return (
    <svg
      aria-hidden="true"
      focusable="false"
      viewBox="0 0 1024 1024"
      className="case-agent-copy-svg"
    >
      <path
        d="M815.5648 53.3504h-384a147.8656 147.8656 0 0 0-147.5072 147.6608v90.5216H208.4352a147.7632 147.7632 0 0 0-147.6096 147.6096v384a147.8144 147.8144 0 0 0 147.6096 147.6608h384a147.8656 147.8656 0 0 0 147.6608-147.6608v-90.6752h75.6224a147.7632 147.7632 0 0 0 147.6096-147.6096v-384a147.8144 147.8144 0 0 0-147.7632-147.5072z m-137.0624 769.6384a86.3232 86.3232 0 0 1-86.2208 86.2208h-384a86.3232 86.3232 0 0 1-86.1696-86.2208v-384a86.272 86.272 0 0 1 86.3232-86.016h384a86.272 86.272 0 0 1 86.2208 86.1696z m223.232-238.1312a86.272 86.272 0 0 1-86.1696 86.1696h-75.6224V439.1424a147.7632 147.7632 0 0 0-147.6608-147.6096h-246.784V201.0112a86.3232 86.3232 0 0 1 86.2208-86.2208h384a86.3232 86.3232 0 0 1 86.1696 86.2208z"
        fill="currentColor"
      />
    </svg>
  );
}

function formatElapsed(ms: number): string {
  const seconds = Math.max(0, ms / 1000);
  return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`;
}

function answerPlainText(answer: CaseAgentAnswer): string {
  if (answer.answer_markdown) {
    return businessText(answer.answer_markdown);
  }
  return answer.content_blocks.map((block) => businessText(block.text)).join('\n');
}

type AnswerMarkdownToken =
  | { type: 'text'; value: string }
  | { type: 'citation'; value: string; label: number };

function tokenizeAnswerMarkdown(text: string): AnswerMarkdownToken[] {
  const tokens: AnswerMarkdownToken[] = [];
  const pattern = /\[(\d{1,2})\]/g;
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    const start = match.index ?? 0;
    if (start > cursor) {
      tokens.push({ type: 'text', value: text.slice(cursor, start) });
    }
    tokens.push({
      type: 'citation',
      value: match[0],
      label: Number(match[1]),
    });
    cursor = start + match[0].length;
  }
  if (cursor < text.length) {
    tokens.push({ type: 'text', value: text.slice(cursor) });
  }
  return tokens;
}

function eventTimeMs(event: CaseAgentEvent): number | null {
  const value = Date.parse(event.created_at);
  return Number.isNaN(value) ? null : value;
}

function eventPayloadString(event: CaseAgentEvent, key: string): string {
  const value = event.payload[key];
  return typeof value === 'string' ? value : '';
}

function eventPayloadLayer(event: CaseAgentEvent): string {
  return eventPayloadString(event, 'layer').toUpperCase();
}

function eventCapability(event: CaseAgentEvent): string {
  return eventPayloadString(event, 'capability') || eventPayloadString(event, 'tool_name');
}

function capabilityProgressLayer(capability: string): AgentProgressLayerKey | null {
  if (L3_CAPABILITY_NAMES.has(capability)) return 'l3';
  if (L2_CAPABILITY_NAMES.has(capability)) return 'l2';
  if (capability.startsWith('query_')) return 'l1';
  return null;
}

function layerProgressLayer(layer: string): AgentProgressLayerKey | null {
  if (layer === 'L1') return 'l1';
  if (layer === 'L2') return 'l2';
  if (layer === 'L3') return 'l3';
  return null;
}

function eventIsLayerStep(event: CaseAgentEvent, layer: AgentProgressLayerKey): boolean {
  if (event.event_type !== 'dag_step_ready') return false;
  return layerProgressLayer(eventPayloadLayer(event)) === layer;
}

function eventIsToolForLayer(event: CaseAgentEvent, layer: AgentProgressLayerKey): boolean {
  if (!TOOL_EVENT_TYPES.has(event.event_type)) return false;
  return capabilityProgressLayer(eventCapability(event)) === layer;
}

function eventIsExpertCapabilityFinished(event: CaseAgentEvent): boolean {
  if (!['capability_complete', 'capability_unavailable', 'capability_failed'].includes(event.event_type)) {
    return false;
  }
  const capability = eventCapability(event);
  if (capability) return capabilityProgressLayer(capability) === 'l3';
  return eventPayloadString(event, 'expert_task_type') === 'policy_analysis';
}

function substepMatchesEvent(substep: AgentProgressSubstepDefinition, event: CaseAgentEvent): boolean {
  if (substep.eventTypes?.includes(event.event_type)) return true;
  return substep.match?.(event) ?? false;
}

function buildProgressSteps(
  events: CaseAgentEvent[],
  elapsedMs: number,
  progressStartedAt: number | null,
): AgentProgressStepView[] {
  const seen = new Set<string>();
  const visits: AgentProgressNodeVisit[] = [];
  const fallbackStart = progressStartedAt ?? Date.now();
  const orderedEvents = events
    .map((event, index) => ({
      event,
      index,
      time: eventTimeMs(event) ?? fallbackStart + index,
    }))
    .sort((left, right) => {
      if (left.time !== right.time) return left.time - right.time;
      return left.index - right.index;
    });

  for (const { event, time } of orderedEvents) {
    AGENT_PROGRESS_NODES.forEach((definition) => {
      if (seen.has(definition.key) || !substepMatchesEvent(definition, event)) return;
      seen.add(definition.key);
      visits.push({ definition, time });
    });
  }

  if (!visits.length) return [];

  let terminalEvent: CaseAgentEvent | undefined;
  for (let index = orderedEvents.length - 1; index >= 0; index -= 1) {
    const event = orderedEvents[index]?.event;
    if (event && ['complete', 'degraded', 'failed', 'waiting_for_user'].includes(event.event_type)) {
      terminalEvent = event;
      break;
    }
  }
  const hasTerminalEvent = terminalEvent !== undefined;
  const failed = terminalEvent?.event_type === 'failed';
  const activeVisitKey = hasTerminalEvent ? '' : visits[visits.length - 1]?.definition.key;
  const currentTime = progressStartedAt === null ? Date.now() : progressStartedAt + elapsedMs;

  const grouped = new Map<AgentProgressStepKey, AgentProgressSubstepView[]>();
  visits.forEach((visit, index) => {
    const isLastVisit = index === visits.length - 1;
    const status: AgentProgressStatus = failed && isLastVisit
      ? 'failed'
      : visit.definition.key === activeVisitKey
        ? 'active'
        : 'completed';
    const groupSubsteps = grouped.get(visit.definition.group) ?? [];
    groupSubsteps.push({
      key: visit.definition.key,
      label: visit.definition.label,
      status,
      time: visit.time,
    });
    grouped.set(visit.definition.group, groupSubsteps);
  });

  const renderedGroups = AGENT_PROGRESS_GROUPS
    .map((group) => ({
      ...group,
      substeps: grouped.get(group.key) ?? [],
    }))
    .filter((group) => group.substeps.length > 0);

  return renderedGroups.map((group, index) => {
    const groupStart = group.substeps[0]?.time;
    const nextGroupStart = renderedGroups[index + 1]?.substeps[0]?.time;
    const groupStatus: AgentProgressStatus = group.substeps.some((substep) => substep.status === 'failed')
      ? 'failed'
      : group.substeps.some((substep) => substep.status === 'active')
        ? 'active'
        : 'completed';
    const groupEnd = groupStatus === 'active'
      ? currentTime
      : nextGroupStart ?? group.substeps[group.substeps.length - 1]?.time;
    return {
      ...group,
      status: groupStatus,
      elapsedMs: groupStart !== undefined && groupEnd !== undefined
        ? Math.max(0, groupEnd - groupStart)
        : undefined,
    };
  });
}

function progressStatusText(step: AgentProgressStepView): string {
  if (step.status === 'completed') {
    return step.elapsedMs === undefined ? '已完成' : `已完成 ${formatElapsed(step.elapsedMs)}`;
  }
  if (step.status === 'active') {
    return step.elapsedMs === undefined ? '进行中' : `进行中 ${formatElapsed(step.elapsedMs)}`;
  }
  if (step.status === 'failed') return '未完成';
  return '已完成';
}

function progressSubstepStatusText(status: AgentProgressStatus): string {
  if (status === 'completed') return '已完成';
  if (status === 'active') return '进行中';
  if (status === 'failed') return '未完成';
  return '已完成';
}

function AgentProgressBubble({
  events,
  elapsedMs,
  progressStartedAt,
}: {
  events: CaseAgentEvent[];
  elapsedMs: number;
  progressStartedAt: number | null;
}) {
  const steps = useMemo(
    () => buildProgressSteps(events, elapsedMs, progressStartedAt),
    [elapsedMs, events, progressStartedAt],
  );
  return (
    <div className="case-agent-progress-bubble">
      <div className="case-agent-progress-head">
        <span>正在处理案件问题</span>
        <span className="case-agent-progress-total">{formatElapsed(elapsedMs)}</span>
      </div>
      <div className="case-agent-progress-flow" aria-label="Agent 处理流程">
        {steps.map((step) => (
          <div
            className={`case-agent-progress-step is-${step.status}`}
            key={step.key}
          >
            <span className="case-agent-progress-node" aria-hidden="true" />
            <span className="case-agent-progress-step-body">
              <span className="case-agent-progress-step-label">{step.label}</span>
              <span className="case-agent-progress-substeps">
                {step.substeps.map((substep) => (
                  <span
                    className={`case-agent-progress-substep is-${substep.status}`}
                    key={substep.key}
                  >
                    <span className="case-agent-progress-substep-text">{substep.label}</span>
                    <span className="case-agent-progress-substep-status">
                      {progressSubstepStatusText(substep.status)}
                    </span>
                  </span>
                ))}
              </span>
            </span>
            <span className="case-agent-progress-step-status">
              {progressStatusText(step)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function AgentAnswerContent({
  answer,
  onOpenSource,
}: {
  answer: CaseAgentAnswer;
  onOpenSource: (source: CaseAgentSource) => void;
}) {
  const sourceMap = useMemo(
    () => new Map(answer.sources.map((source) => [source.source_ref, source])),
    [answer.sources],
  );
  const citationNumberMap = useMemo(() => {
    const refs: string[] = [];
    for (const block of answer.content_blocks) {
      for (const ref of block.source_refs) {
        if (sourceMap.has(ref) && !refs.includes(ref)) {
          refs.push(ref);
        }
      }
    }
    if (!refs.length) {
      for (const source of answer.sources) {
        refs.push(source.source_ref);
      }
    }
    return new Map(refs.map((ref, index) => [ref, index + 1]));
  }, [answer.content_blocks, answer.sources, sourceMap]);
  const citationSourceMap = useMemo(() => {
    const mapped = new Map<number, CaseAgentSource>();
    for (const citation of answer.citations ?? []) {
      const sourceRef = citation.source_refs.find((ref) => sourceMap.has(ref));
      if (!sourceRef) continue;
      const source = sourceMap.get(sourceRef);
      if (source) mapped.set(citation.label, source);
    }
    if (!mapped.size) {
      for (const [ref, label] of citationNumberMap.entries()) {
        const source = sourceMap.get(ref);
        if (source) mapped.set(label, source);
      }
    }
    return mapped;
  }, [answer.citations, citationNumberMap, sourceMap]);
  const markdownParagraphs = useMemo(
    () => (answer.answer_markdown ?? '')
      .split(/\n+/)
      .map((paragraph) => businessText(paragraph).trim())
      .filter(Boolean),
    [answer.answer_markdown],
  );
  return (
    <div className={`case-agent-answer is-${answer.display_mode}`}>
      <div className="case-agent-answer-blocks">
        {markdownParagraphs.length
          ? (
              <>
                {markdownParagraphs.map((paragraph, index) => (
                  <Typography.Paragraph
                    className="case-agent-answer-text"
                    key={`${index}-${paragraph}`}
                  >
                    {tokenizeAnswerMarkdown(paragraph).map((token, tokenIndex) => {
                      if (token.type === 'text') {
                        return <span key={`${index}-${tokenIndex}`}>{token.value}</span>;
                      }
                      const source = citationSourceMap.get(token.label);
                      if (!source) {
                        return <span key={`${index}-${tokenIndex}`}>{token.value}</span>;
                      }
                      return (
                        <button
                          type="button"
                          className="case-agent-inline-source"
                          key={`${index}-${tokenIndex}-${token.label}`}
                          onClick={() => onOpenSource(source)}
                          aria-label={`查看引用 ${token.label}：${businessText(source.title)}`}
                          title={businessText(source.title)}
                        >
                          {token.label}
                        </button>
                      );
                    })}
                  </Typography.Paragraph>
                ))}
                {answer.content_blocks.slice(1).map((block, index) => (
                  <Typography.Paragraph
                    className="case-agent-answer-text"
                    key={`extra-${index}-${block.text}`}
                  >
                    <span>{businessText(block.text)}</span>
                    {block.source_refs.map((ref) => {
                      const source = sourceMap.get(ref);
                      if (!source) return null;
                      return (
                        <button
                          type="button"
                          className="case-agent-inline-source"
                          key={`extra-${index}-${ref}`}
                          onClick={() => onOpenSource(source)}
                          aria-label={`查看引用 ${citationNumberMap.get(ref) ?? 1}：${businessText(source.title)}`}
                          title={businessText(source.title)}
                        >
                          {citationNumberMap.get(ref) ?? 1}
                        </button>
                      );
                    })}
                  </Typography.Paragraph>
                ))}
              </>
            )
          : answer.content_blocks.map((block, index) => (
              <Typography.Paragraph
                className="case-agent-answer-text"
                key={`${index}-${block.text}`}
              >
                <span>{businessText(block.text)}</span>
                {block.source_refs.map((ref) => {
                  const source = sourceMap.get(ref);
                  if (!source) return null;
                  return (
                    <button
                      type="button"
                      className="case-agent-inline-source"
                      key={`${index}-${ref}`}
                      onClick={() => onOpenSource(source)}
                      aria-label={`查看引用 ${citationNumberMap.get(ref) ?? 1}：${businessText(source.title)}`}
                      title={businessText(source.title)}
                    >
                      {citationNumberMap.get(ref) ?? 1}
                    </button>
                  );
                })}
              </Typography.Paragraph>
            ))}
      </div>

      {answer.fallback_notice && (
        <Typography.Text className="case-agent-fallback-notice">
          {businessText(answer.fallback_notice)}
        </Typography.Text>
      )}
    </div>
  );
}

interface CaseAgentComposerProps {
  activeSessionId: string;
  activeStage: WorkflowStepKey;
  answers: PresetAnswer[];
  caseId: string;
  caseType: string;
  hasUserMessage: boolean;
  draft: string;
  running: boolean;
  stopping: boolean;
  onDraftChange: (value: string) => void;
  onSendQuestion: (question: string) => Promise<boolean>;
  onStop: () => void;
  onUnavailableAttachment: () => void;
}

const CaseAgentComposer = memo(function CaseAgentComposer({
  activeSessionId,
  activeStage,
  answers,
  caseId,
  caseType,
  hasUserMessage,
  draft,
  running,
  stopping,
  onDraftChange,
  onSendQuestion,
  onStop,
  onUnavailableAttachment,
}: CaseAgentComposerProps) {
  const taskContext = useMemo(() => ({
    case_id: caseId,
    case_type: caseType,
    stage: activeStage,
  }), [activeStage, caseId, caseType]);
  const shouldShowPrompts = Boolean(activeSessionId)
    && !hasUserMessage
    && !running
    && !draft.trim();

  const submit = useCallback(async () => {
    const content = draft.trim();
    if (!content || !activeSessionId || running) return;
    const accepted = await onSendQuestion(content);
    if (accepted) onDraftChange('');
  }, [activeSessionId, draft, onDraftChange, onSendQuestion, running]);

  return (
    <div className="case-agent-input-area">
      {shouldShowPrompts && (
        <Prompts
          className="case-agent-prompts"
          aria-label="常用案件问题"
          fadeIn={false}
          items={answers.map((answer) => ({
            key: answer.key,
            label: answer.question,
          }))}
          onItemClick={({ data: prompt }) => {
            const answer = answers.find((item) => item.key === prompt.key);
            if (answer) void onSendQuestion(answer.question);
          }}
        />
      )}
      <div className="case-agent-composer">
        <div className="case-agent-composer-input-row">
          <Input.TextArea
            value={draft}
            onChange={(event) => onDraftChange(event.target.value)}
            autoSize={{ minRows: 2, maxRows: 8 }}
            placeholder="请输入案件核验问题"
            disabled={!activeSessionId}
            onPressEnter={(event) => {
              if (event.nativeEvent.isComposing || event.shiftKey) return;
              event.preventDefault();
              if (!running) void submit();
            }}
          />
        </div>
        <div className="case-agent-composer-toolbar">
          <div className="case-agent-composer-tools">
            <Tooltip title="添加附件（未接入）">
              <Button
                type="text"
                shape="circle"
                className="case-agent-tool-button"
                icon={<PlusOutlined />}
                aria-label="添加附件"
                onClick={onUnavailableAttachment}
              />
            </Tooltip>
            <MemoryGovernancePanel taskContext={taskContext} />
          </div>
          <div className="case-agent-composer-actions">
            <Tooltip title={running ? (stopping ? '正在停止任务' : '停止当前任务') : '发送案件核验问题'}>
              <Button
                type="primary"
                shape="circle"
                danger={running}
                className={`case-agent-send-button${running ? ' is-stop' : ''}`}
                icon={running ? <StopOutlined /> : <ArrowUpOutlined />}
                loading={stopping}
                disabled={!activeSessionId || (!running && !draft.trim())}
                aria-label={running ? '停止当前任务' : '发送案件核验问题'}
                onClick={() => (running ? onStop() : void submit())}
              />
            </Tooltip>
          </div>
        </div>
      </div>
    </div>
  );
});

export default function CaseAgentPanel({
  data,
  activeStage = 'evidence_package',
  onClose,
  onStageChange,
}: CaseAgentPanelProps) {
  const answers = useMemo(
    () => (data ? buildPresetAnswers(data, activeStage) : []),
    [activeStage, data],
  );
  const caseId = data?.case.case_id ?? '';
  const [sessions, setSessions] = useState<CaseAgentSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>('');
  const [session, setSession] = useState<CaseAgentSession | null>(null);
  const [loading, setLoading] = useState(false);
  const [runStateBySessionId, setRunStateBySessionId] = useState<Record<string, SessionRunState>>({});
  const [draftBySessionId, setDraftBySessionId] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [sourceDetail, setSourceDetail] = useState<CaseAgentSource | null>(null);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [titleEditing, setTitleEditing] = useState(false);
  const [titleDraft, setTitleDraft] = useState('');
  const [renaming, setRenaming] = useState(false);
  const [deletingSessionIds, setDeletingSessionIds] = useState<Set<string>>(
    () => new Set(),
  );
  const runWatchersRef = useRef<Map<string, () => void>>(new Map());
  const reconnectTimersRef = useRef<Map<string, number>>(new Map());
  const runStateRef = useRef<Record<string, SessionRunState>>({});
  const activeSessionIdRef = useRef('');
  const startRunWatcherRef = useRef<(sessionId: string, runId: string, after?: number) => void>(() => undefined);
  const titleInputRef = useRef<InputRef>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const scrollAfterSendRef = useRef(false);

  useEffect(() => {
    activeSessionIdRef.current = activeSessionId;
  }, [activeSessionId]);

  useEffect(() => {
    runStateRef.current = runStateBySessionId;
  }, [runStateBySessionId]);

  const closeRunWatcher = useCallback((runId: string) => {
    runWatchersRef.current.get(runId)?.();
    runWatchersRef.current.delete(runId);
    const timer = reconnectTimersRef.current.get(runId);
    if (timer !== undefined) {
      window.clearTimeout(timer);
      reconnectTimersRef.current.delete(runId);
    }
  }, []);

  const closeAllRunWatchers = useCallback(() => {
    [...runWatchersRef.current.keys()].forEach(closeRunWatcher);
  }, [closeRunWatcher]);

  const scrollToLatestMessage = useCallback((behavior: ScrollBehavior = 'smooth') => {
    window.requestAnimationFrame(() => {
      const scrollBox = messagesContainerRef.current?.querySelector<HTMLElement>(
        '.ant-bubble-list-scroll-box',
      );
      if (!scrollBox) return;
      scrollBox.scrollTo({
        top: scrollBox.scrollHeight,
        behavior,
      });
    });
  }, []);

  const refreshSession = useCallback(async (sessionId: string) => {
    const next = await fetchCaseAgentSession(sessionId);
    if (activeSessionIdRef.current === sessionId) setSession(next);
    setSessions((previous) => sortCaseAgentSessions(
      previous.map((item) => (item.session_id === sessionId ? { ...item, ...next, messages: [] } : item)),
    ));
    return next;
  }, []);

  const reconcileTerminalRun = useCallback(async (sessionId: string, runId: string) => {
    closeRunWatcher(runId);
    try {
      const run = await fetchCaseAgentRun(runId);
      setRunStateBySessionId((previous) => ({
        ...previous,
        [sessionId]: {
          ...(previous[sessionId] ?? {
            runId,
            events: [],
            startedAt: Date.now(),
            lastSequence: 0,
          }),
          runId,
          status: run.cancel_requested_at && ACTIVE_RUN_STATUSES.has(run.status)
            ? 'cancelling'
            : run.status,
          stopping: Boolean(run.cancel_requested_at && ACTIVE_RUN_STATUSES.has(run.status)),
          pendingMessage: undefined,
        },
      }));
      let refreshed = await refreshSession(sessionId);
      if (
        activeSessionIdRef.current === sessionId
        && document.visibilityState === 'visible'
        && refreshed.has_unread_activity
        && refreshed.latest_run?.run_id === runId
        && refreshed.latest_run.completed_at
      ) {
        refreshed = await markCaseAgentSessionRead(sessionId, runId);
        setSession(refreshed);
        setSessions((previous) => sortCaseAgentSessions(
          previous.map((item) => (item.session_id === sessionId ? { ...item, ...refreshed, messages: [] } : item)),
        ));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '刷新任务状态失败');
    }
  }, [closeRunWatcher, refreshSession]);

  const startRunWatcher = useCallback((sessionId: string, runId: string, after = 0) => {
    if (runWatchersRef.current.has(runId)) return;
    let lastSequence = after;
    let closed = false;
    const closeSource = subscribeCaseAgentEvents(
      runId,
      (event) => {
        lastSequence = Math.max(lastSequence, event.sequence);
        setRunStateBySessionId((previous) => {
          const current = previous[sessionId];
          const events = current?.events.some((item) => item.sequence === event.sequence)
            ? current.events
            : [...(current?.events ?? []), event];
          const terminalStatus: CaseAgentRun['status'] | undefined = event.event_type === 'complete'
            ? 'completed'
            : event.event_type === 'waiting_for_user'
              ? 'waiting_for_user'
              : event.event_type === 'degraded'
                ? 'degraded'
                : event.event_type === 'failed'
                  ? 'failed'
                  : event.event_type === 'cancelled'
                    ? 'cancelled'
                    : undefined;
          return {
            ...previous,
            [sessionId]: {
              ...(current ?? {
                runId,
                status: 'running',
                startedAt: Date.now(),
                stopping: false,
              }),
              runId,
              status: event.event_type === 'cancel_requested'
                ? 'cancelling'
                : terminalStatus ?? current?.status ?? 'running',
              stopping: event.event_type === 'cancel_requested'
                || (current?.stopping ?? false),
              events,
              lastSequence,
            },
          };
        });
        if (['complete', 'waiting_for_user', 'degraded', 'failed', 'cancelled'].includes(event.event_type)) {
          void reconcileTerminalRun(sessionId, runId);
        }
      },
      () => {
        if (closed) return;
        closeRunWatcher(runId);
        void fetchCaseAgentRun(runId)
          .then((run) => {
            if (!ACTIVE_RUN_STATUSES.has(run.status)) {
              void reconcileTerminalRun(sessionId, runId);
              return;
            }
            setRunStateBySessionId((previous) => ({
              ...previous,
              [sessionId]: {
                ...(previous[sessionId] ?? {
                  runId,
                  events: [],
                  startedAt: Date.now(),
                  lastSequence,
                }),
                runId,
                status: run.cancel_requested_at ? 'cancelling' : run.status,
                stopping: Boolean(run.cancel_requested_at),
              },
            }));
            const timer = window.setTimeout(() => {
              reconnectTimersRef.current.delete(runId);
              startRunWatcherRef.current(sessionId, runId, lastSequence);
            }, 1200);
            reconnectTimersRef.current.set(runId, timer);
          })
          .catch(() => {
            const timer = window.setTimeout(() => {
              reconnectTimersRef.current.delete(runId);
              startRunWatcherRef.current(sessionId, runId, lastSequence);
            }, 2400);
            reconnectTimersRef.current.set(runId, timer);
          });
      },
      after,
    );
    runWatchersRef.current.set(runId, () => {
      closed = true;
      closeSource();
    });
  }, [closeRunWatcher, reconcileTerminalRun]);

  useEffect(() => {
    startRunWatcherRef.current = startRunWatcher;
  }, [startRunWatcher]);

  const hydrateRunStates = useCallback((items: CaseAgentSession[]) => {
    setRunStateBySessionId((previous) => {
      const next = { ...previous };
      items.forEach((item) => {
        const latest = item.latest_run;
        if (!latest || !isActiveRunStatus(latest.effective_status)) return;
        next[item.session_id] = {
          ...(next[item.session_id] ?? {
            events: [],
            lastSequence: 0,
            startedAt: latest.started_at ? Date.parse(latest.started_at) : Date.parse(latest.created_at),
          }),
          runId: latest.run_id,
          status: latest.effective_status,
          stopping: latest.effective_status === 'cancelling',
        };
      });
      return next;
    });
    items.forEach((item) => {
      const latest = item.latest_run;
      if (latest && isActiveRunStatus(latest.effective_status)) {
        startRunWatcher(item.session_id, latest.run_id, runStateRef.current[item.session_id]?.lastSequence ?? 0);
      }
    });
  }, [startRunWatcher]);

  const loadSessions = useCallback(async () => {
    if (!caseId) return;
    setLoading(true);
    setError(null);
    try {
      let nextSessions = sortCaseAgentSessions(await fetchCaseAgentSessions(caseId));
      if (nextSessions.length === 0) {
        const created = await createCaseAgentSession(caseId, { title: '默认会话' });
        nextSessions = [created];
      }
      setSessions(nextSessions);
      hydrateRunStates(nextSessions);
      const nextActive =
        nextSessions.find((item) => item.session_id === activeSessionIdRef.current)?.session_id
        ?? nextSessions[0]?.session_id
        ?? '';
      activeSessionIdRef.current = nextActive;
      setActiveSessionId(nextActive);
      if (nextActive) {
        await refreshSession(nextActive);
      }
    } catch (err) {
      setSessions([]);
      setSession(null);
      setActiveSessionId('');
      setError(err instanceof Error ? err.message : 'Case Agent 当前不可用');
    } finally {
      setLoading(false);
    }
  }, [caseId, hydrateRunStates, refreshSession]);

  useEffect(() => {
    closeAllRunWatchers();
    setRunStateBySessionId({});
    setDraftBySessionId({});
    loadSessions();
    return closeAllRunWatchers;
  }, [caseId, closeAllRunWatchers, loadSessions]);

  useEffect(() => {
    if (!activeSessionId) return;
    refreshSession(activeSessionId)
      .then(async (next) => {
        if (
          document.visibilityState === 'visible'
          && next.has_unread_activity
          && next.latest_run?.completed_at
        ) {
          const read = await markCaseAgentSessionRead(activeSessionId, next.latest_run.run_id);
          if (activeSessionIdRef.current === activeSessionId) setSession(read);
          setSessions((previous) => sortCaseAgentSessions(
            previous.map((item) => (item.session_id === activeSessionId ? { ...item, ...read, messages: [] } : item)),
          ));
        }
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : '加载会话失败');
      });
  }, [activeSessionId, refreshSession]);

  useEffect(() => {
    if (titleEditing) {
      window.setTimeout(() => {
        titleInputRef.current?.focus();
        titleInputRef.current?.select();
      }, 0);
    }
  }, [titleEditing]);

  useEffect(() => {
    const activeRun = runStateRef.current[activeSessionId];
    if (!activeRun || !isActiveRunStatus(activeRun.status)) {
      setElapsedMs(0);
      return undefined;
    }
    const updateElapsed = () => setElapsedMs(Date.now() - activeRun.startedAt);
    updateElapsed();
    const timer = window.setInterval(updateElapsed, 200);
    return () => window.clearInterval(timer);
  }, [activeSessionId, runStateBySessionId]);

  const markSessionDeleting = useCallback((sessionId: string, deleting: boolean) => {
    setDeletingSessionIds((previous) => {
      const next = new Set(previous);
      if (deleting) {
        next.add(sessionId);
      } else {
        next.delete(sessionId);
      }
      return next;
    });
  }, []);

  const switchSession = useCallback((sessionId: string) => {
    activeSessionIdRef.current = sessionId;
    setActiveSessionId(sessionId);
    setSession(null);
    setHistoryOpen(false);
  }, []);

  const createSession = async () => {
    if (!caseId) return;
    setLoading(true);
    setError(null);
    try {
      const created = await createCaseAgentSession(caseId, {
        title: `会话 ${sessions.length + 1}`,
      });
      setSessions((previous) => [created, ...previous]);
      setActiveSessionId(created.session_id);
      activeSessionIdRef.current = created.session_id;
      setSession(created);
      setHistoryOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : '创建会话失败');
    } finally {
      setLoading(false);
    }
  };

  const sendQuestion = async (question: string) => {
    const content = question.trim();
    const targetSessionId = activeSessionIdRef.current;
    const currentRun = runStateRef.current[targetSessionId];
    if (!content || !targetSessionId || (currentRun && isActiveRunStatus(currentRun.status))) return false;
    const optimisticStartedAt = Date.now();
    setRunStateBySessionId((previous) => ({
      ...previous,
      [targetSessionId]: {
        runId: `pending:${crypto.randomUUID()}`,
        status: 'created',
        events: [],
        pendingMessage: content,
        startedAt: optimisticStartedAt,
        stopping: false,
        lastSequence: 0,
      },
    }));
    scrollAfterSendRef.current = true;
    setError(null);
    try {
      const run = await sendCaseAgentMessage(targetSessionId, {
        content,
        active_stage: activeStage,
        client_request_id: crypto.randomUUID(),
      });
      setRunStateBySessionId((previous) => ({
        ...previous,
        [targetSessionId]: {
          ...(previous[targetSessionId] ?? {
            events: [],
            startedAt: optimisticStartedAt,
            lastSequence: 0,
          }),
          runId: run.run_id,
          status: run.cancel_requested_at ? 'cancelling' : run.status,
          stopping: Boolean(run.cancel_requested_at),
        },
      }));
      const refreshed = await refreshSession(targetSessionId);
      if (refreshed.messages.some((message) => message.message_id === run.user_message_id)) {
        setRunStateBySessionId((previous) => ({
          ...previous,
          [targetSessionId]: {
            ...previous[targetSessionId],
            pendingMessage: undefined,
          },
        }));
      }
      if (ACTIVE_RUN_STATUSES.has(run.status)) {
        startRunWatcher(targetSessionId, run.run_id);
      } else {
        void reconcileTerminalRun(targetSessionId, run.run_id);
      }
      return true;
    } catch (err) {
      setRunStateBySessionId((previous) => {
        const next = { ...previous };
        delete next[targetSessionId];
        return next;
      });
      setError(err instanceof Error ? err.message : '发送问题失败');
      fetchCaseAgentSessions(caseId)
        .then((items) => {
          const sorted = sortCaseAgentSessions(items);
          setSessions(sorted);
          hydrateRunStates(sorted);
        })
        .catch(() => undefined);
      return false;
    }
  };

  const stopCurrentRun = async () => {
    const targetSessionId = activeSessionIdRef.current;
    const current = runStateRef.current[targetSessionId];
    if (!current || !isActiveRunStatus(current.status) || current.runId.startsWith('pending:')) return;
    setRunStateBySessionId((previous) => ({
      ...previous,
      [targetSessionId]: {
        ...previous[targetSessionId],
        status: 'cancelling',
        stopping: true,
      },
    }));
    setError(null);
    try {
      const run = await cancelCaseAgentRun(current.runId);
      setRunStateBySessionId((previous) => ({
        ...previous,
        [targetSessionId]: {
          ...previous[targetSessionId],
          status: run.cancel_requested_at && ACTIVE_RUN_STATUSES.has(run.status)
            ? 'cancelling'
            : run.status,
          stopping: Boolean(run.cancel_requested_at && ACTIVE_RUN_STATUSES.has(run.status)),
        },
      }));
      if (!ACTIVE_RUN_STATUSES.has(run.status)) {
        await reconcileTerminalRun(targetSessionId, run.run_id);
      }
    } catch (err) {
      setRunStateBySessionId((previous) => ({
        ...previous,
        [targetSessionId]: {
          ...previous[targetSessionId],
          status: current.status,
          stopping: false,
        },
      }));
      setError(err instanceof Error ? err.message : '停止任务失败');
    }
  };

  const activeRunState = runStateBySessionId[activeSessionId];
  const sending = Boolean(activeRunState && isActiveRunStatus(activeRunState.status));
  const events = activeRunState?.events ?? [];
  const pendingUserMessage = activeRunState?.pendingMessage ?? null;
  const progressStartedAt = activeRunState?.startedAt ?? null;
  const activeDraft = draftBySessionId[activeSessionId] ?? '';
  const messages = session?.messages ?? [];
  const hasUserMessage = messages.some((message) => message.role === 'user');
  const hasConversationMessages = messages.length > 0 || Boolean(pendingUserMessage);
  const activeSession = sessions.find((item) => item.session_id === activeSessionId);
  const headerTitle = session?.title ?? activeSession?.title ?? '会话';
  const sourceDetailStage = sourceDetail ? sourceDetailTargetStage(sourceDetail) : null;
  const openSourceDetailStage = () => {
    if (!sourceDetailStage || !onStageChange) return;
    onStageChange(sourceDetailStage);
    setSourceDetail(null);
  };

  const startTitleEdit = () => {
    if (!activeSessionId) return;
    setTitleDraft(headerTitle);
    setTitleEditing(true);
  };

  const cancelTitleEdit = () => {
    setTitleDraft('');
    setTitleEditing(false);
  };

  const commitTitleEdit = async () => {
    if (!activeSessionId) return;
    const nextTitle = titleDraft.trim();
    if (!nextTitle) {
      cancelTitleEdit();
      return;
    }
    if (nextTitle === headerTitle) {
      cancelTitleEdit();
      return;
    }
    setRenaming(true);
    setError(null);
    try {
      const updated = await renameCaseAgentSession(activeSessionId, { title: nextTitle });
      setSession((previous) => (
        previous?.session_id === updated.session_id ? updated : previous
      ));
      setSessions((previous) =>
        sortCaseAgentSessions(
          previous.map((item) => (item.session_id === updated.session_id ? updated : item)),
        ),
      );
      cancelTitleEdit();
    } catch (err) {
      setError(err instanceof Error ? err.message : '修改会话标题失败');
    } finally {
      setRenaming(false);
    }
  };

  const restoreDeletedSession = async (
    deletedSession: CaseAgentSession,
    shouldActivate: boolean,
  ) => {
    const toastKey = `case-agent-delete-${deletedSession.session_id}`;
    try {
      const restored = await restoreCaseAgentSession(deletedSession.session_id);
      setSessions((previous) => sortCaseAgentSessions([
        restored,
        ...previous.filter((item) => item.session_id !== restored.session_id),
      ]));
      if (shouldActivate || !activeSessionId) {
        activeSessionIdRef.current = restored.session_id;
        setActiveSessionId(restored.session_id);
        setSession(restored);
      }
      antdMessage.destroy(toastKey);
      antdMessage.success('已恢复会话');
    } catch (err) {
      setError(err instanceof Error ? err.message : '恢复会话失败');
    }
  };

  const deleteSession = async (target: CaseAgentSession) => {
    if (deletingSessionIds.has(target.session_id)) return;
    const targetRuntime = runStateRef.current[target.session_id];
    const targetRunStatus = targetRuntime?.status ?? target.latest_run?.effective_status;
    if (targetRunStatus && isActiveRunStatus(targetRunStatus)) {
      antdMessage.warning('会话仍有任务正在处理，请先停止任务');
      return;
    }
    const previousSessions = sessions;
    const previousSession = session;
    const wasActive = target.session_id === activeSessionId;
    const nextSessions = previousSessions.filter(
      (item) => item.session_id !== target.session_id,
    );
    markSessionDeleting(target.session_id, true);
    setSessions(nextSessions);
    if (wasActive) {
      const nextActive = nextSessions[0]?.session_id ?? '';
      activeSessionIdRef.current = nextActive;
      setActiveSessionId(nextActive);
      if (nextActive) {
        refreshSession(nextActive).catch((err) => {
          setError(err instanceof Error ? err.message : '加载会话失败');
        });
      } else {
        setSession(null);
      }
    }
    try {
      await archiveCaseAgentSession(target.session_id);
      const toastKey = `case-agent-delete-${target.session_id}`;
      antdMessage.open({
        key: toastKey,
        type: 'success',
        duration: 6,
        content: (
          <span className="case-agent-delete-toast">
            <span>{`已删除“${target.title}”`}</span>
            <Button
              type="link"
              size="small"
              onClick={() => restoreDeletedSession(target, wasActive)}
            >
              撤销
            </Button>
          </span>
        ),
      });
    } catch (err) {
      setSessions(previousSessions);
      setActiveSessionId(activeSessionId);
      setSession(previousSession);
      setError(err instanceof Error ? err.message : '删除会话失败');
    } finally {
      markSessionDeleting(target.session_id, false);
    }
  };

  const historyContent = (
    <div className="case-agent-history-popover" aria-label="历史会话">
      <div className="case-agent-history-list" role="list">
        {sessions.map((item) => {
          const active = item.session_id === activeSessionId;
          const deleting = deletingSessionIds.has(item.session_id);
          const runtime = runStateBySessionId[item.session_id];
          const effectiveStatus = runtime?.status ?? item.latest_run?.effective_status;
          const running = Boolean(effectiveStatus && isActiveRunStatus(effectiveStatus));
          const statusLabel = effectiveStatus === 'created'
            ? '排队中'
            : effectiveStatus === 'running' || effectiveStatus === 'resuming'
              ? '正在核验'
              : effectiveStatus === 'cancelling'
                ? '正在停止'
                : item.attention_type === 'needs_input'
                  ? '待补充'
                  : item.attention_type === 'degraded'
                    ? '降级完成'
                    : item.attention_type === 'failed'
                      ? '处理失败'
                      : '';
          return (
            <div
              className={`case-agent-history-item${active ? ' is-active' : ''}${
                deleting ? ' is-deleting' : ''
              }${running ? ' is-running' : ''}${
                item.has_unread_activity ? ` has-attention is-${item.attention_type}` : ''
              }`}
              key={item.session_id}
              role="listitem"
            >
              <button
                type="button"
                className="case-agent-history-card"
                aria-current={active ? 'true' : undefined}
                disabled={deleting}
                onClick={() => switchSession(item.session_id)}
              >
                <span className="case-agent-history-card-copy">
                  <span className="case-agent-history-title-row">
                    <span className="case-agent-history-card-title">{item.title}</span>
                    {item.has_unread_activity && !active && (
                      <span
                        className="case-agent-history-attention"
                        aria-label={item.attention_type === 'new_result' ? '有新回答' : statusLabel || '需要关注'}
                      />
                    )}
                    {active && (
                      <span className="case-agent-history-current-badge">当前</span>
                    )}
                  </span>
                  <span className="case-agent-history-card-time">
                    {statusLabel && <span className="case-agent-history-status">{statusLabel}</span>}
                    <span>{formatSessionTime(item.updated_at)}</span>
                  </span>
                </span>
              </button>
              <button
                type="button"
                className="case-agent-history-delete"
                aria-label={`删除${item.title}`}
                disabled={deleting || running}
                onClick={(event) => {
                  event.stopPropagation();
                  deleteSession(item);
                }}
              >
                删除
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );

  const sessionToolButtons = (
    <div className="case-agent-session-tools" aria-label="会话工具">
      <Tooltip title="新建会话">
        <Button
          type="text"
          shape="circle"
          className="case-agent-session-tool-button"
          aria-label="新建会话"
          onClick={createSession}
          disabled={!caseId || loading}
        >
          <span className="case-agent-session-icon-box">
            <NewSessionIcon />
          </span>
        </Button>
      </Tooltip>
      <Popover
        arrow={false}
        content={historyContent}
        open={historyOpen}
        onOpenChange={setHistoryOpen}
        placement="bottomRight"
        trigger="click"
        overlayClassName="case-agent-history-overlay"
      >
        <Tooltip title="历史会话">
          <Button
            type="text"
            shape="circle"
            className="case-agent-session-tool-button"
            aria-label="历史会话"
            disabled={!caseId || loading}
          >
            <span className="case-agent-session-icon-box">
              <HistoryOutlined />
            </span>
          </Button>
        </Tooltip>
      </Popover>
    </div>
  );

  const persistedBubbleItems: BubbleItemType[] = (() => {
    if (messages.length) {
      const items = messages.map((message) => {
        if (message.role === 'assistant' && message.answer_payload) {
          const answer = message.answer_payload;
          return {
            key: message.message_id,
            role: 'ai',
            content: (
              <AgentAnswerContent
                answer={answer}
                onOpenSource={setSourceDetail}
              />
            ),
            footer: (
              <div className="case-agent-message-footer">
                <div className="case-agent-message-actions">
                  <Actions.Copy text={answerPlainText(answer)} icon={<CopySoftIcon />} />
                </div>
                <span className="case-agent-message-time">
                  {formatAnswerTime(message.created_at)}
                </span>
              </div>
            ),
          };
        }
        return {
          key: message.message_id,
          role: message.role === 'user' ? 'user' : 'ai',
          content: businessText(message.content),
        };
      });
      if (pendingUserMessage) {
        items.push({
          key: 'pending-user-message',
          role: 'user',
          content: businessText(pendingUserMessage),
        });
      }
      return items;
    }
    if (pendingUserMessage) {
      return [
        {
          key: 'pending-user-message',
          role: 'user',
          content: businessText(pendingUserMessage),
        },
      ];
    }
    return [
      {
        key: 'welcome',
        role: 'ai',
        content: (
          <div className="case-agent-welcome-message">
            当前案件助手已就绪。可选择常用核验问题，或输入需要核验的案件事实、规则依据与 Evidence 分析。
          </div>
        ),
      },
    ];
  })();
  const latestRunStatus = activeRunState?.status ?? session?.latest_run?.effective_status;
  const bubbleItems: BubbleItemType[] = sending
    ? [
        ...persistedBubbleItems,
        {
          key: 'case-agent-progress',
          role: 'ai',
          content: (
            <AgentProgressBubble
              events={events}
              elapsedMs={elapsedMs}
              progressStartedAt={progressStartedAt}
            />
          ),
        },
      ]
    : latestRunStatus === 'cancelled'
      ? [
          ...persistedBubbleItems,
          {
            key: `case-agent-cancelled-${session?.latest_run?.run_id ?? activeRunState?.runId ?? 'latest'}`,
            role: 'ai',
            content: <div className="case-agent-run-notice">该请求已停止，未生成或保存助手回答。</div>,
          },
        ]
      : persistedBubbleItems;

  useEffect(() => {
    if (!scrollAfterSendRef.current) return;
    scrollToLatestMessage('smooth');
  }, [bubbleItems.length, pendingUserMessage, sending, scrollToLatestMessage]);

  useEffect(() => {
    if (pendingUserMessage) {
      scrollToLatestMessage('smooth');
    }
  }, [pendingUserMessage, scrollToLatestMessage]);

  useEffect(() => {
    if (!scrollAfterSendRef.current || sending) return;
    scrollToLatestMessage('smooth');
    scrollAfterSendRef.current = false;
  }, [messages.length, sending, scrollToLatestMessage]);

  return (
    <section className="case-agent-panel" aria-label="案件稽核助手">
      <div className="case-agent-shell">
        <header className="case-agent-header">
          <div className="case-agent-title">
            <AuditAssistantIcon />
            {titleEditing ? (
              <Input
                ref={titleInputRef}
                className="case-agent-title-input"
                value={titleDraft}
                maxLength={80}
                size="small"
                disabled={renaming}
                onChange={(event) => setTitleDraft(event.target.value)}
                onBlur={commitTitleEdit}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    event.preventDefault();
                    commitTitleEdit();
                  }
                  if (event.key === 'Escape') {
                    event.preventDefault();
                    cancelTitleEdit();
                  }
                }}
              />
            ) : (
              <button
                type="button"
                className="case-agent-title-button"
                title={headerTitle}
                disabled={!activeSessionId || renaming}
                onClick={startTitleEdit}
              >
                <span>{headerTitle}</span>
              </button>
            )}
          </div>
          <div className="case-agent-header-actions">
            {sessionToolButtons}
            {onClose && (
              <Tooltip title="关闭案件稽核助手">
                <Button
                  type="text"
                  shape="circle"
                  icon={<CloseOutlined />}
                  aria-label="关闭案件稽核助手"
                  onClick={onClose}
                />
              </Tooltip>
            )}
          </div>
        </header>

        {data ? (
          <div className="case-agent-layout">
            {error && (
              <Alert
                className="case-agent-alert"
                type="warning"
                showIcon
                message={error}
              />
            )}

            <div
              ref={messagesContainerRef}
              className={`case-agent-messages${
                !hasConversationMessages && !sending ? ' case-agent-messages-initial' : ''
              }`}
              aria-live="polite"
            >
              {loading && !session ? (
                <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="正在加载会话" />
              ) : (
                <>
                  <Bubble.List
                    autoScroll={bubbleItems.length > 2}
                    items={bubbleItems}
                    role={{
                      ai: {
                        placement: 'start',
                        variant: 'borderless',
                        className: 'case-agent-ai-bubble',
                      },
                      user: {
                        placement: 'end',
                        variant: 'filled',
                        shape: 'corner',
                        className: 'case-agent-user-bubble',
                      },
                    }}
                  />
                </>
              )}
            </div>

            <CaseAgentComposer
              activeSessionId={activeSessionId}
              activeStage={activeStage}
              answers={answers}
              caseId={data.case.case_id}
              caseType={data.case.case_type}
              hasUserMessage={hasUserMessage}
              draft={activeDraft}
              running={sending}
              stopping={activeRunState?.stopping ?? false}
              onDraftChange={(value) => setDraftBySessionId((previous) => ({
                ...previous,
                [activeSessionId]: value,
              }))}
              onSendQuestion={sendQuestion}
              onStop={() => void stopCurrentRun()}
              onUnavailableAttachment={() => (
                setError('附件上传能力暂未接入，当前仅支持文本核验问题。')
              )}
            />
          </div>
        ) : (
          <Empty
            className="case-agent-empty"
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description="请先选择一个审核案件"
          />
        )}
      </div>

      <Modal
        title={sourceDetail ? businessText(sourceDetail.title) : '依据详情'}
        open={Boolean(sourceDetail)}
        onCancel={() => setSourceDetail(null)}
        footer={[
          sourceDetailStage && onStageChange ? (
            <Button key="open-module" onClick={openSourceDetailStage}>
              查看完整业务模块
            </Button>
          ) : null,
          <Button key="close" type="primary" onClick={() => setSourceDetail(null)}>
            关闭
          </Button>,
        ]}
      >
        {sourceDetail && (
          <div className="case-agent-source-detail">
            <dl className="case-agent-source-detail-list">
              {getSourceDetailDisplayFields(sourceDetail).map((field) => (
                  <div
                    className={`case-agent-source-detail-item is-${field.tone ?? 'value'}`}
                    key={`${field.label}-${field.value}`}
                  >
                    <dt>{businessText(field.label)}</dt>
                    <dd>{formatSourceFieldValue(field.value)}</dd>
                  </div>
              ))}
            </dl>
          </div>
        )}
      </Modal>
    </section>
  );
}
