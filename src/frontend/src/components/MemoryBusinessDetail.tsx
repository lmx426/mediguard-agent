import {
  InfoCircleOutlined,
} from '@ant-design/icons';
import { Collapse, Typography } from 'antd';
import type {
  MemoryPresentationItem,
  MemorySafeDetail,
  MemoryType,
} from '../types';

type JsonRecord = Record<string, unknown>;

type BusinessRow = {
  label: string;
  value: string;
};

const MEMORY_LEVEL_LABELS: Record<string, string> = {
  L1_atomic_memory: '单次经验',
  L2_scenario_memory: '场景经验',
  L3_stable_profile_or_playbook: '稳定偏好或标准流程',
};

const MEMORY_CONSUMER_BUSINESS_LABELS: Record<string, string> = {
  intent_router: '意图识别与任务路由',
  policy_filter_resolver: '政策检索条件整理',
  expert_analysis: '政策与专业依据分析',
  decision_planner: '审核任务步骤规划',
  answer_generator: '审核答复生成',
  recovery_handler: '异常恢复处理',
};

const INTENT_LABELS: Record<string, string> = {
  case_task: '查询当前案件事实',
  expert_task: '查询政策或专业依据',
  general_help: '一般业务帮助',
  clarification_required: '补充任务信息',
};

const CAPABILITY_LABELS: Record<string, string> = {
  query_evidence_package: '读取当前案件的基础证据包',
  query_verification_items: '查询待人工核验事项',
  query_case_judgement: '查询已有案件研判结果',
  query_rule_verification: '查询业务规则核验清单',
  query_case_overview: '查询案件基础信息',
  query_materials: '查询案件材料目录',
  query_claim_details: '查询申报明细',
  ask_policy_expert: '检索并分析相关政策依据',
  derive_filter_sort_topn: '对已取得的信息进行筛选、排序或重点提取',
  derive_aggregation: '对已取得的信息进行汇总统计',
  derive_peer_comparison: '进行同类指标比较',
  explain_rule_metric: '解释规则指标和阈值含义',
  direct_answer: '直接组织回答',
};

const TARGET_LABELS: Record<string, string> = {
  evidence_package: '当前案件的基础证据包',
  evidence_package_query: '当前案件的基础证据包',
  verification_items: '待人工核验事项',
  case_judgement: '已有案件研判结果',
  rule_verification: '业务规则核验信息',
  policy_evidence: '政策依据',
};

const EVIDENCE_LABELS: Record<string, string> = {
  case_fact: '当前案件事实',
  derived_fact: '由当前案件信息派生的结果',
  policy_evidence: '可核验政策依据',
};

const VALIDATION_LABELS: Record<string, string> = {
  invalid_json: '回答格式无法读取',
  invalid_schema: '回答结构不符合要求',
  citation_invalid: '引用依据无效',
  citation_missing_for_grounded: '需要依据的内容缺少引用',
};

const FILTER_LABELS: Record<string, string> = {
  jurisdiction: '适用地区',
  policy_domain: '政策领域',
  document_number: '政策文号',
  document_type: '文件类型',
  catalog: '政策目录',
  effective_date: '生效时间',
};

const STYLE_LABELS: Record<string, string> = {
  conclusion_first: '结论前置',
  list_format: '分点组织',
  table_format: '表格组织',
};

function asRecord(value: unknown): JsonRecord {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as JsonRecord
    : {};
}

function stringValues(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value
      .map((item) => typeof item === 'string' || typeof item === 'number' ? String(item) : '')
      .filter(Boolean);
  }
  if (typeof value === 'string' || typeof value === 'number') return [String(value)];
  return [];
}

function joinValues(value: unknown, labels: Record<string, string> = {}): string {
  return stringValues(value).map((item) => labels[item] ?? item).join('、');
}

function actionParts(memory: Pick<MemorySafeDetail, 'structured_action'>) {
  const root = asRecord(memory.structured_action);
  const matchProfile = asRecord(root.match_profile);
  const nestedAction = asRecord(root.action_payload);
  const action = Object.keys(nestedAction).length ? nestedAction : root;
  return { root, matchProfile, action };
}

function capabilityText(value: unknown): string {
  return joinValues(value, CAPABILITY_LABELS);
}

function targetText(value: unknown): string {
  return joinValues(value, TARGET_LABELS);
}

function intentText(value: unknown): string {
  return joinValues(value, INTENT_LABELS);
}

function filterText(value: unknown): string {
  const filters = asRecord(value);
  return Object.entries(filters)
    .map(([key, item]) => {
      const content = joinValues(item);
      return content ? `${FILTER_LABELS[key] ?? key}：${content}` : '';
    })
    .filter(Boolean)
    .join('；');
}

function styleText(value: unknown): string {
  const preferences = asRecord(value);
  const labels: string[] = [];
  Object.entries(STYLE_LABELS).forEach(([key, label]) => {
    if (preferences[key] === true) labels.push(label);
    if (preferences[key] === false && key === 'table_format') labels.push('不使用表格');
  });
  if (preferences.verbosity === 'concise') labels.push('简洁回答');
  if (preferences.verbosity === 'detailed') labels.push('详细说明');
  return labels.join('、');
}

function genericSummary(memoryType: MemoryType): string {
  return {
    intent_route_hint: '相似案件查询可参考这条已经验证的任务路由。',
    policy_search_hint: '相似政策问题可复用这条已经验证的检索条件和回答要点。',
    failure_hint: '出现相似校验问题时，可参考这条已经验证的恢复方式。',
    answer_style_hint: '生成回答时，可沿用审核人员已经明确表达的组织偏好。',
    decision_plan_hint: '处理相似的多步骤任务时，可参考这条已经验证的执行顺序。',
  }[memoryType];
}

function summaryFromTechnicalText(memory: Pick<MemoryPresentationItem, 'memory_type' | 'summary'>): string {
  if (memory.memory_type === 'intent_route_hint') {
    const route = memory.summary.match(/(?:使用路由|建议使用)\s+(.+)$/)?.[1];
    const capabilities = route?.split(/[,，、\s]+/).filter(Boolean) ?? [];
    if (capabilities.length) {
      return `处理相似案件任务时，系统建议${capabilityText(capabilities)}，为后续审核准备所需依据。`;
    }
  }
  if (memory.memory_type === 'answer_style_hint' && !/[a-z]+_[a-z]+/i.test(memory.summary)) {
    return memory.summary;
  }
  return genericSummary(memory.memory_type);
}

function reviewerMemorySummary(
  memory: Pick<MemoryPresentationItem, 'memory_type' | 'summary'> & Partial<Pick<MemorySafeDetail, 'structured_action'>>,
): string {
  if (!memory.structured_action || !Object.keys(memory.structured_action).length) {
    return summaryFromTechnicalText(memory);
  }
  const { root, matchProfile, action } = actionParts(memory as Pick<MemorySafeDetail, 'structured_action'>);

  if (memory.memory_type === 'intent_route_hint') {
    const task = intentText(action.intent ?? matchProfile.task_type ?? root.route_tags) || '查询当前案件信息';
    const capability = capabilityText(action.capabilities ?? root.capabilities);
    return capability
      ? `当审核人员需要${task}时，系统建议${capability}，为后续审核提供当前案件依据。`
      : `当审核人员需要${task}时，系统可参考这条已经验证的任务路由。`;
  }

  if (memory.memory_type === 'policy_search_hint') {
    const needs = joinValues(action.information_needs ?? root.information_needs);
    const filters = filterText(action.filters ?? root.filters);
    return `查询${needs || '相关'}政策依据时，系统建议${filters ? `按照${filters}进行检索` : '沿用已验证的检索条件'}，并覆盖所需回答要点。`;
  }

  if (memory.memory_type === 'decision_plan_hint') {
    const steps = capabilityText(action.planning_steps ?? root.planning_steps);
    return steps
      ? `处理相似的多步骤审核任务时，系统建议依次${steps}，并在当前案件范围内完成。`
      : genericSummary(memory.memory_type);
  }

  if (memory.memory_type === 'failure_hint') {
    const codes = joinValues(
      matchProfile.validation_codes ?? root.validation_codes ?? root.validation_code,
      VALIDATION_LABELS,
    );
    const recovery = joinValues(action.recover ?? root.recover) || '按照已验证方式修复并重新校验';
    return `当${codes || '回答校验未通过'}时，系统建议${recovery}。`;
  }

  const preferences = styleText(action.style_preferences ?? root.style_preferences ?? action);
  return preferences
    ? `生成审核回答时，按照审核人员偏好采用${preferences}；当前明确要求始终优先。`
    : genericSummary(memory.memory_type);
}

function businessRows(memory: MemorySafeDetail): BusinessRow[] {
  const { root, matchProfile, action } = actionParts(memory);
  const constraints = joinValues(action.constraints ?? root.constraints);

  if (memory.memory_type === 'intent_route_hint') {
    const task = intentText(action.intent ?? matchProfile.task_type ?? root.route_tags) || '案件信息查询';
    const targets = targetText(matchProfile.target_objects);
    const evidence = joinValues(matchProfile.evidence_need, EVIDENCE_LABELS);
    return [
      { label: '适用场景', value: `${task}；${targets || evidence || '当前请求需要查询案件事实'}` },
      { label: '系统操作', value: capabilityText(action.capabilities ?? root.capabilities) || '沿用已经验证的任务路由' },
      { label: '使用边界', value: '为后续审核准备案件信息；当前仅作路由验证，不会直接改变执行流程' },
    ];
  }

  if (memory.memory_type === 'policy_search_hint') {
    const needs = joinValues(action.information_needs ?? root.information_needs);
    const coverage = joinValues(action.coverage_requirements ?? root.coverage_requirements);
    return [
      { label: '适用场景', value: `查询政策、服务指南或审核依据；重点回答${needs || coverage || '当前问题要求的信息点'}` },
      { label: '系统操作', value: filterText(action.filters ?? root.filters) || '根据当前问题确定检索范围' },
      { label: '使用边界', value: joinValues(action.avoid_claims ?? root.avoid_claims) || '历史检索经验只用于辅助检索，不能代替当前有效政策结论' },
    ];
  }

  if (memory.memory_type === 'decision_plan_hint') {
    const complexity = matchProfile.task_complexity === 'multi_step' ? '需要多个步骤完成的审核任务' : '相似审核任务';
    return [
      { label: '适用场景', value: complexity },
      { label: '系统操作', value: capabilityText(action.planning_steps ?? root.planning_steps) || '按照已验证顺序处理' },
      {
        label: '执行边界',
        value: `${Object.keys(asRecord(action.dependencies ?? root.dependencies)).length ? '后续步骤等待前序步骤完成；' : ''}${constraints || '只调用已开放能力，并保持当前案件数据隔离'}`,
      },
    ];
  }

  if (memory.memory_type === 'failure_hint') {
    return [
      { label: '适用场景', value: joinValues(matchProfile.validation_codes ?? root.validation_code, VALIDATION_LABELS) || '回答校验未通过' },
      { label: '系统操作', value: joinValues(action.recover ?? root.recover) || '修复后再次执行完整校验' },
      { label: '使用边界', value: `${joinValues(action.avoid ?? root.avoid) || '不得保留未通过校验的内容'}；不改变案件事实、规则结果或人工决定` },
    ];
  }

  return [
    { label: '适用场景', value: matchProfile.scenario_type === 'all' || !matchProfile.scenario_type ? '审核人员的日常回答' : String(matchProfile.scenario_type) },
    { label: '系统操作', value: styleText(action.style_preferences ?? root.style_preferences ?? action) || '沿用已确认的回答组织偏好' },
    { label: '使用边界', value: joinValues(action.override_rules ?? root.override_rules) || '当前用户明确提出的要求优先于历史偏好' },
  ];
}

function sourceDescription(memory: MemorySafeDetail): string {
  if (memory.memory_level === 'L3_stable_profile_or_playbook') {
    return '由多次稳定场景经验归纳形成，原始案件敏感内容未保存在记忆中';
  }
  if (memory.memory_level === 'L2_scenario_memory') {
    return '由多条相似的单次经验归纳形成，原始问题和案件敏感内容未保存在记忆中';
  }
  return '由一次已完成的受控任务提炼，原始问题和案件敏感内容未保存在记忆中';
}

function formatDateTime(value?: string | null): string {
  if (!value) return '未记录';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '未记录' : date.toLocaleString('zh-CN');
}

function consumerBusinessLabel(value: string): string {
  return MEMORY_CONSUMER_BUSINESS_LABELS[value] ?? '系统内部处理节点';
}

export function MemoryBusinessDetail({
  memory,
  scopeText,
}: {
  memory: MemorySafeDetail;
  scopeText: string;
}) {
  const rows = businessRows(memory);
  const consumers = memory.allowed_consumers.map(consumerBusinessLabel).join('、') || '未限定使用环节';
  const technicalJson = Object.keys(memory.structured_action).length
    ? JSON.stringify(memory.structured_action, null, 2)
    : '暂无内部结构化字段';

  return (
    <div className="memory-business-overview">
      <section className="memory-business-summary" aria-label="记忆摘要">
        <InfoCircleOutlined aria-hidden="true" />
        <Typography.Paragraph>{reviewerMemorySummary(memory)}</Typography.Paragraph>
      </section>

      <section className="memory-business-section" aria-labelledby="memory-business-action-title">
        <Typography.Text id="memory-business-action-title" strong>关键信息</Typography.Text>
        <dl className="memory-business-rows">
          {rows.map((row) => (
            <div key={row.label}>
              <dt>{row.label}</dt>
              <dd>{row.value}</dd>
            </div>
          ))}
        </dl>
      </section>

      <dl className="memory-business-meta" aria-label="记忆适用信息">
        <div><dt>成熟程度</dt><dd>{MEMORY_LEVEL_LABELS[memory.memory_level] ?? memory.memory_level}</dd></div>
        <div><dt>可信程度</dt><dd>{Math.round(memory.confidence * 100)}%</dd></div>
        <div><dt>适用范围</dt><dd>{scopeText}</dd></div>
        <div><dt>使用环节</dt><dd>{consumers}</dd></div>
      </dl>

      <Collapse
        ghost
        className="memory-technical-collapse"
        items={[
          {
            key: 'technical',
            label: '来源与技术信息',
            children: (
              <div className="memory-technical-content">
                <p><strong>形成依据：</strong>{sourceDescription(memory)}</p>
                <p><strong>形成时间：</strong>{formatDateTime(memory.created_at)}</p>
                <p><strong>内部摘要：</strong>{memory.summary}</p>
                <p><strong>内部说明：</strong>{memory.detail || '未提供'}</p>
                <p><strong>脱敏来源编号：</strong>{memory.source_refs.join('、') || '未记录'}</p>
                <p><strong>内部记忆编号：</strong>{memory.memory_id}</p>
                <pre className="memory-json-preview">{technicalJson}</pre>
              </div>
            ),
          },
        ]}
      />
    </div>
  );
}
