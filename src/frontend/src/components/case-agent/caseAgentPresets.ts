import type { CaseFullResponse, WorkflowStepKey } from '../../types';
import { businessText, ruleBusinessName } from '../../utils/businessLabels';

export interface PresetAnswer {
  key: string;
  question: string;
  answer: string;
  sources: string[];
  checks: string[];
}

function compact(items: Array<string | null | undefined>): string[] {
  return items.filter((item): item is string => Boolean(item));
}

const STAGE_CHECKS: Record<WorkflowStepKey, string[]> = {
  case_intake: ['确认接入记录与案件编号一致'],
  fact_base: ['核对结构化字段与已登记材料是否一致'],
  risk_screening: ['确认风险提示构成项与页面展示一致'],
  rule_check: ['核对规则适用范围及对应证据字段'],
  evidence_package: ['确认引用依据完整且可追溯'],
  initial_review: ['提交前复核事实、规则与材料记录'],
  secondary_review: ['确认复审条件和补充材料范围'],
  appeal_handling: ['区分新增事实、冲突点和仍缺失信息'],
  case_result: ['确认展示内容与已记录人工意见一致'],
};

type VisitorSamplePrompt = Pick<PresetAnswer, 'key' | 'question'>;

const VISITOR_SAMPLE_PROMPTS: Record<string, VisitorSamplePrompt[]> = {
  SIM_PERSON_011872: [
    {
      key: 'case_1_remote_principle',
      question: '跨省异地就医中，就医地目录与参保地待遇如何分工？',
    },
    {
      key: 'case_1_emergency_manual',
      question: '北京参保人异地急诊、备案和手工报销如何处理？',
    },
    {
      key: 'case_1_shanghai_payment_scope',
      question: '上海药品、胸部 CT 和医用耗材支付范围如何核验？',
    },
  ],
  SIM_PERSON_006395: [
    {
      key: 'case_2_emergency_vs_outpatient',
      question: '异地急诊抢救与普通门诊如何区分和适用备案规则？',
    },
    {
      key: 'case_2_emergency_material_gap',
      question: '异地急诊留观费用申请手工报销时，现有政策要求核验哪些材料？',
    },
    {
      key: 'case_2_ordinary_manual_materials',
      question: '普通异地门诊手工报销需要哪些材料？',
    },
  ],
  SIM_PERSON_002052: [
    {
      key: 'case_3_no_shanghai_ratio',
      question: '北京参保人在上海门诊就医，为什么不能直接使用上海报销比例？',
    },
    {
      key: 'case_3_beijing_retiree_benefit',
      question: '北京退休职工门急诊待遇应核验哪些参数？',
    },
    {
      key: 'case_3_shanghai_scope',
      question: '上海药品和诊疗项目支付范围如何核验？',
    },
  ],
  SIM_PERSON_014632: [
    {
      key: 'case_4_active_policy_by_service_date',
      question: '2026年5月上海异地门诊案件应使用哪一版北京备案政策？',
    },
    {
      key: 'case_4_replaced_policy',
      question: '北京异地备案旧政策是否已经废止或被新政策替代？',
    },
    {
      key: 'case_4_valid_on_filter',
      question: '政策 RAG 如何按服务日期筛选有效政策？',
    },
  ],
  SIM_PERSON_011569: [
    {
      key: 'case_5_beijing_ct_price',
      question: '北京医疗服务价格表中，CT 项目的项目名称、计价单位和价格如何核验？',
    },
    {
      key: 'case_5_ct_report_film_charge',
      question: 'CT 平扫、图文报告和胶片费能否分别收费，应查什么价格依据？',
    },
    {
      key: 'case_5_fund_supervision_charge',
      question: '医保基金监管中，检查费用应围绕哪些重复、分解或超标准风险点核验？',
    },
  ],
  SIM_PERSON_002204: [
    {
      key: 'case_6_drug_catalog_scope',
      question: '阿莫西林、布洛芬等是否在国家或北京医保药品目录中？',
    },
    {
      key: 'case_6_limited_payment',
      question: '医保药品限定支付范围和诊断用药匹配如何核验？',
    },
    {
      key: 'case_6_drug_ratio_not_violation',
      question: '药品费占比高是否足以直接认定医保不合规？',
    },
  ],
  SIM_PERSON_011394: [
    {
      key: 'case_7_manual_materials',
      question: '北京门诊手工报销需要提交哪些材料？',
    },
    {
      key: 'case_7_material_chain',
      question: '票据、处方、费用明细和门诊病历如何形成手工报销材料链？',
    },
    {
      key: 'case_7_missing_registration',
      question: '缺少挂号或门诊病历时，手工报销应补充核验哪些材料？',
    },
  ],
  SIM_PERSON_013515: [
    {
      key: 'case_8_long_prescription',
      question: '北京慢病长期处方和续方如何核验？',
    },
    {
      key: 'case_8_special_disease',
      question: '北京门诊慢特病备案和病种范围如何核验？',
    },
    {
      key: 'case_8_designated_duplicate',
      question: '北京慢病多机构取药时，定点机构、重复购药和异常开药应分别核验什么？',
    },
  ],
};

export function buildPresetAnswers(
  data: CaseFullResponse,
  stage: WorkflowStepKey,
): PresetAnswer[] {
  const caseInfo = data.case;
  const visitorSamplePrompts = caseInfo.subject_ref
    ? VISITOR_SAMPLE_PROMPTS[caseInfo.subject_ref]
    : undefined;
  if (visitorSamplePrompts) {
    return visitorSamplePrompts.map((prompt) => ({
      ...prompt,
      answer: '',
      sources: [],
      checks: STAGE_CHECKS[stage],
    }));
  }

  const hitRules = caseInfo.rule_hits.filter((rule) => rule.hit);
  const ruleNames = hitRules.map((rule) => ruleBusinessName(rule));
  const missing = data.evidence.missing_information ?? [];
  const citationRefs = (data.evidence.citations ?? []).map((citation) => citation.ref);
  const attentionReasons = caseInfo.model_signal_reasons.map(businessText).join('；');
  const riskScoreText =
    caseInfo.risk_score_breakdown?.display_score ??
    `${Math.round(caseInfo.risk_score * 100)} / 100`;
  const riskBreakdownText = caseInfo.risk_score_breakdown
    ? caseInfo.risk_score_breakdown.components
        .map((component) => `${component.label}${component.score}/${component.max_score}`)
        .join('，')
    : attentionReasons;
  const reviewSummary = data.review
    ? `${data.review.reviewer}已记录“${data.review.decision}”，理由为：${data.review.reason}`
    : '当前尚未提交人工初审意见。';

  const stageAnswers: Record<WorkflowStepKey, Omit<PresetAnswer, 'checks'>[]> = {
    case_intake: [
      {
        key: 'intake-scope',
        question: '这条案件如何进入稽核流程？',
        answer: `案件 ${caseInfo.case_id} 由脱敏医保结算统计记录建案，类型为“${caseInfo.case_type}”。接入阶段只确认接入范围和处理基线，不形成风险或审核结论。`,
        sources: compact([caseInfo.case_id, caseInfo.rule_pool_version]),
      },
      {
        key: 'intake-boundary',
        question: '接入数据有哪些安全边界？',
        answer:
          '当前案件上下文只使用脱敏结构化统计字段和统计材料视图，不应包含真实身份信息、RES 标签、原始票据或生产原始病历。',
        sources: ['项目接入范围'],
      },
    ],
    fact_base: [
      {
        key: 'fact-summary',
        question: '当前有哪些可确认的案件事实？',
        answer: `${businessText(caseInfo.claim_summary)}。事实底座陈列结构化申报、就诊统计和由 81 字段生成的统计材料，不替代医学合理性判断。`,
        sources: compact([caseInfo.model_evidence_ref, '脱敏申报宽表']),
      },
      {
        key: 'fact-missing',
        question: '事实底座还缺什么材料？',
        answer:
          missing.length > 0
            ? `证据包标记的待补充项包括：${missing.map(businessText).join('；')}。`
            : '当前证据包未标记具体缺失材料；如需判断医学合理性，仍应核验合法取得的业务材料和有效政策依据。',
        sources: compact(citationRefs),
      },
    ],
    risk_screening: [
      {
        key: 'risk-reason',
        question: '为什么需要人工关注？',
        answer: `当前风险提示强度为 ${riskScoreText}，风险等级为${
          caseInfo.risk_level === 'high'
            ? '高'
            : caseInfo.risk_level === 'medium'
              ? '中'
              : caseInfo.risk_level === 'low'
                ? '低'
                : '证据不足'
        }。贡献构成：${riskBreakdownText || '当前没有额外关注原因'}。该提示只用于分流，不是欺诈判断。`,
        sources: compact([caseInfo.model_evidence_ref]),
      },
      {
        key: 'risk-next',
        question: '风险筛查后先核对什么？',
        answer: `${caseInfo.evidence_consistency}。建议先进入规则核验，确认筛查线索是否满足确定性业务条件。`,
        sources: compact([caseInfo.model_evidence_ref, ...citationRefs]),
      },
    ],
    rule_check: [
      {
        key: 'rule-hits',
        question: '本案命中了哪些规则？',
        answer:
          hitRules.length > 0
            ? `共发现 ${hitRules.length} 条规则线索：${ruleNames.join('、')}。`
            : '当前没有规则命中记录，应避免仅凭风险提示形成处理结论。',
        sources: compact(hitRules.map((rule) => rule.evidence_ref)),
      },
      {
        key: 'rule-action',
        question: '规则线索建议怎么核验？',
        answer:
          hitRules.length > 0
            ? hitRules
                .map(
                  (rule) =>
                    `${ruleBusinessName(rule)}：${businessText(rule.business_explanation)}`,
                )
                .join('；')
            : '核对输入字段、规则适用范围和数据质量，并保留人工核验记录。',
        sources: compact(hitRules.map((rule) => ruleBusinessName(rule))),
      },
    ],
    evidence_package: [
      {
        key: 'evidence-advice',
        question: '证据包给出的审核建议是什么？',
        answer: businessText(data.evidence.recommendation),
        sources: compact(citationRefs),
      },
      {
        key: 'evidence-extend',
        question: '哪些情况需要扩展核验？',
        answer:
          '涉及新药、新技术或参保地政策差异时应核验最新有效政策；出现医院、药店、参保人或中介协同线索时应开展跨主体关联核验。当前环境未配置联网检索、RAG 或关系图谱。',
        sources: ['扩展核验提醒', ...compact(citationRefs)],
      },
    ],
    initial_review: [
      {
        key: 'initial-focus',
        question: '初审前还需要确认什么？',
        answer:
          missing.length > 0
            ? `优先确认：${missing.map(businessText).join('；')}。同时核对规则适用、来源明细和申报事实是否一致。`
            : '当前未标记具体缺失材料。提交初审前仍需核对规则适用、来源明细和申报事实是否一致。',
        sources: compact(citationRefs),
      },
      {
        key: 'initial-record',
        question: '当前人工初审记录是什么？',
        answer: reviewSummary,
        sources: data.review ? ['人工初审记录'] : ['人工初审待办'],
      },
    ],
    secondary_review: [
      {
        key: 'secondary-entry',
        question: '什么情况下进入人工复审？',
        answer:
          '初审移交、结论争议、重大风险线索，或补充材料可能改变原有判断时，由医保侧上级审核员、审核组长或授权复审人员进入复审。',
        sources: compact(['复审条件', data.review ? '人工初审记录' : null]),
      },
      {
        key: 'secondary-role',
        question: '复审由谁完成？',
        answer:
          '复审属于医保侧分级审核，不由申报人员完成。申报方只负责提交补充或申诉材料。',
        sources: ['审核角色边界'],
      },
    ],
    appeal_handling: [
      {
        key: 'appeal-check',
        question: '收到申诉材料后核对什么？',
        answer:
          '比较新增材料与原事实底座、规则依据和人工意见，标记新增事实、冲突点及仍缺失的信息，并由医保审核人员形成处理意见。',
        sources: compact(['申诉材料', ...citationRefs]),
      },
      {
        key: 'appeal-agent',
        question: '案件稽核助手在申诉阶段能做什么？',
        answer:
          '稽核助手可以整理材料差异和待核验项，但不能读取当前未上传的文件、不能代替复审，也不能修改人工结论。',
        sources: ['稽核助手权限边界'],
      },
    ],
    case_result: [
      {
        key: 'result-current',
        question: '当前案件处理结果是什么？',
        answer: reviewSummary,
        sources: data.review ? ['人工初审记录'] : ['案件结果待办'],
      },
      {
        key: 'result-boundary',
        question: '这个结果可以由稽核助手修改吗？',
        answer:
          '不可以。案件处理结果必须由有权限的医保审核人员确认，稽核助手只能整理证据和提示待核验事项。',
        sources: ['人工决定边界'],
      },
    ],
  };

  return stageAnswers[stage].map((answer) => ({
    ...answer,
    checks: STAGE_CHECKS[stage],
  }));
}
