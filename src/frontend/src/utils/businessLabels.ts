const RULE_BUSINESS_NAMES: Record<string, string> = {
  'OP-R001': '就诊行为一致性核验',
  'OP-R002': '费用与支付结构核验',
  'OP-R003': '药品费用合理性核验',
  'OP-R004': '检查治疗结构核验',
  'OP-R005': '挂号与流程一致性核验',
  'OP-R006': '待遇补助口径核验',
  'OP-R007': '申报数据质量核验',
  'OP-R008': '材料补充核验',
};

const RULE_NAME_REPLACEMENTS: Record<string, string> = {
  高频就诊分级规则: '就诊频次核验',
  '多机构 / 同日多机构分级规则': '多机构就诊核验',
  '高金额 / 高审批金额分级规则': '费用金额核验',
  药品费用组合规则: '药品费用结构核验',
  治疗费用结构规则: '治疗项目费用核验',
  '个人支付 / 基金支付比例异常规则': '医保支付结构核验',
  挂号状态组合规则: '挂号与就诊流程核验',
  数据质量与适用范围提示规则: '数据完整性与适用范围核验',
  就诊行为一致性核验: '就诊行为一致性核验',
  费用与支付结构核验: '费用与支付结构核验',
  药品费用合理性核验: '药品费用合理性核验',
  检查治疗结构核验: '检查治疗结构核验',
  挂号与流程一致性核验: '挂号与流程一致性核验',
  待遇补助口径核验: '待遇补助口径核验',
  申报数据质量核验: '申报数据质量核验',
  材料补充核验: '材料补充核验',
};

const SEVERITY_BUSINESS_LABELS: Record<string, string> = {
  critical: '强关注',
  high: '高关注',
  medium: '中关注',
  low: '低关注',
  info: '提示',
};

const SEVERITY_COLORS: Record<string, string> = {
  critical: 'red',
  high: 'volcano',
  medium: 'orange',
  low: 'blue',
  info: 'default',
};

export function ruleBusinessName(rule: {
  rule_id?: string;
  rule_name?: string;
}): string {
  if (rule.rule_id && RULE_BUSINESS_NAMES[rule.rule_id]) {
    return RULE_BUSINESS_NAMES[rule.rule_id];
  }
  if (rule.rule_name && RULE_NAME_REPLACEMENTS[rule.rule_name]) {
    return RULE_NAME_REPLACEMENTS[rule.rule_name];
  }
  return rule.rule_name?.replace(/规则|分级/g, '').trim() || '业务规则核验';
}

export function ruleReviewTitle(rule: {
  rule_id?: string;
  rule_name?: string;
}): string {
  const name = ruleBusinessName(rule);
  if (name === '申报数据质量核验') return '申报数据质量需确认';
  if (name === '材料补充核验') return '需查看材料已整理';
  return `${name.replace(/核验$/, '')}需核验`;
}

export function sourceTypeLabel(source: string): string {
  if (source === 'model_evidence') return '风险筛查';
  if (source === 'fraud_screening') return '模型识别预警';
  if (source === 'rule_evidence') return '业务规则核验';
  return '业务依据';
}

export function severityBusinessLabel(severity: string): string {
  return SEVERITY_BUSINESS_LABELS[severity] ?? severity;
}

export function severityColor(severity: string): string {
  return SEVERITY_COLORS[severity] ?? 'default';
}

export function citationBusinessLabel(label: string): string {
  const ruleMatch = label.match(/规则 \[(OP-R\d+)\]/);
  if (ruleMatch?.[1]) {
    return RULE_BUSINESS_NAMES[ruleMatch[1]] ?? '业务规则核验';
  }
  if (label.includes('风险提示强度')) return '费用与就诊异常筛查';
  if (label.includes('模型识别预警')) return '模型识别预警';
  return businessText(label);
}

export function evidenceRefLabel(ref: string): string {
  if (ref.startsWith('rule:')) {
    const ruleMatch = ref.match(/rule:(OP-R\d+)/);
    return ruleMatch?.[1] ? `${RULE_BUSINESS_NAMES[ruleMatch[1]]}留痕` : '规则核验留痕';
  }
  if (ref.startsWith('model') || ref.startsWith('risk_score') || ref.includes('model_signal')) {
    return '风险筛查留痕';
  }
  if (ref.startsWith('fraud_screening')) {
    return '模型识别预警留痕';
  }
  if (ref.includes('op-rule-pool')) {
    return '当前业务核验口径';
  }
  return ref;
}

export function businessText(text: string): string {
  let result = text;

  for (const [technicalName, businessName] of Object.entries(RULE_NAME_REPLACEMENTS)) {
    result = result.replaceAll(technicalName, businessName);
  }

  for (const [ruleId, businessName] of Object.entries(RULE_BUSINESS_NAMES)) {
    result = result.replaceAll(ruleId, businessName);
  }

  return result
    .replaceAll('医保结算风险信号引擎（op-rule-pool-v0.1）', '费用与就诊异常筛查')
    .replaceAll('mediredata-baseline-v0.1', '当前统计基线')
    .replaceAll('[risk_signal]', '风险线索')
    .replaceAll('[data_quality_or_applicability]', '数据质量提示')
    .replaceAll('[strong_review_signal]', '重点核验线索')
    .replaceAll('固定规则池', '医保业务核验规则')
    .replaceAll('规则池', '业务核验规则')
    .replaceAll('规则证据', '业务基础依据')
    .replaceAll('op-rule-pool-v0.1', '当前业务核验口径')
    .replaceAll('高优先级人工复核', '建议优先核验')
    .replaceAll('优先完成人工复核', '建议优先核验')
    .replaceAll('进入人工复核', '建议人工核验')
    .replaceAll('人工复核', '人工核验')
    .replaceAll('强复核', '重点核验')
    .replaceAll('抽样复核', '抽样核验')
    .replaceAll('复核线索', '核验线索')
    .replaceAll('风险规则', '业务线索')
    .replaceAll('model_signal:operational:当前业务核验口径', '风险筛查留痕')
    .replaceAll('model_signal:operational:op-rule-pool-v0.1', '风险筛查留痕')
    .replace(/rule:就诊频次核验:[\w-]+/g, '就诊频次核验留痕')
    .replace(/rule:多机构就诊核验:[\w-]+/g, '多机构就诊核验留痕')
    .replace(/rule:费用金额核验:[\w-]+/g, '费用金额核验留痕')
    .replace(/rule:药品费用结构核验:[\w-]+/g, '药品费用结构核验留痕')
    .replace(/rule:治疗项目费用核验:[\w-]+/g, '治疗项目费用核验留痕')
    .replace(/rule:医保支付结构核验:[\w-]+/g, '医保支付结构核验留痕')
    .replace(/rule:挂号与就诊流程核验:[\w-]+/g, '挂号与就诊流程核验留痕')
    .replace(/rule:数据完整性与适用范围核验:[\w-]+/g, '数据完整性与适用范围核验留痕')
    .replace(/rule:OP-R001:[\w-]+/g, '就诊行为一致性核验留痕')
    .replace(/rule:OP-R002:[\w-]+/g, '费用与支付结构核验留痕')
    .replace(/rule:OP-R003:[\w-]+/g, '药品费用合理性核验留痕')
    .replace(/rule:OP-R004:[\w-]+/g, '检查治疗结构核验留痕')
    .replace(/rule:OP-R005:[\w-]+/g, '挂号与流程一致性核验留痕')
    .replace(/rule:OP-R006:[\w-]+/g, '待遇补助口径核验留痕')
    .replace(/rule:OP-R007:[\w-]+/g, '申报数据质量核验留痕')
    .replace(/rule:OP-R008:[\w-]+/g, '材料补充核验留痕');
}

export function evidenceMainText(text: string): string {
  return businessText(text)
    .replace(/风险提示强度 ([0-9.]+)（([^)]+)），/g, '风险提示强度 $1，')
    .replace(/.*当前业务核验口径 基于聚合基线 当前统计基线 完成 8 类.*核验。/g, '已完成本案费用、就诊频次、机构和挂号流程等业务核验。')
    .replaceAll('现行审核口径 基于聚合基线 当前统计基线 完成 8 类医保业务核验规则。', '已完成本案费用、就诊频次、机构和挂号流程等业务核验。')
    .replaceAll('当前业务核验口径 基于聚合基线 当前统计基线 完成 8 类医保业务核验规则。', '已完成本案费用、就诊频次、机构和挂号流程等业务核验。')
    .replaceAll('医保业务核验规则命中', '业务核验发现')
    .replaceAll('数据质量/适用范围', '数据质量或适用范围')
    .replaceAll('证据引用：', '来源留痕：')
    .replaceAll('建议动作：建议人工核验', '建议动作：核对相关材料')
    .replaceAll('建议动作：要求补充材料', '建议动作：补充材料后再核验')
    .replaceAll('建议动作：标记数据质量问题', '建议动作：先确认数据完整性')
    .replace(/；核验口径：[^；。]+/g, '');
}
