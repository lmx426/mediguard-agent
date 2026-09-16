import { useEffect, useState } from 'react';
import { Card, Collapse, Segmented, Space, Tag, Typography } from 'antd';
import type {
  CaseDetail,
  EvidencePackage as EvidencePackageType,
  RuleEvidenceCard,
  RiskScoreComponent,
} from '../types';
import {
  evidenceMainText,
  ruleBusinessName,
  severityBusinessLabel,
} from '../utils/businessLabels';
import { fetchReviewAdvisorStatus } from '../services/api';
import type { ReviewAdvisorStatus } from '../types';
import ReviewAdvisorPanel from './ReviewAdvisorPanel';

interface EvidencePackageProps {
  caseId: string;
  caseInfo: CaseDetail;
  evidence: EvidencePackageType;
}

interface BaseClue {
  title: string;
  source: string;
  status: string;
  statusRank: number;
  severityRank: number;
  domainRank: number;
  sourceRank: number;
}

function uniqueMaterials(cards: RuleEvidenceCard[]) {
  return Array.from(
    new Set(cards.flatMap((card) => card.supplement_materials ?? [])),
  );
}

function riskLevelLabel(value: CaseDetail['risk_level']) {
  const labels: Record<CaseDetail['risk_level'], string> = {
    high: '高风险',
    medium: '中风险',
    low: '低风险',
    insufficient: '证据不足',
  };
  return labels[value] ?? value;
}

function riskDisplayScore(caseInfo: CaseDetail) {
  if (caseInfo.risk_score_breakdown?.display_score) return caseInfo.risk_score_breakdown.display_score;
  return `${Math.round(caseInfo.risk_score * 100)} / 100`;
}

function modelWarningLabel(caseInfo: CaseDetail) {
  return caseInfo.fraud_screening?.label || '未接入';
}

function riskValueClass(value: CaseDetail['risk_level']) {
  return `is-risk-${value}`;
}

function modelWarningClass(caseInfo: CaseDetail) {
  const result = caseInfo.fraud_screening?.result;
  const label = modelWarningLabel(caseInfo);
  if (result === 'suspected' || label === '有预警') return 'is-warning';
  if (result === 'not_suspected' || label === '无预警') return 'is-clear';
  return 'is-muted';
}

function ruleHitCount(caseInfo: CaseDetail) {
  return caseInfo.rule_hits.filter((rule) => rule.hit).length;
}

function materialPromptCount(materials: string[], missingInformation: string[]) {
  return new Set([...materials, ...missingInformation].filter(Boolean)).size;
}

const SAFE_FIELD_SUMMARY_FIELDS = [
  '月就诊天数_MAX',
  '月就诊天数_AVG',
  '月就诊医院数_AVG',
  '就诊次数_SUM',
  '月就诊次数_MAX',
  '月就诊次数_AVG',
  '月统筹金额_MAX',
  '月统筹金额_AVG',
  '月药品金额_MAX',
  '月药品金额_AVG',
  '医院_就诊天数_MAX',
  '医院_就诊天数_AVG',
  '医院_统筹金_MAX',
  '医院_统筹金_AVG',
  '医院_药品_MAX',
  '医院_药品_AVG',
  '个人账户金额_SUM',
  '统筹支付金额_SUM',
  'ALL_SUM',
  '可用账户报销金额_SUM',
  '药品费发生金额_SUM',
  '药品费自费金额_SUM',
  '药品费申报金额_SUM',
  '贵重药品发生金额_SUM',
  '中成药费发生金额_SUM',
  '检查费发生金额_SUM',
  '检查费申报金额_SUM',
  '治疗费发生金额_SUM',
  '治疗费申报金额_SUM',
  '医用材料发生金额_SUM',
  '一次性医用材料申报金额_SUM',
  '起付标准以上自负比例金额_SUM',
  '基本统筹基金支付金额_SUM',
  '非账户支付金额_SUM',
  '本次审批金额_SUM',
  '药品在总金额中的占比',
  '个人支付的药品占比',
  '检查总费用在总金额占比',
  '治疗费用在总金额占比',
];

const severityRank: Record<string, number> = {
  critical: 0,
  high: 1,
  medium: 2,
  low: 3,
  info: 4,
};

function statusRank(status: string) {
  if (status === '命中') return 0;
  if (status === '待核对') return 1;
  if (status === '提示') return 2;
  return 3;
}

function clueDomainRank(title: string) {
  if (title.includes('药品')) return 0;
  if (title.includes('就诊频次') || title.includes('就诊行为')) return 1;
  if (title.includes('挂号') || title.includes('流程')) return 2;
  if (title.includes('费用') || title.includes('支付')) return 3;
  if (title.includes('检查') || title.includes('治疗')) return 4;
  if (title.includes('数据')) return 5;
  return 9;
}

function ruleClueTitle(card: RuleEvidenceCard) {
  const name = ruleBusinessName(card);
  if (name.includes('药品')) return '药品费用结构异常';
  if (name.includes('就诊行为') || name.includes('就诊频次')) return '就诊频次偏高';
  if (name.includes('挂号') || name.includes('材料补充')) return '挂号流程材料缺口';
  if (name.includes('数据')) return '数据完整性或适用范围提示';
  if (name.includes('支付') || name.includes('费用')) return '费用与支付结构异常';
  if (name.includes('检查') || name.includes('治疗')) return '检查治疗结构异常';
  return `${name.replace(/核验$/, '')}提示`;
}

function ruleClueStatus(card: RuleEvidenceCard) {
  if (card.rule_id === 'OP-R006' || card.rule_id === 'OP-R007' || card.rule_id === 'OP-R008') return '待核对';
  if (card.severity === 'info') return '提示';
  return '命中';
}

function componentClue(component: RiskScoreComponent): BaseClue | null {
  if (component.score <= 0) return null;
  if (component.key === 'peer_deviation') {
    return makeBaseClue('就诊频次偏高', '同类偏离指标', '提示', 2, 3);
  }
  if (component.key === 'data_flow') {
    return makeBaseClue('挂号流程材料缺口', '数据完整性规则', '待核对', 3, 1);
  }
  if (component.key === 'model_warning') {
    return makeBaseClue('模型识别预警', '模型预警', '提示', 0, 2);
  }
  return null;
}

function makeBaseClue(
  title: string,
  source: string,
  status: string,
  sourceRank: number,
  severity: number,
): BaseClue {
  return {
    title,
    source,
    status,
    statusRank: statusRank(status),
    severityRank: severity,
    domainRank: clueDomainRank(title),
    sourceRank,
  };
}

function buildBaseClues(caseInfo: CaseDetail, ruleCards: RuleEvidenceCard[]) {
  const fromRules = ruleCards.map((card) => ({
    ...makeBaseClue(
      ruleClueTitle(card),
      `规则 ${card.rule_id}`,
      ruleClueStatus(card),
      1,
      severityRank[card.severity] ?? 4,
    ),
  }));
  const fromComponents = (caseInfo.risk_score_breakdown?.components ?? [])
    .map(componentClue)
    .filter((item): item is BaseClue => Boolean(item));
  const seen = new Set<string>();
  return [...fromRules, ...fromComponents]
    .sort((left, right) => (
      left.statusRank - right.statusRank
      || left.severityRank - right.severityRank
      || left.domainRank - right.domainRank
      || left.sourceRank - right.sourceRank
    ))
    .filter((item) => {
      const key = item.title;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
}

function safeFieldValue(caseInfo: CaseDetail, field: string) {
  const record = {
    ...caseInfo.input_features,
    ...(caseInfo.source_record ?? {}),
  };
  const value = record[field];
  if (value === undefined || value === null || value === '') return null;
  if (typeof value === 'number' && field.includes('占比')) {
    const normalized = value <= 1 ? value * 100 : value;
    return `${Number(normalized.toFixed(2))}%`;
  }
  if (typeof value === 'number') {
    return value.toLocaleString('zh-CN', {
      maximumFractionDigits: 2,
    });
  }
  return String(value);
}

function buildSafeFieldRows(caseInfo: CaseDetail) {
  return SAFE_FIELD_SUMMARY_FIELDS
    .map((field) => ({ field, value: safeFieldValue(caseInfo, field) }))
    .filter((item): item is { field: string; value: string } => item.value !== null);
}

function ruleResultStatus(rule: CaseDetail['rule_hits'][number]) {
  if (rule.hit) return '命中';
  if (rule.severity === 'info' || rule.layer === 'data_quality_or_applicability') return '提示';
  return '未命中';
}

function ruleDomainRank(rule: CaseDetail['rule_hits'][number]) {
  return clueDomainRank(ruleBusinessName(rule));
}

function sortedRuleRows(ruleHits: CaseDetail['rule_hits']) {
  return [...ruleHits].sort((left, right) => (
    statusRank(ruleResultStatus(left)) - statusRank(ruleResultStatus(right))
    || (severityRank[left.severity] ?? 4) - (severityRank[right.severity] ?? 4)
    || ruleDomainRank(left) - ruleDomainRank(right)
    || left.rule_id.localeCompare(right.rule_id)
  ));
}

function modelWarningReason(caseInfo: CaseDetail, evidence: EvidencePackageType) {
  return caseInfo.fraud_screening?.reason
    || caseInfo.model_signal_reasons.filter(Boolean).join('；')
    || evidence.model_evidence
    || '当前未返回模型预警说明。';
}

export default function EvidencePackage({ caseId, caseInfo, evidence }: EvidencePackageProps) {
  const [view, setView] = useState<'base' | 'agent'>('agent');
  const [agentStatus, setAgentStatus] = useState<ReviewAdvisorStatus | null>(null);
  const ruleCards = evidence.rule_cards ?? [];
  const missingInformation = evidence.missing_information ?? [];
  const materials = uniqueMaterials(ruleCards);
  const baseClues = buildBaseClues(caseInfo, ruleCards);
  const safeFieldRows = buildSafeFieldRows(caseInfo);
  const ruleRows = sortedRuleRows(caseInfo.rule_hits);
  const materialRows = [
    ...materials.map((item) => ({ type: '材料提示', text: item })),
    ...missingInformation.map((item) => ({ type: '缺失提示', text: item })),
  ];

  useEffect(() => {
    let active = true;
    fetchReviewAdvisorStatus()
      .then((status) => {
        if (active) setAgentStatus(status);
      })
      .catch(() => {
        if (active) setAgentStatus(null);
      });
    return () => {
      active = false;
    };
  }, []);

  return (
    <Card
      className="evidence-package-card"
      title="证据包与审核建议"
      extra={agentStatus?.enabled ? (
        <Segmented
          options={[
            { label: '基础证据包', value: 'base' },
            { label: '研判建议', value: 'agent' },
          ]}
          value={view}
          onChange={(value) => setView(value as 'base' | 'agent')}
        />
      ) : undefined}
    >
      {view === 'agent' && agentStatus?.enabled ? (
        <ReviewAdvisorPanel
          caseId={caseId}
          riskLevel={caseInfo.risk_level}
          riskScore={caseInfo.risk_score}
          riskScoreBreakdown={caseInfo.risk_score_breakdown}
          status={agentStatus}
        />
      ) : (
      <Space className="evidence-package-stack" orientation="vertical" size={18}>
        <section className="evidence-panel evidence-panel-package">
          <section className="evidence-checklist" aria-label="证据包清单">
            <div className="evidence-checklist-row">
              <Typography.Text className="evidence-checklist-label">基础摘要</Typography.Text>
              <div className="evidence-checklist-content">
                <div className="evidence-base-summary">
                  <div className="evidence-base-summary-line">
                    <Typography.Text type="secondary">综合风险提示强度：</Typography.Text>
                    <Typography.Text className={`evidence-base-summary-value ${riskValueClass(caseInfo.risk_level)}`} strong>
                      {riskLevelLabel(caseInfo.risk_level)} {riskDisplayScore(caseInfo)}
                    </Typography.Text>
                  </div>
                  <div className="evidence-base-summary-line">
                    <Typography.Text type="secondary">模型识别预警：</Typography.Text>
                    <Typography.Text className={`evidence-base-summary-value ${modelWarningClass(caseInfo)}`} strong>
                      {modelWarningLabel(caseInfo)}
                    </Typography.Text>
                  </div>
                  <div className="evidence-base-summary-line">
                    <Typography.Text type="secondary">规则命中：</Typography.Text>
                    <Typography.Text strong>{ruleHitCount(caseInfo)} 条</Typography.Text>
                  </div>
                  <div className="evidence-base-summary-line">
                    <Typography.Text type="secondary">材料提示：</Typography.Text>
                    <Typography.Text strong>{materialPromptCount(materials, missingInformation)} 项</Typography.Text>
                  </div>
                </div>
              </div>
            </div>

            <div className="evidence-checklist-row">
              <Typography.Text className="evidence-checklist-label">系统已发现线索</Typography.Text>
              <div className="evidence-checklist-content">
                {baseClues.length === 0 ? (
                  <Typography.Text type="secondary">当前未发现需要单独列示的线索。</Typography.Text>
                ) : (
                  <Collapse
                    className="evidence-base-clue-collapse"
                    defaultActiveKey={[]}
                    items={baseClues.map((clue) => {
                      const statusClass = clue.status === '命中'
                        ? 'is-hit'
                        : clue.status === '待核对'
                          ? 'is-pending'
                          : 'is-tip';
                      return {
                        key: `${clue.title}-${clue.source}`,
                        label: (
                          <div className="evidence-base-clue-collapse-label">
                            <Typography.Text strong>{evidenceMainText(clue.title)}</Typography.Text>
                            <Tag className={`evidence-base-status ${statusClass}`}>
                              状态：{clue.status}
                            </Tag>
                          </div>
                        ),
                        children: (
                          <div className="evidence-base-clue-detail">
                            <Typography.Text type="secondary">来源：{clue.source}</Typography.Text>
                          </div>
                        ),
                      };
                    })}
                  />
                )}
              </div>
            </div>

            <div className="evidence-checklist-row">
              <Typography.Text className="evidence-checklist-label">来源依据</Typography.Text>
              <div className="evidence-checklist-content">
                <Collapse
                  className="evidence-source-basis-collapse"
                  defaultActiveKey={[]}
                  items={[
                    {
                      key: 'model-warning',
                      label: <Typography.Text strong>模型预警</Typography.Text>,
                      children: (
                        <div className="evidence-source-basis-list">
                          <div className="evidence-source-basis-item">
                            <Typography.Text strong>模型识别预警：{modelWarningLabel(caseInfo)}</Typography.Text>
                            <Typography.Text type="secondary">
                              来源：{caseInfo.fraud_screening?.source || caseInfo.model_signal_source || '风险筛查留痕'}
                            </Typography.Text>
                            <Typography.Text type="secondary">
                              说明：{evidenceMainText(modelWarningReason(caseInfo, evidence))}
                            </Typography.Text>
                          </div>
                        </div>
                      ),
                    },
                    {
                      key: 'rule-results',
                      label: <Typography.Text strong>规则结果</Typography.Text>,
                      children: ruleRows.length === 0 ? (
                        <Typography.Text type="secondary">当前未返回规则核验结果。</Typography.Text>
                      ) : (
                        <div className="evidence-source-basis-list">
                          {ruleRows.map((rule) => (
                            <div className="evidence-source-basis-item" key={rule.rule_id}>
                              <div className="evidence-source-basis-title">
                                <Typography.Text strong>{ruleBusinessName(rule)}</Typography.Text>
                                <Tag className="evidence-source-basis-tag">{ruleResultStatus(rule)}</Tag>
                              </div>
                              <Typography.Text type="secondary">
                                来源：规则 {rule.rule_id}，{severityBusinessLabel(rule.severity)}
                              </Typography.Text>
                              <Typography.Text type="secondary">
                                说明：{evidenceMainText(rule.business_explanation || rule.reason)}
                              </Typography.Text>
                            </div>
                          ))}
                        </div>
                      ),
                    },
                    {
                      key: 'safe-fields',
                      label: <Typography.Text strong>安全字段摘要</Typography.Text>,
                      children: safeFieldRows.length === 0 ? (
                        <Typography.Text type="secondary">当前未返回可展示的安全字段摘要。</Typography.Text>
                      ) : (
                        <div className="evidence-source-field-grid">
                          {safeFieldRows.map((item) => (
                            <div className="evidence-source-field" key={item.field}>
                              <Typography.Text type="secondary">{item.field}</Typography.Text>
                              <Typography.Text strong>{item.value}</Typography.Text>
                            </div>
                          ))}
                        </div>
                      ),
                    },
                    {
                      key: 'material-summary',
                      label: <Typography.Text strong>统计材料摘要</Typography.Text>,
                      children: materialRows.length === 0 ? (
                        <Typography.Text type="secondary">当前暂无材料提示或缺失信息。</Typography.Text>
                      ) : (
                        <div className="evidence-source-basis-list">
                          {materialRows.map((item) => (
                            <div className="evidence-source-basis-item" key={`${item.type}-${item.text}`}>
                              <div className="evidence-source-basis-title">
                                <Typography.Text strong>{item.type}</Typography.Text>
                              </div>
                              <Typography.Text type="secondary">
                                {evidenceMainText(item.text)}
                              </Typography.Text>
                            </div>
                          ))}
                        </div>
                      ),
                    },
                  ]}
                />
              </div>
            </div>
          </section>
        </section>
      </Space>
      )}
    </Card>
  );
}
