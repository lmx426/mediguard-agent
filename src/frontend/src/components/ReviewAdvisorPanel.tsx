import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Button,
  Checkbox,
  Collapse,
  Descriptions,
  Drawer,
  Progress,
  Space,
  Tag,
  Typography,
  message,
} from 'antd';
import {
  fetchCurrentReviewAdvisorAnalysis,
  fetchReviewAdvisorRun,
  fetchLatestReviewAdvisorRun,
  startReviewAdvisorRun,
  subscribeReviewAdvisorEvents,
} from '../services/api';
import type {
  CaseSummary,
  ReviewAdvisorAnalysis,
  ReviewAdvisorCitation,
  ReviewAdvisorClueReview,
  ReviewAdvisorEvidenceReview,
  ReviewAdvisorEvent,
  ReviewAdvisorRun,
  ReviewAdvisorSignalReview,
  ReviewAdvisorStatus,
  ReviewAdvisorStatement,
  ReviewAdvisorSystemRiskPrompt,
  ReviewAdvisorVerificationItem,
  RiskScoreBreakdown,
} from '../types';

interface Props {
  caseId: string;
  riskLevel: CaseSummary['risk_level'];
  riskScore: number;
  riskScoreBreakdown?: RiskScoreBreakdown | null;
  status: ReviewAdvisorStatus;
}

const terminal = new Set(['complete', 'partial', 'failed']);
const relationOrder = ['risk_score_related', 'scan_discovered'] as const;
const priorityOrder = { high: 0, medium: 1, low: 2 } as const;
type ActionRelationType = (typeof relationOrder)[number];
type ActionPriority = keyof typeof priorityOrder;
type SignalReviewViewModel = {
  supported_clues: ReviewAdvisorStatement[];
  insufficient_clues: ReviewAdvisorStatement[];
};

function riskLevelLabel(value: CaseSummary['risk_level']) {
  const labels: Record<CaseSummary['risk_level'], string> = {
    high: '高风险',
    medium: '中风险',
    low: '低风险',
    insufficient: '证据不足',
  };
  return labels[value] ?? value;
}

function riskValueClass(value: CaseSummary['risk_level']) {
  return `is-risk-${value}`;
}

function evidenceRelationClass(value: ReviewAdvisorEvidenceReview['relation']) {
  return `is-relation-${value.replaceAll('_', '-')}`;
}

function eventLabel(event?: ReviewAdvisorEvent) {
  if (!event) return '等待发起研判建议';
  const labels: Record<string, string> = {
    run_created: '已创建研判建议任务',
    base_loaded: '已读取案件基础证据包',
    planning: '正在规划核验路径',
    tool_running: '正在调取只读核验工具',
    ledger_updated: '正在汇总证据来源',
    evidence_saturated: '证据收集已完成',
    generating: '正在生成研判建议',
    validating: '正在校验引用和安全边界',
    repairing: '正在修正研判内容',
    repairing_structure: '正在修正输出结构',
    complete: '研判建议已完成',
    partial: '研判建议部分完成',
    failed: '研判建议生成失败',
  };
  return labels[event.event_type] ?? event.message;
}

function progressPercent(run: ReviewAdvisorRun | null, event?: ReviewAdvisorEvent) {
  if (!run) return 0;
  if (run.status === 'failed') return 100;
  if (run.status === 'complete' || run.status === 'partial') return 100;
  const values: Record<string, number> = {
    run_created: 12,
    base_loaded: 24,
    planning: 36,
    tool_running: 50,
    ledger_updated: 64,
    evidence_saturated: 72,
    generating: 82,
    repairing: 86,
    repairing_structure: 86,
    validating: 92,
  };
  return values[event?.event_type ?? ''] ?? (run.status === 'queued' ? 10 : 24);
}

function uniqueRefs(refs: string[]) {
  return Array.from(new Set(refs.filter(Boolean)));
}

function mergeActions(
  first: ReviewAdvisorVerificationItem[],
  second: ReviewAdvisorVerificationItem[],
) {
  const seen = new Set<string>();
  return [...first, ...second].filter((item) => {
    const key = `${item.title.trim()}|${item.action.trim()}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function actionItemKey(item: ReviewAdvisorVerificationItem) {
  return `${item.title.trim()}|${item.action.trim()}`;
}

function normalizeRelationType(
  item: ReviewAdvisorVerificationItem,
  citations: Map<string, ReviewAdvisorCitation>,
): ActionRelationType {
  if (item.relation_type === 'risk_score_related' || item.relation_type === 'scan_discovered') {
    return item.relation_type;
  }
  const riskRelatedSourceTypes = new Set([
    'risk_breakdown',
    'risk_component',
    'model_warning',
    'rule_result',
    'rule_detail',
    'base_evidence',
  ]);
  return item.source_refs.some((ref) => {
    const citation = citations.get(ref);
    return ref.startsWith('risk:')
      || ref.startsWith('rule:')
      || ref.startsWith('fraud_screening:')
      || (citation ? riskRelatedSourceTypes.has(citation.source_type) : false);
  }) ? 'risk_score_related' : 'scan_discovered';
}

function normalizePriority(item: ReviewAdvisorVerificationItem): ActionPriority {
  if (item.priority === 'high' || item.priority === 'medium' || item.priority === 'low') {
    return item.priority;
  }
  return 'medium';
}

function caseReferencePrefix(caseId: string) {
  const normalized = caseId.trim();
  const match = normalized.match(/([A-Za-z0-9]+)$/);
  return (match?.[1] || normalized).slice(-8).toUpperCase();
}

function readCheckedActions(storageKey: string) {
  try {
    return new Set(JSON.parse(window.localStorage.getItem(storageKey) || '[]') as string[]);
  } catch {
    return new Set<string>();
  }
}

function statementItemKey(item: ReviewAdvisorStatement) {
  return `${item.statement.trim()}|${item.source_refs.join(',')}`;
}

function uniqueStatements(items: ReviewAdvisorStatement[]) {
  const seen = new Set<string>();
  return items.filter((item) => {
    const key = statementItemKey(item);
    if (!item.statement.trim() || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function actionToStatement(item: ReviewAdvisorVerificationItem): ReviewAdvisorStatement {
  return {
    statement: `${item.title}：${item.rationale || item.action}`,
    source_refs: item.source_refs,
  };
}

function hasSignalReviewContent(review?: ReviewAdvisorSignalReview | null) {
  return Boolean(
    review?.case_review_hint?.trim()
    || review?.supported_clues?.length
    || review?.needs_review?.length
    || review?.unconfirmed_items?.length
    || review?.supplementary_review_hints?.length
  );
}

function clueReviewToStatement(item: ReviewAdvisorClueReview): ReviewAdvisorStatement {
  return {
    statement: `${item.title}：${item.explanation}`,
    source_refs: item.source_refs,
  };
}

function normalizeSignalReview(analysis: ReviewAdvisorAnalysis): SignalReviewViewModel {
  if (analysis.clue_reviews?.length) {
    const insufficientStatuses = new Set(['needs_review', 'unconfirmed']);
    return {
      supported_clues: analysis.clue_reviews
        .filter((item) => item.status === 'supported')
        .map(clueReviewToStatement),
      insufficient_clues: analysis.clue_reviews
        .filter((item) => insufficientStatuses.has(item.status))
        .map(clueReviewToStatement),
    };
  }

  if (hasSignalReviewContent(analysis.signal_review)) {
    const fallbackRefs = uniqueRefs([
      ...analysis.risk_overview_source_refs,
      ...analysis.evidence_strength_source_refs,
    ]);
    return {
      supported_clues: uniqueStatements(analysis.signal_review?.supported_clues || []),
      insufficient_clues: uniqueStatements([
        ...analysis.signal_review?.needs_review || [],
        ...uniqueRefs(analysis.signal_review?.unconfirmed_items || []).map((statement) => ({
          statement,
          source_refs: fallbackRefs,
        })),
      ]),
    };
  }

  const supportedClues = uniqueStatements(analysis.supporting_evidence).slice(0, 3);
  const actionHints = uniqueStatements([
    ...analysis.human_review_focus.map(actionToStatement),
    ...analysis.verification_checklist.map(actionToStatement),
  ]);
  const needsReview = uniqueStatements([
    ...analysis.conflicts,
    ...actionHints,
  ]).slice(0, 3);
  const hintRefs = uniqueRefs([
    ...analysis.risk_overview_source_refs,
    ...analysis.evidence_strength_source_refs,
    ...supportedClues.flatMap((item) => item.source_refs),
  ]).slice(0, 8);
  return {
    supported_clues: supportedClues,
    insufficient_clues: uniqueStatements([
      ...needsReview,
      ...uniqueRefs(analysis.missing_information).slice(0, 3).map((statement) => ({
        statement,
        source_refs: hintRefs,
      })),
    ]),
  };
}

function normalizeSystemRiskPrompt(
  analysis: ReviewAdvisorAnalysis,
  riskLevel: CaseSummary['risk_level'],
  riskScore: number,
  breakdown?: RiskScoreBreakdown | null,
): ReviewAdvisorSystemRiskPrompt {
  if (analysis.system_risk_prompt) return analysis.system_risk_prompt;
  const score = breakdown?.total_score ?? Math.round(riskScore * 100);
  return {
    generated_by: 'backend_risk_engine',
    risk_level: riskLevel,
    risk_level_label: breakdown?.level_label ?? riskLevelLabel(riskLevel),
    risk_score: score,
    score_text: breakdown?.display_score ?? `${score} / 100`,
    source_summary: breakdown?.components.map((item) => item.label) ?? [],
    source_refs: [],
  };
}

function normalizeEvidenceReview(analysis: ReviewAdvisorAnalysis): ReviewAdvisorEvidenceReview {
  if (analysis.evidence_review) return analysis.evidence_review;
  const relationByStrength: Record<string, Pick<ReviewAdvisorEvidenceReview, 'relation' | 'label'>> = {
    高: { relation: 'supports', label: '支持' },
    中: { relation: 'partially_supports', label: '部分支持' },
    低: { relation: 'weakly_supports', label: '支撑较弱' },
  };
  const relation = relationByStrength[analysis.evidence_strength]
    ?? { relation: 'insufficient_evidence' as const, label: '证据不足' };
  return {
    ...relation,
    support_level: analysis.evidence_strength === '高'
      || analysis.evidence_strength === '中'
      || analysis.evidence_strength === '低'
      ? analysis.evidence_strength
      : '证据不足',
    summary: analysis.risk_overview,
    source_refs: uniqueRefs([
      ...analysis.evidence_strength_source_refs,
      ...analysis.risk_overview_source_refs,
    ]),
  };
}

export default function ReviewAdvisorPanel({
  caseId,
  riskLevel,
  riskScore,
  riskScoreBreakdown,
  status,
}: Props) {
  const [analysis, setAnalysis] = useState<ReviewAdvisorAnalysis | null>(null);
  const [run, setRun] = useState<ReviewAdvisorRun | null>(null);
  const [events, setEvents] = useState<ReviewAdvisorEvent[]>([]);
  const [loading, setLoading] = useState(status.available);
  const [selectedCitation, setSelectedCitation] = useState<ReviewAdvisorCitation | null>(null);
  const closeStream = useRef<null | (() => void)>(null);
  const [messageApi, contextHolder] = message.useMessage();

  const refreshRun = useCallback(async (runId: string) => {
    const latest = await fetchReviewAdvisorRun(runId);
    setRun(latest);
    if (latest.analysis) setAnalysis(latest.analysis);
    if (terminal.has(latest.status)) closeStream.current?.();
  }, []);

  const watchRun = useCallback((runId: string) => {
    closeStream.current?.();
    closeStream.current = subscribeReviewAdvisorEvents(
      runId,
      (event) => {
        setEvents((current) => [...current.filter((item) => item.sequence !== event.sequence), event]);
        if (terminal.has(event.event_type)) void refreshRun(runId);
      },
      () => {
        closeStream.current?.();
        window.setTimeout(() => void refreshRun(runId), 800);
      },
    );
  }, [refreshRun]);

  useEffect(() => {
    closeStream.current?.();
    setAnalysis(null);
    setRun(null);
    setEvents([]);
    if (!status.available) {
      setLoading(false);
      return;
    }
    let active = true;
    setLoading(true);
    void (async () => {
      try {
        const current = await fetchCurrentReviewAdvisorAnalysis(caseId);
        if (!active) return;
        if (current) {
          setAnalysis(current);
          return;
        }

        const latest = await fetchLatestReviewAdvisorRun(caseId);
        if (!active || !latest) return;
        setRun(latest);
        if (latest.analysis) {
          setAnalysis(latest.analysis);
        }
        if (!terminal.has(latest.status)) {
          watchRun(latest.run_id);
        }
      } catch {
        // 研判建议失败时保留基础证据包；人工可按需重新生成。
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
      closeStream.current?.();
    };
  }, [caseId, status.available, watchRun]);

  const start = useCallback(async (options: { silent?: boolean } = {}) => {
    setLoading(true);
    setEvents([]);
    try {
      const created = await startReviewAdvisorRun(
        caseId,
        { analysis_type: 'comprehensive' },
      );
      setRun(created);
      if (created.analysis) {
        setAnalysis(created.analysis);
        if (!options.silent) messageApi.info('案件事实未变化，已复用现有研判建议。');
      } else {
        watchRun(created.run_id);
      }
    } catch (error) {
      messageApi.error(error instanceof Error ? error.message : '研判建议任务创建失败');
    } finally {
      setLoading(false);
    }
  }, [caseId, messageApi, watchRun]);

  const viewModel = useMemo(() => {
    if (!analysis) return null;
    const citationMap = new Map(analysis.citations.map((item) => [item.citation_id, item]));
    return {
      citationMap,
      systemRiskPrompt: normalizeSystemRiskPrompt(analysis, riskLevel, riskScore, riskScoreBreakdown),
      evidenceReview: normalizeEvidenceReview(analysis),
      signalReview: normalizeSignalReview(analysis),
      actionItems: mergeActions(
        analysis.human_review_focus,
        analysis.verification_checklist,
      ),
    };
  }, [analysis, riskLevel, riskScore, riskScoreBreakdown]);

  if (!status.available) {
    return (
      <Alert
        type="warning"
        showIcon
        message="研判建议当前不可用"
        description="配置、数据库迁移或模型密钥尚未就绪。基础证据包不受影响。"
      />
    );
  }

  const runInProgress = Boolean(run && !terminal.has(run.status));
  const actionLoading = loading || runInProgress;
  const currentEvent = events.at(-1);
  const casePrefix = caseReferencePrefix(caseId);
  const progressText = runInProgress
    ? eventLabel(currentEvent)
    : loading
      ? '正在检查研判建议结果'
      : '研判建议暂未生成';
  const startButtonText = run?.status === 'failed' ? '重新生成研判建议' : '生成研判建议';
  const startCard = status.showcase ? (
    <Alert
      type="warning"
      showIcon
      message="预生成研判内容暂未加载"
      description="公网只读演示不会创建实时模型任务，请刷新案件详情后重试。"
    />
  ) : (
    <section className="agent-evidence-start-card">
      <div className="agent-evidence-start-content">
        {actionLoading ? (
          <div className="agent-evidence-progress">
            <Progress
              percent={progressPercent(run, currentEvent)}
              showInfo={false}
              status="active"
              strokeColor="#1B65B9"
            />
            <Typography.Text type="secondary">
              {progressText}
            </Typography.Text>
          </div>
        ) : (
          <Button type="primary" onClick={() => void start()}>
            {startButtonText}
          </Button>
        )}
      </div>
    </section>
  );

  return (
    <div className="agent-evidence-stack">
      {contextHolder}
      {status.showcase && (
        <Alert
          className="showcase-agent-notice"
          type="info"
          showIcon
          message={analysis?.generated_notice ?? '预生成演示，未调用实时模型'}
          description={analysis?.boundary_notice ?? '内容来自当前案件的脱敏事实和确定性规则，只供面试展示。'}
        />
      )}
      {!analysis ? startCard : null}

      {analysis && viewModel && (
        <>
          <AgentSection
            title="案件研判"
            tone="overview"
          >
            <div className="agent-judgement-summary">
              <div className={`agent-system-risk-band ${riskValueClass(viewModel.systemRiskPrompt.risk_level)}`}>
                <div className="agent-judgement-label-row">
                  <Typography.Text className="agent-section-subtitle">系统综合风险提示</Typography.Text>
                  <Typography.Text className="agent-data-owner">系统计算</Typography.Text>
                </div>
                <Typography.Text className="agent-system-risk-value" strong>
                  {viewModel.systemRiskPrompt.risk_level_label} {viewModel.systemRiskPrompt.score_text}
                </Typography.Text>
                {viewModel.systemRiskPrompt.source_summary.length ? (
                  <Typography.Text className="agent-system-risk-sources">
                    来源：{viewModel.systemRiskPrompt.source_summary.join('、')}
                  </Typography.Text>
                ) : null}
              </div>
              <div className="agent-evidence-review-band">
                <Typography.Text className="agent-section-subtitle">证据复核结果</Typography.Text>
                <div className="agent-evidence-review-facts">
                  <div className="agent-overview-fact">
                    <Typography.Text>复核关系：</Typography.Text>
                    <Typography.Text
                      className={`agent-overview-value ${evidenceRelationClass(viewModel.evidenceReview.relation)}`}
                      strong
                    >
                      {viewModel.evidenceReview.label}
                    </Typography.Text>
                  </div>
                  <div className="agent-overview-fact">
                    <Typography.Text>结论支撑度：</Typography.Text>
                    <Typography.Text className="agent-overview-value is-support" strong>
                      {viewModel.evidenceReview.support_level}
                    </Typography.Text>
                  </div>
                </div>
                <div className="agent-overview-copy">
                  <Typography.Text className="agent-section-subtitle">复核说明</Typography.Text>
                  <Typography.Paragraph>{viewModel.evidenceReview.summary}</Typography.Paragraph>
                </div>
              </div>
            </div>
            <div className="agent-overview-signal-review">
              <Typography.Text className="agent-section-subtitle">线索复核</Typography.Text>
              <SignalReviewPanel
                review={viewModel.signalReview}
                citations={viewModel.citationMap}
                onOpen={setSelectedCitation}
              />
            </div>
          </AgentSection>

          <AgentSection
            title="待人工核验事项"
            tone="actions"
            description="已完成基础证据包范围内的全局只读扫描。优先展示与综合风险提示强度直接相关的人工核验项，同时展示扫描过程中发现的补充确认事项；未筛出需人工介入的正常项不在清单中展示。"
          >
            <ActionList
              items={viewModel.actionItems}
              citations={viewModel.citationMap}
              onOpen={setSelectedCitation}
              storageKey={`mediguard:evidence-agent:checked-actions:${caseId}`}
              casePrefix={casePrefix}
            />
          </AgentSection>
        </>
      )}

      <Drawer
        title="证据来源"
        open={Boolean(selectedCitation)}
        onClose={() => setSelectedCitation(null)}
        width={420}
      >
        {selectedCitation && (
          <Descriptions column={1} bordered size="small">
            <Descriptions.Item label="来源">{selectedCitation.label}</Descriptions.Item>
            <Descriptions.Item label="类型">{selectedCitation.source_type}</Descriptions.Item>
            <Descriptions.Item label="引用标识">{selectedCitation.source_ref}</Descriptions.Item>
            <Descriptions.Item label="版本">{selectedCitation.version || '未提供'}</Descriptions.Item>
            <Descriptions.Item label="当前值">{selectedCitation.current_value || '见来源摘要'}</Descriptions.Item>
            <Descriptions.Item label="核验口径">{selectedCitation.threshold || '未提供'}</Descriptions.Item>
          </Descriptions>
        )}
      </Drawer>
    </div>
  );
}

function AgentSection({
  title,
  description,
  extra,
  tone = 'default',
  children,
}: {
  title: string;
  description?: string;
  extra?: React.ReactNode;
  tone?: 'default' | 'overview' | 'signal-review' | 'actions' | 'evidence';
  children: React.ReactNode;
}) {
  return (
    <section className={`agent-analysis-section is-${tone}`}>
      <div className="agent-analysis-section-head">
        <div className="agent-analysis-section-title">
          <Typography.Title level={5}>{title}</Typography.Title>
          {description && <Typography.Text type="secondary">{description}</Typography.Text>}
        </div>
        {extra && <div className="agent-analysis-section-extra">{extra}</div>}
      </div>
      {children}
    </section>
  );
}

function SignalReviewPanel({
  review,
  citations,
  onOpen,
}: {
  review: SignalReviewViewModel;
  citations: Map<string, ReviewAdvisorCitation>;
  onOpen: (citation: ReviewAdvisorCitation) => void;
}) {
  return (
    <div className="agent-signal-review-panel">
      <Collapse
        className="agent-signal-review-collapse"
        defaultActiveKey={[]}
        items={[
          {
            key: 'supported',
            label: <Typography.Text strong>已支持</Typography.Text>,
            children: (
              <SignalStatementList
                emptyText="暂无已支持线索。"
                items={review.supported_clues}
                citations={citations}
                onOpen={onOpen}
              />
            ),
          },
          {
            key: 'insufficient',
            label: <Typography.Text strong>证据不足</Typography.Text>,
            children: (
              <SignalStatementList
                emptyText="暂无证据不足的线索。"
                items={review.insufficient_clues}
                citations={citations}
                onOpen={onOpen}
              />
            ),
          },
        ]}
      />
    </div>
  );
}

function SignalStatementList({
  title,
  emptyText,
  items,
  citations,
  onOpen,
}: {
  title?: string;
  emptyText: string;
  items: ReviewAdvisorStatement[];
  citations: Map<string, ReviewAdvisorCitation>;
  onOpen: (citation: ReviewAdvisorCitation) => void;
}) {
  return (
    <div className="agent-signal-review-block">
      {title && <Typography.Text strong>{title}</Typography.Text>}
      {items.length ? (
        <ul className="agent-signal-review-list">
          {items.map((item) => (
            <li key={statementItemKey(item)}>
              <Typography.Text>{item.statement}</Typography.Text>
              <CitationTags refs={item.source_refs} citations={citations} onOpen={onOpen} />
            </li>
          ))}
        </ul>
      ) : (
        <Typography.Text type="secondary">{emptyText}</Typography.Text>
      )}
    </div>
  );
}

function ActionList({
  items,
  citations,
  onOpen,
  storageKey,
  casePrefix,
}: {
  items: ReviewAdvisorVerificationItem[];
  citations: Map<string, ReviewAdvisorCitation>;
  onOpen: (citation: ReviewAdvisorCitation) => void;
  storageKey: string;
  casePrefix: string;
}) {
  const [checkedKeys, setCheckedKeys] = useState<Set<string>>(() => readCheckedActions(storageKey));
  const sortedItems = useMemo(() => {
    return items
      .map((item, originalIndex) => ({
        item,
        originalIndex,
        priority: normalizePriority(item),
        relationType: normalizeRelationType(item, citations),
      }))
      .sort((left, right) => {
        const relationDiff = relationOrder.indexOf(left.relationType) - relationOrder.indexOf(right.relationType);
        if (relationDiff !== 0) return relationDiff;
        const priorityDiff = priorityOrder[left.priority] - priorityOrder[right.priority];
        if (priorityDiff !== 0) return priorityDiff;
        return left.originalIndex - right.originalIndex;
      })
      .map((entry, index) => ({ ...entry, displayIndex: index + 1 }));
  }, [citations, items]);

  const groups = useMemo(() => ([
    {
      key: 'risk_score_related',
      title: '风险提示关联核验',
      items: sortedItems.filter((item) => item.relationType === 'risk_score_related'),
    },
    {
      key: 'scan_discovered',
      title: '全局扫描补充发现',
      items: sortedItems.filter((item) => item.relationType === 'scan_discovered'),
    },
  ]), [sortedItems]);

  useEffect(() => {
    const validKeys = new Set(items.map(actionItemKey));
    const stored = readCheckedActions(storageKey);
    const next = new Set([...stored].filter((key) => validKeys.has(key)));
    setCheckedKeys(next);
    window.localStorage.setItem(storageKey, JSON.stringify([...next]));
  }, [items, storageKey]);

  const updateChecked = (key: string, checked: boolean) => {
    setCheckedKeys((current) => {
      const next = new Set(current);
      if (checked) {
        next.add(key);
      } else {
        next.delete(key);
      }
      window.localStorage.setItem(storageKey, JSON.stringify([...next]));
      return next;
    });
  };

  if (!sortedItems.length) {
    return (
      <Typography.Text type="secondary">
        暂无需要人工查验、补充确认或冲突判断的核验事项。
      </Typography.Text>
    );
  }
  return (
    <div className="agent-action-groups">
      {groups.map((group) => group.items.length ? (
        <section className="agent-action-group" key={group.key}>
          <Typography.Text className="agent-section-subtitle">{group.title}</Typography.Text>
          <Collapse
            className="agent-action-item-collapse"
            defaultActiveKey={[]}
            items={group.items.map(({ item, displayIndex }) => {
              const key = actionItemKey(item);
              return {
                key: `${group.key}-${displayIndex}`,
                label: (
                  <div className="agent-action-collapse-label">
                    <Checkbox
                      aria-label={`标记已处理：${item.title}`}
                      checked={checkedKeys.has(key)}
                      onChange={(event) => updateChecked(key, event.target.checked)}
                      onClick={(event) => event.stopPropagation()}
                    />
                    <span className="agent-collapse-index">{casePrefix}-{String(displayIndex).padStart(2, '0')}</span>
                    <Typography.Text strong>{item.title}</Typography.Text>
                  </div>
                ),
                children: (
                  <div className="agent-action-content">
                    <div className="agent-action-detail">
                      <Typography.Text type="secondary">动作</Typography.Text>
                      <Typography.Paragraph>{item.action}</Typography.Paragraph>
                    </div>
                    <div className="agent-action-detail">
                      <Typography.Text type="secondary">原因</Typography.Text>
                      <Typography.Paragraph>{item.rationale}</Typography.Paragraph>
                    </div>
                    <CitationTags refs={item.source_refs} citations={citations} onOpen={onOpen} />
                  </div>
                ),
              };
            })}
          />
        </section>
      ) : null)}
    </div>
  );
}

function CitationTags({
  refs,
  citations,
  onOpen,
}: {
  refs: string[];
  citations: Map<string, ReviewAdvisorCitation>;
  onOpen: (citation: ReviewAdvisorCitation) => void;
}) {
  const unique = uniqueRefs(refs);
  if (!unique.length) return null;
  return (
    <Space size={[4, 4]} wrap className="agent-citation-tags">
      {unique.map((ref) => {
        const citation = citations.get(ref);
        return (
          <Tag key={ref} onClick={() => citation && onOpen(citation)} className={citation ? 'is-clickable' : ''}>
            {citation?.label || ref}
          </Tag>
        );
      })}
    </Space>
  );
}
