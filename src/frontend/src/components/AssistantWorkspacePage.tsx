import { ArrowRightOutlined, FileSearchOutlined } from '@ant-design/icons';
import {
  Button,
  Card,
  Descriptions,
  Empty,
  Grid,
  Space,
  Spin,
  Splitter,
  Typography,
} from 'antd';
import type { CaseFullResponse, CaseSummary } from '../types';
import CaseAgentPanel from './CaseAgentPanel';
import RiskTag from './RiskTag';

const REVIEW_STATUS_LABELS: Record<CaseSummary['review_status'], string> = {
  pending: '待初审',
  reviewed: '已初审',
};

interface AssistantWorkspacePageProps {
  cases: CaseSummary[];
  selectedCaseId: string | null;
  selectedCase: CaseFullResponse | null;
  loading: boolean;
  onSelectCase: (caseId: string) => void;
  onOpenCase: (caseId: string) => void;
}

function CaseSelector({
  cases,
  selectedCaseId,
  onSelectCase,
}: Pick<AssistantWorkspacePageProps, 'cases' | 'selectedCaseId' | 'onSelectCase'>) {
  return (
    <Card
      className="assistant-case-list-card"
      title="审核案件"
      extra={<Typography.Text type="secondary">{cases.length} 件</Typography.Text>}
    >
      {cases.length === 0 ? (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description="当前没有可选择的案件"
        />
      ) : (
        <div className="assistant-case-list" role="list">
          {cases.map((item) => (
            <button
              type="button"
              key={item.case_id}
              className={`assistant-case-item ${
                item.case_id === selectedCaseId ? 'is-selected' : ''
              }`}
              aria-pressed={item.case_id === selectedCaseId}
              onClick={() => onSelectCase(item.case_id)}
            >
              <span className="assistant-case-item-stack">
                <span className="assistant-case-item-head">
                  <RiskTag level={item.risk_level} />
                  <span
                    className={`review-status-indicator review-${
                      item.review_status ?? 'pending'
                    }`}
                  >
                    <span
                      className="review-status-dot"
                      aria-hidden="true"
                    />
                    <span>
                      {
                        REVIEW_STATUS_LABELS[
                          item.review_status ?? 'pending'
                        ]
                      }
                    </span>
                  </span>
                </span>
                <Typography.Text
                  className="assistant-case-item-title"
                  strong
                  title={item.case_title}
                >
                  {item.case_title}
                </Typography.Text>
                <span className="assistant-case-item-meta">
                  <Typography.Text
                    className="assistant-case-item-id"
                    type="secondary"
                    title={item.case_id}
                  >
                    {item.case_id}
                  </Typography.Text>
                  <Typography.Text className="assistant-case-item-score" strong>
                    {item.risk_score.toFixed(2)}
                  </Typography.Text>
                </span>
              </span>
            </button>
          ))}
        </div>
      )}
    </Card>
  );
}

function CaseContext({
  data,
  onOpenCase,
}: {
  data: CaseFullResponse | null;
  onOpenCase: (caseId: string) => void;
}) {
  return (
    <Card
      className="assistant-context-card"
      title={
        <Space size={8}>
          <FileSearchOutlined />
          <span>案件上下文</span>
        </Space>
      }
    >
      {data ? (
        <Space orientation="vertical" size={16} style={{ width: '100%' }}>
          <RiskTag level={data.case.risk_level} />
          <Descriptions
            column={1}
            size="small"
            items={[
              {
                key: 'case_id',
                label: '案件编号',
                children: data.case.case_id,
              },
              {
                key: 'case_title',
                label: '案件摘要',
                children: data.case.case_title,
              },
              {
                key: 'risk_score',
                label: '风险提示强度',
                children: data.case.risk_score_breakdown?.display_score ?? `${Math.round(data.case.risk_score * 100)} / 100`,
              },
              {
                key: 'rules',
                label: '规则线索',
                children: `${data.case.rule_hits.filter((rule) => rule.hit).length} 条`,
              },
              {
                key: 'notes',
                label: '工作笔记',
                children: `${data.notes.length} 条`,
              },
              {
                key: 'review',
                label: '审核状态',
                children: data.review ? '已初审' : '待初审',
              },
            ]}
          />
          <Typography.Text type="secondary">
            助手只能围绕当前所选案件建立上下文。
          </Typography.Text>
          <Button
            className="assistant-context-action"
            type="primary"
            block
            icon={<ArrowRightOutlined />}
            iconPosition="end"
            onClick={() => onOpenCase(data.case.case_id)}
          >
            进入审核详情
          </Button>
        </Space>
      ) : (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description="选择案件后显示只读上下文"
        />
      )}
    </Card>
  );
}

export default function AssistantWorkspacePage({
  cases,
  selectedCaseId,
  selectedCase,
  loading,
  onSelectCase,
  onOpenCase,
}: AssistantWorkspacePageProps) {
  const screens = Grid.useBreakpoint();
  const wide = screens.xl ?? window.innerWidth >= 1200;

  const selector = (
    <CaseSelector
      cases={cases}
      selectedCaseId={selectedCaseId}
      onSelectCase={onSelectCase}
    />
  );
  const agent = loading ? (
    <Card className="assistant-loading-card">
      <Spin description="加载案件上下文..." />
    </Card>
  ) : (
    <CaseAgentPanel data={selectedCase} />
  );
  const context = <CaseContext data={selectedCase} onOpenCase={onOpenCase} />;

  return (
    <main className="assistant-page">
      {wide ? (
        <Splitter className="assistant-workspace-splitter">
          <Splitter.Panel defaultSize={300} min={260} max={380}>
            <div className="assistant-pane-scroll">{selector}</div>
          </Splitter.Panel>
          <Splitter.Panel min={480}>
            <div className="assistant-pane-scroll">{agent}</div>
          </Splitter.Panel>
          <Splitter.Panel defaultSize={340} min={300} max={420}>
            <div className="assistant-pane-scroll">{context}</div>
          </Splitter.Panel>
        </Splitter>
      ) : (
        <div className="assistant-workspace-stack">
          {selector}
          {agent}
          {context}
        </div>
      )}
    </main>
  );
}
