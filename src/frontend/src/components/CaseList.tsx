import { List, Space, Tag, Typography } from 'antd';
import type { CaseSummary } from '../types';

interface CaseListProps {
  cases: CaseSummary[];
  selectedId: string | null;
  onSelect: (caseId: string) => void;
}

const RISK_LABELS: Record<string, string> = {
  low: '低风险',
  medium: '中风险',
  high: '高风险',
  insufficient: '证据不足',
};

const RISK_COLORS: Record<string, string> = {
  low: 'success',
  medium: 'warning',
  high: 'error',
  insufficient: 'default',
};

const REVIEW_STATUS_LABELS: Record<string, string> = {
  pending: '待初审',
  submitted: '已初审',
  reviewed: '已初审',
};

function formatCaseTitle(title: string) {
  return title.trim();
}

export default function CaseList({ cases, selectedId, onSelect }: CaseListProps) {
  return (
    <div className="case-list-panel">
      <Typography.Text type="secondary" className="case-list-heading">
        案件队列
      </Typography.Text>
      <List
        dataSource={cases}
        locale={{ emptyText: '暂无案件' }}
        renderItem={(item) => (
          <List.Item
            className="case-list-entry"
            onClick={() => onSelect(item.case_id)}
          >
            <div className={`case-list-card ${item.case_id === selectedId ? 'selected' : ''}`}>
              <Space orientation="vertical" size={8} style={{ width: '100%' }}>
                <div className="case-list-topline">
                  <Tag className="case-list-risk-tag" color={RISK_COLORS[item.risk_level]}>
                    {RISK_LABELS[item.risk_level] ?? item.risk_level}
                  </Tag>
                  <Tag className="case-list-review-tag">
                    {REVIEW_STATUS_LABELS[item.review_status ?? 'pending'] ?? '待初审'}
                  </Tag>
                </div>
                <Typography.Text strong className="case-list-title">
                  {formatCaseTitle(item.case_title)}
                </Typography.Text>
                <div className="case-list-bottomline">
                  <span className="case-list-score">
                    <Typography.Text type="secondary">风险提示强度</Typography.Text>
                    <Typography.Text strong>{Math.round(item.risk_score * 100)}</Typography.Text>
                  </span>
                  <Typography.Text type="secondary" className="case-list-id">
                    {item.case_id}
                  </Typography.Text>
                </div>
              </Space>
            </div>
          </List.Item>
        )}
      />
    </div>
  );
}
