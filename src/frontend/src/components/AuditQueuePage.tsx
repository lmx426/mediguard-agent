import { ArrowRightOutlined, SearchOutlined } from '@ant-design/icons';
import {
  Alert,
  Button,
  Card,
  Input,
  Pagination,
  Segmented,
  Select,
  Table,
  Typography,
} from 'antd';
import type { TableColumnsType } from 'antd';
import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import type { CaseSummary } from '../types';
import RiskTag from './RiskTag';

export type QueueStatusFilter = 'pending' | 'reviewed' | 'all';
export type QueueRiskFilter = 'all' | CaseSummary['risk_level'];

const QUEUE_PAGE_SIZE = 6;

interface AuditQueuePageProps {
  cases: CaseSummary[];
  filteredCases: CaseSummary[];
  statusFilter: QueueStatusFilter;
  riskFilter: QueueRiskFilter;
  keyword: string;
  highlightCaseId: string | null;
  batchCount: number | null;
  loading: boolean;
  onStatusChange: (status: QueueStatusFilter) => void;
  onRiskChange: (risk: QueueRiskFilter) => void;
  onKeywordChange: (keyword: string) => void;
  onOpenCase: (caseId: string) => void;
}

const REVIEW_STATUS_LABELS: Record<CaseSummary['review_status'], string> = {
  pending: '待初审',
  reviewed: '已初审',
};

export default function AuditQueuePage({
  cases,
  filteredCases,
  statusFilter,
  riskFilter,
  keyword,
  highlightCaseId,
  batchCount,
  loading,
  onStatusChange,
  onRiskChange,
  onKeywordChange,
  onOpenCase,
}: AuditQueuePageProps) {
  const pendingCount = cases.filter((item) => (item.review_status ?? 'pending') === 'pending').length;
  const reviewedCount = cases.filter((item) => item.review_status === 'reviewed').length;
  const [showBatchBanner, setShowBatchBanner] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(highlightCaseId);
  const hasActiveFilters = statusFilter !== 'pending' || riskFilter !== 'all' || keyword.length > 0;

  useEffect(() => {
    setShowBatchBanner(true);
  }, [batchCount]);

  useEffect(() => {
    if (highlightCaseId) {
      setSelectedCaseId(highlightCaseId);
    }
  }, [highlightCaseId]);

  useEffect(() => {
    setCurrentPage(1);
  }, [statusFilter, riskFilter, keyword]);

  useEffect(() => {
    const maxPage = Math.max(1, Math.ceil(filteredCases.length / QUEUE_PAGE_SIZE));
    setCurrentPage((page) => Math.min(page, maxPage));
  }, [filteredCases.length]);

  const pagedCases = filteredCases.slice(
    (currentPage - 1) * QUEUE_PAGE_SIZE,
    currentPage * QUEUE_PAGE_SIZE,
  );

  const resetFilters = () => {
    onStatusChange('pending');
    onRiskChange('all');
    onKeywordChange('');
    setCurrentPage(1);
  };

  const tableWidths = {
    caseId: '15%',
    title: '24%',
    riskLevel: '12%',
    riskScore: '13%',
    ruleSignals: '11%',
    reviewStatus: '12%',
    action: '13%',
  };

  const batchBanner =
    batchCount && batchCount > 0 && showBatchBanner
      ? createPortal(
          <div className="audit-batch-banner" role="status" aria-live="polite">
            <Alert
              className="audit-batch-alert"
              type="info"
              showIcon
              closable
              onClose={() => setShowBatchBanner(false)}
              message={`本批次处理 ${batchCount} 件案件`}
              description="记录已通过后端完整校验；完全重复记录会复用已有案件，不会覆盖原审核结果。"
            />
          </div>,
          document.body,
        )
      : null;

  const columns: TableColumnsType<CaseSummary> = [
    {
      title: '案件编号',
      dataIndex: 'case_id',
      width: tableWidths.caseId,
      responsive: ['md'],
      render: (value: string) => (
        <Typography.Text className="queue-case-id">{value}</Typography.Text>
      ),
    },
    {
      title: '案件摘要',
      dataIndex: 'case_title',
      width: tableWidths.title,
      ellipsis: true,
      render: (value: string, record) => (
        <span className="queue-case-summary-block">
          <Typography.Text className="queue-case-summary">{value}</Typography.Text>
          <Typography.Text className="queue-case-summary-id">{record.case_id}</Typography.Text>
        </span>
      ),
    },
    {
      title: '风险等级',
      dataIndex: 'risk_level',
      width: tableWidths.riskLevel,
      render: (value: CaseSummary['risk_level']) => <RiskTag level={value} />,
    },
    {
      title: '风险提示强度',
      dataIndex: 'risk_score',
      width: tableWidths.riskScore,
      className: 'audit-risk-score-cell',
      responsive: ['md'],
      sorter: (a, b) => a.risk_score - b.risk_score,
      defaultSortOrder: 'descend',
      render: (value: number) => (
        <Typography.Text className="queue-risk-score">{Math.round(value * 100)}</Typography.Text>
      ),
    },
    {
      title: '规则线索数',
      dataIndex: 'rule_signal_count',
      width: tableWidths.ruleSignals,
      responsive: ['lg'],
      render: (value: number) => (
        <Typography.Text className="queue-rule-count">{value ?? 0}</Typography.Text>
      ),
    },
    {
      title: '审核状态',
      dataIndex: 'review_status',
      width: tableWidths.reviewStatus,
      responsive: ['md'],
      render: (value: CaseSummary['review_status'] | undefined) => {
        const status = value ?? 'pending';
        return (
          <span className={`review-status-indicator review-${status}`}>
            <span className="review-status-dot" aria-hidden="true" />
            <span>{REVIEW_STATUS_LABELS[status]}</span>
          </span>
        );
      },
    },
    {
      title: '操作',
      key: 'action',
      width: tableWidths.action,
      render: (_, record) => {
        const status = record.review_status ?? 'pending';

        return (
          <Button
            className={`queue-review-button ${
              status === 'reviewed' ? 'queue-review-button-view' : 'queue-review-button-audit'
            }`}
            icon={<ArrowRightOutlined />}
            type={status === 'reviewed' ? 'text' : 'default'}
            onClick={(event) => {
              event.stopPropagation();
              onOpenCase(record.case_id);
            }}
          >
            {status === 'reviewed' ? '查看详情' : '审核'}
          </Button>
        );
      },
    },
  ];

  return (
    <>
      {batchBanner}
      <main className="audit-page">
        <div className="audit-page-stack">
          <Card className="audit-filter-card">
            <div className="audit-queue-filters">
              <div className="audit-status-filter">
                <Typography.Text className="audit-filter-label">审核状态</Typography.Text>
                <Segmented
                  value={statusFilter}
                  onChange={(value) => onStatusChange(value as QueueStatusFilter)}
                  options={[
                    { label: `待初审 ${pendingCount}`, value: 'pending' },
                    { label: `已初审 ${reviewedCount}`, value: 'reviewed' },
                    { label: `全部 ${cases.length}`, value: 'all' },
                  ]}
                />
              </div>
              <div className="audit-secondary-filters">
                <Select
                  className="audit-risk-filter"
                  value={riskFilter}
                  onChange={(value) => onRiskChange(value as QueueRiskFilter)}
                  options={[
                    { value: 'all', label: '全部风险等级' },
                    { value: 'high', label: '高风险' },
                    { value: 'medium', label: '中风险' },
                    { value: 'low', label: '低风险' },
                    { value: 'insufficient', label: '证据不足' },
                  ]}
                />
                <Input
                  className="audit-keyword-input"
                  allowClear
                  prefix={<SearchOutlined />}
                  placeholder="搜索案件编号或摘要"
                  value={keyword}
                  onChange={(event) => onKeywordChange(event.target.value)}
                />
                <Button
                  className="audit-filter-reset"
                  disabled={!hasActiveFilters}
                  type="link"
                  onClick={resetFilters}
                >
                  重置
                </Button>
              </div>
            </div>
          </Card>

          <Card className="audit-queue-card" title="案件列表">
            <Table<CaseSummary>
              rowKey="case_id"
              className="audit-queue-table"
              loading={loading}
              columns={columns}
              dataSource={pagedCases}
              size="middle"
              tableLayout="fixed"
              locale={{
                emptyText: (
                  <div className="audit-empty-placeholder" aria-label="暂无案件">
                    <Typography.Text type="secondary">
                      没有符合当前筛选条件的案件
                    </Typography.Text>
                  </div>
                ),
              }}
              pagination={false}
              onRow={(record) => ({
                'aria-selected': record.case_id === selectedCaseId,
                onClick: () => setSelectedCaseId(record.case_id),
              })}
              rowClassName={(record) =>
                record.case_id === selectedCaseId
                  ? 'audit-table-row audit-row-selected'
                  : 'audit-table-row'
              }
            />
            <div
              className={
                filteredCases.length === 0
                  ? 'audit-queue-footer audit-queue-footer-empty'
                  : 'audit-queue-footer'
              }
            >
              <Typography.Text className="audit-queue-total">
                共 {filteredCases.length} 件
              </Typography.Text>
              <Pagination
                current={currentPage}
                pageSize={QUEUE_PAGE_SIZE}
                showSizeChanger={false}
                total={filteredCases.length}
                onChange={setCurrentPage}
              />
            </div>
          </Card>
        </div>
      </main>
    </>
  );
}
