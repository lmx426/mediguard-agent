import {
  ArrowRightOutlined,
  CloudUploadOutlined,
  DeleteOutlined,
  FileTextOutlined,
  LockOutlined,
} from '@ant-design/icons';
import {
  Alert,
  Button,
  Card,
  Empty,
  Select,
  Spin,
  Tabs,
  Tag,
  Table,
  Typography,
  Upload,
} from 'antd';
import type { TableColumnsType, UploadFile, UploadProps } from 'antd';
import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  fetchIngestRecord,
  fetchVisitorIngestRecords,
  pushIngestRecord,
  submitIngestBatch,
} from '../services/api';
import type {
  CaseFullResponse,
  IngestRecordInput,
  IngestRecordResponse,
  IngestRecordSummary,
} from '../types';

interface IngestPageProps {
  readOnly?: boolean;
  onCasesGenerated: (cases: CaseFullResponse[]) => void;
}

type VisitorSampleGroup = 'abnormal' | 'normal';

interface VisitorSampleOption {
  recordId: string;
  label: string;
  shortLabel: string;
  categoryLabel: string;
  group: VisitorSampleGroup;
}

interface FieldRow {
  field: string;
  value: string;
}

function sampleOptionLabel(option: VisitorSampleOption) {
  const scenarioText = option.group === 'normal'
    ? option.shortLabel
    : option.categoryLabel
    ? `${option.categoryLabel} · ${option.shortLabel}`
    : option.shortLabel;
  return (
    <span className="ingest-sample-option">
      <span className="ingest-sample-option-code">{option.recordId}</span>
      <span className="ingest-sample-option-title">{scenarioText}</span>
    </span>
  );
}

function sampleCategoryLabel(category?: string | null): string {
  const value = category?.trim();
  if (!value) return '';
  return (value.split('：')[0] || value).trim();
}

function shortSampleLabel(label: string): string {
  const replacements: Array<[string, string]> = [
    ['异地门急诊手工报销，备案状态不明', '异地手工报销'],
    ['申请急诊但材料缺少急诊标识', '急诊材料不足'],
    ['上海侧预估比例不能直接作为北京手工报销待遇', '待遇目录混用'],
    ['线上备案即时生效与旧线下备案表提示冲突', '政策版本冲突'],
    ['检查费占比异常，CT相关收费待核验', '检查费异常'],
    ['药品费占比异常，目录限定支付与诊断匹配待核验', '药品费异常'],
    ['存在费用和审批金额但挂号状态为 0', '挂号链条异常'],
    ['高频多机构慢病取药，重复购药核验点', '高频多机构'],
    ['异地门急诊手工报销材料齐全', '异地材料齐全'],
    ['普通门诊材料齐全', '普通门诊齐全'],
  ];
  const hit = replacements.find(([source]) => label.includes(source));
  return hit?.[1] ?? label.replace(/^正常样本：/, '').replace(/^异常样本：/, '');
}

function buildVisitorOptions(records: IngestRecordSummary[]): VisitorSampleOption[] {
  return records.map((record) => ({
    recordId: record.record_id,
    label: record.sample_label || record.subject_ref || record.record_id,
    shortLabel: shortSampleLabel(record.sample_label || record.sample_category || record.record_id),
    categoryLabel: sampleCategoryLabel(record.sample_category),
    group: record.sample_result === 'normal' ? 'normal' : 'abnormal',
  }));
}

function buildGroupedOptions(options: VisitorSampleOption[]) {
  return [
    {
      label: (
        <span className="ingest-sample-select-group-label">
          <span className="ingest-sample-select-group-title">异常样本</span>
        </span>
      ),
      options: options.filter((option) => option.group === 'abnormal').map((option) => ({
        value: option.recordId,
        label: sampleOptionLabel(option),
      })),
    },
    {
      label: (
        <span className="ingest-sample-select-group-label is-normal">
          <span className="ingest-sample-select-group-title">正常样本</span>
        </span>
      ),
      options: options.filter((option) => option.group === 'normal').map((option) => ({
        value: option.recordId,
        label: sampleOptionLabel(option),
      })),
    },
  ].filter((group) => group.options.length > 0);
}

interface UploadPreview {
  fileName: string;
  rowCount: number;
}

type UploadState = 'empty' | 'ready' | 'error';

function parseCsvText(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = '';
  let inQuotes = false;
  const normalized = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');

  for (let index = 0; index < normalized.length; index += 1) {
    const char = normalized[index];

    if (inQuotes) {
      if (char === '"') {
        if (normalized[index + 1] === '"') {
          cell += '"';
          index += 1;
        } else {
          inQuotes = false;
        }
      } else {
        cell += char;
      }
      continue;
    }

    if (char === '"') {
      inQuotes = true;
      continue;
    }
    if (char === ',') {
      row.push(cell);
      cell = '';
      continue;
    }
    if (char === '\n') {
      row.push(cell);
      rows.push(row);
      row = [];
      cell = '';
      continue;
    }
    cell += char;
  }

  if (inQuotes) {
    throw new Error('CSV 报表存在未闭合的引号');
  }

  row.push(cell);
  rows.push(row);

  return rows.filter((item) => item.some((value) => value.trim()));
}

function parseCsvToIngestRecords(text: string): IngestRecordInput[] {
  const rows = parseCsvText(text);
  if (rows.length < 2) {
    throw new Error('CSV 报表至少需要表头和 1 行数据');
  }

  const [rawHeader, ...dataRows] = rows;
  const header = rawHeader.map((cell, index) =>
    (index === 0 ? cell.replace(/^\uFEFF/, '') : cell).trim(),
  );
  const headerSet = new Set(header);

  if (headerSet.size !== header.length) {
    throw new Error('CSV 报表表头存在重复字段');
  }
  if (header.some((field) => !field)) {
    throw new Error('CSV 报表表头存在空字段');
  }

  return dataRows.map((row, index) => {
    if (row.length !== header.length) {
      throw new Error(`第 ${index + 2} 行列数为 ${row.length}，应为 ${header.length}`);
    }

    const record: Record<string, string> = {};
    header.forEach((field, fieldIndex) => {
      record[field] = row[fieldIndex] ?? '';
    });
    return { record };
  });
}

function formatRecordValue(value: number | string | undefined): string {
  if (value === undefined || value === null || value === '') return '-';
  return String(value);
}

function fieldRowsFromRecord(record: IngestRecordResponse | null): FieldRow[] {
  if (!record) return [];
  return Object.entries(record.record).map(([field, value]) => ({
    field,
    value: formatRecordValue(value),
  }));
}

export default function IngestPage({
  readOnly = false,
  onCasesGenerated,
}: IngestPageProps) {
  const [visitorSampleOptions, setVisitorSampleOptions] = useState<VisitorSampleOption[]>(
    [],
  );
  const [sampleListLoading, setSampleListLoading] = useState(true);
  const [selectedRecordId, setSelectedRecordId] = useState('');
  const [selectedRecord, setSelectedRecord] = useState<IngestRecordResponse | null>(
    null,
  );
  const [recordLoading, setRecordLoading] = useState(true);
  const [sampleSubmitting, setSampleSubmitting] = useState(false);
  const [sampleError, setSampleError] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [uploadPreview, setUploadPreview] = useState<UploadPreview | null>(null);
  const [batchRecords, setBatchRecords] = useState<IngestRecordInput[]>([]);
  const [batchSubmitting, setBatchSubmitting] = useState(false);

  const uploadState: UploadState = uploadError
    ? 'error'
    : uploadPreview
      ? 'ready'
      : 'empty';
  const activeUploadFileName = fileList[0]?.name ?? '等待 CSV 报表';
  const activeUploadRowCount = uploadPreview ? `${uploadPreview.rowCount} 条` : '-';
  const activeUploadResult =
    uploadState === 'error'
      ? '上传内容存在问题'
      : uploadState === 'ready'
        ? '可进入整批校验'
        : '等待上传';
  const activeUploadTag =
    uploadState === 'error'
      ? { color: 'error' as const, text: '校验失败' }
      : uploadState === 'ready'
        ? { color: 'processing' as const, text: '待后端整批校验' }
        : { color: 'default' as const, text: '未上传' };

  const resetUploadState = useCallback(() => {
    setFileList([]);
    setBatchRecords([]);
    setUploadPreview(null);
    setUploadError(null);
  }, []);

  const groupedSampleOptions = useMemo(
    () => buildGroupedOptions(visitorSampleOptions),
    [visitorSampleOptions],
  );

  const fieldRows = useMemo(() => fieldRowsFromRecord(selectedRecord), [selectedRecord]);
  const fieldColumns: TableColumnsType<FieldRow> = useMemo(
    () => [
      {
        title: '字段',
        dataIndex: 'field',
        width: 280,
      },
      {
        title: '当前值',
        dataIndex: 'value',
      },
    ],
    [],
  );

  useEffect(() => {
    let active = true;
    setSampleListLoading(true);
    fetchVisitorIngestRecords()
      .then((records) => {
        if (!active) return;
        const options = buildVisitorOptions(records);
        setVisitorSampleOptions(options);
        setSelectedRecordId((current) => current || options[0]?.recordId || '');
      })
      .catch((err) => {
        if (active) {
          setSampleError(err instanceof Error ? err.message : '加载游客样本失败');
        }
      })
      .finally(() => {
        if (active) {
          setSampleListLoading(false);
        }
      });

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!selectedRecordId) {
      setRecordLoading(false);
      setSelectedRecord(null);
      return undefined;
    }
    let active = true;
    setRecordLoading(true);
    setSampleError(null);
    setSelectedRecord(null);

    fetchIngestRecord(selectedRecordId)
      .then((record) => {
        if (active) {
          setSelectedRecord(record);
        }
      })
      .catch((err) => {
        if (active) {
          setSampleError(err instanceof Error ? err.message : '加载完整脱敏宽表失败');
        }
      })
      .finally(() => {
        if (active) {
          setRecordLoading(false);
        }
      });

    return () => {
      active = false;
    };
  }, [selectedRecordId]);

  const uploadProps: UploadProps = {
    accept: '.csv,text/csv',
    beforeUpload: () => false,
    disabled: readOnly || batchSubmitting,
    fileList,
    maxCount: 1,
    multiple: false,
    onChange: async (info) => {
      const nextFileList = info.fileList.slice(-1);
      setFileList(nextFileList);
      setBatchRecords([]);
      setUploadPreview(null);
      setUploadError(null);

      const originFile = nextFileList[0]?.originFileObj as File | undefined;
      if (!originFile) return;

      if (!originFile.name.toLowerCase().endsWith('.csv')) {
        setUploadError('当前阶段只支持 CSV 报表上传');
        return;
      }

      try {
        const text = await originFile.text();
        const records = parseCsvToIngestRecords(text);
        setBatchRecords(records);
        setUploadPreview({
          fileName: originFile.name,
          rowCount: records.length,
        });
      } catch (err) {
        setUploadError(err instanceof Error ? err.message : 'CSV 报表解析失败');
      }
    },
    onRemove: () => {
      resetUploadState();
      return true;
    },
  };

  async function handleSampleSubmit() {
    if (readOnly || !selectedRecordId) return;
    setSampleSubmitting(true);
    setSampleError(null);
    try {
      const response = await pushIngestRecord(selectedRecordId);
      onCasesGenerated([response]);
    } catch (err) {
      setSampleError(err instanceof Error ? err.message : '样本建案失败');
    } finally {
      setSampleSubmitting(false);
    }
  }

  async function handleBatchSubmit() {
    if (readOnly) return;
    if (!batchRecords.length) {
      setUploadError('请先上传至少 1 行完整脱敏宽表的 CSV 报表');
      return;
    }

    setBatchSubmitting(true);
    setUploadError(null);
    try {
      const response = await submitIngestBatch({ records: batchRecords });
      if (!response.cases.length) {
        throw new Error('批量建案未返回案件结果，请重试');
      }
      onCasesGenerated(response.cases);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : '报表批量建案失败');
    } finally {
      setBatchSubmitting(false);
    }
  }

  return (
    <main className="ingest-page">
      <div className="ingest-page-stack">
        {readOnly && (
          <Alert
            className="showcase-readonly-banner"
            type="info"
            showIcon
            message="公网面试展示为只读模式"
            description="服务器已预置 5 个合成脱敏案件。可查看样本字段，但生成案件和 CSV 上传已关闭。"
          />
        )}
        <Tabs
          className="ingest-mode-tabs"
          defaultActiveKey="sample"
          items={[
            {
              key: 'sample',
              label: (
                <span className="ingest-tab-label">
                  <FileTextOutlined />
                  游客样本
                </span>
              ),
              children: (
                <div className="ingest-sample-tabpane">
                  <Card className="ingest-sample-card" title="游客样本">
                    <div className="ingest-sample-toolbar">
                      <div className="ingest-sample-select-block">
                        <Typography.Text className="ingest-toolbar-label">
                          选择样本
                        </Typography.Text>
                        <Select
                          className="ingest-sample-select"
                          popupClassName="ingest-sample-select-popup"
                          value={selectedRecordId}
                          options={groupedSampleOptions}
                          loading={sampleListLoading || recordLoading}
                          onChange={(value) => setSelectedRecordId(value)}
                        />
                      </div>
                      <Button
                        type="primary"
                        icon={readOnly ? <LockOutlined /> : <ArrowRightOutlined />}
                        disabled={readOnly || !selectedRecord || sampleListLoading}
                        loading={sampleSubmitting}
                        onClick={() => void handleSampleSubmit()}
                      >
                        {readOnly ? '只读演示已预置案件' : '生成案件'}
                      </Button>
                    </div>

                    {sampleError && (
                      <Alert
                        className="ingest-inline-alert"
                        type="error"
                        showIcon
                        message="样本处理失败"
                        description={sampleError}
                      />
                    )}

                    <div className="ingest-sample-field-shell">
                      {recordLoading && (
                        <div className="ingest-loading">
                          <Spin tip="加载样本..." />
                        </div>
                      )}

                      {!recordLoading && selectedRecord && (
                        <div className="ingest-sample-field-stack">
                          <Table
                            className="ingest-field-table"
                            size="small"
                            rowKey="field"
                            columns={fieldColumns}
                            dataSource={fieldRows}
                            pagination={false}
                            scroll={{ y: 'calc(100dvh - 390px)' }}
                          />
                        </div>
                      )}
                      {!recordLoading && !selectedRecord && !sampleError && (
                        <Empty description="请选择一个样本" />
                      )}
                    </div>
                  </Card>
                </div>
              ),
            },
            {
              key: 'upload',
              disabled: readOnly,
              label: (
                <span className="ingest-tab-label">
                  <CloudUploadOutlined />
                  报表上传
                </span>
              ),
              children: (
                <div className="ingest-upload-tabpane">
                  <Card className="ingest-upload-card" title="报表上传">
                    <Alert
                      type="info"
                      showIcon
                      message="批量接入规则"
                      description="CSV 报表中的多条记录会先整体校验；任意一行失败或同批出现完全重复记录时，整批拒绝，不生成部分案件。"
                    />

                    <div className="ingest-upload-stage">
                      <Upload.Dragger
                        className="ingest-upload-dragger"
                        disabled={readOnly}
                        {...uploadProps}
                      >
                        <p className="ant-upload-drag-icon">
                          <CloudUploadOutlined />
                        </p>
                        <p className="ant-upload-text">选择或拖入 CSV 报表</p>
                        <p className="ant-upload-hint">
                          报表需要包含完整脱敏宽表记录；当前阶段仅支持 CSV 文件。
                        </p>
                      </Upload.Dragger>

                      <div className="ingest-upload-summary">
                        <div className="ingest-upload-summary-head">
                          <Typography.Text className="ingest-upload-summary-title">
                            报表状态
                          </Typography.Text>
                          <Tag color={activeUploadTag.color}>{activeUploadTag.text}</Tag>
                        </div>

                        <dl className="ingest-upload-summary-list">
                          <div className="ingest-upload-summary-item">
                            <dt>文件</dt>
                            <dd>{activeUploadFileName}</dd>
                          </div>
                          <div className="ingest-upload-summary-item">
                            <dt>记录数</dt>
                            <dd>{activeUploadRowCount}</dd>
                          </div>
                          <div className="ingest-upload-summary-item">
                            <dt>结果</dt>
                            <dd>{activeUploadResult}</dd>
                          </div>
                        </dl>

                        <Typography.Text className="ingest-upload-summary-note" type="secondary">
                          整批校验全部通过后才会生成案件，不会出现部分成功。
                        </Typography.Text>
                      </div>
                    </div>

                    {uploadError && (
                      <Alert
                        className="ingest-inline-alert"
                        type="error"
                        showIcon
                        message="报表校验失败"
                        description={uploadError}
                      />
                    )}

                    <div className="ingest-upload-actions">
                      <Button
                        icon={<DeleteOutlined />}
                        disabled={!fileList.length && !uploadPreview && !uploadError}
                        onClick={resetUploadState}
                      >
                        清空报表
                      </Button>
                      <Button
                        type="primary"
                        icon={<CloudUploadOutlined />}
                        loading={batchSubmitting}
                        disabled={readOnly || !batchRecords.length}
                        onClick={() => void handleBatchSubmit()}
                      >
                        批量生成案件
                      </Button>
                    </div>
                  </Card>
                </div>
              ),
            },
          ]}
        />
      </div>
    </main>
  );
}
