import {
  ClockCircleOutlined,
  DeleteOutlined,
  FileSearchOutlined,
  FormOutlined,
  LockOutlined,
  PaperClipOutlined,
  PlusOutlined,
} from '@ant-design/icons';
import {
  Alert,
  Button,
  Card,
  Empty,
  Form,
  Input,
  Modal,
  Pagination,
  Select,
  Segmented,
  Space,
  Tag,
  Typography,
} from 'antd';
import { useState, type ReactNode } from 'react';
import { deleteAuditNote, submitAuditNote } from '../services/api';
import type {
  AuditNote,
  AuditNoteSource,
  ReviewAttachment,
  ReviewDecision,
} from '../types';
import AuditAssistantIcon from './AuditAssistantIcon';

interface AuditNotesPanelProps {
  caseId: string;
  notes: AuditNote[];
  review: ReviewDecision | null;
  readOnly?: boolean;
  onNoteSaved: (note: AuditNote) => void;
  onNoteDeleted: (noteId: string) => void;
}

interface AuditNoteFormValues {
  source: AuditNoteSource;
  content?: string;
  material?: Partial<ReviewAttachment>;
}

type AuditNotePanelMode = 'compose' | 'list';
const NOTE_PAGE_SIZE = 3;

const SOURCE_META: Record<
  AuditNoteSource,
  { label: string; icon: ReactNode; className: string }
> = {
  manual: {
    label: '人工研判',
    icon: <FormOutlined />,
    className: 'note-source-manual',
  },
  assistant: {
    label: '稽核助手摘录',
    icon: <AuditAssistantIcon />,
    className: 'note-source-assistant',
  },
  case_agent_adopted: {
    label: '案件助手采纳',
    icon: <AuditAssistantIcon />,
    className: 'note-source-assistant',
  },
  external: {
    label: '外部材料核验',
    icon: <FileSearchOutlined />,
    className: 'note-source-external',
  },
};

const MATERIAL_TYPE_OPTIONS = [
  { value: '诊疗摘要材料', label: '诊疗摘要材料' },
  { value: '费用清单', label: '费用清单' },
  { value: '处方/用药记录', label: '处方/用药记录' },
  { value: '结算材料', label: '结算材料' },
  { value: '其他', label: '其他' },
];

const MATERIAL_STATUS_OPTIONS = [
  { value: '待核验', label: '待核验' },
  { value: '已核验', label: '已核验' },
  { value: '需补充', label: '需补充' },
];

function formatDateTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).format(date);
}

function normalizeMaterial(
  material: Partial<ReviewAttachment> | undefined,
): ReviewAttachment | null {
  const name = material?.name?.trim();
  if (!name) return null;
  return {
    name,
    material_type: material?.material_type || '其他',
    source: material?.source?.trim() || '审核人员登记',
    verification_status: material?.verification_status || '待核验',
    remark: material?.remark?.trim() || '',
  };
}

function materialSummary(material: ReviewAttachment): string {
  return [
    material.material_type,
    material.source,
    material.remark,
  ].filter(Boolean).join(' · ');
}

function materialNoteContent(material: ReviewAttachment): string {
  const remark = material.remark ? `备注：${material.remark}` : '';
  return [
    `外部材料核验：${material.name}`,
    `材料类型：${material.material_type}`,
    `核验状态：${material.verification_status}`,
    remark,
  ].filter(Boolean).join('；');
}

export default function AuditNotesPanel({
  caseId,
  notes,
  review,
  readOnly = false,
  onNoteSaved,
  onNoteDeleted,
}: AuditNotesPanelProps) {
  const [form] = Form.useForm<AuditNoteFormValues>();
  const selectedSource = Form.useWatch('source', form) ?? 'manual';
  const [submitting, setSubmitting] = useState(false);
  const [deletingNoteId, setDeletingNoteId] = useState<string | null>(null);
  const [detailNote, setDetailNote] = useState<AuditNote | null>(null);
  const [mode, setMode] = useState<AuditNotePanelMode>(readOnly ? 'list' : 'compose');
  const [notePage, setNotePage] = useState(1);
  const [error, setError] = useState<string | null>(null);

  async function handleFinish(values: AuditNoteFormValues) {
    setSubmitting(true);
    setError(null);
    try {
      const source = values.source ?? 'manual';
      const material = normalizeMaterial(values.material);
      if (source === 'external' && material === null) {
        throw new Error('请填写关联材料信息');
      }
      const content = source === 'external' && material
        ? materialNoteContent(material)
        : (values.content ?? '').trim();
      const savedNote = await submitAuditNote(caseId, {
        source,
        content,
        materials: source === 'external' && material ? [material] : [],
      });
      form.setFieldValue('content', '');
      form.resetFields(['material']);
      onNoteSaved(savedNote);
      setNotePage(1);
      setMode('list');
    } catch (err) {
      setError(err instanceof Error ? err.message : '保存工作笔记失败');
    } finally {
      setSubmitting(false);
    }
  }

  async function handleDelete(noteId: string) {
    setDeletingNoteId(noteId);
    setError(null);
    try {
      const deletedNote = await deleteAuditNote(caseId, noteId);
      onNoteDeleted(deletedNote.note_id);
      setDetailNote((current) => (
        current?.note_id === deletedNote.note_id ? null : current
      ));
    } catch (err) {
      setError(err instanceof Error ? err.message : '删除工作笔记失败');
    } finally {
      setDeletingNoteId(null);
    }
  }

  const visibleNotes = notes.filter((note) => note.status !== 'voided');
  const orderedNotes = [...visibleNotes].reverse();
  const totalNotePages = Math.max(
    1,
    Math.ceil(orderedNotes.length / NOTE_PAGE_SIZE),
  );
  const currentNotePage = Math.min(notePage, totalNotePages);
  const pagedNotes = orderedNotes.slice(
    (currentNotePage - 1) * NOTE_PAGE_SIZE,
    currentNotePage * NOTE_PAGE_SIZE,
  );

  return (
    <aside
      className={`audit-notes-panel ${
        selectedSource === 'external' ? 'audit-notes-panel-external' : ''
      }`}
      aria-label="审核工作记录"
    >
      <Card
        className="audit-notes-card"
        title={
          <Space size={8}>
            <FormOutlined />
            <span>审核工作记录</span>
          </Space>
        }
        extra={
          <Tag className="case-level-tag">案件级</Tag>
        }
      >
        <Segmented
          className="audit-note-mode-switch"
          value={mode}
          onChange={(value) => setMode(value as AuditNotePanelMode)}
          options={[
            { label: '记笔记', value: 'compose' },
            { label: '笔记列表', value: 'list' },
          ]}
        />
        {mode === 'list' ? (
          <section className="audit-note-section audit-note-list-section">
            <div className="audit-note-list">
              {visibleNotes.length === 0 ? (
                <Empty
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                  description="暂无工作笔记"
                />
              ) : (
                <div className="audit-note-items" role="list">
                  {pagedNotes.map((note) => {
                    const source = SOURCE_META[note.source];
                    return (
                      <article
                        key={note.note_id}
                        className="audit-note-item"
                        role="button"
                        tabIndex={0}
                        onClick={() => setDetailNote(note)}
                        onKeyDown={(event) => {
                          if (event.key === 'Enter' || event.key === ' ') {
                            event.preventDefault();
                            setDetailNote(note);
                          }
                        }}
                      >
                        <div className="audit-note-item-body">
                          <div className="audit-note-item-head">
                            <Space size={6} wrap>
                              <Tag
                                className={`audit-note-source ${source.className}`}
                                icon={source.icon}
                              >
                                {source.label}
                              </Tag>
                            </Space>
                            <Typography.Text className="audit-note-author">
                              {note.author}
                            </Typography.Text>
                          </div>
                          <Typography.Paragraph className="audit-note-content">
                            {note.content}
                          </Typography.Paragraph>
                          {(note.materials ?? []).length > 0 && (
                            <div className="audit-note-materials audit-note-materials-preview">
                              {note.materials?.slice(0, 1).map((material, index) => (
                                <div
                                  className="audit-note-material"
                                  key={`${note.note_id}-material-${index}`}
                                >
                                  <div className="audit-note-material-head">
                                    <PaperClipOutlined />
                                    <span>{material.name}</span>
                                    <Tag>{material.verification_status}</Tag>
                                  </div>
                                </div>
                              ))}
                            </div>
                          )}
                          <div className="audit-note-meta-row">
                            <Typography.Text className="audit-note-time">
                              <ClockCircleOutlined /> {formatDateTime(note.created_at)}
                            </Typography.Text>
                            {!review && !readOnly && (
                              <Button
                                className="audit-note-delete-button"
                                type="text"
                                size="small"
                                icon={<DeleteOutlined />}
                                aria-label="删除工作笔记"
                                title="删除工作笔记"
                                loading={deletingNoteId === note.note_id}
                                onClick={(event) => {
                                  event.stopPropagation();
                                  handleDelete(note.note_id);
                                }}
                              />
                            )}
                          </div>
                        </div>
                      </article>
                    );
                  })}
                </div>
              )}
            </div>
            {visibleNotes.length > NOTE_PAGE_SIZE && (
              <Pagination
                className="audit-note-pagination"
                simple
                size="small"
                current={currentNotePage}
                pageSize={NOTE_PAGE_SIZE}
                total={visibleNotes.length}
                onChange={(page) => setNotePage(page)}
              />
            )}
          </section>
        ) : (
          <section className="audit-note-section audit-note-editor-section">
            {review || readOnly ? (
              <div className="audit-note-readonly">
                <LockOutlined />
                <span>
                  {readOnly
                    ? '公网演示不保存工作笔记'
                    : '人工初审已提交，工作笔记只读'}
                </span>
              </div>
            ) : (
              <Form
                className={`audit-note-form ${
                  selectedSource === 'external' ? 'audit-note-form-external' : ''
                }`}
                form={form}
                layout="vertical"
                initialValues={{ source: 'manual' }}
                onFinish={handleFinish}
              >
                <div className="audit-note-form-fields">
                  {error && <Alert type="error" showIcon title={error} />}
                  <Form.Item label="信息来源" name="source">
                    <Select
                      options={[
                        { value: 'manual', label: '人工研判' },
                        { value: 'assistant', label: '稽核助手摘录（人工确认）' },
                        { value: 'external', label: '外部材料核验' },
                      ]}
                    />
                  </Form.Item>
                  {selectedSource === 'external' ? (
                    <section className="audit-note-material-editor">
                      <Typography.Text className="audit-note-material-title">
                        关联材料
                      </Typography.Text>
                      <Form.Item
                        label="材料名称"
                        name={['material', 'name']}
                        rules={[{ required: true, whitespace: true, message: '请填写材料名称' }]}
                      >
                        <Input placeholder="例如：门诊费用明细" />
                      </Form.Item>
                      <Form.Item
                        label="材料类型"
                        name={['material', 'material_type']}
                        rules={[{ required: true, message: '请选择材料类型' }]}
                      >
                        <Select options={MATERIAL_TYPE_OPTIONS} placeholder="选择材料类型" />
                      </Form.Item>
                      <Form.Item label="来源" name={['material', 'source']}>
                        <Input placeholder="审核人员登记" />
                      </Form.Item>
                      <Form.Item label="核验状态" name={['material', 'verification_status']}>
                        <Select options={MATERIAL_STATUS_OPTIONS} placeholder="待核验" />
                      </Form.Item>
                      <Form.Item label="备注" name={['material', 'remark']}>
                        <Input.TextArea rows={2} placeholder="材料核验备注" />
                      </Form.Item>
                    </section>
                  ) : (
                    <Form.Item
                      className="audit-note-content-field"
                      label="笔记内容"
                      name="content"
                      rules={[{ required: true, whitespace: true, message: '请填写笔记内容' }]}
                    >
                      <Input.TextArea
                        rows={7}
                        maxLength={2000}
                        showCount
                        placeholder="记录研判要点、待核验事项或经人工确认的辅助信息"
                      />
                    </Form.Item>
                  )}
                </div>
                <Button
                  block
                  type="primary"
                  htmlType="submit"
                  icon={<PlusOutlined />}
                  loading={submitting}
                >
                  {selectedSource === 'external' ? '保存材料核验记录' : '保存工作笔记'}
                </Button>
              </Form>
            )}
          </section>
        )}
      </Card>
      <Modal
        title="工作笔记详情"
        open={detailNote !== null}
        footer={null}
        onCancel={() => setDetailNote(null)}
      >
        {detailNote && (
          <Space direction="vertical" size={14} style={{ width: '100%' }}>
            <Space size={8} wrap>
              <Tag
                className={`audit-note-source ${SOURCE_META[detailNote.source].className}`}
                icon={SOURCE_META[detailNote.source].icon}
              >
                {SOURCE_META[detailNote.source].label}
              </Tag>
              <Typography.Text type="secondary">
                {detailNote.author}
              </Typography.Text>
              <Typography.Text type="secondary">
                {formatDateTime(detailNote.created_at)}
              </Typography.Text>
            </Space>
            <Typography.Paragraph className="audit-note-detail-content">
              {detailNote.content}
            </Typography.Paragraph>
            {(detailNote.materials ?? []).length > 0 && (
              <div className="audit-note-detail-materials">
                <Typography.Text strong>关联材料</Typography.Text>
                {detailNote.materials?.map((material, index) => (
                  <div
                    className="audit-note-material"
                    key={`${detailNote.note_id}-detail-material-${index}`}
                  >
                    <div className="audit-note-material-head">
                      <PaperClipOutlined />
                      <span>{material.name}</span>
                      <Tag>{material.verification_status}</Tag>
                    </div>
                    <Typography.Text type="secondary">
                      {materialSummary(material)}
                    </Typography.Text>
                  </div>
                ))}
              </div>
            )}
          </Space>
        )}
      </Modal>
    </aside>
  );
}
