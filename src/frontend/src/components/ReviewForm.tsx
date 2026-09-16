import { DeleteOutlined, PlusOutlined, UploadOutlined } from '@ant-design/icons';
import {
  Alert,
  Button,
  Card,
  Descriptions,
  Form,
  Input,
  Select,
  Space,
  Table,
  Typography,
  Upload,
} from 'antd';
import type { UploadProps } from 'antd';
import { useState } from 'react';
import { submitReview } from '../services/api';
import type {
  AuthenticatedUser,
  ReviewAttachment,
  ReviewDecision,
} from '../types';

interface ReviewFormProps {
  caseId: string;
  existingReview: ReviewDecision | null;
  currentUser: AuthenticatedUser;
  readOnly?: boolean;
  onReviewSubmitted: () => void;
}

interface ReviewFormValues {
  decision?: string;
  reason: string;
  attachments?: ReviewAttachment[];
}

const ATTACHMENT_TYPE_OPTIONS = [
  { value: '诊疗摘要材料', label: '诊疗摘要材料' },
  { value: '费用清单', label: '费用清单' },
  { value: '处方/用药记录', label: '处方/用药记录' },
  { value: '结算材料', label: '结算材料' },
  { value: '其他', label: '其他' },
];

const ATTACHMENT_STATUS_OPTIONS = [
  { value: '待核验', label: '待核验' },
  { value: '已核验', label: '已核验' },
  { value: '需补充', label: '需补充' },
];

function inferMaterialType(file: File): string {
  const text = `${file.name} ${file.type}`.toLowerCase();
  if (/处方|用药|prescription|drug|medicine/.test(text)) return '处方/用药记录';
  if (/费用|清单|invoice|bill|fee|cost/.test(text)) return '费用清单';
  if (/结算|医保|settlement|payment/.test(text)) return '结算材料';
  if (/病历|诊疗|medical|record|clinic/.test(text)) return '诊疗摘要材料';
  return '其他';
}

function formatFileSize(size?: number): string {
  if (typeof size !== 'number' || Number.isNaN(size)) return '大小未知';
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

function formatFileModified(lastModified?: number): string | undefined {
  if (!lastModified) return undefined;
  return new Date(lastModified).toLocaleString('zh-CN', { hour12: false });
}

function fileMetadataFromFile(file: File): Pick<
  ReviewAttachment,
  'file_name' | 'file_size' | 'file_type' | 'file_last_modified'
> {
  return {
    file_name: file.name,
    file_size: file.size,
    file_type: file.type || '未知类型',
    file_last_modified: formatFileModified(file.lastModified),
  };
}

export default function ReviewForm({
  caseId,
  existingReview,
  currentUser,
  readOnly = false,
  onReviewSubmitted,
}: ReviewFormProps) {
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [form] = Form.useForm<ReviewFormValues>();

  function applyLocalFile(index: number, file: File) {
    const current = form.getFieldValue(['attachments', index]) as ReviewAttachment | undefined;
    form.setFieldValue(['attachments', index], {
      ...current,
      name: current?.name?.trim() || file.name,
      material_type: current?.material_type || inferMaterialType(file),
      source: current?.source?.trim() || '审核人员登记',
      verification_status: current?.verification_status || '待核验',
      remark: current?.remark || '',
      ...fileMetadataFromFile(file),
    });
  }

  async function handleFinish(values: ReviewFormValues) {
    setError(null);
    setSubmitting(true);
    try {
      const attachments = (values.attachments ?? []).map((item) => ({
        name: item.name.trim(),
        material_type: item.material_type,
        source: item.source?.trim() || '审核人员登记',
        verification_status: item.verification_status || '待核验',
        remark: item.remark?.trim() || '',
        file_name: item.file_name?.trim() || undefined,
        file_size: Number.isFinite(Number(item.file_size)) ? Number(item.file_size) : undefined,
        file_type: item.file_type?.trim() || undefined,
        file_last_modified: item.file_last_modified?.trim() || undefined,
      }));
      await submitReview(caseId, {
        decision: values.decision || '未指定',
        reason: values.reason.trim(),
        attachments,
      });
      form.resetFields();
      onReviewSubmitted();
    } catch (err) {
      setError(err instanceof Error ? err.message : '提交失败');
    } finally {
      setSubmitting(false);
    }
  }

  if (existingReview) {
    return (
      <Card className="manual-review-card" title="人工初审" extra="人工决定">
        <Space orientation="vertical" size={12} style={{ width: '100%' }}>
          <Descriptions
            column={1}
            bordered
            size="small"
            items={[
              { key: 'reviewer', label: '审核员', children: existingReview.reviewer },
              { key: 'decision', label: '结论', children: existingReview.decision },
              { key: 'reason', label: '理由', children: existingReview.reason },
              { key: 'submitted_at', label: '提交时间', children: existingReview.submitted_at },
            ]}
          />
          {(existingReview.attachments ?? []).length > 0 && (
            <section className="review-attachment-result">
              <Typography.Title level={5}>附件材料登记</Typography.Title>
              <AttachmentTable attachments={existingReview.attachments} />
            </section>
          )}
          <Alert
            type="success"
            showIcon
            title="初审意见由审核人员提交，案件稽核助手和证据包不能修改此记录。"
          />
        </Space>
      </Card>
    );
  }

  if (readOnly) {
    return (
      <Card className="manual-review-card" title="人工初审" extra="只读展示">
        <Space orientation="vertical" size={12} style={{ width: '100%' }}>
          <Alert
            type="info"
            showIcon
            title="公网演示不保存人工初审"
            description="可查看前序风险、规则和证据组织结果；提交意见、登记材料和附件写入仅在本地开发环境开放。"
          />
          <Descriptions
            column={1}
            bordered
            size="small"
            items={[
              { key: 'reviewer', label: '演示账号', children: currentUser.display_name },
              { key: 'status', label: '当前状态', children: '待人工初审（只读）' },
              { key: 'boundary', label: '决定边界', children: 'Agent 不形成或修改最终人工决定' },
            ]}
          />
        </Space>
      </Card>
    );
  }

  return (
    <Card className="manual-review-card" title="人工初审" extra="人工决定">
      <Space orientation="vertical" size={12} style={{ width: '100%' }}>
        {error && <Alert type="error" showIcon title={error} />}
        <Alert
          type="info"
          showIcon
          title={`当前审核人员：${currentUser.display_name}`}
          description={currentUser.department || currentUser.username}
        />
        <Form
          form={form}
          layout="vertical"
          onFinish={handleFinish}
          initialValues={{ decision: '常规处理' }}
        >
          <Space orientation="vertical" size={16} style={{ width: '100%' }}>
            <section className="review-attachment-editor">
              <div className="review-attachment-editor-head">
                <Typography.Title level={5}>审核材料登记</Typography.Title>
              </div>
              <Form.List name="attachments">
                {(fields, { add, remove }) => (
                  <Space orientation="vertical" size={12} style={{ width: '100%' }}>
                    {fields.map((field) => (
                      <div className="review-attachment-row" key={field.key}>
                        <Form.Item
                          {...field}
                          label="材料名称"
                          name={[field.name, 'name']}
                          rules={[{ required: true, whitespace: true, message: '请填写材料名称' }]}
                        >
                          <Input placeholder="例如：就诊统计材料" />
                        </Form.Item>
                        <Form.Item
                          className="review-attachment-file"
                          label="本地文件"
                        >
                          <Space orientation="vertical" size={6} style={{ width: '100%' }}>
                            <Upload
                              beforeUpload={((file) => {
                                applyLocalFile(field.name, file);
                                return false;
                              }) as UploadProps['beforeUpload']}
                              maxCount={1}
                              showUploadList={false}
                            >
                              <Button icon={<UploadOutlined />}>选择文件</Button>
                            </Upload>
                            <Typography.Text type="secondary">
                              仅登记文件元数据【后端流程未接入】
                            </Typography.Text>
                            <Form.Item noStyle shouldUpdate>
                              {() => {
                                const attachment = form.getFieldValue([
                                  'attachments',
                                  field.name,
                                ]) as ReviewAttachment | undefined;
                                return attachment?.file_name ? (
                                  <Typography.Text type="secondary">
                                    {[
                                      attachment.file_name,
                                      formatFileSize(attachment.file_size),
                                      attachment.file_type,
                                    ]
                                      .filter(Boolean)
                                      .join(' · ')}
                                  </Typography.Text>
                                ) : (
                                  <Typography.Text type="secondary">未选择</Typography.Text>
                                );
                              }}
                            </Form.Item>
                          </Space>
                        </Form.Item>
                        <Form.Item
                          {...field}
                          label="材料类型"
                          name={[field.name, 'material_type']}
                          rules={[{ required: true, message: '请选择材料类型' }]}
                        >
                          <Select options={ATTACHMENT_TYPE_OPTIONS} />
                        </Form.Item>
                        <Form.Item
                          {...field}
                          label="来源"
                          name={[field.name, 'source']}
                        >
                          <Input placeholder="审核人员登记" />
                        </Form.Item>
                        <Form.Item
                          {...field}
                          label="核验状态"
                          name={[field.name, 'verification_status']}
                        >
                          <Select options={ATTACHMENT_STATUS_OPTIONS} />
                        </Form.Item>
                        <Form.Item
                          {...field}
                          className="review-attachment-remark"
                          label="备注"
                          name={[field.name, 'remark']}
                        >
                          <Input.TextArea rows={2} placeholder="材料核验备注" />
                        </Form.Item>
                        <Form.Item {...field} hidden name={[field.name, 'file_name']}>
                          <Input />
                        </Form.Item>
                        <Form.Item {...field} hidden name={[field.name, 'file_size']}>
                          <Input />
                        </Form.Item>
                        <Form.Item {...field} hidden name={[field.name, 'file_type']}>
                          <Input />
                        </Form.Item>
                        <Form.Item {...field} hidden name={[field.name, 'file_last_modified']}>
                          <Input />
                        </Form.Item>
                        <Button
                          className="review-attachment-remove"
                          icon={<DeleteOutlined />}
                          aria-label="删除材料登记"
                          onClick={() => remove(field.name)}
                        />
                      </div>
                    ))}
                    <Button
                      icon={<PlusOutlined />}
                      onClick={() =>
                        add({
                          source: '审核人员登记',
                          verification_status: '待核验',
                          remark: '',
                        })
                      }
                    >
                      添加材料登记
                    </Button>
                  </Space>
                )}
              </Form.List>
            </section>
            <section className="review-decision-editor">
              <div className="review-decision-editor-head">
                <Typography.Title level={5}>初审意见提交</Typography.Title>
              </div>
              <Form.Item label="审核结论" name="decision">
                <Select
                  options={[
                    { value: '常规处理', label: '常规处理' },
                    { value: '需进一步调查', label: '需进一步调查' },
                    { value: '需补充材料', label: '需补充材料' },
                    { value: '移交上级审核', label: '移交上级审核' },
                  ]}
                />
              </Form.Item>
              <Form.Item
                label="审核理由"
                name="reason"
                rules={[{ required: true, whitespace: true, message: '审核理由不能为空' }]}
              >
                <Input.TextArea rows={4} placeholder="请填写审核理由" />
              </Form.Item>
              <Button type="primary" htmlType="submit" loading={submitting}>
                提交审核意见
              </Button>
            </section>
          </Space>
        </Form>
      </Space>
    </Card>
  );
}

function AttachmentTable({ attachments }: { attachments: ReviewAttachment[] }) {
  return (
    <Table
      size="small"
      rowKey={(record) => `${record.name}-${record.material_type}-${record.file_name ?? ''}`}
      pagination={false}
      dataSource={attachments}
      columns={[
        { title: '材料名称', dataIndex: 'name' },
        { title: '材料类型', dataIndex: 'material_type', width: 140 },
        {
          title: '本地文件',
          dataIndex: 'file_name',
          width: 220,
          render: (_: string | undefined, record) =>
            record.file_name ? (
              <Space orientation="vertical" size={0}>
                <Typography.Text>{record.file_name}</Typography.Text>
                <Typography.Text type="secondary">
                  {[formatFileSize(record.file_size), record.file_type].filter(Boolean).join(' · ')}
                </Typography.Text>
              </Space>
            ) : (
              <Typography.Text type="secondary">未选择</Typography.Text>
            ),
        },
        { title: '来源', dataIndex: 'source', width: 140 },
        { title: '核验状态', dataIndex: 'verification_status', width: 120 },
        { title: '备注', dataIndex: 'remark' },
      ]}
    />
  );
}
