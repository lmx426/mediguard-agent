import {
  ArrowLeftOutlined,
  CheckCircleOutlined,
  InboxOutlined,
  SafetyCertificateOutlined,
} from '@ant-design/icons';
import {
  Alert,
  Button,
  Card,
  Descriptions,
  Space,
  Tag,
  Typography,
  Upload,
} from 'antd';
import { useState } from 'react';
import type { UploadFile } from 'antd/es/upload/interface';

interface AppealPortalPageProps {
  caseId: string;
  onBack: () => void;
}

export default function AppealPortalPage({
  caseId,
  onBack,
}: AppealPortalPageProps) {
  const [files, setFiles] = useState<UploadFile[]>([]);
  const [submitted, setSubmitted] = useState(false);

  return (
    <main className="appeal-portal-page">
      <div className="appeal-portal-toolbar">
        <Button icon={<ArrowLeftOutlined />} onClick={onBack}>
          返回案件详情
        </Button>
        <Tag icon={<SafetyCertificateOutlined />} className="appeal-role-tag">
          申报方材料入口
        </Tag>
      </div>

      <Card
        className="appeal-portal-card"
        title="补充 / 申诉材料"
        extra={<Tag color="blue">【后端流程未接入】</Tag>}
      >
        <Space orientation="vertical" size={20} style={{ width: '100%' }}>
          <Descriptions
            className="appeal-case-reference"
            size="small"
            column={1}
            items={[
              {
                key: 'case_id',
                label: '关联案件',
                children: <Typography.Text code>{caseId}</Typography.Text>,
              },
            ]}
          />

          <Alert
            type="info"
            showIcon
            title="本入口只用于补充材料"
            description="申报方不参与医保侧人工复审，也不能在此查看审核意见或案件处理状态。"
          />

          <Upload.Dragger
            className="appeal-upload-dragger"
            multiple
            maxCount={8}
            accept=".pdf,.jpg,.jpeg,.png,.doc,.docx,.zip"
            fileList={files}
            beforeUpload={() => false}
            onChange={({ fileList }) => {
              setFiles(fileList);
              setSubmitted(false);
            }}
          >
            <p className="ant-upload-drag-icon">
              <InboxOutlined />
            </p>
            <p className="ant-upload-text">选择或拖入补充材料</p>
            <p className="ant-upload-hint">
              支持 PDF、图片、Word 或 ZIP，最多 8 个文件
            </p>
          </Upload.Dragger>

          <Alert
            type="warning"
            showIcon
            title="材料安全边界"
            description="当前环境仅保留本页文件选择状态，不读取、不上传、不保存文件内容；刷新页面后文件列表会清空。请勿选择真实个人身份材料。【后端流程未接入】"
          />

          {submitted && (
            <Alert
              type="success"
              showIcon
              icon={<CheckCircleOutlined />}
              title="材料已登记到当前页面"
              description={`已选择 ${files.length} 个文件，当前环境未配置网络上传或持久化。【后端流程未接入】`}
            />
          )}

          <div className="appeal-submit-row">
            <Typography.Text type="secondary">
              已选择 {files.length} / 8 个文件
            </Typography.Text>
            <Button
              type="primary"
              disabled={files.length === 0}
              onClick={() => setSubmitted(true)}
            >
              提交材料
            </Button>
          </div>
        </Space>
      </Card>
    </main>
  );
}
