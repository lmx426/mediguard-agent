import { Card, Space, Typography } from 'antd';
import AuditAssistantIcon from './AuditAssistantIcon';

export default function AuditAssistPanel() {
  return (
    <aside className="audit-assist-panel">
      <Card className="audit-assist-card" title="审核辅助信息" extra="辅助证据">
        <Space orientation="vertical" size={10}>
          <AuditAssistantIcon className="agent-icon" />
          <Typography.Text strong>Review Advisor 边界</Typography.Text>
          <Typography.Text type="secondary">
            仅用于当前案件问答、规则解释、证据引用和补充材料提示，不生成自动结论。
          </Typography.Text>
          <div className="assist-note-list">
            <span>模型信号：辅助线索</span>
            <span>规则命中：确定性核验</span>
            <span>人工审核：最终业务入口</span>
          </div>
        </Space>
      </Card>
    </aside>
  );
}
