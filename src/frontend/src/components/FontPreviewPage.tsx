import { CheckOutlined } from '@ant-design/icons';
import { Button, Card, Space, Tag, Typography } from 'antd';
import {
  TYPEFACE_OPTIONS,
  getTypefaceOption,
  type TypefaceKey,
} from '../config/typefaces';

interface FontPreviewPageProps {
  currentTypeface: TypefaceKey;
  onSelectTypeface: (typeface: TypefaceKey) => void;
}

const SAMPLE_METRICS = [
  { label: '风险信号分', value: '0.68' },
  { label: '规则线索数', value: '6' },
  { label: '审核状态', value: '待初审' },
];

export default function FontPreviewPage({
  currentTypeface,
  onSelectTypeface,
}: FontPreviewPageProps) {
  const selected = getTypefaceOption(currentTypeface);

  return (
    <main className="font-preview-page">
      <section className="font-preview-hero">
        <div className="font-preview-copy">
          <Typography.Title level={2}>字体预览</Typography.Title>
          <Typography.Paragraph className="font-preview-intro">
            点选一种字体后，整个平台会同步切换并记住你的选择。建议优先看中文标题、
            正文、数字和英文混排。
          </Typography.Paragraph>
        </div>
        <Card className="font-preview-current" bordered={false}>
          <span className="font-preview-current-label">当前应用字体</span>
          <strong>{selected.label}</strong>
          <span className="font-preview-current-note">{selected.note}</span>
        </Card>
      </section>

      <section className="font-preview-grid" aria-label="字体候选预览">
        {TYPEFACE_OPTIONS.map((option) => {
          const active = option.key === currentTypeface;
          return (
            <Card
              key={option.key}
              className={active ? 'font-preview-card is-active' : 'font-preview-card'}
            >
              <div className="font-preview-card-head">
                <div className="font-preview-card-titleblock">
                  <Typography.Title level={4}>{option.label}</Typography.Title>
                  <Typography.Text>{option.note}</Typography.Text>
                </div>
                {active ? (
                  <Tag color="blue" icon={<CheckOutlined />}>
                    当前
                  </Tag>
                ) : null}
              </div>

              <div
                className="font-preview-sample"
                style={{ fontFamily: option.stack }}
              >
                <h3 className="font-preview-sample-title">
                  医保智能稽核服务平台
                </h3>
                <p className="font-preview-sample-copy">
                  输入一条脱敏后的统计特征记录后，系统会动态生成风险提示、规则命中
                  和证据包，帮助审核员完成初审。
                </p>

                <div className="font-preview-stats">
                  {SAMPLE_METRICS.map((metric) => (
                    <div key={metric.label} className="font-preview-stat">
                      <span>{metric.label}</span>
                      <strong>{metric.value}</strong>
                    </div>
                  ))}
                </div>

                <div className="font-preview-inline">
                  <span>CASE-002</span>
                  <span>MediGuard</span>
                  <span>2026-07-27</span>
                </div>
              </div>

              <Space wrap className="font-preview-actions">
                <Button
                  type={active ? 'primary' : 'default'}
                  icon={active ? <CheckOutlined /> : undefined}
                  onClick={() => onSelectTypeface(option.key)}
                >
                  {active ? '当前字体' : '设为当前字体'}
                </Button>
              </Space>
            </Card>
          );
        })}
      </section>
    </main>
  );
}
