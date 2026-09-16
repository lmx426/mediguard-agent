import { Card, Collapse, Space, Table, Tag, Typography } from 'antd';
import type { RuleCheckItem, RuleHit } from '../types';
import {
  businessText,
  ruleBusinessName,
  severityBusinessLabel,
  severityColor,
} from '../utils/businessLabels';

interface RuleHitsListProps {
  rules: RuleHit[];
}

const MATERIALS_BY_RULE_ID: Record<string, string[]> = {
  'OP-R001': ['就诊流水', '同日就诊明细', '跨机构就诊说明'],
  'OP-R002': ['结算单', '基金支付明细', '费用明细'],
  'OP-R003': ['处方记录', '药品费用清单', '长期用药材料'],
  'OP-R004': ['检查治疗项目明细', '诊疗摘要材料'],
  'OP-R005': ['挂号记录', '就诊流程材料'],
  'OP-R006': ['待遇资格材料', '补助支付明细'],
  'OP-R007': ['字段校验记录', '申报材料说明'],
  'OP-R008': ['按前序核验结果查看对应材料'],
};

function hasDataPrompt(rule: RuleHit) {
  return rule.check_items.some((item) => item.hit);
}

function ruleStatus(rule: RuleHit): { label: string; color: string } {
  if (rule.hit) return { label: '需核验', color: 'error' };
  if (hasDataPrompt(rule)) return { label: '提示', color: 'warning' };
  return { label: '未触发', color: 'default' };
}

export default function RuleHitsList({ rules }: RuleHitsListProps) {
  const reviewCount = rules.filter((rule) => rule.hit).length;
  const promptCount = rules.filter((rule) => !rule.hit && hasDataPrompt(rule)).length;

  return (
    <Card
      className="rule-check-card"
      title="业务规则核验清单"
      extra={<Tag color="blue">8 类业务核验</Tag>}
    >
      <Typography.Paragraph type="secondary">
        按业务场景完成核验：需核验 {reviewCount} 类，材料或数据提示 {promptCount} 类。
      </Typography.Paragraph>
      <Collapse
        items={rules.map((rule) => {
          const status = ruleStatus(rule);
          const materials = MATERIALS_BY_RULE_ID[rule.rule_id] ?? ['相关申报材料'];
          return {
            key: rule.rule_id,
            label: (
              <Space wrap>
                <Tag color={status.color}>{status.label}</Tag>
                <Typography.Text strong>{ruleBusinessName(rule)}</Typography.Text>
                {rule.severity !== 'low' && (
                  <Tag color={severityColor(rule.severity)}>
                    {severityBusinessLabel(rule.severity)}
                  </Tag>
                )}
              </Space>
            ),
            children: (
              <Space orientation="vertical" size={14} style={{ width: '100%' }}>
                <section className="rule-business-summary">
                  <Typography.Paragraph>{businessText(rule.business_explanation)}</Typography.Paragraph>
                  <Typography.Paragraph strong>{businessText(rule.reason)}</Typography.Paragraph>
                  <div className="rule-material-tags">
                    {materials.map((item) => (
                      <Tag key={item}>{item}</Tag>
                    ))}
                  </div>
                </section>
                <Collapse
                  className="rule-basis-collapse"
                  ghost
                  size="small"
                  items={[
                    {
                      key: 'basis',
                      label: '查看依据',
                      children: (
                        <Table<RuleCheckItem>
                          rowKey={(item) => `${rule.rule_id}-${item.label}`}
                          size="small"
                          pagination={false}
                          dataSource={rule.check_items}
                          columns={[
                            { title: '基础依据', dataIndex: 'label', width: 180 },
                            { title: '当前依据', dataIndex: 'current_value' },
                            { title: '判断口径', dataIndex: 'threshold' },
                            {
                              title: '说明',
                              dataIndex: 'explanation',
                              render: (value: string) => businessText(value),
                            },
                          ]}
                        />
                      ),
                    },
                  ]}
                />
              </Space>
            ),
          };
        })}
      />
    </Card>
  );
}
