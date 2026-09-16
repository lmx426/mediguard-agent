import { Tag } from 'antd';
import type { CaseSummary } from '../types';

const RISK_LABELS: Record<CaseSummary['risk_level'], string> = {
  low: '低风险',
  medium: '中风险',
  high: '高风险',
  insufficient: '证据不足',
};

interface RiskTagProps {
  level: CaseSummary['risk_level'];
  prefix?: string;
}

export default function RiskTag({ level, prefix }: RiskTagProps) {
  return (
    <Tag className={`risk-soft-tag risk-${level}`}>
      {prefix}
      {RISK_LABELS[level]}
    </Tag>
  );
}
