import { businessText, evidenceRefLabel } from '../utils/businessLabels';

interface RiskScoreCardProps {
  score: number;
  level: string;
  modelEvidenceRef: string;
  source: string;
  reasons: string[];
}

/** 风险等级对应的颜色 */
function levelColor(level: string): string {
  switch (level) {
    case 'high':
      return '#dc2626';
    case 'medium':
      return '#f59e0b';
    case 'low':
      return '#16a34a';
    case 'critical':
      return '#991b1b';
    default:
      return '#6b7280';
  }
}

/** 风险等级中文 */
function levelLabel(level: string): string {
  switch (level) {
    case 'high':
      return '高风险';
    case 'medium':
      return '中风险';
    case 'low':
      return '低风险';
    default:
      return level;
  }
}

export default function RiskScoreCard({
  score,
  level,
  modelEvidenceRef,
  source,
  reasons,
}: RiskScoreCardProps) {
  const color = levelColor(level);

  return (
    <section className="card risk-score-card">
      <div className="card-header">
        <h3>风险提示强度</h3>
        <span className="badge badge-model">异常筛查</span>
      </div>
      <div className="risk-score-body">
        <div className="risk-score-circle" style={{ borderColor: color, color }}>
          <span className="risk-score-value">{score.toFixed(2)}</span>
          <span className="risk-score-label">{levelLabel(level)}</span>
        </div>
        <div className="risk-score-meta">
          <p className="risk-score-ref">
            筛查留痕：{evidenceRefLabel(modelEvidenceRef)}
          </p>
          <p className="risk-score-ref">
            筛查来源：{businessText(source)}
          </p>
          <p className="risk-score-disclaimer">
            该提示由费用与就诊异常筛查基于单条统计特征生成，非 LLM 生成，Agent 不可修改。
          </p>
          {reasons.length > 0 && (
            <ul className="risk-reason-list">
              {reasons.map((reason) => (
                <li key={reason}>{businessText(reason)}</li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </section>
  );
}
