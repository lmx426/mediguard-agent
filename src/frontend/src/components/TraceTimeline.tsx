import type { TraceNode } from '../types';

interface TraceTimelineProps {
  trace: TraceNode[];
}

function formatTraceMetadataValue(value: unknown): string {
  if (Array.isArray(value)) {
    return value.join('；');
  }
  if (value && typeof value === 'object') {
    return JSON.stringify(value);
  }
  return String(value);
}

function traceStatusLabel(node: TraceNode): string {
  if (node.status === 'completed') {
    return '已完成';
  }
  if (node.node_id.includes('manual_review_pending')) {
    return '当前';
  }
  return '待处理';
}

export default function TraceTimeline({ trace }: TraceTimelineProps) {
  return (
    <section className="card trace-card">
      <div className="card-header">
        <h3>稽核流程追踪</h3>
        <span className="badge badge-trace">审计追踪</span>
      </div>
      {trace.length === 0 ? (
        <p className="empty-hint">暂无 Trace 记录</p>
      ) : (
        <ol className="trace-list">
          {trace
            .slice()
            .sort((a, b) => a.order - b.order)
            .map((node) => (
              <li
                key={node.node_id}
                className={`trace-node ${node.status}`}
              >
                <div className="trace-node-marker">
                  {node.status === 'completed' ? '●' : '○'}
                </div>
                <div className="trace-node-content">
                  <div className="trace-node-header">
                    <span className="trace-node-name">{node.node_name}</span>
                    <span className={`trace-node-status ${node.status}`}>
                      {traceStatusLabel(node)}
                    </span>
                  </div>
                  <p className="trace-node-summary">{node.summary}</p>
                  {Object.keys(node.metadata ?? {}).length > 0 && (
                    <details className="trace-metadata">
                      <summary>查看节点元数据</summary>
                      <dl>
                        {Object.entries(node.metadata).map(([key, value]) => (
                          <div key={key}>
                            <dt>{key}</dt>
                            <dd>{formatTraceMetadataValue(value)}</dd>
                          </div>
                        ))}
                      </dl>
                    </details>
                  )}
                </div>
              </li>
            ))}
        </ol>
      )}
    </section>
  );
}
