import { HistoryOutlined } from '@ant-design/icons';
import { Collapse, Empty, Timeline, Typography } from 'antd';

const EVENT_LABELS: Record<string, string> = {
  MEMORY_CAPTURED: '记忆已创建',
  MEMORY_OBSERVED: '相同经验已再次观测',
  MEMORY_SHADOW_THRESHOLD_REACHED: '影子观察达到确认阈值',
  MEMORY_CONFIRM: '人工确认激活',
  MEMORY_AUTO_ACTIVATE: '系统自动激活',
  MEMORY_REJECT: '记忆已拒绝',
  MEMORY_SNOOZE: '记忆已暂不处理',
  MEMORY_SNOOZE_EXPIRED: '暂存期限已结束',
  MEMORY_ARCHIVE: '记忆已归档',
  MEMORY_AUTO_ARCHIVED: '记忆已自动归档',
  MEMORY_RESTORE: '记忆已恢复',
  MEMORY_REVOKE: '记忆已撤销',
  MEMORY_SUPERSEDED: '记忆已被替代',
  MEMORY_SUPERSEDES: '记忆已替代旧版本',
  MEMORY_CONFLICT_MARKED: '记忆进入冲突核查',
  MEMORY_STATUS_CHANGED: '状态已更新',
  MEMORY_PROMOTED: '成熟度已提升',
  MEMORY_PROJECTION_SYNCED: '检索投影已同步',
  MEMORY_PROJECTION_FAILED: '检索投影同步失败',
};

const STATUS_LABELS: Record<string, string> = {
  candidate: '待确认',
  active: '已激活',
  shadow: '影子观察',
  rejected: '已拒绝',
  snoozed: '暂不处理',
  archived: '已归档',
  superseded: '已被替代',
  revoked: '已撤销',
  tombstoned: '已清除投影',
  conflict_review: '冲突核查中',
};

function formatDateTime(value: unknown): string {
  if (typeof value !== 'string' && !(value instanceof Date)) return '时间未知';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '时间未知' : date.toLocaleString('zh-CN');
}

function actorLabel(value: string): string {
  return /system|worker|service|auto/i.test(value) ? '系统处理' : '授权人员操作';
}

function eventDescription(event: Record<string, unknown>): string | undefined {
  const payload = event.payload;
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return undefined;
  const record = payload as Record<string, unknown>;
  const from = typeof record.from_status === 'string' ? record.from_status : undefined;
  const to = typeof record.to_status === 'string' ? record.to_status : undefined;
  const count = typeof record.observation_count === 'number' ? record.observation_count : undefined;
  if (from && to) {
    return `${STATUS_LABELS[from] ?? from} → ${STATUS_LABELS[to] ?? to}`;
  }
  if (count !== undefined) return `累计观测 ${count} 次`;
  if (typeof record.status === 'string') return `状态：${STATUS_LABELS[record.status] ?? record.status}`;
  return undefined;
}

export function MemoryStatusHistory({
  events,
}: {
  events: Array<Record<string, unknown>>;
}) {
  const items = events.map((event, index) => {
    const eventType = typeof event.event_type === 'string'
      ? event.event_type.toUpperCase()
      : 'MEMORY_EVENT';
    const actor = typeof event.actor_id === 'string' ? event.actor_id : undefined;
    const description = eventDescription(event);
    return {
      key: typeof event.event_id === 'string' ? event.event_id : `${eventType}-${index}`,
      children: (
        <div className="memory-timeline-event">
          <Typography.Text>{EVENT_LABELS[eventType] ?? '记忆状态已更新'}</Typography.Text>
          {description && <Typography.Text type="secondary">{description}</Typography.Text>}
          <Typography.Text type="secondary">
            {formatDateTime(event.created_at)}{actor ? ` · ${actorLabel(actor)}` : ''}
          </Typography.Text>
        </div>
      ),
    };
  });

  return (
    <Collapse
      ghost
      className="memory-detail-collapse"
      items={[
        {
          key: 'history',
          label: <span><HistoryOutlined /> 状态记录（{items.length}）</span>,
          children: items.length ? (
            <Timeline className="memory-status-timeline" items={items} />
          ) : (
            <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无状态记录" />
          ),
        },
      ]}
    />
  );
}

export function MemorySafetyNotice() {
  return (
    <Typography.Text type="secondary" className="memory-safety-note">
      详情仅包含安全治理字段，不展示原始 Prompt、模型内部推理、RES 或原始案件敏感数据。
    </Typography.Text>
  );
}
