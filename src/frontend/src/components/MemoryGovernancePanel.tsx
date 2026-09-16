import {
  CheckOutlined,
  ClockCircleOutlined,
  CloseOutlined,
  EyeOutlined,
  InboxOutlined,
  ReloadOutlined,
  SearchOutlined,
} from '@ant-design/icons';
import {
  Alert,
  Button,
  Collapse,
  Drawer,
  Empty,
  Modal,
  Select,
  Space,
  Spin,
  Tag,
  Tooltip,
  Typography,
  message,
} from 'antd';
import { Brain } from 'lucide-react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { CSSProperties } from 'react';
import {
  fetchMemoryCandidates,
  fetchMemoryDetail,
  fetchMemoryPendingCount,
  previewMemory,
  updateMemoryStatus,
} from '../services/api';
import type {
  MemoryHintPack,
  MemoryPreview,
  MemoryPresentationItem,
  MemorySafeDetail,
} from '../types';
import { MemoryBusinessDetail } from './MemoryBusinessDetail';
import { MemorySafetyNotice, MemoryStatusHistory } from './MemoryStatusHistory';

const TYPE_LABELS: Record<string, string> = {
  intent_route_hint: '任务路由',
  policy_search_hint: '政策检索',
  failure_hint: '异常恢复',
  answer_style_hint: '回答组织',
  decision_plan_hint: '处理流程',
};

const TYPE_COLORS: Record<string, string> = {
  intent_route_hint: '49 115 205',
  policy_search_hint: '41 137 92',
  failure_hint: '193 72 72',
  answer_style_hint: '137 91 174',
  decision_plan_hint: '201 125 35',
};

const LEVEL_OPACITY: Record<string, number> = {
  L1_atomic_memory: 0.24,
  L2_scenario_memory: 0.42,
  L3_stable_profile_or_playbook: 0.62,
};

const LEVEL_LABELS: Record<string, string> = {
  L1_atomic_memory: '单次经验',
  L2_scenario_memory: '场景经验',
  L3_stable_profile_or_playbook: '稳定偏好或标准流程',
};

const SCOPE_LABELS: Record<string, string> = {
  auditor: '个人',
  team: '团队',
  department: '部门',
  global: '全局',
};

const DELIVERY_LABELS: Record<string, string> = {
  control_hints: '路由或流程建议',
  prompt_contexts: '回答参考信息',
  tool_param_hints: '工具使用建议',
  trace_refs: '来源记录',
};

const CONSUMERS = [
  { value: 'intent_router', label: '意图识别与任务路由' },
  { value: 'policy_filter_resolver', label: '政策检索条件整理' },
  { value: 'expert_analysis', label: '政策与专业依据分析' },
  { value: 'decision_planner', label: '审核任务步骤规划' },
  { value: 'answer_generator', label: '审核答复生成' },
  { value: 'recovery_handler', label: '异常恢复处理' },
];

const HINT_PACK_SECTIONS: Array<{
  key: keyof Pick<MemoryHintPack, 'control_hints' | 'prompt_contexts' | 'tool_param_hints' | 'trace_refs'>;
  label: string;
}> = [
  { key: 'control_hints', label: '任务路由或流程建议' },
  { key: 'prompt_contexts', label: '回答时参考的信息' },
  { key: 'tool_param_hints', label: '检索或工具使用建议' },
  { key: 'trace_refs', label: '脱敏来源记录' },
];

const DELIVERY_DESCRIPTIONS: Record<string, string> = {
  control_hints: '帮助系统选择任务路由或安排处理步骤',
  prompt_contexts: '作为回答组织和必要背景的参考',
  tool_param_hints: '作为检索条件或工具参数的候选值',
  trace_refs: '用于核查这条记忆的脱敏来源',
};

type GovernanceAction = 'confirm' | 'reject' | 'snooze' | 'archive';

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function formatDateTime(value: unknown): string {
  if (typeof value !== 'string' && !(value instanceof Date)) return '时间未知';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '时间未知' : date.toLocaleString('zh-CN');
}

function consumerLabel(value: string): string {
  return CONSUMERS.find((item) => item.value === value)?.label ?? '系统内部处理节点';
}

function candidateSummary(item: MemoryPresentationItem): string {
  if (item.memory_type === 'intent_route_hint') {
    if (item.summary.includes('query_evidence_package')) {
      return '处理相似案件查询时，系统建议读取当前案件的基础证据包，为后续审核准备事实依据。';
    }
    if (item.summary.includes('ask_policy_expert')) {
      return '处理相似政策问题时，系统建议进入政策与专业依据分析环节。';
    }
    return '相似案件查询可参考这条已经验证的任务路由。';
  }
  if (item.memory_type === 'policy_search_hint') {
    return '相似政策问题可复用这条已经验证的检索条件和回答要点。';
  }
  if (item.memory_type === 'failure_hint') {
    return '出现相似校验问题时，可参考这条已经验证的恢复方式。';
  }
  if (item.memory_type === 'decision_plan_hint') {
    return '处理相似的多步骤任务时，可参考这条已经验证的执行顺序。';
  }
  return /[a-z]+_[a-z]+/i.test(item.summary)
    ? '生成回答时，可沿用审核人员已经确认的组织偏好。'
    : item.summary;
}

function previewEffect(consumer: string): string {
  return {
    intent_router: '提供任务类型和路由建议；当前仅作影子验证，不会直接改变本次执行路线',
    policy_filter_resolver: '提供可复用的政策检索范围和需要覆盖的信息点',
    expert_analysis: '提醒专业分析需要核验的政策范围、证据重点和回答边界',
    decision_planner: '为相似任务提供处理步骤、前后依赖和执行限制',
    answer_generator: '按照审核人员已经确认的偏好组织回答',
    recovery_handler: '在相同校验问题出现时提供经过验证的恢复建议',
  }[consumer] ?? '在当前处理环节提供经过治理的参考信息';
}

function scopeLabel(item: MemoryPresentationItem): string {
  const label = SCOPE_LABELS[item.scope.scope_type] ?? item.scope.scope_type;
  return item.scope.scope_type === 'auditor' || item.scope.scope_type === 'global'
    ? label
    : `${label} · ${item.scope.scope_id}`;
}

function bubbleSize(observationCount: number, maxObservationCount: number): number {
  if (maxObservationCount <= 1) return 144;
  const ratio = Math.log1p(Math.max(0, observationCount - 1))
    / Math.log1p(maxObservationCount - 1);
  return Math.round(112 + ratio * 78);
}

function observationCount(item: MemoryPresentationItem): number {
  return Number.isFinite(item.observation_count) && item.observation_count > 0
    ? item.observation_count
    : Math.max(1, item.source.length);
}

type PositionedBubble = {
  item: MemoryPresentationItem;
  size: number;
  left: number;
  top: number;
};

function stableHash(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function stableUnit(value: string): number {
  return stableHash(value) / 0xffffffff;
}

function buildBubbleLayout(
  items: MemoryPresentationItem[],
  fieldWidth: number,
): { height: number; bubbles: PositionedBubble[] } {
  const width = Math.max(260, fieldWidth);
  const padding = 8;
  const collisionGap = 10;
  const maxObservationCount = Math.max(...items.map(observationCount), 1);
  const maxResponsiveSize = Math.max(112, width * 0.42);
  const pending = items
    .map((item) => ({
      item,
      size: Math.min(bubbleSize(observationCount(item), maxObservationCount), maxResponsiveSize),
    }))
    .sort((left, right) => (
      right.size - left.size
      || stableHash(left.item.memory_id) - stableHash(right.item.memory_id)
    ));
  const totalArea = pending.reduce((sum, bubble) => sum + bubble.size ** 2, 0);
  let height = Math.max(260, Math.ceil(totalArea / Math.max(width * 0.52, 1)));
  const placed: PositionedBubble[] = [];

  for (const bubble of pending) {
    let position: PositionedBubble | null = null;
    let expansion = 0;
    while (!position && expansion < 12) {
      const availableX = Math.max(0, width - bubble.size - padding * 2);
      const availableY = Math.max(0, height - bubble.size - padding * 2);
      for (let attempt = 0; attempt < 220; attempt += 1) {
        const left = padding + stableUnit(`${bubble.item.memory_id}:${expansion}:${attempt}:x`) * availableX;
        const top = padding + stableUnit(`${bubble.item.memory_id}:${expansion}:${attempt}:y`) * availableY;
        const centerX = left + bubble.size / 2;
        const centerY = top + bubble.size / 2;
        const overlaps = placed.some((existing) => {
          const existingCenterX = existing.left + existing.size / 2;
          const existingCenterY = existing.top + existing.size / 2;
          const minimumDistance = (bubble.size + existing.size) / 2 + collisionGap;
          return Math.hypot(centerX - existingCenterX, centerY - existingCenterY) < minimumDistance;
        });
        if (!overlaps) {
          position = { item: bubble.item, size: bubble.size, left, top };
          break;
        }
      }
      if (!position) {
        height += Math.max(48, Math.round(bubble.size * 0.42));
        expansion += 1;
      }
    }
    if (!position) {
      const availableX = Math.max(0, width - bubble.size - padding * 2);
      const top = height + collisionGap;
      position = {
        item: bubble.item,
        size: bubble.size,
        left: padding + stableUnit(`${bubble.item.memory_id}:fallback`) * availableX,
        top,
      };
      height = top + bubble.size + padding;
    }
    placed.push(position);
  }

  return { height, bubbles: placed };
}

const CandidateBubblePool = memo(function CandidateBubblePool({
  items,
  onOpen,
}: {
  items: MemoryPresentationItem[];
  onOpen: (item: MemoryPresentationItem) => void;
}) {
  const fieldRef = useRef<HTMLDivElement>(null);
  const [fieldWidth, setFieldWidth] = useState(0);
  const layout = useMemo(
    () => buildBubbleLayout(items, fieldWidth || 840),
    [fieldWidth, items],
  );

  useEffect(() => {
    const field = fieldRef.current;
    if (!field) return undefined;
    const updateWidth = () => {
      const nextWidth = Math.round(field.getBoundingClientRect().width);
      setFieldWidth((current) => (current === nextWidth ? current : nextWidth));
    };
    updateWidth();
    const observer = new ResizeObserver(updateWidth);
    observer.observe(field);
    return () => observer.disconnect();
  }, []);

  return (
    <div className="memory-bubble-pool">
      <div className="memory-bubble-legend" aria-label="记忆气泡图例">
        <div className="memory-bubble-legend-group">
          {Object.entries(TYPE_LABELS).map(([type, label]) => (
            <span key={type}>
              <i style={{ background: `rgb(${TYPE_COLORS[type]})` }} />
              {label}
            </span>
          ))}
        </div>
        <div className="memory-bubble-legend-group memory-level-legend">
          {Object.entries(LEVEL_LABELS).map(([level, label]) => (
            <span key={level}>
              <i style={{ '--memory-level-opacity': LEVEL_OPACITY[level] } as CSSProperties} />
              {label.split(' ')[0]}
            </span>
          ))}
        </div>
      </div>

      <div
        ref={fieldRef}
        className="memory-bubble-field"
        style={{ height: layout.height }}
      >
        {layout.bubbles.map(({ item, size, left, top }) => {
          const frequency = observationCount(item);
          const style = {
            '--memory-bubble-size': `${size}px`,
            '--memory-bubble-color': TYPE_COLORS[item.memory_type] ?? '96 112 128',
            '--memory-bubble-opacity': LEVEL_OPACITY[item.memory_level] ?? 0.3,
            left,
            top,
          } as CSSProperties;
          return (
            <Tooltip
              key={item.memory_id}
              placement="top"
              title={
                <div className="memory-bubble-tooltip">
                  <strong>{TYPE_LABELS[item.memory_type] ?? item.memory_type}</strong>
                  <span>{candidateSummary(item)}</span>
                  <span>来源：已完成的案件助手任务</span>
                  <span>时间：{formatDateTime(item.created_at)}</span>
                  <span>置信度：{percent(item.confidence)}</span>
                  <span>适用范围：{scopeLabel(item)}</span>
                </div>
              }
            >
              <button
                type="button"
                className="memory-candidate-bubble"
                style={style}
                aria-label={`${TYPE_LABELS[item.memory_type] ?? item.memory_type}，出现 ${frequency} 次，${candidateSummary(item)}`}
                onClick={() => onOpen(item)}
              >
                <span className="memory-bubble-type">{TYPE_LABELS[item.memory_type] ?? item.memory_type}</span>
                <strong>{candidateSummary(item)}</strong>
                <span className="memory-bubble-frequency">出现 {frequency} 次</span>
              </button>
            </Tooltip>
          );
        })}
      </div>
    </div>
  );
});

function HintPackPreview({ preview }: { preview: MemoryPreview }) {
  const pack = preview.hint_pack;
  if (!pack) return null;
  const sections = HINT_PACK_SECTIONS.filter(({ key }) => pack[key].length > 0);
  if (!sections.length) return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="没有可投放字段" />;

  return (
    <div className="memory-hint-pack">
      <div className="memory-delivery-list">
        {sections.map(({ key, label }) => (
          <div key={key}>
            <div>
              <Typography.Text strong>{label}</Typography.Text>
              <Typography.Text type="secondary">{DELIVERY_DESCRIPTIONS[key]}</Typography.Text>
            </div>
            <Tag>{pack[key].length} 条</Tag>
          </div>
        ))}
      </div>
      <Collapse
        ghost
        className="memory-technical-collapse"
        items={[
          {
            key: 'preview-technical',
            label: '技术投放信息（一般无需查看）',
            children: (
              <div className="memory-technical-content">
                <p><strong>允许字段：</strong>{preview.allowed_fields.join('、') || '无'}</p>
                <pre className="memory-json-preview">{JSON.stringify(pack, null, 2)}</pre>
              </div>
            ),
          },
        ]}
      />
    </div>
  );
}

const MemoryGovernancePanel = memo(function MemoryGovernancePanel({
  taskContext,
}: {
  taskContext: Record<string, unknown>;
}) {
  const [open, setOpen] = useState(false);
  const [pendingCount, setPendingCount] = useState(0);
  const [candidates, setCandidates] = useState<MemoryPresentationItem[]>([]);
  const [selected, setSelected] = useState<MemorySafeDetail | null>(null);
  const [preview, setPreview] = useState<MemoryPreview | null>(null);
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [actingKey, setActingKey] = useState<string | null>(null);
  const [previewConsumer, setPreviewConsumer] = useState('expert_analysis');

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [count, candidateRows] = await Promise.all([
        fetchMemoryPendingCount(),
        fetchMemoryCandidates(),
      ]);
      setPendingCount(count.count);
      setCandidates(candidateRows);
    } catch (error) {
      message.error(error instanceof Error ? error.message : '记忆列表加载失败');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const openDetail = async (item: MemoryPresentationItem) => {
    setDetailLoading(true);
    try {
      setSelected(await fetchMemoryDetail(item.memory_id));
      setPreview(null);
    } catch (error) {
      message.error(error instanceof Error ? error.message : '记忆详情加载失败');
    } finally {
      setDetailLoading(false);
    }
  };

  const act = async (item: MemoryPresentationItem, action: GovernanceAction) => {
    setActingKey(`${item.memory_id}:${action}`);
    try {
      const snoozedUntil = action === 'snooze'
        ? new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString()
        : undefined;
      await updateMemoryStatus(item.memory_id, action, snoozedUntil);
      message.success(
        action === 'confirm'
          ? '记忆已确认并进入激活队列'
          : action === 'reject'
            ? '记忆已拒绝'
            : action === 'snooze'
              ? '记忆已暂存，7 天后可重新处理'
              : '记忆已归档',
      );
      await load();
      if (selected?.memory_id === item.memory_id) setSelected(null);
    } catch (error) {
      message.error(error instanceof Error ? error.message : '记忆治理操作失败');
    } finally {
      setActingKey(null);
    }
  };

  const runPreview = async () => {
    if (!selected) return;
    setPreviewLoading(true);
    try {
      setPreview(await previewMemory(selected.memory_id, previewConsumer, taskContext));
    } catch (error) {
      message.error(error instanceof Error ? error.message : '记忆投放预览失败');
    } finally {
      setPreviewLoading(false);
    }
  };

  return (
    <>
      <Tooltip title={pendingCount ? `${pendingCount} 条待确认记忆` : '记忆'}>
        <span className="memory-entry-control">
          <Button
            type="text"
            shape="circle"
            className="case-agent-tool-button memory-entry-button"
            icon={<Brain aria-hidden="true" />}
            aria-label={pendingCount ? `记忆，${pendingCount} 条待确认` : '记忆'}
            onClick={() => {
              setOpen(true);
              void load();
            }}
          />
          {pendingCount > 0 && (
            <span className="memory-entry-count" aria-hidden="true">
              {Math.min(pendingCount, 99)}
            </span>
          )}
        </span>
      </Tooltip>

      <Modal
        centered
        width={960}
        open={open}
        footer={null}
        rootClassName="memory-governance-modal"
        onCancel={() => setOpen(false)}
        title={
          <div className="memory-modal-title">
            <Typography.Text strong>记忆确认</Typography.Text>
            <Tooltip title="刷新记忆">
              <Button
                size="small"
                className="memory-refresh-button"
                aria-label="刷新记忆"
                icon={<ReloadOutlined />}
                onClick={() => void load()}
              />
            </Tooltip>
          </div>
        }
      >
        <Spin spinning={detailLoading}>
          <section className="memory-candidate-section">
            <header className="memory-candidate-heading">
              <Typography.Text strong>待确认记忆</Typography.Text>
              <Tag color="blue">{pendingCount} 条待确认</Tag>
            </header>
            {loading ? (
              <div className="memory-loading"><Spin /></div>
            ) : candidates.length ? (
              <CandidateBubblePool items={candidates} onOpen={openDetail} />
            ) : (
              <Empty description="暂无待确认记忆" />
            )}
          </section>
        </Spin>
      </Modal>

      <Drawer
        title={selected
          ? `${TYPE_LABELS[selected.memory_type] ?? selected.memory_type} · ${LEVEL_LABELS[selected.memory_level] ?? selected.memory_level}`
          : '记忆详情'}
        placement="right"
        width="min(680px, 100vw)"
        open={Boolean(selected)}
        rootClassName="memory-detail-drawer"
        onClose={() => {
          setSelected(null);
          setPreview(null);
        }}
      >
        {selected && (
          <Space orientation="vertical" size={14} style={{ width: '100%' }}>
            {selected.status === 'candidate' && (
              <section className="memory-deposit-actions">
                <Typography.Text strong>是否保留这条经验</Typography.Text>
                <Space size={[6, 6]} wrap>
                  <Button
                    type="primary"
                    icon={<CheckOutlined />}
                    loading={actingKey === `${selected.memory_id}:confirm`}
                    disabled={Boolean(actingKey) && actingKey !== `${selected.memory_id}:confirm`}
                    onClick={() => void act(selected, 'confirm')}
                  >
                    确认
                  </Button>
                  <Button
                    danger
                    icon={<CloseOutlined />}
                    loading={actingKey === `${selected.memory_id}:reject`}
                    disabled={Boolean(actingKey) && actingKey !== `${selected.memory_id}:reject`}
                    onClick={() => void act(selected, 'reject')}
                  >
                    拒绝
                  </Button>
                  <Button
                    icon={<ClockCircleOutlined />}
                    loading={actingKey === `${selected.memory_id}:snooze`}
                    disabled={Boolean(actingKey) && actingKey !== `${selected.memory_id}:snooze`}
                    onClick={() => void act(selected, 'snooze')}
                  >
                    暂不处理
                  </Button>
                  <Button
                    icon={<InboxOutlined />}
                    loading={actingKey === `${selected.memory_id}:archive`}
                    disabled={Boolean(actingKey) && actingKey !== `${selected.memory_id}:archive`}
                    onClick={() => void act(selected, 'archive')}
                  >
                    归档
                  </Button>
                </Space>
              </section>
            )}
            <MemoryBusinessDetail memory={selected} scopeText={scopeLabel(selected)} />

            <MemoryStatusHistory events={selected.status_events} />

            <Collapse
              ghost
              className="memory-detail-collapse"
              items={[
                {
                  key: 'usage-preview',
                  label: <span><SearchOutlined /> 当前任务如何使用</span>,
                  children: (
                    <section className="memory-preview-section">
                      <div className="memory-preview-toolbar">
                        <Select
                          value={previewConsumer}
                          options={CONSUMERS}
                          onChange={(value) => {
                            setPreviewConsumer(value);
                            setPreview(null);
                          }}
                          aria-label="选择记忆使用环节"
                          className="memory-consumer-select"
                        />
                        <Button
                          type="primary"
                          icon={<EyeOutlined />}
                          loading={previewLoading}
                          onClick={() => void runPreview()}
                        >
                          查看使用效果
                        </Button>
                      </div>

                      {preview && (
                        <Alert
                          className="memory-preview-alert"
                          type={preview.available ? 'success' : 'warning'}
                          message={preview.available
                            ? `将在“${consumerLabel(preview.consumer)}”环节使用`
                            : '当前处理环节不会使用这条记忆'}
                          description={preview.available ? (
                            <div className="memory-preview-policy">
                              <span>本次作用</span>
                              <span>{previewEffect(preview.consumer)}</span>
                              <span>使用内容</span>
                              <Space size={[4, 4]} wrap>
                                {preview.delivery_locations.map((location) => (
                                  <Tag key={location} color="blue">{DELIVERY_LABELS[location] ?? location}</Tag>
                                ))}
                              </Space>
                              <span>执行边界</span>
                              <span>只使用白名单允许的安全字段，不读取原始问题、案件敏感数据或模型内部过程</span>
                            </div>
                          ) : preview.reason}
                          showIcon
                        />
                      )}
                      {preview?.hint_pack && <HintPackPreview preview={preview} />}
                    </section>
                  ),
                },
              ]}
            />

            <MemorySafetyNotice />
          </Space>
        )}
      </Drawer>
    </>
  );
});

export default MemoryGovernancePanel;
