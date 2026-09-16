import {
  ArrowLeftOutlined,
  EyeOutlined,
  InboxOutlined,
  KeyOutlined,
  SafetyCertificateOutlined,
  UndoOutlined,
} from '@ant-design/icons';
import {
  Alert,
  Avatar,
  Button,
  Card,
  Collapse,
  Descriptions,
  Empty,
  Form,
  Input,
  Modal,
  Popconfirm,
  Segmented,
  Select,
  Space,
  Spin,
  Switch,
  Table,
  Tabs,
  Tag,
  Tooltip,
  Typography,
  message as antdMessage,
} from 'antd';
import type { TableColumnsType, TabsProps } from 'antd';
import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  fetchActiveMemories,
  fetchArchivedMemories,
  fetchMemoryDetail,
  fetchMemoryPendingCount,
  fetchPersonalMemoryPreference,
  updatePersonalMemoryPreference,
  updateMemoryStatus,
} from '../services/api';
import type {
  AuthenticatedUser,
  MemoryActivationMode,
  MemoryPresentationItem,
  MemorySafeDetail,
  MemoryType,
} from '../types';
import { MemoryBusinessDetail } from './MemoryBusinessDetail';
import { MemorySafetyNotice, MemoryStatusHistory } from './MemoryStatusHistory';

type PasswordFormValues = {
  currentPassword: string;
  newPassword: string;
  confirmPassword: string;
};

type MemoryTypeFilter = 'all' | MemoryType;
type ActivationModeFilter = 'all' | Extract<MemoryActivationMode, 'human_confirmed' | 'auto_active'>;
type MemoryView = 'active' | 'archived';

export type ReviewerAccountTabKey = 'profile' | 'security' | 'memory';

interface ReviewerAccountPageProps {
  currentUser: AuthenticatedUser;
  activeTab: ReviewerAccountTabKey;
  readOnly?: boolean;
  onBack: () => void;
  onTabChange: (tab: ReviewerAccountTabKey) => void;
}

const ROLE_LABELS: Record<string, string> = {
  admin: '系统管理员',
  auditor: '审核员',
  reviewer: '复核员',
};

const TAB_LABELS: Record<ReviewerAccountTabKey, string> = {
  profile: '人员信息',
  security: '账号安全',
  memory: '记忆管理',
};

const MEMORY_TYPE_LABELS: Record<MemoryType, string> = {
  intent_route_hint: '任务路由',
  policy_search_hint: '政策检索',
  failure_hint: '异常恢复',
  answer_style_hint: '回答组织',
  decision_plan_hint: '处理流程',
};

const MEMORY_LEVEL_LABELS: Record<string, string> = {
  L1_atomic_memory: '单次经验',
  L2_scenario_memory: '场景经验',
  L3_stable_profile_or_playbook: '稳定偏好或标准流程',
};

const CREATION_MODE_LABELS: Record<string, string> = {
  system_extracted: '系统抽取',
  system_consolidated: '系统归纳',
};

const ACTIVATION_MODE_LABELS: Record<string, string> = {
  human_confirmed: '人工确认',
  auto_active: '自动激活',
  pending_review: '待人工确认',
  shadow: '影子观察',
};

const SCOPE_LABELS: Record<string, string> = {
  auditor: '个人',
  team: '团队',
  department: '部门',
  global: '全局',
};

const CONSUMER_LABELS: Record<string, string> = {
  intent_router: '意图识别与任务路由',
  policy_filter_resolver: '政策检索条件整理',
  expert_analysis: '政策与专业依据分析',
  decision_planner: '审核任务步骤规划',
  answer_generator: '审核答复生成',
  recovery_handler: '异常恢复处理',
};

function roleLabel(role: string): string {
  return ROLE_LABELS[role] ?? role;
}

function formatDateTime(value?: string | null): string {
  if (!value) return '未记录';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '未记录' : date.toLocaleString('zh-CN');
}

function scopeLabel(item: MemoryPresentationItem, currentUserId?: string): string {
  const label = SCOPE_LABELS[item.scope.scope_type] ?? item.scope.scope_type;
  if (item.scope.scope_type === 'auditor' && item.scope.scope_id === currentUserId) return label;
  return item.scope.scope_type === 'global' ? label : `${label} · ${item.scope.scope_id}`;
}

function memoryListSummary(item: MemoryPresentationItem): string {
  if (item.memory_type === 'intent_route_hint') {
    if (item.summary.includes('query_evidence_package')) {
      return '处理相似案件查询时，系统建议读取当前案件的基础证据包。';
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

function activationTag(mode?: MemoryActivationMode) {
  return (
    <Tag color={mode === 'auto_active' ? 'blue' : 'green'}>
      {ACTIVATION_MODE_LABELS[mode ?? ''] ?? '来源待确认'}
    </Tag>
  );
}

export default function ReviewerAccountPage({
  currentUser,
  activeTab,
  readOnly = false,
  onBack,
  onTabChange,
}: ReviewerAccountPageProps) {
  const [passwordForm] = Form.useForm<PasswordFormValues>();
  const [passwordModalOpen, setPasswordModalOpen] = useState(false);
  const [memoryTypeFilter, setMemoryTypeFilter] = useState<MemoryTypeFilter>('all');
  const [activationModeFilter, setActivationModeFilter] = useState<ActivationModeFilter>('all');
  const [memoryView, setMemoryView] = useState<MemoryView>('active');
  const [memoryKeyword, setMemoryKeyword] = useState('');
  const [memoryItems, setMemoryItems] = useState<MemoryPresentationItem[]>([]);
  const [archivedMemoryItems, setArchivedMemoryItems] = useState<MemoryPresentationItem[]>([]);
  const [pendingMemoryCount, setPendingMemoryCount] = useState(0);
  const [memoryLoading, setMemoryLoading] = useState(false);
  const [memoryPreferenceLoading, setMemoryPreferenceLoading] = useState(false);
  const [personalMemoryEnabled, setPersonalMemoryEnabled] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [memoryActionKey, setMemoryActionKey] = useState<string | null>(null);
  const [selectedMemoryId, setSelectedMemoryId] = useState<string | null>(null);
  const [selectedMemory, setSelectedMemory] = useState<MemorySafeDetail | null>(null);

  const humanConfirmedCount = memoryItems.filter(
    (item) => item.activation_mode === 'human_confirmed',
  ).length;
  const autoActiveCount = memoryItems.filter(
    (item) => item.activation_mode === 'auto_active',
  ).length;
  const displayedMemoryItems = memoryView === 'active' ? memoryItems : archivedMemoryItems;
  const filteredMemoryItems = useMemo(
    () => {
      const keyword = memoryKeyword.trim().toLowerCase();
      return displayedMemoryItems
        .filter((item) => memoryTypeFilter === 'all' || item.memory_type === memoryTypeFilter)
        .filter((item) => activationModeFilter === 'all' || item.activation_mode === activationModeFilter)
        .filter((item) => {
          if (!keyword) return true;
          const values = [
            MEMORY_TYPE_LABELS[item.memory_type],
            MEMORY_LEVEL_LABELS[item.memory_level],
            CREATION_MODE_LABELS[item.creation_mode],
            ACTIVATION_MODE_LABELS[item.activation_mode],
            item.summary,
            scopeLabel(item, currentUser.id),
            ...item.source,
          ];
          return values.some((value) => value?.toLowerCase().includes(keyword));
        });
    },
    [activationModeFilter, currentUser.id, displayedMemoryItems, memoryKeyword, memoryTypeFilter],
  );
  const selectedMemoryItem = selectedMemory
    ?? memoryItems.find((item) => item.memory_id === selectedMemoryId)
    ?? archivedMemoryItems.find((item) => item.memory_id === selectedMemoryId)
    ?? null;

  const loadMemoryData = useCallback(async () => {
    setMemoryLoading(true);
    try {
      const [pending, activeMemories, archivedMemories, preference] = await Promise.all([
        fetchMemoryPendingCount(),
        fetchActiveMemories(),
        fetchArchivedMemories(),
        fetchPersonalMemoryPreference(),
      ]);
      setPendingMemoryCount(pending.count);
      setMemoryItems(activeMemories);
      setArchivedMemoryItems(archivedMemories);
      setPersonalMemoryEnabled(preference.enabled);
    } catch (error) {
      antdMessage.error(error instanceof Error ? error.message : '记忆数据加载失败');
    } finally {
      setMemoryLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!readOnly && activeTab === 'memory') void loadMemoryData();
  }, [activeTab, loadMemoryData, readOnly]);

  function handlePasswordFinish() {
    passwordForm.resetFields();
    setPasswordModalOpen(false);
    antdMessage.success('密码修改表单校验通过。');
  }

  async function handlePersonalMemoryChange(enabled: boolean) {
    setMemoryPreferenceLoading(true);
    try {
      const preference = await updatePersonalMemoryPreference(enabled);
      setPersonalMemoryEnabled(preference.enabled);
      antdMessage.success(
        preference.enabled
          ? '个人跨案件记忆已开启'
          : '个人跨案件记忆已关闭，已有记忆仍会保留',
      );
    } catch (error) {
      antdMessage.error(error instanceof Error ? error.message : '记忆设置更新失败');
    } finally {
      setMemoryPreferenceLoading(false);
    }
  }

  async function openMemoryDetail(item: MemoryPresentationItem) {
    setSelectedMemoryId(item.memory_id);
    setSelectedMemory(null);
    setDetailLoading(true);
    try {
      setSelectedMemory(await fetchMemoryDetail(item.memory_id));
    } catch (error) {
      setSelectedMemoryId(null);
      antdMessage.error(error instanceof Error ? error.message : '记忆详情加载失败');
    } finally {
      setDetailLoading(false);
    }
  }

  async function governMemory(item: MemoryPresentationItem, action: 'archive' | 'restore') {
    setMemoryActionKey(`${item.memory_id}:${action}`);
    try {
      await updateMemoryStatus(item.memory_id, action);
      const successMessage = {
        archive: '记忆已停用并归档',
        restore: '记忆已恢复并重新参与召回',
      }[action];
      antdMessage.success(successMessage);
      if (selectedMemoryId === item.memory_id) {
        setSelectedMemoryId(null);
        setSelectedMemory(null);
      }
      await loadMemoryData();
    } catch (error) {
      antdMessage.error(error instanceof Error ? error.message : '记忆治理操作失败');
    } finally {
      setMemoryActionKey(null);
    }
  }

  const memoryColumns: TableColumnsType<MemoryPresentationItem> = [
    {
      title: '记忆条目',
      key: 'identity',
      width: 230,
      render: (_, record) => (
        <div className="account-memory-title">
          <Typography.Text strong>{MEMORY_TYPE_LABELS[record.memory_type]}</Typography.Text>
          <Space size={[4, 4]} wrap>
            <Tag>{MEMORY_LEVEL_LABELS[record.memory_level] ?? record.memory_level}</Tag>
            {activationTag(record.activation_mode)}
          </Space>
        </div>
      ),
    },
    {
      title: '记忆内容',
      dataIndex: 'summary',
      width: 420,
      render: (_: string, record) => (
        <Typography.Text className="account-memory-summary">
          {memoryListSummary(record)}
        </Typography.Text>
      ),
    },
    {
      title: '适用范围',
      key: 'scope',
      width: 180,
      render: (_, record) => scopeLabel(record, currentUser.id),
    },
    {
      title: memoryView === 'active' ? '激活时间' : '归档时间',
      key: memoryView === 'active' ? 'activated_at' : 'archived_at',
      width: 170,
      render: (_, record) => formatDateTime(
        memoryView === 'active'
          ? record.activated_at ?? record.last_verified_at ?? record.created_at
          : record.archived_at ?? record.created_at,
      ),
    },
    {
      title: '操作',
      key: 'action',
      width: 240,
      render: (_, record) => (
        <Space size={4} wrap>
          <Button
            type="text"
            size="small"
            icon={<EyeOutlined />}
            onClick={() => void openMemoryDetail(record)}
          >
            查看
          </Button>
          {memoryView === 'active' ? (
            <Popconfirm
              title="停用并归档该记忆"
              description="停用后将不再参与召回，并保留审计记录、来源链和恢复入口。"
              okText="停用"
              cancelText="取消"
              onConfirm={() => void governMemory(record, 'archive')}
            >
              <Button
                type="text"
                size="small"
                icon={<InboxOutlined />}
                loading={memoryActionKey === `${record.memory_id}:archive`}
              >
                停用
              </Button>
            </Popconfirm>
          ) : (
            <Popconfirm
              title="恢复该记忆"
              description="恢复后将重新进入当前可用记忆，并按原适用范围参与召回。"
              okText="恢复"
              cancelText="取消"
              onConfirm={() => void governMemory(record, 'restore')}
            >
              <Button
                type="text"
                size="small"
                icon={<UndoOutlined />}
                loading={memoryActionKey === `${record.memory_id}:restore`}
              >
                恢复
              </Button>
            </Popconfirm>
          )}
        </Space>
      ),
    },
  ];

  const tabItems: TabsProps['items'] = [
    {
      key: 'profile',
      label: TAB_LABELS.profile,
      children: (
        <div className="account-profile-panel">
          <Card className="account-overview-card">
            <div className="account-profile-header">
              <div className="account-profile-identity">
                <Avatar className="account-profile-avatar" size={56}>
                  {currentUser.display_name.trim().slice(0, 1) || '审'}
                </Avatar>
                <div>
                  <Typography.Title level={4}>{currentUser.display_name}</Typography.Title>
                  <Typography.Text type="secondary">{currentUser.username}</Typography.Text>
                </div>
              </div>
              <div className="account-profile-status">
                <span aria-hidden="true" />
                在岗
              </div>
            </div>

            <dl className="account-profile-facts">
              <div>
                <dt>角色权限</dt>
                <dd>
                  <Space wrap size={6}>
                    {currentUser.roles.map((role) => (
                      <Tag className="account-role-tag" bordered={false} key={role}>
                        {roleLabel(role)}
                      </Tag>
                    ))}
                  </Space>
                </dd>
              </div>
              <div>
                <dt>所属部门</dt>
                <dd>{currentUser.department || '未配置'}</dd>
              </div>
              <div>
                <dt>业务小组</dt>
                <dd>医保稽核二组</dd>
              </div>
              <div>
                <dt>上级复核人</dt>
                <dd>王倩</dd>
              </div>
            </dl>
          </Card>
        </div>
      ),
    },
    {
      key: 'security',
      label: TAB_LABELS.security,
      children: (
        <div className="account-security-panel">
          <Card className="account-overview-card">
            <div className="account-security-header">
              <div className="account-security-heading">
                <span className="account-security-mark" aria-hidden="true">
                  <SafetyCertificateOutlined />
                </span>
                <div>
                  <Typography.Title level={4}>账号保护正常</Typography.Title>
                  <Typography.Text type="secondary">
                    密码与内网终端校验已启用
                  </Typography.Text>
                </div>
              </div>
              <Button
                icon={<KeyOutlined />}
                disabled={readOnly}
                onClick={() => setPasswordModalOpen(true)}
              >
                修改密码
              </Button>
            </div>

            <dl className="account-security-facts">
              <div>
                <dt>密码状态</dt>
                <dd>有效</dd>
                <small>有效至 2026-10-22</small>
              </div>
              <div>
                <dt>验证方式</dt>
                <dd>账号密码</dd>
                <small>内网终端校验</small>
              </div>
              <div>
                <dt>当前会话</dt>
                <dd>1 个活跃会话</dd>
                <small>30 分钟无操作后失效</small>
              </div>
            </dl>

            <section className="account-security-activity">
              <Typography.Text strong>最近安全活动</Typography.Text>
              <dl>
                <div>
                  <dt>最近登录</dt>
                  <dd>2026-08-12 08:41 · MG-WS-018 · Chrome</dd>
                </div>
                <div>
                  <dt>异常尝试</dt>
                  <dd>2026-08-11 23:17 · MG-WS-011 · 3 次密码错误，已自动解锁</dd>
                </div>
              </dl>
            </section>
          </Card>
        </div>
      ),
    },
    {
      key: 'memory',
      label: TAB_LABELS.memory,
      children: (
        <div className="account-memory-layout">
          <Card className="account-memory-control-card">
            <div className="account-memory-control">
              <div>
                <Typography.Text strong>个人跨案件记忆</Typography.Text>
                <Typography.Paragraph type="secondary">
                  待确认 {pendingMemoryCount} 条 · 当前可用 {memoryItems.length} 条 ·
                  归档历史 {archivedMemoryItems.length} 条 · 人工确认 {humanConfirmedCount} 条 ·
                  自动激活 {autoActiveCount} 条
                </Typography.Paragraph>
              </div>
              <div className="account-memory-control-actions">
                <Tooltip
                  title={personalMemoryEnabled
                    ? '关闭后将停止跨案件记忆的调用和新增沉淀'
                    : '开启后将恢复跨案件记忆的调用和新增沉淀'}
                >
                  <Switch
                    className="account-memory-toggle"
                    aria-label="个人跨案件记忆开关"
                    checked={personalMemoryEnabled}
                    checkedChildren="已开启"
                    unCheckedChildren="已关闭"
                    loading={memoryPreferenceLoading}
                    onChange={(enabled) => void handlePersonalMemoryChange(enabled)}
                  />
                </Tooltip>
              </div>
            </div>
          </Card>

          <Card
            className="account-memory-table-card"
            title="记忆库"
            extra={(
              <Segmented
                className="account-memory-view-switch"
                value={memoryView}
                onChange={(value) => setMemoryView(value as MemoryView)}
                options={[
                  { label: `当前可用 ${memoryItems.length}`, value: 'active' },
                  { label: `归档历史 ${archivedMemoryItems.length}`, value: 'archived' },
                ]}
              />
            )}
          >
            <div className="account-memory-toolbar">
              <Input.Search
                allowClear
                className="account-memory-search"
                placeholder="搜索记忆类型、范围或内容"
                value={memoryKeyword}
                onChange={(event) => setMemoryKeyword(event.target.value)}
              />
              <Space className="account-memory-toolbar-actions" wrap>
                <Typography.Text type="secondary">
                  匹配 {filteredMemoryItems.length} / {displayedMemoryItems.length}
                </Typography.Text>
                <Select<MemoryTypeFilter>
                  aria-label="按记忆类型筛选"
                  value={memoryTypeFilter}
                  onChange={setMemoryTypeFilter}
                  options={[
                    { label: '全部类型', value: 'all' },
                    ...Object.entries(MEMORY_TYPE_LABELS).map(([value, label]) => ({
                      label,
                      value: value as MemoryType,
                    })),
                  ]}
                />
                <Select<ActivationModeFilter>
                  aria-label="按激活来源筛选"
                  value={activationModeFilter}
                  onChange={setActivationModeFilter}
                  options={[
                    { label: '全部来源', value: 'all' },
                    { label: '人工确认', value: 'human_confirmed' },
                    { label: '自动激活', value: 'auto_active' },
                  ]}
                />
              </Space>
            </div>
            <Table<MemoryPresentationItem>
              rowKey="memory_id"
              className="account-memory-table"
              columns={memoryColumns}
              dataSource={filteredMemoryItems}
              loading={memoryLoading}
              pagination={{
                pageSize: 8,
                showSizeChanger: false,
                showTotal: (total, range) =>
                  `第 ${range[0]}-${range[1]} 条 / 共 ${total} 条`,
              }}
              tableLayout="fixed"
              scroll={{ x: 1240 }}
              locale={{
                emptyText: (
                  <Empty
                    image={Empty.PRESENTED_IMAGE_SIMPLE}
                    description={memoryView === 'active'
                      ? '暂无已激活记忆，候选确认或自动激活后会显示在这里'
                      : '暂无归档记忆'}
                  />
                ),
              }}
            />
          </Card>
        </div>
      ),
    },
  ];
  const visibleTabItems = readOnly
    ? tabItems.filter((item) => item?.key !== 'memory')
    : tabItems;
  const effectiveActiveTab = readOnly && activeTab === 'memory' ? 'profile' : activeTab;

  return (
    <div className="account-page-shell">
      <div className="account-page-hero">
        <div className="account-page-title-row">
          <Tooltip title="返回业务" placement="bottom">
            <Button
              className="account-back-button"
              aria-label="返回业务"
              icon={<ArrowLeftOutlined />}
              type="text"
              onClick={onBack}
            />
          </Tooltip>
          <div className="account-page-heading">
            <Typography.Title level={3}>审核人员中心</Typography.Title>
          </div>
        </div>
      </div>

      {readOnly && (
        <Alert
          className="showcase-readonly-banner"
          type="info"
          showIcon
          message="只读演示账号"
          description="账号修改和跨案件记忆治理未在公网环境开放。"
        />
      )}

      <Tabs
        className="account-tabs"
        activeKey={effectiveActiveTab}
        onChange={(key) => onTabChange(key as ReviewerAccountTabKey)}
        items={visibleTabItems}
      />

      <Modal
        className="account-password-modal"
        title="修改登录密码"
        centered
        open={passwordModalOpen}
        width={460}
        okText="确认修改"
        cancelText="取消"
        onOk={() => passwordForm.submit()}
        onCancel={() => {
          passwordForm.resetFields();
          setPasswordModalOpen(false);
        }}
        destroyOnHidden
      >
        <Form
          className="account-password-form"
          form={passwordForm}
          layout="vertical"
          requiredMark={false}
          onFinish={handlePasswordFinish}
        >
          <Form.Item
            label="当前密码"
            name="currentPassword"
            rules={[{ required: true, message: '请输入当前密码' }]}
          >
            <Input.Password
              autoComplete="current-password"
              placeholder="请输入当前密码"
            />
          </Form.Item>
          <Form.Item
            label="新密码"
            name="newPassword"
            rules={[
              { required: true, message: '请输入新密码' },
              { min: 8, message: '新密码至少 8 位' },
              ({ getFieldValue }) => ({
                validator(_, value) {
                  if (!value || value !== getFieldValue('currentPassword')) {
                    return Promise.resolve();
                  }
                  return Promise.reject(new Error('新密码不能与当前密码相同'));
                },
              }),
            ]}
          >
            <Input.Password autoComplete="new-password" placeholder="至少 8 位" />
          </Form.Item>
          <Form.Item
            label="确认新密码"
            name="confirmPassword"
            dependencies={['newPassword']}
            rules={[
              { required: true, message: '请再次输入新密码' },
              ({ getFieldValue }) => ({
                validator(_, value) {
                  if (!value || getFieldValue('newPassword') === value) {
                    return Promise.resolve();
                  }
                  return Promise.reject(new Error('两次输入的新密码不一致'));
                },
              }),
            ]}
          >
            <Input.Password autoComplete="new-password" placeholder="再次输入新密码" />
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        className="account-memory-detail-modal"
        title={
          selectedMemoryItem
            ? `${MEMORY_TYPE_LABELS[selectedMemoryItem.memory_type]} · ${MEMORY_LEVEL_LABELS[selectedMemoryItem.memory_level]}`
            : '记忆详情'
        }
        centered
        open={Boolean(selectedMemoryId)}
        width={760}
        footer={
          <Button
            type="primary"
            onClick={() => {
              setSelectedMemoryId(null);
              setSelectedMemory(null);
            }}
          >
            关闭
          </Button>
        }
        onCancel={() => {
          setSelectedMemoryId(null);
          setSelectedMemory(null);
        }}
        destroyOnHidden
      >
        <Spin spinning={detailLoading}>
          {selectedMemory && (
            <div className="account-memory-detail">
              <MemoryBusinessDetail
                memory={selectedMemory}
                scopeText={scopeLabel(selectedMemory, currentUser.id)}
              />

              <Collapse
                ghost
                className="memory-detail-collapse"
                items={[
                  {
                    key: 'governance',
                    label: '治理记录',
                    children: (
                      <Descriptions
                        column={1}
                        size="small"
                        bordered
                        items={[
                    { key: 'type', label: '记忆用途', children: MEMORY_TYPE_LABELS[selectedMemory.memory_type] },
                    { key: 'level', label: '成熟程度', children: MEMORY_LEVEL_LABELS[selectedMemory.memory_level] },
                    {
                      key: 'origin',
                      label: '产生与激活',
                      children: (
                        <Space size={4} wrap>
                          <Tag>{CREATION_MODE_LABELS[selectedMemory.creation_mode]}</Tag>
                          {activationTag(selectedMemory.activation_mode)}
                        </Space>
                      ),
                    },
                    {
                      key: 'activated_by',
                      label: '激活操作人',
                      children: selectedMemory.activation_mode === 'auto_active'
                        ? '系统'
                        : selectedMemory.activated_by === currentUser.id
                          ? currentUser.display_name
                          : selectedMemory.activated_by || '未记录',
                    },
                    { key: 'activated_at', label: '激活时间', children: formatDateTime(selectedMemory.activated_at) },
                    ...(selectedMemory.status === 'archived' ? [
                      {
                        key: 'archived_by',
                        label: '归档操作人',
                        children: selectedMemory.archived_by === currentUser.id
                          ? currentUser.display_name
                          : selectedMemory.archived_by || '未记录',
                      },
                      {
                        key: 'archived_at',
                        label: '归档时间',
                        children: formatDateTime(selectedMemory.archived_at),
                      },
                    ] : []),
                    {
                      key: 'consumer',
                      label: '使用环节',
                      children: selectedMemory.allowed_consumers
                        .map((consumer) => CONSUMER_LABELS[consumer] ?? '系统内部处理节点')
                        .join('、') || '未限定',
                    },
                    {
                      key: 'scores',
                      label: '维护指标',
                      children: `置信度 ${Math.round(selectedMemory.confidence * 100)}% · 重要性 ${Math.round(selectedMemory.importance_score * 100)}% · 新鲜度 ${Math.round(selectedMemory.freshness_score * 100)}%`,
                    },
                        ]}
                      />
                    ),
                  },
                ]}
              />

              <MemoryStatusHistory events={selectedMemory.status_events} />
              <MemorySafetyNotice />
            </div>
          )}
        </Spin>
      </Modal>
    </div>
  );
}
