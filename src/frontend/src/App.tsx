import {
  ArrowLeftOutlined,
  FileAddOutlined,
  LeftOutlined,
  LogoutOutlined,
  MenuOutlined,
  RightOutlined,
  UserOutlined,
} from '@ant-design/icons';
import {
  Alert,
  Avatar,
  Button,
  Card,
  ConfigProvider,
  Empty,
  Grid,
  Layout,
  Space,
  Spin,
  Tag,
  Typography,
} from 'antd';
import { useCallback, useEffect, useMemo, useState } from 'react';
import AssistantWorkspacePage from './components/AssistantWorkspacePage';
import AppealPortalPage from './components/AppealPortalPage';
import AuditQueuePage, {
  type QueueRiskFilter,
  type QueueStatusFilter,
} from './components/AuditQueuePage';
import BusinessSidebar, {
  type BusinessModule,
} from './components/BusinessSidebar';
import CaseDetail from './components/CaseDetail';
import FontPreviewPage from './components/FontPreviewPage';
import GlobalAuditAssistant from './components/GlobalAuditAssistant';
import IngestPage from './components/IngestPage';
import LoginPage from './components/LoginPage';
import MediGuardMark from './components/MediGuardMark';
import ReviewerAccountPage, {
  type ReviewerAccountTabKey,
} from './components/ReviewerAccountPage';
import {
  DEFAULT_TYPEFACE,
  TYPEFACE_STORAGE_KEY,
  getTypefaceOption,
  type TypefaceKey,
} from './config/typefaces';
import {
  ApiError,
  fetchCaseDetail,
  fetchCurrentUser,
  fetchHealth,
  fetchCaseWorkflow,
  fetchCases,
  logout,
} from './services/api';
import type {
  AuthenticatedUser,
  AuditNote,
  CaseFullResponse,
  CaseSummary,
  WorkflowResponse,
  WorkflowStepKey,
} from './types';
import {
  DEFAULT_WORKFLOW_STEP,
  WORKFLOW_STEP_KEYS,
} from './workflow';

type AppRoute =
  | 'login'
  | 'ingest'
  | 'audit'
  | 'audit_detail'
  | 'account'
  | 'appeal_portal'
  | 'assistant'
  | 'font_preview';

interface RouteState {
  route: AppRoute;
  caseId: string | null;
  selectedStep: WorkflowStepKey;
  queueStatus: QueueStatusFilter;
  riskFilter: QueueRiskFilter;
  keyword: string;
  highlightCaseId: string | null;
  batchCount: number | null;
  agentOpen: boolean;
  accountTab: ReviewerAccountTabKey;
  accountNext: string | null;
}

function parseStep(search: URLSearchParams): WorkflowStepKey {
  const step = search.get('step') as WorkflowStepKey | null;
  return step && WORKFLOW_STEP_KEYS.includes(step)
    ? step
    : DEFAULT_WORKFLOW_STEP;
}

function parseStatus(
  search: URLSearchParams,
  fallback: QueueStatusFilter,
): QueueStatusFilter {
  const value = search.get('status') as QueueStatusFilter | null;
  return value === 'pending' || value === 'reviewed' || value === 'all'
    ? value
    : fallback;
}

function parseRisk(search: URLSearchParams): QueueRiskFilter {
  const value = search.get('risk') as QueueRiskFilter | null;
  return value === 'low' ||
    value === 'medium' ||
    value === 'high' ||
    value === 'insufficient'
    ? value
    : 'all';
}

function parseBatchCount(search: URLSearchParams): number | null {
  const value = Number(search.get('batch'));
  return Number.isFinite(value) && value > 0 ? value : null;
}

function parseAccountTab(search: URLSearchParams): ReviewerAccountTabKey {
  const value = search.get('tab') as ReviewerAccountTabKey | null;
  return value === 'profile' || value === 'security' || value === 'memory'
    ? value
    : 'profile';
}

function readRouteState(): RouteState {
  const pathname = window.location.pathname;
  const search = new URLSearchParams(window.location.search);
  const shared = {
    selectedStep: parseStep(search),
    queueStatus: parseStatus(search, pathname.startsWith('/audit/') ? 'all' : 'pending'),
    riskFilter: parseRisk(search),
    keyword: search.get('q') ?? '',
    highlightCaseId: search.get('highlight'),
    batchCount: parseBatchCount(search),
    agentOpen: search.get('assistant') === 'open',
    accountTab: parseAccountTab(search),
    accountNext: search.get('next'),
  };

  if (pathname === '/login') {
    return {
      ...shared,
      route: 'login',
      caseId: null,
    };
  }

  if (pathname === '/ingest') {
    return {
      ...shared,
      route: 'ingest',
      caseId: null,
    };
  }

  if (pathname === '/assistant') {
    return {
      ...shared,
      route: 'assistant',
      caseId: search.get('case_id'),
    };
  }

  if (pathname === '/font-preview') {
    return {
      ...shared,
      route: 'font_preview',
      caseId: null,
    };
  }

  if (pathname === '/account') {
    return {
      ...shared,
      route: 'account',
      caseId: null,
    };
  }

  if (pathname.startsWith('/appeal/')) {
    const caseId = decodeURIComponent(
      pathname.replace('/appeal/', '').split('/')[0] ?? '',
    );
    return {
      ...shared,
      route: 'appeal_portal',
      caseId: caseId || null,
    };
  }

  if (pathname.startsWith('/audit/')) {
    const caseId = decodeURIComponent(
      pathname.replace('/audit/', '').split('/')[0] ?? '',
    );
    return {
      ...shared,
      route: 'audit_detail',
      caseId: caseId || null,
    };
  }

  return {
    ...shared,
    route: 'audit',
    caseId: null,
  };
}

function currentPathWithSearch(): string {
  return `${window.location.pathname}${window.location.search}`;
}

function isProtectedRoute(route: AppRoute): boolean {
  return route !== 'login' && route !== 'appeal_portal' && route !== 'font_preview';
}

function buildLoginPath(nextPath: string): string {
  const params = new URLSearchParams();
  if (nextPath && nextPath !== '/login') params.set('next', nextPath);
  const query = params.toString();
  return query ? `/login?${query}` : '/login';
}

function buildAccountPath({
  tab,
  next,
}: {
  tab: ReviewerAccountTabKey;
  next?: string | null;
}) {
  const params = new URLSearchParams();
  params.set('tab', tab);
  if (next) params.set('next', next);
  return `/account?${params.toString()}`;
}

function readLoginNext(): string | null {
  const next = new URLSearchParams(window.location.search).get('next');
  if (!next || !next.startsWith('/') || next.startsWith('//')) return null;
  if (next.startsWith('/login')) return null;
  return next;
}

function addQueueContext(
  params: URLSearchParams,
  {
    status,
    risk,
    keyword,
  }: {
    status: QueueStatusFilter;
    risk: QueueRiskFilter;
    keyword: string;
  },
) {
  params.set('status', status);
  if (risk !== 'all') params.set('risk', risk);
  if (keyword.trim()) params.set('q', keyword.trim());
}

function buildQueuePath({
  status,
  risk,
  keyword,
  highlight,
  batchCount,
}: {
  status: QueueStatusFilter;
  risk: QueueRiskFilter;
  keyword: string;
  highlight?: string | null;
  batchCount?: number | null;
}) {
  const params = new URLSearchParams();
  addQueueContext(params, { status, risk, keyword });
  if (highlight) params.set('highlight', highlight);
  if (batchCount && batchCount > 0) params.set('batch', String(batchCount));
  return `/audit?${params.toString()}`;
}

function buildDetailPath({
  caseId,
  step,
  status,
  risk,
  keyword,
  agentOpen,
}: {
  caseId: string;
  step: WorkflowStepKey;
  status: QueueStatusFilter;
  risk: QueueRiskFilter;
  keyword: string;
  agentOpen: boolean;
}) {
  const params = new URLSearchParams();
  params.set('step', step);
  addQueueContext(params, { status, risk, keyword });
  if (agentOpen) params.set('assistant', 'open');
  return `/audit/${encodeURIComponent(caseId)}?${params.toString()}`;
}

function buildAssistantPath({
  caseId,
  status,
  risk,
  keyword,
}: {
  caseId?: string | null;
  status: QueueStatusFilter;
  risk: QueueRiskFilter;
  keyword: string;
}) {
  const params = new URLSearchParams();
  addQueueContext(params, { status, risk, keyword });
  if (caseId) params.set('case_id', caseId);
  return `/assistant?${params.toString()}`;
}

function buildAppealPath({
  caseId,
  status,
  risk,
  keyword,
  agentOpen,
}: {
  caseId: string;
  status: QueueStatusFilter;
  risk: QueueRiskFilter;
  keyword: string;
  agentOpen: boolean;
}) {
  const params = new URLSearchParams();
  params.set('step', 'appeal_handling');
  addQueueContext(params, { status, risk, keyword });
  if (agentOpen) params.set('assistant', 'open');
  return `/appeal/${encodeURIComponent(caseId)}?${params.toString()}`;
}

function filterAndSortCases(
  cases: CaseSummary[],
  status: QueueStatusFilter,
  risk: QueueRiskFilter,
  keyword: string,
) {
  const normalizedKeyword = keyword.trim().toLowerCase();
  return [...cases]
    .filter(
      (item) =>
        status === 'all' || (item.review_status ?? 'pending') === status,
    )
    .filter((item) => risk === 'all' || item.risk_level === risk)
    .filter((item) => {
      if (!normalizedKeyword) return true;
      return (
        item.case_id.toLowerCase().includes(normalizedKeyword) ||
        item.case_title.toLowerCase().includes(normalizedKeyword)
      );
    })
    .sort(
      (a, b) =>
        b.risk_score - a.risk_score || a.case_id.localeCompare(b.case_id),
    );
}

function readStoredSidebarState(): boolean {
  try {
    return window.localStorage.getItem('mediguard.sidebar.collapsed') === 'true';
  } catch {
    return false;
  }
}

export default function App() {
  const screens = Grid.useBreakpoint();
  const desktopSidebar = screens.lg ?? window.innerWidth >= 992;
  const [routeState, setRouteState] = useState<RouteState>(readRouteState);
  const [typeface, setTypeface] = useState<TypefaceKey>(DEFAULT_TYPEFACE);
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [caseData, setCaseData] = useState<CaseFullResponse | null>(null);
  const [workflow, setWorkflow] = useState<WorkflowResponse | null>(null);
  const [currentUser, setCurrentUser] = useState<AuthenticatedUser | null>(null);
  const [showcaseMode, setShowcaseMode] = useState(false);
  const [authChecked, setAuthChecked] = useState(false);
  const [loadingCases, setLoadingCases] = useState(false);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(
    readStoredSidebarState,
  );
  const [mobileNavigationOpen, setMobileNavigationOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const syncFromLocation = useCallback(() => {
    setRouteState(readRouteState());
  }, []);

  const navigate = useCallback((path: string, replace = false) => {
    if (replace) {
      window.history.replaceState(null, '', path);
    } else {
      window.history.pushState(null, '', path);
    }
    setRouteState(readRouteState());
  }, []);

  const redirectToLogin = useCallback(() => {
    setCurrentUser(null);
    navigate(buildLoginPath(currentPathWithSearch()), true);
  }, [navigate]);

  const loadCases = useCallback(async () => {
    setLoadingCases(true);
    setError(null);
    try {
      const data = await fetchCases();
      setCases(data);
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        redirectToLogin();
        return;
      }
      setError(err instanceof Error ? err.message : '加载案件队列失败');
    } finally {
      setLoadingCases(false);
    }
  }, [redirectToLogin]);

  const loadAuditCase = useCallback(async (caseId: string) => {
    setLoadingDetail(true);
    setError(null);
    setCaseData(null);
    setWorkflow(null);
    try {
      const [detail, workflowData] = await Promise.all([
        fetchCaseDetail(caseId),
        fetchCaseWorkflow(caseId),
      ]);
      setCaseData(detail);
      setWorkflow(workflowData);
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        redirectToLogin();
        return;
      }
      setError(err instanceof Error ? err.message : '加载案件详情失败');
    } finally {
      setLoadingDetail(false);
    }
  }, [redirectToLogin]);

  const loadAssistantCase = useCallback(async (caseId: string) => {
    setLoadingDetail(true);
    setError(null);
    setCaseData(null);
    setWorkflow(null);
    try {
      setCaseData(await fetchCaseDetail(caseId));
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        redirectToLogin();
        return;
      }
      setError(
        err instanceof Error ? err.message : '加载助手案件上下文失败',
      );
    } finally {
      setLoadingDetail(false);
    }
  }, [redirectToLogin]);

  useEffect(() => {
    let active = true;

    async function verifySession() {
      const [userResult, healthResult] = await Promise.allSettled([
        fetchCurrentUser(),
        fetchHealth(),
      ]);
      if (!active) return;
      setCurrentUser(userResult.status === 'fulfilled' ? userResult.value : null);
      setShowcaseMode(
        healthResult.status === 'fulfilled' && healthResult.value.showcase.enabled,
      );
      setAuthChecked(true);
    }

    void verifySession();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!authChecked) return;
    if (isProtectedRoute(routeState.route) && !currentUser) {
      navigate(buildLoginPath(currentPathWithSearch()), true);
      return;
    }
    if (routeState.route === 'login' && currentUser) {
      navigate(
        readLoginNext() ??
          buildQueuePath({ status: 'pending', risk: 'all', keyword: '' }),
        true,
      );
    }
  }, [authChecked, currentUser, navigate, routeState.route]);

  useEffect(() => {
    const pathname = window.location.pathname;
    if (pathname === '/' || pathname === '/workbench') {
      navigate(
        buildQueuePath({ status: 'pending', risk: 'all', keyword: '' }),
        true,
      );
    }
  }, [navigate]);

  useEffect(() => {
    const option = getTypefaceOption(typeface);
    document.documentElement.style.setProperty(
      '--mediguard-font-family',
      option.stack,
    );
    window.localStorage.setItem(TYPEFACE_STORAGE_KEY, typeface);
  }, [typeface]);

  useEffect(() => {
    function handlePopState() {
      syncFromLocation();
    }
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, [syncFromLocation]);

  useEffect(() => {
    if (
      !authChecked ||
      !currentUser ||
      !isProtectedRoute(routeState.route) ||
      routeState.route === 'account'
    ) {
      return;
    }
    void loadCases();
  }, [authChecked, currentUser, loadCases, routeState.route]);

  useEffect(() => {
    if (!authChecked) return;
    if (!currentUser && isProtectedRoute(routeState.route)) {
      setCaseData(null);
      setWorkflow(null);
      setLoadingDetail(false);
      return;
    }
    if (routeState.route === 'audit_detail' && routeState.caseId) {
      void loadAuditCase(routeState.caseId);
      return;
    }
    if (routeState.route === 'assistant' && routeState.caseId) {
      void loadAssistantCase(routeState.caseId);
      return;
    }
    setCaseData(null);
    setWorkflow(null);
    setLoadingDetail(false);
  }, [
    routeState.route,
    routeState.caseId,
    authChecked,
    currentUser,
    loadAuditCase,
    loadAssistantCase,
  ]);

  useEffect(() => {
    setMobileNavigationOpen(false);
  }, [routeState.route]);

  const filteredCases = useMemo(
    () =>
      filterAndSortCases(
        cases,
        routeState.queueStatus,
        routeState.riskFilter,
        routeState.keyword,
      ),
    [
      cases,
      routeState.queueStatus,
      routeState.riskFilter,
      routeState.keyword,
    ],
  );

  const fallbackDetailSequence = useMemo(
    () => filterAndSortCases(cases, 'all', 'all', ''),
    [cases],
  );

  const detailSequence = filteredCases.some(
    (item) => item.case_id === routeState.caseId,
  )
    ? filteredCases
    : fallbackDetailSequence;
  const currentIndex = detailSequence.findIndex(
    (item) => item.case_id === routeState.caseId,
  );
  const previousCase =
    currentIndex > 0 ? detailSequence[currentIndex - 1] : null;
  const nextCase =
    currentIndex >= 0 && currentIndex < detailSequence.length - 1
      ? detailSequence[currentIndex + 1]
      : null;

  const moduleRoute =
    routeState.route !== 'login' &&
    routeState.route !== 'account' &&
    routeState.route !== 'audit_detail' &&
    routeState.route !== 'appeal_portal' &&
    routeState.route !== 'font_preview';
  const activeModule: BusinessModule =
    routeState.route === 'ingest'
      ? 'ingest'
      : routeState.route === 'assistant'
        ? 'assistant'
        : 'audit';

  function navigateQueue(next: Partial<RouteState>, highlight?: string | null) {
    navigate(
      buildQueuePath({
        status: next.queueStatus ?? routeState.queueStatus,
        risk: next.riskFilter ?? routeState.riskFilter,
        keyword: next.keyword ?? routeState.keyword,
        highlight,
      }),
    );
  }

  function navigateModule(module: BusinessModule) {
    if (module === 'ingest') {
      navigate('/ingest');
      return;
    }
    if (module === 'assistant') {
      navigate(
        buildAssistantPath({
          status: routeState.queueStatus,
          risk: routeState.riskFilter,
          keyword: routeState.keyword,
        }),
      );
      return;
    }
    navigate(
      buildQueuePath({
        status: routeState.queueStatus,
        risk: routeState.riskFilter,
        keyword: routeState.keyword,
      }),
    );
  }

  function openCase(caseId: string) {
    navigate(
      buildDetailPath({
        caseId,
        step: DEFAULT_WORKFLOW_STEP,
        status: routeState.queueStatus,
        risk: routeState.riskFilter,
        keyword: routeState.keyword,
        agentOpen: false,
      }),
    );
  }

  function openAdjacent(caseId: string) {
    navigate(
      buildDetailPath({
        caseId,
        step: routeState.selectedStep,
        status: routeState.queueStatus,
        risk: routeState.riskFilter,
        keyword: routeState.keyword,
        agentOpen: routeState.agentOpen,
      }),
    );
  }

  function updateDetailRoute({
    step = routeState.selectedStep,
    agentOpen = routeState.agentOpen,
  }: {
    step?: WorkflowStepKey;
    agentOpen?: boolean;
  }) {
    if (!routeState.caseId) return;
    navigate(
      buildDetailPath({
        caseId: routeState.caseId,
        step,
        status: routeState.queueStatus,
        risk: routeState.riskFilter,
        keyword: routeState.keyword,
        agentOpen,
      }),
      true,
    );
  }

  function handleReviewSubmitted() {
    if (!routeState.caseId) return;
    navigate(
      buildDetailPath({
        caseId: routeState.caseId,
        step: 'case_result',
        status: 'reviewed',
        risk: routeState.riskFilter,
        keyword: routeState.keyword,
        agentOpen: routeState.agentOpen,
      }),
      true,
    );
    void loadCases();
    void loadAuditCase(routeState.caseId);
  }

  function handleNoteSaved(note: AuditNote) {
    setCaseData((current) => {
      if (!current || current.case.case_id !== routeState.caseId) return current;
      return {
        ...current,
        notes: [
          ...(current.notes ?? []).filter((item) => item.note_id !== note.note_id),
          note,
        ],
      };
    });
  }

  function handleNoteDeleted(noteId: string) {
    setCaseData((current) => {
      if (!current || current.case.case_id !== routeState.caseId) return current;
      return {
        ...current,
        notes: (current.notes ?? []).filter((item) => item.note_id !== noteId),
      };
    });
  }

  function handleCasesGenerated(generatedCases: CaseFullResponse[]) {
    const firstCaseId = generatedCases[0]?.case.case_id ?? null;
    setCaseData(null);
    setWorkflow(null);
    setError(null);
    navigate(
      buildQueuePath({
        status: 'pending',
        risk: 'all',
        keyword: '',
        highlight: firstCaseId,
        batchCount: generatedCases.length,
      }),
    );
    void loadCases();
  }

  function handleSidebarCollapse(collapsed: boolean) {
    setSidebarCollapsed(collapsed);
    try {
      window.localStorage.setItem(
        'mediguard.sidebar.collapsed',
        String(collapsed),
      );
    } catch {
      // The layout still works when browser storage is unavailable.
    }
  }

  function handleAssistantQuickEntry() {
    if (routeState.route !== 'audit_detail') return;
    updateDetailRoute({ agentOpen: !routeState.agentOpen });
  }

  function handleLoginSuccess(user: AuthenticatedUser) {
    setCurrentUser(user);
    setAuthChecked(true);
    navigate(
      readLoginNext() ??
        buildQueuePath({ status: 'pending', risk: 'all', keyword: '' }),
      true,
    );
  }

  async function handleLogout() {
    try {
      await logout();
    } finally {
      setCurrentUser(null);
      navigate('/login', true);
    }
  }

  function openAccountCenter() {
    const tab =
      routeState.route === 'account' ? routeState.accountTab : 'profile';
    const next =
      routeState.route === 'account'
        ? routeState.accountNext
        : currentPathWithSearch();
    navigate(
      buildAccountPath({
        tab,
        next,
      }),
    );
  }

  function handleAccountBack() {
    const next = routeState.accountNext;
    if (next && next.startsWith('/')) {
      navigate(next, true);
      return;
    }
    navigate(
      buildQueuePath({
        status: routeState.queueStatus,
        risk: routeState.riskFilter,
        keyword: routeState.keyword,
      }),
      true,
    );
  }

  function openAppealPortal() {
    if (!routeState.caseId) return;
    navigate(
      buildAppealPath({
        caseId: routeState.caseId,
        status: routeState.queueStatus,
        risk: routeState.riskFilter,
        keyword: routeState.keyword,
        agentOpen: routeState.agentOpen,
      }),
    );
  }

  const errorAlert = error ? (
    <Alert
      className="global-error"
      type="error"
      showIcon
      title={`错误：${error}`}
      action={
        <Button size="small" onClick={() => void loadCases()}>
          重试
        </Button>
      }
    />
  ) : null;
  const protectedContentBlocked =
    isProtectedRoute(routeState.route) && (!authChecked || !currentUser);

  const moduleContent = (
    <div className="audit-shell module-audit-shell">
      {errorAlert}

      {routeState.route === 'ingest' && (
        <IngestPage
          readOnly={showcaseMode}
          onCasesGenerated={handleCasesGenerated}
        />
      )}

      {routeState.route === 'audit' && (
        <AuditQueuePage
          cases={cases}
          filteredCases={filteredCases}
          statusFilter={routeState.queueStatus}
          riskFilter={routeState.riskFilter}
          keyword={routeState.keyword}
          highlightCaseId={routeState.highlightCaseId}
          batchCount={routeState.batchCount}
          loading={loadingCases}
          onStatusChange={(queueStatus) => navigateQueue({ queueStatus })}
          onRiskChange={(riskFilter) => navigateQueue({ riskFilter })}
          onKeywordChange={(keyword) => navigateQueue({ keyword })}
          onOpenCase={openCase}
        />
      )}

      {routeState.route === 'assistant' && (
        <AssistantWorkspacePage
          cases={fallbackDetailSequence}
          selectedCaseId={routeState.caseId}
          selectedCase={caseData}
          loading={loadingDetail}
          onSelectCase={(caseId) =>
            navigate(
              buildAssistantPath({
                caseId,
                status: routeState.queueStatus,
                risk: routeState.riskFilter,
                keyword: routeState.keyword,
              }),
            )
          }
          onOpenCase={openCase}
        />
      )}
    </div>
  );

  return (
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: '#1b65b9',
          colorInfo: '#1b65b9',
          colorSuccess: '#27864f',
          colorWarning: '#b56c0b',
          colorError: '#c63c3c',
          colorText: '#303133',
          colorTextSecondary: '#606266',
          colorBorder: '#dcdfe6',
          colorBgLayout: '#f5f7fa',
          fontFamily: getTypefaceOption(typeface).stack,
          fontSize: 16,
          fontSizeSM: 14,
          controlHeight: 40,
          controlHeightSM: 34,
          borderRadius: 4,
          wireframe: false,
        },
        components: {
          Button: {
            fontWeight: 600,
          },
          Card: {
            headerFontSize: 18,
          },
          Table: {
            headerBg: '#f3f5f7',
            headerColor: '#303133',
            rowHoverBg: '#eef5fb',
          },
        },
      }}
    >
      <Layout className={routeState.route === 'login' ? 'app-shell login-app-shell' : 'app-shell'}>
        {routeState.route !== 'login' && (
        <Layout.Header className="app-topbar">
          <div className="app-topbar-inner">
            {moduleRoute && !desktopSidebar && (
              <Button
                className="mobile-navigation-button"
                type="text"
                icon={<MenuOutlined />}
                aria-label="打开业务导航"
                onClick={() => setMobileNavigationOpen(true)}
              />
            )}
            <button
              className="app-brand"
              type="button"
              aria-label="进入案件审核队列"
              onClick={() =>
                navigate(
                  buildQueuePath({
                    status: 'pending',
                    risk: 'all',
                    keyword: '',
                  }),
                )
              }
            >
              <span className="app-brand-mark">
                <MediGuardMark />
              </span>
              <span className="app-brand-copy">
                <strong>医保智能稽核服务平台</strong>
                <span className="app-brand-english" aria-label="MediGuard">
                  <span aria-hidden="true">M</span>
                  <span aria-hidden="true">e</span>
                  <span aria-hidden="true">d</span>
                  <span aria-hidden="true">i</span>
                  <span aria-hidden="true">G</span>
                  <span aria-hidden="true">u</span>
                  <span aria-hidden="true">a</span>
                  <span aria-hidden="true">r</span>
                  <span aria-hidden="true">d</span>
                </span>
              </span>
            </button>
            {showcaseMode && (
              <Tag className="showcase-mode-tag">只读演示</Tag>
            )}
            <div
              className="reviewer-account"
              aria-label={
                currentUser
                  ? `当前审核人员：${currentUser.display_name}`
                  : routeState.route === 'appeal_portal'
                    ? '当前入口：申报材料'
                    : '当前状态：未登录'
              }
            >
              {currentUser ? (
                <div className="reviewer-account-actions">
                  <Button
                    className="reviewer-account-button"
                    type="text"
                    icon={<Avatar size={28} icon={<UserOutlined />} />}
                    aria-label={`审核人员中心：${currentUser.display_name}`}
                    onClick={openAccountCenter}
                  >
                    {currentUser.display_name}
                  </Button>
                  <Button
                    className="topbar-logout reviewer-account-logout"
                    type="text"
                    icon={<LogoutOutlined />}
                    aria-label="退出登录"
                    onClick={() => void handleLogout()}
                  >
                    退出
                  </Button>
                </div>
              ) : (
                <span className="reviewer-account-static">
                  <Avatar
                    size={32}
                    icon={
                      routeState.route === 'appeal_portal' ? (
                        <FileAddOutlined />
                      ) : (
                        <UserOutlined />
                      )
                    }
                  />
                  <span>
                    {routeState.route === 'appeal_portal'
                      ? '申报材料入口'
                      : '未登录'}
                  </span>
                </span>
              )}
            </div>
          </div>
        </Layout.Header>
        )}

        {routeState.route === 'login' ? (
          <LoginPage
            showcaseMode={showcaseMode}
            onLogin={handleLoginSuccess}
          />
        ) : protectedContentBlocked ? (
          <Layout.Content className="audit-shell">
            <Card>
              <Spin description="正在确认审核人员登录状态..." />
            </Card>
          </Layout.Content>
        ) : routeState.route === 'font_preview' ? (
          <Layout.Content className="audit-shell font-preview-shell">
            <FontPreviewPage
              currentTypeface={typeface}
              onSelectTypeface={setTypeface}
            />
          </Layout.Content>
        ) : routeState.route === 'account' && currentUser ? (
          <Layout.Content className="account-shell">
            <ReviewerAccountPage
              currentUser={currentUser}
              activeTab={routeState.accountTab}
              readOnly={showcaseMode}
              onBack={handleAccountBack}
              onTabChange={(tab) =>
                navigate(
                  buildAccountPath({
                    tab,
                    next: routeState.accountNext ?? currentPathWithSearch(),
                  }),
                  true,
                )
              }
            />
          </Layout.Content>
        ) : routeState.route === 'appeal_portal' && routeState.caseId ? (
          <Layout.Content className="appeal-audit-shell">
            <AppealPortalPage
              caseId={routeState.caseId}
              onBack={() =>
                navigate(
                  buildDetailPath({
                    caseId: routeState.caseId!,
                    step: 'appeal_handling',
                    status: routeState.queueStatus,
                    risk: routeState.riskFilter,
                    keyword: routeState.keyword,
                    agentOpen: routeState.agentOpen,
                  }),
                )
              }
            />
          </Layout.Content>
        ) : moduleRoute ? (
          <Layout className="module-shell">
            <BusinessSidebar
              activeModule={activeModule}
              collapsed={sidebarCollapsed}
              desktop={desktopSidebar}
              mobileOpen={mobileNavigationOpen}
              onCollapse={handleSidebarCollapse}
              onMobileClose={() => setMobileNavigationOpen(false)}
              onNavigate={navigateModule}
            />
            <Layout.Content className="module-content">
              {moduleContent}
            </Layout.Content>
          </Layout>
        ) : (
          <Layout.Content className="audit-shell detail-audit-shell">
            {errorAlert}
            <main className="case-detail-page">
              <Space orientation="vertical" size={16} style={{ width: '100%' }}>
                <Card className="case-detail-toolbar">
                  <div className="detail-toolbar-content">
                    <Space wrap>
                      <Button
                        icon={<ArrowLeftOutlined />}
                        onClick={() =>
                          navigate(
                            buildQueuePath({
                              status: routeState.queueStatus,
                              risk: routeState.riskFilter,
                              keyword: routeState.keyword,
                              highlight: routeState.caseId,
                            }),
                          )
                        }
                      >
                        返回审核队列
                      </Button>
                      <Button
                        icon={<LeftOutlined />}
                        disabled={!previousCase}
                        onClick={() =>
                          previousCase && openAdjacent(previousCase.case_id)
                        }
                      >
                        上一案
                      </Button>
                      <Button
                        icon={<RightOutlined />}
                        disabled={!nextCase}
                        onClick={() =>
                          nextCase && openAdjacent(nextCase.case_id)
                        }
                      >
                        下一案
                      </Button>
                    </Space>
                    {caseData && (
                      <Typography.Text className="detail-queue-position">
                        当前队列第 {currentIndex >= 0 ? currentIndex + 1 : '-'} /{' '}
                        {detailSequence.length} 案
                      </Typography.Text>
                    )}
                  </div>
                </Card>

                {loadingDetail && !caseData && (
                  <Card>
                    <Spin description="加载案件详情..." />
                  </Card>
                )}
                {!loadingDetail && !caseData && !error && (
                  <Card>
                    <Empty description="当前案件不存在或尚未加载。" />
                  </Card>
                )}
                {caseData && workflow && currentUser && (
                  <CaseDetail
                    data={caseData}
                    workflow={workflow}
                    currentUser={currentUser}
                    showcaseMode={showcaseMode}
                    selectedStep={routeState.selectedStep}
                    agentOpen={routeState.agentOpen}
                    onAgentOpenChange={(agentOpen) =>
                      updateDetailRoute({ agentOpen })
                    }
                    onStepChange={(step) => updateDetailRoute({ step })}
                    onOpenAppealPortal={openAppealPortal}
                    onReviewSubmitted={handleReviewSubmitted}
                    onNoteSaved={handleNoteSaved}
                    onNoteDeleted={handleNoteDeleted}
                  />
                )}
              </Space>
            </main>
          </Layout.Content>
        )}

        <GlobalAuditAssistant
          hidden={
            routeState.route !== 'audit_detail' || routeState.agentOpen
          }
          active={routeState.route === 'audit_detail' && routeState.agentOpen}
          onActivate={handleAssistantQuickEntry}
        />
      </Layout>
    </ConfigProvider>
  );
}
