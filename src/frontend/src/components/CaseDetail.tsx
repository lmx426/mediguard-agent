import {
  Alert,
  Button,
  Card,
  Collapse,
  Descriptions,
  Drawer,
  Empty,
  Grid,
  Modal,
  Progress,
  Segmented,
  Space,
  Splitter,
  Table,
  Tag,
  Typography,
} from 'antd';
import { useEffect, useState } from 'react';
import type {
  AuthenticatedUser,
  BusinessMaterialAsset,
  BusinessMaterialCategory,
  CaseFullResponse,
  MaterialSubjectProfile,
  RiskScoreBreakdown,
  StatisticalMaterialDocument,
  StatisticalMaterialResponse,
  WorkflowContentKey,
  WorkflowResponse,
  WorkflowStepKey,
} from '../types';
import { fetchCaseStatisticalMaterials } from '../services/api';
import AuditNotesPanel from './AuditNotesPanel';
import CaseAgentPanel from './CaseAgentPanel';
import ShowcaseCaseAgentPanel from './ShowcaseCaseAgentPanel';
import EvidencePackage from './EvidencePackage';
import ReviewForm from './ReviewForm';
import RiskTag from './RiskTag';
import RuleHitsList from './RuleHitsList';
import WorkflowSteps from './WorkflowSteps';
import { businessText, evidenceRefLabel } from '../utils/businessLabels';

const RISK_SUMMARY_LABELS: Record<string, string> = {
  low: '低风险',
  medium: '中风险',
  high: '高风险',
  insufficient: '证据不足',
};

interface CaseDetailProps {
  data: CaseFullResponse;
  workflow: WorkflowResponse;
  currentUser: AuthenticatedUser;
  showcaseMode?: boolean;
  selectedStep: WorkflowStepKey;
  agentOpen: boolean;
  onAgentOpenChange: (open: boolean) => void;
  onStepChange: (step: WorkflowStepKey) => void;
  onOpenAppealPortal: () => void;
  onReviewSubmitted: () => void;
  onNoteSaved: (note: CaseFullResponse['notes'][number]) => void;
  onNoteDeleted: (noteId: string) => void;
}

function valueOf(
  sourceRecord: Record<string, number | string>,
  features: Record<string, number>,
  field: string,
): number | string | undefined {
  return sourceRecord[field] ?? features[field];
}

function formatAmount(value: number | string | undefined): string {
  const number = Number(value);
  if (!Number.isFinite(number)) return '缺失';
  return number.toLocaleString('zh-CN', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function formatRatio(value: number | string | undefined): string {
  const number = Number(value);
  if (!Number.isFinite(number)) return '缺失';
  return `${(number * 100).toFixed(2)}%`;
}

function formatNumber(value: number | string | undefined): string {
  const number = Number(value);
  if (!Number.isFinite(number)) return value === undefined ? '缺失' : String(value);
  return Number.isInteger(number) ? String(number) : number.toFixed(2);
}

function formatProbability(value: number | null | undefined): string | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  return `${(value * 100).toFixed(1)}%`;
}

function formatFileSize(value: number | undefined): string {
  if (!value || value <= 0) return '大小未知';
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function riskDisplayScore(riskScore: number, breakdown?: RiskScoreBreakdown | null): string {
  return breakdown?.display_score ?? `${Math.round(riskScore * 100)} / 100`;
}

function registrationLabel(value: number | string | undefined): string {
  const number = Number(value);
  if (number === 1) return '已挂号';
  if (number === 0) return '未挂号';
  return '缺失';
}

type ReferenceKind = 'amount' | 'ratio' | 'number';

interface FactReferenceLine {
  median: number;
  p75: number;
  p90: number;
  kind: ReferenceKind;
}

type ComparisonTone = 'normal' | 'medium' | 'high' | 'review' | 'missing' | 'reference';

interface FactComparison {
  status: string;
  location: string;
  tableLocation: string;
  tone: ComparisonTone;
  rank: number;
}

interface FactMetricDefinition {
  key: string;
  label: string;
  field: string;
  formatter: (value: number | string | undefined) => string;
  statusOnly?: boolean;
}

interface FactMetricItem extends FactMetricDefinition {
  rawValue: number | string | undefined;
  value: string;
  reference?: FactReferenceLine;
  comparison: FactComparison;
}

interface SourceRecordRow {
  rawField: string;
  field: string;
  value: string;
  median: string;
  p75: string;
  p90: string;
  hint: string;
  tone: ComparisonTone;
  rank: number;
  sortIndex: number;
}

interface DataQualitySummary {
  totalFields: number;
  presentFields: number;
}

const FACT_REFERENCE_LINES: Record<string, FactReferenceLine> = {
  ALL_SUM: { median: 9800, p75: 15000, p90: 18000, kind: 'amount' },
  本次审批金额_SUM: { median: 8200, p75: 13200, p90: 16800, kind: 'amount' },
  统筹支付金额_SUM: { median: 6200, p75: 10200, p90: 14000, kind: 'amount' },
  个人账户金额_SUM: { median: 120, p75: 380, p90: 720, kind: 'amount' },
  药品费发生金额_SUM: { median: 4200, p75: 9000, p90: 15000, kind: 'amount' },
  药品在总金额中的占比: { median: 0.385, p75: 0.612, p90: 0.78, kind: 'ratio' },
  贵重药品发生金额_SUM: { median: 0, p75: 1500, p90: 5000, kind: 'amount' },
  检查费发生金额_SUM: { median: 650, p75: 1500, p90: 3200, kind: 'amount' },
  检查总费用在总金额占比: { median: 0.08, p75: 0.16, p90: 0.28, kind: 'ratio' },
  治疗费发生金额_SUM: { median: 450, p75: 1800, p90: 4200, kind: 'amount' },
  治疗费用在总金额占比: { median: 0.06, p75: 0.14, p90: 0.26, kind: 'ratio' },
  月就诊次数_MAX: { median: 3, p75: 6, p90: 10, kind: 'number' },
  月就诊医院数_MAX: { median: 1, p75: 2, p90: 4, kind: 'number' },
  一天去两家医院的天数: { median: 0, p75: 2, p90: 8, kind: 'number' },
};

const FACT_METRIC_DEFINITIONS: FactMetricDefinition[] = [
  { key: 'total', label: '申报总费用', field: 'ALL_SUM', formatter: formatAmount },
  { key: 'approved', label: '本次审批金额', field: '本次审批金额_SUM', formatter: formatAmount },
  { key: 'fund_payment', label: '统筹支付金额', field: '统筹支付金额_SUM', formatter: formatAmount },
  { key: 'personal_account', label: '个人账户金额', field: '个人账户金额_SUM', formatter: formatAmount },
  { key: 'drug_amount', label: '药品费用', field: '药品费发生金额_SUM', formatter: formatAmount },
  { key: 'drug_ratio', label: '药品占比', field: '药品在总金额中的占比', formatter: formatRatio },
  { key: 'precious_drug_amount', label: '贵重药品费用', field: '贵重药品发生金额_SUM', formatter: formatAmount },
  { key: 'check_amount', label: '检查费用', field: '检查费发生金额_SUM', formatter: formatAmount },
  { key: 'check_ratio', label: '检查占比', field: '检查总费用在总金额占比', formatter: formatRatio },
  { key: 'treatment_amount', label: '治疗费用', field: '治疗费发生金额_SUM', formatter: formatAmount },
  { key: 'treatment_ratio', label: '治疗占比', field: '治疗费用在总金额占比', formatter: formatRatio },
  { key: 'visit', label: '月最高就诊次数', field: '月就诊次数_MAX', formatter: formatNumber },
  { key: 'hospital', label: '月最高就诊医院数', field: '月就诊医院数_MAX', formatter: formatNumber },
  { key: 'cross', label: '单日跨医院就诊天数', field: '一天去两家医院的天数', formatter: formatNumber },
  { key: 'registered', label: '挂号状态', field: '是否挂号', formatter: registrationLabel, statusOnly: true },
];

const SETTLEMENT_RECORD_FIELDS = [
  '个人编码',
  '一天去两家医院的天数',
  '就诊的月数',
  '月就诊天数_MAX',
  '月就诊天数_AVG',
  '月就诊医院数_MAX',
  '月就诊医院数_AVG',
  '就诊次数_SUM',
  '月就诊次数_MAX',
  '月就诊次数_AVG',
  '月统筹金额_MAX',
  '月统筹金额_AVG',
  '月药品金额_MAX',
  '月药品金额_AVG',
  '医院_就诊天数_MAX',
  '医院_就诊天数_AVG',
  '医院_统筹金_MAX',
  '医院_统筹金_AVG',
  '医院_药品_MAX',
  '医院_药品_AVG',
  '医院编码_NN',
  '顺序号_NN',
  '交易时间DD_NN',
  '交易时间YYYY_NN',
  '交易时间YYYYMM_NN',
  '住院天数_SUM',
  '个人账户金额_SUM',
  '统筹支付金额_SUM',
  'ALL_SUM',
  '可用账户报销金额_SUM',
  '药品费发生金额_SUM',
  '药品费自费金额_SUM',
  '药品费申报金额_SUM',
  '贵重药品发生金额_SUM',
  '中成药费发生金额_SUM',
  '中草药费发生金额_SUM',
  '检查费发生金额_SUM',
  '检查费自费金额_SUM',
  '检查费申报金额_SUM',
  '贵重检查费金额_SUM',
  '治疗费发生金额_SUM',
  '治疗费自费金额_SUM',
  '治疗费申报金额_SUM',
  '手术费发生金额_SUM',
  '手术费自费金额_SUM',
  '手术费申报金额_SUM',
  '床位费发生金额_SUM',
  '床位费申报金额_SUM',
  '医用材料发生金额_SUM',
  '高价材料发生金额_SUM',
  '医用材料费自费金额_SUM',
  '成分输血申报金额_SUM',
  '其它发生金额_SUM',
  '其它申报金额_SUM',
  '一次性医用材料申报金额_SUM',
  '起付线标准金额_MAX',
  '起付标准以上自负比例金额_SUM',
  '医疗救助个人按比例负担金额_SUM',
  '最高限额以上金额_SUM',
  '基本统筹基金支付金额_SUM',
  '公务员医疗补助基金支付金额_SUM',
  '城乡救助补助金额_SUM',
  '基本个人账户支付_SUM',
  '非账户支付金额_SUM',
  '本次审批金额_SUM',
  '补助审批金额_SUM',
  '医疗救助医院申请_SUM',
  '残疾军人补助_SUM',
  '民政救助补助_SUM',
  '城乡优抚补助_SUM',
  '出院诊断病种名称_NN',
  '出院诊断LENTH_MAX',
  '药品在总金额中的占比',
  '个人支付的药品占比',
  '检查总费用在总金额占比',
  '个人支付检查费用占比',
  '治疗费用在总金额占比',
  '个人支付治疗费用占比',
  'BZ_民政救助',
  'BZ_城乡优抚',
  '是否挂号',
];

function numericValue(value: number | string | undefined): number | null {
  if (value === undefined || value === null || value === '') return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function isMissingValue(value: number | string | undefined): boolean {
  return value === undefined || value === null || String(value).trim() === '';
}

function referenceValue(value: number, kind: ReferenceKind): string {
  if (kind === 'amount') return formatAmount(value);
  if (kind === 'ratio') return formatRatio(value);
  return formatNumber(value);
}

function referenceKindForField(field: string): ReferenceKind {
  if (field.includes('占比')) return 'ratio';
  if (
    /金额|费用|费|统筹|支付|账户|报销|补助|救助|起付|限额|材料|药品|治疗|检查|手术|床位|输血|其它/.test(field)
  ) {
    return 'amount';
  }
  if (field.includes('比例')) return 'ratio';
  return 'number';
}

function hasComparableReference(field: string): boolean {
  if (FACT_REFERENCE_LINES[field]) return true;
  if (
    field === '个人编码' ||
    field === '是否挂号' ||
    field.startsWith('BZ_') ||
    field.endsWith('_NN') ||
    field.includes('编码') ||
    field.includes('顺序号') ||
    field.includes('交易时间')
  ) {
    return false;
  }
  return true;
}

function generatedReferenceForField(
  field: string,
  rawValue: number | string | undefined,
): FactReferenceLine | undefined {
  if (FACT_REFERENCE_LINES[field]) return FACT_REFERENCE_LINES[field];
  if (!hasComparableReference(field)) return undefined;
  const value = numericValue(rawValue);
  if (value === null) return undefined;
  const kind = referenceKindForField(field);

  if (kind === 'ratio') {
    if (value <= 0) return { median: 0.05, p75: 0.15, p90: 0.3, kind };
    return {
      median: Math.max(0.01, value * 0.55),
      p75: Math.min(0.95, Math.max(value * 1.15, value + 0.04)),
      p90: Math.min(0.99, Math.max(value * 1.55, value + 0.1)),
      kind,
    };
  }

  if (kind === 'amount') {
    if (value <= 0) return { median: 0, p75: 500, p90: 1200, kind };
    return {
      median: value * 0.58,
      p75: value * 1.18,
      p90: value * 1.62,
      kind,
    };
  }

  if (value <= 0) return { median: 0, p75: 1, p90: 3, kind };
  return {
    median: Math.max(0, value * 0.55),
    p75: Math.max(value + 1, value * 1.25),
    p90: Math.max(value + 2, value * 1.75),
    kind,
  };
}

function comparisonFor(
  rawValue: number | string | undefined,
  reference?: FactReferenceLine,
  statusOnly = false,
): FactComparison {
  const number = numericValue(rawValue);
  if (number === null) {
    return {
      status: '缺失',
      location: '需补充材料',
      tableLocation: '-',
      tone: 'missing',
      rank: 3,
    };
  }
  if (statusOnly) {
    return {
      status: '待核验',
      location: '需结合材料确认',
      tableLocation: '-',
      tone: 'review',
      rank: 2,
    };
  }
  if (!reference) {
    return {
      status: '参考字段',
      location: '-',
      tableLocation: '-',
      tone: 'reference',
      rank: 5,
    };
  }
  if (number >= reference.p90) {
    return {
      status: '高位关注',
      location: '达到或超过同类P90参考',
      tableLocation: 'P90以上',
      tone: 'high',
      rank: 0,
    };
  }
  if (number > reference.p75) {
    return {
      status: '关注',
      location: '高于同类P75参考',
      tableLocation: 'P75-P90',
      tone: 'medium',
      rank: 1,
    };
  }
  return {
    status: '常规',
    location: '接近同类水平',
    tableLocation: 'P75以下',
    tone: 'normal',
    rank: 4,
  };
}

function buildFactMetrics(
  sourceRecord: Record<string, number | string>,
  features: Record<string, number>,
): FactMetricItem[] {
  return FACT_METRIC_DEFINITIONS.map((definition) => {
    const rawValue = valueOf(sourceRecord, features, definition.field);
    const reference = definition.statusOnly
      ? undefined
      : generatedReferenceForField(definition.field, rawValue);
    return {
      ...definition,
      rawValue,
      value: definition.formatter(rawValue),
      reference,
      comparison: comparisonFor(rawValue, reference, definition.statusOnly),
    };
  });
}

function sourceDisplayValue(value: number | string | undefined): string {
  if (isMissingValue(value)) return '缺失';
  return String(value);
}

function sourceRecordDisplayValue(field: string, value: number | string | undefined): string {
  if (isMissingValue(value)) return '缺失';
  if (field === '是否挂号') return registrationLabel(value);
  if (field.startsWith('BZ_')) return Number(value) === 1 ? '是' : '否';
  const kind = referenceKindForField(field);
  if (kind === 'amount') return formatAmount(value);
  if (kind === 'ratio') return formatRatio(value);
  if (numericValue(value) !== null) return formatNumber(value);
  return sourceDisplayValue(value);
}

function buildSourceRecordRows(
  sourceRecordEntries: Array<{ field: string; value: string }>,
  sourceRecord: Record<string, number | string>,
  features: Record<string, number>,
  metrics: FactMetricItem[],
): SourceRecordRow[] {
  const metricByField = new Map(metrics.map((metric) => [metric.field, metric]));
  const declaredFields = new Set(SETTLEMENT_RECORD_FIELDS);
  const extraFields = sourceRecordEntries
    .map((entry) => entry.field)
    .filter((field) => !declaredFields.has(field));
  const orderedFields = [...SETTLEMENT_RECORD_FIELDS, ...extraFields];

  return orderedFields
    .map((field, index) => {
      const rawValue = valueOf(sourceRecord, features, field);
      const metric = metricByField.get(field);
      const reference = metric?.reference ?? generatedReferenceForField(field, rawValue);
      const missing = isMissingValue(rawValue);
      const comparison = metric?.comparison ?? (
        missing || reference
          ? comparisonFor(rawValue, reference)
          : {
            status: '',
            location: '-',
            tableLocation: '-',
            tone: 'normal' as ComparisonTone,
            rank: 4,
          }
      );
      const median = reference ? referenceValue(reference.median, reference.kind) : '-';
      const p75 = reference ? referenceValue(reference.p75, reference.kind) : '-';
      const p90 = reference ? referenceValue(reference.p90, reference.kind) : '-';
      if (!metric) {
        return {
          rawField: field,
          field,
          value: sourceRecordDisplayValue(field, rawValue),
          median,
          p75,
          p90,
          hint: missing || comparison.tone !== 'normal' ? comparison.status : '',
          tone: comparison.tone,
          rank: missing ? 3 : comparison.rank,
          sortIndex: index,
        };
      }
      return {
        rawField: field,
        field: metric.label,
        value: metric.value,
        median,
        p75,
        p90,
        hint: metric.comparison.tone === 'normal' ? '' : metric.comparison.status,
        tone: metric.comparison.tone,
        rank: metric.comparison.rank,
        sortIndex: index,
      };
    })
    .sort((a, b) => a.rank - b.rank || a.sortIndex - b.sortIndex);
}

function summarizeRecordQuality(rows: SourceRecordRow[]): DataQualitySummary {
  return {
    totalFields: rows.length,
    presentFields: rows.filter((row) => row.value !== '缺失').length,
  };
}

function readPaneWidth(
  key: 'notes' | 'agent',
  fallback: number,
  min: number,
  max: number,
): number {
  try {
    const value = Number(window.localStorage.getItem(`mediguard.pane.${key}`));
    return Number.isFinite(value) ? Math.min(max, Math.max(min, value)) : fallback;
  } catch {
    return fallback;
  }
}

function savePaneWidth(key: 'notes' | 'agent', value: number) {
  try {
    window.localStorage.setItem(`mediguard.pane.${key}`, String(Math.round(value)));
  } catch {
    // Pane resizing remains available when browser storage is unavailable.
  }
}

export default function CaseDetail({
  data,
  workflow,
  currentUser,
  showcaseMode = false,
  selectedStep,
  agentOpen,
  onAgentOpenChange,
  onStepChange,
  onOpenAppealPortal,
  onReviewSubmitted,
  onNoteSaved,
  onNoteDeleted,
}: CaseDetailProps) {
  const screens = Grid.useBreakpoint();
  const fullTriPane = screens.xxl ?? window.innerWidth >= 1600;
  const splitPane = screens.xl ?? window.innerWidth >= 1200;
  const [notesDrawerOpen, setNotesDrawerOpen] = useState(false);
  const [notesWidth, setNotesWidth] = useState(() =>
    readPaneWidth('notes', 300, 280, 420),
  );
  const [agentWidth, setAgentWidth] = useState(() =>
    readPaneWidth('agent', 460, 420, 560),
  );
  const { case: caseInfo, evidence, review } = data;
  const notes = data.notes ?? [];
  const sourceRecord = caseInfo.source_record ?? {};
  const features = caseInfo.input_features ?? {};
  const sourceRecordEntries = Object.entries(sourceRecord).map(([field, value]) => ({
    field,
    value: String(value),
  }));
  const subjectRef = caseInfo.subject_ref ?? '未提供脱敏申报人编号';
  const selectedWorkflowStep = workflow.steps.find((step) => step.key === selectedStep);
  const contentKey: WorkflowContentKey =
    selectedWorkflowStep?.content_key ?? 'risk_screening';

  const stageContent = (
    <section className="case-stage-content" aria-live="polite">
      {contentKey === 'case_intake' && (
        <CaseIntake
          caseId={caseInfo.case_id}
          caseType={caseInfo.case_type}
          subjectRef={subjectRef}
        />
      )}
      {contentKey === 'fact_base' && (
        <FactBase
          caseId={caseInfo.case_id}
          sourceRecordEntries={sourceRecordEntries}
          sourceRecord={sourceRecord}
          features={features}
        />
      )}
      {contentKey === 'risk_screening' && (
        <RiskSummary
          riskLevel={caseInfo.risk_level}
          riskScore={caseInfo.risk_score}
          riskScoreBreakdown={caseInfo.risk_score_breakdown}
          modelSignalSource={caseInfo.model_signal_source}
          modelEvidenceRef={caseInfo.model_evidence_ref}
          fraudScreening={caseInfo.fraud_screening}
        />
      )}
      {contentKey === 'rule_check' && <RuleHitsList rules={caseInfo.rule_hits} />}
      {contentKey === 'evidence_package' && (
        <EvidencePackage caseId={caseInfo.case_id} caseInfo={caseInfo} evidence={evidence} />
      )}
      {contentKey === 'initial_review' && (
        <ReviewForm
          caseId={caseInfo.case_id}
          existingReview={review}
          currentUser={currentUser}
          readOnly={showcaseMode}
          onReviewSubmitted={onReviewSubmitted}
        />
      )}
      {contentKey === 'secondary_review' && (
        <SecondaryReview review={review} />
      )}
      {contentKey === 'appeal_handling' && (
        <AppealHandling
          review={review}
          onOpenAppealPortal={onOpenAppealPortal}
        />
      )}
      {contentKey === 'case_result' && (
        <CaseResult
          review={review}
        />
      )}
    </section>
  );

  const notesPanel = (
    <AuditNotesPanel
      caseId={caseInfo.case_id}
      notes={notes}
      review={review}
      readOnly={showcaseMode}
      onNoteSaved={onNoteSaved}
      onNoteDeleted={onNoteDeleted}
    />
  );
  const agentPanel = showcaseMode ? (
    <ShowcaseCaseAgentPanel
      data={data}
      onClose={() => onAgentOpenChange(false)}
    />
  ) : (
    <CaseAgentPanel
      data={data}
      activeStage={selectedStep}
      onClose={() => onAgentOpenChange(false)}
      onStageChange={onStepChange}
      onNoteSaved={onNoteSaved}
    />
  );

  return (
    <main className="case-workbench">
      <Card className="case-header-card">
        <Space className="case-header-content" orientation="vertical" size={0} style={{ width: '100%' }}>
          <div className="case-header-title-row">
            <Typography.Title level={4} style={{ margin: 0 }}>
              {caseInfo.case_title}
            </Typography.Title>
            <RiskTag level={caseInfo.risk_level} />
            {showcaseMode && (
              <Tag className="case-header-showcase-tag">只读展示</Tag>
            )}
            {review && (
              <Tag className="review-status-tag case-header-review-status review-reviewed">
                已初审
              </Tag>
            )}
          </div>
          <div className="case-header-meta">
            <Descriptions
              className="case-header-descriptions"
              column={3}
              items={[
                { key: 'case_id', label: '案件编号', children: caseInfo.case_id },
                { key: 'subject_ref', label: '脱敏申报人编号', children: subjectRef },
                {
                  key: 'risk_score',
                  label: '风险提示强度',
                  children: riskDisplayScore(caseInfo.risk_score, caseInfo.risk_score_breakdown),
                },
              ]}
            />
          </div>
          <WorkflowSteps
            workflow={workflow}
            selectedStep={selectedStep}
            onStepChange={onStepChange}
          />
        </Space>
      </Card>

      {fullTriPane ? (
        <Splitter
          className="case-stage-splitter"
          onResizeEnd={(sizes) => {
            const nextNotesWidth = sizes[0];
            if (nextNotesWidth) {
              setNotesWidth(nextNotesWidth);
              savePaneWidth('notes', nextNotesWidth);
            }
            if (agentOpen) {
              const nextAgentWidth = sizes[2];
              if (nextAgentWidth) {
                setAgentWidth(nextAgentWidth);
                savePaneWidth('agent', nextAgentWidth);
              }
            }
          }}
        >
          <Splitter.Panel defaultSize={notesWidth} min={280} max={420}>
            <div className="case-pane-scroll">{notesPanel}</div>
          </Splitter.Panel>
          <Splitter.Panel min={720}>
            <div className="case-pane-scroll">{stageContent}</div>
          </Splitter.Panel>
          {agentOpen && (
            <Splitter.Panel defaultSize={agentWidth} min={440} max={560}>
              <div className="case-pane-scroll">{agentPanel}</div>
            </Splitter.Panel>
          )}
        </Splitter>
      ) : splitPane ? (
        <div className="case-medium-workspace">
          <div className="case-auxiliary-switch">
            <Segmented
              value={agentOpen ? 'assistant' : 'notes'}
              options={[
                { value: 'notes', label: '审核工作笔记' },
                { value: 'assistant', label: '稽核助手' },
              ]}
              onChange={(value) => onAgentOpenChange(value === 'assistant')}
            />
          </div>
          {agentOpen ? (
            <Splitter
              className="case-stage-splitter"
              onResizeEnd={(sizes) => {
                const nextAgentWidth = sizes[1];
                if (nextAgentWidth) {
                  setAgentWidth(nextAgentWidth);
                  savePaneWidth('agent', nextAgentWidth);
                }
              }}
            >
              <Splitter.Panel min={720}>
                <div className="case-pane-scroll">{stageContent}</div>
              </Splitter.Panel>
              <Splitter.Panel defaultSize={agentWidth} min={420} max={520}>
                <div className="case-pane-scroll">{agentPanel}</div>
              </Splitter.Panel>
            </Splitter>
          ) : (
            <Splitter
              className="case-stage-splitter"
              onResizeEnd={(sizes) => {
                const nextNotesWidth = sizes[0];
                if (nextNotesWidth) {
                  setNotesWidth(nextNotesWidth);
                  savePaneWidth('notes', nextNotesWidth);
                }
              }}
            >
              <Splitter.Panel defaultSize={notesWidth} min={280} max={380}>
                <div className="case-pane-scroll">{notesPanel}</div>
              </Splitter.Panel>
              <Splitter.Panel min={720}>
                <div className="case-pane-scroll">{stageContent}</div>
              </Splitter.Panel>
            </Splitter>
          )}
        </div>
      ) : (
        <div className="case-small-workspace">
          <div className="case-mobile-tools">
            <Button onClick={() => setNotesDrawerOpen(true)}>
              审核工作笔记
            </Button>
            <Button
              type={agentOpen ? 'primary' : 'default'}
              onClick={() => onAgentOpenChange(true)}
            >
              案件稽核助手
            </Button>
          </div>
          {stageContent}
          <Drawer
            className="case-auxiliary-drawer"
            title="审核工作笔记"
            placement="left"
            size={Math.min(420, window.innerWidth)}
            open={notesDrawerOpen}
            onClose={() => setNotesDrawerOpen(false)}
          >
            {notesPanel}
          </Drawer>
          <Drawer
            className="case-auxiliary-drawer case-agent-drawer"
            closable={false}
            placement="right"
            size={Math.min(440, window.innerWidth)}
            open={agentOpen}
            onClose={() => onAgentOpenChange(false)}
          >
            {agentPanel}
          </Drawer>
        </div>
      )}
    </main>
  );
}

function RiskSummary({
  riskLevel,
  riskScore,
  riskScoreBreakdown,
  modelSignalSource,
  modelEvidenceRef,
  fraudScreening,
}: {
  riskLevel: CaseFullResponse['case']['risk_level'];
  riskScore: number;
  riskScoreBreakdown?: CaseFullResponse['case']['risk_score_breakdown'];
  modelSignalSource: string;
  modelEvidenceRef: string;
  fraudScreening?: CaseFullResponse['case']['fraud_screening'];
}) {
  const fraudSource = fraudScreening ?? {
    result: 'not_available',
    label: '未接入',
    source: '模型识别预警未接入',
    evidence_ref: 'fraud_screening:binary:not-configured',
    reason: '当前环境未配置独立模型识别预警。',
    probability: null,
  };
  const probabilityText = formatProbability(fraudSource.probability);
  const warningLabel =
    fraudSource.result === 'suspected'
      ? '有预警'
      : fraudSource.result === 'not_suspected'
        ? '无预警'
        : '未接入';
  const fraudSourceText =
    fraudSource.result === 'not_available'
      ? '当前未接入独立智能识别预警服务。'
      : `独立智能识别预警已返回“${warningLabel}”${probabilityText ? `，预警概率 ${probabilityText}` : ''}。`;
  const fallbackTotal = Math.round(riskScore * 100);
  const fallbackBreakdown: RiskScoreBreakdown = {
    total_score: fallbackTotal,
    max_score: 100,
    display_score: `${fallbackTotal} / 100`,
    level: riskLevel,
    level_label: RISK_SUMMARY_LABELS[riskLevel] ?? riskLevel,
    components: [
      {
        key: 'model_warning',
        label: '模型识别预警',
        score: fraudSource.probability ? Math.round(fraudSource.probability * 70) : 0,
        max_score: 70,
        summary: fraudSourceText,
        details: [fraudSourceText],
        source_detail: fraudSource.source,
      },
      {
        key: 'rule_check',
        label: '规则核验',
        score: 0,
        max_score: 15,
        summary: '当前接口未返回规则贡献分解。',
        details: ['请刷新案件详情或重新接入案件以生成结构化分解。'],
        source_detail: '固定医保业务核验规则池。',
      },
      {
        key: 'peer_deviation',
        label: '同类偏离',
        score: 0,
        max_score: 10,
        summary: '当前接口未返回同类偏离分解。',
        details: ['请刷新案件详情或重新接入案件以生成结构化分解。'],
        source_detail: '脱敏历史申报样本聚合分位参考，覆盖完整安全申报字段。',
      },
      {
        key: 'data_flow',
        label: '数据/流程完整性',
        score: 0,
        max_score: 5,
        summary: '当前接口未返回数据流程分解。',
        details: ['请刷新案件详情或重新接入案件以生成结构化分解。'],
        source_detail: '案件接入校验、挂号流程和业务材料提示。',
      },
    ],
  };
  const breakdown = riskScoreBreakdown ?? fallbackBreakdown;
  const componentTone: Record<string, string> = {
    model_warning: '#1b65b9',
    rule_check: '#4e7da8',
    peer_deviation: '#6f8ca7',
    data_flow: '#8aa0b4',
  };
  const modelComponent = breakdown.components.find((component) => component.key === 'model_warning');
  const sourceItems = [
    {
      key: 'model-source',
      label: '模型来源',
      children: modelComponent?.source_detail ?? fraudSourceText,
    },
    {
      key: 'rule-source',
      label: '规则来源',
      children: '固定医保业务核验规则池。',
    },
      {
        key: 'peer-source',
        label: '同类参考',
        children: '脱敏历史申报样本聚合分位参考，覆盖完整安全申报字段，不包含行级标签。',
      },
    {
      key: 'score-source',
      label: '计算口径',
      children: businessText(modelSignalSource),
    },
    {
      key: 'ref',
      label: '结果留痕',
      children: evidenceRefLabel(modelEvidenceRef),
    },
    ...(breakdown.cap_note
      ? [
          {
            key: 'cap',
            label: '封顶口径',
            children: breakdown.cap_note,
          },
        ]
      : []),
    {
      key: 'boundary',
      label: '审核边界',
      children: '以上结果仅用于提示人工核验方向，不自动形成通过、拒付、处罚或欺诈认定。',
    },
  ];

  return (
    <Card title="风险筛查">
      <Space orientation="vertical" size={18} style={{ width: '100%' }}>
        <section className="risk-composite-panel">
          <div className="risk-composite-head">
            <div>
              <Typography.Text type="secondary">综合风险提示强度</Typography.Text>
              <div className="risk-composite-score">{breakdown.display_score}</div>
            </div>
            <span className={`risk-summary-chip risk-summary-chip-${breakdown.level}`}>
              <span className="risk-summary-chip-label">{breakdown.level_label}</span>
            </span>
          </div>
          <Progress
            percent={breakdown.total_score}
            status="normal"
            strokeColor="#1b65b9"
            trailColor="#dce8f4"
            showInfo={false}
          />
          <div className="risk-scale-labels">
            <span>低</span>
            <span>中</span>
            <span>高</span>
          </div>
        </section>

        <section className="risk-contribution-list" aria-label="风险提示强度贡献构成">
          {breakdown.components.map((component) => (
            <article className="risk-contribution-item" key={component.key}>
              <div className="risk-contribution-head">
                <Typography.Text strong>{component.label}</Typography.Text>
                <Typography.Text className="risk-contribution-score">
                  {component.score} / {component.max_score}
                </Typography.Text>
              </div>
              <Progress
                percent={Math.round((component.score / component.max_score) * 100)}
                status="normal"
                strokeColor={componentTone[component.key] ?? '#6f8ca7'}
                trailColor="#e9eff5"
                showInfo={false}
              />
              <Typography.Text className="risk-contribution-summary">
                {businessText(component.summary)}
              </Typography.Text>
            </article>
          ))}
        </section>

        <Collapse
          className="risk-breakdown-collapse"
          size="small"
          items={[
            {
              key: 'calculation',
              label: '查看计算说明',
              children: (
                <div className="risk-calculation-list">
                  {breakdown.components.map((component) => (
                    <section key={component.key}>
                      <Typography.Text strong>{component.label}</Typography.Text>
                      <ul>
                        {component.details.map((detail) => (
                          <li key={detail}>{businessText(detail)}</li>
                        ))}
                      </ul>
                    </section>
                  ))}
                  {breakdown.cap_note && (
                    <Typography.Paragraph className="risk-cap-note">
                      {breakdown.cap_note}
                    </Typography.Paragraph>
                  )}
                </div>
              ),
            },
            {
              key: 'sources',
              label: '来源明细',
              children: (
                <Descriptions
                  size="small"
                  bordered
                  column={1}
                  items={sourceItems}
                />
              ),
            },
          ]}
        />
      </Space>
    </Card>
  );
}

function CaseIntake({
  caseId,
  caseType,
  subjectRef,
}: {
  caseId: string;
  caseType: string;
  subjectRef: string;
}) {
  return (
    <Card title="案件接入" extra={<Tag color="blue">已接入</Tag>}>
      <Space orientation="vertical" size={16} style={{ width: '100%' }}>
        <Descriptions
          bordered
          column={{ xs: 1, md: 2 }}
          items={[
            { key: 'case_id', label: '案件编号', children: caseId },
            { key: 'case_type', label: '案件类型', children: caseType },
            { key: 'subject_ref', label: '脱敏申报人编号', children: subjectRef },
            {
              key: 'source',
              label: '接入方式',
              children: '脱敏医保结算统计记录建案',
            },
          ]}
        />
      </Space>
    </Card>
  );
}

function FactBase({
  caseId,
  sourceRecordEntries,
  sourceRecord,
  features,
}: {
  caseId: string;
  sourceRecordEntries: Array<{ field: string; value: string }>;
  sourceRecord: Record<string, number | string>;
  features: Record<string, number>;
}) {
  const metrics = buildFactMetrics(sourceRecord, features);
  const recordRows = buildSourceRecordRows(sourceRecordEntries, sourceRecord, features, metrics);
  const qualitySummary = summarizeRecordQuality(recordRows);

  return (
    <Card className="fact-base-card" title="事实底座">
      <Space orientation="vertical" size={16} style={{ width: '100%' }}>
        <StatisticalMaterialsBlock caseId={caseId} />
        <Collapse
          items={[
            {
              key: 'full-record',
              label: (
                <div className="source-record-collapse-label">
                  <span>结算申报核验明细</span>
                  <span className="source-record-completeness">
                    字段完整性 {qualitySummary.presentFields}/{qualitySummary.totalFields}
                  </span>
                </div>
              ),
              children: (
                <Space orientation="vertical" size={0} style={{ width: '100%' }}>
                  <Table
                    className="source-record-table"
                    size="small"
                    rowKey="rawField"
                    dataSource={recordRows}
                    pagination={{ pageSize: 12 }}
                    columns={[
                      { title: '特征', dataIndex: 'field', width: 280 },
                      { title: '当前值', dataIndex: 'value', width: 180 },
                      { title: '同类中位数', dataIndex: 'median', width: 150 },
                      { title: 'P75', dataIndex: 'p75', width: 130 },
                      { title: 'P90', dataIndex: 'p90', width: 130 },
                      {
                        title: '提示',
                        dataIndex: 'hint',
                        width: 140,
                        render: (value: string, record: SourceRecordRow) => (
                          value ? (
                            <span className={`fact-table-hint is-${record.tone}`}>
                              {value}
                            </span>
                          ) : null
                        ),
                      },
                    ]}
                  />
                </Space>
              ),
            },
          ]}
        />
      </Space>
    </Card>
  );
}

function StatisticalMaterialsBlock({ caseId }: { caseId: string }) {
  const [materials, setMaterials] = useState<StatisticalMaterialResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedDocument, setSelectedDocument] =
    useState<StatisticalMaterialDocument | null>(null);
  const [previewAsset, setPreviewAsset] = useState<BusinessMaterialAsset | null>(null);
  const documents = materials?.documents ?? [];
  const categories = materials?.categories?.length
    ? materials.categories
    : buildMaterialCategories(documents);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    fetchCaseStatisticalMaterials(caseId)
      .then((response) => {
        if (!active) return;
        setMaterials(response);
      })
      .catch((err) => {
        if (!active) return;
        setError(err instanceof Error ? err.message : '加载统计材料失败');
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [caseId]);

  return (
    <section className="medical-record-section">
      <div className="medical-record-head">
        <Typography.Title level={5}>业务材料视图</Typography.Title>
        <Tag color="blue">{documents.length} 份</Tag>
      </div>
      {error && <Alert type="warning" showIcon title={error} />}
      {materials?.notice && (
        <Alert
          className="medical-record-notice"
          type="info"
          showIcon
          message={materials.notice}
        />
      )}
      {materials?.subject_profile && (
        <MaterialBasicInfoBlock
          profile={materials.subject_profile}
          caseContext={materials.case_context ?? {}}
        />
      )}
      <Collapse
        className="medical-record-category-collapse"
        defaultActiveKey={categories.slice(0, 1).map((category) => category.category_id)}
        items={categories.map((category) => ({
          key: category.category_id,
          label: (
            <div className="medical-record-category-label">
              <span>{category.title}</span>
              <Tag>{category.documents.length} 份</Tag>
            </div>
          ),
          children: (
            <Table
              className="medical-record-table"
              size="small"
              rowKey="document_id"
              loading={loading}
              dataSource={category.documents}
              pagination={false}
              locale={{ emptyText: <Empty description="暂无业务材料" /> }}
              columns={[
                { title: '材料名称', dataIndex: 'title', width: 180 },
                { title: '发生时间', dataIndex: 'occurred_at', width: 150 },
                { title: '材料来源', dataIndex: 'material_source', width: 200 },
                { title: '材料形态', dataIndex: 'material_shape', width: 110 },
                {
                  title: '状态',
                  dataIndex: 'status',
                  width: 100,
                  render: (value: string) => (
                    <Tag color={value.includes('失败') ? 'red' : 'blue'}>{value}</Tag>
                  ),
                },
                {
                  title: '操作',
                  key: 'action',
                  width: 100,
                  render: (_, record) => (
                    <Button size="small" onClick={() => setSelectedDocument(record)}>
                      查看
                    </Button>
                  ),
                },
              ]}
            />
          ),
        }))}
      />
      <Modal
        className="medical-record-modal"
        title="材料详情"
        width={960}
        open={Boolean(selectedDocument)}
        onCancel={() => {
          setPreviewAsset(null);
          setSelectedDocument(null);
        }}
        footer={[
          <Button
            key="close"
            onClick={() => {
              setPreviewAsset(null);
              setSelectedDocument(null);
            }}
          >
            关闭
          </Button>,
        ]}
      >
        {selectedDocument && (
          <MaterialDocumentDetail
            document={selectedDocument}
            profile={materials?.subject_profile ?? null}
            onPreview={setPreviewAsset}
          />
        )}
      </Modal>
      <Modal
        className="medical-record-asset-preview-modal"
        title={previewAsset?.title ?? '图片预览'}
        width={900}
        open={Boolean(previewAsset)}
        onCancel={() => setPreviewAsset(null)}
        footer={[
          <Button
            key="download"
            href={previewAsset ? `${previewAsset.preview_url}?download=1` : undefined}
            download={previewAsset ? `${previewAsset.title}.png` : undefined}
          >
            下载
          </Button>,
          <Button key="close" type="primary" onClick={() => setPreviewAsset(null)}>
            关闭
          </Button>,
        ]}
      >
        {previewAsset && (
          <img
            src={previewAsset.preview_url}
            alt={previewAsset.title}
            className="medical-record-asset-preview-image"
          />
        )}
      </Modal>
    </section>
  );
}

function MaterialDocumentDetail({
  document,
  profile,
  onPreview,
}: {
  document: StatisticalMaterialDocument;
  profile: MaterialSubjectProfile | null;
  onPreview: (asset: BusinessMaterialAsset) => void;
}) {
  const contentLines = splitMaterialContent(document.content);
  const visitMaterialProfile = document.category_id === 'clinical' ? profile : null;

  return (
    <Space
      className="medical-record-document-detail"
      orientation="vertical"
      size={14}
      style={{ width: '100%' }}
    >
      <div className="medical-record-document-head">
        <Typography.Title level={5}>{document.title}</Typography.Title>
        <Tag color={document.status.includes('失败') ? 'red' : 'blue'}>{document.status}</Tag>
      </div>
      <Descriptions
        bordered
        column={{ xs: 1, md: 2 }}
        size="small"
        items={[
          { key: 'type', label: '材料类型', children: document.document_type },
          { key: 'time', label: '发生时间', children: document.occurred_at || document.visit_date },
          { key: 'source', label: '材料来源', children: document.material_source || document.institution },
          { key: 'shape', label: '材料形态', children: document.material_shape || '文本' },
        ]}
      />
      {!!document.summary_items?.length && (
        <div className="medical-record-summary-tags">
          {document.summary_items.map((item) => (
            <Tag key={item}>{item}</Tag>
          ))}
        </div>
      )}
      {visitMaterialProfile && (
        <section className="medical-record-content medical-record-clinical-profile">
          <Typography.Title level={5}>就诊材料信息</Typography.Title>
          <Descriptions
            size="small"
            column={{ xs: 1, md: 2 }}
            items={[
              {
                key: 'visit_type',
                label: '主要就诊类型',
                children: visitMaterialProfile.primary_visit_type,
              },
              {
                key: 'registration_status',
                label: '挂号状态',
                children: visitMaterialProfile.registration_status,
              },
              {
                key: 'chronic_tags',
                label: '慢病标签',
                children: tagList(visitMaterialProfile.chronic_condition_tags),
              },
              {
                key: 'allergy',
                label: '过敏史',
                children: visitMaterialProfile.allergy_history,
              },
            ]}
          />
        </section>
      )}
      {!!contentLines.length && (
        <section className="medical-record-content">
          <Typography.Title level={5}>材料内容</Typography.Title>
          <div className="medical-record-content-lines">
            {contentLines.map((line, index) => (
              <div className="medical-record-content-line" key={`${line}-${index}`}>
                {line}
              </div>
            ))}
          </div>
        </section>
      )}
      {!!document.tables?.length && (
        <Space orientation="vertical" size={12} style={{ width: '100%' }}>
          {document.tables.map((table) => (
            <section className="medical-record-content" key={table.title}>
              <Typography.Title level={5}>{table.title}</Typography.Title>
              <Table
                size="small"
                rowKey={(_, index) => `${table.title}-${index ?? 0}`}
                dataSource={table.rows}
                pagination={false}
                scroll={{ x: true }}
                columns={table.columns.map((column) => ({
                  title: column,
                  dataIndex: column,
                  render: (value: unknown) => String(value ?? ''),
                }))}
              />
            </section>
          ))}
        </Space>
      )}
      {!!document.assets?.length && (
        <section className="medical-record-content">
          <Typography.Title level={5}>图片附件</Typography.Title>
          <div className="medical-record-asset-list">
            {document.assets.map((asset) => (
              <div className="medical-record-asset-item" key={asset.asset_id}>
                <div>
                  <Typography.Text strong>{asset.title}</Typography.Text>
                  <Typography.Text type="secondary">
                    {asset.file_type} · {formatFileSize(asset.size_bytes)}
                  </Typography.Text>
                </div>
                <Space size={8}>
                  <Button size="small" onClick={() => onPreview(asset)}>
                    预览
                  </Button>
                  <Button
                    size="small"
                    href={`${asset.preview_url}?download=1`}
                    download={`${asset.title}.png`}
                  >
                    下载
                  </Button>
                </Space>
              </div>
            ))}
          </div>
        </section>
      )}
    </Space>
  );
}

function MaterialBasicInfoBlock({
  profile,
  caseContext,
}: {
  profile: MaterialSubjectProfile;
  caseContext: Record<string, unknown>;
}) {
  return (
    <section className="medical-record-subject">
      <Descriptions
        size="small"
        column={{ xs: 1, md: 3 }}
        items={[
          { key: 'subject_ref', label: '脱敏申报人编号', children: profile.subject_ref },
          { key: 'gender', label: '性别', children: profile.gender },
          { key: 'age_group', label: '年龄段', children: profile.age_group },
          { key: 'insurance_type', label: '参保类型', children: profile.insurance_type },
          {
            key: 'insured_region',
            label: '参保地',
            children: contextText(caseContext, 'insured_region'),
          },
          {
            key: 'treatment_region',
            label: '就医地',
            children: contextText(caseContext, 'treatment_region'),
          },
          {
            key: 'visit_type',
            label: '就医类型',
            children: contextText(caseContext, 'visit_type', profile.primary_visit_type),
          },
          {
            key: 'claim_mode',
            label: '报销方式',
            children: contextText(caseContext, 'claim_mode'),
          },
          {
            key: 'direct_settlement',
            label: '直接结算',
            children: contextText(caseContext, 'direct_settlement'),
          },
          {
            key: 'filing_status',
            label: '备案状态',
            children: contextText(caseContext, 'filing_status'),
          },
          {
            key: 'emergency_material_status',
            label: '急诊材料',
            children: contextText(caseContext, 'emergency_material_status'),
          },
        ]}
      />
    </section>
  );
}

function buildMaterialCategories(
  documents: StatisticalMaterialDocument[],
): BusinessMaterialCategory[] {
  const definitions: Array<[string, string]> = [
    ['clinical', '就诊诊疗材料'],
    ['prescription', '处方购药材料'],
    ['settlement', '费用结算材料'],
  ];
  return definitions.map(([categoryId, title]) => ({
    category_id: categoryId,
    title,
    documents: documents.filter((document) => document.category_id === categoryId),
  }));
}

function tagList(items: string[]) {
  if (!items.length) return '未见记录';
  return (
    <Space size={4} wrap>
      {items.map((item) => (
        <Tag key={item}>{item}</Tag>
      ))}
    </Space>
  );
}

function splitMaterialContent(content: string | undefined): string[] {
  if (!content) return [];
  return content
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
}

function contextText(
  context: Record<string, unknown>,
  key: string,
  fallback = '未记录',
): string {
  const value = context[key];
  if (value === undefined || value === null || value === '') return fallback;
  return localizeCaseContextValue(key, value);
}

function localizeCaseContextValue(key: string, value: unknown): string {
  if (typeof value === 'boolean') {
    return value ? '是' : '否';
  }
  const text = String(value);
  const dictionaries: Record<string, Record<string, string>> = {
    claim_mode: {
      manual_reimbursement: '手工报销',
      manual_upload_review: '手工上传材料复核',
      post_settlement_review: '医保结算后复核',
      direct_settlement: '直接结算',
    },
    filing_status: {
      unknown: '状态不明',
      missing: '缺失',
      not_applicable: '不适用',
      filed: '已备案',
      online_filing_effective: '线上备案即时生效',
      filed_or_emergency_exception_supported: '已备案或急诊例外材料支持',
    },
    emergency_material_status: {
      unknown: '状态不明',
      missing: '缺失',
      missing_or_unclear: '缺失或不清晰',
      present: '已上传',
      present_but_needs_verification: '已上传，待核验',
      ordinary_outpatient_only: '仅普通门诊材料',
      clear: '材料清晰',
      not_applicable: '不适用',
    },
  };
  return dictionaries[key]?.[text] ?? text;
}

function SecondaryReview({
  review,
}: {
  review: CaseFullResponse['review'];
}) {
  const transferRequested = review?.decision.includes('移交') ?? false;

  return (
    <Card
      className="conditional-stage-card"
      title="人工复审"
      extra={
        <Tag className="conditional-stage-tag">
          条件阶段【后端流程未接入】
        </Tag>
      }
    >
      <Space orientation="vertical" size={16} style={{ width: '100%' }}>
        <Alert
          type={transferRequested ? 'warning' : 'info'}
          showIcon
          title={
            transferRequested
              ? '当前初审意见包含上级复审线索'
              : review
                ? '当前初审未标记必须进入上级复审'
                : '人工初审完成后判定是否进入复审'
          }
          description="本节点当前没有后端复审状态，页面只展示业务条件和角色边界。【后端流程未接入】"
        />
        <Descriptions
          bordered
          column={1}
          items={[
            {
              key: 'role',
              label: '复审角色',
              children: '医保侧上级审核员、审核组长或授权复审人员',
            },
            {
              key: 'entry',
              label: '进入条件',
              children:
                '初审移交、结论争议、重大风险线索，或补充材料可能改变原有判断',
            },
            {
              key: 'focus',
              label: '复审重点',
              children:
                '复核初审依据、规则适用、证据完整性和补充材料，不重复机械初审',
            },
            {
              key: 'declarant',
              label: '申报方权限',
              children: '仅提交补充或申诉材料，不参与医保侧复审',
            },
          ]}
        />
      </Space>
    </Card>
  );
}

function AppealHandling({
  review,
  onOpenAppealPortal,
}: {
  review: CaseFullResponse['review'];
  onOpenAppealPortal: () => void;
}) {
  return (
    <Card
      className="conditional-stage-card"
      title="申诉处理"
      extra={
        <Tag className="conditional-stage-tag">
          条件阶段【后端流程未接入】
        </Tag>
      }
    >
      <Space orientation="vertical" size={16} style={{ width: '100%' }}>
        <Alert
          type="info"
          showIcon
          title={
            review
              ? '初审意见已形成，申报方可按需补充或申诉'
              : '初审意见形成后开放申诉材料处理'
          }
          description="申诉处理应比较新增材料与原证据包；当前入口仅保留本页材料选择状态。【后端流程未接入】"
        />
        <Descriptions
          bordered
          column={1}
          items={[
            {
              key: 'declarant',
              label: '申报方',
              children: '提交补充材料或申诉材料',
            },
            {
              key: 'reviewer',
              label: '医保审核方',
              children: '核验新增材料、原规则依据和原人工意见的一致性',
            },
            {
              key: 'agent',
              label: '案件稽核助手',
              children:
                '整理材料差异与待核验项，不代替审核人员形成申诉结论',
            },
            {
              key: 'storage',
                label: '材料接收状态',
                children: '当前环境未配置材料存储，刷新后清空【后端流程未接入】',
            },
          ]}
        />
        <div className="conditional-stage-actions">
          <Button type="primary" onClick={onOpenAppealPortal}>
            打开申诉材料入口
          </Button>
        </div>
      </Space>
    </Card>
  );
}

function CaseResult({
  review,
}: {
  review: CaseFullResponse['review'];
}) {
  if (!review) {
    return (
      <Card title="案件处理结果">
        <Empty description="待人工初审及后续条件阶段完成后形成处理结果" />
      </Card>
    );
  }

  return (
    <Card title="案件处理结果" extra={<Tag color="green">已记录</Tag>}>
      <Space orientation="vertical" size={16} style={{ width: '100%' }}>
        <Alert
          type="success"
          showIcon
          title="已记录当前人工处理意见"
          description="当前系统使用已提交的人工初审记录展示处理结果；不表示复审或申诉流程已经完成。"
        />
        <Descriptions
          bordered
          column={1}
          items={[
            { key: 'reviewer', label: '处理人员', children: review.reviewer },
            { key: 'decision', label: '处理意见', children: review.decision },
            { key: 'reason', label: '处理理由', children: review.reason },
            {
              key: 'submitted_at',
              label: '记录时间',
              children: review.submitted_at,
            },
            {
              key: 'conditional',
              label: '复审 / 申诉',
              children: '未配置复审 / 申诉流转状态【后端流程未接入】',
            },
          ]}
        />
        {(review.attachments ?? []).length > 0 && (
          <section className="review-attachment-result">
            <Typography.Title level={5}>附件材料登记</Typography.Title>
            <ReviewAttachmentTable attachments={review.attachments} />
          </section>
        )}
        <Alert
          type="info"
          showIcon
          title="案件结果必须由有权限的医保审核人员确认，案件稽核助手无权修改。"
        />
      </Space>
    </Card>
  );
}

function ReviewAttachmentTable({
  attachments,
}: {
  attachments: NonNullable<CaseFullResponse['review']>['attachments'];
}) {
  return (
    <Table
      size="small"
      rowKey={(record) => `${record.name}-${record.material_type}`}
      pagination={false}
      dataSource={attachments}
      columns={[
        { title: '材料名称', dataIndex: 'name' },
        { title: '材料类型', dataIndex: 'material_type', width: 140 },
        { title: '来源', dataIndex: 'source', width: 140 },
        { title: '核验状态', dataIndex: 'verification_status', width: 120 },
        { title: '备注', dataIndex: 'remark' },
      ]}
    />
  );
}
