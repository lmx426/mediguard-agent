import type { WorkflowStepKey } from './types';

export const DEFAULT_WORKFLOW_STEP: WorkflowStepKey = 'risk_screening';

export const WORKFLOW_STEP_KEYS: WorkflowStepKey[] = [
  'case_intake',
  'fact_base',
  'risk_screening',
  'rule_check',
  'evidence_package',
  'initial_review',
  'secondary_review',
  'appeal_handling',
  'case_result',
];
