import { CheckCircleFilled, LoadingOutlined } from '@ant-design/icons';
import { Steps, Typography } from 'antd';
import type { WorkflowResponse, WorkflowStepKey } from '../types';

interface WorkflowStepsProps {
  workflow: WorkflowResponse;
  selectedStep: WorkflowStepKey;
  onStepChange: (step: WorkflowStepKey) => void;
}

const STATUS_LABELS: Record<string, string> = {
  pending: '待处理',
  conditional: '条件进入',
};

function toAntdStatus(status: string) {
  if (status === 'current') return 'process';
  if (status === 'pending') return 'wait';
  if (status === 'conditional') return 'wait';
  return 'finish';
}

function renderStatusDescription(status: string) {
  if (status === 'completed' || status === 'recorded') {
    const label = status === 'completed' ? '已完成' : '已记录';
    return (
      <span
        className={`workflow-status-icon workflow-status-icon-${status}`}
        aria-label={label}
        title={label}
      >
        <CheckCircleFilled aria-hidden="true" />
      </span>
    );
  }

  if (status === 'current') {
    return (
      <span
        className="workflow-status-icon workflow-status-icon-current"
        aria-label="进行中"
        title="进行中"
      >
        <LoadingOutlined aria-hidden="true" spin />
      </span>
    );
  }

  return (
    <span className={`workflow-status-text is-${status}`}>
      {STATUS_LABELS[status] ?? status}
    </span>
  );
}

export default function WorkflowSteps({
  workflow,
  selectedStep,
  onStepChange,
}: WorkflowStepsProps) {
  const current = Math.max(
    workflow.steps.findIndex((step) => step.key === selectedStep),
    0,
  );

  return (
    <section className="workflow-card" aria-label="稽核流程追踪">
      <div className="workflow-card-header">
        <Typography.Title level={5}>稽核流程追踪</Typography.Title>
      </div>
      <Steps
        progressDot
        responsive={false}
        size="small"
        current={current}
        onChange={(index) => {
          const step = workflow.steps[index];
          if (step) onStepChange(step.key);
        }}
        items={workflow.steps.map((step) => {
          const isViewing = step.key === selectedStep;

          return {
            key: step.key,
            className: isViewing ? 'is-viewing' : undefined,
            title: (
              <span
                className={`workflow-step-shell${isViewing ? ' is-viewing' : ''}`}
                aria-current={isViewing ? 'step' : undefined}
              >
                <span className="workflow-step-title-text">{step.title}</span>
                <span className="workflow-step-status-slot">
                  {renderStatusDescription(step.status)}
                </span>
              </span>
            ),
            status: toAntdStatus(step.status),
          };
        })}
      />
    </section>
  );
}
