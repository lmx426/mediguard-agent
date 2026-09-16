import {
  CheckCircleOutlined,
  CloseOutlined,
  DatabaseOutlined,
  SafetyCertificateOutlined,
} from '@ant-design/icons';
import {
  Alert,
  Button,
  Collapse,
  Empty,
  Skeleton,
  Space,
  Tag,
  Tooltip,
  Typography,
} from 'antd';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { fetchShowcaseCaseAgent } from '../services/api';
import type {
  CaseFullResponse,
  ShowcaseCaseAgentProjection,
} from '../types';
import AuditAssistantIcon from './AuditAssistantIcon';

interface ShowcaseCaseAgentPanelProps {
  data: CaseFullResponse;
  onClose?: () => void;
}

export default function ShowcaseCaseAgentPanel({
  data,
  onClose,
}: ShowcaseCaseAgentPanelProps) {
  const caseId = data.case.case_id;
  const [projection, setProjection] = useState<ShowcaseCaseAgentProjection | null>(null);
  const [activeKey, setActiveKey] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadProjection = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const next = await fetchShowcaseCaseAgent(caseId);
      setProjection(next);
      setActiveKey(next.answers[0]?.key ?? '');
    } catch (err) {
      setProjection(null);
      setError(err instanceof Error ? err.message : '预生成案件助手内容加载失败');
    } finally {
      setLoading(false);
    }
  }, [caseId]);

  useEffect(() => {
    void loadProjection();
  }, [loadProjection]);

  const activeAnswer = useMemo(
    () => projection?.answers.find((item) => item.key === activeKey) ?? projection?.answers[0],
    [activeKey, projection],
  );
  const sourceMap = useMemo(
    () => new Map(projection?.sources.map((source) => [source.source_ref, source]) ?? []),
    [projection],
  );

  return (
    <section className="case-agent-panel showcase-case-agent-panel" aria-label="案件稽核助手只读演示">
      <div className="case-agent-shell">
        <header className="case-agent-header">
          <div className="case-agent-title showcase-case-agent-title">
            <AuditAssistantIcon />
            <span>{projection?.title ?? '案件稽核助手 · 预生成会话'}</span>
          </div>
          {onClose && (
            <Tooltip title="关闭案件稽核助手">
              <Button
                type="text"
                shape="circle"
                icon={<CloseOutlined />}
                aria-label="关闭案件稽核助手"
                onClick={onClose}
              />
            </Tooltip>
          )}
        </header>

        <div className="showcase-case-agent-body">
          {loading && !projection ? (
            <Skeleton active paragraph={{ rows: 8 }} />
          ) : error ? (
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description={error}
            >
              <Button onClick={() => void loadProjection()}>重新加载</Button>
            </Empty>
          ) : projection && activeAnswer ? (
            <>
              <Alert
                className="showcase-agent-notice"
                type="info"
                showIcon
                message={projection.generated_notice}
                description={projection.boundary_notice}
              />

              <div className="showcase-case-agent-questions" aria-label="预生成问题">
                {projection.answers.map((answer) => (
                  <Button
                    key={answer.key}
                    type={answer.key === activeAnswer.key ? 'primary' : 'default'}
                    onClick={() => setActiveKey(answer.key)}
                  >
                    {answer.question}
                  </Button>
                ))}
              </div>

              <section className="showcase-case-agent-conversation">
                <div className="showcase-case-agent-user-message">
                  <Typography.Text>{activeAnswer.question}</Typography.Text>
                </div>
                <div className="showcase-case-agent-answer">
                  <div className="showcase-case-agent-answer-label">
                    <AuditAssistantIcon />
                    <Typography.Text strong>案件助手</Typography.Text>
                    <Tag bordered={false}>预生成</Tag>
                  </div>
                  <Typography.Paragraph>{activeAnswer.answer}</Typography.Paragraph>
                  <div className="showcase-case-agent-checks">
                    <Typography.Text strong>建议核验清单</Typography.Text>
                    {activeAnswer.checks.map((check) => (
                      <div key={check} className="showcase-case-agent-check">
                        <CheckCircleOutlined />
                        <span>{check}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </section>

              <Collapse
                className="showcase-case-agent-sources"
                size="small"
                items={[
                  {
                    key: 'sources',
                    label: (
                      <Space size={8}>
                        <DatabaseOutlined />
                        <span>查看本回答依据</span>
                        <Tag>{activeAnswer.source_refs.length}</Tag>
                      </Space>
                    ),
                    children: (
                      <div className="showcase-case-agent-source-list">
                        {activeAnswer.source_refs.map((ref) => {
                          const source = sourceMap.get(ref);
                          if (!source) return null;
                          return (
                            <div className="showcase-case-agent-source" key={ref}>
                              <Typography.Text strong>{source.label}</Typography.Text>
                              <Typography.Text type="secondary">{source.summary}</Typography.Text>
                            </div>
                          );
                        })}
                      </div>
                    ),
                  },
                ]}
              />

              <div className="showcase-case-agent-boundary">
                <SafetyCertificateOutlined />
                <Typography.Text>
                  展示内容不连接外部模型，不保存对话，也不替代人工审核。
                </Typography.Text>
              </div>
            </>
          ) : (
            <Empty description="当前案件没有可展示的预生成助手内容" />
          )}
        </div>
      </div>
    </section>
  );
}
