import { GithubOutlined } from '@ant-design/icons';
import { Tag, Typography } from 'antd';

export const PROJECT_DISCLAIMER =
  '本项目为个人学习、竞赛及面试演示作品，非政府网站，非国家医保服务平台，与任何医疗保障行政部门不存在隶属或授权关系。系统仅使用合成脱敏数据，不提供真实医保业务办理、医学诊断或自动裁决服务。';

const PRIMARY_TECH_STACK = [
  'React',
  'FastAPI',
  'LangGraph',
  'RAG',
  'Ragas',
  'Mem0',
  'PostgreSQL',
  'Milvus',
  'Redis',
  'Nginx',
  'Docker',
  'BGE-M3',
  'BM25',
  'RRF',
  'XGBoost',
  'Text-to-SQL',
  'Multi-Agent',
  'MCP',
];
const PROJECT_GITHUB_URL =
  import.meta.env.VITE_PROJECT_GITHUB_URL?.trim() ||
  'https://github.com/lmx426/mediguard-agent';

interface ProjectOverviewProps {
  showcaseMode?: boolean;
}

export function ProjectOverview({ showcaseMode = false }: ProjectOverviewProps) {
  return (
    <section className="project-showcase" aria-label="项目介绍">
      {showcaseMode && (
        <div className="project-showcase-badges">
          <Tag color="blue">面试只读演示</Tag>
        </div>
      )}

      <p className="project-showcase-description">
        <strong>项目描述：</strong>
        面向医保智能稽核场景，融合风险模型、规则引擎与 Multi-Agent；以确定性九阶段流程承载业务状态，将 CaseR
        作为嵌入式智能助手，通过“规划–执行–总结”工作流与状态路由编排专家 Agents，完成风险筛查、证据复核、分析溯源与结果校验。
      </p>

      <div className="project-showcase-repository">
        <GithubOutlined />
        <strong>GitHub：</strong>
        <Typography.Link href={PROJECT_GITHUB_URL} target="_blank" rel="noreferrer">
          {PROJECT_GITHUB_URL}
        </Typography.Link>
      </div>

      <div className="project-showcase-stack">
        <strong>核心技术栈：</strong>
        <div className="project-showcase-tags">
          {PRIMARY_TECH_STACK.map((item) => (
            <Tag key={item}>{item}</Tag>
          ))}
        </div>
      </div>
    </section>
  );
}
