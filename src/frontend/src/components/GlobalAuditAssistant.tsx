import { FloatButton } from 'antd';
import AuditAssistantIcon from './AuditAssistantIcon';

interface GlobalAuditAssistantProps {
  hidden?: boolean;
  active?: boolean;
  onActivate: () => void;
}

export default function GlobalAuditAssistant({
  hidden = false,
  active = false,
  onActivate,
}: GlobalAuditAssistantProps) {
  if (hidden) return null;

  return (
    <FloatButton
      className={`global-agent-button ${active ? 'is-active' : ''}`}
      icon={<AuditAssistantIcon />}
      tooltip={active ? '关闭稽核助手' : '打开稽核助手'}
      aria-label={active ? '关闭稽核助手' : '打开稽核助手'}
      onClick={onActivate}
    />
  );
}
