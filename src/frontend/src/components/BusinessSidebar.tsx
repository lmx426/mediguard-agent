import {
  AuditOutlined,
  CloudUploadOutlined,
} from '@ant-design/icons';
import { Drawer, Layout, Menu } from 'antd';
import type { MenuProps } from 'antd';
import AuditAssistantIcon from './AuditAssistantIcon';

export type BusinessModule = 'audit' | 'ingest' | 'assistant';

interface BusinessSidebarProps {
  activeModule: BusinessModule;
  collapsed: boolean;
  desktop: boolean;
  mobileOpen: boolean;
  onCollapse: (collapsed: boolean) => void;
  onMobileClose: () => void;
  onNavigate: (module: BusinessModule) => void;
}

const MENU_ITEMS: MenuProps['items'] = [
  {
    key: 'ingest',
    icon: <CloudUploadOutlined />,
    label: '上游接入',
  },
  {
    key: 'audit',
    icon: <AuditOutlined />,
    label: '审核管理',
  },
  {
    key: 'assistant',
    icon: <AuditAssistantIcon />,
    label: '稽核助手',
  },
];

function SidebarMenu({
  activeModule,
  onNavigate,
}: Pick<BusinessSidebarProps, 'activeModule' | 'onNavigate'>) {
  return (
    <Menu
      className="business-menu"
      mode="inline"
      selectedKeys={[activeModule]}
      items={MENU_ITEMS}
      onClick={({ key }) => onNavigate(key as BusinessModule)}
    />
  );
}

export default function BusinessSidebar({
  activeModule,
  collapsed,
  desktop,
  mobileOpen,
  onCollapse,
  onMobileClose,
  onNavigate,
}: BusinessSidebarProps) {
  if (!desktop) {
    return (
      <Drawer
        className="business-mobile-drawer"
        title="业务导航"
        placement="left"
        size={288}
        open={mobileOpen}
        onClose={onMobileClose}
      >
        <SidebarMenu
          activeModule={activeModule}
          onNavigate={(module) => {
            onNavigate(module);
            onMobileClose();
          }}
        />
      </Drawer>
    );
  }

  return (
    <Layout.Sider
      className="business-sidebar"
      theme="light"
      width={208}
      collapsedWidth={64}
      collapsible
      collapsed={collapsed}
      onCollapse={onCollapse}
    >
      <div className="business-sidebar-heading">
        {collapsed ? '业务' : '业务办理'}
      </div>
      <SidebarMenu activeModule={activeModule} onNavigate={onNavigate} />
    </Layout.Sider>
  );
}
