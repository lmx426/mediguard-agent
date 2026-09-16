import {
  LockOutlined,
  SafetyCertificateOutlined,
  UserOutlined,
} from '@ant-design/icons';
import {
  Alert,
  Button,
  Card,
  Form,
  Input,
  Space,
  Typography,
} from 'antd';
import { useState } from 'react';
import { login } from '../services/api';
import type { AuthenticatedUser, LoginInput } from '../types';
import MediGuardMark from './MediGuardMark';
import {
  PROJECT_DISCLAIMER,
  ProjectOverview,
} from './ProjectShowcase';

interface LoginPageProps {
  showcaseMode?: boolean;
  onLogin: (user: AuthenticatedUser) => void;
}

const DEMO_REVIEWER_USERNAME = 'default_auditor';
const DEMO_REVIEWER_PASSWORD = 'mediguard123';
const LOGIN_FAILURE_TITLE = '登录失败';
const LOGIN_FAILURE_DESCRIPTION = '请确认审核人员账号、密码是否正确，且账号已开通审核权限。';

export default function LoginPage({ showcaseMode = false, onLogin }: LoginPageProps) {
  const [form] = Form.useForm<LoginInput>();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleFinish(values: LoginInput) {
    setSubmitting(true);
    setError(null);
    try {
      const response = await login({
        username: values.username.trim(),
        password: values.password,
      });
      onLogin(response.user);
    } catch {
      setError(LOGIN_FAILURE_TITLE);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="login-shell">
      <header className="login-page-header">
        <div className="login-header-inner">
          <div className="login-header-brand">
            <span className="login-header-mark">
              <MediGuardMark />
            </span>
            <span className="login-header-text">
              <strong>医保智能稽核服务平台</strong>
              <span className="login-header-english" aria-label="MediGuard">
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
          </div>
          <nav className="login-header-nav" aria-label="登录页导航">
            <a className="login-header-link" href="/">
              首页
            </a>
            <span className="login-header-link is-active" aria-current="page">
              登录
            </span>
          </nav>
        </div>
      </header>

      <section className="login-stage" aria-label="审核人员登录">
        <div className="login-brand-panel">
          <ProjectOverview showcaseMode={showcaseMode} />
          <div className="login-boundary-note">
            <SafetyCertificateOutlined />
            <div>
              <strong>演示边界与合规声明</strong>
              <span>{PROJECT_DISCLAIMER}</span>
            </div>
          </div>
        </div>

        <Card className="login-card">
          <Space orientation="vertical" size={18} style={{ width: '100%' }}>
            <div className="login-card-head">
              <Typography.Title level={2}>账号登录</Typography.Title>
              <Typography.Text>演示审核员账号登录</Typography.Text>
            </div>
            {error && (
              <Alert
                type="error"
                showIcon
                title={error}
                description={LOGIN_FAILURE_DESCRIPTION}
              />
            )}
            <Form
              form={form}
              initialValues={{
                username: DEMO_REVIEWER_USERNAME,
                password: DEMO_REVIEWER_PASSWORD,
              }}
              layout="vertical"
              requiredMark={false}
              onFinish={handleFinish}
            >
              <Form.Item
                label="账号"
                name="username"
                rules={[{ required: true, whitespace: true, message: '请输入审核人员账号' }]}
              >
                <Input
                  autoComplete="username"
                  prefix={<UserOutlined />}
                  placeholder={DEMO_REVIEWER_USERNAME}
                  size="large"
                />
              </Form.Item>
              <Form.Item
                label="密码"
                name="password"
                rules={[{ required: true, message: '请输入密码' }]}
              >
                <Input.Password
                  autoComplete="current-password"
                  prefix={<LockOutlined />}
                  placeholder="请输入密码"
                  size="large"
                />
              </Form.Item>
              <Button
                block
                type="primary"
                htmlType="submit"
                size="large"
                loading={submitting}
              >
                登录并进入系统
              </Button>
            </Form>
          </Space>
        </Card>
      </section>
    </main>
  );
}
