"""Add structured cross-turn task state to Case Agent sessions."""

from alembic import op


revision = "20260909_0013"
down_revision = "20260904_0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE case_agent_sessions "
        "ADD COLUMN IF NOT EXISTS task_state JSONB NOT NULL DEFAULT '{}'::jsonb"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE case_agent_sessions DROP COLUMN IF EXISTS task_state")
