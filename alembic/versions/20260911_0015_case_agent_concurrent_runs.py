"""Add Case Agent concurrent-run lifecycle and read markers."""

from alembic import op


revision = "20260911_0015"
down_revision = "20260910_0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE case_agent_sessions "
        "ADD COLUMN IF NOT EXISTS last_read_at TIMESTAMPTZ"
    )
    op.execute(
        "ALTER TABLE case_agent_runs "
        "ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ, "
        "ADD COLUMN IF NOT EXISTS cancel_requested_at TIMESTAMPTZ, "
        "ADD COLUMN IF NOT EXISTS cancelled_at TIMESTAMPTZ, "
        "ADD COLUMN IF NOT EXISTS cancel_reason VARCHAR(80), "
        "ADD COLUMN IF NOT EXISTS client_request_id VARCHAR(120), "
        "ADD COLUMN IF NOT EXISTS case_context_fingerprint VARCHAR(80), "
        "ADD COLUMN IF NOT EXISTS lease_owner VARCHAR(120), "
        "ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ"
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_case_agent_runs_session_client_request "
        "ON case_agent_runs (session_id, client_request_id) "
        "WHERE client_request_id IS NOT NULL"
    )
    op.execute(
        "WITH ranked AS ("
        " SELECT id, ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY created_at DESC, id DESC) AS rn"
        " FROM case_agent_runs WHERE status IN ('created', 'running', 'resuming')"
        ") "
        "UPDATE case_agent_runs SET status = 'failed', current_node = 'migration_conflict', "
        "error_code = 'legacy_concurrent_run', completed_at = NOW() "
        "WHERE id IN (SELECT id FROM ranked WHERE rn > 1)"
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_case_agent_runs_one_active_per_session "
        "ON case_agent_runs (session_id) "
        "WHERE status IN ('created', 'running', 'resuming')"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_case_agent_runs_one_active_per_session")
    op.execute("DROP INDEX IF EXISTS uq_case_agent_runs_session_client_request")
    op.execute(
        "ALTER TABLE case_agent_runs "
        "DROP COLUMN IF EXISTS lease_expires_at, "
        "DROP COLUMN IF EXISTS lease_owner, "
        "DROP COLUMN IF EXISTS case_context_fingerprint, "
        "DROP COLUMN IF EXISTS client_request_id, "
        "DROP COLUMN IF EXISTS cancel_reason, "
        "DROP COLUMN IF EXISTS cancelled_at, "
        "DROP COLUMN IF EXISTS cancel_requested_at, "
        "DROP COLUMN IF EXISTS started_at"
    )
    op.execute("ALTER TABLE case_agent_sessions DROP COLUMN IF EXISTS last_read_at")
