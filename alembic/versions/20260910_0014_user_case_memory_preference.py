"""Add the reviewer-level cross-case memory preference."""

from alembic import op


revision = "20260910_0014"
down_revision = "20260909_0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE users "
        "ADD COLUMN IF NOT EXISTS case_memory_enabled BOOLEAN NOT NULL DEFAULT TRUE"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS case_memory_enabled")
