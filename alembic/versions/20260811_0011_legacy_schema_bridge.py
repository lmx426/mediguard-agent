"""Compatibility bridge for the historical schema revision.

The original 20260811_0011 migration is not present in this checkout, but
local PostgreSQL databases may already be stamped at that revision. Keeping a
no-op bridge lets Alembic resolve the existing revision and apply the Case
Memory migration without pretending to recreate the historical schema.

The application still calls ``Base.metadata.create_all`` for the legacy
tables when bootstrapping a local database. A deployment that owns the full
historical migration tree should replace this bridge with that migration.
"""

from alembic import op


revision = "20260811_0011"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
