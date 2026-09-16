"""Add the CaseMemoryService truth ledger and projection tables."""

from alembic import op
import sqlalchemy as sa

revision = "20260904_0012"
down_revision = "20260811_0011"
branch_labels = None
depends_on = None


def _json() -> sa.JSON:
    return sa.JSON()


def upgrade() -> None:
    op.create_table(
        "case_memories",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("memory_id", sa.String(80), nullable=False),
        sa.Column("memory_type", sa.String(60), nullable=False),
        sa.Column("memory_level", sa.String(60), nullable=False),
        sa.Column("scope_type", sa.String(20), nullable=False),
        sa.Column("scope_id", sa.String(160), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("payload", _json(), nullable=False),
        sa.Column("allowed_consumers", _json(), nullable=False),
        sa.Column("consumer_view_policy_id", sa.String(100), nullable=False),
        sa.Column("admission_policy", sa.String(30), nullable=False),
        sa.Column("admission_confidence", sa.Float(), nullable=False),
        sa.Column("shadow_observation_count", sa.Integer(), nullable=False),
        sa.Column("importance_score", sa.Float(), nullable=False),
        sa.Column("freshness_score", sa.Float(), nullable=False),
        sa.Column("usage_count", sa.Integer(), nullable=False),
        sa.Column("success_count", sa.Integer(), nullable=False),
        sa.Column("conflict_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_verified_at", sa.DateTime(timezone=True)),
        sa.Column("last_used_at", sa.DateTime(timezone=True)),
        sa.Column("review_after", sa.DateTime(timezone=True)),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column("snoozed_until", sa.DateTime(timezone=True)),
        sa.Column("memo_memory_id", sa.String(120)),
        sa.Column("projection_sync_status", sa.String(30), nullable=False),
        sa.Column("supersedes_id", sa.String(80)),
        sa.Column("conflict_with_ids", _json(), nullable=False),
        sa.Column("source_event_ids", _json(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("memory_id"),
        if_not_exists=True,
    )
    op.create_index("ix_case_memories_recall", "case_memories", ["status", "scope_type", "scope_id", "memory_type", "memory_level"], if_not_exists=True)
    op.create_index("ix_case_memories_freshness", "case_memories", ["expires_at", "review_after"], if_not_exists=True)

    op.create_table(
        "memory_events",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("memory_id", sa.String(80), nullable=False),
        sa.Column("event_type", sa.String(80), nullable=False),
        sa.Column("aggregate_type", sa.String(40), nullable=False),
        sa.Column("actor_id", sa.String(80)),
        sa.Column("source_run_id", sa.String(120)),
        sa.Column("source_event_id", sa.String(120)),
        sa.Column("payload", _json(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )
    op.create_index("ix_memory_events_memory_created", "memory_events", ["memory_id", "created_at"], if_not_exists=True)

    op.create_table(
        "memory_outbox",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("event_type", sa.String(80), nullable=False),
        sa.Column("aggregate_type", sa.String(40), nullable=False),
        sa.Column("aggregate_id", sa.String(80), nullable=False),
        sa.Column("idempotency_key", sa.String(160), nullable=False),
        sa.Column("payload", _json(), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("locked_until", sa.DateTime(timezone=True)),
        sa.Column("last_error", sa.Text()),
        sa.Column("processed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("idempotency_key"),
        if_not_exists=True,
    )
    op.create_index("ix_memory_outbox_status_next", "memory_outbox", ["status", "next_attempt_at"], if_not_exists=True)

    op.create_table(
        "memory_embeddings",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("memory_id", sa.String(80), nullable=False),
        sa.Column("embedding_text_hash", sa.String(64), nullable=False),
        sa.Column("embedding_model", sa.String(160), nullable=False),
        sa.Column("embedding_dimension", sa.Integer(), nullable=False),
        sa.Column("embedding_vector", _json()),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("memory_id"),
        if_not_exists=True,
    )
    op.create_table(
        "case_memory_index",
        sa.Column("memory_id", sa.String(80), nullable=False),
        sa.Column("search_text", sa.Text(), nullable=False),
        sa.Column("search_vector", sa.Text()),
        sa.Column("last_indexed_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("memory_id"),
        if_not_exists=True,
    )

    op.create_table(
        "memory_graph_nodes",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("node_type", sa.String(60), nullable=False),
        sa.Column("node_key_hash", sa.String(128), nullable=False),
        sa.Column("display_label", sa.String(240), nullable=False),
        sa.Column("safe_attrs", _json(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("node_type", "node_key_hash"),
        if_not_exists=True,
    )
    op.create_table(
        "memory_graph_edges",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("source_node_id", sa.UUID(), nullable=False),
        sa.Column("target_node_id", sa.UUID(), nullable=False),
        sa.Column("edge_type", sa.String(60), nullable=False),
        sa.Column("memory_id", sa.String(80)),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("source", sa.String(60), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("supporting_refs", _json(), nullable=False),
        sa.Column("created_by_rule_or_llm", sa.String(60), nullable=False),
        sa.Column("last_verified_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["source_node_id"], ["memory_graph_nodes.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["target_node_id"], ["memory_graph_nodes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_node_id", "target_node_id", "edge_type"),
        if_not_exists=True,
    )
    op.create_index("ix_memory_graph_edges_memory", "memory_graph_edges", ["memory_id", "status"], if_not_exists=True)


def downgrade() -> None:
    op.drop_index("ix_memory_graph_edges_memory", table_name="memory_graph_edges")
    op.drop_table("memory_graph_edges")
    op.drop_table("memory_graph_nodes")
    op.drop_table("case_memory_index")
    op.drop_table("memory_embeddings")
    op.drop_index("ix_memory_outbox_status_next", table_name="memory_outbox")
    op.drop_table("memory_outbox")
    op.drop_index("ix_memory_events_memory_created", table_name="memory_events")
    op.drop_table("memory_events")
    op.drop_index("ix_case_memories_freshness", table_name="case_memories")
    op.drop_index("ix_case_memories_recall", table_name="case_memories")
    op.drop_table("case_memories")
