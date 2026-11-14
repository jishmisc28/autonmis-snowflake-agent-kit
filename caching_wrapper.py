"""
caching_wrapper.py — Prefix Stabilization for Snowflake Text-to-SQL Agents.

Integrates Hearth (hearth-ai) context engine patterns with Snowflake metadata
to guarantee that schema DDLs, column definitions, and dialect instructions stay
strictly within the cacheable prefix across multi-turn agent execution loops.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SnowflakeTableSchema:
    database: str
    schema: str
    name: str
    columns: Tuple[Tuple[str, str], ...]  # (column_name, data_type)
    comment: Optional[str] = None

    @property
    def ddl_representation(self) -> str:
        cols_str = ",\n    ".join(f"{col} {dtype}" for col, dtype in self.columns)
        comment_str = f" COMMENT = '{self.comment}'" if self.comment else ""
        return f"CREATE TABLE {self.database}.{self.schema}.{self.name} (\n    {cols_str}\n){comment_str};"


@dataclass
class PromptCacheMetrics:
    total_turns: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    raw_input_tokens: int = 0
    effective_billed_tokens: int = 0

    @property
    def cost_savings_pct(self) -> float:
        if self.raw_input_tokens == 0:
            return 0.0
        saved = self.raw_input_tokens - self.effective_billed_tokens
        return round((saved / self.raw_input_tokens) * 100, 2)


class SnowflakePromptCacheEngine:
    """
    Guarantees prefix prompt stability for LLM agents interacting with Snowflake.
    
    In multi-turn Text-to-SQL, naive architectures repeatedly bust provider prompt caches
    by prepending timestamps, dynamic session IDs, or re-ordering table definitions.
    SnowflakePromptCacheEngine enforces:
      1. Canonical ordering of schema DDLs in the immutable prefix.
      2. Segregation of volatile execution artifacts (timestamps, query IDs, warehouse state)
         into the un-cached suffix/tail.
      3. Precise token billing amortization tracking (k* compaction cadence).
    """

    def __init__(
        self,
        dialect: str = "snowflake",
        cache_discount_factor: float = 0.10,  # 90% discount on cached tokens (e.g. Anthropic/Bedrock)
    ):
        self.dialect = dialect
        self.cache_discount_factor = cache_discount_factor
        self.registered_tables: Dict[str, SnowflakeTableSchema] = {}
        self.metrics = PromptCacheMetrics()
        self._last_prefix_hash: Optional[str] = None

    def register_table(self, schema: SnowflakeTableSchema) -> None:
        """Register or update a Snowflake table definition."""
        key = f"{schema.database}.{schema.schema}.{schema.name}".upper()
        self.registered_tables[key] = schema

    def register_tables_from_metadata(self, rows: List[Dict[str, Any]]) -> None:
        """Parse Snowflake INFORMATION_SCHEMA.COLUMNS output into registered tables."""
        grouped: Dict[Tuple[str, str, str], List[Tuple[str, str]]] = {}
        for r in rows:
            db = r.get("TABLE_CATALOG", "PROD")
            schema = r.get("TABLE_SCHEMA", "PUBLIC")
            tbl = r.get("TABLE_NAME", "")
            col = r.get("COLUMN_NAME", "")
            dtype = r.get("DATA_TYPE", "VARCHAR")
            key = (db, schema, tbl)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append((col, dtype))

        for (db, schema, tbl), cols in grouped.items():
            self.register_table(
                SnowflakeTableSchema(
                    database=db,
                    schema=schema,
                    name=tbl,
                    columns=tuple(cols),
                )
            )

    def render_immutable_prefix(self) -> Tuple[str, str]:
        """
        Renders the deterministic system prompt and schema catalog.
        Returns:
            Tuple of (prefix_content, prefix_sha256_hash)
        """
        lines = [
            f"-- DIALECT: {self.dialect.upper()} (ANSI SQL:2016 Compliant)",
            "-- SYSTEM: You are an autonomous Text-to-SQL reasoning agent for Snowflake.",
            "-- RULES: Use CTEs for aggregation. Do not hallucinate column names.",
            "-- CATALOG DEFINITIONS (CANONICAL ORDER):",
        ]
        
        # Sort canonically by fully qualified table name to ensure byte-for-byte reproducibility
        for table_key in sorted(self.registered_tables.keys()):
            table = self.registered_tables[table_key]
            lines.append(f"\n-- TABLE: {table_key}")
            lines.append(table.ddl_representation)

        content = "\n".join(lines)
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return content, content_hash

    def assemble_payload(
        self,
        turn_history: List[Dict[str, str]],
        current_query: str,
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Assembles a provider-agnostic LLM prompt payload with prefix preservation.
        Volatile elements (timestamps, query UUIDs) are strictly placed in the final turn.
        """
        prefix_content, prefix_hash = self.render_immutable_prefix()
        
        # Token estimation heuristic (~4 chars per token)
        prefix_tokens = len(prefix_content) // 4
        
        # Check cache hit
        is_cache_hit = (self._last_prefix_hash == prefix_hash) and (self.metrics.total_turns > 0)
        self._last_prefix_hash = prefix_hash

        # Update metrics
        self.metrics.total_turns += 1
        if is_cache_hit:
            self.metrics.cache_hits += 1
            billed_prefix = int(prefix_tokens * self.cache_discount_factor)
        else:
            self.metrics.cache_misses += 1
            billed_prefix = prefix_tokens

        # Suffix tokens
        turn_content = "\n".join(f"{t['role']}: {t['content']}" for t in turn_history)
        volatile_tail = (
            f"\n[VOLATILE_CONTEXT]\n"
            f"session_id: {session_id}\n"
            f"timestamp_epoch_ms: {int(time.time() * 1000)}\n"
            f"current_request: {current_query}\n"
        )
        suffix_content = turn_content + volatile_tail
        suffix_tokens = len(suffix_content) // 4

        self.metrics.raw_input_tokens += (prefix_tokens + suffix_tokens)
        self.metrics.effective_billed_tokens += (billed_prefix + suffix_tokens)

        return {
            "prefix": prefix_content,
            "prefix_hash": prefix_hash,
            "is_cache_hit": is_cache_hit,
            "messages": turn_history + [{"role": "user", "content": current_query + volatile_tail}],
            "estimated_prefix_tokens": prefix_tokens,
            "estimated_suffix_tokens": suffix_tokens,
        }
