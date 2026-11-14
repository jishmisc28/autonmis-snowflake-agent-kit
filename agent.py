"""
agent.py — Production-Grade Snowflake Text-to-SQL Agent with Connection Pooling.

Directly adapted from Autonmis production architecture (snowflakeDirectExecutor.ts),
featuring cryptographic connection pool hashing, resource concurrency throttling,
and zero-leak stream iteration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, Generator, List, Optional

from caching_wrapper import SnowflakePromptCacheEngine, SnowflakeTableSchema

logger = logging.getLogger("autonmis.snowflake.agent")


class MockSnowflakeConnection:
    """Simulates a live snowflake-sdk or snowflake-connector-python connection."""

    def __init__(self, connection_id: str, account: str, warehouse: str):
        self.connection_id = connection_id
        self.account = account
        self.warehouse = warehouse
        self.is_closed = False

    def execute_stream(self, sql: str) -> Generator[Dict[str, Any], None, None]:
        """Simulates statement.streamRows() with backpressure."""
        if self.is_closed:
            raise RuntimeError("Cannot execute on closed Snowflake connection.")
        # Simulated streaming records
        yield {"STATUS": "SUCCESS", "QUERY_SQL": sql, "ROW_ID": 1, "VALUE": 42.50}
        yield {"STATUS": "SUCCESS", "QUERY_SQL": sql, "ROW_ID": 2, "VALUE": 189.90}

    def close(self) -> None:
        self.is_closed = True


class SnowflakeConnectionPool:
    """
    Thread-safe connection pool for Snowflake sessions.
    
    Ported from Autonmis `lib/data-plane/executors/snowflakeDirectExecutor.ts`.
    Uses SHA-256 credential hashing to pool connections safely across multi-tenant
    agent invocations, capping warehouse concurrency to prevent credit burn.
    """

    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self._pools: Dict[str, queue.Queue[MockSnowflakeConnection]] = {}
        self._pool_counts: Dict[str, int] = {}
        self._lock = threading.Lock()

    def _compute_key(self, config: Dict[str, Any]) -> str:
        payload = {
            "account": config.get("account", "").lower(),
            "user": config.get("user", "").lower(),
            "database": config.get("database", "").upper(),
            "schema": config.get("schema", "PUBLIC").upper(),
            "warehouse": config.get("warehouse", "").upper(),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def acquire(self, config: Dict[str, Any]) -> MockSnowflakeConnection:
        key = self._compute_key(config)
        with self._lock:
            if key not in self._pools:
                self._pools[key] = queue.Queue(maxsize=self.max_connections)
                self._pool_counts[key] = 0

        pool = self._pools[key]
        try:
            return pool.get_nowait()
        except queue.Empty:
            with self._lock:
                if self._pool_counts[key] < self.max_connections:
                    self._pool_counts[key] += 1
                    conn_id = f"conn_{key[:8]}_{self._pool_counts[key]}"
                    return MockSnowflakeConnection(
                        connection_id=conn_id,
                        account=config.get("account", ""),
                        warehouse=config.get("warehouse", "COMPUTE_WH"),
                    )
            # If max reached, wait for an available connection
            return pool.get(timeout=10.0)

    def release(self, config: Dict[str, Any], conn: MockSnowflakeConnection) -> None:
        key = self._compute_key(config)
        if key in self._pools and not conn.is_closed:
            try:
                self._pools[key].put_nowait(conn)
            except queue.Full:
                conn.close()

    def destroy_all(self) -> None:
        with self._lock:
            for key, q in self._pools.items():
                while not q.empty():
                    try:
                        conn = q.get_nowait()
                        conn.close()
                    except queue.Empty:
                        break
            self._pools.clear()
            self._pool_counts.clear()


class AutonmisSnowflakeAgent:
    """
    Autonomous Text-to-SQL Agent backed by Snowflake connection pooling
    and Hearth prompt cache stabilization.
    """

    def __init__(
        self,
        snowflake_config: Dict[str, Any],
        pool: Optional[SnowflakeConnectionPool] = None,
        cache_engine: Optional[SnowflakePromptCacheEngine] = None,
    ):
        self.config = snowflake_config
        self.pool = pool or SnowflakeConnectionPool(max_connections=10)
        self.cache_engine = cache_engine or SnowflakePromptCacheEngine()
        self.history: List[Dict[str, str]] = []

    def bootstrap_catalog(self, mock_tables: Optional[List[SnowflakeTableSchema]] = None) -> None:
        """Loads enterprise data catalog definitions into the prompt cache engine."""
        if mock_tables:
            for tbl in mock_tables:
                self.cache_engine.register_table(tbl)
        else:
            # Default schema: Financial ledger & ERP orders
            self.cache_engine.register_table(
                SnowflakeTableSchema(
                    database=self.config.get("database", "FINANCE_DB"),
                    schema=self.config.get("schema", "LEDGER"),
                    name="GENERAL_JOURNAL",
                    columns=(
                        ("ENTRY_ID", "VARCHAR(64)"),
                        ("POSTING_DATE", "TIMESTAMP_NTZ"),
                        ("ACCOUNT_CODE", "VARCHAR(32)"),
                        ("DEBIT_AMOUNT", "NUMBER(18,4)"),
                        ("CREDIT_AMOUNT", "NUMBER(18,4)"),
                        ("CURRENCY", "VARCHAR(3)"),
                        ("RECONCILIATION_STATUS", "VARCHAR(16)"),
                    ),
                    comment="Core double-entry journal entries for corporate balance sheet.",
                )
            )
            self.cache_engine.register_table(
                SnowflakeTableSchema(
                    database=self.config.get("database", "FINANCE_DB"),
                    schema=self.config.get("schema", "LEDGER"),
                    name="PAYMENT_SETTLEMENTS",
                    columns=(
                        ("SETTLEMENT_ID", "VARCHAR(64)"),
                        ("MERCHANT_ID", "VARCHAR(64)"),
                        ("BATCH_DATE", "DATE"),
                        ("GROSS_AMOUNT", "NUMBER(18,2)"),
                        ("NET_FEE_AMOUNT", "NUMBER(18,2)"),
                        ("PAYOUT_STATUS", "VARCHAR(16)"),
                    ),
                    comment="Daily payment gateway settlement batches.",
                )
            )

    def run_turn(self, user_prompt: str, session_id: str) -> Dict[str, Any]:
        """
        Executes an agent reasoning turn:
        1. Formulates the prompt with immutable prefix caching.
        2. Simulates SQL synthesis.
        3. Executes against Snowflake connection pool via streaming.
        4. Updates conversation history and metrics.
        """
        # 1. Assemble payload with warm prefix
        payload = self.cache_engine.assemble_payload(
            turn_history=self.history,
            current_query=user_prompt,
            session_id=session_id,
        )

        # 2. Simulated deterministic SQL generation
        generated_sql = (
            f"SELECT ACCOUNT_CODE, SUM(DEBIT_AMOUNT) AS TOTAL_DEBIT "
            f"FROM {self.config.get('database', 'FINANCE_DB')}.{self.config.get('schema', 'LEDGER')}.GENERAL_JOURNAL "
            f"WHERE RECONCILIATION_STATUS = 'PENDING' "
            f"GROUP BY ACCOUNT_CODE ORDER BY TOTAL_DEBIT DESC LIMIT 10;"
        )

        # 3. Connection acquisition & streaming query execution
        conn = self.pool.acquire(self.config)
        records: List[Dict[str, Any]] = []
        try:
            for row in conn.execute_stream(generated_sql):
                records.append(row)
        finally:
            self.pool.release(self.config, conn)

        # 4. History preservation
        self.history.append({"role": "user", "content": user_prompt})
        self.history.append({"role": "assistant", "content": f"Executed SQL: {generated_sql}"})

        return {
            "sql": generated_sql,
            "row_count": len(records),
            "rows": records,
            "cache_hit": payload["is_cache_hit"],
            "prefix_hash": payload["prefix_hash"],
            "metrics": {
                "total_turns": self.cache_engine.metrics.total_turns,
                "cache_hits": self.cache_engine.metrics.cache_hits,
                "cost_savings_pct": self.cache_engine.metrics.cost_savings_pct,
            },
        }
