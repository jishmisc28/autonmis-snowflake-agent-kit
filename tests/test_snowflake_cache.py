"""
test_snowflake_cache.py — Unit and integration tests for autonmis-snowflake-agent-kit.
"""

import unittest
from caching_wrapper import SnowflakePromptCacheEngine, SnowflakeTableSchema
from agent import SnowflakeConnectionPool, AutonmisSnowflakeAgent


class TestSnowflakeConnectionPool(unittest.TestCase):
    def setUp(self):
        self.pool = SnowflakeConnectionPool(max_connections=3)
        self.config_a = {
            "account": "xy12345.us-east-1",
            "user": "AUTONMIS_AGENT",
            "database": "ANALYTICS_PROD",
            "schema": "CORE",
            "warehouse": "QUERY_WH",
        }
        self.config_b = {
            "account": "xy12345.us-east-1",
            "user": "AUTONMIS_AGENT",
            "database": "FINANCE_PROD",
            "schema": "LEDGER",
            "warehouse": "FIN_WH",
        }

    def tearDown(self):
        self.pool.destroy_all()

    def test_pool_keying_deterministic(self):
        key1 = self.pool._compute_key(self.config_a)
        key2 = self.pool._compute_key(self.config_a.copy())
        self.assertEqual(key1, key2)

        key_diff = self.pool._compute_key(self.config_b)
        self.assertNotEqual(key1, key_diff)

    def test_connection_reuse(self):
        conn1 = self.pool.acquire(self.config_a)
        conn1_id = conn1.connection_id
        self.pool.release(self.config_a, conn1)

        conn2 = self.pool.acquire(self.config_a)
        self.assertEqual(conn1_id, conn2.connection_id)
        self.pool.release(self.config_a, conn2)

    def test_pool_concurrency_ceiling(self):
        conns = [self.pool.acquire(self.config_a) for _ in range(3)]
        self.assertEqual(len(conns), 3)
        self.assertEqual(self.pool._pool_counts[self.pool._compute_key(self.config_a)], 3)
        for c in conns:
            self.pool.release(self.config_a, c)


class TestPromptCacheEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SnowflakePromptCacheEngine()
        self.tbl_orders = SnowflakeTableSchema(
            database="ECOM_DB",
            schema="PUBLIC",
            name="ORDERS",
            columns=(("ORDER_ID", "VARCHAR(32)"), ("TOTAL_AMOUNT", "NUMBER(10,2)")),
        )
        self.tbl_customers = SnowflakeTableSchema(
            database="ECOM_DB",
            schema="PUBLIC",
            name="CUSTOMERS",
            columns=(("CUST_ID", "VARCHAR(32)"), ("NAME", "VARCHAR(128)")),
        )

    def test_canonical_sorting_guarantees_hash_stability(self):
        # Register in order: ORDERS then CUSTOMERS
        self.engine.register_table(self.tbl_orders)
        self.engine.register_table(self.tbl_customers)
        content_1, hash_1 = self.engine.render_immutable_prefix()

        # New engine with reverse insertion order: CUSTOMERS then ORDERS
        engine_rev = SnowflakePromptCacheEngine()
        engine_rev.register_table(self.tbl_customers)
        engine_rev.register_table(self.tbl_orders)
        content_2, hash_2 = engine_rev.render_immutable_prefix()

        self.assertEqual(content_1, content_2)
        self.assertEqual(hash_1, hash_2)

    def test_volatile_tail_does_not_bust_prefix(self):
        self.engine.register_table(self.tbl_orders)
        _, initial_hash = self.engine.render_immutable_prefix()

        turn1 = self.engine.assemble_payload([], "Show me high-value orders", "sess-1")
        self.assertEqual(turn1["prefix_hash"], initial_hash)
        self.assertFalse(turn1["is_cache_hit"])  # First turn is cache creation (miss)

        # Second turn with different timestamp and query
        turn2 = self.engine.assemble_payload(
            [{"role": "user", "content": "Show me high-value orders"}],
            "Filter for orders over $500",
            "sess-1",
        )
        self.assertEqual(turn2["prefix_hash"], initial_hash)
        self.assertTrue(turn2["is_cache_hit"])  # Second turn hits cache!

    def test_multi_turn_amortization_savings(self):
        self.engine.register_table(self.tbl_orders)
        self.engine.register_table(self.tbl_customers)

        for i in range(10):
            self.engine.assemble_payload([], f"Query turn {i}", "sess-amort")

        self.assertEqual(self.engine.metrics.total_turns, 10)
        self.assertEqual(self.engine.metrics.cache_hits, 9)
        self.assertEqual(self.engine.metrics.cache_misses, 1)
        self.assertGreater(self.engine.metrics.cost_savings_pct, 50.0)


class TestAutonmisSnowflakeAgent(unittest.TestCase):
    def test_end_to_end_agent_execution(self):
        config = {
            "account": "test_acct",
            "user": "test_user",
            "database": "FINANCE_DB",
            "schema": "LEDGER",
            "warehouse": "COMPUTE_WH",
        }
        agent = AutonmisSnowflakeAgent(snowflake_config=config)
        agent.bootstrap_catalog()

        # Turn 1
        res1 = agent.run_turn("Find all un-reconciled debit transactions", "sess-42")
        self.assertEqual(res1["row_count"], 2)
        self.assertFalse(res1["cache_hit"])

        # Turn 2
        res2 = agent.run_turn("Group by account and order descending", "sess-42")
        self.assertEqual(res2["row_count"], 2)
        self.assertTrue(res2["cache_hit"])
        self.assertEqual(res2["metrics"]["cache_hits"], 1)


if __name__ == "__main__":
    unittest.main()
