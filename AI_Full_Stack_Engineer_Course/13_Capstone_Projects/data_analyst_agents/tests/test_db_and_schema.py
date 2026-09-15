"""The read-only database layer and the schema catalog."""

import math
import sqlite3

import pytest

from conftest import INJECTION, make_db
from data_analyst_agents.schema import BM25, read_descriptions, tokenize


def test_select_returns_columns_and_rows(shop_db):
    result = shop_db.execute("SELECT status, COUNT(*) FROM orders GROUP BY status ORDER BY status")
    assert result.ok and result.columns == ["status", "COUNT(*)"] and result.rows == [("paid", 9), ("refunded", 3)]


@pytest.mark.parametrize("sql", [
    "INSERT INTO orders (order_id) VALUES (99)", "UPDATE orders SET amount = 0", "DELETE FROM orders", "DROP TABLE orders",
    "CREATE TABLE t (a)", "ATTACH DATABASE ':memory:' AS m", "PRAGMA table_info(orders)", "WITH x AS (SELECT 1) DELETE FROM orders",
    "SELECT 1; DROP TABLE orders", "SELECT load_extension('evil')", "CREATE TEMP TABLE t AS SELECT * FROM orders",
])
def test_everything_but_reads_is_refused(shop_db, shop_path, sql):
    result = shop_db.execute(sql)
    assert not result.ok and result.error_kind in {"rejected", "not_authorized"}, result.error
    assert sqlite3.connect(shop_path).execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 12


def test_tables_outside_the_allowlist_are_invisible(shop_db):
    assert "secrets" not in shop_db.tables
    for sql in ["SELECT token FROM secrets", "SELECT name FROM sqlite_master"]:
        assert shop_db.execute(sql).error_kind == "not_authorized"


def test_pii_columns_read_as_null_even_in_filters(shop_db):
    result = shop_db.execute("SELECT name, email FROM customers ORDER BY customer_id")
    assert [row[1] for row in result.rows] == [None, None, None] and result.masked_columns == ["customers.email"]
    assert shop_db.execute("SELECT COUNT(*) FROM customers WHERE email LIKE '%@%'").rows == [(0,)]


def test_the_file_is_read_only_even_without_the_authorizer(shop_db):
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        shop_db._connect().execute("INSERT INTO orders (order_id) VALUES (100)")


def test_row_limit_and_timeout(shop_path):
    db = make_db(shop_path, max_rows=5, timeout_s=0.2)
    rows = db.execute("SELECT * FROM orders")
    assert len(rows.rows) == 5 and rows.truncated
    slow = db.execute("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) SELECT COUNT(*) FROM n")
    assert slow.error_kind == "timeout" and slow.elapsed_s < 2


def test_ctes_work_but_cannot_shadow_a_hidden_table(shop_db):
    grouped = shop_db.execute("WITH t AS (SELECT status, amount FROM orders) SELECT status, SUM(amount) FROM t GROUP BY status ORDER BY status")
    assert grouped.rows == [("paid", 540.0), ("refunded", 240.0)]
    assert shop_db.execute("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n WHERE x < 5) SELECT SUM(x) FROM n").rows == [(15,)]
    assert shop_db.execute("WITH x AS (SELECT token FROM secrets) SELECT token FROM x").error_kind == "not_authorized"  # hidden table inside a CTE
    shadow = shop_db.execute("WITH secrets AS (SELECT 1 AS token) SELECT token FROM secrets")  # a CTE named like the hidden table …
    assert shadow.rows == [(1,)] and "not for agents" not in str(shadow.rows)  # … reads the CTE, never the real table


def test_cost_estimate_flags_full_scans(shop_path):
    db = make_db(shop_path, expensive_scan_rows=2)
    cost = db.estimate_cost("SELECT COUNT(*) FROM orders AS o JOIN customers c ON o.customer_id = c.customer_id")
    assert cost["expensive"] and cost["scanned_rows"] >= 3 and cost["plan"]
    assert not db.estimate_cost("SELECT * FROM customers WHERE customer_id = 1")["expensive"]


def test_count_rows_and_a_reserved_word_table(shop_db):
    assert shop_db.count_rows("SELECT * FROM orders WHERE status = 'paid'") == 9
    assert shop_db.execute('SELECT COUNT(*) FROM "order"').rows == [(1,)]


def test_tokenize_splits_camel_and_snake_case():
    assert tokenize("GasStationID and link_to_member Stations") == ["gas", "station", "id", "link", "member", "station"]


def test_bm25_matches_the_formula_by_hand():
    docs = [["loan", "amount"], ["loan", "status", "loan"], ["district", "name"]]
    bm25 = BM25(docs, k1=1.5, b=0.75)
    idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    expected = idf * 2 * 2.5 / (2 + 1.5 * (1 - 0.75 + 0.75 * 3 / (7 / 3)))
    scores = bm25.scores(["loan"])
    assert math.isclose(scores[1], expected) and scores[2] == 0 and scores[1] > scores[0]


def test_descriptions_decode_utf8_bom_and_windows_1252(descriptions):
    parsed = read_descriptions(descriptions)
    assert parsed["orders"]["amount"]["description"] == "order value in € (euro)"
    assert parsed["customers"]["segment"]["values"] == "SME: small business, KAM: key account"


def test_render_shows_meaning_samples_masking_and_drops_injected_samples(catalogs):
    catalog = catalogs["shop"]
    text = catalog.render()
    assert "order value in €" in text and "'paid'" in text and "[MASKED personal data" in text
    assert "example.com" not in text and "secrets" not in text and 'a reserved word: write "order"' in text
    assert INJECTION not in text and catalog.dropped_samples == 1


def test_retrieval_and_bridge_tables(catalogs):
    catalog = catalogs["shop"]
    assert catalog.retrieve("total order amount by payment status")[0][0] == "orders"
    assert catalog.connect_tables(["customers", "refunds"]) == ["customers", "refunds", "orders"]
    assert catalog.resolve('"ORDERS"') == "orders" and catalog.resolve("payments") is None
