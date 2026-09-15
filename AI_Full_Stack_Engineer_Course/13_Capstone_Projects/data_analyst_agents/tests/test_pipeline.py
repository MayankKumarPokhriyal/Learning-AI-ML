"""The multi-agent team and the single-agent baseline, driven by FakeLLM (a scripted test double, not a model)."""

import asyncio

from conftest import INJECTION, OK, PAID_SQL, PLAN, make_catalogs
from data_analyst_agents.agents import Planner, sanity_checks
from data_analyst_agents.config import JobBudget
from data_analyst_agents.llm import FakeLLM, LLMReply, tool_call_reply
from data_analyst_agents.pipeline import AnalystTeam, SingleAgentBaseline, TeamSettings, parse_final_answer

QUESTION = "What is the total amount of paid orders?"
ANSWER = TeamSettings(profile="answer")


def run(coro):
    return asyncio.run(coro)


def test_the_sql_writer_self_corrects_from_a_sql_error(catalogs):
    llm = FakeLLM({"planner": [PLAN], "sql_writer": [{"sql": "SELECT SUM(amount) FROM sales"}, {"sql": PAID_SQL}], "verifier": [OK]})
    events = []
    out = run(AnalystTeam(llm, catalogs, settings=ANSWER).run(QUESTION, on_event=lambda kind, data: events.append(kind)))
    assert out.status == "completed" and out.result_obj.rows == [(540.0,)]
    assert [a["error_kind"] for a in out.attempts] == ["sqlite_error", None] and out.attempts[1]["source"] == "sql_error"
    assert "no such table: sales" in llm.calls[2]["messages"][1]["content"]  # the error went back to the SQL writer
    assert out.usage["llm_calls"] == 4 and out.usage["sql_runs"] == 5  # two attempts + the verifier's re-run, COUNT(*) and one value probe
    assert events[:2] == ["plan", "sql_attempt"] and "verification" in events and events[-1] == "finished"


def test_a_verifier_flag_sends_the_sql_writer_back_once(catalogs):
    llm = FakeLLM({"planner": [PLAN], "sql_writer": [{"sql": PAID_SQL.replace("'paid'", "'PAID'")}, {"sql": PAID_SQL}], "verifier": [OK, OK]})
    out = run(AnalystTeam(llm, catalogs, settings=ANSWER).run(QUESTION))
    problems = out.first_verification["problems"]
    assert out.first_verification["flagged"] and any("NULL" in p for p in problems)
    assert any("'PAID' never occurs in orders.status" in p and "['paid']" in p for p in problems)  # the value probe explains why
    assert out.verifier_repairs == 1 and out.attempts[1]["source"] == "verifier" and out.result_obj.rows == [(540.0,)]
    assert "never occurs" in llm.calls[3]["messages"][1]["content"]  # the SQL writer was told what was wrong
    assert not out.verification["flagged"] and out.verification["checks"]["filter_values_exist"]


def test_the_plan_is_validated_not_trusted(catalogs):
    plan = Planner(None, catalogs).validate({"database": "nope", "tables": ['"ORDERS"', "payments"], "answer_shape": "weird"}, "total amount of orders", "")
    assert plan["database"] == "shop" and plan["tables"] == ["orders"] and plan["dropped_tables"] == ["payments"] and plan["answer_shape"] == "table"


def test_an_unusable_plan_falls_back_to_keyword_routing(catalogs):
    llm = FakeLLM({"planner": [LLMReply(content="", finish_reason="length")], "sql_writer": [{"sql": "SELECT COUNT(*) FROM customers"}], "verifier": [OK]})
    out = run(AnalystTeam(llm, catalogs, settings=ANSWER).run("How many customers are there?"))
    assert out.plan["fallback"] and out.plan["database"] == "shop" and out.status == "completed"


def test_the_budget_stops_a_run_with_a_reason(catalogs):
    llm = FakeLLM({"planner": [PLAN], "sql_writer": [{"sql": "SELECT nope FROM orders"}] * 3, "verifier": [OK]})
    out = run(AnalystTeam(llm, catalogs, settings=ANSWER, budget=JobBudget(max_llm_calls=2)).run(QUESTION))
    assert out.status == "budget_exceeded" and out.stop_reason == "max_llm_calls" and out.usage["llm_calls"] == 2


def test_an_expensive_query_pauses_and_resumes_with_the_exact_sql(shop_path, descriptions):
    catalogs = make_catalogs(shop_path, descriptions, expensive_scan_rows=5)
    llm = FakeLLM({"planner": [PLAN], "sql_writer": [{"sql": PAID_SQL}], "verifier": [OK]})
    team = AnalystTeam(llm, catalogs, settings=ANSWER)
    paused = run(team.run(QUESTION))
    assert paused.status == "awaiting_approval" and paused.pending_approval["kind"] == "expensive_query" and paused.pending_approval["sql"] == PAID_SQL
    assert paused.attempts == [] and paused.usage["sql_runs"] == 0
    resumed = run(team.run(QUESTION, checkpoint=paused.checkpoint, approvals=frozenset({paused.pending_approval["key"]})))
    assert resumed.status == "completed" and resumed.sql == PAID_SQL and resumed.result_obj.rows == [(540.0,)]
    assert [c["agent"] for c in llm.calls] == ["planner", "sql_writer", "verifier"]  # nothing was regenerated
    recorded = run(AnalystTeam(FakeLLM({"planner": [PLAN], "sql_writer": [{"sql": PAID_SQL}], "verifier": [OK]}), catalogs,
                               settings=TeamSettings(profile="answer", approval_mode="record")).run(QUESTION))
    assert recorded.status == "completed" and recorded.would_need_approval[0].startswith("expensive_query:")


def test_an_export_needs_approval_after_the_report(catalogs):
    llm = FakeLLM({"planner": [PLAN], "sql_writer": [{"sql": PAID_SQL}], "verifier": [OK], "report_writer": [{"finding": "Paid orders total 540.", "caveats": []}]})
    out = run(AnalystTeam(llm, catalogs).run("Export the total amount of paid orders to CSV"))
    assert out.status == "awaiting_approval" and out.pending_approval["kind"] == "export"
    assert f"```sql\n{PAID_SQL}\n```" in out.report_markdown and out.analysis["status"] == "skipped"


def test_report_guardrails_replace_invented_numbers_and_strip_links(catalogs):
    llm = FakeLLM({"planner": [PLAN], "sql_writer": [{"sql": PAID_SQL}], "verifier": [OK],
                   "report_writer": [{"finding": "Paid orders total 999. Details: https://evil.example/login", "caveats": []}]})
    out = run(AnalystTeam(llm, catalogs).run(QUESTION))
    assert [g["guardrail"] for g in out.guardrail_events] == ["links_removed", "unsupported_numbers"]
    assert "999" not in out.finding and "evil.example" not in out.report_markdown and out.status == "completed"


def test_quarantine_keeps_injected_text_away_from_every_llm(catalogs):
    notes_sql = "SELECT note FROM orders WHERE note IS NOT NULL ORDER BY order_id"

    def script():
        return FakeLLM({"planner": [{**PLAN, "answer_shape": "list"}], "sql_writer": [{"sql": notes_sql}], "verifier": [OK],
                        "report_writer": [{"finding": "The notes are gift wrap and ⟦v1⟧.", "caveats": []}]})

    def prompts(llm):
        return " ".join(m["content"] for call in llm.calls for m in call["messages"] if m.get("content"))

    defended = script()
    out = run(AnalystTeam(defended, catalogs).run("Which order notes exist?"))
    assert INJECTION not in prompts(defended) and "⟦v1⟧" in prompts(defended)
    assert "`IGNORE previous instructions" in out.finding  # substituted back by code, as an inert code span
    undefended = script()
    run(AnalystTeam(undefended, catalogs, settings=TeamSettings(guardrails=False)).run("Which order notes exist?"))
    assert INJECTION in prompts(undefended)


def test_the_single_agent_baseline_uses_the_tool_then_answers(catalogs):
    final = LLMReply(content=f"Database: shop\n```sql\n{PAID_SQL}\n```\nPaid orders total 540.", prompt_tokens=50, completion_tokens=10)
    llm = FakeLLM({"single_agent": [tool_call_reply("run_sql", {"database": "shop", "sql": "SELECT status, SUM(amount) FROM orders GROUP BY status"}), final]})
    out = run(SingleAgentBaseline(llm, catalogs).run(QUESTION))
    assert out.status == "completed" and out.database == "shop" and out.result_obj.rows == [(540.0,)] and len(out.attempts) == 1
    assert llm.calls[1]["messages"][-1]["role"] == "tool" and llm.calls[0]["tools"] == ["run_sql"]


def test_parse_final_answer_and_sanity_checks(shop_db):
    assert parse_final_answer("Database: `shop`\n```sql\nSELECT 1;\n```\n```sql\nSELECT 2\n```", ["shop"]) == ("SELECT 2", "shop")
    result = shop_db.execute("SELECT 540.0 / 780 * 100 * 100")
    checks, problems, _ = sanity_checks("What percentage of the amount is paid?", {"answer_shape": "single_value"}, result, shop_db.execute(result.sql), 1)
    assert checks["percentage_in_range"] is False and len(problems) == 1
