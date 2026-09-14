import urllib.error

import pytest

from conftest import FEED
from research_agent.arxiv import ArxivClient, ArxivError, RateLimitedError, RateLimiter, SnapshotMissingError, normalize_arxiv_id, parse_feed
from research_agent.corpus import fts_query


@pytest.mark.parametrize("raw, expected", [
    ("2210.03629", "2210.03629"),
    ("arXiv:2210.03629v3", "2210.03629"),
    ("https://arxiv.org/abs/1706.03762v7", "1706.03762"),
    ("hep-th/9901001", "hep-th/9901001"),
    ("math.GT/0309136v1", "math.GT/0309136"),
])
def test_normalize_accepts_real_identifier_formats(raw, expected):
    assert normalize_arxiv_id(raw) == expected


@pytest.mark.parametrize("raw", ["", "../etc/passwd", "2210.036", "12345.67890", "2210.03629; DROP TABLE papers", "cs.CL"])
def test_normalize_rejects_everything_else(raw):
    with pytest.raises(ValueError):
        normalize_arxiv_id(raw)


def test_parse_real_feed_fixture(fixture_papers):
    by_id = {p.arxiv_id: p for p in fixture_papers}
    react = by_id["2210.03629"]
    assert react.title == "ReAct: Synergizing Reasoning and Acting in Language Models"
    assert react.authors[0] == "Shunyu Yao" and len(react.authors) == 7
    assert (react.version, react.primary_category, react.published, react.year) == (3, "cs.CL", "2022-10-06", 2022)
    assert "HotpotQA" in react.abstract and "\n" not in react.abstract
    assert parse_feed(FEED)[1] == 2


def test_rate_limit_body_and_error_feed_are_errors():
    with pytest.raises(RateLimitedError):
        parse_feed(b"Rate exceeded.")
    error_feed = (b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/api/errors#incorrect_id_format</id>'
                  b"<title>Error</title><summary>incorrect id format for 1234</summary></entry></feed>")
    with pytest.raises(ArxivError, match="incorrect id format"):
        parse_feed(error_feed)


def test_rate_limiter_spaces_requests_three_seconds_apart():
    now, sleeps = [0.0], []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(3.0, clock=lambda: now[0], sleep=sleep)
    assert limiter.wait() == 0.0
    now[0] += 1.0
    assert limiter.wait() == pytest.approx(2.0)
    assert limiter.wait() == pytest.approx(3.0)
    assert sleeps == [pytest.approx(2.0), pytest.approx(3.0)]


def test_build_url_matches_the_arxiv_api_format(tmp_path):
    client = ArxivClient(tmp_path)
    assert client.build_url(id_list=["2210.03629", "1706.03762"], max_results=20) == \
        "https://export.arxiv.org/api/query?id_list=2210.03629,1706.03762&max_results=20"
    assert client.build_url(search_query='abs:"prompt injection"', max_results=5, sort_by="relevance") == \
        "https://export.arxiv.org/api/query?search_query=abs:%22prompt%20injection%22&max_results=5&sortBy=relevance"


def test_replay_mode_never_touches_the_network(tmp_path):
    client = ArxivClient(tmp_path, mode="replay", opener=lambda url: pytest.fail("network used in replay mode"))
    with pytest.raises(SnapshotMissingError):
        client.get_by_ids(["2210.03629"])


def test_record_then_replay_is_reproducible(tmp_path):
    calls = []
    recorder = ArxivClient(tmp_path, mode="record", opener=lambda url: calls.append(url) or FEED, limiter=RateLimiter(0))
    first = recorder.get_by_ids(["1706.03762", "2210.03629"])
    recorder.get_by_ids(["1706.03762", "2210.03629"])
    assert len(calls) == 1 and recorder.stats == {**recorder.stats, "live_requests": 1, "snapshot_hits": 1}
    replay = ArxivClient(tmp_path, mode="replay", opener=lambda url: pytest.fail("network used in replay mode"))
    assert replay.get_by_ids(["1706.03762", "2210.03629"]) == first
    manifest = recorder.store.manifest()
    assert manifest[0]["url"] == calls[0] and manifest[0]["bytes"] == len(FEED)


def test_rate_limits_and_server_errors_are_retried_with_backoff(tmp_path):
    responses = [b"Rate exceeded.", urllib.error.HTTPError("u", 503, "busy", None, None), FEED]

    def opener(url):
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    sleeps = []
    client = ArxivClient(tmp_path, opener=opener, limiter=RateLimiter(0), backoff_s=10, max_retries=3, sleep=sleeps.append)
    assert len(client.get_by_ids(["2210.03629"])) == 2
    assert sleeps == [10, 20] and client.stats["retries"] == 2


def test_gives_up_after_max_retries_without_saving_a_snapshot(tmp_path):
    client = ArxivClient(tmp_path, opener=lambda url: b"Rate exceeded.", limiter=RateLimiter(0), max_retries=2, sleep=lambda s: None)
    with pytest.raises(RateLimitedError):
        client.get_by_ids(["2210.03629"])
    assert client.stats["live_requests"] == 3 and client.store.manifest() == []


def test_client_errors_are_not_retried(tmp_path):
    calls = []

    def opener(url):
        calls.append(url)
        raise urllib.error.HTTPError(url, 400, "bad request", None, None)

    with pytest.raises(ArxivError, match="HTTP 400"):
        ArxivClient(tmp_path, opener=opener, limiter=RateLimiter(0)).get_by_ids(["2210.03629"])
    assert len(calls) == 1


def test_fts_query_quotes_terms_and_drops_stopwords():
    assert fts_query('the ReAct paper: "reasoning" OR acting*') == '"react" OR "reasoning" OR "acting"'
    assert fts_query("the of and") == ""


def test_index_search_and_filters(index):
    assert index.search("reasoning and acting")[0].arxiv_id == "2210.03629"
    assert [p.arxiv_id for p in index.search("", author="Vaswani")] == ["1706.03762"]
    assert [p.arxiv_id for p in index.search("", category="cs.CL", year_to=2018)] == ["1706.03762"]
    assert index.search("", year_from=2030) == []
    assert index.get("9999.99999") is None and index.count() == 2


def test_copy_to_leaves_the_original_untouched(index, tmp_path):
    import dataclasses

    copy = index.copy_to(tmp_path / "copy.sqlite")
    paper = copy.get("2210.03629")
    copy.upsert([dataclasses.replace(paper, abstract=paper.abstract + " INJECTED")], source="controlled-test")
    assert copy.get("2210.03629").abstract.endswith("INJECTED")
    assert not index.get("2210.03629").abstract.endswith("INJECTED")
