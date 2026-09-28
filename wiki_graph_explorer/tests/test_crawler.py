"""Unit tests for the MediaWiki client and the BFS crawler.

All HTTP traffic is mocked: a :class:`FakeSession` replays canned MediaWiki
payloads, so the suite never touches the network and is fully deterministic.
"""

from __future__ import annotations

import json

import pytest

import crawler.wiki_api as wiki_api
from crawler.bfs import build_crawl, node_budget_for_depth
from crawler.wiki_api import (
    ArticleRef,
    WikiApiClient,
    WikiApiError,
    cache_key,
    normalise_title,
    parse_article_input,
)

FIXTURE_LINKS: dict[str, list[str]] = {
    "Root": ["Charlie", "Alpha", "Bravo", "Alpha (disambiguation)"],
    "Alpha": ["Delta", "Echo"],
    "Bravo": ["Charlie"],
    "Charlie": ["Delta"],
    "Delta": ["Foxtrot"],
    "Echo": ["Foxtrot"],
    "Foxtrot": ["Root"],
}


def _links_payload(title: str, links: list[str]) -> dict:
    """Build a minimal ``prop=links`` response.

    Parameters
    ----------
    title
        Article title of the page.
    links
        Out-link titles returned by the API.

    Returns
    -------
    dict
        JSON payload shaped like the real MediaWiki answer.
    """
    return {
        "query": {
            "pages": {
                "1": {
                    "pageid": 1,
                    "ns": 0,
                    "title": title,
                    "links": [{"ns": 0, "title": link} for link in links],
                }
            }
        }
    }


def _missing_payload(title: str) -> dict:
    """Build a response for a non-existing article.

    Parameters
    ----------
    title
        Article title that does not exist.

    Returns
    -------
    dict
        JSON payload with a ``missing`` page.
    """
    return {"query": {"pages": {"-1": {"ns": 0, "title": title, "missing": ""}}}}


class FakeResponse:
    """Canned HTTP response.

    Parameters
    ----------
    payload
        Decoded JSON body.
    status_code
        HTTP status code.
    headers
        Response headers.
    """

    def __init__(
        self, payload: dict, status_code: int = 200, headers: dict | None = None
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def json(self) -> dict:
        """Return the decoded body.

        Returns
        -------
        dict
            The canned payload.
        """
        return self.payload


class FakeSession:
    """Minimal ``requests.Session`` stand-in driven by :data:`FIXTURE_LINKS`.

    Parameters
    ----------
    links
        Article -> out-links mapping used to answer ``prop=links``.
    errors
        Number of leading requests that must fail with HTTP 429.
    """

    def __init__(
        self, links: dict[str, list[str]] | None = None, errors: int = 0
    ) -> None:
        self.links = links if links is not None else FIXTURE_LINKS
        self.errors = errors
        self.headers: dict[str, str] = {}
        self.calls: list[dict] = []

    def get(
        self, url: str, params: dict | None = None, timeout: float | None = None
    ) -> FakeResponse:
        """Answer one API call.

        Parameters
        ----------
        url
            Requested endpoint (unused by the fake).
        params
            Query parameters of the call.
        timeout
            Timeout of the call (unused by the fake).

        Returns
        -------
        FakeResponse
            The canned answer.
        """
        self.calls.append(dict(params or {}))
        if self.errors > 0:
            self.errors -= 1
            return FakeResponse({}, status_code=429, headers={"Retry-After": "0"})
        params = params or {}
        title = params.get("titles", "")
        if "prop" in params and params["prop"] == "info":
            return FakeResponse({"query": {"pages": {"1": {"title": title}}}})
        if title not in self.links:
            return FakeResponse(_missing_payload(title))
        return FakeResponse(_links_payload(title, self.links[title]))


def make_client(
    tmp_path, session: FakeSession | None = None, **kwargs
) -> WikiApiClient:
    """Build a :class:`WikiApiClient` with caching and delays disabled.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory used as cache directory.
    session
        Optional fake session.
    **kwargs
        Extra client options.

    Returns
    -------
    WikiApiClient
        The configured client.
    """
    options = {"delay": 0.0, "cache_dir": tmp_path}
    options.update(kwargs)
    return WikiApiClient(session=session or FakeSession(), **options)


# --------------------------------------------------------------- input parsing
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "https://en.wikipedia.org/wiki/Information_retrieval",
            ArticleRef("en", "Information retrieval"),
        ),
        (
            "https://ru.wikipedia.org/wiki/Информационный_поиск",
            ArticleRef("ru", "Информационный поиск"),
        ),
        (
            "https://en.m.wikipedia.org/wiki/Graph_theory",
            ArticleRef("en", "Graph theory"),
        ),
        (
            "https://de.wikipedia.org/w/index.php?title=Zahlentheorie",
            ArticleRef("de", "Zahlentheorie"),
        ),
        ("ru:Информационный_поиск", ArticleRef("ru", "Информационный поиск")),
        ("fr Théorie des graphes", ArticleRef("fr", "Théorie des graphes")),
        ("Information retrieval", ArticleRef("en", "Information retrieval")),
    ],
)
def test_parse_article_input(raw: str, expected: ArticleRef) -> None:
    """Every accepted input format maps to the expected article reference."""
    assert parse_article_input(raw) == expected


@pytest.mark.parametrize("raw", ["", "   "])
def test_parse_article_input_rejects_empty(raw: str) -> None:
    """An empty input raises a friendly ``ValueError``."""
    with pytest.raises(ValueError):
        parse_article_input(raw)


def test_normalise_title_and_cache_key() -> None:
    """Titles are normalised and cache keys are stable SHA-1 identifiers."""
    assert normalise_title("  Deep__learning ") == "Deep learning"
    assert cache_key("en", "Root") == cache_key("en", "Root")
    assert cache_key("en", "Root") != cache_key("ru", "Root")
    assert cache_key("en", "Root").startswith("en_")
    assert cache_key("en", "Root").endswith(".json")


# ------------------------------------------------------------------ fetching
def test_fetch_links_sorted_filtered_and_truncated(tmp_path) -> None:
    """Links are de-duplicated, sorted, disambiguation-free and truncated."""
    client = make_client(tmp_path)
    assert client.fetch_links("Root", max_children=2) == ["Alpha", "Bravo"]


def test_fetch_links_uses_cache(tmp_path) -> None:
    """A second call is served from disk without touching the session."""
    session = FakeSession()
    client = make_client(tmp_path, session=session)
    first = client.fetch_links("Root", max_children=3)
    calls_after_first = len(session.calls)
    second = client.fetch_links("Root", max_children=3)
    assert first == second == ["Alpha", "Bravo", "Charlie"]
    assert len(session.calls) == calls_after_first
    assert client.stats == {"fetched": 1, "cached": 1, "failed": 0}
    cache_file = tmp_path / cache_key("en", "Root")
    assert cache_file.exists()
    assert json.loads(cache_file.read_text(encoding="utf-8"))["links"] == first


def test_fetch_links_retries_on_429(tmp_path, monkeypatch) -> None:
    """HTTP 429 is retried with backoff until the request succeeds."""
    monkeypatch.setattr(wiki_api.time, "sleep", lambda *_: None)
    session = FakeSession(errors=2)
    client = make_client(tmp_path, session=session)
    assert client.fetch_links("Root", max_children=1) == ["Alpha"]
    assert len(session.calls) == 3


def test_fetch_links_raises_for_missing_article(tmp_path) -> None:
    """A non-existing article raises :class:`WikiApiError`."""
    client = make_client(tmp_path)
    with pytest.raises(WikiApiError):
        client.fetch_links("Does not exist")


def test_resolve_title(tmp_path) -> None:
    """Title resolution returns the canonical title from the API."""
    client = make_client(tmp_path)
    assert client.resolve_title("Information_Retrieval") == "Information Retrieval"
    assert client.article_url("Root") == "https://en.wikipedia.org/wiki/Root"


def test_user_agent_is_set(tmp_path) -> None:
    """The session advertises the polite ``User-Agent`` required by Wikipedia."""
    client = make_client(tmp_path)
    assert (
        client.session.headers["User-Agent"]
        == "WikiGraphExplorer/1.0 (educational project)"
    )


# ----------------------------------------------------------------------- BFS
def test_bfs_depth_one_only_fetches_root(tmp_path) -> None:
    """At depth 1 only the root article is expanded."""
    session = FakeSession()
    result = build_crawl(
        make_client(tmp_path, session=session), "Root", depth=1, max_children=3
    )
    assert sorted(result.depths) == ["Alpha", "Bravo", "Charlie", "Root"]
    assert list(result.links) == ["Root"]
    assert result.depths["Root"] == 0
    assert result.depths["Alpha"] == 1
    assert result.node_count == 4
    assert result.edge_count == 3


def test_bfs_depth_two_expands_first_level(tmp_path) -> None:
    """At depth 2 the direct neighbours are expanded as well."""
    result = build_crawl(make_client(tmp_path), "Root", depth=2, max_children=3)
    assert sorted(result.links) == ["Alpha", "Bravo", "Charlie", "Root"]
    assert "Delta" in result.depths
    assert result.depths["Delta"] == 2
    assert result.depths["Root"] == 0
    assert result.lang == "en"
    assert result.root == "Root"


def test_bfs_respects_max_children(tmp_path) -> None:
    """No article keeps more than ``max_children`` out-links."""
    result = build_crawl(make_client(tmp_path), "Root", depth=3, max_children=1)
    assert all(len(children) <= 1 for children in result.links.values())
    assert result.links["Root"] == ["Alpha"]
    assert result.max_children == 1


def test_bfs_is_deterministic(tmp_path) -> None:
    """Two identical crawls produce exactly the same result."""
    first = build_crawl(make_client(tmp_path), "Root", depth=3, max_children=2)
    second = build_crawl(make_client(tmp_path), "Root", depth=3, max_children=2)
    assert first.links == second.links
    assert list(first.depths) == list(second.depths)
    assert first.depths == second.depths


def test_bfs_skips_failing_articles(tmp_path) -> None:
    """Articles that cannot be read are recorded and do not abort the crawl."""
    result = build_crawl(make_client(tmp_path), "Nonexistent", depth=2, max_children=2)
    assert "Nonexistent" in result.errors
    assert "Root" not in result.depths
    assert result.node_count == 1


def test_bfs_reports_progress(tmp_path) -> None:
    """The progress callback is invoked with monotonically growing counters."""
    seen: list[tuple[int, int]] = []
    build_crawl(
        make_client(tmp_path),
        "Root",
        depth=2,
        max_children=2,
        progress=lambda snapshot: seen.append((snapshot.fetched, snapshot.queued)),
    )
    assert seen
    assert seen[-1] == (3, 0)
    assert [item[0] for item in seen] == sorted(item[0] for item in seen)


def test_bfs_truncates_at_node_budget(tmp_path) -> None:
    """The node budget stops the crawl and sets the ``truncated`` flag."""
    result = build_crawl(
        make_client(tmp_path), "Root", depth=3, max_children=5, node_budget=3
    )
    assert result.truncated is True
    assert result.node_count == 3


def test_node_budget_for_depth() -> None:
    """The configured node budget is depth dependent."""
    assert node_budget_for_depth(1) == 60
    assert node_budget_for_depth(2) == 200
    assert node_budget_for_depth(3) == 1000
    assert node_budget_for_depth(99) == 1000
