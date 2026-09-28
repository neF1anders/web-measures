"""End-to-end smoke test of the Streamlit script (no network, no browser).

``streamlit.testing.v1.AppTest`` executes ``app.py`` exactly like the real
server does - including ``st.set_page_config``, the sidebar widgets, the pyvis
rendering and the details panel - while the MediaWiki client is monkeypatched
with a canned article graph.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

import config
from crawler.wiki_api import WikiApiClient

APP = str(config.BASE_DIR / "app.py")

FIXTURE_LINKS: dict[str, list[str]] = {
    "Information retrieval": ["Precision and recall", "Search engine", "Ranking"],
    "Precision and recall": ["Confusion matrix"],
    "Search engine": ["Ranking"],
    "Ranking": ["Search engine"],
    "Confusion matrix": ["Information retrieval"],
}


def fake_resolve_title(self: WikiApiClient, title: str) -> str:
    """Pretend the API resolves a title to itself.

    Parameters
    ----------
    self
        The client instance.
    title
        The requested title.

    Returns
    -------
    str
        The normalised title.
    """
    return " ".join(title.replace("_", " ").split())


def fake_fetch_links(
    self: WikiApiClient, title: str, max_children: int = 15
) -> list[str]:
    """Return the canned out-links of an article.

    Parameters
    ----------
    self
        The client instance.
    title
        Article to inspect.
    max_children
        Fan-out cap.

    Returns
    -------
    list of str
        Sorted, truncated out-links.
    """
    return sorted(FIXTURE_LINKS.get(title, []))[:max_children]


@pytest.fixture()
def app(monkeypatch) -> AppTest:
    """Return a fresh :class:`AppTest` with the crawler replaced by fixtures.

    Parameters
    ----------
    monkeypatch
        Pytest monkeypatch fixture.

    Returns
    -------
    streamlit.testing.v1.AppTest
        The test harness for ``app.py``.
    """
    monkeypatch.setattr(WikiApiClient, "resolve_title", fake_resolve_title)
    monkeypatch.setattr(WikiApiClient, "fetch_links", fake_fetch_links)
    return AppTest.from_file(APP, default_timeout=120)


def test_app_renders_idle_state(app: AppTest) -> None:
    """Without a request the app shows the input hint and no exception."""
    app.run()
    assert not app.exception
    assert app.text_input[0].label == "Wikipedia article URL"
    assert app.number_input[0].label == "Depth"
    assert app.number_input[0].value == config.DEFAULT_DEPTH
    assert app.number_input[1].label == "Max children per node"
    assert app.number_input[1].value == config.DEFAULT_MAX_CHILDREN
    assert app.button[0].label == "Build graph"
    assert app.info


def test_app_builds_graph_and_details_panel(app: AppTest) -> None:
    """Pressing "Build graph" renders the graph, the table and the details panel."""
    app.run()
    app.text_input[0].set_value("https://en.wikipedia.org/wiki/Information_retrieval")
    app.number_input[0].set_value(1)
    app.number_input[1].set_value(3)
    app.button[0].click().run()

    assert not app.exception
    assert not app.error
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Articles"] == "4"
    assert metrics["Links"] == "3"
    assert metrics["Depth"] == "1"
    assert "In-degree" in metrics
    assert app.selectbox[0].label == "Node size metric"
    assert app.selectbox[1].label == "Inspect article"
    assert app.selectbox[1].value == "Information retrieval"
    # dataframe[0] is the eight-measure table of the details panel, the
    # searchable article table comes last.
    assert app.dataframe[1].value.shape[0] == 4
    assert "Article" in app.dataframe[1].value.columns


def test_app_reports_bad_input(app: AppTest) -> None:
    """An unusable input produces a visible error instead of a traceback."""
    app.run()
    app.text_input[0].set_value("   ")
    app.button[0].click().run()
    assert not app.exception
    assert any(
        "enter a wikipedia article" in error.value.lower() for error in app.error
    )


def test_app_selects_another_article(app: AppTest) -> None:
    """Changing the picker updates the details panel of the chosen article."""
    app.run()
    app.text_input[0].set_value("Information retrieval")
    app.number_input[0].set_value(2)
    app.button[0].click().run()
    assert not app.exception

    options = app.selectbox[1].options
    app.selectbox[1].set_value("Ranking")
    app.run()
    assert not app.exception
    assert "Ranking" in options
    headers = [markdown.value for markdown in app.markdown]
    assert any("Ranking" in header for header in headers)
