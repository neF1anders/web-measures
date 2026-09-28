"""Unit tests for graph construction, the pyvis renderer and the UI helpers."""

from __future__ import annotations

import json

import networkx as nx
import pytest

import config
from crawler.bfs import CrawlResult
from crawler.wiki_api import ArticleRef
from graph.builder import build_digraph, raw_degrees, to_table_rows
from graph.centrality import (
    BETWEENNESS,
    MEASURE_ORDER,
    PAGERANK,
    compute_all_centralities,
)
from ui.details_panel import measure_bar_figure, measure_frame
from viz.legends import legend_html, measure_reference_markdown
from viz.pyvis_render import (
    PICKER_ID,
    build_network,
    centrality_json,
    inject_node_picker,
    node_colors,
    render_html,
    size_measure_options,
    winner_summary,
)

LINKS = {
    "Root": ["Alpha", "Bravo"],
    "Alpha": ["Charlie"],
    "Bravo": ["Charlie"],
    "Charlie": ["Outside"],
}
DEPTHS = {"Root": 0, "Alpha": 1, "Bravo": 1, "Charlie": 2}


@pytest.fixture()
def crawl() -> CrawlResult:
    """Return a small synthetic crawl result.

    Returns
    -------
    CrawlResult
        Four articles, where ``Charlie`` links to a node outside the subtree.
    """
    return CrawlResult(
        ref=ArticleRef(lang="en", title="Root"),
        links=LINKS,
        depths=DEPTHS,
        max_children=2,
        depth=2,
    )


@pytest.fixture()
def built(crawl: CrawlResult):
    """Return ``(graph, centralities)`` for the synthetic crawl.

    Parameters
    ----------
    crawl
        The synthetic crawl result.

    Returns
    -------
    tuple
        ``(networkx.DiGraph, graph.centrality.CentralityResult)``.
    """
    graph = build_digraph(crawl)
    return graph, compute_all_centralities(graph)


# ------------------------------------------------------------------- builder
def test_build_digraph_nodes_and_edges(crawl: CrawlResult) -> None:
    """Nodes carry their metadata and out-of-subtree links are dropped."""
    graph = build_digraph(crawl)
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 4
    assert set(graph.nodes) == set(DEPTHS)
    assert graph.nodes["Root"]["is_root"] is True
    assert graph.nodes["Alpha"]["is_root"] is False
    assert graph.nodes["Root"]["depth"] == 0
    assert graph.nodes["Charlie"]["depth"] == 2
    assert graph.nodes["Alpha"]["url"] == "https://en.wikipedia.org/wiki/Alpha"
    assert graph.nodes["Root"]["lang"] == "en"
    assert not graph.has_edge("Charlie", "Outside")


def test_build_digraph_is_deterministic(crawl: CrawlResult) -> None:
    """Two builds produce the same node and edge sets."""
    first, second = build_digraph(crawl), build_digraph(crawl)
    assert list(first.nodes) == list(second.nodes)
    assert list(first.edges) == list(second.edges)


def test_raw_degrees(crawl: CrawlResult) -> None:
    """Raw degrees are integers keyed by article title."""
    degrees = raw_degrees(build_digraph(crawl))
    assert degrees["Root"] == {"in": 0, "out": 2}
    assert degrees["Charlie"] == {"in": 2, "out": 0}


def test_to_table_rows_sorted_by_measure(built) -> None:
    """Table rows are sorted by the chosen measure, ties broken alphabetically."""
    graph, centralities = built
    rows = to_table_rows(graph, centralities, PAGERANK)
    assert len(rows) == 4
    assert rows[0]["Article"] in graph.nodes
    values = [row["PageRank"] for row in rows]
    assert values == sorted(values, reverse=True)
    for key in MEASURE_ORDER:
        assert centralities.label_of(key) in rows[0]


# ------------------------------------------------------------------ renderer
def test_build_network(built) -> None:
    """The pyvis network has one node per article with a size and a colour."""
    graph, centralities = built
    network = build_network(graph, centralities, PAGERANK)
    assert network.num_nodes() == graph.number_of_nodes()
    assert network.num_edges() == graph.number_of_edges()
    for title in graph.nodes:
        node = next(item for item in network.nodes if item["id"] == title)
        assert config.NODE_SIZE_MIN <= node["size"] <= config.NODE_SIZE_MAX
        assert node["color"].startswith("#")
        assert "PageRank" in node["title"]


def test_node_colors_only_winners(built) -> None:
    """Exactly the per-measure winners are coloured; the rest is light grey."""
    graph, centralities = built
    colors = node_colors(centralities)
    assert set(colors) == set(centralities.winners().values())
    for winner in centralities.winners().values():
        assert colors[winner] != config.COLOR_REGULAR


def test_render_html_contains_picker(built) -> None:
    """The rendered document embeds the node picker and the vis-network bundle."""
    graph, centralities = built
    document = render_html(graph, centralities, PAGERANK, "Alpha")
    assert PICKER_ID in document
    assert "vis.Network" in document
    assert '<option value="Alpha" selected>' in document
    assert (
        "searchParams.set(&quot;node&quot;" in document
        or "searchParams.set(" in document
    )


def test_inject_node_picker_escapes_titles() -> None:
    """HTML metacharacters in titles cannot break out of the option element."""
    document = inject_node_picker(
        "<html><body></body></html>", ['<img src=x onerror="alert(1)">']
    )
    assert "onerror=&quot;alert(1)&quot;" in document
    assert "<img src=x" not in document


def test_inject_node_picker_without_body() -> None:
    """A document without ``</body>`` still receives the picker."""
    assert PICKER_ID in inject_node_picker("<div></div>", ["A"])


def test_size_measure_options() -> None:
    """PageRank is the default sizing metric and is offered first."""
    labels = size_measure_options()
    assert labels[0] == "PageRank"
    assert len(labels) == len(MEASURE_ORDER)


def test_winner_summary_and_json(built) -> None:
    """The helper strings contain every measure and valid JSON."""
    _, centralities = built
    summary = winner_summary(centralities)
    assert "PageRank" in summary
    assert set(json.loads(centrality_json(centralities))) == set(MEASURE_ORDER)


# --------------------------------------------------------------------- legend
def test_legend_html_lists_every_measure(built) -> None:
    """The legend has one card per measure plus the regular-node entry."""
    _, centralities = built
    legend = legend_html(centralities)
    for key in MEASURE_ORDER:
        assert centralities.label_of(key) in legend
    assert "Regular article" in legend
    assert "top-1" in legend


def test_legend_html_without_centralities() -> None:
    """The legend can be rendered before a graph exists."""
    assert "Regular article" in legend_html()


def test_measure_reference_markdown() -> None:
    """All six canonical references are cited in the bibliography."""
    text = measure_reference_markdown()
    for author in ("Brin", "Kleinberg", "Newman", "Boldi", "Langville", "Kempe"):
        assert author in text


# -------------------------------------------------------------- details panel
def test_measure_frame(built) -> None:
    """The details table has one row per measure with value, percentile, rank."""
    graph, centralities = built
    frame = measure_frame(centralities, "Alpha")
    assert list(frame["Measure"]) == [
        centralities.label_of(key) for key in MEASURE_ORDER
    ]
    assert frame["Value"].dtype.kind == "f"
    assert frame["Rank"].min() >= 1
    assert frame["Percentile"].between(0.0, 1.0).all()
    best = frame.sort_values("Rank").iloc[0]
    assert best["Rank"] == 1
    assert frame.loc[frame["Measure"] == "Betweenness centrality", "Value"].iloc[0] == (
        centralities.values(BETWEENNESS)["Alpha"]
    )


def test_measure_bar_figure(built) -> None:
    """The bar chart is built without a display server."""
    _, centralities = built
    figure = measure_bar_figure(measure_frame(centralities, "Alpha"))
    assert len(figure.axes) == 1
    assert len(figure.axes[0].get_yticklabels()) == len(MEASURE_ORDER)
    assert figure.axes[0].get_xlim() == (0, 1)


# ------------------------------------------------------------------ renderer 2
def test_render_html_is_deterministic(built) -> None:
    """Rendering twice produces the same document (no random layout seeds)."""
    graph, centralities = built
    first = render_html(graph, centralities, PAGERANK)
    second = render_html(graph, centralities, PAGERANK)
    assert first == second


def test_build_network_handles_empty_graph() -> None:
    """An empty graph does not crash the renderer."""
    graph = nx.DiGraph()
    network = build_network(graph, compute_all_centralities(graph), PAGERANK)
    assert network.num_nodes() == 0
