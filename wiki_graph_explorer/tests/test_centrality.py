"""Unit tests for the centrality measures.

The expectations are analytic values of small, well known graphs (a directed
path ``a -> b -> c -> d`` and a directed star ``c -> {l1, l2, l3}``), so a
regression in the normalisation of any measure is caught immediately.
"""

from __future__ import annotations

import networkx as nx
import pytest

from graph.centrality import (
    DEGREE_IN,
    DEGREE_OUT,
    HITS_AUTHORITY,
    HITS_HUB,
    MEASURE_ORDER,
    PAGERANK,
    betweenness_centrality,
    classical_closeness_centrality,
    closeness_centrality,
    compute_all_centralities,
    degree_centralities,
    eigenvector_centrality,
    hits,
    normalise_sizes,
    pagerank,
    percentile_ranks,
)

PATH = [("a", "b"), ("b", "c"), ("c", "d")]
STAR = [("c", "l1"), ("c", "l2"), ("c", "l3")]


def path_graph() -> nx.DiGraph:
    """Return the directed path ``a -> b -> c -> d``.

    Returns
    -------
    networkx.DiGraph
        Four nodes, three edges.
    """
    return nx.DiGraph(PATH)


def star_graph() -> nx.DiGraph:
    """Return the out-star ``c -> {l1, l2, l3}``.

    Returns
    -------
    networkx.DiGraph
        Four nodes, three edges.
    """
    return nx.DiGraph(STAR)


# ------------------------------------------------------------------ 1. degree
def test_degree_centrality_values() -> None:
    """Degree centralities equal the raw degrees divided by ``N - 1`` (Freeman)."""
    degrees = degree_centralities(path_graph())
    assert degrees[DEGREE_OUT] == {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3, "d": 0.0}
    assert degrees[DEGREE_IN] == {"a": 0.0, "b": 1 / 3, "c": 1 / 3, "d": 1 / 3}


def test_degree_centrality_star() -> None:
    """In the out-star the centre has all the out-links, the leaves all the in-links."""
    degrees = degree_centralities(star_graph())
    assert degrees[DEGREE_OUT]["c"] == 1.0
    assert degrees[DEGREE_IN]["l1"] == pytest.approx(1 / 3)
    assert degrees[DEGREE_IN]["c"] == 0.0


# -------------------------------------------------------------- 2. betweenness
def test_betweenness_path_known_values() -> None:
    """On ``a -> b -> c -> d`` the two inner nodes carry 1/3 each."""
    values = betweenness_centrality(path_graph())
    assert values["a"] == pytest.approx(0.0)
    assert values["b"] == pytest.approx(1 / 3)
    assert values["c"] == pytest.approx(1 / 3)
    assert values["d"] == pytest.approx(0.0)


def test_betweenness_bridge_node_is_top() -> None:
    """The centre of a bidirectional star lies on all six leaf-to-leaf paths."""
    graph = nx.DiGraph()
    for leaf in ("l1", "l2", "l3"):
        graph.add_edge("c", leaf)
        graph.add_edge(leaf, "c")
    values = betweenness_centrality(graph)
    assert values["c"] == pytest.approx(1.0)
    assert values["l1"] == pytest.approx(0.0)


def test_betweenness_out_star_is_zero() -> None:
    """A pure out-star has no path between two leaves, so betweenness is zero."""
    values = betweenness_centrality(star_graph())
    assert set(values.values()) == {0.0}


def test_betweenness_unnormalised() -> None:
    """Without normalisation the raw shortest-path counts are returned."""
    values = betweenness_centrality(path_graph(), normalized=False)
    assert values["b"] == pytest.approx(2.0)


def test_betweenness_too_small_graph() -> None:
    """Graphs with fewer than three nodes have zero betweenness."""
    assert betweenness_centrality(nx.DiGraph([("a", "b")])) == {"a": 0.0, "b": 0.0}


# ------------------------------------------------------------ 3. closeness
def test_harmonic_closeness_path_values() -> None:
    """Harmonic centrality of a path equals the sum of inverse out-distances."""
    values = closeness_centrality(path_graph())
    assert values["a"] == pytest.approx(1 + 1 / 2 + 1 / 3)
    assert values["b"] == pytest.approx(1 + 1 / 2)
    assert values["c"] == pytest.approx(1.0)
    assert values["d"] == 0.0
    assert values["a"] > values["b"] > values["c"] > values["d"]


def test_classical_closeness_collapses_on_directed_path() -> None:
    """The classical measure is meaningless on a directed path (no back-links).

    Only ``a`` reaches every other node, so the sink ``d`` - which cannot reach
    anything - collapses to ``0`` although it is a perfectly well connected
    article.  This is exactly why the harmonic variant is used in the app.
    """
    values = classical_closeness_centrality(path_graph())
    assert values["a"] == pytest.approx(0.5)
    assert values["d"] == 0.0
    assert values["b"] < values["a"]


def test_harmonic_closeness_star() -> None:
    """In the out-star only the centre reaches the leaves."""
    values = closeness_centrality(star_graph())
    assert values["c"] == pytest.approx(3.0)
    assert values["l1"] == pytest.approx(0.0)


def test_eigenvector_path_peaks_in_the_middle() -> None:
    """On the symmetrised path the two inner articles are the most central."""
    values = eigenvector_centrality(path_graph())
    assert sum(values.values()) == pytest.approx(1.0)
    assert values["b"] == pytest.approx(values["c"])
    assert values["b"] > values["a"]


# --------------------------------------------------------- 4. eigenvector
def test_eigenvector_star_centre_wins() -> None:
    """In the out-star the centre is the most central node."""
    values = eigenvector_centrality(star_graph())
    assert max(values, key=lambda node: values[node]) == "c"
    assert sum(values.values()) == pytest.approx(1.0)


def test_eigenvector_fallback_on_ambiguous_solution(monkeypatch) -> None:
    """An ambiguous directed eigenproblem falls back to the power iteration."""

    def boom(*args, **kwargs):
        raise nx.AmbiguousSolution("does not give consistent results")

    monkeypatch.setattr(nx, "eigenvector_centrality_numpy", boom)
    values = eigenvector_centrality(star_graph())
    assert sum(values.values()) == pytest.approx(1.0)
    assert all(value >= 0.0 for value in values.values())
    assert max(values, key=lambda node: values[node]) == "c"


# -------------------------------------------------------------- 5. PageRank
def test_pagerank_is_a_distribution() -> None:
    """PageRank values form a probability distribution."""
    for graph in (path_graph(), star_graph()):
        values = pagerank(graph, alpha=0.85)
        assert sum(values.values()) == pytest.approx(1.0)
        assert all(0.0 <= value <= 1.0 for value in values.values())


def test_pagerank_star_leaves_win() -> None:
    """In an out-star the sinks collect the rank, not the hub that links out.

    This is the classic difference to HITS: PageRank follows the *in*-edges, so
    an article that many other articles point at scores higher than the article
    that points at them.
    """
    values = pagerank(star_graph())
    assert max(values, key=lambda node: values[node]) == "l1"
    assert values["l1"] == pytest.approx(values["l2"])
    assert values["l1"] > values["c"]


def test_pagerank_path_increases_downstream() -> None:
    """Along a directed path PageRank grows with the number of in-links.

    ``a`` is a dangling node (nothing links to it), so it only receives its
    teleportation mass, while ``d`` collects the rank of the whole upstream.
    """
    values = pagerank(path_graph())
    assert values["a"] < values["b"] < values["c"] < values["d"]


# ------------------------------------------------------------------ 6. HITS
def test_hits_separates_hubs_and_authorities() -> None:
    """The star centre is the hub, its leaves are the authorities."""
    scores = hits(star_graph())
    assert max(scores[HITS_HUB], key=lambda node: scores[HITS_HUB][node]) == "c"
    assert max(
        scores[HITS_AUTHORITY], key=lambda node: scores[HITS_AUTHORITY][node]
    ) in {"l1", "l2", "l3"}
    assert sum(scores[HITS_HUB].values()) == pytest.approx(1.0)
    assert sum(scores[HITS_AUTHORITY].values()) == pytest.approx(1.0)


# --------------------------------------------------------------- aggregation
def test_compute_all_centralities_covers_every_measure() -> None:
    """``compute_all_centralities`` returns the eight documented measures."""
    graph = nx.DiGraph(PATH + STAR)
    result = compute_all_centralities(graph)
    assert list(result.measures) == list(MEASURE_ORDER)
    assert result.node_count == graph.number_of_nodes()
    for values in result.measures.values():
        assert set(values) == set(graph.nodes)
    assert result.top_node(PAGERANK) in graph.nodes
    assert set(result.winners()) == set(MEASURE_ORDER)
    assert result.max_value(PAGERANK) > 0
    assert result.label_of(PAGERANK) == "PageRank"
    assert result.label_of("nope") == "nope"
    assert result.values("nope") == {}


def test_centralities_are_deterministic() -> None:
    """Computing the measures twice yields bit-identical results.

    The traversal, the node order and every solver start vector are fixed, so
    two runs on the same graph agree down to the last floating point bit - this
    is what the "reproducible demo" requirement of the report relies on.
    """
    graph = nx.DiGraph(PATH + STAR)
    first = compute_all_centralities(graph)
    second = compute_all_centralities(graph)
    for key in MEASURE_ORDER:
        assert first.values(key) == second.values(key)


def test_empty_graph_is_handled() -> None:
    """An empty graph produces empty result sets instead of raising."""
    result = compute_all_centralities(nx.DiGraph())
    assert result.node_count == 0
    assert all(values == {} for values in result.measures.values())
    assert result.top_node(PAGERANK) is None
    assert result.winners() == {}


def test_as_frame() -> None:
    """The pandas view has one row per article and one column per measure."""
    frame = compute_all_centralities(nx.DiGraph(PATH)).as_frame()
    assert list(frame.index) == ["a", "b", "c", "d"]
    assert len(frame.columns) == len(MEASURE_ORDER)
    assert frame.loc["b", "PageRank"] == pytest.approx(pagerank(path_graph())["b"])


# --------------------------------------------------------------- percentiles
def test_percentile_ranks() -> None:
    """Percentiles are the share of nodes scoring strictly lower."""
    assert percentile_ranks({"a": 1.0, "b": 2.0, "c": 3.0}) == {
        "a": 0.0,
        "b": 0.5,
        "c": 1.0,
    }
    assert percentile_ranks({"a": 1.0, "b": 1.0}) == {"a": 0.0, "b": 0.0}
    assert percentile_ranks({}) == {}
    assert percentile_ranks({"only": 5.0}) == {"only": 1.0}


def test_normalise_sizes() -> None:
    """Node sizes are min-max rescaled into the pyvis range."""
    assert normalise_sizes({"a": 0.0, "b": 1.0}, 10, 20) == {"a": 10.0, "b": 20.0}
    assert normalise_sizes({"a": 5.0, "b": 5.0}, 10, 20) == {"a": 15.0, "b": 15.0}
    assert normalise_sizes({}) == {}
