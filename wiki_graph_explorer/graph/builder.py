"""Construction of a :class:`networkx.DiGraph` from a crawl result.

Edge semantics follow the web-graph convention used throughout this project::

    u -> v   <=>   the article ``u`` contains a hyperlink to the article ``v``

References
----------
.. [1] Kleinberg, J. M. (1999). Authoritative sources in a hyperlinked
       environment. *Journal of the ACM*, 46(5), 604-632.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx

from crawler.bfs import CrawlResult
from graph.centrality import PAGERANK, CentralityResult

LOGGER = logging.getLogger("wge.graph.builder")


def build_digraph(crawl: CrawlResult) -> nx.DiGraph:
    """Build the directed web graph of a crawl result.

    Nodes are keyed by the canonical Wikipedia title.  Each node carries the
    attributes ``title``, ``url``, ``lang``, ``depth`` and ``is_root``; each
    edge carries ``label`` (always ``"link"``) which the visualiser uses for
    arrow styling.

    Parameters
    ----------
    crawl
        Output of :func:`crawler.bfs.build_crawl`.

    Returns
    -------
    networkx.DiGraph
        Graph with ``crawl.node_count`` nodes and at most ``crawl.edge_count``
        edges.  Links pointing outside the crawled subtree are dropped, so the
        result is always self-contained.

    Examples
    --------
    >>> graph = build_digraph(crawl)                 # doctest: +SKIP
    >>> graph.number_of_nodes(), graph.number_of_edges()   # doctest: +SKIP
    (148, 512)
    """
    graph = nx.DiGraph()
    for title, depth in sorted(
        crawl.depths.items(), key=lambda item: (item[1], item[0])
    ):
        graph.add_node(
            title,
            title=title,
            url=crawl.article_url(title),
            lang=crawl.lang,
            depth=depth,
            is_root=(title == crawl.root),
        )
    for source, children in sorted(crawl.links.items()):
        if source not in graph:
            continue
        for target in children:
            if target in graph:
                graph.add_edge(source, target, label="link")
    LOGGER.info(
        "Built DiGraph for '%s': %d nodes / %d edges",
        crawl.root,
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph


def raw_degrees(graph: nx.DiGraph) -> dict[str, dict[str, int]]:
    """Return the raw in/out degree of every node.

    Parameters
    ----------
    graph
        The directed web graph.

    Returns
    -------
    dict
        ``{title: {"in": indegree, "out": outdegree}}`` for every node.

    Examples
    --------
    >>> raw_degrees(nx.DiGraph([("A", "B")]))
    {'A': {'in': 0, 'out': 1}, 'B': {'in': 1, 'out': 0}}
    """
    return {
        str(node): {
            "in": int(graph.in_degree(node)),
            "out": int(graph.out_degree(node)),
        }
        for node in graph.nodes
    }


def to_table_rows(
    graph: nx.DiGraph,
    result: CentralityResult,
    sort_measure: str = PAGERANK,
) -> list[dict[str, Any]]:
    """Flatten a :class:`graph.centrality.CentralityResult` into table rows.

    The rows are sorted by the measure currently highlighted in the UI so that
    the most important articles come first; ties are broken alphabetically,
    which keeps the ordering deterministic across runs.

    Parameters
    ----------
    graph
        The directed web graph.
    result
        A :class:`graph.centrality.CentralityResult`.
    sort_measure
        Key of the measure used for ordering (default PageRank).

    Returns
    -------
    list of dict
        One dictionary per node with the title, URL, degrees, every centrality
        value and its percentile rank inside the graph.

    Examples
    --------
    >>> rows = to_table_rows(graph, result, PAGERANK)   # doctest: +SKIP
    >>> rows[0]["Article"]                              # doctest: +SKIP
    'Information retrieval'
    """
    degrees = raw_degrees(graph)
    values = result.values(sort_measure)
    order = sorted(graph.nodes, key=lambda node: (-values.get(node, 0.0), str(node)))
    rows: list[dict[str, Any]] = []
    for node in order:
        row: dict[str, Any] = {
            "Article": str(node),
            "Depth": int(graph.nodes[node].get("depth", 0)),
            "In-deg": degrees[node]["in"],
            "Out-deg": degrees[node]["out"],
        }
        for key, measure_values in result.measures.items():
            row[result.label_of(key)] = float(measure_values.get(node, 0.0))
        rows.append(row)
    return rows
