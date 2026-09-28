"""Details panel of the selected article (centrality values, bars, links)."""

from __future__ import annotations

import logging
from typing import Any

import matplotlib
import networkx as nx
import pandas as pd
import streamlit as st

from graph.centrality import MEASURE_ORDER, MEASURES, CentralityResult, percentile_ranks

LOGGER = logging.getLogger("wge.ui.details")

FLOAT_FORMAT = "%.6f"
"""Every centrality value is reported with six decimals, as in the report."""


def measure_frame(centralities: CentralityResult, title: str) -> pd.DataFrame:
    """Build the per-measure table of one article.

    Parameters
    ----------
    centralities
        All centrality values of the graph.
    title
        Article to describe.

    Returns
    -------
    pandas.DataFrame
        One row per measure with the value, the percentile rank inside the
        graph, the rank and the value normalised by the best article of that
        measure.

    Examples
    --------
    >>> frame = measure_frame(centralities, "Information retrieval")
    >>> list(frame.columns)
    ['Measure', 'Value', 'Percentile', 'Rank']
    """
    rows: list[dict[str, Any]] = []
    for key in MEASURE_ORDER:
        if key not in centralities.measures:
            continue
        values = centralities.values(key)
        percentiles = percentile_ranks(values)
        ordered = sorted(values.items(), key=lambda item: (-item[1], str(item[0])))
        rank = next(
            (index + 1 for index, (node, _) in enumerate(ordered) if node == title),
            0,
        )
        best = ordered[0][1] if ordered else 0.0
        rows.append(
            {
                "Measure": MEASURES[key].label,
                "Value": float(values.get(title, 0.0)),
                "Percentile": float(percentiles.get(title, 0.0)),
                "Rank": int(rank),
                "Colour": MEASURES[key].color,
                "Relative to best": (
                    float(values.get(title, 0.0) / best) if best > 0 else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def measure_bar_figure(frame: pd.DataFrame) -> matplotlib.figure.Figure:
    """Draw the horizontal bar chart comparing the eight measures.

    Bars are coloured like the graph (winner colour per measure) and show the
    percentile of the article inside the current graph, which makes the eight
    otherwise incomparable scales directly comparable.

    Parameters
    ----------
    frame
        Output of :func:`measure_frame`.

    Returns
    -------
    matplotlib.figure.Figure
        Figure ready for ``st.pyplot``.
    """
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = frame.iloc[::-1]
    figure, axes = plt.subplots(figsize=(6.0, 3.4), dpi=140)
    axes.barh(
        frame["Measure"],
        frame["Percentile"],
        color=frame["Colour"],
        edgecolor="#9aa0aa",
        height=0.62,
    )
    axes.set_xlim(0, 1)
    axes.set_xlabel("percentile within the crawled graph")
    axes.set_title("Centrality profile", fontsize=11)
    axes.grid(axis="x", linestyle=":", alpha=0.45)
    for index, value in enumerate(frame["Percentile"]):
        axes.text(
            min(float(value) + 0.015, 0.965),
            index,
            f"{float(value):.2f}",
            va="center",
            fontsize=8,
        )
    for spine in ("top", "right", "left"):
        axes.spines[spine].set_visible(False)
    figure.tight_layout()
    return figure


def _neighbour_list(
    graph: nx.DiGraph, title: str, direction: str, limit: int = 12
) -> str:
    """Render a compact markdown list of neighbours.

    Parameters
    ----------
    graph
        The crawled web graph.
    title
        Article whose neighbours are listed.
    direction
        ``"out"`` for articles it links to, ``"in"`` for articles linking to it.
    limit
        Maximum number of entries shown.

    Returns
    -------
    str
        Markdown list, or a placeholder when there is no neighbour.
    """
    neighbours = (
        list(graph.successors(title))
        if direction == "out"
        else list(graph.predecessors(title))
    )
    neighbours = sorted(neighbours, key=str)[:limit]
    if not neighbours:
        return "_none in the crawled graph_"
    lines = [f"- {name}" for name in neighbours]
    if len(neighbours) >= limit:
        lines.append(f"- _... ({graph.degree(title)} in total)_")
    return "\n".join(lines)


def render_details_panel(
    graph: nx.DiGraph,
    centralities: CentralityResult,
    title: str,
) -> None:
    """Render everything about the selected article.

    Parameters
    ----------
    graph
        The crawled web graph.
    centralities
        All centrality values of the graph.
    title
        Selected article; nothing is rendered when it is unknown to the graph.

    Notes
    -----
    The panel shows the article title and a link to Wikipedia, the raw
    in/out-degree, all eight centrality values with six decimals, their
    percentile ranks and a horizontal bar for visual comparison.
    """
    if title not in graph:
        st.info("Select an article in the graph, in the dropdown or in the table.")
        return
    attributes = graph.nodes[title]
    url = attributes.get("url", "")
    degrees = {
        "in": int(graph.in_degree(title)),
        "out": int(graph.out_degree(title)),
    }
    st.markdown(f"#### {title}")
    if url:
        st.markdown(f"[Open on Wikipedia]({url})")
    badges = [
        MEASURES[key].label
        for key, node in centralities.winners().items()
        if node == title
    ]
    if badges:
        st.success("Top-1 for: " + ", ".join(badges))
    columns = st.columns(4)
    columns[0].metric("BFS depth", int(attributes.get("depth", 0)))
    columns[1].metric("In-degree", degrees["in"])
    columns[2].metric("Out-degree", degrees["out"])
    columns[3].metric(
        "Nodes / edges", f"{graph.number_of_nodes()} / {graph.number_of_edges()}"
    )

    frame = measure_frame(centralities, title)
    st.dataframe(
        frame[["Measure", "Value", "Percentile", "Rank"]].style.format(
            {"Value": FLOAT_FORMAT, "Percentile": FLOAT_FORMAT}
        ),
        width="stretch",
        hide_index=True,
    )
    st.pyplot(measure_bar_figure(frame), width="stretch")
    st.caption(
        "Values are printed with six decimals. *Percentile* is the share of "
        "articles in this graph scoring strictly below this one, which makes "
        "the eight different scales comparable."
    )
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Links to ({degrees['out']})**")
        st.markdown(_neighbour_list(graph, title, "out"))
    with right:
        st.markdown(f"**Linked from ({degrees['in']})**")
        st.markdown(_neighbour_list(graph, title, "in"))
