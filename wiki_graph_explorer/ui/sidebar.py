"""Sidebar controls: article input, crawl parameters and the build button."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import streamlit as st

import config
from crawler.bfs import node_budget_for_depth
from crawler.wiki_api import ArticleRef, parse_article_input
from viz.legends import legend_html, measure_reference_markdown

LOGGER = logging.getLogger("wge.ui.sidebar")

DEFAULT_INPUT = "https://en.wikipedia.org/wiki/Information_retrieval"
"""Pre-filled example so that the demo can be launched with a single click."""


@dataclass(frozen=True)
class CrawlRequest:
    """Validated user request for a new crawl.

    Attributes
    ----------
    ref
        Resolved root article (language + canonical title).
    depth
        Maximum BFS depth.
    max_children
        Maximum number of out-links kept per article.
    """

    ref: ArticleRef
    depth: int
    max_children: int


def render_controls(default_input: str = DEFAULT_INPUT) -> CrawlRequest | None:
    """Render the sidebar widgets and return a request when "Build graph" is hit.

    Parameters
    ----------
    default_input
        Pre-filled value of the URL text field.

    Returns
    -------
    CrawlRequest or None
        The validated request, or ``None`` when the button was not pressed or
        the input could not be parsed (an error is shown in place).

    Notes
    -----
    The widgets are keyed so that Streamlit keeps their state across reruns
    caused by a node selection; the crawl itself only runs when the button is
    pressed, which keeps the API politeness delay under user control.
    """
    with st.sidebar:
        st.markdown("### Crawl parameters")
        raw_input = st.text_input(
            "Wikipedia article URL",
            value=default_input,
            key="wge_input",
            help=(
                "Full URL (https://en.wikipedia.org/wiki/Information_retrieval), "
                "'ru:Информационный_поиск' or a bare title."
            ),
        )
        depth = int(
            st.number_input(
                "Depth",
                min_value=config.MIN_DEPTH,
                max_value=config.MAX_DEPTH,
                value=config.DEFAULT_DEPTH,
                step=1,
                key="wge_depth",
                help="BFS depth: 1 = the article and its direct links.",
            )
        )
        with st.expander("Advanced", expanded=False):
            max_children = int(
                st.number_input(
                    "Max children per node",
                    min_value=1,
                    max_value=config.MAX_MAX_CHILDREN,
                    value=config.DEFAULT_MAX_CHILDREN,
                    step=1,
                    key="wge_children",
                    help=(
                        "Upper bound on the out-links kept per article "
                        "(alphabetically first links are kept, which makes the "
                        "graph reproducible)."
                    ),
                )
            )
        budget = node_budget_for_depth(depth)
        st.caption(
            f"Node budget for depth {depth}: **{budget}** articles "
            f"(API calls &asymp; {min(budget, (max_children + 1) ** depth)})."
        )
        if depth == config.MAX_DEPTH:
            st.warning(
                f"Depth {depth} crawls up to {budget} articles; the crawl is "
                "truncated once the budget is reached.",
                icon="⚠️",
            )
        pressed = st.button("Build graph", type="primary", use_container_width=True)

    if not pressed:
        return None
    try:
        ref = parse_article_input(raw_input)
    except ValueError as error:
        st.error(str(error))
        return None
    LOGGER.info(
        "Crawl requested: %s (depth=%d, max_children=%d)", ref, depth, max_children
    )
    return CrawlRequest(ref=ref, depth=depth, max_children=max_children)


def render_legend_panel(centralities=None) -> None:
    """Render the colour legend and the bibliography in the sidebar.

    Parameters
    ----------
    centralities
        Optional :class:`graph.centrality.CentralityResult` used to print the
        top-ranked article of every measure.
    """
    with st.sidebar:
        with st.expander("Legend: what each colour means", expanded=False):
            st.markdown(legend_html(centralities), unsafe_allow_html=True)
        with st.expander("References", expanded=False):
            st.markdown(measure_reference_markdown())
