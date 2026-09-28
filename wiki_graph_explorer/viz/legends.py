"""HTML colour legend and measure descriptions.

The legend explains, in one sentence each, what the eight centrality measures
capture and which colour marks the top-ranked article of that measure.  It is
rendered with ``st.markdown(..., unsafe_allow_html=True)`` so that the colour
swatches match the pyvis drawing exactly.
"""

from __future__ import annotations

import html
from typing import Mapping

import config
from graph.centrality import MEASURE_ORDER, MEASURES, CentralityResult

_CARD_STYLE = (
    "border-left:4px solid {color};background:#ffffff;padding:8px 10px;"
    "border-radius:6px;box-shadow:0 1px 2px rgba(16,24,40,0.06);"
    "margin-bottom:8px;"
)
_HEADER_STYLE = "font-weight:600;color:#111827;margin:0 0 2px 0;font-size:15px;"
_META_STYLE = "color:#6b7280;font-size:12px;margin:0 0 2px 0;"
_TEXT_STYLE = "color:#374151;font-size:13px;margin:0;"


def legend_html(centralities: CentralityResult | None = None) -> str:
    """Render the colour legend as an HTML fragment.

    Parameters
    ----------
    centralities
        Optional centrality values; when given, the top-ranked article of each
        measure is printed next to its colour.

    Returns
    -------
    str
        HTML fragment with one card per measure plus the "regular node" entry.

    Examples
    --------
    >>> html = legend_html()
    >>> "PageRank" in html
    True
    """
    winners: Mapping[str, str] = centralities.winners() if centralities else {}
    cards = []
    for key in MEASURE_ORDER:
        spec = MEASURES[key]
        winner = winners.get(key)
        winner_html = (
            f"<p class='wge-meta' style='{_META_STYLE}'>top-1: "
            f"<b>{html.escape(winner)}</b></p>"
            if winner
            else ""
        )
        cards.append(
            f"<div style='{_CARD_STYLE.format(color=spec.color)}'>"
            f"<div style='display:flex;align-items:center;gap:8px'>"
            f"<span style='width:16px;height:16px;border-radius:50%;"
            f"background:{spec.color};display:inline-block'></span>"
            f"<p style='{_HEADER_STYLE}'>{html.escape(spec.label)}</p></div>"
            f"<p style='{_META_STYLE}'><code>{html.escape(spec.formula)}</code> "
            f"&middot; {html.escape(spec.reference)}</p>"
            f"{winner_html}"
            f"<p style='{_TEXT_STYLE}'>{html.escape(spec.description)}</p>"
            f"</div>"
        )
    cards.append(
        f"<div style='{_CARD_STYLE.format(color=config.COLOR_REGULAR)}'>"
        f"<div style='display:flex;align-items:center;gap:8px'>"
        f"<span style='width:16px;height:16px;border-radius:50%;"
        f"background:{config.COLOR_REGULAR};display:inline-block'></span>"
        f"<p style='{_HEADER_STYLE}'>Regular article</p></div>"
        f"<p style='{_META_STYLE}'>node size &asymp; selected measure "
        f"(PageRank by default)</p>"
        f"<p style='{_TEXT_STYLE}'>Not ranked first by any of the eight "
        f"measures.</p></div>"
    )
    return (
        "<div style='font-family:-apple-system,Segoe UI,Roboto,Helvetica,"
        "Arial,sans-serif;'>" + "".join(cards) + "</div>"
    )


def measure_reference_markdown() -> str:
    """Return the bibliography of the centrality literature as Markdown.

    Returns
    -------
    str
        Markdown list of the six canonical references used in the docstrings.

    Examples
    --------
    >>> "Kleinberg" in measure_reference_markdown()
    True
    """
    return (
        "1. Brin, S. & Page, L. (1998). *The anatomy of a large-scale "
        "hypertextual information retrieval system.* Computer Networks and "
        "Systems, 30(1-7), 301-309.\n"
        "2. Kleinberg, J. M. (1999). *Authoritative sources in a hyperlinked "
        "environment.* Journal of the ACM, 46(5), 604-632.\n"
        "3. Newman, M. E. J. (2005). *Networks: An Introduction.* Oxford "
        "University Press.\n"
        "4. Boldi, P. & Vigna, C. (2014). *Axioms for centrality.* WWW '14, "
        "275-286.\n"
        "5. Langville, A. N. & Meyer, C. D. (2006). *Google's PageRank and Its "
        "Friends.* Now Publishers.\n"
        "6. Kempe, J., Kleinberg, J. & Tardos, E. (2003). *Maximizing the "
        "spread of influence through a social network.* KDD '03, 137-146."
    )
