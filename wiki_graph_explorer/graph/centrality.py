"""All centrality measures of the web graph.

Every measure is a *pure function* returning a dictionary that maps an article
title to a float, so the functions can be unit tested in isolation and reused
by the visualisation layer.  The mathematical definition of each measure is
repeated in the docstring because these docstrings are quoted verbatim in the
accompanying academic report.

Notation used throughout
------------------------
``G = (V, E)`` is a directed graph with ``N = |V|`` nodes and adjacency matrix
``A`` where ``A[i][j] = 1`` iff ``i -> j``.  ``d(i, j)`` is the length of a
shortest directed path from ``i`` to ``j`` (``inf`` when unreachable).

References
----------
.. [1] Brin, S., & Page, L. (1998). The anatomy of a large-scale hypertextual
       information retrieval system. *Computer Networks and Systems*,
       30(1-7), 301-309.
.. [2] Kleinberg, J. M. (1999). Authoritative sources in a hyperlinked
       environment. *Journal of the ACM*, 46(5), 604-632.
.. [3] Newman, M. E. J. (2005). *Networks: An Introduction*. Oxford
       University Press.
.. [4] Boldi, P., & Vigna, C. (2014). Axioms for centrality. In *Proceedings
       of the 21st International Conference on the World Wide Web (WWW)*
       (pp. 275-286). ACM.
.. [5] Langville, A. N., & Meyer, C. D. (2006). *Google's PageRank and Its
       Friends: A History of Major Algorithms and the Computational Power of
       the Web*. Now Publishers.
.. [6] Kempe, J., Kleinberg, J., & Tardos, E. (2003). Maximizing the spread of
       influence through a social network. In *Proceedings of the 9th ACM
       SIGKDD International Conference on Knowledge Discovery and Data Mining*
       (pp. 137-146). ACM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import networkx as nx
import numpy as np

import config

LOGGER = logging.getLogger("wge.graph.centrality")

# --------------------------------------------------------------------------
# Measure identifiers
# --------------------------------------------------------------------------
DEGREE_IN = "degree_in"
"""Key of the in-degree centrality measure."""

DEGREE_OUT = "degree_out"
"""Key of the out-degree centrality measure."""

BETWEENNESS = "betweenness"
"""Key of the (Brandes) betweenness centrality measure."""

CLOSENESS = "closeness"
"""Key of the harmonic closeness centrality measure."""

EIGENVECTOR = "eigenvector"
"""Key of the eigenvector centrality measure."""

PAGERANK = "pagerank"
"""Key of the PageRank measure."""

HITS_AUTHORITY = "hits_authority"
"""Key of the HITS authority score."""

HITS_HUB = "hits_hub"
"""Key of the HITS hub score."""

MEASURE_ORDER: tuple[str, ...] = (
    DEGREE_IN,
    DEGREE_OUT,
    BETWEENNESS,
    CLOSENESS,
    EIGENVECTOR,
    PAGERANK,
    HITS_AUTHORITY,
    HITS_HUB,
)
"""Canonical display / reporting order of the eight measures."""


@dataclass(frozen=True)
class MeasureSpec:
    """Static metadata of one centrality measure.

    Attributes
    ----------
    key
        Identifier used as dictionary key.
    label
        Human readable name shown in the UI.
    color
        Hex colour of the "winner" node for this measure.
    description
        One-sentence explanation used in the legend panel.
    formula
        Plain-text mathematical definition.
    reference
        Short citation key of the originating publication.
    """

    key: str
    label: str
    color: str
    description: str
    formula: str
    reference: str


MEASURES: dict[str, MeasureSpec] = {
    DEGREE_IN: MeasureSpec(
        key=DEGREE_IN,
        label="Degree centrality (in)",
        color=config.COLOR_DEGREE_IN,
        description=(
            "Articles that are linked to by many other articles; measures the "
            "popularity of an article as a target of citation."
        ),
        formula="C_D^in(i) = indegree(i) / (N - 1)",
        reference="Newman (2005)",
    ),
    DEGREE_OUT: MeasureSpec(
        key=DEGREE_OUT,
        label="Degree centrality (out)",
        color=config.COLOR_DEGREE_OUT,
        description=(
            "Articles that link to many other articles; measures how "
            "broadcasting / enumerative a page is."
        ),
        formula="C_D^out(i) = outdegree(i) / (N - 1)",
        reference="Newman (2005)",
    ),
    BETWEENNESS: MeasureSpec(
        key=BETWEENNESS,
        label="Betweenness centrality",
        color=config.COLOR_BETWEENNESS,
        description=(
            "Bridge articles that lie on many shortest paths; highlights "
            "hubs connecting otherwise separate parts of the graph."
        ),
        formula="C_B(i) = sum_{s != i != t} sigma_st(i) / sigma_st",
        reference="Brandes (2001), Newman (2005)",
    ),
    CLOSENESS: MeasureSpec(
        key=CLOSENESS,
        label="Closeness centrality (harmonic)",
        color=config.COLOR_CLOSENESS,
        description=(
            "Articles that are quickly reachable from (or reach) most other "
            "articles; generalises closeness to disconnected graphs."
        ),
        formula="C_H(i) = sum_{j != i} 1 / d(i, j)",
        reference="Boldi & Vigna (2014)",
    ),
    EIGENVECTOR: MeasureSpec(
        key=EIGENVECTOR,
        label="Eigenvector centrality",
        color=config.COLOR_EIGENVECTOR,
        description=(
            "Articles connected to other important articles; a page inherits "
            "centrality from its neighbourhood."
        ),
        formula="C_E(i) = x_i / sum(x),  A x = lambda x",
        reference="Newman (2005)",
    ),
    PAGERANK: MeasureSpec(
        key=PAGERANK,
        label="PageRank",
        color=config.COLOR_PAGERANK,
        description=(
            "Global importance derived from the recursive structure of the "
            "link graph and a random surfer model."
        ),
        formula="PR = alpha A^T D^-1 PR + (1 - alpha) 1/N",
        reference="Brin & Page (1998); Langville & Meyer (2006)",
    ),
    HITS_AUTHORITY: MeasureSpec(
        key=HITS_AUTHORITY,
        label="HITS authority",
        color=config.COLOR_HITS_AUTHORITY,
        description=(
            "Articles cited by the most authoritative articles, i.e. the "
            "best sources of the sub-web."
        ),
        formula="a = L^T L a",
        reference="Kleinberg (1999); Langville & Meyer (2006)",
    ),
    HITS_HUB: MeasureSpec(
        key=HITS_HUB,
        label="HITS hub",
        color=config.COLOR_HITS_HUB,
        description=(
            "Articles that point to many authorities, i.e. index / directory "
            "pages of the sub-web."
        ),
        formula="h = L L^T h",
        reference="Kleinberg (1999); Langville & Meyer (2006)",
    ),
}


# --------------------------------------------------------------------------
# 1. Degree centrality
# --------------------------------------------------------------------------
def degree_centralities(graph: nx.DiGraph) -> dict[str, dict[str, float]]:
    """In-degree and out-degree centrality of every node.

    For a directed graph with ``N`` nodes the degree centralities are
    Freeman's normalised degrees [1]_::

        C_D^in(i)  = indegree(i)  / (N - 1)
        C_D^out(i) = outdegree(i) / (N - 1)

    The ``1 / (N - 1)`` factor makes the in- and the out-degree comparable and
    maps a node linked to (or linking from) every other node to ``1.0``.

    Parameters
    ----------
    graph
        Directed web graph.

    Returns
    -------
    dict
        ``{"degree_in": {title: value}, "degree_out": {title: value}}`` with
        values in ``[0, 1]``.

    References
    ----------
    .. [1] Freeman, L. C. (1979). Centrality in social networks: Conceptual
           measurement. *Social Networks*, 1(3), 215-239.
    .. [2] Newman, M. E. J. (2005). *Networks: An Introduction*. Oxford
           University Press.
    """
    if graph.number_of_nodes() == 0:
        return {DEGREE_IN: {}, DEGREE_OUT: {}}
    return {
        DEGREE_IN: {
            str(node): float(value)
            for node, value in nx.in_degree_centrality(graph).items()
        },
        DEGREE_OUT: {
            str(node): float(value)
            for node, value in nx.out_degree_centrality(graph).items()
        },
    }


def raw_degrees(graph: nx.DiGraph) -> dict[str, dict[str, int]]:
    """Raw (un-normalised) in- and out-degree of every node.

    Parameters
    ----------
    graph
        Directed web graph.

    Returns
    -------
    dict
        ``{title: {"in": int, "out": int}}``.

    Examples
    --------
    >>> import networkx as nx
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


def _normalise_non_negative(scores: Mapping[Any, float]) -> dict[str, float]:
    """Turn a score vector into a clean non-negative probability distribution.

    HITS solves a symmetric eigenproblem whose solvers may return tiny negative
    values (``-0.0``) or an arbitrary overall sign; taking the absolute value
    and re-normalising keeps the reported numbers meaningful and reproducible
    without changing the ranking.

    Parameters
    ----------
    scores
        Mapping ``node -> score``.

    Returns
    -------
    dict
        ``{title: non-negative score}`` summing to ``1`` (or an empty dict for
        an empty input).

    Examples
    --------
    >>> _normalise_non_negative({"a": 2.0, "b": -2.0})
    {'a': 0.5, 'b': 0.5}
    """
    absolute = {str(node): abs(float(value)) for node, value in scores.items()}
    total = sum(absolute.values())
    if total == 0.0:
        uniform = 1.0 / len(absolute) if absolute else 0.0
        return {node: uniform for node in absolute}
    return {node: value / total for node, value in absolute.items()}


# --------------------------------------------------------------------------
# 2. Betweenness centrality
# --------------------------------------------------------------------------
def betweenness_centrality(
    graph: nx.DiGraph,
    normalized: bool = config.BETWEENNESS_NORMALIZED,
) -> dict[str, float]:
    """Betweenness centrality via Brandes' algorithm.

    For a node ``i`` the betweenness centrality is the fraction of all
    shortest paths between all *ordered* pairs of other nodes that pass
    through ``i``::

        C_B(i) = sum_{s != i != t}  sigma_st(i) / sigma_st

    where ``sigma_st`` is the number of shortest ``s -> t`` paths and
    ``sigma_st(i)`` the number of those paths that contain ``i``.  Brandes'
    algorithm accumulates this quantity in a single multi-source BFS sweep per
    source node, i.e. in ``O(N * M)`` time for unweighted graphs and
    ``O(N * (M + N log N))`` with a priority queue; the space complexity is
    ``O(N + M)`` [1]_.  Normalisation divides by the number of ordered pairs,
    which yields values in ``[0, 1]``.

    Parameters
    ----------
    graph
        Directed web graph.
    normalized
        Divide by the number of node pairs (NetworkX convention).

    Returns
    -------
    dict
        ``{title: betweenness}``.

    References
    ----------
    .. [1] Brandes, U. (2001). A faster algorithm for betweenness centrality.
           *Journal of Mathematical Sociology*, 25(2), 163-177.
    .. [2] Newman, M. E. J. (2005). *Networks: An Introduction*. Oxford
           University Press. (Chapter 13: centrality.)
    """
    if graph.number_of_nodes() < 3:
        return {str(node): 0.0 for node in graph.nodes}
    return {
        str(node): float(value)
        for node, value in nx.betweenness_centrality(
            graph, normalized=normalized, weight=None
        ).items()
    }


# --------------------------------------------------------------------------
# 3. Closeness / harmonic centrality
# --------------------------------------------------------------------------
def classical_closeness_centrality(graph: nx.DiGraph) -> dict[str, float]:
    """Classical closeness centrality on out-distances.

    ``C_C(i) = ( (1 / (N - 1)) * sum_{j != i} d(i, j) )^-1``

    The definition is only meaningful for **connected** graphs: as soon as some
    ``j`` is unreachable, ``d(i, j) = inf``, and with the Wasserman-Faust
    correction used by NetworkX the whole measure collapses to ``0`` [1]_.  A
    Wikipedia web graph is a forest of directed acyclic sub-graphs, so most
    articles have no path to most of the crawled sub-web and this measure
    degenerates; it is therefore only kept for reference and testing, while the
    application uses :func:`closeness_centrality`.

    Parameters
    ----------
    graph
        Directed web graph.

    Returns
    -------
    dict
        ``{title: closeness}`` with ``0.0`` for the nodes that cannot reach the
        rest of the graph.

    References
    ----------
    .. [1] Newman, M. E. J. (2005). *Networks: An Introduction*. Oxford
           University Press.
    .. [2] Wasserman, S., & Faust, K. (1995). Centrality measures in social
           networks. *Sociological Methodology*, 25, 239-271.
    """
    if graph.number_of_nodes() < 2:
        return {str(node): 0.0 for node in graph.nodes}
    return {
        str(node): float(value)
        for node, value in nx.closeness_centrality(graph.reverse()).items()
    }


def closeness_centrality(graph: nx.DiGraph) -> dict[str, float]:
    """Harmonic centrality, the robust closeness measure for disconnected graphs.

    Instead of averaging distances, the harmonic centrality sums the inverse
    distances::

        C_H(i) = sum_{j != i} 1 / d(i, j)

    Terms with ``d(i, j) = inf`` simply contribute ``0``, so the measure is
    well defined on *any* graph, connected or not.  It is a monotone transform
    of the classical closeness on connected graphs and, per Boldi & Vigna, it
    is the unique strictly monotone centrality on the class of
    ``gamma -> 1`` centralities satisfying an additivity axiom, which is why we
    use it for the web graph [1]_, [2]_.

    Direction convention: the sum runs over the *out*-distances of ``i``, i.e.
    "how fast can ``i`` reach the rest of the sub-web".  NetworkX implements the
    incoming direction (``sum_v 1 / d(v, u)``), so the graph is reversed before
    calling it.

    Parameters
    ----------
    graph
        Directed web graph.

    Returns
    -------
    dict
        ``{title: harmonic_centrality}`` with values in ``[0, N - 1]``.

    References
    ----------
    .. [1] Boldi, P., & Vigna, C. (2014). Axioms for centrality.
           In *Proceedings of the 21st International Conference on the World
           Wide Web (WWW)* (pp. 275-286). ACM.
    .. [2] Rochat, V. (2009). The notion of closeness centrality and related
           concepts. In *Encyclopedia of Social Network Analysis and Mining*
           (pp. 39-59). IGI Global.
    """
    if graph.number_of_nodes() < 2:
        return {str(node): 0.0 for node in graph.nodes}
    return {
        str(node): float(value)
        for node, value in nx.harmonic_centrality(graph.reverse()).items()
    }


# --------------------------------------------------------------------------
# 4. Eigenvector centrality
# --------------------------------------------------------------------------
def _power_iteration_eigenvector(
    matrix: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1.0e-14,
) -> np.ndarray:
    """Dominant eigenvector of a symmetric matrix by shifted power iteration.

    The plain iteration ``x_{k+1} = A x_k / ||A x_k||`` only converges when the
    dominant eigenvalue dominates the *moduli* of the others.  A symmetrised
    link graph is often bipartite, and then the spectrum is symmetric around
    zero (``lambda_i = -lambda_j``), so the iteration oscillates forever.  The
    standard fix is to add a shift ``s = max_i sum_j |A[i][j]| >= rho(A)`` (the
    maximum row sum dominates the spectral radius), which maps the spectrum to
    ``[s - rho, s + rho]``: the algebraically largest eigenvalue is now also the
    largest in modulus, the iteration converges, and - because the shift is a
    multiple of the identity - the eigenvectors are unchanged [1]_, [2]_.

    Parameters
    ----------
    matrix
        Square symmetric (adjacency) matrix as a NumPy array.
    max_iter
        Maximum number of iterations.
    tol
        Stopping tolerance on the L2 norm of the update.

    Returns
    -------
    numpy.ndarray
        Unit-norm dominant eigenvector (or the last iterate if the iteration
        does not converge, in which case a warning is logged).

    Notes
    -----
    The iteration is started from the all-ones vector, so for a non-negative
    matrix the iterates stay non-negative and the result is the Perron
    eigenvector, i.e. the unique dominant eigenvector of an irreducible
    non-negative matrix.

    References
    ----------
    .. [1] Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations*
           (4th ed.). Johns Hopkins University Press. (Chapter 8: the power
           method.)
    .. [2] Perron, O. (1909). Über die Eigenwerte einer Monothetischen
           Matrix. *Mathematische Annalen*, 64(1), 99-115.
    """
    size = matrix.shape[0]
    shift = float(np.abs(matrix).sum(axis=1).max())
    shifted = matrix + shift * np.eye(size)
    vector = np.full(size, 1.0 / np.sqrt(size), dtype=float)
    for _ in range(max_iter):
        product = shifted @ vector
        norm = float(np.linalg.norm(product))
        if norm == 0.0:
            return vector
        updated = product / norm
        if float(np.linalg.norm(updated - vector)) < tol:
            return updated
        vector = updated
    LOGGER.warning("Power iteration did not converge in %d steps", max_iter)
    return vector


def eigenvector_centrality(graph: nx.DiGraph) -> dict[str, float]:
    """Eigenvector centrality, normalised to sum to one.

    Find the dominant eigenvector ``x`` of the adjacency matrix, i.e. the
    solution of ``A x = lambda x`` for the largest ``lambda``, and report::

        C_E(i) = x_i / sum_j x_j

    ``x_i`` is large for nodes that are connected to *other* important nodes,
    which is why eigenvector centrality is the classic "reputation inherits
    reputation" measure; the final normalisation makes the vector a
    probability distribution, as in Kleinberg's original presentation [1]_,
    [2]_.  ``networkx.eigenvector_centrality_numpy`` is used first (dense
    LAPACK eigen-decomposition, ``O(N^3)``).  A crawled Wikipedia sub-graph is
    a forest of directed acyclic sub-graphs, whose adjacency matrix is
    *reducible* and usually has a non-unique dominant eigenvalue - NetworkX
    raises :class:`networkx.AmbiguousSolution` for exactly that case.  The
    fallback is therefore the hand-written power iteration on the symmetrised
    adjacency matrix ``A + A^T``: by the Perron-Frobenius theorem a symmetric
    non-negative matrix has a unique positive dominant eigenvector, which is
    the standard way of symmetrising the web graph for this measure
    (Newman, *Networks*, ch. 14).

    Parameters
    ----------
    graph
        Directed web graph.

    Returns
    -------
    dict
        ``{title: centrality}``; the values are non-negative and sum to ``1``.

    References
    ----------
    .. [1] Newman, M. E. J. (2005). *Networks: An Introduction*. Oxford
           University Press.
    .. [2] Langville, A. N., & Meyer, C. D. (2006). *Google's PageRank and Its
           Friends*. Now Publishers.
    """
    nodes = [str(node) for node in graph.nodes]
    if not nodes:
        return {}
    try:
        raw = nx.eigenvector_centrality_numpy(graph, max_iter=1000, weight=None)
    except (
        nx.AmbiguousSolution,
        nx.PowerIterationFailedConvergence,
        np.linalg.LinAlgError,
    ) as error:
        LOGGER.info(
            "Eigenvector centrality via symmetrised power iteration (%s)", error
        )
        matrix = nx.to_numpy_array(graph.to_undirected(as_view=False), weight=None)
        vector = _power_iteration_eigenvector(matrix)
        raw = dict(zip(nodes, (float(value) for value in vector), strict=False))
    total = float(sum(abs(value) for value in raw.values()))
    if total == 0.0:
        uniform = 1.0 / len(nodes)
        return {node: uniform for node in nodes}
    return {str(node): float(abs(value) / total) for node, value in raw.items()}


# --------------------------------------------------------------------------
# 5. PageRank
# --------------------------------------------------------------------------
def pagerank(
    graph: nx.DiGraph,
    alpha: float = config.PAGERANK_ALPHA,
    max_iter: int = config.PAGERANK_MAX_ITER,
    tol: float = config.PAGERANK_TOL,
) -> dict[str, float]:
    """PageRank of every node.

    PageRank is the stationary distribution of the random surfer model.  With
    adjacency matrix ``A``, out-degree matrix ``D`` and damping factor
    ``alpha`` the score vector solves the linear system [1]_, [2]_, [3]_::

        PR = alpha * A^T * D^-1 * PR + (1 - alpha) * (1/N) * 1

    which is the stationary form of the random walk

        PR(t+1) = alpha * A^T * D^-1 * PR(t) + (1 - alpha) * r,   r = 1/N

    where ``A^T D^-1`` is the column-stochastic transition matrix of a click
    (``A^T D^-1 PR`` spreads the rank of ``i`` over its out-neighbours) and the
    teleportation term ``(1 - alpha) r`` makes the chain irreducible, which
    guarantees a unique stationary distribution and damps the "rank sinks"
    effect.  ``alpha = 0.85`` is the value proposed by Brin & Page.

    Parameters
    ----------
    graph
        Directed web graph.
    alpha
        Damping factor (``0 < alpha < 1``).
    max_iter
        Maximum number of power iterations.
    tol
        Convergence tolerance of the L1 change between two iterations.

    Returns
    -------
    dict
        ``{title: pagerank}``; the values sum to ``1``.

    References
    ----------
    .. [1] Brin, S., & Page, L. (1998). The anatomy of a large-scale
           hypertextual information retrieval system. *Computer Networks and
           Systems*, 30(1-7), 301-309.
    .. [2] Page, L., et al. (1999). The PageRank citation ranking: bringing
           order to the Web. Technical report, Stanford InfoLab.
    .. [3] Langville, A. N., & Meyer, C. D. (2006). *Google's PageRank and Its
           Friends*. Now Publishers.
    """
    if graph.number_of_nodes() == 0:
        return {}
    return {
        str(node): float(value)
        for node, value in nx.pagerank(
            graph, alpha=alpha, max_iter=max_iter, tol=tol, weight=None
        ).items()
    }


# --------------------------------------------------------------------------
# 6. HITS
# --------------------------------------------------------------------------
def hits(
    graph: nx.DiGraph,
    max_iter: int = config.HITS_MAX_ITER,
    tol: float = 1.0e-8,
) -> dict[str, dict[str, float]]:
    """HITS hub and authority scores of every node.

    Let ``L`` be the adjacency matrix of the web graph.  HITS alternates
    between a hub update and an authority update [1]_, [2]_::

        a <- A^T a        (authority: "cited by good hubs")
        h <- A h          (hub:      "points at good authorities")

    In matrix form the two score vectors are the dominant eigenvectors of
    ``L^T L`` (authorities) and ``L L^T`` (hubs)::

        a = L^T L a        h = L L^T h

    Hubs and authorities are mutually recursive, which is why the measure
    separates *index* pages (hubs) from *authoritative* pages (authorities).
    NetworkX solves the underlying singular value decomposition
    ``L = U S V^T`` with ARPACK and reports the first left/right singular
    vectors, which is equivalent to the power iteration above.  Two arguments
    are passed explicitly for robustness and reproducibility: a uniform
    ``nstart`` (without it ARPACK draws a random start vector, so the scores
    would differ in the last floating point bits between runs) and
    ``normalized=False`` - NetworkX normalises by dividing by the sum of the
    vector, which is ``0`` for the (perfectly legitimate) singular vectors of a
    bipartite link graph, and the division is done here instead.

    Parameters
    ----------
    graph
        Directed web graph.
    max_iter
        Maximum number of power iterations (kept for API symmetry).
    tol
        Convergence tolerance.

    Returns
    -------
    dict
        ``{"hits_authority": {title: value}, "hits_hub": {title: value}}``;
        each sub-dictionary sums to ``1``.

    References
    ----------
    .. [1] Kleinberg, J. M. (1999). Authoritative sources in a hyperlinked
           environment. *Journal of the ACM*, 46(5), 604-632.
    .. [2] Langville, A. N., & Meyer, C. D. (2006). *Google's PageRank and Its
           Friends*. Now Publishers.
    .. [3] Kempe, J., Kleinberg, J., & Tardos, E. (2003). Maximizing the spread
           of influence through a social network. In *Proceedings of the 9th
           ACM SIGKDD International Conference on KDD* (pp. 137-146). ACM.
    """
    if graph.number_of_nodes() == 0:
        return {HITS_AUTHORITY: {}, HITS_HUB: {}}
    try:
        hubs, authorities = nx.hits(
            graph,
            max_iter=max_iter,
            tol=tol,
            normalized=False,
            nstart={node: 1.0 for node in graph.nodes},
        )
    except nx.PowerIterationFailedConvergence:  # pragma: no cover - defensive
        LOGGER.warning("HITS failed to converge; falling back to plain degrees")
        degrees = raw_degrees(graph)
        in_total = sum(value["in"] for value in degrees.values()) or 1
        out_total = sum(value["out"] for value in degrees.values()) or 1
        return {
            HITS_AUTHORITY: {
                node: value["in"] / in_total for node, value in degrees.items()
            },
            HITS_HUB: {
                node: value["out"] / out_total for node, value in degrees.items()
            },
        }
    return {
        HITS_AUTHORITY: _normalise_non_negative(authorities),
        HITS_HUB: _normalise_non_negative(hubs),
    }


# --------------------------------------------------------------------------
# Aggregation helpers
# --------------------------------------------------------------------------
def percentile_ranks(values: Mapping[str, float]) -> dict[str, float]:
    """Convert a score vector into percentile ranks inside the same graph.

    The percentile of a value ``v`` is the empirical fraction of nodes that
    score strictly below it::

        pct(i) = |{ j : value(j) < value(i) }| / (N - 1)

    This is a rank-based, scale-free way of comparing the eight measures with
    each other: PageRank lives in ``(0, 1)`` and cannot be compared to a raw
    betweenness sum, but their percentiles can.

    Parameters
    ----------
    values
        Mapping ``node -> score`` as returned by the measures above.

    Returns
    -------
    dict
        ``{node: percentile in [0, 1]}``; ``0.0`` for the lowest and ``1.0``
        for the highest scoring node (ties share the same percentile).

    Examples
    --------
    >>> percentile_ranks({"a": 1.0, "b": 2.0, "c": 3.0})
    {'a': 0.0, 'b': 0.5, 'c': 1.0}
    """
    if not values:
        return {}
    ordered = sorted(values.items(), key=lambda item: (item[1], str(item[0])))
    size = len(ordered)
    ranks: dict[str, float] = {}
    position = 0
    while position < size:
        end = position
        while end + 1 < size and ordered[end + 1][1] == ordered[position][1]:
            end += 1
        percentile = position / (size - 1) if size > 1 else 1.0
        for index in range(position, end + 1):
            ranks[ordered[index][0]] = float(percentile)
        position = end + 1
    return ranks


@dataclass
class CentralityResult:
    """Container with the eight centrality vectors of one graph.

    Attributes
    ----------
    measures
        Ordered mapping ``measure key -> {title: value}`` following
        :data:`MEASURE_ORDER`.
    node_count
        Number of nodes the values refer to.
    """

    measures: dict[str, dict[str, float]]
    node_count: int

    def values(self, measure_key: str) -> dict[str, float]:
        """Return the raw values of one measure.

        Parameters
        ----------
        measure_key
            One of :data:`MEASURE_ORDER`.

        Returns
        -------
        dict
            ``{title: value}`` (empty dict for an unknown key).
        """
        return self.measures.get(measure_key, {})

    def label_of(self, measure_key: str) -> str:
        """Return the display label of a measure key.

        Parameters
        ----------
        measure_key
            One of :data:`MEASURE_ORDER`.

        Returns
        -------
        str
            Human readable label, or the key itself when unknown.
        """
        spec = MEASURES.get(measure_key)
        return spec.label if spec else measure_key

    def percentiles(self, measure_key: str) -> dict[str, float]:
        """Return the percentile ranks of one measure.

        Parameters
        ----------
        measure_key
            One of :data:`MEASURE_ORDER`.

        Returns
        -------
        dict
            ``{title: percentile}``.
        """
        return percentile_ranks(self.values(measure_key))

    def top_node(self, measure_key: str) -> str | None:
        """Return the highest scoring node of a measure.

        Ties are broken alphabetically, which keeps the "winner" colour stable
        across runs.

        Parameters
        ----------
        measure_key
            One of :data:`MEASURE_ORDER`.

        Returns
        -------
        str or None
            The winning title, or ``None`` for an empty graph.
        """
        values = self.values(measure_key)
        if not values:
            return None
        return min(values.items(), key=lambda item: (-item[1], str(item[0])))[0]

    def winners(self) -> dict[str, str]:
        """Return the winning node of every measure.

        Returns
        -------
        dict
            ``{measure key: title}``; empty for a graph without nodes.
        """
        return {
            key: winner
            for key in MEASURE_ORDER
            if (winner := self.top_node(key)) is not None
        }

    def max_value(self, measure_key: str) -> float:
        """Return the largest value attained by a measure.

        Parameters
        ----------
        measure_key
            One of :data:`MEASURE_ORDER`.

        Returns
        -------
        float
            Maximum value, or ``0.0`` for an empty measure.
        """
        values = self.values(measure_key)
        return max(values.values()) if values else 0.0

    def as_frame(self) -> "Any":
        """Return the measures as a :class:`pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            One row per article (sorted by title) and one column per measure,
            ready to be displayed with ``st.dataframe``.  Columns keep the
            display labels, values are raw floats.
        """
        import pandas as pd

        titles = sorted(
            {title for values in self.measures.values() for title in values}
        )
        data: dict[str, list[float]] = {
            MEASURES[key].label: [
                float(self.measures[key].get(title, 0.0)) for title in titles
            ]
            for key in MEASURE_ORDER
            if key in self.measures
        }
        frame = pd.DataFrame(data, index=pd.Index(titles, name="Article"))
        return frame


def compute_all_centralities(
    graph: nx.DiGraph,
    alpha: float = config.PAGERANK_ALPHA,
) -> CentralityResult:
    """Compute all eight centrality measures for ``graph``.

    Parameters
    ----------
    graph
        Directed web graph built by :func:`graph.builder.build_digraph`.
    alpha
        PageRank damping factor.

    Returns
    -------
    CentralityResult
        Every measure, keyed by the identifiers in :data:`MEASURE_ORDER`.

    Examples
    --------
    >>> import networkx as nx
    >>> result = compute_all_centralities(nx.DiGraph([("A", "B"), ("B", "C")]))
    >>> result.top_node(PAGERANK) in {"A", "B", "C"}
    True
    """
    measures: dict[str, dict[str, float]] = {}
    degrees = degree_centralities(graph)
    measures[DEGREE_IN] = degrees[DEGREE_IN]
    measures[DEGREE_OUT] = degrees[DEGREE_OUT]
    measures[BETWEENNESS] = betweenness_centrality(graph)
    measures[CLOSENESS] = closeness_centrality(graph)
    measures[EIGENVECTOR] = eigenvector_centrality(graph)
    measures[PAGERANK] = pagerank(graph, alpha=alpha)
    measures.update(hits(graph))
    ordered = {key: measures[key] for key in MEASURE_ORDER if key in measures}
    return CentralityResult(measures=ordered, node_count=graph.number_of_nodes())


def measure_keys(labels: Iterable[str] = MEASURE_ORDER) -> list[str]:
    """Return the measure keys, optionally reordered/filtered by label.

    Parameters
    ----------
    labels
        Display labels in the desired order.

    Returns
    -------
    list of str
        Matching measure keys.

    Examples
    --------
    >>> measure_keys(["PageRank"])
    ['pagerank']
    """
    by_label = {spec.label: key for key, spec in MEASURES.items()}
    return [by_label[label] for label in labels if label in by_label]


def normalise_sizes(
    values: Mapping[str, float],
    size_min: float = config.NODE_SIZE_MIN,
    size_max: float = config.NODE_SIZE_MAX,
) -> dict[str, float]:
    """Min-max rescale a score vector into the pyvis node-size range.

    Parameters
    ----------
    values
        Mapping ``node -> score``.
    size_min
        Smallest drawn size.
    size_max
        Largest drawn size.

    Returns
    -------
    dict
        ``{node: size}`` in ``[size_min, size_max]``.  When all values are equal
        (or there is a single node) every node gets the mean of the two bounds.

    Examples
    --------
    >>> normalise_sizes({"a": 0.0, "b": 1.0}, 10, 20)
    {'a': 10.0, 'b': 20.0}
    """
    if not values:
        return {}
    smallest = min(values.values())
    largest = max(values.values())
    if largest - smallest <= 0:
        constant = (size_min + size_max) / 2.0
        return {node: constant for node in values}
    span = size_max - size_min
    return {
        node: size_min + span * (value - smallest) / (largest - smallest)
        for node, value in values.items()
    }


def build_callback_map(
    measures: Mapping[str, Mapping[str, float]],
) -> dict[str, Callable[[str], float]]:
    """Return ``{measure key: accessor}`` for templating / debugging.

    Parameters
    ----------
    measures
        Mapping ``measure key -> {node: value}``.

    Returns
    -------
    dict
        ``{measure key: function(node) -> value}``.
    """
    return {
        key: (lambda node, values=values: float(values.get(node, 0.0)))
        for key, values in measures.items()
    }
