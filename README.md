# Wiki Graph Centrality Explorer

A containerised **Streamlit** application that crawls a small subtree of
Wikipedia through the MediaWiki API and applies the classic *Social Network
Analysis* centrality measures to the resulting **web graph**
(`u -> v`  <=>  "article *u* hyperlinks to article *v*").

It is built as a live demo for the academic report *"Social Network Analysis
methods in Web information retrieval"*: every measure is a separate, pure,
fully documented function whose docstring contains the mathematical definition
and the original reference, so the text of `graph/centrality.py` can be quoted
in the paper.

---

## 1. Features

| Requirement | Implementation |
| --- | --- |
| Article input | Full URL (`https://ru.wikipedia.org/wiki/Информационный_поиск`), `lang:Title`, `lang Title` or a bare title |
| Crawl control | `Depth` (1-3) and `Max children per node` (1-60) widgets + **Build graph** button |
| Live progress | `st.progress` + status line `X pages fetched, Y queued` |
| Interactive graph | `pyvis` (`Network(notebook=False, cdn_resources="in_line")`) inside `st.components.v1.html` |
| Node selection | Click a node **or** use the `<select>` under the graph **or** the searchable table |
| Details panel | Title + Wikipedia link, in/out-degree, all 8 measures with 6 decimals, percentile bars |
| Measures | Degree (in/out), betweenness, closeness (harmonic), eigenvector, PageRank, HITS (authority/hub) |
| Colours | The top-1 article of every measure gets that measure's colour, everything else is light grey |
| Reproducibility | Sorted (alphabetical) out-links, FIFO BFS, fixed solver start vectors -> bit-identical results |
| Politeness | `User-Agent`, 0.1 s delay, retry/backoff on HTTP 429, on-disk JSON cache |

---

## 2. Quick start

```bash
git clone https://github.com/neF1anders/web-measures.git
cd wiki_graph_explorer/
pip install -r requirements.txt
python app.py
```

`app.py` is a *dual mode* entry point: under plain `python app.py` it starts
`streamlit run app.py --server.port $STREAMLIT_PORT` (default `8080`,
`0.0.0.0`, headless); when it is executed by Streamlit itself it renders the
UI. Nothing else is needed - the deps are already in the image.

Run the tests (all of them, no network access required):

```bash
python -m pytest -v"
```

---

## 3. Architecture

```
                    +---------------------------------------------------+
   browser  ----->  |  app.py            (thin orchestrator, <200 lines)  |
                    |                                                   |
                    |  ui/sidebar.py ......... input, depth, max_children |
                    |  ui/details_panel.py ... measure table + bar chart  |
                    +--------------------------+------------------------+
                                               |
             +---------------------------------+---------------------------------+
             |                                                                   |
   +---------v----------+  +----------------+  +-------------------+  +--------v--------+
   | crawler/wiki_api.py|  | crawler/bfs.py |  | graph/builder.py  |  | viz/pyvis_render |
   | MediaWiki api.php  |->| deterministic |->| networkx.DiGraph  |->| pyvis + picker   |
   | 429 retry, cache/  |  | BFS, budgets   |  | node/edge attrs   |  | viz/legends.py   |
   | User-Agent, delay  |  | progress cbs   |  +---------+---------+  +-----------------+
   +--------------------+  +----------------+            |
                                                        |  graph/centrality.py
                                                        |  8 pure functions,
                                                        |  NumPy-style docstrings
                                                        v
                                          +-------------------------------+
                                          | config.py   constants         |
                                          | logging -> logs/app.log      |
                                          | cache/    -> cache/*.json    |
                                          +-------------------------------+
```

Data flow of one **Build graph** click:

1. `ui.sidebar.render_controls` -> `CrawlRequest(lang, title, depth, max_children)`.
2. `crawler.wiki_api.WikiApiClient.resolve_title` (follows redirects).
3. `crawler.bfs.build_crawl` -> FIFO BFS, at most `max_children` out-links per
   article, stops at `depth` or the node budget -> `CrawlResult`.
4. `graph.builder.build_digraph` -> `networkx.DiGraph` (links outside the
   crawled subtree are dropped).
5. `graph.centrality.compute_all_centralities` -> `CentralityResult` with the
   eight measures, cached in `st.session_state`.
6. `viz.pyvis_render.render_html` -> HTML document, `st.components.v1.html`.
7. `ui.details_panel.render_details_panel` -> numbers, percentiles, bar chart.

---

## 4. File tree

```
wiki_graph_explorer/
├── app.py                      # Streamlit entry point + port launcher
├── config.py                   # constants: API, limits, colours, ports
├── crawler/
│   ├── __init__.py
│   ├── wiki_api.py             # MediaWiki client: fetch_links, resolve_title, cache, retry
│   └── bfs.py                  # deterministic BFS, node budgets, progress callback
├── graph/
│   ├── __init__.py
│   ├── builder.py              # DiGraph construction, degrees, table rows
│   └── centrality.py           # the 8 measures + percentiles (pure functions)
├── viz/
│   ├── __init__.py
│   ├── pyvis_render.py         # node sizing, winner colours, node picker
│   └── legends.py              # colour legend + bibliography HTML
├── ui/
│   ├── __init__.py
│   ├── sidebar.py              # controls
│   └── details_panel.py        # selected article details
├── tests/
│   ├── test_crawler.py         # mocked HTTP, depth/max_children/determinism
│   ├── test_centrality.py      # analytic values on small graphs
│   ├── test_builder.py         # graph, pyvis, legend, details table
│   └── test_app_smoke.py       # AppTest end-to-end run of app.py
├── requirements.txt
└── README.md
```

Runtime artefacts (created on first use, safe to delete):
`cache/{lang}_{sha1(title)}.json` (MediaWiki cache) and `logs/app.log`.

---

## 5. The eight measures

| # | Measure | Formula (see the docstrings) | Reference | Winner colour |
| --- | --- | --- | --- | --- |
| 1 | Degree (in) | `C_D^in(i) = indegree(i) / (N - 1)` | Freeman 1979; Newman 2005 | red `#e41a1c` |
| 2 | Degree (out) | `C_D^out(i) = outdegree(i) / (N - 1)` | Freeman 1979; Newman 2005 | orange `#ff7f00` |
| 3 | Betweenness | `C_B(i) = sum sigma_st(i) / sigma_st` | Brandes 2001; Newman 2005 | purple `#984ea3` |
| 4 | Closeness (harmonic) | `C_H(i) = sum_{j != i} 1 / d(i, j)` | Boldi & Vigna 2014 | green `#4daf4a` |
| 5 | Eigenvector | `A x = lambda x`, `C_E(i) = x_i / sum(x)` | Newman 2005; Langville & Meyer 2006 | brown `#a65628` |
| 6 | PageRank | `PR = alpha A^T D^-1 PR + (1 - alpha) 1/N` | Brin & Page 1998; Langville & Meyer 2006 | blue `#377eb8` |
| 7 | HITS authority | `a = L^T L a` | Kleinberg 1999; Kempe et al. 2003 | magenta `#e377c2` |
| 8 | HITS hub | `h = L L^T h` | Kleinberg 1999; Kempe et al. 2003 | cyan `#17becf` |

`alpha = 0.85`, HITS `max_iter = 100`, betweenness normalised (Brandes'
`O(N*M)` for unweighted graphs), PageRank `max_iter = 300`, `tol = 1e-10`.

### What each measure highlights in a Wikipedia graph

* **Degree (in)** - *popularity / citation*: overview and survey articles that
  many sub-topic pages point at (`Apache Accumulo` in the worked example).
* **Degree (out)** - *broadcasting*: overview and navigational pages that
  enumerate the field (`Academic freedom`, a page that links out to everything
  in scope).
* **Betweenness** - *bridges between communities*: articles that connect
  otherwise separate clusters (information retrieval, statistics, NLP); in the
  example the root article itself, which is the only link between the
  "theoretical IR" and the "IR systems" clusters.
* **Closeness (harmonic)** - *gatekeepers / navigation hubs*: articles that
  reach most of the sub-web quickly; because the graph is a DAG the harmonic
  form is the only one that is defined for the sinks as well.
* **Eigenvector** - *reputation inherits reputation*: pages that are linked
  from other important pages (`Apache Lucene`, linked from every IR-system
  page, so it inherits the rank of its neighbours).
* **PageRank** - *global importance*: the same class as eigenvector but with
  the random-surfer damping, so long "citation chains" (`ArXiv`) can still
  score high even with a low in-degree.
* **HITS authority** - *best sources*: pages cited by many hubs; identical to
  the in-degree ranking on a graph this small, which is a useful observation
  for the report.
* **HITS hub** - *index / directory pages*: pages that point at many
  authorities (`Apache Lucene` again - a search-engine product page that
  points at dozens of IR sources).

### Worked example (measured, `Information retrieval`, depth 2, `max_children` 15)

152 articles, 170 links, weakly connected:

| Measure | top-1 article | value |
| --- | --- | --- |
| Degree (in) | Apache Accumulo | 0.013245 |
| Degree (out) | Academic freedom | 0.099338 |
| Betweenness | Information retrieval | 0.006004 |
| Closeness (harmonic) | Information retrieval | 83.000000 |
| Eigenvector | Apache Lucene | 0.052231 |
| PageRank | ArXiv | 0.011720 |
| HITS authority | Apache Accumulo | 0.066667 |
| HITS hub | Apache Lucene | 0.500000 |

---

## 6. Design decisions (documented choices, as requested)

1. **Deterministic out-links.** The MediaWiki API returns `prop=links` sorted
   by title, and the crawler additionally sorts and de-duplicates, then keeps
   the first `max_children` titles. Same input -> same graph, always.
2. **Link-list pagination.** At most `MAX_LINK_PAGES = 4` (2 000 links) are
   read per article. Because the API sorts by title, the kept links are the
   alphabetical prefix; for a hub with more than 2 000 links the selection is
   therefore *not* a uniform sample of its out-neighbourhood (caveat 6.2).
3. **Namespace filter.** `plnamespace=0` keeps only main-namespace articles;
   files, categories, templates and talk pages never enter the queue.
4. **Disambiguation / fragments.** Titles ending in `(disambiguation)` and
   titles starting with `#` are dropped heuristically. Redirects are *not*
   resolved per link (that would double the number of API calls); link targets
   therefore keep the title as written in the source article, which is the
   standard web-graph convention and the reason `1890 US census` and
   `1890 United States census` can both appear.
5. **Harmonic direction.** `nx.harmonic_centrality` in NetworkX >= 3.6 sums the
   *in*-distances (`sum_v 1/d(v,u)`); the project implements the formula of the
   task, i.e. the *out*-distances, by reversing the graph before the call. The
   direction is documented in the docstring.
6. **Eigenvector on a reducible matrix.** A crawled Wikipedia sub-graph is a
   forest of DAGs, so its adjacency matrix is reducible and NetworkX raises
   `AmbiguousSolution`. The fallback is a hand-written, shifted power iteration
   on the symmetrised matrix `A + A^T`, which converges for bipartite graphs
   too (the shift is a multiple of the identity, so the eigenvectors are
   unchanged) and matches `numpy.linalg.eigh` to machine precision.
7. **HITS start vector and normalisation.** `nx.hits` uses ARPACK with a
   *random* start vector unless `nstart` is given, and it normalises by
   dividing by the sum of the singular vector, which is `0` for the perfectly
   legitimate dominant singular vector of a bipartite link graph (NetworkX
   then returns `inf`/`nan` and prints a `RuntimeWarning`). Both problems are
   removed by passing a uniform `nstart` and `normalized=False`; the
   normalisation is done here instead. Two crawls of the same article now
   produce bit-identical values for all eight measures.
8. **Node budgets.** 60 / 200 / 1 000 articles for depth 1 / 2 / 3. Depth 2
   with `max_children = 15` stays below the 200-article target (152 measured
   above). Exceeding a budget sets a flag and shows a `st.warning`.
9. **Node selection & iframe round trip.** A `pyvis` network lives inside a
   `st.components.v1.html` iframe and cannot call Python directly, so the
   injected `<select>` (and a click on a node) writes the title into the
   `node` query parameter of the parent page and dispatches `popstate`; the
   script reads it back at the beginning of the next run
   (`app._sync_selection`). Two guaranteed paths exist as well and are the ones
   to use if a browser blocks the round trip: the Streamlit **Inspect article**
   dropdown and the row selection of the article table.
10. **Percentiles.** The eight measures live on eight different scales, so
    every value is additionally reported as the percentile of the article
    inside the current graph: `|{j : value(j) < value(i)}| / (N - 1)`.

---

## 7. Tests

```bash
python -m pytest -v"
# 69 passed
```

* `tests/test_crawler.py` - 16 tests: input parsing (7 URL/title formats),
  cache write/read, HTTP 429 retry, missing article, BFS depth limit,
  `max_children` cap, node budget, determinism, progress counters.
  All HTTP is mocked with a `FakeSession`; no network access.
* `tests/test_centrality.py` - 22 tests with **analytic** expectations:
  `a -> b -> c -> d` has betweenness `1/3` for the inner nodes, degree
  `1/3`; the out-star has PageRank maximum at the *leaves* but HITS hub
  maximum at the centre; the harmonic sum equals `1 + 1/2 + 1/3`; the
  eigenvector fallback is triggered and still normalised; computing all
  measures twice yields **bit-identical** values (no tolerance needed).
* `tests/test_builder.py` - 17 tests: node attributes, out-of-subtree links
  dropped, table ordering, pyvis payload (size/colour/tooltip), HTML escaping
  of article titles, legend completeness, details table and matplotlib figure.
* `tests/test_app_smoke.py` - 4 end-to-end tests with
  `streamlit.testing.v1.AppTest`: idle state, build graph -> graph + table +
  details panel, invalid input -> visible error, changing the inspected
  article. The MediaWiki client is monkeypatched, so the suite is offline.

---

## 8. Caveats

1. **Wikipedia API etiquette.** The client sends a descriptive `User-Agent`,
   waits 0.1 s between calls and retries with exponential backoff. Keep the
   depth small; a depth-3 crawl is ~1 000 requests.
2. **Alphabetical prefix selection.** See 6.2 - for articles with more than
   2 000 links the kept neighbours are the alphabetically first ones, which
   biases fan-out towards early letters.
3. **The graph is a sample, not the web.** With `max_children = 15` the
   in/out degrees are truncated, so raw degrees are *lower bounds*. Centrality
   values describe the crawled sub-graph only; they are comparable *within* one
   graph (hence the percentile display), not across crawls.
4. **Disconnected and acyclic structure.** Wikipedia sub-graphs are mostly
   DAGs; the classical closeness centrality degenerates (0 for every sink),
   which is exactly why the harmonic variant is used.
5. **Size of the matrix.** Eigenvector centrality is `O(N^3)`; with the
   1 000-article cap the dense solve takes a few seconds at most.
6. **Rendering.** The pyvis document is fully self-contained
   (`cdn_resources="in_line"`, ~1 MB) but the pyvis template still links
   Bootstrap from a CDN for the (unused) built-in menus; without internet the
   graph itself still renders, only the menu styling is missing.
7. **Cache and logs** grow on disk (`cache/`, `logs/app.log`); both are safe
   to delete.

---

## 9. References

1. Brin, S. & Page, L. (1998). *The anatomy of a large-scale hypertextual
   information retrieval system.* Computer Networks and Systems, 30(1-7),
   301-309.
2. Kleinberg, J. M. (1999). *Authoritative sources in a hyperlinked
   environment.* Journal of the ACM, 46(5), 604-632.
3. Newman, M. E. J. (2005). *Networks: An Introduction.* Oxford University
   Press.
4. Boldi, P. & Vigna, C. (2014). *Axioms for centrality.* WWW '14, 275-286.
5. Langville, A. N. & Meyer, C. D. (2006). *Google's PageRank and Its
   Friends.* Now Publishers.
6. Kempe, J., Kleinberg, J. & Tardos, E. (2003). *Maximizing the spread of
   influence through a social network.* KDD '03, 137-146.
7. Brandes, U. (2001). *A faster algorithm for betweenness centrality.*
   Journal of Mathematical Sociology, 25(2), 163-177.
8. Freeman, L. C. (1979). *Centrality in social networks: Conceptual
   measurement.* Social Networks, 1(3), 215-239.
