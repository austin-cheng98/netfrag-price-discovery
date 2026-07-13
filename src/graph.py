from __future__ import annotations
import sys
from typing import Any, Optional
import numpy as np
import pandas as pd
import networkx as nx
sys.path.insert(0, '/Users/austincheng/Desktop/netfrag-price-discovery/src')
from contracts import RedditItem, GraphSnapshot, FragmentationScore
from config import CFG
try:
    import igraph as _ig
    import leidenalg as _la
    _HAS_LEIDEN = True
except Exception:
    _HAS_LEIDEN = False
_INVALID_AUTHORS = {'', '[deleted]', '[removed]', 'AutoModerator', None}

def _valid_author(a: Optional[str]) -> bool:
    return a is not None and a not in _INVALID_AUTHORS

def _fullname(item: RedditItem) -> str:
    iid = item.id or ''
    if iid.startswith(('t1_', 't3_')):
        return iid
    prefix = 't3_' if item.kind == 'submission' else 't1_'
    return f'{prefix}{iid}'

def _undirected_projection(G: nx.Graph) -> nx.Graph:
    if not G.is_directed():
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        for u, v, d in G.edges(data=True):
            if u == v:
                continue
            w = float(d.get('weight', 1.0))
            if H.has_edge(u, v):
                H[u][v]['weight'] += w
            else:
                H.add_edge(u, v, weight=w)
        return H
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        if u == v:
            continue
        w = float(d.get('weight', 1.0))
        if H.has_edge(u, v):
            H[u][v]['weight'] += w
        else:
            H.add_edge(u, v, weight=w)
    return H

def _largest_cc_subgraph(H: nx.Graph) -> nx.Graph:
    if H.number_of_nodes() == 0:
        return H
    comps = list(nx.connected_components(H))
    if not comps:
        return H
    largest = max(comps, key=len)
    return H.subgraph(largest).copy()

def detect_communities(G) -> dict[str, int]:
    if G is None:
        return {}
    H = _undirected_projection(G) if G.is_directed() else _undirected_projection(G)
    nodes = list(H.nodes())
    if len(nodes) == 0:
        return {}
    if len(nodes) == 1:
        return {nodes[0]: 0}
    if H.number_of_edges() == 0:
        return {n: i for i, n in enumerate(nodes)}
    if _HAS_LEIDEN:
        try:
            idx = {n: i for i, n in enumerate(nodes)}
            edges = []
            weights = []
            for u, v, d in H.edges(data=True):
                edges.append((idx[u], idx[v]))
                weights.append(float(d.get('weight', 1.0)))
            g = _ig.Graph(n=len(nodes), edges=edges, directed=False)
            part = _la.find_partition(g, _la.ModularityVertexPartition, weights=weights, seed=int(CFG.seed))
            membership = part.membership
            return {nodes[i]: int(membership[i]) for i in range(len(nodes))}
        except Exception:
            pass
    try:
        comms = nx.community.louvain_communities(H, weight='weight', seed=int(CFG.seed))
    except Exception:
        comms = nx.community.greedy_modularity_communities(H, weight='weight')
    labels: dict[str, int] = {}
    for cid, com in enumerate(comms):
        for n in com:
            labels[n] = cid
    nxt = len(comms)
    for n in nodes:
        if n not in labels:
            labels[n] = nxt
            nxt += 1
    return labels

def _add_reply_edges(G: nx.DiGraph, items: list[RedditItem]) -> None:
    id_to_author: dict[str, str] = {}
    for it in items:
        if _valid_author(it.author):
            id_to_author[_fullname(it)] = it.author
    for it in items:
        child = it.author
        if not _valid_author(child):
            continue
        pid = it.parent_id
        if not pid:
            continue
        parent_author = id_to_author.get(pid)
        if not _valid_author(parent_author):
            continue
        if parent_author == child:
            continue
        if G.has_edge(child, parent_author):
            G[child][parent_author]['weight'] += 1.0
        else:
            G.add_edge(child, parent_author, weight=1.0)

def _add_thread_edges(G: nx.DiGraph, items: list[RedditItem]) -> None:
    threads: dict[str, list[str]] = {}
    for it in items:
        if not _valid_author(it.author):
            continue
        link = it.link_id or (_fullname(it) if it.kind == 'submission' else None)
        if not link:
            continue
        seq = threads.setdefault(link, [])
        if it.author not in seq:
            seq.append(it.author)
    for link, authors in threads.items():
        if len(authors) < 2:
            continue
        for a, b in zip(authors[:-1], authors[1:]):
            if a == b:
                continue
            for u, v in ((a, b), (b, a)):
                if G.has_edge(u, v):
                    G[u][v]['weight'] += 1.0
                else:
                    G.add_edge(u, v, weight=1.0)

def _author_mean_embeddings(items: list[RedditItem], authors: list[str], embedder) -> Optional[np.ndarray]:
    if embedder is None:
        return None
    by_author: dict[str, list[str]] = {a: [] for a in authors}
    for it in items:
        if _valid_author(it.author) and it.author in by_author:
            txt = (it.body or '').strip()
            if txt:
                by_author[it.author].append(txt)
    flat_texts: list[str] = []
    slices: dict[str, tuple[int, int]] = {}
    for a in authors:
        msgs = by_author.get(a, [])
        start = len(flat_texts)
        flat_texts.extend(msgs)
        slices[a] = (start, len(flat_texts))
    if not flat_texts:
        return None
    try:
        emb = np.asarray(embedder.encode(flat_texts), dtype=np.float64)
    except Exception:
        return None
    if emb.ndim != 2 or emb.shape[0] != len(flat_texts):
        return None
    dim = emb.shape[1]
    out = np.zeros((len(authors), dim), dtype=np.float64)
    for i, a in enumerate(authors):
        s, e = slices[a]
        if e > s:
            m = emb[s:e].mean(axis=0)
        else:
            m = np.zeros(dim)
        n = np.linalg.norm(m)
        out[i] = m / n if n > 0 else m
    return out

def _add_semantic_edges(G: nx.DiGraph, authors: list[str], emb: np.ndarray, threshold: float) -> None:
    if emb is None or emb.shape[0] != len(authors) or len(authors) < 2:
        return
    sims = emb @ emb.T
    n = len(authors)
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sims[i, j])
            if s > threshold:
                a, b = (authors[i], authors[j])
                w = s
                for u, v in ((a, b), (b, a)):
                    if G.has_edge(u, v):
                        G[u][v]['weight'] += w
                    else:
                        G.add_edge(u, v, weight=w)

def build_graph(items: list[RedditItem], event_id: str, embedder=None, cfg=CFG) -> GraphSnapshot:
    items = items or []
    authors: list[str] = []
    seen = set()
    for it in items:
        a = it.author
        if _valid_author(a) and a not in seen:
            seen.add(a)
            authors.append(a)
    n_comments = sum((1 for it in items if it.kind == 'comment'))
    G = nx.DiGraph()
    G.add_nodes_from(authors)
    mode = (cfg.edge_mode or 'thread').lower()
    emb = None
    if embedder is not None and authors:
        emb = _author_mean_embeddings(items, authors, embedder)
    if mode == 'reply':
        _add_reply_edges(G, items)
    elif mode == 'thread':
        _add_thread_edges(G, items)
    elif mode == 'semantic':
        if emb is not None:
            _add_semantic_edges(G, authors, emb, cfg.semantic_sim_threshold)
    elif mode == 'hybrid':
        _add_reply_edges(G, items)
        if emb is not None:
            _add_semantic_edges(G, authors, emb, cfg.semantic_sim_threshold)
    else:
        _add_thread_edges(G, items)
    community_labels = detect_communities(G)
    n_communities = len(set(community_labels.values())) if community_labels else 0
    return GraphSnapshot(event_id=event_id, n_nodes=G.number_of_nodes(), n_edges=G.number_of_edges(), directed=True, community_labels=community_labels, n_communities=n_communities, graph=G, embeddings=emb)

def _modularity(H: nx.Graph, labels: dict[str, int]) -> float:
    if H.number_of_edges() == 0 or H.number_of_nodes() < 2:
        return float('nan')
    groups: dict[int, set] = {}
    for n in H.nodes():
        c = labels.get(n, -1)
        groups.setdefault(c, set()).add(n)
    communities = list(groups.values())
    if len(communities) < 1:
        return float('nan')
    try:
        return float(nx.community.modularity(H, communities, weight='weight'))
    except Exception:
        return float('nan')

def _normalized_laplacian_lambda2(H: nx.Graph) -> float:
    if H.number_of_nodes() < 2 or H.number_of_edges() == 0:
        return float('nan')
    if not nx.is_connected(H):
        largest = max(nx.connected_components(H), key=len)
        H = H.subgraph(largest).copy()
    cc = H
    n = cc.number_of_nodes()
    if n < 2 or cc.number_of_edges() == 0:
        return float('nan')
    nodes = list(cc.nodes())
    L = nx.normalized_laplacian_matrix(cc, nodelist=nodes, weight='weight')
    L = L.astype(np.float64)
    if n <= 500:
        Ld = L.toarray()
        Ld = 0.5 * (Ld + Ld.T)
        vals = np.linalg.eigvalsh(Ld)
        vals = np.sort(np.real(vals))
        lam2 = float(vals[1])
        return max(lam2, 0.0)
    from scipy.sparse.linalg import eigsh
    try:
        k = min(3, n - 1)
        vals = eigsh(L, k=k, which='SM', return_eigenvectors=False, maxiter=5000, tol=1e-06)
        vals = np.sort(np.real(vals))
        return max(float(vals[1]), 0.0)
    except Exception:
        try:
            vals = eigsh(L, k=min(3, n - 1), sigma=1e-08, which='LM', return_eigenvectors=False, maxiter=5000)
            vals = np.sort(np.real(vals))
            return max(float(vals[1]), 0.0)
        except Exception:
            Ld = L.toarray()
            Ld = 0.5 * (Ld + Ld.T)
            vals = np.sort(np.real(np.linalg.eigvalsh(Ld)))
            return max(float(vals[1]), 0.0)

def _conductance(H: nx.Graph, labels: dict[str, int]) -> float:
    if H.number_of_edges() == 0 or H.number_of_nodes() < 2:
        return float('nan')
    nodes = list(H.nodes())
    node_set = set(nodes)

    def cond_of(S: set) -> Optional[float]:
        S = S & node_set
        if not S or len(S) == len(nodes):
            return None
        Sc = node_set - S
        try:
            return float(nx.conductance(H, S, Sc, weight='weight'))
        except Exception:
            return None
    candidates: list[float] = []
    groups: dict[int, set] = {}
    for n in nodes:
        groups.setdefault(labels.get(n, -1), set()).add(n)
    if len(groups) >= 2:
        for S in groups.values():
            c = cond_of(S)
            if c is not None and np.isfinite(c):
                candidates.append(c)
    if not candidates:
        cc = _largest_cc_subgraph(H)
        if cc.number_of_nodes() >= 2 and cc.number_of_edges() > 0:
            try:
                fied = nx.fiedler_vector(cc, weight='weight', method='lanczos', seed=int(CFG.seed))
                cc_nodes = list(cc.nodes())
                S = {cc_nodes[i] for i in range(len(cc_nodes)) if fied[i] >= 0}
                c = cond_of(S)
                if c is not None and np.isfinite(c):
                    candidates.append(c)
            except Exception:
                pass
    if not candidates:
        return float('nan')
    return float(min(candidates))

def _effective_resistance(cc: nx.Graph, seed: int, cap: int=300) -> float:
    n = cc.number_of_nodes()
    if n < 2 or cc.number_of_edges() == 0:
        return float('nan')
    nodes = list(cc.nodes())
    if n > cap:
        rng = np.random.default_rng(seed)
        pick = rng.choice(n, size=cap, replace=False)
        nodes = [nodes[i] for i in pick]
        sub = cc.subgraph(nodes).copy()
        sub = _largest_cc_subgraph(sub)
        nodes = list(sub.nodes())
        if len(nodes) < 2:
            return float('nan')
        cc = sub
    L = nx.laplacian_matrix(cc, nodelist=nodes, weight='weight').toarray().astype(np.float64)
    try:
        Linv = np.linalg.pinv(L)
    except Exception:
        return float('nan')
    diag = np.diag(Linv)
    m = len(nodes)
    R = diag[:, None] + diag[None, :] - 2.0 * Linv
    iu = np.triu_indices(m, k=1)
    vals = R[iu]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float('nan')
    return float(np.mean(vals))

def _community_entropy(labels: dict[str, int], n_nodes: int) -> float:
    if not labels:
        return float('nan')
    sizes = pd.Series(list(labels.values())).value_counts().to_numpy(dtype=np.float64)
    n_comm = len(sizes)
    if n_comm < 2:
        return float('nan')
    p = sizes / sizes.sum()
    p = p[p > 0]
    H = -float(np.sum(p * np.log(p)))
    denom = np.log(n_comm)
    if denom <= 0:
        return float('nan')
    return H / denom

def _embedding_variance(emb: Optional[np.ndarray]) -> float:
    if emb is None:
        return float('nan')
    E = np.asarray(emb, dtype=np.float64)
    if E.ndim != 2 or E.shape[0] < 2:
        return float('nan')
    norms = np.linalg.norm(E, axis=1)
    mask = norms > 0
    if mask.sum() < 2:
        return float('nan')
    En = E[mask] / norms[mask][:, None]
    sims = En @ En.T
    m = En.shape[0]
    iu = np.triu_indices(m, k=1)
    cos = np.clip(sims[iu], -1.0, 1.0)
    dist = 1.0 - cos
    if dist.size == 0:
        return float('nan')
    return float(np.mean(dist))

def fragmentation(snap: GraphSnapshot, cfg=CFG) -> FragmentationScore:
    G = snap.graph
    labels = snap.community_labels or {}
    n_nodes = snap.n_nodes
    n_comments = None
    score = FragmentationScore(event_id=snap.event_id, n_nodes=n_nodes, n_communities=snap.n_communities)
    if G is None or n_nodes < 2 or G.number_of_edges() == 0:
        score.modularity = float('nan')
        score.spectral_gap = float('nan')
        score.conductance = float('nan')
        score.effective_resistance = float('nan')
        score.community_entropy = _community_entropy(labels, n_nodes)
        score.embedding_variance = _embedding_variance(snap.embeddings)
        return score
    H = _undirected_projection(G)
    cc = _largest_cc_subgraph(H)
    score.modularity = _modularity(H, labels)
    score.spectral_gap = _normalized_laplacian_lambda2(H)
    score.conductance = _conductance(H, labels)
    score.effective_resistance = _effective_resistance(cc, seed=int(cfg.seed))
    score.community_entropy = _community_entropy(labels, n_nodes)
    score.embedding_variance = _embedding_variance(snap.embeddings)
    return score
__all__ = ['build_graph', 'detect_communities', 'fragmentation']
