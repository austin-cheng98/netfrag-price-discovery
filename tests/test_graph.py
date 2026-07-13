from __future__ import annotations
import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import networkx as nx
import pytest
import graph as gmod
from graph import build_graph, detect_communities, fragmentation
from contracts import RedditItem, GraphSnapshot
from config import CFG
UTC = timezone.utc
T0 = pd.Timestamp('2024-01-01T00:00:00Z')

def _snap_from_undirected(H: nx.Graph, event_id: str) -> GraphSnapshot:
    D = nx.DiGraph()
    D.add_nodes_from(H.nodes())
    for u, v, d in H.edges(data=True):
        D.add_edge(u, v, weight=float(d.get('weight', 1.0)))
    labels = detect_communities(D)
    return GraphSnapshot(event_id=event_id, n_nodes=D.number_of_nodes(), n_edges=D.number_of_edges(), directed=True, community_labels=labels, n_communities=len(set(labels.values())) if labels else 0, graph=D)

def two_disjoint_cliques(k: int=5) -> nx.Graph:
    A = nx.complete_graph([f'a{i}' for i in range(k)])
    B = nx.complete_graph([f'b{i}' for i in range(k)])
    H = nx.union(A, B)
    nx.set_edge_attributes(H, 1.0, 'weight')
    return H

def single_complete(n: int=10) -> nx.Graph:
    H = nx.complete_graph([f'c{i}' for i in range(n)])
    nx.set_edge_attributes(H, 1.0, 'weight')
    return H

def ring(n: int=12) -> nx.Graph:
    H = nx.cycle_graph([f'r{i}' for i in range(n)])
    nx.set_edge_attributes(H, 1.0, 'weight')
    return H

def test_modularity_ordering():
    sa = fragmentation(_snap_from_undirected(two_disjoint_cliques(5), 'a'))
    sb = fragmentation(_snap_from_undirected(single_complete(10), 'b'))
    sc = fragmentation(_snap_from_undirected(ring(12), 'c'))
    assert sa.modularity > sb.modularity
    assert sa.modularity > 0.4
    assert sb.modularity < 0.15
    assert math.isfinite(sc.modularity)

def test_spectral_gap_ordering():
    sa = fragmentation(_snap_from_undirected(two_disjoint_cliques(5), 'a'))
    sb = fragmentation(_snap_from_undirected(single_complete(10), 'b'))
    sc = fragmentation(_snap_from_undirected(ring(12), 'c'))
    assert sa.spectral_gap > sc.spectral_gap
    assert sb.spectral_gap > sc.spectral_gap
    for s in (sa, sb, sc):
        assert math.isfinite(s.spectral_gap)
        assert s.spectral_gap > 0.0

def test_disjoint_cliques_high_fragmentation_signature():
    H = two_disjoint_cliques(5)
    snap = _snap_from_undirected(H, 'frag')
    s = fragmentation(snap)
    assert s.n_communities == 2
    assert s.community_entropy is not None
    assert s.community_entropy > 0.99
    assert math.isfinite(s.effective_resistance)
    assert math.isfinite(s.conductance)

def test_complete_graph_low_fragmentation_signature():
    s = fragmentation(_snap_from_undirected(single_complete(10), 'comp'))
    assert s.n_communities == 1 or s.modularity < 0.15
    sr = fragmentation(_snap_from_undirected(ring(12), 'ring'))
    assert s.effective_resistance < sr.effective_resistance

def test_effective_resistance_ordering():
    s_comp = fragmentation(_snap_from_undirected(single_complete(10), 'comp'))
    s_ring = fragmentation(_snap_from_undirected(ring(12), 'ring'))
    assert s_ring.effective_resistance > s_comp.effective_resistance

def test_single_node_is_nan_safe():
    D = nx.DiGraph()
    D.add_node('solo')
    snap = GraphSnapshot(event_id='solo', n_nodes=1, n_edges=0, directed=True, community_labels={'solo': 0}, n_communities=1, graph=D)
    s = fragmentation(snap)
    assert s.n_nodes == 1
    for v in (s.modularity, s.spectral_gap, s.conductance, s.effective_resistance):
        assert v is None or math.isnan(v)

def test_empty_graph_is_nan_safe():
    D = nx.DiGraph()
    snap = GraphSnapshot(event_id='empty', n_nodes=0, n_edges=0, directed=True, community_labels={}, n_communities=0, graph=D)
    s = fragmentation(snap)
    assert math.isnan(s.modularity)
    assert math.isnan(s.spectral_gap)
    assert math.isnan(s.community_entropy)

def test_no_edges_multiple_nodes_nan_safe():
    D = nx.DiGraph()
    D.add_nodes_from(['x', 'y', 'z'])
    labels = detect_communities(D)
    snap = GraphSnapshot(event_id='noedge', n_nodes=3, n_edges=0, directed=True, community_labels=labels, n_communities=len(set(labels.values())), graph=D)
    s = fragmentation(snap)
    assert math.isnan(s.spectral_gap)
    assert math.isnan(s.modularity)
    assert s.n_communities == 3

def _mk(id_, author, kind='comment', parent_id=None, link_id=None, body='hi'):
    return RedditItem(id=id_, kind=kind, subreddit='stocks', author=author, created_utc=T0, body=body, parent_id=parent_id, link_id=link_id)

def test_build_graph_reply_mode():
    cfg = _cfg(edge_mode='reply')
    items = [_mk('S', 'op', kind='submission'), _mk('c1', 'alice', parent_id='t3_S', link_id='t3_S'), _mk('c2', 'bob', parent_id='t1_c1', link_id='t3_S'), _mk('c3', 'alice', parent_id='t1_c2', link_id='t3_S')]
    snap = build_graph(items, 'ev1', cfg=cfg)
    G = snap.graph
    assert snap.directed
    assert G.has_edge('alice', 'op')
    assert G.has_edge('bob', 'alice')
    assert G.has_edge('alice', 'bob')
    assert not any((u == v for u, v in G.edges()))
    assert snap.n_nodes == 3

def test_build_graph_reply_weight_accumulates():
    cfg = _cfg(edge_mode='reply')
    items = [_mk('p', 'op', kind='submission'), _mk('a1', 'alice', parent_id='t3_p', link_id='t3_p'), _mk('a2', 'alice', parent_id='t3_p', link_id='t3_p')]
    snap = build_graph(items, 'ev', cfg=cfg)
    assert snap.graph['alice']['op']['weight'] == 2.0

def test_build_graph_reply_skips_self_reply():
    cfg = _cfg(edge_mode='reply')
    items = [_mk('p', 'op', kind='submission'), _mk('c1', 'op', parent_id='t3_p', link_id='t3_p')]
    snap = build_graph(items, 'ev', cfg=cfg)
    assert snap.graph.number_of_edges() == 0

def test_build_graph_thread_mode_links_participants():
    cfg = _cfg(edge_mode='thread')
    items = [_mk('S', 'op', kind='submission', link_id='t3_S'), _mk('c1', 'alice', link_id='t3_S'), _mk('c2', 'bob', link_id='t3_S'), _mk('c3', 'carol', link_id='t3_S')]
    snap = build_graph(items, 'ev', cfg=cfg)
    H = gmod._undirected_projection(snap.graph)
    assert nx.number_connected_components(H) == 1
    assert snap.n_nodes == 4
    assert H.number_of_edges() == 3

def test_build_graph_ignores_deleted_authors():
    cfg = _cfg(edge_mode='reply')
    items = [_mk('p', 'op', kind='submission'), _mk('c1', '[deleted]', parent_id='t3_p'), _mk('c2', 'AutoModerator', parent_id='t3_p'), _mk('c3', 'alice', parent_id='t3_p')]
    snap = build_graph(items, 'ev', cfg=cfg)
    assert set(snap.graph.nodes()) == {'op', 'alice'}

def test_build_graph_empty_input():
    snap = build_graph([], 'ev')
    assert snap.n_nodes == 0
    assert snap.n_edges == 0
    s = fragmentation(snap)
    assert math.isnan(s.modularity)

def test_semantic_mode_with_fake_embedder():
    cfg = _cfg(edge_mode='semantic', semantic_sim_threshold=0.5)

    class FakeEmbedder:
        name = 'fake'
        dim = 3

        def encode(self, texts):
            out = []
            for t in texts:
                if t.startswith('g1'):
                    out.append([1.0, 0.0, 0.0])
                else:
                    out.append([0.0, 1.0, 0.0])
            return np.asarray(out, dtype=np.float64)
    items = [_mk('1', 'u1', body='g1 apple'), _mk('2', 'u2', body='g1 banana'), _mk('3', 'u3', body='g2 zebra'), _mk('4', 'u4', body='g2 yak')]
    snap = build_graph(items, 'ev', embedder=FakeEmbedder(), cfg=cfg)
    G = snap.graph
    assert G.has_edge('u1', 'u2') or G.has_edge('u2', 'u1')
    assert G.has_edge('u3', 'u4') or G.has_edge('u4', 'u3')
    assert not (G.has_edge('u1', 'u3') or G.has_edge('u3', 'u1'))
    assert snap.embeddings is not None
    assert snap.embeddings.shape[0] == 4
    s = fragmentation(snap)
    assert s.n_communities == 2
    assert not math.isnan(s.embedding_variance)

def test_detect_communities_deterministic():
    H = two_disjoint_cliques(6)
    D = nx.DiGraph()
    D.add_nodes_from(H.nodes())
    for u, v in H.edges():
        D.add_edge(u, v, weight=1.0)
    l1 = detect_communities(D)
    l2 = detect_communities(D)

    def groups(l):
        g = {}
        for n, c in l.items():
            g.setdefault(c, set()).add(n)
        return set((frozenset(s) for s in g.values()))
    assert groups(l1) == groups(l2)
    assert len(groups(l1)) == 2

def _cfg(**overrides):
    import copy
    c = copy.copy(CFG)
    for k, v in overrides.items():
        setattr(c, k, v)
    return c

def test_ollama_smoke_optional():
    import requests

    class OllamaEmbedder:
        name = 'ollama-all-minilm'
        dim = 384

        def encode(self, texts):
            r = requests.post('http://localhost:11434/api/embed', json={'model': 'all-minilm', 'input': list(texts)}, timeout=10)
            r.raise_for_status()
            arr = np.asarray(r.json()['embeddings'], dtype=np.float64)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return arr / norms
    try:
        emb = OllamaEmbedder().encode(['hello world', 'goodbye world'])
    except Exception as e:
        pytest.skip(f'Ollama not reachable: {e}')
    assert emb.shape == (2, 384)
    cfg = _cfg(edge_mode='semantic', semantic_sim_threshold=0.3)
    items = [_mk('1', 'u1', body='Nvidia earnings beat expectations, stock ripping'), _mk('2', 'u2', body='NVDA crushed the quarter, huge beat on revenue'), _mk('3', 'u3', body='I had a sandwich for lunch today it was fine')]
    snap = build_graph(items, 'ev', embedder=OllamaEmbedder(), cfg=cfg)
    assert snap.embeddings is not None
    s = fragmentation(snap)
    assert math.isfinite(s.embedding_variance)
