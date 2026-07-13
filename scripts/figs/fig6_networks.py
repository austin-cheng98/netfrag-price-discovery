import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import networkx as nx
sys.path.insert(0, 'src')
from config import CFG
import earnings as earnings_mod
from reddit_data import fetch_event_window
from graph import build_graph, _undirected_projection
panel = pd.read_parquet('data/processed/event_panel_final.parquet')
band = panel[(panel['n_nodes'] >= 45) & (panel['n_nodes'] <= 95)].copy()
lo_row = band.loc[band['modularity'].idxmin()]
hi_row = band.loc[band['modularity'].idxmax()]
CFG.reddit_sources = ['arctic_shift']
CFG.reddit_query_by_ticker = False
CFG.subreddits = ['stocks', 'investing']
CFG.reddit_pre_hours = 3
CFG.reddit_post_hours = 15
CFG.max_items_per_window = 3000
CFG.rate_limit_sleep = 0
events = earnings_mod.get_all_earnings(CFG.tickers, CFG.start_date, CFG.end_date, cfg=CFG)
by_id = {ev.event_id: ev for ev in events}

def reconstruct(row):
    ev = by_id[row['event_id']]
    items = fetch_event_window(ev, cfg=CFG)
    snap = build_graph(items, ev.event_id, embedder=None, cfg=CFG)
    return (ev, snap)
lo_ev, lo_snap = reconstruct(lo_row)
hi_ev, hi_snap = reconstruct(hi_row)

def draw(ax, ev, snap, q, title_prefix):
    H = _undirected_projection(snap.graph)
    labels = snap.community_labels or {}
    comm_ids = sorted(set((labels.get(n, -1) for n in H.nodes())))
    cmap = cm.get_cmap('tab10')
    cid_to_color = {c: cmap(i % 10) for i, c in enumerate(comm_ids)}
    node_colors = [cid_to_color[labels.get(n, -1)] for n in H.nodes()]
    deg = dict(H.degree())
    node_sizes = [30 + 45 * deg.get(n, 0) for n in H.nodes()]
    pos = nx.spring_layout(H, seed=CFG.seed, k=None)
    nx.draw_networkx_edges(H, pos, ax=ax, edge_color='lightgrey', width=0.6, alpha=0.7)
    nx.draw_networkx_nodes(H, pos, ax=ax, node_color=node_colors, node_size=node_sizes, linewidths=0.4, edgecolors='white')
    ax.set_title(f'{title_prefix}: {ev.ticker} {ev.announce_utc.date().isoformat()}\nQ = {q:.3f}   (n = {H.number_of_nodes()} nodes)', fontsize=11)
    ax.axis('off')
fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
draw(axes[0], lo_ev, lo_snap, float(lo_row['modularity']), 'Low fragmentation')
draw(axes[1], hi_ev, hi_snap, float(hi_row['modularity']), 'High fragmentation')
fig.suptitle('Low- vs high-fragmentation discussion networks', fontsize=13, y=1.0)
fig.tight_layout()
out = 'results/fig6_networks.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
assert os.path.getsize(out) > 0
print('SAVED', out, os.path.getsize(out))
print('LOW', lo_row['event_id'], 'Q=%.4f' % lo_row['modularity'], 'n_nodes=', int(lo_row['n_nodes']), 'recon_nodes=', lo_snap.n_nodes)
print('HIGH', hi_row['event_id'], 'Q=%.4f' % hi_row['modularity'], 'n_nodes=', int(hi_row['n_nodes']), 'recon_nodes=', hi_snap.n_nodes)
