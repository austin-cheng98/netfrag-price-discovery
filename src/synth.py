from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from contracts import RedditItem, PANEL_COLUMNS
from config import CFG

def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=0)
    return (x - x.mean()) / sd if sd > 0 else x * 0.0

def generate_synthetic_panel(n_events: int=400, beta: float=1.5, seed: int=CFG.seed) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = int(n_events)
    n_comments = rng.integers(30, 5000, size=n).astype(float)
    n_comments_z = _z(n_comments)
    frag_latent = 0.75 * n_comments_z + rng.normal(0, 1.0, size=n)
    frag_index = _z(frag_latent)
    abs_surprise = np.abs(rng.normal(0, 3, size=n))
    surprise_pct = rng.normal(0, 5, size=n)
    log_mktcap = rng.normal(23, 1.5, size=n)
    pre_vol = np.abs(rng.normal(0.02, 0.01, size=n))
    log_volume = rng.normal(18, 1.0, size=n)
    prior_return = rng.normal(0, 0.05, size=n)
    news_intensity = np.abs(rng.normal(0.5, 0.3, size=n))
    sector = rng.choice(['Information Technology', 'Communication Services', 'Consumer Discretionary', 'Financials', 'Industrials'], size=n)
    noise = rng.normal(0, 1.0, size=n)
    halflife_days = 2.0 + beta * frag_index - 0.3 * abs_surprise + 0.8 * n_comments_z + noise
    pead_abs = np.abs(0.01 + 0.006 * frag_index + rng.normal(0, 0.004, size=n))
    pead = (0.006 * frag_index + rng.normal(0, 0.004, size=n)) * np.sign(surprise_pct + 1e-09)
    adjustment_speed_k = np.log(2.0) / np.clip(halflife_days, 0.25, None)
    variance_ratio = 1.0 + 0.05 * frag_index + rng.normal(0, 0.05, size=n)
    post_vol = np.abs(0.02 + 0.004 * frag_index + rng.normal(0, 0.003, size=n))
    car_event = 0.002 * surprise_pct + rng.normal(0, 0.01, size=n)
    modularity = 0.5 + 0.3 * frag_index + rng.normal(0, 0.2, size=n)
    spectral_gap = -(0.4 * frag_index) + rng.normal(0, 0.2, size=n)
    conductance = 0.3 + 0.2 * frag_index + rng.normal(0, 0.2, size=n)
    effective_resistance = 1.0 + 0.5 * frag_index + rng.normal(0, 0.3, size=n)
    community_entropy = 1.5 + 0.4 * frag_index + rng.normal(0, 0.2, size=n)
    embedding_variance = 0.2 + 0.1 * frag_index + rng.normal(0, 0.1, size=n)
    n_nodes = rng.integers(20, 500, size=n)
    n_communities = rng.integers(2, 20, size=n)
    tickers = np.array(CFG.tickers)
    ticker = tickers[rng.integers(0, len(tickers), size=n)]
    base = pd.Timestamp('2022-08-01T21:00:00Z')
    announce = [base + pd.Timedelta(days=int(d)) for d in rng.integers(0, 850, size=n)]
    event_id = [f'{ticker[i]}:{announce[i].date().isoformat()}#{i}' for i in range(n)]
    panel = pd.DataFrame({'event_id': event_id, 'ticker': ticker, 'sector': sector, 'announce_utc': [a.isoformat() for a in announce], 'frag_index': frag_index, 'modularity': modularity, 'spectral_gap': spectral_gap, 'conductance': conductance, 'effective_resistance': effective_resistance, 'community_entropy': community_entropy, 'embedding_variance': embedding_variance, 'n_communities': n_communities, 'n_nodes': n_nodes, 'n_comments': n_comments, 'pead': pead, 'pead_abs': pead_abs, 'halflife_days': halflife_days, 'adjustment_speed_k': adjustment_speed_k, 'variance_ratio': variance_ratio, 'post_vol': post_vol, 'car_event': car_event, 'surprise_pct': surprise_pct, 'abs_surprise': abs_surprise, 'log_mktcap': log_mktcap, 'pre_vol': pre_vol, 'log_volume': log_volume, 'prior_return': prior_return, 'news_intensity': news_intensity})
    panel = panel.reindex(columns=list(dict.fromkeys(PANEL_COLUMNS + list(panel.columns))))
    did_long = _make_did_long(panel, beta_did=0.9 * beta / 1.5, seed=seed)
    return (panel, did_long)

def _make_did_long(panel: pd.DataFrame, beta_did: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    rows = []
    frag = panel['frag_index'].to_numpy(dtype=float)
    ni = _z(panel['news_intensity'].astype(float).to_numpy())
    for i, ev in enumerate(panel['event_id'].tolist()):
        alpha_i = rng.normal(0, 0.4)
        f = float(frag[i])
        n_i = float(ni[i])
        for period in (0, 1):
            base = 6.0 + alpha_i + 0.2 * n_i
            effect = beta_did * f if period == 1 else 0.0
            ineff = base + effect + rng.normal(0, 0.5)
            rows.append({'event_id': ev, 'period': period, 'inefficiency': abs(ineff), 'frag_index': f, 'news_intensity': n_i})
    return pd.DataFrame(rows)

def generate_synthetic_reddit(fragmented: bool, n_authors: int=30, seed: int=CFG.seed, ticker: str='AAPL', subreddit: str='stocks') -> list[RedditItem]:
    rng = np.random.default_rng(seed)
    n = int(n_authors)
    authors = [f'user_{i:03d}' for i in range(n)]
    t0 = pd.Timestamp('2024-03-01T14:00:00Z')
    body_tail = f' thoughts on ${ticker} after earnings'
    items: list[RedditItem] = []
    uid = [0]

    def _new_id() -> str:
        uid[0] += 1
        return f'c{uid[0]:05d}'
    if fragmented:
        k = max(2, n // 6)
        clusters = [authors[i::k] for i in range(k)]
        for ci, members in enumerate(clusters):
            if len(members) < 2:
                continue
            link_id = f't3_sub{ci:03d}'
            root_author = members[0]
            sub_id = f'sub{ci:03d}'
            items.append(RedditItem(id=sub_id, kind='submission', subreddit=subreddit, author=root_author, created_utc=t0 + pd.Timedelta(minutes=int(rng.integers(0, 30))), body=f'${ticker} cluster {ci} discussion' + body_tail, parent_id=None, link_id=f't3_{sub_id}', score=int(rng.integers(1, 50)), source='synth'))
            posted: list[str] = [root_author]
            posted_fullnames: list[str] = [f't3_{sub_id}']
            for a in members[1:]:
                for _ in range(int(rng.integers(2, 5))):
                    j = int(rng.integers(0, len(posted)))
                    parent_fn = posted_fullnames[j]
                    cid = _new_id()
                    items.append(RedditItem(id=cid, kind='comment', subreddit=subreddit, author=a, created_utc=t0 + pd.Timedelta(minutes=int(rng.integers(30, 2000))), body=f'reply within cluster {ci}' + body_tail, parent_id=parent_fn, link_id=link_id, score=int(rng.integers(0, 30)), source='synth'))
                    posted.append(a)
                    posted_fullnames.append(f't1_{cid}')
    else:
        link_id = 't3_sub_all'
        items.append(RedditItem(id='sub_all', kind='submission', subreddit=subreddit, author=authors[0], created_utc=t0, body=f'${ticker} megathread' + body_tail, parent_id=None, link_id='t3_sub_all', score=100, source='synth'))
        posted_fullnames = ['t3_sub_all']
        posted_authors = [authors[0]]
        for a in authors[1:]:
            for _ in range(int(rng.integers(3, 7))):
                j = int(rng.integers(0, len(posted_fullnames)))
                parent_fn = posted_fullnames[j]
                cid = _new_id()
                items.append(RedditItem(id=cid, kind='comment', subreddit=subreddit, author=a, created_utc=t0 + pd.Timedelta(minutes=int(rng.integers(1, 3000))), body='cross reply megathread' + body_tail, parent_id=parent_fn, link_id=link_id, score=int(rng.integers(0, 30)), source='synth'))
                posted_fullnames.append(f't1_{cid}')
                posted_authors.append(a)
    return items
__all__ = ['generate_synthetic_panel', 'generate_synthetic_reddit']
