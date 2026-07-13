from __future__ import annotations
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from config import CFG, PROCESSED
from contracts import PANEL_COLUMNS
import analysis
import synth
log = logging.getLogger('assemble')
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
PANEL_PATH = PROCESSED / 'event_panel.parquet'
DID_PATH = PROCESSED / 'did_long.parquet'

def _coerce(v):
    return np.nan if v is None else v

def _row_from_parts(event, frag, outcome, controls) -> dict:
    row: dict = {'event_id': event.event_id, 'ticker': event.ticker, 'announce_utc': event.announce_utc.isoformat(), 'sector': getattr(controls, 'sector', None)}
    for f in ('modularity', 'spectral_gap', 'conductance', 'effective_resistance', 'community_entropy', 'embedding_variance', 'n_communities', 'n_nodes', 'frag_index'):
        row[f] = _coerce(getattr(frag, f, None))
    row['n_comments'] = _coerce(getattr(controls, 'n_comments', None) if getattr(controls, 'n_comments', None) is not None else getattr(frag, 'n_comments', None))
    for f in ('pead', 'pead_abs', 'halflife_days', 'adjustment_speed_k', 'variance_ratio', 'post_vol', 'car_event'):
        row[f] = _coerce(getattr(outcome, f, None))
    for f in ('surprise_pct', 'abs_surprise', 'log_mktcap', 'pre_vol', 'log_volume', 'prior_return', 'news_intensity'):
        row[f] = _coerce(getattr(controls, f, None))
    return row

def build_panel(cfg=CFG, tickers=None, pilot=None) -> pd.DataFrame:
    from earnings import get_all_earnings
    from reddit_data import fetch_event_window
    from prices import get_prices, get_market
    from embeddings import get_embedder
    from graph import build_graph, fragmentation
    from outcomes import compute_outcomes
    from features import compute_controls
    tickers = list(tickers) if tickers else list(cfg.tickers)
    if pilot is not None:
        tickers = tickers[:int(pilot)]
    log.info('build_panel: %d tickers, window %s..%s', len(tickers), cfg.start_date, cfg.end_date)
    events = get_all_earnings(tickers, cfg.start_date, cfg.end_date, cfg=cfg)
    log.info('fetched %d earnings events', len(events))
    embedder = get_embedder(cfg)
    log.info('embedder = %s (dim=%s)', getattr(embedder, 'name', '?'), getattr(embedder, 'dim', '?'))
    price_start = (pd.Timestamp(cfg.start_date) - pd.Timedelta(days=cfg.price_lookback_days)).date().isoformat()
    try:
        market = get_prices(cfg.market_index, price_start, cfg.end_date, cfg=cfg)
    except Exception as exc:
        log.warning('get_market failed (%s); abnormal returns will be NaN', exc)
        market = None
    price_cache: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    n_skipped = 0
    for ev in events:
        try:
            try:
                items = fetch_event_window(ev, cfg=cfg)
            except Exception as exc:
                log.info('skip %s: reddit fetch failed (%s)', ev.event_id, exc)
                n_skipped += 1
                continue
            if items is None or len(items) < cfg.min_event_comments:
                log.info('skip %s: too little chatter (%d < %d)', ev.event_id, 0 if items is None else len(items), cfg.min_event_comments)
                n_skipped += 1
                continue
            snap = build_graph(items, ev.event_id, embedder=embedder, cfg=cfg)
            frag = fragmentation(snap, cfg=cfg)
            frag.n_comments = len(items)
            prices = price_cache.get(ev.ticker)
            if prices is None:
                try:
                    prices = get_prices(ev.ticker, price_start, cfg.end_date, cfg=cfg)
                except Exception as exc:
                    log.info('skip %s: price fetch failed (%s)', ev.event_id, exc)
                    n_skipped += 1
                    continue
                price_cache[ev.ticker] = prices
            if prices is None or len(prices) == 0:
                log.info('skip %s: no prices', ev.event_id)
                n_skipped += 1
                continue
            outcome = compute_outcomes(ev, prices, market, cfg=cfg)
            controls = compute_controls(ev, prices, market, items, cfg=cfg)
            rows.append(_row_from_parts(ev, frag, outcome, controls))
        except Exception as exc:
            log.warning('skip %s: unexpected error (%s)', getattr(ev, 'event_id', '?'), exc)
            n_skipped += 1
            continue
    log.info('assembled %d rows (%d skipped)', len(rows), n_skipped)
    if not rows:
        raise RuntimeError('build_panel produced 0 rows — no events cleared the chatter/price filters. Check keys, the sample window, or run --synthetic offline.')
    raw_panel = pd.DataFrame(rows)
    raw_panel = raw_panel.reindex(columns=list(dict.fromkeys(PANEL_COLUMNS + list(raw_panel.columns))))
    panel = analysis.prep(raw_panel, cfg=cfg)
    did_long = _build_did_long_real(panel)
    _write(panel, did_long)
    return panel

def _build_did_long_real(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in panel.iterrows():
        frag = r.get('frag_index', np.nan)
        if not np.isfinite(frag):
            continue
        pre = r.get('prior_return', np.nan)
        post = r.get('pead_abs', np.nan)
        if not np.isfinite(post):
            post = r.get('post_vol', np.nan)
        if not (np.isfinite(pre) or np.isfinite(post)):
            continue
        ni = r.get('news_intensity', np.nan)
        rows.append({'event_id': r['event_id'], 'period': 0, 'inefficiency': abs(pre) if np.isfinite(pre) else np.nan, 'frag_index': frag, 'news_intensity': ni})
        rows.append({'event_id': r['event_id'], 'period': 1, 'inefficiency': abs(post) if np.isfinite(post) else np.nan, 'frag_index': frag, 'news_intensity': ni})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=['inefficiency'])
    return df

def assemble_from_synth(cfg=CFG, n_events: int=400, beta: float=1.5) -> pd.DataFrame:
    panel, did_long = synth.generate_synthetic_panel(n_events=n_events, beta=beta, seed=cfg.seed)
    panel = analysis.prep(panel, cfg=cfg)
    _write(panel, did_long)
    log.info('wrote synthetic panel (%d rows) + did_long (%d rows)', len(panel), len(did_long))
    return panel

def _write(panel: pd.DataFrame, did_long: pd.DataFrame) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    try:
        panel.to_parquet(PANEL_PATH, index=False)
    except Exception as exc:
        log.warning('parquet write failed (%s); falling back to CSV', exc)
        panel.to_csv(PANEL_PATH.with_suffix('.csv'), index=False)
    try:
        if did_long is not None and len(did_long) > 0:
            did_long.to_parquet(DID_PATH, index=False)
    except Exception as exc:
        log.warning('did_long parquet write failed (%s); CSV fallback', exc)
        did_long.to_csv(DID_PATH.with_suffix('.csv'), index=False)

def load_panel() -> pd.DataFrame:
    return pd.read_parquet(PANEL_PATH)

def load_did_long() -> pd.DataFrame | None:
    if DID_PATH.exists():
        return pd.read_parquet(DID_PATH)
    return None
__all__ = ['build_panel', 'assemble_from_synth', 'load_panel', 'load_did_long', 'PANEL_PATH', 'DID_PATH']
