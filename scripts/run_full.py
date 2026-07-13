import argparse, json, sys, time, traceback
from pathlib import Path
sys.path.insert(0, 'src')
import numpy as np
import pandas as pd
from config import CFG, PROCESSED, INTERIM, RESULTS
from contracts import PANEL_COLUMNS
import glob as _glob
import analysis
import assemble as asm
CKPT = INTERIM / 'panel_rows.jsonl'
FAILLOG = INTERIM / 'event_failures.jsonl'

def _all_ckpts() -> list[Path]:
    return [Path(p) for p in _glob.glob(str(INTERIM / 'panel_rows*.jsonl'))]

def _load_done() -> set[str]:
    done = set()
    for path in _all_ckpts():
        for line in path.open():
            try:
                done.add(json.loads(line)['event_id'])
            except Exception:
                pass
    return done

def _append(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as fh:
        fh.write(json.dumps(obj, default=str) + '\n')
        fh.flush()

def collect(tickers, cfg=CFG) -> None:
    from earnings import get_all_earnings
    from reddit_data import fetch_event_window
    from prices import get_prices, get_market
    from embeddings import get_embedder
    from graph import build_graph, fragmentation
    from outcomes import compute_outcomes
    from features import compute_controls
    done = _load_done()
    print(f'[init] {len(done)} events already checkpointed', flush=True)
    events = get_all_earnings(tickers, cfg.start_date, cfg.end_date, cfg=cfg)
    events = [e for e in events if e.event_id not in done]
    print(f'[init] {len(events)} events to process across {len(tickers)} tickers', flush=True)
    embedder = get_embedder(cfg)
    price_start = (pd.Timestamp(cfg.start_date) - pd.Timedelta(days=cfg.price_lookback_days)).date().isoformat()
    try:
        market = get_prices(cfg.market_index, price_start, cfg.end_date, cfg=cfg)
    except Exception as exc:
        print(f'[warn] market fetch failed ({exc}); abnormal returns degrade', flush=True)
        market = None
    price_cache: dict[str, pd.DataFrame] = {}
    n_ok = n_skip = 0
    for i, ev in enumerate(events, 1):
        t0 = time.time()
        try:
            items = fetch_event_window(ev, cfg=cfg)
            if items is None or len(items) < cfg.min_event_comments:
                _append(FAILLOG, {'event_id': ev.event_id, 'reason': 'low_chatter', 'n': 0 if items is None else len(items)})
                n_skip += 1
                print(f'[{i}/{len(events)}] {ev.event_id} skip: chatter={(0 if items is None else len(items))}', flush=True)
                continue
            snap = build_graph(items, ev.event_id, embedder=embedder, cfg=cfg)
            frag = fragmentation(snap, cfg=cfg)
            frag.n_comments = len(items)
            prices = price_cache.get(ev.ticker)
            if prices is None:
                prices = get_prices(ev.ticker, price_start, cfg.end_date, cfg=cfg)
                price_cache[ev.ticker] = prices
            if prices is None or len(prices) == 0:
                _append(FAILLOG, {'event_id': ev.event_id, 'reason': 'no_prices'})
                n_skip += 1
                continue
            outcome = compute_outcomes(ev, prices, market, cfg=cfg)
            controls = compute_controls(ev, prices, market, items, cfg=cfg)
            row = asm._row_from_parts(ev, frag, outcome, controls)
            _append(CKPT, row)
            n_ok += 1
            print(f'[{i}/{len(events)}] {ev.event_id} OK: {len(items)} items, {snap.n_nodes} authors, Q={frag.modularity}, pead={outcome.pead}, hl={outcome.halflife_days} ({time.time() - t0:.0f}s)', flush=True)
        except Exception as exc:
            _append(FAILLOG, {'event_id': ev.event_id, 'reason': 'error', 'err': repr(exc)})
            n_skip += 1
            print(f'[{i}/{len(events)}] {ev.event_id} ERR: {exc!r}', flush=True)
            traceback.print_exc()
            continue
    print(f'\n[collect done] {n_ok} ok, {n_skip} skipped, {len(_load_done())} total checkpointed', flush=True)

def assemble_and_analyze(cfg=CFG) -> None:
    shards = _all_ckpts()
    if not shards:
        print('[assemble] no checkpoint rows yet', flush=True)
        return
    rows = [json.loads(l) for p in shards for l in p.open() if l.strip()]
    raw = pd.DataFrame(rows)
    if 'event_id' in raw.columns:
        raw = raw.drop_duplicates(subset='event_id', keep='last').reset_index(drop=True)
    raw = raw.reindex(columns=list(dict.fromkeys(PANEL_COLUMNS + list(raw.columns))))
    panel = analysis.prep(raw, cfg=cfg)
    did_long = asm._build_did_long_real(panel)
    asm._write(panel, did_long)
    print(f'[assemble] wrote panel ({len(panel)} events) + did_long ({len(did_long)} rows) to {PROCESSED}', flush=True)
    try:
        key = analysis.run_all(panel, did_long, cfg=cfg)
        print('\n=== REAL-DATA HEADLINE COEFFICIENTS ===', flush=True)
        for k, v in key.items():
            if not str(k).startswith('_'):
                print(f'  {k}: {v}', flush=True)
        print(f"\ntables -> {RESULTS / 'regression_tables.txt'}", flush=True)
    except Exception as exc:
        print(f'[assemble] analysis failed (likely too few events yet): {exc!r}', flush=True)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tickers', default=None, help='CSV subset; default = all 35')
    ap.add_argument('--assemble-only', action='store_true')
    ap.add_argument('--sleep', type=float, default=0.5, help='polite per-page delay (s)')
    ap.add_argument('--cap', type=int, default=8000, help='max items per (sub,kind) window')
    ap.add_argument('--subs', default='stocks,investing', help='subreddits (comma-separated)')
    ap.add_argument('--pre', type=int, default=6, help='reddit pre-window hours')
    ap.add_argument('--post', type=int, default=24, help='reddit post-window hours')
    ap.add_argument('--query', action='store_true', help='use server-side ticker query (pullpush/arctic body=) instead of bulk drain')
    ap.add_argument('--out', default=None, help='checkpoint shard name (e.g. A); enables concurrent workers writing panel_rows_<name>.jsonl. Assembly unions all shards.')
    args = ap.parse_args()
    if args.out:
        globals()['CKPT'] = INTERIM / f'panel_rows_{args.out}.jsonl'
        globals()['FAILLOG'] = INTERIM / f'event_failures_{args.out}.jsonl'
    if args.query:
        CFG.reddit_sources = ['pullpush', 'arctic_shift']
        CFG.reddit_query_by_ticker = True
    else:
        CFG.reddit_sources = ['arctic_shift']
        CFG.reddit_query_by_ticker = False
    CFG.subreddits = [s.strip() for s in args.subs.split(',') if s.strip()]
    CFG.reddit_pre_hours = args.pre
    CFG.reddit_post_hours = args.post
    CFG.rate_limit_sleep = args.sleep
    CFG.max_items_per_window = args.cap
    tickers = args.tickers.split(',') if args.tickers else list(CFG.tickers)
    if not args.assemble_only:
        collect(tickers, cfg=CFG)
    assemble_and_analyze(cfg=CFG)
