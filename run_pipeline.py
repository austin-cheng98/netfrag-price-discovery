from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def _print_analysis_summary(out: dict, beta_true: float | None=None) -> None:
    print('\n' + '=' * 70)
    print('ANALYSIS SUMMARY')
    print('=' * 70)
    outcome = out.get('outcome', '?')
    coef = out.get('baseline_frag_coef', float('nan'))
    p = out.get('baseline_frag_p', float('nan'))
    print(f'outcome                    : {outcome}')
    print(f'baseline frag_index coef   : {coef:.4f}   (p={p:.3g})')
    if 'baseline_frag_coef_no_confounder' in out:
        print(f"  (confounder OMITTED coef): {out['baseline_frag_coef_no_confounder']:.4f}  <- biased, for comparison")
    if beta_true is not None:
        print(f'TRUE beta                  : {beta_true:.4f}')
        print(f'recovery error             : {coef - beta_true:+.4f}')
    if 'placebo_p_perm' in out:
        print(f"placebo permutation p      : {out['placebo_p_perm']:.4f}")
    if 'did_beta' in out:
        print(f"DiD interaction beta       : {out['did_beta']:.4f}  (p={out.get('did_beta_p', float('nan')):.3g})")
    print(f"tables  -> {out.get('_tables_path')}")
    print(f"coefs   -> {out.get('_coefficients_path')}")
    print('=' * 70)

def run_synthetic(beta_true: float=1.5, n_events: int=400) -> dict:
    import assemble
    import analysis
    from config import CFG
    print(f'[synthetic] generating panel: n_events={n_events}, beta_true={beta_true}')
    panel = assemble.assemble_from_synth(cfg=CFG, n_events=n_events, beta=beta_true)
    long_df = assemble.load_did_long()
    out = analysis.run_all(panel, long_df=long_df, cfg=CFG)
    coef = out.get('baseline_frag_coef', float('nan'))
    ci_lo = ci_hi = float('nan')
    within = None
    try:
        prepped = analysis.prep(panel, cfg=CFG)
        res = analysis.baseline_ols(prepped, outcome=out.get('outcome', 'halflife_days'), cfg=CFG)
        key = "Q('frag_index')"
        ci = res.conf_int().loc[key]
        ci_lo, ci_hi = (float(ci[0]), float(ci[1]))
        within = bool(ci_lo <= beta_true <= ci_hi)
    except Exception as exc:
        print(f'[synthetic] CI computation failed: {exc}')
    _print_analysis_summary(out, beta_true=beta_true)
    print(f'\n>>> RECOVERED beta_hat = {coef:.4f}   (true beta = {beta_true})   95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]')
    if within is not None:
        verdict = 'WITHIN CI  ✓' if within else 'OUTSIDE CI  ✗'
        print(f'>>> beta_true {verdict}')
    out['_beta_true'] = beta_true
    out['_beta_hat'] = coef
    out['_ci'] = (ci_lo, ci_hi)
    out['_within_ci'] = within
    return out

def run_pilot(n: int, tickers: list[str] | None=None) -> dict:
    import assemble
    import analysis
    from config import CFG
    print(f'[pilot] building REAL panel on first {n} tickers (live)')
    panel = assemble.build_panel(cfg=CFG, tickers=tickers, pilot=n)
    long_df = assemble.load_did_long()
    out = analysis.run_all(panel, long_df=long_df, cfg=CFG)
    _print_analysis_summary(out)
    return out

def run_stage(stage: str, tickers: list[str] | None) -> None:
    from config import CFG
    tickers = tickers or CFG.tickers[:3]
    start, end = (CFG.start_date, CFG.end_date)
    if stage == 'earnings':
        from earnings import get_all_earnings
        evs = get_all_earnings(tickers, start, end, cfg=CFG)
        print(f'[earnings] {len(evs)} events for {tickers}')
        for e in evs[:10]:
            print(f'  {e.event_id}  session={e.session}  surprise_pct={e.surprise_pct}')
    elif stage == 'reddit':
        from earnings import get_all_earnings
        from reddit_data import fetch_event_window
        evs = get_all_earnings(tickers, start, end, cfg=CFG)
        if not evs:
            print('[reddit] no events to window')
            return
        ev = evs[0]
        items = fetch_event_window(ev, cfg=CFG)
        print(f'[reddit] {ev.event_id}: {len(items)} items in window')
    elif stage == 'prices':
        from prices import get_prices, get_market
        for t in tickers:
            try:
                df = get_prices(t, start, end, cfg=CFG)
                print(f'[prices] {t}: {len(df)} rows  {df.index.min()}..{df.index.max()}')
            except Exception as exc:
                print(f'[prices] {t}: FAILED ({exc})')
        try:
            mk = get_market(cfg=CFG)
            print(f'[prices] market {CFG.market_index}: {len(mk)} rows')
        except Exception as exc:
            print(f'[prices] market FAILED ({exc})')
    elif stage == 'graph':
        import synth
        from graph import build_graph, fragmentation
        import copy
        cfg = copy.copy(CFG)
        cfg.edge_mode = 'reply'
        for frag_flag in (True, False):
            items = synth.generate_synthetic_reddit(frag_flag, n_authors=30, seed=CFG.seed)
            snap = build_graph(items, f'synth_{frag_flag}', cfg=cfg)
            score = fragmentation(snap, cfg=cfg)
            print(f'[graph] fragmented={frag_flag}: n_nodes={snap.n_nodes} n_comm={snap.n_communities} modularity={score.modularity:.3f}')
    elif stage == 'outcomes':
        from earnings import get_all_earnings
        from prices import get_prices, get_market
        from outcomes import compute_outcomes
        evs = get_all_earnings(tickers, start, end, cfg=CFG)
        market = get_market(cfg=CFG)
        for ev in evs[:5]:
            prices = get_prices(ev.ticker, start, end, cfg=CFG)
            om = compute_outcomes(ev, prices, market, cfg=CFG)
            print(f'[outcomes] {ev.event_id}: halflife={om.halflife_days} pead_abs={om.pead_abs} car_event={om.car_event}')
    elif stage == 'assemble':
        import assemble
        panel = assemble.build_panel(cfg=CFG, tickers=tickers)
        print(f'[assemble] panel rows={len(panel)} -> {assemble.PANEL_PATH}')
    elif stage == 'analyze':
        import assemble
        import analysis
        panel = assemble.load_panel()
        long_df = assemble.load_did_long()
        out = analysis.run_all(panel, long_df=long_df, cfg=CFG)
        _print_analysis_summary(out)
    else:
        raise SystemExit(f'unknown stage: {stage}')

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='run_pipeline.py', description='Communication-Network-Fragmentation -> Price-Discovery pipeline driver.', formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--synthetic', action='store_true', help='assemble synthetic panels then analyze (recovers beta).')
    p.add_argument('--pilot', type=int, metavar='N', help='run the REAL pipeline on the first N tickers, then analyze.')
    p.add_argument('--stage', choices=['earnings', 'reddit', 'prices', 'graph', 'outcomes', 'assemble', 'analyze'], help='run a single pipeline stage in isolation.')
    p.add_argument('--tickers', type=str, default=None, help='comma-separated ticker list to scope --stage/--pilot.')
    p.add_argument('--beta', type=float, default=1.5, help='true beta for --synthetic (default 1.5).')
    p.add_argument('--n-events', type=int, default=400, help='synthetic event count for --synthetic (default 400).')
    return p

def main(argv: list[str] | None=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    tickers = [t.strip().upper() for t in args.tickers.split(',')] if args.tickers else None
    if args.synthetic:
        run_synthetic(beta_true=args.beta, n_events=args.n_events)
        return 0
    if args.pilot is not None:
        run_pilot(args.pilot, tickers=tickers)
        return 0
    if args.stage:
        run_stage(args.stage, tickers=tickers)
        return 0
    parser.print_help()
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
