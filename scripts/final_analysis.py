import sys, json, glob, warnings
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import CFG, RESULTS, PROCESSED
from contracts import PANEL_COLUMNS, zscore
import analysis
FRAG_MEASURES = ['modularity', 'spectral_gap', 'conductance', 'effective_resistance', 'community_entropy', 'embedding_variance']
OUTCOMES = [('pead_abs', '|PEAD| (CAR[+2,+30])'), ('post_vol', 'Post-event volatility'), ('halflife_days', 'Adjustment half-life (days)'), ('vr_ineff', '|Variance ratio − 1|'), ('car_event', 'Announcement CAR[0,+1]')]

def load_panel():
    rows = [json.loads(l) for f in glob.glob('data/interim/panel_rows_*.jsonl') for l in open(f) if l.strip()]
    raw = pd.DataFrame(rows).drop_duplicates(subset='event_id', keep='last').reset_index(drop=True)
    raw = raw.reindex(columns=list(dict.fromkeys(PANEL_COLUMNS + list(raw.columns))))
    panel = analysis.prep(raw, cfg=CFG)
    panel['frag_index'] = zscore(panel['frag_index'])
    panel['vr_ineff'] = (panel['variance_ratio'] - 1.0).abs()
    return panel

def frag_row(res):
    key = [k for k in res.params.index if 'frag_index' in k]
    if not key:
        return (np.nan, np.nan, np.nan, int(res.nobs))
    k = key[0]
    return (res.params[k], res.bse[k], res.pvalues[k], int(res.nobs))

def main():
    panel = load_panel()
    panel.to_parquet(PROCESSED / 'event_panel_final.parquet', index=False)
    n = len(panel)
    lines = []

    def w(s=''):
        lines.append(s)
        print(s)
    w('=' * 74)
    w(f'COMMUNICATION FRAGMENTATION & PRICE DISCOVERY — FINAL ANALYSIS  (N={n} events)')
    w('=' * 74)
    w('\n[1] SAMPLE COMPOSITION')
    w(f"  events={n} | tickers={panel['ticker'].nunique()} | date range {pd.to_datetime(panel['announce_utc']).min().date()} .. {pd.to_datetime(panel['announce_utc']).max().date()}")
    w(f"  median comments/event={panel['n_comments'].median():.0f} | median authors={panel['n_nodes'].median():.0f} | median communities={panel['n_communities'].median():.0f}")
    w('\n[2] FRAGMENTATION MEASURES (mean / sd / min / max)')
    for m in FRAG_MEASURES:
        s = panel[m].dropna()
        if len(s):
            w(f'  {m:22} {s.mean():8.3f} {s.std():8.3f} {s.min():8.3f} {s.max():8.3f}')
    w('\n[3] MAIN RESULTS — frag_index (per 1 SD) -> inefficiency, sector FE + controls, HC1 SE')
    w(f"  {'outcome':30} {'coef':>10} {'se':>9} {'t':>7} {'p':>7} {'n':>4}")
    main_res = {}
    for col, label in OUTCOMES:
        sub = panel.dropna(subset=[col, 'frag_index'])
        if len(sub) < 20:
            continue
        try:
            res = analysis.baseline_ols(sub, outcome=col, treat='frag_index', cfg=CFG)
            c, se, p, nn = frag_row(res)
            t = c / se if se else np.nan
            star = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
            w(f'  {label:30} {c:>10.4f} {se:>9.4f} {t:>7.2f} {p:>7.3f} {nn:>4} {star}')
            main_res[col] = (c, se, p, nn)
        except Exception as e:
            w(f'  {label:30} ERR {e!r}')
    w('\n[4] ROBUSTNESS — each measure -> |PEAD|, signed so +=more fragmented (raw r | controlled)')
    signs = {'spectral_gap': -1.0}
    for m in FRAG_MEASURES:
        sub = panel.dropna(subset=[m, 'pead_abs']).copy()
        if len(sub) < 20:
            continue
        sub['treat'] = signs.get(m, 1.0) * zscore(sub[m])
        raw = sub[['treat', 'pead_abs']].corr().iloc[0, 1]
        sub = sub.drop(columns=[c for c in ['frag_index'] if c in sub.columns])
        try:
            res = analysis.baseline_ols(sub, outcome='pead_abs', treat='treat', cfg=CFG)
            key = [k for k in res.params.index if 'treat' in k][0]
            c, se, p = (res.params[key], res.bse[key], res.pvalues[key])
            star = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
            w(f'  {m:22} raw_r={raw:+.3f}  coef={c:>8.4f} se={se:>7.4f} p={p:>6.3f} {star}')
        except Exception as e:
            w(f'  {m:22} ERR {e!r}')
    w('\n[5] IDENTIFICATION')
    did_long = pd.read_parquet(PROCESSED / 'did_long.parquet') if (PROCESSED / 'did_long.parquet').exists() else None
    try:
        key = analysis.run_all(panel, did_long, cfg=CFG)
        for k in ['did_beta', 'did_beta_p', 'placebo_t_true', 'placebo_p_perm', 'pretrend_frag_coef', 'pretrend_frag_p', 'iv_frag_coef', 'iv_first_stage_F', 'iv_illustrative']:
            if k in key:
                w(f'  {k:24} {key[k]}')
    except Exception as e:
        w(f'  run_all ERR {e!r}')
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].hist(panel['frag_index'].dropna(), bins=25, color='#3b6', edgecolor='k', alpha=0.8)
    ax[0].set_title('Fragmentation index (standardized)')
    ax[0].set_xlabel('frag_index')
    ax[1].hist(panel['modularity'].dropna(), bins=25, color='#69c', edgecolor='k', alpha=0.8)
    ax[1].set_title('Community modularity Q')
    ax[1].set_xlabel('modularity')
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig1_fragmentation_dist.png', dpi=140)
    plt.close(fig)
    sub = panel.dropna(subset=['frag_index', 'pead_abs']).copy()
    sub['bin'] = pd.qcut(sub['frag_index'], q=min(10, max(3, len(sub) // 8)), duplicates='drop')
    g = sub.groupby('bin', observed=True).agg(x=('frag_index', 'mean'), y=('pead_abs', 'mean')).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(g['x'], g['y'], s=40, color='#c33')
    if len(g) > 2:
        b = np.polyfit(g['x'], g['y'], 1)
        xs = np.linspace(g['x'].min(), g['x'].max(), 50)
        ax.plot(xs, np.polyval(b, xs), '--', color='k', lw=1)
    ax.set_xlabel('Communication fragmentation (SD)')
    ax.set_ylabel('|PEAD|  (post-earnings drift)')
    ax.set_title(f'Fragmentation vs. price-discovery inefficiency (N={len(sub)})')
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig2_binscatter_pead.png', dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ys = [lab for col, lab in OUTCOMES if col in main_res]
    cs = [main_res[col][0] for col, lab in OUTCOMES if col in main_res]
    es = [1.96 * main_res[col][1] for col, lab in OUTCOMES if col in main_res]
    stds = []
    for col, lab in OUTCOMES:
        if col in main_res:
            sd = panel[col].std() or 1.0
            stds.append(sd)
    csz = [c / s for c, s in zip(cs, stds)]
    esz = [e / s for e, s in zip(es, stds)]
    yy = np.arange(len(ys))
    ax.errorbar(csz, yy, xerr=esz, fmt='o', color='#225', capsize=3)
    ax.axvline(0, color='grey', lw=0.8)
    ax.set_yticks(yy)
    ax.set_yticklabels(ys, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Std. coef of fragmentation (per 1 SD) ± 95% CI')
    ax.set_title('Fragmentation → inefficiency, across outcomes')
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig3_coef_plot.png', dpi=140)
    plt.close(fig)
    w('\n[6] figures -> results/fig1_fragmentation_dist.png, fig2_binscatter_pead.png, fig3_coef_plot.png')
    (RESULTS / 'final_analysis.txt').write_text('\n'.join(lines))
    print(f"\nsaved -> {RESULTS / 'final_analysis.txt'} and 3 figures; panel N={n}")
if __name__ == '__main__':
    main()
