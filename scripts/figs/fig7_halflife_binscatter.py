import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, 'src')
from config import CFG
from contracts import zscore
import analysis
OUT = 'results/fig7_halflife_binscatter.png'
os.makedirs('results', exist_ok=True)
raw = pd.read_parquet('data/processed/event_panel_final.parquet')
p = analysis.prep(raw, cfg=CFG)
p['frag_index'] = zscore(p['frag_index'])
p['log_hl'] = np.log(p['halflife_days'].clip(lower=0.1))
res = analysis.baseline_ols(p, outcome='log_hl', treat='frag_index', cfg=CFG)
key = 'frag_index' if 'frag_index' in res.params.index else "Q('frag_index')"
slope = res.params[key]
pval = res.pvalues[key]
d = p[['frag_index', 'log_hl']].dropna()
N = len(d)
x = d['frag_index'].values
y = d['log_hl'].values
nbins = 8
try:
    bins = pd.qcut(d['frag_index'], nbins, duplicates='drop')
except Exception:
    bins = pd.cut(d['frag_index'], nbins)
g = d.groupby(bins, observed=True)
bx = g['frag_index'].mean().values
by = g['log_hl'].mean().values
bslope, bint = np.polyfit(bx, by, 1)
xline = np.linspace(x.min(), x.max(), 100)
yline = bint + bslope * xline
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
ax.scatter(bx, by, s=70, color='#0072B2', zorder=3, label='Quantile-bin means')
ax.plot(xline, yline, color='#D55E00', lw=2, zorder=2, label=f'OLS fit (event-level slope = {slope:.3f}, p = {pval:.3f})')
ax.set_xlabel('Communication fragmentation (SD)')
ax.set_ylabel('log adjustment half-life')
ax.set_title(f'Adjustment half-life vs. communication fragmentation (N = {N})')
ax.legend(frameon=False, fontsize=8, loc='best')
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(OUT)
plt.close(fig)
print('saved', OUT, os.path.getsize(OUT))
print(f'slope={slope:.4f} pval={pval:.4f} N={N}')
