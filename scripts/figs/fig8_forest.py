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
from analysis import FRAG_MEASURES
PANEL = 'data/processed/event_panel_final.parquet'
OUT = 'results/fig8_forest.png'
raw = pd.read_parquet(PANEL)
p = analysis.prep(raw, cfg=CFG)
p['frag_index'] = zscore(p['frag_index'])
p['halflife_std'] = p['halflife_days'] / p['halflife_days'].std()
p['vr_ineff'] = (p['variance_ratio'] - 1.0).abs()

def coef_ci(res, key, outcome_sd):
    b = float(res.params.get(key, np.nan))
    se = float(res.bse.get(key, np.nan))
    return (b / outcome_sd, (b - 1.96 * se) / outcome_sd, (b + 1.96 * se) / outcome_sd)
rows = []
a_specs = [('Index -> |PEAD|', 'pead_abs'), ('Index -> half-life (SD)', 'halflife_std'), ('Index -> post-vol', 'post_vol'), ('Index -> |VR-1|', 'vr_ineff')]
for label, outcome in a_specs:
    sd = float(p[outcome].std())
    res = analysis.baseline_ols(p, outcome=outcome, treat='frag_index', cfg=CFG)
    b, lo, hi = coef_ci(res, "Q('frag_index')", sd)
    rows.append((label, b, lo, hi, 'a'))
sd_pead = float(p['pead_abs'].std())
for col, sign in FRAG_MEASURES.items():
    d = p.copy()
    d['treat'] = sign * zscore(d[col].astype(float))
    d = d.drop(columns=['frag_index'])
    res = analysis.baseline_ols(d, outcome='pead_abs', treat='treat', cfg=CFG)
    b, lo, hi = coef_ci(res, "Q('treat')", sd_pead)
    rows.append((f'{col} -> |PEAD|', b, lo, hi, 'b'))
a_rows = [r for r in rows if r[4] == 'a']
b_rows = [r for r in rows if r[4] == 'b']
ypos = []
labels = []
data = []
y = 0.0
gap = 1.0
ordered = list(reversed(b_rows)) + [None] + list(reversed(a_rows))
for item in ordered:
    if item is None:
        y += gap
        continue
    ypos.append(y)
    labels.append(item[0])
    data.append(item)
    y += 1.0
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
c_a = '#0072B2'
c_b = '#D55E00'
for yy, item in zip(ypos, data):
    label, b, lo, hi, panel = item
    color = c_a if panel == 'a' else c_b
    excl = lo > 0 or hi < 0
    ax.plot([lo, hi], [yy, yy], color=color, lw=2, zorder=2)
    ax.plot(b, yy, 'o', color=color, ms=7, markeredgecolor='black', markeredgewidth=0.6 if excl else 0.0, zorder=3)
ax.axvline(0, color='0.35', lw=1.2, ls='--', zorder=1)
ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('SD of outcome per 1 SD of fragmentation (95% CI)')
ax.set_title('Fragmentation coefficient across specifications\n(standardized, 95% CI)', fontsize=11)
from matplotlib.lines import Line2D
leg = [Line2D([0], [0], color=c_a, marker='o', lw=2, label='(a) composite index -> outcomes'), Line2D([0], [0], color=c_b, marker='o', lw=2, label='(b) each measure -> |PEAD|')]
ax.legend(handles=leg, fontsize=8, loc='lower right', framealpha=0.9)
ax.grid(axis='x', alpha=0.25)
ax.margins(y=0.06)
fig.tight_layout()
fig.savefig(OUT)
print('saved', OUT, os.path.getsize(OUT))
excl_specs = [r[0] for r in rows if r[2] > 0 or r[3] < 0]
print('EXCLUDE_0:', excl_specs)
for r in rows:
    print(f"  {r[0]:28s} coef={r[1]:+.3f}  CI=[{r[2]:+.3f},{r[3]:+.3f}]  {('*' if r[2] > 0 or r[3] < 0 else '')}")
