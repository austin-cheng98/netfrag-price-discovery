import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
MEASURES = ['modularity', 'spectral_gap', 'conductance', 'effective_resistance', 'community_entropy', 'embedding_variance']
SHORT = ['Modularity', 'Spectral\ngap', 'Conductance', 'Effective\nresistance', 'Community\nentropy', 'Embedding\nvariance']
df = pd.read_parquet('data/processed/event_panel_final.parquet')
X = df[MEASURES].copy()
X['spectral_gap'] = -X['spectral_gap']
C = X.corr(method='pearson').values
n = len(MEASURES)
best_pos = (-2.0, None)
best_neg = (2.0, None)
for i in range(n):
    for j in range(i + 1, n):
        r = C[i, j]
        if r > best_pos[0]:
            best_pos = (r, (i, j))
        if r < best_neg[0]:
            best_neg = (r, (i, j))
fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
im = ax.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(SHORT, fontsize=8)
ax.set_yticklabels(SHORT, fontsize=8)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
for i in range(n):
    for j in range(n):
        val = C[i, j]
        color = 'white' if abs(val) > 0.6 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)
ax.set_title('Correlations among fragmentation measures', fontsize=12, pad=12)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Pearson correlation', fontsize=9)
fig.tight_layout()
os.makedirs('results', exist_ok=True)
out = 'results/fig5_measure_corr.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
pi, pj = best_pos[1]
ni, nj = best_neg[1]
print('SAVED', out, os.path.getsize(out))
print(f'MOST_POS {MEASURES[pi]}~{MEASURES[pj]} r={best_pos[0]:.2f}')
print(f'MOST_NEG {MEASURES[ni]}~{MEASURES[nj]} r={best_neg[0]:.2f}')
