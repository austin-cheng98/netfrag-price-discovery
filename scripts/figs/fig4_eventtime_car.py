import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
OUT = 'results/fig4_eventtime_car.png'
df = pd.read_parquet('data/processed/event_time_ar.parquet')
ev = df.groupby('event_id')['frag_index'].first()
labels = ['Low', 'Mid', 'High']
terciles = pd.qcut(ev, 3, labels=labels)
ev_ter = terciles.to_dict()
df = df.copy()
df['tercile'] = df['event_id'].map(ev_ter)
taus = np.arange(-5, 31)
colors = {'Low': '#0072B2', 'Mid': '#E69F00', 'High': '#D55E00'}
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
terminal = {}
for lab in labels:
    sub = df[df['tercile'] == lab]
    mean_ar = sub.groupby('tau')['ar'].mean().reindex(taus)
    car = mean_ar.cumsum()
    terminal[lab] = car.iloc[-1]
    n = sub['event_id'].nunique()
    ax.plot(taus, car.values * 100, label=f'{lab} (n={n})', color=colors[lab], linewidth=1.8)
ax.axvline(0, linestyle='--', color='0.4', linewidth=1)
ax.axhline(0, color='0.7', linewidth=0.8)
ax.set_xlabel('Event time (trading days)')
ax.set_ylabel('Cumulative abnormal return (%)')
ax.set_title('Post-announcement drift by communication fragmentation')
ax.legend(title='Fragmentation tercile', frameon=False)
ax.margins(x=0.01)
fig.tight_layout()
fig.savefig(OUT)
print('terminal CAR (fraction):', {k: round(v, 4) for k, v in terminal.items()})
print('terminal CAR (%):', {k: round(v * 100, 2) for k, v in terminal.items()})
print('exists:', os.path.exists(OUT), 'size:', os.path.getsize(OUT))
