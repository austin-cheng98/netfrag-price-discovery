import sys, warnings
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from config import CFG, PROCESSED
from contracts import zscore
import analysis
from earnings import get_all_earnings
from prices import get_prices
from outcomes import market_model, locate_t0
TAU_LO, TAU_HI = (-5, 30)
panel = analysis.prep(pd.read_parquet(PROCESSED / 'event_panel_final.parquet'), cfg=CFG)
panel['frag_index'] = zscore(panel['frag_index'])
FIRMS = sorted(panel['ticker'].unique())
events = {e.event_id: e for e in get_all_earnings(FIRMS, CFG.start_date, CFG.end_date, cfg=CFG)}
price_start = (pd.Timestamp(CFG.start_date) - pd.Timedelta(days=CFG.price_lookback_days)).date().isoformat()
mkt = get_prices(CFG.market_index, price_start, CFG.end_date, cfg=CFG)
rows = []
for _, r in panel.iterrows():
    ev = events.get(r['event_id'])
    if ev is None:
        continue
    try:
        px = get_prices(ev.ticker, price_start, CFG.end_date, cfg=CFG)
        mret = mkt['adj_close'].reindex(px.index).pct_change()
        sret = px['adj_close'].pct_change()
        t0 = locate_t0(ev, px)
        if t0 is None:
            continue
        est_end = t0 - CFG.estimation_gap
        est_start = est_end - CFG.estimation_days
        if est_start < 1 or t0 + TAU_HI >= len(px):
            continue
        mm = market_model(sret.iloc[est_start:est_end], mret.iloc[est_start:est_end])
        alpha, beta = (mm.alpha, mm.beta)
        for tau in range(TAU_LO, TAU_HI + 1):
            j = t0 + tau
            if j < 1 or j >= len(px):
                continue
            s_j, m_j = (sret.iloc[j], mret.iloc[j])
            if pd.isna(s_j) or pd.isna(m_j):
                continue
            ar = float(s_j - (alpha + beta * m_j))
            rows.append({'event_id': r['event_id'], 'tau': tau, 'ar': ar, 'frag_index': float(r['frag_index'])})
    except Exception:
        continue
ar = pd.DataFrame(rows)
ar.to_parquet(PROCESSED / 'event_time_ar.parquet', index=False)
n_ev = ar['event_id'].nunique()
print(f'event-time AR panel: {len(ar)} rows, {n_ev} events, tau in [{TAU_LO},{TAU_HI}]')
ar['tercile'] = pd.qcut(ar.groupby('event_id')['frag_index'].transform('first'), 3, labels=['Low', 'Mid', 'High'])
post = ar[ar['tau'] >= 1].sort_values('tau')
car = post.groupby(['tercile', 'tau'])['ar'].mean().groupby(level=0).cumsum()
final = car.groupby(level=0).last()
print('terminal CAR[+1,+30] by fragmentation tercile:')
for k, v in final.items():
    print(f'  {k}: {v:+.4f}')
