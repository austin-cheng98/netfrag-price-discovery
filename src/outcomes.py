from __future__ import annotations
import sys
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
sys.path.insert(0, '/Users/austincheng/Desktop/netfrag-price-discovery/src')
from config import CFG, Config
from contracts import EarningsEvent, OutcomeMeasures
try:
    from scipy.optimize import curve_fit
except Exception:
    curve_fit = None

def _price_col(df: pd.DataFrame) -> str:
    for c in ('adj_close', 'adjclose', 'adj close', 'close', 'Close', 'Adj Close'):
        if c in df.columns:
            return c
    raise KeyError(f"No usable price column in {list(df.columns)}; expected 'adj_close' or 'close'.")

def _simple_returns(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) == 0:
        return pd.Series(dtype='float64')
    d = df.sort_index()
    px = pd.to_numeric(d[_price_col(d)], errors='coerce')
    ret = px.pct_change()
    return ret

def _to_utc_ts(x) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    else:
        ts = ts.tz_convert('UTC')
    return ts

@dataclass
class MarketModel:
    alpha: float
    beta: float
    resid_std: float

def market_model(stock_ret: pd.Series, mkt_ret: pd.Series, est_window: Optional[tuple]=None) -> MarketModel:
    s = stock_ret.copy()
    m = mkt_ret.copy()
    if est_window is not None:
        lo, hi = est_window
        s = s.loc[lo:hi]
        m = m.loc[lo:hi]
    df = pd.concat([s.rename('y'), m.rename('x')], axis=1).dropna()
    if len(df) < 3:
        return MarketModel(np.nan, np.nan, np.nan)
    x = df['x'].to_numpy(dtype=float)
    y = df['y'].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    try:
        coef, _res, rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return MarketModel(np.nan, np.nan, np.nan)
    if rank < 2:
        return MarketModel(np.nan, np.nan, np.nan)
    alpha, beta = (float(coef[0]), float(coef[1]))
    resid = y - X @ coef
    dof = len(y) - 2
    resid_std = float(np.sqrt(np.sum(resid ** 2) / dof)) if dof > 0 else np.nan
    return MarketModel(alpha, beta, resid_std)

def abnormal_returns(prices: pd.DataFrame, market: pd.DataFrame, t0_idx: int, cfg: Config=CFG) -> pd.Series:
    stock_ret = _simple_returns(prices)
    mkt_ret = _simple_returns(market)
    if stock_ret.empty or mkt_ret.empty:
        return pd.Series(dtype='float64')
    joined = pd.concat([stock_ret.rename('s'), mkt_ret.rename('m')], axis=1).dropna()
    if joined.empty:
        return pd.Series(dtype='float64')
    if not 0 <= t0_idx < len(prices):
        return pd.Series(dtype='float64')
    t0_label = prices.sort_index().index[t0_idx]
    if t0_label not in joined.index:
        after = joined.index[joined.index >= t0_label]
        if len(after) == 0:
            return pd.Series(dtype='float64')
        t0_label = after[0]
    t0_pos = joined.index.get_loc(t0_label)
    est_end = t0_pos - cfg.estimation_gap
    est_start = est_end - cfg.estimation_days
    if est_start < 0 or est_end <= est_start:
        return pd.Series(dtype='float64')
    est = joined.iloc[est_start:est_end]
    mm = market_model(est['s'], est['m'])
    if not np.isfinite(mm.alpha) or not np.isfinite(mm.beta):
        return pd.Series(dtype='float64')
    horizon_end = min(len(joined) - 1, t0_pos + cfg.car_end)
    horizon = joined.iloc[est_start:horizon_end + 1]
    ar = horizon['s'] - (mm.alpha + mm.beta * horizon['m'])
    event_days = np.arange(est_start - t0_pos, horizon_end - t0_pos + 1)
    ar.index = event_days
    ar.name = 'AR'
    return ar

def pead(AR: pd.Series, direction: float, cfg: Config=CFG) -> tuple[float, float]:
    if AR is None or len(AR) == 0:
        return (np.nan, np.nan)
    lo, hi = (cfg.car_start, cfg.car_end)
    window = AR[(AR.index >= lo) & (AR.index <= hi)]
    if window.empty:
        return (np.nan, np.nan)
    car = float(window.sum())
    sign = np.sign(direction) if direction is not None and np.isfinite(direction) and (direction != 0) else 1.0
    return (float(sign * car), float(abs(car)))

def _exp_decay(t, a, k):
    return a * np.exp(-k * t)

def halflife_fit(abs_AR_or_gap: pd.Series, cfg: Config=CFG) -> tuple[float, float, float]:
    nan3 = (np.nan, np.nan, np.nan)
    if curve_fit is None or abs_AR_or_gap is None:
        return nan3
    s = pd.Series(abs_AR_or_gap).dropna()
    if s.empty:
        return nan3
    s = s[(s.index >= 0) & (s.index <= cfg.halflife_max_day)]
    if len(s) < 3:
        return nan3
    t = s.index.to_numpy(dtype=float)
    y = s.to_numpy(dtype=float)
    if not np.all(np.isfinite(y)) or np.allclose(y, 0.0):
        return nan3
    a0 = float(y[0]) if np.isfinite(y[0]) and y[0] != 0 else float(np.nanmax(np.abs(y)) or 1.0)
    p0 = [a0, 0.3]
    try:
        popt, _pcov = curve_fit(_exp_decay, t, y, p0=p0, maxfev=10000, bounds=([-np.inf, 0.0], [np.inf, np.inf]))
    except Exception:
        return nan3
    a_hat, k_hat = (float(popt[0]), float(popt[1]))
    _MAX_HALFLIFE = 100.0 * float(cfg.halflife_max_day)
    k_min = np.log(2.0) / _MAX_HALFLIFE
    if not np.isfinite(k_hat) or k_hat <= k_min:
        return (k_hat if np.isfinite(k_hat) else np.nan, np.nan, np.nan)
    yhat = _exp_decay(t, a_hat, k_hat)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    halflife = float(np.log(2.0) / k_hat)
    return (k_hat, halflife, r2)

def variance_ratio(returns: pd.Series, q: int=5) -> tuple[float, float]:
    if returns is None:
        return (np.nan, np.nan)
    x = pd.Series(returns).dropna().to_numpy(dtype=float)
    n = x.size
    q = int(q)
    if q < 2 or n < q + 1:
        return (np.nan, np.nan)
    mu = x.mean()
    var_1 = np.sum((x - mu) ** 2) / (n - 1)
    if var_1 <= 0:
        return (np.nan, np.nan)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    q_sums = cs[q:] - cs[:-q]
    m = q * (n - q + 1) * (1.0 - q / n)
    if m <= 0:
        return (np.nan, np.nan)
    var_q = np.sum((q_sums - q * mu) ** 2) / m
    vr = float(var_q / var_1)
    e2 = (x - mu) ** 2
    denom = np.sum(e2) ** 2
    if denom <= 0:
        return (vr, np.nan)
    phi = 0.0
    for j in range(1, q):
        num = np.sum(e2[j:] * e2[:-j])
        delta_j = num / denom
        w = 2.0 * (q - j) / q
        phi += w ** 2 * delta_j
    if not np.isfinite(phi) or phi <= 0:
        return (vr, np.nan)
    z = float((vr - 1.0) / np.sqrt(phi))
    return (vr, z)

def locate_t0(event: EarningsEvent, prices: pd.DataFrame) -> Optional[int]:
    if prices is None or len(prices) == 0:
        return None
    idx = prices.sort_index().index
    if idx.tz is None:
        idx_utc = idx.tz_localize('UTC')
    else:
        idx_utc = idx.tz_convert('UTC')
    announce = _to_utc_ts(event.announce_utc)
    announce_day = announce.normalize()
    on_or_after = np.where(idx_utc.normalize() >= announce_day)[0]
    if on_or_after.size == 0:
        return None
    same_session = int(on_or_after[0])
    session = (event.session or '').lower()
    if session == 'bmo':
        return same_session
    nxt = same_session + 1
    if nxt >= len(idx_utc):
        return None
    return nxt

def compute_outcomes(event: EarningsEvent, prices: pd.DataFrame, market: pd.DataFrame, cfg: Config=CFG) -> OutcomeMeasures:
    om = OutcomeMeasures(event_id=event.event_id, ticker=event.ticker, source='outcomes.py')
    try:
        t0_idx = locate_t0(event, prices)
        if t0_idx is None:
            return om
        ar = abnormal_returns(prices, market, t0_idx, cfg)
        if ar.empty:
            return om
        immed = ar[(ar.index >= 0) & (ar.index <= 1)]
        if not immed.empty:
            om.car_event = float(immed.sum())
        direction = None
        for cand in (event.surprise_pct, event.surprise, om.car_event):
            if cand is not None and np.isfinite(cand) and (cand != 0):
                direction = float(cand)
                break
        p, p_abs = pead(ar, direction if direction is not None else 0.0, cfg)
        om.pead = p
        om.pead_abs = p_abs
        post_ar = ar[ar.index >= 0]
        k, hl, r2 = halflife_fit(post_ar.abs(), cfg)
        om.adjustment_speed_k = k
        om.halflife_days = hl
        om.car_r2 = r2
        post = ar[ar.index >= 0]
        if len(post) >= 6:
            vr, z = variance_ratio(post, q=5)
            om.variance_ratio = vr
            om.vr_stat = z
        post_win = ar[(ar.index >= cfg.car_start) & (ar.index <= cfg.car_end)]
        if len(post_win) >= 2:
            om.post_vol = float(post_win.std(ddof=1))
    except Exception:
        return om
    return om
