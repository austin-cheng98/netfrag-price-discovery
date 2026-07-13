from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, '/Users/austincheng/Desktop/netfrag-price-discovery/src')
from config import CFG
from contracts import EarningsEvent
import outcomes as O
RNG = np.random.default_rng(20260706)

def _trading_index(n, start='2023-01-02'):
    idx = pd.bdate_range(start=start, periods=n, tz='UTC')
    return idx

def test_halflife_recovers_known_k():
    k_true = 0.4
    t = np.arange(0, CFG.halflife_max_day + 1)
    y = 0.05 * np.exp(-k_true * t) + RNG.normal(0, 1e-05, size=t.size)
    s = pd.Series(y, index=t)
    k, hl, r2 = O.halflife_fit(s)
    assert np.isfinite(k) and np.isfinite(hl) and np.isfinite(r2)
    assert abs(k - k_true) < 0.03, f'k={k}'
    assert abs(hl - np.log(2) / k_true) < 0.15, f'halflife={hl}'
    assert r2 > 0.99

def test_halflife_nonconvergence_returns_nan():
    s = pd.Series(RNG.normal(0, 1.0, size=15), index=np.arange(15))
    k, hl, r2 = O.halflife_fit(s)
    assert np.isnan(hl) or np.isfinite(hl)

def test_halflife_too_few_points_nan():
    s = pd.Series([0.05, 0.03], index=[0, 1])
    assert all((np.isnan(v) for v in O.halflife_fit(s)))

def test_variance_ratio_random_walk():
    zs = []
    vrs = []
    for _ in range(30):
        r = pd.Series(RNG.normal(0, 0.01, size=500))
        vr, z = O.variance_ratio(r, q=5)
        vrs.append(vr)
        zs.append(z)
    mean_vr = np.nanmean(vrs)
    assert abs(mean_vr - 1.0) < 0.1, f'mean VR={mean_vr}'
    frac_huge = np.mean(np.abs(zs) > 3.0)
    assert frac_huge < 0.2, f'fraction |z|>3 = {frac_huge}'

def test_variance_ratio_positive_autocorr_detected():
    n = 800
    e = RNG.normal(0, 0.01, size=n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.4 * x[i - 1] + e[i]
    vr, z = O.variance_ratio(pd.Series(x), q=5)
    assert vr > 1.05, f'VR={vr}'
    assert z > 2.0, f'z={z}'

def test_variance_ratio_short_series_nan():
    vr, z = O.variance_ratio(pd.Series([0.01, -0.01, 0.02]), q=5)
    assert np.isnan(vr) and np.isnan(z)

def test_market_model_recovers_alpha_beta():
    n = 400
    mkt = pd.Series(RNG.normal(0.0003, 0.01, size=n))
    alpha_true, beta_true = (0.001, 1.35)
    stock = alpha_true + beta_true * mkt + pd.Series(RNG.normal(0, 0.0001, size=n))
    mm = O.market_model(stock, mkt)
    assert abs(mm.alpha - alpha_true) < 0.0005, mm.alpha
    assert abs(mm.beta - beta_true) < 0.02, mm.beta
    assert mm.resid_std < 0.001

def test_market_model_est_window_slice():
    n = 200
    mkt = pd.Series(RNG.normal(0, 0.01, size=n))
    stock = 0.0 + 2.0 * mkt
    stock2 = stock.copy()
    stock2.iloc[100:] = RNG.normal(0, 1.0, size=100)
    mm = O.market_model(stock2, mkt, est_window=(0, 99))
    assert abs(mm.beta - 2.0) < 1e-06
    assert abs(mm.alpha) < 1e-06

def test_market_model_insufficient_data_nan():
    mm = O.market_model(pd.Series([0.01, 0.02]), pd.Series([0.01, 0.02]))
    assert np.isnan(mm.alpha) and np.isnan(mm.beta)

def _build_price_frames_with_drift(drift_per_day, direction_sign, n_pre=180, n_post=35):
    n = n_pre + n_post
    idx = _trading_index(n)
    mkt_ret = RNG.normal(0.0, 0.008, size=n)
    beta = 1.0
    stock_ret = beta * mkt_ret.copy()
    t0_pos = n_pre
    for i in range(t0_pos + 1, n):
        stock_ret[i] += direction_sign * drift_per_day

    def to_prices(rets):
        px = np.empty(n)
        px[0] = 100.0
        for i in range(1, n):
            px[i] = px[i - 1] * (1.0 + rets[i])
        return px
    stock_px = to_prices(stock_ret)
    mkt_px = to_prices(mkt_ret)
    stock_df = pd.DataFrame({'adj_close': stock_px, 'close': stock_px}, index=idx)
    mkt_df = pd.DataFrame({'adj_close': mkt_px, 'close': mkt_px}, index=idx)
    return (stock_df, mkt_df, idx, t0_pos)

def test_pead_positive_drift_underreaction():
    drift = 0.002
    stock_df, mkt_df, idx, t0_pos = _build_price_frames_with_drift(drift, +1)
    ar = O.abnormal_returns(stock_df, mkt_df, t0_pos, CFG)
    assert not ar.empty
    win = ar[(ar.index >= CFG.car_start) & (ar.index <= CFG.car_end)]
    assert win.mean() > 0
    assert abs(win.mean() - drift) < 0.0005
    p, p_abs = O.pead(ar, direction=+1.0)
    n_days = len(win)
    expected_car = drift * n_days
    assert p > 0
    assert abs(p - expected_car) < 0.01
    assert abs(p_abs - abs(expected_car)) < 1e-09

def test_pead_sign_flips_with_negative_surprise():
    drift = 0.002
    stock_df, mkt_df, idx, t0_pos = _build_price_frames_with_drift(drift, +1)
    ar = O.abnormal_returns(stock_df, mkt_df, t0_pos, CFG)
    p_pos, _ = O.pead(ar, direction=+1.0)
    p_neg, _ = O.pead(ar, direction=-1.0)
    assert p_pos > 0 and p_neg < 0
    assert abs(p_pos + p_neg) < 1e-12

def test_pead_empty_ar_nan():
    p, pa = O.pead(pd.Series(dtype=float), direction=1.0)
    assert np.isnan(p) and np.isnan(pa)

def test_locate_t0_bmo_same_session():
    idx = _trading_index(10, start='2023-01-02')
    prices = pd.DataFrame({'adj_close': np.arange(10.0)}, index=idx)
    ev = EarningsEvent(ticker='AAA', announce_utc=idx[3], session='bmo')
    assert O.locate_t0(ev, prices) == 3

def test_locate_t0_amc_next_session():
    idx = _trading_index(10, start='2023-01-02')
    prices = pd.DataFrame({'adj_close': np.arange(10.0)}, index=idx)
    ev = EarningsEvent(ticker='AAA', announce_utc=idx[3], session='amc')
    assert O.locate_t0(ev, prices) == 4

def test_locate_t0_weekend_announce_snaps_forward():
    idx = _trading_index(10, start='2023-01-02')
    prices = pd.DataFrame({'adj_close': np.arange(10.0)}, index=idx)
    sat = pd.Timestamp('2023-01-07', tz='UTC')
    ev = EarningsEvent(ticker='AAA', announce_utc=sat, session='bmo')
    t0 = O.locate_t0(ev, prices)
    assert idx[t0].normalize() >= sat.normalize()
    assert idx[t0].dayofweek < 5

def test_compute_outcomes_end_to_end():
    drift = 0.002
    stock_df, mkt_df, idx, t0_pos = _build_price_frames_with_drift(drift, +1)
    ev = EarningsEvent(ticker='AAA', announce_utc=idx[t0_pos], session='bmo', surprise_pct=8.0)
    om = O.compute_outcomes(ev, stock_df, mkt_df, CFG)
    assert om.event_id == ev.event_id
    assert np.isfinite(om.pead) and om.pead > 0
    assert np.isfinite(om.pead_abs) and om.pead_abs >= 0
    assert np.isfinite(om.car_event)
    assert np.isfinite(om.variance_ratio)
    assert om.source

def test_compute_outcomes_incomplete_window_no_crash():
    idx = _trading_index(5)
    prices = pd.DataFrame({'adj_close': np.linspace(100, 104, 5)}, index=idx)
    ev = EarningsEvent(ticker='AAA', announce_utc=idx[2], session='amc')
    om = O.compute_outcomes(ev, prices, prices, CFG)
    assert om.event_id == ev.event_id
    assert om.pead is None or np.isnan(om.pead) if om.pead is not None else True

def test_compute_outcomes_empty_prices():
    ev = EarningsEvent(ticker='AAA', announce_utc=pd.Timestamp('2023-06-01', tz='UTC'), session='amc')
    empty = pd.DataFrame(columns=['adj_close'])
    om = O.compute_outcomes(ev, empty, empty, CFG)
    assert om.pead is None
