from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
_SRC = str(Path(__file__).resolve().parent.parent / 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from contracts import EarningsEvent, RedditItem
from config import CFG
import features as F

def _mk_event(session: str, announce: str, **kw) -> EarningsEvent:
    return EarningsEvent(ticker=kw.pop('ticker', 'AAPL'), announce_utc=pd.Timestamp(announce, tz='UTC'), session=session, **kw)

def _biz_index(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=periods, freq='B', tz='UTC')

def test_sector_map_covers_all_tickers():
    missing = [t for t in CFG.tickers if t not in F.SECTOR_MAP]
    assert not missing, f'SECTOR_MAP missing tickers: {missing}'
    assert all((isinstance(v, str) and v for v in F.SECTOR_MAP.values()))
    assert F.sector_for('ZZZZ') == 'Unknown'
    assert F.sector_for('aapl') == 'Information Technology'

def test_event_reddit_window_bounds_and_tz():
    ev = _mk_event('amc', '2023-05-04 20:30:00')
    w = F.event_reddit_window(ev, CFG)
    ann = pd.Timestamp('2023-05-04 20:30:00', tz='UTC')
    assert w['comm_after'] == ann - pd.Timedelta(hours=CFG.reddit_pre_hours)
    assert w['comm_before'] == ann + pd.Timedelta(hours=CFG.reddit_post_hours)
    assert w['treat_after'] == ann
    assert w['treat_before'] == ann + pd.Timedelta(hours=CFG.treat_window_hours)
    for v in w.values():
        assert v.tzinfo is not None and str(v.tz) == 'UTC'

def test_event_reddit_window_localizes_naive():
    ev = EarningsEvent(ticker='MSFT', announce_utc=pd.Timestamp('2023-01-10 12:00:00'), session='bmo')
    w = F.event_reddit_window(ev, CFG)
    assert str(w['treat_after'].tz) == 'UTC'
    assert w['treat_after'] == pd.Timestamp('2023-01-10 12:00:00', tz='UTC')

def test_bmo_maps_to_same_day():
    idx = _biz_index('2023-05-01', 15)
    ev = _mk_event('bmo', '2023-05-04 11:00:00')
    t0 = F.event_trading_day(ev, idx)
    assert t0.normalize() == pd.Timestamp('2023-05-04', tz='UTC')

def test_dmt_maps_to_same_day():
    idx = _biz_index('2023-05-01', 15)
    ev = _mk_event('dmt', '2023-05-04 17:00:00')
    t0 = F.event_trading_day(ev, idx)
    assert t0.normalize() == pd.Timestamp('2023-05-04', tz='UTC')

def test_amc_maps_to_next_trading_day():
    idx = _biz_index('2023-05-01', 15)
    ev = _mk_event('amc', '2023-05-04 21:00:00')
    t0 = F.event_trading_day(ev, idx)
    assert t0.normalize() == pd.Timestamp('2023-05-05', tz='UTC')

def test_amc_friday_maps_over_weekend_to_monday():
    idx = _biz_index('2023-05-01', 15)
    ev = _mk_event('amc', '2023-05-05 21:00:00')
    t0 = F.event_trading_day(ev, idx)
    assert t0.normalize() == pd.Timestamp('2023-05-08', tz='UTC')

def test_holiday_gap_bmo_skips_to_next_available():
    idx = _biz_index('2023-05-01', 15)
    idx = idx[idx.normalize() != pd.Timestamp('2023-05-04', tz='UTC')]
    ev = _mk_event('bmo', '2023-05-04 11:00:00')
    t0 = F.event_trading_day(ev, idx)
    assert t0.normalize() == pd.Timestamp('2023-05-05', tz='UTC')

def test_unknown_session_treated_as_next_day():
    idx = _biz_index('2023-05-01', 15)
    ev = _mk_event('unknown', '2023-05-04 11:00:00')
    t0 = F.event_trading_day(ev, idx)
    assert t0.normalize() == pd.Timestamp('2023-05-05', tz='UTC')

def test_trading_day_accepts_dataframe_and_naive_index():
    idx = pd.date_range('2023-05-01', periods=10, freq='B')
    df = pd.DataFrame({'close': np.arange(10.0)}, index=idx)
    ev = _mk_event('bmo', '2023-05-04 11:00:00')
    t0 = F.event_trading_day(ev, df)
    assert t0.normalize() == pd.Timestamp('2023-05-04', tz='UTC')

def _linear_price_frame(t0_date: str, n_pre: int, n_post: int, close0: float, step: float, vol: float) -> pd.DataFrame:
    total = n_pre + 1 + n_post
    start = pd.Timestamp(t0_date, tz='UTC') - pd.tseries.offsets.BDay(n_pre)
    idx = pd.date_range(start=start, periods=total, freq='B', tz='UTC')
    offsets = np.arange(-n_pre, n_post + 1)
    closes = close0 + step * offsets
    df = pd.DataFrame({'close': closes, 'volume': np.full(total, vol)}, index=idx)
    return df

def test_compute_controls_prior_return_hand_value():
    n_pre, n_post = (65, 5)
    total = n_pre + 1 + n_post
    t0_date = '2023-06-01'
    start = pd.Timestamp(t0_date, tz='UTC') - pd.tseries.offsets.BDay(n_pre)
    idx = pd.date_range(start=start, periods=total, freq='B', tz='UTC')
    offsets = np.arange(-n_pre, n_post + 1)
    closes = 100.0 * 1.01 ** offsets
    df = pd.DataFrame({'close': closes, 'volume': np.full(total, 1000000.0)}, index=idx)
    ev = _mk_event('bmo', '2023-06-01 11:00:00')
    cv = F.compute_controls(ev, df, None, [], CFG, allow_network=False)
    rets = df['close'].pct_change()
    lo = n_pre - 60
    hi = n_pre - 2
    expected = float(np.prod(1.0 + rets.iloc[lo:hi + 1].dropna().values) - 1.0)
    assert cv.prior_return == pytest.approx(expected, rel=1e-09)
    assert cv.prior_return > 0.5

def test_compute_controls_pre_vol_hand_value():
    n_pre, n_post = (40, 5)
    total = n_pre + 1 + n_post
    t0_date = '2023-06-01'
    start = pd.Timestamp(t0_date, tz='UTC') - pd.tseries.offsets.BDay(n_pre)
    idx = pd.date_range(start=start, periods=total, freq='B', tz='UTC')
    offsets = np.arange(-n_pre, n_post + 1)
    closes = 100.0 * 1.01 ** offsets
    df = pd.DataFrame({'close': closes, 'volume': np.full(total, 1000000.0)}, index=idx)
    ev = _mk_event('bmo', '2023-06-01 11:00:00')
    cv = F.compute_controls(ev, df, None, [], CFG, allow_network=False)
    assert cv.pre_vol == pytest.approx(0.0, abs=1e-12)

def test_compute_controls_pre_vol_alternating():
    n_pre, n_post = (40, 5)
    total = n_pre + 1 + n_post
    t0_date = '2023-06-01'
    start = pd.Timestamp(t0_date, tz='UTC') - pd.tseries.offsets.BDay(n_pre)
    idx = pd.date_range(start=start, periods=total, freq='B', tz='UTC')
    rng = np.random.default_rng(42)
    daily = rng.normal(0.0, 0.02, size=total)
    closes = 100.0 * np.cumprod(1.0 + daily)
    df = pd.DataFrame({'close': closes, 'volume': np.full(total, 1000000.0)}, index=idx)
    ev = _mk_event('bmo', '2023-06-01 11:00:00')
    cv = F.compute_controls(ev, df, None, [], CFG, allow_network=False)
    rets = df['close'].pct_change()
    t0_pos = n_pre
    expected = float(rets.iloc[t0_pos - 30:t0_pos - 6 + 1].dropna().std(ddof=1))
    assert cv.pre_vol == pytest.approx(expected, rel=1e-09)

def test_compute_controls_log_volume_hand_value():
    df = _linear_price_frame('2023-06-01', n_pre=10, n_post=5, close0=100.0, step=0.0, vol=1000000.0)
    ev = _mk_event('bmo', '2023-06-01 11:00:00')
    cv = F.compute_controls(ev, df, None, [], CFG, allow_network=False)
    assert cv.log_volume == pytest.approx(np.log(100000000.0), rel=1e-12)

def test_compute_controls_surprise_and_metadata():
    ev = _mk_event('amc', '2023-06-01 21:00:00', ticker='NVDA', surprise=0.25, surprise_pct=12.5)
    df = _linear_price_frame('2023-06-02', n_pre=65, n_post=5, close0=400.0, step=1.0, vol=2000000.0)
    items = [RedditItem(id='s1', kind='submission', subreddit='stocks', author='a', created_utc=pd.Timestamp('2023-06-01 21:30:00', tz='UTC'), body='x'), RedditItem(id='s2', kind='submission', subreddit='stocks', author='b', created_utc=pd.Timestamp('2023-06-02 10:00:00', tz='UTC'), body='y'), RedditItem(id='c1', kind='comment', subreddit='stocks', author='c', created_utc=pd.Timestamp('2023-06-01 22:00:00', tz='UTC'), body='z'), RedditItem(id='s3', kind='submission', subreddit='stocks', author='d', created_utc=pd.Timestamp('2023-05-01 00:00:00', tz='UTC'), body='old')]
    cv = F.compute_controls(ev, df, None, items, CFG, allow_network=False)
    assert cv.surprise_pct == 12.5
    assert cv.abs_surprise == pytest.approx(0.25)
    assert cv.sector == 'Information Technology'
    assert cv.n_comments == 4
    assert np.isnan(cv.log_mktcap)
    hours = CFG.reddit_pre_hours + CFG.reddit_post_hours
    assert cv.news_intensity == pytest.approx(2.0 / hours)

def test_compute_controls_empty_prices_graceful():
    ev = _mk_event('bmo', '2023-06-01 11:00:00', surprise=0.1, surprise_pct=5.0)
    cv = F.compute_controls(ev, pd.DataFrame(), None, None, CFG, allow_network=False)
    assert cv.surprise_pct == 5.0
    assert np.isnan(cv.pre_vol) and np.isnan(cv.prior_return) and np.isnan(cv.log_volume)
    assert cv.n_comments is None
    assert cv.news_intensity is None
    assert cv.sector == 'Information Technology'

def test_module_imports_only_allowed_deps():
    import importlib
    mod = importlib.reload(F)
    src = Path(mod.__file__).read_text()
    forbidden = ['import sources', 'import graphbuild', 'import outcomes', 'import fragmentation', 'import assemble', 'from sources', 'from outcomes', 'from fragmentation', 'from graphbuild']
    for f in forbidden:
        assert f not in src, f'features.py illegally imports: {f}'
