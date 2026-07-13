from __future__ import annotations
import os
import sys
import pandas as pd
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import CFG
from contracts import EarningsEvent
import earnings as E

def _utc(y, m, d, h, mi):
    return pd.Timestamp(year=y, month=m, day=d, hour=h, minute=mi, tz='UTC')

def test_infer_session_boundaries():
    assert E.infer_session(_utc(2024, 3, 1, 12, 0), True) == 'bmo'
    assert E.infer_session(_utc(2024, 3, 1, 14, 29), True) == 'bmo'
    assert E.infer_session(_utc(2024, 3, 1, 14, 30), True) == 'dmt'
    assert E.infer_session(_utc(2024, 3, 1, 18, 0), True) == 'dmt'
    assert E.infer_session(_utc(2024, 3, 1, 20, 59), True) == 'dmt'
    assert E.infer_session(_utc(2024, 3, 1, 21, 0), True) == 'amc'
    assert E.infer_session(_utc(2024, 3, 2, 0, 30), True) == 'amc'
    assert E.infer_session(_utc(2024, 1, 8, 20, 30), True) == 'dmt'
    assert E.infer_session(_utc(2024, 1, 9, 2, 0), True) == 'amc'
    assert E.infer_session(_utc(2024, 7, 8, 20, 30), True) == 'amc'
    assert E.infer_session(_utc(2024, 3, 1, 12, 0), False) == 'unknown'

def test_infer_session_requires_tzaware():
    naive = pd.Timestamp('2024-03-01 12:00:00')
    with pytest.raises(ValueError):
        E.infer_session(naive, True)

def test_infer_session_converts_non_utc():
    ts = pd.Timestamp('2024-06-03 09:00', tz='America/New_York')
    assert E.infer_session(ts, True) == 'bmo'
    ts2 = pd.Timestamp('2024-06-03 16:05', tz='America/New_York')
    assert E.infer_session(ts2, True) == 'amc'

def test_compute_surprise_basic():
    s, sp = E.compute_surprise(1.53, 1.5)
    assert s == pytest.approx(0.03)
    assert sp == pytest.approx(100 * 0.03 / 1.5)

def test_compute_surprise_negative_estimate_uses_abs():
    s, sp = E.compute_surprise(-0.1, -0.2)
    assert s == pytest.approx(0.1)
    assert sp == pytest.approx(100 * 0.1 / 0.2)

def test_compute_surprise_zero_and_none_estimate():
    s, sp = E.compute_surprise(0.5, 0.0)
    assert s == pytest.approx(0.5)
    assert sp is None
    s, sp = E.compute_surprise(0.5, None)
    assert s is None and sp is None
    s, sp = E.compute_surprise(None, 1.0)
    assert s is None and sp is None

def test_to_float_coercions():
    assert E._to_float('$1,234.5') == pytest.approx(1234.5)
    assert E._to_float('(0.12)') == pytest.approx(-0.12)
    assert E._to_float('--') is None
    assert E._to_float('') is None
    assert E._to_float(float('nan')) is None
    assert E._to_float(None) is None
    assert E._to_float('12%') == pytest.approx(12.0)

def _fake_yf_df():
    idx = pd.DatetimeIndex([pd.Timestamp('2024-05-02 16:30', tz='America/New_York'), pd.Timestamp('2024-02-01 07:00', tz='America/New_York'), pd.Timestamp('2019-01-01 16:00', tz='America/New_York')], name='Earnings Date')
    return pd.DataFrame({'EPS Estimate': [1.5, 2.1, 0.9], 'Reported EPS': [1.53, 2.0, 0.95], 'Surprise(%)': [2.0, -4.76, 5.55]}, index=idx)

def test_yfinance_parse_tzaware_utc_and_session():
    evs = E.YFinanceEarnings.parse('aapl', _fake_yf_df(), '2022-07-01', '2024-12-31')
    assert len(evs) == 2
    by_date = {e.announce_utc.date().isoformat(): e for e in evs}
    amc = by_date['2024-05-02']
    assert amc.announce_utc.tzinfo is not None
    assert str(amc.announce_utc.tz) == 'UTC'
    assert amc.announce_utc.hour == 20 and amc.announce_utc.minute == 30
    assert amc.session == 'amc'
    assert amc.ticker == 'AAPL'
    assert amc.eps_actual == pytest.approx(1.53)
    assert amc.eps_estimate == pytest.approx(1.5)
    assert amc.surprise == pytest.approx(0.03)
    assert amc.surprise_pct == pytest.approx(2.0)
    assert amc.event_id == 'AAPL:2024-05-02'
    assert amc.source == 'yfinance'
    bmo = by_date['2024-02-01']
    assert bmo.session == 'bmo'
    assert bmo.surprise == pytest.approx(-0.1)

def test_yfinance_parse_falls_back_to_feed_surprise_pct():
    idx = pd.DatetimeIndex([pd.Timestamp('2024-05-02 16:30', tz='America/New_York')], name='Earnings Date')
    df = pd.DataFrame({'EPS Estimate': [float('nan')], 'Reported EPS': [1.53], 'Surprise(%)': [7.7]}, index=idx)
    ev = E.YFinanceEarnings.parse('NVDA', df, '2022-07-01', '2024-12-31')[0]
    assert ev.eps_estimate is None
    assert ev.surprise is None
    assert ev.surprise_pct == pytest.approx(7.7)

def test_finnhub_parse():
    data = [{'period': '2024-05-02', 'actual': 1.53, 'estimate': 1.5, 'symbol': 'AAPL', 'quarter': 2, 'year': 2024}, {'period': '2018-01-01', 'actual': 1.0, 'estimate': 0.9}]
    evs = E.FinnhubEarnings.parse('AAPL', data, '2022-07-01', '2024-12-31')
    assert len(evs) == 1
    ev = evs[0]
    assert ev.session == 'unknown'
    assert str(ev.announce_utc.tz) == 'UTC'
    assert ev.fiscal_period == '2024Q2'
    assert ev.surprise_pct == pytest.approx(100 * 0.03 / 1.5)
    assert ev.source == 'finnhub'

def test_finnhub_no_key_raises():
    src = E.FinnhubEarnings(token=None)
    with pytest.raises(E.SourceUnavailable):
        src.get_events('AAPL', '2022-07-01', '2024-12-31')

def test_nasdaq_parse_and_session_hint():
    payload = {'data': {'rows': [{'symbol': 'AAPL', 'eps': '$1.53', 'epsForecast': '$1.50', 'time': 'time-after-hours'}, {'symbol': 'MSFT', 'eps': '2.90', 'epsForecast': '2.80', 'time': 'time-pre-market'}]}}
    evs = E.NasdaqEarnings.parse('AAPL', '2024-05-02', payload)
    assert len(evs) == 1
    ev = evs[0]
    assert ev.ticker == 'AAPL'
    assert ev.session == 'amc'
    assert ev.eps_actual == pytest.approx(1.53)
    assert ev.eps_estimate == pytest.approx(1.5)
    assert str(ev.announce_utc.tz) == 'UTC'

def test_nasdaq_per_ticker_defers():
    with pytest.raises(E.SourceUnavailable):
        E.NasdaqEarnings().get_events('AAPL', '2022-07-01', '2024-12-31')

class _FakeSource:

    def __init__(self, name, events=None, exc=None):
        self.name = name
        self._events = events or []
        self._exc = exc

    def get_events(self, ticker, start, end):
        if self._exc is not None:
            raise self._exc
        return list(self._events)

def _ev(ticker, date, session='unknown', actual=None, estimate=None, source='x'):
    return E._make_event(ticker, pd.Timestamp(date, tz='UTC'), has_time=session != 'unknown', actual=actual, estimate=estimate, source=source) if session == 'unknown' else EarningsEvent(ticker=ticker, announce_utc=pd.Timestamp(date, tz='UTC'), session=session, eps_actual=actual, eps_estimate=estimate, source=source)

def test_fallback_first_working_source_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(E, 'RAW', tmp_path)
    e_good = EarningsEvent(ticker='AAPL', announce_utc=pd.Timestamp('2024-05-02', tz='UTC'), session='unknown', eps_actual=1.53, eps_estimate=1.5, source='yfinance')
    factories = {'finnhub': _FakeSource('finnhub', exc=E.SourceUnavailable('no key')), 'nasdaq': _FakeSource('nasdaq', exc=E.SourceUnavailable('by-date only')), 'yfinance': _FakeSource('yfinance', events=[e_good])}
    out = E.get_earnings('AAPL', '2022-07-01', '2024-12-31', cfg=CFG, use_cache=False, _factories=factories)
    assert len(out) == 1 and out[0].source == 'yfinance'
    assert (tmp_path / 'earnings' / 'AAPL.json').exists()

def test_fallback_all_fail_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(E, 'RAW', tmp_path)
    factories = {'finnhub': _FakeSource('finnhub', exc=E.SourceUnavailable('no key')), 'nasdaq': _FakeSource('nasdaq', exc=E.SourceUnavailable('defer')), 'yfinance': _FakeSource('yfinance', exc=E.SourceUnavailable('429'))}
    with pytest.raises(E.SourceUnavailable):
        E.get_earnings('AAPL', '2022-07-01', '2024-12-31', cfg=CFG, use_cache=False, _factories=factories)

def test_dedup_prefers_complete_record():
    sparse = EarningsEvent(ticker='AAPL', announce_utc=pd.Timestamp('2024-05-02 20:30', tz='UTC'), session='amc', eps_actual=None, eps_estimate=None, source='finnhub')
    full = EarningsEvent(ticker='AAPL', announce_utc=pd.Timestamp('2024-05-02 20:31', tz='UTC'), session='amc', eps_actual=1.53, eps_estimate=1.5, source='yfinance')
    out = E._dedup([sparse, full])
    assert len(out) == 1
    assert out[0].source == 'yfinance'
    assert out[0].eps_actual == pytest.approx(1.53)

def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(E, 'RAW', tmp_path)
    ev = EarningsEvent(ticker='AAPL', announce_utc=pd.Timestamp('2024-05-02 20:30', tz='UTC'), session='amc', eps_actual=1.53, eps_estimate=1.5, surprise=0.03, surprise_pct=2.0, source='yfinance')
    E._write_cache('AAPL', [ev], '2015-01-01', '2025-01-01')
    back = E._read_cache('AAPL')
    assert back is not None and len(back) == 1
    r = back[0]
    assert r.ticker == 'AAPL'
    assert str(r.announce_utc.tz) == 'UTC'
    assert r.announce_utc.hour == 20 and r.announce_utc.minute == 30
    assert r.eps_actual == pytest.approx(1.53)
    assert r.session == 'amc'

def test_get_earnings_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(E, 'RAW', tmp_path)
    ev = EarningsEvent(ticker='AAPL', announce_utc=pd.Timestamp('2024-05-02', tz='UTC'), session='unknown', eps_actual=1.53, eps_estimate=1.5, source='yfinance')
    E._write_cache('AAPL', [ev], '2015-01-01', '2025-01-01')
    boom = {'finnhub': _FakeSource('finnhub', exc=RuntimeError('should not be called'))}
    out = E.get_earnings('AAPL', '2022-07-01', '2024-12-31', cfg=CFG, use_cache=True, _factories=boom)
    assert len(out) == 1 and out[0].eps_actual == pytest.approx(1.53)

def test_get_earnings_refilters_cache_to_window(tmp_path, monkeypatch):
    monkeypatch.setattr(E, 'RAW', tmp_path)
    in_win = EarningsEvent(ticker='AAPL', announce_utc=pd.Timestamp('2024-02-05', tz='UTC'), session='unknown', eps_actual=1.5, eps_estimate=1.4, source='yfinance')
    out_win = EarningsEvent(ticker='AAPL', announce_utc=pd.Timestamp('2024-05-02', tz='UTC'), session='unknown', eps_actual=1.53, eps_estimate=1.5, source='yfinance')
    E._write_cache('AAPL', [in_win, out_win], '2015-01-01', '2025-01-01')
    boom = {'finnhub': _FakeSource('finnhub', exc=RuntimeError('should not be called'))}
    out = E.get_earnings('AAPL', '2024-02-01', '2024-02-15', cfg=CFG, use_cache=True, _factories=boom)
    assert len(out) == 1
    assert out[0].announce_utc.date().isoformat() == '2024-02-05'

def test_get_all_earnings_skips_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(E, 'RAW', tmp_path)
    e_good = EarningsEvent(ticker='AAPL', announce_utc=pd.Timestamp('2024-05-02', tz='UTC'), session='unknown', eps_actual=1.53, eps_estimate=1.5, source='yfinance')

    def factory_for(tkr):
        if tkr == 'AAPL':
            return {'yfinance': _FakeSource('yfinance', events=[e_good])}
        return {'yfinance': _FakeSource('yfinance', exc=E.SourceUnavailable('429'))}

    class _PerTicker:
        name = 'yfinance'

        def get_events(self, ticker, start, end):
            if ticker == 'AAPL':
                return [e_good]
            raise E.SourceUnavailable('429')
    factories = {'yfinance': _PerTicker()}
    import dataclasses
    cfg2 = dataclasses.replace(CFG, earnings_sources=['yfinance'])
    out = E.get_all_earnings(['AAPL', 'TSLA'], '2022-07-01', '2024-12-31', cfg=cfg2, use_cache=False, _factories=factories)
    tickers = {e.ticker for e in out}
    assert tickers == {'AAPL'}

@pytest.mark.skipif(os.getenv('NETFRAG_LIVE') != '1', reason='live smoke disabled (set NETFRAG_LIVE=1 to enable)')
def test_live_yfinance_smoke():
    src = E.YFinanceEarnings(limit=8)
    evs = src.get_events('AAPL', '2022-07-01', '2024-12-31')
    assert evs and all((str(e.announce_utc.tz) == 'UTC' for e in evs))
    assert all((isinstance(e, EarningsEvent) for e in evs))
