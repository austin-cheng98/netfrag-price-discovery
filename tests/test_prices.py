from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
SRC = Path(__file__).resolve().parent.parent / 'src'
sys.path.insert(0, str(SRC))
import prices
from prices import AlphaVantagePrices, NasdaqPrices, StooqPrices, TiingoPrices, YFinancePrices, OHLCV_COLUMNS, daily_returns, get_prices, _normalize_frame

def _ohlcv(prices_list, adj=None):
    n = len(prices_list)
    idx = pd.date_range('2024-01-02', periods=n, freq='D', tz='UTC')
    return pd.DataFrame({'open': prices_list, 'high': [p + 1 for p in prices_list], 'low': [p - 1 for p in prices_list], 'close': prices_list, 'adj_close': prices_list if adj is None else adj, 'volume': [1000 * (i + 1) for i in range(n)]}, index=idx)

def test_daily_returns_values():
    df = _ohlcv([100.0, 110.0, 99.0, 99.0])
    r = daily_returns(df)
    assert len(r) == 3
    assert r.name == 'return'
    np.testing.assert_allclose(r.values, [0.1, -0.1, 0.0], rtol=1e-09)
    assert str(r.index.tz) == 'UTC'

def test_daily_returns_uses_adj_close_not_close():
    df = _ohlcv([100.0, 100.0, 100.0], adj=[50.0, 55.0, 60.5])
    r = daily_returns(df)
    np.testing.assert_allclose(r.values, [0.1, 0.1], rtol=1e-09)

def test_daily_returns_adj_close_nan_falls_back_to_close():
    df = _ohlcv([100.0, 120.0, 132.0])
    df['adj_close'] = [np.nan, np.nan, np.nan]
    r = daily_returns(df)
    np.testing.assert_allclose(r.values, [0.2, 0.1], rtol=1e-09)

def test_daily_returns_empty():
    r = daily_returns(prices._empty_frame())
    assert len(r) == 0
    assert r.name == 'return'

def test_normalize_fills_missing_adj_close_from_close():
    df = pd.DataFrame({'open': [1.0], 'high': [2.0], 'low': [0.5], 'close': [1.5], 'volume': [10]}, index=pd.to_datetime(['2024-03-01']))
    out = _normalize_frame(df)
    assert list(out.columns) == OHLCV_COLUMNS
    assert out['adj_close'].iloc[0] == 1.5
    assert str(out.index.tz) == 'UTC'

def test_normalize_sorts_and_dedupes():
    idx = pd.to_datetime(['2024-01-03', '2024-01-01', '2024-01-01'])
    df = pd.DataFrame({'open': [3, 1, 9], 'high': [3, 1, 9], 'low': [3, 1, 9], 'close': [3.0, 1.0, 9.0], 'adj_close': [3.0, 1.0, 9.0], 'volume': [1, 1, 1]}, index=idx)
    out = _normalize_frame(df)
    assert out.index.is_monotonic_increasing
    assert len(out) == 2
    assert out.loc[out.index[0], 'close'] == 9.0

def test_normalize_naive_index_localized_to_utc():
    df = _ohlcv([1.0, 2.0])
    df.index = df.index.tz_convert(None)
    out = _normalize_frame(df)
    assert str(out.index.tz) == 'UTC'

def test_nasdaq_parse():
    payload = {'data': {'tradesTable': {'rows': [{'date': '01/03/2024', 'open': '$100.00', 'high': '$102.00', 'low': '$99.00', 'close': '$101.00', 'volume': '1,234,567'}, {'date': '01/02/2024', 'open': '$98.00', 'high': '$99.00', 'low': '$97.00', 'close': '$98.50', 'volume': '1,000,000'}]}}}
    df = NasdaqPrices._parse(payload)
    assert list(df.columns) == OHLCV_COLUMNS
    assert df.index.is_monotonic_increasing
    assert df['close'].iloc[0] == 98.5
    assert df['volume'].iloc[0] == 1000000
    assert df['adj_close'].iloc[0] == 98.5
    assert str(df.index.tz) == 'UTC'

def test_tiingo_parse():
    rows = [{'date': '2024-01-02T00:00:00.000Z', 'open': 10, 'high': 11, 'low': 9, 'close': 10.5, 'adjClose': 5.25, 'volume': 500}, {'date': '2024-01-03T00:00:00.000Z', 'open': 10.5, 'high': 12, 'low': 10, 'close': 11.0, 'adjClose': 5.5, 'volume': 600}]
    df = TiingoPrices._parse(rows)
    assert df['adj_close'].iloc[0] == 5.25
    assert df['close'].iloc[1] == 11.0
    assert str(df.index.tz) == 'UTC'

def test_alpha_vantage_parse():
    payload = {'Time Series (Daily)': {'2024-01-02': {'1. open': '100.0', '2. high': '101.0', '3. low': '99.0', '4. close': '100.5', '5. adjusted close': '50.25', '6. volume': '1000'}}}
    df = AlphaVantagePrices._parse(payload)
    assert df['adj_close'].iloc[0] == 50.25
    assert df['volume'].iloc[0] == 1000

def test_alpha_vantage_throttle_note_raises():
    with pytest.raises(prices.SourceUnavailable):
        AlphaVantagePrices._parse({'Note': 'rate limited'})

def test_stooq_parse():
    text = 'Date,Open,High,Low,Close,Volume\n2024-01-02,100,101,99,100.5,1000\n2024-01-03,100.5,102,100,101.0,1100\n'
    df = StooqPrices._parse(text)
    assert list(df.columns) == OHLCV_COLUMNS
    assert df['close'].iloc[0] == 100.5
    assert df['adj_close'].iloc[0] == 100.5
    assert str(df.index.tz) == 'UTC'

def test_stooq_error_response_raises():
    with pytest.raises(prices.SourceUnavailable):
        StooqPrices._parse('No data')

def test_get_prices_reads_cache_without_network(tmp_path, monkeypatch):
    ticker = 'TESTX'
    monkeypatch.setattr(prices, '_PRICE_CACHE_DIR', tmp_path)
    cached = _normalize_frame(_ohlcv([10.0, 11.0, 12.0]))
    prices._write_cache(ticker, cached, '2024-01-01', '2024-12-31')

    class _Boom:
        name = 'boom'

        def get_daily(self, *a, **k):
            raise AssertionError('network hit despite valid cache')
    monkeypatch.setattr(prices, '_ADAPTERS', {n: _Boom for n in prices.CFG.price_sources})
    out = get_prices(ticker, '2024-01-01', '2024-12-31')
    assert len(out) == 3
    assert out['close'].iloc[-1] == 12.0
    assert str(out.index.tz) == 'UTC'

def test_get_prices_cache_miss_falls_through_and_writes(tmp_path, monkeypatch):
    ticker = 'MISSX'
    monkeypatch.setattr(prices, '_PRICE_CACHE_DIR', tmp_path)
    fake = _normalize_frame(_ohlcv([20.0, 21.0]))

    class _Good:
        name = 'good'

        def get_daily(self, t, s, e):
            return fake.copy()

    class _Skip:
        name = 'skip'

        def get_daily(self, t, s, e):
            raise prices.SourceUnavailable('no key')
    monkeypatch.setattr(prices.CFG, 'price_sources', ['skip', 'good'], raising=False)
    monkeypatch.setattr(prices, '_ADAPTERS', {'skip': _Skip, 'good': _Good})
    out = get_prices(ticker, '2024-01-01', '2024-12-31')
    assert len(out) == 2
    assert (tmp_path / f'{ticker}.parquet').exists()

def test_get_prices_all_sources_fail_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, '_PRICE_CACHE_DIR', tmp_path)

    class _Fail:
        name = 'fail'

        def get_daily(self, t, s, e):
            raise prices.SourceUnavailable('down')
    monkeypatch.setattr(prices.CFG, 'price_sources', ['fail'], raising=False)
    monkeypatch.setattr(prices, '_ADAPTERS', {'fail': _Fail})
    with pytest.raises(prices.SourceUnavailable):
        get_prices('NOPE', '2024-01-01', '2024-12-31')

def test_keyed_sources_skip_when_no_key():
    if prices.KEYS.tiingo is None:
        with pytest.raises(prices.SourceUnavailable):
            TiingoPrices().get_daily('AAPL', '2024-01-01', '2024-02-01')
    if prices.KEYS.alpha_vantage is None:
        with pytest.raises(prices.SourceUnavailable):
            AlphaVantagePrices().get_daily('AAPL', '2024-01-01', '2024-02-01')

def test_adapter_names():
    assert YFinancePrices().name == 'yfinance'
    assert NasdaqPrices().name == 'nasdaq'
    assert TiingoPrices().name == 'tiingo'
    assert AlphaVantagePrices().name == 'alpha_vantage'
    assert StooqPrices().name == 'stooq'

def test_live_yfinance_smoke():
    try:
        df = YFinancePrices().get_daily('AAPL', '2024-01-02', '2024-01-12')
    except Exception as exc:
        pytest.skip(f'live yfinance unavailable: {exc!r}')
    if len(df) == 0:
        pytest.skip('live yfinance returned empty (rate-limited)')
    assert list(df.columns) == OHLCV_COLUMNS
    assert str(df.index.tz) == 'UTC'
    assert df.index.is_monotonic_increasing
