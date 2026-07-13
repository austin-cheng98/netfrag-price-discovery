from __future__ import annotations
import io
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from config import CFG, KEYS, RAW
OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
_PRICE_CACHE_DIR = RAW / 'prices'

class RateLimited(Exception):
    pass

class SourceUnavailable(Exception):
    pass

def _empty_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz='UTC', name='date')
    return pd.DataFrame(columns=OHLCV_COLUMNS, index=idx)

def _to_utc_index(idx) -> pd.DatetimeIndex:
    di = pd.DatetimeIndex(pd.to_datetime(idx))
    if di.tz is None:
        di = di.tz_localize('UTC')
    else:
        di = di.tz_convert('UTC')
    di = di.normalize()
    di.name = 'date'
    return di

def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return _empty_frame()
    df = df.copy()
    df.index = _to_utc_index(df.index)
    for col in OHLCV_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[OHLCV_COLUMNS]
    for col in OHLCV_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    missing_adj = df['adj_close'].isna()
    df.loc[missing_adj, 'adj_close'] = df.loc[missing_adj, 'close']
    df = df[df['close'].notna()]
    df = df[~df.index.duplicated(keep='last')]
    df = df.sort_index()
    return df

def _within(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if len(df) == 0:
        return df
    s = pd.Timestamp(start, tz='UTC').normalize()
    e = pd.Timestamp(end, tz='UTC').normalize()
    return df[(df.index >= s) & (df.index <= e)]

def _session():
    s = requests.Session()
    s.headers.update({'User-Agent': CFG.user_agent})
    return s

def _retry():
    return retry(reraise=True, retry=retry_if_exception_type((RateLimited, requests.exceptions.RequestException)), wait=wait_exponential(multiplier=CFG.backoff_base, min=1, max=60), stop=stop_after_attempt(CFG.max_retries))

class YFinancePrices:
    name = 'yfinance'

    def get_daily(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        import yfinance as yf
        end_excl = (pd.Timestamp(end).normalize() + pd.Timedelta(days=1)).date().isoformat()
        raw = yf.download(ticker, start=start, end=end_excl, auto_adjust=False, progress=False, threads=False)
        if raw is None or len(raw) == 0:
            raise SourceUnavailable(f'yfinance returned no rows for {ticker}')
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.droplevel(1, axis=1)
        rename = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'}
        raw = raw.rename(columns=rename)
        df = _normalize_frame(raw)
        return _within(df, start, end)

class NasdaqPrices:
    name = 'nasdaq'
    _URL = 'https://api.nasdaq.com/api/quote/{t}/historical'

    def get_daily(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        sess = _session()
        sess.headers.update({'Accept': 'application/json', 'Origin': 'https://www.nasdaq.com', 'Referer': 'https://www.nasdaq.com/'})
        fromdate = pd.Timestamp(start).date().isoformat()
        todate = pd.Timestamp(end).date().isoformat()
        params = {'assetclass': 'stocks', 'fromdate': fromdate, 'todate': todate, 'limit': 9999}

        @_retry()
        def _pull() -> dict:
            r = sess.get(self._URL.format(t=ticker), params=params, timeout=CFG.http_timeout)
            if r.status_code == 429:
                raise RateLimited('nasdaq 429')
            r.raise_for_status()
            return r.json()
        payload = _pull()
        return _within(self._parse(payload), start, end)

    @staticmethod
    def _parse(payload: dict) -> pd.DataFrame:
        data = (payload or {}).get('data') or {}
        trades = data.get('tradesTable') or {}
        rows = trades.get('rows') or []
        if not rows:
            raise SourceUnavailable('nasdaq returned no rows')

        def _num(x) -> float:
            if x is None:
                return np.nan
            s = str(x).replace('$', '').replace(',', '').strip()
            if s in ('', 'N/A', '--'):
                return np.nan
            try:
                return float(s)
            except ValueError:
                return np.nan
        recs = []
        for row in rows:
            recs.append({'date': row.get('date'), 'open': _num(row.get('open')), 'high': _num(row.get('high')), 'low': _num(row.get('low')), 'close': _num(row.get('close')), 'adj_close': np.nan, 'volume': _num(row.get('volume'))})
        df = pd.DataFrame(recs)
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
        df = df.set_index('date')
        return _normalize_frame(df)

class TiingoPrices:
    name = 'tiingo'
    _URL = 'https://api.tiingo.com/tiingo/daily/{t}/prices'

    def get_daily(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        if KEYS.tiingo is None:
            raise SourceUnavailable('tiingo: no API key configured')
        sess = _session()
        params = {'startDate': pd.Timestamp(start).date().isoformat(), 'endDate': pd.Timestamp(end).date().isoformat(), 'format': 'json', 'token': KEYS.tiingo}

        @_retry()
        def _pull() -> list:
            r = sess.get(self._URL.format(t=ticker), params=params, timeout=CFG.http_timeout)
            if r.status_code == 429:
                raise RateLimited('tiingo 429')
            r.raise_for_status()
            return r.json()
        rows = _pull()
        return _within(self._parse(rows), start, end)

    @staticmethod
    def _parse(rows: list) -> pd.DataFrame:
        if not rows:
            raise SourceUnavailable('tiingo returned no rows')
        df = pd.DataFrame(rows)
        out = pd.DataFrame({'open': df.get('open'), 'high': df.get('high'), 'low': df.get('low'), 'close': df.get('close'), 'adj_close': df.get('adjClose'), 'volume': df.get('volume')})
        out.index = pd.to_datetime(df['date'], errors='coerce')
        return _normalize_frame(out)

class AlphaVantagePrices:
    name = 'alpha_vantage'
    _URL = 'https://www.alphavantage.co/query'

    def get_daily(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        if KEYS.alpha_vantage is None:
            raise SourceUnavailable('alpha_vantage: no API key configured')
        sess = _session()
        params = {'function': 'TIME_SERIES_DAILY_ADJUSTED', 'symbol': ticker, 'outputsize': 'full', 'apikey': KEYS.alpha_vantage}

        @_retry()
        def _pull() -> dict:
            r = sess.get(self._URL, params=params, timeout=CFG.http_timeout)
            if r.status_code == 429:
                raise RateLimited('alpha_vantage 429')
            r.raise_for_status()
            return r.json()
        payload = _pull()
        return _within(self._parse(payload), start, end)

    @staticmethod
    def _parse(payload: dict) -> pd.DataFrame:
        payload = payload or {}
        if 'Time Series (Daily)' not in payload:
            msg = payload.get('Note') or payload.get('Information') or payload.get('Error Message') or 'no time series'
            raise SourceUnavailable(f'alpha_vantage: {msg}')
        ts = payload['Time Series (Daily)']
        recs = {}
        for date_str, fields in ts.items():
            recs[date_str] = {'open': fields.get('1. open'), 'high': fields.get('2. high'), 'low': fields.get('3. low'), 'close': fields.get('4. close'), 'adj_close': fields.get('5. adjusted close'), 'volume': fields.get('6. volume')}
        df = pd.DataFrame.from_dict(recs, orient='index')
        df.index = pd.to_datetime(df.index, errors='coerce')
        return _normalize_frame(df)

class StooqPrices:
    name = 'stooq'
    _URL = 'https://stooq.com/q/d/l/'

    def get_daily(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        sess = _session()
        symbol = self._stooq_symbol(ticker)
        params = {'s': symbol, 'd1': pd.Timestamp(start).strftime('%Y%m%d'), 'd2': pd.Timestamp(end).strftime('%Y%m%d'), 'i': 'd'}

        @_retry()
        def _pull() -> str:
            r = sess.get(self._URL, params=params, timeout=CFG.http_timeout)
            if r.status_code == 429:
                raise RateLimited('stooq 429')
            r.raise_for_status()
            return r.text
        text = _pull()
        return _within(self._parse(text), start, end)

    @staticmethod
    def _stooq_symbol(ticker: str) -> str:
        t = ticker.lower()
        return t if '.' in t else f'{t}.us'

    @staticmethod
    def _parse(text: str) -> pd.DataFrame:
        text = (text or '').strip()
        if not text or 'Date' not in text.splitlines()[0]:
            raise SourceUnavailable(f"stooq: {text[:80] or 'empty response'}")
        df = pd.read_csv(io.StringIO(text))
        if df.empty:
            raise SourceUnavailable('stooq returned no rows')
        df = df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        df['adj_close'] = np.nan
        df['adj_close'] = df['close']
        df.index = pd.to_datetime(df['date'], errors='coerce')
        return _normalize_frame(df)
_ADAPTERS = {YFinancePrices.name: YFinancePrices, NasdaqPrices.name: NasdaqPrices, TiingoPrices.name: TiingoPrices, AlphaVantagePrices.name: AlphaVantagePrices, StooqPrices.name: StooqPrices}

def _cache_path(ticker: str) -> Path:
    return _PRICE_CACHE_DIR / f'{ticker.upper()}.parquet'

def _read_cache(ticker: str) -> Optional[pd.DataFrame]:
    path = _cache_path(ticker)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    return _normalize_frame(df)

def _meta_path(ticker: str) -> Path:
    return _PRICE_CACHE_DIR / f'{ticker.upper()}.meta.json'

def _write_cache(ticker: str, df: pd.DataFrame, req_start: str | None=None, req_end: str | None=None) -> None:
    _PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(_cache_path(ticker))
        if req_start is not None and req_end is not None:
            import json
            _meta_path(ticker).write_text(json.dumps({'req_start': req_start, 'req_end': req_end}))
    except Exception:
        pass

def _cache_covers(ticker: str, start: str, end: str) -> bool:
    p = _meta_path(ticker)
    if not p.exists():
        return False
    try:
        import json
        m = json.loads(p.read_text())
        return m.get('req_start', '9999') <= start and m.get('req_end', '0000') >= end
    except Exception:
        return False

def get_prices(ticker: str, start: str, end: str, cfg=CFG) -> pd.DataFrame:
    if _cache_covers(ticker, start, end):
        cached = _read_cache(ticker)
        if cached is not None:
            return _within(cached, start, end)
    last_err: Optional[Exception] = None
    for src_name in cfg.price_sources:
        adapter_cls = _ADAPTERS.get(src_name)
        if adapter_cls is None:
            continue
        adapter = adapter_cls()
        try:
            df = adapter.get_daily(ticker, start, end)
        except Exception as exc:
            last_err = exc
            continue
        if df is not None and len(df) > 0:
            _write_cache(ticker, df, start, end)
            return _within(df, start, end)
    if last_err is not None:
        raise SourceUnavailable(f'All price sources failed for {ticker}: {last_err!r}')
    return _empty_frame()

def get_market(cfg=CFG) -> pd.DataFrame:
    return get_prices(cfg.market_index, cfg.start_date, cfg.end_date, cfg=cfg)

def daily_returns(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) == 0:
        return pd.Series(dtype='float64', name='return')
    df = _normalize_frame(df)
    px = df['adj_close'].copy()
    px = px.where(px.notna(), df['close'])
    ret = px.pct_change()
    ret = ret.dropna()
    ret.name = 'return'
    return ret
