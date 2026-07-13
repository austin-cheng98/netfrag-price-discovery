from __future__ import annotations
import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from config import CFG, KEYS, RAW
from contracts import EarningsEvent
_EXCHANGE_TZ = 'America/New_York'
_MKT_OPEN_LOCAL = (9, 30)
_MKT_CLOSE_LOCAL = (16, 0)

class RetriableHTTPError(Exception):
    pass

class SourceUnavailable(Exception):
    pass

def infer_session(announce_dt_utc: pd.Timestamp, has_time: bool) -> str:
    if not has_time:
        return 'unknown'
    ts = pd.Timestamp(announce_dt_utc)
    if ts.tzinfo is None:
        raise ValueError('announce_dt_utc must be tz-aware UTC')
    local = ts.tz_convert(_EXCHANGE_TZ)
    minutes = local.hour * 60 + local.minute
    open_m = _MKT_OPEN_LOCAL[0] * 60 + _MKT_OPEN_LOCAL[1]
    close_m = _MKT_CLOSE_LOCAL[0] * 60 + _MKT_CLOSE_LOCAL[1]
    if minutes < open_m:
        return 'bmo'
    if minutes >= close_m:
        return 'amc'
    return 'dmt'

def compute_surprise(actual: Optional[float], estimate: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    a = _to_float(actual)
    e = _to_float(estimate)
    if a is None or e is None:
        surprise = None
    else:
        surprise = a - e
    if surprise is None or e is None or e == 0:
        surprise_pct = None
    else:
        surprise_pct = 100.0 * surprise / abs(e)
    return (surprise, surprise_pct)

def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, str):
        s = x.strip()
        if s in ('', '--', 'N/A', 'n/a', 'NA', 'null', 'None'):
            return None
        s = s.replace('$', '').replace('%', '').replace(',', '')
        if s.startswith('(') and s.endswith(')'):
            s = '-' + s[1:-1]
        try:
            return float(s)
        except ValueError:
            return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v

def _fiscal_period(ts: pd.Timestamp) -> str:
    q = (ts.month - 1) // 3 + 1
    return f'{ts.year}Q{q}'

def _within(ts: pd.Timestamp, start: str, end: str) -> bool:
    lo = pd.Timestamp(start, tz='UTC')
    hi = pd.Timestamp(end, tz='UTC') + pd.Timedelta(days=1)
    return lo <= ts < hi

def _make_event(ticker: str, announce_utc: pd.Timestamp, has_time: bool, actual: Optional[float], estimate: Optional[float], source: str) -> EarningsEvent:
    surprise, surprise_pct = compute_surprise(actual, estimate)
    return EarningsEvent(ticker=ticker.upper(), announce_utc=announce_utc, session=infer_session(announce_utc, has_time), eps_actual=_to_float(actual), eps_estimate=_to_float(estimate), surprise=surprise, surprise_pct=surprise_pct, fiscal_period=_fiscal_period(announce_utc), source=source)

def _headers(extra: Optional[dict[str, str]]=None) -> dict[str, str]:
    h = {'User-Agent': CFG.user_agent, 'Accept': 'application/json'}
    if extra:
        h.update(extra)
    return h

@retry(retry=retry_if_exception_type(RetriableHTTPError), stop=stop_after_attempt(CFG.max_retries), wait=wait_exponential(multiplier=CFG.backoff_base, min=CFG.backoff_base, max=60), reraise=True)
def _get_json(url: str, params: dict[str, Any], headers: dict[str, str]) -> Any:
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=CFG.http_timeout)
    except requests.RequestException as exc:
        raise RetriableHTTPError(f'request error for {url}: {exc}') from exc
    if resp.status_code == 429 or 500 <= resp.status_code < 600:
        raise RetriableHTTPError(f'HTTP {resp.status_code} for {url}')
    if resp.status_code != 200:
        raise SourceUnavailable(f'HTTP {resp.status_code} for {url}')
    try:
        return resp.json()
    except ValueError as exc:
        raise SourceUnavailable(f'non-JSON body from {url}: {exc}') from exc

class FinnhubEarnings:
    name = 'finnhub'
    URL = 'https://finnhub.io/api/v1/stock/earnings'

    def __init__(self, token: Optional[str]=None):
        self.token = token if token is not None else KEYS.finnhub

    def get_events(self, ticker: str, start: str, end: str) -> list[EarningsEvent]:
        if not self.token:
            raise SourceUnavailable('finnhub: no API key configured')
        data = _get_json(self.URL, params={'symbol': ticker.upper(), 'token': self.token}, headers=_headers())
        rows = self.parse(ticker, data, start, end)
        if not rows:
            raise SourceUnavailable(f'finnhub: no events for {ticker}')
        return rows

    @staticmethod
    def parse(ticker: str, data: Any, start: str, end: str) -> list[EarningsEvent]:
        if isinstance(data, dict):
            data = data.get('earningsCalendar') or data.get('data') or []
        events: list[EarningsEvent] = []
        for row in data or []:
            period = row.get('period') or row.get('date')
            if not period:
                continue
            try:
                announce = pd.Timestamp(period, tz='UTC')
            except (ValueError, TypeError):
                continue
            if not _within(announce, start, end):
                continue
            ev = _make_event(ticker, announce, has_time=False, actual=row.get('actual'), estimate=row.get('estimate'), source='finnhub')
            if row.get('quarter') and row.get('year'):
                ev.fiscal_period = f"{row['year']}Q{row['quarter']}"
            events.append(ev)
        return events

class NasdaqEarnings:
    name = 'nasdaq'
    CAL_URL = 'https://api.nasdaq.com/api/calendar/earnings'

    def get_events(self, ticker: str, start: str, end: str) -> list[EarningsEvent]:
        raise SourceUnavailable('nasdaq: per-ticker history unsupported (calendar is by-date only)')

    def get_events_for_date(self, ticker: str, date: str) -> list[EarningsEvent]:
        data = _get_json(self.CAL_URL, params={'date': date}, headers=_headers({'Accept': 'application/json, text/plain, */*', 'Origin': 'https://www.nasdaq.com', 'Referer': 'https://www.nasdaq.com/'}))
        return self.parse(ticker, date, data)

    @staticmethod
    def parse(ticker: str, date: str, data: Any) -> list[EarningsEvent]:
        rows = ((data or {}).get('data') or {}).get('rows') or []
        tkr = ticker.upper()
        day = pd.Timestamp(date, tz='UTC')
        out: list[EarningsEvent] = []
        for row in rows:
            if (row.get('symbol') or '').upper() != tkr:
                continue
            session_hint = (row.get('time') or '').lower()
            has_time = bool(session_hint)
            actual = row.get('eps')
            estimate = row.get('epsForecast') or row.get('consensusForecast')
            ev = _make_event(tkr, day, has_time=False, actual=actual, estimate=estimate, source='nasdaq')
            if 'pre-market' in session_hint or 'before' in session_hint:
                ev.session = 'bmo'
            elif 'after-hours' in session_hint or 'after' in session_hint:
                ev.session = 'amc'
            out.append(ev)
        return out

class YFinanceEarnings:
    name = 'yfinance'

    def __init__(self, limit: int=60):
        self.limit = limit

    def get_events(self, ticker: str, start: str, end: str) -> list[EarningsEvent]:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise SourceUnavailable(f'yfinance not installed: {exc}') from exc
        try:
            df = yf.Ticker(ticker.upper()).get_earnings_dates(limit=self.limit)
        except Exception as exc:
            raise SourceUnavailable(f'yfinance: fetch failed for {ticker}: {exc}') from exc
        if df is None or len(df) == 0:
            raise SourceUnavailable(f'yfinance: empty frame for {ticker}')
        rows = self.parse(ticker, df, start, end)
        if not rows:
            raise SourceUnavailable(f'yfinance: no in-window events for {ticker}')
        return rows

    @staticmethod
    def parse(ticker: str, df: pd.DataFrame, start: str, end: str) -> list[EarningsEvent]:
        cols = {c.lower().strip(): c for c in df.columns}
        est_col = cols.get('eps estimate')
        act_col = cols.get('reported eps')
        spct_col = cols.get('surprise(%)') or cols.get('surprise (%)')
        events: list[EarningsEvent] = []
        for idx, row in df.iterrows():
            ts = pd.Timestamp(idx)
            if ts.tzinfo is None:
                ts = ts.tz_localize('America/New_York')
            announce = ts.tz_convert('UTC')
            if not _within(announce, start, end):
                continue
            actual = row[act_col] if act_col else None
            estimate = row[est_col] if est_col else None
            ev = _make_event(ticker, announce, has_time=True, actual=actual, estimate=estimate, source='yfinance')
            if spct_col is not None:
                feed_spct = _to_float(row[spct_col])
                if feed_spct is not None and ev.surprise_pct is None:
                    ev.surprise_pct = feed_spct
            events.append(ev)
        return events
_SOURCE_FACTORIES = {'finnhub': FinnhubEarnings, 'nasdaq': NasdaqEarnings, 'yfinance': YFinanceEarnings}

def _dedup(events: list[EarningsEvent]) -> list[EarningsEvent]:
    best: dict[tuple[str, str], EarningsEvent] = {}
    for ev in events:
        key = (ev.ticker.upper(), ev.announce_utc.date().isoformat())
        cur = best.get(key)
        if cur is None or _completeness(ev) > _completeness(cur):
            best[key] = ev
    return sorted(best.values(), key=lambda e: e.announce_utc)

def _completeness(ev: EarningsEvent) -> tuple[int, int, int]:
    return (int(ev.eps_actual is not None), int(ev.eps_estimate is not None), int(ev.session != 'unknown'))

def _cache_path(ticker: str):
    d = RAW / 'earnings'
    d.mkdir(parents=True, exist_ok=True)
    return d / f'{ticker.upper()}.json'
_HIST_START = '2015-01-01'

def _hist_end() -> str:
    return (datetime.now(timezone.utc) + timedelta(days=120)).date().isoformat()

def _write_cache(ticker: str, events: list[EarningsEvent], fetch_start: str, fetch_end: str) -> None:
    path = _cache_path(ticker)
    payload = {'ticker': ticker.upper(), 'fetched_utc': datetime.now(timezone.utc).isoformat(), 'fetch_start': fetch_start, 'fetch_end': fetch_end, 'events': [ev.to_row() for ev in events]}
    path.write_text(json.dumps(payload, indent=2, default=str))

def _read_cache_payload(ticker: str) -> Optional[dict[str, Any]]:
    path = _cache_path(ticker)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return None

def _read_cache(ticker: str) -> Optional[list[EarningsEvent]]:
    payload = _read_cache_payload(ticker)
    if payload is None:
        return None
    events = [_row_to_event(r) for r in payload.get('events', [])]
    return [e for e in events if e is not None]

def _row_to_event(row: dict[str, Any]) -> Optional[EarningsEvent]:
    try:
        announce = pd.Timestamp(row['announce_utc'])
        if announce.tzinfo is None:
            announce = announce.tz_localize('UTC')
        else:
            announce = announce.tz_convert('UTC')
        return EarningsEvent(ticker=row['ticker'], announce_utc=announce, session=row.get('session', 'unknown'), eps_actual=row.get('eps_actual'), eps_estimate=row.get('eps_estimate'), surprise=row.get('surprise'), surprise_pct=row.get('surprise_pct'), sue=row.get('sue'), fiscal_period=row.get('fiscal_period'), source=row.get('source', ''))
    except (KeyError, ValueError, TypeError):
        return None

def get_earnings(ticker: str, start: str, end: str, cfg=CFG, use_cache: bool=True, _factories: Optional[dict[str, Any]]=None) -> list[EarningsEvent]:
    hist_start, hist_end = (_HIST_START, _hist_end())
    if use_cache:
        payload = _read_cache_payload(ticker)
        if payload is not None:
            fs = payload.get('fetch_start')
            fe = payload.get('fetch_end')
            if fs is not None and fe is not None and (fs <= start) and (fe >= end):
                cached = [e for e in (_row_to_event(r) for r in payload.get('events', [])) if e is not None]
                in_window = [ev for ev in cached if _within(ev.announce_utc, start, end)]
                return _dedup(in_window)
    factories = _factories if _factories is not None else _SOURCE_FACTORIES
    errors: list[str] = []
    for name in cfg.earnings_sources:
        factory = factories.get(name)
        if factory is None:
            errors.append(f'{name}: not implemented')
            continue
        source = factory() if isinstance(factory, type) else factory
        try:
            events = source.get_events(ticker, hist_start, hist_end)
        except Exception as exc:
            errors.append(f'{name}: {exc}')
            continue
        if events:
            deduped = _dedup(events)
            _write_cache(ticker, deduped, hist_start, hist_end)
            return _dedup([ev for ev in deduped if _within(ev.announce_utc, start, end)])
        errors.append(f'{name}: empty')
    raise SourceUnavailable(f'all earnings sources failed for {ticker}: ' + ' | '.join(errors))

def get_all_earnings(tickers: list[str], start: str, end: str, cfg=CFG, use_cache: bool=True, _factories: Optional[dict[str, Any]]=None) -> list[EarningsEvent]:
    out: list[EarningsEvent] = []
    for t in tickers:
        try:
            out.extend(get_earnings(t, start, end, cfg=cfg, use_cache=use_cache, _factories=_factories))
        except SourceUnavailable:
            continue
    return out
