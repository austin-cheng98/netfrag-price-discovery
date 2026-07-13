from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from contracts import ControlVars, EarningsEvent, RedditItem
from config import CFG, Config, RAW
SECTOR_MAP: dict[str, str] = {'AAPL': 'Information Technology', 'MSFT': 'Information Technology', 'NVDA': 'Information Technology', 'AMD': 'Information Technology', 'INTC': 'Information Technology', 'MU': 'Information Technology', 'QCOM': 'Information Technology', 'AVGO': 'Information Technology', 'CRM': 'Information Technology', 'PLTR': 'Information Technology', 'SHOP': 'Information Technology', 'SMCI': 'Information Technology', 'META': 'Communication Services', 'GOOGL': 'Communication Services', 'NFLX': 'Communication Services', 'DIS': 'Communication Services', 'SNAP': 'Communication Services', 'AMC': 'Communication Services', 'TSLA': 'Consumer Discretionary', 'AMZN': 'Consumer Discretionary', 'GME': 'Consumer Discretionary', 'BABA': 'Consumer Discretionary', 'F': 'Consumer Discretionary', 'NIO': 'Consumer Discretionary', 'RIVN': 'Consumer Discretionary', 'LCID': 'Consumer Discretionary', 'DKNG': 'Consumer Discretionary', 'UBER': 'Consumer Discretionary', 'COIN': 'Financials', 'SOFI': 'Financials', 'PYPL': 'Financials', 'HOOD': 'Financials', 'MARA': 'Financials', 'RIOT': 'Financials', 'BA': 'Industrials'}

def sector_for(ticker: str) -> str:
    return SECTOR_MAP.get(ticker.upper(), 'Unknown')

def _to_utc_ts(x) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tzinfo is None or ts.tz is None:
        ts = ts.tz_localize('UTC')
    else:
        ts = ts.tz_convert('UTC')
    return ts

def _price_index_utc(price_index) -> pd.DatetimeIndex:
    if isinstance(price_index, pd.DataFrame):
        idx = price_index.index
    else:
        idx = price_index
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    else:
        idx = idx.tz_convert('UTC')
    return idx.sort_values()

def event_reddit_window(event: EarningsEvent, cfg: Config=CFG) -> dict[str, pd.Timestamp]:
    announce = _to_utc_ts(event.announce_utc)
    return {'comm_after': announce - pd.Timedelta(hours=cfg.reddit_pre_hours), 'comm_before': announce + pd.Timedelta(hours=cfg.reddit_post_hours), 'treat_after': announce, 'treat_before': announce + pd.Timedelta(hours=cfg.treat_window_hours)}

def event_trading_day(event: EarningsEvent, price_index) -> pd.Timestamp:
    idx = _price_index_utc(price_index)
    if len(idx) == 0:
        raise ValueError('price_index is empty; cannot map trading day')
    announce = _to_utc_ts(event.announce_utc)
    announce_date = announce.normalize()
    session = (event.session or '').strip().lower()
    same_day_sessions = {'bmo', 'dmt', 'during', 'before'}
    if session in same_day_sessions:
        target = announce_date
    else:
        target = announce_date + pd.Timedelta(days=1)
    idx_dates = idx.normalize()
    mask = np.asarray(idx_dates >= target)
    if not mask.any():
        raise ValueError(f'No trading day at/after {target.date()} in price index (last available {idx[-1].date()})')
    pos = int(np.argmax(mask))
    return idx[pos]

def _returns_from_close(close: pd.Series) -> pd.Series:
    return close.pct_change()

def _shares_outstanding(ticker: str) -> float | None:
    try:
        import yfinance as yf
    except Exception:
        return None
    try:
        tk = yf.Ticker(ticker)
        try:
            fi = tk.fast_info
            so = getattr(fi, 'shares_outstanding', None)
            if so is None and isinstance(fi, dict):
                so = fi.get('shares_outstanding') or fi.get('sharesOutstanding')
            if so and float(so) > 0:
                return float(so)
        except Exception:
            pass
        try:
            sf = tk.get_shares_full(start=None, end=None)
            if sf is not None and len(sf) > 0:
                val = float(pd.Series(sf).dropna().iloc[-1])
                if val > 0:
                    return val
        except Exception:
            pass
    except Exception:
        return None
    return None

def compute_controls(event: EarningsEvent, prices: pd.DataFrame, market: pd.DataFrame | None, items: list[RedditItem] | None, cfg: Config=CFG, *, allow_network: bool=True) -> ControlVars:
    ticker = event.ticker
    surprise_pct = event.surprise_pct
    abs_surprise = abs(event.surprise) if event.surprise is not None else None
    if prices is None or len(prices) == 0:
        return ControlVars(event_id=event.event_id, ticker=ticker, surprise_pct=surprise_pct, abs_surprise=abs_surprise, log_mktcap=np.nan, pre_vol=np.nan, log_volume=np.nan, prior_return=np.nan, sector=sector_for(ticker), n_comments=len(items) if items is not None else None, news_intensity=_news_intensity(event, items, cfg))
    px = prices.copy()
    px.index = _price_index_utc(px)
    px = px[~px.index.duplicated(keep='last')].sort_index()
    try:
        t0 = event_trading_day(event, px.index)
        t0_pos = int(px.index.get_loc(t0))
    except Exception:
        t0 = None
        t0_pos = None
    close = px['close'].astype(float) if 'close' in px.columns else None
    volume = px['volume'].astype(float) if 'volume' in px.columns else None
    pre_vol = np.nan
    log_volume = np.nan
    prior_return = np.nan
    log_mktcap = np.nan
    if t0_pos is not None and close is not None:
        rets = _returns_from_close(close)
        n = len(px)

        def _slice(lo_off: int, hi_off: int) -> slice:
            lo = max(0, t0_pos + lo_off)
            hi = min(n - 1, t0_pos + hi_off)
            return slice(lo, hi + 1)
        pv = rets.iloc[_slice(-30, -6)].dropna()
        if len(pv) >= 2:
            pre_vol = float(pv.std(ddof=1))
        pr = rets.iloc[_slice(-60, -2)].dropna()
        if len(pr) >= 1:
            prior_return = float(np.prod(1.0 + pr.values) - 1.0)
        if volume is not None:
            win = _slice(-5, 1)
            c = close.iloc[win].values
            v = volume.iloc[win].values
            dollar = c * v
            dollar = dollar[np.isfinite(dollar) & (dollar > 0)]
            if len(dollar) >= 1:
                log_volume = float(np.mean(np.log(dollar)))
        if allow_network:
            so = _shares_outstanding(ticker)
            if so is not None:
                px_t0 = float(close.iloc[t0_pos])
                if np.isfinite(px_t0) and px_t0 > 0:
                    log_mktcap = float(np.log(so * px_t0))
    return ControlVars(event_id=event.event_id, ticker=ticker, surprise_pct=surprise_pct, abs_surprise=abs_surprise, log_mktcap=log_mktcap, pre_vol=pre_vol, log_volume=log_volume, prior_return=prior_return, sector=sector_for(ticker), n_comments=len(items) if items is not None else None, news_intensity=_news_intensity(event, items, cfg))

def _news_intensity(event: EarningsEvent, items: list[RedditItem] | None, cfg: Config) -> float | None:
    if items is None:
        return None
    win = event_reddit_window(event, cfg)
    after = win['comm_after']
    before = win['comm_before']
    hours = (before - after) / pd.Timedelta(hours=1)
    if hours <= 0:
        return None
    n_sub = 0
    for it in items:
        if getattr(it, 'kind', None) != 'submission':
            continue
        ts = it.created_utc
        if ts is None:
            continue
        ts = _to_utc_ts(ts)
        if after <= ts <= before:
            n_sub += 1
    return float(n_sub) / float(hours)
__all__ = ['SECTOR_MAP', 'sector_for', 'event_reddit_window', 'event_trading_day', 'compute_controls']
