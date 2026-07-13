from __future__ import annotations
import hashlib
import re
import time
import warnings
from datetime import datetime, timezone
from typing import Optional
import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from config import CFG, RAW
from contracts import RedditItem

class RedditSourceError(RuntimeError):
    pass

class _RateLimited(Exception):
    pass

def _to_utc(dt: datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None or ts.tz is None:
        ts = ts.tz_localize('UTC')
    else:
        ts = ts.tz_convert('UTC')
    return ts

def _to_unix(dt: datetime | pd.Timestamp) -> int:
    return int(_to_utc(dt).timestamp())

def _from_unix(sec: float) -> pd.Timestamp:
    return pd.Timestamp(datetime.fromtimestamp(float(sec), tz=timezone.utc))

def _fmt_stamp(dt: datetime | pd.Timestamp) -> str:
    return _to_utc(dt).strftime('%Y%m%dT%H%M%SZ')
_NAME_ALIASES: dict[str, list[str]] = {'AAPL': ['apple'], 'MSFT': ['microsoft'], 'NVDA': ['nvidia'], 'AMD': [], 'TSLA': ['tesla'], 'AMZN': ['amazon'], 'META': [], 'GOOGL': ['google', 'alphabet'], 'NFLX': ['netflix'], 'INTC': ['intel'], 'PLTR': ['palantir'], 'AMC': [], 'GME': ['gamestop'], 'COIN': ['coinbase'], 'BABA': ['alibaba'], 'DIS': ['disney'], 'BA': ['boeing'], 'SOFI': [], 'MU': ['micron'], 'QCOM': ['qualcomm'], 'AVGO': ['broadcom'], 'CRM': ['salesforce'], 'PYPL': ['paypal'], 'SHOP': ['shopify'], 'SNAP': ['snapchat'], 'F': ['ford'], 'NIO': [], 'RIVN': ['rivian'], 'LCID': ['lucid'], 'MARA': ['marathon digital'], 'RIOT': ['riot platforms', 'riot blockchain'], 'SMCI': ['supermicro', 'super micro'], 'DKNG': ['draftkings'], 'HOOD': ['robinhood'], 'UBER': []}
_AMBIGUOUS: frozenset[str] = frozenset({'F', 'A', 'ON', 'MU', 'DIS', 'SO', 'IT', 'ALL', 'BA', 'SHOP', 'NIO', 'AMC', 'META', 'SNAP', 'COIN', 'HOOD', 'RIOT', 'DIS', 'CRM', 'UBER'})

def _build_aliases() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for t in CFG.tickers:
        pats = [t, f'${t}']
        pats += _NAME_ALIASES.get(t, [])
        out[t] = pats
    return out
TICKER_ALIASES: dict[str, list[str]] = _build_aliases()

def _compile_matchers(ticker: str) -> tuple[list[re.Pattern], list[re.Pattern]]:
    t = ticker.upper()
    ci: list[re.Pattern] = []
    cs: list[re.Pattern] = []
    ci.append(re.compile(f"(?<![\\w$]){re.escape('$' + t)}\\b", re.IGNORECASE))
    if t in _AMBIGUOUS:
        cs.append(re.compile(f'\\b{re.escape(t)}\\b'))
    else:
        ci.append(re.compile(f'\\b{re.escape(t)}\\b', re.IGNORECASE))
    for name in _NAME_ALIASES.get(t, []):
        ci.append(re.compile(f'\\b{re.escape(name)}\\b', re.IGNORECASE))
    return (ci, cs)
_MATCHER_CACHE: dict[str, tuple[list[re.Pattern], list[re.Pattern]]] = {}

def _matchers(ticker: str) -> tuple[list[re.Pattern], list[re.Pattern]]:
    key = ticker.upper()
    if key not in _MATCHER_CACHE:
        _MATCHER_CACHE[key] = _compile_matchers(key)
    return _MATCHER_CACHE[key]

def mentions_ticker(text: str, ticker: str) -> bool:
    if not text:
        return False
    ci, cs = _matchers(ticker)
    for pat in ci:
        if pat.search(text):
            return True
    for pat in cs:
        if pat.search(text):
            return True
    return False

def _retryer(cfg=CFG):
    return retry(retry=retry_if_exception_type((_RateLimited, requests.RequestException)), stop=stop_after_attempt(cfg.max_retries), wait=wait_exponential(multiplier=cfg.backoff_base, min=cfg.backoff_base, max=60), reraise=True)

def _get_json(url: str, params: dict, cfg=CFG) -> dict:

    @_retryer(cfg)
    def _do() -> dict:
        resp = requests.get(url, params=params, headers={'User-Agent': cfg.user_agent, 'Accept': 'application/json'}, timeout=cfg.http_timeout)
        if resp.status_code == 429:
            raise _RateLimited(f'429 from {url}')
        resp.raise_for_status()
        return resp.json()
    return _do()

class ArcticShiftSource:
    name = 'arctic_shift'
    BASE = 'https://arctic-shift.photon-reddit.com/api'

    def __init__(self, cfg=CFG):
        self.cfg = cfg

    def _query_param(self, kind: str) -> str | None:
        return 'body' if kind == 'comment' else 'selftext'

    def _endpoint(self, kind: str) -> str:
        if kind == 'comment':
            return f'{self.BASE}/comments/search'
        if kind == 'submission':
            return f'{self.BASE}/posts/search'
        raise ValueError(f'unknown kind {kind!r}')

    @staticmethod
    def _parse(row: dict, kind: str, subreddit: str) -> Optional[RedditItem]:
        rid = row.get('id')
        created = row.get('created_utc')
        if rid is None or created is None:
            return None
        if kind == 'submission':
            title = row.get('title') or ''
            selftext = row.get('selftext') or ''
            body = (title + '\n' + selftext).strip()
        else:
            body = row.get('body') or ''
        return RedditItem(id=str(rid), kind=kind, subreddit=row.get('subreddit') or subreddit, author=row.get('author') or '', created_utc=_from_unix(created), body=body, parent_id=row.get('parent_id'), link_id=row.get('link_id'), score=row.get('score'), source='arctic_shift')

    def search(self, subreddit: str, after: datetime, before: datetime, kind: str='comment', query: str | None=None, limit: int | None=None) -> list[RedditItem]:
        return _paged_search_descending(source=self, endpoint=self._endpoint(kind), subreddit=subreddit, after=after, before=before, kind=kind, query=query, limit=limit)

class PullPushSource:
    name = 'pullpush'
    BASE = 'https://api.pullpush.io/reddit/search'

    def __init__(self, cfg=CFG):
        self.cfg = cfg

    def _query_param(self, kind: str) -> str | None:
        return 'q'

    def _endpoint(self, kind: str) -> str:
        if kind == 'comment':
            return f'{self.BASE}/comment/'
        if kind == 'submission':
            return f'{self.BASE}/submission/'
        raise ValueError(f'unknown kind {kind!r}')

    @staticmethod
    def _parse(row: dict, kind: str, subreddit: str) -> Optional[RedditItem]:
        rid = row.get('id')
        created = row.get('created_utc')
        if rid is None or created is None:
            return None
        if kind == 'submission':
            title = row.get('title') or ''
            selftext = row.get('selftext') or ''
            body = (title + '\n' + selftext).strip()
        else:
            body = row.get('body') or ''
        return RedditItem(id=str(rid), kind=kind, subreddit=row.get('subreddit') or subreddit, author=row.get('author') or '', created_utc=_from_unix(created), body=body, parent_id=row.get('parent_id'), link_id=row.get('link_id'), score=row.get('score'), source='pullpush')

    def search(self, subreddit: str, after: datetime, before: datetime, kind: str='comment', query: str | None=None, limit: int | None=None) -> list[RedditItem]:
        return _paged_search(source=self, endpoint=self._endpoint(kind), subreddit=subreddit, after=after, before=before, kind=kind, query=query, limit=limit, page_param='size', sort_ascending=True)
_SOURCE_REGISTRY = {ArcticShiftSource.name: ArcticShiftSource, PullPushSource.name: PullPushSource}

def _paged_search(source, endpoint: str, subreddit: str, after: datetime, before: datetime, kind: str, query: str | None, limit: int | None, page_param: str='limit', sort_ascending: bool=True) -> list[RedditItem]:
    cfg = source.cfg
    after_u = _to_unix(after)
    before_u = _to_unix(before)
    if before_u <= after_u:
        return []
    base_page_size = 100
    _MAX_PAGE_SIZE = 10000
    page_size = base_page_size
    collected: dict[str, RedditItem] = {}
    cursor = after_u
    seen_max = after_u - 1
    while cursor < before_u:
        params = {'subreddit': subreddit, 'after': cursor, 'before': before_u, page_param: page_size}
        if page_param != 'limit':
            if sort_ascending:
                params['sort'] = 'asc'
        if query:
            qparam = source._query_param(kind) if hasattr(source, '_query_param') else 'q'
            if qparam:
                params[qparam] = query
        try:
            payload = _get_json(endpoint, params, cfg=cfg)
        except Exception as exc:
            raise RedditSourceError(f'{source.name} failed on {endpoint} ({subreddit}, {kind}): {exc}') from exc
        rows = payload.get('data') or []
        if not rows:
            break
        page_max = cursor
        page_min: int | None = None
        added_this_page = 0
        for row in rows:
            item = source._parse(row, kind, subreddit)
            if item is None:
                continue
            c = _to_unix(item.created_utc)
            if c > page_max:
                page_max = c
            if page_min is None or c < page_min:
                page_min = c
            if after_u <= c <= before_u and item.id not in collected:
                collected[item.id] = item
                added_this_page += 1
        if page_max <= cursor and added_this_page == 0:
            break
        seen_max = max(seen_max, page_max)
        if len(rows) < page_size:
            break
        if page_min is not None and page_min == page_max:
            if page_size < _MAX_PAGE_SIZE:
                page_size = min(page_size * 10, _MAX_PAGE_SIZE)
                time.sleep(cfg.rate_limit_sleep)
                continue
            warnings.warn(f'reddit pagination: >={_MAX_PAGE_SIZE} items share created_utc={page_max} in r/{subreddit} ({kind}); some rows in that second may be dropped.', RuntimeWarning, stacklevel=2)
        if page_min is not None and page_min < page_max:
            new_cursor = max(page_max, cursor + 1)
        else:
            new_cursor = max(page_max + 1, cursor + 1)
        if new_cursor <= cursor:
            break
        cursor = new_cursor
        page_size = base_page_size
        if limit is not None and len(collected) >= limit:
            break
        time.sleep(cfg.rate_limit_sleep)
    items = sorted(collected.values(), key=lambda it: _to_unix(it.created_utc))
    if limit is not None:
        items = items[:limit]
    return items

def _paged_search_descending(source, endpoint: str, subreddit: str, after: datetime, before: datetime, kind: str, query: str | None, limit: int | None, page_param: str='limit') -> list[RedditItem]:
    cfg = source.cfg
    after_u = _to_unix(after)
    before_u = _to_unix(before)
    if before_u <= after_u:
        return []
    base_page_size = 100
    _MAX_PAGE_SIZE = 10000
    page_size = base_page_size
    collected: dict[str, RedditItem] = {}
    cursor = before_u
    while cursor > after_u:
        params = {'subreddit': subreddit, 'after': after_u, 'before': cursor, page_param: page_size}
        if query:
            qparam = source._query_param(kind) if hasattr(source, '_query_param') else None
            if qparam:
                params[qparam] = query
        try:
            payload = _get_json(endpoint, params, cfg=cfg)
        except Exception as exc:
            raise RedditSourceError(f'{source.name} failed on {endpoint} ({subreddit}, {kind}): {exc}') from exc
        rows = payload.get('data') or []
        if not rows:
            break
        page_max: int | None = None
        page_min = cursor
        added_this_page = 0
        for row in rows:
            item = source._parse(row, kind, subreddit)
            if item is None:
                continue
            c = _to_unix(item.created_utc)
            page_max = c if page_max is None else max(page_max, c)
            if c < page_min:
                page_min = c
            if after_u <= c <= before_u and item.id not in collected:
                collected[item.id] = item
                added_this_page += 1
        if page_min >= cursor and added_this_page == 0:
            break
        if len(rows) < page_size:
            break
        if page_max is not None and page_min == page_max:
            if page_size < _MAX_PAGE_SIZE:
                page_size = min(page_size * 10, _MAX_PAGE_SIZE)
                time.sleep(cfg.rate_limit_sleep)
                continue
            warnings.warn(f'reddit pagination: >={_MAX_PAGE_SIZE} items share created_utc={page_min} in r/{subreddit} ({kind}); some rows in that second may be dropped.', RuntimeWarning, stacklevel=2)
        if page_max is not None and page_min < page_max:
            new_cursor = min(page_min, cursor - 1)
        else:
            new_cursor = min(page_min - 1, cursor - 1)
        if new_cursor >= cursor:
            break
        cursor = new_cursor
        page_size = base_page_size
        if limit is not None and len(collected) >= limit:
            break
        time.sleep(cfg.rate_limit_sleep)
    items = sorted(collected.values(), key=lambda it: _to_unix(it.created_utc))
    if limit is not None:
        items = items[-limit:]
    return items

def _cache_path(subreddit: str, kind: str, after: pd.Timestamp, before: pd.Timestamp, query: str | None=None, cap: int | None=None) -> 'pd.Path':
    d = RAW / 'reddit'
    d.mkdir(parents=True, exist_ok=True)
    name = f'{subreddit}_{kind}_{_fmt_stamp(after)}_{_fmt_stamp(before)}'
    if query:
        qtok = hashlib.sha1(query.encode('utf-8')).hexdigest()[:12]
        name += f'_q{qtok}'
    if cap:
        name += f'_c{int(cap)}'
    return d / f'{name}.parquet'

def _items_to_frame(items: list[RedditItem]) -> pd.DataFrame:
    rows = []
    for it in items:
        rows.append({'id': it.id, 'kind': it.kind, 'subreddit': it.subreddit, 'author': it.author, 'created_utc': it.created_utc, 'body': it.body, 'parent_id': it.parent_id, 'link_id': it.link_id, 'score': it.score, 'source': it.source})
    df = pd.DataFrame(rows, columns=['id', 'kind', 'subreddit', 'author', 'created_utc', 'body', 'parent_id', 'link_id', 'score', 'source'])
    if not df.empty:
        df['created_utc'] = pd.to_datetime(df['created_utc'], utc=True)
    return df

def _frame_to_items(df: pd.DataFrame) -> list[RedditItem]:
    items: list[RedditItem] = []
    for r in df.itertuples(index=False):
        items.append(RedditItem(id=str(r.id), kind=str(r.kind), subreddit=str(r.subreddit), author='' if r.author is None else str(r.author), created_utc=_to_utc(r.created_utc), body='' if r.body is None else str(r.body), parent_id=None if pd.isna(r.parent_id) else str(r.parent_id), link_id=None if pd.isna(r.link_id) else str(r.link_id), score=None if pd.isna(r.score) else int(r.score), source=str(r.source)))
    return items

def fetch_window(subreddit: str, after_utc: datetime, before_utc: datetime, kind: str, query: str | None=None, cfg=CFG, max_items: int | None=None) -> list[RedditItem]:
    after_ts = _to_utc(after_utc)
    before_ts = _to_utc(before_utc)
    path = _cache_path(subreddit, kind, after_ts, before_ts, query=query, cap=max_items)
    if path.exists():
        try:
            df = pd.read_parquet(path)
            return _frame_to_items(df)
        except Exception:
            pass
    last_err: Optional[Exception] = None
    for src_name in cfg.reddit_sources:
        cls = _SOURCE_REGISTRY.get(src_name)
        if cls is None:
            continue
        src = cls(cfg=cfg)
        try:
            items = src.search(subreddit=subreddit, after=after_ts, before=before_ts, kind=kind, query=query, limit=max_items)
        except RedditSourceError as exc:
            last_err = exc
            continue
        _items_to_frame(items).to_parquet(path, index=False)
        return items
    raise RedditSourceError(f'all reddit sources failed for {subreddit}/{kind} [{after_ts.isoformat()}..{before_ts.isoformat()}]: {last_err}')

def fetch_event_window(event, cfg=CFG) -> list[RedditItem]:
    announce = _to_utc(event.announce_utc)
    after = announce - pd.Timedelta(hours=cfg.reddit_pre_hours)
    before = announce + pd.Timedelta(hours=cfg.reddit_post_hours)
    query = event.ticker if getattr(cfg, 'reddit_query_by_ticker', True) else None
    cap = getattr(cfg, 'max_items_per_window', None)
    bulk_ok = set(getattr(cfg, 'bulk_fallback_subs', ()))
    bulk_cap = getattr(cfg, 'bulk_max_items', cap)
    by_id: dict[str, RedditItem] = {}
    for subreddit in cfg.subreddits:
        for kind in ('comment', 'submission'):
            items = None
            try:
                items = fetch_window(subreddit, after, before, kind, query=query, cfg=cfg, max_items=cap)
            except RedditSourceError:
                if query is not None and subreddit in bulk_ok:
                    try:
                        items = fetch_window(subreddit, after, before, kind, query=None, cfg=cfg, max_items=bulk_cap)
                    except RedditSourceError:
                        items = None
            if items is None:
                continue
            for it in items:
                if it.id in by_id:
                    continue
                if it.is_deleted():
                    continue
                if not mentions_ticker(it.body, event.ticker):
                    continue
                by_id[it.id] = it
    return sorted(by_id.values(), key=lambda it: _to_unix(it.created_utc))
__all__ = ['ArcticShiftSource', 'PullPushSource', 'RedditSourceError', 'TICKER_ALIASES', 'mentions_ticker', 'fetch_window', 'fetch_event_window']
