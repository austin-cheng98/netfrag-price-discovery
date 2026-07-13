from __future__ import annotations
import os
from datetime import datetime, timezone
import pandas as pd
import pytest
import reddit_data as rd
from config import CFG
from contracts import RedditItem

def _ts(y, m, d, h=0, mi=0, s=0) -> pd.Timestamp:
    return pd.Timestamp(datetime(y, m, d, h, mi, s, tzinfo=timezone.utc))

def _item(id_, created, body='AAPL to the moon', kind='comment', author='user', subreddit='stocks') -> RedditItem:
    return RedditItem(id=id_, kind=kind, subreddit=subreddit, author=author, created_utc=created, body=body, source='test')

def test_ambiguous_F_no_false_positive_on_word_if():
    assert rd.mentions_ticker('what if I do this', 'F') is False
    assert rd.mentions_ticker('the fund is fine', 'F') is False

def test_ambiguous_F_cashtag_and_upper_and_name():
    assert rd.mentions_ticker('bought $F today', 'F') is True
    assert rd.mentions_ticker('$f lowercase cashtag', 'F') is True
    assert rd.mentions_ticker('F is ripping', 'F') is True
    assert rd.mentions_ticker('Ford earnings beat', 'F') is True

def test_aapl_apple_alias_matches():
    assert rd.mentions_ticker('i love my apple products', 'AAPL') is True
    assert rd.mentions_ticker('AAPL calls printing', 'AAPL') is True
    assert rd.mentions_ticker('aapl lowercase ok too', 'AAPL') is True

def test_word_boundary_no_substring_false_positive():
    assert rd.mentions_ticker('pineapple juice', 'AAPL') is False
    assert rd.mentions_ticker('snap a quick photo', 'SNAP') is False
    assert rd.mentions_ticker('snapchat filters', 'SNAP') is True

def test_meta_polysemy_guarded():
    assert rd.mentions_ticker('in a meta sense', 'META') is False
    assert rd.mentions_ticker('$META puts', 'META') is True
    assert rd.mentions_ticker('META reported today', 'META') is True

def test_empty_text_is_false():
    assert rd.mentions_ticker('', 'AAPL') is False
    assert rd.mentions_ticker(None, 'AAPL') is False

def test_ticker_aliases_cover_universe():
    for t in CFG.tickers:
        assert t in rd.TICKER_ALIASES
        pats = rd.TICKER_ALIASES[t]
        assert t in pats and f'${t}' in pats

class _FakeSource:

    def __init__(self, name, roster, cfg=CFG):
        self.name = name
        self.cfg = cfg
        self._roster = roster

    def search(self, subreddit, after, before, kind='comment', query=None, limit=None):
        a = rd._to_unix(after)
        b = rd._to_unix(before)
        out = []
        for it in self._roster:
            if it.kind != kind or it.subreddit != subreddit:
                continue
            c = rd._to_unix(it.created_utc)
            if a <= c <= b:
                out.append(it)
        return out

def test_window_inclusive_endpoints(monkeypatch, tmp_path):
    announce = _ts(2024, 6, 3, 12)
    pre = announce - pd.Timedelta(hours=CFG.reddit_pre_hours)
    post = announce + pd.Timedelta(hours=CFG.reddit_post_hours)
    roster = [_item('before_edge', pre, kind='comment', subreddit='stocks'), _item('after_edge', post, kind='comment', subreddit='stocks'), _item('too_early', pre - pd.Timedelta(seconds=1), subreddit='stocks'), _item('too_late', post + pd.Timedelta(seconds=1), subreddit='stocks'), _item('inside', announce, subreddit='stocks')]

    def fake_fetch_window(subreddit, after_utc, before_utc, kind, query=None, cfg=CFG, max_items=None):
        src = _FakeSource('fake', roster, cfg=cfg)
        return src.search(subreddit, after_utc, before_utc, kind=kind, limit=max_items)
    monkeypatch.setattr(rd, 'fetch_window', fake_fetch_window)
    monkeypatch.setattr(CFG, 'subreddits', ['stocks'], raising=False)

    class Ev:
        ticker = 'AAPL'
        announce_utc = announce
    items = rd.fetch_event_window(Ev(), cfg=CFG)
    ids = {it.id for it in items}
    assert 'before_edge' in ids and 'after_edge' in ids and ('inside' in ids)
    assert 'too_early' not in ids and 'too_late' not in ids

def test_event_window_dedup_deleted_and_ticker_filter(monkeypatch):
    announce = _ts(2024, 6, 3, 12)
    roster = [_item('keep1', announce, body='AAPL smashing it', subreddit='stocks'), _item('keep1', announce, body='AAPL smashing it', subreddit='stocks'), _item('deleted', announce, body='[deleted]', subreddit='stocks'), _item('removed', announce, body='[removed]', subreddit='stocks'), _item('automod', announce, body='AAPL bot text', author='AutoModerator', subreddit='stocks'), _item('nomention', announce, body='just talking about bonds', subreddit='stocks'), _item('keep2', announce, body='i love apple', subreddit='investing'), _item('submission_keep', announce, body='AAPL earnings thread', kind='submission', subreddit='stocks')]

    def fake_fetch_window(subreddit, after_utc, before_utc, kind, query=None, cfg=CFG, max_items=None):
        src = _FakeSource('fake', roster, cfg=cfg)
        return src.search(subreddit, after_utc, before_utc, kind=kind, limit=max_items)
    monkeypatch.setattr(rd, 'fetch_window', fake_fetch_window)
    monkeypatch.setattr(CFG, 'subreddits', ['stocks', 'investing'], raising=False)

    class Ev:
        ticker = 'AAPL'
        announce_utc = announce
    items = rd.fetch_event_window(Ev(), cfg=CFG)
    ids = {it.id for it in items}
    assert ids == {'keep1', 'keep2', 'submission_keep'}
    assert sum((1 for it in items if it.id == 'keep1')) == 1
    assert isinstance(items, list)

def _make_rows(start_unix, n, id_prefix, step=1):
    return [{'id': f'{id_prefix}{i}', 'created_utc': start_unix + i * step, 'author': 'u', 'body': 'AAPL text', 'subreddit': 'stocks', 'parent_id': None, 'link_id': None, 'score': 1} for i in range(n)]

def test_pagination_advances_cursor_and_dedups(monkeypatch):
    after = _ts(2024, 6, 1, 0)
    before = _ts(2024, 6, 5, 0)
    after_u = rd._to_unix(after)
    page1 = _make_rows(after_u, 100, 'a', step=1)
    p2_start = after_u + 99
    page2 = [dict(page1[-1])] + _make_rows(p2_start + 1, 99, 'b', step=1)
    page3 = _make_rows(p2_start + 100, 10, 'c', step=1)
    calls = {'n': 0}

    def fake_get_json(url, params, cfg=CFG):
        calls['n'] += 1
        cur = params['after']
        if calls['n'] == 1:
            return {'data': page1}
        elif calls['n'] == 2:
            return {'data': page2}
        else:
            return {'data': page3}
    monkeypatch.setattr(rd, '_get_json', fake_get_json)
    monkeypatch.setattr(CFG, 'rate_limit_sleep', 0, raising=False)
    src = rd.PullPushSource(cfg=CFG)
    items = src.search('stocks', after, before, kind='comment')
    ids = [it.id for it in items]
    assert len(ids) == len(set(ids))
    assert len(ids) == 100 + 99 + 10
    ts = [rd._to_unix(it.created_utc) for it in items]
    assert ts == sorted(ts)
    assert calls['n'] == 3

def test_pagination_descending_drains_and_dedups(monkeypatch):
    after = _ts(2024, 6, 1, 0)
    before = _ts(2024, 6, 5, 0)
    before_u = rd._to_unix(before)
    page1 = _make_rows(before_u - 100, 100, 'a', step=1)
    page2 = [dict(page1[0])] + _make_rows(before_u - 199, 99, 'b', step=1)
    page3 = _make_rows(before_u - 209, 10, 'c', step=1)
    calls = {'n': 0}

    def fake_get_json(url, params, cfg=CFG):
        calls['n'] += 1
        return {'data': [page1, page2, page3][min(calls['n'] - 1, 2)]}
    monkeypatch.setattr(rd, '_get_json', fake_get_json)
    monkeypatch.setattr(CFG, 'rate_limit_sleep', 0, raising=False)
    src = rd.ArcticShiftSource(cfg=CFG)
    items = src.search('stocks', after, before, kind='comment')
    ids = [it.id for it in items]
    assert len(ids) == len(set(ids))
    assert len(ids) == 100 + 99 + 10
    ts = [rd._to_unix(it.created_utc) for it in items]
    assert ts == sorted(ts)
    assert calls['n'] == 3

def test_pagination_stops_on_empty_first_page(monkeypatch):
    after = _ts(2024, 6, 1)
    before = _ts(2024, 6, 2)

    def fake_get_json(url, params, cfg=CFG):
        return {'data': []}
    monkeypatch.setattr(rd, '_get_json', fake_get_json)
    src = rd.ArcticShiftSource(cfg=CFG)
    assert src.search('stocks', after, before, kind='comment') == []

def test_pagination_respects_limit(monkeypatch):
    after = _ts(2024, 6, 1, 0)
    before = _ts(2024, 6, 5, 0)
    after_u = rd._to_unix(after)
    page1 = _make_rows(after_u, 100, 'a', step=1)
    monkeypatch.setattr(rd, '_get_json', lambda url, params, cfg=CFG: {'data': page1})
    monkeypatch.setattr(CFG, 'rate_limit_sleep', 0, raising=False)
    src = rd.ArcticShiftSource(cfg=CFG)
    items = src.search('stocks', after, before, kind='comment', limit=25)
    assert len(items) == 25

def test_pagination_does_not_drop_same_second_burst(monkeypatch):
    after = _ts(2024, 6, 1, 0)
    before = _ts(2024, 6, 5, 0)
    after_u = rd._to_unix(after)
    burst_sec = after_u + 10
    burst = _make_rows(burst_sec, 150, 'burst', step=0)
    for i, r in enumerate(burst):
        r['id'] = f'burst{i}'
    tail = _make_rows(burst_sec + 5, 3, 'tail', step=1)
    universe = burst + tail

    def fake_get_json(url, params, cfg=CFG):
        cur = params['after']
        cap = params.get('limit') or params.get('size') or 100
        rows = [r for r in universe if r['created_utc'] >= cur]
        rows = sorted(rows, key=lambda r: r['created_utc'])
        return {'data': rows[:cap]}
    monkeypatch.setattr(rd, '_get_json', fake_get_json)
    monkeypatch.setattr(CFG, 'rate_limit_sleep', 0, raising=False)
    src = rd.ArcticShiftSource(cfg=CFG)
    items = src.search('stocks', after, before, kind='comment')
    ids = {it.id for it in items}
    assert len(ids) == 153
    assert all((f'burst{i}' in ids for i in range(150)))

def test_before_le_after_returns_empty():
    after = _ts(2024, 6, 2)
    before = _ts(2024, 6, 1)
    src = rd.ArcticShiftSource(cfg=CFG)
    assert src.search('stocks', after, before, kind='comment') == []

def test_parse_submission_concats_title_selftext():
    row = {'id': 'x1', 'created_utc': 1700000000, 'title': 'Earnings', 'selftext': 'AAPL beat', 'author': 'u', 'subreddit': 'stocks', 'num_comments': 3}
    it = rd.ArcticShiftSource._parse(row, 'submission', 'stocks')
    assert it.kind == 'submission'
    assert 'Earnings' in it.body and 'AAPL beat' in it.body
    assert it.created_utc.tzinfo is not None
    assert str(it.created_utc.tz) == 'UTC'

def test_parse_comment_and_missing_fields():
    good = {'id': 'c1', 'created_utc': 1700000000, 'body': 'hi', 'author': 'u'}
    it = rd.ArcticShiftSource._parse(good, 'comment', 'stocks')
    assert it.body == 'hi' and it.subreddit == 'stocks'
    assert rd.ArcticShiftSource._parse({'created_utc': 1}, 'comment', 'stocks') is None
    assert rd.ArcticShiftSource._parse({'id': 'z'}, 'comment', 'stocks') is None

def test_fetch_window_fallback_then_cache(monkeypatch, tmp_path):
    after = _ts(2024, 6, 1)
    before = _ts(2024, 6, 2)

    def fake_cache_path(subreddit, kind, a, b, query=None, cap=None):
        return tmp_path / f'{subreddit}_{kind}.parquet'
    monkeypatch.setattr(rd, '_cache_path', fake_cache_path)
    made = _item('z1', _ts(2024, 6, 1, 6), body='AAPL', subreddit='stocks')

    class FailFirst:
        name = 'arctic_shift'

        def __init__(self, cfg=CFG):
            self.cfg = cfg

        def search(self, **kw):
            raise rd.RedditSourceError('down')

    class OkSecond:
        name = 'pullpush'

        def __init__(self, cfg=CFG):
            self.cfg = cfg

        def search(self, subreddit, after, before, kind='comment', query=None, limit=None):
            return [made]
    monkeypatch.setattr(rd, '_SOURCE_REGISTRY', {'arctic_shift': FailFirst, 'pullpush': OkSecond})
    monkeypatch.setattr(CFG, 'reddit_sources', ['arctic_shift', 'pullpush'], raising=False)
    items = rd.fetch_window('stocks', after, before, 'comment', cfg=CFG)
    assert [it.id for it in items] == ['z1']
    assert (tmp_path / 'stocks_comment.parquet').exists()
    monkeypatch.setattr(rd, '_SOURCE_REGISTRY', {})
    cached = rd.fetch_window('stocks', after, before, 'comment', cfg=CFG)
    assert [it.id for it in cached] == ['z1']
    assert cached[0].created_utc.tzinfo is not None

def test_fetch_window_all_sources_fail_raises(monkeypatch, tmp_path):
    after = _ts(2024, 6, 1)
    before = _ts(2024, 6, 2)
    monkeypatch.setattr(rd, '_cache_path', lambda s, k, a, b, query=None, cap=None: tmp_path / 'nope.parquet')

    class Fail:
        name = 'arctic_shift'

        def __init__(self, cfg=CFG):
            self.cfg = cfg

        def search(self, **kw):
            raise rd.RedditSourceError('boom')
    monkeypatch.setattr(rd, '_SOURCE_REGISTRY', {'arctic_shift': Fail})
    monkeypatch.setattr(CFG, 'reddit_sources', ['arctic_shift'], raising=False)
    with pytest.raises(rd.RedditSourceError):
        rd.fetch_window('stocks', after, before, 'comment', cfg=CFG)

def test_get_json_retries_on_429_then_succeeds(monkeypatch):
    calls = {'n': 0}

    class Resp:

        def __init__(self, code, payload=None):
            self.status_code = code
            self._payload = payload or {}

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400 and self.status_code != 429:
                raise rd.requests.HTTPError(str(self.status_code))

    def fake_get(url, params=None, headers=None, timeout=None):
        calls['n'] += 1
        if calls['n'] == 1:
            return Resp(429)
        return Resp(200, {'data': [{'id': 'ok', 'created_utc': 1}]})
    monkeypatch.setattr(rd.requests, 'get', fake_get)
    monkeypatch.setattr(CFG, 'backoff_base', 0.001, raising=False)
    monkeypatch.setattr(CFG, 'max_retries', 3, raising=False)
    out = rd._get_json('http://x', {'a': 1}, cfg=CFG)
    assert out == {'data': [{'id': 'ok', 'created_utc': 1}]}
    assert calls['n'] == 2

@pytest.mark.skipif(os.getenv('NETFRAG_LIVE') != '1', reason='live smoke disabled (set NETFRAG_LIVE=1 to enable)')
def test_live_arctic_shift_smoke():
    after = _ts(2024, 1, 10, 12)
    before = _ts(2024, 1, 10, 13)
    src = rd.ArcticShiftSource(cfg=CFG)
    items = src.search('stocks', after, before, kind='comment', limit=20)
    assert isinstance(items, list)
    for it in items[:5]:
        assert isinstance(it, RedditItem)
        assert it.created_utc.tzinfo is not None
        assert after <= it.created_utc <= before
