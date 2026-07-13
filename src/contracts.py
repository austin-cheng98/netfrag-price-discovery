from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Iterable, Optional, Protocol, runtime_checkable
import numpy as np
import pandas as pd

@dataclass
class EarningsEvent:
    ticker: str
    announce_utc: pd.Timestamp
    session: str
    eps_actual: Optional[float] = None
    eps_estimate: Optional[float] = None
    surprise: Optional[float] = None
    surprise_pct: Optional[float] = None
    sue: Optional[float] = None
    fiscal_period: Optional[str] = None
    source: str = ''

    @property
    def event_id(self) -> str:
        return f'{self.ticker}:{self.announce_utc.date().isoformat()}'

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        d['event_id'] = self.event_id
        d['announce_utc'] = self.announce_utc.isoformat()
        return d

@dataclass
class RedditItem:
    id: str
    kind: str
    subreddit: str
    author: str
    created_utc: pd.Timestamp
    body: str
    parent_id: Optional[str] = None
    link_id: Optional[str] = None
    score: Optional[int] = None
    source: str = ''

    def is_deleted(self) -> bool:
        b = (self.body or '').strip().lower()
        return b in {'', '[deleted]', '[removed]'} or self.author in {'[deleted]', 'AutoModerator'}

@dataclass
class GraphSnapshot:
    event_id: str
    n_nodes: int
    n_edges: int
    directed: bool
    community_labels: dict[str, int]
    n_communities: int
    graph: Any = field(default=None, repr=False)
    embeddings: Optional[np.ndarray] = field(default=None, repr=False)

@dataclass
class FragmentationScore:
    event_id: str
    modularity: Optional[float] = None
    spectral_gap: Optional[float] = None
    conductance: Optional[float] = None
    effective_resistance: Optional[float] = None
    community_entropy: Optional[float] = None
    embedding_variance: Optional[float] = None
    n_communities: Optional[int] = None
    n_nodes: Optional[int] = None
    n_comments: Optional[int] = None
    frag_index: Optional[float] = None

    def to_row(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class OutcomeMeasures:
    event_id: str
    ticker: str
    car_event: Optional[float] = None
    pead: Optional[float] = None
    pead_abs: Optional[float] = None
    halflife_days: Optional[float] = None
    adjustment_speed_k: Optional[float] = None
    variance_ratio: Optional[float] = None
    vr_stat: Optional[float] = None
    post_vol: Optional[float] = None
    car_r2: Optional[float] = None
    source: str = ''

    def to_row(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class ControlVars:
    event_id: str
    ticker: str
    surprise_pct: Optional[float] = None
    abs_surprise: Optional[float] = None
    log_mktcap: Optional[float] = None
    pre_vol: Optional[float] = None
    log_volume: Optional[float] = None
    prior_return: Optional[float] = None
    sector: Optional[str] = None
    n_comments: Optional[int] = None
    news_intensity: Optional[float] = None

    def to_row(self) -> dict[str, Any]:
        return asdict(self)

@runtime_checkable
class PriceSource(Protocol):
    name: str

    def get_daily(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        ...

@runtime_checkable
class EarningsSource(Protocol):
    name: str

    def get_events(self, ticker: str, start: str, end: str) -> list[EarningsEvent]:
        ...

@runtime_checkable
class RedditSource(Protocol):
    name: str

    def search(self, subreddit: str, after: datetime, before: datetime, kind: str='comment', query: str | None=None, limit: int | None=None) -> list[RedditItem]:
        ...

@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        ...
PANEL_COLUMNS = ['event_id', 'ticker', 'sector', 'announce_utc', 'frag_index', 'modularity', 'spectral_gap', 'conductance', 'effective_resistance', 'community_entropy', 'embedding_variance', 'n_communities', 'n_nodes', 'n_comments', 'pead', 'pead_abs', 'halflife_days', 'adjustment_speed_k', 'variance_ratio', 'post_vol', 'car_event', 'surprise_pct', 'abs_surprise', 'log_mktcap', 'pre_vol', 'log_volume', 'prior_return', 'news_intensity']

def winsorize(s: pd.Series, p: float=0.01) -> pd.Series:
    lo, hi = (s.quantile(p), s.quantile(1 - p))
    return s.clip(lo, hi)

def zscore(s: pd.Series) -> pd.Series:
    mu, sd = (s.mean(), s.std(ddof=0))
    return (s - mu) / sd if sd and (not np.isnan(sd)) else s * 0.0
