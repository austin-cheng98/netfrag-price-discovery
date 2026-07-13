from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / '.env')
except Exception:
    pass
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
RAW = DATA / 'raw'
INTERIM = DATA / 'interim'
PROCESSED = DATA / 'processed'
RESULTS = ROOT / 'results'
for _d in (RAW, INTERIM, PROCESSED, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class Keys:
    finnhub: str | None = os.getenv('FINNHUB_API_KEY')
    tiingo: str | None = os.getenv('TIINGO_API_KEY')
    alpha_vantage: str | None = os.getenv('ALPHAVANTAGE_API_KEY')
    polygon: str | None = os.getenv('POLYGON_API_KEY')
    fmp: str | None = os.getenv('FMP_API_KEY')
    reddit_client_id: str | None = os.getenv('REDDIT_CLIENT_ID')
    reddit_client_secret: str | None = os.getenv('REDDIT_CLIENT_SECRET')
    reddit_user_agent: str = os.getenv('REDDIT_USER_AGENT', 'netfrag-research/0.1')
KEYS = Keys()

@dataclass
class Config:
    tickers: list[str] = field(default_factory=lambda: ['AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA', 'AMZN', 'META', 'GOOGL', 'NFLX', 'INTC', 'PLTR', 'AMC', 'GME', 'COIN', 'BABA', 'DIS', 'BA', 'SOFI', 'MU', 'QCOM', 'AVGO', 'CRM', 'PYPL', 'SHOP', 'SNAP', 'F', 'NIO', 'RIVN', 'LCID', 'MARA', 'RIOT', 'SMCI', 'DKNG', 'HOOD', 'UBER'])
    subreddits: list[str] = field(default_factory=lambda: ['stocks', 'investing'])
    start_date: str = '2022-07-01'
    end_date: str = '2024-12-31'
    reddit_pre_hours: int = 24
    reddit_post_hours: int = 48
    treat_window_hours: int = 48
    estimation_days: int = 120
    estimation_gap: int = 5
    price_lookback_days: int = 320
    car_start: int = 2
    car_end: int = 30
    halflife_max_day: int = 20
    edge_mode: str = 'thread'
    semantic_sim_threshold: float = 0.55
    min_component_nodes: int = 5
    min_event_comments: int = 30
    embed_backend: str = 'ollama'
    embed_model: str = 'all-minilm'
    embed_dim: int = 384
    price_sources: list[str] = field(default_factory=lambda: ['yfinance', 'nasdaq', 'tiingo', 'alpha_vantage', 'stooq'])
    earnings_sources: list[str] = field(default_factory=lambda: ['finnhub', 'nasdaq', 'yfinance', 'fmp'])
    reddit_sources: list[str] = field(default_factory=lambda: ['pullpush', 'arctic_shift'])
    reddit_query_by_ticker: bool = True
    max_items_per_window: int = 1500
    bulk_fallback_subs: list[str] = field(default_factory=lambda: ['stocks', 'investing'])
    bulk_max_items: int = 5000
    http_timeout: int = 30
    max_retries: int = 5
    backoff_base: float = 2.0
    rate_limit_sleep: float = 1.1
    user_agent: str = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36'
    cache_expire_hours: int = 24 * 30
    winsorize_pct: float = 0.01
    cluster_by: str = 'ticker'
    market_index: str = 'SPY'
    seed: int = 20260706
CFG = Config()
