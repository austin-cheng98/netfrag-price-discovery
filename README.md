# Price Discovery in Fragmented Communication Networks

The pipeline connects scheduled earnings announcements to the surrounding Reddit
discussion, reconstructs the author reply graph for each event, measures its
fragmentation (modularity, spectral gap, conductance, effective resistance,
community entropy, interpretive dispersion), and relates a standardized
fragmentation index to price-discovery outcomes (post-earnings drift, adjustment
half-life, variance ratio) with sector fixed effects, an event-window
difference-in-differences design, a permutation placebo, and a pre-trend test.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # optional API keys; the pipeline runs keyless
```
Message embeddings use a local [Ollama](https://ollama.com) server
(`ollama pull all-minilm`). Prices/earnings use yfinance; historical Reddit uses
the keyless Arctic Shift and pullpush mirrors.

## Reproduce

```bash
# validate the econometrics offline (recovers a known injected effect)
python3 run_pipeline.py --synthetic

# build the event panel (r/stocks + r/investing), then analyze
python3 scripts/run_full.py --tickers NVDA,AMD,TSLA,AAPL
python3 scripts/final_analysis.py
```

## Layout

- `src/` — pipeline modules (config, contracts, data sources, embeddings, graph,
  fragmentation, outcomes, controls, panel assembly, econometrics).
- `scripts/` — collection, assembly, analysis, and figure generation.
- `tests/` — unit tests, including an offline synthetic effect-recovery test.

## Data and results

Data pulls, the assembled panel, and figures are written under `data/` and
`results/` (git-ignored). Results are reported honestly as suggestive: the
fragmentation effect is directionally consistent and significant for the
adjustment half-life and a permutation placebo, but weak and mixed across
measures.
