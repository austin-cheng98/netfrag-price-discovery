from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, '/Users/austincheng/Desktop/netfrag-price-discovery/src')
import analysis
from analysis import prep, baseline_ols, did_panel, iv_2sls, placebo, pretrends, robustness_table, run_all, DEFAULT_CONTROLS, FRAG_MEASURES
from config import CFG, RESULTS
SEED = 20260706
N = 600
SECTORS = ['Tech', 'Consumer', 'Energy', 'Financials']

def make_panel(n=N, seed=SEED):
    rng = np.random.default_rng(seed)
    n_comments = rng.integers(30, 5000, size=n).astype(float)
    n_comments_z = (n_comments - n_comments.mean()) / n_comments.std(ddof=0)
    frag_index = 0.75 * n_comments_z + rng.normal(0, 1.0, size=n)
    frag_index = (frag_index - frag_index.mean()) / frag_index.std(ddof=0)
    abs_surprise = np.abs(rng.normal(0, 3, size=n))
    log_mktcap = rng.normal(23, 1.5, size=n)
    pre_vol = np.abs(rng.normal(0.02, 0.01, size=n))
    log_volume = rng.normal(18, 1.0, size=n)
    prior_return = rng.normal(0, 0.05, size=n)
    sector = rng.choice(SECTORS, size=n)
    noise = rng.normal(0, 1.0, size=n)
    outcome = 2.0 + 1.5 * frag_index - 0.3 * abs_surprise + 0.8 * n_comments_z + noise
    prior_return_pre = rng.normal(0, 0.05, size=n)
    df = pd.DataFrame({'event_id': [f'TCK{i}:2024-01-01' for i in range(n)], 'ticker': [f'TCK{i % 35}' for i in range(n)], 'sector': sector, 'frag_index': frag_index, 'n_comments': n_comments, 'abs_surprise': abs_surprise, 'log_mktcap': log_mktcap, 'pre_vol': pre_vol, 'log_volume': log_volume, 'prior_return': prior_return_pre, 'halflife_days': outcome, 'modularity': 0.5 + 0.3 * frag_index + rng.normal(0, 0.2, size=n), 'spectral_gap': -(0.4 * frag_index) + rng.normal(0, 0.2, size=n), 'conductance': 0.3 + 0.2 * frag_index + rng.normal(0, 0.2, size=n), 'effective_resistance': 1.0 + 0.5 * frag_index + rng.normal(0, 0.3, size=n), 'community_entropy': 1.5 + 0.4 * frag_index + rng.normal(0, 0.2, size=n), 'embedding_variance': 0.2 + 0.1 * frag_index + rng.normal(0, 0.1, size=n), 'n_nodes': rng.integers(20, 500, size=n), 'n_communities': rng.integers(2, 20, size=n)})
    return df

def test_prep_builds_frag_index_and_confounder():
    df = make_panel()
    df2 = df.drop(columns=['frag_index'])
    out = prep(df2)
    assert 'frag_index' in out.columns
    assert 'n_comments_z' in out.columns
    assert out['frag_index'].notna().all()
    r = np.corrcoef(out['frag_index'], df['frag_index'])[0, 1]
    assert abs(r) > 0.8, f'reconstructed frag_index corr too low: {r}'
    assert 'frag_index' not in df2.columns

def test_prep_does_not_mutate_input():
    df = make_panel()
    before = df['halflife_days'].copy()
    _ = prep(df)
    pd.testing.assert_series_equal(df['halflife_days'], before)

def test_baseline_recovers_coef_when_confounder_controlled():
    df = prep(make_panel())
    res = baseline_ols(df, outcome='halflife_days')
    key = "Q('frag_index')"
    coef = res.params[key]
    se = res.bse[key]
    assert abs(coef - 1.5) <= 2 * se, f'coef={coef:.3f} se={se:.3f} not within 2SE of 1.5'
    assert "Q('n_comments_z')" in res.params.index
    assert res.pvalues[key] < 0.01

def test_omitting_confounder_biases_estimate():
    df = prep(make_panel())
    controlled = baseline_ols(df, outcome='halflife_days')
    no_conf_controls = [c for c in DEFAULT_CONTROLS if c != 'n_comments_z']
    biased = baseline_ols(df, outcome='halflife_days', controls=no_conf_controls)
    key = "Q('frag_index')"
    c_ctrl = controlled.params[key]
    c_bias = biased.params[key]
    assert c_bias > c_ctrl + 0.15, f'expected upward bias: controlled={c_ctrl:.3f} biased={c_bias:.3f}'
    assert abs(c_bias - 1.5) > abs(c_ctrl - 1.5)

def test_placebo_small_p_for_true_effect():
    df = prep(make_panel())
    res = placebo(df, outcome='halflife_days', B=150)
    assert res['B'] > 100
    assert np.isfinite(res['t_true'])
    assert res['p_perm'] < 0.05, f"expected small permutation p, got {res['p_perm']}"

def test_placebo_deterministic():
    df = prep(make_panel())
    a = placebo(df, outcome='halflife_days', B=60)
    b = placebo(df, outcome='halflife_days', B=60)
    assert a['p_perm'] == b['p_perm']
    assert a['t_true'] == b['t_true']

def test_placebo_null_effect_large_p():
    df = prep(make_panel())
    rng = np.random.default_rng(1)
    df = df.copy()
    df['frag_index'] = rng.normal(0, 1, size=len(df))
    res = placebo(df, outcome='halflife_days', B=150)
    assert res['p_perm'] > 0.05, f"noise treatment should give large p, got {res['p_perm']}"

def test_pretrends_null():
    df = prep(make_panel())
    res = pretrends(df)
    key = "Q('frag_index')"
    assert res.pvalues[key] > 0.05, f'pretrend should be null, p={res.pvalues[key]}'

def test_robustness_table_shape():
    df = prep(make_panel())
    tbl = robustness_table(df, outcome='halflife_days')
    assert 'frag_index' in tbl.index
    for m in FRAG_MEASURES:
        assert m in tbl.index
    for col in ('coef', 'se', 'p', 'n'):
        assert col in tbl.columns
    assert tbl.loc['frag_index', 'p'] < 0.05

def test_iv_2sls_illustrative_flag_and_first_stage():
    df = prep(make_panel())
    res = iv_2sls(df)
    assert getattr(res, 'illustrative_instrument') is True
    assert np.isfinite(getattr(res, 'first_stage_F'))
    assert 'frag_index' in res.params.index

def make_long(n_events=120, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_events):
        frag = rng.normal(0, 1)
        ni = rng.normal(0, 1)
        alpha_i = rng.normal(0, 0.4)
        for period in (0, 1):
            base = 6.0 + alpha_i + 0.2 * ni
            effect = 0.9 * frag if period == 1 else 0.0
            ineff = base + effect + rng.normal(0, 0.5)
            rows.append({'event_id': f'E{i}', 'period': period, 'inefficiency': abs(ineff), 'frag_index': frag, 'news_intensity': ni})
    return pd.DataFrame(rows)

def test_did_panel_recovers_interaction():
    long_df = make_long()
    res = did_panel(long_df)
    beta = res.params['frag_x_post']
    se = res.std_errors['frag_x_post']
    assert abs(beta - 0.9) <= 3 * se, f'DiD beta={beta:.3f} se={se:.3f} off from 0.9'
    assert res.pvalues['frag_x_post'] < 0.05

def test_did_panel_accepts_multiindex():
    long_df = make_long().set_index(['event_id', 'period'])
    res = did_panel(long_df)
    assert 'frag_x_post' in res.params.index

def test_run_all_writes_outputs():
    df = make_panel()
    long_df = make_long()
    out = run_all(df, long_df=long_df, cfg=CFG)
    assert (RESULTS / 'regression_tables.txt').exists()
    assert (RESULTS / 'coefficients.csv').exists()
    assert 'baseline_frag_coef' in out
    assert abs(out['baseline_frag_coef'] - 1.5) < 0.5
    assert out['baseline_frag_coef_no_confounder'] > out['baseline_frag_coef']
    csv = pd.read_csv(RESULTS / 'coefficients.csv')
    assert {'spec', 'term', 'coef', 'se', 'p', 'n'}.issubset(csv.columns)
