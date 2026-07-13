from __future__ import annotations
import warnings
from typing import Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from contracts import winsorize, zscore
from config import CFG, RESULTS
FRAG_MEASURES: dict[str, float] = {'modularity': +1.0, 'spectral_gap': -1.0, 'conductance': +1.0, 'effective_resistance': +1.0, 'community_entropy': +1.0, 'embedding_variance': +1.0}
CONTINUOUS_VARS = ['modularity', 'spectral_gap', 'conductance', 'effective_resistance', 'community_entropy', 'embedding_variance', 'pead', 'pead_abs', 'halflife_days', 'adjustment_speed_k', 'variance_ratio', 'post_vol', 'car_event', 'surprise_pct', 'abs_surprise', 'log_mktcap', 'pre_vol', 'log_volume', 'prior_return', 'news_intensity']
DEFAULT_CONTROLS = ['n_comments_z', 'abs_surprise', 'log_mktcap', 'pre_vol', 'log_volume', 'prior_return']

def prep(panel: pd.DataFrame, cfg=CFG) -> pd.DataFrame:
    df = panel.copy()
    p = float(cfg.winsorize_pct)
    for col in CONTINUOUS_VARS:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            s = df[col]
            if s.notna().sum() >= 3 and s.nunique(dropna=True) > 1:
                df[col] = winsorize(s, p)
    if 'n_comments' in df.columns:
        df['n_comments_z'] = zscore(df['n_comments'].astype(float))
    elif 'n_comments_z' not in df.columns:
        df['n_comments_z'] = 0.0
    need_index = 'frag_index' not in df.columns or df.get('frag_index', pd.Series(dtype=float)).isna().all()
    if need_index:
        components = []
        for col, sign in FRAG_MEASURES.items():
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                z = zscore(df[col].astype(float))
                components.append(sign * z)
        if components:
            comp = pd.concat(components, axis=1)
            df['frag_index'] = comp.mean(axis=1, skipna=True)
        else:
            raise ValueError('prep: no fragmentation measures present to build frag_index and no precomputed frag_index column.')
    return df

def _present_controls(panel: pd.DataFrame, controls) -> list[str]:
    out = []
    for c in controls:
        if c in panel.columns and panel[c].notna().sum() > 1 and (panel[c].nunique(dropna=True) > 1):
            out.append(c)
    return out

def _build_ols_frame(panel, outcome, treat, controls):
    cols = [outcome, treat] + list(controls)
    has_sector = 'sector' in panel.columns and panel['sector'].nunique(dropna=True) > 1
    if has_sector:
        cols = cols + ['sector']
    df = panel[[c for c in cols if c in panel.columns]].copy()
    df = df.dropna()
    return (df, has_sector)

def baseline_ols(panel: pd.DataFrame, outcome: str='halflife_days', treat: str='frag_index', controls=DEFAULT_CONTROLS, cfg=CFG):
    controls = _present_controls(panel, controls)
    df, has_sector = _build_ols_frame(panel, outcome, treat, controls)
    if len(df) < len(controls) + 3:
        raise ValueError(f'baseline_ols: too few complete rows ({len(df)}) for outcome={outcome!r}.')
    rhs = [treat] + controls
    formula = f"Q('{outcome}') ~ " + ' + '.join((f"Q('{c}')" for c in rhs))
    if has_sector:
        formula += ' + C(sector)'
    model = smf.ols(formula, data=df)
    res = model.fit(cov_type='HC1')
    return res

def did_panel(long_df: pd.DataFrame, cfg=CFG):
    from linearmodels.panel import PanelOLS
    df = long_df.copy()
    if not isinstance(df.index, pd.MultiIndex):
        if {'event_id', 'period'}.issubset(df.columns):
            df = df.set_index(['event_id', 'period'])
        else:
            raise ValueError('did_panel: long_df needs a (event_id, period) MultiIndex or those columns.')
    dep = 'inefficiency' if 'inefficiency' in df.columns else None
    if dep is None:
        raise ValueError("did_panel: expected a dependent column named 'inefficiency'.")
    if 'frag_index' not in df.columns:
        raise ValueError("did_panel: 'frag_index' column required to form the interaction.")
    period_vals = df.index.get_level_values(1).astype(float)
    df = df.assign(post=period_vals)
    df['frag_x_post'] = df['frag_index'].astype(float) * df['post']
    candidate_ctrls = ['news_intensity', 'log_volume', 'pre_vol', 'n_comments_z']
    ctrls = [c for c in candidate_ctrls if c in df.columns and df[c].notna().sum() > 1]
    exog_cols = ['frag_x_post'] + ctrls
    model_df = df[[dep] + exog_cols].dropna()
    y = model_df[dep]
    X = model_df[exog_cols]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=False)
        res = model.fit(cov_type='clustered', cluster_entity=True)
    return res

def iv_2sls(panel: pd.DataFrame, instrument: str='Z_disruption', cfg=CFG):
    from linearmodels.iv import IV2SLS
    df = panel.copy()
    dep = None
    for cand in ('inefficiency', 'pead_abs', 'halflife_days'):
        if cand in df.columns and df[cand].notna().sum() > 3:
            dep = cand
            break
    if dep is None:
        raise ValueError('iv_2sls: no usable dependent variable (inefficiency/pead_abs/halflife_days).')
    if 'frag_index' not in df.columns:
        raise ValueError("iv_2sls: 'frag_index' endogenous regressor required.")
    illustrative = instrument not in df.columns
    if illustrative:
        rng = np.random.default_rng(cfg.seed)
        base = pd.Series(0.0, index=df.index)
        n = 0
        for col in ('n_nodes', 'community_entropy', 'n_communities'):
            if col in df.columns and df[col].notna().any():
                base = base.add(zscore(df[col].astype(float)).fillna(0.0), fill_value=0.0)
                n += 1
        if n == 0:
            base = zscore(df['frag_index'].astype(float)).fillna(0.0)
        base = base / max(n, 1)
        noise = pd.Series(rng.standard_normal(len(df)), index=df.index)
        df[instrument] = base + 0.5 * noise
        warnings.warn(f"iv_2sls: constructing an ILLUSTRATIVE placeholder instrument '{instrument}'. This is NOT a validated instrument; the 2SLS estimate is for mechanics only, not causal inference.", stacklevel=2)
    controls = _present_controls(df, DEFAULT_CONTROLS)
    cols = [dep, 'frag_index', instrument] + controls
    est = df[[c for c in cols if c in df.columns]].dropna().copy()
    if len(est) < len(controls) + 4:
        raise ValueError(f'iv_2sls: too few complete rows ({len(est)}).')
    y = est[[dep]]
    endog = est[['frag_index']]
    instr = est[[instrument]]
    exog = sm.add_constant(est[controls]) if controls else pd.DataFrame({'const': np.ones(len(est))}, index=est.index)
    res = IV2SLS(y, exog, endog, instr).fit(cov_type='robust')
    fs_y = est[['frag_index']]
    fs_X = sm.add_constant(pd.concat([instr, est[controls]], axis=1)) if controls else sm.add_constant(instr)
    fs = sm.OLS(fs_y, fs_X).fit()
    tval = fs.tvalues.get(instrument, np.nan)
    first_stage_F = float(tval ** 2) if np.isfinite(tval) else float('nan')
    if np.isfinite(first_stage_F) and first_stage_F < 10:
        warnings.warn(f'iv_2sls: WEAK instrument — first-stage F={first_stage_F:.2f} < 10.', stacklevel=2)
    try:
        res.first_stage_F = first_stage_F
        res.illustrative_instrument = illustrative
    except Exception:
        pass
    return res

def placebo(panel: pd.DataFrame, outcome: str, treat: str='frag_index', B: int=500, cfg=CFG) -> dict:
    controls = _present_controls(panel, DEFAULT_CONTROLS)
    df, _ = _build_ols_frame(panel, outcome, treat, controls)
    if len(df) < len(controls) + 3:
        raise ValueError(f'placebo: too few complete rows ({len(df)}).')
    true_res = baseline_ols(df, outcome=outcome, treat=treat, controls=controls, cfg=cfg)
    key = f"Q('{treat}')"
    t_true = float(true_res.tvalues.get(key, np.nan))
    coef_true = float(true_res.params.get(key, np.nan))
    rng = np.random.default_rng(cfg.seed)
    treat_vals = df[treat].to_numpy()
    n = len(df)
    count_ge = 0
    valid = 0
    for _ in range(int(B)):
        perm = df.copy()
        perm[treat] = treat_vals[rng.permutation(n)]
        try:
            r = baseline_ols(perm, outcome=outcome, treat=treat, controls=controls, cfg=cfg)
            t_perm = float(r.tvalues.get(key, np.nan))
        except Exception:
            continue
        if np.isfinite(t_perm):
            valid += 1
            if abs(t_perm) >= abs(t_true):
                count_ge += 1
    p_perm = (count_ge + 1) / (valid + 1) if valid > 0 else float('nan')
    return {'t_true': t_true, 'coef_true': coef_true, 'p_perm': float(p_perm), 'B': valid}

def pretrends(panel: pd.DataFrame, cfg=CFG):
    pre_outcome = None
    for cand in ('pre_car', 'pre_ar', 'prior_return', 'pre_vol'):
        if cand in panel.columns and panel[cand].notna().sum() > 3 and (panel[cand].nunique(dropna=True) > 1):
            pre_outcome = cand
            break
    if pre_outcome is None:
        raise ValueError('pretrends: no pre-event outcome column found (pre_car/pre_ar/prior_return/pre_vol).')
    controls = _present_controls(panel, ['n_comments_z', 'abs_surprise', 'log_mktcap'])
    return baseline_ols(panel, outcome=pre_outcome, treat='frag_index', controls=controls, cfg=cfg)

def robustness_table(panel: pd.DataFrame, outcome: str, cfg=CFG) -> pd.DataFrame:
    df = panel.copy()
    controls = _present_controls(df, DEFAULT_CONTROLS)
    rows = []
    specs: list[tuple[str, str]] = [('frag_index', 'frag_index')]
    for col, sign in FRAG_MEASURES.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            zc = f'_z_{col}'
            df[zc] = sign * zscore(df[col].astype(float))
            specs.append((col, zc))
    for label, treat_col in specs:
        try:
            res = baseline_ols(df, outcome=outcome, treat=treat_col, controls=controls, cfg=cfg)
            key = f"Q('{treat_col}')"
            rows.append({'measure': label, 'coef': float(res.params.get(key, np.nan)), 'se': float(res.bse.get(key, np.nan)), 'p': float(res.pvalues.get(key, np.nan)), 'n': int(res.nobs)})
        except Exception as e:
            rows.append({'measure': label, 'coef': np.nan, 'se': np.nan, 'p': np.nan, 'n': 0, 'error': str(e)})
    out = pd.DataFrame(rows).set_index('measure')
    return out

def run_all(panel: pd.DataFrame, long_df: Optional[pd.DataFrame]=None, cfg=CFG) -> dict:
    panel = prep(panel, cfg=cfg)
    outcome = 'halflife_days' if 'halflife_days' in panel.columns and panel['halflife_days'].notna().sum() > 3 else None
    if outcome is None:
        for cand in ('pead_abs', 'variance_ratio', 'post_vol'):
            if cand in panel.columns and panel[cand].notna().sum() > 3:
                outcome = cand
                break
    if outcome is None:
        raise ValueError('run_all: no usable outcome column in panel.')
    sections: list[str] = []
    coef_rows: list[dict] = []
    key_coefs: dict = {'outcome': outcome}

    def _add_hdr(title):
        sections.append('=' * 78)
        sections.append(title)
        sections.append('=' * 78)
    _add_hdr(f'BASELINE OLS  (outcome = {outcome})  [HC1 robust SE]')
    base = baseline_ols(panel, outcome=outcome, cfg=cfg)
    sections.append(str(base.summary()))
    sections.append('')
    fkey = "Q('frag_index')"
    key_coefs['baseline_frag_coef'] = float(base.params.get(fkey, np.nan))
    key_coefs['baseline_frag_p'] = float(base.pvalues.get(fkey, np.nan))
    coef_rows.append({'spec': 'baseline_ols', 'outcome': outcome, 'term': 'frag_index', 'coef': key_coefs['baseline_frag_coef'], 'se': float(base.bse.get(fkey, np.nan)), 'p': key_coefs['baseline_frag_p'], 'n': int(base.nobs)})
    _add_hdr('BASELINE OLS — n_comments OMITTED (biased; for comparison)')
    controls_no_conf = [c for c in DEFAULT_CONTROLS if c != 'n_comments_z']
    try:
        biased = baseline_ols(panel, outcome=outcome, controls=controls_no_conf, cfg=cfg)
        sections.append(str(biased.summary()))
        key_coefs['baseline_frag_coef_no_confounder'] = float(biased.params.get(fkey, np.nan))
        coef_rows.append({'spec': 'baseline_ols_no_confounder', 'outcome': outcome, 'term': 'frag_index', 'coef': key_coefs['baseline_frag_coef_no_confounder'], 'se': float(biased.bse.get(fkey, np.nan)), 'p': float(biased.pvalues.get(fkey, np.nan)), 'n': int(biased.nobs)})
    except Exception as e:
        sections.append(f'[skipped: {e}]')
    sections.append('')
    _add_hdr('PLACEBO (permutation of treatment)')
    try:
        pl = placebo(panel, outcome=outcome, B=min(500, 500), cfg=cfg)
        sections.append(f"t_true = {pl['t_true']:.4f}   coef_true = {pl['coef_true']:.4f}   p_perm = {pl['p_perm']:.4f}   (B={pl['B']})")
        key_coefs['placebo_p_perm'] = pl['p_perm']
        key_coefs['placebo_t_true'] = pl['t_true']
    except Exception as e:
        sections.append(f'[skipped: {e}]')
    sections.append('')
    _add_hdr('PRE-TRENDS (pre-event outcome ~ frag_index; expect null)')
    try:
        pt = pretrends(panel, cfg=cfg)
        sections.append(str(pt.summary()))
        key_coefs['pretrend_frag_coef'] = float(pt.params.get(fkey, np.nan))
        key_coefs['pretrend_frag_p'] = float(pt.pvalues.get(fkey, np.nan))
        coef_rows.append({'spec': 'pretrends', 'outcome': pt.model.endog_names, 'term': 'frag_index', 'coef': key_coefs['pretrend_frag_coef'], 'se': float(pt.bse.get(fkey, np.nan)), 'p': key_coefs['pretrend_frag_p'], 'n': int(pt.nobs)})
    except Exception as e:
        sections.append(f'[skipped: {e}]')
    sections.append('')
    _add_hdr('ROBUSTNESS (each raw measure, sign-corrected)')
    try:
        rob = robustness_table(panel, outcome=outcome, cfg=cfg)
        sections.append(rob.to_string())
        for measure, r in rob.iterrows():
            coef_rows.append({'spec': 'robustness', 'outcome': outcome, 'term': str(measure), 'coef': r.get('coef', np.nan), 'se': r.get('se', np.nan), 'p': r.get('p', np.nan), 'n': int(r.get('n', 0) or 0)})
    except Exception as e:
        sections.append(f'[skipped: {e}]')
    sections.append('')
    _add_hdr('IV 2SLS  (ILLUSTRATIVE placeholder instrument — NOT causal)')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            iv = iv_2sls(panel, cfg=cfg)
        sections.append(str(iv.summary))
        sections.append(f"\nfirst_stage_F = {getattr(iv, 'first_stage_F', float('nan')):.3f}   illustrative_instrument = {getattr(iv, 'illustrative_instrument', True)}")
        key_coefs['iv_frag_coef'] = float(iv.params.get('frag_index', np.nan))
        key_coefs['iv_first_stage_F'] = float(getattr(iv, 'first_stage_F', np.nan))
        key_coefs['iv_illustrative'] = bool(getattr(iv, 'illustrative_instrument', True))
        coef_rows.append({'spec': 'iv_2sls_illustrative', 'outcome': iv.model.dependent.cols[0], 'term': 'frag_index', 'coef': key_coefs['iv_frag_coef'], 'se': float(iv.std_errors.get('frag_index', np.nan)), 'p': float(iv.pvalues.get('frag_index', np.nan)), 'n': int(iv.nobs)})
    except Exception as e:
        sections.append(f'[skipped: {e}]')
    sections.append('')
    if long_df is not None:
        _add_hdr('DiD PANEL (EntityEffects + TimeEffects, clustered by event)')
        try:
            did = did_panel(long_df, cfg=cfg)
            sections.append(str(did.summary))
            bkey = 'frag_x_post'
            key_coefs['did_beta'] = float(did.params.get(bkey, np.nan))
            key_coefs['did_beta_p'] = float(did.pvalues.get(bkey, np.nan))
            coef_rows.append({'spec': 'did_panel', 'outcome': 'inefficiency', 'term': 'frag_x_post', 'coef': key_coefs['did_beta'], 'se': float(did.std_errors.get(bkey, np.nan)), 'p': key_coefs['did_beta_p'], 'n': int(did.nobs)})
        except Exception as e:
            sections.append(f'[skipped: {e}]')
        sections.append('')
    tables_path = RESULTS / 'regression_tables.txt'
    coefs_path = RESULTS / 'coefficients.csv'
    tables_path.write_text('\n'.join(sections))
    pd.DataFrame(coef_rows).to_csv(coefs_path, index=False)
    key_coefs['_tables_path'] = str(tables_path)
    key_coefs['_coefficients_path'] = str(coefs_path)
    return key_coefs
