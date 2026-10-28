"""
src/utils.py
General utility functions
"""
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests
from src.config import STRUCTURAL_ROI_PREFIXES, STRUCTURAL_ROI_SUFFIXES, FDR_ALPHA, RANDOM_STATE
from pathlib import Path 
import pandas as pd
import numpy as np
from scipy.stats import norm

# ==========================================================
# DATA IMPORT & VALIDATION 
# ==========================================================

def find_input_file() -> Path:
    candidates=[
        Path(__file__).resolve().parent / "ACEs_Neuroimages.xlsx",
        Path(__file__).resolve().parent.parent / "data" / "ACEs_Neuroimages.xlsx",
        Path(__file__).resolve().parent.parent / "data" / "raw" / "ACEs_Neuroimages.xlsx"
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find ACEs_Neuroimages.xlsx in the script folder or in ./data/."
    )

def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing: 
        raise KeyError("Missing required columns: "+ ", ".join(missing))

def get_brain_regions(columns: list[str]) -> list[str]: 
    return [
        col for col in columns if col.startswith(STRUCTURAL_ROI_PREFIXES) 
        or col.endswith(STRUCTURAL_ROI_SUFFIXES)]

# ==========================================================
# FORMATTING
# ==========================================================

def format_p_value(p: float, standard_stars: bool = True) -> str:
    if pd.isna(p):
        return "N/A"
    
    if standard_stars:
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = ""
    else:
        sig = "*" if p < 0.05 else ""
        
    return f"{p:.4f}{sig}"

def format_pct(value: float) -> str:
    return f"{100 * value:.1f}%"

def print_title(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

# ==========================================================
# NORMALIZATION & VARIANCE INFLATION FACTOR
# ==========================================================

def zscore_series(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    sd = series.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index = series.index)
    return (series - series.mean()) / sd

def get_vif_max(model) -> float:
    try:
        x_matrix = model.model.exog
        if x_matrix.shape[1] <= 2:
            return 1.0

        vifs = [
            variance_inflation_factor(x_matrix, j) for j in range(x_matrix.shape[1])
            if model.model.exog_names[j] not in ["Intercept", "const"]
            ]
        
        return float(max(vifs)) if vifs else 1.0
    except Exception:
        return np.nan

# ==========================================================
# RUBIN'S RULES POOLING & MONTE CARLO SIMULATION
# ==========================================================

def pool_rubin_scalar(q_list: list[float], se_list: list[float]) -> dict[str, float]:
    m = len(q_list)
    q_bar = np.mean(q_list)
    
    se_arr = np.array(se_list, dtype=float)
    w = np.mean(se_arr**2)
    b = np.var(q_list, ddof=1) if m > 1 else 0.0 

    t = w + (1 + 1/m) * b
    se_pool = np.sqrt(t) if t > 0 else 1e-15

    z = q_bar / se_pool if se_pool > 0 else 0.0
    p_val = 2*(1 - norm.cdf(abs(z)))
    
    summary = {
        "estimate": float(q_bar),
        "se": float(se_pool),
        "z": float(z),
        "p_value": float(p_val)
    }
    
    return summary

def monte_carlo_indirect(
    paths: list[tuple[float,float]], n_sim: int = 10000, seed: int = RANDOM_STATE) -> dict[str, float]:
    rng = np.random.default_rng(seed)

    prod_samples = np.ones(n_sim)
    prod_estimate = 1.0

    for est, se in paths: 
        prod_samples *= rng.normal(loc=est, scale=se, size=n_sim)
        prod_estimate *= est

    ci_low = np.percentile(prod_samples, 2.5)
    ci_high = np.percentile(prod_samples, 97.5)

    tail_prob = min(np.mean(prod_samples <= 0), np.mean(prod_samples >= 0))
    p_val = min(1.0, 2.0 * tail_prob)

    summary = {
        "estimate": float(prod_estimate),
        "ci_high": float(ci_high),
        "ci_low": float(ci_low),
        "p_value": float(p_val)
    }

    return summary

# ==========================================================
# FDR CORRECTION
# ==========================================================

def apply_fdr_correction(
    df:pd.DataFrame, p_cols: list[str] | str = "p_ab",
    alpha: float = FDR_ALPHA, group_by: str | None = None) -> pd.DataFrame:
    df = df.copy()
    if df.empty: 
        return df

    if isinstance(p_cols, str): 
        p_cols = [p_cols]

    suffix = f"_{group_by}" if group_by else ""

    for col in p_cols: 
        if col not in df.columns:
            continue

        fdr_col = f"{col}_fdr{suffix}"
        sig_col = f"sig_{col}{suffix}"

        if group_by and group_by in df.columns:
            df[fdr_col] = np.nan
            df[sig_col] = False

            for _, grp in df.groupby(group_by):
                rej, p_corr, _, _ = multipletests(grp[col], alpha=alpha, method="fdr_bh")
                df.loc[grp.index, fdr_col] = p_corr
                df.loc[grp.index, sig_col] = rej

        else: 
            rej, p_corr, _, _ = multipletests(df[col], alpha=alpha, method="fdr_bh")
            df[fdr_col] = p_corr
            df[sig_col] = rej 

            if col == "p_ab":
                df["sig_fdr"] = rej
    
    return df