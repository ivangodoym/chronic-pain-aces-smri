"""
src/modeling.py
Modeling: Mediation modeling with Rubin's rules and Monte Carlo simulation
"""

from src.config import RANDOM_STATE, OUTPUTS_TABLES_DIR, DATA_PROCESSED_DIR
from src.utils import get_vif_max, apply_fdr_correction, pool_rubin_scalar, monte_carlo_indirect
import pandas as pd 
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import norm 

# ==========================================================
# MEDIATION MODEL
# ==========================================================

def fit_ols_paths(df: pd.DataFrame, x: str, m: str, y: str, covariates: list[str] = None) -> dict[str, float]:
    if covariates:
        covariates = [c for c in covariates if c in df.columns]        
    cov_str = (" + " + " + ".join(covariates)) if covariates else ""
    
    form_a = f"{m} ~ {x}{cov_str}"
    form_b = f"{y} ~ {x} + {m}{cov_str}"
    form_total = f"{y} ~ {x}{cov_str}"

    formulas = {
        "a": form_a, 
        "b": form_b,
        "total": form_total
    }
    models = {name: smf.ols(form, data=df).fit() for name, form in formulas.items()}

    results = {
        "a": models["a"].params[x],
        "se_a": models["a"].bse[x],       
        "b": models["b"].params[m],
        "se_b": models["b"].bse[m],       
        "c_prime": models["b"].params[x],
        "se_c_prime": models["b"].bse[x],       
        "c_total": models["total"].params[x],
        "se_c_total": models["total"].bse[x]
    }

    results["ab"] = results["a"] * results["b"]
    results["n_obs"] = int(models["b"].nobs)

    vif_a = get_vif_max(models["a"])
    vif_b = get_vif_max(models["b"])
    results["vif_max"] = float(np.nanmax([vif_a, vif_b]))

    return results 
    
def pool_mediation_imputations(fit_list: list[dict], n_sim: int = 10000, seed: int = RANDOM_STATE) -> dict:
    q_a = [f["a"] for f in fit_list]
    se_a = [f["se_a"] for f in fit_list]
    
    q_b = [f["b"] for f in fit_list]
    
    se_b = [f["se_b"] for f in fit_list]

    q_c_prime = [f["c_prime"] for f in fit_list]
    se_c_prime = [f["se_c_prime"] for f in fit_list]

    q_c_total = [f["c_total"] for f in fit_list]
    se_c_total = [f["se_c_total"] for f in fit_list]

    res_a = pool_rubin_scalar(q_a, se_a)    
    res_b = pool_rubin_scalar(q_b, se_b) 
    res_c_prime = pool_rubin_scalar(q_c_prime, se_c_prime)
    res_c_total = pool_rubin_scalar(q_c_total, se_c_total)

    res_ab = monte_carlo_indirect(paths=[
        (res_a["estimate"], res_a["se"]),
        (res_b["estimate"], res_b["se"])], 
        n_sim=n_sim, seed=seed)

    n_obs_mean = int(np.mean([f["n_obs"] for f in fit_list]))
    vifs = [f["vif_max"] for f in fit_list if not np.isnan(f.get("vif_max", np.nan))]
    vif_mean = round(float(np.mean(vifs)), 2) if vifs else np.nan

    results = {
        "a": res_a["estimate"],
        "se_a": res_a["se"],
        "p_a": res_a["p_value"],

        "b": res_b["estimate"],
        "se_b": res_b["se"],
        "p_b": res_b["p_value"],

        "c_prime": res_c_prime["estimate"],
        "se_c_prime": res_c_prime["se"],
        "p_c_prime": res_c_prime["p_value"],

        "c_total": res_c_total["estimate"],
        "se_c_total": res_c_total["se"],
        "p_c_total": res_c_total["p_value"],

        "ab": res_ab["estimate"],
        "ci_low_ab": res_ab["ci_low"],
        "ci_high_ab": res_ab["ci_high"],
        "p_ab": res_ab["p_value"],

        "vif_max": vif_mean,
        "n_obs": n_obs_mean
    }

    return results 

# ==========================================================
# MAIN PIPELINE
# ==========================================================

def run_mediation_pipeline(df: pd.DataFrame, outcome: str, predictors: list[str], 
    mediators: list[str], covariates: list[str]) -> pd.DataFrame: 
    
    rows = []
    imputation_dfs = [group for _, group in df.groupby("_imputation")]
    
    for predictor in predictors: 
        for mediator in mediators: 
            fits = [fit_ols_paths(imp_df, x=predictor, m=mediator, y=outcome, covariates=covariates)
            for imp_df in imputation_dfs]

            pooled = pool_mediation_imputations(fits)
            row = {"predictor": predictor, "mediator": mediator, "outcome": outcome, **pooled}
            rows.append(row)

    results = pd.DataFrame(rows)
    results["stage"] = (
        results["predictor"].str.extract(r"_(childhood|adulthood|recent)", expand=False).str.capitalize()
    )
    results = apply_fdr_correction(results, p_cols=["p_a", "p_b", "p_ab"], group_by="stage")
    results = results.sort_values("p_ab").reset_index(drop=True)

    OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = OUTPUTS_TABLES_DIR / f"mediation_{outcome}_results.csv"
    results.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    return results 

if __name__ == "__main__": 
    df_path = DATA_PROCESSED_DIR / "analysis_ready.csv"
    if df_path.exists():
        df_ready = pd.read_csv(df_path)

        qc_path = DATA_PROCESSED_DIR / "qc_predictors_report.csv"
        if qc_path.exists():
            qc_df = pd.read_csv(qc_path)

            candidate_predictors = qc_df.loc[qc_df["QC_Status"] == "PASSED", "Predictor"].tolist()
        else: 
            candidate_predictors = [c for c in df_ready.columns if c.startswith("Cumulative_")
            and c.endswith("_Z")]

        candidate_predictors = [ p for p in candidate_predictors if not p.startswith("Cumulative_risk")]

        outcomes = ["CP_Intensity_Z", "n_regions_Z"]
        
        for outcome_var in outcomes:
            if outcome_var in df_ready.columns: 
                print(f"Running mediation pipeline for: {outcome_var}")

            candidate_mediators = [
                m for m in ["Thalamus_bilateral_Z", "Ant_Insula_bilateral_Z", "Post_Insula_bilateral_Z"]
                if m in df_ready.columns
                ]

            covar_list = [
                c for c in ["demo_age_Z", "demo_sex", "clinical_diagnosis", "TIV_cm3_Z"]
                if c in df_ready.columns
            ]

            run_mediation_pipeline(
                df=df_ready, 
                outcome=outcome_var, 
                predictors=candidate_predictors,
                mediators=candidate_mediators, 
                covariates=covar_list
            )

    else: 
        print(f"No such file: {df_path}. First execute features.py")