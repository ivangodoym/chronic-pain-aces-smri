"""
src/modeling.py
Modeling: Mediation modeling with Rubin's rules and Monte Carlo simulation
"""

from src.config import FDR_ALPHA
from src.config import RANDOM_STATE, OUTPUTS_TABLES_DIR, DATA_PROCESSED_DIR
from src.utils import get_vif_max, apply_fdr_correction, pool_rubin_scalar, monte_carlo_indirect, print_title
import pandas as pd 
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# ==========================================================
# MEDIATION MODEL
# ==========================================================

def fit_and_pool_model(df: pd.DataFrame, formula: str, family: str ="ols") -> dict: 
    imputation_dfs = [grp for _, grp in df.groupby("_imputation")] if "_imputation" in df.columns else [df]
    m_imputations = len(imputation_dfs)

    fits = []
    vifs = []
    nobs_list = []

    for imp_df in imputation_dfs:
        if family == "ols":
            model = smf.ols(formula, data=imp_df).fit()
            vifs.append(get_vif_max(model))
        elif family == "logit":
            try: 
                model = smf.logit(formula, data=imp_df).fit(disp=False, maxiter=100)
            except np.linalg.LinAlgError:
                model = smf.logit(formula, data=imp_df).fit(method="bfgs", disp=False, maxiter=100)
        else: 
            raise ValueError(f"Model family not supported: {family}. Use 'ols' or 'logit'.")

        fits.append(model)
        nobs_list.append(int(model.nobs))

    all_terms = fits[0].params.index.tolist()
    pooled_terms = {}

    for term in all_terms:
        q_list = [f.params[term] for f in fits]
        se_list = [f.bse[term] for f in fits]

        if m_imputations > 1:
            rubin = pool_rubin_scalar(q_list, se_list)
            pooled_terms[term] = {
                    "estimate": rubin["estimate"],
                    "se": rubin["se"],
                    "stat": rubin.get("t", rubin.get("z", np.nan)),
                    "p_value": rubin["p_value"]
                }
        else: 
            pooled_terms[term] = {
                "estimate": fits[0].params[term],
                "se": fits[0].bse[term],
                "stat": fits[0].tvalues[term] if hasattr(fits[0], "tvalues") else fits[0].zvalues[term],
                "p_value": fits[0].pvalues[term]
            }

    vif_mean = round(float(np.nanmean(vifs)), 2) if vifs and not np.all(np.isnan(vifs)) else np.nan
    n_obs_mean = int(np.mean(nobs_list))

    return {
        "terms": pooled_terms, 
        "n_obs": n_obs_mean,
        "vif_max": vif_mean
    }

def run_moderation_pipeline(df: pd.DataFrame, predictors: list[str], rois: list[str], pain_moderators: list[str],
    covariates: list[str], save_csv: bool = True) -> dict[str, pd.DataFrame]:
    df_pain = df[df["CP_1"] == 1].copy() if "CP_1" in df.columns else df.copy()

    covs = [c for c in covariates if c in df_pain.columns]
    cov_str = (" + " + " + ".join(covs)) if covs else ""

    moderation_results = {}

    for mod in pain_moderators: 
        if mod not in df_pain.columns:
            continue

        rows = []
        for roi in rois:
            if roi not in df_pain.columns:
                continue

            for pred in predictors: 
                if pred not in df_pain.columns:
                    continue 

                formula = f"{roi} ~ {pred} * {mod} {cov_str}"
                res = fit_and_pool_model(df_pain, formula, family="ols")

                term_pred = res["terms"].get(pred, {})
                term_mod = res["terms"].get(mod, {})
                term_int = res["terms"].get(f"{pred}:{mod}", {})

                rows.append({
                    "roi": roi,
                    "predictor": pred,
                    "moderator": mod,
                    "b_predictor": term_pred.get("estimate", np.nan),
                    "se_predictor": term_pred.get("se", np.nan),
                    "p_predictor": term_pred.get("p_value", np.nan),
                    "b_moderator": term_mod.get("estimate", np.nan),
                    "se_moderator": term_mod.get("se", np.nan),
                    "p_moderator": term_mod.get("p_value", np.nan),
                    "b_interaction": term_int.get("estimate", np.nan),
                    "se_interaction": term_int.get("se", np.nan),
                    "p_interaction": term_int.get("p_value", np.nan),
                    "n_obs": res["n_obs"],
                    "vif_max": res["vif_max"]
                })
        
        mod_df = pd.DataFrame(rows)
        if not mod_df.empty:
            mod_df["stage"] = (
                mod_df["predictor"].str.extract(r"_(childhood|adulthood|recent)", expand=False).str.capitalize()
                )
            mod_df = apply_fdr_correction(
                    mod_df, p_cols=["p_predictor", "p_moderator", "p_interaction"], group_by="predictor"
                )
            mod_df = mod_df.sort_values("p_interaction").reset_index(drop=True)
        
            if save_csv: 
                OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
                csv_path = OUTPUTS_TABLES_DIR / f"moderation_{mod}_results.csv"
                mod_df.to_csv(csv_path, index=False)
                print(f"Moderation results saved to: {csv_path}")

            moderation_results[mod] = mod_df

    return moderation_results   

def run_logistic_pipeline(df: pd.DataFrame, predictors: list[str], rois: list[str], 
    covariates: list[str], outcome: str = "CP_1", save_csv: bool = True) -> pd.DataFrame: 
    if outcome not in df.columns: 
        raise KeyError(f"Binary outcome '{outcome}' not found in DataFrame")

    covs = [c for c in covariates if c in df.columns]
    cov_str = (" + " + " + ".join(covs)) if covs else ""

    rows = []

    for roi in rois: 
        if roi not in df.columns: 
            continue 
        for pred in predictors: 
            if pred not in df.columns: 
                continue

            formula = f"{outcome} ~ {pred} + {roi}{cov_str}"
            res = fit_and_pool_model(df, formula, family="logit")

            term_pred = res["terms"].get(pred, {})
            term_roi = res["terms"].get(roi, {})
            
            b_pred = term_pred.get("estimate", np.nan)
            se_pred = term_pred.get("se", np.nan)
            p_pred = term_pred.get("p_value", np.nan)

            b_roi = term_roi.get("estimate", np.nan)
            se_roi = term_roi.get("se", np.nan)
            p_roi = term_roi.get("p_value", np.nan)

            or_pred = np.exp(b_pred) if not np.isnan(b_pred) else np.nan
            or_ci_low_pred = np.exp(b_pred - 1.96 * se_pred) if not np.isnan(b_pred) else np.nan
            or_ci_high_pred = np.exp(b_pred + 1.96 * se_pred) if not np.isnan(b_pred) else np.nan

            or_roi = np.exp(b_roi) if not np.isnan(b_roi) else np.nan
            or_ci_low_roi = np.exp(b_roi - 1.96 * se_roi) if not np.isnan(b_roi) else np.nan
            or_ci_high_roi = np.exp(b_roi + 1.96 * se_roi) if not np.isnan(b_roi) else np.nan

            rows.append({
                    "outcome": outcome,
                    "roi": roi,
                    "predictor": pred,
                    "b_predictor": b_pred,
                    "se_predictor": se_pred,
                    "p_predictor": p_pred,
                    "or_predictor": or_pred,
                    "or_ci_low_predictor": or_ci_low_pred,
                    "or_ci_high_predictor": or_ci_high_pred,
                    "b_roi": b_roi,
                    "se_roi": se_roi,
                    "p_roi": p_roi,
                    "or_roi": or_roi,
                    "or_ci_low_roi": or_ci_low_roi,
                    "or_ci_high_roi": or_ci_high_roi,
                    "n_obs": res["n_obs"]
                })

    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        pred_unique = results_df.drop_duplicates(subset=["predictor"]).copy()
        rej_pred, p_fdr_pred, _, _ = multipletests(pred_unique["p_predictor"], alpha=FDR_ALPHA, method="fdr_bh")
        results_df["p_predictor_fdr"] = results_df["predictor"].map(dict(zip(pred_unique["predictor"], p_fdr_pred)))
        results_df["sig_p_predictor"] = results_df["predictor"].map(dict(zip(pred_unique["predictor"], rej_pred)))

        roi_unique = results_df.drop_duplicates(subset=["roi"]).copy()
        rej_roi, p_fdr_roi, _, _ = multipletests(roi_unique["p_roi"], alpha=FDR_ALPHA, method="fdr_bh")
        results_df["p_roi_fdr"] = results_df["roi"].map(dict(zip(roi_unique["roi"], p_fdr_roi)))
        results_df["sig_p_roi"] = results_df["roi"].map(dict(zip(roi_unique["roi"], rej_roi)))

        if save_csv: 
            OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
            csv_path = OUTPUTS_TABLES_DIR / f"logistic_{outcome}_results.csv"
            results_df.to_csv(csv_path, index=False)
            print(f"Logistic regression results saved to: {csv_path}")

    return results_df

def run_mediation_pipeline(df: pd.DataFrame, predictors: list[str], mediators: list[str], outcomes: list[str],
    covariates: list[str], prefix: str = "mediation", save_csv: bool = True) -> dict[str, pd.DataFrame]:
    covs = [c for c in covariates if c in df.columns]
    cov_str = (" + " + " + ".join(covs)) if covs else ""

    results_by_outcome = {}

    for outcome in outcomes: 
        if outcome not in df.columns: 
            continue

        rows = []
        for pred in predictors: 
            if pred not in df.columns: 
                continue
            
            for med in mediators: 
                if med not in df.columns:
                    continue

                form_a = f"{med} ~ {pred}{cov_str}"
                res_a = fit_and_pool_model(df, form_a, family="ols")
                t_a = res_a["terms"].get(pred, {})

                form_b = f"{outcome} ~ {pred} + {med}{cov_str}"
                res_b = fit_and_pool_model(df, form_b, family="ols")
                t_b = res_b["terms"].get(med, {})
                t_cp = res_b["terms"].get(pred, {})

                form_tot = f"{outcome} ~ {pred}{cov_str}"
                res_tot = fit_and_pool_model(df, form_tot, family="ols")
                t_tot = res_tot["terms"].get(pred, {})

                est_a, se_a = t_a.get("estimate", np.nan), t_a.get("se", np.nan)
                est_b, se_b = t_b.get("estimate", np.nan), t_b.get("se", np.nan)

                if not np.isnan(est_a) and not np.isnan(est_b):
                    mc = monte_carlo_indirect(paths=[(est_a, se_a), (est_b, se_b)], seed=RANDOM_STATE)
                else: 
                    mc = {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}

                vif_max = float(np.nanmax([res_a["vif_max"], res_b["vif_max"]]))

                rows.append({
                    "predictor": pred,
                    "mediator": med,
                    "outcome": outcome,
                    "a": est_a, "se_a": se_a, "p_a": t_a.get("p_value", np.nan),
                    "b": est_b, "se_b": se_b, "p_b": t_b.get("p_value", np.nan),
                    "c_prime": t_cp.get("estimate", np.nan), "se_c_prime": t_cp.get("se", np.nan), "p_c_prime": t_cp.get("p_value", np.nan),
                    "c_total": t_tot.get("estimate", np.nan), "se_c_total": t_tot.get("se", np.nan), "p_c_total": t_tot.get("p_value", np.nan),
                    "ab": mc["estimate"], "ci_low_ab": mc["ci_low"], "ci_high_ab": mc["ci_high"], "p_ab": mc["p_value"],
                    "vif_max": vif_max,
                    "n_obs": res_b["n_obs"]
                })

        res_df = pd.DataFrame(rows)
        if not res_df.empty: 
            res_df["stage"] = (
                res_df["predictor"].str.extract(r"_(childhood|adulthood|recent)", expand=False).str.capitalize()
            )
            res_df = apply_fdr_correction(res_df, p_cols=["p_a", "p_b", "p_ab"], group_by="predictor")
            res_df = res_df.sort_values("p_ab").reset_index(drop=True)

            if save_csv: 
                OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
                csv_path = OUTPUTS_TABLES_DIR / f"{prefix}_{outcome}_results.csv"
                res_df.to_csv(csv_path, index=False)
                print(f"Mediation results saved to: {csv_path}")

            results_by_outcome[outcome] = res_df
    
    return results_by_outcome


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def run_modeling() -> None: 
    df_path = DATA_PROCESSED_DIR / "analysis_ready.csv"
    if not df_path.exists():
        raise FileNotFoundError(f"{df_path} not found. Run features.py first.")

    df_ready = pd.read_csv(df_path)
    qc_path = DATA_PROCESSED_DIR / "qc_predictors_report.csv"
    if qc_path.exists():
        qc_df = pd.read_csv(qc_path)

        candidate_predictors = qc_df.loc[qc_df["QC_Status"] == "PASSED", "Predictor"].tolist()
    else: 
        candidate_predictors = [c for c in df_ready.columns if c.startswith("Cumulative_")
        and c.endswith("_Z")]

    candidate_predictors = [p for p in candidate_predictors if not p.startswith("Cumulative_risk")]

    brain_rois = [
        m for m in [
            "Thalamus_bilateral_Z", "Ant_Insula_bilateral_Z", "Post_Insula_bilateral_Z",
            "Hippocampus_bilateral_Z", "ParaHippocampal_bilateral_Z", "Calcarine_bilateral_Z",
            "Cingulum_Ant_bilateral_Z"
        ] if m in df_ready.columns
    ]

    covar_linear = [
        c for c in ["demo_age_Z", "demo_sex", "clinical_diagnosis", "TIV_cm3_Z", "site"]
        if c in df_ready.columns
    ]

    covar_logistic = [c for c in covar_linear if c != "site"]

    pain_continous = [
        p for p in ["CP_Intensity_Z", "n_regions_Z", "Pain_Latent_Z"]
        if p in df_ready.columns 
    ]

    df_pain = df_ready[df_ready["CP_1"] == 1].copy()

    print_title(f"EXECUTING LOGISTIC REGRESSION: PAIN PRESENCE (N = {df_ready["ID"].nunique()})")
    run_logistic_pipeline(
        df=df_ready, 
        predictors=candidate_predictors, 
        rois=brain_rois, 
        covariates=covar_logistic,
        outcome="CP_1",
        save_csv=True
    )

    print_title(f"EXECUTING DIRECT MEDIATION (N = {df_pain["ID"].nunique()})")
    run_mediation_pipeline(
        df=df_pain,
        predictors=candidate_predictors, 
        mediators=brain_rois, 
        outcomes=pain_continous,
        covariates=covar_linear,
        prefix="mediation", 
        save_csv=True
    )

    print_title(f"EXECUTING INVERSE MEDIATION (N = {df_pain["ID"].nunique()})")
    for pain_med in pain_continous:
        inv_dict = run_mediation_pipeline(
            df=df_pain,
            predictors=candidate_predictors, 
            mediators=[pain_med],
            outcomes=brain_rois, 
            covariates=covar_linear,
            prefix="temp", 
            save_csv=False
        )
        if inv_dict: 
            df_inv = pd.concat(inv_dict.values(), ignore_index=True)
            df_inv = apply_fdr_correction(df_inv, p_cols=["p_a", "p_b", "p_ab"], group_by="predictor")
            df_inv = df_inv.sort_values("p_ab").reset_index(drop=True)
            out_csv = OUTPUTS_TABLES_DIR / f"inverse_mediation_{pain_med}_results.csv"
            df_inv.to_csv(out_csv, index=False)
            print(f"Inverse mediation results saved to: {out_csv}")

    
    print_title(f"EXECUTING MODERATION (N = {df_pain["ID"].nunique()})")
    run_moderation_pipeline(
        df=df_pain,
        predictors=candidate_predictors,
        rois=brain_rois, 
        pain_moderators=pain_continous,
        covariates=covar_linear,
        save_csv=True
    )
    

if __name__ == "__main__": 
    run_modeling()
