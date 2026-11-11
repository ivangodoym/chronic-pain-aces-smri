"""
src/sem_modeling.py
Complementary analysis using Structural Equation Modeling
"""
import pandas as pd 
import numpy as np 
import semopy 
from statsmodels.stats.multitest import multipletests
from src.config import DATA_PROCESSED_DIR, OUTPUTS_TABLES_DIR, RANDOM_STATE
from src.utils import print_title, zscore_series, pool_rubin_scalar

# ==========================================================
# SEM MODEL SYNTAX
# ==========================================================

def build_sem_syntax(rois: list[str] = None, pain_outcome: str = "Pain_Latent_Z") -> str:
    if rois is None:
        rois = [
            "Thalamus_bilateral_Z", "Ant_Insula_bilateral_Z", "Post_Insula_bilateral_Z",
            "Cingulum_Ant_bilateral_Z", "Hippocampus_bilateral_Z", "ParaHippocampal_bilateral_Z"
        ]

    covars_brain = "demo_age_Z + demo_sex + TIV_cm3_Z + clinical_diagnosis"
    covars_pain = "demo_age_Z + demo_sex + clinical_diagnosis"
    adversity_preds = (
        "Threat_Child + Depriv_Child + " "Cumulative_threat_adulthood_Z + Cumulative_deprivation_adulthood_Z"
    )

    lines = [
        "# 1. Measurement Model (Childhood Latent Variables)",
        "Threat_Child =~ Physical_threat_childhood_Z + Psychological_threat_childhood_Z",
        "Depriv_Child =~ Asset_deprivation_childhood_Z + Basic_needs_childhood_Z",
        "Threat_Child ~~ Depriv_Child",
        "", 
        "# 2. Life-Course Adversity Cascade (Childhood -> Adulthood)",
        "Cumulative_threat_adulthood_Z ~ Threat_Child + Depriv_Child", 
        "Cumulative_deprivation_adulthood_Z ~ Threat_Child + Depriv_Child",
        "Cumulative_threat_adulthood_Z ~~ Cumulative_deprivation_adulthood_Z",
        "",
        "# 3. Brain Structural ROIs (Parallel Neurostructural Hubs)"
    ]

    for roi in rois: 
        eq = f"{roi} ~ {adversity_preds} + {covars_brain}"
        lines.append(eq)

    lines.append("")
    lines.append("# Residual Covariances among Brain ROIs")

    for i in range(len(rois)):
        for j in range(i + 1, len(rois)):
            lines.append(f"{rois[i]} ~~ {rois[j]}")

    brain_terms = " + ".join(rois)
    lines.extend([
        "",
        "# 4. Clinical Pain Outcome",
        f"{pain_outcome} ~ {brain_terms} + {adversity_preds} + {covars_pain}"
    ])

    return "\n".join(lines)

# ==========================================================
# SEM MODEL SYNTAX
# ==========================================================

def fit_imputed_sem(df: pd.DataFrame, syntax: str = None, pain_only: bool = True) -> tuple[pd.DataFrame, dict]:
    if syntax is None: 
        syntax = build_sem_syntax()

    if pain_only and "CP_1" in df.columns: 
        df = df[df["CP_1"] == 1].copy()

    imputation_dfs = [grp for _, grp in df.groupby("_imputation")] if "_imputation" in df.columns else [df]
    m_imputations = len(imputation_dfs)

    all_estimates = []
    all_fit_stats = []

    print_title(f"Fitting SEM model on {m_imputations} imputations (N = {df["ID"].nunique() if "ID" in df.columns else len(df)})")
    
    for i, imp_df in enumerate(imputation_dfs, start=1):
        try:
            model = semopy.Model(syntax)
            model.fit(imp_df)

            params = model.inspect()
            all_estimates.append(params)

            stats = semopy.calc_stats(model)
            all_fit_stats.append(stats) 
            print(f" Imputation {i}/{m_imputations} fitted successfully")
        except Exception as e: 
            print(f"Warning on imputation {i}: {e}")

    if not all_estimates:
        raise RuntimeError("SEM model couldn't converge in any imputations.")

    template = all_estimates[0].copy()
    pooled_rows = []

    for _, row in template.iterrows():
        lval, op, rval = row["lval"], row["op"], row["rval"]

        q_list = []
        se_list = []

        for est_df in all_estimates: 
            match = est_df[(est_df["lval"] == lval) & (est_df["rval"] == rval) & (est_df["op"] == op)]

            if not match.empty: 
                q_list.append(match.iloc[0]["Estimate"])
                se_list.append(match.iloc[0]["Std. Err"])

        if q_list and se_list:
            try:
                pooled = pool_rubin_scalar(q_list, se_list)
                pooled_rows.append({
                    "lval": lval,
                    "op": op,
                    "rval": rval,
                    "Estimate": pooled["estimate"],
                    "Std. Err": pooled["se"],
                    "z-value": pooled["z"],
                    "p-value": pooled["p_value"],
                    "n_imputations": m_imputations
                })
            except (ValueError, TypeError): 
                pooled_rows.append({
                    "lval": lval,
                    "op": op,
                    "rval": rval,
                    "Estimate": float(q_list[0]) if q_list[0] != "-" else np.nan,
                    "Std. Err": np.nan,
                    "z-value": np.nan,
                    "p-value": np.nan,
                    "n_imputations": m_imputations
                })

    pooled_df = pd.DataFrame(pooled_rows)

    reg_mask = (pooled_df["op"] == "~") & (pooled_df["p-value"].notna())
    pooled_df["p_fdr"] = np.nan
    pooled_df["sig_fdr"]= False
    if reg_mask.sum() > 0: 
        _, p_adj, _, _ = multipletests(pooled_df.loc[reg_mask, "p-value"], method="fdr_bh")
        pooled_df.loc[reg_mask, "p_fdr"] = p_adj
        pooled_df.loc[reg_mask, "sig_fdr"] = p_adj < 0.05

    pooled_df["ci_lower"] = pooled_df["Estimate"] - 1.96 * pooled_df["Std. Err"]
    pooled_df["ci_upper"] = pooled_df["Estimate"] + 1.96 * pooled_df["Std. Err"]

    fit_summary = {}
    if all_fit_stats: 
        fit_concat = pd.concat(all_fit_stats)

        col_map = {c.lower(): c for c in fit_concat.columns}
        for target in ["chi2", "dof", "cfi", "tli", "rmsea", "aic", "bic"]:
            if target in col_map:
                actual_col = col_map[target]
                vals = fit_concat[actual_col].dropna()
                fit_summary[f"{target.upper()}_mean"] = round(float(vals.mean()), 4)


    return pooled_df, fit_summary

# ==========================================================
# INDIRECT EFFECTS (Monte Carlo Simulation)
# ==========================================================

def calculate_sem_indirects(pooled_df: pd.DataFrame, n_sim: int = 10000, seed: int = RANDOM_STATE,
    rois: list[str] = None, pain_outcome: str = "Pain_Latent_Z") -> pd.DataFrame:

    if rois is None: 
        rois = [
            "Thalamus_bilateral_Z", "Ant_Insula_bilateral_Z", "Post_Insula_bilateral_Z", "Cingulum_Ant_bilateral_Z",
            "Hippocampus_bilateral_Z", "ParaHippocampal_bilateral_Z"
        ]

    param_lookup = {
        (row["lval"], row["op"], row["rval"]): (float(row["Estimate"]), float(row["Std. Err"]))
        for _, row in pooled_df.iterrows()
    }

    rng = np.random.default_rng(seed)
    records = []

    hypotheses = []

    for roi in rois: 
        hypotheses.append({
            "category" : "Childhood_Neurodevelopment",
            "name": f"ChildThreat_{roi}_{pain_outcome}",
            "desc": f"Threat_Child -> {roi} -> {pain_outcome}",
            "steps": [(roi, "~", "Threat_Child"), (pain_outcome, "~", roi)]
        })

        hypotheses.append({
            "category": "Childhood_Neurodevelopment", 
            "name": f"ChildDepriv_{roi}_{pain_outcome}",
            "desc": f"Depriv_Child -> {roi} -> {pain_outcome}",
            "steps": [(roi, "~", "Depriv_Child"), (pain_outcome, "~", roi)]
        })

    hypotheses.append({
        "category": "Life_Course_Cascade",
        "name": f"Cascade_Threat_{pain_outcome}",
        "desc": f"Threat_Child -> Cumulative_threat_adulthood_Z -> {pain_outcome}",
        "steps": [
            ("Cumulative_threat_adulthood_Z", "~", "Threat_Child"),
            (pain_outcome, "~", "Cumulative_threat_adulthood_Z")
        ]
    })

    hypotheses.append({
        "category": "Life_Course_Cascade",
        "name": f"Cascade_Depriv_{pain_outcome}",
        "desc": f"Depriv_Child -> Cumulative_deprivation_adulthood_Z -> {pain_outcome}", 
        "steps": [
            ("Cumulative_deprivation_adulthood_Z", "~", "Depriv_Child"),
            (pain_outcome, "~", "Cumulative_deprivation_adulthood_Z")
        ]
    })
        
    for roi in rois: 
        hypotheses.append({
            "category": "Adult_Neurostructural", 
            "name": f"AdultThreat_{roi}_{pain_outcome}",
            "desc": f"Threat_Adulthood -> {roi} -> {pain_outcome}",
            "steps": [(roi, "~", "Cumulative_threat_adulthood_Z"), (pain_outcome, "~", roi)]   
        })

        hypotheses.append({
            "category": "Adult_Neurostructural", 
            "name": f"AdultDepriv_{roi}_{pain_outcome}",
            "desc": f"Depriv_Adulthood -> {roi} -> {pain_outcome}",
            "steps": [(roi, "~", "Cumulative_deprivation_adulthood_Z"), (pain_outcome, "~", roi)]
        })

    for hyp in hypotheses: 
        step_params = [param_lookup.get(step) for step in hyp["steps"]]

        if any(p is None for p in step_params):
            continue

        point_est = np.prod([p[0] for p in step_params])
        sim_draws = np.prod([rng.normal(p[0], p[1], n_sim) for p in step_params], axis=0)

        se_sim = float(np.std(sim_draws))
        ci_low, ci_high = np.percentile(sim_draws, [2.5, 97.5])
        p_val = 2.0 * min(float((sim_draws <= 0).mean()), float((sim_draws >= 0).mean()))
        p_val = min(p_val, 1.0)

        records.append({
            "category": hyp["category"],
            "effect_name": hyp["name"],
            "pathway": hyp["desc"],
            "point_estimate": round(point_est, 5),
            "se": round(se_sim, 5),
            "ci_low": round(ci_low, 5),
            "ci_high": round(ci_high, 5),
            "p_value": round(p_val, 5)
        })

    indirect_df = pd.DataFrame(records)

    if not indirect_df.empty: 
        indirect_df["p_fdr"] = np.nan
        for cat, grp in indirect_df.groupby("category"):
            _, p_adj, _, _ = multipletests(grp["p_value"], method="fdr_bh")
            indirect_df.loc[grp.index, "p_fdr"] = p_adj

        indirect_df["sig_fdr"] = indirect_df["p_fdr"] < 0.05
        indirect_df = indirect_df.sort_values("p_value").reset_index(drop=True)

    return indirect_df

# ==========================================================
# MAIN PIPELINE
# ==========================================================

def run_sem_pipeline(save_csv: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df_path = DATA_PROCESSED_DIR / "analysis_ready.csv"
    if not df_path.exists():
        raise FileNotFoundError(f"{df_path} not found. Execute src.features first.")

    df_ready = pd.read_csv(df_path)

    target_rois = [
        "Thalamus_bilateral_Z", "Ant_Insula_bilateral_Z", "Post_Insula_bilateral_Z", 
        "Cingulum_Ant_bilateral_Z", "Hippocampus_bilateral_Z", "ParaHippocampal_bilateral_Z"
    ]
    available_rois = [r for r in target_rois if r in df_ready.columns]

    pain_outcomes = [p for p in ["Pain_Latent_Z", "CP_Intensity_Z", "n_regions_Z"] if p in df_ready.columns]

    all_sem_results = {}

    for pain_target in pain_outcomes: 
        print_title(f"RUNNING SEM PIPELINE FOR OUTCOME: {pain_target}")
            
        syntax = build_sem_syntax(rois=available_rois, pain_outcome=pain_target)
        pooled_params, fit_stats = fit_imputed_sem(df_ready, syntax=syntax, pain_only=True)
        indirect_effects = calculate_sem_indirects(pooled_params, pain_outcome=pain_target, rois=available_rois)

        if save_csv: 
            OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

            p_path = OUTPUTS_TABLES_DIR / f"sem_pooled_parameters_{pain_target}.csv"
            pooled_params.to_csv(p_path, index=False)
            print(f"SEM combined parameters saved in: {p_path}")

            ie_path = OUTPUTS_TABLES_DIR / f"sem_indirect_effects_{pain_target}.csv"
            indirect_effects.to_csv(ie_path, index=False)
            print(f"Indirect SEM effects saved in: {ie_path}")

            if fit_stats: 
                fit_df = pd.DataFrame([fit_stats])
                fit_path = OUTPUTS_TABLES_DIR / f"sem_fit_indices_{pain_target}.csv"
                fit_df.to_csv(fit_path, index=False)
                print(f"Goodness-of-Fit Index (GFI) values saved in: {fit_path}")
            
        all_sem_results[pain_target] = {
            "parameters": pooled_params,
            "indirect_effects": indirect_effects,
            "fit_stats": fit_stats
        }

    return all_sem_results

if __name__ == "__main__":
    run_sem_pipeline()