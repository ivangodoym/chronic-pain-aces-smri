"""
src/features.py
Feature preparation: Create ACE-related, pain phenotypes and structural neuroimaging features
"""
from src.config import (
    ADVERSITY_DOMAINS, ADVERSITY_ITEMS, TEMPORAL_BLOCKS, RANDOM_STATE,
    ADVERSITY_DIMENSIONAL_MAP, DATA_PROCESSED_DIR, PAIN_REGIONS
    ) 
from src.utils import zscore_series, get_brain_regions, print_title
import pandas as pd 
import numpy as np 
from sklearn.decomposition import FactorAnalysis

# ==========================================================
# CLINICAL VARIABLES
# ==========================================================

def calculate_adversity_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for domain, cols in ADVERSITY_DOMAINS.items():
        df[domain] = df[cols].sum(axis=1)
    
    for stage, domain in TEMPORAL_BLOCKS.items():
        df[f"Cumulative_risk_{stage.lower()}"] = df[domain].sum(axis=1)

    for score_name, subdomains in ADVERSITY_DIMENSIONAL_MAP.items():
        df[f"{score_name}"] = df[subdomains].sum(axis=1)
    
    df["Threat_adulthood_any"] = (df["Cumulative_threat_adulthood"] > 0).astype(float)

    return df

# ==========================================================
# PAIN VARIABLES    
# ==========================================================

def prepare_pain_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["n_regions"] = df[PAIN_REGIONS].sum(axis=1, min_count=len(PAIN_REGIONS))
    
    df.loc[df["n_regions"].notna() & ((df["n_regions"] < 0) | (df["n_regions"] > 17)), "n_regions"] = np.nan
    df.loc[df["CP_1"] == 0, "n_regions"] = 0

    freq_map_weekly = {0: 0.0, 1: 7.0, 2: 1.5, 3: 3.5, 4: 8.0}
    freq_days = df["CP_Frequency"].map(freq_map_weekly)

    df["CP_Frequency"] = df["CP_Frequency"].map({2:1, 3:2, 4:3, 1:4})
    df.loc[df["CP_Intensity"].notna() & ((df["CP_Intensity"] < 0) | (df["CP_Intensity"] > 10)), "CP_Intensity"] = np.nan

    int_01 = (df["CP_Intensity"] / 10.0).clip(0, 1)
    frq_01 = (freq_days / 8.0).clip(0, 1)
    sites_01 = (df["n_regions"] / 17.0).clip(0, 1)

    df["Pain_Latent"] = np.nan 
    pain_mask = (df["CP_1"] == 1) & int_01.notna() & frq_01.notna() & sites_01.notna()

    if pain_mask.sum() > 5:
        X_pain = np.column_stack([int_01[pain_mask], frq_01[pain_mask], sites_01[pain_mask]])
        fa = FactorAnalysis(n_components=1, random_state=RANDOM_STATE)
        latent_scores = fa.fit_transform(X_pain).flatten()
    
        if np.corrcoef(latent_scores, int_01[pain_mask].values)[0, 1] < 0:
            latent_scores = -latent_scores

        df.loc[pain_mask, "Pain_Latent"] = latent_scores

    return df

# ==========================================================
# BRAIN STRUCTURE VARIABLES
# ==========================================================

def prepare_structural_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    brain_structures = [
        "Thalamus",
        "Ant_Insula",
        "Post_Insula",
        "ParaHippocampal",
        "Hippocampus",
        "Cingulum_Ant",
        "Calcarine"
    ]
    
    for structure in brain_structures: 
        col_l, col_r = f"{structure}_L", f"{structure}_R"
        if col_l in df.columns and col_r in df.columns:
            df[f"{structure}_bilateral"] = df[col_l] + df[col_r]

    return df 

# ==========================================================
# STANDARDIZATION
# ==========================================================

def standardize_variables(df: pd.DataFrame, columns_to_zscore: list[str] = None) -> pd.DataFrame:
    df = df.copy()

    if columns_to_zscore is None:
        brain_cols = get_brain_regions(df.columns)
        bilateral_cols = [
            c for c in [
                "Thalamus_bilateral", "Ant_Insula_bilateral", "Post_Insula_bilateral",
                "ParaHippocampal_bilateral", "Hippocampus_bilateral", "Calcarine_bilateral",
                "Cingulum_Ant_bilateral"]
            if c in df.columns]
        
        adversity_cols = (
            list(ADVERSITY_DOMAINS.keys()) 
            + 
            list(ADVERSITY_DIMENSIONAL_MAP.keys()) + [f"Cumulative_risk_{s.lower()}" for s in TEMPORAL_BLOCKS]
            +
            ["Threat_adulthood_any"]
            )
        pain_cols = [c for c in ["CP_Intensity", "n_regions", "Pain_Latent"] if c in df.columns]
        covar_cols = [c for c in ["TIV_cm3", "demo_age"] if c in df.columns]

        columns_to_zscore = brain_cols + bilateral_cols + adversity_cols + pain_cols + covar_cols
        
    z_dict = {}
    valid_cols = [c for c in columns_to_zscore if c in df.columns]

    if "_imputation" in df.columns:
        for col in valid_cols: 
                z_dict[f"{col}_Z"] = df.groupby("_imputation")[col].transform(zscore_series)
    else: 
        for col in valid_cols:
                z_dict[f"{col}_Z"] = zscore_series(df[col])

    z_df = pd.DataFrame(z_dict, index=df.index)
    df = pd.concat([df, z_df], axis=1)

    return df

# ==========================================================
# PREDICTOR QUALITY CONTROL
# ==========================================================

def filter_predictors_qc(
    df: pd.DataFrame, candidate_predictors: list[str],
    max_skew: float = 2.5, max_floor_pct: float = 0.85) -> tuple[list[str], pd.DataFrame]:

    passed_predictors = []
    qc_records = []

    for var in candidate_predictors:
        if var not in df.columns:
            continue

        series = df[var].dropna()
        skew = series.skew()
        min_val = series.min()
        floor_pct = (series == min_val).mean()

        passes_skew = abs(skew) <= max_skew
        passes_floor = floor_pct <= max_floor_pct
        passes_qc = passes_skew and passes_floor

        if passes_qc:
            passed_predictors.append(var)
        
        qc_records.append({
            "Predictor": var, 
            "Skewness": round(skew, 2),
            "Floor %": f"{floor_pct * 100:.1f}%",
            "Outliers_3SD": (series.abs() > 3.0).sum(),
            "QC_Status": "PASSED" if passes_qc else "FAILED"
        })

    summary_df = pd.DataFrame(qc_records)
    print_title("QUALITY CONTROL GATE: PREDICTOR VALIDATION")
    print(summary_df.to_string(index=False))
    print(f"\n Predictors approved for modeling: {len(passed_predictors)}/{len(candidate_predictors)}")

    return passed_predictors, summary_df

# ==========================================================
# MAIN PIPELINE
# ==========================================================

def apply_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = calculate_adversity_scores(df)
    df = prepare_pain_outcomes(df)
    df = prepare_structural_variables(df)
    df = standardize_variables(df)

    return df

def run_feature_engineering(
    input_filename: str = "imputed_datasets.csv", output_filename: str = "analysis_ready.csv"
) -> pd.DataFrame: 
    print_title("RUNNING FEATURE ENGINEERING PIPELINE")
    input_path = DATA_PROCESSED_DIR / input_filename

    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found. Execute preprocessing.py first")

    df_imputed = pd.read_csv(input_path)
    df_ready = apply_feature_pipeline(df_imputed)

    candidate_predictors = (
        [f"{dim}_Z" for dim in ADVERSITY_DIMENSIONAL_MAP.keys()]  
        + ["Threat_adulthood_any_Z"] 
        + [f"Cumulative_risk_{s.lower()}_Z" for s in TEMPORAL_BLOCKS]
    )

    passed_predictors, qc_summary = filter_predictors_qc(df_ready, candidate_predictors)
    qc_summary.to_csv(DATA_PROCESSED_DIR / "qc_predictors_report.csv", index=False)

    output_path = DATA_PROCESSED_DIR / output_filename
    df_ready.to_csv(output_path, index=False)
    print(f"Analysis-ready dataset saved to: {output_path}")

    cc_path = DATA_PROCESSED_DIR / "complete_cases.csv"
    if cc_path.exists():
        df_cc = pd.read_csv(cc_path)
        df_cc_ready = apply_feature_pipeline(df_cc)
        df_cc_ready.to_csv(DATA_PROCESSED_DIR / "analysis_ready_complete_cases.csv", index=False)
        print(f" Analysis-ready complete cases saved")

    return df_ready

if __name__ == "__main__":
    run_feature_engineering()