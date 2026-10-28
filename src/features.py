"""
src/features.py
Feature preparation: Create ACE-related, pain phenotypes and structural neuroimaging features
"""
from src.config import (
    ADVERSITY_DOMAINS, ADVERSITY_ITEMS, TEMPORAL_BLOCKS,
    ADVERSITY_DIMENSIONAL_MAP, DATA_PROCESSED_DIR, PAIN_REGIONS
    ) 
from src.utils import zscore_series, get_brain_regions, print_title
import pandas as pd 
import numpy as np 

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

    return df

# ==========================================================
# PAIN VARIABLES
# ==========================================================

def prepare_pain_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["n_regions"] = df[PAIN_REGIONS].sum(axis=1, min_count=len(PAIN_REGIONS))
    
    df.loc[df["n_regions"].notna() & ((df["n_regions"] < 0) | (df["n_regions"] > 17)), "n_regions"] = np.nan
    df.loc[df["CP_1"] == 0, "n_regions"] = 0

    df["CP_Frequency"] = df["CP_Frequency"].map({2:1, 3:2, 4:3, 1:4})
    df.loc[df["CP_Intensity"].notna() & ((df["CP_Intensity"] < 0) | (df["CP_Intensity"] > 10)), "CP_Intensity"] = np.nan 

    return df

# ==========================================================
# BRAIN STRUCTURE VARIABLES
# ==========================================================

def prepare_structural_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Thalamus_L" in df.columns and "Thalamus_R" in df.columns:
        df["Thalamus_bilateral"] = df["Thalamus_L"] + df["Thalamus_R"]

    if "Ant_Insula_L" in df.columns and "Ant_Insula_R" in df.columns:
        df["Ant_Insula_bilateral"] = df["Ant_Insula_L"] + df["Ant_Insula_R"]

    if "Post_Insula_L" in df.columns and "Post_Insula_R" in df.columns:
        df["Post_Insula_bilateral"] = df["Post_Insula_L"] + df["Post_Insula_R"]

    return df 

# ==========================================================
# STANDARDIZATION
# ==========================================================

def standardize_variables(df: pd.DataFrame, columns_to_zscore: list[str] = None) -> pd.DataFrame:
    df = df.copy()

    if columns_to_zscore is None:
        brain_cols = get_brain_regions(df.columns)
        bilateral_cols = [
            c for c in ["Thalamus_bilateral", "Ant_Insula_bilateral", "Post_Insula_bilateral"]
            if c in df.columns]
        
        adversity_cols = (
            list(ADVERSITY_DOMAINS.keys()) 
            + 
            list(ADVERSITY_DIMENSIONAL_MAP.keys()) + [f"Cumulative_risk_{s.lower()}" for s in TEMPORAL_BLOCKS]
            )
        pain_cols = [c for c in ["CP_Intensity", "n_regions"] if c in df.columns]
        covar_cols = [c for c in ["TIV_cm3", "demo_age"] if c in df.columns]

        columns_to_zscore = brain_cols + bilateral_cols + adversity_cols + pain_cols + covar_cols
        
    if "_imputation" in df.columns:
        for col in columns_to_zscore:
            if col in df.columns: 
                df[f"{col}_Z"] = df.groupby("_imputation")[col].transform(zscore_series)
    else: 
        for col in columns_to_zscore:
            if col in df.columns:
                df[f"{col}_Z"] = zscore_series(df[col])

    return df

# ==========================================================
# PREDICTOR QUALITY CONTROL
# ==========================================================

def filter_predictors_qc(
    df: pd.DataFrame, candidate_predictors: list[str],
    max_skew: float = 2.5, max_floor_pct: float = 0.8) -> tuple[list[str], pd.DataFrame]:

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

    candidate_predictors = [
        f"{dim}_Z" for dim in ADVERSITY_DIMENSIONAL_MAP.keys()] + [f"Cumulative_risk_{s.lower()}_Z"
        for s in TEMPORAL_BLOCKS
    ]

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