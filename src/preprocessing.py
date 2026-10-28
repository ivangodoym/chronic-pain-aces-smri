"""
src/preprocessing.py
Data handling: quality control, data cleaning, and multiple imputation
"""
from src.config import (
    HEADER_ROW, DIAGNOSES_TO_KEEP, SHEET_NAME, PAIN_REGIONS_GROUPS, 
    ADVERSITY_ITEMS, MAX_ITEM_MISSING, ADVERSITY_DOMAINS, TEMPORAL_BLOCKS, N_IMPUTATIONS,
    RANDOM_STATE, PAIN_REGIONS, DATA_PROCESSED_DIR
)
from src.utils import find_input_file, require_columns, get_brain_regions, print_title, format_pct
import pandas as pd 
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from pathlib import Path

# ==========================================================
# DATA PREPARATION
# ==========================================================

def load_and_filter() -> pd.DataFrame:
    input_file = find_input_file()
    df = pd.read_excel(input_file, header=HEADER_ROW, sheet_name=SHEET_NAME)
    df.columns = df.columns.str.strip().str.replace(r"\s+","_", regex=True)
    df.replace(-99, np.nan, inplace=True)

    required = (
        ["ID", "demo_age", "demo_sex", "clinical_diagnosis", "CP_1"]
        + ADVERSITY_ITEMS
        + [item for cols in PAIN_REGIONS_GROUPS.values() for item in cols]
        + ["CP_Frequency", "CP_Intensity"]
    )
    require_columns(df,required)

    brain_regions = get_brain_regions(df.columns)
    if not brain_regions: 
        raise ValueError("Neuroimaging columns not found with the expected prefixes/suffixes")

    n_initial = len(df)
    cp_missing = df["CP_1"].isna().sum()
    df = df.dropna(subset=["CP_1"]).copy()
    df = df[df["clinical_diagnosis"].isin(DIAGNOSES_TO_KEEP)].copy()
    df["clinical_diagnosis"] = df["clinical_diagnosis"].map({"CN": 0, "AD": 1})

    print_title("DATA FILTERS")
    print(f"Input file: {input_file}")
    print(f"Initial rows: {n_initial}")
    print(f"Rows dropped because CP_1 is missing: {cp_missing}")
    print(f"Rows kept after CN/AD filter: {len(df)}")
    print(df["clinical_diagnosis"].value_counts(dropna=False).to_string())

    return df

def report_ace_missingness(df: pd.DataFrame) -> None:
    print_title("ACE MISSINGNESS BEFORE PRIMARY-ANALYSIS EXCLUSION")
    item_missing = df[ADVERSITY_ITEMS].isna().sum().sort_values(ascending=False)
    print("Missingness by item")
    for item, n_missing in item_missing.items():
        print(f"{item}: {n_missing}/{len(df)} ({format_pct(n_missing/len(df))})")

    row_missing = df[ADVERSITY_ITEMS].isna().sum(axis=1)
    print(row_missing.value_counts().sort_index().to_string())
    print(
        f"\n Subjects with any item missing: {(row_missing > 0).sum()}/{len(df)}"
        f"({format_pct((row_missing > 0).mean())})"
    )

def exclude_large_missingness(df:pd.DataFrame, max_missing: float = MAX_ITEM_MISSING,) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    exclude_mask = pd.Series(False, index = df.index)

    for items in ADVERSITY_DOMAINS.values():
        missing_prop_domain = df[items].isna().mean(axis=1)
        exclude_mask = exclude_mask | (missing_prop_domain > max_missing)

    for domain_names in TEMPORAL_BLOCKS.values():
        stage_items = [item for d in domain_names for item in ADVERSITY_DOMAINS[d]]
        missing_prop_stage = df[stage_items].isna().mean(axis=1)
        exclude_mask = exclude_mask | (missing_prop_stage > max_missing)

    df["item_missing_n"] = df[ADVERSITY_ITEMS].isna().sum(axis=1)
    df["item_missing_prop"] = df["item_missing_n"] / len(ADVERSITY_ITEMS)
    exclude_mask = exclude_mask | (df["item_missing_prop"] > max_missing)

    excluded = df.loc[exclude_mask].copy()
    kept = df.loc[~exclude_mask].copy()

    return kept, excluded

# ==========================================================
# IMPUTATION
# ==========================================================

def multiple_imputation(df: pd.DataFrame, m: int = N_IMPUTATIONS) -> list[pd.DataFrame]:
    features = ADVERSITY_ITEMS + ["demo_age", "demo_sex", "TIV_cm3", "CP_1", "clinical_diagnosis", "CP_Intensity", "CP_Frequency"] + PAIN_REGIONS
    output = []

    for i in range (m):
        imputer = IterativeImputer(
            estimator = BayesianRidge(),
            max_iter = 10,
            initial_strategy="most_frequent",
            sample_posterior=True,
            random_state=RANDOM_STATE + i
        )
        imputed = pd.DataFrame(imputer.fit_transform(df[features]), columns=features, index=df.index)
        
        binary_features = ADVERSITY_ITEMS + ["clinical_diagnosis", "CP_1"] + PAIN_REGIONS
        for cols in binary_features:
            imputed[cols] = imputed[cols].round().clip(0, 1)

        if "demo_sex" in df.columns:
            imputed["demo_sex"] = imputed["demo_sex"].round().clip(1, 2)

        imputed.loc[imputed["CP_1"] == 0, ["CP_Frequency", "CP_Intensity"]] = np.nan
        imputed["CP_Frequency"] = imputed["CP_Frequency"].clip(1, 4)
        imputed["CP_Intensity"] = imputed["CP_Intensity"].clip(0, 10)

        results = df.copy()
        for cols in features:
            results[cols] = imputed[cols]

        output.append(results)

    print(f"\n Created {len(output)} multiple imputations")
    return output 

def save_imputations(
    imputations: list[pd.DataFrame], output_dir: Path = DATA_PROCESSED_DIR,
    filename: str = "imputed_datasets.csv") -> Path: 
    output_dir.mkdir(parents=True, exist_ok=True)

    stacked_list =[]
    for i, df_imp in enumerate(imputations, start=1):
        df_copy = df_imp.copy()
        df_copy["_imputation"] = i
        stacked_list.append(df_copy)
        
    combined = pd.concat(stacked_list, ignore_index=True)
    output_path = output_dir / filename 
    combined.to_csv(output_path.with_suffix(".csv"), index=False)

    print(f" Imputations saved succesfully to: {output_path}")

    return output_path

# ==========================================================
# MAIN PIPELINE
# ==========================================================

def run_preprocessing() -> list[pd.DataFrame]:
    print_title("RUNNING PREPROCESSING PIPELINE")
    df = load_and_filter()
    report_ace_missingness(df)
    kept, excluded = exclude_large_missingness(df)
    
    excluded.to_csv(DATA_PROCESSED_DIR/"excluded_subjects.csv", index=False)

    brain_cols = get_brain_regions(df.columns)
    cc_features = ["demo_age", "demo_sex", "clinical_diagnosis", "TIV_cm3", "CP_1"] + ADVERSITY_ITEMS + PAIN_REGIONS + brain_cols

    df_cc = kept.dropna(subset=[c for c in cc_features if c in kept.columns]).copy()
    df_cc.to_csv(DATA_PROCESSED_DIR/"complete_cases.csv", index=False)
    print(f"Complete cases saved: {len(df_cc)} subjects with full data")

    imputations = multiple_imputation(kept)
    save_imputations(imputations)

    return imputations

if __name__ == "__main__":
    run_preprocessing()