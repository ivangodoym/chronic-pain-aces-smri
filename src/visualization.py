"""
src/visualization.py
Table and figure generation: Forest Plot, Heatmap and others.
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from src.config import OUTPUTS_FIGS_DIR, OUTPUTS_TABLES_DIR

# ==========================================================
# LABEL CLEANING & TRANSFORMATION
# ==========================================================

def clean_variables_label(var: str) -> str:
    v = var[:-2] if var.endswith("_Z") else var

    if v.startswith("Cumulative_"):
        parts = v.split("_")
        if len(parts) == 3:
            type = "Cumulative Risk" if parts[1] == "risk" else parts[1].capitalize()
            stage = parts[2].capitalize()
            return f"{type} ({stage})"

    if "_bilateral" in v:
        base = v.replace("_bilateral", "")
        base = base.replace("Ant_Insula", "Anterior Insula").replace("Post_Insula", "Posterior Insula")
        return f"Bilateral {base}"

    pain_map = {
        "CP_Intensity": "Pain Intensity",
        "CP_Frequency": "Pain Frequency",
        "n_regions": "Pain Spatial Extent"
        }

    return pain_map.get(v, v.replace("_", " ").title())

def transform_heatmap_data(df: pd.DataFrame, metrics: list[str] = None) -> dict[str, pd.DataFrame]:
    if metrics is None:
        metrics = [
            "a", "p_a", "p_a_fdr_stage",
            "b", "p_b", "p_b_fdr_stage",
            "ab", "p_ab", "p_ab_fdr_stage"]

    matrices = {
        col: df.pivot(index="predictor", columns="mediator", values=col).rename(
            index=clean_variables_label, columns=clean_variables_label
        )
        for col in metrics
    }

    return matrices 

# ==========================================================
# FOREST PLOT
# ==========================================================

def prepare_forest_data(results_df: pd.DataFrame, outcome: str, 
fdr_col: str = "sig_p_ab_stage") -> pd.DataFrame:
    df = results_df[results_df["outcome"] == outcome].copy()

    df["label"] = (
        df["predictor"].apply(clean_variables_label) 
        + " → " + df["mediator"].apply(clean_variables_label)
        )
    df = df.sort_values("ab", ascending=True).reset_index(drop=True)

    df["significance"] = "Not significant"
    df.loc[df["p_ab"] < 0.05, "significance"] = "Nominal (p < 0.05)"

    target_fdr = fdr_col if fdr_col in df.columns else ("sig_fdr" if "sig_fdr" in df.columns else None)
    if target_fdr:
        df.loc[df[target_fdr] == True, "significance"] = "FDR significant (q < 0.05)"

    return df 


def plot_mediation_forest(plot_df: pd.DataFrame, outcome: str, save_fig: bool = True) -> plt.Figure:
    palette = {
        "FDR significant (q < 0.05)": "#2b5c8f",
        "Nominal (p < 0.05)": "#e66101",
        "Not significant": "#999999"  
    }

    fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.4)), dpi=300)

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)

    for category, color in palette.items():
        sub_df = plot_df[plot_df["significance"] == category] 
        if sub_df.empty:
            continue

        xerr = np.array([
            sub_df["ab"] - sub_df["ci_low_ab"],
            sub_df["ci_high_ab"] - sub_df["ab"]
        ])

        ax.errorbar(
            x=sub_df["ab"],
            y=sub_df.index,
            xerr=xerr,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.8,
            capsize=3.5,
            capthick=1.2,
            markersize=6,
            label=category
        )

    ax.set_yticks(plot_df.index)
    ax.set_yticklabels(plot_df["label"], fontsize=10)
    ax.set_ylim(-0.8, len(plot_df) - 0.2)

    clean_outcome = clean_variables_label(outcome)
    ax.set_xlabel("Standardized Indirect Effect (ab) [95% Monte Carlo CI]", fontsize=11)
    ax.set_title(f"Simple Mediation: Indirect Effects on {clean_outcome}", fontsize= 12, pad=12, fontweight="bold")

    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, loc="best", fontsize=9)

    plt.tight_layout()

    if save_fig:
        OUTPUTS_FIGS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUTS_FIGS_DIR / f"forest_plot_{outcome}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

        print(f"Forest plot saved to: {output_path}")

    plt.close(fig)

    return fig

# ==========================================================
# HEATMAP
# ==========================================================

def make_annot_matrix(coef_df: pd.DataFrame, pval_df: pd.DataFrame) -> pd.DataFrame:
    annot = coef_df.copy().astype(str)
    for r in coef_df.index:
        for c in coef_df.columns:
            beta = coef_df.loc[r, c]
            p = pval_df.loc[r, c]

            stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            annot.loc[r, c] = f"{beta:.2f}{stars}"
    
    return annot 

def plot_mediation_heatmap(
    results_df: pd.DataFrame, outcome: str, 
    save_fig: bool = True, use_fdr: bool = False) -> plt.Figure:
    sub_df = results_df[results_df["outcome"] == outcome]
    matrices = transform_heatmap_data(sub_df)

    p_a_key = "p_a_fdr_stage" if use_fdr and "p_a_fdr_stage" in matrices else "p_a"
    p_b_key = "p_b_fdr_stage" if use_fdr and "p_b_fdr_stage" in matrices else "p_b"
    p_ab_key = "p_ab_fdr_stage" if use_fdr and "p_ab_fdr_stage" in matrices else "p_ab"

    mat_a = matrices["a"]
    mat_b = matrices["b"]
    mat_ab = matrices["ab"]

    annot_a = make_annot_matrix(mat_a, matrices[p_a_key])
    annot_b = make_annot_matrix(mat_b, matrices[p_b_key])
    annot_ab = make_annot_matrix(mat_ab, matrices[p_ab_key])

    clean_outcome = clean_variables_label(outcome)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, max(5, len(mat_a) * 0.7)), sharey=True, dpi=300)

    sns.heatmap(
        mat_a, annot=annot_a, fmt="", cmap="vlag", center=0, linewidths=0.8, linecolor="white",
        cbar_kws={"label": "Standardized Beta (a)"}, ax=ax1
    )

    ax1.set_title("Path a: Adversity → Brain Structure", fontsize=12, fontweight="bold", pad= 12)
    ax1.set_xlabel("")
    ax1.set_ylabel("")

    sns.heatmap(
        mat_b, annot=annot_b, fmt="", cmap="vlag", center=0, linewidths=0.8, linecolor="white",
        cbar_kws={"label": "Standardized Beta (b)"}, ax=ax2
    )
    ax2.set_title(f"Path b: Brain Structure → {clean_outcome}", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("")
    ax2.set_ylabel("")

    sns.heatmap(
        mat_ab, annot=annot_ab, fmt="", cmap="vlag", center=0, linewidths=0.8, linecolor="white",
        cbar_kws={"label": "Indirect Effect (ab)"}, ax=ax3
    )
    ax3.set_title("Indirect Effect (ab): Mediated Pathway", fontsize=12, fontweight="bold", pad=12)
    ax3.set_xlabel("")
    ax3.set_ylabel("")

    for ax in (ax1, ax2, ax3):
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=10)
    ax1.tick_params(axis="y", rotation=0)

    plt.tight_layout()

    if save_fig:
        OUTPUTS_FIGS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUTS_FIGS_DIR / f"heatmap_{outcome}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        
        print(f"Heatmap saved to: {output_path}")
    
    plt.close(fig)

    return fig

# ==========================================================
# MAIN PIPELINE
# ==========================================================

if __name__ == "__main__":
    outcomes = ["CP_Intensity_Z", "n_regions_Z"]

    for outcome_var in outcomes:
        results_path = OUTPUTS_TABLES_DIR / f"mediation_{outcome_var}_results.csv"
        print(f"\nGenerating figures for: {outcome_var}.")
        if results_path.exists():
            results_df = pd.read_csv(results_path)

            print("\n Generating Forest Plot...")
            plot_df = prepare_forest_data(results_df, outcome_var)
            plot_mediation_forest(plot_df, outcome_var)

            print("\nGenerating Heatmap...")
            plot_mediation_heatmap(results_df, outcome_var)

        else: 
            print(f"No results found at {results_path}. First run src/modeling.py")