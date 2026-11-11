"""
src/visualization.py
Table and figure generation: Forest Plot, Heatmap and others.
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from src.config import OUTPUTS_FIGS_DIR, OUTPUTS_TABLES_DIR, BRAIN_HUBS
from src.utils import print_title

# ==========================================================
# AESTHETIC SETTINGS
# ==========================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "ytick.color": "#333333",
    "xtick.color": "#333333"
})

# ==========================================================
# LABEL CLEANING & TRANSFORMATION
# ==========================================================

def clean_variables_label(var: str) -> str:
    v = var[:-2] if var.endswith("_Z") else var

    name_map = {
        "Threat_adulthood_any": "Adult Threat",
        "Cumulative_threat_adulthood": "Adult Threat",
        "Cumulative_deprivation_adulthood": "Adult Deprivation",
        "Cumulative_deprivation_recent": "Recent Deprivation",
        "Cumulative_threat_childhood": "Childhood Threat",
        "Cumulative_deprivation_childhood": "Childhood Deprivation",
        "Threat_Child": "Childhood Threat",
        "Depriv_Child": "Childhood Deprivation",
        "Thalamus_bilateral": "Bilateral Thalamus",
        "Ant_Insula_bilateral": "Bilateral Anterior Insula",
        "Post_Insula_bilateral": "Bilateral Posterior Insula",
        "Cingulum_Ant_bilateral": "Bilateral Anterior Cingulate (ACC)",
        "Hippocampus_bilateral": "Bilateral Hippocampus",
        "ParaHippocampal_bilateral": "Bilateral Parahippocampus",
        "Calcarine_bilateral": "Bilateral Calcarine (Control)",
        "Pain_Latent": "Latent Pain Severity",
        "CP_Intensity": "Pain Intensity",
        "n_regions": "Pain Spatial Extent"
    }

    return name_map.get(v, v.replace("_", " ").title())

# ==========================================================
# FIGURE 1: LOGISTIC REGRESSION FOREST PLOT
# ==========================================================

def plot_logistic_forest(save_fig: bool = True) -> plt.Figure:
    csv_path = OUTPUTS_TABLES_DIR / "logistic_CP_1_results.csv"
    if not csv_path.exists():
        print(f"File not found: {csv_path}. Execute src.modeling first.")
        return None

    df = pd.read_csv(csv_path)

    pred_df = (
        df.drop_duplicates(subset=["predictor"]).sort_values("or_predictor", ascending=False).copy()
    )

    pred_rows = []
    for _, r in pred_df.iterrows():
        pred_rows.append({
            "group": "Adversidad a lo largo de la vida",
            "label": clean_variables_label(r["predictor"]),
            "or": r["or_predictor"],
            "ci_low": r["or_ci_low_predictor"],
            "ci_high": r["or_ci_high_predictor"],
            "p_val": r["p_predictor"],
            "sig_fdr": r.get("sig_p_predictor", False)
        })

    roi_df = (
        df.drop_duplicates(subset=["roi"]).sort_values("or_roi", ascending=True).copy()
    )

    roi_rows = []
    for _, r in roi_df.iterrows():
        roi_rows.append({
            "group": "Estructura Cerebral (sMRI)",
            "label": clean_variables_label(r["roi"]),
            "or": r["or_roi"],
            "ci_low": r["or_ci_low_roi"],
            "ci_high": r["or_ci_high_roi"],
            "p_val": r["p_roi"],
            "sig_fdr": r.get("sig_p_roi", False)
        })

    plot_df = pd.DataFrame(pred_rows + roi_rows)

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    ax.axvline(1.0, color="#222222", linestyle="--", linewidth=1.2, alpha=0.7)
    y_positions = list(range(len(plot_df)))

    n_preds = len(pred_rows)
    adjusted_y = [y if i < n_preds else y + 1.3 for i, y in enumerate(y_positions)]

    for idx, (_, row) in enumerate(plot_df.iterrows()):
        y_pos = adjusted_y[idx]

        if row["sig_fdr"]:
            color = "#1f77b4"
            marker = "o"
        elif row["p_val"] < 0.05:
            color = "#e66101"
            marker = "D"
        else: 
            color = "#7f7f7f"
            marker = "s"

        xerr = [[row["or"] - row["ci_low"]], [row["ci_high"] - row["or"]]]
        ax.errorbar(
            row["or"], y_pos, xerr=xerr, fmt=marker, color=color, ecolor=color, linewidth=2, capsize=4,
            capthick=1.2, markersize=7
        )

        annotation = f"OR = {row["or"]:.2f} [{row["ci_low"]:.2f}, {row["ci_high"]:.2f}]"
        if row["sig_fdr"]:
            annotation += " *"
        ax.text(row["ci_high"] + 0.04, y_pos, annotation, va="center", fontsize=9.5, color="#222222")

    split_y = adjusted_y[n_preds - 1] + 0.65
    ax.axhline(split_y, color="#cccccc", linestyle=":", linewidth=1)

    ax.set_yticks(adjusted_y)
    ax.set_yticklabels(plot_df["label"], fontsize=10.5)

    ax.text(0.52, adjusted_y[-1] + 0.45, "BRAIN STRUCTURE (sMRI)", fontsize=9.5, 
    fontweight="bold", color="#555555")
    ax.text(0.52, adjusted_y[n_preds - 1] + 0.45, "ADVERSITY (LIFE-COURSE)", fontsize=9.5, 
    fontweight="bold", color="#555555")

    ax.set_xlabel("Odds Ratio (OR) [IC 95%]", fontsize=11, fontweight="bold")

    ax.set_xlim(0.5, 2.2)
    ax.set_ylim(-0.8, adjusted_y[-1] + 0.8)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_fig:
        OUTPUTS_FIGS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUTS_FIGS_DIR / "figure_1_forest_plot_logistic_odds_ratios.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Figure 1 saved successfully in: {out_path}")

    plt.close(fig)
    return fig

# ==========================================================
# FIGURE 2: PATHWAY DISSOCIATION PLOT
# ==========================================================

def plot_pathway_dissociation(pain_outcome: str = "Pain_Latent_Z", save_fig: bool = True) -> plt.Figure: 
    med_path = OUTPUTS_TABLES_DIR / f"mediation_{pain_outcome}_results.csv"
    if not med_path.exists():
        print(f"File not found: {med_path}. Execute src.modeling first.")
        return None
    
    df_med = pd.read_csv(med_path)

    direct_df = (
        df_med.drop_duplicates(subset=["predictor"]).sort_values("c_prime", ascending=True)
        .reset_index(drop=True).copy()
    )
    direct_df["label"] = direct_df["predictor"].apply(clean_variables_label)
    direct_df["ci_low"] = direct_df["c_prime"] - 1.96 * direct_df["se_c_prime"]
    direct_df["ci_high"] = direct_df["c_prime"] + 1.96 * direct_df["se_c_prime"]
    direct_df["is_sig"] = direct_df["p_c_prime"] < 0.05

    threat_df = df_med[df_med["predictor"].str.contains("Threat_adulthood", case=False)].copy()
    if threat_df.empty:
        threat_df = df_med.drop_duplicates(subset=["mediator"]).copy()

    threat_df = threat_df[~threat_df["mediator"].str.contains("Calcarine", case=False)]
    threat_df["label"] = threat_df["mediator"].apply(clean_variables_label)
    threat_df = threat_df.sort_values("ab", ascending=True).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    ax1.axvline(0, color="#222222", linestyle="--", linewidth=1.2, alpha=0.7)
    for idx, row in direct_df.iterrows():
        color = "#e66101" if row["is_sig"] else "#7f7f7f"
        marker = "D" if row["is_sig"] else "s"
        xerr = [[row["c_prime"] - row["ci_low"]], [row["ci_high"] - row["c_prime"]]]

        ax1.errorbar(
            row["c_prime"], idx, xerr=xerr, fmt=marker, color=color, ecolor=color, elinewidth=2, capsize=4,
            capthick=1.2, markersize=7
        )

        annotation = f"β = {row['c_prime']:.2f} [{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
        ax1.text(row["ci_high"] + 0.02, idx, annotation, va="center", fontsize=9.5, color="#222222")

    ax1.set_yticks(range(len(direct_df)))
    ax1.set_yticklabels(direct_df["label"], fontsize=10.5)
    ax1.set_xlabel("Direct Effect Standardized (c') [IC 95%]", fontsize=11, fontweight="bold")
    ax1.set_xlim(-0.35, 0.55)
    ax1.grid(axis="x", linestyle=":", alpha=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.axvline(0, color="#222222", linestyle="--", linewidth=1.2, alpha=0.7)
    for idx, row in threat_df.iterrows():
        is_sig_ab = row.get("sig_p_ab_predictor", row.get("sig_p_ab_stage", False))
        color = "#1f77b4" if is_sig_ab else "#7f7f7f"
        marker = "o" if is_sig_ab else "s"
        xerr = [[row["ab"] - row["ci_low_ab"]], [row["ci_high_ab"] - row["ab"]]]

        ax2.errorbar(
            row["ab"], idx, xerr=xerr, fmt=marker, color=color, ecolor=color, elinewidth=2, capsize=4,
            capthick=1.2, markersize=7
        )

        annotation = f"ab = {row["ab"]:.3f} [{row["ci_low_ab"]:.3f}, {row["ci_high_ab"]:.3f}]"
        if is_sig_ab:
            annotation += " *"
        ax2.text(row["ci_high_ab"] + 0.005, idx, annotation, va="center", fontsize=9.5, color="#555555")

    ax2.set_yticks(range(len(threat_df)))
    ax2.set_yticklabels(threat_df["label"], fontsize=10.5)
    ax2.set_xlabel("Indirect Mediated Effect (ab) [IC 95% Monte Carlo]", fontsize=12, fontweight="bold")
    ax2.set_xlim(-0.08, 0.12)
    ax2.grid(axis="x", linestyle=":", alpha=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_fig:
        OUTPUTS_FIGS_DIR.mkdir(parents=True, exist_ok=True)
        prefix = "figure_S2" if pain_outcome != "Pain_Latent_Z" else "figure_2"
        out_fig = OUTPUTS_FIGS_DIR / f"{prefix}_forest_plot_mediation_{pain_outcome}.png"
        fig.savefig(out_fig, dpi=300, bbox_inches="tight")
        print(f"Figure 2 saved successfully in: {out_fig}")

    plt.close(fig)
    return fig 

# ==========================================================
# FIGURE 3: LIFE-COURSE ADVERSITY 
# ==========================================================

def plot_sem_architecture(sem_df: pd.DataFrame, pain_outcome: str = "Pain_Latent_Z", save_fig: bool = True) -> plt.Figure:
    covars = ["demo_age_Z", "demo_sex", "TIV_cm3_Z", "clinical_diagnosis"]
    rois = [
        "Thalamus_bilateral_Z", "Ant_Insula_bilateral_Z", "Post_Insula_bilateral_Z", 
        "Cingulum_Ant_bilateral_Z", "Hippocampus_bilateral_Z", "ParaHippocampal_bilateral_Z"
    ]

    df = sem_df[(sem_df["op"] == "~") & (sem_df["p-value"].notna())].copy()
    df = df[~df["rval"].isin(covars)]

    if "ci_lower" not in df.columns or df["ci_lower"].isna().all():
        df["ci_lower"] = df["Estimate"] - 1.96 * df["Std. Err"]
        df["ci_upper"] = df["Estimate"] + 1.96 * df["Std. Err"]

    df["is_sig"] = df["sig_fdr"] if "sig_fdr" in df.columns else (df["p_fdr"] < 0.05)
    
    name_map = {
        "Threat_Child": "Childhood Threat",
        "Depriv_Child": "Childhood Deprivation",
        "Cumulative_threat_adulthood_Z": "Adult Threat",
        "Cumulative_deprivation_adulthood_Z": "Adult Deprivation",
        "Thalamus_bilateral_Z": "Thalamus",
        "Ant_Insula_bilateral_Z": "Ant. Insula",
        "Post_Insula_bilateral_Z": "Post. Insula",
        "Cingulum_Ant_bilateral_Z": "Ant. Cingulate (ACC)",
        "Hippocampus_bilateral_Z": "Hippocampus",
        "ParaHippocampal_bilateral_Z": "Parahippocampus",
        pain_outcome: "Latent Pain"
    }

    is_pain = df["lval"] == pain_outcome
    is_cascade = df["lval"].isin(["Cumulative_threat_adulthood_Z", "Cumulative_deprivation_adulthood_Z"]) & \
        df["rval"].isin(["Threat_Child", "Depriv_Child"])

    core_df = df[is_pain | is_cascade].copy()
    core_df["label"] = core_df.apply(
        lambda r: f"{name_map.get(r["rval"], r["rval"])} → {name_map.get(r["lval"], r["lval"])}",
        axis=1
    )

    core_df = core_df.sort_values("Estimate", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    ax.axvline(0, color="#222222", linestyle="--", linewidth=1.2, alpha=0.7)

    for idx, row in core_df.iterrows():
        is_fdr = row["is_sig"]
        is_nominal = (not is_fdr) and (row["p-value"] < 0.05)

        if is_fdr:
            color = "#1f77b4"
            marker = "o"
        elif is_nominal:
            color = "#e66101"
            marker = "D"
        else:
            color = "#7f7f7f"
            marker = "s"

        xerr = [[row["Estimate"] - row["ci_lower"]], [row["ci_upper"] - row["Estimate"]]]

        ax.errorbar(
            row["Estimate"], idx, xerr=xerr, fmt=marker, color=color, ecolor=color, elinewidth=2, 
            capsize=4, capthick=1.2, markersize=7
        )

        annotation = f"β = {row['Estimate']:.2f} [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]"
        if is_fdr:
            annotation += " *"
        ax.text(row["ci_upper"] + 0.03, idx, annotation, va="center", fontsize=9.5, color="#222222")
        
    ax.set_yticks(range(len(core_df)))
    ax.set_yticklabels(core_df["label"], fontsize=10)
    ax.set_xlabel("Standardized SEM Path Coefficient (β) [95% CI]", fontsize=10.5, fontweight="bold")
    ax.set_xlim(-0.65, 1.45)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_fig: 
        OUTPUTS_FIGS_DIR.mkdir(parents=True, exist_ok=True)
        out_fig = OUTPUTS_FIGS_DIR / f"figure_3_forest_plot_sem_network_{pain_outcome}.png"
        fig.savefig(out_fig, dpi=300, bbox_inches="tight")
        print(f"Figure 3 saved successfully in: {out_fig}")

    plt.close(fig)

    supp_df = df[df["lval"].isin(rois)].copy()
    if not supp_df.empty and save_fig:
        supp_df["label"] = supp_df.apply(
            lambda r: f"{name_map.get(r['rval'], r['rval'])} → {name_map.get(r['lval'], r['lval'])}", axis=1
        )
        supp_df = supp_df.sort_values("Estimate", ascending=True).reset_index(drop=True)
        
        fig_s3, ax_s3 = plt.subplots(figsize=(10, 9), dpi=300)
        ax_s3.axvline(0, color="#222222", linestyle="--", linewidth=1.2, alpha=0.7)

        for idx, row in supp_df.iterrows():
            xerr = [[row["Estimate"] - row["ci_lower"]], [row["ci_upper"] - row["Estimate"]]]
            ax_s3.errorbar(
                row["Estimate"], idx, xerr=xerr, fmt="s", color="#7f7f7f", ecolor="#7f7f7f",
                elinewidth=1.8, capsize=3.5, capthick=1.0, markersize=5.5
            )
            ann = f"β = {row['Estimate']:.2f} [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]"

            ax_s3.text(row["ci_upper"] + 0.02, idx, ann, va="center", fontsize=8.5, color="#444444")

        ax_s3.set_yticks(range(len(supp_df)))
        ax_s3.set_yticklabels(supp_df["label"], fontsize=9)
        ax_s3.set_xlabel("Standardized SEM Path Coefficient (β) [95% CI]", fontsize=11, fontweight="bold")
        ax_s3.set_title("Figure S3: SEM Afferent Macrostructure (Adversity → Brain ROIs)", fontsize=12, fontweight="bold", pad=15)
        ax_s3.set_xlim(-0.65, 0.65)
        ax_s3.grid(axis="x", linestyle=":", alpha=0.5)
        ax_s3.spines["top"].set_visible(False)
        ax_s3.spines["right"].set_visible(False)

        plt.tight_layout()
        out_s3 = OUTPUTS_FIGS_DIR / f"figure_S3_forest_plot_adversity_to_smri_{pain_outcome}.png"
        fig_s3.savefig(out_s3, dpi=300, bbox_inches="tight")
        print(f"Figure S3 saved successfully in: {out_s3}")
        plt.close(fig_s3)

    return fig

# ==========================================================
# MAIN PIPELINE
# ==========================================================
def run_visualization() -> None: 
    print_title("EXECUTING MAIN VISUALIZATION PIPELINE")

    print("\nGenerating Figure 1: Odds Ratios for Pain Presence (CP_1)")
    plot_logistic_forest(save_fig=True)

    print("\nGenerating Figure 2: Pathway Dissociation Forest Plot (Pain_Latent_Z)")
    plot_pathway_dissociation(pain_outcome="Pain_Latent_Z", save_fig=True)

    for outcome in ["CP_Intensity_Z", "n_regions_Z"]:
        med_path = OUTPUTS_TABLES_DIR / f"mediation_{outcome}_results.csv"
        if med_path.exists():
            print(f"Generating Supplementary Figure for {outcome}")
            plot_pathway_dissociation(pain_outcome=outcome, save_fig=True)

    print("\nGenerating Figure 3: SEM Architecture Forest Plot (Pain_Latent_Z)")
    sem_path = OUTPUTS_TABLES_DIR / "sem_pooled_parameters_Pain_Latent_Z.csv"
    if sem_path.exists():
        sem_df = pd.read_csv(sem_path)
        plot_sem_architecture(sem_df, pain_outcome="Pain_Latent_Z", save_fig=True)
    else: 
        print(f"Warning: File not found: {sem_path}. Execute src.sem_modeling first.")
    
    print_title("VISUALIZATION PIPELINE COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    run_visualization()