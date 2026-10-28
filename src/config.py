"""
src/config.py
Global settings: routes and variable definitions
"""
import json
from pathlib import Path

# ==========================================================
# PROJECT ROUTES
# ==========================================================

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"

CLINICAL_AND_STRUCTURAL_DATA_PATH = DATA_RAW_DIR / "ACEs_Neuroimages.xlsx"
ADVERSITY_DOMAINS_PATH = SRC_DIR / "adversity_domains.json"

OUTPUTS_DIR = ROOT_DIR / "outputs"
OUTPUTS_FIGS_DIR = OUTPUTS_DIR / "figures"
OUTPUTS_TABLES_DIR = OUTPUTS_DIR / "tables"

with open(ADVERSITY_DOMAINS_PATH, "r", encoding="utf-8") as f:
    ADVERSITY_DOMAINS = json.load(f)

# ==========================================================
# PARAMETERS
# ==========================================================

SHEET_NAME = "Sheet1"
ID_COL = "ID"
HEADER_ROW = 0
DIAGNOSES_TO_KEEP = ["CN", "AD"]
N_IMPUTATIONS = 10
RANDOM_STATE = 42
FDR_ALPHA = 0.05
MAX_ITEM_MISSING = 0.20

# ==========================================================
# VARIABLES & COVARIABLES DEFINITION
# ==========================================================

COVARIATES = ["demo_age", "demo_sex", "TIV_cm3"]

PAIN_VARS = {
    "intensity": "CP_Intensity",
    "frequency": "CP_Frequency",
    "spatial_extent": "n_regions"
}

PAIN_REGIONS_GROUPS = {
    "axial": ["CP_Neck_1","CP_UpperBack_1","CP_LowerBack_1","CP_Chest_1","CP_Abdomen_1"],
    "upper": ["CP_Shoulders_1","CP_Arms_1","CP_Elbows_1","CP_wrists_1","CP_Hands_1"],
    "lower": ["CP_Legs_1","CP_Knees_1","CP_Ankles_1","CP_Feet_1"],
    "head": ["CP_Head_1","CP_Face_1"],
    "other": ["CP_Other_1"],
}

PAIN_REGIONS = [c for cols in PAIN_REGIONS_GROUPS.values() for c in cols]

ADVERSITY_DIMENSIONAL_MAP = {
    "Cumulative_threat_childhood": [
        "Physical_threat_childhood",
        "Psychological_threat_childhood"
    ],
    "Cumulative_deprivation_childhood": [
        "Asset_deprivation_childhood",
        "Basic_needs_childhood"
    ],
    "Cumulative_threat_adulthood": [
        "Physical_threat_adulthood",
        "Psychological_threat_adulthood"
    ],
    "Cumulative_deprivation_adulthood": [
        "Asset_deprivation_adulthood",
        "Basic_needs_adulthood"
    ],
    "Cumulative_threat_recent": [
        "Physical_threat_rec",
        "Psychological_threat_rec"
    ],
    "Cumulative_deprivation_recent": [
        "Asset_deprivation_rec",
        "Basic_needs_rec"
    ]
}

ADVERSITY_ITEMS = [c for cols in ADVERSITY_DOMAINS.values() for c in cols]
TEMPORAL_BLOCKS = {}
for dim_name, domain_list in ADVERSITY_DIMENSIONAL_MAP.items():
    stage = dim_name.split("_")[-1]
    TEMPORAL_BLOCKS.setdefault(stage,[]).extend(domain_list)

CLINICAL_DIMENSIONAL_VARS = list(ADVERSITY_DIMENSIONAL_MAP.keys())

STRUCTURAL_ROI_SUFFIXES = ("_L", "_R")
STRUCTURAL_ROI_PREFIXES = ("Vermis_",)