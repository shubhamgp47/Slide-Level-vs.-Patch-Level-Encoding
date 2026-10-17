# per_organ_reg_only.py
# Computes REG metric per organ + overall using REG2025_Inference.eval

import os
import json
import pandas as pd
from pathlib import Path
import sys

# ---------- Path setup: HistGen + REG2025_Inference ----------

CURRENT_DIR = Path.cwd().resolve()
p = CURRENT_DIR

# Find parent that has BOTH HistGen and REG2025_Inference
while p != p.parent and not ((p / "HistGen").exists() and (p / "REG2025_Inference").exists()):
    p = p.parent

PROJECT_ROOT = p                              # e.g. /home/.../iwi5204h
HISTGEN_ROOT = PROJECT_ROOT / "HistGen"       # e.g. /home/.../iwi5204h/HistGen

# 1) Make parent visible (so REG2025_Inference is importable)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2) Make HistGen visible (if you ever want to reuse modules from there)
if str(HISTGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(HISTGEN_ROOT))

print("PROJECT_ROOT:", PROJECT_ROOT)
print("HISTGEN_ROOT added to sys.path:", HISTGEN_ROOT)

from REG2025_Inference.eval import REG_Evaluator

# ---------- Proxy for REG (BioBERT / spaCy downloads) ----------

os.environ["http_proxy"] = "http://proxy:80"
os.environ["https_proxy"] = "http://proxy:80"

# ---------- Config ----------

EMBEDDING_MODEL = "dmis-lab/biobert-v1.1"
SPACY_MODEL = "en_core_sci_lg"

ORGANS = [
    "Breast",
    "Urinary bladder",
    "Uterine cervix",
    "Colon",
    "Lung",
    "Prostate",
    "Stomach",
    "Rectum",
]

# ---------- Helpers ----------

def infer_organ(report_text: str) -> str:
    if pd.isna(report_text):
        return "Unknown"
    text = str(report_text).strip()
    for organ in ORGANS:
        if text.lower().startswith(organ.lower()):
            return organ
    return "Unknown"

def load_reports_from_json(gt_json_path, pred_json_path) -> pd.DataFrame:
    """
    Load GT and prediction JSONs of the form:
    [
      {"id": "PIT_01_00002_01.tiff", "report": "..."},
      ...
    ]
    and return a unified DataFrame with:
    Case ID, Ground Truths, Generated Reports
    """
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt_list = json.load(f)
    with open(pred_json_path, "r", encoding="utf-8") as f:
        pred_list = json.load(f)

    gt_df = pd.DataFrame(gt_list).rename(columns={"id": "Case ID", "report": "Ground Truths"})
    pred_df = pd.DataFrame(pred_list).rename(columns={"id": "Case ID", "report": "Generated Reports"})

    df = gt_df.merge(pred_df, on="Case ID", how="inner")
    return df

# ---------- REG evaluation ----------

def evaluate_reg_per_organ(gt_json_path, pred_json_path, output_csv_path):
    df = load_reports_from_json(gt_json_path, pred_json_path)

    required_cols = ["Case ID", "Generated Reports", "Ground Truths"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in merged DataFrame: {missing}")

    df["Organ"] = df["Ground Truths"].apply(infer_organ)

    evaluator = REG_Evaluator(
        embedding_model=EMBEDDING_MODEL,
        spacy_model=SPACY_MODEL,
    )

    summary_rows = []

    # Overall REG score
    overall_pairs = list(zip(df["Ground Truths"].tolist(), df["Generated Reports"].tolist()))
    overall_score = evaluator.evaluate_dummy(overall_pairs)

    summary_rows.append({
        "Organ": "Overall",
        "Num Reports": len(df),
        "REG_AverageRanking": overall_score,
    })

    # Per-organ REG scores
    for organ, group in df.groupby("Organ"):
        if len(group) == 0:
            continue

        organ_pairs = list(zip(group["Ground Truths"].tolist(), group["Generated Reports"].tolist()))
        organ_score = evaluator.evaluate_dummy(organ_pairs)

        summary_rows.append({
            "Organ": organ,
            "Num Reports": len(group),
            "REG_AverageRanking": organ_score,
        })

    reg_df = pd.DataFrame(summary_rows)

    os.makedirs(Path(output_csv_path).parent, exist_ok=True)
    reg_df.to_csv(output_csv_path, index=False)

    print(reg_df)
    return reg_df

# ---------- Main ----------

if __name__ == "__main__":
    GT_PATH = "ground-truths.json"
    PRED_PATH = "pred.json"
    OUTPUT_CSV = "reg_metrics_per_organ_and_overall.csv"

    evaluate_reg_per_organ(GT_PATH, PRED_PATH, OUTPUT_CSV)