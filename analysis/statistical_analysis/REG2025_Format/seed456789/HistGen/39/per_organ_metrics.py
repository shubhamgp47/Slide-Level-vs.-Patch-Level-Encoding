# per_organ_nlg_only.py
# Computes BLEU / METEOR / ROUGE-L per organ + overall using HistGen.metrics

import os
import json
import pandas as pd
from pathlib import Path
import sys

# ---------- Path setup: HistGen only ----------

CURRENT_DIR = Path.cwd().resolve()
p = CURRENT_DIR

# Find parent that has HistGen
while p != p.parent and not (p / "HistGen").exists():
    p = p.parent

PROJECT_ROOT = p
HISTGEN_ROOT = PROJECT_ROOT / "HistGen"

if str(HISTGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(HISTGEN_ROOT))

print("PROJECT_ROOT:", PROJECT_ROOT)
print("HISTGEN_ROOT added to sys.path:", HISTGEN_ROOT)

from modules.metrics import compute_scores  # uses local pycocoevalcap + METEOR

# ---------- Organ logic ----------

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

def infer_organ(report_text):
    if pd.isna(report_text):
        return "Unknown"
    text = str(report_text).strip()
    for organ in ORGANS:
        if text.lower().startswith(organ.lower()):
            return organ
    return "Unknown"

def build_score_dict(texts):
    return {i: [t] for i, t in enumerate(texts)}

# ---------- JSON loading ----------

def load_reports_from_json(gt_json_path, pred_json_path):
    """
    Load GT and prediction JSONs of the form:
    [
      {"id": "PIT_01_00002_01.tiff", "report": "..."},
      ...
    ]
    Return DataFrame with: Case ID, Ground Truths, Generated Reports
    """
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt_list = json.load(f)
    with open(pred_json_path, "r", encoding="utf-8") as f:
        pred_list = json.load(f)

    gt_df = pd.DataFrame(gt_list).rename(columns={"id": "Case ID", "report": "Ground Truths"})
    pred_df = pd.DataFrame(pred_list).rename(columns={"id": "Case ID", "report": "Generated Reports"})

    df = gt_df.merge(pred_df, on="Case ID", how="inner")
    return df

# ---------- Evaluation ----------

def evaluate_nlg_metrics_per_organ(gt_json_path, pred_json_path, output_csv_path):
    df = load_reports_from_json(gt_json_path, pred_json_path)

    required_cols = ["Case ID", "Generated Reports", "Ground Truths"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in merged DataFrame: {missing}")

    df["Organ"] = df["Ground Truths"].apply(infer_organ)

    summary_rows = []

    # Overall NLG metrics
    overall_gts = build_score_dict(df["Ground Truths"].fillna("").tolist())
    overall_res = build_score_dict(df["Generated Reports"].fillna("").tolist())
    overall_scores = compute_scores(overall_gts, overall_res)

    overall_row = {"Organ": "Overall", "Num Reports": len(df)}
    overall_row.update(overall_scores)
    summary_rows.append(overall_row)

    # Per-organ NLG metrics
    for organ, group in df.groupby("Organ"):
        if len(group) == 0:
            continue

        gts = build_score_dict(group["Ground Truths"].fillna("").tolist())
        res = build_score_dict(group["Generated Reports"].fillna("").tolist())
        scores = compute_scores(gts, res)

        row = {
            "Organ": organ,
            "Num Reports": len(group),
            **scores,
        }
        summary_rows.append(row)

    metrics_df = pd.DataFrame(summary_rows)

    os.makedirs(Path(output_csv_path).parent, exist_ok=True)
    metrics_df.to_csv(output_csv_path, index=False)

    print(metrics_df)
    return metrics_df

# ---------- Main ----------

if __name__ == "__main__":
    GT_PATH = "ground-truths.json"
    PRED_PATH = "pred.json"
    OUTPUT_CSV = "nlg_metrics_per_organ_and_overall.csv"

    evaluate_nlg_metrics_per_organ(GT_PATH, PRED_PATH, OUTPUT_CSV)