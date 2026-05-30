import json
import os
import random
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
LABELED_DATA_FILE = "labeled_main_dataset.json"
ERROR_RATES_FILE = "value_error_rates.json"
OUTPUT_CSV = "group_level_prevalences.csv"

NUM_GROUPS = 50  # 10,303 pairs / 50 ≈ 206 pairs per group
RANDOM_SEED = 42

FEATURES = [
    'Self-direction: thought', 'Self-direction: action', 'Stimulation', 'Hedonism', 'Achievement',
    'Power: dominance', 'Power: resources', 'Face', 'Security: personal', 'Security: societal', 'Tradition',
    'Conformity: rules', 'Conformity: interpersonal', 'Humility', 'Benevolence: caring',
    'Benevolence: dependability', 'Universalism: concern', 'Universalism: nature', 'Universalism: tolerance',
    'Universalism: objectivity'
]

# ==========================================
# 1. LOAD DATA
# ==========================================
print("Loading files...")
with open(LABELED_DATA_FILE, "r", encoding="utf-8") as f:
    labeled_data = json.load(f)

with open(ERROR_RATES_FILE, "r", encoding="utf-8") as f:
    error_rates = json.load(f)

# Shuffle data to ensure random distribution across groups
random.seed(RANDOM_SEED)
random.shuffle(labeled_data)

# ==========================================
# 2. DEFINE GROUPS
# ==========================================
print(f"Dividing {len(labeled_data)} pairs into {NUM_GROUPS} random groups...")
groups = np.array_split(labeled_data, NUM_GROUPS)

# ==========================================
# 3. QUANTIFICATION PIPELINE (ACC FORMULA)
# ==========================================
def run_acc_correction(raw_prevalence, tpr, fpr):
    """Applies the Adjusted Classify and Count formula with standard clipping."""
    denominator = tpr - fpr
    if denominator == 0:
        return raw_prevalence  # Avoid division by zero if class is completely uncalibrated
    
    corrected = (raw_prevalence - fpr) / denominator
    return float(np.clip(corrected, 0.0, 1.0))  # Standard practice: clip to [0, 1] range

group_rows = []

print("Running quantification across all groups and classes...")
for group_idx, group in enumerate(groups):
    group_size = len(group)
    row_data = {
        "group_id": group_idx + 1,
        "group_size": group_size
    }
    
    # Process each of the 20 value categories
    for feat_idx, feature in enumerate(FEATURES):
        tpr = error_rates[feature]["tpr"]
        fpr = error_rates[feature]["fpr"]
        
        # 1. Extract raw predictions (binary 1s or 0s) for this group
        op_preds = [item["op_context"]["predicted_binary"][feat_idx] for item in group]
        win_preds = [item["winning_rebuttal"]["predicted_binary"][feat_idx] for item in group]
        lose_preds = [item["losing_rebuttal"]["predicted_binary"][feat_idx] for item in group]
        
        # 2. Calculate uncorrected (Classify and Count) raw prevalences
        op_raw_prev = sum(op_preds) / group_size
        win_raw_prev = sum(win_preds) / group_size
        lose_raw_prev = sum(lose_preds) / group_size
        
        # 3. Apply the ACC corrections
        op_corr_prev = run_acc_correction(op_raw_prev, tpr, fpr)
        win_corr_prev = run_acc_correction(win_raw_prev, tpr, fpr)
        lose_corr_prev = run_acc_correction(lose_raw_prev, tpr, fpr)
        
        # 4. Store values in our row dictionary
        safe_feature_name = feature.lower().replace(":", "").replace(" ", "_").replace("-", "_")
        
        row_data[f"op_{safe_feature_name}_raw"] = op_raw_prev
        row_data[f"op_{safe_feature_name}_corrected"] = op_corr_prev
        
        row_data[f"win_{safe_feature_name}_raw"] = win_raw_prev
        row_data[f"win_{safe_feature_name}_corrected"] = win_corr_prev
        
        row_data[f"lose_{safe_feature_name}_raw"] = lose_raw_prev
        row_data[f"lose_{safe_feature_name}_corrected"] = lose_corr_prev
        
    group_rows.append(row_data)

# ==========================================
# 4. SAVE GROUP-LEVEL CSV
# ==========================================
df_output = pd.DataFrame(group_rows)
df_output.to_csv(OUTPUT_CSV, index=False)
print(f"Quantification completed. Output saved to: {OUTPUT_CSV}")