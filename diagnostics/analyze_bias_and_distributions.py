import json
import os
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
LABELED_DATA_FILE = "labeled_main_dataset.json"
GOLD_STANDARD_FILE = "Archive/complete_dataset_test.json"
ERROR_RATES_FILE = "value_error_rates.json"
OUTPUT_CSV = "model_bias_audit.csv"

FEATURES = [
    'Self-direction: thought', 'Self-direction: action', 'Stimulation', 'Hedonism', 'Achievement',
    'Power: dominance', 'Power: resources', 'Face', 'Security: personal', 'Security: societal', 'Tradition',
    'Conformity: rules', 'Conformity: interpersonal', 'Humility', 'Benevolence: caring',
    'Benevolence: dependability', 'Universalism: concern', 'Universalism: nature', 'Universalism: tolerance',
    'Universalism: objectivity'
]

# ==========================================
# 1. LOAD DATASETS
# ==========================================
print("Loading datasets...")
if not os.path.exists(LABELED_DATA_FILE):
    raise FileNotFoundError(f"Missing labeled main dataset: {LABELED_DATA_FILE}")
if not os.path.exists(GOLD_STANDARD_FILE):
    raise FileNotFoundError(f"Missing gold standard dataset: {GOLD_STANDARD_FILE}")
if not os.path.exists(ERROR_RATES_FILE):
    raise FileNotFoundError(f"Missing error rates file: {ERROR_RATES_FILE}")

with open(LABELED_DATA_FILE, "r", encoding="utf-8") as f:
    main_dataset = json.load(f)

with open(GOLD_STANDARD_FILE, "r", encoding="utf-8") as f:
    gold_dataset = json.load(f)

with open(ERROR_RATES_FILE, "r", encoding="utf-8") as f:
    error_rates = json.load(f)

# ==========================================
# 2. CALCULATE HUMAN GOLD STANDARD DISTRIBUTION
# ==========================================
print("Calculating Human Gold Standard distributions...")
gold_counts = {feat: 0 for feat in FEATURES}
total_gold_texts = len(gold_dataset)

for item in gold_dataset:
    for val in item.get("values", []):
        if val in gold_counts:
            gold_counts[val] += 1

gold_prevalence = {feat: (count / total_gold_texts) for feat, count in gold_counts.items()}

# ==========================================
# 3. CALCULATE RAW MODEL INFERENCE DISTRIBUTION
# ==========================================
print("Calculating Raw Model Inference distributions...")
raw_counts = {feat: 0 for feat in FEATURES}
# Each pair contains 3 texts: OP, Winning Rebuttal, Losing Rebuttal
total_main_texts = len(main_dataset) * 3 

for item in main_dataset:
    for feat_idx, feature in enumerate(FEATURES):
        raw_counts[feature] += item["op_context"]["predicted_binary"][feat_idx]
        raw_counts[feature] += item["winning_rebuttal"]["predicted_binary"][feat_idx]
        raw_counts[feature] += item["losing_rebuttal"]["predicted_binary"][feat_idx]

raw_prevalence = {feat: (count / total_main_texts) for feat, count in raw_counts.items()}

# ==========================================
# 4. COMPILING THE COMPARATIVE METRICS
# ==========================================
print("Compiling comparative metrics and applying ACC correction...")
audit_rows = []

for feature in FEATURES:
    human_prev = gold_prevalence[feature]
    raw_model_prev = raw_prevalence[feature]
    
    tpr = error_rates[feature]["tpr"]
    fpr = error_rates[feature]["fpr"]
    
    # Apply global ACC correction formula
    denominator = tpr - fpr
    if denominator != 0:
        corrected_prev = (raw_model_prev - fpr) / denominator
        corrected_prev = float(np.clip(corrected_prev, 0.0, 1.0))
    else:
        corrected_prev = raw_model_prev
        
    # Calculate Systematic Bias Ratio (Raw model prevalence divided by Human Baseline)
    # Ratio > 1.0 means the model over-predicts. Ratio < 1.0 means under-predicts.
    bias_ratio = raw_model_prev / human_prev if human_prev > 0 else np.nan
    
    audit_rows.append({
        "Value Category": feature,
        "Human Standard (%)": human_prev * 100,
        "Raw Model Pred (%)": raw_model_prev * 100,
        "ACC Calibrated (%)": corrected_prev * 100,
        "Over/Under Ratio": bias_ratio,
        "Model TPR (Recall)": tpr,
        "Model FPR": fpr
    })

# ==========================================
# 5. EXPORT AND PRINT TABLE
# ==========================================
df_audit = pd.DataFrame(audit_rows)
df_audit.to_csv(OUTPUT_CSV, index=False)

print("\n" + "="*95)
print("                      COMPREHENSIVE VALUE PREVALENCE & BIAS AUDIT")
print("="*95)
print(f"{'VALUE CATEGORY':<28} | {'HUMAN %':<8} | {'RAW MODEL %':<11} | {'CALIBRATED %':<12} | {'BIAS RATIO':<10} | {'TPR/FPR'}")
print("-" * 95)

for _, r in df_audit.iterrows():
    ratio_str = f"{r['Over/Under Ratio']:.2f}x" if not np.isnan(r['Over/Under Ratio']) else "N/A"
    tpr_fpr_str = f"{r['Model TPR (Recall)']:.2f}/{r['Model FPR']:.2f}"
    print(f"{r['Value Category']:<28} | {r['Human Standard (%)']:<8.2f} | {r['Raw Model Pred (%)']:<11.2f} | {r['ACC Calibrated (%)']:<12.2f} | {r['Over/Under Ratio']:<10.2f}f | {tpr_fpr_str}")
print("="*95)
print(f"Results saved to: {OUTPUT_CSV}\n")