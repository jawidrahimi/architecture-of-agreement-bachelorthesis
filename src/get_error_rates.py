import json
import numpy as np
from datasets import load_dataset
from setfit import SetFitModel
from sklearn.metrics import confusion_matrix

# ==========================================
# CONFIGURATION
# ==========================================
COMPLETE_DATASET_FILE = "Archive/complete_dataset_test.json"  # Your human-labeled file
MODEL_PATH = "Archive/setfit-modernBERT-base-full-dataset"  # Your fine-tuned model path
OUTPUT_FILE = "value_error_rates.json"  # File to save TPR and FPR for QuaPy

FEATURES = [
    'Self-direction: thought', 'Self-direction: action', 'Stimulation', 'Hedonism', 'Achievement',
    'Power: dominance', 'Power: resources', 'Face', 'Security: personal', 'Security: societal', 'Tradition',
    'Conformity: rules', 'Conformity: interpersonal', 'Humility', 'Benevolence: caring',
    'Benevolence: dependability', 'Universalism: concern', 'Universalism: nature', 'Universalism: tolerance',
    'Universalism: objectivity'
]

# ==========================================
# 1. RECONSTRUCT THE EXACT TEST SPLIT
# ==========================================
print("Reconstructing the test split...")
dataset = load_dataset("json", data_files=COMPLETE_DATASET_FILE, split="train")

def encode_labels(record):
    encoded = [0.0] * len(FEATURES)
    for val in record["values"]:
        if val in FEATURES:
            encoded[FEATURES.index(val)] = 1.0
    return {"label": encoded}

dataset = dataset.map(encode_labels)

# Using the exact same seed (3407) ensures an identical 20% split
dataset = dataset.train_test_split(test_size=0.2, seed=3407)
test_dataset = dataset["test"]
print(f"Reconstructed test set size: {len(test_dataset)} comments.")

# ==========================================
# 2. RUN MODEL PREDICTIONS
# ==========================================
print(f"Loading model from: {MODEL_PATH}")
model = SetFitModel.from_pretrained(MODEL_PATH)

X_test = test_dataset["argument"]
y_true = np.array(test_dataset["label"])

print("Running predictions on the test set...")
y_pred = np.array(model.predict(X_test))

# ==========================================
# 3. CALCULATE TPR AND FPR PER CLASS
# ==========================================
print("Calculating TPR and FPR for each of the 20 classes...")
error_rates = {}

for idx, feature in enumerate(FEATURES):
    # Retrieve true and predicted columns for the current class
    class_true = y_true[:, idx]
    class_pred = y_pred[:, idx]
    
    # Generate confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(class_true, class_pred, labels=[0, 1]).ravel()
    
    # Calculate rates safely handling division-by-zero
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    error_rates[feature] = {
        "tpr": float(tpr),
        "fpr": float(fpr)
    }

# ==========================================
# 4. SAVE OUTPUT FOR QUAPY
# ==========================================
print(f"Saving error rates to: {OUTPUT_FILE}")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(error_rates, f, indent=4)

print("\nProcessing complete. Calculated rates:")
for feature, rates in error_rates.items():
    print(f"  {feature:<30} | TPR (Recall): {rates['tpr']:.4f} | FPR: {rates['fpr']:.4f}")