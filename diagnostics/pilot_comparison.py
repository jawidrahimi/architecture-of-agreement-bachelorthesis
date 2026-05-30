import numpy as np
from datasets import load_dataset
from setfit import SetFitModel
from sklearn.metrics import f1_score, precision_score, recall_score

COMPLETE_DATASET_FILE = "complete_dataset_test.json"
MODEL_PATH = "Archive/setfit-modernBERT-base-full-dataset"

FEATURES = [
    'Self-direction: thought', 'Self-direction: action', 'Stimulation', 'Hedonism', 'Achievement',
    'Power: dominance', 'Power: resources', 'Face', 'Security: personal', 'Security: societal', 'Tradition',
    'Conformity: rules', 'Conformity: interpersonal', 'Humility', 'Benevolence: caring',
    'Benevolence: dependability', 'Universalism: concern', 'Universalism: nature', 'Universalism: tolerance',
    'Universalism: objectivity'
]

# 1. Reconstruct the exact 100-comment test set
dataset = load_dataset("json", data_files=COMPLETE_DATASET_FILE, split="train")

def encode_labels(record):
    encoded = [0.0] * len(FEATURES)
    for val in record["values"]:
        if val in FEATURES:
            encoded[FEATURES.index(val)] = 1.0
    return {"label": encoded}

dataset = dataset.map(encode_labels)
dataset = dataset.train_test_split(test_size=0.2, seed=3407)
test_dataset = dataset["test"]

X_test = test_dataset["argument"]
y_true = np.array(test_dataset["label"])

# 2. Load model
model = SetFitModel.from_pretrained(MODEL_PATH)
model.to("cpu")  # CPU is perfectly fine and fast for just 100 comments

def evaluate_with_length(max_len):
    print(f"\nRunning evaluation with max_seq_length = {max_len}...")
    model.model_body.max_seq_length = max_len
    
    y_pred = np.array(model.predict(X_test))
    
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    precision = precision_score(y_true, y_pred, average="micro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="micro", zero_division=0)
    
    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "precision": precision,
        "recall": recall
    }

results_256 = evaluate_with_length(256)
results_512 = evaluate_with_length(512)

# 3. Print Comparison Table
print("\n" + "="*55)
print(f"{'METRIC':<20} | {'LIMIT: 256 TOKENS':<15} | {'LIMIT: 512 TOKENS':<15}")
print("="*55)
print(f"{'Macro-F1':<20} | {results_256['macro_f1']:<15.4f} | {results_512['macro_f1']:.4f}")
print(f"{'Micro-F1 (Overall)':<20} | {results_256['micro_f1']:<15.4f} | {results_512['micro_f1']:.4f}")
print(f"{'Precision (Micro)':<20} | {results_256['precision']:<15.4f} | {results_512['precision']:.4f}")
print(f"{'Recall (Micro)':<20} | {results_256['recall']:<15.4f} | {results_512['recall']:.4f}")
print("="*55)