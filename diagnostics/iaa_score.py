import json
from collections import defaultdict
from itertools import combinations

def extract_labels(item):
    """
    Normalizes the inconsistent transcription formats across the three files.
    """
    tx = item.get('transcription', [])
    labels = set()
    
    # Format 1: Dirk's file {"choices": [["Label1"], ["Label2"]]}
    if isinstance(tx, dict):
        choices = tx.get('choices', [])
        for choice in choices:
            if isinstance(choice, list) and len(choice) > 0:
                labels.add(choice[0])
    
    # Format 2: Edwin/Jawid's file ["Label1", "Label2"]
    elif isinstance(tx, list):
        for choice in tx:
            if isinstance(choice, str):
                labels.add(choice)
            elif isinstance(choice, list) and len(choice) > 0:
                labels.add(choice[0])
                
    return labels

def calculate_f1(set_a, set_b):
    """Calculates the F1 score between two sets of labels."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a.intersection(set_b))
    if intersection == 0:
        return 0.0
    precision = intersection / len(set_a)
    recall = intersection / len(set_b)
    return 2 * (precision * recall) / (precision + recall)

def process_files(file_paths):
    # Load data
    data = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data.append(json.load(f))

    # Ensure all files have the same number of items
    lengths = [len(d) for d in data]
    if len(set(lengths)) > 1:
        print(f"Warning: Files have different lengths! {lengths}. Using the minimum ({min(lengths)}).")
    num_items = min(lengths)

    all_values = set()
    annotator_labels = [] # List of lists: [ [set_item1, set_item2...], [annotator2...], ... ]

    for d in data:
        labels_for_this_annotator = []
        for item in d:
            lbls = extract_labels(item)
            labels_for_this_annotator.append(lbls)
            all_values.update(lbls)
        annotator_labels.append(labels_for_this_annotator)

    # 1. Calculate Overall Mean Pairwise F1
    # Pairs: (0,1), (0,2), (1,2)
    total_f1 = 0
    comparisons = 0
    for i in range(len(annotator_labels)):
        for j in range(i + 1, len(annotator_labels)):
            f1_sum = sum(calculate_f1(annotator_labels[i][k], annotator_labels[j][k]) for k in range(num_items))
            total_f1 += (f1_sum / num_items)
            comparisons += 1
    
    overall_iaa = total_f1 / comparisons

    # 2. Calculate Score Per Value (Cohen's Kappa equivalent for binary presence)
    # We calculate the % of arguments where all annotators agreed on the presence/absence
    value_scores = {}
    for val in sorted(all_values):
        agreements = 0
        for k in range(num_items):
            # Check if all annotators agree on this specific label for this item
            presence = [val in annotator_labels[i][k] for i in range(len(data))]
            if all(presence) or not any(presence):
                agreements += 1
        value_scores[val] = agreements / num_items

    # 3. Count label frequencies across all annotators and items (for class imbalance check)
    total_annotations = len(data) * num_items  # total (annotator, item) pairs
    value_freqs = {}
    for val in sorted(all_values):
        count = sum(
            val in annotator_labels[i][k]
            for i in range(len(data))
            for k in range(num_items)
        )
        value_freqs[val] = (count, count / total_annotations)

    # 4. Generate confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))

    for k in range(num_items):
        for i, j in combinations(range(len(annotator_labels)), 2):
            only_i = annotator_labels[i][k] - annotator_labels[j][k]
            only_j = annotator_labels[j][k] - annotator_labels[i][k]
            for li in only_i:
                for lj in only_j:
                    confusion[li][lj] += 1
                    confusion[lj][li] += 1

    return overall_iaa, value_scores, value_freqs, confusion

# File names (ensure these match your local file names)
files = ['First_150_Dirk.json', 'First_150_Edwin.json', 'First_150_Jawid.json']

try:
    overall, per_val, freqs, confusion = process_files(files)

    print(f"{'='*40}")
    print(f"INTER-ANNOTATOR AGREEMENT RESULTS")
    print(f"{'='*40}")
    print(f"Overall Pairwise F1 Score: {overall:.4f}")
    print(f" (Interpreted as average similarity between annotators)")
    print(f"{'-'*40}")
    print(f"{'Value Name':<25} | {'Consensus Score'}")
    print(f"{'-'*40}")
    for val, score in per_val.items():
        print(f"{val:<25} | {score:.4f}")
    print(f"{'='*40}")

    print(f"\n{'='*40}")
    print(f"LABEL FREQUENCY (CLASS IMBALANCE)")
    print(f"{'='*40}")
    print(f"{'Value Name':<25} | {'Count':>7} | {'% of all annotations'}")
    print(f"{'-'*40}")
    for val, (count, pct) in freqs.items():
        print(f"{val:<25} | {count:>7} | {pct:.2%} | {per_val[val]:.4f}")
    print(f"{'='*40}")

except FileNotFoundError as e:
    print(f"Error: Could not find files. Please ensure the JSON files are in the same directory. {e}")