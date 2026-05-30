import json
import os
import numpy as np
from transformers import AutoTokenizer

INPUT_FILE = "triage_pairs.json"
MODEL_PATH = "Archive/setfit-modernBERT-base-full-dataset"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Could not find input file: {INPUT_FILE}")

print("Loading dataset...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Loading local tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

op_lengths = []
win_lengths = []
lose_lengths = []

print("Analyzing token lengths (this will take a few seconds)...")
for item in data:
    # Tokenize and count length (without adding special tokens like [CLS]/[SEP] for raw count)
    op_len = len(tokenizer.encode(item["op_context"]["text"], add_special_tokens=False))
    win_len = len(tokenizer.encode(item["winning_rebuttal"]["text"], add_special_tokens=False))
    lose_len = len(tokenizer.encode(item["losing_rebuttal"]["text"], add_special_tokens=False))
    
    op_lengths.append(op_len)
    win_lengths.append(win_len)
    lose_lengths.append(lose_len)

all_lengths = op_lengths + win_lengths + lose_lengths

def print_stats(name, lengths):
    lengths = np.array(lengths)
    pct_under_256 = (lengths <= 256).mean() * 100
    pct_under_512 = (lengths <= 512).mean() * 100
    
    print(f"\n=== Distribution for {name} ===")
    print(f"  Median (50th percentile): {int(np.percentile(lengths, 50))} tokens")
    print(f"  75th percentile:          {int(np.percentile(lengths, 75))} tokens")
    print(f"  90th percentile:          {int(np.percentile(lengths, 90))} tokens")
    print(f"  95th percentile:          {int(np.percentile(lengths, 95))} tokens")
    print(f"  Percentage <= 256 tokens: {pct_under_256:.2f}%")
    print(f"  Percentage <= 512 tokens: {pct_under_512:.2f}%")

print_stats("Original Posts (OPs)", op_lengths)
print_stats("Winning Rebuttals", win_lengths)
print_stats("Losing Rebuttals", lose_lengths)
print_stats("Combined Overall Corpus", all_lengths)