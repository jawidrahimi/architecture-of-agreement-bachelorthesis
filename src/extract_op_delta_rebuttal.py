import json
import time

INPUT_FILE = "pairs.jsonl"
OUTPUT_FILE = "op_and_delta_rebuttals.json"

def extract_delta_winning_comment(delta_comment_obj):
    """Return the comment body that actually has delta=True."""
    for comment in delta_comment_obj["comments"]:
        if comment.get("delta"):
            return comment["body"]
    return delta_comment_obj["comments"][0]["body"] if delta_comment_obj["comments"] else ""

start = time.time()

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    content = f.read()

decoder = json.JSONDecoder()
entries = []
pos = 0
while pos < len(content):
    while pos < len(content) and content[pos] in " \t\n\r":
        pos += 1
    if pos >= len(content):
        break
    obj, pos = decoder.raw_decode(content, pos)
    entries.append(obj)

results = []
for entry in entries:
    submission = entry["submission"]
    delta_obj = entry["delta_comment"]
    results.append({
        "submission_id":  entry["submission_id"],
        "title":          submission["title"],
        "op_author":      submission["author"],
        "op_text":        submission["selftext"],
        "delta_author":   delta_obj["author"],
        "delta_rebuttal": extract_delta_winning_comment(delta_obj),
    })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

elapsed = time.time() - start
print(f"Saved {len(results)} entries to {OUTPUT_FILE} in {elapsed:.2f}s")