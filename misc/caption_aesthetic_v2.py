"""
Convert JSONL to CSV format.
- "filename" in JSONL -> "image" in CSV
- "text" in JSONL -> "prompt" in CSV
Samples 10k rows for test metadata CSV; the rest go to train metadata CSV.
"""

import json
import csv
import random

input_jsonl_path = "/miele/brian/data/Aesthetic-Train-V2/metadata.jsonl"
output_train_csv_path = "/local/brian/Aesthetic-Train-V2/metadata_train.csv"
output_test_csv_path = "/local/brian/Aesthetic-Train-V2/metadata_test.csv"
NUM_TEST = 10_000
random.seed(42)

# Read all entries
rows = []
with open(input_jsonl_path, 'r') as jsonl_file:
    for line in jsonl_file:
        data = json.loads(line.strip())
        filename = data.get('filename', data.get('file_name', ''))
        text = data.get('text', '')
        rows.append([filename, text])

# Sample 10k for test, rest for train
random.shuffle(rows)
test_rows = rows[:NUM_TEST]
train_rows = rows[NUM_TEST:]

def write_csv(path, rows):
    with open(path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['image', 'prompt'])
        writer.writerows(rows)

write_csv(output_test_csv_path, test_rows)
write_csv(output_train_csv_path, train_rows)

print(f"Total entries: {len(rows)}")
print(f"Test:  {len(test_rows)} -> {output_test_csv_path}")
print(f"Train: {len(train_rows)} -> {output_train_csv_path}")