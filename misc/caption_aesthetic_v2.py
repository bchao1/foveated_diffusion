"""
Convert JSONL to CSV format.
- "filename" in JSONL -> "image" in CSV
- "text" in JSONL -> "prompt" in CSV
"""

import json
import csv

input_jsonl_path = "/miele/brian/data/Aesthetic-Train-V2/metadata.jsonl"
output_csv_path = "/local/brian/Aesthetic-Train-V2/metadata.csv"

# Read JSONL and write to CSV
with open(input_jsonl_path, 'r') as jsonl_file, open(output_csv_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    
    # Write header
    writer.writerow(['image', 'prompt'])
    
    # Process each line (only include images with "0000" in the name)
    count = 0
    for line in jsonl_file:
        data = json.loads(line.strip())
        filename = data.get('filename', data.get('file_name', ''))  # handle both "filename" and "file_name"
        
        # Only include images with "0000" in the filename
        #if 'images/0000/' not in filename:
        #    continue
            
        text = data.get('text', '')
        writer.writerow([filename, text])
        count += 1

print(f"Converted {count} entries from {input_jsonl_path} to {output_csv_path}")