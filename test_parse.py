import json
import sys
import csv

input_file = 'output/MIMIC/synth_mimic_small_100pct_20260221_180714.json'
output_file = 'test_train.jsonl'
eval_file = 'data/mimic/mimic_m.json'
eval_output_file = 'test_eval.jsonl'
csv_file = 'data/mimic/ICD10_descriptions_mimic.csv'

def stream_json_array(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            char = f.read(1)
            if not char or char == '[':
                break
        buffer = ""
        decoder = json.JSONDecoder()
        while True:
            chunk = f.read(2048 * 2048)
            if not chunk:
                break
            buffer += chunk
            while True:
                buffer = buffer.lstrip(' \n\r\t,')
                if not buffer or buffer.startswith(']'):
                    break
                try:
                    obj, idx = decoder.raw_decode(buffer)
                    yield obj
                    buffer = buffer[idx:]
                except json.JSONDecodeError:
                    break

def process_file(in_path, out_path, desc_to_code, sorted_descs, limit=5):
    valid_entries = 0
    total_entries = 0
    with open(out_path, 'w', encoding='utf-8') as fout:
        for entry in stream_json_array(in_path):
            total_entries += 1
            if 'output' in entry:
                entry['original_text'] = entry.pop('output')
            if 'output1' in entry:
                entry['generated_text'] = entry.pop('output1')
            elif 'output_1' in entry:
                entry['generated_text'] = entry.pop('output_1')
            elif 'original_text' in entry and 'generated_text' not in entry:
                entry['generated_text'] = entry['original_text']
            
            input_text = entry.get('input', '')
            codes = []
            for desc in sorted_descs:
                if desc in input_text:
                    codes.append(desc_to_code[desc])
                    input_text = input_text.replace(desc, '')
            entry['icd10'] = codes
            entry['mode'] = 'cares'
            if len(codes) > 0:
                fout.write(json.dumps(entry) + '\n')
                valid_entries += 1
            if total_entries >= limit: break
    return valid_entries, total_entries

desc_to_code = {}
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['icd_version'] == '10': desc_to_code[row['long_title']] = row['icd_code']

sorted_descs = sorted(desc_to_code.keys(), key=len, reverse=True)

train_v, train_t = process_file(input_file, output_file, desc_to_code, sorted_descs, limit=5)
print(f"Train: mapped {train_v} / {train_t} items to {output_file}")
eval_v, eval_t = process_file(eval_file, eval_output_file, desc_to_code, sorted_descs, limit=5)
print(f"Eval: mapped {eval_v} / {eval_t} items to {eval_output_file}")
