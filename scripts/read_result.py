import os
import re
import csv
from collections import defaultdict

BASE_DIR = 'taskutils/memory_eval/results'
FILE_TEMPLATE = "eval-memocr_train_qwen2d5_7b_vl_triple_MemBudget{}.txt"

BUDGETS = [1024, 256, 64, 16]

def parse_line_info(line):
    parts = line.strip().split()
    if len(parts) < 4:
        return None
    
    full_key = parts[0]
    try:
        metric = float(parts[3])
    except ValueError:
        return None

    clean_key = re.sub(r'_MemBudget\d+$', '', full_key)
    
    match = re.search(r'eval_(.*)_(\d+)$', clean_key)
    if match:
        dataset_name = match.group(1)
        step_number = int(match.group(2))
        return dataset_name, step_number, metric
        
    return None

def main(file_template, output_csv):
    data = defaultdict(lambda: defaultdict(dict))
    
    all_steps = set()

    print(f"Reading files for budgets: {BUDGETS}...")

    for budget in BUDGETS:
        filename = file_template.format(budget)
        filepath = os.path.join(BASE_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filename}")
            continue
            
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    
                    parsed = parse_line_info(line)
                    if parsed:
                        dataset, step, metric = parsed
                        
                        data[dataset][budget][step] = metric
                        
                        all_steps.add(step)
                        
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    output_path = os.path.join(BASE_DIR, output_csv)
    print(f"\nWriting transposed results to {output_path}...")
    
    sorted_steps = sorted(list(all_steps))
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        header = ['Dataset', 'Budget'] + [str(s) for s in sorted_steps]
        writer.writerow(header)
        
        dataset_names = sorted(data.keys(), key=lambda x: (0 if 'hotpot' in x else 1, x))
        
        for dataset in dataset_names:
            for budget in BUDGETS:
                row = [dataset, budget]
                
                for step in sorted_steps:
                    val = data[dataset][budget].get(step, '')
                    row.append(val)
                
                writer.writerow(row)

    print("Done!")

    for budget in BUDGETS:
        line_str = f'{budget},'
        for ds in ['hotpotqa', '2wikimultihopqa', 'nq', 'triviaqa']:
            for step in sorted_steps:
                val = data[ds][budget].get(step, '-')
                line_str += f'{val},'
        print(line_str)

if __name__ == "__main__":
    file_template = FILE_TEMPLATE.replace('1024', '{}')
    output_csv = file_template.replace('{}.txt', '_aggregated.csv')
    main(file_template, output_csv)