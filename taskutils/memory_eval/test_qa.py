# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os, json
import argparse
import time
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

from utils import extract_solution,update_answer
from utils.envs import DATAROOT
import psutil

MIN_PIXELS_28_28 = int(os.getenv("MIN_PIXELS_28_28", 8))
MAX_PIXELS_28_28 = int(os.getenv("MAX_PIXELS_28_28", 512))

def calc_metrics(predictions, goldens):
    assert len(predictions) == len(goldens)
    metrics = {'f1': 0, 'prec': 0, 'recall': 0, 'em': 0, 'sub_em': 0, 'total_num': 0}
    for pred, gold in zip(predictions, goldens):
        update_answer(metrics, pred, gold)
    for k, _ in metrics.items():
        if k == 'total_num':
            continue
        metrics[k] = round((metrics[k]/metrics['total_num']), 2)
    return metrics


def get_pred(data, args, out_file):
    model = args.model
    print(f'Using API: {args.api}')
    if "gpt" in model or "o1" in model or "o3" in model or "o4" in model or "gemini" in model or "claude" in model:
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct', trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if args.api == "openai":
        from utils.openai_api import async_query_llm
        from utils import extract_answer
        if 'filter' in args.save_file:
            do_filter = True
        else:
            do_filter = False
    elif args.api == "memos":
        from utils.memos_api import async_query_llm, query_llm
        from utils import extract_answer
    elif args.api == "recurrent":
        from utils.recurrent import async_query_llm
        from utils import extract_answer
    elif args.api == "recurrent-boxed":
        from utils.recurrent_boxed import async_query_llm
        from utils import extract_boxed_answer as extract_answer
    elif args.api == "memocr_md":
        from utils.memocr_md import async_query_llm
        from utils import extract_boxed_answer as extract_answer
    elif args.api == "memocr_html":
        from utils.memocr_html import async_query_llm
        from utils import extract_boxed_answer as extract_answer
    elif args.api == "boxed":
        from utils.boxed import async_query_llm
        from utils import extract_boxed_answer as extract_answer
    else:
        print(f"Invalid API: {args.api}")
        raise ValueError
    coros = []
    all_memory_usage = []
    if args.n_proc == 1:
        print("Processing data in single process")
        outputs = []
        for item in tqdm(data, desc="Processing data"):
            start_time = time.time()
            response = query_llm(item, model, tokenizer, temperature=0.7, top_p=0.95)
            end_time = time.time()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            all_memory_usage.append(memory_usage)
            outputs.append(response)
    else:
        for item in data:
            start_time = time.time()
            if args.api in ["recurrent_revisit", "rememr1"]:
                raise ValueError(f"Invalid API: {args.api}")
            elif args.api == "openai" and do_filter:
                coro = async_query_llm(item, model, tokenizer, temperature=0.7, top_p=0.95, do_filter=True)
            else:
                coro = async_query_llm(item, model, tokenizer, temperature=0.7, top_p=0.95)
            end_time = time.time()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            all_memory_usage.append(memory_usage)
            coros.append(coro)
        from utils.aio import async_main, close_async_client
        import uvloop
        outputs = uvloop.run(async_main(coros, args.n_proc))
        uvloop.run(async_main([close_async_client()]))
    from collections import defaultdict
    scores = defaultdict(list)
    fout = open(out_file, 'w' if args.force else 'a', encoding='utf-8')
    metric_fout = open(out_file.replace('.jsonl', '_metric.json'), 'w' if args.force else 'a', encoding='utf-8')
    overall_readout_file = os.path.join(os.path.dirname(os.path.dirname(out_file)), os.path.splitext(os.path.basename(out_file))[0] + ".txt")
    overall_readout_fout = open(overall_readout_file, 'w' if args.force else 'a', encoding='utf-8')

    all_performance_metrics = []
    for i, (output, item) in enumerate(zip(outputs, data)):
        if isinstance(output, tuple):
            output, performance_metrics = output
        else:
            performance_metrics = {}
        if output == '' or '$LANG' in output:
            continue
        all_performance_metrics.append(performance_metrics)
        response = output.strip()
        pred, _ = extract_solution(response)
        item['response'] = response
        
        # Standard QA evaluation
        item['answer'] = item["answers"][0]
        item['pred'] = extract_answer(pred) if pred else extract_answer(response)
        if not item["pred"]:
            item['judge_f1'] = 0
            item['judge_em'] = 0
            item['judge_sub_em'] = 0
        else:
            all_scores = [calc_metrics([item["pred"]], [answer]) for answer in item["answers"]]
            item['judge_f1'] = max(score['f1'] for score in all_scores)
            item['judge_em'] = max(score['em'] for score in all_scores)
            item['judge_sub_em'] = max(score['sub_em'] for score in all_scores)
        
        scores['f1'].append(item['judge_f1'])
        scores['em'].append(item['judge_em'])
        scores['sub_em'].append(item['judge_sub_em'])
        item.pop('context');fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        if i == 0:
            print("="*40 + "New Item Start" + "="*40)
            print(item['response'])
            print("-"*80)
            print(item['pred'])
            print("-"*80)
            print(item['answer'])
            print("-"*80)
            print(item['judge_sub_em'])
            print("="*40 + "New Item End" + "="*40)

    average_performance_metrics = {}
    for item in all_performance_metrics:
        for k, v in item.items():
            if k not in average_performance_metrics:
                average_performance_metrics[k] = []
            average_performance_metrics[k].append(v)
    if 'trajectory' in average_performance_metrics:
        all_trajectories = average_performance_metrics['trajectory']
        all_f1s = scores['f1']
        all_ems = scores['em']
        all_sub_ems = scores['sub_em']
        trajectory_file = out_file.replace('.jsonl', '_trajectory.jsonl')
        with open(trajectory_file, 'w', encoding='utf-8') as f:
            for i, trajectory in enumerate(all_trajectories):
                f.write(json.dumps([{'f1': all_f1s[i], 'em': all_ems[i], 'sub_em': all_sub_ems[i]}] + trajectory, ensure_ascii=False) + '\n')
        print(f"Trajectory saved to {trajectory_file}")
        del average_performance_metrics['trajectory']
    for k, v in average_performance_metrics.items():
        average_performance_metrics[k] = round(sum(v) / len(v), 6)
    print(f"Running [{args.name}]")
    for k, v in scores.items():
        print(f"{k}: {round(sum(v) * 100 /len(v), 2)}")
    metric_scores = {k: round(sum(v) * 100 /len(v), 6) for k, v in scores.items()}
    metric_scores['time_per_item'] = round(average_performance_metrics.get('total_time', 0), 6)
    metric_scores['average_memory'] = round(sum(all_memory_usage) / len(all_memory_usage), 6)
    metric_scores['retrieval_time'] = round(average_performance_metrics.get('retrieval_time', 0), 6)
    metric_scores['corpus_memory_kb'] = round(average_performance_metrics.get('corpus_memory', 0)/1024, 6)
    print(f"Average time: {metric_scores['time_per_item']}")
    print(f"Average memory: {sum(all_memory_usage) / len(all_memory_usage)} MB")
    metric_fout.write(json.dumps(metric_scores, ensure_ascii=False, indent=4) + '\n')
    
    # Prepare overall readout line
    overall_readout_fout.write(f"{os.path.basename(os.path.dirname(out_file))}_MemBudget{MAX_PIXELS_28_28}\t{metric_scores['f1']}\t{metric_scores['em']}\t{metric_scores['sub_em']}\ttime={metric_scores['time_per_item']}s\tmemory={metric_scores['average_memory']}MB,retrieval_time={metric_scores['retrieval_time']}s,corpus_memory={metric_scores['corpus_memory_kb']}KB\n")
    
    print(f"Total: {len(data)}")
    print(f"Max image pixels: {MAX_PIXELS_28_28}")
    print(f"Min image pixels: {MIN_PIXELS_28_28}")

def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    out_file = os.path.join(args.save_dir, args.save_file + ".jsonl")

    dataset = concatenate_datasets([
            load_dataset("json", data_files=f"{DATAROOT}/{args.name}.json", split="train"),
        ])
        
    print(f"original data len {len(dataset)}")
    # 通过深拷贝生成新数据集
    import copy
    dataset = [copy.deepcopy(item) for _ in range(args.sampling) for item in dataset]
    print(f"sampling data len {len(dataset)}")

    data_all = []
    for idx, item in enumerate(dataset):
        item["_id"] = idx  # 现在每个 item 是独立对象
        data_all.append(item)

    print(data_all[0]["_id"])
    print(data_all[-1]["_id"])

    # cache
    # has_data = {}
    # if os.path.exists(out_file):
    #     with open(out_file, encoding='utf-8') as f:
    #         has_data = {json.loads(line)["_id"]: 0 for line in f}
    # data = []
    # for item in data_all:
    #     if item["_id"] not in has_data or args.force:
    #         data.append(item)
    #     elif args.force:
    #         data.append(item)
    data = data_all

    get_pred(data, args, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="eval_hotpotqa_50")
    parser.add_argument("--save_dir", "-s", type=str, default="results/ruler_hqa")
    parser.add_argument("--save_file", "-f", type=str, default="Qwen2.5-7B-Instruct-recurrent")
    parser.add_argument("--model", "-m", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer", "-t", type=str, default="/mnt/hdfs/hongli/model/Qwen2.5-7B-Instruct")
    parser.add_argument("--n_proc", "-n", type=int, default=64)
    parser.add_argument("--api", "-a", type=str, default="recurrent")
    parser.add_argument("--sampling", "-p", type=int, default=1)
    parser.add_argument('--force', action='store_true', help='force to overrite')
    args = parser.parse_args()
    main()