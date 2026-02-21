
"""
Data processing script for multi-hop QA datasets.

Supports:
- 2wikimultihopqa, hotpotqa (from RUC-NLPIR/FlashRAG_datasets)
- Booksum (from YuWangX/Memalpha)

Usage examples:
  # For 2wikimultihopqa or hotpotqa:
  python process_test.py --data_sources 2wikimultihopqa --local_dir ./data/test --n_subset 100
  
  # For Booksum:
  python process_test.py --data_sources booksum --local_dir ./data/test --n_subset 100
"""

import re
import os
import datasets

import argparse
import json
from transformers import AutoTokenizer
import random
from tqdm import tqdm

# Global variables
QAS = None
DOCS = None
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")


def generate_input_output(index, num_docs, use_sim=False):
    global QAS, DOCS
    curr_q = QAS[index]['question']
    curr_a = QAS[index]['outputs']
    curr_docs = QAS[index]['context']
    curr_more = QAS[index].get('more_context', [])
    evidence_idx = QAS[index]['evidence_idx']
    
    if num_docs < len(DOCS):
        if False:
            if len(DOCS) > 100 * num_docs:
                # down-sample DOCS to 100 * num_docs
                sample_doc_indices = list(random.sample(range(len(DOCS)), 100 * num_docs))
            else:
                sample_doc_indices = list(range(len(DOCS)))
            similar_docs = _select_similar_docs(curr_docs, sample_doc_indices, num_docs)
            print(f'num of padding: {num_docs - len(curr_docs)}, num of similar docs: {len(similar_docs)}')
            all_docs = curr_docs + random.sample(similar_docs, num_docs - len(curr_docs))
        else:
            if (num_docs - len(curr_docs)) > len(curr_more):
                addition_docs = [i for i, d in enumerate(DOCS) if i not in curr_docs + curr_more]
                all_docs = curr_docs + curr_more + random.sample(addition_docs, max(0, num_docs - len(curr_docs) - len(curr_more)))
            else:
                all_docs = curr_docs + random.sample(curr_more, num_docs - len(curr_docs))
        all_docs = [DOCS[idx] for idx in all_docs]
    else:
        all_docs = DOCS
    perm = list(range(len(all_docs)))
    random.Random(4).shuffle(perm)
    all_docs = [all_docs[i] for i in perm]
    new_evidence_idx = [perm.index(i) for i in evidence_idx]
    is_ordered = new_evidence_idx == sorted(new_evidence_idx)
    is_reverse_ordered = new_evidence_idx == sorted(new_evidence_idx, reverse=True)
    sorted_evidence_idx = sorted(new_evidence_idx)
    min_distance = min(abs(sorted_evidence_idx[i+1] - sorted_evidence_idx[i]) for i in range(len(sorted_evidence_idx) - 1))

    DOCUMENT_PROMPT = "Document {i}:\n{document}"
    context = '\n\n'.join([DOCUMENT_PROMPT.format(i=i+1, document=d) for i, d in enumerate(all_docs)])
    
    formatted_output = {
        'context': context,
        'input': curr_q,
        'answers': curr_a,
        'num_docs': num_docs,
        'index': index,
        'is_ordered': is_ordered,
        'is_reverse_ordered': is_reverse_ordered,
        'min_distance': min_distance,
        'evidence_idx': new_evidence_idx,
        'level': QAS[index]['level'],
        'type': QAS[index]['type'],
        'id': QAS[index]['id'],
    }
    return formatted_output

def generate_dataset_booksum(n_subset: int, save_dir: str, data_source: str, incremental: int = 10, qas=None, docs=None, use_sim=False):
    DOCUMENT_PROMPT = "Document {i}:\n{document}"
    write_jsons = [
        {
            'context': '\n\n'.join([DOCUMENT_PROMPT.format(i=i+1, document=d) for i, d in enumerate(data_item['context'])]),
            'input': data_item['question'],
            'answers': data_item['outputs'],
            'num_docs': 1,
            'index': data_item['id'],
            'is_ordered': False,
            'is_reverse_ordered': False,
            'min_distance': 0,
            'evidence_idx': data_item['evidence_idx'],
            'level': data_item['level'],
            'type': data_item['type'],
            'id': data_item['id'],
        } for data_item in qas
    ]
    if len(write_jsons) > n_subset: # down-sample if too many samples
        write_jsons = random.sample(write_jsons, n_subset)
    print(f'{data_source} {len(write_jsons)} samples')
    with open(save_dir + ".json", 'w') as f:
        json.dump(write_jsons, f, ensure_ascii=False, indent=4)
    return write_jsons

def generate_dataset(n_subset: int, save_dir: str, data_source: str, incremental: int = 10, qas=None, docs=None, use_sim=False):
    global QAS, DOCS
    if qas is None or docs is None:
        raise ValueError("QAS and DOCS must be provided.")

    if len(qas) < n_subset:
        sub_index = list(range(len(qas)))
    else:
        sub_index = random.sample(range(len(qas)), n_subset)
    assert len(sub_index) <= n_subset, f'{len(sub_index)} < {n_subset}'

    QAS = qas
    DOCS = docs

    print("start")
    
    from utils import TqdmExecutor
    if incremental > 10000:
        write_jsons = TqdmExecutor(max_workers=os.cpu_count()).run(generate_index_only, sub_index, num_docs=incremental, use_sim=use_sim)
    else:
        write_jsons = TqdmExecutor(max_workers=os.cpu_count()).run(generate_input_output, sub_index, num_docs=incremental, use_sim=use_sim)
        tokens = [len(x) for x in tokenizer([j['context'] for j in write_jsons])['input_ids']]
        print(f'{data_source} max(tokens), min(tokens), sum(tokens) / len(tokens): {max(tokens)}, {min(tokens)}, {sum(tokens) / len(tokens)}')

    with open(save_dir + ".json", 'w') as f:
        json.dump(write_jsons[:n_subset], f, ensure_ascii=False, indent=4)
    return write_jsons
    write_jsons_forward = [d for d in write_jsons if d['is_ordered']]
    write_jsons_reverse = [d for d in write_jsons if not d['is_ordered']]
    write_jsons_reverse_ordered = [d for d in write_jsons_reverse if d['is_reverse_ordered']]

    with open(save_dir + "_forward.json", 'w') as f:
        json.dump(write_jsons_forward[:n_subset], f, ensure_ascii=False, indent=4)
    with open(save_dir + "_reverse.json", 'w') as f:
        json.dump(write_jsons_reverse[:n_subset], f, ensure_ascii=False, indent=4)
    with open(save_dir + "_reverse_ordered.json", 'w') as f:
        json.dump(write_jsons_reverse_ordered[:n_subset], f, ensure_ascii=False, indent=4)
    return write_jsons

    write_jsons_min_distance_100 = [d for d in write_jsons if (d['min_distance'] >=100 and d['is_reverse_ordered'])]
    write_jsons_min_distance_200 = [d for d in write_jsons if (d['min_distance'] >=200 and d['is_reverse_ordered'])]
    write_jsons_min_distance_400 = [d for d in write_jsons if (d['min_distance'] >=400 and d['is_reverse_ordered'])]
    write_jsons_min_distance_800 = [d for d in write_jsons if (d['min_distance'] >=800 and d['is_reverse_ordered'])]
    write_jsons_min_distance_1600 = [d for d in write_jsons if (d['min_distance'] >=1600 and d['is_reverse_ordered'])]

    # print(f'{data_source} {len(write_jsons_forward)} forward, {len(write_jsons_reverse)} reverse')
    print(f'{data_source}-{incremental} {len(write_jsons_min_distance_100)} min_distance >= 100')
    print(f'{data_source}-{incremental} {len(write_jsons_min_distance_200)} min_distance >= 200')
    print(f'{data_source}-{incremental} {len(write_jsons_min_distance_400)} min_distance >= 400')
    print(f'{data_source}-{incremental} {len(write_jsons_min_distance_800)} min_distance >= 800')
    print(f'{data_source}-{incremental} {len(write_jsons_min_distance_1600)} min_distance >= 1600')

    with open(save_dir + "_min_distance_100.json", 'w') as f:
        json.dump(write_jsons_min_distance_100[:n_subset], f, ensure_ascii=False, indent=4)
    with open(save_dir + "_min_distance_200.json", 'w') as f:
        json.dump(write_jsons_min_distance_200[:n_subset], f, ensure_ascii=False, indent=4)
    with open(save_dir + "_min_distance_400.json", 'w') as f:
        json.dump(write_jsons_min_distance_400[:n_subset], f, ensure_ascii=False, indent=4)
    with open(save_dir + "_min_distance_800.json", 'w') as f:
        json.dump(write_jsons_min_distance_800[:n_subset], f, ensure_ascii=False, indent=4)
    with open(save_dir + "_min_distance_1600.json", 'w') as f:
        json.dump(write_jsons_min_distance_1600[:n_subset], f, ensure_ascii=False, indent=4)

    return write_jsons

def generate_index_only(index, num_docs, use_sim=False):
    global QAS, DOCS
    curr_q = QAS[index]['question']
    curr_a = QAS[index]['outputs']
    curr_docs = QAS[index]['context']
    curr_more = QAS[index].get('more_context', [])
    evidence_idx = QAS[index]['evidence_idx']
    

    if num_docs < len(DOCS):
        if False:
            if len(DOCS) > 100 * num_docs:
                # down-sample DOCS to 100 * num_docs
                sample_doc_indices = list(random.sample(range(len(DOCS)), 100 * num_docs))
            else:
                sample_doc_indices = list[int](range(len(DOCS)))
            similar_docs = _select_similar_docs(curr_docs, sample_doc_indices, num_docs)
            all_docs = curr_docs + random.sample(similar_docs, num_docs - len(curr_docs))
        else:
            if (num_docs - len(curr_docs)) > len(curr_more):
                addition_docs = [i for i, d in enumerate(DOCS) if i not in curr_docs + curr_more]
                all_docs = curr_docs + curr_more + random.sample(addition_docs, max(0, num_docs - len(curr_docs) - len(curr_more)))
            else:
                all_docs = curr_docs + random.sample(curr_more, num_docs - len(curr_docs))
    else:
        all_docs = list(range(len(DOCS)))

    perm = list(range(len(all_docs)))
    random.Random(4).shuffle(perm)
    all_docs = [all_docs[i] for i in perm]
    new_evidence_idx = [perm.index(e) for e in evidence_idx]
    is_ordered = new_evidence_idx == sorted(new_evidence_idx)
    is_reverse_ordered = new_evidence_idx == sorted(new_evidence_idx, reverse=True)
    formatted_output = {
        'context': all_docs,
        'input': curr_q,
        'answers': curr_a,
        'num_docs': num_docs,
        'index': index,
        'is_ordered': is_ordered,
        'is_reverse_ordered': is_reverse_ordered,
        'evidence_idx': new_evidence_idx,
        'level': QAS[index]['level'],
        'type': QAS[index]['type'],
        'id': QAS[index]['id'],
    }
    return formatted_output

def read_dataset_booksum(split='test'):
    """Read the Booksum dataset from Memalpha"""
    dataset = datasets.load_dataset('YuWangX/Memalpha', split=split)
    # filter 'data_source'=='booksum'
    dataset = [ex for ex in dataset if ex.get('data_source') == 'booksum']
    print(f'Using the Memalpha {split} dataset with {len(dataset)} samples...')
    
    total_docs = []
    total_qas = []
    
    for example in tqdm(dataset):
        context = example['chunks'] # in json format
        context = json.loads(context) # in list format
        total_docs.extend(context)
        questions_and_answers = example['questions_and_answers']
        questions_and_answers = json.loads(questions_and_answers)
        

        # Process each question-answer pair
        for qa_idx, qa in enumerate(questions_and_answers):
            data_item = dict()
            data_item['question'] = qa['question'].strip()
            
            # Answers should be in list format
            answer = qa['answer']
            assert isinstance(answer, str), f'{answer} is not a string'
            answer = answer.split(', ')
            assert len(answer) > 0, f'{answer} is not a valid answer'
            data_item['outputs'] = answer
            
            # All chunks for this example are relevant context
            data_item['context'] = context
            
            # For booksum, we treat all chunks as evidence since we don't have explicit supporting facts
            # Use all chunks as evidence
            data_item['evidence_idx'] = 0
            
            # Add metadata
            data_item['level'] = None
            data_item['type'] = None
            data_item['id'] = f"{example['instance_id']}_q{qa_idx}"
            
            # Add more_context as empty for now (could be populated with chunks from other examples)
            data_item['more_context'] = []
            
            total_qas.append(data_item)
    
    print(f'Booksum: {len(total_qas)} qas, {len(total_docs)} docs')
    return total_qas, total_docs


def read_dataset(data_source):
    dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_source)

    if 'test' in dataset:
        test_dataset = dataset['test']
        print(f'Using the {data_source} test dataset with {len(test_dataset)} samples...')
    elif 'dev' in dataset:
        test_dataset = dataset['dev']
        print(f'Using the {data_source} dev dataset with {len(test_dataset)} samples...')
    else:
        test_dataset = dataset['train']
        print(f'Using the {data_source} train dataset with {len(test_dataset)} samples...')

    total_docs = []
    for example in test_dataset:
        context = example['metadata']['context']
        titles = context['title']
        sentences = context['sentences'] if 'sentences' in context else context['content']
        for t, s in zip(titles, sentences):
            total_docs.append(f"{t}\n{''.join(s)}")
    total_docs = sorted(list(set(total_docs)))
    doc_to_idx = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    print(f'{data_source} {len(test_dataset)} examples')
    for example in tqdm(test_dataset):
        data_item = dict()
        data_item['question'] = example['question'].strip()
        if data_item['question'][-1] != '?':
            data_item['question'] += '?'
        data_item['outputs'] = example['golden_answers']
        assert isinstance(data_item['outputs'], list)

        context = example['metadata']['context']
        titles = context['title']
        sentences = context['sentences'] if 'sentences' in context else context['content']
        data_item['context'] = [doc_to_idx[f"{t}\n{''.join(p)}"] for t, p in zip(titles, sentences)]

        supporting_list = example['metadata']['supporting_facts']['title']

        if not all(title in context['title'] for title in supporting_list):
            continue
        
        data_item['evidence_idx'] = [context['title'].index(title) for title in supporting_list]
        data_item['level'] = example['metadata']['level'] if 'level' in example['metadata'] else None
        data_item['type'] = example['metadata']['type'] if 'type' in example['metadata'] else None
        data_item['id'] = example['id']

        total_qas.append(data_item)
    print(f'{data_source} {len(total_qas)} qas')
    return total_qas, total_docs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/test')
    parser.add_argument('--data_sources', default='nq')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_subset', type=int, default=0, help='number of samples to subset')
    parser.add_argument('--use_sim', type=bool, default=False, help='use similarity-based selection')
    args = parser.parse_args()

    data_sources = args.data_sources.split(',')
    random.seed(args.seed)

    for data_source in data_sources:
        # Check if it's booksum dataset
        if data_source.lower() == 'booksum':
            total_qas, total_docs = read_dataset_booksum(split='test')
            generate_dataset_booksum(args.n_subset, f"{args.local_dir}/eval_{data_source}", data_source, incremental=-1, qas=total_qas, docs=total_docs, use_sim=args.use_sim)
            continue
        else:
            total_qas, total_docs = read_dataset(data_source)

        for length in [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]:
            generate_dataset(args.n_subset, f"{args.local_dir}/eval_{data_source}_{length}", data_source, incremental=length, qas=total_qas, docs=total_docs, use_sim=args.use_sim)

        # clear global variables
        QAS = None
        DOCS = None