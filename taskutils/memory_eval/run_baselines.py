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
import os
import time
from dataclasses import dataclass
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
DASH_PORT = os.getenv("DASH_PORT", "8265")
SERVE_PORT = os.getenv("SERVE_PORT", "8000")
REVERSED = os.getenv("REVERSED", 0)


@dataclass
class ENV:
    # config for direct generation
    MAX_INPUT_LEN: int = 120000
    MAX_OUTPUT_LEN: int = 10000
    # Config for memory agent
    RECURRENT_MAX_CONTEXT_LEN: int = None
    RECURRENT_CHUNK_SIZE: int = None
    RECURRENT_MAX_NEW: int = None

    def setenv(self):
        if not hasattr(self, "_environ"):
            self._environ = {}
        for k, v in self.__dict__.items():
            if v is not None and k != "_environ":
                os.environ[k] = str(v)
                self._environ[k] = str(v)
                print(f"set {k}={v}")

    def unsetenv(self):
        for k in self._environ:
            os.environ[k] = self._environ[k]
        self._environ = {}


# for ruler hqa, we just control the number of distractive wiki items instead the context length
# 50~7K tokens, 100~14K tokens and so on.
RULER_HQA_TESTS = [50, 100, 200, 400, 800, 1600, 3200, 6400]
RULER_HQA_TESTS_OVER_1M = [12800, 25600]
# for other ruler task, we use the standard synthetic scripts for convenient and control the context length.
RULER_TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "fwe",
    "qa_1",
]
RULER_PROMPT_LENGTH = [8192, 16384, 32768, 65536, 131072, 262144, 524288]
RULER_GENERRAL_TESTS = [(task, length) for task in RULER_TASKS for length in RULER_PROMPT_LENGTH]
import subprocess

# TEST_NUM_DOCS = [50, 100, 200, 400, 800, 1600, 3200, 6400]
# TEST_NUM_DOCS = [50, 100, 200, 400] # don't run too much
TEST_NUM_DOCS = [50, 100, 200, 400, 800, 1600] # run all
# TEST_NUM_DOCS = [50] # for debug
RULER_TEST_TASKS = [
    f"eval_{ds}_{num_docs}" \
    for ds in ['hotpotqa', '2wikimultihopqa'] \
    for num_docs in TEST_NUM_DOCS
]

RULER_TEST_TASKS_set2 = [
    f"eval_{ds}_{num_docs}" \
    for ds in ['hotpotqa', '2wikimultihopqa', 'bamboogle', 'musique', 'nq', 'triviaqa'] \
    for num_docs in [50, 200, 800]
]
enable_more_ds = os.getenv("ENABLE_MORE_DS", "0")
if enable_more_ds == "1":
    print("Running with more datasets")
    RULER_TEST_TASKS = RULER_TEST_TASKS_set2

class Config:
    SERVE_TAG = "__serve"

    def __init__(self, name, ckpt, tp, method, env, concur=1024):
        MAX_PIXELS_28_28 = os.getenv("MAX_PIXELS_28_28", None)
        NAME_VERSION = os.getenv("NAME_VERSION", "v1")
        self.name = name + f"_{NAME_VERSION}_MemBudget{MAX_PIXELS_28_28}" if MAX_PIXELS_28_28 else name
        self.ckpt = ckpt
        from pathlib import Path

        if Path(self.ckpt).is_dir():
            self.model = Path(self.ckpt).name
        else:
            self.model = self.ckpt
        self.method = method
        self.tp = tp
        self.env = env
        self.concur = concur
        self.test_process = {}

    def serve(self, wait=True):
        GPU_NUM = 8
        self.dp = int(GPU_NUM/int(self.tp))
        cmd = f"python -m vllm.entrypoints.openai.api_server --model {self.ckpt} --tensor-parallel-size {self.tp} --data-parallel-size {self.dp} --served-model-name {Path(self.ckpt).name}"
        # cmd = f"python -m sglang.launch_server --model-path {self.ckpt} --tensor-parallel-size {self.tp} --served-model-name {Path(self.ckpt).name} --port {SERVE_PORT} --data-parallel-size {self.pp}"
        print("serving command:")
        print(cmd)
        if wait:
            p = subprocess.run(["curl", "-m", "10", f"http://127.0.0.1:{SERVE_PORT}/v1/models"], capture_output=True)
            if p.returncode == 0:
                print("server already started")
                return
            else:
                os.system(f"yes | serve shutdown -a http://localhost:{DASH_PORT}")
                # setsid so that it can be interrupted
                serve_p = subprocess.Popen(cmd.split(), preexec_fn=os.setsid)
                self.test_process[self.SERVE_TAG] = serve_p
            while True:
                print("try to conntect...")
                p = subprocess.run(["curl", "-m", "100000000", f"http://127.0.0.1:{SERVE_PORT}/v1/models"], capture_output=True)
                if p.returncode != 0:
                    print("waiting...")
                    time.sleep(5)
                elif rf'"id":"{self.model}"' not in p.stdout.decode():
                    print("model not found, maybe shutting down previous server...")
                    time.sleep(5)
                else:
                    print("connected")
                    break
        else:
            p = subprocess.run(["curl", "-m", "10", f"http://127.0.0.1:{SERVE_PORT}/v1/models"], capture_output=True)
            if p.returncode != 0:
                print("server not started")
                exit(1)
        print(p.stdout)

    def run(self, tests, serve=True, force=False):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        self.env.setenv()
        self.serve(serve)
        concur = self.concur
        for test in tests:
            if test in RULER_HQA_TESTS:
                cmd = f"""python ruler_hqa.py --model {self.model}\
                    --length {test} \
                    --save_dir results/ruler_hqa_{test} \
                    --save_file {self.name} \
                    --tokenizer {self.ckpt} \
                    --api {self.method} \
                    --n_proc {concur}"""
            elif test in RULER_TEST_TASKS:
                cmd = f"""python test_qa.py --model {self.model}\
                    --name {test} \
                    --save_dir results/{test} \
                    --save_file {self.name} \
                    --tokenizer {self.ckpt} \
                    --api {self.method} \
                    --n_proc {concur}"""
            elif test in RULER_GENERRAL_TESTS:
                cmd = f"""python ruler_general.py --model {self.model}\
                    --split {test[0]} \
                    --length {test[1]} \
                    --save_dir results/ruler_{test[0]}_{test[1]} \
                    --save_file {self.name} \
                    --tokenizer {self.ckpt} \
                    --api {self.method} \
                    --n_proc {concur}"""
            elif test in RULER_HQA_TESTS_OVER_1M:
                cmd = f"""python ruler_hqa_over1m.py --model {self.model}\
                    --length {test} \
                    --save_dir results/ruler_hqa_{test} \
                    --save_file {self.name} \
                    --tokenizer {self.ckpt} \
                    --api {self.method} \
                    --n_proc {concur}"""
            else:
                print("=" * 20 + f"Not Implemented Task {test}, please check" + "=" * 20)
                continue
            if force:
                cmd += " --force"
            p = subprocess.Popen(cmd, shell=True)
            self.test_process[test] = p
            p.wait()
            self.test_process[test].wait()
        self.env.unsetenv()
        if serve and self.SERVE_TAG in self.test_process:
            os.killpg(os.getpgid(self.test_process[self.SERVE_TAG].pid), 2)
            try:
                self.test_process[self.SERVE_TAG].wait(30)
            except:
                self.test_process[self.SERVE_TAG].kill()
        print("all tests finished")

    def __del__(self):
        for k, p in self.test_process.items():
            if k == self.SERVE_TAG:
                os.killpg(os.getpgid(p.pid), 2)
            else:
                p.kill()


L1 = Config(
    name="L1-120k+10k",
    ckpt=f"Tongyi-Zhiwen/QwenLong-L1-32B",
    tp=4,
    method="openai",
    concur=128,
    env=ENV(),
)

R1_14B = Config(
    name="R1-14B-120k+10k-openai",
    ckpt=f"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    tp=4,
    method="openai",
    concur=256,
    env=ENV(),
)

R1_7B = Config(
    name="R1-7B-120k+10k",
    ckpt=f"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    tp=4,
    method="openai",
    concur=256,
    env=ENV(),
)

Qwen25_7B_1M = Config(
    name="Qwen-7B-1M",
    ckpt=f"Qwen/Qwen2.5-7B-Instruct-1M",
    tp=4,
    method="openai",
    concur=256,
    env=ENV(MAX_INPUT_LEN=990000, MAX_OUTPUT_LEN=10000),
)

Qwen25_14B_1M = Config(
    name="Qwen-14B-1M",
    ckpt=f"Qwen/Qwen2.5-14B-Instruct-1M",
    tp=4,
    method="openai",
    concur=256,
    env=ENV(MAX_INPUT_LEN=990000, MAX_OUTPUT_LEN=10000),
)

Qwen3_4B_128k = Config(
    name="Qwen3-4B-128k",
    ckpt=f"Qwen/Qwen3-4B",
    tp=8,
    method="openai",
    concur=256,
    env=ENV(),
)

Qwen3_8B_128k = Config(
    name="Qwen3-8B-128k",
    ckpt=f"Qwen/Qwen3-8B",
    tp=8,
    method="openai",
    concur=256,
    env=ENV(),
)

# This model's path is special
Qwen25_7B_128k = Config(
    name="Qwen-7B-128k",
    ckpt="Qwen/Qwen2.5-7B-Instruct",
    tp=4,
    method="openai",
    concur=256,
    env=ENV(),
)

Qwen25_14B_128k = Config(
    name="Qwen-14B-128k",
    ckpt=f"Qwen/Qwen2.5-14B-Instruct",
    tp=4,
    method="openai",
    concur=256,
    env=ENV(),
)

Qwen25_14B_5k_1k = Config(
    name="Qwen-14B-5k-1k",
    ckpt=f"Qwen/Qwen2.5-14B-Instruct",
    tp=4,
    method="recurrent",
    concur=256,
    env=ENV(RECURRENT_MAX_CONTEXT_LEN=100000000000, RECURRENT_CHUNK_SIZE=5000, RECURRENT_MAX_NEW=1024),
)

Qwen25_7B_5k_1k = Config(
    name="Qwen-7B-5k-1k",
    ckpt="Qwen/Qwen2.5-7B-Instruct",
    tp=4,
    method="recurrent",
    concur=256,
    env=ENV(RECURRENT_MAX_CONTEXT_LEN=100000000000, RECURRENT_CHUNK_SIZE=5000, RECURRENT_MAX_NEW=1024),
)

Qwen25_3B_5k_1k = Config(
    name="Qwen-3B-5k-1k",
    ckpt="Qwen/Qwen2.5-3B-Instruct",
    tp=4,
    method="recurrent",
    concur=256,
    env=ENV(RECURRENT_MAX_CONTEXT_LEN=100000000000, RECURRENT_CHUNK_SIZE=5000, RECURRENT_MAX_NEW=1024),
)

MemAgent_7B_vanilla_em = Config(
    name="MemAgent-7B-vanilla-em",
    ckpt=f"BytedTsinghua-SIA/RL-MemoryAgent-7B",
    tp=2,
    method="recurrent",
    concur=256,
    env=ENV(RECURRENT_MAX_CONTEXT_LEN=100000000000, RECURRENT_CHUNK_SIZE=5000, RECURRENT_MAX_NEW=1024),
)

CONFIGS = [
    MemAgent_7B_vanilla_em,
    R1_7B,
    Qwen25_7B_1M,
    Qwen25_7B_128k,
]
if REVERSED == '1':
    CONFIGS = CONFIGS[::-1]
# Reverse the test models

def run_ruler_hqa():
    for c in CONFIGS:
        task = RULER_HQA_TESTS
        if c.name.startswith("MemoryAgent"):
            task += RULER_HQA_TESTS_OVER_1M
        c.run(task, serve=True, force=False)


def run_ood_tasks():
    for c in CONFIGS:
        subset = [
            "niah_single_1",
            "niah_single_2",
            "niah_single_3",
            "niah_multikey_1",
            "niah_multikey_2",
            "niah_multikey_3",
            "niah_multivalue",
            "niah_multiquery",
            "vt",
            "fwe",
            "qa_1",
        ]
        lengths = [8192, 16384, 32768, 65536, 131072, 262144, 524288]
        task = [(s, l) for s in subset for l in lengths if not (s == "qa_1" and l > 262144)]
        c.run(task, serve=True, force=False)

def run_test_tasks():
    for c in CONFIGS:
        task = RULER_TEST_TASKS
        c.run(task, serve=True, force=False)

if __name__ == "__main__":
    print(f"{SERVE_PORT=}, {DASH_PORT=}, {MODEL_ROOT=}, {REVERSED=}")
    run_test_tasks()
