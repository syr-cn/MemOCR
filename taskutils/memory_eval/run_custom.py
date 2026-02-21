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

RULER_TEST_TASKS = [
    f"eval_{ds}_{num_docs}" \
    for ds in ['hotpotqa', '2wikimultihopqa', 'nq', 'triviaqa'] \
    for num_docs in [50, 200, 800]
]

class Config:
    SERVE_TAG = "__serve"

    def __init__(self, name, ckpt, tp, method, env, concur=1024):
        self.name = name
        self.ckpt = ckpt

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
        # serve_script = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "serve/llm070.py"))
        # cmd = f"python {serve_script} --model {self.ckpt} --tp {self.tp}"
        GPU_NUM = 8
        self.dp = int(GPU_NUM/int(self.tp))
        cmd = f"python3 -m vllm.entrypoints.openai.api_server --model {self.ckpt} --tensor-parallel-size {self.tp} --data-parallel-size {self.dp} --served-model-name {Path(self.ckpt).name}"
        # cmd = f"python3 -m sglang.launch_server --model-path {self.ckpt} --tensor-parallel-size {self.tp} --served-model-name {Path(self.ckpt).name} --port {SERVE_PORT} --data-parallel-size {self.pp}"
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
            elif test in RULER_TEST_TASKS_OVER_1M:
                assert False, "Not implemented"
                cmd = f"""python test_qa_over1m.py --model {self.model}\
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
            print(f'Running Command: {cmd}')
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


exp_name = os.getenv("EXP_NAME")
model_name = os.getenv("MODEL_NAME")
model_tp = os.getenv("MODEL_TP")


if "_html_" in model_name.lower():
    method_name = "memocr_html"
elif "_md_" in model_name.lower() or '_triple_' in model_name.lower():
    method_name = "memocr_md"
else:
    method_name = "memocr_md"
print(f'using method {method_name}')
    # raise ValueError(f"Invalid model name: {model_name}")

MemoryAgent_Custom = Config(
    name=exp_name,
    ckpt=model_name,
    tp=model_tp,
    method=method_name,
    concur=32,
    env=ENV(RECURRENT_MAX_CONTEXT_LEN=100000000000, RECURRENT_CHUNK_SIZE=5000, RECURRENT_MAX_NEW=2048),
)

CONFIGS = [
    MemoryAgent_Custom,
]

def run_ruler_hqa():
    for c in CONFIGS:
        task = RULER_HQA_TESTS[:]
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
    print(f"{SERVE_PORT=}, {DASH_PORT=}")
    run_test_tasks()
