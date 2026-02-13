import json
import os
import subprocess
import sys
import time
from threading import Semaphore, Thread

import yaml

TRAIN_DIR = sys.argv[1]
TASK_DIR = sys.argv[2]
DATA_DIR = sys.argv[3]
MIN_EPOCH = int(sys.argv[4])
MAX_EPOCH = int(sys.argv[5])
WIDTH_MULT = float(sys.argv[6])
NUM_WORKERS = int(sys.argv[7])
SCALE = sys.argv[8]
USE_MUP = sys.argv[9]
ATTN_MULT = float(sys.argv[10])
OUTPUT_MULT = float(sys.argv[11])
MAX_PARALLEL = int(sys.argv[12])
COMPUTE_LOSS = float(sys.argv[13])
MODEL_ARCH = sys.argv[14]

if USE_MUP == "true":
    MUP_FLAG = "--use-mup"
else:
    MUP_FLAG = "--no-use-mup"

with open(TASK_DIR) as file:
    tasks = yaml.safe_load(file)
task_set = set(tasks.keys())

if WIDTH_MULT in [0.5, 1.0]:
    BATCH_SIZE = 1024
elif WIDTH_MULT in [2.0]:
    BATCH_SIZE = 512
else:
    BATCH_SIZE = 2048

models = [f"epoch_{i}.pt" for i in range(MIN_EPOCH, MAX_EPOCH + 1)]
completed_models = set()
semaphore = Semaphore(MAX_PARALLEL)


def is_output_valid(log_file_name):
    result_set = set()
    if os.path.isfile(log_file_name):
        with open(log_file_name, "r") as f:
            for line in f:
                each_result = json.loads(line)
                result_set.add(each_result["key"])

    return result_set == task_set


def run_and_check(MODEL_NAME):
    with semaphore:
        FULL_TRAIN_DIR = f"{TRAIN_DIR}/checkpoints"

        cmd = f"datascaler-datacomp evaluate \
                --scale {SCALE} \
                --task-dir {TASK_DIR} \
                --data-dir {DATA_DIR} \
                --batch-size {BATCH_SIZE} \
                --num-workers {NUM_WORKERS} \
                --width-mult {WIDTH_MULT} \
                --train-output-dir {FULL_TRAIN_DIR} \
                --model-checkpoint {MODEL_NAME} \
                --attn-mult {ATTN_MULT} \
                --output-mult {OUTPUT_MULT} \
                --compute-loss {COMPUTE_LOSS} \
                --model-arch {MODEL_ARCH} \
                {MUP_FLAG}"

        print(f"Running {MODEL_NAME}")
        subprocess.run(cmd, shell=True)

        model_id = MODEL_NAME.split("_")[1].split(".")[0]
        log_file_name = f"{FULL_TRAIN_DIR}/eval_results_{model_id}.jsonl"
        if is_output_valid(log_file_name):
            print(f"✅  {MODEL_NAME} succeeded")
            completed_models.add(MODEL_NAME)
        else:
            print(f"❌  {MODEL_NAME} failed — will retry")


while len(completed_models) < len(models):
    threads = []
    for model in models:
        if model not in completed_models:
            t = Thread(target=run_and_check, args=(model,))
            t.start()
            threads.append(t)
    for t in threads:
        t.join()

    remaining = len(models) - len(completed_models)
    print(f"🔁  Remaining: {remaining} models — retrying in 2s...\n")
    if remaining > 0:
        time.sleep(2)

print("All models successfully evaluated!")
