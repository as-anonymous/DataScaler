import json
import pathlib
import pickle
import re
import time
import warnings

import numpy as np
import yaml

import datascaler_datacomp.eval_utils.main
import datascaler_datacomp.params

warnings.filterwarnings("ignore", message="Length of IterableDataset")


def save_info_file(params, epoch_num):
    info = {
        "scale_config": {"model": params.model_arch},
        "checkpoint": params.train_output_dir / params.model_checkpoint,
    }
    with open(params.output_dir / f"info_{epoch_num}.pkl", "wb") as f:
        pickle.dump(info, f)


def main(params: datascaler_datacomp.params.DatacompEvaluateParams):
    if params.output_dir is None:
        params.output_dir = params.train_output_dir

    params.train_output_dir = params.output_dir

    # Generate barebones info.pkl
    pathlib.Path.mkdir(params.output_dir, parents=True, exist_ok=True)

    match = re.match(r"epoch_(\d+)\.pt", params.model_checkpoint)
    if not match:
        raise ValueError("Invalid checkpoint name format")
    epoch_num = match.group(1)
    save_info_file(params, epoch_num)

    # Read training information
    train_info_filename = params.train_output_dir / f"info_{epoch_num}.pkl"
    train_info = pickle.load(open(train_info_filename, "rb"))

    results_filename = params.output_dir / f"eval_results_{epoch_num}.jsonl"

    # Get list of datasets
    with open(params.task_dir) as f:
        tasks = yaml.safe_load(f)

    # Check for cached results
    results = {}
    cached_train_info_filename = params.output_dir / f"info_{epoch_num}.pkl"
    if params.output_dir.exists() and cached_train_info_filename.exists():
        # If the output directory already exists, the training information should match.
        cached_train_info = pickle.load(open(cached_train_info_filename, "rb"))
        error_message = (
            "Error: output directory exists, but the training configs do not match. "
            "If you are re-using an output directory for evals, please be sure that "
            "the training output directory is consistent."
        )
        assert cached_train_info == train_info, error_message

        # Read existing results
        if results_filename.exists():
            with open(results_filename, "r") as f:
                lines = [json.loads(s) for s in f.readlines()]
                for line in lines:
                    if line["key"] not in tasks:
                        continue
                    results[line["dataset"]] = line
            print(f"Found {len(results)} eval result(s) in {results_filename}.")
    else:
        pathlib.Path.mkdir(params.output_dir, parents=True, exist_ok=True)
        pickle.dump(train_info, open(cached_train_info_filename, "wb"))

    try:
        exists = pathlib.Path(train_info["checkpoint"]).exists()
    except Exception:
        exists = False
    if not exists and params.model_checkpoint is None:
        print(
            "Warning, did not find or could not read checkpoint at",
            train_info["checkpoint"],
        )
        default_checkpoint_name = params.train_output_dir / "checkpoints" / "epoch_latest.pt"
        print("Defaulting to", default_checkpoint_name)
        train_info["checkpoint"] = default_checkpoint_name

    print("Evaluating")

    starttime = int(time.time())

    for task_key in tasks:
        task_name = tasks[task_key].get("name", task_key)
        if task_name in results:
            print(f"Skipping {task_name} since results are already in {results_filename}")
        else:
            print(f"Evaluating on {task_name}")
            metrics = datascaler_datacomp.eval_utils.main.evaluate_model(
                task_key,
                train_info,
                params.data_dir,
                tasks[task_key].get("size"),
                train_output_dir=params.train_output_dir,
                use_mup=params.use_mup,
                batch_size=params.batch_size,
                num_workers=params.num_workers,
                width_mult=params.width_mult,
                attn_mult=params.attn_mult,
                output_mult=params.output_mult,
                compute_loss=params.compute_loss,
            )
            metrics["main_metric"] = metrics.get(  # type: ignore
                tasks[task_key].get("main_metric", "acc1")
            )
            results[task_name] = {
                "key": task_key,
                "dataset": task_name,
                "metrics": metrics,
            }
            with open(results_filename, "a+") as f:
                f.write(json.dumps(results[task_name]) + "\n")

        if results[task_name]["metrics"]["main_metric"] is not None:
            print(f"Score: {results[task_name]['metrics']['main_metric']:.4f}")
        else:
            print("Score: No summary metric")

    elapsed = int(time.time()) - starttime
    print(
        f"Evaluation time: {elapsed // 3600} hour(s) {elapsed % 3600 // 60} minute(s) {elapsed % 60} second(s)"
    )
    print()
    print("=== Final results ===")
    for line in results.values():
        print(f"{line['dataset']}: {line['metrics']['main_metric']}")

    print("=====================")
    average = np.mean(
        [
            val["metrics"]["main_metric"]
            for val in results.values()
            if val["metrics"]["main_metric"] is not None
        ]
    )
    print(f"Average: {average}")
