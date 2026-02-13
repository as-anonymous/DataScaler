# Main branching point for evaluating on different datasets
from datascaler_datacomp.eval_utils.fairness_eval import (
    evaluate_dollar_street_dataset,
    evaluate_geode_dataset,
)
from datascaler_datacomp.eval_utils.retr_eval import evaluate_retrieval_dataset
from datascaler_datacomp.eval_utils.wds_eval import evaluate_webdataset
from datascaler_datacomp.eval_utils.wilds_eval import evaluate_wilds_dataset
from datascaler_datacomp.eval_utils.wino_eval import evaluate_winogavil_dataset


def evaluate_model(
    task_key,
    train_info,
    data_root,
    dataset_size,
    train_output_dir,
    use_mup=False,
    batch_size=64,
    num_workers=64,
    width_mult=1.0,
    attn_mult=1.0,
    output_mult=1.0,
    compute_loss=0.0,
):
    if task_key.startswith("retrieval/"):
        metrics = evaluate_retrieval_dataset(
            task_key,
            train_info["scale_config"]["model"],
            train_info["checkpoint"],
            data_root=data_root,
            train_output_dir=train_output_dir,
            use_mup=use_mup,
            batch_size=batch_size,
            num_workers=num_workers,
            width_mult=width_mult,
            attn_mult=attn_mult,
            output_mult=output_mult,
            compute_loss=compute_loss,
        )
    elif task_key.startswith("wilds/"):
        metrics = evaluate_wilds_dataset(
            task_key,
            train_info["scale_config"]["model"],
            train_info["checkpoint"],
            data_root=data_root,
            dataset_len=dataset_size,
            train_output_dir=train_output_dir,
            use_mup=use_mup,
            batch_size=batch_size,
            num_workers=num_workers,
            width_mult=width_mult,
            attn_mult=attn_mult,
            output_mult=output_mult,
        )
    elif task_key.startswith("fairness/"):
        eval_fn = {
            "fairness/dollar_street": evaluate_dollar_street_dataset,
            "fairness/geode": evaluate_geode_dataset,
        }.get(task_key)
        if eval_fn is not None:
            metrics = eval_fn(
                task_key,
                train_info["scale_config"]["model"],
                train_info["checkpoint"],
                data_root=data_root,
                dataset_len=dataset_size,
                train_output_dir=train_output_dir,
                use_mup=use_mup,
                batch_size=batch_size,
                num_workers=num_workers,
                width_mult=width_mult,
                attn_mult=attn_mult,
                output_mult=output_mult,
            )
        else:
            metrics = {}
    elif task_key.startswith("misc/"):
        if task_key == "misc/winogavil":
            metrics = evaluate_winogavil_dataset(
                train_info["scale_config"]["model"],
                train_info["checkpoint"],
                data_root=data_root,
                train_output_dir=train_output_dir,
                use_mup=use_mup,
                batch_size=batch_size,
                num_workers=num_workers,
                width_mult=width_mult,
                attn_mult=attn_mult,
                output_mult=output_mult,
            )
        else:
            metrics = {}
    else:
        metrics = evaluate_webdataset(
            task_key,
            train_info["scale_config"]["model"],
            train_info["checkpoint"],
            data_root=data_root,
            dataset_len=dataset_size,
            train_output_dir=train_output_dir,
            use_mup=use_mup,
            batch_size=batch_size,
            num_workers=num_workers,
            width_mult=width_mult,
            attn_mult=attn_mult,
            output_mult=output_mult,
            compute_loss=compute_loss,
        )
    return metrics
