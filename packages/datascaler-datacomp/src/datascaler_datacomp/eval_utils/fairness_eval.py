import torch
from wilds.common.grouper import CombinatorialGrouper

from datascaler_datacomp.eval_utils.wilds_eval import (
    EVALUATORS,
    MyAccuracy,
    WILDSEvaluator,
    create_metadata_loader,
    evaluate_webdataset,
)

# Dollar Street


class TopKAccuracy(MyAccuracy):
    def __init__(self, prediction_fn=None, name=None):
        if name is None:
            name = "acc_topk"
        super().__init__(name=name)

    def _compute_element_wise(self, y_pred, y_true):
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)  # typee: ignore
        return (y_pred == y_true.unsqueeze(-1)).any(-1).float()


class DollarStreetEvaluator(WILDSEvaluator):
    def __init__(self, metadata):
        super().__init__(metadata)
        self._metadata_fields = ["income_ds", "income_meta", "region"]
        self._eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=["income_ds"])

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        metric = TopKAccuracy(prediction_fn=prediction_fn, name="acc_top5")
        return self.standard_group_eval(metric, self._eval_grouper, y_pred, y_true, metadata)


EVALUATORS["fairness/dollar_street"] = DollarStreetEvaluator


def evaluate_dollar_street_dataset(
    task,
    model_arch,
    model_path,
    data_root,
    dataset_len,
    train_output_dir,
    batch_size=64,
    num_workers=4,
    use_mup=False,
    width_mult=1.0,
    attn_mult=1.0,
    output_mult=1.0,
):
    """Evaluate CLIP model on Dollar Street classification task."""

    # Evaluate
    metrics, y_pred, y_target = evaluate_webdataset(
        task.replace("fairness/", ""),
        model_arch,
        model_path,
        data_root,
        dataset_len,
        train_output_dir,
        batch_size,
        num_workers,
        return_preds=True,
        return_topk=5,  # type: ignore
        use_mup=use_mup,
        width_mult=width_mult,
        attn_mult=attn_mult,
        output_mult=output_mult,
    )

    # Load additional metadata
    print("Reading additional metadata")
    metadata_loader = create_metadata_loader(
        task.replace("fairness/", ""), data_root, dataset_len, batch_size, num_workers
    )
    # Check metadata
    y_array = []
    metadata_array = []
    for label, metadata in metadata_loader:
        y_array.append(label)
        metadata_array.append(metadata)
    # assert (y_target == np.array(y_array)).all(), "Labels do not match"
    metadata = torch.cat(metadata_array)

    # Compute additional metrics
    evaluator = EVALUATORS[task](metadata)
    metrics.update(evaluator.eval(y_pred, y_target, metadata)[0])  # type: ignore

    return metrics


# GeoDE


class GeoDEEvaluator(WILDSEvaluator):
    def __init__(self, metadata):
        super().__init__(metadata)
        self._metadata_fields = ["region", "country"]
        self._eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=["region"])

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        metric = MyAccuracy(prediction_fn=prediction_fn)
        return self.standard_group_eval(metric, self._eval_grouper, y_pred, y_true, metadata)


EVALUATORS["fairness/geode"] = GeoDEEvaluator


def evaluate_geode_dataset(
    task,
    model_arch,
    model_path,
    data_root,
    dataset_len,
    train_output_dir,
    batch_size=64,
    num_workers=4,
    use_mup=False,
    width_mult=1.0,
    attn_mult=1.0,
    output_mult=1.0,
):
    """Evaluate CLIP model on GeoDE classification task."""

    # Evaluate
    metrics, y_pred, y_target = evaluate_webdataset(
        task.replace("fairness/", ""),
        model_arch,
        model_path,
        data_root,
        dataset_len,
        train_output_dir,
        batch_size,
        num_workers,
        return_preds=True,
        use_mup=use_mup,
        width_mult=width_mult,
        attn_mult=attn_mult,
        output_mult=output_mult,
    )

    # Load additional metadata
    print("Reading additional metadata")
    metadata_loader = create_metadata_loader(
        task.replace("fairness/", ""), data_root, dataset_len, batch_size, num_workers
    )
    # Check metadata
    y_array = []
    metadata_array = []
    for label, metadata in metadata_loader:
        y_array.append(label)
        metadata_array.append(metadata)
    # assert (y_target == np.array(y_array)).all(), "Labels do not match"
    metadata = torch.cat(metadata_array)

    # Compute additional metrics
    evaluator = EVALUATORS[task](metadata)
    metrics.update(evaluator.eval(y_pred, y_target, metadata)[0])  # type: ignore

    return metrics
