import pathlib
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field

import datascaler_datacomp.scale


class DatacompPrecision(StrEnum):
    amp = "amp"
    amp_bf16 = "amp_bf16"
    amp_bfloat16 = "amp_bfloat16"
    bf16 = "bf16"
    fp16 = "fp16"
    fp32 = "fp32"


class DatacompTrainParams(BaseModel):
    scale: Annotated[
        datascaler_datacomp.scale.DatacompScale,
        Field(
            description="Competition scale.",
        ),
    ]
    data_dir: Annotated[
        str,
        Field(
            description='Path to directory where the data is stored. Multiple paths can be used, separated by "::".',
        ),
    ]
    data_weights: Annotated[
        str | None,
        Field(
            "When using multiple data sources with webdataset and sampling with replacement, which weight to use for sampling the different data sources. "
            "Similar to --data-dir, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) "
            "By default, datapoints are sampled uniformly regardless of the dataset sizes."
        ),
    ] = None
    output_dir: Annotated[
        str,
        Field(
            description="Path to directory where outputs will be stored.",
        ),
    ]
    val_ratio: Annotated[
        float,
        Field("Ratio of validation set."),
    ] = 0.05
    val_batch_size: Annotated[
        int,
        Field(
            description="Val batch size.",
        ),
    ] = 128
    exp_name: Annotated[str | None, Field(description="Name of the experiment for logging.")] = None
    use_cached_shards: Annotated[
        bool,
        Field(
            description="If true, re-use the re-sharded data if possible.",
        ),
    ] = False
    wandb_project_name: Annotated[
        str,
        Field(
            description="Name of the project if logging with wandb.",
        ),
    ] = "datanet"
    workers: Annotated[int, Field(description="Number of workers for open_clip.")] = 4
    precision: Annotated[
        DatacompPrecision,
        Field(
            description="Floating point precision.",
        ),
    ] = DatacompPrecision.amp
    num_checkpoints: Annotated[
        int,
        Field(
            description="Number of times we save checkpoints during training.",
        ),
    ] = 5
    seed: Annotated[int, Field(description="Random seed.")] = 0
    report_to_wandb: Annotated[
        bool,
        Field(
            description="If True, report to wandb.",
        ),
    ] = False
    accum_freq: Annotated[
        int,
        Field(
            description="Update the model every --acum-freq steps.",
        ),
    ] = 1
    log_every_n_steps: Annotated[
        int,
        Field(
            description="Log every n steps to tensorboard/console/wandb.",
        ),
    ] = 10
    resume: Annotated[
        str,
        Field(
            description="Path to checkpoint to resume from (default: latest checkpoint in the training directory).",
        ),
    ] = "latest"
    imagenet_val: Annotated[
        str | None,
        Field(
            description="Optional path to imagenet val set for conducting zero shot evaluation.",
        ),
    ] = None
    # Strange argument in original DataComp, unsupported by downstream OpenCLIP training script
    # blur_field: Annotated[
    #     str | None,
    #     Field(
    #         description="Name of the field in the webdataset json files with bounding boxes to blur.",
    #     )
    # ] = None
    grad_clip_norm: Annotated[
        float | None,
        Field(
            description=None,
        ),
    ] = None
    save_frequency: Annotated[
        int,
        Field(
            description=None,
        ),
    ] = 0
    lr_scheduler: Annotated[
        str,
        Field(
            description="LR scheduler. One of: 'cosine', 'cosine_with_end', 'linear', 'linear_with_end', 'const' (constant), 'const-cooldown' (constant w/ cooldown),. Default: cosine",
        ),
    ] = "cosine"
    use_mup: Annotated[
        bool,
        Field(
            description="If True, use mup.",
        ),
    ] = False
    width_mult: Annotated[
        float,
        Field(
            description="Width multiplier.",
        ),
    ] = 1.0
    min_width_mult: Annotated[
        float,
        Field(
            description="Minimum value of width multiplier for base shape generation.",
        ),
    ] = 1.0
    attn_mult: Annotated[
        float,
        Field(
            description="attn is multiplied by sqrt(attn_mult)/head_dim.",
        ),
    ] = 1.0
    output_mult: Annotated[
        float,
        Field(
            description="output is multiplied by sqrt(output_mult/d_model).",
        ),
    ] = 1.0
    subset_file: Annotated[
        str | None,
        Field(
            description="Path to trie pickle file.",
        ),
    ] = None
    learning_rate: Annotated[
        float,
        Field(
            description="Learning rate.",
        ),
    ] = 5e-4
    train_num_samples: Annotated[
        int,
        Field(
            description="Number of samples seen during training.",
        ),
    ] = 12_800_000
    model_arch: Annotated[
        str,
        Field(
            description="Architecture of the trained CLIP model. (ex: ViT-B-32, ViT-B-16)",
            examples=["ViT-B-32", "ViT-B-16"],
        ),
    ] = "ViT-B-32"


class DatacompEvaluateParams(BaseModel):
    scale: Annotated[
        datascaler_datacomp.scale.DatacompScale,
        Field(
            description="Competition scale.",
        ),
    ]
    train_output_dir: Annotated[
        pathlib.Path,
        Field(
            description="Path to output directory from training.",
        ),
    ]
    task_dir: Annotated[
        pathlib.Path,
        Field(
            description="Path to directory containing evaluation tasks.",
        ),
    ]
    model_checkpoint: Annotated[
        str,
        Field(
            description="Name of model checkpoint.",
        ),
    ]
    model_arch: Annotated[
        str,
        Field(
            description="Architecture of the trained CLIP model. (ex: ViT-B-32, ViT-B-16)",
            examples=["ViT-B-32", "ViT-B-16"],
        ),
    ] = "ViT-B-32"
    output_dir: Annotated[
        pathlib.Path | None,
        Field(
            description="Path to output directory to use for evaluation. If nothing is passed, use the training output dir.",
        ),
    ] = None
    data_dir: Annotated[
        str | None,
        Field(
            description="(Optional) Path to directory containing downloaded evaluation datasets.",
        ),
    ] = None
    batch_size: Annotated[
        int,
        Field(
            description="Batch size.",
        ),
    ] = 64
    use_mup: Annotated[
        bool,
        Field(
            description="If True, use mup.",
        ),
    ] = False
    num_workers: Annotated[
        int,
        Field(
            description="Number of workers.",
        ),
    ] = 64
    width_mult: Annotated[
        float,
        Field(
            description="Width multiplier.",
        ),
    ] = 1.0
    attn_mult: Annotated[
        float,
        Field(
            description="attn is multiplied by sqrt(attn_mult)/head_dim.",
        ),
    ] = 1.0
    output_mult: Annotated[
        float,
        Field(
            description="output is multiplied by sqrt(output_mult/d_model).",
        ),
    ] = 1.0
    compute_loss: Annotated[
        float,
        Field(
            description="If 1.0, compute zero-shot contrastive loss.",
        ),
    ] = 0.0
