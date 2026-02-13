import ast
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import BaseModel, BeforeValidator, Field, model_validator


def parse_aug_cfg(raw_aug_cfg: list[str] | None) -> dict[str, Any]:
    if not raw_aug_cfg:
        return {}
    aug_cfg = {}
    for raw_value in raw_aug_cfg:
        key, value = raw_value.split("=")
        try:
            aug_cfg[key] = ast.literal_eval(value)
        except ValueError:
            aug_cfg[key] = value
    return aug_cfg


class DatasetType(StrEnum):
    webdataset = "webdataset"
    csv = "csv"
    synthetic = "synthetic"
    auto = "auto"


class Precision(StrEnum):
    amp = "amp"
    amp_bf16 = "amp_bf16"
    amp_bfloat16 = "amp_bfloat16"
    bf16 = "bf16"
    fp16 = "fp16"
    pure_bf16 = "pure_bf16"
    pure_fp16 = "pure_fp16"
    fp32 = "fp32"


class ImageInterpolation(StrEnum):
    bicubic = "bicubic"
    bilinear = "bilinear"
    random = "random"


class ImageResizeMode(StrEnum):
    shortest = "shortest"
    longest = "longest"
    squash = "squash"


class RemoteSyncProtocol(StrEnum):
    s3 = "s3"
    fsspec = "fsspec"


class OpenClipTrainParams(BaseModel):
    train_data: Annotated[
        str | list | None,
        Field(
            description="Path to file(s) with training data. When using webdataset, multiple datasources can be combined using the `::` separator.",
        ),
    ] = None
    train_data_upsampling_factors: Annotated[
        str | None,
        Field(
            description=(
                "When using multiple data sources with webdataset and sampling with replacement, this can be used to upsample specific data sources. "
                "Similar to --train-data, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) "
                "By default, datapoints are sampled uniformly regardless of the dataset sizes."
            ),
        ),
    ] = None
    val_data: Annotated[
        str | list | None,
        Field(
            description="Path to file(s) with validation data",
        ),
    ] = None
    train_num_samples: Annotated[
        int | None,
        Field(
            description="Number of samples in dataset. Required for webdataset if not available in info file.",
        ),
    ] = None
    val_num_samples: Annotated[
        int | None,
        Field(
            description="Number of samples in dataset. Useful for webdataset if not available in info file.",
        ),
    ] = None
    dataset_type: Annotated[
        DatasetType,
        Field(
            description="Which type of dataset to process.",
        ),
    ] = DatasetType.auto
    dataset_resampled: Annotated[
        bool,
        Field(
            description="Whether to use sampling with replacement for webdataset shard selection.",
        ),
    ] = False
    csv_separator: Annotated[
        str,
        Field(
            description="For csv-like datasets, which separator to use.",
        ),
    ] = "\t"
    csv_img_key: Annotated[
        str,
        Field(
            description="For csv-like datasets, the name of the key for the image paths.",
        ),
    ] = "filepath"
    csv_caption_key: Annotated[
        str,
        Field(
            description="For csv-like datasets, the name of the key for the captions.",
        ),
    ] = "title"
    imagenet_val: Annotated[
        str | None,
        Field(
            description="Path to imagenet val set for conducting zero shot evaluation.",
        ),
    ] = None
    imagenet_v2: Annotated[
        str | None,
        Field(
            description="Path to imagenet v2 for conducting zero shot evaluation.",
        ),
    ] = None
    cache_dir: Annotated[
        str | None,
        Field(
            description="Override system default cache path for model & tokenizer file downloads.",
        ),
    ] = None
    logs: Annotated[
        str,
        Field(
            description="Where to store tensorboard logs. Use None to avoid storing logs.",
        ),
    ] = "./logs/"
    log_local: Annotated[
        bool,
        Field(
            description="log files on local master, otherwise global master only.",
        ),
    ] = False
    name: Annotated[
        str | None,
        Field(
            description="Optional identifier for the experiment when storing logs. Otherwise use current time.",
        ),
    ] = None
    workers: Annotated[
        int,
        Field(
            description="Number of dataloader workers per GPU.",
        ),
    ] = 4
    batch_size: Annotated[
        int,
        Field(
            description="Batch size per GPU.",
        ),
    ] = 64
    val_batch_size: Annotated[
        int,
        Field(
            description="Val batch size per GPU.",
        ),
    ] = 64
    epochs: Annotated[
        int,
        Field(
            description="Number of epochs to train for.",
        ),
    ] = 32
    epochs_cooldown: Annotated[
        int | None,
        Field(
            description="When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs onwards.",
        ),
    ] = None
    lr: Annotated[
        float | None,
        Field(
            description="Learning rate.",
        ),
    ] = None
    beta1: Annotated[
        float | None,
        Field(
            description="Adam beta 1.",
        ),
    ] = None
    beta2: Annotated[
        float | None,
        Field(
            description="Adam beta 2.",
        ),
    ] = None
    eps: Annotated[
        float | None,
        Field(
            description="Adam epsilon.",
        ),
    ] = None
    wd: Annotated[
        float,
        Field(
            description="Weight decay.",
        ),
    ] = 0.2
    momentum: Annotated[
        float | None,
        Field(
            description="Momentum (for timm optimizers).",
        ),
    ] = None
    warmup: Annotated[
        int,
        Field(
            description="Number of steps to warmup for.",
        ),
    ] = 10000
    opt: Annotated[
        str,
        Field(
            description="Which optimizer to use. Choices are ['adamw', or any timm optimizer 'timm/{opt_name}'].",
        ),
    ] = "adamw"
    use_bn_sync: Annotated[
        bool,
        Field(
            description="Whether to use batch norm sync.",
        ),
    ] = False
    skip_scheduler: Annotated[
        bool,
        Field(
            description="Use this flag to skip the learning rate decay.",
        ),
    ] = False
    lr_scheduler: Annotated[
        str,
        Field(
            description="LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown' (constant w/ cooldown). Default: cosine",
        ),
    ] = "cosine"
    lr_cooldown_end: Annotated[
        float,
        Field(
            description="End learning rate for cooldown schedule. Default: 0",
        ),
    ] = 0.0
    lr_cooldown_power: Annotated[
        float,
        Field(
            description="Power for polynomial cooldown schedule. Default: 1.0 (linear decay)",
        ),
    ] = 1.0
    save_frequency: Annotated[
        int,
        Field(
            description="How often to save checkpoints.",
        ),
    ] = 1
    save_most_recent: Annotated[
        bool,
        Field(
            description="Always save the most recent model trained to epoch_latest.pt.",
        ),
    ] = False
    zeroshot_frequency: Annotated[
        int,
        Field(
            description="How often to run zero shot.",
        ),
    ] = 2
    val_frequency: Annotated[
        int,
        Field(
            description="How often to run evaluation with val data.",
        ),
    ] = 1
    resume: Annotated[
        str | None,
        Field(
            description="path to latest checkpoint (default: none)",
        ),
    ] = None
    precision: Annotated[
        Precision,
        Field(
            description="Floating point precision.",
        ),
    ] = Precision.amp
    model: Annotated[
        str,
        Field(
            description="Name of the vision backbone to use.",
        ),
    ] = "RN50"
    pretrained: Annotated[
        str,
        Field(
            description="Use a pretrained CLIP model weights with the specified tag or file path.",
        ),
    ] = ""
    pretrained_image: Annotated[
        bool,
        Field(
            description="Load imagenet pretrained weights for image tower backbone if available.",
        ),
    ] = False
    lock_image: Annotated[
        bool,
        Field(
            description="Lock full image tower by disabling gradients.",
        ),
    ] = False
    lock_image_unlocked_groups: Annotated[
        int,
        Field(
            description="Leave last n image tower layer groups unlocked.",
        ),
    ] = 0
    lock_image_freeze_bn_stats: Annotated[
        bool,
        Field(
            description="Freeze BatchNorm running stats in image tower for any locked layers.",
        ),
    ] = False
    image_mean: Annotated[
        tuple[float] | None,
        Field(
            description="Override default image mean value of dataset",
        ),
    ] = None
    image_std: Annotated[
        tuple[float] | None,
        Field(
            description="Override default image std deviation of of dataset",
        ),
    ] = None
    image_interpolation: Annotated[
        str | None,
        Field(
            description="Override default image resize interpolation",
        ),
    ] = None
    image_resize_mode: Annotated[
        str | None,
        Field(
            description="Override default image resize (& crop) mode during inference",
        ),
    ] = None
    aug_cfg: Annotated[
        dict[str, Any],  # TODO open_clip.transform.AugmentationCfg
        Field(
            description=None,
        ),
    ] = {}
    grad_checkpointing: Annotated[
        bool,
        Field(
            description="Enable gradient checkpointing.",
        ),
    ] = False
    local_loss: Annotated[
        bool,
        Field(
            description="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)",
        ),
    ] = False
    gather_with_grad: Annotated[
        bool,
        Field(
            description="enable full distributed gradient for feature gather",
        ),
    ] = False
    force_image_size: Annotated[
        int | tuple[int, int] | None,
        Field(
            description="Override default image size",
        ),
        BeforeValidator(lambda el: el[0] if isinstance(el, list) and len(el) == 1 else el),
    ] = None
    force_quick_gelu: Annotated[
        bool,
        Field(
            description="Force use of QuickGELU activation for non-OpenAI transformer models.",
        ),
    ] = False
    force_patch_dropout: Annotated[
        float | None,
        Field(
            description="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper",
        ),
    ] = None
    force_custom_text: Annotated[
        bool,
        Field(
            description="Force use of CustomTextCLIP model (separate text-tower).",
        ),
    ] = False
    torchscript: Annotated[
        bool,
        Field(
            description="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
        ),
    ] = False
    torchcompile: Annotated[
        bool,
        Field(
            description="torch.compile() the model, requires pytorch 2.0 or later.",
        ),
    ] = False
    trace: Annotated[
        bool,
        Field(
            description="torch.jit.trace the model for inference / eval only",
        ),
    ] = False
    accum_freq: Annotated[
        int,
        Field(
            description="Update the model every --acum-freq steps.",
        ),
    ] = 1
    device: Annotated[
        str,
        Field(
            description="Accelerator to use.",
        ),
    ] = "cuda"
    dist_url: Annotated[
        str | None,
        Field(
            description="url used to set up distributed training",
        ),
    ] = None
    dist_backend: Annotated[
        str | None,
        Field(
            description='distributed backend. "nccl" for GPU, "hccl" for Ascend NPU',
        ),
    ] = None
    report_to: Annotated[
        str,
        Field(
            description="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']",
        ),
    ] = ""
    wandb_notes: Annotated[
        str,
        Field(
            description="Notes if logging with wandb",
        ),
    ] = ""
    wandb_project_name: Annotated[
        str,
        Field(
            description="Name of the project if logging with wandb.",
        ),
    ] = "open-clip"
    debug: Annotated[
        bool,
        Field(
            description="If true, more information is logged.",
        ),
    ] = False
    copy_codebase: Annotated[
        bool,
        Field(
            description="If true, we copy the entire base on the log directory, and execute from there.",
        ),
    ] = False
    horovod: Annotated[
        bool,
        Field(
            description="Use horovod for distributed training.",
        ),
    ] = False
    ddp_static_graph: Annotated[
        bool,
        Field(
            description="Enable static graph optimization for DDP in PyTorch >= 1.11.",
        ),
    ] = False
    no_set_device_rank: Annotated[
        bool,
        Field(
            description="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
        ),
    ] = False
    seed: Annotated[
        int,
        Field(
            description="Default random seed.",
        ),
    ] = 0
    grad_clip_norm: Annotated[
        float | None,
        Field(
            description="Gradient clip.",
        ),
    ] = None
    lock_text: Annotated[
        bool,
        Field(
            description="Lock full text tower by disabling gradients.",
        ),
    ] = False
    lock_text_unlocked_layers: Annotated[
        int,
        Field(
            description="Leave last n text tower layer groups unlocked.",
        ),
    ] = 0
    lock_text_freeze_layer_norm: Annotated[
        bool,
        Field(
            description="Freeze LayerNorm running stats in text tower for any locked layers.",
        ),
    ] = False
    log_every_n_steps: Annotated[
        int,
        Field(
            description="Log every n steps to tensorboard/console/wandb.",
        ),
    ] = 10
    coca_caption_loss_weight: Annotated[
        float,
        Field(
            description="Weight assigned to caption loss in CoCa.",
        ),
    ] = 2.0
    coca_contrastive_loss_weight: Annotated[
        float,
        Field(
            description="Weight assigned to contrastive loss when training CoCa.",
        ),
    ] = 1.0
    remote_sync: Annotated[
        str | None,
        Field(
            description="Optinoally sync with a remote path specified by this arg",
        ),
    ] = None
    remote_sync_frequency: Annotated[
        int,
        Field(
            description="How frequently to sync to a remote directly if --remote-sync is not None.",
        ),
    ] = 300
    remote_sync_protocol: Annotated[
        RemoteSyncProtocol,
        Field(
            description="How to do the remote sync backup if --remote-sync is not None.",
        ),
    ] = RemoteSyncProtocol.s3
    delete_previous_checkpoint: Annotated[
        bool,
        Field(
            description="If true, delete previous checkpoint after storing a new one.",
        ),
    ] = False
    distill_model: Annotated[
        str | None,
        Field(
            description="Which model arch to distill from, if any.",
        ),
    ] = None
    distill_pretrained: Annotated[
        str | None,
        Field(
            description="Which pre-trained weights to distill from, if any.",
        ),
    ] = None
    use_bnb_linear: Annotated[
        str | None,
        Field(
            description="Replace the network linear layers from the bitsandbytes library. Allows int8 training/inference, etc.",
        ),
    ] = None
    siglip: Annotated[
        bool,
        Field(
            description="Use SigLip (sigmoid) loss.",
        ),
    ] = False
    loss_dist_impl: Annotated[
        str | None,
        Field(
            description="A string to specify a specific distributed loss implementation.",
        ),
    ] = None
    use_mup: Annotated[
        bool,
        Field(
            description="Use MUP.",
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

    @model_validator(mode="after")
    def set_default_values(self) -> Self:
        # open_clip_train.params.get_default_params
        if "timm" not in self.opt:
            # get default parameters from paper (https://arxiv.org/pdf/2103.00020.pdf)
            if "vit" in self.model.lower():
                self.lr = 5.0e-4 if self.lr is None else self.lr
                self.beta1 = 0.9 if self.beta1 is None else self.beta1
                self.beta2 = 0.98 if self.beta2 is None else self.beta2
                self.eps = 1.0e-6 if self.eps is None else self.eps
            else:
                self.lr = 5.0e-4 if self.lr is None else self.lr
                self.beta1 = 0.9 if self.beta1 is None else self.beta1
                self.beta2 = 0.999 if self.beta2 is None else self.beta2
                self.eps = 1.0e-8 if self.eps is None else self.eps
        return self


class OpenClipTrainInternalParams(OpenClipTrainParams):
    """Parameters handled internally by OpenCLIP"""

    rank: int  # global rank
    local_rank: int
    world_size: int
    device: str = "cuda"
    distributed: bool

    name: str = "default_name"
    log_path: str | None = None
    log_level: int | str = "INFO"
    wandb: bool = False
    tensorboard: bool = False
    checkpoint_path: str = "./logs/default_name/checkpoints"
    tensorboard_path: str = ""  # empty string indicates no tensorboard
    resume: str | None = None  # special handling for "latest"
    distill: bool = False
    force_image_size: int | tuple[int, int] | None = None
    save_logs: bool = False
    train_sz: int = -1
    val_sz: int = -1
