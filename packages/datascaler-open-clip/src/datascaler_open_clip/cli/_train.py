from typing import Annotated

import typer

import datascaler_open_clip.params
import datascaler_open_clip.train

app = typer.Typer()


@app.command(name="train")
def train_model(
    train_data: Annotated[
        str | None,
        typer.Option(
            help="Path to file(s) with training data. When using webdataset, multiple datasources can be combined using the `::` separator.",
        ),
    ] = None,
    train_data_upsampling_factors: Annotated[
        str | None,
        typer.Option(
            help=(
                "When using multiple data sources with webdataset and sampling with replacement, this can be used to upsample specific data sources. "
                "Similar to --train-data, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) "
                "By default, datapoints are sampled uniformly regardless of the dataset sizes."
            )
        ),
    ] = None,
    val_data: Annotated[
        str | None,
        typer.Option(
            help="Path to file(s) with validation data",
        ),
    ] = None,
    train_num_samples: Annotated[
        int | None,
        typer.Option(
            help="Number of samples in dataset. Required for webdataset if not available in info file.",
        ),
    ] = None,
    val_num_samples: Annotated[
        int | None,
        typer.Option(
            help="Number of samples in dataset. Useful for webdataset if not available in info file.",
        ),
    ] = None,
    dataset_type: Annotated[
        datascaler_open_clip.params.DatasetType,
        typer.Option(
            help="Which type of dataset to process.",
        ),
    ] = datascaler_open_clip.params.DatasetType.auto,
    dataset_resampled: Annotated[
        bool,
        typer.Option(
            help="Whether to use sampling with replacement for webdataset shard selection."
        ),
    ] = False,
    csv_separator: Annotated[
        str,
        typer.Option(help="For csv-like datasets, which separator to use."),
    ] = "\t",
    csv_img_key: Annotated[
        str,
        typer.Option(help="For csv-like datasets, the name of the key for the image paths."),
    ] = "filepath",
    csv_caption_key: Annotated[
        str,
        typer.Option(
            help="For csv-like datasets, the name of the key for the captions.",
        ),
    ] = "title",
    imagenet_val: Annotated[
        str | None,
        typer.Option(
            help="Path to imagenet val set for conducting zero shot evaluation.",
        ),
    ] = None,
    imagenet_v2: Annotated[
        str | None,
        typer.Option(
            help="Path to imagenet v2 for conducting zero shot evaluation.",
        ),
    ] = None,
    cache_dir: Annotated[
        str | None,
        typer.Option(
            help="Override system default cache path for model & tokenizer file downloads.",
        ),
    ] = None,
    logs: Annotated[
        str,
        typer.Option(
            help="Where to store tensorboard logs. Use None to avoid storing logs.",
        ),
    ] = "./logs/",
    log_local: Annotated[
        bool,
        typer.Option(
            help="log files on local master, otherwise global master only.",
        ),
    ] = False,
    name: Annotated[
        str | None,
        typer.Option(
            help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
        ),
    ] = None,
    workers: Annotated[
        int,
        typer.Option(
            help="Number of dataloader workers per GPU.",
        ),
    ] = 4,
    batch_size: Annotated[
        int,
        typer.Option(
            help="Batch size per GPU.",
        ),
    ] = 64,
    epochs: Annotated[
        int,
        typer.Option(
            help="Number of epochs to train for.",
        ),
    ] = 32,
    epochs_cooldown: Annotated[
        int | None,
        typer.Option(
            help="When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs onwards.",
        ),
    ] = None,
    lr: Annotated[
        float | None,
        typer.Option(
            help="Learning rate.",
        ),
    ] = None,
    beta1: Annotated[
        float | None,
        typer.Option(
            help="Adam beta 1.",
        ),
    ] = None,
    beta2: Annotated[
        float | None,
        typer.Option(
            help="Adam beta 2.",
        ),
    ] = None,
    eps: Annotated[
        float | None,
        typer.Option(
            help="Adam epsilon.",
        ),
    ] = None,
    wd: Annotated[
        float,
        typer.Option(
            help="Weight decay.",
        ),
    ] = 0.2,
    momentum: Annotated[
        float | None,
        typer.Option(
            help="Momentum (for timm optimizers).",
        ),
    ] = None,
    warmup: Annotated[
        int,
        typer.Option(
            help="Number of steps to warmup for.",
        ),
    ] = 10000,
    opt: Annotated[
        str,
        typer.Option(
            help="Which optimizer to use. Choices are ['adamw', or any timm optimizer 'timm/{opt_name}'].",
        ),
    ] = "adamw",
    use_bn_sync: Annotated[
        bool,
        typer.Option(
            help="Whether to use batch norm sync.",
        ),
    ] = False,
    skip_scheduler: Annotated[
        bool,
        typer.Option(
            help="Use this flag to skip the learning rate decay.",
        ),
    ] = False,
    lr_scheduler: Annotated[
        str,
        typer.Option(
            help="LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown' (constant w/ cooldown). Default: cosine",
        ),
    ] = "cosine",
    lr_cooldown_end: Annotated[
        float,
        typer.Option(
            help="End learning rate for cooldown schedule. Default: 0",
        ),
    ] = 0.0,
    lr_cooldown_power: Annotated[
        float,
        typer.Option(
            help="Power for polynomial cooldown schedule. Default: 1.0 (linear decay)",
        ),
    ] = 1.0,
    save_frequency: Annotated[
        int,
        typer.Option(
            help="How often to save checkpoints.",
        ),
    ] = 1,
    save_most_recent: Annotated[
        bool,
        typer.Option(
            help="Always save the most recent model trained to epoch_latest.pt.",
        ),
    ] = False,
    zeroshot_frequency: Annotated[
        int,
        typer.Option(
            help="How often to run zero shot.",
        ),
    ] = 2,
    val_frequency: Annotated[
        int,
        typer.Option(
            help="How often to run evaluation with val data.",
        ),
    ] = 1,
    resume: Annotated[
        str | None,
        typer.Option(
            help="path to latest checkpoint (default: none)",
        ),
    ] = None,
    precision: Annotated[
        datascaler_open_clip.params.Precision,
        typer.Option(
            help="Floating point precision.",
        ),
    ] = datascaler_open_clip.params.Precision.amp,
    model: Annotated[
        str,
        typer.Option(
            help="Name of the vision backbone to use.",
        ),
    ] = "RN50",
    pretrained: Annotated[
        str,
        typer.Option(
            help="Use a pretrained CLIP model weights with the specified tag or file path.",
        ),
    ] = "",
    pretrained_image: Annotated[
        bool,
        typer.Option(
            help="Load imagenet pretrained weights for image tower backbone if available.",
        ),
    ] = False,
    lock_image: Annotated[
        bool,
        typer.Option(
            help="Lock full image tower by disabling gradients.",
        ),
    ] = False,
    lock_image_unlocked_groups: Annotated[
        int,
        typer.Option(
            help="Leave last n image tower layer groups unlocked.",
        ),
    ] = 0,
    lock_image_freeze_bn_stats: Annotated[
        bool,
        typer.Option(
            help="Freeze BatchNorm running stats in image tower for any locked layers.",
        ),
    ] = False,
    image_mean: Annotated[
        list[float] | None,
        typer.Option(
            help="Override default image mean value of dataset",
            metavar="MEAN",
        ),
    ] = None,
    image_std: Annotated[
        list[float] | None,
        typer.Option(
            help="Override default image std deviation of of dataset",
            metavar="STD",
        ),
    ] = None,
    image_interpolation: Annotated[
        datascaler_open_clip.params.ImageInterpolation | None,
        typer.Option(
            help="Override default image resize interpolation",
        ),
    ] = None,
    image_resize_mode: Annotated[
        datascaler_open_clip.params.ImageResizeMode | None,
        typer.Option(
            help="Override default image resize (& crop) mode during inference",
        ),
    ] = None,
    raw_aug_cfg: Annotated[
        list[str] | None,
        typer.Option(
            "--aug-cfg",
            help=None,
        ),
    ] = None,
    grad_checkpointing: Annotated[
        bool,
        typer.Option(
            help="Enable gradient checkpointing.",
        ),
    ] = False,
    local_loss: Annotated[
        bool,
        typer.Option(
            help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)",
        ),
    ] = False,
    gather_with_grad: Annotated[
        bool,
        typer.Option(
            help="enable full distributed gradient for feature gather",
        ),
    ] = False,
    force_image_size: Annotated[
        list[int] | None,
        typer.Option(
            help="Override default image size",
        ),
    ] = None,
    force_quick_gelu: Annotated[
        bool,
        typer.Option(
            help="Force use of QuickGELU activation for non-OpenAI transformer models.",
        ),
    ] = False,
    force_patch_dropout: Annotated[
        float | None,
        typer.Option(
            help="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper",
        ),
    ] = None,
    force_custom_text: Annotated[
        bool,
        typer.Option(
            help="Force use of CustomTextCLIP model (separate text-tower).",
        ),
    ] = False,
    torchscript: Annotated[
        bool,
        typer.Option(
            help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
        ),
    ] = False,
    torchcompile: Annotated[
        bool,
        typer.Option(
            help="torch.compile() the model, requires pytorch 2.0 or later.",
        ),
    ] = False,
    trace: Annotated[
        bool,
        typer.Option(
            help="torch.jit.trace the model for inference / eval only",
        ),
    ] = False,
    accum_freq: Annotated[
        int,
        typer.Option(
            help="Update the model every --acum-freq steps.",
        ),
    ] = 1,
    device: Annotated[
        str,
        typer.Option(
            help="Accelerator to use.",
        ),
    ] = "cuda",
    dist_url: Annotated[
        str | None,
        typer.Option(
            help="url used to set up distributed training",
        ),
    ] = None,
    dist_backend: Annotated[
        str | None,
        typer.Option(
            help='distributed backend. "nccl" for GPU, "hccl" for Ascend NPU',
        ),
    ] = None,
    report_to: Annotated[
        str,
        typer.Option(
            help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']",
        ),
    ] = "",
    wandb_notes: Annotated[
        str,
        typer.Option(
            help="Notes if logging with wandb",
        ),
    ] = "",
    wandb_project_name: Annotated[
        str,
        typer.Option(
            help="Name of the project if logging with wandb.",
        ),
    ] = "open-clip",
    debug: Annotated[
        bool,
        typer.Option(
            help="If true, more information is logged.",
        ),
    ] = False,
    copy_codebase: Annotated[
        bool,
        typer.Option(
            help="If true, we copy the entire base on the log directory, and execute from there.",
        ),
    ] = False,
    horovod: Annotated[
        bool,
        typer.Option(
            help="Use horovod for distributed training.",
        ),
    ] = False,
    ddp_static_graph: Annotated[
        bool,
        typer.Option(
            help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
        ),
    ] = False,
    no_set_device_rank: Annotated[
        bool,
        typer.Option(
            "--no-set-device-rank/--set-device-rank",
            help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
        ),
    ] = False,
    seed: Annotated[
        int,
        typer.Option(
            help="Default random seed.",
        ),
    ] = 0,
    grad_clip_norm: Annotated[
        float | None,
        typer.Option(
            help="Gradient clip.",
        ),
    ] = None,
    lock_text: Annotated[
        bool,
        typer.Option(
            help="Lock full text tower by disabling gradients.",
        ),
    ] = False,
    lock_text_unlocked_layers: Annotated[
        int,
        typer.Option(
            help="Leave last n text tower layer groups unlocked.",
        ),
    ] = 0,
    lock_text_freeze_layer_norm: Annotated[
        bool,
        typer.Option(
            help="Freeze LayerNorm running stats in text tower for any locked layers.",
        ),
    ] = False,
    log_every_n_steps: Annotated[
        int,
        typer.Option(
            help="Log every n steps to tensorboard/console/wandb.",
        ),
    ] = 10,
    coca_caption_loss_weight: Annotated[
        float,
        typer.Option(
            help="Weight assigned to caption loss in CoCa.",
        ),
    ] = 2.0,
    coca_contrastive_loss_weight: Annotated[
        float,
        typer.Option(
            help="Weight assigned to contrastive loss when training CoCa.",
        ),
    ] = 1.0,
    remote_sync: Annotated[
        str | None,
        typer.Option(
            help="Optionally sync with a remote path specified by this arg",
        ),
    ] = None,
    remote_sync_frequency: Annotated[
        int,
        typer.Option(
            help="How frequently to sync to a remote directly if --remote-sync is not None.",
        ),
    ] = 300,
    remote_sync_protocol: Annotated[
        datascaler_open_clip.params.RemoteSyncProtocol,
        typer.Option(
            help="How to do the remote sync backup if --remote-sync is not None.",
        ),
    ] = datascaler_open_clip.params.RemoteSyncProtocol.s3,
    delete_previous_checkpoint: Annotated[
        bool,
        typer.Option(
            help="If true, delete previous checkpoint after storing a new one.",
        ),
    ] = False,
    distill_model: Annotated[
        str | None,
        typer.Option(
            help="Which model arch to distill from, if any.",
        ),
    ] = None,
    distill_pretrained: Annotated[
        str | None,
        typer.Option(
            help="Which pre-trained weights to distill from, if any.",
        ),
    ] = None,
    use_bnb_linear: Annotated[
        str | None,
        typer.Option(
            help="Replace the network linear layers from the bitsandbytes library. Allows int8 training/inference, etc.",
        ),
    ] = None,
    siglip: Annotated[
        bool,
        typer.Option(
            help="Use SigLip (sigmoid) loss.",
        ),
    ] = False,
    loss_dist_impl: Annotated[
        str | None,
        typer.Option(
            help="A string to specify a specific distributed loss implementation.",
        ),
    ] = None,
    use_mup: Annotated[
        bool,
        typer.Option(
            help="Use MUP.",
        ),
    ] = False,
    width_mult: Annotated[
        float,
        typer.Option(
            help="Width multiplier.",
        ),
    ] = 1.0,
    min_width_mult: Annotated[
        float,
        typer.Option(
            help="Minimum value of width multiplier for base shape generation.",
        ),
    ] = 1.0,
    attn_mult: Annotated[
        float,
        typer.Option(
            help="attn is multiplied by sqrt(attn_mult)/head_dim.",
        ),
    ] = 1.0,
    output_mult: Annotated[
        float,
        typer.Option(
            help="output is multiplied by sqrt(output_mult/d_model).",
        ),
    ] = 1.0,
    subset_file: Annotated[
        str | None,
        typer.Option(
            help="Path to trie pickle file.",
        ),
    ] = None,
):
    aug_cfg = datascaler_open_clip.params.parse_aug_cfg(raw_aug_cfg)

    params = datascaler_open_clip.params.OpenClipTrainParams(
        train_data=train_data,
        train_data_upsampling_factors=train_data_upsampling_factors,
        val_data=val_data,
        train_num_samples=train_num_samples,
        val_num_samples=val_num_samples,
        dataset_type=dataset_type,
        dataset_resampled=dataset_resampled,
        csv_separator=csv_separator,
        csv_img_key=csv_img_key,
        csv_caption_key=csv_caption_key,
        imagenet_val=imagenet_val,
        imagenet_v2=imagenet_v2,
        cache_dir=cache_dir,
        logs=logs,
        log_local=log_local,
        name=name,
        workers=workers,
        batch_size=batch_size,
        epochs=epochs,
        epochs_cooldown=epochs_cooldown,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        wd=wd,
        momentum=momentum,
        warmup=warmup,
        opt=opt,
        use_bn_sync=use_bn_sync,
        skip_scheduler=skip_scheduler,
        lr_scheduler=lr_scheduler,
        lr_cooldown_end=lr_cooldown_end,
        lr_cooldown_power=lr_cooldown_power,
        save_frequency=save_frequency,
        save_most_recent=save_most_recent,
        zeroshot_frequency=zeroshot_frequency,
        val_frequency=val_frequency,
        resume=resume,
        precision=precision,
        model=model,
        pretrained=pretrained,
        pretrained_image=pretrained_image,
        lock_image=lock_image,
        lock_image_unlocked_groups=lock_image_unlocked_groups,
        lock_image_freeze_bn_stats=lock_image_freeze_bn_stats,
        image_mean=image_mean,  # type: ignore  # handled by Pydantic
        image_std=image_std,  # type: ignore  # handled by Pydantic
        image_interpolation=image_interpolation,
        image_resize_mode=image_resize_mode,
        aug_cfg=aug_cfg,
        grad_checkpointing=grad_checkpointing,
        local_loss=local_loss,
        gather_with_grad=gather_with_grad,
        force_image_size=force_image_size,  # type: ignore  # handled by Pydantic
        force_quick_gelu=force_quick_gelu,
        force_patch_dropout=force_patch_dropout,
        force_custom_text=force_custom_text,
        torchscript=torchscript,
        torchcompile=torchcompile,
        trace=trace,
        accum_freq=accum_freq,
        device=device,
        dist_url=dist_url,
        dist_backend=dist_backend,
        report_to=report_to,
        wandb_notes=wandb_notes,
        wandb_project_name=wandb_project_name,
        debug=debug,
        copy_codebase=copy_codebase,
        horovod=horovod,
        ddp_static_graph=ddp_static_graph,
        no_set_device_rank=no_set_device_rank,
        seed=seed,
        grad_clip_norm=grad_clip_norm,
        lock_text=lock_text,
        lock_text_unlocked_layers=lock_text_unlocked_layers,
        lock_text_freeze_layer_norm=lock_text_freeze_layer_norm,
        log_every_n_steps=log_every_n_steps,
        coca_caption_loss_weight=coca_caption_loss_weight,
        coca_contrastive_loss_weight=coca_contrastive_loss_weight,
        remote_sync=remote_sync,
        remote_sync_frequency=remote_sync_frequency,
        remote_sync_protocol=remote_sync_protocol,
        delete_previous_checkpoint=delete_previous_checkpoint,
        distill_model=distill_model,
        distill_pretrained=distill_pretrained,
        use_bnb_linear=use_bnb_linear,
        siglip=siglip,
        loss_dist_impl=loss_dist_impl,
        use_mup=use_mup,
        width_mult=width_mult,
        min_width_mult=min_width_mult,
        attn_mult=attn_mult,
        output_mult=output_mult,
        subset_file=subset_file,
    )

    exit_code = datascaler_open_clip.train.main(params)
    exit_code = 0 if exit_code is None else 1
    typer.Exit(code=exit_code)
