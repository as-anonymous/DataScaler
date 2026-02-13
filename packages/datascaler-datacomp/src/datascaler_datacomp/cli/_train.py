from typing import Annotated

import typer

import datascaler_datacomp.params
import datascaler_datacomp.scale
import datascaler_datacomp.train

app = typer.Typer()


@app.command(name="train")
def train_model(
    scale: Annotated[
        datascaler_datacomp.scale.DatacompScale,
        typer.Option(
            help="Competition scale.",
        ),
    ],
    data_dir: Annotated[
        str,
        typer.Option(
            help='Path to directory where the data is stored. Multiple paths can be used, separated by "::".',
        ),
    ],
    output_dir: Annotated[
        str,
        typer.Option(
            help="Path to directory where outputs will be stored.",
        ),
    ],
    val_ratio: Annotated[
        float,
        typer.Option(help="Ratio of validation set."),
    ] = 0.05,
    val_batch_size: Annotated[
        int,
        typer.Option(help="Val batch size."),
    ] = 128,
    data_weights: Annotated[
        str | None,
        typer.Option(
            help=None,
        ),
    ] = None,
    exp_name: Annotated[
        str | None,
        typer.Option(
            help="Name of the experiment for logging.",
        ),
    ] = None,
    use_cached_shards: Annotated[
        bool,
        typer.Option(
            help="If true, re-use the re-sharded data if possible.",
        ),
    ] = False,
    wandb_project_name: Annotated[
        str,
        typer.Option(
            help="Name of the project if logging with wandb.",
        ),
    ] = "datanet",
    workers: Annotated[
        int,
        typer.Option(
            help="Number of workers for open_clip.",
        ),
    ] = 4,
    precision: Annotated[
        datascaler_datacomp.params.DatacompPrecision,
        typer.Option(
            help="Floating point precision.",
        ),
    ] = datascaler_datacomp.params.DatacompPrecision.amp,
    num_checkpoints: Annotated[
        int,
        typer.Option(
            help="Number of times we save checkpoints during training.",
        ),
    ] = 5,
    seed: Annotated[
        int,
        typer.Option(
            help="Random seed.",
        ),
    ] = 0,
    report_to_wandb: Annotated[
        bool,
        typer.Option(
            help="If True, report to wandb.",
        ),
    ] = False,
    accum_freq: Annotated[
        int,
        typer.Option(
            help="Update the model every --acum-freq steps.",
        ),
    ] = 1,
    log_every_n_steps: Annotated[
        int,
        typer.Option(
            help="Log every n steps to tensorboard/console/wandb.",
        ),
    ] = 10,
    resume: Annotated[
        str,
        typer.Option(
            help="Path to checkpoint to resume from (default: latest checkpoint in the training directory).",
        ),
    ] = "latest",
    imagenet_val: Annotated[
        str | None,
        typer.Option(
            help="Optional path to imagenet val set for conducting zero shot evaluation.",
        ),
    ] = None,
    grad_clip_norm: Annotated[
        float | None,
        typer.Option(
            help=None,
        ),
    ] = None,
    save_frequency: Annotated[
        int,
        typer.Option(
            help=None,
        ),
    ] = 0,
    lr_scheduler: Annotated[
        str,
        typer.Option(
            help="LR scheduler. One of: 'cosine', 'cosine_with_end', 'const' (constant), 'const-cooldown' (constant w/ cooldown),. Default: cosine",
        ),
    ] = "cosine",
    model_arch: Annotated[
        str,
        typer.Option(
            help="Architecture of the trained CLIP model.",
        ),
    ] = "ViT-B-32",
    use_mup: Annotated[
        bool,
        typer.Option(
            help="If True, use mup.",
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
    learning_rate: Annotated[
        float,
        typer.Option(
            help="Learning rate.",
        ),
    ] = 5e-4,
    train_num_samples: Annotated[
        int,
        typer.Option(
            help="Number of samples seen during training.",
        ),
    ] = 12_800_000,
):
    params = datascaler_datacomp.params.DatacompTrainParams(
        scale=scale,
        data_dir=data_dir,
        data_weights=data_weights,
        output_dir=output_dir,
        val_ratio=val_ratio,
        val_batch_size=val_batch_size,
        exp_name=exp_name,
        use_cached_shards=use_cached_shards,
        wandb_project_name=wandb_project_name,
        workers=workers,
        precision=precision,
        num_checkpoints=num_checkpoints,
        seed=seed,
        report_to_wandb=report_to_wandb,
        accum_freq=accum_freq,
        log_every_n_steps=log_every_n_steps,
        resume=resume,
        imagenet_val=imagenet_val,
        grad_clip_norm=grad_clip_norm,
        save_frequency=save_frequency,
        lr_scheduler=lr_scheduler,
        model_arch=model_arch,
        use_mup=use_mup,
        width_mult=width_mult,
        min_width_mult=min_width_mult,
        attn_mult=attn_mult,
        output_mult=output_mult,
        subset_file=subset_file,
        learning_rate=learning_rate,
        train_num_samples=train_num_samples,
    )
    datascaler_datacomp.train.main(params)
