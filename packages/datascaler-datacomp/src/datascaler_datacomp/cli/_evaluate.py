from typing import Annotated

import typer

import datascaler_datacomp.evaluate
import datascaler_datacomp.params
import datascaler_datacomp.scale

app = typer.Typer()


@app.command(name="evaluate")
def evaluate_model(
    scale: Annotated[
        datascaler_datacomp.scale.DatacompScale,
        typer.Option(
            help="Competition scale.",
        ),
    ],
    train_output_dir: Annotated[
        str,
        typer.Option(
            help="Path to output directory from training.",
        ),
    ],
    task_dir: Annotated[
        str,
        typer.Option(
            help="Path to directory containing evaluation tasks.",
        ),
    ],
    model_checkpoint: Annotated[
        str,
        typer.Option(
            help="Name of model checkpoint.",
        ),
    ],
    model_arch: Annotated[
        str,
        typer.Option(
            help="Architecture of the trained CLIP model.",
        ),
    ] = "ViT-B-32",
    output_dir: Annotated[
        str | None,
        typer.Option(
            help="Path to output directory to use for evaluation. If nothing is passed, use the training output dir.",
        ),
    ] = None,
    data_dir: Annotated[
        str | None,
        typer.Option(
            help="(Optional) Path to directory containing downloaded evaluation datasets.",
        ),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            help="Batch size.",
        ),
    ] = 64,
    use_mup: Annotated[
        bool,
        typer.Option(
            help="If True, use mup.",
        ),
    ] = False,
    num_workers: Annotated[
        int,
        typer.Option(
            help="Number of workers.",
        ),
    ] = 64,
    width_mult: Annotated[
        float,
        typer.Option(
            help="Width multiplier.",
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
    compute_loss: Annotated[
        float,
        typer.Option(
            help="If 1.0, compute zero-shot contrastive loss.",
        ),
    ] = 0.0,
):
    params = datascaler_datacomp.params.DatacompEvaluateParams(
        scale=scale,
        train_output_dir=train_output_dir,  # type: ignore # handled by Pydantic
        task_dir=task_dir,  # type: ignore # handled by Pydantic
        model_checkpoint=model_checkpoint,
        model_arch=model_arch,
        output_dir=output_dir,  # type: ignore # handled by Pydantic
        data_dir=data_dir,
        batch_size=batch_size,
        use_mup=use_mup,
        num_workers=num_workers,
        width_mult=width_mult,
        attn_mult=attn_mult,
        output_mult=output_mult,
        compute_loss=compute_loss,
    )
    datascaler_datacomp.evaluate.main(params)
