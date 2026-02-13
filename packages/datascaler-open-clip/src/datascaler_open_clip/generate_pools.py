import argparse
import os

import torch
import webdataset as wds
from open_clip.factory import create_model_and_transforms
from open_clip_train.data import (
    filter_no_caption_or_no_image,
    get_dataset_size,
    log_and_continue,
    tarfile_to_samples_nothrow,
)

from datascaler_open_clip.filter_utils.apply_filter import apply_filter
from datascaler_open_clip.filter_utils.clip import get_clip_scores
from datascaler_open_clip.filter_utils.dbp import get_dbp_idx, get_dbp_scores
from datascaler_open_clip.filter_utils.dfn import get_dfn_scores
from datascaler_open_clip.filter_utils.sieve import get_sieve_scores
from datascaler_open_clip.filter_utils.tmars import get_tmars_scores


def load_dataloader(args):
    input_shards = args.data_dir
    assert os.path.exists(os.path.dirname(input_shards)), (
        f"Parent directory does not exist: {os.path.dirname(input_shards)}"
    )
    num_samples, num_shards = get_dataset_size(input_shards)
    print(f"Num of Shards {num_shards} - Num of Samples {num_samples}")

    device = torch.device(args.device)
    _, _, preprocess_val = create_model_and_transforms(
        pretrained="openai",
        model_name="ViT-L-14",
        precision="fp32",
        device=device,
        jit=True,
        output_dict=True,
        force_quick_gelu=True,
    )

    pipeline = [wds.SimpleShardList(args.data_dir)]
    pipeline.extend([wds.split_by_node])  # type: ignore
    pipeline.extend([wds.split_by_worker])  # type: ignore
    pipeline.extend([tarfile_to_samples_nothrow])  # type: ignore
    pipeline.extend(
        [  # type: ignore
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(
                image="jpg;png;jpeg;webp",
                text="txt",
                uid="json",
                original_width="json",
                original_height="json",
            ),
            wds.map_dict(
                image=preprocess_val,
                uid=lambda data: data["uid"],
                original_width=lambda data: data["original_width"],
                original_height=lambda data: data["original_height"],
            ),
            wds.to_tuple("__key__", "uid", "image", "text", "original_width", "original_height"),
            wds.batched(args.batch_size, partial=True),
        ]
    )
    dataset = wds.DataPipeline(*pipeline)
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    return dataloader


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    if not (args.verbose):
        setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def main():
    parser = argparse.ArgumentParser(description="Description of your program.")
    # Add arguments to the parser
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Path to shard files in .tar extension",
    )
    parser.add_argument("--output-dir", default="inference_results/", type=str)
    parser.add_argument(
        "--data-scale",
        default="small",
        type=str,
        help="Dataset scale.",
    )
    parser.add_argument("--num-workers", default=32, type=int, help="Number of workers")
    parser.add_argument("--batch-size", default=128, type=int, help="Global inference batch size")
    parser.add_argument(
        "--method",
        default="clip",
        type=str,
        help="Select scoring method among clip, t_mars, sieve, dbp, and dfn",
    )
    parser.add_argument("--min-fraction", default=0.0, type=float)
    parser.add_argument("--fraction", default=1.0, type=float)
    parser.add_argument(
        "--save-all-captions",
        action="store_true",
        help="save all generated captions from VLM decoder",
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)"
    )
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument(
        "--dist-url", default="env://", type=str, help="url used to set up distributed training"
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument("--verbose", help="print what each process is seeing", action="store_true")

    # Parse the command-line arguments
    args = parser.parse_args()
    init_distributed_mode(args)

    combined_file_name = f"{args.data_scale}_{args.method}.jsonl.gz"
    combined_file_path = os.path.join(args.output_dir, combined_file_name)
    if not os.path.isfile(combined_file_path):
        # end_num = args.data_dir.split("/")[-1].split(".")[-2][:8]
        file_name = f"{args.data_scale}_{args.method}.jsonl.gz"
        file_path = os.path.join(args.output_dir, file_name)

        if not os.path.isfile(file_path):
            dataloader = load_dataloader(args)

            if args.method == "clip":
                get_clip_scores(args, dataloader, file_path)
            if args.method == "dbp":
                get_dbp_scores(args, dataloader, file_path)
            if args.method == "dfn":
                get_dfn_scores(args, dataloader, file_path)
            if args.method == "sieve":
                get_sieve_scores(args, dataloader, file_path)
            if args.method == "tmars":
                get_tmars_scores(args, file_path)  # Uses custom dataloader inside tmars.py

    else:
        if args.method == "dbp":
            dataloader = load_dataloader(args)
            get_dbp_idx(args, dataloader, combined_file_path)
        else:
            apply_filter(args, combined_file_path)


if __name__ == "__main__":
    main()
