import glob as glob
import multiprocessing as mp
from functools import partial
from typing import Any, List, Tuple

import fsspec
import numpy as np
import pandas as pd
import pyarrow.json
from datascaler_open_clip.filter_utils.utils import (
    build_and_save_trie,
    worker_threadpool,
    worker_threadpool_only_uid_int,
)


def my_worker(path: str, columns: list[str]):
    table = pyarrow.json.read_json(path, pyarrow.json.ReadOptions(block_size=10 << 20))
    df = table.select(columns).to_pandas()
    return df


def load_metadata(
    metadata_dir_path: str,
    num_workers: int,
    columns: List[str] = None,  # type: ignore
) -> pd.DataFrame:
    """load metadata for many parquets

    Args:
        metadata_dir_path (str): directory where metadata is stored
        num_workers (int): number of cpu workers, each of which processes a parquet
        columns (List[str], optional): list of columns to retain from the parquet. Defaults to None.

    Returns:
        pd.DataFrame: loaded parquet columns
    """
    fs, url = fsspec.core.url_to_fs(metadata_dir_path)
    jsonl_paths = [str(x) for x in fs.ls(url) if ".jsonl.gz" in x]
    worker = partial(my_worker, columns=columns)

    return worker_threadpool_only_uid_int(worker, pd.concat, jsonl_paths, num_workers)  # type: ignore


# min/max normalized
def normalize_df(df, df_min, df_max):
    return (df - df_min) / (df_max - df_min)


# this function is a generic function that fuses pruning signals
def load_uids_generic_filter(
    metadata_dir_path: str,
    min_fraction: float,
    fraction: float,
    key: str,
    num_workers: int,
):
    df = load_metadata(metadata_dir_path, num_workers=num_workers, columns=[key])
    prune_signals_min = []
    prune_signals_max = []

    prune_signals_min, prune_signals_max = df[key].min(), df[key].max()
    df[key] = normalize_df(df[key], df_min=prune_signals_min, df_max=prune_signals_max)

    sorted_df = -np.sort(-df[key].values)  # type: ignore
    n = int(len(df[key]) * min_fraction)
    min_threshold = sorted_df[n]

    n = int(len(df[key]) * fraction)
    threshold = sorted_df[n - 1]

    print(f"Pruning signal minimum: {prune_signals_min}")
    print(f"Pruning signal maximum: {prune_signals_max}")
    print(f"Threshold: {threshold}")
    print(f"Min threshold: {min_threshold}")
    worker = partial(
        load_uids_generic_filter_helper,
        key=key,
        prune_signals_min=prune_signals_min,
        prune_signals_max=prune_signals_max,
        min_threshold=min_threshold,
        threshold=threshold,
    )
    fs, url = fsspec.core.url_to_fs(metadata_dir_path)
    jsonl_paths = [(fs, str(x)) for x in fs.ls(url) if ".jsonl.gz" in x]

    return worker_threadpool(worker, np.concatenate, jsonl_paths, num_workers)  # type: ignore


def load_uids_generic_filter_helper(
    fs_url: Tuple[Any, str],
    key: str,
    prune_signals_min: float,
    prune_signals_max: float,
    min_threshold: float,
    threshold: float,
):
    fs, url = fs_url
    columns = ["key", key]

    table = pyarrow.json.read_json(url, pyarrow.json.ReadOptions(block_size=10 << 20))
    df = table.select(columns).to_pandas()

    df[key] = normalize_df(df[key], df_min=prune_signals_min, df_max=prune_signals_max)
    mask_values = (df[key] >= threshold) & (df[key] <= min_threshold)
    keys = df["key"][mask_values]

    return keys


def apply_filter(args: Any, file_path: str) -> None:
    """function to route the args to the proper baseline function

    Args:
        args (Any): commandline args

    Raises:
        ValueError: unsupported name
    """
    mp.set_start_method("spawn", force=True)

    keys = load_uids_generic_filter(
        metadata_dir_path=file_path,
        min_fraction=args.min_fraction,
        fraction=args.fraction,
        key=args.method,
        num_workers=args.num_workers,
    )

    number_of_samples = len(keys)
    trie_file_path = (
        args.output_dir
        + f"/{args.data_scale}_{args.method}_{args.min_fraction}_{args.fraction}"
        + "_trie.pkl"
    )
    print(f"saving {trie_file_path} with {number_of_samples} entries")
    build_and_save_trie(data=keys, file_path=trie_file_path)  # type: ignore
