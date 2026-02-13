import pickle
import random
from multiprocessing import Pool
from typing import Any, List

import numpy as np
import torch
from tqdm import tqdm


def dicttrie(arr):
    trie = {}
    for p in tqdm(arr):
        key = p.strip()
        trie[key] = True
    return trie


def trie_search(trie, uid):
    try:
        if trie[uid]:
            return True
    except KeyError:
        return False


def build_and_save_trie(data: list, file_path: str):
    print("Building trie structue for the pruned data")
    mytrie = dicttrie(data)
    print("Saving trie.pickle ...")
    with open(file_path, "wb") as f:
        pickle.dump(mytrie, f)
    print("Saved")


def random_seed(seed: int = 0) -> None:
    """set seed

    Args:
        seed (int, optional): seed value. Defaults to 0.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def worker_threadpool(
    worker_fn: Any, concat_fn: Any, paths: List[str], n_workers: int
) -> np.ndarray:
    """get filtered uids

    Args:
        worker_fn (Any): function to map over the pool
        concat_fn (Any): function to use to collate the results
        paths (List[str]): metadata paths to process
        n_workers (int): number of cpu workers

    Returns:
        np.ndarray: filtered uids
    """
    print("creating thread pool for processing")
    with Pool(n_workers) as pool:
        uid_int_list = []
        for uid_int in tqdm(
            pool.imap_unordered(worker_fn, paths),
            total=len(paths),
        ):
            uid_int_list.append(uid_int)

    # save both uid_int (low, high) and original uid_str
    return concat_fn(uid_int_list)  # type: ignore


def worker_threadpool_only_uid_int(
    worker_fn: Any, concat_fn: Any, paths: List[str], n_workers: int
) -> np.ndarray:
    """get filtered uids

    Args:
        worker_fn (Any): function to map over the pool
        concat_fn (Any): function to use to collate the results
        paths (List[str]): metadata paths to process
        n_workers (int): number of cpu workers

    Returns:
        np.ndarray: filtered uids
    """
    print("creating thread pool for processing")
    with Pool(n_workers) as pool:
        uids = []
        for u in tqdm(
            pool.imap_unordered(worker_fn, paths),
            total=len(paths),
        ):
            uids.append(u)

    return concat_fn(uids)
