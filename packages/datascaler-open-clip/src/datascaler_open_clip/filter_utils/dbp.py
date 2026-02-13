import gzip
import json
import os

import faiss
import numpy as np
import torch
from datascaler_open_clip.filter_utils.utils import (
    build_and_save_trie,
)
from open_clip.factory import create_model_and_transforms
from qpsolvers import solve_qp


def semdedup(embedding, eps, detailed=False):
    pair_w_sim_matrix = embedding @ embedding.T
    del embedding
    triu_sim_mat = torch.triu(pair_w_sim_matrix, diagonal=1)

    M = torch.max(triu_sim_mat, dim=0)[0]
    del triu_sim_mat

    return (M < 1 - eps), None if not detailed else {
        "M": M,
        "delected_idx": torch.where(M >= 1 - eps),
    }


def get_dbp_scores(args, dataloader, file_path):
    with torch.no_grad(), gzip.open(file_path, "wt") as fw:
        for idx, sample in enumerate(dataloader):
            if idx % 10 == 0:
                print(f"Current batch {idx}")
            captions = sample[3]
            keys = sample[0]
            uids = sample[1]
            orig_width = sample[4].tolist()
            orig_height = sample[5].tolist()

            for i in range(len(uids)):
                result = {
                    "uid": uids[i],
                    "key": keys[i],
                    "original_width": orig_width[i],
                    "original_height": orig_height[i],
                    "text": captions[i],
                    f"{args.method}": 1.0,
                }
                json.dump(result, fw, separators=(",", ":"))
                fw.write("\n")


# dbp score
@torch.no_grad()
def get_dbp_idx(args, dataloader, file_path):
    device = torch.device(args.device)
    ratio = args.fraction
    if not os.path.exists("./cache/clip_embedding.npy"):
        device = torch.device(args.device)
        clip_model_name = "ViT-L-14"
        clip_model, _, preprocess_val = create_model_and_transforms(
            pretrained="openai",
            model_name=clip_model_name,
            precision="fp32",
            device=device,
            jit=True,
            output_dict=True,
            force_quick_gelu=True,
        )
        clip_model.eval()

        if args.distributed and args.sync_bn:
            clip_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(clip_model)

        clip_model_without_ddp = clip_model
        if args.distributed:
            clip_model = torch.nn.parallel.DistributedDataParallel(
                clip_model, device_ids=[args.gpu]
            )
            clip_model_without_ddp = clip_model.module

        print(f"Number of GPUS: {torch.cuda.device_count()}")
        print(f"Batch size: {args.batch_size}")
        print(f"Model name is: {clip_model_name}")

        embedding_list = []
        with torch.no_grad():
            for idx, sample in enumerate(dataloader):
                images = sample[2]
                if idx % 10 == 0:
                    print(f"Current batch {idx}")
                img = images.to(device)
                img_f = clip_model_without_ddp.encode_image(img)  # type: ignore
                img_f = img_f / img_f.norm(dim=-1, keepdim=True)
                embedding_list.append(img_f)

        embeddings = torch.cat(embedding_list).cpu().numpy()
        np.save("./cache/clip_embedding.npy", embeddings)
    else:
        embeddings = np.load("./cache/clip_embedding.npy")

    # SemDeDup
    classes = max(int(pow(embeddings.shape[0] / 10, 1 / 2)), 10)
    k_means = faiss.Kmeans(
        embeddings.shape[1], classes, niter=100, verbose=True, spherical=True, gpu=True
    )
    k_means.train(embeddings)
    D, cluster_index = k_means.index.search(embeddings, 1)  # type: ignore
    cluster_index = cluster_index.squeeze()

    idx_list = []
    for i in range(classes):
        idx = cluster_index == i
        orig_size = idx.sum()
        if orig_size < 30000:
            cls_embedding = torch.Tensor(embeddings[cluster_index == i]).to(device)
        else:
            cls_embedding = torch.Tensor(embeddings[cluster_index == i])
        idx, _ = semdedup(cls_embedding, 0.01, detailed=True)
        idx = torch.arange(embeddings.shape[0], device=device)[cluster_index == i][idx]

        idx_list.append(idx.cpu().numpy())

    deduped_idx = np.concatenate(idx_list)
    deduped_array = embeddings[deduped_idx]
    original_all_size = embeddings.shape[0]
    all_size = deduped_array.shape[0]

    print(f"ORG SIZE: {original_all_size} / DEDUP SIZE: {all_size}")

    # k=100 for DataComp Medium
    # k=500 for LAION-DeDup-280M
    classes = 100
    # dbp
    k_means = faiss.Kmeans(
        embeddings.shape[1], classes, niter=100, verbose=True, spherical=True, gpu=True
    )
    k_means.train(deduped_array)
    D, cluster_index = k_means.index.search(deduped_array, 1)  # type: ignore
    cluster_index = cluster_index.squeeze()

    cluster_centroids = torch.Tensor(k_means.centroids).to(device)

    # calculate intra-cluster distance
    d_intra = torch.zeros(classes, device=device)
    d_size = np.zeros(classes)
    d_idx = {}
    sorted_distance_dict = {}

    for i in range(classes):
        idx = cluster_index == i
        d_size[i] = idx.sum()
        d_idx[i] = np.arange(all_size)[idx]
        cls_embedding = torch.Tensor(deduped_array[idx]).to(device)
        distance = 1 - (cls_embedding @ cluster_centroids[i].T)
        sorted_distance_dict[i] = torch.sort(distance, descending=True)[1].cpu().numpy()
        d_intra[i] = distance.mean()

    cls_dis = 1 - (cluster_centroids @ cluster_centroids.T)
    d_inter = (
        cls_dis[[i for i in range(classes) for _ in range(5)], cls_dis.argsort()[:, 1:6].ravel()]  # type: ignore
        .reshape(classes, 5)
        .mean(axis=1)
    )

    temper = 0.1
    d_i = d_intra * d_inter
    probs = torch.softmax(torch.Tensor(d_i) / temper, dim=0).cpu().numpy()

    P = np.eye(classes)
    q = -probs * round(original_all_size * ratio)
    A = np.array([1.0] * classes)
    b = np.array([round(original_all_size * ratio)])
    bounds = np.array([(1, d_size[i]) for i in range(classes)])

    x = solve_qp(P=P, q=q, A=A, b=b, lb=bounds[:, 0], ub=bounds[:, 1], solver="osqp")
    x = np.rint(x).astype(int)  # type: ignore

    for i in range(classes):
        print(f"{i} classes: {x[i]}/{int(d_size[i])}")

    subset_idx_list = [0 for _ in range(classes)]
    for i in range(classes):
        subset_idx_list[i] = deduped_idx[d_idx[i][sorted_distance_dict[i][: x[i]]]]  # type: ignore
    subset_idx = np.concatenate(subset_idx_list)
    subset_idx = list(map(str, subset_idx.tolist()))
    subset_idx = ["0" * (12 - len(x)) + x for x in subset_idx]

    # SAVE SUBSET INDEX AS TRIE
    number_of_samples = len(subset_idx)
    trie_file_path = (
        args.output_dir + f"/{args.data_scale}_{args.method}_{args.fraction}" + "_trie.pkl"
    )
    print(f"saving {trie_file_path} with {number_of_samples} entries")
    build_and_save_trie(data=subset_idx, file_path=trie_file_path)

    return subset_idx
