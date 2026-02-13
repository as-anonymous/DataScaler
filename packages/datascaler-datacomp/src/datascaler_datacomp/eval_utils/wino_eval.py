"""Evaluate on WinoGAViL dataset."""

import os

import datasets
import numpy as np
import open_clip
import torch

# from collections import Counter
from sklearn.metrics import jaccard_score
from tqdm import tqdm

from datascaler_datacomp.eval_utils.wds_eval import create_model


class WinoDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None, text_transform=None):
        super().__init__()
        self._dataset = hf_dataset
        self.transform = (lambda x: x) if transform is None else transform
        self.text_transform = (lambda x: x) if text_transform is None else text_transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index: int):
        example = self._dataset[index]
        return (
            self.transform(example["candidate_images"]),
            self.text_transform(example["cue"]),
            np.isin(example["candidates"], example["associations"]),
        )


def evaluate_winogavil_dataset(
    model_arch,
    model_path,
    data_root,
    train_output_dir,
    num_workers=4,
    batch_size=None,
    use_mup=False,
    width_mult=1.0,
    attn_mult=1.0,
    output_mult=1.0,
):
    model, transform, device = create_model(
        model_arch,
        model_path,
        use_mup,
        train_output_dir,
        width_mult,
        attn_mult,
        output_mult,
    )
    tokenizer = open_clip.get_tokenizer(model_arch)

    # Load data
    dataset = WinoDataset(
        datasets.load_dataset(
            "nlphuji/winogavil",
            split="test",
            cache_dir=os.path.join(data_root, "hf_cache") if data_root is not None else None,
            trust_remote_code=False,
        ),
        transform=lambda imgs: torch.stack([transform(img) for img in imgs]),  # type: ignore
        text_transform=lambda text: tokenizer([get_clip_prompt(text)]),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: batch[0],
    )

    all_groups = []
    all_scores = []

    # Iterate WinoGAViL Instances
    for idx, (images, text, y_true) in enumerate(tqdm(dataloader)):
        # Get example
        n_images = len(images)
        n_assoc = y_true.sum()
        # Featurize
        with torch.no_grad(), torch.amp.autocast("cuda"):  # type: ignore
            image_features = model.encode_image(images.to(device), normalize=True)
            text_features = model.encode_text(text.to(device), normalize=True)
            # Compute similarities
            image_logits = (text_features @ image_features.T).squeeze(0).cpu().numpy()
        # Select topk
        topk_indices = np.argsort(image_logits)[-n_assoc:]
        y_pred = np.isin(np.arange(n_images), topk_indices)

        # Evaluate with Jaccard
        score = jaccard_score(y_true, y_pred)
        all_scores.append(score)
        all_groups.append(n_images)

        # if idx > 0 and idx % 100 == 0:
        #     print(f"idx: {idx}, current Jaccard index average: {np.mean(all_scores)}")

    all_groups = np.array(all_groups)
    all_scores = np.array(all_scores)
    return {
        "avg_jaccard_score": all_scores.mean(),
        "jaccard_score_5": all_scores[all_groups == 5].mean(),
        "jaccard_score_6": all_scores[all_groups == 6].mean(),
        "jaccard_score_10": all_scores[all_groups == 10].mean(),
        "jaccard_score_12": all_scores[all_groups == 12].mean(),
        "jaccard_score_5-6": all_scores[all_groups <= 6].mean(),
        "jaccard_score_10-12": all_scores[all_groups >= 10].mean(),
    }


def get_clip_prompt(item):
    item = item.lower()
    vowels = ["a", "e", "i", "o", "u"]
    if item[0] in vowels:
        clip_txt = f"An {item}"
    else:
        clip_txt = f"A {item}"
    return clip_txt
