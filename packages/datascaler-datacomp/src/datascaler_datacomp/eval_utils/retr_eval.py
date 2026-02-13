"""Evaluate on image-text retrieval datasets."""

import os

import datasets
import open_clip
import torch
import torch.nn.functional as F
from clip_benchmark.datasets.builder import image_captions_collate_fn
from clip_benchmark.metrics import zeroshot_retrieval as zsr
from open_clip import get_input_dtype
from open_clip_train.precision import get_autocast
from tqdm import tqdm

from datascaler_datacomp.eval_utils.clip_loss import ClipLoss
from datascaler_datacomp.eval_utils.wds_eval import create_model


class RetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        super().__init__()
        self._dataset = hf_dataset
        self.transform = (lambda x: x) if transform is None else transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index: int):
        return (
            self.transform(self._dataset[index]["image"]),
            self._dataset[index]["caption"],
        )


def evaluate_retrieval_dataset(
    task,
    model_arch,
    model_path,
    data_root,
    train_output_dir,
    use_mup=False,
    batch_size=64,
    num_workers=4,
    width_mult=1.0,
    attn_mult=1.0,
    output_mult=1.0,
    compute_loss=0.0,
):
    """Evaluate CLIP model on retrieval task."""

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

    dataset = RetrievalDataset(
        datasets.load_dataset(
            f"nlphuji/{task.replace('retrieval/', '')}",
            split="test",
            cache_dir=os.path.join(data_root, "hf_cache") if data_root is not None else None,
            trust_remote_code=False,
        ),
        transform=transform,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=image_captions_collate_fn,
    )

    if compute_loss == 0.0:
        metrics = zsr.evaluate(
            model, dataloader, tokenizer, recall_k_list=[1, 5, 10], device=device
        )
        metrics["mean_recall@1"] = 0.5 * (
            metrics["text_retrieval_recall@1"] + metrics["image_retrieval_recall@1"]
        )
    else:
        autocast = get_autocast("amp", device_type="cuda")
        input_dtype = get_input_dtype("amp")

        loss_fn = ClipLoss(
            local_loss=False,
            gather_with_grad=False,
            cache_labels=True,
            rank=1,
            world_size=1,
            use_horovod=False,
        )

        total_loss = torch.tensor(0.0, device=device)
        total_samples = torch.tensor(0.0, device=device)
        with torch.no_grad(), autocast():
            for i, batch in enumerate(tqdm(dataloader)):
                image_batch, text_batch = batch

                images = image_batch.to(device=device, dtype=input_dtype, non_blocking=True)
                image_features = model.encode_image(images, normalize=True)

                text_features = []
                for texts in text_batch:
                    texts = tokenizer(list(texts)).to(device)  # tokenize
                    text_embeddings = model.encode_text(texts)
                    text_embeddings = F.normalize(text_embeddings, dim=-1).mean(dim=0)
                    text_embeddings /= text_embeddings.norm()
                    text_features.append(text_embeddings)
                text_features = torch.stack(text_features, dim=0).to(device)

                logit_scale = model.logit_scale.exp()

                losses = loss_fn(image_features, text_features, logit_scale, output_dict=True)
                loss_val = sum(losses.values())

                total_loss += loss_val * len(images)
                total_samples += len(images)

        avg_loss = (total_loss / total_samples).item()
        print(f"Loss: {avg_loss}, total_samples: {total_samples}")

        metrics = {
            "loss": avg_loss,
        }

    return metrics
