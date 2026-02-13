"""Evaluate on standard classification webdatasets."""

import os

import datascaler_open_clip.factory
import open_clip
import torch
import torch.nn.functional as F
from clip_benchmark.datasets.builder import build_dataset
from clip_benchmark.metrics import zeroshot_classification as zsc
from open_clip import get_input_dtype
from open_clip_train.precision import get_autocast
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from datascaler_datacomp.eval_utils.clip_loss import ClipLoss


def create_model(
    model_arch, model_path, use_mup, train_output_dir, width_mult, attn_mult, output_mult
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    model_path = str(model_path)

    # Modified part to load our CLIP models
    model, _, transform = datascaler_open_clip.factory.create_model_and_transforms(
        model_arch,
        pretrained=model_path,
        use_mup=use_mup,
        base_shape_path=train_output_dir,
        width_mult=width_mult,
        attn_mult=attn_mult,
        output_mult=output_mult,
    )
    model.eval()
    # model.half()
    model = model.to(device)

    return model, transform, device


def create_webdataset(
    task, transform, data_root=None, dataset_len=None, batch_size=64, num_workers=4
):
    data_folder = f"wds_{task.replace('/', '-')}_test"
    if data_root is None:
        data_root = f"https://huggingface.co/datasets/djghosh/{data_folder}/tree/main"
    else:
        data_root = os.path.join(data_root, data_folder)
    dataset = build_dataset(
        dataset_name=f"wds/{task}",
        root=data_root,
        transform=transform,
        split="test",
        download=False,
    )
    if dataset_len:
        dataset = dataset.with_length((dataset_len + batch_size - 1) // batch_size)
    dataloader = torch.utils.data.DataLoader(
        dataset.batched(batch_size),
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
    )
    return dataset, dataloader


def evaluate_webdataset(
    task,
    model_arch,
    model_path,
    data_root,
    dataset_len,
    train_output_dir,
    batch_size=64,
    num_workers=4,
    return_preds=False,
    return_topk=False,
    use_mup=False,
    width_mult=1.0,
    attn_mult=1.0,
    output_mult=1.0,
    compute_loss=0.0,
):
    """Evaluate CLIP model on classification task."""

    # Create model
    model, transform, device = create_model(
        model_arch,
        model_path,
        use_mup,
        train_output_dir,
        width_mult,
        attn_mult,
        output_mult,
    )

    # Load data
    dataset, dataloader = create_webdataset(
        task, transform, data_root, dataset_len, batch_size, num_workers
    )

    zeroshot_templates = dataset.templates if hasattr(dataset, "templates") else None
    classnames = dataset.classes if hasattr(dataset, "classes") else None

    assert zeroshot_templates is not None and classnames is not None, (
        "Dataset does not support classification"
    )

    if compute_loss == 0.0:
        # Evaluate
        classifier = zsc.zero_shot_classifier(
            model,
            open_clip.get_tokenizer(model_arch),
            classnames,
            zeroshot_templates,
            device,
        )

        logits, target = zsc.run_classification(model, classifier, dataloader, device, amp=False)

        with torch.no_grad():
            pred = logits.argmax(axis=1).cpu()  # type: ignore
            target = target.cpu()

        # Compute metrics
        if len(dataset.classes) >= 5:
            acc1, acc5 = zsc.accuracy(logits, target, topk=(1, 5))
        else:
            (acc1,) = zsc.accuracy(logits, target, topk=(1,))
            acc5 = None
        mean_per_class_recall = balanced_accuracy_score(target, pred)
        metrics = {
            "acc1": acc1,
            "acc5": acc5,
            "mean_per_class_recall": mean_per_class_recall,
        }
    else:
        autocast = get_autocast("amp", device_type="cuda")
        input_dtype = get_input_dtype("amp")
        tokenizer = open_clip.get_tokenizer(model_arch)

        loss_fn = ClipLoss(
            local_loss=False,
            gather_with_grad=False,
            cache_labels=True,
            rank=1,
            world_size=1,
            use_horovod=False,
        )

        model.eval()
        model = model.to(device)

        text_feature_arr = []
        with torch.no_grad(), autocast():
            for classname in tqdm(classnames):
                if isinstance(zeroshot_templates, dict):
                    # class-specific prompts (e.g., CuPL https://arxiv.org/abs/2209.03320)
                    texts = zeroshot_templates[classname]
                elif isinstance(zeroshot_templates, list):
                    # generic prompts tht are specialized for each class by replacing {c} with the class name
                    texts = [template.format(c=classname) for template in zeroshot_templates]
                else:
                    raise ValueError("templates must be a list or a dict")
                texts = tokenizer(texts).to(device)  # tokenize
                class_embeddings = model.encode_text(texts)
                class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                class_embedding /= class_embedding.norm()
                text_feature_arr.append(class_embedding)

        total_loss = torch.tensor(0.0, device=device)
        total_samples = torch.tensor(0.0, device=device)
        with torch.no_grad(), autocast():
            for i, batch in enumerate(tqdm(dataloader)):
                images, texts = batch

                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                image_features = model.encode_image(images, normalize=True)

                text_features = [text_feature_arr[text] for text in texts]
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

    if return_preds:
        if return_topk:
            with torch.no_grad():
                _, topk_pred = torch.topk(logits, int(return_topk), dim=1)
                topk_pred = topk_pred.cpu()
            return metrics, topk_pred, target
        return metrics, pred, target
    return metrics
