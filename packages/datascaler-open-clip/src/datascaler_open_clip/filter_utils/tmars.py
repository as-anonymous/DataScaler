import gzip
import json
import os
from functools import reduce

import cv2
import datascaler_open_clip.filter_utils.easyocr
import numpy as np
import torch
import torchvision.transforms.functional as TF
import webdataset as wds
from open_clip.factory import create_model_and_transforms, get_tokenizer
from open_clip_train.data import (
    filter_no_caption_or_no_image,
    get_dataset_size,
    log_and_continue,
    tarfile_to_samples_nothrow,
)
from torchvision import transforms


def load_dataloader(args):
    input_shards = args.data_dir
    assert os.path.exists(os.path.dirname(input_shards)), (
        f"Parent directory does not exist: {os.path.dirname(input_shards)}"
    )
    num_samples, num_shards = get_dataset_size(input_shards)
    print(f"Num of Shards {num_shards} - Num of Samples {num_samples}")

    resize_transform = transforms.Compose(
        [
            transforms.Resize((600, 600)),
            transforms.ToTensor(),
        ]
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
                image=resize_transform,
                # text=lambda data: data,
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


def get_tmars_scores(args, file_path):
    device = torch.device(args.device)
    dataloader = load_dataloader(args)

    def clip_transform_batch(batch: torch.Tensor) -> torch.Tensor:
        batch = TF.resize(batch, size=[224, 224], interpolation=TF.InterpolationMode.BICUBIC)
        batch = batch / 255

        CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=batch.device).view(
            1, 3, 1, 1
        )
        CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=batch.device).view(
            1, 3, 1, 1
        )

        return (batch - CLIP_MEAN) / CLIP_STD

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
    tokenizer = get_tokenizer(clip_model_name)
    clip_model.eval()
    ocr = datascaler_open_clip.filter_utils.easyocr.Reader(["en"], gpu=True)

    with torch.no_grad(), gzip.open(file_path, "wt") as fw:
        for idx, inputs in enumerate(dataloader):
            images = inputs[2]
            if idx % 10 == 0:
                print(f"Current batch {idx}")
            captions = inputs[3]
            keys = inputs[0]
            uids = inputs[1]
            orig_width = inputs[4]
            orig_height = inputs[5]

            ocr_input = np.array(images).swapaxes(1, 3) * 255
            OCR_outputs1, OCR_outputs2 = ocr.readtext_batched(ocr_input, batch_size=len(images))

            for n, img in enumerate(images):
                img = np.array(img).swapaxes(0, 2).swapaxes(0, 1)
                img_int = np.array(img * 255).astype(np.uint8).copy()
                img_shape = img.shape
                for rec in OCR_outputs1[n]:
                    # Todo: cleanup code
                    rec = np.array(rec).astype(np.int32)
                    rec = [
                        max(min(rec[0], 599), 0),
                        max(min(rec[1], 599), 0),
                        max(min(rec[2], 599), 0),
                        max(min(rec[3], 599), 0),
                    ]
                    img = np.array(img)
                    color = [0, 0, 0]
                    color += img_int[rec[2], rec[0]]
                    color += img_int[rec[3], rec[0]]
                    color += img_int[rec[2], rec[1]]
                    color += img_int[rec[3], rec[1]]
                    color = np.array(color / 4).astype(np.uint8).tolist()
                    rec = np.array(
                        [
                            [rec[0], rec[2]],
                            [rec[1], rec[2]],
                            [rec[1], rec[3]],
                            [rec[0], rec[3]],
                        ]
                    )

                    cv2.fillPoly(img_int, [rec], color)

                for rec in OCR_outputs2[n]:
                    rec = np.array(rec).astype(np.int32)
                    img = np.array(img)
                    color = (
                        (
                            reduce(
                                lambda x, y: x + y,
                                [
                                    img_int[np.clip(rec[i][1], 0, img_shape[0] - 1)][
                                        np.clip(rec[i][0], 0, img_shape[1] - 1)
                                    ].astype(int)
                                    for i in range(4)
                                ],
                            )
                            / 4
                        )
                        .astype(np.uint8)
                        .tolist()
                    )
                    cv2.fillPoly(img_int, [rec], color)
                images[n] = torch.Tensor(img_int.swapaxes(0, 2).astype(np.float32))

                # Debug-Line
                # cv2.imwrite(f"{n}.png", img_int)

            tokenized_captions = torch.stack([tokenizer(cap)[0] for cap in captions], dim=0)
            images = clip_transform_batch(images.to(device))
            img = images.to(device)
            txt = tokenized_captions.to(device)
            img_f = clip_model.encode_image(img)  # type: ignore
            txt_f = clip_model.encode_text(txt)  # type: ignore
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
            clip_scores = torch.diag(img_f @ txt_f.T)

            orig_width = orig_width.tolist()
            orig_height = orig_height.tolist()
            clip_scores = clip_scores.cpu().tolist()

            for n in range(len(uids)):
                result = {
                    "uid": uids[n],
                    "key": keys[n],
                    "original_width": orig_width[n],
                    "original_height": orig_height[n],
                    "text": captions[n],
                    f"{args.method}": clip_scores[n],
                }
                json.dump(result, fw, separators=(",", ":"))
                fw.write("\n")
