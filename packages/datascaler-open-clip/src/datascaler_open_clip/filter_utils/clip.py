import gzip
import json

import torch
from open_clip.factory import create_model_and_transforms, get_tokenizer


def get_clip_scores(args, dataloader, file_path):
    # model_size = args.clip_model_size
    device = torch.device(args.device)

    # need for preprocessing function in openclip
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

    if args.distributed and args.sync_bn:
        clip_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(clip_model)

    clip_model_without_ddp = clip_model
    if args.distributed:
        clip_model = torch.nn.parallel.DistributedDataParallel(clip_model, device_ids=[args.gpu])
        clip_model_without_ddp = clip_model.module

    print(f"Number of GPUS: {torch.cuda.device_count()}")
    print(f"Batch size: {args.batch_size}")

    # model_name = f"CLIP_{model_size}"
    print(f"Model name is: {clip_model_name}")

    with torch.no_grad(), gzip.open(file_path, "wt") as fw:
        for idx, sample in enumerate(dataloader):
            images = sample[2]
            if idx % 10 == 0:
                print(f"Current batch {idx}")
            captions = sample[3]
            keys = sample[0]
            uids = sample[1]
            orig_width = sample[4]
            orig_height = sample[5]
            # save openclip scores
            tokenized_captions = torch.stack([tokenizer(cap)[0] for cap in captions], dim=0)
            img = images.to(device)
            txt = tokenized_captions.to(device)
            img_f = clip_model_without_ddp.encode_image(img)  # type: ignore
            txt_f = clip_model_without_ddp.encode_text(txt)  # type: ignore
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
            clip_current_scores = torch.diag(img_f @ txt_f.T)

            orig_width = orig_width.tolist()
            orig_height = orig_height.tolist()
            clip_current_scores = clip_current_scores.cpu().numpy().tolist()

            for i in range(len(uids)):
                result = {
                    "uid": uids[i],
                    "key": keys[i],
                    "original_width": orig_width[i],
                    "original_height": orig_height[i],
                    "text": captions[i],
                    f"{args.method}": clip_current_scores[i],
                }
                json.dump(result, fw, separators=(",", ":"))
                fw.write("\n")
