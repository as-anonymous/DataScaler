import gzip
import json

import torch
from transformers import CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPTokenizer


def score_func(image_emb, text_emb):
    image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    clip_matrices = 100 * image_emb @ text_emb.T
    clip_scores = clip_matrices.diag()

    return clip_scores


# dfn score
@torch.no_grad()
def get_dfn_scores(args, dataloader, file_path):
    model = CLIPModel.from_pretrained("apple/DFN-public")
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("apple/DFN-public")
    processor = CLIPProcessor(image_processor=image_processor, tokenizer=tokenizer)
    device = torch.device(args.device)

    model.to(device)  # type: ignore
    model.eval()

    dfn_list = []
    with gzip.open(file_path, "wt") as fw:
        for i, inputs in enumerate(dataloader):
            captions = inputs[3]
            keys = inputs[0]
            uids = inputs[1]
            orig_width = inputs[4]
            orig_height = inputs[5]
            texts = processor(
                text=inputs[3],
                images=None,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )

            outputs = model(
                pixel_values=inputs[2].to(device),
                input_ids=texts["input_ids"].to(device),  # type: ignore
            )

            clip_scores = score_func(
                outputs.image_embeds.to(device), outputs.text_embeds.to(device)
            )
            dfn_list = clip_scores.cpu().tolist()

            orig_width = orig_width.tolist()
            orig_height = orig_height.tolist()

            for n in range(len(uids)):
                result = {
                    "uid": uids[n],
                    "key": keys[n],
                    "original_width": orig_width[n],
                    "original_height": orig_height[n],
                    "text": captions[n],
                    f"{args.method}": dfn_list[n],
                }
                json.dump(result, fw, separators=(",", ":"))
                fw.write("\n")

    return
