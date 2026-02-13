import gzip
import json
import re
import sys

import torch
import torch.nn.functional as F
from open_clip.factory import create_model_and_transforms, get_tokenizer
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel

# BLIP from https://github.com/facebookresearch/SIEVE/
sys.path.insert(0, "./BLIP")
from BLIP.models.blip import blip_decoder

NUM_CAPTIONS = 8

REMOVE_PHRASES_LONG = [
    "an image of",
    "a photo of",
    "an icon of",
    "an illustration of",
    "a template of",
    "a thumbnail of",
    "a vector of",
    "photo stock",
    "stock photo",
    "a photo",
    "an image",
    "an icon",
    "an illustration",
    "a template",
    "a thumbnail",
    "image",
    "photo",
    "icon",
    "illustration",
    "template",
    "vector",
    "thumbnail",
    "free",
    "print",
    "sale",
    "quot",
    "png",
    "jpeg",
    "jpg",
]

REMOVE_PHRASES = [
    "an image of",
    "a photo of",
    "stock photo",
    "photo stock",
    "a photo",
    "an image",
    "image",
    "photo",
]

# print(f"The list of phrases to exclude from captions and generated captions: {REMOVE_PHRASES}")


def remove_phrases(sentences, phrases_to_remove=REMOVE_PHRASES):
    modified_sentences = []
    # ignore case sensitivity
    phrases_to_remove = [re.compile(phrase, re.IGNORECASE) for phrase in phrases_to_remove]
    for sentence in sentences:
        for phrase in phrases_to_remove:
            sentence = phrase.sub("", sentence)
        modified_sentences.append(sentence)
    return modified_sentences


# sieve score
@torch.no_grad()
def get_sieve_scores(args, dataloader, file_path):
    encoder_model_name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(encoder_model_name)

    device = torch.device(args.device)
    model.to(device)  # type: ignore
    model.eval()
    clip_model_name = "ViT-L-14"
    clip_tokenizer = get_tokenizer(clip_model_name)
    clip_model, _, _ = create_model_and_transforms(
        pretrained="openai",
        model_name=clip_model_name,
        precision="fp32",
        device=device,
        jit=True,
        output_dict=True,
        force_quick_gelu=True,
    )
    clip_model.eval()

    blip_model = blip_decoder(
        pretrained="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_14M.pth",
        image_size=224,
        vit="base",
    ).to(device)

    sen_model = SentenceTransformer(
        "sentence-transformers/distiluse-base-multilingual-cased-v1"
    ).to(device)

    blip_model.eval()
    sen_model.eval()
    import time

    with torch.no_grad(), gzip.open(file_path, "wt") as fw:
        for idx, sample in enumerate(dataloader):
            if idx % 10 == 0:
                print(f"Current batch {idx}")
            images = sample[2]
            captions = sample[3]
            keys = sample[0]
            uids = sample[1]
            orig_width = sample[4]
            orig_height = sample[5]

            # Generate Captions (8 Candidates)
            generated_text_list = []
            st = time.time()
            # for im in images:
            #     generated_caption = blip_model.generate(
            #         torch.stack([im]).to(device),
            #         sample=True,
            #         top_p=0.9,
            #         max_length=20,
            #         min_length=5,
            #     )
            #     generated_text_list.append(generated_caption)

            generated_text_list = blip_model.generate(
                images.to(device),
                sample=True,
                top_p=0.9,
                max_length=20,
                min_length=5,
            )
            print(time.time() - st)
            # generated_text_list.append(generated_caption)

            # Remove Medium Phrases
            generated_text_list = [
                remove_phrases(array)[0:NUM_CAPTIONS] for array in generated_text_list
            ]

            # Get CLIP Embeddings for Captions
            tokenized_captions = torch.stack([clip_tokenizer(cap)[0] for cap in captions], dim=0)
            tokenized_captions = tokenized_captions.to(device)
            cap_embs = clip_model.encode_text(tokenized_captions)
            cap_embs = cap_embs / cap_embs.norm(dim=-1, keepdim=True)

            # Get CLIP Embeddings for Generated Captions
            gen_emb_list = []
            for gen_texts in generated_text_list:
                tokenized_captions = torch.stack(
                    [clip_tokenizer(cap)[0] for cap in gen_texts], dim=0
                )
                tokenized_captions = tokenized_captions.to(device)
                gen_embs = clip_model.encode_text(tokenized_captions)
                gen_embs = gen_embs / gen_embs.norm(dim=-1, keepdim=True)
                gen_emb_list.append(gen_embs)

            max_sim_list = []

            # Compute Max Cosine Similarity between Captions, Generated Captions
            for n in range(len(cap_embs)):
                sim_list = []
                for i in range(NUM_CAPTIONS):
                    cosine_sim = F.cosine_similarity(cap_embs[n], gen_emb_list[n][i], dim=0)
                    cosine_sim = cosine_sim.cpu().numpy().tolist()
                    sim_list.append(cosine_sim)
                max_sim_list.append(max(sim_list))

            orig_width = orig_width.tolist()
            orig_height = orig_height.tolist()

            for i in range(len(uids)):
                result = {
                    "uid": uids[i],
                    "key": keys[i],
                    "original_width": orig_width[i],
                    "original_height": orig_height[i],
                    "text": captions[i],
                    f"{args.method}": max_sim_list[i],
                }
                json.dump(result, fw, separators=(",", ":"))
                fw.write("\n")

    return
