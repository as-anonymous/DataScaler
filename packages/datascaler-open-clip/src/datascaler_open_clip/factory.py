import copy
import dataclasses
import json
import logging
import pathlib
import re
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import mup
import open_clip.factory
import open_clip.model
import open_clip.pretrained
import open_clip.transform
import open_clip.transformer
import torch
from calflops import calculate_flops

import datascaler_open_clip.model
from datascaler_open_clip.tokenizer import (
    DEFAULT_CONTEXT_LENGTH,
    HFTokenizer,
    SigLipTokenizer,
    SimpleTokenizer,
)

logger = logging.getLogger(__name__)

HF_HUB_PREFIX = "hf-hub:"
_MODEL_CONFIG_PATHS = [Path(__file__).parent / "model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = (".json",)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f"*{ext}"))

    for cf in config_files:
        with open(cf, "r") as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ("embed_dim", "vision_cfg", "text_cfg")):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {
        k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))
    }


_rescan_model_configs()  # initial populate of model config registry


def get_model_config(model_name):
    """Fetch model config from builtin (local library) configs."""
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None


def get_tokenizer(
    model_name: str = "",
    context_length: Optional[int] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
):
    if model_name.startswith(HF_HUB_PREFIX):
        model_name = model_name[len(HF_HUB_PREFIX) :]
        try:
            config = _get_hf_config(model_name, cache_dir=cache_dir)["model_cfg"]
        except Exception:
            tokenizer = HFTokenizer(
                model_name,
                context_length=context_length or DEFAULT_CONTEXT_LENGTH,
                cache_dir=cache_dir,
                **kwargs,
            )
            return tokenizer
    else:
        config = get_model_config(model_name)
        assert config is not None, f"No valid model config found for {model_name}."

    text_config = config.get("text_cfg", {})
    if "tokenizer_kwargs" in text_config:
        tokenizer_kwargs = dict(text_config["tokenizer_kwargs"], **kwargs)
    else:
        tokenizer_kwargs = kwargs

    if context_length is None:
        context_length = text_config.get("context_length", DEFAULT_CONTEXT_LENGTH)

    model_name = model_name.lower()
    if text_config.get("hf_tokenizer_name", ""):
        tokenizer = HFTokenizer(
            text_config["hf_tokenizer_name"],
            context_length=context_length,
            cache_dir=cache_dir,
            **tokenizer_kwargs,
        )
    elif "siglip" in model_name:
        tn = "gemma" if "siglip2" in model_name else "mc4" if "i18n" in model_name else "c4-en"
        tokenizer = SigLipTokenizer(
            tn,
            context_length=context_length,
            # **tokenizer_kwargs,
        )
    else:
        tokenizer = SimpleTokenizer(
            context_length=context_length,
            **tokenizer_kwargs,
        )

    return tokenizer


def _get_hf_config(
    model_id: str,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Fetch model config from HuggingFace Hub."""
    config_path = open_clip.pretrained.download_pretrained_from_hf(
        model_id,
        filename="open_clip_config.json",
        cache_dir=cache_dir,
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def create_model(
    model_name: str,
    pretrained: str | None = None,
    precision: str = "fp32",
    device: str | torch.device = "cpu",
    jit: bool = False,
    force_quick_gelu: bool = False,
    force_custom_text: bool = False,
    force_patch_dropout: float | None = None,
    force_image_size: int | tuple[int, int] | None = None,
    force_preprocess_cfg: dict[str, Any] | None = None,
    pretrained_image: bool = False,
    pretrained_hf: bool = True,
    cache_dir: str | None = None,
    output_dict: bool | None = None,
    require_pretrained: bool = False,
    load_weights_only: bool = True,
    use_mup: bool = False,
    width_mult: float = 1.0,
    min_width_mult: float = 1.0,
    attn_mult: float = 1.0,
    output_mult: float = 1.0,
    base_shape_path: str | None = None,
    batch_size: int = 4096,
    **model_kwargs,
):
    """Creates and configures a contrastive vision-language model.

    Args:
        model_name: Name of the model architecture to create. Can be a local model name
            or a Hugging Face model ID prefixed with 'hf-hub:'.
        pretrained: Tag/path for pretrained model weights. Can be:
            - A pretrained tag name (e.g., 'openai')
            - A path to local weights
            - None to initialize with random weights
        precision: Model precision/AMP configuration. Options:
            - 'fp32': 32-bit floating point
            - 'fp16'/'bf16': Mixed precision with FP32 for certain layers
            - 'pure_fp16'/'pure_bf16': Pure 16-bit precision
        device: Device to load the model on ('cpu', 'cuda', or torch.device object)
        jit: If True, JIT compile the model
        force_quick_gelu: Force use of QuickGELU activation
        force_custom_text: Force use of custom text encoder
        force_patch_dropout: Override default patch dropout value
        force_image_size: Override default image size for vision encoder
        force_preprocess_cfg: Override default preprocessing configuration
        pretrained_image: Load pretrained weights for timm vision models
        pretrained_hf: Load pretrained weights for HF text models when not loading CLIP weights
        cache_dir: Override default cache directory for downloaded model files
        output_dict: If True and model supports it, return dictionary of features
        require_pretrained: Raise error if pretrained weights cannot be loaded
        load_weights_only: Only deserialize model weights and unpickling torch checkpoints (for safety)
        use_mup: Use MUP
        **model_kwargs: Additional keyword arguments passed to model constructor

    Returns:
        Created and configured model instance

    Raises:
        RuntimeError: If model config is not found or required pretrained weights
            cannot be loaded

    Examples:
        # Create basic CLIP model
        model = create_model('ViT-B/32')

        # Create CLIP model with mixed precision on GPU
        model = create_model('ViT-B/32', precision='fp16', device='cuda')

        # Load pretrained OpenAI weights
        model = create_model('ViT-B/32', pretrained='openai')

        # Load Hugging Face model
        model = create_model('hf-hub:organization/model-name')
    """

    force_preprocess_cfg = force_preprocess_cfg or {}
    preprocess_cfg = dataclasses.asdict(open_clip.transform.PreprocessCfg())
    has_hf_hub_prefix = model_name.startswith(HF_HUB_PREFIX)
    if has_hf_hub_prefix:
        model_id = model_name[len(HF_HUB_PREFIX) :]
        checkpoint_path = open_clip.pretrained.download_pretrained_from_hf(
            model_id, cache_dir=cache_dir
        )
        config = _get_hf_config(model_id, cache_dir=cache_dir)
        preprocess_cfg = open_clip.transform.merge_preprocess_dict(
            preprocess_cfg, config["preprocess_cfg"]
        )
        model_cfg = config["model_cfg"]
        pretrained_hf = False  # override, no need to load original HF text weights
    else:
        model_name = model_name.replace(
            "/", "-"
        )  # for callers using old naming with / in ViT names
        checkpoint_path = None
        model_cfg = None

    if isinstance(device, str):
        device = torch.device(device)

    # model_cfg = model_cfg or open_clip.factory.get_model_config(model_name)
    model_cfg = model_cfg or get_model_config(model_name)
    if model_cfg is not None:
        logger.info(f"Loaded {model_name} model config.")
    else:
        logger.error(
            f"Model config for {model_name} not found; available models {open_clip.factory.list_models()}."
        )
        raise RuntimeError(f"Model config for {model_name} not found.")

    if force_quick_gelu:
        # override for use of QuickGELU on non-OpenAI transformer models
        model_cfg["quick_gelu"] = True

    if force_patch_dropout is not None:
        # override the default patch dropout value
        model_cfg["vision_cfg"]["patch_dropout"] = force_patch_dropout

    if force_image_size is not None:
        # override model config's image size
        model_cfg["vision_cfg"]["image_size"] = force_image_size

    is_timm_model = "timm_model_name" in model_cfg.get("vision_cfg", {})
    if pretrained_image:
        if is_timm_model:
            # pretrained weight loading for timm models set via vision_cfg
            model_cfg["vision_cfg"]["timm_model_pretrained"] = True
        else:
            assert False, "pretrained image towers currently only supported for timm models"

    # cast_dtype set for fp16 and bf16 (manual mixed-precision), not set for 'amp' or 'pure' modes
    cast_dtype = open_clip.model.get_cast_dtype(precision)
    is_hf_model = "hf_model_name" in model_cfg.get("text_cfg", {})
    if is_hf_model:
        # load pretrained weights for HF text model IFF no CLIP weights being loaded
        model_cfg["text_cfg"]["hf_model_pretrained"] = pretrained_hf and not pretrained
    custom_text = model_cfg.pop("custom_text", False) or force_custom_text or is_hf_model

    model_cfg = dict(model_cfg, **model_kwargs)  # merge cfg dict w/ kwargs (kwargs overrides cfg)

    if use_mup:
        assert base_shape_path is not None, "Base shape path must be provided for MUP"

        def gen_model(model_cfg, SP, attn_mult, output_mult, base_shape_name):
            model = datascaler_open_clip.model.CLIP(
                **model_cfg,
                cast_dtype=cast_dtype,
                SP=SP,
                attn_mult=attn_mult,
                output_mult=output_mult,
            )
            mup.set_base_shapes(model, base_shape_name)
            return model

        def make_base_file(
            width_mult, min_width_mult, model_cfg, SP, attn_mult, output_mult, base_shape_name
        ):
            base_model_cfg = copy.deepcopy(model_cfg)

            base_model_cfg["vision_cfg"]["width"] = round(
                (min_width_mult / width_mult) * base_model_cfg["vision_cfg"]["width"]
            )
            base_model_cfg["text_cfg"]["width"] = round(
                (min_width_mult / width_mult) * base_model_cfg["text_cfg"]["width"]
            )

            base_model = datascaler_open_clip.model.CLIP(
                **base_model_cfg,
                cast_dtype=cast_dtype,
                SP=SP,
                attn_mult=attn_mult,
                output_mult=output_mult,
            )
            base_shapes = mup.get_shapes(base_model)

            delta_model = datascaler_open_clip.model.CLIP(
                **model_cfg,
                cast_dtype=cast_dtype,
                SP=SP,
                attn_mult=attn_mult,
                output_mult=output_mult,
            )
            delta_shapes = mup.get_shapes(delta_model)

            mup.make_base_shapes(base_shapes, delta_shapes, savefile=base_shape_name)
            return

        model_cfg["vision_cfg"]["width"] = round(width_mult * model_cfg["vision_cfg"]["width"])
        model_cfg["text_cfg"]["width"] = round(width_mult * model_cfg["text_cfg"]["width"])

        if model_name == "ViT-B-32":
            model_cfg["vision_cfg"]["layers"] = round(
                width_mult * model_cfg["vision_cfg"]["layers"]
            )
            model_cfg["text_cfg"]["layers"] = round(width_mult * model_cfg["text_cfg"]["layers"])
            model_cfg["text_cfg"]["heads"] = round(width_mult * model_cfg["text_cfg"]["heads"])
            model_cfg["embed_dim"] = round(width_mult * model_cfg["embed_dim"])

        base_shape_name = pathlib.Path(base_shape_path) / "clip.bsh"
        if not base_shape_name.is_file():
            make_base_file(
                width_mult,
                min_width_mult,
                model_cfg,
                not use_mup,
                attn_mult,
                output_mult,
                base_shape_name=str(base_shape_name),
            )
        model = gen_model(
            model_cfg,
            not use_mup,
            attn_mult,
            output_mult,
            base_shape_name=str(base_shape_name),
        )
    else:
        if custom_text:
            if "multimodal_cfg" in model_cfg:
                model = open_clip.coca_model.CoCa(**model_cfg, cast_dtype=cast_dtype)
            else:
                model = open_clip.model.CustomTextCLIP(**model_cfg, cast_dtype=cast_dtype)
        else:
            model_cfg["vision_cfg"]["width"] = round(width_mult * model_cfg["vision_cfg"]["width"])
            model_cfg["text_cfg"]["width"] = round(width_mult * model_cfg["text_cfg"]["width"])
            model = open_clip.model.CLIP(**model_cfg, cast_dtype=cast_dtype)

    input_shape = (1, 3, 224, 224)
    flops, macs, params = calculate_flops(
        model=model,
        input_shape=input_shape,
        output_as_string=False,
        print_results=False,
        print_detailed=False,
    )
    logger.info(f"Params: {params}, FLOPs: {flops}, MACs: {macs}")

    if precision in ("fp16", "bf16"):
        dtype = torch.float16 if "fp16" in precision else torch.bfloat16
        # manual mixed precision that matches original OpenAI behaviour
        if is_timm_model:
            # FIXME this is a bit janky, create timm based model in low-precision and
            # then cast only LayerNormFp32 instances back to float32 so they don't break.
            # Why? The convert_weights_to_lp fn only works with native models.
            model.to(device=device, dtype=dtype)

            def _convert_ln(m):
                if isinstance(m, open_clip.transformer.LayerNormFp32):
                    m.weight.data = m.weight.data.to(torch.float32)
                    m.bias.data = m.bias.data.to(torch.float32)

            model.apply(_convert_ln)
        else:
            model.to(device=device)
            open_clip.model.convert_weights_to_lp(model, dtype=dtype)
    elif precision in ("pure_fp16", "pure_bf16"):
        dtype = torch.float16 if "fp16" in precision else torch.bfloat16
        model.to(device=device, dtype=dtype)
    else:
        model.to(device=device)

    pretrained_loaded = False
    if pretrained:
        checkpoint_path = ""
        pretrained_cfg = open_clip.pretrained.get_pretrained_cfg(model_name, pretrained)
        if pretrained_cfg:
            checkpoint_path = open_clip.pretrained.download_pretrained(
                pretrained_cfg, cache_dir=cache_dir
            )
            preprocess_cfg = open_clip.transform.merge_preprocess_dict(
                preprocess_cfg, pretrained_cfg
            )
            pretrained_quick_gelu = pretrained_cfg.get("quick_gelu", False)
            model_quick_gelu = model_cfg.get("quick_gelu", False)
            if pretrained_quick_gelu and not model_quick_gelu:
                warnings.warn(
                    "These pretrained weights were trained with QuickGELU activation but the model config does "
                    'not have that enabled. Consider using a model config with a "-quickgelu" suffix or enable with a flag.'
                )
            elif not pretrained_quick_gelu and model_quick_gelu:
                warnings.warn(
                    "The pretrained weights were not trained with QuickGELU but this activation is enabled in the "
                    "model config, consider using a model config without QuickGELU or disable override flags."
                )
        elif pathlib.Path(pretrained).exists():
            checkpoint_path = pretrained

        if checkpoint_path:
            logger.info(f"Loading pretrained {model_name} weights ({pretrained}).")
            open_clip.factory.load_checkpoint(
                model,  # type: ignore  # from original code
                checkpoint_path,
                weights_only=load_weights_only,
            )
        else:
            error_str = (
                f"Pretrained weights ({pretrained}) not found for model {model_name}."
                f" Available pretrained tags ({open_clip.pretrained.list_pretrained_tags_by_model(model_name)}."
            )
            logger.warning(error_str)
            raise RuntimeError(error_str)
        pretrained_loaded = True
    elif has_hf_hub_prefix:
        logger.info(f"Loading pretrained {model_name} weights ({checkpoint_path}).")
        assert checkpoint_path is not None
        open_clip.factory.load_checkpoint(
            model,  # type: ignore  # from original code
            checkpoint_path,
            weights_only=load_weights_only,
        )
        pretrained_loaded = True

    if require_pretrained and not pretrained_loaded:
        # callers of create_model_from_pretrained always expect pretrained weights
        raise RuntimeError(
            f"Pretrained weights were required for (model: {model_name}, pretrained: {pretrained}) but not loaded."
        )

    if output_dict and hasattr(model, "output_dict"):
        model.output_dict = True  # type: ignore  # chcked has attribute

    if jit:
        model = torch.jit.script(model)

    # set image preprocessing configuration in model attributes for convenience
    if getattr(model.visual, "image_size", None) is not None:
        # use image_size set on model creation (via config or force_image_size arg)
        force_preprocess_cfg["size"] = model.visual.image_size
    open_clip.model.set_model_preprocess_cfg(
        model, open_clip.transform.merge_preprocess_dict(preprocess_cfg, force_preprocess_cfg)
    )

    return model


def create_model_and_transforms(
    model_name: str,
    pretrained: str | None = None,
    precision: str = "fp32",
    device: str | torch.device = "cpu",
    jit: bool = False,
    force_quick_gelu: bool = False,
    force_custom_text: bool = False,
    force_patch_dropout: float | None = None,
    force_image_size: int | tuple[int, int] | None = None,
    image_mean: tuple[float, ...] | None = None,
    image_std: tuple[float, ...] | None = None,
    image_interpolation: str | None = None,
    image_resize_mode: str | None = None,  # only effective for inference
    aug_cfg: dict[str, Any] | open_clip.transform.AugmentationCfg | None = None,
    pretrained_image: bool = False,
    pretrained_hf: bool = True,
    cache_dir: str | None = None,
    output_dict: bool | None = None,
    load_weights_only: bool = True,
    use_mup: bool = False,
    width_mult: float = 1.0,
    min_width_mult: float = 1.0,
    attn_mult: float = 1.0,
    output_mult: float = 1.0,
    base_shape_path: str | None = None,
    batch_size: int = 4096,
    **model_kwargs,
):
    force_preprocess_cfg = open_clip.transform.merge_preprocess_dict(
        {},
        {
            "mean": image_mean,
            "std": image_std,
            "interpolation": image_interpolation,
            "resize_mode": image_resize_mode,
        },
    )

    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_patch_dropout=force_patch_dropout,
        force_image_size=force_image_size,
        force_preprocess_cfg=force_preprocess_cfg,
        pretrained_image=pretrained_image,
        pretrained_hf=pretrained_hf,
        cache_dir=cache_dir,
        output_dict=output_dict,
        load_weights_only=load_weights_only,
        use_mup=use_mup,
        width_mult=width_mult,
        min_width_mult=min_width_mult,
        attn_mult=attn_mult,
        output_mult=output_mult,
        base_shape_path=base_shape_path,
        batch_size=batch_size,
        **model_kwargs,
    )

    pp_cfg = open_clip.transform.PreprocessCfg(**model.visual.preprocess_cfg)  # type: ignore  # from original code

    preprocess_train = open_clip.transform.image_transform_v2(
        pp_cfg,
        is_train=True,
        aug_cfg=aug_cfg,
    )
    preprocess_val = open_clip.transform.image_transform_v2(
        pp_cfg,
        is_train=False,
    )

    return model, preprocess_train, preprocess_val
