from enum import StrEnum

from pydantic import BaseModel


class DatacompScale(StrEnum):
    debug = "debug"
    small = "small"
    medium = "medium"
    large = "large"
    xlarge = "xlarge"


class DatacompScaleConfig(BaseModel):
    batch_size: int
    learning_rate: float
    train_num_samples: int
    warmup: int
    model: str
    beta2: float | None


SCALE_CONFIGS = {
    "debug": DatacompScaleConfig(
        batch_size=128,
        learning_rate=1e-4,
        train_num_samples=128_000,
        warmup=1,
        model="ViT-B-32",
        beta2=None,
    ),
    "small": DatacompScaleConfig(
        batch_size=4096,
        learning_rate=5e-4,
        train_num_samples=12_800_000,
        warmup=500,
        model="ViT-B-32",
        beta2=None,
    ),
    "medium": DatacompScaleConfig(
        batch_size=4096,
        learning_rate=5e-4,
        train_num_samples=128_000_000,
        warmup=500,
        model="ViT-B-32",
        beta2=None,
    ),
    "large": DatacompScaleConfig(
        batch_size=8192,
        learning_rate=5e-4,
        train_num_samples=1_280_000_000,
        warmup=500,
        model="ViT-B-16",
        beta2=None,
    ),
    "xlarge": DatacompScaleConfig(
        batch_size=90112,
        learning_rate=1e-3,
        train_num_samples=12_800_000_000,
        warmup=10000,
        model="ViT-L-14",
        beta2=0.95,
    ),
}

SIMPLE_NAMES = ["debug", "small", "medium", "large", "xlarge"]


def available_scales(simple_names: bool = False):
    if simple_names:
        return SIMPLE_NAMES
    else:
        return sorted(list(SCALE_CONFIGS.keys()))


def get_scale_config(scale: str):
    if scale not in SCALE_CONFIGS:
        raise ValueError(f"Unknown scale: {scale}. Please use one of {available_scales()}")
    return SCALE_CONFIGS[scale]
