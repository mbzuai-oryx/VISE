"""Configuration for VISE (Visual Invariance Self-Evolution)."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

# Default LoRA target modules: attention projections, MLP, and the
# multimodal projector. The vision encoder is kept frozen.
DEFAULT_LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "mm_projector",
)


@dataclass
class Config:
    # Model
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"

    # Coordinate space (boxes are normalized to [0, coordinate_scale])
    coordinate_scale: int = 1000

    # Reward weights
    geo_weight: float = 0.5      # weight for R_geo (Geometric Invariance reward)
    sem_weight: float = 0.5      # weight for R_sem (Semantic Invariance reward)

    # Ghosting parameters (semantic invariance perturbation)
    ghost_blur_sigma: float = 25.0
    ghost_method: str = "blur"         # "blur" or "mean"

    # Geometric transformation parameters
    num_views: int = 2
    transform_types: Tuple[str, ...] = ("affine", "crop", "flip")
    translate_range: Tuple[int, int] = (-50, 50)
    scale_range: Tuple[float, float] = (0.9, 1.1)
    rotate_range: Tuple[float, float] = (-10, 10)  # degrees

    # Device / precision
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16"
    device_map: str = "auto"

    # Training
    total_steps: int = 16180
    batch_size: int = 1
    lr: float = 1e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    temp: float = 1.0
    top_p: float = 1.0
    max_new_tokens_question: int = 64
    max_new_tokens_grounding: int = 64
    max_new_tokens_verify: int = 16

    # Adaptive KL
    kl_coef: float = 1e-3
    kl_target: float = 0.020
    kl_adapt_rate: float = 0.10

    # Data / IO
    data_dir: str = "/workspace/grounding/data/images"
    output_dir: str = "./runs"
    save_every: int = 500
    max_checkpoints: int = 2

    # Freezing
    freeze_vision: bool = True

    # Reproducibility
    seed: int = 42

    # LoRA options
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = DEFAULT_LORA_TARGETS
    load_adapter: Optional[str] = None
    start_step: int = 0

    # Weights & Biases options
    wandb_mode: str = "disabled"
    wandb_project: str = "vise"
    wandb_entity: Optional[str] = None
    wandb_run_name: str = "vise_run_1"
    wandb_log_images_every: int = 100

    # OOM guard
    clear_cache_every: int = 10
