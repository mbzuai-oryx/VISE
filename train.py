"""Command-line entry point for VISE training.

Example:
    python train.py --data_dir /path/to/images --wandb_mode online --use_lora
"""

import argparse
import dataclasses
import warnings

warnings.filterwarnings("ignore")

from vise.config import Config, DEFAULT_LORA_TARGETS
from vise.trainer import VISETrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("VISE: Visual Invariance Self-Evolution")

    # Model
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")

    # Data
    p.add_argument("--data_dir", type=str, default="/workspace/grounding/data/images")
    p.add_argument("--output_dir", type=str, default="./runs")

    # Training
    p.add_argument("--total_steps", type=int, default=16180)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--max_checkpoints", type=int, default=2)
    p.add_argument("--start_step", type=int, default=0)
    p.add_argument("--load_adapter", type=str, default=None)

    # Rewards
    p.add_argument("--geo_weight", type=float, default=0.5,
                   help="Weight for the geometric invariance reward R_geo.")
    p.add_argument("--sem_weight", type=float, default=0.5,
                   help="Weight for the semantic invariance reward R_sem.")

    # Coordinate space
    p.add_argument("--coordinate_scale", type=int, default=1000)

    # Ghosting (semantic invariance perturbation)
    p.add_argument("--ghost_blur_sigma", type=float, default=25.0)
    p.add_argument("--ghost_method", type=str, default="blur", choices=["blur", "mean"])

    # Transforms
    p.add_argument("--transform_types", type=str, default="affine,crop,flip")
    p.add_argument("--translate_range", type=str, default="-50,50")
    p.add_argument("--scale_range", type=str, default="0.9,1.1")
    p.add_argument("--rotate_range", type=str, default="-10,10")

    # KL
    p.add_argument("--kl_target", type=float, default=0.020)
    p.add_argument("--kl_adapt_rate", type=float, default=0.10)

    # LoRA
    p.add_argument("--use_lora", action="store_true", default=True)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_targets", type=str, default=",".join(DEFAULT_LORA_TARGETS))

    # Weights & Biases
    p.add_argument("--wandb_mode", type=str, default="disabled",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_project", type=str, default="vise")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default="vise_run_1")
    p.add_argument("--wandb_log_images_every", type=int, default=100)

    # Memory
    p.add_argument("--clear_cache_every", type=int, default=10)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze_vision", action="store_true", default=True)

    return p.parse_args()


def build_config_from_args(args: argparse.Namespace) -> Config:
    transform_types = tuple(s.strip() for s in args.transform_types.split(",") if s.strip())
    translate_range = tuple(int(x) for x in args.translate_range.split(","))
    scale_range = tuple(float(x) for x in args.scale_range.split(","))
    rotate_range = tuple(float(x) for x in args.rotate_range.split(","))
    lora_targets = tuple(s.strip() for s in args.lora_targets.split(",") if s.strip())

    return Config(
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        total_steps=args.total_steps,
        lr=args.lr,
        save_every=args.save_every,
        load_adapter=args.load_adapter,
        start_step=args.start_step,
        max_checkpoints=args.max_checkpoints,
        geo_weight=args.geo_weight,
        sem_weight=args.sem_weight,
        coordinate_scale=args.coordinate_scale,
        ghost_blur_sigma=args.ghost_blur_sigma,
        ghost_method=args.ghost_method,
        transform_types=transform_types,
        translate_range=translate_range,
        scale_range=scale_range,
        rotate_range=rotate_range,
        kl_target=args.kl_target,
        kl_adapt_rate=args.kl_adapt_rate,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_targets,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_log_images_every=args.wandb_log_images_every,
        clear_cache_every=args.clear_cache_every,
        seed=args.seed,
        freeze_vision=args.freeze_vision,
    )


def main():
    args = parse_args()
    cfg = build_config_from_args(args)

    print("=" * 80)
    print("VISE: Visual Invariance Self-Evolution")
    print("=" * 80)
    for key, value in dataclasses.asdict(cfg).items():
        print(f"  {key}: {value}")
    print("=" * 80)

    trainer = VISETrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
