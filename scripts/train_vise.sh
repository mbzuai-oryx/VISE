#!/usr/bin/env bash
# VISE: Visual Invariance Self-Evolution training launcher.
# Trains a single Qwen3-VL policy on raw, unlabeled images with the geometric
# and semantic invariance rewards. No annotations or external reward models.

set -euo pipefail

ulimit -Sc 0
ulimit -Hc 0

python train.py \
  --data_dir /workspace/grounding/images \
  --output_dir ./runs \
  --model_name Qwen/Qwen3-VL-2B-Instruct \
  --wandb_mode online \
  --wandb_project vise \
  --wandb_run_name vise_2b \
  --use_lora \
  --lora_targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,mm_projector \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --total_steps 4000 \
  --lr 1e-6 \
  --kl_target 0.020 \
  --kl_adapt_rate 0.10 \
  --geo_weight 0.5 \
  --sem_weight 0.5 \
  --coordinate_scale 1000 \
  --ghost_blur_sigma 25.0 \
  --ghost_method blur \
  --transform_types affine,crop,flip \
  --translate_range="-50,50" \
  --scale_range="0.9,1.1" \
  --rotate_range="-10,10" \
  --clear_cache_every 10 \
  --save_every 500 \
  --max_checkpoints 2 \
  --wandb_log_images_every 100 \
  --seed 42 \
  --freeze_vision
