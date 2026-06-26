"""The VISE objective and training loop.

Contains the two invariance rewards, the KL-regularized REINFORCE updater, and
``VISETrainer``, the single-model self-evolving loop. Each step runs the full
self-questioning cycle and one policy update:

    1. self-question  ->  query q
    2. ground on x    ->  B_orig
    3. transform x    ->  x', matrix M
    4. ground on x'   ->  B_new
    5. R_geo          =  GIoU(project(B_orig, M), B_new) normalized
    6. R_sem          =  ghost(B_orig) visibility test
    7. R_total        =  geo_weight*R_geo + sem_weight*R_sem
    8. REINFORCE update on the grounding completion
"""

import dataclasses
import gc
import json
import os
import random
import re
import shutil
import time
from typing import Dict, Tuple

from PIL import Image

import torch
from transformers import set_seed

from .config import Config
from .model import ImagePool, VLMCore, VLMRole
from .prompts import (
    build_grounding_prompt,
    build_self_question_prompt,
    build_verification_prompt,
)
from .utils import (
    apply_affine_transform,
    apply_crop_transform,
    apply_flip_transform,
    apply_ghosting,
    clip_grad_norm_multi_device,
    compute_giou,
    denormalize_box,
    parse_box,
    parse_visibility,
    strip_tags,
    transform_box,
)

try:
    import wandb
    HAS_WANDB = True
except Exception:
    HAS_WANDB = False


# ----------------------------------------------------------------------
# Invariance rewards
# ----------------------------------------------------------------------
def geometric_invariance_reward(box_proj: Tuple[float, ...],
                                box_new: Tuple[float, ...]) -> float:
    """R_geo = (GIoU(B_proj, B_new) + 1) / 2, mapped from [-1, 1] to [0, 1].

    Maximized when the box predicted on the transformed view agrees with the
    analytic projection of the original prediction.
    """
    giou = compute_giou(box_proj, box_new)
    return (giou + 1.0) / 2.0


def semantic_invariance_reward(model: VLMRole, image: Image.Image,
                               box: Tuple[int, int, int, int], query: str,
                               *, ghost_method: str = "blur",
                               ghost_sigma: float = 25.0,
                               max_new_tokens: int = 16) -> float:
    """R_sem = 1 iff the object is visible originally and NOT visible after ghosting.

    The region is judged before and after the predicted box is degraded. A model
    relying on language priors stays insensitive to the removal and is penalized.
    """
    verify_prompt = build_verification_prompt(query)

    # 1. Visibility on the original image.
    verify_out_orig = model.generate(
        image=image, prompt=verify_prompt,
        max_new_tokens=max_new_tokens, temperature=0.0,
    )
    visible_orig = parse_visibility(verify_out_orig)

    # 2. Ghost (perturb) the predicted region.
    ghosted_img = apply_ghosting(image, box, method=ghost_method, sigma=ghost_sigma)

    # 3. Visibility on the perturbed image.
    verify_out_ghost = model.generate(
        image=ghosted_img, prompt=verify_prompt,
        max_new_tokens=max_new_tokens, temperature=0.0,
    )
    visible_ghost = parse_visibility(verify_out_ghost)

    # 4. Reward correct evidence sensitivity only.
    if visible_orig and not visible_ghost:
        return 1.0  # present before, absent after -> correctly conditioned
    return 0.0


# ----------------------------------------------------------------------
# Policy updater (KL-regularized REINFORCE)
# ----------------------------------------------------------------------
class PolicyUpdater:
    """Single-step REINFORCE update against a frozen reference policy.

    Loss:  L = -A_t * log p_theta(y|x,q) + beta_t * (log p_theta - log p_ref)
    where the advantage ``A_t = R_t - b_t`` uses an EMA baseline (managed by the
    trainer) and ``beta_t`` adapts to hold the policy near a target divergence.
    """

    def __init__(self, role: VLMRole, role_ref: VLMRole, cfg: Config):
        self.role = role
        self.role_ref = role_ref
        self.cfg = cfg

        params = [p for p in role.core.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(
            params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        self.kl_coef = cfg.kl_coef
        self._step = 0

    def step(self, image: Image.Image, prompt: str, completion: str,
             reward: float, baseline: float) -> Dict[str, float]:
        """Perform one REINFORCE update and adapt the KL coefficient."""
        self._step += 1

        ce_loss, log_prob = self.role.forward_with_logprobs(image, prompt, completion)

        with torch.no_grad():
            _, log_prob_ref = self.role_ref.forward_with_logprobs(image, prompt, completion)

        kl_div = log_prob - log_prob_ref
        kl_val = float(kl_div.item())

        advantage = reward - baseline

        loss_rl = -advantage * log_prob
        loss_kl = self.kl_coef * kl_div
        loss_total = loss_rl + loss_kl

        self.opt.zero_grad()
        loss_total.backward()
        clip_grad_norm_multi_device(self.role.core.model, self.cfg.grad_clip)
        self.opt.step()

        # Adapt beta_t toward the target divergence budget.
        beta_before = self.kl_coef
        if abs(kl_val) > self.cfg.kl_target:
            self.kl_coef *= (1.0 + self.cfg.kl_adapt_rate)
        else:
            self.kl_coef *= (1.0 - self.cfg.kl_adapt_rate)
        self.kl_coef = max(1e-6, self.kl_coef)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        return {
            "ce_loss": float(ce_loss.item()),
            "kl_loss": kl_val,
            "advantage": advantage,
            "kl_coef_before": beta_before,
            "kl_coef_after": float(self.kl_coef),
            "loss_total": float(loss_total.item()),
        }


# ----------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------
class VISETrainer:
    """VISE: Visual Invariance Self-Evolution Trainer."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        set_seed(cfg.seed)
        random.seed(cfg.seed)

        print("[Models] Single model for self-questioning and visual interpretation.")
        core = VLMCore(cfg.model_name, cfg.device, cfg.dtype, cfg, apply_lora=cfg.use_lora)
        self.model = VLMRole(core)

        # Reference model (frozen) for the KL regularizer.
        core_ref = VLMCore(cfg.model_name, cfg.device, cfg.dtype, cfg, apply_lora=False)
        self.model_ref = VLMRole(core_ref)

        self.updater = PolicyUpdater(self.model, self.model_ref, cfg)

        self.pool = ImagePool(cfg)

        # EMA baseline for the advantage.
        self.baseline = 0.0
        self.momentum = 0.9

        self.run_name = cfg.wandb_run_name
        self.run_dir = os.path.join(cfg.output_dir, self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        self.wandb_run = None
        if HAS_WANDB and cfg.wandb_mode != "disabled":
            self.wandb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=self.run_name,
                mode=cfg.wandb_mode,
                config=dataclasses.asdict(cfg),
            )

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def train_step(self, step: int):
        """Single training iteration (one image, one policy update)."""
        cfg = self.cfg

        image, meta = self.pool.sample_by_iter(step)
        self._last_image_path = meta.get("path")

        # PHASE 1: SELF-QUESTIONING
        question_prompt = build_self_question_prompt()
        question_out = self.model.generate(
            image=image,
            prompt=question_prompt,
            max_new_tokens=cfg.max_new_tokens_question,
            temperature=cfg.temp,
            top_p=cfg.top_p,
        )
        query = strip_tags(question_out, "query")
        if not query:
            query = "the main object"

        # PHASE 2: GROUNDING ON ORIGINAL IMAGE
        ground_prompt = build_grounding_prompt(query)
        ground_out_orig = self.model.generate(
            image=image,
            prompt=ground_prompt,
            max_new_tokens=cfg.max_new_tokens_grounding,
            temperature=cfg.temp,
            top_p=cfg.top_p,
        )

        box_orig_norm = parse_box(ground_out_orig)
        if box_orig_norm is None:
            print(f"[Step {step:05d}] Failed to parse box, skipping...")
            return

        # PHASE 3: GEOMETRIC TRANSFORMATION
        transform_type = random.choice(cfg.transform_types)
        try:
            if transform_type == "affine":
                image_new, M = apply_affine_transform(image, cfg)
            elif transform_type == "crop":
                image_new, M = apply_crop_transform(image, cfg)
            else:  # flip
                image_new, M = apply_flip_transform(image)
        except Exception as e:
            print(f"[Step {step:05d}] Transform failed: {e}, skipping...")
            return

        # PHASE 4: GROUNDING ON TRANSFORMED IMAGE
        ground_out_new = self.model.generate(
            image=image_new,
            prompt=ground_prompt,
            max_new_tokens=cfg.max_new_tokens_grounding,
            temperature=cfg.temp,
            top_p=cfg.top_p,
        )

        box_new_norm = parse_box(ground_out_new)
        if box_new_norm is None:
            print(f"[Step {step:05d}] Failed to parse transformed box, skipping...")
            return

        # PHASE 5: COMPUTE R_geo (Geometric Invariance Reward)
        try:
            box_proj_norm = transform_box(box_orig_norm, M, scale=cfg.coordinate_scale)
            r_geo = geometric_invariance_reward(box_proj_norm, box_new_norm)
            giou_raw = compute_giou(box_proj_norm, box_new_norm)
        except Exception as e:
            print(f"[Step {step:05d}] R_geo computation failed: {e}, skipping...")
            return

        # PHASE 6: COMPUTE R_sem (Semantic Invariance Reward)
        w, h = image.size
        box_orig_pix = denormalize_box(box_orig_norm, w, h, cfg.coordinate_scale)
        try:
            r_sem = semantic_invariance_reward(
                self.model, image, box_orig_pix, query,
                ghost_method=cfg.ghost_method,
                ghost_sigma=cfg.ghost_blur_sigma,
                max_new_tokens=cfg.max_new_tokens_verify,
            )
        except Exception as e:
            print(f"[Step {step:05d}] R_sem computation failed: {e}, using r_sem=0")
            r_sem = 0.0

        # PHASE 7: COMPOSITE REWARD
        reward_total = cfg.geo_weight * r_geo + cfg.sem_weight * r_sem

        # PHASE 8: UPDATE MODEL
        stats_orig = self.updater.step(
            image=image,
            prompt=ground_prompt,
            completion=ground_out_orig,
            reward=reward_total,
            baseline=self.baseline,
        )

        # Update EMA baseline.
        self.baseline = self.momentum * self.baseline + (1 - self.momentum) * reward_total

        # PHASE 9: LOGGING
        print(f"[Step {step:05d}] query='{query}' | "
              f"R_geo={r_geo:.3f} R_sem={r_sem:.3f} R_total={reward_total:.3f} | "
              f"GIoU={giou_raw:.3f}")

        if self.wandb_run:
            metrics = {
                "train/step": step,
                "train/r_geo": r_geo,
                "train/r_sem": r_sem,
                "train/r_total": reward_total,
                "train/baseline": self.baseline,
                "train/giou_raw": giou_raw,
                "model/ce_loss": stats_orig["ce_loss"],
                "model/kl_loss": stats_orig["kl_loss"],
                "model/kl_coef": stats_orig["kl_coef_after"],
                "model/advantage": stats_orig["advantage"],
                "text/query": query,
                "data/transform_type": transform_type,
            }
            if cfg.wandb_log_images_every > 0 and (step % cfg.wandb_log_images_every) == 0:
                try:
                    metrics["vis/image_orig"] = wandb.Image(image, caption=f"Step {step}")
                    metrics["vis/image_transformed"] = wandb.Image(image_new, caption=f"{transform_type}")
                except Exception:
                    pass
            wandb.log(metrics, step=step)

        # PHASE 10: CHECKPOINTING
        if step % 500 == 0:
            self._save_checkpoint(step, is_epoch_checkpoint=True)
        if cfg.save_every and (step % cfg.save_every) == 0 and step % 500 != 0:
            self._save_checkpoint(step, is_epoch_checkpoint=False)

        if cfg.clear_cache_every > 0 and (step % cfg.clear_cache_every) == 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _save_checkpoint(self, step: int, is_epoch_checkpoint: bool = False):
        """Atomically save a model checkpoint (adapter or full)."""
        if is_epoch_checkpoint:
            checkpoint_name = f"epoch_{step:05d}"
        else:
            checkpoint_name = f"step_{step:05d}"

        final_dir = os.path.join(self.run_dir, checkpoint_name)
        tmp_dir = final_dir + ".tmp"

        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)

        solver_dir = os.path.join(tmp_dir, "solver")
        os.makedirs(solver_dir, exist_ok=True)

        try:
            if self.model.core.is_lora:
                self.model.core.model.save_pretrained(solver_dir, save_adapter=True)
            else:
                self.model.core.model.save_pretrained(solver_dir)
        except Exception as e:
            print(f"[Checkpoint] Model save failed: {e}")
            return

        try:
            if self.model.core.tokenizer is not None:
                self.model.core.tokenizer.save_pretrained(solver_dir)
            if self.model.core.processor is not None:
                self.model.core.processor.save_pretrained(solver_dir)
        except Exception:
            pass

        meta = {
            "model_name": self.model.core.model_name,
            "is_lora": self.model.core.is_lora,
            "step": step,
            "is_epoch_checkpoint": is_epoch_checkpoint,
            "time": int(time.time()),
        }
        with open(os.path.join(solver_dir, "checkpoint_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        with open(os.path.join(tmp_dir, "SAVE_OK"), "w") as f:
            f.write("ok\n")

        try:
            os.replace(tmp_dir, final_dir)
            checkpoint_type = "Epoch" if is_epoch_checkpoint else "Regular"
            print(f"[Checkpoint] Saved ({checkpoint_type}): {os.path.basename(final_dir)}")
        except Exception:
            print(f"[Checkpoint] Rename failed, kept: {tmp_dir}")

        if not is_epoch_checkpoint:
            self._prune_checkpoints()

    def _prune_checkpoints(self):
        """Keep only the last K regular checkpoints (epoch checkpoints are kept)."""
        if not os.path.isdir(self.run_dir):
            return

        step_dirs = []
        for d in os.listdir(self.run_dir):
            if d.startswith("step_") and os.path.isfile(os.path.join(self.run_dir, d, "SAVE_OK")):
                m = re.match(r"step_(\d+)", d)
                if m:
                    step_dirs.append((int(m.group(1)), os.path.join(self.run_dir, d)))

        step_dirs.sort(key=lambda x: x[0])

        if len(step_dirs) > self.cfg.max_checkpoints:
            to_delete = step_dirs[:-self.cfg.max_checkpoints]
            for _step, path in to_delete:
                try:
                    shutil.rmtree(path, ignore_errors=True)
                    print(f"[Checkpoint] Pruned: {os.path.basename(path)}")
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def train(self):
        """Run the main self-evolving training loop."""
        cfg = self.cfg
        print(f"Starting VISE training for {cfg.total_steps} steps.")
        print(f"Data: {cfg.data_dir}")
        print(f"Output: {self.run_dir}")
        print("=" * 80)

        for step in range(cfg.start_step + 1, cfg.total_steps + 1):
            try:
                self.train_step(step)
            except Exception as e:
                print(f"[Step {step:05d}] ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

        print("=" * 80)
        print("Training complete!")

        if self.wandb_run:
            wandb.finish()
