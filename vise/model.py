"""Runtime components: the image pool (data) and the VLM wrapper (model).

- ``ImagePool`` deterministically samples raw, unlabeled images.
- ``VLMCore`` loads the base VLM (optionally with LoRA) and freezes the encoder.
- ``VLMRole`` provides generation and log-prob scoring over a ``VLMCore``.
"""

import os
import random
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
)

from .config import Config
from .utils import safe_dtype

try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False


# ----------------------------------------------------------------------
# Data loader
# ----------------------------------------------------------------------
class ImagePool:
    """Recursively indexes raw images under ``cfg.data_dir`` and samples them.

    No captions, boxes, or labels are read; VISE trains on pixels only.
    """

    DEFAULT_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.paths: List[str] = []

        root = os.path.abspath(cfg.data_dir)
        if not os.path.isdir(root):
            raise RuntimeError(f"[ImagePool] data_dir not found: {root}")

        def _is_img(fn: str) -> bool:
            fnl = fn.lower()
            return fnl.endswith(self.DEFAULT_EXTS) and not os.path.basename(fnl).startswith(".")

        for r, _dirs, files in os.walk(root):
            for fn in files:
                if _is_img(fn):
                    full = os.path.join(r, fn)
                    self.paths.append(full)

        if not self.paths:
            raise RuntimeError(f"[ImagePool] No images found under: {root}")

        self.paths.sort()
        print(f"[ImagePool] Found {len(self.paths)} images under: {root}")

        self.indices = list(range(len(self.paths)))
        rnd = random.Random(cfg.seed)
        rnd.shuffle(self.indices)

        self._root = root

    def __len__(self) -> int:
        return len(self.paths)

    def _build_meta(self, p: str) -> dict:
        rel = os.path.relpath(p, self._root)
        return {
            "dataset": "coco",
            "split": "train",
            "path": p,
            "rel_path": rel,
        }

    def sample_by_iter(self, iter_no: int) -> Tuple[Image.Image, dict]:
        """Deterministic sample by (shuffled) iteration number."""
        idx = self.indices[(max(1, int(iter_no)) - 1) % len(self.paths)]
        p = self.paths[idx]
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            return self.sample_by_iter(iter_no + 1)
        meta = self._build_meta(p)
        return img, meta


# ----------------------------------------------------------------------
# VLM core + role
# ----------------------------------------------------------------------
class VLMCore:
    """Loads the base VLM, freezes the vision encoder, and (optionally) adds LoRA."""

    def __init__(self, model_name: str, device: str, dtype: str, cfg: Config,
                 *, apply_lora: bool = False):
        self.device = device
        self.dtype = safe_dtype(dtype)
        self.model_name = model_name
        self.cfg = cfg

        print(f"[Load] {model_name} on {device} ({self.dtype}), device_map={cfg.device_map}")
        self.model: PreTrainedModel = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=cfg.device_map,
        )

        self.processor = AutoProcessor.from_pretrained(model_name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        except Exception:
            self.tokenizer = getattr(self.processor, "tokenizer", None)

        if cfg.freeze_vision:
            for n, p in self.model.named_parameters():
                if "vision" in n.lower() or "visual" in n.lower():
                    p.requires_grad_(False)

        self.is_lora = False

        if apply_lora and HAS_PEFT:
            if cfg.load_adapter:
                try:
                    self.model = PeftModel.from_pretrained(self.model, cfg.load_adapter)
                    self.is_lora = True
                    print(f"[LoRA] Loaded adapter from: {cfg.load_adapter}")

                    for name, param in self.model.named_parameters():
                        if "lora_" in name:
                            param.requires_grad = True

                    print("[LoRA] Enabled training mode for adapter parameters")

                except Exception as e:
                    print(f"[LoRA] Failed to load adapter: {e}")
            else:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=cfg.lora_r,
                    lora_alpha=cfg.lora_alpha,
                    lora_dropout=cfg.lora_dropout,
                    target_modules=list(cfg.lora_target_modules),
                )
                self.model = get_peft_model(self.model, lora_config)
                self.is_lora = True
                print(f"[LoRA] Created new adapter (r={cfg.lora_r}, alpha={cfg.lora_alpha})")

        if self.is_lora:
            self.model.print_trainable_parameters()


class VLMRole:
    """A thin role over a :class:`VLMCore` providing generation and log-prob scoring."""

    def __init__(self, core: VLMCore):
        self.core = core

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 128,
                 temperature: float = 1.0, top_p: float = 1.0) -> str:
        """Generate text conditioned on an image + prompt."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.core.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.core.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.core.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.core.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=(temperature > 0),
            )

        input_ids = inputs["input_ids"]
        generated_ids = [
            output_ids[i][len(input_ids[i]):]
            for i in range(len(output_ids))
        ]
        output_text = self.core.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text

    def forward_with_logprobs(self, image: Image.Image, prompt: str,
                              completion: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (CE loss, mean token log-prob) for the completion.

        Used both for the REINFORCE policy gradient and for the KL proxy against
        the frozen reference model.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.core.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = text + completion

        inputs = self.core.processor(
            text=[full_text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.core.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        input_ids = inputs["input_ids"]

        outputs = self.core.model(**inputs)
        logits = outputs.logits  # (1, seq_len, vocab_size)

        prompt_ids = self.core.processor.tokenizer(text, return_tensors="pt").input_ids
        prompt_len = prompt_ids.shape[1]

        shift_logits = logits[:, prompt_len - 1:-1, :]
        shift_labels = input_ids[:, prompt_len:]

        ce_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="mean",
        )

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=2,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)

        avg_log_prob = token_log_probs.mean()

        return ce_loss, avg_log_prob
