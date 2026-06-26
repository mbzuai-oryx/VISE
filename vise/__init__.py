"""VISE: Visual Invariance Self-Evolution.

A purely unsupervised, single-model self-evolving framework that strengthens a
large multimodal model's visual conditioning through two complementary
invariance rewards (geometric + semantic).

Modules
-------
- ``config``   : the ``Config`` dataclass + LoRA defaults
- ``utils``    : stateless helpers (tag parsing, box geometry, image transforms)
- ``prompts``  : self-question / grounding / verification prompt templates
- ``model``    : ``ImagePool`` (data) and ``VLMCore`` / ``VLMRole`` (model)
- ``trainer``  : invariance rewards, ``PolicyUpdater``, and ``VISETrainer``
"""

from .config import Config, DEFAULT_LORA_TARGETS
from .model import ImagePool, VLMCore, VLMRole
from .trainer import (
    PolicyUpdater,
    VISETrainer,
    geometric_invariance_reward,
    semantic_invariance_reward,
)

__version__ = "1.0.0"

__all__ = [
    "Config",
    "DEFAULT_LORA_TARGETS",
    "ImagePool",
    "VLMCore",
    "VLMRole",
    "PolicyUpdater",
    "geometric_invariance_reward",
    "semantic_invariance_reward",
    "VISETrainer",
]
