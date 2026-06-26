"""Stateless helpers for VISE.

Grouped here (rather than scattered across many tiny modules):
  * generic helpers:     dtype selection, tag parsing, gradient clipping
  * bounding-box utils:  parsing, (de)normalization, IoU/GIoU, projection
  * image transforms:    affine / crop / flip (geometric branch) + ghosting
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False
    print("[WARNING] opencv-python not found. Install with: pip install opencv-python")


# ----------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------
def clip_grad_norm_multi_device(model: nn.Module, max_norm: float) -> None:
    """Clip gradient norm per-device (supports ``device_map='auto'`` sharding)."""
    by_dev: Dict[torch.device, List[nn.Parameter]] = {}
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            by_dev.setdefault(p.grad.device, []).append(p)
    for _dev, params in by_dev.items():
        nn.utils.clip_grad_norm_(params, max_norm)


def safe_dtype(dtype: str) -> torch.dtype:
    """Resolve a dtype string to a torch dtype, falling back when unsupported."""
    if dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if dtype == "float16" and torch.cuda.is_available():
        return torch.float16
    return torch.float32


def strip_tags(text: str, tag: str) -> Optional[str]:
    """Extract the content between ``<tag>`` and ``</tag>``."""
    lt, rt = f"<{tag}>", f"</{tag}>"
    if lt in text and rt in text:
        s = text.split(lt, 1)[1]
        s = s.split(rt, 1)[0]
        return s.strip()
    return None


def parse_visibility(text: str) -> bool:
    """Parse a binary visibility judgment from a ``<visible>...</visible>`` tag."""
    visible_str = strip_tags(text, "visible")
    if visible_str:
        return visible_str.lower().strip() in ["yes", "true", "1"]
    return False


# ----------------------------------------------------------------------
# Bounding-box utilities
# ----------------------------------------------------------------------
def parse_box(text: str) -> Optional[Tuple[float, float, float, float]]:
    """Extract box coordinates from model output like ``<box>x1,y1,x2,y2</box>``."""
    box_str = strip_tags(text, "box")
    if not box_str:
        return None
    try:
        coords_str = box_str.replace(" ", ",")
        coords = [float(x.strip()) for x in coords_str.split(",") if x.strip()]
        if len(coords) == 4:
            return tuple(coords)
    except Exception:
        pass
    return None


def normalize_box(box: Tuple[float, ...], img_width: int, img_height: int,
                  scale: int = 1000) -> Tuple[float, float, float, float]:
    """Normalize pixel-space box coordinates to the ``[0, scale]`` range."""
    x1, y1, x2, y2 = box
    x1_norm = (x1 / img_width) * scale
    y1_norm = (y1 / img_height) * scale
    x2_norm = (x2 / img_width) * scale
    y2_norm = (y2 / img_height) * scale
    return (x1_norm, y1_norm, x2_norm, y2_norm)


def denormalize_box(box: Tuple[float, ...], img_width: int, img_height: int,
                    scale: int = 1000) -> Tuple[int, int, int, int]:
    """Convert a normalized box back to pixel coordinates."""
    x1, y1, x2, y2 = box
    x1_pix = int((x1 / scale) * img_width)
    y1_pix = int((y1 / scale) * img_height)
    x2_pix = int((x2 / scale) * img_width)
    y2_pix = int((y2 / scale) * img_height)
    return (x1_pix, y1_pix, x2_pix, y2_pix)


def compute_iou(box1: Tuple[float, ...], box2: Tuple[float, ...]) -> float:
    """Compute Intersection over Union of two axis-aligned boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / max(union, 1e-6)


def compute_giou(box1: Tuple[float, ...], box2: Tuple[float, ...]) -> float:
    """Compute Generalized Intersection over Union (GIoU)."""
    iou = compute_iou(box1, box2)

    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1_c = min(x1_1, x1_2)
    y1_c = min(y1_1, y1_2)
    x2_c = max(x2_1, x2_2)
    y2_c = max(y2_1, y2_2)

    c_area = (x2_c - x1_c) * (y2_c - y1_c)

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - iou * max(area1 + area2, 1e-6)

    giou = iou - (c_area - union) / max(c_area, 1e-6)
    return giou


def transform_box(box: Tuple[float, ...], M: np.ndarray,
                  scale: int = 1000) -> Tuple[float, float, float, float]:
    """Project a box through a 3x3 homogeneous matrix ``M`` (B_orig -> B_proj).

    The four corners are lifted to homogeneous coordinates, mapped by ``M``, and
    the axis-aligned bounding box of the result is returned (clamped to ``[0, scale]``).
    """
    x1, y1, x2, y2 = box

    corners = np.array([
        [x1, y1, 1],
        [x2, y1, 1],
        [x2, y2, 1],
        [x1, y2, 1],
    ]).T  # 3x4

    transformed_corners = M @ corners  # 3x4

    x_coords = transformed_corners[0, :]
    y_coords = transformed_corners[1, :]

    x1_new = float(np.min(x_coords))
    y1_new = float(np.min(y_coords))
    x2_new = float(np.max(x_coords))
    y2_new = float(np.max(y_coords))

    x1_new = max(0, min(scale, x1_new))
    y1_new = max(0, min(scale, y1_new))
    x2_new = max(0, min(scale, x2_new))
    y2_new = max(0, min(scale, y2_new))

    return (x1_new, y1_new, x2_new, y2_new)


# ----------------------------------------------------------------------
# Image transformations
# ----------------------------------------------------------------------
def apply_affine_transform(image: Image.Image, cfg) -> Tuple[Image.Image, np.ndarray]:
    """Apply a random affine transform; return transformed image + 3x3 matrix."""
    if not HAS_CV2:
        return image, np.eye(3)

    img_array = np.array(image)
    h, w = img_array.shape[:2]

    tx = random.uniform(cfg.translate_range[0], cfg.translate_range[1])
    ty = random.uniform(cfg.translate_range[0], cfg.translate_range[1])
    scale = random.uniform(cfg.scale_range[0], cfg.scale_range[1])
    angle = random.uniform(cfg.rotate_range[0], cfg.rotate_range[1])

    center = (w / 2, h / 2)
    M_rotate_scale = cv2.getRotationMatrix2D(center, angle, scale)

    M_rotate_scale[0, 2] += tx
    M_rotate_scale[1, 2] += ty

    transformed = cv2.warpAffine(img_array, M_rotate_scale, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(128, 128, 128))

    M_full = np.vstack([M_rotate_scale, [0, 0, 1]])

    return Image.fromarray(transformed), M_full


def apply_crop_transform(image: Image.Image, cfg) -> Tuple[Image.Image, np.ndarray]:
    """Apply a random crop (resized back); return cropped image + 3x3 matrix."""
    w, h = image.size

    crop_ratio = random.uniform(0.8, 1.0)
    crop_w = int(w * crop_ratio)
    crop_h = int(h * crop_ratio)

    left = random.randint(0, max(1, w - crop_w))
    top = random.randint(0, max(1, h - crop_h))

    cropped = image.crop((left, top, left + crop_w, top + crop_h))
    resized = cropped.resize((w, h), Image.BILINEAR)

    scale_x = w / max(crop_w, 1)
    scale_y = h / max(crop_h, 1)

    M = np.array([
        [scale_x, 0, -left * scale_x],
        [0, scale_y, -top * scale_y],
        [0, 0, 1],
    ])

    return resized, M


def apply_flip_transform(image: Image.Image) -> Tuple[Image.Image, np.ndarray]:
    """Apply a horizontal flip; return flipped image + 3x3 matrix."""
    flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
    w, h = image.size

    M = np.array([
        [-1, 0, w],
        [0, 1, 0],
        [0, 0, 1],
    ])

    return flipped, M


def apply_ghosting(image: Image.Image, box: Tuple[int, int, int, int],
                   method: str = "blur", sigma: float = 25.0) -> Image.Image:
    """Ghost (blur or mean-fill) the contents inside the bounding box.

    Used by the semantic invariance reward: degrading the predicted region
    removes the evidence for the queried object, so a well-conditioned model
    should report the object as no longer visible.
    """
    if not HAS_CV2:
        return image

    img_array = np.array(image).copy()
    x1, y1, x2, y2 = box

    x1 = max(0, min(img_array.shape[1], x1))
    x2 = max(0, min(img_array.shape[1], x2))
    y1 = max(0, min(img_array.shape[0], y1))
    y2 = max(0, min(img_array.shape[0], y2))

    if x2 <= x1 or y2 <= y1:
        return image

    if method == "blur":
        roi = img_array[y1:y2, x1:x2].copy()
        ksize = max(3, int(sigma) * 2 + 1)
        blurred = cv2.GaussianBlur(roi, (ksize, ksize), sigma)
        img_array[y1:y2, x1:x2] = blurred
    else:  # "mean"
        roi = img_array[y1:y2, x1:x2]
        mean_color = roi.mean(axis=(0, 1)).astype(np.uint8)
        img_array[y1:y2, x1:x2] = mean_color

    return Image.fromarray(img_array)
