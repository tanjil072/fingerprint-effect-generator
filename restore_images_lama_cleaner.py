"""
Run deep-learning inpainting with LaMa (via lama-cleaner backend) using your masks.

This script:
- Reads scratched images from output/processed_images
- Reads binary masks from output/label_masks (white=to inpaint)
- Writes restored images to output/restored_images

It selects CPU/MPS automatically. For Apple Silicon, this runs on CPU; MPS is used by PyTorch, but
lama-cleaner’s internal code primarily runs standard torch ops which map to MPS when available.
"""
from __future__ import annotations

import os
import argparse
import numpy as np
import cv2
import torch

from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import HDStrategy, Config


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def load_mask_bin(path: str, target_hw: tuple[int, int]) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    if m.shape[:2] != target_hw:
        m = cv2.resize(m, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    _, m = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return m


def torch_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def process_all(processed_dir: str, mask_dir: str, restored_dir: str, dilate_ksize: int = 3, dilate_iter: int = 1) -> None:
    ensure_dir(restored_dir)

    device = torch_device()
    print(f"Using device: {device}")

    # Initialize LaMa model via lama-cleaner
    manager = ModelManager(name="lama", device=device)
    # Basic config; we supply mask and image; white means inpaint region
    cfg = Config(
        ldm_steps=25,
        hd_strategy=HDStrategy.CROP,
        hd_strategy_crop_margin=32,
        hd_strategy_crop_trigger_size=512,
        hd_strategy_resize_limit=1024,
        prompt="",
    )

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = sorted([f for f in os.listdir(processed_dir) if f.lower().endswith(exts)])
    if not files:
        print(f"No images in {processed_dir}")
        return

    for fname in files:
        ip = os.path.join(processed_dir, fname)
        mp = os.path.join(mask_dir, fname)
        op = os.path.join(restored_dir, fname)
        if not os.path.exists(mp):
            print(f"[skip] no mask for {fname}")
            continue

        try:
            bgr = load_image_bgr(ip)
            mask = load_mask_bin(mp, bgr.shape[:2])
            if dilate_ksize > 1 and dilate_iter > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
                mask = cv2.dilate(mask, k, iterations=dilate_iter)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # lama-cleaner expects uint8 RGB and mask uint8 0/255
            out_rgb = manager.model.forward(rgb, mask, cfg)
            out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(op, out_bgr):
                print(f"[err] save failed: {op}")
            else:
                print(f"[ok] {fname} -> {op}")
        except Exception as e:
            print(f"[err] {fname}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Deep learning inpainting with LaMa (lama-cleaner)")
    ap.add_argument("--processed_dir", default="output/processed_images")
    ap.add_argument("--mask_dir", default="output/label_masks")
    ap.add_argument("--restored_dir", default="output/restored_images")
    ap.add_argument("--dilate_ksize", type=int, default=1)
    ap.add_argument("--dilate_iter", type=int, default=0)
    args = ap.parse_args()

    process_all(args.processed_dir, args.mask_dir, args.restored_dir, args.dilate_ksize, args.dilate_iter)


if __name__ == "__main__":
    main()
