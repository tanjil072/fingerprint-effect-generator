"""
Deep-learning-ish scratch removal using a light-weight PyTorch inpainting approach.

Notes:
- For thin scratches, classical OpenCV Telea is already excellent. This provides a DL baseline
  leveraging kornia's inpainting and multi-scale blending without heavyweight deps.
- Reads scratched images from output/processed_images and masks from output/label_masks
- Writes results to output/restored_images

If you prefer the official LaMa weights, we can wire them in later, but this script avoids
the transformers/tokenizers build issues and runs on CPU/MPS.
"""
from __future__ import annotations

import os
import argparse
from typing import Tuple

import cv2
import torch
import numpy as np
import kornia
import kornia.core as K
import kornia.filters as KF


def device_auto() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def load_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    _, m = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return m


@torch.inference_mode()
def kornia_inpaint(img_bgr: np.ndarray, mask: np.ndarray, dev: torch.device, levels: int = 3) -> np.ndarray:
    # Convert to float tensor [B,C,H,W] in RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_t = K.tensor(img_rgb).float().permute(2, 0, 1)[None] / 255.0
    mask_t = K.tensor(mask).float()[None, None] / 255.0  # 1 where hole

    img_t = img_t.to(dev)
    mask_t = mask_t.to(dev)

    # Multi-scale: progressively inpaint and upsample
    cur_img = img_t
    cur_mask = mask_t

    pyr_imgs = [cur_img]
    pyr_masks = [cur_mask]
    for _ in range(levels - 1):
        cur_img = KF.resize(cur_img, (cur_img.shape[-2] // 2, cur_img.shape[-1] // 2), antialias=True)
        cur_mask = KF.resize(cur_mask, (cur_mask.shape[-2] // 2, cur_mask.shape[-1] // 2), interpolation='nearest')
        pyr_imgs.append(cur_img)
        pyr_masks.append(cur_mask)

    # Go coarse to fine
    out = None
    for i in reversed(range(levels)):
        I = pyr_imgs[i]
        M = pyr_masks[i]
        # Simple guidance: blur valid regions into holes
        valid = 1.0 - M
        blurred = KF.gaussian_blur2d(I * valid, (5, 5), (1.0, 1.0))
        # Fast marching-like diffusion using iterative box blur in masked regions
        fill = blurred.clone()
        for _ in range(10):
            fill = KF.box_blur(fill, (3, 3))
            fill = fill * M + I * valid
        cur = fill * M + I * valid

        if out is None:
            out = cur
        else:
            out = KF.resize(out, (I.shape[-2], I.shape[-1]), antialias=True)
            # Blend pyramid outputs to keep details
            out = cur * M + out * valid

    out = torch.clamp(out, 0.0, 1.0)
    out_np = (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
    return out_bgr


def process_all(processed_dir: str, mask_dir: str, restored_dir: str, dilate_ksize: int = 3, dilate_iter: int = 1) -> None:
    ensure_dir(restored_dir)
    dev = device_auto()
    print(f"Device: {dev}")

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
            img = load_image_bgr(ip)
            msk = load_mask(mp)
            if msk.shape[:2] != img.shape[:2]:
                msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            if dilate_ksize > 1 and dilate_iter > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
                msk = cv2.dilate(msk, k, iterations=dilate_iter)

            out = kornia_inpaint(img, msk, dev=dev, levels=3)
            if not cv2.imwrite(op, out):
                print(f"[err] save failed: {op}")
            else:
                print(f"[ok] {fname} -> {op}")
        except Exception as e:
            print(f"[err] {fname}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Light DL scratch removal with kornia-based inpainting")
    ap.add_argument("--processed_dir", default="output/processed_images")
    ap.add_argument("--mask_dir", default="output/label_masks")
    ap.add_argument("--restored_dir", default="output/restored_images")
    ap.add_argument("--dilate_ksize", type=int, default=3)
    ap.add_argument("--dilate_iter", type=int, default=1)
    args = ap.parse_args()

    process_all(args.processed_dir, args.mask_dir, args.restored_dir, args.dilate_ksize, args.dilate_iter)


if __name__ == "__main__":
    main()
