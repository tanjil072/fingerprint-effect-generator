"""
Restore images using OpenCV inpainting (great for thin scratches) with provided masks.

- Reads scratched images from output/processed_images
- Reads binary masks from output/label_masks (white=to inpaint)
- Writes restored images to output/restored_images

You can choose method (telea or ns) and radius. Optionally dilate masks to cover edges.
"""
from __future__ import annotations

import os
import argparse
from typing import Tuple

import cv2
import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def load_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    # Normalize to 0/255 binary
    _, mask_bin = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return mask_bin


def maybe_resize_mask(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    h, w = target_shape
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def dilate_mask(mask: np.ndarray, ksize: int, iterations: int) -> np.ndarray:
    if ksize <= 1 or iterations <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(mask, k, iterations=iterations)


def inpaint_image(img_bgr: np.ndarray, mask: np.ndarray, radius: float, method: str) -> np.ndarray:
    cv_method = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    # OpenCV expects mask 8-bit, non-zero values mark inpaint region
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return cv2.inpaint(img_bgr, mask, inpaintRadius=float(radius), flags=cv_method)


def process_all(
    processed_dir: str,
    mask_dir: str,
    restored_dir: str,
    radius: float = 3.0,
    method: str = "telea",
    dilate_ksize: int = 3,
    dilate_iter: int = 1,
) -> None:
    ensure_dir(restored_dir)

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = [f for f in os.listdir(processed_dir) if f.lower().endswith(exts)]
    files.sort()

    if not files:
        print(f"No images found in {processed_dir}.")
        return

    print(
        f"Inpainting {len(files)} image(s) using method={method}, radius={radius}, "
        f"dilate_ksize={dilate_ksize}, dilate_iter={dilate_iter}"
    )

    for fname in files:
        inp_path = os.path.join(processed_dir, fname)
        msk_path = os.path.join(mask_dir, fname)
        out_path = os.path.join(restored_dir, fname)

        if not os.path.exists(msk_path):
            print(f"[skip] Mask not found for {fname}")
            continue

        try:
            img = load_image_bgr(inp_path)
            mask = load_mask(msk_path)
            mask = maybe_resize_mask(mask, img.shape[:2])
            if dilate_ksize > 1 and dilate_iter > 0:
                mask = dilate_mask(mask, dilate_ksize, dilate_iter)

            restored = inpaint_image(img, mask, radius, method)
            ok = cv2.imwrite(out_path, restored)
            if ok:
                print(f"[ok] {fname} -> {out_path}")
            else:
                print(f"[err] Failed to save {out_path}")
        except Exception as e:
            print(f"[err] {fname}: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Restore scratched images via OpenCV inpainting")
    p.add_argument("--processed_dir", default="output/processed_images", help="Path to scratched images")
    p.add_argument("--mask_dir", default="output/label_masks", help="Path to binary masks (white=to inpaint)")
    p.add_argument("--restored_dir", default="output/restored_images", help="Output folder for restored images")
    p.add_argument("--method", choices=["telea", "ns"], default="telea", help="Inpainting method")
    p.add_argument("--radius", type=float, default=3.0, help="Inpainting radius in pixels")
    p.add_argument("--dilate_ksize", type=int, default=3, help="Elliptical kernel size for mask dilation")
    p.add_argument("--dilate_iter", type=int, default=1, help="Iterations for mask dilation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    process_all(
        processed_dir=args.processed_dir,
        mask_dir=args.mask_dir,
        restored_dir=args.restored_dir,
        radius=args.radius,
        method=args.method,
        dilate_ksize=args.dilate_ksize,
        dilate_iter=args.dilate_iter,
    )


if __name__ == "__main__":
    main()
