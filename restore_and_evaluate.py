"""
Unified restoration + evaluation runner.

Features:
- Methods: opencv (Telea/NS), lama (via lama-cleaner), or both
- Reads scratched images from output/processed_images and masks from output/label_masks
- Writes restored images to output/restored_images/<method>
- Evaluates PSNR/SSIM vs input_images and writes per-method reports and a combined summary
"""
from __future__ import annotations

import os
import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.io import imread


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_images(path: str) -> List[str]:
    files = [f for f in os.listdir(path) if f.lower().endswith(IMG_EXTS)]
    files.sort()
    return files


def load_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    _, m = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return m


def opencv_inpaint(input_path: str, mask_path: str, output_path: str, method: str, radius: float, dilate_ksize: int, dilate_iter: int) -> None:
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(input_path)
    mask = load_mask(mask_path)
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    if dilate_ksize > 1 and dilate_iter > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
        mask = cv2.dilate(mask, k, iterations=dilate_iter)
    flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    out = cv2.inpaint(img, mask.astype(np.uint8), float(radius), flag)
    cv2.imwrite(output_path, out)


def lama_inpaint(input_path: str, mask_path: str, output_path: str, dilate_ksize: int, dilate_iter: int) -> None:
    try:
        from lama_cleaner.model_manager import ModelManager
        from lama_cleaner.schema import HDStrategy, Config
    except Exception as e:
        raise RuntimeError("lama-cleaner not installed; install it in a Python 3.11 env") from e

    import torch

    bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(input_path)
    mask = load_mask(mask_path)
    if mask.shape[:2] != bgr.shape[:2]:
        mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    if dilate_ksize > 1 and dilate_iter > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
        mask = cv2.dilate(mask, k, iterations=dilate_iter)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    manager = ModelManager(name="lama", device=device)
    cfg = Config(
        ldm_steps=25,
        hd_strategy=HDStrategy.CROP,
        hd_strategy_crop_margin=32,
        hd_strategy_crop_trigger_size=512,
        hd_strategy_resize_limit=1024,
        prompt="",
    )
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out_rgb = manager.model.forward(rgb, mask, cfg)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, out_bgr)


@dataclass
class EvalResult:
    filename: str
    psnr: float
    ssim: float


def eval_dir(gt_dir: str, pred_dir: str) -> Tuple[List[EvalResult], float, float]:
    results: List[EvalResult] = []
    for fname in list_images(gt_dir):
        gt_path = os.path.join(gt_dir, fname)
        pr_path = os.path.join(pred_dir, fname)
        if not os.path.exists(pr_path):
            continue
        gt = imread(gt_path)
        pr = imread(pr_path)
        if pr.shape != gt.shape:
            from skimage.transform import resize
            pr = resize(pr, gt.shape, preserve_range=True, anti_aliasing=True).astype(gt.dtype)
        if gt.ndim == 3:
            gt = np.mean(gt, axis=2)
        if pr.ndim == 3:
            pr = np.mean(pr, axis=2)
        data_range = 255 if gt.dtype == np.uint8 else gt.max() - gt.min()
        psnr = peak_signal_noise_ratio(gt, pr, data_range=data_range)
        ssim = structural_similarity(gt, pr, data_range=data_range)
        results.append(EvalResult(fname, psnr, ssim))

    if results:
        avg_psnr = sum(r.psnr for r in results) / len(results)
        avg_ssim = sum(r.ssim for r in results) / len(results)
    else:
        avg_psnr = 0.0
        avg_ssim = 0.0
    return results, avg_psnr, avg_ssim


def write_report(path: str, results: List[EvalResult], avg_psnr: float, avg_ssim: float) -> None:
    with open(path, 'w') as f:
        for r in results:
            f.write(f"{r.filename}: PSNR={r.psnr:.2f}, SSIM={r.ssim:.4f}\n")
        f.write(f"Average PSNR: {avg_psnr:.2f}\nAverage SSIM: {avg_ssim:.4f}\n")


def write_combined(combined_path: str, by_method: Dict[str, Tuple[List[EvalResult], float, float]]) -> None:
    # Build set of filenames
    names = set()
    for m, (res, _, _) in by_method.items():
        names.update(r.filename for r in res)
    names = sorted(names)
    with open(combined_path, 'w') as f:
        f.write("filename,method,psnr,ssim\n")
        for m, (res, avg_p, avg_s) in by_method.items():
            for r in res:
                f.write(f"{r.filename},{m},{r.psnr:.4f},{r.ssim:.6f}\n")
            f.write(f"AVERAGE,{m},{avg_p:.4f},{avg_s:.6f}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Restore scratched images and evaluate PSNR/SSIM")
    ap.add_argument("--method", choices=["opencv", "lama", "both"], default="both")
    ap.add_argument("--input_dir", default="input_images", help="Ground truth images for evaluation")
    ap.add_argument("--processed_dir", default="output/processed_images")
    ap.add_argument("--mask_dir", default="output/label_masks")
    ap.add_argument("--restored_dir", default="output/restored_images")
    # OpenCV params
    ap.add_argument("--opencv_kind", choices=["telea", "ns"], default="telea")
    ap.add_argument("--opencv_radius", type=float, default=3.0)
    ap.add_argument("--opencv_dilate_ksize", type=int, default=3)
    ap.add_argument("--opencv_dilate_iter", type=int, default=1)
    # LaMa params
    ap.add_argument("--lama_dilate_ksize", type=int, default=1)
    ap.add_argument("--lama_dilate_iter", type=int, default=0)
    args = ap.parse_args()

    ensure_dir(args.restored_dir)

    methods = [args.method] if args.method != "both" else ["opencv", "lama"]
    eval_results: Dict[str, Tuple[List[EvalResult], float, float]] = {}

    files = list_images(args.processed_dir)
    if not files:
        print(f"No images found in {args.processed_dir}")
        sys.exit(0)

    for m in methods:
        out_dir = os.path.join(args.restored_dir, m)
        ensure_dir(out_dir)
        print(f"\n== Restoring with {m} ==")
        for fname in files:
            ip = os.path.join(args.processed_dir, fname)
            mp = os.path.join(args.mask_dir, fname)
            op = os.path.join(out_dir, fname)
            if not os.path.exists(mp):
                print(f"[skip] mask missing: {fname}")
                continue
            try:
                if m == "opencv":
                    opencv_inpaint(ip, mp, op, args.opencv_kind, args.opencv_radius, args.opencv_dilate_ksize, args.opencv_dilate_iter)
                else:
                    lama_inpaint(ip, mp, op, args.lama_dilate_ksize, args.lama_dilate_iter)
                print(f"[ok] {fname}")
            except Exception as e:
                print(f"[err] {fname}: {e}")

        # Evaluate
        res, avg_p, avg_s = eval_dir(args.input_dir, out_dir)
        eval_results[m] = (res, avg_p, avg_s)
        report_path = os.path.join(args.restored_dir, f"psnr_ssim_{m}.txt")
        write_report(report_path, res, avg_p, avg_s)
        print(f"Report written: {report_path}")

    # Combined CSV
    combined_path = os.path.join(args.restored_dir, "psnr_ssim_combined.csv")
    write_combined(combined_path, eval_results)
    print(f"Combined report: {combined_path}")


if __name__ == "__main__":
    main()
