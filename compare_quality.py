"""
Compare image quality between baseline and optimized GPS-Gaussian outputs.

Computes per-frame and aggregate PSNR, SSIM, and LPIPS metrics.
Also produces a summary table and per-frame CSV.

Usage:
    python compare_quality.py \
        --baseline_dir ./test_out \
        --compare_dirs ./fast_out_compiled ./fast_out_hybrid \
        [--output_csv quality_report.csv]
"""
import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    return img.astype(np.float32)


def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(255.0 ** 2 / mse)


def compute_ssim(img1, img2):
    from skimage.metrics import structural_similarity
    return structural_similarity(
        img1, img2,
        channel_axis=2,
        data_range=255.0,
    )


class LPIPSMetric:
    def __init__(self, net='alex'):
        import lpips
        self.model = lpips.LPIPS(net=net, verbose=False).cuda().eval()

    @torch.no_grad()
    def compute(self, img1_np, img2_np):
        """Takes two HxWx3 uint8-range float images, returns scalar LPIPS."""
        t1 = torch.from_numpy(img1_np).permute(2, 0, 1).unsqueeze(0).cuda() / 255.0
        t2 = torch.from_numpy(img2_np).permute(2, 0, 1).unsqueeze(0).cuda() / 255.0
        t1 = 2.0 * t1 - 1.0
        t2 = 2.0 * t2 - 1.0
        return self.model(t1, t2).item()


def compare_directories(baseline_dir, compare_dir, use_lpips=True):
    baseline_files = sorted([
        f for f in os.listdir(baseline_dir)
        if f.endswith(('.jpg', '.png'))
    ])
    compare_files = sorted([
        f for f in os.listdir(compare_dir)
        if f.endswith(('.jpg', '.png'))
    ])

    common = sorted(set(baseline_files) & set(compare_files))
    if not common:
        logging.error(f"No common files between {baseline_dir} and {compare_dir}")
        return []

    if len(common) < len(baseline_files):
        logging.warning(
            f"Only {len(common)}/{len(baseline_files)} files in common "
            f"(baseline has {len(baseline_files)}, compare has {len(compare_files)})"
        )

    lpips_metric = LPIPSMetric() if use_lpips else None

    results = []
    for fname in common:
        img_base = load_image(os.path.join(baseline_dir, fname))
        img_comp = load_image(os.path.join(compare_dir, fname))

        if img_base.shape != img_comp.shape:
            logging.warning(f"Shape mismatch for {fname}: {img_base.shape} vs {img_comp.shape}, skipping")
            continue

        psnr = compute_psnr(img_base, img_comp)
        ssim = compute_ssim(img_base, img_comp)
        lpips_val = lpips_metric.compute(img_base, img_comp) if lpips_metric else None

        results.append({
            'filename': fname,
            'psnr': psnr,
            'ssim': ssim,
            'lpips': lpips_val,
        })

    return results


def print_summary(label, results):
    psnrs = [r['psnr'] for r in results if r['psnr'] != float('inf')]
    ssims = [r['ssim'] for r in results]
    lpipss = [r['lpips'] for r in results if r['lpips'] is not None]

    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"  {len(results)} frames compared")
    print(f"{'=' * 65}")
    print(f"  {'Metric':<10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-' * 55}")

    if psnrs:
        print(f"  {'PSNR (dB)':<10} {np.mean(psnrs):>10.2f} {np.std(psnrs):>10.2f} "
              f"{np.min(psnrs):>10.2f} {np.max(psnrs):>10.2f}")
    if ssims:
        print(f"  {'SSIM':<10} {np.mean(ssims):>10.4f} {np.std(ssims):>10.4f} "
              f"{np.min(ssims):>10.4f} {np.max(ssims):>10.4f}")
    if lpipss:
        print(f"  {'LPIPS':<10} {np.mean(lpipss):>10.4f} {np.std(lpipss):>10.4f} "
              f"{np.min(lpipss):>10.4f} {np.max(lpipss):>10.4f}")
    print(f"{'=' * 65}")

    identical = sum(1 for r in results if r['psnr'] == float('inf'))
    if identical > 0:
        print(f"  ({identical} frames are pixel-identical to baseline)")


def save_csv(all_results, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mode', 'filename', 'psnr_db', 'ssim', 'lpips'])
        for mode_label, results in all_results.items():
            for r in results:
                writer.writerow([
                    mode_label,
                    r['filename'],
                    f"{r['psnr']:.4f}" if r['psnr'] != float('inf') else 'inf',
                    f"{r['ssim']:.6f}",
                    f"{r['lpips']:.6f}" if r['lpips'] is not None else '',
                ])
    logging.info(f"Per-frame CSV saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare GPS-Gaussian output quality')
    parser.add_argument('--baseline_dir', type=str, required=True,
                        help='Directory with baseline (FP32) output images')
    parser.add_argument('--compare_dirs', type=str, nargs='+', required=True,
                        help='One or more directories with optimized output images')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='Labels for each compare_dir (defaults to dirname)')
    parser.add_argument('--output_csv', type=str, default='quality_report.csv',
                        help='Path for per-frame CSV output')
    parser.add_argument('--no_lpips', action='store_true',
                        help='Skip LPIPS (saves GPU memory and time)')
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.compare_dirs):
        parser.error("--labels must have same count as --compare_dirs")

    labels = args.labels or [os.path.basename(d.rstrip('/')) for d in args.compare_dirs]

    logging.info(f"Baseline: {args.baseline_dir}")
    all_results = {}

    for label, cdir in zip(labels, args.compare_dirs):
        logging.info(f"Comparing: {label} ({cdir})")
        results = compare_directories(
            args.baseline_dir, cdir,
            use_lpips=not args.no_lpips,
        )
        all_results[label] = results
        print_summary(f"Baseline vs {label}", results)

    save_csv(all_results, args.output_csv)

    print(f"\n{'#' * 65}")
    print(f"  SUMMARY TABLE")
    print(f"{'#' * 65}")
    header = f"  {'Mode':<25} {'PSNR (dB)':>10} {'SSIM':>10}"
    if not args.no_lpips:
        header += f" {'LPIPS':>10}"
    print(header)
    print(f"  {'-' * (55 if not args.no_lpips else 45)}")

    for label, results in all_results.items():
        psnrs = [r['psnr'] for r in results if r['psnr'] != float('inf')]
        ssims = [r['ssim'] for r in results]
        row = f"  {label:<25} {np.mean(psnrs):>10.2f} {np.mean(ssims):>10.4f}"
        if not args.no_lpips:
            lpipss = [r['lpips'] for r in results if r['lpips'] is not None]
            row += f" {np.mean(lpipss):>10.4f}" if lpipss else f" {'N/A':>10}"
        print(row)

    print(f"{'#' * 65}\n")
