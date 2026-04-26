#!/usr/bin/env python3
"""
Offline quality evaluation for the GPS-Gaussian + LiVoGS streaming pipeline.

Loads decoded Gaussian parameter .pt files saved by streaming_pipeline.py,
re-renders novel views using the Gaussian rasterizer, and compares against
baseline renders (from test_real_data_fast.py) to measure quality degradation
introduced by LiVoGS compression.

Metrics reported: per-frame and average PSNR and SSIM.

Usage:
  python scripts/evaluate_streaming.py \
      --decoded_dir ./streaming_out/decoded_frames \
      --baseline_dir ./fast_out_compiled \
      --test_data_root '/path/to/real_data' \
      --ckpt_path './models/GPS-GS_stage2_final.pth' \
      --src_view 0 1
"""
from __future__ import print_function, division

import argparse
import csv
import logging
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from gaussian_renderer import render
from lib.human_loader import StereoHumanDataset
from lib.utils import get_novel_calib
from config.stereo_human_config import ConfigStereoHuman as config


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def psnr(img1, img2):
    """PSNR between two [C, H, W] tensors in [0, 1]."""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return -10.0 * math.log10(mse)


def _gaussian_kernel_1d(size, sigma):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def ssim(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    """Structural Similarity between two [C, H, W] tensors in [0, 1].

    Uses a Gaussian sliding window per-channel and returns the mean SSIM.
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    channels = img1.shape[1]
    kernel_1d = _gaussian_kernel_1d(window_size, 1.5).to(img1.device)
    kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
    window = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()

    mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size // 2,
                                     groups=channels)
    mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size // 2,
                                     groups=channels)
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(
        img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(
        img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(
        img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


# ---------------------------------------------------------------------------
# Render from decoded params
# ---------------------------------------------------------------------------

def render_from_params(params, novel_view_data, bg_color):
    """Render a novel view from decoded Gaussian parameters.

    Args:
        params: dict with {means, quats, scales, opacities, colors} on GPU
        novel_view_data: dict with camera params (FovX, FovY, width, height,
                         world_view_transform, full_proj_transform,
                         camera_center)
        bg_color: list of 3 floats

    Returns:
        rendered image tensor [3, H, W] in [0, 1]
    """
    data_for_render = {'novel_view': novel_view_data}

    rendered = render(
        data_for_render,
        idx=0,
        pts_xyz=params['means'],
        pts_rgb=params['colors'],
        rotations=params['quats'],
        scales=params['scales'],
        opacity=params['opacities'].unsqueeze(1),
        bg_color=bg_color,
    )
    return rendered


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def _save_render(tensor, path):
    """Save a [3, H, W] float tensor in [0, 1] as a JPEG."""
    img = (tensor.clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def evaluate(decoded_dir, baseline_dir, dataset, cfg_dataset,
             view_select, ratio=0.5, output_csv=None, save_renders_dir=None):
    """Run quality evaluation on all decoded frames."""

    total_frames = len(os.listdir(
        os.path.join(cfg_dataset.test_data_root, 'img')))

    logging.info(f"Preparing camera data for {total_frames} frames …")
    camera_data_list = []
    frame_names = []

    for idx in tqdm(range(total_frames), desc="Preparing cameras"):
        item = dataset.get_test_item(idx, source_id=view_select)
        frame_name = item['name']
        frame_names.append(frame_name)

        for view in ('lmain', 'rmain'):
            for key in item[view]:
                item[view][key] = item[view][key].unsqueeze(0)
        for key in item['novel_view']:
            if torch.is_tensor(item['novel_view'][key]):
                item['novel_view'][key] = item['novel_view'][key].unsqueeze(0)

        data_gpu = {}
        for view in ('lmain', 'rmain'):
            data_gpu[view] = {}
            for key, val in item[view].items():
                data_gpu[view][key] = val.cuda() if torch.is_tensor(val) else val
        data_gpu['novel_view'] = {}
        for key, val in item['novel_view'].items():
            data_gpu['novel_view'][key] = val.cuda() if torch.is_tensor(val) else val
        data_gpu['name'] = frame_name

        data_gpu = get_novel_calib(data_gpu, cfg_dataset, ratio=ratio,
                                   intr_key='intr_ori', extr_key='extr_ori')

        nv = {}
        for key, val in data_gpu['novel_view'].items():
            nv[key] = val.cpu() if torch.is_tensor(val) else val
        camera_data_list.append(nv)

        del data_gpu

    logging.info("Camera data ready.")

    if save_renders_dir:
        os.makedirs(save_renders_dir, exist_ok=True)
        logging.info(f"Saving renders to {save_renders_dir}")

    bg_color = cfg_dataset.bg_color
    csv_rows = []
    all_psnr, all_ssim = [], []
    skipped = 0

    for idx in tqdm(range(total_frames), desc="Evaluating"):
        frame_name = frame_names[idx]
        pt_path = os.path.join(decoded_dir, f'{frame_name}.pt')
        baseline_path = os.path.join(baseline_dir, f'{frame_name}_novel.jpg')

        if not os.path.exists(pt_path):
            skipped += 1
            continue
        if not os.path.exists(baseline_path):
            logging.warning(f"Baseline image not found: {baseline_path}")
            skipped += 1
            continue

        params = torch.load(pt_path, map_location='cuda', weights_only=True)

        nv_gpu = {}
        for key, val in camera_data_list[idx].items():
            nv_gpu[key] = val.cuda() if torch.is_tensor(val) else val

        with torch.no_grad():
            rendered = render_from_params(params, nv_gpu, bg_color)

        baseline_bgr = cv2.imread(baseline_path)
        baseline_rgb = cv2.cvtColor(baseline_bgr, cv2.COLOR_BGR2RGB)
        baseline_tensor = torch.from_numpy(
            baseline_rgb.astype(np.float32) / 255.0
        ).permute(2, 0, 1).cuda()

        p = psnr(rendered, baseline_tensor)
        s = ssim(rendered, baseline_tensor)

        if save_renders_dir:
            _save_render(rendered,
                         os.path.join(save_renders_dir, f'{frame_name}_decoded.jpg'))
            _save_render(baseline_tensor,
                         os.path.join(save_renders_dir, f'{frame_name}_baseline.jpg'))

        all_psnr.append(p)
        all_ssim.append(s)

        csv_rows.append({
            'frame_name': frame_name,
            'psnr': p,
            'ssim': s,
            'num_gaussians': params['means'].shape[0],
        })

    if skipped > 0:
        logging.info(f"Skipped {skipped} frames (missing .pt or baseline)")

    if not csv_rows:
        print("No frames evaluated.")
        return

    avg_psnr = np.mean(all_psnr)
    avg_ssim = np.mean(all_ssim)

    print(f"\n{'=' * 60}")
    print("Quality Evaluation: LiVoGS-decoded vs. GPS-Gaussian baseline")
    print(f"{'=' * 60}")
    print(f"  Frames evaluated: {len(csv_rows)}")
    print(f"  PSNR  mean={avg_psnr:.2f} dB  "
          f"std={np.std(all_psnr):.2f}  "
          f"min={np.min(all_psnr):.2f}  max={np.max(all_psnr):.2f}")
    print(f"  SSIM  mean={avg_ssim:.4f}  "
          f"std={np.std(all_ssim):.4f}  "
          f"min={np.min(all_ssim):.4f}  max={np.max(all_ssim):.4f}")
    print(f"{'=' * 60}\n")

    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
        with open(output_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            w.writeheader()
            for r in csv_rows:
                w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v)
                            for k, v in r.items()})
        print(f"Per-frame CSV saved to {output_csv}")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    p = argparse.ArgumentParser(
        description="Offline quality evaluation for streaming pipeline")
    p.add_argument('--decoded_dir', type=str, required=True,
                   help='Directory with decoded .pt files from streaming_pipeline.py')
    p.add_argument('--baseline_dir', type=str, required=True,
                   help='Directory with baseline rendered images (*_novel.jpg)')
    p.add_argument('--test_data_root', type=str, required=True)
    p.add_argument('--ckpt_path', type=str, required=True,
                   help='GPS-Gaussian checkpoint (only used for config)')
    p.add_argument('--src_view', type=int, nargs='+', required=True)
    p.add_argument('--ratio', type=float, default=0.5)
    p.add_argument('--output_csv', type=str, default=None,
                   help='Path for per-frame CSV output')
    p.add_argument('--save_renders', type=str, default=None,
                   help='Directory to save decoded and baseline render images')

    args = p.parse_args()

    cfg = config()
    cfg.load(os.path.join(_PROJECT_ROOT, 'config', 'stage2.yaml'))
    cfg = cfg.get_cfg()
    cfg.defrost()
    cfg.batch_size = 1
    cfg.dataset.test_data_root = args.test_data_root
    cfg.dataset.use_processed_data = False
    cfg.restore_ckpt = args.ckpt_path
    cfg.freeze()

    dataset = StereoHumanDataset(cfg.dataset, phase='test')

    save_renders_dir = args.save_renders
    if save_renders_dir:
        os.makedirs(save_renders_dir, exist_ok=True)

    evaluate(
        decoded_dir=args.decoded_dir,
        baseline_dir=args.baseline_dir,
        dataset=dataset,
        cfg_dataset=cfg.dataset,
        view_select=args.src_view,
        ratio=args.ratio,
        output_csv=args.output_csv,
        save_renders_dir=save_renders_dir,
    )

    print("Done.")
