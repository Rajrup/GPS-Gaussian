#!/usr/bin/env python3
"""
Live GPS-Gaussian + LiVoGS Streaming Pipeline.

Generates Gaussian splats with GPS-Gaussian, compresses/decompresses each
frame with LiVoGS in real-time, and saves decoded frames for offline
quality evaluation.

Modes:
  sequential:  gen -> enc -> dec per frame (accurate per-stage timing)
  pipelined:   3 threads with queues (max throughput, tests bottleneck hypothesis)

Usage:
  python scripts/streaming_pipeline.py \
      --test_data_root '/path/to/real_data' \
      --ckpt_path './models/GPS-GS_stage2_final.pth' \
      --src_view 0 1 \
      --mode compiled \
      --mode_pipeline sequential \
      --output_dir ./streaming_out
"""
from __future__ import print_function, division

import argparse
import csv
import logging
import os
import sys
import time
import threading
import queue
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_LIVOGS_COMPRESSION = os.path.join(_PROJECT_ROOT, "LiVoGS", "compression")
if _LIVOGS_COMPRESSION not in sys.path:
    sys.path.insert(0, _LIVOGS_COMPRESSION)

from test_real_data_fast import StereoHumanRenderFast, CUDATimer
from config.stereo_human_config import ConfigStereoHuman as config
from compress_decompress import encode_livogs, decode_livogs


# ---------------------------------------------------------------------------
# Cross-device transfer utility (handles broken P2P)
# ---------------------------------------------------------------------------

_P2P_OK = None  # lazily detected

def _test_p2p(src_device, dst_device):
    """Return True if direct GPU-to-GPU .to() works correctly."""
    probe = torch.ones(64, device=src_device)
    torch.cuda.synchronize(src_device)
    out = probe.to(dst_device)
    torch.cuda.synchronize(dst_device)
    return out.sum().item() == 64.0


def transfer_to_device(params, dst_device, src_device=None):
    """Transfer a dict of tensors to *dst_device*.

    Automatically detects broken P2P and falls back to CPU staging
    (src → pinned-CPU → dst) when direct transfers produce zeros.
    """
    global _P2P_OK
    if src_device is None:
        first_val = next(iter(params.values()))
        src_device = str(first_val.device)
    if str(src_device) == str(dst_device):
        return params

    if _P2P_OK is None:
        _P2P_OK = _test_p2p(src_device, dst_device)
        if _P2P_OK:
            logging.info("P2P direct GPU transfer OK")
        else:
            logging.warning(
                "P2P GPU transfer produces zeros — using CPU-staged transfer "
                f"({src_device} -> CPU -> {dst_device})")

    if _P2P_OK:
        return {k: v.to(dst_device) for k, v in params.items()}

    return {k: v.cpu().to(dst_device) for k, v in params.items()}


# ---------------------------------------------------------------------------
# Format Bridge: GPS-Gaussian output -> LiVoGS input
# ---------------------------------------------------------------------------

def extract_gaussian_params(data):
    """Convert GPS-Gaussian per-pixel output maps to a flat LiVoGS params dict.

    All operations are GPU tensor slicing/masking/concat -- zero CPU transfers.

    GPS-Gaussian produces per-view 2D maps for rot, scale, opacity and a
    per-pixel 3D point cloud.  This function flattens them, filters by the
    validity mask, concatenates both views, and returns the dict that
    ``encode_livogs`` expects.
    """
    all_xyz, all_rgb, all_rot, all_scale, all_opacity = [], [], [], [], []

    for view in ('lmain', 'rmain'):
        valid = data[view]['pts_valid'][0]                         # [H*W]
        xyz = data[view]['xyz'][0][valid]                          # [M, 3]

        rgb = data[view]['img'][0].permute(1, 2, 0).reshape(-1, 3)[valid]
        rgb = rgb * 0.5 + 0.5                                     # [-1,1] -> [0,1]

        rot = data[view]['rot_maps'][0].permute(1, 2, 0).reshape(-1, 4)[valid]
        scale = data[view]['scale_maps'][0].permute(1, 2, 0).reshape(-1, 3)[valid]
        opacity = data[view]['opacity_maps'][0].permute(1, 2, 0).reshape(-1, 1)[valid]

        all_xyz.append(xyz)
        all_rgb.append(rgb)
        all_rot.append(rot)
        all_scale.append(scale)
        all_opacity.append(opacity)

    return {
        'means':     torch.cat(all_xyz, dim=0),         # [N, 3]
        'quats':     torch.cat(all_rot, dim=0),          # [N, 4]
        'scales':    torch.cat(all_scale, dim=0),        # [N, 3]
        'opacities': torch.cat(all_opacity, dim=0).squeeze(1),  # [N]
        'colors':    torch.cat(all_rgb, dim=0),          # [N, 3]
    }


# ---------------------------------------------------------------------------
# Background Saving (runs in ThreadPoolExecutor -- only CPU<->GPU transfer)
# ---------------------------------------------------------------------------

def save_frame_worker(decoded_params_gpu, frame_name, save_dir):
    """Move decoded params to CPU and persist as .pt file."""
    params_cpu = {k: v.detach().cpu() for k, v in decoded_params_gpu.items()}
    torch.save(params_cpu, os.path.join(save_dir, f'{frame_name}.pt'))


# ---------------------------------------------------------------------------
# Generation helper (shared by both modes)
# ---------------------------------------------------------------------------

def _generate_one_frame(renderer, data, use_fp16):
    """Run GPS-Gaussian forward pass and extract Gaussian params."""
    with torch.no_grad():
        renderer._run_forward(data, use_fp16)
        if use_fp16:
            renderer._cast_to_fp32(data)
    return extract_gaussian_params(data)


# ---------------------------------------------------------------------------
# Sequential Mode
# ---------------------------------------------------------------------------

def run_sequential(renderer, livogs_cfg, output_dir,
                   save_interval=1, warmup=5):
    total_frames = len(renderer.preloaded_data)
    use_fp16 = renderer.mode in ('compiled', 'tensorrt', 'hybrid')
    timer = CUDATimer()

    gen_device = livogs_cfg['gen_device']
    codec_device = livogs_cfg['codec_device']
    codec_device_id = livogs_cfg['codec_device_id']
    encode_kwargs = livogs_cfg['encode_kwargs']
    same_device = (gen_device == codec_device)

    save_dir = os.path.join(output_dir, 'decoded_frames')
    os.makedirs(save_dir, exist_ok=True)
    save_executor = ThreadPoolExecutor(max_workers=2)
    save_futures = []
    csv_rows = []

    # -- warmup (not timed) --
    logging.info(f"Warming up {min(warmup, total_frames)} frames …")
    for idx in range(min(warmup, total_frames)):
        torch.cuda.set_device(gen_device)
        data = renderer._get_gpu_frame(idx)
        params = _generate_one_frame(renderer, data, use_fp16)
        if not same_device:
            params = transfer_to_device(params, codec_device, gen_device)
        compressed = encode_livogs(params, device=codec_device,
                                   device_id=codec_device_id, **encode_kwargs)
        decode_livogs(compressed, device=codec_device,
                      device_id=codec_device_id)
    torch.cuda.synchronize()
    logging.info("Warmup done.")

    # -- timed loop --
    for idx in tqdm(range(total_frames), desc="Sequential"):
        torch.cuda.set_device(gen_device)
        data = renderer._get_gpu_frame(idx)
        frame_name = data['name']

        timer.start('e2e')

        timer.start('generation')
        params = _generate_one_frame(renderer, data, use_fp16)
        if not same_device:
            params = transfer_to_device(params, codec_device, gen_device)
        timer.stop('generation')

        num_gaussians = params['means'].shape[0]

        timer.start('encode')
        compressed = encode_livogs(params, device=codec_device,
                                   device_id=codec_device_id, **encode_kwargs)
        timer.stop('encode')

        num_voxels = compressed['Nvox']
        compressed_bytes = compressed['total_compressed_bytes']

        timer.start('decode')
        decoded = decode_livogs(compressed, device=codec_device,
                                device_id=codec_device_id)
        timer.stop('decode')

        timer.stop('e2e')
        timer.sync_and_collect()

        if save_interval > 0 and idx % save_interval == 0:
            fut = save_executor.submit(save_frame_worker, decoded,
                                       frame_name, save_dir)
            save_futures.append(fut)

        csv_rows.append({
            'frame_name': frame_name,
            'gen_ms': timer.timings['generation'][-1],
            'enc_ms': timer.timings['encode'][-1],
            'dec_ms': timer.timings['decode'][-1],
            'e2e_ms': timer.timings['e2e'][-1],
            'num_gaussians': num_gaussians,
            'num_voxels': num_voxels,
            'compressed_bytes': compressed_bytes,
        })

    for f in save_futures:
        f.result()
    save_executor.shutdown(wait=True)

    timer.report(skip_first_n=warmup)
    _write_csv(csv_rows, os.path.join(output_dir, 'benchmark_sequential.csv'))
    _print_summary(csv_rows, warmup, 'Sequential')
    return csv_rows


# ---------------------------------------------------------------------------
# Pipelined Mode -- three threads, bounded queues, zero-copy on same device
# ---------------------------------------------------------------------------

_SENTINEL = None  # poison pill to drain the pipeline


class _GenThread(threading.Thread):
    """Generates Gaussian splat parameters from GPS-Gaussian."""

    def __init__(self, renderer, output_queue, gen_device, use_fp16):
        super().__init__(daemon=True, name='GenThread')
        self.renderer = renderer
        self.output_queue = output_queue
        self.gen_device = gen_device
        self.use_fp16 = use_fp16
        self.frame_times = []

    def run(self):
        total = len(self.renderer.preloaded_data)
        torch.cuda.set_device(self.gen_device)
        gen_stream = torch.cuda.Stream(device=self.gen_device)

        for idx in range(total):
            torch.cuda.set_device(self.gen_device)

            with torch.cuda.stream(gen_stream):
                data = self.renderer._get_gpu_frame(idx)
                frame_name = data['name']

                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)

                start_ev.record(gen_stream)
                params = _generate_one_frame(self.renderer, data,
                                             self.use_fp16)
                end_ev.record(gen_stream)

            end_ev.synchronize()
            gen_ms = start_ev.elapsed_time(end_ev)
            self.frame_times.append((frame_name, gen_ms))

            self.output_queue.put((frame_name, params))

        self.output_queue.put(_SENTINEL)


class _TransferThread(threading.Thread):
    """Transfers Gaussian params from gen_device to codec_device via CPU staging.

    Uses a dedicated stream on gen_device for GPU→CPU copies to avoid
    interleaving with GenThread's compute stream.
    """

    def __init__(self, input_queue, output_queue,
                 gen_device, codec_device):
        super().__init__(daemon=True, name='TransferThread')
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.gen_device = gen_device
        self.codec_device = codec_device
        self.frame_times = []

    def run(self):
        d2h_stream = torch.cuda.Stream(device=self.gen_device)
        h2d_stream = torch.cuda.Stream(device=self.codec_device)

        while True:
            item = self.input_queue.get()
            if item is _SENTINEL:
                self.output_queue.put(_SENTINEL)
                break

            frame_name, params = item

            t0 = time.perf_counter()

            with torch.cuda.stream(d2h_stream):
                cpu_params = {k: v.cpu() for k, v in params.items()}
            d2h_stream.synchronize()
            del params

            with torch.cuda.stream(h2d_stream):
                params_dst = {k: v.to(self.codec_device, non_blocking=True)
                              for k, v in cpu_params.items()}
            h2d_stream.synchronize()

            t1 = time.perf_counter()
            self.frame_times.append((frame_name, (t1 - t0) * 1000))

            self.output_queue.put((frame_name, params_dst))


class _EncThread(threading.Thread):
    """Compresses Gaussian parameters with LiVoGS."""

    def __init__(self, input_queue, output_queue,
                 device, device_id, encode_kwargs):
        super().__init__(daemon=True, name='EncThread')
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.device = device
        self.device_id = device_id
        self.encode_kwargs = encode_kwargs
        self.frame_times = []

    def run(self):
        while True:
            item = self.input_queue.get()
            if item is _SENTINEL:
                self.output_queue.put(_SENTINEL)
                break

            frame_name, params = item
            torch.cuda.set_device(self.device)

            torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()

            compressed = encode_livogs(
                params, device=self.device, device_id=self.device_id,
                **self.encode_kwargs)

            torch.cuda.synchronize(self.device)
            t1 = time.perf_counter()
            self.frame_times.append((frame_name, (t1 - t0) * 1000))

            num_gaussians = compressed['N_original']
            num_voxels = compressed['Nvox']
            compressed_bytes = compressed['total_compressed_bytes']

            self.output_queue.put((frame_name, compressed,
                                   num_gaussians, num_voxels,
                                   compressed_bytes))


class _DecThread(threading.Thread):
    """Decompresses bitstream with LiVoGS."""

    def __init__(self, input_queue, output_queue, device, device_id):
        super().__init__(daemon=True, name='DecThread')
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.device = device
        self.device_id = device_id
        self.frame_times = []

    def run(self):
        while True:
            item = self.input_queue.get()
            if item is _SENTINEL:
                self.output_queue.put(_SENTINEL)
                break

            frame_name, compressed, n_gauss, n_vox, comp_bytes = item
            torch.cuda.set_device(self.device)

            torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()

            decoded = decode_livogs(compressed, device=self.device,
                                    device_id=self.device_id)

            torch.cuda.synchronize(self.device)
            t1 = time.perf_counter()
            self.frame_times.append((frame_name, (t1 - t0) * 1000))

            self.output_queue.put((frame_name, decoded,
                                   n_gauss, n_vox, comp_bytes))


def run_pipelined(renderer, livogs_cfg, output_dir,
                  save_interval=1, warmup=5):
    total_frames = len(renderer.preloaded_data)
    use_fp16 = renderer.mode in ('compiled', 'tensorrt', 'hybrid')

    gen_device = livogs_cfg['gen_device']
    codec_device = livogs_cfg['codec_device']
    codec_device_id = livogs_cfg['codec_device_id']
    encode_kwargs = livogs_cfg['encode_kwargs']
    same_device = (gen_device == codec_device)

    save_dir = os.path.join(output_dir, 'decoded_frames')
    os.makedirs(save_dir, exist_ok=True)
    save_executor = ThreadPoolExecutor(max_workers=2)
    save_futures = []

    # -- warmup (sequential, ensures JIT / kernel caches are warm) --
    logging.info(f"Warming up {min(warmup, total_frames)} frames …")
    for idx in range(min(warmup, total_frames)):
        torch.cuda.set_device(gen_device)
        data = renderer._get_gpu_frame(idx)
        params = _generate_one_frame(renderer, data, use_fp16)
        if not same_device:
            params = transfer_to_device(params, codec_device, gen_device)
        compressed = encode_livogs(params, device=codec_device,
                                   device_id=codec_device_id, **encode_kwargs)
        decode_livogs(compressed, device=codec_device,
                      device_id=codec_device_id)
    torch.cuda.synchronize()
    logging.info("Warmup done.")

    # -- build pipeline --
    # Dual-GPU: 4-stage (gen → transfer → enc → dec) so transfer overlaps gen
    # Single-GPU: 3-stage (gen → enc → dec) with no transfer needed
    decode_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=4)

    xfer_t = None
    if same_device:
        encode_queue = queue.Queue(maxsize=1)
        gen_t = _GenThread(renderer, encode_queue, gen_device, use_fp16)
    else:
        transfer_queue = queue.Queue(maxsize=1)
        encode_queue = queue.Queue(maxsize=1)
        gen_t = _GenThread(renderer, transfer_queue, gen_device, use_fp16)
        xfer_t = _TransferThread(transfer_queue, encode_queue,
                                 gen_device, codec_device)

    enc_t = _EncThread(encode_queue, decode_queue,
                       codec_device, codec_device_id, encode_kwargs)
    dec_t = _DecThread(decode_queue, result_queue,
                       codec_device, codec_device_id)

    torch.cuda.synchronize()
    wall_start = time.perf_counter()

    gen_t.start()
    if xfer_t is not None:
        xfer_t.start()
    enc_t.start()
    dec_t.start()

    # -- main thread: collect decoded frames, dispatch background saves --
    csv_rows = []
    frame_count = 0

    while True:
        item = result_queue.get()
        if item is _SENTINEL:
            break

        frame_name, decoded, n_gauss, n_vox, comp_bytes = item
        frame_count += 1

        if save_interval > 0 and (frame_count - 1) % save_interval == 0:
            fut = save_executor.submit(save_frame_worker, decoded,
                                       frame_name, save_dir)
            save_futures.append(fut)

        csv_rows.append({
            'frame_name': frame_name,
            'num_gaussians': n_gauss,
            'num_voxels': n_vox,
            'compressed_bytes': comp_bytes,
        })

    gen_t.join()
    if xfer_t is not None:
        xfer_t.join()
    enc_t.join()
    dec_t.join()

    wall_end = time.perf_counter()

    for f in save_futures:
        f.result()
    save_executor.shutdown(wait=True)

    # -- merge per-thread timings --
    gen_map = {name: ms for name, ms in gen_t.frame_times}
    xfer_map = ({name: ms for name, ms in xfer_t.frame_times}
                if xfer_t is not None else {})
    enc_map = {name: ms for name, ms in enc_t.frame_times}
    dec_map = {name: ms for name, ms in dec_t.frame_times}

    for row in csv_rows:
        fn = row['frame_name']
        row['gen_ms'] = gen_map.get(fn, 0.0)
        row['xfer_ms'] = xfer_map.get(fn, 0.0)
        row['enc_ms'] = enc_map.get(fn, 0.0)
        row['dec_ms'] = dec_map.get(fn, 0.0)
        row['e2e_ms'] = (row['gen_ms'] + row['xfer_ms']
                         + row['enc_ms'] + row['dec_ms'])

    elapsed = wall_end - wall_start
    pipeline_fps = total_frames / elapsed if elapsed > 0 else 0.0

    print(f"\n{'=' * 70}")
    print("Pipelined Mode Results")
    print(f"{'=' * 70}")
    print(f"  gen_device={gen_device}  codec_device={codec_device}")
    if not same_device:
        print(f"  Pipeline:            4-stage (gen → xfer → enc → dec)")
    print(f"  Total frames:        {total_frames}")
    print(f"  Wall-clock time:     {elapsed:.3f} s")
    print(f"  Pipeline throughput: {pipeline_fps:.1f} FPS "
          f"({elapsed * 1000 / max(total_frames, 1):.2f} ms/frame)")
    _print_summary(csv_rows, warmup, 'Pipelined')
    _write_csv(csv_rows, os.path.join(output_dir, 'benchmark_pipelined.csv'))
    return csv_rows


# ---------------------------------------------------------------------------
# Multiprocessing Pipelined Mode -- separate GILs for gen and codec
# ---------------------------------------------------------------------------

def _codec_worker(data_q, result_q, cfg):
    """Codec child process: upload → encode → decode → save on codec_device.

    Runs in a separate process with its own GIL so that the heavy CPU work
    inside LiVoGS native extensions cannot slow down GPU kernel launches
    in the generation process.
    """
    import time
    import torch

    project_root = cfg['project_root']
    livogs_path = os.path.join(project_root, "LiVoGS", "compression")
    if livogs_path not in sys.path:
        sys.path.insert(0, livogs_path)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from compress_decompress import encode_livogs, decode_livogs

    codec_device = cfg['codec_device']
    codec_device_id = cfg['codec_device_id']
    encode_kwargs = cfg['encode_kwargs']
    save_dir = cfg['save_dir']
    save_interval = cfg['save_interval']

    torch.cuda.set_device(codec_device)

    frame_count = 0
    while True:
        item = data_q.get()
        if item is None:
            break

        frame_name, params_cpu, t_enter = item

        t0 = time.perf_counter()
        params = {k: v.to(codec_device) for k, v in params_cpu.items()}
        torch.cuda.synchronize(codec_device)
        t_upload = time.perf_counter()

        compressed = encode_livogs(
            params, device=codec_device, device_id=codec_device_id,
            **encode_kwargs)
        torch.cuda.synchronize(codec_device)
        t_enc = time.perf_counter()

        decoded = decode_livogs(
            compressed, device=codec_device, device_id=codec_device_id)
        torch.cuda.synchronize(codec_device)
        t_dec = time.perf_counter()

        if save_dir and save_interval > 0 and frame_count % save_interval == 0:
            decoded_cpu = {k: v.detach().cpu() for k, v in decoded.items()}
            torch.save(decoded_cpu, os.path.join(save_dir, f'{frame_name}.pt'))

        frame_count += 1
        result_q.put({
            'frame_name': frame_name,
            'num_gaussians': compressed['N_original'],
            'num_voxels': compressed['Nvox'],
            'compressed_bytes': compressed['total_compressed_bytes'],
            'upload_ms': (t_upload - t0) * 1000,
            'enc_ms': (t_enc - t_upload) * 1000,
            'dec_ms': (t_dec - t_enc) * 1000,
            'e2e_wall_ms': (t_dec - t_enter) * 1000,
        })

    result_q.put(None)


def run_pipelined_mp(renderer, livogs_cfg, output_dir,
                     save_interval=1, warmup=5):
    """Multiprocessing pipeline: gen (Process 1) + codec (Process 2).

    Generation runs in the main process on gen_device.  A background copy
    thread (same process, minimal GIL impact) ships CPU tensors through a
    multiprocessing Queue to a child process that runs encode + decode on
    codec_device with its own GIL.
    """
    total_frames = len(renderer.preloaded_data)
    use_fp16 = renderer.mode in ('compiled', 'tensorrt', 'hybrid')
    gen_device = livogs_cfg['gen_device']
    codec_device = livogs_cfg['codec_device']

    save_dir = os.path.join(output_dir, 'decoded_frames')
    os.makedirs(save_dir, exist_ok=True)

    ctx = mp.get_context('spawn')
    data_q = ctx.Queue(maxsize=4)
    result_q = ctx.Queue()

    codec_cfg = {
        'project_root': _PROJECT_ROOT,
        'codec_device': codec_device,
        'codec_device_id': livogs_cfg['codec_device_id'],
        'encode_kwargs': livogs_cfg['encode_kwargs'],
        'save_dir': save_dir,
        'save_interval': save_interval,
    }

    # -- warmup gen (main process) --
    logging.info(f"Warming up generation ({min(warmup, total_frames)} frames) …")
    for idx in range(min(warmup, total_frames)):
        torch.cuda.set_device(gen_device)
        data = renderer._get_gpu_frame(idx)
        _generate_one_frame(renderer, data, use_fp16)
    torch.cuda.synchronize()
    logging.info("Gen warmup done.")

    # -- start codec child process --
    codec_proc = ctx.Process(
        target=_codec_worker, args=(data_q, result_q, codec_cfg), daemon=True)
    codec_proc.start()

    # -- warmup codec: send a few frames through the codec process --
    logging.info("Warming up codec process …")
    for idx in range(min(warmup, total_frames)):
        torch.cuda.set_device(gen_device)
        data = renderer._get_gpu_frame(idx)
        params = _generate_one_frame(renderer, data, use_fp16)
        torch.cuda.synchronize(gen_device)
        params_cpu = {k: v.cpu() for k, v in params.items()}
        data_q.put(('__warmup__', params_cpu, time.perf_counter()))
    for _ in range(min(warmup, total_frames)):
        result_q.get()
    logging.info("Codec warmup done.")

    # -- background copy thread (same process, uses copy_stream) --
    copy_q = queue.Queue(maxsize=2)
    copy_stream = torch.cuda.Stream(device=gen_device)

    def _copy_worker():
        while True:
            item = copy_q.get()
            if item is None:
                data_q.put(None)
                break
            frame_name, params_gpu, t_enter = item
            with torch.cuda.stream(copy_stream):
                params_cpu = {k: v.cpu() for k, v in params_gpu.items()}
            copy_stream.synchronize()
            data_q.put((frame_name, params_cpu, t_enter))

    copy_thread = threading.Thread(target=_copy_worker, daemon=True,
                                   name='CopyWorker')
    copy_thread.start()

    # -- generation loop --
    gen_times = []
    torch.cuda.synchronize()
    wall_start = time.perf_counter()

    for idx in range(total_frames):
        torch.cuda.set_device(gen_device)
        data = renderer._get_gpu_frame(idx)
        frame_name = data['name']

        t_enter = time.perf_counter()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

        start_ev.record()
        params = _generate_one_frame(renderer, data, use_fp16)
        end_ev.record()
        end_ev.synchronize()

        gen_ms = start_ev.elapsed_time(end_ev)
        gen_times.append((frame_name, gen_ms))

        copy_q.put((frame_name, params, t_enter))

    copy_q.put(None)
    copy_thread.join()

    # -- collect results from codec process --
    csv_rows = []
    while True:
        item = result_q.get()
        if item is None:
            break
        csv_rows.append(item)

    wall_end = time.perf_counter()
    codec_proc.join(timeout=30)

    # -- merge gen timings --
    gen_map = {name: ms for name, ms in gen_times}
    for row in csv_rows:
        fn = row['frame_name']
        row['gen_ms'] = gen_map.get(fn, 0.0)
        row['e2e_sum_ms'] = (row['gen_ms'] + row['upload_ms']
                             + row['enc_ms'] + row['dec_ms'])

    elapsed = wall_end - wall_start
    pipeline_fps = total_frames / elapsed if elapsed > 0 else 0.0

    print(f"\n{'=' * 70}")
    print("Multiprocessing Pipelined Mode Results")
    print(f"{'=' * 70}")
    print(f"  gen_device={gen_device}  codec_device={codec_device}")
    print(f"  Pipeline: multiprocessing (gen process + codec process)")
    print(f"  Total frames:        {total_frames}")
    print(f"  Wall-clock time:     {elapsed:.3f} s")
    print(f"  Pipeline throughput: {pipeline_fps:.1f} FPS "
          f"({elapsed * 1000 / max(total_frames, 1):.2f} ms/frame)")

    _print_summary_mp(csv_rows, warmup)
    _write_csv(csv_rows, os.path.join(output_dir, 'benchmark_pipelined_mp.csv'))
    return csv_rows


def _print_summary_mp(rows, skip_n):
    valid = rows[skip_n:] if len(rows) > skip_n else rows
    if not valid:
        return

    print(f"\n  Per-stage summary (skipping first {skip_n}):")
    print(f"  {'Stage':<15} {'Mean ms':>9} {'Std ms':>9} {'FPS':>9}")
    print(f"  {'-' * 45}")
    for key, label in [('gen_ms', 'generation'), ('upload_ms', 'upload'),
                       ('enc_ms', 'encode'), ('dec_ms', 'decode'),
                       ('e2e_sum_ms', 'e2e (sum)'),
                       ('e2e_wall_ms', 'e2e (wall)')]:
        vals = [r[key] for r in valid]
        m, s = np.mean(vals), np.std(vals)
        fps = 1000.0 / m if m > 0 else float('inf')
        print(f"  {label:<15} {m:>9.2f} {s:>9.2f} {fps:>9.1f}")

    avg_g = np.mean([r['num_gaussians'] for r in valid])
    avg_v = np.mean([r['num_voxels'] for r in valid])
    avg_c = np.mean([r['compressed_bytes'] for r in valid])
    avg_u = avg_g * (3 + 4 + 3 + 1 + 3) * 4
    print(f"\n  Avg Gaussians: {avg_g:.0f} -> {avg_v:.0f} voxels")
    print(f"  Avg compressed: {avg_c / 1024:.1f} KB "
          f"(ratio: {avg_u / max(avg_c, 1):.1f}x)")
    print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _write_csv(rows, path):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.2f}' if isinstance(v, float) else v)
                        for k, v in r.items()})
    logging.info(f"CSV saved to {path}")


def _print_summary(rows, skip_n, mode_label):
    valid = rows[skip_n:] if len(rows) > skip_n else rows
    if not valid:
        return
    n = len(valid)

    print(f"\n  {mode_label} per-stage summary (skipping first {skip_n}):")
    print(f"  {'Stage':<15} {'Mean ms':>9} {'Std ms':>9} {'FPS':>9}")
    print(f"  {'-' * 45}")
    stages = [('gen_ms', 'generation'), ('enc_ms', 'encode'),
              ('dec_ms', 'decode'), ('e2e_ms', 'e2e')]
    if 'xfer_ms' in valid[0] and any(r['xfer_ms'] > 0 for r in valid):
        stages.insert(1, ('xfer_ms', 'transfer'))
    for key, label in stages:
        vals = [r[key] for r in valid]
        m, s = np.mean(vals), np.std(vals)
        fps = 1000.0 / m if m > 0 else float('inf')
        print(f"  {label:<15} {m:>9.2f} {s:>9.2f} {fps:>9.1f}")

    avg_g = np.mean([r['num_gaussians'] for r in valid])
    avg_v = np.mean([r['num_voxels'] for r in valid])
    avg_c = np.mean([r['compressed_bytes'] for r in valid])
    avg_u = avg_g * (3 + 4 + 3 + 1 + 3) * 4  # 14 floats * 4 bytes
    print(f"\n  Avg Gaussians: {avg_g:.0f} -> {avg_v:.0f} voxels")
    print(f"  Avg compressed: {avg_c / 1024:.1f} KB "
          f"(ratio: {avg_u / max(avg_c, 1):.1f}x)")
    print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Live GPS-Gaussian + LiVoGS Streaming Pipeline")

    g = p.add_argument_group("GPS-Gaussian")
    g.add_argument('--test_data_root', type=str, required=True)
    g.add_argument('--ckpt_path', type=str, required=True)
    g.add_argument('--src_view', type=int, nargs='+', required=True)
    g.add_argument('--ratio', type=float, default=0.5)
    g.add_argument('--mode', type=str, default='compiled',
                   choices=['baseline', 'compiled', 'tensorrt', 'hybrid'])
    g.add_argument('--trt_dir', type=str, default='./trt_engines')

    g = p.add_argument_group("LiVoGS")
    g.add_argument('--J', type=int, default=15)
    g.add_argument('--quantize_step', type=float, default=0.0001)
    g.add_argument('--sh_color_space', type=str, default='yuv',
                   choices=['rgb', 'yuv', 'klt'])
    g.add_argument('--rlgr_block_size', type=int, default=4096)
    g.add_argument('--nvcomp_algorithm', type=str, default='ANS',
                   choices=['None', 'LZ4', 'Snappy', 'GDeflate', 'Deflate',
                            'zStandard', 'Cascaded', 'Bitcomp', 'ANS'])

    g = p.add_argument_group("Pipeline")
    g.add_argument('--mode_pipeline', type=str, default='sequential',
                   choices=['sequential', 'pipelined', 'pipelined_mp'])
    g.add_argument('--gen_device', type=str, default='cuda:0')
    g.add_argument('--codec_device', type=str, default=None,
                   help='LiVoGS device (default: same as --gen_device)')
    g.add_argument('--save_interval', type=int, default=1,
                   help='Save every Nth decoded frame (0 = no saving)')
    g.add_argument('--output_dir', type=str, default='./streaming_out')
    g.add_argument('--warmup', type=int, default=5)

    return p.parse_args()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    args = _parse_args()

    gen_device = args.gen_device
    codec_device = args.codec_device if args.codec_device else gen_device
    gen_device_id = int(gen_device.split(':')[1]) if ':' in gen_device else 0
    codec_device_id = int(codec_device.split(':')[1]) if ':' in codec_device else 0

    cfg = config()
    cfg.load(os.path.join(_PROJECT_ROOT, 'config', 'stage2.yaml'))
    cfg = cfg.get_cfg()
    cfg.defrost()
    cfg.batch_size = 1
    cfg.dataset.test_data_root = args.test_data_root
    cfg.dataset.use_processed_data = False
    cfg.restore_ckpt = args.ckpt_path
    cfg.test_out_path = os.path.join(args.output_dir, 'renders')
    cfg.freeze()

    os.makedirs(args.output_dir, exist_ok=True)
    Path(cfg.test_out_path).mkdir(exist_ok=True, parents=True)

    torch.cuda.set_device(gen_device)
    renderer = StereoHumanRenderFast(cfg, phase='test', mode=args.mode,
                                     trt_dir=args.trt_dir)
    total_frames = renderer.preload_all_frames(args.src_view, ratio=args.ratio)

    nvcomp_alg = None if args.nvcomp_algorithm == 'None' else args.nvcomp_algorithm
    qs = args.quantize_step
    livogs_cfg = {
        'gen_device': gen_device,
        'codec_device': codec_device,
        'codec_device_id': codec_device_id,
        'encode_kwargs': {
            'J': args.J,
            'sh_color_space': args.sh_color_space,
            'quantize_step': {
                'quats': qs, 'scales': qs, 'opacity': qs,
                'sh_dc': qs, 'sh_rest': [],
            },
            'rlgr_block_size': args.rlgr_block_size,
            'nvcomp_algorithm': nvcomp_alg,
        },
    }

    print("=" * 70)
    print("Live GPS-Gaussian + LiVoGS Streaming Pipeline")
    print("=" * 70)
    print(f"  Pipeline mode:     {args.mode_pipeline}")
    print(f"  GPS-Gaussian mode: {args.mode}")
    print(f"  Gen device:        {gen_device}")
    print(f"  Codec device:      {codec_device}")
    print(f"  Total frames:      {total_frames}")
    print(f"  Save interval:     {args.save_interval}")
    print(f"  LiVoGS J={args.J}  qs={qs}  "
          f"color_space={args.sh_color_space}  nvcomp={nvcomp_alg}")
    print("=" * 70)

    if args.mode_pipeline == 'sequential':
        run_sequential(renderer, livogs_cfg, args.output_dir,
                       save_interval=args.save_interval,
                       warmup=args.warmup)
    elif args.mode_pipeline == 'pipelined':
        run_pipelined(renderer, livogs_cfg, args.output_dir,
                      save_interval=args.save_interval,
                      warmup=args.warmup)
    elif args.mode_pipeline == 'pipelined_mp':
        run_pipelined_mp(renderer, livogs_cfg, args.output_dir,
                         save_interval=args.save_interval,
                         warmup=args.warmup)

    print("Done.")
