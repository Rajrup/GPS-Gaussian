"""
Optimized GPS-Gaussian inference script with benchmarking.

Supports multiple optimization modes:
  - baseline:  Original PyTorch inference (for comparison)
  - compiled:  torch.compile + full FP16 + pre-loaded data
  - tensorrt:  TensorRT engines (requires Phase 2 setup)

Usage:
  python test_real_data_fast.py \
      --test_data_root '/path/to/real_data' \
      --ckpt_path './models/GPS-GS_stage2_final.pth' \
      --src_view 0 1 \
      --mode compiled
"""
from __future__ import print_function, division

import argparse
import logging
import numpy as np
import cv2
import os
import time
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from lib.human_loader import StereoHumanDataset
from lib.network import RtStereoHumanModel
from config.stereo_human_config import ConfigStereoHuman as config
from lib.utils import get_novel_calib
from lib.GaussianRender import pts2render

import torch
import torch.backends.cudnn
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class TRTEngine:
    """TensorRT engine wrapper with pre-allocated GPU buffers."""

    def __init__(self, engine_path):
        import tensorrt as trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream().cuda_stream
        self._setup_bindings()

    def _setup_bindings(self):
        import tensorrt as trt
        self.input_names = []
        self.output_names = []
        self.output_shapes = {}
        self.output_dtypes = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = torch.float16 if dtype == trt.float16 else torch.float32

            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
                self.output_shapes[name] = tuple(shape)
                self.output_dtypes[name] = torch_dtype

        self.outputs = {}
        for name in self.output_names:
            self.outputs[name] = torch.empty(
                self.output_shapes[name],
                dtype=self.output_dtypes[name],
                device='cuda'
            )

    def __call__(self, **inputs):
        for name in self.input_names:
            tensor = inputs[name].contiguous()
            self.context.set_tensor_address(name, tensor.data_ptr())
        for name in self.output_names:
            self.context.set_tensor_address(name, self.outputs[name].data_ptr())

        self.context.execute_async_v3(self.stream)
        return tuple(self.outputs[name] for name in self.output_names)


class CUDATimer:
    """Accurate GPU timing using CUDA events."""

    def __init__(self):
        self.timings = defaultdict(list)
        self._start_events = {}
        self._end_events = {}

    def start(self, name):
        if name not in self._start_events:
            self._start_events[name] = torch.cuda.Event(enable_timing=True)
            self._end_events[name] = torch.cuda.Event(enable_timing=True)
        self._start_events[name].record()

    def stop(self, name):
        self._end_events[name].record()

    def sync_and_collect(self):
        torch.cuda.synchronize()
        for name in self._start_events:
            if name in self._end_events:
                elapsed = self._start_events[name].elapsed_time(self._end_events[name])
                self.timings[name].append(elapsed)

    def report(self, skip_first_n=5):
        print("\n" + "=" * 70)
        print(f"{'Component':<30} {'Mean (ms)':>10} {'Std (ms)':>10} {'FPS':>10}")
        print("=" * 70)
        for name, times in self.timings.items():
            t = times[skip_first_n:] if len(times) > skip_first_n else times
            if not t:
                continue
            mean_ms = np.mean(t)
            std_ms = np.std(t)
            fps = 1000.0 / mean_ms if mean_ms > 0 else float('inf')
            print(f"{name:<30} {mean_ms:>10.2f} {std_ms:>10.2f} {fps:>10.1f}")
        print("=" * 70)

        total_key = 'total'
        if total_key in self.timings:
            t = self.timings[total_key][skip_first_n:]
            if t:
                mean_ms = np.mean(t)
                print(f"\nOverall: {mean_ms:.2f} ms/frame = {1000.0/mean_ms:.1f} FPS")
        print()


class StereoHumanRenderFast:
    def __init__(self, cfg_file, phase, mode='baseline', trt_dir='./trt_engines'):
        self.cfg = cfg_file
        self.bs = self.cfg.batch_size
        self.mode = mode
        self.trt_dir = trt_dir

        self.model = RtStereoHumanModel(self.cfg, with_gs_render=True)
        self.dataset = StereoHumanDataset(self.cfg.dataset, phase=phase)
        self.model.cuda()
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        self.model.eval()

        self.trt_gs = None
        self.trt_enc = None

        if mode == 'compiled':
            self._apply_compile_optimizations()
        elif mode == 'tensorrt':
            self._load_trt_engines()
        elif mode == 'hybrid':
            self._load_trt_engines()
            self._compile_raft_stereo()

    def _load_trt_engines(self):
        gs_path = os.path.join(self.trt_dir, 'gs_regresser_fp16.engine')
        enc_path = os.path.join(self.trt_dir, 'img_encoder_fp16.engine')

        if os.path.exists(gs_path):
            logging.info(f"Loading TRT engine: {gs_path}")
            self.trt_gs = TRTEngine(gs_path)
        else:
            logging.warning(f"TRT engine not found: {gs_path}, falling back to PyTorch")

        if os.path.exists(enc_path):
            logging.info(f"Loading TRT engine: {enc_path}")
            self.trt_enc = TRTEngine(enc_path)
        else:
            logging.warning(f"TRT engine not found: {enc_path}, falling back to PyTorch")

    def _compile_raft_stereo(self):
        try:
            self.model.raft_stereo = torch.compile(
                self.model.raft_stereo,
                mode="max-autotune",
                fullgraph=False,
            )
            logging.info("  raft_stereo compiled for hybrid mode")
        except Exception as e:
            logging.warning(f"  raft_stereo compile failed: {e}")

    def _apply_compile_optimizations(self):
        logging.info("Applying torch.compile optimizations...")
        try:
            self.model.img_encoder = torch.compile(
                self.model.img_encoder,
                mode="max-autotune",
                fullgraph=False,
            )
            logging.info("  img_encoder compiled")
        except Exception as e:
            logging.warning(f"  img_encoder compile failed: {e}")

        try:
            self.model.gs_parm_regresser = torch.compile(
                self.model.gs_parm_regresser,
                mode="max-autotune",
                fullgraph=False,
            )
            logging.info("  gs_parm_regresser compiled")
        except Exception as e:
            logging.warning(f"  gs_parm_regresser compile failed: {e}")

        try:
            self.model.raft_stereo = torch.compile(
                self.model.raft_stereo,
                mode="max-autotune",
                fullgraph=False,
            )
            logging.info("  raft_stereo compiled")
        except Exception as e:
            logging.warning(f"  raft_stereo compile failed: {e}")

    def preload_all_frames(self, view_select, ratio=0.5):
        """Pre-compute stereo rectification and novel camera calibration.

        All data stays on CPU pinned memory for fast async GPU transfers.
        This eliminates per-frame disk I/O and stereo rectification cost.
        """
        total_frames = len(os.listdir(os.path.join(self.cfg.dataset.test_data_root, 'img')))
        logging.info(f"Pre-computing {total_frames} frames (pinned CPU)...")

        self.preloaded_data = []
        for idx in tqdm(range(total_frames), desc="Pre-loading"):
            item = self.dataset.get_test_item(idx, source_id=view_select)
            for view in ['lmain', 'rmain']:
                for key in item[view].keys():
                    item[view][key] = item[view][key].unsqueeze(0)
            for key in item['novel_view'].keys():
                if torch.is_tensor(item['novel_view'][key]):
                    item['novel_view'][key] = item['novel_view'][key].unsqueeze(0)

            gpu_item = self._to_gpu(item)
            gpu_item = get_novel_calib(gpu_item, self.cfg.dataset, ratio=ratio,
                                       intr_key='intr_ori', extr_key='extr_ori')
            nv_cpu = {}
            for key, val in gpu_item['novel_view'].items():
                if torch.is_tensor(val):
                    nv_cpu[key] = val.cpu().pin_memory()
                else:
                    nv_cpu[key] = val
            item['novel_view'] = nv_cpu
            del gpu_item

            for view in ['lmain', 'rmain']:
                for key in item[view].keys():
                    if torch.is_tensor(item[view][key]):
                        item[view][key] = item[view][key].pin_memory()

            self.preloaded_data.append(item)

        logging.info(f"Pre-computed {len(self.preloaded_data)} frames")
        return total_frames

    def _to_gpu(self, item):
        """Move a preloaded CPU item to GPU."""
        data = {'name': item['name'], 'novel_view': {}}
        for view in ['lmain', 'rmain']:
            data[view] = {}
            for key, val in item[view].items():
                data[view][key] = val.cuda(non_blocking=True) if torch.is_tensor(val) else val
        for key, val in item['novel_view'].items():
            data['novel_view'][key] = val.cuda(non_blocking=True) if torch.is_tensor(val) else val
        return data

    def _get_gpu_frame(self, idx):
        """Get a single frame on GPU with novel view calibration."""
        return self._to_gpu(self.preloaded_data[idx])

    def infer_benchmark(self, view_select, ratio=0.5, warmup=10, save_images=True):
        """Run inference with detailed per-component benchmarking."""
        total_frames = self.preload_all_frames(view_select, ratio)
        timer = CUDATimer()

        use_fp16 = self.mode in ('compiled', 'tensorrt', 'hybrid')
        logging.info(f"Running benchmark: mode={self.mode}, fp16={use_fp16}, "
                     f"frames={total_frames}, warmup={warmup}")

        for idx in tqdm(range(total_frames), desc=f"Inference ({self.mode})"):
            data = self._get_gpu_frame(idx)

            timer.start('total')

            with torch.no_grad():
                if use_fp16:
                    with torch.autocast("cuda", dtype=torch.float16):
                        self._forward_timed(data, timer)
                else:
                    self._forward_timed(data, timer)

                if use_fp16:
                    self._cast_to_fp32(data)
                timer.start('render')
                data = pts2render(data, bg_color=self.cfg.dataset.bg_color)
                timer.stop('render')

            timer.stop('total')
            timer.sync_and_collect()

            if save_images:
                render_novel = self.tensor2np(data['novel_view']['img_pred'])
                cv2.imwrite(self.cfg.test_out_path + '/%s_novel.jpg' % (data['name']),
                            render_novel)

        timer.report(skip_first_n=warmup)

    @staticmethod
    def _cast_to_fp32(data):
        """Cast model outputs to FP32 for the rasterizer (no FP16 support)."""
        for view in ['lmain', 'rmain']:
            for key in ('xyz', 'rot_maps', 'scale_maps', 'opacity_maps', 'img'):
                if key in data[view] and data[view][key].dtype == torch.float16:
                    data[view][key] = data[view][key].float()

    def _forward_timed(self, data, timer):
        """Model forward pass with per-component timing."""
        if self.mode in ('compiled', 'hybrid'):
            torch.compiler.cudagraph_mark_step_begin()

        bs = data['lmain']['img'].shape[0]
        image = torch.cat([data['lmain']['img'], data['rmain']['img']], dim=0)

        timer.start('img_encoder')
        if self.trt_enc is not None:
            feat1, feat2, feat3 = self.trt_enc(image=image.half())
            img_feat = (feat1, feat2, feat3)
        else:
            img_feat = self.model.img_encoder(image)
        timer.stop('img_encoder')

        timer.start('raft_stereo')
        flow_up = self.model.raft_stereo(img_feat[2], iters=self.model.val_iters, test_mode=True)
        timer.stop('raft_stereo')

        data['lmain']['flow_pred'] = flow_up[0]
        data['rmain']['flow_pred'] = flow_up[1]

        timer.start('gs_parm_regress')
        if self.trt_gs is not None:
            self._flow2gsparms_trt(image, img_feat, data, bs)
        else:
            data = self.model.flow2gsparms(image, img_feat, data, bs)
        timer.stop('gs_parm_regress')

    def _run_forward(self, data, use_fp16):
        """Dispatch forward pass based on mode."""
        if self.mode in ('compiled', 'hybrid'):
            torch.compiler.cudagraph_mark_step_begin()

        uses_trt = self.mode in ('tensorrt', 'hybrid')
        if uses_trt:
            if use_fp16:
                with torch.autocast("cuda", dtype=torch.float16):
                    self._forward_notimed(data)
            else:
                self._forward_notimed(data)
        else:
            if use_fp16:
                with torch.autocast("cuda", dtype=torch.float16):
                    data, _, _ = self.model(data, is_train=False)
            else:
                data, _, _ = self.model(data, is_train=False)

    def _forward_notimed(self, data):
        """Forward pass without timing (for fast inference loop)."""
        bs = data['lmain']['img'].shape[0]
        image = torch.cat([data['lmain']['img'], data['rmain']['img']], dim=0)

        if self.trt_enc is not None:
            feat1, feat2, feat3 = self.trt_enc(image=image.half())
            img_feat = (feat1, feat2, feat3)
        else:
            img_feat = self.model.img_encoder(image)

        flow_up = self.model.raft_stereo(img_feat[2], iters=self.model.val_iters, test_mode=True)
        data['lmain']['flow_pred'] = flow_up[0]
        data['rmain']['flow_pred'] = flow_up[1]

        if self.trt_gs is not None:
            self._flow2gsparms_trt(image, img_feat, data, bs)
        else:
            data = self.model.flow2gsparms(image, img_feat, data, bs)

    def _flow2gsparms_trt(self, lr_img, lr_img_feat, data, bs):
        """flow2gsparms using TensorRT for the GSRegresser."""
        from lib.utils import flow2depth, depth2pc

        for view in ['lmain', 'rmain']:
            data[view]['depth'] = flow2depth(data[view])
            data[view]['xyz'] = depth2pc(
                data[view]['depth'], data[view]['extr'], data[view]['intr']
            ).view(bs, -1, 3)
            valid = data[view]['depth'] != 0.0
            data[view]['pts_valid'] = valid.view(bs, -1)

        lr_depth = torch.concat([data['lmain']['depth'], data['rmain']['depth']], dim=0)
        rot_maps, scale_maps, opacity_maps = self.trt_gs(
            img=lr_img.half(),
            depth=lr_depth.half(),
            feat1=lr_img_feat[0].half(),
            feat2=lr_img_feat[1].half(),
            feat3=lr_img_feat[2].half(),
        )

        data['lmain']['rot_maps'], data['rmain']['rot_maps'] = torch.split(rot_maps, [bs, bs])
        data['lmain']['scale_maps'], data['rmain']['scale_maps'] = torch.split(scale_maps, [bs, bs])
        data['lmain']['opacity_maps'], data['rmain']['opacity_maps'] = torch.split(opacity_maps, [bs, bs])

    def _to_gpu_on_stream(self, item, stream):
        """Transfer a preloaded CPU item to GPU on a specific stream."""
        data = {'name': item['name'], 'novel_view': {}}
        with torch.cuda.stream(stream):
            for view in ['lmain', 'rmain']:
                data[view] = {}
                for key, val in item[view].items():
                    data[view][key] = val.cuda(non_blocking=True) if torch.is_tensor(val) else val
            for key, val in item['novel_view'].items():
                data['novel_view'][key] = val.cuda(non_blocking=True) if torch.is_tensor(val) else val
        return data

    def infer_sequence_fast(self, view_select, ratio=0.5, save_images=True):
        """Optimized inference with pipelined CPU->GPU transfer."""
        total_frames = self.preload_all_frames(view_select, ratio)
        use_fp16 = self.mode in ('compiled', 'tensorrt', 'hybrid')
        copy_stream = torch.cuda.Stream()

        warmup = min(10, total_frames)
        logging.info(f"Warming up {warmup} frames...")
        for idx in range(warmup):
            data = self._get_gpu_frame(idx)
            with torch.no_grad():
                self._run_forward(data, use_fp16)
                if use_fp16:
                    self._cast_to_fp32(data)
                data = pts2render(data, bg_color=self.cfg.dataset.bg_color)

        torch.cuda.synchronize()

        next_data = self._to_gpu_on_stream(self.preloaded_data[0], copy_stream)

        start_time = time.perf_counter()

        for idx in tqdm(range(total_frames), desc=f"Fast inference ({self.mode})"):
            torch.cuda.current_stream().wait_stream(copy_stream)
            data = next_data

            if idx + 1 < total_frames:
                next_data = self._to_gpu_on_stream(self.preloaded_data[idx + 1], copy_stream)

            with torch.no_grad():
                self._run_forward(data, use_fp16)
                if use_fp16:
                    self._cast_to_fp32(data)
                data = pts2render(data, bg_color=self.cfg.dataset.bg_color)

            if save_images:
                render_novel = self.tensor2np(data['novel_view']['img_pred'])
                cv2.imwrite(self.cfg.test_out_path + '/%s_novel.jpg' % (data['name']),
                            render_novel)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        fps = total_frames / elapsed
        ms_per_frame = elapsed * 1000.0 / total_frames

        print(f"\n{'=' * 50}")
        print(f"Mode: {self.mode}")
        print(f"Total frames: {total_frames}")
        print(f"Total time: {elapsed:.3f}s")
        print(f"Per frame: {ms_per_frame:.2f} ms")
        print(f"FPS: {fps:.1f}")
        print(f"{'=' * 50}\n")

    def tensor2np(self, img_tensor):
        img_np = img_tensor.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
        img_np = img_np * 255
        img_np = img_np[:, :, ::-1].astype(np.uint8)
        return img_np

    def fetch_data(self, data):
        for view in ['lmain', 'rmain']:
            for item in data[view].keys():
                data[view][item] = data[view][item].cuda().unsqueeze(0)
        return data

    def load_ckpt(self, load_path):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        self.model.load_state_dict(ckpt['network'], strict=True)
        logging.info(f"Parameter loading done")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_root', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--src_view', type=int, nargs='+', required=True)
    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--mode', type=str, default='compiled',
                        choices=['baseline', 'compiled', 'tensorrt', 'hybrid'],
                        help='Optimization mode (hybrid = TRT engines + compiled raft)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run per-component benchmark (slower due to sync)')
    parser.add_argument('--no_save', action='store_true',
                        help='Skip saving images (pure speed test)')
    parser.add_argument('--trt_dir', type=str, default='./trt_engines',
                        help='Directory containing TensorRT engines')
    arg = parser.parse_args()

    cfg = config()
    cfg_for_train = os.path.join('./config', 'stage2.yaml')
    cfg.load(cfg_for_train)
    cfg = cfg.get_cfg()

    cfg.defrost()
    cfg.batch_size = 1
    cfg.dataset.test_data_root = arg.test_data_root
    cfg.dataset.use_processed_data = False
    cfg.restore_ckpt = arg.ckpt_path
    cfg.test_out_path = f'./fast_out_{arg.mode}'
    Path(cfg.test_out_path).mkdir(exist_ok=True, parents=True)
    cfg.freeze()

    render = StereoHumanRenderFast(cfg, phase='test', mode=arg.mode, trt_dir=arg.trt_dir)

    if arg.benchmark:
        render.infer_benchmark(view_select=arg.src_view, ratio=arg.ratio,
                               save_images=not arg.no_save)
    else:
        render.infer_sequence_fast(view_select=arg.src_view, ratio=arg.ratio,
                                   save_images=not arg.no_save)
