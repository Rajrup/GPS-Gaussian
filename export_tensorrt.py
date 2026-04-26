"""Export GPS-Gaussian sub-networks to ONNX and build TensorRT engines.

Focuses on the GSRegresser (gs_parm_regresser) since it's the dominant bottleneck
(~67% of inference time). The img_encoder is also exported since it's straightforward.

Usage:
    python export_tensorrt.py \
        --ckpt_path ./models/GPS-GS_stage2_final.pth \
        --output_dir ./trt_engines
"""
import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import onnx

from lib.network import RtStereoHumanModel
from config.stereo_human_config import ConfigStereoHuman as config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')


class GSRegresserWrapper(nn.Module):
    """Wraps GSRegresser to accept flat tensor inputs (no tuples) for ONNX export."""

    def __init__(self, gs_regresser):
        super().__init__()
        self.gs = gs_regresser

    def forward(self, img, depth, feat1, feat2, feat3):
        img_feat = (feat1, feat2, feat3)
        rot, scale, opacity = self.gs(img, depth, img_feat)
        return rot, scale, opacity


class ImgEncoderWrapper(nn.Module):
    """Wraps UnetExtractor to return 3 tensors instead of a tuple."""

    def __init__(self, encoder):
        super().__init__()
        self.enc = encoder

    def forward(self, x):
        f1, f2, f3 = self.enc(x)
        return f1, f2, f3


def export_gs_regresser_onnx(model, output_path, src_res=1024):
    logging.info("Exporting GSRegresser to ONNX...")
    wrapper = GSRegresserWrapper(model.gs_parm_regresser).cuda().eval()

    B = 2
    dummy_img = torch.randn(B, 3, src_res, src_res, device='cuda', dtype=torch.float16)
    dummy_depth = torch.randn(B, 1, src_res, src_res, device='cuda', dtype=torch.float16)
    dummy_f1 = torch.randn(B, 32, src_res // 2, src_res // 2, device='cuda', dtype=torch.float16)
    dummy_f2 = torch.randn(B, 48, src_res // 4, src_res // 4, device='cuda', dtype=torch.float16)
    dummy_f3 = torch.randn(B, 96, src_res // 8, src_res // 8, device='cuda', dtype=torch.float16)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        torch.onnx.export(
            wrapper,
            (dummy_img, dummy_depth, dummy_f1, dummy_f2, dummy_f3),
            output_path,
            input_names=['img', 'depth', 'feat1', 'feat2', 'feat3'],
            output_names=['rot', 'scale', 'opacity'],
            opset_version=17,
            do_constant_folding=True,
        )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logging.info(f"GSRegresser ONNX saved to {output_path}")
    return output_path


def export_img_encoder_onnx(model, output_path, src_res=1024):
    logging.info("Exporting ImgEncoder to ONNX...")
    wrapper = ImgEncoderWrapper(model.img_encoder).cuda().eval()

    B = 2
    dummy_img = torch.randn(B, 3, src_res, src_res, device='cuda', dtype=torch.float16)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        torch.onnx.export(
            wrapper,
            (dummy_img,),
            output_path,
            input_names=['image'],
            output_names=['feat1', 'feat2', 'feat3'],
            opset_version=17,
            do_constant_folding=True,
        )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logging.info(f"ImgEncoder ONNX saved to {output_path}")
    return output_path


def build_trt_engine(onnx_path, engine_path, fp16=True, workspace_gb=4):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    logging.info(f"Parsing ONNX: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logging.error(f"  ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parsing failed")

    config_trt = builder.create_builder_config()
    config_trt.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    if fp16:
        config_trt.set_flag(trt.BuilderFlag.FP16)
        logging.info("  FP16 mode enabled")

    logging.info(f"Building TensorRT engine (this may take several minutes)...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config_trt)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    elapsed = time.time() - t0
    logging.info(f"Engine built in {elapsed:.1f}s")

    with open(engine_path, 'wb') as f:
        f.write(serialized)
    logging.info(f"TensorRT engine saved to {engine_path} ({os.path.getsize(engine_path) / 1e6:.1f} MB)")
    return engine_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./trt_engines')
    parser.add_argument('--workspace_gb', type=int, default=4)
    parser.add_argument('--skip_img_encoder', action='store_true')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    cfg = config()
    cfg.load('./config/stage2.yaml')
    cfg = cfg.get_cfg()
    cfg.defrost()
    cfg.batch_size = 1
    cfg.restore_ckpt = args.ckpt_path
    cfg.freeze()

    model = RtStereoHumanModel(cfg, with_gs_render=True).cuda()
    ckpt = torch.load(args.ckpt_path, map_location='cuda', weights_only=False)
    model.load_state_dict(ckpt['network'], strict=True)
    model.eval()
    logging.info("Model loaded")

    gs_onnx = os.path.join(args.output_dir, 'gs_regresser.onnx')
    export_gs_regresser_onnx(model, gs_onnx, src_res=cfg.dataset.src_res)
    gs_engine = os.path.join(args.output_dir, 'gs_regresser_fp16.engine')
    build_trt_engine(gs_onnx, gs_engine, fp16=True, workspace_gb=args.workspace_gb)

    if not args.skip_img_encoder:
        enc_onnx = os.path.join(args.output_dir, 'img_encoder.onnx')
        export_img_encoder_onnx(model, enc_onnx, src_res=cfg.dataset.src_res)
        enc_engine = os.path.join(args.output_dir, 'img_encoder_fp16.engine')
        build_trt_engine(enc_onnx, enc_engine, fp16=True, workspace_gb=args.workspace_gb)

    logging.info("All exports complete!")
