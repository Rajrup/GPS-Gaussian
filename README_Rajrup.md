## Setup

```bash
conda env create --file environment.yml
conda activate gps_gaussian
git clone --recursive https://github.com/Rajrup/GPS-Gaussian
git submodule update --init --recursive

git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting/

LD_LIBRARY_PATH=/home/rajrup/miniconda3/envs/gps_gaussian/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH \
pip install -e submodules/diff-gaussian-rasterization --no-build-isolation

git clone https://github.com/princeton-vl/RAFT-Stereo.git
cd RAFT-Stereo/sampler
LD_LIBRARY_PATH=/home/rajrup/miniconda3/envs/gps_gaussian/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH \
python setup.py install
cd ../..
```

### Testing

- Real-world data: download the test data ```real_data``` from [Baidu Netdisk](https://pan.baidu.com/s/1sX9m8wRDSQAI9d78wST7mw?pwd=rax4) or [OneDrive](https://hiteducn0-my.sharepoint.com/:f:/g/personal/sawyer0503_hit_edu_cn/EkE2GFd2saBCh_XkY3TsoV0BVTmK1UiTTKJDYje3U3vdkw?e=YazWdd). Then, run the following code for synthesizing a fixed novel view between ```src_view``` 0 and 1, the position of novel viewpoint between source views is adjusted with a ```ratio``` ranging from 0 to 1.

```bash
python test_real_data.py \
--test_data_root 'PATH/TO/REAL_DATA' \
--ckpt_path 'PATH/TO/GPS-GS_stage2_final.pth' \
--src_view 0 1 \
--ratio=0.5

python test_real_data.py \
--test_data_root '/synology/rajrup/GPS-Gaussian/real_data' \
--ckpt_path './models/GPS-GS_stage2_final.pth' \
--src_view 0 1 \
--ratio=0.5
```

- Freeview rendering: run the following code to interpolate freeview between source views, and modify the ```novel_view_nums``` to set a specific number of novel viewpoints.

```bash
python test_view_interp.py \
--test_data_root 'PATH/TO/RENDER_DATA/val' \
--ckpt_path 'PATH/TO/GPS-GS_stage2_final.pth' \
--novel_view_nums 5

python test_view_interp.py \
--test_data_root '/synology/rajrup/GPS-Gaussian/render_data/val' \
--ckpt_path './models/GPS-GS_stage2_final.pth' \
--novel_view_nums 5
```

### Pipeline

**Convert GPS-Gaussian model to TRT engine:**

```bash
LD_LIBRARY_PATH=/home/rajrup/miniconda3/envs/gps_gaussian/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH \
python export_tensorrt.py \
  --ckpt_path ./models/GPS-GS_stage2_final.pth \
  --output_dir ./trt_engines
```

**Multiprocessing pipeline (dual-GPU, ~21.8 FPS without save, ~18.3 FPS with save):**

Runs GPS-Gaussian generation on `cuda:0` and LiVoGS encode/decode on `cuda:1` in separate processes to avoid GIL contention.

```bash
# With decoded frame saving (for evaluation)
LD_LIBRARY_PATH=/home/rajrup/miniconda3/envs/gps_gaussian/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH \
python scripts/streaming_pipeline.py \
  --test_data_root '/synology/rajrup/GPS-Gaussian/real_data' \
  --ckpt_path './models/GPS-GS_stage2_final.pth' \
  --src_view 0 1 \
  --mode compiled \
  --mode_pipeline pipelined_mp \
  --gen_device cuda:0 \
  --codec_device cuda:1 \
  --output_dir ./gps_livogs_streaming_out \
  --warmup 5 \
  --save_interval 1

# Without saving (max throughput)
LD_LIBRARY_PATH=/home/rajrup/miniconda3/envs/gps_gaussian/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH \
conda run -n gps_gaussian python scripts/streaming_pipeline.py \
  --test_data_root /synology/rajrup/GPS-Gaussian/real_data \
  --ckpt_path ./models/GPS-GS_stage2_final.pth \
  --src_view 0 1 \
  --mode compiled \
  --mode_pipeline pipelined_mp \
  --gen_device cuda:0 \
  --codec_device cuda:1 \
  --output_dir ./gps_livogs_streaming_out_nosave \
  --warmup 5 \
  --save_interval 0
```

**Quality evaluation (decoded vs baseline):**

First generate baseline renders (uncompressed GPS-Gaussian novel views):

```bash
LD_LIBRARY_PATH=/home/rajrup/miniconda3/envs/gps_gaussian/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH \
python test_real_data_fast.py \
  --test_data_root /synology/rajrup/GPS-Gaussian/real_data \
  --ckpt_path ./models/GPS-GS_stage2_final.pth \
  --src_view 0 1 \
  --mode compiled 
```

Then evaluate the MP pipeline decoded frames against the baseline, with side-by-side render saving:

```bash
LD_LIBRARY_PATH=/home/rajrup/miniconda3/envs/gps_gaussian/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH \
python scripts/evaluate_streaming.py \
  --decoded_dir ./gps_livogs_streaming_out/decoded_frames \
  --baseline_dir ./fast_out_compiled \
  --test_data_root '/synology/rajrup/GPS-Gaussian/real_data' \
  --ckpt_path './models/GPS-GS_stage2_final.pth' \
  --src_view 0 1 \
  --output_csv ./gps_livogs_streaming_out/quality_eval.csv \
  --save_renders ./gps_livogs_streaming_out/eval_renders
```

Results: PSNR 54.30 dB, SSIM 0.9991 (near-lossless). Saved renders: `{frame}_baseline.jpg` and `{frame}_decoded.jpg`.
