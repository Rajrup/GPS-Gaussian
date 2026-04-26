# GPS-Gaussian: Technical Explanation

This document provides a detailed explanation of how GPS-Gaussian works, particularly during inference time with multiple fixed cameras.

**Reference**: [GPS-Gaussian: Generalizable Pixel-wise 3D Gaussian Splatting for Real-time Human Novel View Synthesis (CVPR 2024)](https://shunyuanzheng.github.io/GPS-Gaussian)

---

## Overview

GPS-Gaussian is a **feed-forward** method for real-time novel view synthesis of dynamic humans. Unlike traditional 3D Gaussian Splatting which requires per-subject optimization (minutes of training per scene), GPS-Gaussian directly **regresses** Gaussian parameters in a single forward pass, enabling instant rendering of unseen performers.

### Key Features
- Real-time rendering: 2K resolution at 25+ FPS
- No per-subject optimization required
- Works with sparse camera setups (6-8 cameras)
- Generalizes to unseen performers without fine-tuning

---

## Core Innovation: Pixel-wise Gaussian Parameter Maps

Instead of optimizing unstructured 3D point clouds, GPS-Gaussian defines **2D Gaussian parameter maps** on source view image planes:

- Each **foreground pixel** corresponds to a **single 3D Gaussian point**
- This enables efficient 2D convolution networks rather than expensive 3D operators
- The Gaussian parameters are: **Position (X)**, **Color (c)**, **Rotation (r)**, **Scaling (s)**, **Opacity (α)**

---

## Inference Pipeline with Multiple Cameras (e.g., 10 Cameras)

Here's how inference works step-by-step when you have multiple fixed cameras:

### Step 1: View Selection (Two-View Stereo)

**Key insight**: For any given novel viewpoint, only **two adjacent cameras** are used, not all cameras.

```
Given:
- N input cameras {C₁, C₂, ..., Cₙ} arranged in a circle
- Target novel viewpoint Cₜₐᵣ
- Scene center O

Algorithm:
1. Compute view vectors: Vₙ = Cₙ - O (camera position relative to scene center)
2. Compute target view vector: Vₜₐᵣ = Cₜₐᵣ - O
3. Find two nearest cameras by dot product similarity
4. Select (Vₗ, Vᵣ) as "left" and "right" working set
```

So if you want to render a view between camera 3 and camera 4, those two cameras form your stereo pair.

### Step 2: Feature Extraction

The two selected source images `Iₗ, Iᵣ` are:
1. **Rectified** (standard stereo rectification)
2. Fed through a **shared image encoder** `Eᵢₘ𝓰`

```
{fₗˢ}³ₛ₌₁, {fᵣˢ}³ₛ₌₁ = Eᵢₘ𝓰(Iₗ, Iᵣ)
```

The encoder produces multi-scale features at **1/2, 1/4, 1/8** resolution with 32, 48, and 96 channels.

### Step 3: Depth Estimation

This is the **bridge** between 2D image planes and 3D Gaussian representation.

**Process:**
1. Build a **3D correlation volume** from the feature maps:
   ```
   C(fₗˢ, fᵣˢ), Cᵢⱼₖ = Σₕ (fₗˢ)ᵢⱼₕ · (fᵣˢ)ᵢₖₕ
   ```

2. **Iteratively refine** depth using GRU units (inspired by RAFT-Stereo):
   - T=4 iterations of refinement
   - Outputs depth maps `Dₗ, Dᵣ` for **both** source views

3. **Convex upsampling** to full resolution

**Novel aspect**: Unlike traditional stereo methods that only estimate depth for one "reference" view, GPS-Gaussian estimates depth for **both** views symmetrically, enabling a compact parallelized implementation.

### Step 4: Gaussian Parameter Map Prediction

For each source view, five parameter maps are computed:

| Parameter | Map | How Obtained |
|-----------|-----|--------------|
| **Position** | Mₚ | Unprojection: `Mₚ(x) = Π⁻¹ₚ(x, D(x))` using depth + camera intrinsics |
| **Color** | Mᵧ | Directly from source RGB: `Mᵧ(x) = I(x)` (no SH coefficients needed for diffuse humans) |
| **Rotation** | Mᵣ | Neural network head with **normalization** (quaternion) |
| **Scaling** | Mₛ | Neural network head with **Softplus** activation |
| **Opacity** | Mα | Neural network head with **Sigmoid** activation |

**Architecture for rotation/scaling/opacity:**
```
Γ = Dₚₐᵣₘ(Eᵢₘ𝓰(I) ⊕ Eₐₑₚₜₕ(D))
```
- Image features + depth features are concatenated at all levels
- U-Net decoder produces pixel-wise Gaussian features
- Separate prediction heads (2 conv layers each) for each parameter

### Step 5: Unproject and Aggregate

1. **Unproject** the 2D Gaussian parameter maps from both source views to 3D space
2. **Aggregate** both sets of Gaussian points into a single representation

```
G₁ = Gaussians from left view (unprojected)
G₂ = Gaussians from right view (unprojected)
G = G₁ ∪ G₂  (combined representation)
```

### Step 6: Differentiable Gaussian Splatting

Render the novel view using standard 3D Gaussian Splatting:

1. **Project** 3D Gaussians to 2D image plane:
   ```
   Σ' = JWΣWᵀJᵀ  (2D covariance matrix)
   ```

2. **Alpha-blending** (similar to NeRF volume rendering):
   ```
   Cₒₗₒᵣ = Σᵢ cᵢ · αᵢ · Πⱼ₌₁ⁱ⁻¹(1 - αⱼ)
   ```

---

## Runtime Performance

This is where GPS-Gaussian shines for multi-view setups:

| Component | Time |
|-----------|------|
| **View-independent** (Gaussian parameter maps) | **27ms** |
| **View-dependent** (per novel view) | **0.8ms** |

**Key advantage**: The Gaussian representation is computed **once** per frame on source views. After that, rendering to any number of novel viewpoints is nearly instantaneous (0.8ms each). This enables:
- Multiple viewers seeing different viewpoints simultaneously
- 2K resolution at 25+ FPS

---

## Why Two Views Instead of All Cameras?

1. **Stereo matching works best with adjacent views** - Large baselines cause severe occlusions and correspondence failures
2. **Efficiency** - Processing all views would be much slower
3. **The angle between adjacent cameras is ~45°** (with 8 cameras), which is manageable for stereo

---

## How the Learned Opacity Helps

The jointly-trained opacity map learns to:
- Assign **low opacity** to depth ambiguity regions (margins, occluded areas)
- This effectively **eliminates artifacts** from noisy depth estimation
- Without this, depth errors at boundaries would cause visible noise in renderings

---

## Architecture Summary

```
N Cameras Capturing Human
          │
          ▼
    [View Selection]
          │
    Select 2 adjacent cameras for target viewpoint
          │
          ▼
┌─────────┴─────────┐
│   Iₗ (left)       │   Iᵣ (right)   │
└─────────┬─────────┘
          │
    [Shared Image Encoder]
          │
          ▼
    [Depth Estimation via Stereo Matching]
          │
          ▼
    [Gaussian Parameter Prediction]
    (Position, Color, Rotation, Scaling, Opacity)
          │
          ▼
    [Unproject to 3D & Aggregate]
          │
          ▼
    [3D Gaussian Splatting to Novel View]
          │
          ▼
    Rendered Novel View Image
```

---

## Network Architecture Details

### Image Encoder
- Similar to RAFT-Stereo feature encoder
- 5×5 convolution at input, followed by residual blocks
- Group normalization (instead of batch normalization)
- Produces features at 3 levels: 1/2, 1/4, 1/8 resolution
- Channels: 32, 48, 96 respectively

### Depth Estimation Module
- Builds 3D correlation volume from feature maps
- GRU-based iterative refinement (T=4 iterations)
- Symmetric: estimates depth for **both** views simultaneously
- ~30% efficiency gain from shared computation

### Gaussian Parameter Prediction Module
- Depth encoder (same architecture as image encoder)
- U-Net decoder with skip connections
- Three prediction heads for rotation, scaling, opacity
- Position from depth unprojection, color from RGB

---

## Training Details

- **Two-stage training**:
  1. Pre-train depth estimation module for 40k iterations
  2. Joint training of both modules for 100k iterations
- **Optimizer**: AdamW with initial learning rate 2e-4
- **Batch size**: 2
- **Training time**: ~15 hours on single RTX 3090
- **Training data**: 1700 Twindom + 526 THuman2.0 scans

### Loss Functions
```
L = Lᵣₑₙₐₑᵣ + Lₐᵢₛₚ

Lᵣₑₙₐₑᵣ = 0.8 × L₁ + 0.2 × SSIM
Lₐᵢₛₚ = Σₜ 0.9^(T-t) × ||dₐₜ - dₜ||₁
```

---

## Limitations

1. Requires accurate **foreground matting** as preprocessing
2. Cannot perfectly handle very large disparity (when target region is totally invisible in both source views)
3. Best suited for ~45° angle between adjacent cameras

---

# Codebase Analysis: Inference Code Flow

This section explains how the GPS-Gaussian codebase implements inference, based on analysis of the actual implementation.

---

## Input Data Structure

The real data folder (`real_data/`) has the following structure:

```
real_data/
├── img/
│   ├── 0001/           # Sample/frame name
│   │   ├── 0.jpg       # Camera 0 image
│   │   ├── 1.jpg       # Camera 1 image
│   │   ├── ...
│   │   └── 15.jpg      # Camera 15 image (16 cameras total)
│   ├── 0002/
│   └── ...
├── mask/
│   ├── 0001/
│   │   ├── 0.png       # Foreground mask for camera 0
│   │   └── ...
│   └── ...
└── parm/
    ├── 0001/
    │   ├── 0_intrinsic.npy   # 3x3 intrinsic matrix for camera 0
    │   ├── 0_extrinsic.npy   # 3x4 extrinsic matrix for camera 0
    │   └── ...
    └── ...
```

**Key points:**
- Each sample (frame) has images from multiple cameras (e.g., 16 cameras: 0-15)
- Foreground masks are required for masking out the background
- Camera parameters (intrinsic & extrinsic) are stored as numpy files

---

## Core Network Architecture (`lib/network.py`)

The main model is `RtStereoHumanModel`:

```python
class RtStereoHumanModel(nn.Module):
    def __init__(self, cfg, with_gs_render=False):
        # Three main components:
        self.img_encoder = UnetExtractor(...)      # Shared image feature extractor
        self.raft_stereo = RAFTStereoHuman(...)    # Stereo depth estimation (RAFT-based)
        self.gs_parm_regresser = GSRegresser(...)  # Gaussian parameter regression
```

### Forward Pass (Inference Mode)

```python
def forward(self, data, is_train=False):
    # 1. Concatenate left and right images
    image = torch.cat([data['lmain']['img'], data['rmain']['img']], dim=0)
    
    # 2. Extract image features (shared encoder for both views)
    img_feat = self.img_encoder(image)  # Returns 3 scales: 1/2, 1/4, 1/8
    
    # 3. Run RAFT-Stereo depth estimation
    flow_up = self.raft_stereo(img_feat[2], iters=self.val_iters, test_mode=True)
    # flow_up contains disparity predictions for both views
    
    # 4. Convert disparity to Gaussian parameters
    data = self.flow2gsparms(image, img_feat, data, bs)
    
    return data, None, None
```

### Flow to Gaussian Parameters (`flow2gsparms`)

```python
def flow2gsparms(self, lr_img, lr_img_feat, data, bs):
    for view in ['lmain', 'rmain']:
        # Convert disparity (flow) to depth
        data[view]['depth'] = flow2depth(data[view])
        
        # Unproject depth to 3D point cloud
        data[view]['xyz'] = depth2pc(depth, extrinsic, intrinsic)
        
        # Track valid pixels (non-zero depth)
        data[view]['pts_valid'] = (depth != 0.0)
    
    # Concatenate depths from both views
    lr_depth = torch.concat([data['lmain']['depth'], data['rmain']['depth']], dim=0)
    
    # Regress rotation, scale, opacity maps
    rot_maps, scale_maps, opacity_maps = self.gs_parm_regresser(lr_img, lr_depth, lr_img_feat)
    
    # Split back to left/right views
    data['lmain']['rot_maps'], data['rmain']['rot_maps'] = torch.split(rot_maps, [bs, bs])
    # ... same for scale and opacity
```

---

## Depth Estimation: Disparity to Depth (`lib/utils.py`)

```python
def flow2depth(data):
    # Disparity = offset - flow_prediction
    offset = data['ref_intr'][:, 0, 2] - data['intr'][:, 0, 2]
    disparity = offset - data['flow_pred']
    
    # Depth = -disparity / Tf_x (baseline * focal_length)
    depth = -disparity / data['Tf_x']
    
    # Apply foreground mask
    depth *= data['mask'][:, :1, :, :]
    return depth
```

## Depth to Point Cloud (`lib/utils.py`)

```python
def depth2pc(depth, extrinsic, intrinsic):
    # Create pixel grid
    y, x = torch.meshgrid(...)
    pts_2d = torch.stack([x, y, ones], dim=-1)
    
    # Apply inverse intrinsic (pixel to camera coordinates)
    pts_2d[..., 2] = 1.0 / (depth + 1e-8)
    pts_2d[..., 0] -= intrinsic[:, 0, 2]  # cx
    pts_2d[..., 1] -= intrinsic[:, 1, 2]  # cy
    pts_2d[..., 0] /= intrinsic[:, 0, 0]  # fx
    pts_2d[..., 1] /= intrinsic[:, 1, 1]  # fy
    
    # Apply inverse extrinsic (camera to world coordinates)
    rot_t = rot.permute(0, 2, 1)  # R^T
    pts = torch.bmm(rot_t, pts_2d) - torch.bmm(rot_t, trans)
    
    return pts  # Shape: [B, H*W, 3]
```

---

## Gaussian Parameter Regression (`lib/gs_parm_network.py`)

```python
class GSRegresser(nn.Module):
    def __init__(self, cfg, rgb_dim=3, depth_dim=1):
        # Depth encoder (same architecture as image encoder)
        self.depth_encoder = UnetExtractor(in_channel=1, ...)
        
        # U-Net style decoder
        self.decoder3 = ...  # Fuses img_feat + depth_feat at 1/8 scale
        self.decoder2 = ...  # Fuses at 1/4 scale
        self.decoder1 = ...  # Fuses at 1/2 scale
        
        # Prediction heads
        self.rot_head = nn.Sequential(Conv, ReLU, Conv)      # Output: 4 channels (quaternion)
        self.scale_head = nn.Sequential(Conv, ReLU, Conv, Softplus)  # Output: 3 channels
        self.opacity_head = nn.Sequential(Conv, ReLU, Conv, Sigmoid) # Output: 1 channel
    
    def forward(self, img, depth, img_feat):
        # Encode depth
        depth_feat1, depth_feat2, depth_feat3 = self.depth_encoder(depth)
        
        # Fuse image and depth features at each scale
        feat3 = concat(img_feat3, depth_feat3)
        feat2 = concat(img_feat2, depth_feat2)
        feat1 = concat(img_feat1, depth_feat1)
        
        # Decode with skip connections
        up3 = self.decoder3(feat3)
        up2 = self.decoder2(concat(upsample(up3), feat2))
        up1 = self.decoder1(concat(upsample(up2), feat1))
        
        # Final output at full resolution
        out = concat(upsample(up1), img, depth)
        out = self.out_conv(out)
        
        # Predict Gaussian parameters
        rot_out = normalize(self.rot_head(out))      # Normalized quaternion
        scale_out = clamp_max(self.scale_head(out), 0.01)  # Max scale = 0.01
        opacity_out = self.opacity_head(out)         # [0, 1]
        
        return rot_out, scale_out, opacity_out
```

---

## Novel View Rendering (`lib/GaussianRender.py`)

```python
def pts2render(data, bg_color):
    for i in range(batch_size):
        xyz_valid, rgb_valid, rot_valid, scale_valid, opacity_valid = [], [], [], [], []
        
        # Collect valid Gaussians from BOTH source views
        for view in ['lmain', 'rmain']:
            valid_mask = data[view]['pts_valid'][i]
            xyz = data[view]['xyz'][i][valid_mask]
            rgb = data[view]['img'][i].view(-1, 3)[valid_mask]
            rot = data[view]['rot_maps'][i].view(-1, 4)[valid_mask]
            scale = data[view]['scale_maps'][i].view(-1, 3)[valid_mask]
            opacity = data[view]['opacity_maps'][i].view(-1, 1)[valid_mask]
            
            # Append to lists
            xyz_valid.append(xyz)
            # ... etc
        
        # Concatenate Gaussians from both views
        pts_xyz = torch.concat(xyz_valid, dim=0)
        pts_rgb = torch.concat(rgb_valid, dim=0)
        # ... etc
        
        # Render using diff-gaussian-rasterization
        rendered_image = render(data, i, pts_xyz, pts_rgb, rot, scale, opacity, bg_color)
    
    data['novel_view']['img_pred'] = rendered_images
    return data
```

---

## Gaussian Splatting Renderer (`gaussian_renderer/__init__.py`)

```python
def render(data, idx, pts_xyz, pts_rgb, rotations, scales, opacity, bg_color):
    # Set up rasterization settings
    raster_settings = GaussianRasterizationSettings(
        image_height=data['novel_view']['height'][idx],
        image_width=data['novel_view']['width'][idx],
        tanfovx=tan(FovX * 0.5),
        tanfovy=tan(FovY * 0.5),
        viewmatrix=data['novel_view']['world_view_transform'][idx],
        projmatrix=data['novel_view']['full_proj_transform'][idx],
        campos=data['novel_view']['camera_center'][idx],
        ...
    )
    
    rasterizer = GaussianRasterizer(raster_settings)
    
    # Rasterize Gaussians to image
    rendered_image, radii, _ = rasterizer(
        means3D=pts_xyz,
        colors_precomp=pts_rgb,  # Direct RGB, no SH
        opacities=opacity,
        scales=scales,
        rotations=rotations,
    )
    
    return rendered_image
```

---

## How `test_real_data.py` Works

This script renders a **single novel view** at a specified interpolation ratio between two source cameras.

### Command Line Usage
```bash
python test_real_data.py \
    --test_data_root '/path/to/real_data' \
    --ckpt_path './models/GPS-GS_stage2_final.pth' \
    --src_view 0 1 \
    --ratio 0.5
```

### Code Flow

```python
class StereoHumanRender:
    def __init__(self, cfg_file, phase):
        # 1. Create model with Gaussian rendering enabled
        self.model = RtStereoHumanModel(self.cfg, with_gs_render=True)
        
        # 2. Load dataset
        self.dataset = StereoHumanDataset(self.cfg.dataset, phase='test')
        
        # 3. Load checkpoint
        self.load_ckpt(self.cfg.restore_ckpt)
        self.model.eval()
    
    def infer_sequence(self, view_select, ratio=0.5):
        for idx in range(total_frames):
            # 1. Load data for the two source views
            item = self.dataset.get_test_item(idx, source_id=view_select)
            data = self.fetch_data(item)  # Move to GPU
            
            # 2. Compute novel camera pose (interpolated between two views)
            data = get_novel_calib(data, ratio=ratio, ...)
            
            # 3. Run inference
            with torch.no_grad():
                data, _, _ = self.model(data, is_train=False)
                
                # 4. Render novel view
                data = pts2render(data, bg_color=...)
            
            # 5. Save result
            render_novel = data['novel_view']['img_pred']
            cv2.imwrite(f'{name}_novel.jpg', render_novel)
```

### Novel Camera Pose Interpolation (`lib/utils.py`)

```python
def get_novel_calib(data, opt, ratio=0.5, ...):
    # Get camera parameters from both source views
    intr0, extr0 = data['lmain']['intr_ori'], data['lmain']['extr_ori']
    intr1, extr1 = data['rmain']['intr_ori'], data['rmain']['extr_ori']
    
    # Interpolate rotation using SLERP (Spherical Linear Interpolation)
    rot0, rot1 = extr0[:3, :3], extr1[:3, :3]
    rots = Rotation.from_matrix([rot0, rot1])
    slerp = Slerp([0, 1], rots)
    rot_interp = slerp(ratio)  # Smooth rotation interpolation
    
    # Linearly interpolate translation
    trans_interp = (1.0 - ratio) * extr0[:3, 3] + ratio * extr1[:3, 3]
    
    # Linearly interpolate intrinsics
    intr_interp = (1.0 - ratio) * intr0 + ratio * intr1
    
    # Compute projection matrices for Gaussian splatting
    world_view_transform = getWorld2View2(R, T, ...)
    projection_matrix = getProjectionMatrix(znear, zfar, K, h, w)
    full_proj_transform = world_view_transform @ projection_matrix
    
    data['novel_view']['world_view_transform'] = world_view_transform
    data['novel_view']['full_proj_transform'] = full_proj_transform
    data['novel_view']['camera_center'] = world_view_transform.inverse()[3, :3]
    
    return data
```

---

## How `test_view_interp.py` Works

This script generates **multiple novel views** between two source cameras for view interpolation.

### Command Line Usage
```bash
python test_view_interp.py \
    --test_data_root '/path/to/render_data/val' \
    --ckpt_path './models/GPS-GS_stage2_final.pth' \
    --novel_view_nums 5
```

### Code Flow

```python
def infer_static(self, view_select, novel_view_nums):
    for idx in range(total_samples):
        # Load data for source views [0, 1]
        item = self.dataset.get_test_item(idx, source_id=view_select)
        data = self.fetch_data(item)
        
        # Generate multiple novel views at different ratios
        for i in range(novel_view_nums):
            # Compute ratio: evenly spaced between the two views
            # e.g., for 5 views: 0.1, 0.3, 0.5, 0.7, 0.9
            ratio_tmp = (i + 0.5) * (1 / novel_view_nums)
            
            # Get interpolated camera pose
            data_i = get_novel_calib(data, ratio=ratio_tmp, ...)
            
            # Run inference and render
            with torch.no_grad():
                data_i, _, _ = self.model(data_i, is_train=False)
                data_i = pts2render(data_i, bg_color=...)
            
            # Save with index suffix
            cv2.imwrite(f'{name}_novel{i:02d}.jpg', render_novel)
```

### Key Difference from `test_real_data.py`

| Aspect | `test_real_data.py` | `test_view_interp.py` |
|--------|---------------------|----------------------|
| **Output** | Single novel view per frame | Multiple novel views per frame |
| **Ratio** | Fixed (user-specified, default 0.5) | Evenly spaced (0.1, 0.3, 0.5, 0.7, 0.9 for 5 views) |
| **Source views** | User-specified `--src_view` | Hardcoded `[0, 1]` |
| **Output folder** | `./test_out/` | `./interp_out/` |
| **Use case** | Render specific viewpoint | Generate smooth view interpolation |

---

## Complete Inference Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT DATA                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Camera 0    │  │ Camera 1    │  │ Camera 2    │  │ Camera N    │        │
│  │ img + mask  │  │ img + mask  │  │ img + mask  │  │ img + mask  │        │
│  │ intr + extr │  │ intr + extr │  │ intr + extr │  │ intr + extr │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 1: VIEW SELECTION                                   │
│  Select two adjacent source views based on target novel viewpoint           │
│  e.g., --src_view 0 1  →  left=Camera0, right=Camera1                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 2: STEREO RECTIFICATION                             │
│  StereoHumanDataset.get_rectified_stereo_data()                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  cv2.stereoRectify(intr0, intr1, R, T)  →  R0, R1, P0, P1           │   │
│  │  cv2.initUndistortRectifyMap()  →  rectify_maps                      │   │
│  │  cv2.remap(img, rectify_maps)  →  rectified_images                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  Output: Rectified images where epipolar lines are horizontal               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 3: IMAGE FEATURE EXTRACTION                         │
│  UnetExtractor (shared for both views)                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Input: [img_left, img_right]  (concatenated)                        │   │
│  │  Conv 5×5 → ResBlocks → 3 scales of features                         │   │
│  │  Output: feat1 (1/2), feat2 (1/4), feat3 (1/8)                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 4: STEREO DEPTH ESTIMATION                          │
│  RAFTStereoHuman (RAFT-based iterative refinement)                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  1. Build correlation volume: C[i,j,k] = Σ f_left[i,j] · f_right[i,k]│   │
│  │  2. Initialize disparity flow to 0                                   │   │
│  │  3. Iterate T times (T=32 for validation):                           │   │
│  │     - Look up correlation at current flow estimate                   │   │
│  │     - GRU update: delta_flow = GRU(corr, flow, context)              │   │
│  │     - Update: flow = flow + delta_flow                               │   │
│  │  4. Convex upsample to full resolution                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  Output: flow_left, flow_right (disparity maps for both views)              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 5: DISPARITY → DEPTH → 3D POINTS                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  flow2depth():                                                       │   │
│  │    disparity = offset - flow_pred                                    │   │
│  │    depth = -disparity / Tf_x                                         │   │
│  │                                                                       │   │
│  │  depth2pc():                                                         │   │
│  │    pts_cam = K⁻¹ · [u, v, 1] · depth                                 │   │
│  │    pts_world = R^T · pts_cam - R^T · t                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  Output: xyz coordinates for each pixel (H×W×3)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 6: GAUSSIAN PARAMETER REGRESSION                    │
│  GSRegresser                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  1. Encode depth map: depth_feat = DepthEncoder(depth)               │   │
│  │  2. Fuse features: feat = concat(img_feat, depth_feat) at each level │   │
│  │  3. U-Net decode: up3 → up2 → up1 (with skip connections)            │   │
│  │  4. Predict heads:                                                   │   │
│  │     - Rotation: normalize(rot_head(out))  →  [H, W, 4] quaternion    │   │
│  │     - Scale: clamp(scale_head(out), max=0.01)  →  [H, W, 3]          │   │
│  │     - Opacity: sigmoid(opacity_head(out))  →  [H, W, 1]              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  Output: Per-pixel Gaussian parameters (rot, scale, opacity)                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 7: COMPUTE NOVEL CAMERA POSE                        │
│  get_novel_calib()                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Given ratio (0.0 = left view, 1.0 = right view):                    │   │
│  │  - Rotation: SLERP(R_left, R_right, ratio)                           │   │
│  │  - Translation: LERP(t_left, t_right, ratio)                         │   │
│  │  - Intrinsics: LERP(K_left, K_right, ratio)                          │   │
│  │                                                                       │   │
│  │  Compute Gaussian splatting matrices:                                │   │
│  │  - world_view_transform = [R|t]                                      │   │
│  │  - projection_matrix = perspective(fov, near, far)                   │   │
│  │  - full_proj_transform = world_view × projection                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 8: AGGREGATE GAUSSIANS FROM BOTH VIEWS              │
│  pts2render()                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  For each view (left, right):                                        │   │
│  │    - Filter by valid mask (depth > 0)                                │   │
│  │    - Collect: xyz, rgb (from image), rot, scale, opacity             │   │
│  │                                                                       │   │
│  │  Concatenate all valid Gaussians from both views:                    │   │
│  │    pts_xyz = concat(xyz_left, xyz_right)                             │   │
│  │    pts_rgb = concat(rgb_left, rgb_right)                             │   │
│  │    ... (same for rot, scale, opacity)                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  Output: Combined Gaussian point cloud (N_total × attributes)               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 9: GAUSSIAN SPLATTING RENDER                        │
│  diff_gaussian_rasterization.GaussianRasterizer                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Input:                                                              │   │
│  │    - means3D: [N, 3] world positions                                 │   │
│  │    - colors_precomp: [N, 3] RGB colors (no SH)                       │   │
│  │    - rotations: [N, 4] quaternions                                   │   │
│  │    - scales: [N, 3] scale factors                                    │   │
│  │    - opacities: [N, 1] opacity values                                │   │
│  │    - viewmatrix, projmatrix: novel view camera matrices              │   │
│  │                                                                       │   │
│  │  Process:                                                            │   │
│  │    1. Project 3D Gaussians to 2D                                     │   │
│  │    2. Compute 2D covariance from 3D covariance + projection          │   │
│  │    3. Sort by depth                                                  │   │
│  │    4. Alpha-blend front-to-back                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  Output: Rendered image [H, W, 3]                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT                                            │
│                    Novel view image saved to disk                           │
│                    e.g., ./test_out/0001_novel.jpg                          │
│                    or    ./interp_out/0001_novel00.jpg                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary: Key Code Files

| File | Purpose |
|------|---------|
| `test_real_data.py` | Single novel view inference script |
| `test_view_interp.py` | Multiple novel views (view interpolation) script |
| `lib/network.py` | Main model (`RtStereoHumanModel`) |
| `lib/human_loader.py` | Dataset class (`StereoHumanDataset`) |
| `lib/gs_parm_network.py` | Gaussian parameter regression network |
| `lib/GaussianRender.py` | Aggregates Gaussians and calls renderer |
| `lib/utils.py` | Helper functions (depth2pc, flow2depth, get_novel_calib) |
| `core/raft_stereo_human.py` | RAFT-based stereo depth estimation |
| `core/extractor.py` | Feature extraction networks |
| `gaussian_renderer/__init__.py` | Gaussian splatting renderer wrapper |

---

## References

- Original paper: Zheng et al., "GPS-Gaussian: Generalizable Pixel-wise 3D Gaussian Splatting for Real-time Human Novel View Synthesis", CVPR 2024
- Project page: https://shunyuanzheng.github.io/GPS-Gaussian
- Related work: RAFT-Stereo, 3D Gaussian Splatting, PIFu
