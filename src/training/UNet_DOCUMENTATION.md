# UNet Time Evolution Operator - Documentation

## Contents

- [UNet Time Evolution Operator - Documentation](#unet-time-evolution-operator---documentation)
  - [Contents](#contents)
  - [Overview](#overview)
  - [Notebook Structure](#notebook-structure)
  - [Data Handling and Structure](#data-handling-and-structure)
    - [Data Directory Structure](#data-directory-structure)
    - [Data Loading Process](#data-loading-process)
    - [How Sampling Works](#how-sampling-works)
      - [Timestep Sampling](#timestep-sampling)
      - [Sample Generation](#sample-generation)
      - [Total Dataset Size](#total-dataset-size)
      - [Time Sequence Handling](#time-sequence-handling)
    - [Train/Test Split](#traintest-split)
  - [UNet Architecture Explanation](#unet-architecture-explanation)
    - [Overview](#overview-1)
    - [Architecture Components](#architecture-components)
      - [1. **Encoder (Downsampling Path)**](#1-encoder-downsampling-path)
      - [2. **Decoder (Upsampling Path)**](#2-decoder-upsampling-path)
      - [3. **DoubleConv Block**](#3-doubleconv-block)
    - [Key Design Features](#key-design-features)
      - [1. **Residual Connection**](#1-residual-connection)
      - [2. **Output Masking**](#2-output-masking)
      - [3. **Parameter Injection**](#3-parameter-injection)
      - [4. **Input Format**](#4-input-format)
    - [Forward Pass Flow](#forward-pass-flow)
    - [Why UNet for Fluid Dynamics?](#why-unet-for-fluid-dynamics)
    - [Model Size](#model-size)
  - [Comparison with Reference CFDBench Implementation](#comparison-with-reference-cfdbench-implementation)
    - [Model Size Comparison](#model-size-comparison)
    - [Task Difference: Multi-Step vs Single-Step Training](#task-difference-multi-step-vs-single-step-training)
    - [Same Dataset, Different Approach](#same-dataset-different-approach)
    - [Summary of Key Differences](#summary-of-key-differences)
    - [Recommendations for This Notebook](#recommendations-for-this-notebook)
  - [Key Hyperparameters for Post-Tuning](#key-hyperparameters-for-post-tuning)
    - [Dataset Parameters](#dataset-parameters)
    - [Training Parameters](#training-parameters)
    - [Model Architecture Parameters](#model-architecture-parameters)
    - [Loss Function Parameters](#loss-function-parameters)
    - [Learning Rate Scheduler](#learning-rate-scheduler)
  - [Training Process](#training-process)
    - [How Time Evolution Works](#how-time-evolution-works)
    - [Training Tips](#training-tips)
  - [Evaluation](#evaluation)
    - [Evaluation Metrics](#evaluation-metrics)
    - [Running Evaluation](#running-evaluation)
    - [Visualization](#visualization)
  - [Post-Tuning Guide](#post-tuning-guide)
    - [Step-by-Step Tuning Process](#step-by-step-tuning-process)
    - [Common Tuning Scenarios](#common-tuning-scenarios)
  - [Important Notes](#important-notes)
    - [Data Handling](#data-handling)
    - [Model Behavior](#model-behavior)
    - [Checkpointing](#checkpointing)
    - [Device Configuration](#device-configuration)
  - [Troubleshooting](#troubleshooting)
    - [Issue: Loss becomes NaN or Inf](#issue-loss-becomes-nan-or-inf)
    - [Issue: Out of memory errors](#issue-out-of-memory-errors)
    - [Issue: Predictions diverge over long rollouts](#issue-predictions-diverge-over-long-rollouts)
    - [Issue: Model predictions are all zeros](#issue-model-predictions-are-all-zeros)
    - [Issue: Training is very slow](#issue-training-is-very-slow)
  - [File Structure](#file-structure)
  - [References](#references)
  - [Quick Reference: Key Parameters](#quick-reference-key-parameters)

## Overview

This notebook implements a UNet-based neural time integrator for 2D cylinder flow prediction using the CFDBench dataset. The model learns to predict the evolution of Navier-Stokes flow fields over time by learning the discrete flow map:

$$u^{t+\Delta t}(x) = G_\theta(u^t(x), BC, \nu, \rho, geometry)$$

**Key Features:**
- **One-step predictor**: The model predicts the next timestep from the current state
- **Autoregressive rollout**: Multi-step predictions are made by iteratively applying the model
- **Residual connections**: Helps with training stability and gradient flow
- **Output masking**: Enforces boundary conditions at the cylinder geometry
- **Parameter conditioning**: Injects physical parameters (velocity, density, viscosity, geometry) into the model

**Advantage**: UNet is faster on Mac systems since it doesn't require FFT operations (unlike FNO).

---

## Notebook Structure

The notebook is organized into 8 main sections:

1. **Data Loading** - Loads CFDBench dataset with JSON metadata
2. **Geometry Mask Creation** - Creates binary masks for cylinder geometry
3. **UNet Architecture Implementation** - Defines the model architecture
4. **Data Preprocessing** - Prepares input channels and dataset
5. **Train/Test Split** - 70% training, 30% evaluation split
6. **Training Code Block** - Training loop with checkpointing
7. **Evaluation Code Block** - Metrics computation and evaluation
8. **Visualization** - Predicted vs Ground Truth comparison with difference plots

---

## Data Handling and Structure

### Data Directory Structure

The CFDBench dataset is organized in a folder structure with different subsets:

```
data/
└── bc/              # Boundary conditions subset
    ├── case0000/
    │   ├── u.npy    # U-velocity field: (T, H, W) = (620, 64, 64)
    │   ├── v.npy    # V-velocity field: (T, H, W) = (620, 64, 64)
    │   └── params.json  # Physical parameters
    ├── case0001/
    └── ...
```

**Key Points:**
- **`bc/` directory**: Contains boundary condition variations (different inlet velocities)
- Each case has a unique identifier (e.g., `case0000`, `case0001`)
- Data files: `u.npy` and `v.npy` contain velocity fields over time
- **Shape**: `(620, 64, 64)` = (timesteps, height, width)
- **Parameters**: Stored in `params.json` with physical properties

### Data Loading Process

**Location**: Section 1 (Data Loading)

```python
from data_handle.data import get_cylinder_auto_datasets

bc_train_data, bc_dev_data, bc_test_data = get_cylinder_auto_datasets(
    data_dir=Path("../../../"),  # Relative path to data folder (3 levels up from notebook)
    subset_name="bc",            # Load from bc/ directory
    norm_props=True,              # Normalize properties
    norm_bc=True,                 # Normalize boundary conditions
    case_fraction=1.0,            # Use all cases (1.0 = 100%)
    load_splits=["train", "dev", "test"]
)
```

**What happens:**
1. Searches for data in `data_dir/bc/` directory
2. Loads cases from train/dev/test splits (predefined in dataset)
3. **Note**: Only train and dev data are combined and used. Test data (`bc_test_data`) is loaded but not used in the final dataset.
4. Each case contains:
   - `u_data`: U-velocity array `(620, 64, 64)`
   - `v_data`: V-velocity array `(620, 64, 64)`
   - `case_params`: Dictionary with physical parameters:
     - `vel_in`: Inlet velocity
     - `density`: Fluid density (ρ)
     - `viscosity`: Dynamic viscosity (μ)
     - `radius`: Cylinder radius
     - `x_min`, `x_max`, `y_min`, `y_max`: Domain boundaries

### How Sampling Works

**Location**: Section 4 (Data Preprocessing) - `CylinderFlowDataset` class

The dataset uses **lazy loading** - samples are generated on-the-fly, not pre-loaded into memory.

#### Timestep Sampling

For each case with `T=620` timesteps and `rollout_steps=10`:

1. **Available starting points**: `range(0, T - rollout_steps)` = `range(0, 610)`
   - We need `rollout_steps` future timesteps, so we can't start at timestep `T - rollout_steps` or later
   - With `rollout_steps=10` and `T=620`: can start from timesteps 0 to 609 (610 possible starting points)

2. **Stride sampling** (`timestep_stride`):
   ```python
   timesteps = list(range(0, T - rollout_steps, timestep_stride))
   ```
   - `stride=1`: Use all timesteps → 610 samples per case
   - `stride=2`: Every other timestep → 305 samples per case
   - `stride=3`: Every third timestep → 203 samples per case

3. **Limit per case** (`max_samples_per_case`):
   ```python
   if max_samples_per_case is not None:
       timesteps = timesteps[:max_samples_per_case]
   ```
   - Limits the number of samples per case
   - Example: `max_samples_per_case=200` → Use first 200 timesteps from the stride-sampled list

#### Sample Generation

For each `(case_idx, timestep_t)` pair:

```python
# Input: Current state at timestep t
u_t = u_data[t]      # Shape: (64, 64)
v_t = v_data[t]      # Shape: (64, 64)

# Target: Future states (K steps ahead)
target_u = u_data[t+1:t+rollout_steps+1]  # Shape: (10, 64, 64)
target_v = v_data[t+1:t+rollout_steps+1]  # Shape: (10, 64, 64)
```

**Example with `rollout_steps=10`:**
- Input at `t=0` → Predict `t=1` through `t=10`
- Input at `t=1` → Predict `t=2` through `t=11`
- Input at `t=2` → Predict `t=3` through `t=12`
- ...

#### Total Dataset Size

```
Total samples = (Number of cases) × (Samples per case)

Example:
- 45 cases
- max_samples_per_case = 200
- Total = 45 × 200 = 9,000 samples
```

#### Time Sequence Handling

**Important**: The dataset does **not** explicitly encode time. Time ordering is **implicit** in the array structure:

- First dimension of `u_data` and `v_data` arrays = time axis
- `u_data[0]` = timestep 0, `u_data[1]` = timestep 1, etc.
- The model learns temporal dynamics through the autoregressive training process

### Train/Test Split

**Location**: Section 5 (Train/Test Split)

The split is done at the **sample level**, not case level:

```python
n_samples = len(full_dataset)  # Total samples across all cases
n_train = int(n_samples * 0.7)  # 70% for training

# Random shuffle
indices = np.arange(n_samples)
np.random.shuffle(indices)

train_indices = indices[:n_train]
eval_indices = indices[n_train:]
```

**Implications:**
- Samples from the same case can be in both train and eval sets
- This is intentional - provides better generalization
- The model sees different temporal segments from the same simulation

---

## UNet Architecture Explanation

### Overview

The UNet is a convolutional encoder-decoder network originally designed for image segmentation, adapted here for **spatial-temporal prediction** of fluid flow fields.

### Architecture Components

#### 1. **Encoder (Downsampling Path)**

The encoder extracts hierarchical features from the input velocity field:

```
Input [B, 2, 64, 64]  (u, v velocities) - format: [batch, channels, height, width]
    ↓ Add mask channel
Input [B, 3, 64, 64]  (u, v velocities + mask)
    ↓
DoubleConv → [B, 64, 64, 64]   (x1) - [batch, 64 channels, 64 height, 64 width]
    ↓ MaxPool (stride=2)
DoubleConv → [B, 128, 32, 32]  (x2) - [batch, 128 channels, 32 height, 32 width]
    ↓ MaxPool (stride=2)
DoubleConv → [B, 256, 16, 16]  (x3) - [batch, 256 channels, 16 height, 16 width]
    ↓ MaxPool (stride=2)
DoubleConv → [B, 512, 8, 8]    (x4) - [batch, 512 channels, 8 height, 8 width]
    ↓ MaxPool (stride=2)
DoubleConv → [B, 1024, 4, 4]   (x5) - Bottleneck [batch, 1024 channels, 4 height, 4 width]
```

**Note**: Input to `in_conv` is `[B, 3, 64, 64]` because mask is concatenated after residual extraction:
- Original input: `[B, 2, 64, 64]` (velocities)
- Residual extracted: `[B, 2, 64, 64]` (saved for later)
- Mask added: `[B, 2, 64, 64]` + `[B, 1, 64, 64]` = `[B, 3, 64, 64]` (input to in_conv)

**Purpose**: 
- Captures features at multiple scales
- Lower resolution = larger receptive field (sees more of the flow field)
- Each level captures different patterns (local vortices, global flow patterns, etc.)

#### 2. **Decoder (Upsampling Path)**

The decoder reconstructs the output at full resolution using skip connections:

```
Bottleneck [B, 1024, 4, 4] - format: [batch, channels, height, width]
    ↓ Upsample (2x) + Concatenate with x4 [B, 512, 8, 8]
DoubleConv → [B, 512, 8, 8]
    ↓ Upsample (2x) + Concatenate with x3 [B, 256, 16, 16]
DoubleConv → [B, 256, 16, 16]
    ↓ Upsample (2x) + Concatenate with x2 [B, 128, 32, 32]
DoubleConv → [B, 128, 32, 32]
    ↓ Upsample (2x) + Concatenate with x1 [B, 64, 64, 64]
DoubleConv → [B, 64, 64, 64]
    ↓
Output Conv → [B, 2, 64, 64]  (predicted u and v increments) - [batch, 2 channels, 64 height, 64 width]
    ↓ Residual connection (adds input velocities [B, 2, 64, 64])
    ↓ Output masking (zeros cylinder region)
    ↓ Permute (0, 2, 3, 1) - swap channels to last dimension
Final Output → [B, 64, 64, 2]  (predicted velocities at next timestep) - [batch, 64 height, 64 width, 2 channels]
```

**Skip Connections**:
- Concatenate encoder features with decoder features at the same resolution
- Preserves fine-grained spatial details
- Helps with gradient flow during training

#### 3. **DoubleConv Block**

Each convolution block consists of two 3×3 convolutions:

```python
DoubleConv:
    Conv2d(3×3) → BatchNorm → ReLU
    Conv2d(3×3) → BatchNorm → ReLU
```

**Features**:
- **Replicate padding**: Handles boundaries better than zero-padding
- **Batch normalization**: Stabilizes training, speeds up convergence
- **ReLU activation**: Non-linearity for learning complex patterns

### Key Design Features

#### 1. **Residual Connection**

```python
if use_residual:
    preds = preds + residual  # residual = input velocities
```

**Purpose**:
- Model predicts **increments** (Δu, Δv) rather than absolute values
- Adds current state: `u_{t+1} = u_t + Δu`
- Helps with training stability and learning rate
- Similar to ResNet philosophy

#### 2. **Output Masking**

```python
if use_output_mask:
    preds = preds * mask  # Zero out predictions inside cylinder
```

**Purpose**:
- Enforces boundary conditions at cylinder geometry
- Mask = 0 inside cylinder, 1 in fluid domain
- Prevents model from predicting flow inside solid geometry

#### 3. **Parameter Injection**

Physical parameters can be injected at two locations:

**Option A: At Input** (`insert_case_params_at="input"`)
```python
# Concatenate parameters as additional channels
x = [velocities, mask, case_params]  # All as spatial channels
```

**Option B: At Hidden Layer** (`insert_case_params_at="hidden"`) - **Default**
```python
# Add to bottleneck features
bottleneck = bottleneck + case_params_fc(case_params)
```

**Purpose**:
- Conditions the model on physical parameters (velocity, density, viscosity, geometry)
- Allows model to adapt predictions based on flow conditions
- Hidden injection is typically more effective

#### 4. **Input Format**

The model supports two input formats:

**Legacy Format** (11 channels):
```python
[u, v, u_B, ρ, μ, d, x_min, x_max, y_min, y_max, mask]
```
- All information baked into channels
- Simpler but less flexible

**New Format** (separate inputs):
```python
velocities: [B, H, W, 2]      # u and v only (channels last)
case_params: [B, 8]          # Physical parameters
mask: [B, H, W]              # Geometry mask
```
- More flexible and modular
- Better for parameter conditioning
- **Note**: Model automatically converts `[B, H, W, 2]` to `[B, 2, H, W]` (channels first) internally

### Forward Pass Flow

```
1. Input: Current velocity field [u_t, v_t]
2. Add mask and case parameters (if at input)
3. Encoder: Extract multi-scale features
4. Inject case parameters at bottleneck (if at hidden)
5. Decoder: Reconstruct with skip connections
6. Output: Predicted velocity increments [Δu, Δv]
7. Residual: Add current state → [u_{t+1}, v_{t+1}]
8. Mask: Zero out cylinder region
9. Return: Next timestep velocities
```

### Why UNet for Fluid Dynamics?

1. **Spatial Structure**: UNet preserves spatial relationships through skip connections
2. **Multi-scale Features**: Encoder captures features at different scales (local vortices, global flow patterns)
3. **Efficiency**: No FFT operations (unlike FNO) - faster on Mac/CPU
4. **Flexibility**: Easy to add conditioning (parameters, masks)
5. **Proven**: Works well for image-to-image tasks, similar to flow field prediction

### Model Size

With default parameters `features=[64, 128, 256, 512]`:
- **Total parameters**: ~31 million
- **Memory**: ~120 MB (model weights)
- **Inference speed**: Fast on GPU/MPS, moderate on CPU

---

## Comparison with Reference CFDBench Implementation

This notebook implementation differs significantly from the standard CFDBench UNet implementation (`unet.py` and `train_auto.py`). Understanding these differences helps explain design choices and trade-offs.

### Model Size Comparison

**Reference Implementation:**
- **Base dimension**: `dim=8` → features scale as `[8, 16, 32, 64, 128]`
- **Total parameters**: ~1-2 million
- **Model size**: ~8-16 MB

**Notebook Implementation:**
- **Base dimension**: `features=[64, 128, 256, 512]` → bottleneck: 1024
- **Total parameters**: ~31 million
- **Model size**: ~120 MB

**Why the difference?**
The notebook uses **8x larger base features** (64 vs 8), which creates a **quadratic scaling** in parameters:
- For Conv2d layers, parameters scale roughly with \(C_{in}\times C_{out}\) (times kernel area).
- If you scale *all* channel widths by a factor \(s\), convolution parameters scale roughly like \(s^2\).
- Here \(s=64/8=8\), so **comparable conv blocks are ~64× larger** in parameter count (and the deepest blocks dominate total params).

**Key layers contributing to size:**
- **Bottleneck `Down` block (`DoubleConv`)**:
  - Notebook `DoubleConv(512→1024)` has ~**14.16M** conv-weight params (two 3×3 convs: \(9·512·1024 + 9·1024·1024\); biases/BatchNorm add only a small extra).
  - Reference `DoubleConv(64→128)` has ~**0.221M** conv-weight params (\(9·64·128 + 9·128·128\)).
- **Up path layers**: Each upsampling layer with larger channels adds millions of parameters
- **Case-params FC (hidden injection)**:
  - Notebook: `Linear(8→1024)` has **9,216** params (\(8·1024 + 1024\)).
  - Reference: `Linear(5→128)` has **768** params (\(5·128 + 128\)).

### Task Difference: Multi-Step vs Single-Step Training

**Reference Implementation:**
- **Training**: Single-step prediction (predicts `t+1` from `t`)
- **Multi-step inference**: Done via `generate_many()` method (separate from training)
- **Loss function**: Standard NMSE/MSE computed inside model
- **Error accumulation**: Not explicitly addressed during training

**Notebook Implementation:**
- **Training**: Multi-step autoregressive rollout (10 steps by default)
- **Autoregressive training**: Model sees its own predictions as inputs during training
- **Loss function**: Custom physics-informed loss (velocity MSE + 0.1 × energy conservation)
- **Error accumulation**: Explicitly trained to handle prediction errors over multiple steps

**Why multi-step training?**
1. **Realistic inference**: Since inference uses autoregressive rollout, training should match this behavior
2. **Error handling**: Model learns to correct and stabilize its own predictions
3. **Long-term stability**: Better performance on 10+ step predictions
4. **Physics consistency**: Energy conservation term helps maintain physical properties

**Trade-off:**
- **Slower training**: 10 forward passes per batch vs 1 (10x slower per iteration)
- **Better long-term predictions**: Model is explicitly trained for multi-step stability

### Same Dataset, Different Approach

Both implementations use the **same CFDBench dataset** but with different data handling:

**Reference Implementation:**
- **Target format**: Single timestep `[B, 2, H, W]` (one step ahead)
- **Case parameters**: 5 parameters (different subset)
- **Data format**: Channels first `[B, C, H, W]`
- **Mask handling**: Extracted from input channels

**Notebook Implementation:**
- **Target format**: Multi-step `[B, K, H, W, 2]` where K=10 (ten steps ahead)
- **Case parameters**: 8 parameters (includes geometry bounds: x_min, x_max, y_min, y_max)
- **Data format**: Channels last `[B, H, W, C]` (converted internally)
- **Mask handling**: Separate input tensor, not extracted from channels

**Dataset usage:**
- Both use the same `.npy` files (`u.npy`, `v.npy`) and `params.json`
- Both perform train/test splits (70/30)
- Notebook uses `max_samples_per_case=200` to reduce dataset size (vs full dataset in reference)

### Summary of Key Differences

| Aspect | Reference | Notebook | Impact |
|--------|-----------|----------|--------|
| **Model size** | 1-2M params | 31M params | 15-30x larger |
| **Training steps** | 1 step | 10 steps | 10x slower training |
| **Loss function** | Standard NMSE | Physics-informed | Better physics consistency |
| **Multi-step** | Inference only | Training + inference | Better long-term stability |
| **Flexibility** | Fixed architecture | Configurable flags | Easier experimentation |
| **Memory** | Low (~8-16 MB) | High (~120 MB) | Requires more GPU memory |
| **Overfitting risk** | Low | High | Needs regularization |



### Recommendations for This Notebook

Given the current setup (10-step rollout, reduced data):
1. **Multi-step training is appropriate**: Matches inference needs
2. **Consider reducing model size**: `features=[32, 64, 128, 256]` (~8M params) balances capacity and efficiency
3. **Increase regularization**: With large model + reduced data, use higher `weight_decay` or dropout
4. **Physics-informed loss is beneficial**: Energy conservation helps long-term stability

---

## Key Hyperparameters for Post-Tuning

### Dataset Parameters

**Location**: Section 4 (Data Preprocessing)

```python
full_dataset = CylinderFlowDataset(
    all_u_data, all_v_data, all_params, masks_list, 
    rollout_steps=10,              # Number of future timesteps to predict
    timestep_stride=1,              # Sample every Nth timestep (1 = all)
    max_samples_per_case=200        # Limit samples per case (None = all)
)
```

- **`rollout_steps`** (class default: 3, current usage: 10)
  - Number of timesteps to predict ahead
  - **Tuning guide**: 
    - Start with 3-5 for initial training
    - Increase to 10+ for longer-term predictions
    - Higher values = more challenging, slower training
    - Watch for error accumulation in longer rollouts

- **`timestep_stride`** (default: 1)
  - Sample every Nth timestep from the data
  - **Tuning guide**:
    - `1` = use all timesteps (maximum data)
    - `2` = use every other timestep (faster training, less data)
    - Higher values reduce dataset size and training time

- **`max_samples_per_case`** (class default: None, current usage: 200)
  - Maximum number of samples per simulation case
  - **Tuning guide**:
    - `None` = use all available samples (~610 per case with 620 timesteps and rollout_steps=10)
    - `200` = reduced dataset for faster training (current configuration)
    - Lower values = faster training but less data
    - Higher values = more data but slower training

### Training Parameters

**Location**: Section 6 (Training Code Block)

```python
batch_size = 16
n_epochs = 50
rollout_steps = 10
learning_rate = 1e-3
weight_decay = 1e-4
```

- **`batch_size`** (default: 16)
  - Number of samples per batch
  - **Tuning guide**:
    - Increase if you have more GPU memory (32, 64)
    - Decrease if running out of memory (8, 4)
    - Larger batches = more stable gradients but slower per iteration

- **`n_epochs`** (default: 50)
  - Number of training epochs
  - **Tuning guide**:
    - Monitor training loss to determine if more epochs needed
    - Use early stopping if loss plateaus
    - Typical range: 30-100 epochs

- **`learning_rate`** (default: 1e-3)
  - Initial learning rate for Adam optimizer
  - **Tuning guide**:
    - Too high: loss may explode or oscillate
    - Too low: slow convergence
    - Try: 5e-4, 1e-3, 2e-3
    - Learning rate scheduler automatically reduces LR on plateau

- **`weight_decay`** (default: 1e-4)
  - L2 regularization strength
  - **Tuning guide**:
    - Increase (1e-3) if overfitting
    - Decrease (1e-5) if underfitting
    - Helps prevent overfitting

### Model Architecture Parameters

**Location**: Section 6 (Training Code Block)

```python
model = UNet(
    in_channels=2,                  # Velocity components (u, v)
    out_channels=2,                 # Output velocity components
    features=[64, 128, 256, 512],  # Encoder/decoder feature sizes
    n_case_params=8,                # Number of case parameters (vel_in, density, viscosity, diameter, x_min, x_max, y_min, y_max)
    insert_case_params_at="hidden", # "input" or "hidden"
    bilinear=False,                 # Upsampling method
    use_residual=True,              # Enable residual connections
    use_output_mask=True            # Apply geometry mask at output
)
```

- **`features`** (default: [64, 128, 256, 512])
  - Feature channel sizes in encoder/decoder
  - **Tuning guide**:
    - Larger: [128, 256, 512, 1024] = more capacity, slower
    - Smaller: [32, 64, 128, 256] = faster, less capacity
    - Adjust based on model size vs. performance tradeoff

- **`insert_case_params_at`** (default: "hidden")
  - Where to inject physical parameters
  - Options: `"input"` or `"hidden"` (bottleneck)
  - **Tuning guide**:
    - `"hidden"` = typically better for conditioning
    - `"input"` = simpler, may work for simpler problems

- **`use_residual`** (default: True)
  - Add input velocities to predictions (predicts increments)
  - **Tuning guide**:
    - `True` = recommended, helps with stability
    - `False` = model predicts absolute values (harder)

- **`use_output_mask`** (default: True)
  - Zero out predictions inside cylinder geometry
  - **Tuning guide**:
    - `True` = recommended, enforces boundary conditions
    - `False` = model must learn boundaries (harder)

- **`bilinear`** (default: False)
  - Use bilinear upsampling vs. transposed convolution
  - **Tuning guide**:
    - `False` = transposed conv (more parameters, sharper)
    - `True` = bilinear (smoother, fewer parameters)

### Loss Function Parameters

**Location**: Section 6 (Training loop)

```python
velocity_error = torch.mean((pred_next - target_k) ** 2)
energy_error = torch.mean((kinetic_energy_pred - kinetic_energy_target) ** 2)
step_loss = velocity_error + 0.1 * energy_error
```

- **Energy weight** (default: 0.1)
  - Weight for kinetic energy conservation term
  - **Tuning guide**:
    - Increase (0.2, 0.5) if energy conservation is important
    - Decrease (0.05) or remove (0.0) if focusing on velocity accuracy
    - Helps enforce physics constraints

### Learning Rate Scheduler

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)
```

- **`factor`** (default: 0.5)
  - Multiply LR by this when plateau detected
  - **Tuning guide**: 0.5 = halve LR, 0.1 = reduce more aggressively

- **`patience`** (default: 10)
  - Epochs to wait before reducing LR
  - **Tuning guide**: Lower (5) = reduce LR faster, Higher (15) = wait longer

---

## Training Process

### How Time Evolution Works

1. **Single-step prediction**: Model takes current velocity field `[u_t, v_t]` and predicts `[u_{t+1}, v_{t+1}]`

2. **Autoregressive rollout**: For `rollout_steps=10`:
   ```
   t=0:  [u_0, v_0] → model → [u_1_pred, v_1_pred]
   t=1:  [u_1_pred, v_1_pred] → model → [u_2_pred, v_2_pred]
   ...
   t=9:  [u_9_pred, v_9_pred] → model → [u_10_pred, v_10_pred]
   ```

3. **Loss computation**: Loss is averaged over all rollout steps

4. **Residual connection**: Model predicts increments, then adds current state:
   ```
   u_{t+1} = u_t + Δu_predicted
   ```

### Training Tips

1. **Monitor training loss**: Should decrease steadily
   - If loss explodes: reduce learning rate or use gradient clipping (already implemented)
   - If loss plateaus: increase learning rate or model capacity

2. **Checkpointing**: 
   - Best model saved to `checkpoints/best_model.pth`
   - Latest model saved to `checkpoints/latest_model.pth`
   - Periodic checkpoints every 5 epochs

3. **Gradient clipping**: Already set to `max_norm=1.0` to prevent explosions

4. **Early stopping**: Implemented for catastrophic failures (loss > 1000)

---

## Evaluation

### Evaluation Metrics

**Location**: Section 7 (Evaluation Code Block)

The evaluation computes three metrics:

1. **L2 Error**: Mean squared error between predicted and ground truth velocities
   ```python
   l2_error = torch.mean((pred - target) ** 2)
   ```

2. **Kinetic Energy Drift**: Mean absolute difference in kinetic energy
   ```python
   ke_drift = mean(|KE_predicted - KE_ground_truth|)
   ```

3. **Relative Error**: Mean relative error with numerical stability
   ```python
   relative_error = mean(|pred - target| / (|target| + 1e-6))
   ```
   - The `1e-6` term prevents division by zero

### Running Evaluation

1. Load a trained model checkpoint:
   ```python
   checkpoint = torch.load('checkpoints/best_model.pth')
   model.load_state_dict(checkpoint['model_state_dict'])
   ```

2. Run evaluation loop (automatically uses `eval_dataset`)

3. Metrics are averaged over all evaluation samples and rollout steps

### Visualization

**Location**: Section 8 (Visualization)

The visualization shows:
- **Row 0**: Predicted and Ground Truth for t+1 (u and v components)
- **Row 1**: Predicted and Ground Truth for t+rollout_steps (u and v components) - currently t+10
- **Row 2**: **Differences** (Prediction - Ground Truth) for t+1 and t+rollout_steps

**Color scheme**:
- Red: Positive differences (prediction > ground truth)
- Blue: Negative differences (prediction < ground truth)
- White: Near-zero differences (good predictions)

---

## Post-Tuning Guide

### Step-by-Step Tuning Process

1. **Start with baseline**:
   - Use default parameters
   - Train for a few epochs to establish baseline performance

2. **Adjust data parameters**:
   - If training is slow: reduce `max_samples_per_case` or increase `timestep_stride`
   - If you need longer predictions: increase `rollout_steps` gradually (3 → 5 → 10)

3. **Tune model capacity**:
   - If underfitting: increase `features` sizes
   - If overfitting: add more `weight_decay` or reduce model size

4. **Optimize training**:
   - Adjust `learning_rate` based on loss behavior
   - Modify `batch_size` based on available memory
   - Adjust `n_epochs` based on convergence

5. **Fine-tune loss**:
   - Adjust energy weight if energy conservation is important
   - Try different loss combinations

### Common Tuning Scenarios

**Scenario 1: Training too slow**
- Reduce `max_samples_per_case` (e.g., 200 → 100)
- Increase `timestep_stride` (e.g., 1 → 2)
- Reduce `batch_size` if memory constrained
- Reduce `rollout_steps` during initial training

**Scenario 2: Model not learning**
- Increase `learning_rate` (e.g., 1e-3 → 2e-3)
- Check if data is loading correctly
- Verify model architecture is correct
- Try removing energy loss term (set weight to 0)

**Scenario 3: Overfitting**
- Increase `weight_decay` (e.g., 1e-4 → 1e-3)
- Reduce model capacity (`features`)
- Use more training data (increase `max_samples_per_case`)

**Scenario 4: Poor long-term predictions**
- Start with shorter `rollout_steps` (3-5) and gradually increase
- Increase energy loss weight to enforce physics
- Check if residual connections are enabled
- Verify output masking is working

**Scenario 5: Unstable training**
- Reduce `learning_rate`
- Gradient clipping is already enabled (max_norm=1.0)
- Check for NaN/Inf in loss (already monitored)
- Try smaller `batch_size`

---

## Important Notes

### Data Handling

- **Time sequence**: The dataset uses implicit time ordering - the first dimension of data arrays is time
- **70/30 split**: Applied at the **sample level**, not case level
  - Samples from the same case can be in both train and eval sets
  - This is intentional for better generalization

### Model Behavior

- **One-step predictor**: The model architecture itself only predicts one step ahead
- **Multi-step via autoregression**: Longer predictions are achieved by iteratively applying the model
- **Error accumulation**: Errors compound over longer rollouts - this is expected

### Checkpointing

- Models are saved in `checkpoints/` directory
- `best_model.pth`: Best model (only saved if evaluation is run with eval_loss provided)
- `latest_model.pth`: Most recent checkpoint (saved every epoch)
- `checkpoint_epoch_XXXX.pth`: Periodic checkpoints every 5 epochs
- **Note**: During training, `best_model.pth` may not be created since `eval_loss=None`. Use `latest_model.pth` or periodic checkpoints.

### Device Configuration

- Automatically detects CUDA if available
- Falls back to CPU if no GPU
- Uses `pin_memory=True` for GPU to speed up data loading

---

## Troubleshooting

### Issue: Loss becomes NaN or Inf
- **Solution**: Reduce learning rate, check data for invalid values
- Already handled: Code stops training if catastrophic loss detected

### Issue: Out of memory errors
- **Solution**: Reduce `batch_size`, reduce `max_samples_per_case`, or use gradient accumulation

### Issue: Predictions diverge over long rollouts
- **Solution**: This is expected - try shorter `rollout_steps` or increase energy loss weight

### Issue: Model predictions are all zeros
- **Solution**: Check if output masking is too aggressive, verify model is actually training (loss decreasing)

### Issue: Training is very slow
- **Solution**: Reduce dataset size (`max_samples_per_case`, `timestep_stride`), reduce `rollout_steps`, use smaller model

---

## File Structure

```
src/data_handle/
├── UNet.ipynb                    # Main training/evaluation notebook
├── UNet_DOCUMENTATION.md        # This file
├── checkpoints/                 # Saved model checkpoints
│   ├── best_model.pth
│   ├── latest_model.pth
│   └── checkpoint_epoch_XXXX.pth
└── data/                        # CFDBench dataset
```

---

## References

- **Dataset**: CFDBench - 2D Cylinder Flow dataset
- **Architecture**: UNet with residual connections and parameter conditioning
- **Task**: Neural time integrator for Navier-Stokes equations

---

## Quick Reference: Key Parameters

| Parameter | Location | Default | Tuning Range |
|-----------|----------|---------|--------------|
| `rollout_steps` | Dataset creation | 10 (current) | 3-20 |
| `max_samples_per_case` | Dataset creation | 200 (current) | 50-600 |
| `batch_size` | Training | 16 | 4-64 |
| `learning_rate` | Training | 1e-3 | 5e-4 to 2e-3 |
| `n_epochs` | Training | 50 | 30-100 |
| `features` | Model | [64,128,256,512] | Varies |
| Energy weight | Loss function | 0.1 | 0.0-0.5 |

---

**Last Updated**: Based on notebook with 10-step rollout and reduced dataset configuration.
