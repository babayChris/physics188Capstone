from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import random
import torch
import numpy as np
import json

# --- 1. Placeholder/Base Classes (Required for the function definition) ---

# Mock base classes - assume full implementation is elsewhere
class CfdDataset: pass
class CfdAutoDataset: pass

# Mock implementation of the CylinderFlowAutoDataset class
class CylinderFlowAutoDataset(CfdAutoDataset):
    data_delta_time = 0.001
    
    def __init__(self, case_dirs: List[Path], split: str, **kwargs):
        self.case_dirs = case_dirs
        self.split = split
        # Mock parameter to easily display which cases were loaded
        self.case_names = [p.name for p in case_dirs]
        
        # Load data is mocked to avoid file system dependency in this example
        self.time_step_size = int(kwargs.get('delta_time', 0.01) / self.data_delta_time)
        self.load_data(case_dirs, self.time_step_size)

    def load_data(self, case_dirs, time_step_size: int):
        print(f"  [Loader]: Processing data for split '{self.split}' from {len(case_dirs)} case(s).")
        
        if not case_dirs:
            raise ValueError("No case directories provided")
        
        # Initialize lists to store data for all cases
        self.u_data = []
        self.v_data = []
        self.case_params = []  # Store JSON metadata for each case
        
        # Use first 620 timesteps for consistency across all cases
        max_timesteps = 620
        
        # Process all cases
        for case_dir in case_dirs:
            u_path = case_dir / "u.npy"
            v_path = case_dir / "v.npy"
            
            # Check if files exist
            if not u_path.exists():
                raise FileNotFoundError(f"u.npy not found in {case_dir}")
            if not v_path.exists():
                raise FileNotFoundError(f"v.npy not found in {case_dir}")
            
            # Load data
            u_data = np.load(u_path)
            v_data = np.load(v_path)
            
            # Validate u and v have matching shapes
            if u_data.shape != v_data.shape:
                raise ValueError(
                    f"Shape mismatch in {case_dir}: u.npy shape {u_data.shape} != v.npy shape {v_data.shape}"
                )
            
            # Truncate to first max_timesteps if needed
            original_timesteps = u_data.shape[0]
            if original_timesteps > max_timesteps:
                u_data = u_data[:max_timesteps]
                v_data = v_data[:max_timesteps]
                print(f"    ⚠ {case_dir.name}: Truncated from {original_timesteps} to {max_timesteps} timesteps")
            elif original_timesteps < max_timesteps:
                print(f"    ⚠ {case_dir.name}: Only {original_timesteps} timesteps available (less than {max_timesteps})")
            
            # Ensure spatial dimensions match expected (64x64)
            if len(u_data.shape) == 3 and u_data.shape[1:] != (64, 64):
                raise ValueError(
                    f"Spatial dimension mismatch in {case_dir}: expected (T, 64, 64), got {u_data.shape}"
                )
            
            # Store the data in memory
            self.u_data.append(u_data)
            self.v_data.append(v_data)
            
            # Load JSON metadata
            json_path = case_dir / "case.json"
            if json_path.exists():
                with open(json_path, 'r') as f:
                    case_params = json.load(f)
                self.case_params.append(case_params)
            else:
                raise FileNotFoundError(f"case.json not found in {case_dir}")
            
            print(f"    ✓ {case_dir.name}: u.npy shape {u_data.shape}, v.npy shape {v_data.shape}, params: {case_params}")
        
        # Store detected shape (using actual shape from first case after truncation)
        if self.u_data:
            self.data_shape = self.u_data[0].shape
            print(f"  [Loader]: Using data shape: {self.data_shape} (time_steps={self.data_shape[0]}, spatial={self.data_shape[1]}x{self.data_shape[2]})")
        
        # Mock tensors/params to fulfill internal requirements
        self.inputs = torch.randn(10, 3, 64, 64) 
        self.labels = torch.randn(10, 3, 64, 64) 
        self.case_ids = [0] * 10


# --- 2. The Modified Dataset Loading Function ---

def get_cylinder_auto_datasets(
    data_dir: Path,
    subset_name: str,
    norm_props: bool,
    norm_bc: bool,
    delta_time: float = 0.01,
    stable_state_diff: float = 0.001,
    seed: int = 0,
    load_splits: List[str] = ["train", "dev", "test"],
    case_fraction: float = 1.0, 
) -> Tuple[
    Optional[CylinderFlowAutoDataset],
    Optional[CylinderFlowAutoDataset],
    Optional[CylinderFlowAutoDataset],
]:
    # Check if data directory exists
    data_dir = Path(data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}. Please check the path.")
    
    print(f"\nSearching in subset: {subset_name}")
    print(f"Data directory: {data_dir}")
    
    case_dirs = []
    
    # Case collection: only loads cases from the specified subset_name (e.g., 'bc')
    for name in ["prop", "bc", "geo"]:
        if name == subset_name:
            case_dir = data_dir / name
            if not case_dir.exists():
                raise FileNotFoundError(f"Subset directory not found: {case_dir}. Check data_dir path.")
            # Glob all cases and sort them numerically
            this_case_dirs = sorted(
                case_dir.glob("case*"), key=lambda x: int(x.name[4:])
            )
            case_dirs += this_case_dirs

    assert case_dirs, f"No cases found in {data_dir}/{subset_name}. Check data_dir path."

    random.seed(seed)
    random.shuffle(case_dirs)

    # 🌟 LOGIC TO SELECT FRACTION OF CASES
    num_cases_total = len(case_dirs)
    if case_fraction < 1.0:
        num_cases_to_use = int(num_cases_total * case_fraction)
        
        # Ensure at least one case is selected
        if num_cases_to_use == 0 and num_cases_total > 0:
            num_cases_to_use = 1 

        case_dirs = case_dirs[:num_cases_to_use]
        print(f"Loading partial data: Using {num_cases_to_use} out of {num_cases_total} total cases ({case_fraction*100:.2f}%)")

    # Split into train, dev, test
    num_cases = len(case_dirs)
    num_train = int(num_cases * 0.8)
    num_dev = int(num_cases * 0.1)
    
    # Adjust split for single case load (forces 1 Train, 0 Dev, 0 Test)
    if num_cases == 1:
        num_train, num_dev = 1, 0
    elif num_cases >= 2:
        if num_train + num_dev == num_cases: 
            num_dev -= 1
        elif num_train == num_cases: 
            num_train -= 1

    train_case_dirs = case_dirs[:num_train]
    dev_case_dirs = case_dirs[num_train : num_train + num_dev]
    test_case_dirs = case_dirs[num_train + num_dev :]
    
    print("==== Number of cases in different splits (Subset) ====")
    print(
        f"train: {len(train_case_dirs)}, "
        f"dev: {len(dev_case_dirs)}, "
        f"test: {len(test_case_dirs)}"
    )
    print("=============================================")
    
    kwargs: dict[str, Any] = dict(
        delta_time=delta_time,
        stable_state_diff=stable_state_diff,
        norm_props=norm_props,
        norm_bc=norm_bc,
        cache_dir=Path("../../..", subset_name),
    )
    
    train_data, dev_data, test_data = None, None, None
    
    if len(train_case_dirs) > 0 and "train" in load_splits:
        train_data = CylinderFlowAutoDataset(train_case_dirs, split="train", **kwargs)
    if len(dev_case_dirs) > 0 and "dev" in load_splits:
        dev_data = CylinderFlowAutoDataset(dev_case_dirs, split="dev", **kwargs)
    if len(test_case_dirs) > 0 and "test" in load_splits:
        test_data = CylinderFlowAutoDataset(test_case_dirs, split="test", **kwargs)
        
    return train_data, dev_data, test_data


# --- 3. Execution (The three calls you requested) ---

# ⚠️ IMPORTANT: Update this path to your actual CFDBench data directory
DATA_DIR = Path("../../../")
MIN_FRACTION = 0.01 # Forces the selection of a single case

# 1. Load one case from the Boundary Condition (bc) subset
bc_train_data, _, _ = get_cylinder_auto_datasets(
    data_dir=DATA_DIR,
    subset_name="bc", 
    norm_props=True,
    norm_bc=True,
    case_fraction=MIN_FRACTION 
)

# 2. Load one case from the Geometry (geo) subset
geo_train_data, _, _ = get_cylinder_auto_datasets(
    data_dir=DATA_DIR,
    subset_name="geo", 
    norm_props=True,
    norm_bc=True,
    case_fraction=MIN_FRACTION 
)

# 3. Load one case from the Property (prop) subset
prop_train_data, _, _ = get_cylinder_auto_datasets(
    data_dir=DATA_DIR,
    subset_name="prop", 
    norm_props=True,
    norm_bc=True,
    case_fraction=MIN_FRACTION 
)

# Print u and v matrices for the first case
if bc_train_data and hasattr(bc_train_data, 'u_data') and len(bc_train_data.u_data) > 0:
    case_idx = 0
    u_matrix = bc_train_data.u_data[case_idx]  # Shape: (time_steps, 64, 64)
    v_matrix = bc_train_data.v_data[case_idx]  # Shape: (time_steps, 64, 64)
    
    print(f"\n=== U and V Matrices for {bc_train_data.case_names[case_idx]} ===")
    print(f"U matrix shape: {u_matrix.shape}")
    print(f"V matrix shape: {v_matrix.shape}")
    
    # Print a specific timestep (e.g., timestep 0)
    timestep = 0
    print(f"\nU matrix at timestep {timestep} (first 10x10):")
    print(u_matrix[timestep][:10, :10])  # Print first 10x10 for readability
    
    print(f"\nV matrix at timestep {timestep} (first 10x10):")
    print(v_matrix[timestep][:10, :10])  # Print first 10x10 for readability
    
    # Print summary statistics
    print(f"\nU matrix statistics:")
    print(f"  Min: {u_matrix.min():.6f}, Max: {u_matrix.max():.6f}, Mean: {u_matrix.mean():.6f}")
    print(f"\nV matrix statistics:")
    print(f"  Min: {v_matrix.min():.6f}, Max: {v_matrix.max():.6f}, Mean: {v_matrix.mean():.6f}")
else:
    print("bc_train_data not available or data not loaded")

# --- Summary Output ---
print("\n--- Summary of Loaded Datasets ---")
print("These are minimal datasets, each containing one case trajectory.")
print(f"BC Subset case: {bc_train_data.case_names if bc_train_data else 'None'}")
print(f"GEO Subset case: {geo_train_data.case_names if geo_train_data else 'None'}")
print(f"PROP Subset case: {prop_train_data.case_names if prop_train_data else 'None'}")