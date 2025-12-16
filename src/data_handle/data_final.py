import numpy as np
from pathlib import Path
import json
from torch.utils.data import Dataset, Subset
import torch
from typing import Optional
from functools import lru_cache


class CFDBenchDataset(Dataset):
    SUPPORTED_GEOMETRIES = ['cylinder', 'cavity', 'dam', 'tube']
    def __init__(
        self, 
        data_root: str | Path,
        geometries: list[str] | str = 'all',
        rollout_steps: int = 1,
        grid_size: int = 64,
        normalize: bool = True,
        cache_size: int = 32  # Number of cases to keep in memory
    ):
        self.data_root = Path(data_root)
        self.rollout_steps = rollout_steps
        self.grid_size = grid_size
        self.normalize = normalize
        self.cache_size = cache_size
        
        # Handle geometry selection
        if geometries == 'all':
            self.geometries = self.SUPPORTED_GEOMETRIES
        elif isinstance(geometries, str):
            self.geometries = [geometries]
        else:
            self.geometries = geometries
        
        for g in self.geometries:
            if g not in self.SUPPORTED_GEOMETRIES:
                raise ValueError(f"Unknown geometry: {g}. Supported: {self.SUPPORTED_GEOMETRIES}")
        
        # Storage - only paths and metadata, NOT actual arrays
        self.case_info = []  # {path, geometry, metadata, n_timesteps}
        self.samples = []    # (case_idx, timestep)
        
        # Normalization stats
        self.stats = {
            'u_mean': 0.0, 'u_std': 1.0,
            'v_mean': 0.0, 'v_std': 1.0,
            'sdf_max': 1.0,
        }
        
        self._index_all_data()
        
        if self.normalize:
            self._compute_normalization_stats_streaming()
    
    def _index_all_data(self):
        """Index all cases without loading arrays into memory."""
        for geometry in self.geometries:
            geom_path = self.data_root / geometry
            if not geom_path.exists():
                print(f"Warning: {geom_path} does not exist, skipping")
                continue
            
            self._index_geometry(geometry, geom_path)
        
        print(f"Indexed {len(self.case_info)} cases, {len(self.samples)} samples")
    
    def _index_geometry(self, geometry: str, geom_path: Path):
        """Index all cases for a specific geometry type."""
        case_dirs = []
        
        for item in sorted(geom_path.iterdir()):
            if item.is_dir():
                if (item / 'case.json').exists():
                    case_dirs.append(item)
                else:
                    # Subdirectory (bc/geo/prop)
                    for case_dir in sorted(item.iterdir()):
                        if case_dir.is_dir() and (case_dir / 'case.json').exists():
                            case_dirs.append(case_dir)
        
        for case_dir in case_dirs:
            self._index_case(geometry, case_dir)
    
    def _index_case(self, geometry: str, case_dir: Path):
        """Index a single case - store paths only."""
        u_path = case_dir / 'u.npy'
        v_path = case_dir / 'v.npy'
        json_path = case_dir / 'case.json'
        
        if not all(p.exists() for p in [u_path, v_path, json_path]):
            return
        
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            u_mmap = np.load(u_path, mmap_mode='r')
            n_timesteps = u_mmap.shape[0]
            del u_mmap  # Release memory map
            
            # Store case info
            case_idx = len(self.case_info)
            self.case_info.append({
                'case_dir': case_dir,
                'geometry': geometry,
                'metadata': metadata,
                'n_timesteps': n_timesteps,
                'case_id': case_dir.name
            })
            
            # Create sample indices
            for t in range(n_timesteps - self.rollout_steps):
                self.samples.append((case_idx, t))
                
        except Exception as e:
            print(f"Error indexing {case_dir}: {e}")
    
    #NOTE: Lru_cache stores output of the function below so if the same case_idx is loaded, the ouptu is taken from the cache and the function is not run.  This optimized performance.
    @lru_cache(maxsize=32)
    def _load_case(self, case_idx: int):
        """Load case data with LRU caching."""
        info = self.case_info[case_idx]
        case_dir = info['case_dir']
        
        u = np.load(case_dir / 'u.npy').astype(np.float32)
        v = np.load(case_dir / 'v.npy').astype(np.float32)
        sdf = self._compute_sdf(info['geometry'], info['metadata'])
        params = self._extract_params(info['geometry'], info['metadata'])
        
        return u, v, sdf, params
    
    def _compute_normalization_stats_streaming(self):
        """Compute stats by streaming through data (memory-efficient)."""
        print("Computing normalization stats (streaming)...")
        n = 0
        u_mean, u_m2 = 0.0, 0.0
        v_mean, v_m2 = 0.0, 0.0
        sdf_max = 0.0
        sample_cases = min(50, len(self.case_info))
        case_indices = np.random.choice(len(self.case_info), sample_cases, replace=False)
        
        for case_idx in case_indices:
            info = self.case_info[case_idx]
            case_dir = info['case_dir']

            u = np.load(case_dir / 'u.npy', mmap_mode='r')
            v = np.load(case_dir / 'v.npy', mmap_mode='r')
 
            for frame in range(0, u.shape[0], 10):  # Sample every 10th frame
                u_flat = u[frame].flatten().astype(np.float64)
                v_flat = v[frame].flatten().astype(np.float64)
                
                for val in u_flat:
                    n += 1
                    delta = val - u_mean
                    u_mean += delta / n
                    u_m2 += delta * (val - u_mean)
                
                for val in v_flat:
                    delta = val - v_mean
                    v_mean += delta / n
                    v_m2 += delta * (val - v_mean)

            sdf = self._compute_sdf(info['geometry'], info['metadata'])
            sdf_max = max(sdf_max, np.abs(sdf).max())
            
            del u, v  # Release memory map
        
        self.stats['u_mean'] = float(u_mean)
        self.stats['u_std'] = float(np.sqrt(u_m2 / n)) + 1e-8
        self.stats['v_mean'] = float(v_mean)
        self.stats['v_std'] = float(np.sqrt(v_m2 / n)) + 1e-8
        self.stats['sdf_max'] = float(sdf_max) + 1e-8
        
        print(f"Stats: u={self.stats['u_mean']:.4f}±{self.stats['u_std']:.4f}, "
              f"v={self.stats['v_mean']:.4f}±{self.stats['v_std']:.4f}")
    
    def _compute_sdf(self, geometry: str, meta: dict) -> np.ndarray:
        """Compute Signed Distance Function for the geometry."""
        H, W = self.grid_size, self.grid_size
        
        if geometry == 'cylinder':
            return self._sdf_cylinder(meta, H, W)
        elif geometry == 'cavity':
            return self._sdf_cavity(meta, H, W)
        elif geometry == 'dam':
            return self._sdf_dam(meta, H, W)
        elif geometry == 'tube':
            return self._sdf_tube(meta, H, W)
        else:
            return np.zeros((H, W), dtype=np.float32)
    
    def _sdf_cylinder(self, meta: dict, H: int, W: int) -> np.ndarray:
        """SDF for flow past cylinder centered at origin."""
        x_min, x_max = meta['x_min'], meta['x_max']
        y_min, y_max = meta['y_min'], meta['y_max']
        radius = meta['radius']
        
        x = np.linspace(x_min, x_max, W)
        y = np.linspace(y_min, y_max, H)
        X, Y = np.meshgrid(x, y, indexing='xy')
        
        # Cylinder at origin
        sdf = np.sqrt(X**2 + Y**2) - radius
        
        # Normalize by domain size
        domain_scale = max(x_max - x_min, y_max - y_min)
        sdf = sdf / domain_scale
        
        return sdf.astype(np.float32)
    
    def _sdf_cavity(self, meta: dict, H: int, W: int) -> np.ndarray:
        """SDF for lid-driven cavity."""
        x = np.linspace(0, 1, W)
        y = np.linspace(0, 1, H)
        X, Y = np.meshgrid(x, y, indexing='xy')
        
        # Distance to nearest wall
        sdf = np.minimum(
            np.minimum(X, 1 - X),
            np.minimum(Y, 1 - Y)
        )
        
        if meta.get('rotated', False):
            sdf = np.rot90(sdf)
        
        return sdf.astype(np.float32)
    
    def _sdf_dam(self, meta: dict, H: int, W: int) -> np.ndarray:
        """SDF for dam break with barrier."""
        barrier_height = meta['barrier_height']
        barrier_width = meta['barrier_width']
        domain_height = meta['height']
        domain_width = meta['width']
        
        x = np.linspace(0, 1, W)
        y = np.linspace(0, 1, H)
        X, Y = np.meshgrid(x, y, indexing='xy')
        
        # Normalized barrier dimensions
        bh = barrier_height / domain_height
        bw = barrier_width / domain_width
        
        # Barrier centered at bottom
        barrier_x_start = 0.5 - bw / 2
        barrier_x_end = 0.5 + bw / 2
        
        # Inside barrier check
        inside_x = (X >= barrier_x_start) & (X <= barrier_x_end)
        inside_y = Y <= bh
        inside_barrier = inside_x & inside_y
        
        # Simple SDF approximation
        sdf = np.ones((H, W), dtype=np.float32) * 0.5
        sdf[inside_barrier] = -0.1
        
        return sdf.astype(np.float32)
    
    def _sdf_tube(self, meta: dict, H: int, W: int) -> np.ndarray:
        """SDF for tube/channel flow."""
        y = np.linspace(0, 1, H)
        Y = np.broadcast_to(y[:, None], (H, W))
        
        # Distance to top and bottom walls
        sdf = np.minimum(Y, 1 - Y)
        
        return sdf.astype(np.float32)
    
    def _extract_params(self, geometry: str, meta: dict) -> np.ndarray:
        """Extract physical parameters as spatial fields."""
        H, W = self.grid_size, self.grid_size
        params = []
        
        # Density (log scale)
        rho = meta.get('density', 1.0)
        params.append(np.full((H, W), np.log10(rho + 1e-8), dtype=np.float32))
        
        # Viscosity (log scale)
        mu = meta.get('viscosity', 0.01)
        params.append(np.full((H, W), np.log10(mu + 1e-10), dtype=np.float32))
        
        # Boundary velocity
        if geometry == 'cylinder':
            vel_bc = meta.get('vel_in', 0.0)
        elif geometry == 'cavity':
            vel_bc = meta.get('vel_top', 0.0)
        elif geometry == 'dam':
            vel_bc = meta.get('velocity', 0.0)
        elif geometry == 'tube':
            vel_bc = meta.get('vel_in', 0.0)
        else:
            vel_bc = 0.0
        params.append(np.full((H, W), vel_bc, dtype=np.float32))
        
        # Geometry one-hot encoding
        geom_idx = self.SUPPORTED_GEOMETRIES.index(geometry)
        for i in range(len(self.SUPPORTED_GEOMETRIES)):
            val = 1.0 if i == geom_idx else 0.0
            params.append(np.full((H, W), val, dtype=np.float32))
        
        return np.stack(params, axis=-1)  # [H, W, 7]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        case_idx, t = self.samples[index]
        
        # Load case (from preload or cache)
        u, v, sdf, params = self._get_case(case_idx)
        
        # Get current timestep
        u_t = u[t].copy()
        v_t = v[t].copy()
        
        # Normalize
        if self.normalize:
            u_t = (u_t - self.stats['u_mean']) / self.stats['u_std']
            v_t = (v_t - self.stats['v_mean']) / self.stats['v_std']
            sdf_norm = sdf / self.stats['sdf_max']
        else:
            sdf_norm = sdf
        
        # Stack input: [u, v, sdf, params...] -> [H, W, 10]
        x = np.concatenate([
            u_t[..., None],
            v_t[..., None],
            sdf_norm[..., None],
            params
        ], axis=-1).astype(np.float32)
        
        # Target
        target_u = u[t+1 : t+1+self.rollout_steps].copy()
        target_v = v[t+1 : t+1+self.rollout_steps].copy()
        
        if self.normalize:
            target_u = (target_u - self.stats['u_mean']) / self.stats['u_std']
            target_v = (target_v - self.stats['v_mean']) / self.stats['v_std']
        
        y = np.stack([target_u, target_v], axis=-1).astype(np.float32)
        
        if self.rollout_steps == 1:
            y = y.squeeze(0)
        
        return torch.from_numpy(x), torch.from_numpy(y)
    
    def clear_cache(self):
        """Clear the LRU cache to free memory."""
        self._load_case.cache_clear()
    
    def preload_all(self):
        """Load all data into RAM for fast training."""
        print(f"Preloading {len(self.case_info)} cases into RAM...")
        self._preloaded = {}
        for case_idx in range(len(self.case_info)):
            # Bypass LRU cache, store directly
            info = self.case_info[case_idx]
            case_dir = info['case_dir']
            
            u = np.load(case_dir / 'u.npy').astype(np.float32)
            v = np.load(case_dir / 'v.npy').astype(np.float32)
            sdf = self._compute_sdf(info['geometry'], info['metadata'])
            params = self._extract_params(info['geometry'], info['metadata'])
            
            self._preloaded[case_idx] = (u, v, sdf, params)
            
            if (case_idx + 1) % 20 == 0:
                print(f"  Loaded {case_idx + 1}/{len(self.case_info)} cases")
        
        print(f"Preload complete. Estimated RAM usage: {self._estimate_ram_usage():.1f} GB")
    
    def _estimate_ram_usage(self):
        """Estimate RAM usage in GB."""
        if not hasattr(self, '_preloaded') or not self._preloaded:
            return 0
        
        total_bytes = 0
        for u, v, sdf, params in self._preloaded.values():
            total_bytes += u.nbytes + v.nbytes + sdf.nbytes + params.nbytes
        
        return total_bytes / (1024**3)
    
    def _get_case(self, case_idx: int):
        """Get case from preloaded cache or load lazily."""
        if hasattr(self, '_preloaded') and case_idx in self._preloaded:
            return self._preloaded[case_idx]
        return self._load_case(case_idx)
    
    def get_geometry_indices(self, geometry: str) -> list[int]:
        """Get sample indices for a specific geometry."""
        indices = []
        for i, (case_idx, t) in enumerate(self.samples):
            if self.case_info[case_idx]['geometry'] == geometry:
                indices.append(i)
        return indices
    
    def get_geometry_breakdown(self) -> dict:
        """Get count of samples per geometry."""
        breakdown = {g: 0 for g in self.SUPPORTED_GEOMETRIES}
        for case_idx, t in self.samples:
            geom = self.case_info[case_idx]['geometry']
            breakdown[geom] += 1
        return breakdown
    
    def set_stats(self, stats: dict):
        """Set normalization stats externally."""
        self.stats = stats
    
    def denormalize(self, u: torch.Tensor, v: torch.Tensor):
        """Denormalize predictions."""
        u = u * self.stats['u_std'] + self.stats['u_mean']
        v = v * self.stats['v_std'] + self.stats['v_mean']
        return u, v


def stratified_split(
    dataset: CFDBenchDataset,
    val_fraction: float = 0.15,
    seed: int = 42
) -> tuple[Subset, Subset]:
    """
    Stratified split by case (not sample)
    Ensures each geometry is represented in both splits.
    """
    rng = np.random.default_rng(seed)
    
    train_indices = []
    val_indices = []
    
    # Group cases by geometry
    geom_to_cases = {g: [] for g in dataset.SUPPORTED_GEOMETRIES}
    for case_idx, info in enumerate(dataset.case_info):
        geom_to_cases[info['geometry']].append(case_idx)
    
    # Split each geometry
    val_cases = set()
    for geom, case_list in geom_to_cases.items():
        if len(case_list) == 0:
            continue
        
        case_arr = np.array(case_list)
        rng.shuffle(case_arr)
        
        n_val = max(1, int(len(case_arr) * val_fraction))
        val_cases.update(case_arr[:n_val])
    
    # Assign samples to train/val based on case
    for i, (case_idx, t) in enumerate(dataset.samples):
        if case_idx in val_cases:
            val_indices.append(i)
        else:
            train_indices.append(i)
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices)