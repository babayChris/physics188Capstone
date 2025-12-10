import numpy as np
from pathlib import Path
import json
from torch.utils.data import Dataset
import torch
import sys


class Data(Dataset):
    def __init__(self, geometry: str = 'cylinder', rollout_steps: int = 5):
        if geometry not in ['cylinder', 'cavity', 'dam', 'tube']:
            print('enter valid geometry')
            sys.exit()
        
        self.rollout_steps = rollout_steps
        file_dir = Path(__file__).parent
        self.path = file_dir / '../../data/CFDBench' / geometry
        
        self.case_data = []  
        self.samples = [] 
        
        self._load_data()

    def _load_data(self):
        for type_dir in self.path.iterdir():
            if not type_dir.is_dir():
                continue
            
            for case_dir in type_dir.iterdir():
                if not case_dir.is_dir():
                    continue
                
                u_path = case_dir / 'u.npy'
                v_path = case_dir / 'v.npy'
                
                if not u_path.exists() or not v_path.exists():
                    continue
                
                u = np.load(u_path)
                v = np.load(v_path)
                metadata = self._extract_metadata(case_dir)
                
                if metadata is None:
                    continue
                
                # Store case data once
                case_idx = len(self.case_data)
                self.case_data.append((u, v, metadata))
                
                # Create sample indices
                T = u.shape[0]
                for t in range(T - self.rollout_steps):
                    self.samples.append((case_idx, t))
        
        print(f"Loaded {len(self.case_data)} cases, {len(self.samples)} samples")

    def __getitem__(self, index):
        case_idx, t = self.samples[index]
        u, v, metadata = self.case_data[case_idx]
        
        uv_t = np.stack([u[t], v[t]], axis=-1)
        x = np.concatenate([uv_t, metadata], axis=-1).astype(np.float32)
        
        target_u = u[t+1 : t+1+self.rollout_steps]
        target_v = v[t+1 : t+1+self.rollout_steps]
        y = np.stack([target_u, target_v], axis=-1).astype(np.float32)
        
        return torch.from_numpy(x), torch.from_numpy(y)
    
    def __len__(self):
        return len(self.samples)
    
    def _extract_metadata(self, case_dir):
        """
        Returns: [H, W, 4] channels-last
        """
        try:
            with open(case_dir / 'case.json', 'r') as f:
                raw = json.load(f)

            return np.stack([
                np.full((64, 64), raw['vel_in']),
                np.full((64, 64), raw['density']),
                np.full((64, 64), raw['viscosity']),
                np.full((64, 64), raw['radius']),
                np.full((64, 64), raw['x_min']),
                np.full((64, 64), raw['x_max']),
                np.full((64, 64), raw['y_min']),
                np.full((64, 64), raw['y_max'])
            ], axis=-1).astype(np.float32)  # [64, 64, 8]
                
        except (json.JSONDecodeError, KeyError, IOError) as e:
            print(f"Error loading {case_dir}: {e}")
            return None