import numpy as np
from pathlib import Path
import json
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, geometry : str = 'cylinder'):
        if geometry not in ['cylinder', 'cavity', 'dam', 'tube']:
            print('enter valid geometry')
            sys.exit()
        self.samples = []
        self.path = Path(f'../../data/CFDbench/{geometry}')
        self.data = self._load_data()
        
        self.data_type_split = {'bc': 0, 'geo' : 0, 'prop' : 0}

    def __len__(self):
        return len(self.samples)

    def _load_data(self):
        """
            returns [num samples, 6, 64, 64]
        """
        for i, type in enumerate(self.path.iterdir()):
            #log where the types are
            self.data_type_split[type.name] = i
            type_path = Path(f'{self.path}/{type}')
            for case in type_path.iterdir():
                case_dir = f'{type_path}/{case}'
                #check dir
                if not os.path.exists(case_dir):
                    print(f"Error: File not found at '{case_dir}'")
                    return None
                
                u = np.load(case_dir / 'u.npy') #(620, 64, 64)
                v = np.load(case_dir / 'v.npy') #(620, 64, 64)
                uv = np.stack([u, v], axis=1).astype(np.float32)
                #stack u and v 
                metadata = self._extract_metadata(case_dir) #(4, 64, 64)
                for t in range(uv.shape[0]):
                    self.samples.append((uv, metadata, t))
    
    def __getitem__(self, index):
        """
        for autoregressive case (predict next timestep)
            x : (6, 64, 64)
            y : (2, 64, 64)
        """
        uv, metadata, t = self.samples[index]
        x = np.concatenate([uv[t], metadata], axis=0).astype(np.float32)
        y = uv[t + 1]
        return x, y


    def _extract_metadata(self, case_dir):
        """
            returns arr (4, 64, 64) with v_in, density, viscocity, radius in order and broadcasted over input array size
        """
        try:
            with open(case_dir / 'case.json', 'r') as f:
                raw = json.load(f)

            return np.stack([
                np.full((64, 64), raw['vel_in']),
                np.full((64, 64), raw['density']),
                np.full((64, 64), raw['viscosity']),
                np.full((64, 64), raw['radius']),
            ]).astype(np.float32)
                
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from the file at '{case_dir}'")
            return None
        except IOError as e:
            print(f"Error reading file: {e}")
            return None


        
