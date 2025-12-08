import sys
import numpy as np
from pathlib import Path
import json
import os
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, geometry : str = 'cylinder'):
        if geometry not in ['cylinder', 'cavity', 'dam', 'tube']:
            print('enter valid geometry')
            sys.exit()

        self.path = Path(f'../../data/CFDbench/{geometry}')
        self.data = self._load_data()
        
        self.data_type_split = {'bc': 0, 'geo' : 0, 'prop' : 0}

    def __len__(self):
        return len(self.data)

    def _load_data(self):
        """
            returns [num samples, 6, 64, 64]
        """
        data = []
        metadata = {}
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
                
                u = np.load(case_dir / 'u.npy')
                v = np.load(case_dir / 'v.npy')

                #stack u and v 
                metadata = self._extract_metadata(case_dir)
                uv = np.stack([u, v], axis=1).astype(np.float32)
                combine = np.concat(uv, metadata)
                data.append(combine)
        return data
    
    def __getitem__(self, index):
        keys = list(self.data.keys())
        key = keys[index]
        return self.data[key]

    def _extract_metadata(self, case_dir):
        """
            returns arr (4, 64, 64) with v_in, density, viscocity, radius in order and broadcasted over input array size
        """
        try:
            with open(case_dir / 'case.json', 'r') as file:
                raw_data = json.load(file)
                v_in = np.full((64,64), raw_data['vel_in'])
                density = np.full((64,64), raw_data['density'])
                viscosity = np.full((64,64), raw_data['viscosity'])
                radius = np.full((64,64), raw_data['radius'])
                return np.vstack([v_in, density, viscosity, radius])
                
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from the file at '{case_dir}'")
            return None
        except IOError as e:
            print(f"Error reading file: {e}")
            return None


        
