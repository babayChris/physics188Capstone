import sys
import numpy as np
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self):
        self.path = '../../data/CFDbench'
        self.

    def __len__(self):
        return len(self.)

    def fetch(self, geometry):
        if geometry not in ['cylinder', 'cavity', 'dam', 'tube']:
            print('enter valid geometry')
            sys.exit()

        u, v = np.load(f'{self.path}/{geometry}/')