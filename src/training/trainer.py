import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm



class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        dataset,
        loss=torch.nn.MSELoss(),
        optimizer=None,
        scheduler=None,
        batch_size: int = 32,
        lr: float = 1e-4,
        val_split: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu',
        checkpoint_dir: str = '../../chkpts'):
        
        self.device = device
        self.model = model.to(device)
        self.loss = loss
        self.checkpoint_dir = Path(checkpoint_dir)

        #optimizer
        if optimizer == None: 
            self.optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
        else:
            self.optimizer = optimizer

        #data
        val_size = int(val_split * len(dataset))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
