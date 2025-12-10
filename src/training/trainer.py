import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import time
from torchvision import transforms
import matplotlib.pyplot as plt



class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        dataset,
        epochs = 50,
        optimizer=None,
        optimizer_params = None,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_params=None,
        batch_size: int = 16,
        lr: float = 1e-4,
        val_split: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu',
        checkpoint_dir: str = '../../chkpts'):
        
        self.epochs = epochs
        self.device = device
        self.model = model.to(device)
        print(device)
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir)

        #optimizer
        if optimizer == None: 
            self.optimizer = torch.optim.AdamW(model.parameters(), lr = lr, **(optimizer_params or {}))
        else:
            self.optimizer = optimizer

        #data
        if scheduler:
            self.scheduler = scheduler(self.optimizer, **(scheduler_params or {}))

        val_size = int(val_split * len(dataset))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        #NOTE: can change num_workers to speed / slow training epoch
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True) 
        self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.batches = batch_size
        self.losses = []

    def train_autoregressive(self, rollout_steps = 5, clip_grad = True, show_loss = True):
        print('start training')
        print(f"Total epochs: {self.epochs}, Batches per epoch: {self.batches}")

        for e in range(self.epochs):
            self.model.train()
            e_loss = 0
            e_time = time.time()

            for i, (x, y) in enumerate(self.train_loader):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()

                curr_loss = 0.0
                curr_state = x
                #iter forward in time
                for k in range(rollout_steps):
                    pred = self.model(curr_state) # [B, H, W, 2] (v, u)

                    target = y[:, k, :, :, :]
                    step_loss = self._energy_weighted_loss(pred, target)
                    curr_loss += step_loss
                    curr_state[:, :, :, 0] = pred[:, :, :, 0] #u_x
                    curr_state[:, :, :, 1] = pred[:, :, :, 1] #u_y
                loss = curr_loss / rollout_steps

                #backward
                loss.backward()
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                e_loss += loss.item()
            avg_loss = e_loss / len(self.train_loader)
            self.losses.append(avg_loss)
            if self.scheduler:
                self.scheduler.step(avg_loss)
            epoch_time = time.time() - e_time
            print(f"Epoch [{e+1}/{self.epochs}] completed - Avg Loss: {avg_loss:.6f}, "
            f"LR: {self.optimizer.param_groups[0]['lr']:.6f}, Time: {epoch_time:.2f}s")
        if show_loss:
            self.show_loss()
        print("Training completed!")


    def _energy_weighted_loss(self, pred, target, energy_weight = 0.1):
        """
            x: [B, H, W, 2] input u, v
            y: [B, H, W, 2] output u, v
        """
        generic_loss = torch.mean((pred - target) ** 2)
        u_pred, v_pred = pred[:, :, :, 0], pred[:, :, :, 1]
        u_target, v_target = target[:, :, :, 0], target[:, :, :, 1]
        kinetic_energy_pred = 0.5 * (u_pred**2 + v_pred**2)
        kinetic_energy_target = 0.5 * (u_target**2 + v_target**2)
        energy_error = torch.mean((kinetic_energy_pred - kinetic_energy_target) ** 2)
        return generic_loss + energy_weight * energy_error
                
    def show_loss(self, log_scale = True):
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        if log_scale:
            plt.yscale('log')
        plt.grid(True)
        plt.show()


        
            