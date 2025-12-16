import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from data_handle.data_final import stratified_split


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        dataset,
        epochs=50,
        optimizer=None,
        optimizer_params=None,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_params=None,
        batch_size: int = 16,
        lr: float = 1e-4,
        val_split: float = 0.1,
        device: str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu',
        checkpoint_dir: str = '/root/programs/physics188/physics188Capstone/src/training/epochs/FNO_large'
    ):
        self.epochs = epochs
        self.device = device
        self.model = model.to(device)
        print(f"Using device: {device}")
        self.checkpoint_dir = Path(checkpoint_dir)

        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, **(optimizer_params or {}))
        else:
            self.optimizer = optimizer

        # Scheduler
        if scheduler:
            self.scheduler = scheduler(self.optimizer, **(scheduler_params or {}))
        else:
            self.scheduler = None

        # Data splits - FIXED: properly handle subset indices
        train_val_set, self.test_set = stratified_split(dataset, val_fraction=0.1, seed=42)
        
        # Second split from train_val indices only
        train_val_indices = list(train_val_set.indices)
        n_val = int(len(train_val_indices) * 0.12)  # ~10% of original for val
        
        rng = np.random.default_rng(43)
        rng.shuffle(train_val_indices)
        
        val_indices = train_val_indices[:n_val]
        train_indices = train_val_indices[n_val:]
        
        self.train_set = Subset(dataset, train_indices)
        self.val_set = Subset(dataset, val_indices)
        
        print(f"Train: {len(self.train_set)}, Val: {len(self.val_set)}, Test: {len(self.test_set)}")

        # DataLoaders - use num_workers=0 for preloaded data (shared memory)
        # or num_workers=2 with persistent_workers if not preloading
        self.train_loader = DataLoader(
            self.train_set, batch_size=batch_size, shuffle=True, 
            num_workers=0, pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_set, batch_size=batch_size, shuffle=False, 
            num_workers=0, pin_memory=True
        )
        self.test_loader = DataLoader(
            self.test_set, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True
        )
        
        self.batch_size = batch_size
        self.losses = []
        self.val_losses = []

    def train_single_step(self, epochs=None, clip_grad=True, show_loss=True):
        """Single-step training without autoregressive rollout."""
        epochs = epochs or self.epochs
        print(f'Starting single-step training for {epochs} epochs')
        
        best_val_loss = float('inf')
        
        for e in range(epochs):
            self.model.train()
            e_loss = 0
            e_time = time.time()

            for x, y in tqdm(self.train_loader, desc=f"Epoch {e+1}/{epochs}"):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)  # [B, H, W, 2]

                self.optimizer.zero_grad()
                pred = self.model(x)
                loss = self._energy_weighted_loss(pred, y) + self.compressible_loss(pred)

                loss.backward()
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                e_loss += loss.item()

            avg_loss = e_loss / len(self.train_loader)
            val_loss = self._validate()
            
            self.losses.append(avg_loss)
            self.val_losses.append(val_loss)
            
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 
                          f'{self.checkpoint_dir}/best_model.pt')
            
            # Save periodic checkpoint
            torch.save(self.model.state_dict(), 
                      f'{self.checkpoint_dir}/model_weights_{e}.pt')
            
            epoch_time = time.time() - e_time
            print(f"Epoch [{e+1}/{epochs}] - Train: {avg_loss:.6f}, Val: {val_loss:.6f}, "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}, Time: {epoch_time:.2f}s")
        
        if show_loss:
            self.show_loss()
        print("Training completed!")




    def train_autoregressive(self, epochs=None, load_weights=None, rollout_steps=5, 
                             clip_grad=True, show_loss=True):
        """Autoregressive training with multi-step rollout."""
        epochs = epochs or self.epochs
        print(f'Starting autoregressive training for {epochs} epochs, rollout={rollout_steps}')
        
        if load_weights:
            self.model.load_state_dict(torch.load(load_weights, weights_only=True))
        
        best_val_loss = float('inf')
        
        for e in range(epochs):
            self.model.train()
            e_loss = 0
            e_time = time.time()

            for x, y in tqdm(self.train_loader, desc=f"Epoch {e+1}/{epochs}"):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()

                curr_loss = 0.0
                curr_state = x.clone()
                
                # Iterate forward in time
                for k in range(rollout_steps):
                    pred = self.model(curr_state)  # [B, H, W, 2]
                    target = y[:, k, :, :, :]
                    step_loss = self._energy_weighted_loss(pred, target) + self.compressible_loss(pred)
                    curr_loss += step_loss
                    
                    # Update state: replace velocity channels, keep other features
                    curr_state = torch.cat([pred, curr_state[:, :, :, 2:]], dim=-1)
                
                loss = curr_loss / rollout_steps

                loss.backward()
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                e_loss += loss.item()
            
            avg_loss = e_loss / len(self.train_loader)
            val_loss = self._validate_autoregressive(rollout_steps)
            
            self.losses.append(avg_loss)
            self.val_losses.append(val_loss)
            
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 
                          f'{self.checkpoint_dir}/best_model_ar.pt')
            
            torch.save(self.model.state_dict(), 
                      f'{self.checkpoint_dir}/model_weights_ar_{e+20}.pt')
            
            epoch_time = time.time() - e_time
            print(f"Epoch [{e+1}/{epochs}] - Train: {avg_loss:.6f}, Val: {val_loss:.6f}, "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}, Time: {epoch_time:.2f}s")
        
        if show_loss:
            self.show_loss()
        print("Training completed!")

    def _validate(self):
        """Validation for single-step training."""
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                loss = self._energy_weighted_loss(pred, y)
                val_loss += loss.item()
        return val_loss / len(self.val_loader)

    def _validate_autoregressive(self, rollout_steps):
        """Validation for autoregressive training."""
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                curr_state = x.clone()
                batch_loss = 0
                
                for k in range(rollout_steps):
                    pred = self.model(curr_state)
                    target = y[:, k, :, :, :]
                    batch_loss += self._energy_weighted_loss(pred, target).item()
                    curr_state = torch.cat([pred, curr_state[:, :, :, 2:]], dim=-1)
                
                val_loss += batch_loss / rollout_steps
        
        return val_loss / len(self.val_loader)

    def compressible_loss(self, pred, weight=0.01):
        """
        Penalize divergence (continuity violation) for incompressible flow.
        ∇·u = ∂u/∂x + ∂v/∂y ≈ 0
        """
        u = pred[:, :, :, 0]  # [B, H, W]
        v = pred[:, :, :, 1]
        
        du_dx = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / 2  # [B, H-2, W-2]
        dv_dy = (v[:, 2:, 1:-1] - v[:, :-2, 1:-1]) / 2  # [B, H-2, W-2]
        
        # Divergence (should be ~0 for incompressible)
        div = du_dx + dv_dy
        return weight * torch.mean(div ** 2)

    def _energy_weighted_loss(self, pred, target, energy_weight=0.1):
        """
        MSE + kinetic energy conservation loss.
        pred/target: [B, H, W, 2] with channels [u, v]
        """
        generic_loss = torch.mean((pred - target) ** 2)
        
        u_pred, v_pred = pred[:, :, :, 0], pred[:, :, :, 1]
        u_target, v_target = target[:, :, :, 0], target[:, :, :, 1]
        
        kinetic_energy_pred = 0.5 * (u_pred**2 + v_pred**2)
        kinetic_energy_target = 0.5 * (u_target**2 + v_target**2)
        energy_error = torch.mean((kinetic_energy_pred - kinetic_energy_target) ** 2)
        
        return generic_loss + energy_weight * energy_error

    def show_loss(self, log_scale=True):
        """Plot training and validation loss."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses, label='Train')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        if log_scale:
            plt.yscale('log')
        plt.grid(True)
        plt.savefig(f'{self.checkpoint_dir}/loss_curve.png', dpi=150)
        plt.show()

    def evaluate(self, dataset_split='test'):
        """Evaluate model on test set and return metrics."""
        loader = {
            'train': self.train_loader,
            'val': self.val_loader,
            'test': self.test_loader
        }[dataset_split]
        
        self.model.eval()
        total_mse = 0
        total_mae = 0
        total_energy_err = 0
        n_samples = 0
        
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                
                mse = torch.mean((pred - y) ** 2).item()
                mae = torch.mean(torch.abs(pred - y)).item()
                
                # Energy error
                ke_pred = 0.5 * (pred[:,:,:,0]**2 + pred[:,:,:,1]**2)
                ke_true = 0.5 * (y[:,:,:,0]**2 + y[:,:,:,1]**2)
                energy_err = torch.mean(torch.abs(ke_pred - ke_true)).item()
                
                batch_size = x.shape[0]
                total_mse += mse * batch_size
                total_mae += mae * batch_size
                total_energy_err += energy_err * batch_size
                n_samples += batch_size
        
        metrics = {
            'mse': total_mse / n_samples,
            'rmse': np.sqrt(total_mse / n_samples),
            'mae': total_mae / n_samples,
            'energy_mae': total_energy_err / n_samples
        }
        
        print(f"\n{dataset_split.upper()} Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")
        
        return metrics