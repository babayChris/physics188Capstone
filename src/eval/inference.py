import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from typing import Optional, Union, Tuple, Dict
from tqdm import tqdm


class FNOInference:
    def __init__(
        self,
        model: torch.nn.Module,
        checkpoint_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu',
        output_dir: str = './inference_results'
    ):
        self.device = device
        self.model = model.to(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        self.model.eval()
        print(f"Inference running on: {device}")
    
    def load_checkpoint(self, checkpoint_path: str):
        print(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully!")
    
    def predict_single_step(
        self, 
        initial_state: Union[torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        self.model.eval()
        
        if isinstance(initial_state, np.ndarray):
            initial_state = torch.from_numpy(initial_state).float()
        
        if initial_state.ndim == 3:
            initial_state = initial_state.unsqueeze(0)
        
        initial_state = initial_state.to(self.device)
        
        with torch.no_grad():
            pred = self.model(initial_state)
        
        pred = pred.cpu().numpy()
        
        if pred.shape[0] == 1:
            pred = pred[0]
        
        return pred
    
    def predict_autoregressive(
        self,
        initial_state: Union[torch.Tensor, np.ndarray],
        num_steps: int,
        return_all_steps: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
        self.model.eval()
        
        if isinstance(initial_state, np.ndarray):
            initial_state = torch.from_numpy(initial_state).float()
        
        single_sample = initial_state.ndim == 3
        if single_sample:
            initial_state = initial_state.unsqueeze(0)
        
        curr_state = initial_state.to(self.device)
        all_predictions = []
        
        with torch.no_grad():
            for step in range(num_steps):
                pred = self.model(curr_state)
                
                if return_all_steps:
                    all_predictions.append(pred.cpu().numpy())
                
                if curr_state.shape[-1] > 2:
                    curr_state = torch.cat([pred, curr_state[:, :, :, 2:]], dim=-1)
                else:
                    curr_state = pred
        
        final_pred = pred.cpu().numpy()
        
        if single_sample:
            final_pred = final_pred[0]
            if return_all_steps:
                all_predictions = [p[0] for p in all_predictions]
        
        if return_all_steps:
            return final_pred, all_predictions
        else:
            return final_pred
    
    def predict_dataset(
        self,
        dataset,
        batch_size: int = 16,
        max_samples: Optional[int] = None,
        use_first_step_only: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        all_preds = []
        all_targets = []
        all_inputs = []
        
        samples_processed = 0
        
        with torch.no_grad():
            for x, y in tqdm(loader, desc="Running inference"):
                if max_samples and samples_processed >= max_samples:
                    break
                
                x = x.to(self.device)
                pred = self.model(x)
                
                pred_np = pred.cpu().numpy()
                y_np = y.numpy()
                x_np = x.cpu().numpy()
                
                if y_np.ndim == 5:
                    if use_first_step_only:
                        y_np = y_np[:, 0, :, :, :]
                
                pred_np = self._ensure_batch_channels_last(pred_np, expected_channels=2)
                y_np = self._ensure_batch_channels_last(y_np, expected_channels=2)
                x_np = self._ensure_batch_channels_last(x_np)
                
                all_preds.append(pred_np)
                all_targets.append(y_np)
                all_inputs.append(x_np)
                
                samples_processed += x.shape[0]
        
        predictions = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        inputs = np.concatenate(all_inputs, axis=0)
        
        if max_samples:
            predictions = predictions[:max_samples]
            targets = targets[:max_samples]
            inputs = inputs[:max_samples]
        
        print(f"Final shapes: preds={predictions.shape}, targets={targets.shape}, inputs={inputs.shape}")
        
        return predictions, targets, inputs
    
    def evaluate_autoregressive_with_gt(
        self,
        dataset,
        batch_size: int = 8,
        max_samples: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        if dataset.rollout_steps <= 1:
            raise ValueError("Dataset must have rollout_steps > 1 for multi-step evaluation")
        
        self.model.eval()
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        rollout_steps = dataset.rollout_steps
        
        mse_per_step = np.zeros(rollout_steps)
        mae_per_step = np.zeros(rollout_steps)
        energy_error_per_step = np.zeros(rollout_steps)
        n_samples = 0
        
        with torch.no_grad():
            for x, y_multi in tqdm(loader, desc="Multi-step evaluation"):
                if max_samples and n_samples >= max_samples:
                    break
                
                x = x.to(self.device)
                y_multi = y_multi.to(self.device)
                
                batch_size_curr = x.shape[0]
                curr_state = x.clone()
                
                for t in range(rollout_steps):
                    pred = self.model(curr_state)
                    target = y_multi[:, t, :, :, :]
                    
                    mse = torch.mean((pred - target) ** 2).item()
                    mae = torch.mean(torch.abs(pred - target)).item()
                    
                    ke_pred = 0.5 * (pred[..., 0]**2 + pred[..., 1]**2)
                    ke_true = 0.5 * (target[..., 0]**2 + target[..., 1]**2)
                    energy_err = torch.mean(torch.abs(ke_pred - ke_true)).item()
                    
                    mse_per_step[t] += mse * batch_size_curr
                    mae_per_step[t] += mae * batch_size_curr
                    energy_error_per_step[t] += energy_err * batch_size_curr
                    
                    if curr_state.shape[-1] > 2:
                        curr_state = torch.cat([pred, curr_state[:, :, :, 2:]], dim=-1)
                    else:
                        curr_state = pred
                
                n_samples += batch_size_curr
        
        mse_per_step /= n_samples
        mae_per_step /= n_samples
        energy_error_per_step /= n_samples
        
        results = {
            'mse_per_step': mse_per_step,
            'rmse_per_step': np.sqrt(mse_per_step),
            'mae_per_step': mae_per_step,
            'energy_error_per_step': energy_error_per_step,
            'n_samples': n_samples,
            'rollout_steps': rollout_steps
        }
        
        return results
    
    def compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        pred_mag = np.sqrt(predictions[..., 0]**2 + predictions[..., 1]**2)
        true_mag = np.sqrt(targets[..., 0]**2 + targets[..., 1]**2)
        mag_error = np.mean(np.abs(pred_mag - true_mag))
        
        ke_pred = 0.5 * (predictions[..., 0]**2 + predictions[..., 1]**2)
        ke_true = 0.5 * (targets[..., 0]**2 + targets[..., 1]**2)
        energy_mae = np.mean(np.abs(ke_pred - ke_true))
        
        rel_l2 = np.sqrt(np.sum((predictions - targets)**2)) / np.sqrt(np.sum(targets**2))
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'magnitude_mae': float(mag_error),
            'energy_mae': float(energy_mae),
            'relative_l2': float(rel_l2)
        }
        
        return metrics
    
    def visualize_prediction(
        self,
        prediction: np.ndarray,
        target: Optional[np.ndarray] = None,
        input_field: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        title: str = "Velocity Field Prediction",
        show_streamlines: bool = True
    ):
        prediction = self._ensure_channels_last(prediction, expected_channels=2)
        if target is not None:
            target = self._ensure_channels_last(target, expected_channels=2)
        if input_field is not None:
            input_field = self._ensure_channels_last(input_field)
        
        num_plots = 1 + (target is not None) + (input_field is not None)
        fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 5))
        
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        if input_field is not None:
            input_velocity = self._extract_velocity_from_input(input_field)
            self._plot_velocity_field(
                axes[plot_idx], 
                input_velocity,
                "Input (t)", 
                show_streamlines
            )
            plot_idx += 1
        
        self._plot_velocity_field(
            axes[plot_idx], 
            prediction, 
            "Prediction (t+1)", 
            show_streamlines
        )
        plot_idx += 1
        
        if target is not None:
            self._plot_velocity_field(
                axes[plot_idx], 
                target, 
                "Ground Truth (t+1)", 
                show_streamlines
            )
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def visualize_rollout(
        self,
        predictions: list,
        targets: Optional[list] = None,
        save_dir: Optional[str] = None,
        prefix: str = "rollout"
    ):
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        for i, pred in enumerate(predictions):
            target = targets[i] if targets is not None else None
            
            save_path = None
            if save_dir:
                save_path = save_dir / f"{prefix}_step_{i:03d}.png"
            
            self.visualize_prediction(
                pred,
                target,
                save_path=save_path,
                title=f"Timestep {i+1}",
                show_streamlines=True
            )
    
    def visualize_error_map(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        save_path: Optional[str] = None
    ):
        prediction = self._ensure_channels_last(prediction, expected_channels=2)
        target = self._ensure_channels_last(target, expected_channels=2)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        u_error = np.abs(prediction[..., 0] - target[..., 0])
        im1 = axes[0, 0].imshow(u_error, cmap='Reds', origin='lower')
        axes[0, 0].set_title('U-velocity Error')
        plt.colorbar(im1, ax=axes[0, 0])
        
        v_error = np.abs(prediction[..., 1] - target[..., 1])
        im2 = axes[0, 1].imshow(v_error, cmap='Reds', origin='lower')
        axes[0, 1].set_title('V-velocity Error')
        plt.colorbar(im2, ax=axes[0, 1])
        
        pred_mag = np.sqrt(prediction[..., 0]**2 + prediction[..., 1]**2)
        true_mag = np.sqrt(target[..., 0]**2 + target[..., 1]**2)
        mag_error = np.abs(pred_mag - true_mag)
        im3 = axes[1, 0].imshow(mag_error, cmap='Reds', origin='lower')
        axes[1, 0].set_title('Magnitude Error')
        plt.colorbar(im3, ax=axes[1, 0])
        
        rel_error = mag_error / (true_mag + 1e-8)
        im4 = axes[1, 1].imshow(rel_error, cmap='Reds', origin='lower', vmax=0.5)
        axes[1, 1].set_title('Relative Error')
        plt.colorbar(im4, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved error map to {save_path}")
        
        plt.show()
    
    def visualize_rollout_comparison(
        self,
        dataset,
        sample_idx: int,
        save_dir: Optional[str] = None
    ):
        if dataset.rollout_steps <= 1:
            raise ValueError("Dataset must have rollout_steps > 1")
        
        x, y_multi = dataset[sample_idx]
        
        final_pred, predictions = self.predict_autoregressive(
            x.numpy(),
            num_steps=dataset.rollout_steps,
            return_all_steps=True
        )
        
        ground_truth = [y_multi[t].numpy() for t in range(dataset.rollout_steps)]
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        for t in range(len(predictions)):
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            self._plot_velocity_field(
                axes[0],
                predictions[t],
                f"Prediction (t={t+1})",
                show_streamlines=False
            )
            
            self._plot_velocity_field(
                axes[1],
                ground_truth[t],
                f"Ground Truth (t={t+1})",
                show_streamlines=False
            )
            
            error = np.abs(predictions[t] - ground_truth[t])
            u_err, v_err = error[..., 0], error[..., 1]
            mag_err = np.sqrt(u_err**2 + v_err**2)
            
            im = axes[2].imshow(mag_err, cmap='Reds', origin='lower')
            plt.colorbar(im, ax=axes[2], label='Error Magnitude')
            axes[2].set_title(f"Error (t={t+1})")
            axes[2].set_xlabel('X')
            axes[2].set_ylabel('Y')
            
            mse = np.mean(error ** 2)
            mae = np.mean(np.abs(error))
            fig.suptitle(f'Timestep {t+1}/{len(predictions)} - MSE: {mse:.6f}, MAE: {mae:.6f}',
                        fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            if save_dir:
                save_path = save_dir / f'comparison_t{t+1:03d}.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        
        if save_dir:
            print(f"Saved {len(predictions)} comparison frames to {save_dir}")
    
    def plot_error_accumulation(
        self,
        results: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        steps = np.arange(1, results['rollout_steps'] + 1)
        
        axes[0, 0].plot(steps, results['mse_per_step'], 'o-', linewidth=2)
        axes[0, 0].set_xlabel('Timestep')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].set_title('Mean Squared Error Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        axes[0, 1].plot(steps, results['mae_per_step'], 'o-', linewidth=2, color='orange')
        axes[0, 1].set_xlabel('Timestep')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('Mean Absolute Error Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        axes[1, 0].plot(steps, results['rmse_per_step'], 'o-', linewidth=2, color='green')
        axes[1, 0].set_xlabel('Timestep')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].set_title('Root Mean Squared Error Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        axes[1, 1].plot(steps, results['energy_error_per_step'], 'o-', linewidth=2, color='red')
        axes[1, 1].set_xlabel('Timestep')
        axes[1, 1].set_ylabel('Energy Error')
        axes[1, 1].set_title('Kinetic Energy Error Over Time')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
        
        plt.suptitle(f'Error Accumulation Over {results["rollout_steps"]} Timesteps\n'
                     f'({results["n_samples"]} samples)', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved error accumulation plot to {save_path}")
        
        plt.show()
    
    def save_predictions(
        self,
        predictions: np.ndarray,
        save_path: str,
        metadata: Optional[dict] = None
    ):
        save_path = Path(save_path)
        
        if save_path.suffix == '.npz':
            if metadata:
                np.savez(save_path, predictions=predictions, **metadata)
            else:
                np.savez(save_path, predictions=predictions)
        else:
            np.save(save_path, predictions)
        
        print(f"Saved predictions to {save_path}")
    
    def _extract_velocity_from_input(self, input_field: np.ndarray) -> np.ndarray:
        if input_field.shape[-1] < 2:
            raise ValueError(f"Input field must have at least 2 channels (u, v), got {input_field.shape[-1]}")
        
        velocity = input_field[..., :2]
        return velocity
    
    def _ensure_channels_last(self, array: np.ndarray, expected_channels: Optional[int] = None) -> np.ndarray:
        if array.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {array.shape}")
        
        if array.shape[0] < 20 and array.shape[0] < min(array.shape[1], array.shape[2]):
            array = array.transpose(1, 2, 0)
        
        if expected_channels is not None and array.shape[-1] != expected_channels:
            if array.shape[-1] >= expected_channels:
                array = array[..., :expected_channels]
            else:
                raise ValueError(f"Expected {expected_channels} channels, got {array.shape[-1]}")
        
        return array
    
    def _ensure_batch_channels_last(self, array: np.ndarray, expected_channels: Optional[int] = None) -> np.ndarray:
        if array.ndim == 3:
            array = array[np.newaxis, ...]
        
        if array.ndim != 4:
            raise ValueError(f"Expected 4D array [N, H, W, C] or [N, C, H, W], got shape {array.shape}")
        
        if array.shape[1] < 20 and array.shape[1] < min(array.shape[2], array.shape[3]):
            array = array.transpose(0, 2, 3, 1)
        
        if expected_channels is not None and array.shape[-1] != expected_channels:
            if array.shape[-1] >= expected_channels:
                array = array[..., :expected_channels]
            else:
                raise ValueError(f"Expected {expected_channels} channels, got {array.shape[-1]}")
        
        return array
    
    def _plot_velocity_field(self, ax, velocity, title, show_streamlines=True):
        if velocity.ndim != 3 or velocity.shape[-1] != 2:
            raise ValueError(f"Expected velocity shape [H, W, 2], got {velocity.shape}")
        
        u = velocity[:,:,0]
        v = velocity[:,:, 1]
        magnitude = np.sqrt(u**2 + v**2)
        
        im = ax.imshow(magnitude, cmap='viridis', origin='lower')
        plt.colorbar(im, ax=ax, label='Velocity Magnitude')
        
        if show_streamlines:
            H, W = u.shape
            y, x = np.mgrid[0:H, 0:W]
            ax.streamplot(
                x, y, u, v,
                color='white',
                linewidth=0.5,
                density=1.5,
                arrowsize=0.8,
                
            )
        
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')