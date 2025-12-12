import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 
                                   dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 
                                   dtype=torch.cfloat)
        )

    def forward(self, x):
        batchsize = x.shape[0]
        device = x.device        
        # FFT
        x_ft = torch.fft.rfft2(x)
        
        # Output tensor on same device
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
            dtype=torch.cfloat, device=device
        )
        
        # Spectral multiplication
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -self.modes1:, :self.modes2],
            self.weights2
        )
        
        # Inverse FFT
        x_out = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        
        return x_out


class FNOBlock(nn.Module):
    """FNO Block: Spectral Convolution + Pointwise Convolution + Residual"""
    def __init__(self, width: int, modes1: int, modes2: int):
        super(FNOBlock, self).__init__()
        self.conv = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)  # Pointwise convolution
        self.activation = nn.GELU()
        self.norm = nn.InstanceNorm2d(width)
    
    def forward(self, x):
        # Residual connection around the block
        return x + self.norm(self.activation(self.conv(x) + self.w(x)))


class FNO(nn.Module):
    """
    Fourier Neural Operator for 2D Navier-Stokes time evolution
    
    Input channels: [u, v, sdf, log_rho, log_mu, vel_bc, geom_onehot x4] = 10 channels
    Output channels: [u, v] = 2 channels
    """
    def __init__(
        self,
        in_channels: int = 10,
        out_channels: int = 2,
        width: int = 32,
        modes1: int = 12,
        modes2: int = 12,
        n_layers: int = 4,
        padding: int = 8
    ):
        super(FNO, self).__init__()
        self.padding = padding
        
        # Lifting layer: project to higher dimension
        self.fc0 = nn.Linear(in_channels, width)
        
        # Fourier layers
        self.fno_blocks = nn.ModuleList([
            FNOBlock(width, modes1, modes2) for _ in range(n_layers)
        ])
        
        # Projection layer: back to output dimension
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)
        self.activation = nn.GELU()
    
    def forward(self, x):
        """
        x: [B, H, W, in_channels]
        Returns: [B, H, W, out_channels]
        """
        # Zero padding for boundary conditions
        if self.padding > 0:
            # Pad H and W dimensions: (left, right, top, bottom, front, back) for last 3 dims
            # For [B, H, W, C]: pad W, then H, then C (but we want C=0 padding)
            x = torch.nn.functional.pad(
                x, 
                (0, 0, self.padding, self.padding, self.padding, self.padding), 
                mode='constant', 
                value=0
            )
        
        # Lifting: input is [B, H, W, C], fc0 acts on last dim
        x = self.fc0(x)  # [B, H, W, width]
        x = x.permute(0, 3, 1, 2)  # [B, width, H, W] for conv layers
        
        # Fourier layers (each block has internal residual connection)
        for fno_block in self.fno_blocks:
            x = fno_block(x)
        
        # Projection
        x = x.permute(0, 2, 3, 1)  # [B, H, W, width]
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)  # [B, H, W, out_channels]
        
        # Crop back to original size
        if self.padding > 0:
            x = x[:, self.padding:-self.padding, self.padding:-self.padding, :]
        
        return x