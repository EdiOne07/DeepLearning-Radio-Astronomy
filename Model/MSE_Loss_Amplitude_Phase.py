import torch
import torch.nn as nn
import torch.nn.functional as F


def polar_to_cartesian(output,amplitude_scale=1.0):
    """
    output: [B, 2, H, W]
        Channel 0: predicted log amplitude
        Channel 1: predicted phase

    Returns: [B, 2, H, W]
        Channel 0: real
        Channel 1: imaginary
    """
    raw_amplitude=output[:,0]
    phase=output[:,1]
    amplitude = amplitude_scale * torch.nn.functional.softplus(raw_amplitude)
    
    real_part=amplitude*torch.cos(phase)
    imag_part=amplitude*torch.sin(phase)

    return torch.stack((real_part,imag_part),dim=1)

class MSELoss_Amplitude_Phase(nn.Module):
    def __init__(self, amplitude_scale: float = 1.0):
        super().__init__()
        self.amplitude_scale=amplitude_scale
    def forward(self, output, target, mask):
        """
        output: [B, 2, H, W] predicted polar values
        target: [B, 2, H, W] original real/imag
        mask:   [B, 1, H, W] occupied positions
        """
        prediction=polar_to_cartesian(output=output,amplitude_scale=self.amplitude_scale)
        error=(prediction-target).square()
        error=error.sum(dim=1,keepdim=True)
        masked_error=mask*error
        return masked_error.sum() / mask.sum().clamp_min(1)