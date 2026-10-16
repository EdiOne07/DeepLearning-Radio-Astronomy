import torch
import torch.nn as nn

def polar_to_cartesian(output,amplitude_scale=1.0):
    """
    output: [B, 2, H, W]
        Channel 0: predicted log amplitude
        Channel 1: predicted phase

    Returns: [B, 2, H, W]
        Channel 0: real
        Channel 1: imaginary
    """
    log_amplitude=output[:,0]
    phase=output[:,1]
    log_amplitude = torch.clamp(
        log_amplitude, min=0.0, max=20.0
    )
    amplitude = (
        torch.expm1(log_amplitude) * amplitude_scale
    )
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
