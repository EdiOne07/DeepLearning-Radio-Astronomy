import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedVisibilityLoss(nn.Module):
    def __init__(self, lambda_smooth: float = 1e-4):
        super().__init__()
        self.lambda_smooth = lambda_smooth
    def forward(self, pred:torch.Tensor, target_grid:torch.Tensor)->torch.Tensor:
        measured_vis=target_grid[:,0:2,:,:]
        mask=target_grid[:,2:3,:,:]

        diff = (pred - measured_vis) * mask
        fidelity_loss = torch.sum(diff ** 2) / (torch.sum(mask) * 2 + 1e-8)

        tv_h = torch.mean(torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]))
        smoothness_loss = tv_h + tv_w

        return fidelity_loss + self.lambda_smooth * smoothness_loss