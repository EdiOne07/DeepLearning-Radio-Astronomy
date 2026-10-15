
import sys
import glob
import copy
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from Self_Supervising_Loss import MaskedVisibilityLoss
from Spatial_KFold import SpatialKfoldMask  # Adjust filename capitalization if needed
from rat_stack import TimeSlicedVisDataset
import Data_Gridding

from enum import Enum

chuchichaestli_src = Path(
    "/home/edione/Projects/chuchichaestli/src"
)
if str(chuchichaestli_src) not in sys.path:
    sys.path.insert(0, str(chuchichaestli_src))

from chuchichaestli.models.autoencoder import Autoencoder

###################
# CONFIGURATION
###################
WINDOW_SIZE=8
TARGET_SIZE=1
STRIDE=1
EPOCHS=30
LEARNING_RATE=1e-3
LAMBDA_SMOOTHNESS=1e-6
K_FOLDS=5
GRID_SIZE=256
MAX_UV=70000
BLOCK_SIZE = 32
HOLDOUT_RATIO = 0.20
SEED=42
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class Model_Type(Enum):
    AUTOENCODER_BASE="AUTOENCODER_BASE"
    FUTURE_AUTOENCODER="FUTURE_AUTOENCODER"

###################
# DATA_HANDLING
###################

transform=Data_Gridding(GRID_SIZE,MAX_UV)

class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset=dataset
        self.transform=transform
    def __getitem__(self, index):
        item=self.dataset[index]
        return self.transform(self.dataset[index])
    def __len__(self):
        return len(self.dataset)

def makeDataset(paths:list[str],use_cuda):
    dataset=TimeSlicedVisDataset(
        paths=paths,
        window_size=WINDOW_SIZE,
        target_size=TARGET_SIZE,
        stride=STRIDE,
        pin_memory=use_cuda,
        read_uvw=True,
    )
    return TransformDataset(dataset=dataset,transform=transform)

###################
# MODEL
###################

def makeModel(device,model_type:Model_Type):
    if model_type==Model_Type.AUTOENCODER_BASE:
        return Autoencoder.build(
                dimensions=2,
                in_channels=3,
                out_channels=2,
                latent_dim=16,
                res_act_fn="silu",
                res_norm_type="group",
                encoder_args={"n_channels": 16},
            ).to(device)
    elif Model_Type==Model_Type.FUTURE_AUTOENCODER:
        # tbd
        return 0

###################
# METRICS
###################

@torch.no_grad()
def compute_reconstruction_metrics(pred, target, mask):
    """
    Evaluate only the deliberately hidden observed cells.

    pred, target: [B, 2, H, W]
    mask:         [B, 1, H, W]
    """
    mask = mask.expand_as(target)

    squared_error = ((pred - target) ** 2 * mask).sum()
    target_power = (target ** 2 * mask).sum()

    nmse = (
        squared_error / target_power.clamp_min(1e-8)
    ).item()

    snr_db = (
        10.0
        * torch.log10(
            target_power
            / squared_error.clamp_min(1e-8)
        )
    ).item()

    return nmse, snr_db

###################
# MASKING
###################
def make_spatial_masker():
    return SpatialKfoldMask(
        grid_size=GRID_SIZE,
        block_size=BLOCK_SIZE,
        split_ratio=HOLDOUT_RATIO,
    )

def make_model_and_input_target(grid,heldout_mask):
    original_mask=grid[:,2:3,:,:]
    context_mask=original_mask.clone()
    context_mask[heldout_mask>0]=0.0

    model_input=grid.clone()
    model_input[:,0:2,:,:]*=context_mask
    model_input[:,2:3,:,:]=context_mask

    loss_target=grid.clone()
    loss_target[:,2:3,:,:]=heldout_mask

    return model_input, loss_target, context_mask

###################
# EVALUATION
###################

@torch.no_grad()
def validate_model(model,val_loader,criterion,device):
    model.eval()

    masker=make_spatial_masker()

    total_error = 0.0
    total_power = 0.0
    total_loss = 0.0
    total_batches = 0
    total_hidden_cells = 0

    mask_generator = torch.Generator(device="cpu")
    mask_generator.manual_seed(SEED + 1000)

    for batch in val_loader:
        grid_data=batch["input_grid"].to(device)
        original_mask=grid_data[:,2:3,:,:]

        _,heldout_mask=masker.train_validation_split(mask=original_mask,generator=mask_generator)
        model_input,loss_target,_=make_model_and_input_target(grid=grid_data,heldout_mask=heldout_mask)

        output=model(model_input)
        loss=criterion(output,loss_target)

        target=grid_data[:,0:2,:,:]
        expanded_mask = heldout_mask.expand_as(target)

        error = (
                    (output - target).square() * expanded_mask
                ).sum()
        
        power = (
                    target.square() * expanded_mask
                ).sum()
        
        total_error += error.item()
        total_power += power.item()
        total_loss += loss.item()
        total_batches += 1
        total_hidden_cells += int(heldout_mask.sum().item())
        
    if total_hidden_cells == 0:
                raise RuntimeError(
                    "Validation has zero held-out observed cells. "
                    "Check the occupancy mask and spatial split."
                )
        
    nmse = total_error / max(total_power, 1e-8)
    snr_db = 10.0 * torch.log10(
                torch.tensor(
                    total_power / max(total_error, 1e-8)
                )
            ).item()
        
    avg_loss = total_loss / max(total_batches, 1)
        
    return avg_loss, nmse, snr_db
        
###################
# TRAINING_ONE_EPOCH
###################

def train_one_fold(
    fold_number,
    train_paths,
    val_paths,
    output_dir,
    device,
):
    print("\n" + "=" * 70)
    print(f"FOLD {fold_number}/5")
    print("Training observations:", train_paths)
    print("Validation observation:", val_paths)
    print("=" * 70)
     
     