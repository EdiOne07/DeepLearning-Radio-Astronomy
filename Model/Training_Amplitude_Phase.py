
import sys
import glob
import copy
from pathlib import Path
import math

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from Self_Supervising_Loss import MaskedVisibilityLoss
from Spatial_KFold import SpatialKfoldMask  # Adjust filename capitalization if needed
from rat_stack import TimeSlicedVisDataset
from Data_Gridding_Patches import UV_Patch_Transform
from MSE_Loss_Amplitude_Phase import MaskedComplexMSE,polar_to_cartesian
from Data_Loading_and_Vizualization.ReconstructVisualization import visualize_reconstruction
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
WINDOW_SIZE=1
TARGET_SIZE=1
STRIDE=1
BATCH_SIZE=1
EPOCHS=25
LEARNING_RATE=1e-4
LAMBDA_SMOOTHNESS=1e-6
K_FOLDS=1
GRID_SIZE=8192
MAX_UV=70000
BLOCK_SIZE = 32
HOLDOUT_RATIO = 0.20
SEED=42
AMPLITUDE_SCALE=0.1
WEIGHT_DECAY = 1e-4
PATCH_SIZE=16
HANDLE_COLLISION="mean"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class Model_Type(Enum):
    AUTOENCODER_BASE="AUTOENCODER_BASE"
    FUTURE_AUTOENCODER="FUTURE_AUTOENCODER"

###################
# DATA_HANDLING
###################

transform=UV_Patch_Transform(GRID_SIZE,PATCH_SIZE,MAX_UV,AMPLITUDE_SCALE,HANDLE_COLLISION)

class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset=dataset
        self.transform=transform
    def __getitem__(self, index):
        item=self.dataset[index]
        return self.transform(item)
    def __len__(self):
        return len(self.dataset)

def makeDataset(paths:list[str],use_cuda):
    dataset=TimeSlicedVisDataset(
        paths=paths,
        window_size=WINDOW_SIZE,
        target_size=TARGET_SIZE,
        stride=STRIDE,
        pin_memory=False,
        read_uvw=True,
    )
    return TransformDataset(dataset=dataset,transform=transform)
#In case the DataLoader adds one more dimension
def collate_single_timestep(batch):
    if len(batch) != 1:
        raise ValueError("Expected exactly one timestep")

    return batch[0]

###################
# MODEL
###################

#This wrapper was designed to handle some compression mechanism, but right now it is not useful. Need to fix baseline first
#class PatchAutoencoder(nn.Module):
   # def __init__(self, latent_size=512):
       # super().__init__()
        #self.feature_projection = nn.Linear(66, 64)

        #self.latent_queries = nn.Parameter(
          #  torch.randn(8, 1, 64) * 0.02
        #)

        #self.attention = nn.MultiheadAttention(
           # embed_dim=64,
           # num_heads=4,
           # batch_first=True
        #)

        #self.autoencoder = Autoencoder.build(
          #  dimensions=2,
           # in_channels=3,
            #out_channels=2,
            #latent_dim=16,
            #res_act_fn="silu",
           # res_norm_type="group",
           # encoder_args={"n_channels": 16},
       # )

        #self.latent_size = latent_size

    #def encode_patches(self, patches,patch_positions):
        #encoder=self.autoencoder.encode(patches)
        #flattened_patches=encoder.flatten(start_dim=1)
        #patch_positions_normalized = (
           # patch_positions.to(
              #  device=flattened_patches.device,
                #dtype=flattened_patches.dtype,
           # ) / 511.0
       # )
    
        #final_encoding=torch.cat((flattened_patches,patch_positions_normalized),dim=1)
        #return final_encoding
    #

#Model Builder
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
    elif model_type==Model_Type.FUTURE_AUTOENCODER:
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
    total_error = 0.0
    total_power = 0.0
    total_observed = 0.0

    for batch in val_loader:
        input_patch=batch["input_patch"].to(device)
        target_patch=batch["target_patch"].to(device)
        occupancy_mask=input_patch[:,2:3,:,:]


        output=model(input_patch)
       # prediction=polar_to_cartesian(output,amplitude_scale=criterion.amplitude_scale)
        prediction=output
        squared_error = (
            (prediction - target_patch)
            .square()
            .sum(dim=1, keepdim=True)
        )
        target_power = (
            target_patch
            .square()
            .sum(dim=1, keepdim=True)
        )
        total_error += (
            squared_error * occupancy_mask
        ).sum().item()

        total_power += (
            target_power * occupancy_mask
        ).sum().item()

        total_observed += occupancy_mask.sum().item()

    if total_observed == 0:
        raise RuntimeError(
            "Validation contains no observed UV cells."
        )
    val_loss = total_error / total_observed
    if total_power <= 0:
        raise RuntimeError(
            "Validation target has zero signal power."
        )

    nmse = total_error / total_power

    snr_db = (
        -10.0 * math.log10(nmse)
        if nmse > 0
        else float("inf")
    )

    return val_loss, nmse, snr_db


###################
# TRAIN ONE FOLD
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

    use_cuda = device.type == "cuda"

    train_dataset = makeDataset(train_paths, use_cuda)
    val_dataset = makeDataset(val_paths, use_cuda)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(
            f"Fold {fold_number} has an empty dataset. "
            "Check file paths and time-window settings."
        )
    train_dataset = Subset(train_dataset, [0])
    val_dataset=train_dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_single_timestep,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_single_timestep
    )

    model = makeModel(device,Model_Type.AUTOENCODER_BASE)

    criterion = MaskedComplexMSE()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_nmse = float("inf")
    best_snr = float("-inf")
    best_state = None
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_nmse": [],
        "val_snr": [],
    }

    for epoch in range(EPOCHS):
        model.train()

        total_train_loss = 0.0
        total_train_batches = 0

        for batch in train_loader:
            input_patch = batch["input_patch"].to(device)
            target_patch=batch["target_patch"].to(device)
            original_mask = input_patch[:, 2:3, :, :]
            if epoch == 0 and total_train_batches == 0:
                print("Input patches:", input_patch.shape)
                print("Target patches:", target_patch.shape)
                print("Occupied cells:", original_mask.sum().item())
                print("Number of patches:", input_patch.shape[0])

            optimizer.zero_grad()

            outputs = model(input_patch)
            loss = criterion(outputs, target_patch,original_mask)

            if not torch.isfinite(loss):
                print("\n--- NON-FINITE LOSS DIAGNOSTIC ---")
                print("Input finite:", torch.isfinite(input_patch).all().item())
                print("Target finite:", torch.isfinite(target_patch).all().item())
                print("Output finite:", torch.isfinite(outputs).all().item())

                print("Output min:", outputs.detach().amin().item())
                print("Output max:", outputs.detach().amax().item())
                print("Predicted real maximum:", outputs[:, 0].detach().amax().item())
                print("Predicted imaginary maximum:", outputs[:, 1].detach().amax().item())
                print("Target absolute maximum:", target_patch.detach().abs().amax().item())

                raise RuntimeError(
                    f"Non-finite loss in fold {fold_number}, epoch {epoch + 1}, "
                    f"batch {total_train_batches + 1}"
                )


            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
                error_if_nonfinite=True,
            )
            #Just to inspect the loss
            if epoch == 0 and total_train_batches == 0:
                print(f"Gradient norm before clipping: {grad_norm.item():.4f}")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(
                            name,
                            "gradient norm:",
                            param.grad.norm().item()
                        )
            optimizer.step()

            total_train_loss += loss.item()
            total_train_batches += 1

        avg_train_loss = (
            total_train_loss / max(total_train_batches, 1)
        )

        val_loss, val_nmse, val_snr = validate_model(
            model,
            val_loader,
            criterion,
            device,
        )

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_nmse"].append(val_nmse)
        history["val_snr"].append(val_snr)

        print(
            f"Fold {fold_number}/5 | "
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {avg_train_loss:.6f} | "
            f"Val loss: {val_loss:.6f} | "
            f"Val NMSE: {val_nmse:.6f} | "
            f"Val SNR: {val_snr:.2f} dB"
        )

 
        if val_nmse < best_nmse:
            best_nmse = val_nmse
            best_snr = val_snr
            best_state = copy.deepcopy(model.state_dict())

            checkpoint_path = (
                output_dir / f"fold_{fold_number}_best.pt"
            )

            torch.save(
                {
                    "fold": fold_number,
                    "epoch": epoch + 1,
                    "model_state_dict": best_state,
                    "best_val_nmse": best_nmse,
                    "best_val_snr":best_snr,
                    "val_snr_db": val_snr,
                    "train_paths": train_paths,
                    "val_paths": val_paths,
                },
                checkpoint_path,
            )
        #diagnose_reconstruction(
                       # model,
                        #train_loader,
                        #device,
                        #transform.amplitude_scale,
                    #)
    
    if best_state is not None:
        model.load_state_dict(best_state)
        #visualize_reconstruction(
         #   model=model,
          #  dataset=val_dataset,
           # device=device,
           # amplitude_scale=transform.amplitude_scale,
            #output_dir=output_dir,
            #fold_number=fold_number,
            #sample_index=0,
        #)

    print(
        f"\nFold {fold_number} best NMSE: {best_nmse:.6f}"
    )

    return model, history, best_nmse


############################
# Five-Fold Cross Validation
###########################

def train_model(paths):
    if len(paths) != 5:
        raise ValueError(
            f"Expected 5 independent .vis observations, "
            f"but received {len(paths)} paths. "
            "Pass exactly one path per observation."
        )

    paths = sorted(paths)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("Device:", device)
    print("Number of observations:", len(paths))

    output_dir = Path("cross_validation_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []

    for val_idx in range(K_FOLDS):
        val_paths = [paths[val_idx]]
        train_paths = [
            path
            for idx, path in enumerate(paths)
            if idx != val_idx
        ]

        fold_number = val_idx + 1

        _, history, best_nmse = train_one_fold(
                fold_number=fold_number,
                train_paths=train_paths,
                val_paths=val_paths,
                output_dir=output_dir,
                device=device,
            )

        fold_results.append({
                "fold": fold_number,
                "validation_path": val_paths[0],
                "best_nmse": best_nmse,
                "history": history,
            })

    scores = torch.tensor(
            [result["best_nmse"] for result in fold_results],
            dtype=torch.float64,
        )

    print("\n" + "=" * 70)
    print("FIVE-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    for result in fold_results:
        print(
                f"Fold {result['fold']}: "
                f"{result['validation_path']} | "
                f"Best NMSE: {result['best_nmse']:.6f}"
            )

    print(f"Mean best NMSE: {scores.mean().item():.6f}")
    print(f"Std best NMSE:  {scores.std(unbiased=False).item():.6f}")
    print("=" * 70)

    summary_path = output_dir / "fold_summary.pt"
    torch.save(
            [
                {
                    "fold": result["fold"],
                    "validation_path": result["validation_path"],
                    "best_nmse": result["best_nmse"],
                }
                for result in fold_results
            ],
            summary_path,
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for result in fold_results:
            epochs = range(
                1, len(result["history"]["val_nmse"]) + 1
            )

            axes[0].plot(
                epochs,
                result["history"]["val_nmse"],
                label=f"Fold {result['fold']}",
            )

            axes[1].plot(
                epochs,
                result["history"]["val_snr"],
                label=f"Fold {result['fold']}",
            )

    axes[0].set_title("Validation NMSE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("NMSE")
    axes[0].legend()

    axes[1].set_title("Validation SNR")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("SNR (dB)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
            output_dir / "cross_validation_metrics.png",
            dpi=300,
        )
    plt.show()

    return fold_results

###################
# TESTING FUNCTIONS
###################
def check_model_functionalities(path:list[str]):
    model=makeModel(device=device,model_type=Model_Type.AUTOENCODER_BASE)
    model.eval()
    data=makeDataset(paths,use_cuda=use_cuda)
    sample=data[1]
    x=sample["input_grid"].unsqueeze(0).to(device)
    print("Input shape:",x.shape)
    print("\n")
    with torch.no_grad():
        latent=model.encode(x)
        print("Latent type:", type(latent))
        print("Latent shape:", getattr(latent, "shape", None))
        if not isinstance(latent, torch.Tensor):
            raise TypeError(
                "Inspect the encode() return value before decoding."
        )
        reconstruct=model.decode(latent)
        direct_output=model(x)
    print("Input:", x.shape)
    print("Latent:", latent.shape)
    print("Reconstructed:", reconstruct.shape)

    assert reconstruct.shape == (1, 2, 256, 256)
    assert direct_output.shape == reconstruct.shape

    difference = (
        direct_output - reconstruct
    ).abs().max().item()

    print("Maximum forward/decode difference:", difference)

    original = sample["target_grid"]

    original_bytes = original.numel() * original.element_size()
    latent_bytes = latent.numel() * latent.element_size()

    print(
        "Tensor-size ratio:",
        original_bytes / latent_bytes
    )

#Loss testing funciton for MSE_Loss_Amplitude_Phase
def test_loss(path:list[str]):
    data=makeDataset(paths,use_cuda=use_cuda)
    criterion = MSELoss_Amplitude_Phase(
    amplitude_scale=data.transform.amplitude_scale
)

    sample = data[0]

    input_grid = sample["input_grid"].unsqueeze(0)
    target = sample["target_grid"].unsqueeze(0)

    mask = input_grid[:, 2:3]

    perfect_output = input_grid[:, 0:2]

    loss = criterion(perfect_output, target, mask)

    print("Perfect reconstruction loss:", loss.item())

    wrong_output = perfect_output.clone()
    wrong_output[:, 1] += 0.5

    wrong_loss = criterion(wrong_output, target, mask)

    print("Wrong reconstruction loss:", wrong_loss.item())

    assert wrong_loss > loss

##################
#Diagnostics
#################

@torch.no_grad()
def diagnose_reconstruction(model, loader, device, amplitude_scale):
    model.eval()
    was_training = model.training
    batch = next(iter(loader))
    x = batch["input_grid"].to(device)
    target = batch["target_grid"].to(device)

    output = model(x)

    mask = x[:, 2].bool()

    raw_amp = output[:, 0][mask]
    predicted_phase = output[:, 1][mask]

    true_log_amp = x[:, 0][mask]

    prediction = polar_to_cartesian(
        output,
        amplitude_scale=amplitude_scale,
    )

    pred_complex = torch.complex(
        prediction[:, 0],
        prediction[:, 1],
    )

    true_complex = torch.complex(
        target[:, 0],
        target[:, 1],
    )

    pred_amp = pred_complex.abs()[mask]
    true_amp = true_complex.abs()[mask]

    error = (pred_complex[mask] - true_complex[mask]).abs().square()
    power = true_complex[mask].abs().square()

    print("\n--- Reconstruction diagnostic ---")
    print("Occupancy:", mask.float().mean().item())

    print(
        "True amplitude (mean/max):",
        true_amp.mean().item(),
        true_amp.max().item(),
    )

    print(
        "Predicted amplitude (mean/max):",
        pred_amp.mean().item(),
        pred_amp.max().item(),
    )

    print(
        "True log amplitude (mean):",
        true_log_amp.mean().item(),
    )

    print(
        "Raw amplitude output (min/mean/max):",
        raw_amp.min().item(),
        raw_amp.mean().item(),
        raw_amp.max().item(),
    )

    print(
        "Predicted phase (min/mean/max):",
        predicted_phase.min().item(),
        predicted_phase.mean().item(),
        predicted_phase.max().item(),
    )

    print("Batch NMSE:", (error.sum() / power.sum()).item())
    model.train(was_training)

    
if __name__ == "__main__":
   
    paths = sorted({
        str(Path(p).resolve())
        for arg in sys.argv[1:]
        for p in (glob.glob(arg) or [arg])
        if Path(p).is_file()
    })

    if len(paths) != 5:
        print(len(paths))
        raise ValueError("Provide exactly five .vis observations")

    train_model(paths=paths)
    