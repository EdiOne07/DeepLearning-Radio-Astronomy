
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

from torch.utils.data import Subset, ConcatDataset, DataLoader

from Self_Supervising_Loss import MaskedVisibilityLoss
from Spatial_KFold import SpatialKfoldMask  # Adjust filename capitalization if needed
from rat_stack import TimeSlicedVisDataset
from Data_Gridding_Patches import UV_Patch_Transform
from MSE_Loss_Amplitude_Phase import MaskedComplexMSE,polar_to_cartesian
from Data_Loading_and_Vizualization.ReconstructVisualization import visualize_reconstruction
from enum import Enum
from Visibility_Standerdizer import VisibilityStandardizer
from Polarization_Standardization import PolarizationStandardization
import torch.nn.functional as F

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
STRIDE=8
BATCH_SIZE=1
EPOCHS=50
LEARNING_RATE=1e-4
LAMBDA_SMOOTHNESS=1e-6
K_FOLDS=5
GRID_SIZE=8192
MAX_UV=70000
BLOCK_SIZE = 32
HOLDOUT_RATIO = 0.20
SEED=42
AMPLITUDE_SCALE=0.1
WEIGHT_DECAY = 1e-4
PATCH_SIZE=16

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class Model_Type(Enum):
    AUTOENCODER_BASE="AUTOENCODER_BASE"
    FUTURE_AUTOENCODER="FUTURE_AUTOENCODER"

class Training_Strategy(Enum):
    ONE_VISIBILITY="ONE_VISIBILITY"
    FIVE_FOLD_CROSS="FIVE_FOLD_CROSS"
    POOLED_STRATIFIED_SPLIT="POOLED_STRATIFIED_SPLIT"

###################
# DATA_HANDLING
###################

class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform, standardizer):
        self.dataset=dataset
        self.transform=transform
        self.standardizer=standardizer
    def __getitem__(self, index):
        sample = self.dataset[index]
        normalized = self.standardizer.transform(sample["vis"])
        return self.transform(normalized).squeeze(0)
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
    return dataset
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
                in_channels=8,
                out_channels=8,
                latent_dim=16,
                res_act_fn="silu",
                res_norm_type="group",
                encoder_args={
                    "n_channels": 16,
                    "down_block_types": (
                        "AutoencoderDownBlock",
                        "AutoencoderDownBlock",
                        "AttnAutoencoderDownBlock",
                    ),
                    "block_out_channel_mults": (1, 2, 2),
                    "num_layers_per_block": 1,
                },
                decoder_args={
                    "up_block_types": (
                        "AttnAutoencoderUpBlock",
                        "AutoencoderUpBlock",
                        "AutoencoderUpBlock",
                    ),
                    "block_out_channel_mults": (1, 2, 2),
                    "num_layers_per_block": 1,
                },
            ).to(device)
    elif model_type==Model_Type.FUTURE_AUTOENCODER:
        # tbd
        return 0

###################
# METRICS
###################
def channels_to_complex(channels):
    # Input: [1, 8, F, B]
    assert channels.ndim == 4
    assert channels.shape[0] == 1
    assert channels.shape[1] == 8

    _, _, freq, baselines = channels.shape

    components = channels.squeeze(0).reshape(
        4, 2, freq, baselines
    )

    components = components.permute(2, 3, 0, 1)

    return torch.complex(
        components[..., 0].contiguous(),
        components[..., 1].contiguous(),
    )

@torch.no_grad()
def compute_reconstruction_metrics(pred, target, standardizer):
    # pred / target: [1, 8, F, B], standardized
    pred_complex = channels_to_complex(pred.cpu())
    target_complex = channels_to_complex(target.cpu())

    pred_original = standardizer.inverse_transform(pred_complex)
    target_original = standardizer.inverse_transform(target_complex)

    error = (
        (pred_original - target_original)
        .abs()
        .square()
        .sum()
        .item()
    )

    power = (
        target_original
        .abs()
        .square()
        .sum()
        .item()
    )

    return error, power

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
def validate_model(model, val_loader, criterion, device, standardizer):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    total_error = 0.0
    total_power = 0.0

    for batch in val_loader:
        x = batch.to(device)

        height, width = x.shape[-2:]
        pad_h = (-height) % 4
        pad_w = (-width) % 4

        padded = F.pad(x, (0, pad_w, 0, pad_h))

        output = model(padded)
        output = output[..., :height, :width]

        if total_samples == 0:
            print(
                "Validation input power:",
                x.square().mean().item()
            )
            print(
                "Validation predicted power:",
                output.square().mean().item()
            )
            print(
                "Validation predicted/input power ratio:",
                (
                    output.square().sum()
                    / x.square().sum().clamp_min(1e-12)
                ).item()
            )

        loss = criterion(output, x)

        total_loss += loss.item() * x.shape[0]
        total_samples += x.shape[0]

        error, power = compute_reconstruction_metrics(
            output, x, standardizer
        )

        total_error += error
        total_power += power

    if total_samples == 0 or total_power <= 0:
        raise RuntimeError("Invalid validation dataset")

    val_loss = total_loss / total_samples
    nmse = total_error / total_power
    snr_db = -10.0 * math.log10(nmse) if nmse > 0 else math.inf

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
    training_strategy=Training_Strategy.FIVE_FOLD_CROSS
):
    print("\n" + "=" * 70)
    print(f"FOLD {fold_number}/5")
    print("Training observations:", train_paths)
    print("Validation observation:", val_paths)
    print("=" * 70)

    use_cuda = device.type == "cuda"

    train_dataset = makeDataset(train_paths, use_cuda)
    val_dataset = makeDataset(val_paths, use_cuda)
    if training_strategy==Training_Strategy.ONE_VISIBILITY:
        raw_dataset = train_dataset

        train_count = min(32, len(raw_dataset) - 1)
        val_count = min(8, len(raw_dataset) - train_count)

        if train_count < 1 or val_count < 1:
            raise RuntimeError("Not enough windows for testing")

        train_dataset = Subset(
            raw_dataset,
            list(range(train_count))
        )
        val_dataset = Subset(
            raw_dataset,
            list(range(train_count, train_count + val_count))
        )

    elif training_strategy==Training_Strategy.POOLED_STRATIFIED_SPLIT:
            train_parts = []
            val_parts = []
    
            for path in train_paths:
                dataset = makeDataset([path], use_cuda)
    
                split = int(0.8 * len(dataset))
    
                train_parts.append(
                    Subset(dataset, range(0, split))
                )
    
                val_parts.append(
                    Subset(dataset, range(split, len(dataset)))
                )
    
            train_dataset = ConcatDataset(train_parts)
            val_dataset = ConcatDataset(val_parts)
    elif training_strategy!=Training_Strategy.FIVE_FOLD_CROSS:
        raise ValueError("Unknown Strategy or didn't select strategy")


    standardizer = VisibilityStandardizer()


    num_stats_samples = min(32, len(train_dataset))

    indices = torch.linspace(
            0,
            len(train_dataset) - 1,
            steps=num_stats_samples
        ).long()

    statistics_data = torch.cat(
            [train_dataset[i]["vis"] for i in indices.tolist()],
            dim=0
        )

    standardizer.fit(statistics_data)
    del statistics_data


    channel_transform = PolarizationStandardization()

    train_dataset = TransformDataset(
        train_dataset, channel_transform, standardizer
    )

    val_dataset = TransformDataset(
        val_dataset, channel_transform, standardizer
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(
            f"Fold {fold_number} has an empty dataset. "
            "Check file paths and time-window settings."
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )

    model = makeModel(device,Model_Type.AUTOENCODER_BASE)

    criterion = nn.MSELoss()

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
            x = batch.to(device)

            height, width = x.shape[-2:]

            pad_h = (-height) % 4
            pad_w = (-width) % 4

            x_padded = F.pad(x, (0, pad_w, 0, pad_h))

            optimizer.zero_grad()

            outputs = model(x_padded)
            outputs = outputs[..., :height, :width]
            if epoch == 0 and total_train_batches == 0:
                print("Input mean:", x.mean(dim=(0, 2, 3)))
                print("Input std:", x.std(dim=(0, 2, 3), correction=0))

                print("Output mean:", outputs.mean(dim=(0, 2, 3)))
                print("Output std:", outputs.std(dim=(0, 2, 3), correction=0))

            loss = criterion(outputs, x)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
                error_if_nonfinite=True,
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
                standardizer,
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
                    "normalization": {
                    "real_mean": standardizer.real_mean.cpu(),
                    "real_std": standardizer.real_std.cpu(),
                    "imag_mean": standardizer.imag_mean.cpu(),
                    "imag_std": standardizer.imag_std.cpu(),
                    },
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

def train_model(paths,training_strategy):

    paths = sorted(paths)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path("results") / training_strategy.value.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    if training_strategy == Training_Strategy.ONE_VISIBILITY:
        if len(paths) != 1:
            raise ValueError("ONE_VISIBILITY requires one observation")

        train_one_fold(
            1, paths, paths, output_dir, device, training_strategy
        )
        return []

    if training_strategy == Training_Strategy.POOLED_STRATIFIED_SPLIT:
        if len(paths) != 5:
            raise ValueError("Pooled training requires five observations")

        train_one_fold(
            1, paths, paths, output_dir, device, training_strategy
        )
        return []

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("Device:", device)
    print("Number of observations:", len(paths))
    if training_strategy != Training_Strategy.FIVE_FOLD_CROSS:
        raise ValueError(f"Unsupported strategy: {training_strategy}")

    if len(paths) != 5:
        raise ValueError("Five-fold cross-validation requires five observations")

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
                training_strategy=training_strategy
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


def test_overfit_single_slice(model, model_input, device, steps=100):
    import torch
    import torch.nn.functional as F

    model = model.to(device)
    model.train()

    x = model_input.to(device)
    height, width = x.shape[-2:]

    pad_h = (-height) % 4
    pad_w = (-width) % 4
    x_padded = F.pad(x, (0, pad_w, 0, pad_h))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.0
    )

    losses = []

    for step in range(steps):
        optimizer.zero_grad()

        reconstruction = model(x_padded)
        reconstruction = reconstruction[..., :height, :width]

        loss = F.mse_loss(reconstruction, x)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        if step == 0 or (step + 1) % 10 == 0:
            print(f"Step {step + 1}: MSE = {loss.item():.6f}")

    initial = losses[0]
    final = losses[-1]

    improvement = (initial - final) / initial * 100

    print(f"\nInitial MSE: {initial:.6f}")
    print(f"Final MSE:   {final:.6f}")
    print(f"Improvement: {improvement:.2f}%")

    return losses




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
    train_model(paths=paths,training_strategy=Training_Strategy.POOLED_STRATIFIED_SPLIT)
    