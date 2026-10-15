
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

chuchichaestli_src = Path(
    "/home/edione/Projects/chuchichaestli/src"
)
if str(chuchichaestli_src) not in sys.path:
    sys.path.insert(0, str(chuchichaestli_src))

from chuchichaestli.models.autoencoder import Autoencoder


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

GRID_SIZE = 256
BATCH_SIZE = 16
WINDOW_SIZE = 8
TARGET_SIZE = 1
STRIDE = 1

EPOCHS = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

BLOCK_SIZE = 32
HOLDOUT_RATIO = 0.20
LAMBDA_SMOOTH = 1e-6

SEED = 42

transform = Data_Gridding.UVDataGridding(
    grid_size=GRID_SIZE,
    max_uv=70000.0,
)


# ---------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------

class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self.transform(item)


def make_dataset(paths, use_cuda):
    raw_dataset = TimeSlicedVisDataset(
        paths=paths,
        window_size=WINDOW_SIZE,
        target_size=TARGET_SIZE,
        stride=STRIDE,
        pin_memory=use_cuda,
        read_uvw=True,
    )

    return TransformerDataset(raw_dataset, transform)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

def build_model(device):
    return Autoencoder.build(
        dimensions=2,
        in_channels=3,
        out_channels=2,
        latent_dim=16,
        res_act_fn="silu",
        res_norm_type="group",
        encoder_args={"n_channels": 16},
    ).to(device)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Masking
# ---------------------------------------------------------

def make_spatial_masker():
    return SpatialKfoldMask(
        grid_size=GRID_SIZE,
        block_size=BLOCK_SIZE,
        split_ratio=HOLDOUT_RATIO,
    )


def make_model_input_and_target(grid, heldout_mask):
    """
    grid:         original [B, 3, H, W] grid
    heldout_mask: observed cells hidden from the model

    Returns:
        model_input: original observed context, with held-out
                     visibilities removed
        loss_target: original visibilities, with a mask that
                     selects only the held-out measurements
    """
    original_mask = grid[:, 2:3, :, :]

    context_mask = original_mask.clone()
    context_mask[heldout_mask > 0] = 0.0

    model_input = grid.clone()
    model_input[:, 0:2, :, :] *= context_mask
    model_input[:, 2:3, :, :] = context_mask

    loss_target = grid.clone()
    loss_target[:, 2:3, :, :] = heldout_mask

    return model_input, loss_target, context_mask


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

@torch.no_grad()
def validate_model(model, val_loader, criterion, device):
    model.eval()

    masker = make_spatial_masker()

    total_error = 0.0
    total_power = 0.0
    total_loss = 0.0
    total_batches = 0
    total_hidden_cells = 0

    # Reset the RNG each validation pass so the same
    # deterministic loader gets the same masks each epoch.
    mask_generator = torch.Generator(device="cpu")
    mask_generator.manual_seed(SEED + 1000)

    for batch in val_loader:
        grid = batch["input_grid"].to(device)
        original_mask = grid[:, 2:3, :, :]

        _, heldout_mask = masker.train_validation_split(
            original_mask,
            generator=mask_generator,
        )

        model_input, loss_target, _ = (
            make_model_input_and_target(grid, heldout_mask)
        )

        outputs = model(model_input)
        loss = criterion(outputs, loss_target)

        target = grid[:, 0:2, :, :]
        expanded_mask = heldout_mask.expand_as(target)

        error = (
            (outputs - target).square() * expanded_mask
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


# ---------------------------------------------------------
# Training one fold
# ---------------------------------------------------------

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

    train_dataset = make_dataset(train_paths, use_cuda)
    val_dataset = make_dataset(val_paths, use_cuda)

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
        pin_memory=use_cuda,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=use_cuda,
    )

    model = build_model(device)

    criterion = MaskedVisibilityLoss(
        lambda_smooth=LAMBDA_SMOOTH
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    masker = make_spatial_masker()

    best_nmse = float("inf")
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
            grid = batch["input_grid"].to(device)
            original_mask = grid[:, 2:3, :, :]

            context_mask, heldout_mask = (
                masker.train_validation_split(original_mask)
            )

            training_inputs = grid.clone()
            training_inputs[:, 0:2, :, :] *= context_mask
            training_inputs[:, 2:3, :, :] = context_mask

            loss_target = grid.clone()
            loss_target[:, 2:3, :, :] = heldout_mask

            optimizer.zero_grad()

            outputs = model(training_inputs)
            loss = criterion(outputs, loss_target)

            loss.backward()
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
                    "val_snr_db": val_snr,
                    "train_paths": train_paths,
                    "val_paths": val_paths,
                },
                checkpoint_path,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    print(
        f"\nFold {fold_number} best NMSE: {best_nmse:.6f}"
    )

    return model, history, best_nmse


# ---------------------------------------------------------
# Five-fold cross-validation
# ---------------------------------------------------------

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

    print("Device:", device)
    print("Number of observations:", len(paths))

    output_dir = Path("cross_validation_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []

    for val_idx in range(len(paths)):
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

    # Save the aggregate results for later comparison.
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

    # Plot validation performance for all folds.
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


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]

    paths = sorted({
        str(Path(p).resolve())
        for arg in args
        for p in (glob.glob(arg) or [arg])
        if Path(p).is_file()
    })

    if len(paths) != 5:
        sys.exit(
            "Usage: python Model/Training.py "
            "'/path/to/observations/*.vis'\n"
            "Provide exactly five independent .vis files."
        )

    train_model(paths)
