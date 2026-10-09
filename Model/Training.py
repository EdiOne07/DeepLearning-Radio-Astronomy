import sys
import glob
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from Self_Supervising_Loss import MaskedVisibilityLoss

# Ensure chuchichaestli import path
chuchichaestli_src = Path("/home/edione/Projects/chuchichaestli/src")
if str(chuchichaestli_src) not in sys.path:
    sys.path.insert(0, str(chuchichaestli_src))

from chuchichaestli.models.autoencoder import Autoencoder
from rat_stack import TimeSlicedVisDataset
import Data_Gridding

transform = Data_Gridding.UVDataGridding(grid_size=256, max_uv=70000.0)

class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self.transform(item)


def compute_reconstruction_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """
    Computes NMSE and SNR (dB) across both Real and Imaginary channels at sampled points.
    """
    with torch.no_grad():
        diff = (pred - target) * mask
        target_masked = target * mask

        mse = torch.sum(diff ** 2)
        target_power = torch.sum(target_masked ** 2) + 1e-8

        nmse = (mse / target_power).item()
        snr_db = 10.0 * torch.log10(target_power / (mse + 1e-8)).item()

        return nmse, snr_db


def train_model(paths: list[str]) -> None:
    use_cuda = torch.cuda.is_available()
    dataset = TimeSlicedVisDataset(
        paths=paths,
        window_size=8,
        target_size=1,
        stride=1,
        pin_memory=use_cuda,
        read_uvw=True,
    )

    train_dataset = TransformerDataset(dataset, transform)
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=2
    )

    device = torch.device("cuda" if use_cuda else "cpu")
    model = Autoencoder.build(
        dimensions=2,
        in_channels=3,
        out_channels=2,
        latent_dim=16,
        res_act_fn="silu",
        res_norm_type="group",
        encoder_args={"n_channels": 16},
    ).to(device)

    criterion = MaskedVisibilityLoss(lambda_smooth=1e-6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    model.train()

    for epoch in range(10):
        running_loss = 0.0
        running_nmse = 0.0
        running_snr = 0.0
        total_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            inputs = batch["input_grid"].to(device) 
            targets = inputs[:, 0:2, :, :]         
            mask = inputs[:, 2:3, :, :]   

            optimizer.zero_grad()

            outputs = model(inputs)
            loss=criterion(outputs,inputs)
            
            loss.backward()
            optimizer.step()

            nmse, snr_db = compute_reconstruction_metrics(outputs, targets, mask)

            running_loss += loss.item()
            running_nmse += nmse
            running_snr += snr_db

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/5] | Batch [{batch_idx}/{total_batches}] | "
                    f"Loss: {loss.item():.6f} | Batch NMSE: {nmse:.4f} | Batch SNR: {snr_db:.2f} dB"
                )

        epoch_loss = running_loss / total_batches
        epoch_nmse = running_nmse / total_batches
        epoch_snr = running_snr / total_batches

        print("\n" + "=" * 65)
        print(f" END OF EPOCH {epoch+1} SUMMARY:")
        print(f"  - Average Loss: {epoch_loss:.6f}")
        print(f"  - Average NMSE: {epoch_nmse:.4f} (Lower is better)")
        print(f"  - Average SNR:  {epoch_snr:.2f} dB (Higher is better)")
        print("=" * 65 + "\n")

    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        input_grid = sample_batch["input_grid"].to(device)
        output_grid = model(input_grid)

        input_real = input_grid[0, 0].cpu().numpy()
        pred_real = output_grid[0, 0].cpu().numpy()
        mask = input_grid[0, 2].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(input_real, cmap="coolwarm", origin="lower")
    axes[0].set_title("Sparse Input $V_{real}$ (Sampled Points)")

    im1 = axes[1].imshow(pred_real, cmap="coolwarm", origin="lower")
    axes[1].set_title("Autoencoder Completed Grid $\hat{V}_{real}$")

    im2 = axes[2].imshow(mask, cmap="gray_r", origin="lower")
    axes[2].set_title("Input Sampling Mask")

    plt.tight_layout()
    plt.savefig("self_supervised_completion.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    args = sys.argv[1:]
    paths = [p for arg in args for p in (glob.glob(arg) or [arg])]

    if not paths:
        sys.exit("Usage: python Model/Training.py <file0.vis> [file1.vis ...]")

    train_model(paths)