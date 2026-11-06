import argparse
import glob
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from Training_Amplitude_Phase import (
    Model_Type,
    TransformDataset,
    makeDataset,
    makeModel,
    validate_model,
)
from Polarization_Standardization import PolarizationStandardization
from Visibility_Standerdizer import VisibilityStandardizer


def standardizer_from_checkpoint(checkpoint):
    norm = checkpoint["normalization"]
    standardizer = VisibilityStandardizer()
    standardizer.real_mean = norm["real_mean"]
    standardizer.real_std = norm["real_std"]
    standardizer.imag_mean = norm["imag_mean"]
    standardizer.imag_std = norm["imag_std"]
    return standardizer


def spread_indices(n_total, count):
    count = min(count, n_total)
    return torch.linspace(0, n_total - 1, steps=count).round().long().tolist()


def refit_standardizer(dataset, n_samples):
    indices = spread_indices(len(dataset), n_samples)
    data = torch.cat([dataset[i]["vis"] for i in indices], dim=0)
    standardizer = VisibilityStandardizer()
    standardizer.fit(data)
    return standardizer


def evaluate(model, dataset, indices, standardizer, device,uvw_scale=None):
    loader = DataLoader(
        TransformDataset(Subset(dataset, indices),
                         PolarizationStandardization(), standardizer,uvw_scale),
        batch_size=1, shuffle=False, num_workers=0,
    )
    _, nmse, snr, nmse_pol = validate_model(
        model, loader, nn.MSELoss(), device, standardizer
    )
    return nmse, snr, nmse_pol


def fmt_pol(values):
    return " ".join(f"{v:.4f}" for v in values.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vis", nargs="+", required=True,
                        help="Observation files or quoted globs")
    parser.add_argument("--windows", type=int, default=256,
                        help="Windows to evaluate per observation, spread over time")
    parser.add_argument("--refit-samples", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    model = makeModel(device, Model_Type.AUTOENCODER_BASE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    source_std = standardizer_from_checkpoint(checkpoint)
    source_uvw_scale = checkpoint["normalization"].get("uvw_scale")
    print("Checkpoint:", args.checkpoint)
    print("Trained on:", checkpoint.get("train_paths"),
          "| epoch", checkpoint.get("epoch"),
          "| best val NMSE", checkpoint.get("best_val_nmse"))

    paths = sorted({str(Path(p).resolve()) for pattern in args.vis
                    for p in (glob.glob(pattern) or [pattern])
                    if Path(p).is_file()})
    if not paths:
        parser.error("No matching observation files")

    rows = []
    reference_uvw=None
    for path in paths:
        name = Path(path).name
        print("\n" + "=" * 70)
        print("Observation:", name)

      

        dataset = makeDataset([path], use_cuda=False)
        indices = spread_indices(len(dataset), args.windows)
        uvw0 = dataset[indices[0]]["uvw"].float()
        if reference_uvw is None:
            reference_uvw = uvw0
        else:
            print(f"max|UVW - first observation's UVW| = "
                          f"{(uvw0 - reference_uvw).abs().max().item():.3e}")
            
        print(f"Windows: {len(dataset)} total, evaluating {len(indices)}")

        vis = dataset[indices[0]]["vis"]
        diff = (vis[..., 1] - vis[..., 2]).abs().max().item()
        scale = vis[..., 1].abs().max().item()
        print(f"max|P1 - P2| = {diff:.3e}  (max|P1| = {scale:.3e})")

        target_std = refit_standardizer(dataset, args.refit_samples)

        print("Std ratio target/source, real [P0 P1 P2 P3]:",
              fmt_pol(target_std.real_std / source_std.real_std))
        print("Std ratio target/source, imag [P0 P1 P2 P3]:",
              fmt_pol(target_std.imag_std / source_std.imag_std))

        for label, standardizer in (("A: source stats", source_std),
                                    ("B: refit stats ", target_std)):
            nmse, snr, nmse_pol = evaluate(
                model, dataset, indices, standardizer, device, source_uvw_scale
            )
            print(f"{label} | NMSE {nmse:.4f} | SNR {snr:.2f} dB | "
                  f"per pol [P0 P1 P2 P3]: {fmt_pol(nmse_pol)}")
            rows.append((name, label, nmse, nmse_pol))

    print("\n" + "=" * 70)
    print("SUMMARY")
    for name, label, nmse, nmse_pol in rows:
        print(f"{name[:45]:45s} {label} NMSE {nmse:.4f} | {fmt_pol(nmse_pol)}")


if __name__ == "__main__":
    main()