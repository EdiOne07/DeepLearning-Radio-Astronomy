#!/usr/bin/env python3
"""Plots for a trained per-baseline checkpoint. No training.

Figures (saved to --out-dir):
  1. training_curve.png      val NMSE vs epoch from the SLURM logs (+ PCA references)
  2. spectra.png             amplitude/phase vs frequency, original vs reconstructed,
                             for a short, a medium and a long baseline
  3. dirty_images.png        dirty image (Stokes I ~ (XX+YY)/2) of original,
                             reconstruction and residual, same colour scale
  4. nmse_by_length.png      NMSE per projected-baseline-length quartile

The observation is normalized with its own statistics (as in LEAVE_ONE_OUT
validation), so the numbers match the training log for the held-out sky.

Example (from the project root):
    python Testing/visualize_results.py \
        --checkpoint results_cluster/leave_one_out/physical_perbaseline_lat4_uvw_shift0.05/s40/fold_1_best.pt \
        --vis "<path>/gleam_ra0.0_dec-27.0_fov5.0_s40_AA2_i00,01,02,03,04.vis" \
        --logs cluster_logs/perbaseline-loo-cpu-164127.log \
               cluster_logs/perbaseline-loo-cpu-164477.log \
               cluster_logs/perbaseline-loo-cpu-164903.log \
        --ablation-logs cluster_logs/perbaseline-loo-cpu-164232.log \
        --out-dir figures_s40
"""

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT, PROJECT_ROOT / "Model"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset

import Model.Training_Amplitude_Phase as tap
from Model.Polarization_Standardization import PolarizationStandardization
from Model.Visibility_Standerdizer import VisibilityStandardizer

C = 299_792_458.0
EPOCH_RE = re.compile(r"Fold \d+/\d+ \| Epoch (\d+)/\d+ .*?Val NMSE: ([0-9.]+)")


# ---------------------------------------------------------------- training curve
def parse_logs(paths):
    """Concatenate val NMSE over several logs (resumed runs), in the given order."""
    values = []
    for path in paths:
        for line in Path(path).read_text(errors="ignore").splitlines():
            m = EPOCH_RE.search(line)
            if m:
                values.append(float(m.group(2)))
    return values


def plot_training_curve(logs, ablation_logs, out):
    main = parse_logs(logs)
    if not main:
        print("No epoch lines found in --logs; skipping training curve.")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(main) + 1), main, label="per-baseline AE, shift 0.05", linewidth=2)
    if ablation_logs:
        abl = parse_logs(ablation_logs)
        if abl:
            ax.plot(range(1, len(abl) + 1), abl, label="per-baseline AE, no shift", linestyle="--")
    ax.axhline(0.37, color="gray", linestyle=":", label="PCA, fitted on another sky (0.37)")
    ax.axhline(0.26, color="black", linestyle=":", label="PCA, fitted on this sky (0.26)")
    ax.set_xlabel("Epoch (resumed runs concatenated)")
    ax.set_ylabel("Validation NMSE (held-out sky)")
    ax.set_title("8x compression, held-out observation")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Training curve: best val NMSE {min(main):.4f} after {len(main)} epochs -> {out}")


# ---------------------------------------------------------------- model + data
def load_model(checkpoint, device):
    state = checkpoint["model_state_dict"]
    # Infer architecture settings from the weights, so env vars don't have to match.
    for name, w in state.items():
        if name.endswith("latent_proj.weight"):
            tap.LATENT_DIM = w.shape[0]
        if name.endswith("conv_in.weight") and name.startswith("encoder"):
            tap.IN_CHANNELS = w.shape[1]
    model = tap.makeModel(device, tap.Model_Type.AUTOENCODER_BASE)
    model.load_state_dict(state)
    model.eval()
    print(f"Model: in_channels={tap.IN_CHANNELS}, latent_dim={tap.LATENT_DIM}, "
          f"epoch {checkpoint.get('epoch')}, best val NMSE {checkpoint.get('best_val_nmse')}")
    return model


def fit_own_standardizer(raw, n=32):
    idx = torch.linspace(0, len(raw) - 1, steps=min(n, len(raw))).long().tolist()
    s = VisibilityStandardizer()
    s.fit(torch.cat([raw[i]["vis"] for i in idx], dim=0))
    return s


@torch.no_grad()
def reconstruct(model, raw, index, standardizer, uvw_scale, device):
    """Returns original and reconstructed visibilities [F, B, 4] (complex, physical units)
    and the window's uvw [B, 3] in metres."""
    ds = tap.TransformDataset(Subset(raw, [index]), PolarizationStandardization(),
                              standardizer, uvw_scale)
    x = tap.to_baselines(ds[0][None].to(device))              # [B, C, F]
    out = model(x)                                            # [B, 8, F]
    target = x[:, :tap.N_VIS_CHANNELS]
    to_phys = lambda t: standardizer.inverse_transform(
        tap.channels_to_complex(tap.to_slice(t).cpu()))
    uvw = raw[index]["uvw"][0].double()
    return to_phys(target), to_phys(out), uvw


# ---------------------------------------------------------------- plots
def plot_spectra(orig, rec, uvw, freqs_mhz, out):
    length = torch.linalg.vector_norm(uvw[:, :2], dim=-1)
    order = torch.argsort(length)
    picks = [order[int(q * (len(order) - 1))].item() for q in (0.05, 0.5, 0.95)]
    names = ["short", "medium", "long"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5), sharex=True)
    for col, (b, name) in enumerate(zip(picks, names)):
        o, r = orig[:, b, 0].numpy(), rec[:, b, 0].numpy()        # XX
        err = np.sum(np.abs(o - r) ** 2) / max(np.sum(np.abs(o) ** 2), 1e-30)
        axes[0, col].plot(freqs_mhz, np.abs(o), label="original")
        axes[0, col].plot(freqs_mhz, np.abs(r), "--", label="reconstructed")
        axes[0, col].set_title(f"{name} baseline ({length[b]:.0f} m), XX NMSE {err:.3f}")
        axes[1, col].plot(freqs_mhz, np.angle(o), ".", label="original")
        axes[1, col].plot(freqs_mhz, np.angle(r), "x", markersize=4, label="reconstructed")
        axes[1, col].set_xlabel("Frequency [MHz]")
    axes[0, 0].set_ylabel("|V| (XX)")
    axes[1, 0].set_ylabel("phase [rad] (XX)")
    axes[0, 0].legend()
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Spectra ->", out)


def dirty_image(u, v, vals, lim, n):
    grid = np.zeros((n, n), dtype=np.complex128)
    weights = np.zeros((n, n))
    for sign, data in ((1, vals), (-1, np.conjugate(vals))):
        col = np.rint((sign * u / (2 * lim) + 0.5) * (n - 1)).astype(int)
        row = np.rint((sign * v / (2 * lim) + 0.5) * (n - 1)).astype(int)
        np.add.at(grid, (row, col), data)
        np.add.at(weights, (row, col), 1)
    img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))) * (n ** 2 / weights.sum())
    return img.real


def plot_dirty_images(samples, freqs_hz, channel, out, n=256):
    u_all, v_all, i_orig, i_rec = [], [], [], []
    for orig, rec, uvw in samples:
        scale = freqs_hz[channel] / C                            # metres -> wavelengths
        u_all.append(uvw[:, 0].numpy() * scale)
        v_all.append(uvw[:, 1].numpy() * scale)
        i_orig.append(0.5 * (orig[channel, :, 0] + orig[channel, :, 3]).numpy())
        i_rec.append(0.5 * (rec[channel, :, 0] + rec[channel, :, 3]).numpy())
    u, v = np.concatenate(u_all), np.concatenate(v_all)
    vo, vr = np.concatenate(i_orig), np.concatenate(i_rec)
    lim = np.quantile(np.maximum(np.abs(u), np.abs(v)), 0.999)
    keep = (np.abs(u) <= lim) & (np.abs(v) <= lim)
    u, v, vo, vr = u[keep], v[keep], vo[keep], vr[keep]

    img_o = dirty_image(u, v, vo, lim, n)
    img_r = dirty_image(u, v, vr, lim, n)
    img_d = img_o - img_r
    vmax = np.max(np.abs(img_o))
    rms_ratio = np.sqrt(np.mean(img_d ** 2)) / np.sqrt(np.mean(img_o ** 2))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, img, title in zip(axes, (img_o, img_r, img_d),
                              ("original", "reconstructed (8x)", "residual (original - reconstructed)")):
        im = ax.imshow(img, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"Dirty image, Stokes I ~ (XX+YY)/2, {freqs_hz[channel] / 1e6:.1f} MHz, "
                 f"{len(samples)} time windows | residual RMS / image RMS = {rms_ratio:.3f}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Dirty images (residual/image RMS {rms_ratio:.3f}) ->", out)


def plot_nmse_by_length(samples, out):
    lengths = torch.cat([torch.linalg.vector_norm(uvw[:, :2], dim=-1) for _, _, uvw in samples])
    edges = torch.quantile(lengths, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
    err, power = torch.zeros(4, dtype=torch.float64), torch.zeros(4, dtype=torch.float64)
    for orig, rec, uvw in samples:
        bins = torch.bucketize(torch.linalg.vector_norm(uvw[:, :2], dim=-1), edges)
        e = (orig - rec).abs().square().sum(dim=(0, 2)).double()      # per baseline
        p = orig.abs().square().sum(dim=(0, 2)).double()
        err.index_add_(0, bins, e)
        power.index_add_(0, bins, p)
    nmse = (err / power).tolist()
    e = edges.tolist()
    labels = [f"< {e[0]:.0f} m", f"{e[0]:.0f}-{e[1]:.0f} m", f"{e[1]:.0f}-{e[2]:.0f} m", f"> {e[2]:.0f} m"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, nmse)
    for i, val in enumerate(nmse):
        ax.text(i, val, f"{val:.3f}", ha="center", va="bottom")
    ax.set_ylabel("NMSE")
    ax.set_xlabel("Projected baseline length (quartiles)")
    ax.set_title("Reconstruction error by baseline length")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("NMSE by length:", " | ".join(f"{l}: {v:.3f}" for l, v in zip(labels, nmse)), "->", out)


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vis", required=True, help="Observation to reconstruct (normally the held-out one)")
    parser.add_argument("--logs", nargs="*", default=[], help="Training logs, in chronological order")
    parser.add_argument("--ablation-logs", nargs="*", default=[])
    parser.add_argument("--window", type=int, default=225, help="Window for the spectra plot")
    parser.add_argument("--image-windows", type=int, default=16,
                        help="Windows combined for dirty images and length bins")
    parser.add_argument("--channel", type=int, default=32, help="Frequency channel for the images")
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.logs:
        plot_training_curve(args.logs, args.ablation_logs, out / "training_curve.png")

    device = torch.device("cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = load_model(checkpoint, device)
    uvw_scale = checkpoint["normalization"].get("uvw_scale")

    raw = tap.makeDataset([args.vis], use_cuda=False)
    standardizer = fit_own_standardizer(raw)
    freqs_hz = tap.FREQS_HZ.numpy()

    window = min(args.window, len(raw) - 1)
    orig, rec, uvw = reconstruct(model, raw, window, standardizer, uvw_scale, device)
    total = ((orig - rec).abs().square().sum() / orig.abs().square().sum()).item()
    print(f"Window {window}: NMSE {total:.4f}")
    plot_spectra(orig, rec, uvw, freqs_hz / 1e6, out / "spectra.png")

    idx = torch.linspace(0, len(raw) - 1, steps=min(args.image_windows, len(raw))).long().tolist()
    samples = [reconstruct(model, raw, i, standardizer, uvw_scale, device) for i in idx]
    plot_dirty_images(samples, freqs_hz, args.channel, out / "dirty_images.png")
    plot_nmse_by_length(samples, out / "nmse_by_length.png")
    print("Figures in", out.resolve())


if __name__ == "__main__":
    main()