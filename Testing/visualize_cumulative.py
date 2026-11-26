#!/usr/bin/env python3
"""Cumulative dirty-image animation: original vs reconstruction vs residual.

Runs a trained per-baseline checkpoint over the whole observation (every
--stride-th window). Each frame adds that window's visibilities to the UV grid,
so the image sharpens as the Earth rotates and the UV coverage fills in.

Outputs (in --out-dir):
  cumulative.gif               always (Pillow, ships with matplotlib)
  cumulative.mp4               if ffmpeg is available
  final_frame.png              the last (fully accumulated) frame
  residual_vs_time.png         residual RMS / image RMS as coverage grows
  frames/frame_XXX.png         only with --save-frames

Stokes I ~ (XX+YY)/2 at one frequency channel. The observation is normalized
with its own statistics, as in LEAVE_ONE_OUT validation.

Example (from the project root):
    python Testing/visualize_cumulative.py \
        --checkpoint results/Shifting_Good/physical_perbaseline_lat4_uvw_shift0.05/s40/fold_1_best.pt \
        --vis "/path/gleam_ra0.0_dec-27.0_fov5.0_s40_AA2_i00,01,02,03,04.vis" \
        --stride 8 --out-dir results/Shifting_Good/cumulative_s40
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT, PROJECT_ROOT / "Model"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import torch
from torch.utils.data import Subset

import Model.Training_Amplitude_Phase as tap
from Model.Polarization_Standardization import PolarizationStandardization
from Model.Visibility_Standerdizer import VisibilityStandardizer

C = 299_792_458.0


def load_model(checkpoint, device):
    state = checkpoint["model_state_dict"]
    for name, w in state.items():               # infer settings from the weights
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
    ds = tap.TransformDataset(Subset(raw, [index]), PolarizationStandardization(),
                              standardizer, uvw_scale)
    x = tap.to_baselines(ds[0][None].to(device))           # [B, C, F]
    out = model(x)
    target = x[:, :tap.N_VIS_CHANNELS]
    to_phys = lambda t: standardizer.inverse_transform(
        tap.channels_to_complex(tap.to_slice(t).cpu()))    # [F, B, 4] complex
    return to_phys(target), to_phys(out), raw[index]["uvw"][0].double()


def grid_indices(u, v, lim, n):
    col = np.rint((u / (2 * lim) + 0.5) * (n - 1)).astype(int)
    row = np.rint((v / (2 * lim) + 0.5) * (n - 1)).astype(int)
    return row, col


def to_image(grid, weights, n):
    total = weights.sum()
    if total == 0:
        return np.zeros((n, n))
    return (np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))) * (n ** 2 / total)).real


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vis", required=True)
    parser.add_argument("--stride", type=int, default=8, help="Use every N-th window")
    parser.add_argument("--channel", type=int, default=32, help="Frequency channel (0-63)")
    parser.add_argument("--grid", type=int, default=256, help="Image size in pixels")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--out-dir", default="cumulative")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = load_model(checkpoint, device)
    uvw_scale = checkpoint["normalization"].get("uvw_scale")

    raw = tap.makeDataset([args.vis], use_cuda=False)
    standardizer = fit_own_standardizer(raw)
    freq_hz = float(tap.FREQS_HZ[args.channel])
    to_lambda = freq_hz / C
    windows = list(range(0, len(raw), args.stride))
    print(f"{len(raw)} windows, using {len(windows)} (stride {args.stride}), "
          f"channel {args.channel} = {freq_hz / 1e6:.1f} MHz")

    # 1) Reconstruct every selected window once.
    per_window = []
    err_tot = pow_tot = 0.0
    for k, i in enumerate(windows):
        orig, rec, uvw = reconstruct(model, raw, i, standardizer, uvw_scale, device)
        err_tot += (orig - rec).abs().square().sum().item()
        pow_tot += orig.abs().square().sum().item()
        per_window.append((
            uvw[:, 0].numpy() * to_lambda,
            uvw[:, 1].numpy() * to_lambda,
            (0.5 * (orig[args.channel, :, 0] + orig[args.channel, :, 3])).numpy(),
            (0.5 * (rec[args.channel, :, 0] + rec[args.channel, :, 3])).numpy(),
        ))
        if k % 10 == 0 or k == len(windows) - 1:
            print(f"  reconstructed {k + 1}/{len(windows)}", flush=True)
    print(f"NMSE over the used windows (all pols, all channels): {err_tot / pow_tot:.4f}")

    # 2) One fixed UV extent for every frame.
    all_r = np.concatenate([np.maximum(np.abs(u), np.abs(v)) for u, v, _, _ in per_window])
    lim = np.quantile(all_r, 0.999)
    n = args.grid

    # 3) Accumulate frame by frame (and the conjugate points).
    g_o = np.zeros((n, n), dtype=np.complex128)
    g_r = np.zeros((n, n), dtype=np.complex128)
    w = np.zeros((n, n))
    frames, ratios = [], []
    for u, v, io, ir in per_window:
        keep = (np.abs(u) <= lim) & (np.abs(v) <= lim)
        u, v, io, ir = u[keep], v[keep], io[keep], ir[keep]
        for sign, ao, ar in ((1, io, ir), (-1, np.conjugate(io), np.conjugate(ir))):
            row, col = grid_indices(sign * u, sign * v, lim, n)
            np.add.at(g_o, (row, col), ao)
            np.add.at(g_r, (row, col), ar)
            np.add.at(w, (row, col), 1)
        img_o, img_r = to_image(g_o, w, n), to_image(g_r, w, n)
        res = img_o - img_r
        frames.append((img_o, img_r, res))
        ratios.append(np.sqrt(np.mean(res ** 2)) / np.sqrt(np.mean(img_o ** 2)))

    # 4) Animation.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    vmax0 = np.max(np.abs(frames[0][0])) or 1.0
    ims = []
    for ax, title in zip(axes, ("original", f"reconstructed ({tap.N_VIS_CHANNELS * 64 // (tap.LATENT_DIM * 16)}x)",
                                "residual")):
        im = ax.imshow(np.zeros((n, n)), origin="lower", cmap="RdBu_r", vmin=-vmax0, vmax=vmax0)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        ims.append(im)
    suptitle = fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    def update(k):
        img_o, img_r, res = frames[k]
        vmax = np.max(np.abs(img_o)) or 1.0
        for im, img in zip(ims, (img_o, img_r, res)):
            im.set_data(img)
            im.set_clim(-vmax, vmax)
        suptitle.set_text(
            f"Cumulative dirty image, Stokes I, {freq_hz / 1e6:.1f} MHz | "
            f"windows 0-{windows[k]} of {len(raw) - 1} ({k + 1} frames) | "
            f"residual/image RMS {ratios[k]:.3f}")
        return ims + [suptitle]

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / args.fps, blit=False)
    anim.save(out / "cumulative.gif", writer=animation.PillowWriter(fps=args.fps), dpi=90)
    print("GIF ->", out / "cumulative.gif")
    if animation.writers.is_available("ffmpeg"):
        anim.save(out / "cumulative.mp4", writer=animation.FFMpegWriter(fps=args.fps), dpi=120)
        print("MP4 ->", out / "cumulative.mp4")
    else:
        print("ffmpeg not found; skipped MP4")

    update(len(frames) - 1)
    fig.savefig(out / "final_frame.png", dpi=150)
    if args.save_frames:
        (out / "frames").mkdir(exist_ok=True)
        for k in range(len(frames)):
            update(k)
            fig.savefig(out / "frames" / f"frame_{k:03d}.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([windows[k] for k in range(len(ratios))], ratios, marker="o")
    ax.set_xlabel("Last window included")
    ax.set_ylabel("Residual RMS / image RMS")
    ax.set_title("Image-domain error as UV coverage accumulates")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "residual_vs_time.png", dpi=150)
    plt.close(fig)
    print(f"Final residual/image RMS: {ratios[-1]:.3f}")
    print("Done ->", out.resolve())


if __name__ == "__main__":
    main()