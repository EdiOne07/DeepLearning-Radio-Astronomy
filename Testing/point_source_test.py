#!/usr/bin/env python3
"""Point-source sanity check for the project's existing UV_Patch_Transform.

Run from the project with modules rat_stack and Data_Gridding_Patches importable:

    python point_source_test.py '/path/to/gleam*_s42_*.vis'
    python point_source_test.py '/path/to/gleam*_s42_*.vis' --grid-size 512 --max-uv 70000

This deliberately uses ONLY frequency 0 and polarization 0, matching the
current UV_Patch_Transform. It uses its TARGET patches (complex real/imag),
not the log-amplitude / phase model INPUT patches.

This validates mapping, patch reassembly and Fourier convention in this test.
It does NOT verify whether raw UVW are in meters or wavelengths, or prove that
some other dirty-imaging code uses the same pipeline.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Match the import setup in the existing training script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rat_stack import TimeSlicedVisDataset
from Model.Data_Gridding_Patches import UV_Patch_Transform


def make_dataset(path: str) -> TimeSlicedVisDataset:
    return TimeSlicedVisDataset(
        paths=[path],
        window_size=1,
        target_size=1,
        stride=8,
        pin_memory=False,
        read_uvw=True,
    )


def patches_to_uv_grid(result: dict, grid_size: int, patch_size: int):
    """Reassemble complex target_patch plus counts into a full UV grid."""
    position = result["patch_position"].long().cpu()
    target = result["target_patch"].cpu()    # [n_patches, 2, P, P]
    counts = result["counts"].cpu()          # [n_patches, P, P]
    p = patch_size
    n = position.shape[0]

    local_row = torch.arange(p).reshape(1, p, 1)
    local_col = torch.arange(p).reshape(1, 1, p)
    rows = (position[:, 0, None, None] * p + local_row).expand(n, p, p)
    cols = (position[:, 1, None, None] * p + local_col).expand(n, p, p)
    mask = counts > 0

    grid = torch.zeros((grid_size, grid_size), dtype=torch.complex64)
    hit_counts = torch.zeros((grid_size, grid_size), dtype=torch.float32)
    values = torch.complex(target[:, 0], target[:, 1])

    grid[rows[mask], cols[mask]] = values[mask]
    hit_counts[rows[mask], cols[mask]] = counts[mask]
    return grid, hit_counts


def add_hermitian_counterparts(uv: torch.Tensor, counts: torch.Tensor):
    """Add V(-u,-v)=conj(V(u,v)) using FFT-centered integer-index reflection.

    Reuses the existing transform's pixel assignments. The transform maps
    zero to a half-pixel center for even N (via N-1), so this is an approximate
    counterpart mapping; a separate coordinate-scaling audit is still needed.
    """
    n = uv.shape[0]
    rows, cols = torch.where(counts > 0)
    flat_idx = rows * n + cols
    mirror_idx = ((-rows) % n) * n + ((-cols) % n)

    summed = torch.zeros(n * n, dtype=torch.complex64)
    weights = torch.zeros(n * n, dtype=torch.float32)
    contrib = uv[rows, cols] * counts[rows, cols]
    weight = counts[rows, cols]

    summed.index_add_(0, flat_idx, contrib)
    summed.index_add_(0, mirror_idx, contrib.conj())
    weights.index_add_(0, flat_idx, weight)
    weights.index_add_(0, mirror_idx, weight)

    combined = torch.zeros_like(summed)
    good = weights > 0
    combined[good] = summed[good] / weights[good]
    return combined.reshape(n, n), weights.reshape(n, n)


def image_from_gridded(uv: torch.Tensor, counts: torch.Tensor) -> np.ndarray:
    # Counts are natural weights: sum repeated visibilities before imaging.
    weighted_uv = uv * counts
    image = torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(weighted_uv))
    )
    return image.numpy()


def grid_sample(transform, sample: dict, grid_size: int, patch_size: int):
    result = transform(sample)
    uv, counts = patches_to_uv_grid(result, grid_size, patch_size)
    return add_hermitian_counterparts(uv, counts)


def synthetic_sample(sample: dict, max_uv: float, dx: float, dy: float):
    """Inject a unit source at a known FFT-image offset (dx, dy) pixels.

    UV coordinates and max_uv are in the SAME units as the old transform.
    Thus this checks the grid mapping internally; physical sky offsets require
    converting UVW to wavelengths and verifying the corresponding image scale.
    """
    obj = dict(sample)
    vis = torch.as_tensor(sample["vis"]).clone().cpu()
    uvw = torch.as_tensor(sample["uvw"]).cpu()
    assert vis.ndim == 4 and vis.shape[0] == 1
    assert uvw.shape[:2] == (vis.shape[0], vis.shape[2])

    u = uvw[:, :, 0]
    v = uvw[:, :, 1]
    l = dx / (2.0 * max_uv)
    m = dy / (2.0 * max_uv)
    phase = -2.0 * torch.pi * (u * l + v * m)
    value = torch.polar(torch.ones_like(phase), phase).to(vis.dtype)
    vis[:, 0, :, 0] = value
    obj["vis"] = vis
    obj["uvw"] = uvw
    return obj


def print_peak(label: str, image: np.ndarray, n: int):
    peak_y, peak_x = np.unravel_index(np.argmax(image.real), image.shape)
    print(
        f"{label} peak (x,y) = ({peak_x},{peak_y}); "
        f"offset from image center = ({peak_x - n // 2}, {peak_y - n // 2})",
        flush=True,
    )
    return peak_x - n // 2, peak_y - n // 2


def run(args):
    matches = glob.glob(args.path)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one .vis file, found {len(matches)}: {matches}")
    path = matches[0]
    dataset = make_dataset(path)
    if not (0 <= args.window < len(dataset)):
        raise ValueError(f"Window {args.window} invalid: dataset has {len(dataset)} windows")
    sample = dataset[args.window]
    vis = torch.as_tensor(sample["vis"]).cpu()
    uvw = torch.as_tensor(sample["uvw"]).cpu()
    if not vis.is_complex():
        raise TypeError("Expected complex visibilities")
    if vis.shape[0] != 1 or vis.ndim != 4 or uvw.shape[:2] != (1, vis.shape[2]):
        raise ValueError(f"Unexpected shapes: vis={vis.shape}, uvw={uvw.shape}")

    u = uvw[:, :, 0].flatten()
    v = uvw[:, :, 1].flatten()
    valid = (
        torch.isfinite(u) & torch.isfinite(v)
        & (u.abs() <= args.max_uv) & (v.abs() <= args.max_uv)
    )
    print(f"Observation: {Path(path).name}")
    print(f"Window {args.window} of {len(dataset)}; vis shape: {tuple(vis.shape)}; UVW shape: {tuple(uvw.shape)}")
    print(f"UVW u range: [{u.min().item():.3g}, {u.max().item():.3g}]; v range: [{v.min().item():.3g}, {v.max().item():.3g}]")
    print(f"In bounds (max_uv={args.max_uv:g}): {int(valid.sum())} / {valid.numel()} baselines")
    if valid.sum() == 0:
        raise RuntimeError("No baselines are inside the UV bounds. Check units and --max-uv")

    transform = UV_Patch_Transform(
        grid_size=args.grid_size,
        patch_size=args.patch_size,
        max_uv=args.max_uv,
    )
    center_sample = synthetic_sample(sample, args.max_uv, dx=0, dy=0)
    shifted_sample = synthetic_sample(sample, args.max_uv, dx=args.dx, dy=args.dy)

    center_grid, center_counts = grid_sample(transform, center_sample, args.grid_size, args.patch_size)
    shifted_grid, shifted_counts = grid_sample(transform, shifted_sample, args.grid_size, args.patch_size)
    real_grid, real_counts = grid_sample(transform, sample, args.grid_size, args.patch_size)

    center_image = image_from_gridded(center_grid, center_counts)
    shifted_image = image_from_gridded(shifted_grid, shifted_counts)
    real_image = image_from_gridded(real_grid, real_counts)
    beam_peak = float(center_image.real.max())
    if beam_peak <= 0:
        raise RuntimeError("Non-positive synthetic beam peak")

    # All images use same normalization so real-image amplitude is comparable
    # to the unit source's dirty-beam amplitude.
    center_image /= beam_peak
    shifted_image /= beam_peak
    real_image /= beam_peak

    center_offset = print_peak("Centered source", center_image, args.grid_size)
    shifted_offset = print_peak("Shifted source", shifted_image, args.grid_size)
    print(f"Requested off-axis shift: (dx,dy) = ({args.dx:g},{args.dy:g}) pixels")
    print(f"Observed off-axis peak: {shifted_offset}")
    print(f"Populated UV cells (including conjugates): {int((real_counts > 0).sum())}")
    print(f"Dirty-image max real: {np.max(real_image.real):.6g}; min real: {np.min(real_image.real):.6g}")
    imag_ratio = np.max(np.abs(real_image.imag)) / max(np.max(np.abs(real_image.real)), 1e-30)
    print(f"Dirty-image imaginary/real max ratio: {imag_ratio:.6g}")

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    occ = np.log1p(real_counts.numpy())
    axs[0, 0].imshow(occ, origin="lower", cmap="magma", interpolation="nearest")
    axs[0, 0].set_title("UV occupancy (log1p sample count)")

    axs[0, 1].imshow(center_image.real, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
    axs[0, 1].set_title(f"Synthetic phase-center source; peak offset {center_offset}")
    axs[0, 1].plot(args.grid_size // 2, args.grid_size // 2, marker="+", color="yellow")

    vmax = max(float(np.max(np.abs(shifted_image.real))), 1e-8)
    axs[1, 0].imshow(shifted_image.real, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axs[1, 0].plot(args.grid_size // 2 + args.dx, args.grid_size // 2 + args.dy, marker="+", color="yellow")
    axs[1, 0].set_title(f"Known off-axis source; expected ({args.dx:g}, {args.dy:g}), observed {shifted_offset}")

    vmax_real = max(float(np.max(np.abs(real_image.real))), 1e-8)
    axs[1, 1].imshow(real_image.real, origin="lower", cmap="RdBu_r", vmin=-vmax_real, vmax=vmax_real)
    axs[1, 1].set_title("Original observed visibilities: dirty image")
    for ax in axs.flat:
        ax.set_xlabel("Image/UV column")
        ax.set_ylabel("Image/UV row")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(f"Point-source sanity check: {Path(path).name}, window={args.window}, freq=0, pol=0", fontsize=11)
    fig.savefig(output, dpi=160)
    print(f"Saved: {output.resolve()}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="One .vis file or a quoted wildcard matching exactly one file")
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--max-uv", type=float, default=70000.0)
    parser.add_argument("--dx", type=float, default=14.0, help="Synthetic source x offset in pixels")
    parser.add_argument("--dy", type=float, default=-10.0, help="Synthetic source y offset in pixels")
    parser.add_argument("--output", default="diagnostics/point_source_s42.png")
    args = parser.parse_args()
    if args.grid_size % args.patch_size:
        parser.error("--grid-size must be divisible by --patch-size")
    run(args)
