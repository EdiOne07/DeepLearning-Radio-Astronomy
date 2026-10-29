#!/usr/bin/env python3
"""Inspect temporal variation in complex radio visibilities without training a model.

Save next to the training script, in the directory where `rat_stack` and
`Visibility_Standerdizer` are importable, then run for one or more .vis files:

    python visibility_diagnostics.py '/path/to/gleam*_s41_*.vis'
    python visibility_diagnostics.py '/path/to/gleam*_s41_*.vis'

Outputs one CSV and one four-panel PNG per observation under diagnostics/.
Uses the same TimeSlicedVisDataset settings as the current training pipeline.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rat_stack import TimeSlicedVisDataset
from Model.Visibility_Standerdizer import VisibilityStandardizer

WINDOW_SIZE = 1
TARGET_SIZE = 1
STRIDE = 8
TRAIN_FRACTION = 0.8
N_TRAIN_SAMPLES = 32
N_VAL_SAMPLES = 8


def make_dataset(path: str):
    return TimeSlicedVisDataset(
        paths=[path],
        window_size=WINDOW_SIZE,
        target_size=TARGET_SIZE,
        stride=STRIDE,
        pin_memory=False,
        read_uvw=True,
    )


def indices_spread(start: int, end_inclusive: int, count: int) -> list[int]:
    available = end_inclusive - start + 1
    if available < count:
        raise ValueError(f"Need {count} windows, have only {available}")
    return torch.linspace(start, end_inclusive, steps=count).round().long().tolist()


def get_vis(dataset, index: int) -> tuple[torch.Tensor, dict]:
    sample = dataset[index]
    vis = torch.as_tensor(sample["vis"]).cpu()
    if not vis.is_complex() or vis.ndim != 4 or vis.shape[-1] != 4:
        raise ValueError(f"Unexpected vis shape/dtype at {index}: {vis.shape}, {vis.dtype}")
    return vis, sample


def uvw_positions(sample: dict):
    if "uvw" not in sample or sample["uvw"] is None:
        return None
    xyz = torch.as_tensor(sample["uvw"]).cpu()
    if xyz.shape[-1] != 3:
        return None
    return xyz.reshape(-1, 3).to(torch.float64)


def complex_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    # Subsample regularly to make correlation cheap. The same positions are
    # used for every window, so this is comparable over the observation.
    av = a[:, ::4, ::16, :].reshape(-1).to(torch.complex128)
    bv = b[:, ::4, ::16, :].reshape(-1).to(torch.complex128)
    denom = (torch.linalg.vector_norm(av) * torch.linalg.vector_norm(bv)).item()
    return float(torch.abs(torch.vdot(av, bv)).item() / denom) if denom > 0 else float("nan")


def mean_or_nan(values: list[float]) -> float:
    finite = [x for x in values if math.isfinite(x)]
    return sum(finite) / len(finite) if finite else float("nan")


@torch.no_grad()
def analyze_observation(path: str, output_dir: Path):
    dataset = make_dataset(path)
    count = len(dataset)
    split = int(TRAIN_FRACTION * count)
    if split < N_TRAIN_SAMPLES or count - split < N_VAL_SAMPLES:
        raise ValueError(f"Not enough windows in {path}: {count}")

    train_indices = indices_spread(0, split - 1, N_TRAIN_SAMPLES)
    val_indices = indices_spread(split, count - 1, N_VAL_SAMPLES)
    print(f"\nObservation: {Path(path).name}", flush=True)
    print(f"Windows: {count}; training region [0, {split - 1}]; validation region [{split}, {count - 1}]", flush=True)
    print("Distributed training indices:", train_indices, flush=True)
    print("Distributed validation indices:", val_indices, flush=True)

    # Reproduce exactly the normalization fitting scheme for the distributed
    # single-observation experiment: all 32 selected TRAIN samples only.
    train_vis = []
    for idx in train_indices:
        vis, _ = get_vis(dataset, idx)
        if not torch.isfinite(vis).all():
            raise RuntimeError(f"Non-finite training sample at window {idx}")
        train_vis.append(vis)
    fit_data = torch.cat(train_vis, dim=0)
    standardizer = VisibilityStandardizer()
    standardizer.fit(fit_data)
    del fit_data, train_vis

    first, first_sample = get_vis(dataset, 0)
    prev = first
    first_uvw = uvw_positions(first_sample)
    uvw_covered = first_uvw is not None
    rows = []

    for i in range(count):
        vis, sample = get_vis(dataset, i)
        finite_mask = torch.isfinite(vis)
        bad = (~finite_mask).sum().item()
        row = {
            "window": i,
            "region": "train_region" if i < split else "val_region",
            "selected_train": int(i in train_indices),
            "selected_val": int(i in val_indices),
            "nonfinite_count": bad,
        }
        if bad:
            print(f"WARNING: {bad} invalid entries at window {i}", flush=True)
            # Preserve the time position; metrics are blank for invalid samples.
            rows.append(row)
            continue

        squared_amplitude = vis.abs().to(torch.float64).square()
        per_pol_power = squared_amplitude.mean(dim=(0, 1, 2)).tolist()
        row.update({f"power_pol{p}": float(per_pol_power[p]) for p in range(4)})
        row["raw_power"] = float(squared_amplitude.mean().item())
        row["max_amplitude"] = float(vis.abs().max().item())
        sampled_amplitudes = vis.abs()[:, ::2, ::32, :].flatten().float()
        row["approx_amplitude_p99"] = float(torch.quantile(sampled_amplitudes, 0.99).item())
        row["zero_fraction"] = float((vis == 0).to(torch.float32).mean().item())
        row["similarity_to_first"] = complex_similarity(first, vis)
        row["similarity_to_previous"] = complex_similarity(prev, vis)

        # Shape [T, F, B, 4, 2] -> [all measurements, 4, 2].
        standardized = standardizer.transform(vis)
        real_imag = torch.view_as_real(standardized).reshape(-1, 4, 2).to(torch.float64)
        chan_mean = real_imag.mean(dim=0).flatten().tolist()
        chan_power = real_imag.square().mean(dim=0).flatten().tolist()
        row["normalized_power"] = float(real_imag.square().mean().item())
        for c in range(8):
            row[f"normalized_mean_ch{c}"] = float(chan_mean[c])
            row[f"normalized_power_ch{c}"] = float(chan_power[c])

        current_uvw = uvw_positions(sample)
        if (uvw_covered and current_uvw is not None
                and current_uvw.shape == first_uvw.shape):
            delta = current_uvw[:, :2] - first_uvw[:, :2]
            row["uvw_mean_uv_displacement"] = float(torch.linalg.vector_norm(delta, dim=-1).mean().item())
            reference_scale = torch.linalg.vector_norm(first_uvw[:, :2], dim=-1).mean().item()
            row["uvw_relative_displacement"] = row["uvw_mean_uv_displacement"] / max(reference_scale, 1e-12)

        prev = vis
        rows.append(row)
        if i == 0 or (i + 1) % 50 == 0 or i + 1 == count:
            print(f"  inspected {i + 1}/{count} windows", flush=True)

    tag_match = re.search(r"_s\d+(?:_|\.)", Path(path).name)
    tag = tag_match.group(0).strip("_.") if tag_match else re.sub(r"[^A-Za-z0-9_-]+", "_", Path(path).stem)[:60]
    csv_file = output_dir / f"{tag}_visibility_audit.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_file.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def series(key):
        return [r.get(key, float("nan")) for r in rows]

    x = [r["window"] for r in rows]
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True, constrained_layout=True)
    plots = [
        ("raw_power", "Raw mean |V|²"),
        ("normalized_power", "Standardized mean squared value"),
        ("similarity_to_first", "Abs. complex cosine similarity to window 0"),
        ("uvw_relative_displacement", "Mean UV displacement / mean initial UV radius"),
    ]
    for ax, (key, title) in zip(axes, plots):
        y = series(key)
        ax.plot(x, y, linewidth=1.5)
        ax.axvline(split - 0.5, color="darkorange", linestyle="--", label="80/20 boundary")
        ax.grid(alpha=0.2)
        ax.set_ylabel(title)
        if key in ("raw_power", "normalized_power") and all(v > 0 for v in y if math.isfinite(v)):
            ax.set_yscale("log")
        ax.legend(loc="best")
    axes[-1].set_xlabel("Time-window index (dataset STRIDE=8)")
    axes[0].set_title(f"Visibility time variation: {Path(path).name}")
    png_file = output_dir / f"{tag}_visibility_audit.png"
    fig.savefig(png_file, dpi=150)
    plt.close(fig)

    print("\nRegion summary (entire training/validation regions, not just selected 32/8):")
    for region in ("train_region", "val_region"):
        data = [r for r in rows if r["region"] == region]
        print(f"  {region}: raw power={mean_or_nan([r.get('raw_power', float('nan')) for r in data]):.6g}, "
              f"normalized power={mean_or_nan([r.get('normalized_power', float('nan')) for r in data]):.6g}, "
              f"similarity to first={mean_or_nan([r.get('similarity_to_first', float('nan')) for r in data]):.5f}, "
              f"bad windows={sum(r['nonfinite_count'] > 0 for r in data)}")
    print("CSV:", csv_file)
    print("Plot:", png_file, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Visibility files or quoted globs")
    parser.add_argument("--output-dir", default="diagnostics")
    args = parser.parse_args()
    paths = sorted({str(Path(p).resolve()) for pattern in args.paths
                    for p in (glob.glob(pattern) or [pattern]) if Path(p).is_file()})
    if not paths:
        parser.error("No matching visibility files found")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        analyze_observation(path, output_dir)


if __name__ == "__main__":
    main()
