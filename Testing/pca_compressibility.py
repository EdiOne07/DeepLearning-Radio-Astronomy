#!/usr/bin/env python3
"""PCA compressibility test for per-baseline visibility spectra. No training.

Each sample is one baseline in one time window: its 64 frequencies x 4
polarizations = 256 complex values. A complex PCA keeps the top k components,
so the compression ratio is 256 / k (k = 32 is about the 8x of the autoencoder;
the shared basis is not counted).

For every observation the script reports NMSE vs k for:
  self  : PCA fitted on that observation, tested on other windows of it
          (best case for any linear compressor on this data)
  cross : PCA fitted on the source observation (e.g. s42), tested on it
          (does the structure transfer between skies?)
and the same split by projected baseline length (quartiles).

Each observation is scaled by its own RMS (like option B in the zero-shot
test); NMSE is a ratio, so it is reported in physical terms.

Place in Model/ or Testing/ and run, e.g.:

    python Testing/pca_compressibility.py \
        --source "/path/gleam*_s42_*.vis" \
        --targets "/path/gleam*_s40_*.vis" "/path/gleam*_s41_*.vis"
"""

import argparse
import glob
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
import torch

from Model.Training_Amplitude_Phase import makeDataset

KS = [0, 1, 2, 4, 8, 16, 32, 64, 128]


def resolve(patterns):
    return sorted({str(Path(p).resolve()) for pattern in patterns
                   for p in (glob.glob(pattern) or [pattern]) if Path(p).is_file()})


def tag_of(path):
    match = re.search(r"_(s\d+)_", Path(path).name)
    return match.group(1) if match else Path(path).stem[:20]


def split_indices(n_total, n_each):
    """Interleaved, disjoint fit/test windows spread over the whole observation."""
    idx = torch.linspace(0, n_total - 1, steps=min(2 * n_each, n_total)).round().long()
    idx = sorted(set(idx.tolist()))
    return idx[0::2], idx[1::2]


def load_window(dataset, index):
    sample = dataset[index]
    vis = sample["vis"][0]                                   # [F, B, P] complex
    x = vis.permute(1, 0, 2).reshape(vis.shape[1], -1)       # [B, F*P]
    uv = sample["uvw"][0, :, :2].double()                    # [B, 2]
    return x.to(torch.complex64), torch.linalg.vector_norm(uv, dim=-1)


def load_observation(path, n_each):
    dataset = makeDataset([path], use_cuda=False)
    fit_idx, test_idx = split_indices(len(dataset), n_each)
    fit = [load_window(dataset, i) for i in fit_idx]
    test = [load_window(dataset, i) for i in test_idx]

    # Per-observation RMS from the fit windows only.
    energy = sum(x.to(torch.complex128).abs().square().sum().item() for x, _ in fit)
    count = sum(x.numel() for x, _ in fit)
    rms = (energy / count) ** 0.5

    scale = lambda windows: [((x.to(torch.complex128) / rms).to(torch.complex64), l)
                             for x, l in windows]
    print(f"  {tag_of(path)}: {len(dataset)} windows, fit {len(fit_idx)}, "
          f"test {len(test_idx)}, RMS |V| = {rms:.4g}", flush=True)
    return {"tag": tag_of(path), "fit": scale(fit), "test": scale(test)}


def fit_pca(windows):
    dim = windows[0][0].shape[1]
    mean = torch.zeros(dim, dtype=torch.complex128)
    rows = 0
    for x, _ in windows:
        mean += x.to(torch.complex128).sum(0)
        rows += x.shape[0]
    mean /= rows

    cov = torch.zeros(dim, dim, dtype=torch.complex128)      # sum of x x^H
    for x, _ in windows:
        xc = x.to(torch.complex128) - mean
        cov += xc.T @ xc.conj()
    evals, evecs = torch.linalg.eigh(cov)                    # ascending, orthonormal columns
    order = torch.argsort(evals, descending=True)
    return mean, evecs[:, order]


@torch.no_grad()
def evaluate(mean, basis, windows, edges):
    n_bins = len(edges) + 1
    n_comp = basis.shape[1]
    err = torch.zeros(n_bins, n_comp + 1, dtype=torch.float64)
    power = torch.zeros(n_bins, dtype=torch.float64)

    for x, length in windows:
        xd = x.to(torch.complex128)
        xc = xd - mean
        coeffs = xc @ basis.conj()                           # c_j = v_j^H x, [B, K]
        captured = torch.cumsum(coeffs.abs().square(), dim=1)
        total = xc.abs().square().sum(1, keepdim=True)
        resid = torch.cat([total, total - captured], dim=1).clamp_min(0)  # column k = k comps kept
        bins = torch.bucketize(length, edges)
        err.index_add_(0, bins, resid)
        power.index_add_(0, bins, xd.abs().square().sum(1))

    return err.sum(0) / power.sum(), err / power[:, None]


def print_table(rows, dim):
    head = "".join(f"{('k=' + str(k)):>9}" for k in KS)
    ratio = "".join(f"{(str(dim // k) + 'x') if k else '-':>9}" for k in KS)
    print(f"\n{'':24}{head}\n{'compression':24}{ratio}")
    for label, nmse, _ in rows:
        print(f"{label:24}" + "".join(f"{nmse[k].item():9.4f}" for k in KS))


def print_bins(rows, k, edges):
    names = (["< " + f"{edges[0]:.0f} m"]
             + [f"{edges[i]:.0f}-{edges[i + 1]:.0f} m" for i in range(len(edges) - 1)]
             + ["> " + f"{edges[-1]:.0f} m"])
    print(f"\nNMSE at k={k} by projected baseline length:")
    print(f"{'':24}" + "".join(f"{n:>16}" for n in names))
    for label, _, by_bin in rows:
        print(f"{label:24}" + "".join(f"{by_bin[b, k].item():16.4f}" for b in range(len(names))))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Observation the 'cross' PCA is fitted on")
    parser.add_argument("--targets", nargs="*", default=[], help="Observations to test on")
    parser.add_argument("--windows", type=int, default=32,
                        help="Windows per split (fit and test each) per observation")
    parser.add_argument("--out", default="pca_compressibility.png")
    args = parser.parse_args()

    source_paths = resolve([args.source])
    if len(source_paths) != 1:
        parser.error(f"--source must match exactly one file, got {source_paths}")
    target_paths = [p for p in resolve(args.targets) if p != source_paths[0]]

    print("Loading observations:", flush=True)
    source = load_observation(source_paths[0], args.windows)
    targets = [load_observation(p, args.windows) for p in target_paths]

    # Fixed length bins (quartiles of the source's projected baseline lengths).
    all_lengths = torch.cat([l for _, l in source["fit"]])
    edges = torch.quantile(all_lengths, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
    dim = source["fit"][0][0].shape[1]

    print("\nFitting PCA ...", flush=True)
    src_mean, src_basis = fit_pca(source["fit"])

    rows = []
    nmse, by_bin = evaluate(src_mean, src_basis, source["test"], edges)
    rows.append((f"{source['tag']} self", nmse, by_bin))
    for target in targets:
        mean, basis = fit_pca(target["fit"])
        nmse, by_bin = evaluate(mean, basis, target["test"], edges)
        rows.append((f"{target['tag']} self", nmse, by_bin))
        nmse, by_bin = evaluate(src_mean, src_basis, target["test"], edges)
        rows.append((f"{target['tag']} cross ({source['tag']} PCA)", nmse, by_bin))

    print_table(rows, dim)
    print_bins(rows, 32, edges.tolist())
    print("\nReference: autoencoder ~8x (k=32): s42 val 0.08; s40 zero-shot B 1.03.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ks = [k for k in KS if k > 0]
    for label, nmse, _ in rows:
        ax.plot(ks, [nmse[k].item() for k in ks], marker="o",
                linestyle="--" if "cross" in label else "-", label=label)
    ax.axvline(32, color="gray", linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Complex components kept per baseline (k)  —  compression = 256/k")
    ax.set_ylabel("NMSE")
    ax.set_title("PCA on per-baseline spectra")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("Plot:", Path(args.out).resolve())


if __name__ == "__main__":
    main()