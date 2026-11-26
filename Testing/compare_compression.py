#!/usr/bin/env python3
"""Compare runs with different compression ratios from their SLURM logs.

Reads each log's "LATENT_DIM=..." line and its "Fold .. | Epoch .. | ... Val NMSE: x
| Val NMSE per pol [P0 P1 P2 P3]: a b c d" lines. If several logs share a
LATENT_DIM, the one with the most epochs is used (and the others are listed).

Outputs (in --out-dir):
  curves.png              validation NMSE vs epoch, one line per compression ratio
  rate_distortion.png     best validation NMSE vs compression ratio, AE vs PCA
  summary.csv             the numbers behind both plots

Example:
    python Testing/compare_compression.py results/compression_sweep/logs/*.log \
        --out-dir results/compression_sweep/figures
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LATENT_RE = re.compile(r"LATENT_DIM=(\d+)")
EPOCH_RE = re.compile(
    r"Fold \d+/\d+ \| Epoch (\d+)/\d+ .*?Val NMSE: ([0-9.]+)"
    r"(?: \| Val NMSE per pol \[P0 P1 P2 P3\]: ([0-9.]+) ([0-9.]+) ([0-9.]+) ([0-9.]+))?"
)

# PCA fitted on s42, tested on s40 (pca_compressibility.py), keyed by compression ratio.
PCA_CROSS = {4: 0.0308, 8: 0.3650, 16: 0.5723, 32: 0.6990}
INPUT_VALUES = 8 * 64          # per baseline
LATENT_POSITIONS = 16          # 64 channels after two 2x downsamplings


def parse_log(path):
    text = Path(path).read_text(errors="ignore")
    m = LATENT_RE.search(text)
    if not m:
        return None
    epochs = {}
    for line in text.splitlines():
        e = EPOCH_RE.search(line)
        if e:
            # keyed by epoch number, so duplicated lines (copy/paste, resumes) count once
            epochs[int(e.group(1))] = (float(e.group(2)),
                                       float(e.group(3)) if e.group(3) else None)
    if not epochs:
        return None
    order = sorted(epochs)
    return {"path": str(path), "latent": int(m.group(1)),
            "nmse": [epochs[k][0] for k in order], "xx": [epochs[k][1] for k in order]}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--out-dir", default="compression_figures")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = {}
    for path in args.logs:
        r = parse_log(path)
        if r is None:
            print(f"skipped (no LATENT_DIM or no epochs): {path}")
            continue
        prev = runs.get(r["latent"])
        if prev is None or len(r["nmse"]) > len(prev["nmse"]):
            if prev is not None:
                print(f"LATENT_DIM={r['latent']}: using {r['path']} ({len(r['nmse'])} epochs) "
                      f"instead of {prev['path']} ({len(prev['nmse'])} epochs)")
            runs[r["latent"]] = r
        else:
            print(f"LATENT_DIM={r['latent']}: ignoring {path} ({len(r['nmse'])} epochs, fewer)")

    if not runs:
        raise SystemExit("No usable logs found.")

    rows = []
    for latent in sorted(runs, reverse=True):          # 4x, 8x, 16x, 32x
        r = runs[latent]
        ratio = INPUT_VALUES / (latent * LATENT_POSITIONS)
        best_i = min(range(len(r["nmse"])), key=lambda i: r["nmse"][i])
        rows.append({
            "latent_dim": latent,
            "compression": f"{ratio:g}x",
            "epochs": len(r["nmse"]),
            "best_epoch": best_i + 1,
            "best_val_nmse": r["nmse"][best_i],
            "xx_at_best": r["xx"][best_i],
            "last_val_nmse": r["nmse"][-1],
            "pca_cross": PCA_CROSS.get(int(round(ratio))),
            "log": r["path"],
        })

    print(f"\n{'compression':>11} {'epochs':>6} {'best NMSE':>10} {'@epoch':>6} "
          f"{'last NMSE':>10} {'PCA cross':>10}")
    for row in rows:
        pca = f"{row['pca_cross']:.4f}" if row["pca_cross"] is not None else "-"
        print(f"{row['compression']:>11} {row['epochs']:>6} {row['best_val_nmse']:>10.4f} "
              f"{row['best_epoch']:>6} {row['last_val_nmse']:>10.4f} {pca:>10}")

    with open(out / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 1) training curves
    fig, ax = plt.subplots(figsize=(9, 5))
    for row in rows:
        r = runs[row["latent_dim"]]
        ax.plot(range(1, len(r["nmse"]) + 1), r["nmse"], label=f"{row['compression']} (latent {row['latent_dim']}x16)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation NMSE (held-out sky)")
    ax.set_yscale("log")
    ax.set_title("Per-baseline autoencoder: training curves by compression ratio")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=150)
    plt.close(fig)

    # 2) rate-distortion: best NMSE vs compression, AE vs PCA
    fig, ax = plt.subplots(figsize=(7, 5))
    ratios = [INPUT_VALUES / (row["latent_dim"] * LATENT_POSITIONS) for row in rows]
    ax.plot(ratios, [row["best_val_nmse"] for row in rows], "o-", linewidth=2, label="per-baseline autoencoder")
    for x, row in zip(ratios, rows):
        ax.annotate(f"{row['best_val_nmse']:.3f}", (x, row["best_val_nmse"]),
                    textcoords="offset points", xytext=(0, 8), ha="center")
    pca_x = sorted(PCA_CROSS)
    ax.plot(pca_x, [PCA_CROSS[k] for k in pca_x], "s--", color="gray", label="PCA fitted on another sky")
    ax.set_xscale("log", base=2)
    ax.set_xticks(pca_x)
    ax.set_xticklabels([f"{k}x" for k in pca_x])
    ax.set_yscale("log")
    ax.set_xlabel("Compression ratio")
    ax.set_ylabel("Best validation NMSE (held-out sky, lower is better)")
    ax.set_title("Reconstruction error vs compression")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "rate_distortion.png", dpi=150)
    plt.close(fig)

    print("\nFigures and summary.csv in", out.resolve())


if __name__ == "__main__":
    main()