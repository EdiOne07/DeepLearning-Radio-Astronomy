import torch
from rat_stack import TimeSlicedVisDataset
import sys
import glob
from pathlib import Path

import torch
import matplotlib.pyplot as plt


import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def test_polarization_power(sample):
    vis = sample["vis"]

    # [T, F, B, P] -> [P]
    power = vis.abs().square().sum(dim=(0, 1, 2))
    power_fraction = power / power.sum()

    print("Signal power by polarization:")

    for p in range(vis.shape[-1]):
        print(
            f"Polarization {p}: "
            f"{100 * power_fraction[p].item():.4f}%"
        )

    assert torch.isfinite(power_fraction).all()
    assert torch.isclose(
        power_fraction.sum(),
        torch.tensor(1.0, device=power.device),
        atol=1e-5,
    )

    print("Polarization power test passed.")

@torch.no_grad()
def analyze_frequency_polarization_distribution(sample):
    vis = sample["vis"]

    assert vis.ndim == 4, f"Expected [T, F, B, P], got {vis.shape}"
    assert torch.is_complex(vis), "Expected complex visibility data"
    assert torch.isfinite(vis.real).all()
    assert torch.isfinite(vis.imag).all()

    # [T, F, B, P] -> [F, P]
    mean_real = vis.real.mean(dim=(0, 2))
    std_real = vis.real.std(dim=(0, 2))
    mean_imag = vis.imag.mean(dim=(0, 2))
    std_imag = vis.imag.std(dim=(0, 2))

    amplitude = vis.abs()
    mean_amplitude = amplitude.mean(dim=(0, 2))

    amplitude_flat = amplitude.permute(1, 3, 0, 2).reshape(
        vis.shape[1], vis.shape[3], -1
    )

    p99_amplitude = torch.quantile(
        amplitude_flat, 0.99, dim=-1
    )

    metrics = {
        "Mean real": mean_real,
        "Std real": std_real,
        "Mean imaginary": mean_imag,
        "Std imaginary": std_imag,
        "Mean amplitude": mean_amplitude,
        "P99 amplitude": p99_amplitude,
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for ax, (name, values) in zip(axes.flat, metrics.items()):
        values = values.cpu().numpy()

        im = ax.imshow(
            values.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest"
        )

        ax.set_title(name)
        ax.set_xlabel("Frequency channel")
        ax.set_ylabel("Polarization index")
        ax.set_yticks(range(vis.shape[3]))

        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(
        "frequency_polarization_distributions.png",
        dpi=200
    )
    plt.close(fig)

    print("Visibility shape:", tuple(vis.shape))

    for name, values in metrics.items():
        print(
            f"{name}: "
            f"min={values.min().item():.6f}, "
            f"max={values.max().item():.6f}, "
            f"mean={values.mean().item():.6f}"
        )

    print("Saved frequency_polarization_distributions.png")

@torch.no_grad()
def analyze_visibility_distribution(sample):
    vis = sample["vis"]

    values = vis[:, 0, :, 0].reshape(-1)

    values = values[
        torch.isfinite(values.real)
        & torch.isfinite(values.imag)
    ]

    distributions = {
        "Real": values.real.float(),
        "Imaginary": values.imag.float(),
        "Amplitude": values.abs().float(),
        "Phase": torch.angle(values).float(),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, (name, data) in zip(axes.flat, distributions.items()):
        data = data.cpu()

        quantiles = torch.quantile(
            data,
            torch.tensor([0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
        )

        print(f"\n{name}")
        print(f"Mean: {data.mean().item():.6f}")
        print(f"Std:  {data.std().item():.6f}")
        print(f"Min:  {data.min().item():.6f}")
        print(f"Max:  {data.max().item():.6f}")
        print(
            "Quantiles [0, 1, 25, 50, 75, 99, 100]%:",
            [round(v, 6) for v in quantiles.tolist()]
        )

        ax.hist(data.numpy(), bins=100)
        ax.set_title(f"{name} distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig("visibility_distributions.png", dpi=200)
    plt.close(fig)

    print("\nSaved visibility_distributions.png")

import numpy as np
import matplotlib.pyplot as plt


def visualize_dirty_images(sample, grid_size=512, frequency_idx=0):
    vis = sample["vis"].detach().cpu().numpy()
    uvw = sample["uvw"].detach().cpu().numpy()

    # UV coordinates for all timesteps and baselines.
    # A single frequency is selected, so a common UV scale is enough
    # to inspect relative structure (not calibrated sky positions).
    u = uvw[..., 0].reshape(-1)
    v = uvw[..., 1].reshape(-1)

    finite_uv = np.isfinite(u) & np.isfinite(v)
    radius = np.maximum(np.abs(u[finite_uv]), np.abs(v[finite_uv]))
    uv_limit = np.quantile(radius, 0.999)

    if uv_limit <= 0:
        raise ValueError("Invalid UV coordinate range.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for row, pol in enumerate([0, 3]):
        values = vis[:, frequency_idx, :, pol].reshape(-1)

        valid = (
            finite_uv
            & np.isfinite(values.real)
            & np.isfinite(values.imag)
            & (np.abs(u) <= uv_limit)
            & (np.abs(v) <= uv_limit)
        )

        uu = u[valid]
        vv = v[valid]
        zz = values[valid]

        if len(zz) == 0:
            raise ValueError("No valid UV measurements.")

        grid = np.zeros((grid_size, grid_size), dtype=np.complex128)
        weights = np.zeros((grid_size, grid_size), dtype=np.float64)

        def add_samples(u_values, v_values, samples):
            col = np.rint(
                (u_values / (2 * uv_limit) + 0.5) * (grid_size - 1)
            ).astype(int)
            row_idx = np.rint(
                (v_values / (2 * uv_limit) + 0.5) * (grid_size - 1)
            ).astype(int)

            np.add.at(grid, (row_idx, col), samples)
            np.add.at(weights, (row_idx, col), 1)

        # Original measurements and their conjugate counterparts.
        add_samples(uu, vv, zz)
        add_samples(-uu, -vv, np.conjugate(zz))

        dirty_complex = np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(grid))
        ) * (grid_size**2 / weights.sum())

        dirty = dirty_complex.real

        ax_uv = axes[row, 0]
        ax_img = axes[row, 1]

        ax_uv.imshow(
            np.log1p(np.abs(grid)),
            origin="lower",
            cmap="magma"
        )
        ax_uv.set_title(f"Polarization {pol}: UV coverage/amplitude")

        vmax = np.max(np.abs(dirty))

        im = ax_img.imshow(
            dirty,
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax
        )
        ax_img.set_title(f"Polarization {pol}: dirty image")
        fig.colorbar(im, ax=ax_img, shrink=0.8)

        center = grid_size // 2
        yy, xx = np.indices(dirty.shape)
        outside_center = (xx - center)**2 + (yy - center)**2 > 10**2

        background_rms = np.sqrt(
            np.mean(dirty[outside_center] ** 2)
        )

        peak = np.unravel_index(np.argmax(dirty), dirty.shape)
        peak_value = dirty[peak]
        peak_to_rms = peak_value / max(background_rms, 1e-15)

        print(f"\nPolarization {pol}")
        print(f"Valid samples: {len(zz)}")
        print(f"Peak position (row, col): {peak}")
        print(f"Peak value: {peak_value:.6g}")
        print(f"Off-center RMS: {background_rms:.6g}")
        print(f"Peak / off-center RMS: {peak_to_rms:.3f}")

    plt.tight_layout()
    plt.savefig("dirty_image_inspection.png", dpi=200)
    plt.close(fig)

    print("\nSaved dirty_image_inspection.png")
    plot_peak_zoom(dirty=dirty,peak=peak,radius=30)

def plot_peak_zoom(dirty, peak, radius=30):
    row, col = peak

    y0 = max(row - radius, 0)
    y1 = min(row + radius + 1, dirty.shape[0])
    x0 = max(col - radius, 0)
    x1 = min(col + radius + 1, dirty.shape[1])

    plt.figure(figsize=(7, 6))
    plt.imshow(
        dirty[y0:y1, x0:x1],
        origin="lower",
        extent=(x0, x1, y0, y1),
        cmap="RdBu_r",
        vmin=-np.max(np.abs(dirty)),
        vmax=np.max(np.abs(dirty)),
    )
    plt.scatter(
        [col], [row],
        marker="+",
        color="black",
        s=100,
    )
    plt.colorbar(label="Dirty-image intensity")
    plt.title(f"Peak zoom — ({col}, {row})")
    plt.tight_layout()
    plt.savefig("dirty_image_peak_zoom.png", dpi=200)
    plt.close()
    
def test_raw_visibility_sample(path:list[str]):

    data=TimeSlicedVisDataset(paths=path,window_size=8,target_size=1,stride=1,pin_memory=False,read_uvw=True)
    sample=data[10]
  

    assert "vis" in sample, "Missing visibility data"
    assert "uvw" in sample, "Missing UVW coordinates"

    vis = sample["vis"]
    uvw = sample["uvw"]

    assert torch.is_complex(vis), "Expected complex visibilities"
    assert vis.ndim == 4, f"Unexpected visibility shape: {vis.shape}"
    assert uvw.ndim == 3, f"Unexpected UVW shape: {uvw.shape}"

    assert vis.shape[0] == uvw.shape[0], "Time dimensions differ"
    assert vis.shape[2] == uvw.shape[1], "Baseline dimensions differ"
    assert uvw.shape[-1] == 3, "UVW must have three coordinates"

    selected = vis[:, 0, :, 0]

    finite = (
        torch.isfinite(selected.real)
        & torch.isfinite(selected.imag)
    )

    assert finite.any(), "No finite visibility measurements"

    print("Visibility shape:", tuple(vis.shape))
    print("UVW shape:", tuple(uvw.shape))
    print("Selected measurements:", selected.numel())
    print("Finite measurements:", finite.sum().item())
    print("Finite percentage:", finite.float().mean().item() * 100)

    print("Raw sample integrity test passed.")
    visualize_dirty_images(sample=sample,grid_size=512,frequency_idx=0)

if __name__ == "__main__":
    args = sys.argv[1:]

    paths = sorted({
        str(Path(p).resolve())
        for arg in args
        for p in (glob.glob(arg) or [arg])
        if Path(p).is_file()
    })
    test_raw_visibility_sample(paths)  