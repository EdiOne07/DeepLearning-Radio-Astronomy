
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from Model.MSE_Loss_Amplitude_Phase import polar_to_cartesian


def dirty_image(uv_grid):
    """
    Compute a simple inverse Fourier transform of a UV grid.

    This is a diagnostic dirty image, not a fully
    calibrated or deconvolved astronomical image.
    """
    return np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(uv_grid)
        )
    )


@torch.no_grad()
def visualize_reconstruction(
    model,
    dataset,
    device,
    amplitude_scale,
    output_dir,
    fold_number,
    sample_index=0,
):
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = dataset[sample_index]

    input_grid = (
        sample["input_grid"]
        .unsqueeze(0)
        .to(device)
    )

    target_grid = (
        sample["target_grid"]
        .unsqueeze(0)
        .to(device)
    )

    latent = model.encode(input_grid)
    output = model.decode(latent)

    prediction = polar_to_cartesian(
        output,
        amplitude_scale=amplitude_scale,
    )

    mask = input_grid[0, 2].bool().cpu().numpy()

    original = torch.complex(
        target_grid[0, 0],
        target_grid[0, 1],
    ).cpu().numpy()

    reconstructed = torch.complex(
        prediction[0, 0],
        prediction[0, 1],
    ).cpu().numpy()

    original = np.where(mask, original, 0)
    reconstructed = np.where(mask, reconstructed, 0)

    residual = original - reconstructed

###########################
# Reconstruction of metrics
##########################

    signal_power = np.sum(np.abs(original[mask]) ** 2)
    error_power = np.sum(np.abs(residual[mask]) ** 2)

    nmse = error_power / max(signal_power, 1e-20)

    snr = (
        -10.0 * np.log10(nmse)
        if nmse > 0
        else float("inf")
    )

    print(f"Fold {fold_number} visualization:")
    print(f"  Observed UV cells: {mask.sum()}")
    print(f"  Sample NMSE: {nmse:.9f}")
    print(f"  Sample SNR: {snr:.3f} dB")

###################
# Figure1:UV Recons
###################

    original_amp = np.log1p(np.abs(original))
    reconstructed_amp = np.log1p(np.abs(reconstructed))
    residual_amp = np.log1p(np.abs(residual))

    common_max = max(
        original_amp.max(),
        reconstructed_amp.max(),
        1e-12,
    )

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5),
        constrained_layout=True,
    )

    images = [
        original_amp,
        reconstructed_amp,
        residual_amp,
    ]

    titles = [
        "Original UV amplitude",
        "Reconstructed UV amplitude",
        "Absolute UV residual",
    ]

    for i, (ax, image, title) in enumerate(
        zip(axes, images, titles)
    ):
        im = ax.imshow(
            image,
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=common_max if i < 2 else None,
            interpolation="nearest",
        )

        ax.set_title(title)
        ax.set_xlabel("u grid index")
        ax.set_ylabel("v grid index")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        f"Fold {fold_number}: UV reconstruction\n"
        f"NMSE = {nmse:.6f}, SNR = {snr:.2f} dB"
    )

    fig.savefig(
        output_dir / f"fold_{fold_number}_uv_reconstruction.png",
        dpi=200,
    )

    plt.close(fig)

###################
# Figure2:Dirty Image
###################

    original_dirty = dirty_image(original)
    reconstructed_dirty = dirty_image(reconstructed)

    # Compare real components using a common scale.
    reference_image = original_dirty.real
    prediction_image = reconstructed_dirty.real
    image_residual = reference_image - prediction_image

    limit = max(
        np.abs(reference_image).max(),
        np.abs(prediction_image).max(),
        1e-12,
    )

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5),
        constrained_layout=True,
    )

    images = [
        reference_image,
        prediction_image,
        image_residual,
    ]

    titles = [
        "Reference dirty image",
        "Reconstructed dirty image",
        "Dirty-image residual",
    ]

    for i, (ax, image, title) in enumerate(
        zip(axes, images, titles)
    ):
        im = ax.imshow(
            image,
            origin="lower",
            cmap="RdBu_r",
            vmin=-limit if i < 2 else None,
            vmax=limit if i < 2 else None,
            interpolation="nearest",
        )

        if i == 2:
            residual_limit = max(
                np.abs(image).max(), 1e-12
            )
            im.set_clim(
                -residual_limit, residual_limit
            )

        ax.set_title(title)
        ax.set_xlabel("Image pixel X")
        ax.set_ylabel("Image pixel Y")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        f"Fold {fold_number}: Dirty-image comparison"
    )

    fig.savefig(
        output_dir / f"fold_{fold_number}_dirty_image.png",
        dpi=200,
    )

    plt.close(fig)

  
    imag_ratio = (
        np.linalg.norm(original_dirty.imag)
        / max(np.linalg.norm(original_dirty), 1e-20)
    )

    print(
        f"  Reference dirty-image imaginary fraction: "
        f"{imag_ratio:.6f}"
    )

    return {
        "nmse": float(nmse),
        "snr_db": float(snr),
        "observed_cells": int(mask.sum()),
        "imaginary_fraction": float(imag_ratio),
    }
