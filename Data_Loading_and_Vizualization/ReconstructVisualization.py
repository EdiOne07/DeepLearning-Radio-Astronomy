"""Sparse patch UV reconstruction and diagnostic dirty-image visualizations."""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Model.MSE_Loss_Amplitude_Phase import polar_to_cartesian


def dirty_image(uv_grid):
    """Uncalibrated FFT diagnostic, not a physical or deconvolved sky image."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(uv_grid)))


def _rebin_sparse(patches, positions, mask, grid_size, patch_size, display_size=1024):
    """Scatter observed 8192-grid values onto a smaller display/FFT grid.

    Several virtual cells may map to one displayed cell; average them. This
    is deliberately a visualization product, not a full-resolution imaging step.
    """
    h, w = patches.shape[-2:]
    if (h, w) != (patch_size, patch_size):
        raise ValueError('Patch-size mismatch')
    n = len(patches)
    if not np.any(mask):
        return np.zeros((display_size, display_size), dtype=np.complex64)
    patch_index, local_r, local_c = np.nonzero(mask)
    virtual_r = positions[patch_index, 0] * patch_size + local_r
    virtual_c = positions[patch_index, 1] * patch_size + local_c
    display_r = np.clip((virtual_r.astype(np.int64) * display_size) // grid_size, 0, display_size - 1)
    display_c = np.clip((virtual_c.astype(np.int64) * display_size) // grid_size, 0, display_size - 1)
    values = patches[patch_index, local_r, local_c]
    flat_idx = display_r * display_size + display_c
    sums_real = np.bincount(flat_idx, weights=values.real, minlength=display_size**2)
    sums_imag = np.bincount(flat_idx, weights=values.imag, minlength=display_size**2)
    counts = np.bincount(flat_idx, minlength=display_size**2)
    nonzero = counts > 0
    result = sums_real + 1j * sums_imag
    result[nonzero] /= counts[nonzero]
    return result.reshape(display_size, display_size)


@torch.no_grad()
def visualize_reconstruction(model, dataset, device, amplitude_scale, output_dir,
                             fold_number, sample_index=0, patch_microbatch=128,
                             grid_size=8192, patch_size=16, display_grid_size=1024):
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = dataset[sample_index]
    inputs = sample['input_patch']
    targets = sample['target_patch']
    positions = sample['patch_position'].cpu().numpy()
    all_predictions = []
    for start in range(0, len(inputs), patch_microbatch):
        x = inputs[start:start + patch_microbatch].to(device)
        output = model(x)
        prediction = polar_to_cartesian(output, amplitude_scale=amplitude_scale)
        all_predictions.append(prediction.cpu())
    prediction = torch.cat(all_predictions, dim=0)
    occupancy = inputs[:, 2].bool()
    original_complex = torch.complex(targets[:, 0], targets[:, 1])
    prediction_complex = torch.complex(prediction[:, 0], prediction[:, 1])
    orig_observed = original_complex[occupancy]
    pred_observed = prediction_complex[occupancy]
    error_power = (orig_observed - pred_observed).abs().square().sum().item()
    signal_power = orig_observed.abs().square().sum().item()
    nmse = error_power / max(signal_power, 1e-20)
    snr_db = -10 * np.log10(nmse) if nmse > 0 else float('inf')

    densest = int(occupancy.sum(dim=(1, 2)).argmax().item())
    mask_patch = occupancy[densest].cpu().numpy()
    orig_patch = original_complex[densest].cpu().numpy()
    pred_patch = prediction_complex[densest].cpu().numpy()
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for row, (label, component) in enumerate([('Real', np.real), ('Imaginary', np.imag)]):
        orig = np.where(mask_patch, component(orig_patch), np.nan)
        pred = np.where(mask_patch, component(pred_patch), np.nan)
        delta = orig - pred
        scale = max(np.nanmax(abs(orig)), np.nanmax(abs(pred)), 1e-12)
        for col, (data, title) in enumerate([(orig, 'Reference'), (pred, 'Reconstructed'), (delta, 'Residual')]):
            im = axes[row, col].imshow(data, origin='lower', cmap='RdBu_r',
                                        vmin=-scale, vmax=scale, interpolation='nearest')
            axes[row, col].set_title(f'{title} {label}')
            fig.colorbar(im, ax=axes[row, col])
    fig.suptitle(f'Fold {fold_number} densest patch | NMSE across sample: {nmse:.5f}')
    fig.savefig(output_dir / 'patch_reconstruction.png', dpi=200)
    plt.close(fig)

    orig_np = original_complex.cpu().numpy()
    pred_np = prediction_complex.cpu().numpy()
    mask_np = occupancy.cpu().numpy()
    original_uv = _rebin_sparse(orig_np, positions, mask_np, grid_size, patch_size, display_grid_size)
    reconstructed_uv = _rebin_sparse(pred_np, positions, mask_np, grid_size, patch_size, display_grid_size)
    uv_residual = original_uv - reconstructed_uv
    panels = [np.log1p(abs(original_uv)), np.log1p(abs(reconstructed_uv)), np.log1p(abs(uv_residual))]
    common_max = max(panels[0].max(), panels[1].max(), 1e-12)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for i, (panel, title) in enumerate(zip(panels, ['Reference UV amplitude', 'Reconstructed UV amplitude', 'Absolute UV residual'])):
        im = axes[i].imshow(panel, origin='lower', cmap='viridis', vmin=0,
                            vmax=common_max if i < 2 else None, interpolation='nearest')
        axes[i].set_title(title)
        fig.colorbar(im, ax=axes[i])
    fig.suptitle(f'Fold {fold_number}: {display_grid_size}×{display_grid_size} rebinned UV display | '
                 f'Occupied-cell NMSE {nmse:.5f}; SNR {snr_db:.2f} dB')
    fig.savefig(output_dir / 'uv_reconstruction.png', dpi=200)
    plt.close(fig)

    original_dirty = dirty_image(original_uv).real
    predicted_dirty = dirty_image(reconstructed_uv).real
    dirty_residual = original_dirty - predicted_dirty
    limit = max(abs(original_dirty).max(), abs(predicted_dirty).max(), 1e-12)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for i, (panel, title) in enumerate(zip([original_dirty, predicted_dirty, dirty_residual],
                                            ['Reference dirty image', 'Reconstructed dirty image', 'Dirty-image residual'])):
        scale = limit if i < 2 else max(abs(panel).max(), 1e-12)
        im = axes[i].imshow(panel, origin='lower', cmap='RdBu_r', vmin=-scale, vmax=scale,
                            interpolation='nearest')
        axes[i].set_title(title)
        fig.colorbar(im, ax=axes[i])
    fig.suptitle(f'Fold {fold_number}: FFT of rebinned sparse UV samples (diagnostic only)')
    fig.savefig(output_dir / 'dirty_image.png', dpi=200)
    plt.close(fig)
    print(f'Fold {fold_number} reconstruction images saved: sample {sample_index}; '
          f'occupied cells {int(occupancy.sum())}, NMSE={nmse:.6f}, SNR={snr_db:.2f} dB', flush=True)
    return {'nmse': float(nmse), 'snr_db': float(snr_db), 'occupied_cells': int(occupancy.sum())}