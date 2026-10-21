
    
  
"""Inspect and pack occupied patches of a high-resolution UV grid.

No dense GRID_SIZE x GRID_SIZE tensor is allocated.
Run: python Model/Test_UV_Patches.py /path/to/observation.vis
"""

import sys
import torch
from rat_stack import TimeSlicedVisDataset

GRID_SIZE = 8192
MAX_UV = 70000.0
AMPLITUDE_SCALE = 0.1
PATCH_SIZES = (16, 32, 64, 128)
SELECTED_PATCH_SIZE = 32


def select_visibilities(sample):
    vis = torch.as_tensor(sample["vis"])
    uvw = torch.as_tensor(sample["uvw"])

    values = vis[:, 0, :, 0].reshape(-1)  # first channel/polarization
    u = uvw[:, :, 0].reshape(-1)
    v = uvw[:, :, 1].reshape(-1)

    valid = (
        torch.isfinite(u)
        & torch.isfinite(v)
        & torch.isfinite(values.real)
        & torch.isfinite(values.imag)
        & (u >= -MAX_UV)
        & (u <= MAX_UV)
        & (v >= -MAX_UV)
        & (v <= MAX_UV)
    )

    values, u, v = values[valid], u[valid], v[valid]
    x = ((u + MAX_UV) / (2 * MAX_UV) * (GRID_SIZE - 1)).long()
    y = ((v + MAX_UV) / (2 * MAX_UV) * (GRID_SIZE - 1)).long()
    x = x.clamp(0, GRID_SIZE - 1)
    y = y.clamp(0, GRID_SIZE - 1)
    return values, x, y, int(valid.numel())


def patch_assignment(x, y, patch_size):
    # Coordinates in the patch grid, not the high-resolution UV grid.
    n_per_axis = (GRID_SIZE + patch_size - 1) // patch_size
    patch_id = (y // patch_size) * n_per_axis + (x // patch_size)
    active_ids, inverse, vis_per_patch = torch.unique(
        patch_id, sorted=True, return_inverse=True, return_counts=True
    )
    patch_y = torch.div(active_ids, n_per_axis, rounding_mode="floor")
    patch_x = active_ids % n_per_axis
    positions = torch.stack((patch_y, patch_x), dim=1)
    return positions, inverse, vis_per_patch


def pack_patches(values, x, y, patch_size):
    """Return only occupied patches, preserving patch-grid positions.

    Values landing in the same fine UV cell are averaged, exactly as with
    regular gridding. The integer counts are retained for diagnostics.
    """
    positions, patch_inverse, _ = patch_assignment(x, y, patch_size)
    n_patches = positions.shape[0]
    cells_per_patch = patch_size * patch_size

    local_x = x % patch_size
    local_y = y % patch_size
    flat_index = patch_inverse * cells_per_patch + local_y * patch_size + local_x
    total_cells = n_patches * cells_per_patch

    sums_re = torch.zeros(total_cells, dtype=torch.float32)
    sums_im = torch.zeros_like(sums_re)
    counts = torch.zeros_like(sums_re)
    sums_re.index_add_(0, flat_index, values.real.float())
    sums_im.index_add_(0, flat_index, values.imag.float())
    counts.index_add_(0, flat_index, torch.ones_like(values.real.float()))

    denom = counts.clamp_min(1.0)
    mean_re = (sums_re / denom).reshape(n_patches, patch_size, patch_size)
    mean_im = (sums_im / denom).reshape(n_patches, patch_size, patch_size)
    mask = (counts > 0).float().reshape(n_patches, patch_size, patch_size)
    counts = counts.reshape(n_patches, patch_size, patch_size)

    amplitude = torch.hypot(mean_re, mean_im)
    log_amplitude = torch.log1p(amplitude / AMPLITUDE_SCALE)
    phase = torch.atan2(mean_im, mean_re)

    # Same channel convention as the existing 2D amplitude/phase model.
    input_patches = torch.stack((log_amplitude, phase, mask), dim=1)
    target_patches = torch.stack((mean_re, mean_im), dim=1)
    return positions, input_patches, target_patches, counts


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python Model/Test_UV_Patches.py observation.vis")

    dataset = TimeSlicedVisDataset(
        paths=[sys.argv[1]],
        window_size=1,
        target_size=1,
        stride=1,
        pin_memory=False,
        read_uvw=True,
    )
    sample = dataset[0]
    values, x, y, total = select_visibilities(sample)
    retained = len(values)
    fine_cell_ids = y * GRID_SIZE + x
    unique_fine_cells = torch.unique(fine_cell_ids).numel()

    print("\n=== SPARSE UV PATCH DIAGNOSTIC ===")
    print("Observation:", sys.argv[1])
    print("Full grid:", f"{GRID_SIZE} x {GRID_SIZE}")
    print("Original visibilities:", total)
    print("Retained visibilities:", retained)
    print("Occupied high-res grid cells:", unique_fine_cells)
    print("Fine-grid collisions:", retained - unique_fine_cells)

    print("\n" + "=" * 105)
    print(
        f"{'Patch':>6} {'Active':>8} {'All':>9} {'Cells processed':>16} "
        f"{'Grid used %':>12} {'Patch fill %':>13} "
        f"{'Input MiB':>11} {'16ch MiB':>11}"
    )
    print("-" * 105)

    for patch_size in PATCH_SIZES:
        if GRID_SIZE % patch_size:
            raise ValueError("Choose patch sizes that divide GRID_SIZE")
        positions, _, vis_per_patch = patch_assignment(x, y, patch_size)
        active = positions.shape[0]
        all_patches = (GRID_SIZE // patch_size) ** 2
        processed_cells = active * patch_size**2
        grid_used_pct = 100.0 * processed_cells / (GRID_SIZE**2)
        patch_fill_pct = 100.0 * unique_fine_cells / max(processed_cells, 1)
        input_mib = processed_cells * 3 * 4 / (1024**2)
        feature_mib = processed_cells * 16 * 4 / (1024**2)
        print(
            f"{patch_size:>6} {active:>8} {all_patches:>9} "
            f"{processed_cells:>16} {grid_used_pct:>11.4f}% "
            f"{patch_fill_pct:>12.3f}% "
            f"{input_mib:>11.2f} {feature_mib:>11.2f}"
        )
        print(
            f"       Visibilities per active patch: "
            f"mean={vis_per_patch.float().mean():.2f}, "
            f"max={int(vis_per_patch.max())}"
        )

    print(f"\n=== BUILDING {SELECTED_PATCH_SIZE}x{SELECTED_PATCH_SIZE} ACTIVE PATCHES ===")
    positions, input_patches, targets, counts = pack_patches(
        values, x, y, SELECTED_PATCH_SIZE
    )
    print("Patch coordinates [row, col]:", tuple(positions.shape))
    print("Inputs [patches, 3, P, P]:", tuple(input_patches.shape))
    print("Targets [patches, 2, P, P]:", tuple(targets.shape))
    print("Occupied cells:", int((counts > 0).sum().item()))
    print("Measurements represented:", int(counts.sum().item()))

    assert int(counts.sum().item()) == retained
    assert int((counts > 0).sum().item()) == unique_fine_cells
    assert torch.isfinite(input_patches).all()
    assert torch.isfinite(targets).all()
    print("PASS: All valid measurements accounted for; no dense full-grid allocation.")
    print("NOTE: Fine-grid cell collisions are still averaged; counts are retained.")


if __name__ == "__main__":
    main()

