
import sys
from pathlib import Path

import torch

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent

sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from rat_stack import TimeSlicedVisDataset
from Data_Gridding_Amplitude_Phase import UVDataGridding_Phase


GRID_SIZE = 256
MAX_UV = 70000
AMPLITUDE_SCALE = 0.1


def main():
    if len(sys.argv) != 2:
        raise ValueError(
            "Usage: python Model/Test_UV_Gridding.py observation.vis"
        )

    path = sys.argv[1]

    dataset = TimeSlicedVisDataset(
        paths=[path],
        window_size=1,
        target_size=1,
        stride=1,
        pin_memory=False,
        read_uvw=True,
    )

    item = dataset[0]

    vis = item["vis"]
    uvw = item["uvw"]

    # Same selection as the training transform.
    selected_vis = vis[:, 0, :, 0].reshape(-1)
    u = uvw[:, :, 0].reshape(-1)
    v = uvw[:, :, 1].reshape(-1)

    initial_count = selected_vis.numel()

    finite_mask = (
        torch.isfinite(u)
        & torch.isfinite(v)
        & torch.isfinite(selected_vis.real)
        & torch.isfinite(selected_vis.imag)
    )

    finite_count = finite_mask.sum().item()

    range_mask = (
        (u >= -MAX_UV)
        & (u <= MAX_UV)
        & (v >= -MAX_UV)
        & (v <= MAX_UV)
    )

    valid_mask = finite_mask & range_mask

    valid_count = valid_mask.sum().item()

    u_valid = u[valid_mask]
    v_valid = v[valid_mask]

    u_idx = (
        (u_valid + MAX_UV)
        / (2 * MAX_UV)
        * (GRID_SIZE - 1)
    ).long()

    v_idx = (
        (v_valid + MAX_UV)
        / (2 * MAX_UV)
        * (GRID_SIZE - 1)
    ).long()

    flat_idx = v_idx * GRID_SIZE + u_idx

    unique_cells, cell_counts = torch.unique(
        flat_idx,
        return_counts=True,
    )

    unique_count = unique_cells.numel()
    collisions = valid_count - unique_count

    print("\n=== UV GRIDDING DIAGNOSTIC ===")
    print("Input visibility shape:", tuple(vis.shape))
    print("UVW shape:", tuple(uvw.shape))

    print("\n--- Filtering ---")
    print("Initial measurements:", initial_count)
    print("Finite measurements:", finite_count)
    print("After UV-range filtering:", valid_count)
    print("Removed as non-finite:", initial_count - finite_count)
    print("Removed by UV-range:", finite_count - valid_count)

    print("\n--- Coordinates ---")
    print("u min/max:", u_valid.min().item(), u_valid.max().item())
    print("v min/max:", v_valid.min().item(), v_valid.max().item())

    print("\n--- Gridding ---")
    print("Valid measurements:", valid_count)
    print("Unique occupied cells:", unique_count)
    print("Additional measurements sharing cells:", collisions)

    print(
        "Maximum measurements in one cell:",
        cell_counts.max().item(),
    )

    print(
        "Mean measurements per occupied cell:",
        cell_counts.float().mean().item(),
    )

    print(
        "Grid occupancy:",
        f"{100 * unique_count / (GRID_SIZE ** 2):.4f}%",
    )

    print("\n--- Actual transform comparison ---")

    transform = UVDataGridding_Phase(
        GRID_SIZE,
        MAX_UV,
        AMPLITUDE_SCALE,
    )

    transformed = transform(item)

    actual_occupied = int(
        transformed["input_grid"][2].sum().item()
    )

    print("Occupied cells from transform:", actual_occupied)

    if actual_occupied != unique_count:
        print(
            "WARNING: Calculated occupancy differs "
            "from the actual transform!"
        )
    else:
        print("Occupancy matches the transform.")


if __name__ == "__main__":
    main()
