
import sys
from pathlib import Path

import torch

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent

sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from rat_stack import TimeSlicedVisDataset
from Data_Gridding_Amplitude_Phase import UVDataGridding_Phase


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

GRID_SIZE = 256
MAX_UV = 70000
AMPLITUDE_SCALE = 0.1

CONFIGURATIONS = [
    (256, 70000),    # Current baseline
    (256, 65000),
    (256, 40000),
    (512, 70000),
    (512, 65000),
    (512, 40000),
    (1024, 70000),
    (1024, 65000),
    (2048, 70000),
]


# --------------------------------------------------
# GRIDDING ANALYSIS
# --------------------------------------------------

def analyze_gridding(u, v, selected_vis, grid_size, max_uv):
    """
    Analyze the number of measurements retained and
    their distribution across UV grid cells.

    Does not allocate a dense grid.
    """

    total = selected_vis.numel()

    finite_mask = (
        torch.isfinite(u)
        & torch.isfinite(v)
        & torch.isfinite(selected_vis.real)
        & torch.isfinite(selected_vis.imag)
    )

    finite_count = int(finite_mask.sum().item())

    range_mask = (
        (u >= -max_uv)
        & (u <= max_uv)
        & (v >= -max_uv)
        & (v <= max_uv)
    )

    valid_mask = finite_mask & range_mask

    retained = int(valid_mask.sum().item())

    if retained == 0:
        return {
            "grid_size": grid_size,
            "max_uv": max_uv,
            "total": total,
            "finite": finite_count,
            "retained": 0,
            "occupied": 0,
            "collisions": 0,
            "max_per_cell": 0,
            "mean_per_cell": 0.0,
            "occupancy_percent": 0.0,
            "unique_percent": 0.0,
            "retained_percent": 0.0,
        }

    u_valid = u[valid_mask]
    v_valid = v[valid_mask]

    # Same coordinate mapping as the actual transform.
    u_idx = (
        (u_valid + max_uv)
        / (2 * max_uv)
        * (grid_size - 1)
    ).long()

    v_idx = (
        (v_valid + max_uv)
        / (2 * max_uv)
        * (grid_size - 1)
    ).long()

    flat_idx = v_idx * grid_size + u_idx

    unique_cells, cell_counts = torch.unique(
        flat_idx,
        return_counts=True,
    )

    occupied = int(unique_cells.numel())
    collisions = retained - occupied

    return {
        "grid_size": grid_size,
        "max_uv": max_uv,
        "total": total,
        "finite": finite_count,
        "retained": retained,
        "occupied": occupied,
        "collisions": collisions,
        "max_per_cell": int(cell_counts.max().item()),
        "mean_per_cell": cell_counts.float().mean().item(),
        "occupancy_percent": 100 * occupied / (grid_size ** 2),
        "unique_percent": 100 * occupied / total,
        "retained_percent": 100 * retained / total,
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

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

    finite_count = int(finite_mask.sum().item())

    print("\n=== UV GRIDDING DIAGNOSTIC ===")
    print("Observation:", path)
    print("Input visibility shape:", tuple(vis.shape))
    print("UVW shape:", tuple(uvw.shape))

    print("\n--- Input measurements ---")
    print("Initial measurements:", initial_count)
    print("Finite measurements:", finite_count)
    print("Non-finite measurements:", initial_count - finite_count)

    if finite_count == 0:
        raise RuntimeError("No finite UV measurements found.")

    print("\n--- Coordinate ranges ---")
    print(
        "u min/max:",
        u[finite_mask].min().item(),
        u[finite_mask].max().item(),
    )
    print(
        "v min/max:",
        v[finite_mask].min().item(),
        v[finite_mask].max().item(),
    )

    # --------------------------------------------------
    # CONFIGURATION SWEEP
    # --------------------------------------------------

    print("\n=== GRID CONFIGURATION SWEEP ===")

    print(
        f"{'Grid':>6} "
        f"{'MAX_UV':>8} "
        f"{'Retained':>9} "
        f"{'Occupied':>9} "
        f"{'Collisions':>11} "
        f"{'Kept %':>9} "
        f"{'Unique %':>10} "
        f"{'Max/Cell':>9}"
    )

    print("-" * 94)

    results = []

    for grid_size, max_uv in CONFIGURATIONS:
        result = analyze_gridding(
            u,
            v,
            selected_vis,
            grid_size,
            max_uv,
        )

        results.append(result)

        print(
            f"{result['grid_size']:>6} "
            f"{result['max_uv']:>8} "
            f"{result['retained']:>9} "
            f"{result['occupied']:>9} "
            f"{result['collisions']:>11} "
            f"{result['retained_percent']:>8.2f}% "
            f"{result['unique_percent']:>9.2f}% "
            f"{result['max_per_cell']:>9}"
        )

    # --------------------------------------------------
    # BASELINE DETAILS
    # --------------------------------------------------

    baseline = analyze_gridding(
        u,
        v,
        selected_vis,
        GRID_SIZE,
        MAX_UV,
    )

    print("\n=== BASELINE DETAILS ===")

    print("Grid size:", GRID_SIZE)
    print("MAX_UV:", MAX_UV)

    print("Initial measurements:", baseline["total"])
    print("Finite measurements:", baseline["finite"])
    print("After UV filtering:", baseline["retained"])

    print(
        "Removed as non-finite:",
        baseline["total"] - baseline["finite"],
    )

    print(
        "Removed by UV range:",
        baseline["finite"] - baseline["retained"],
    )

    print("Unique occupied cells:", baseline["occupied"])
    print("Additional measurements sharing cells:", baseline["collisions"])

    print(
        "Maximum measurements in one cell:",
        baseline["max_per_cell"],
    )

    print(
        "Mean measurements per occupied cell:",
        baseline["mean_per_cell"],
    )

    print(
        "Grid occupancy:",
        f"{baseline['occupancy_percent']:.4f}%",
    )

    # --------------------------------------------------
    # VERIFY AGAINST ACTUAL TRANSFORM
    # --------------------------------------------------

    print("\n=== ACTUAL TRANSFORM COMPARISON ===")

    transform = UVDataGridding_Phase(
        GRID_SIZE,
        MAX_UV,
        AMPLITUDE_SCALE,
    )

    transformed = transform(item)

    actual_occupied = int(
        transformed["input_grid"][2].sum().item()
    )

    print("Calculated occupied cells:", baseline["occupied"])
    print("Occupied cells from transform:", actual_occupied)

    if actual_occupied != baseline["occupied"]:
        print("WARNING: Occupancy does not match the transform!")
    else:
        print("Occupancy matches the transform.")

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    print("\n=== CONFIGURATION SUMMARY ===")

    # Rank configurations by the number of original
    # measurements mapped to distinct UV cells.
    ranked = sorted(
        results,
        key=lambda r: r["occupied"],
        reverse=True,
    )

    for result in ranked[:3]:
        print(
            f"{result['grid_size']}x{result['grid_size']} "
            f"MAX_UV={result['max_uv']} | "
            f"Retained={result['retained']} | "
            f"Occupied={result['occupied']} | "
            f"Collisions={result['collisions']}"
        )

    print(
        "\nNote: Higher occupancy does not automatically "
        "mean better compression or reconstruction."
    )


if __name__ == "__main__":
    main()
