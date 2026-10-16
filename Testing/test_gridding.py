from rat_stack import TimeSlicedVisDataset
import sys
import glob
from pathlib import Path
def test_gridding(paths:list[str]):
    dataset = TimeSlicedVisDataset(
        paths=paths,
        window_size=1,
        target_size=1,
        read_uvw=True,
        pin_memory=False
    )

    print("Frequency channels:", dataset.num_channels)
    print("Polarization products:", dataset.num_pols)
    print("Baselines:", dataset.num_baselines)

    sample = dataset[0]

    print("Visibility shape:", sample["vis"].shape)
    print("UVW shape:", sample["uvw"].shape)

if __name__ == "__main__":
    args = sys.argv[1:]

    paths = sorted({
        str(Path(p).resolve())
        for arg in args
        for p in (glob.glob(arg) or [arg])
        if Path(p).is_file()
    })


    test_gridding(paths)  