import torch
from rat_stack import TimeSlicedVisDataset
import sys
import glob
from pathlib import Path
from Visibility_Standerdizer import VisibilityStandardizer
class PolarizationStandardization:
   def __call__(self, standardized_vis_tensor,time_step=0):
       time_step=standardized_vis_tensor[time_step]
       real_part=time_step.real
       imag_part=time_step.imag

       standardized_vis_tensor=torch.stack((real_part,imag_part),dim=-1)
       standardized_vis_tensor=torch.permute(standardized_vis_tensor,dims=(2,3,0,1))
       channel_shaping=torch.reshape(standardized_vis_tensor,(8,64,2278))

       return torch.unsqueeze(channel_shaping,dim=0)
       

def test_polarization_representation(transform, standardized_vis):
    import torch

    model_input = transform(standardized_vis)

    T, F, B, P = standardized_vis.shape

    assert model_input.shape == (1, 2 * P, F, B), (
        f"Expected {(1, 2 * P, F, B)}, "
        f"got {tuple(model_input.shape)}"
    )

    assert not torch.is_complex(model_input)
    assert torch.isfinite(model_input).all()

    for p in range(P):
        expected_real = standardized_vis[0, :, :, p].real
        expected_imag = standardized_vis[0, :, :, p].imag

        actual_real = model_input[0, 2 * p]
        actual_imag = model_input[0, 2 * p + 1]

        assert torch.allclose(actual_real, expected_real), (
            f"Real component mismatch at polarization {p}"
        )

        assert torch.allclose(actual_imag, expected_imag), (
            f"Imaginary component mismatch at polarization {p}"
        )

    print("Input shape:", tuple(model_input.shape))
    print("All 8 channels contain the correct values.")
    print("Polarization representation test passed.")


if __name__ == "__main__":
    args = sys.argv[1:]

    paths = sorted({
        str(Path(p).resolve())
        for arg in args
        for p in (glob.glob(arg) or [arg])
        if Path(p).is_file()
    })
    standardizer = VisibilityStandardizer()
    data=TimeSlicedVisDataset(paths=paths,window_size=8,target_size=1,stride=1,pin_memory=False,read_uvw=True)
    standardizer.fit(data[0]["vis"])

    normalized = standardizer.transform(data[0]["vis"])

    transform = PolarizationStandardization()

    test_polarization_representation(transform, normalized)
