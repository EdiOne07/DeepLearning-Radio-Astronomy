import torch
from rat_stack import TimeSlicedVisDataset
import sys
import glob
from pathlib import Path

class VisibilityStandardizer:

    def fit(self, vis):
      self.real_mean=torch.mean(vis.real,dim=(0,1,2))
      self.imag_mean=torch.mean(vis.imag,dim=(0,1,2))

      self.real_std=torch.std(vis.real,dim=(0,1,2),correction=0)
      self.imag_std=torch.std(vis.imag,dim=(0,1,2),correction=0)

      self.real_std=torch.clamp_min(self.real_std,1.19209e-07)
      self.imag_std=torch.clamp_min(self.imag_std,1.19209e-07)
      

    def transform(self, vis):
       real=(vis.real-self.real_mean)/self.real_std
       imag=(vis.imag-self.imag_mean)/self.imag_std

       return torch.complex(real,imag)

    def inverse_transform(self, vis):
        real=vis.real*self.real_std + self.real_mean
        imag=vis.imag*self.imag_std + self.imag_mean

        return torch.complex(real,imag)


def test_transform(sample):
    vis = sample["vis"]

    standardizer = VisibilityStandardizer()
    standardizer.fit(vis)

    normalized = standardizer.transform(vis)

    assert normalized.shape == vis.shape
    assert torch.is_complex(normalized)

    for p in range(vis.shape[-1]):
        real = normalized[..., p].real
        imag = normalized[..., p].imag

        assert abs(real.mean().item()) < 1e-4
        assert abs(imag.mean().item()) < 1e-4

        assert abs(real.std(correction=0).item() - 1) < 1e-3
        assert abs(imag.std(correction=0).item() - 1) < 1e-3

    print("Transform test passed.")

def test_inverse_transform(sample):
    vis = sample["vis"]

    standardizer = VisibilityStandardizer()
    standardizer.fit(vis)

    normalized = standardizer.transform(vis)
    restored = standardizer.inverse_transform(normalized)

    assert restored.shape == vis.shape
    assert torch.is_complex(restored)
    assert torch.isfinite(restored.real).all()
    assert torch.isfinite(restored.imag).all()

    absolute_error = (restored - vis).abs()

    print("Maximum absolute error:", absolute_error.max().item())
    print("Mean absolute error:", absolute_error.mean().item())

    assert torch.allclose(
        restored,
        vis,
        atol=1e-6,
        rtol=1e-5,
    ), "Round-trip reconstruction failed"

    print("Inverse transform test passed.")

def test_standardizer_reuse(dataset):
    standardizer = VisibilityStandardizer()
    standardizer.fit(dataset[0]["vis"])

    original_mean = standardizer.real_mean.clone()
    original_std = standardizer.real_std.clone()

    other = dataset[1]["vis"]
    normalized = standardizer.transform(other)
    restored = standardizer.inverse_transform(normalized)

    assert torch.equal(standardizer.real_mean, original_mean)
    assert torch.equal(standardizer.real_std, original_std)
    assert torch.allclose(restored, other, atol=1e-6, rtol=1e-5)

    print("Standardizer reuse test passed.")

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
    sample=data[10]
    test_standardizer_reuse(data)
