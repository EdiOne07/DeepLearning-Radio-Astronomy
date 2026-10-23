import torch

class UV_Patch_Transform:
    def __init__(self, grid_size: int = 8192, patch_size: int =16,max_uv: float = 70000.0,amplitude_scale=0.1,handle_collision_strategy: str="mean"):
        if grid_size%patch_size!=0:
            raise Exception("Grid size has to be perfectly divided by the patch size")
        self.grid_size = grid_size
        self.patch_size=patch_size
        self.max_uv = max_uv
        self.amplitude_scale=amplitude_scale
        self.handle_collision_strategy=handle_collision_strategy

    def __call__(self, item: dict) -> dict:

        uvw = item["uvw"]
        vis = item["vis"]

        u = uvw[:, :, 0].flatten()
        v = uvw[:, :, 1].flatten()
        visibility = vis[:, 0, :, 0].flatten()

        valid_points = (
        torch.isfinite(u)
        & torch.isfinite(v)
        & torch.isfinite(visibility.real)
        & torch.isfinite(visibility.imag)
        & (u >= -self.max_uv)
        & (u <= self.max_uv)
        & (v >= -self.max_uv)
        & (v <= self.max_uv)
        )

        u= u[valid_points]
        v = v[valid_points]
        visibility = visibility[valid_points]

        u_idx = ((u + self.max_uv)/(2 * self.max_uv)*(self.grid_size - 1)).long()
        v_idx = ((v + self.max_uv)/(2 * self.max_uv)*(self.grid_size - 1)).long()

        #Patch logic
        patch_row=v_idx//self.patch_size
        patch_col=u_idx//self.patch_size

        local_row=v_idx%self.patch_size
        local_col=u_idx%self.patch_size

        patch_axis=self.grid_size//self.patch_size

        patch_id=patch_row*patch_axis+patch_col

        unique_patches,inverse=torch.unique(patch_id,sorted=True,return_inverse=True)

        patch_positions=torch.stack(
            [
                unique_patches//patch_axis,
                unique_patches%patch_axis,
            ],
            dim=1
        )

        n_patches=unique_patches.numel()

        real=torch.zeros((n_patches, self.patch_size,self.patch_size),dtype=torch.float32)
        imaginary=torch.zeros_like(real)
        counts=torch.zeros_like(real)

        real.index_put_((inverse,local_row,local_col),visibility.real.float(),accumulate=True)
        imaginary.index_put_((inverse,local_row,local_col),visibility.imag.float(),accumulate=True)
        counts.index_put_((inverse,local_row,local_col), torch.ones_like(v_idx, dtype=torch.float32), accumulate=True)

        input_grid = torch.zeros((n_patches, 3 ,self.patch_size, self.patch_size), dtype=torch.float32)
        
        mask = (counts > 0)
     
        real[mask] /= counts[mask]
        imaginary[mask] /= counts[mask]
        target_grid=torch.stack((real,imaginary),dim=1)

        complex_grid = torch.complex(real, imaginary)
        #Amplitude and Phase tensors
        amplitude=torch.abs(complex_grid)
        phase=torch.atan2(imaginary,real)
        amplitude_normalization=torch.log1p((amplitude/self.amplitude_scale))

        input_grid[:,0][mask]=amplitude_normalization[mask]
        input_grid[:,1][mask]=phase[mask]
        input_grid[:,2] = mask.float()  
        return {
            "input_patch": input_grid,
            "target_patch":target_grid,
            "patch_position":patch_positions,
            "counts":counts,
        }