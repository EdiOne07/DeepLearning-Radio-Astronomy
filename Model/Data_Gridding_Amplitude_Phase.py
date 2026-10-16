import torch

class UVDataGridding_Phase:
    def __init__(self, grid_size: int = 256, max_uv: float = 70000.0,amplitude_scale=1.0):
        self.grid_size = grid_size
        self.max_uv = max_uv
        self.amplitude_scale=amplitude_scale

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

        real=torch.zeros((self.grid_size,self.grid_size),dtype=torch.float32)
        imaginary=torch.zeros((self.grid_size,self.grid_size),dtype=torch.float32)
        counts = torch.zeros((self.grid_size, self.grid_size), dtype=torch.float32)


        real.index_put_((v_idx,u_idx),visibility.real.float(),accumulate=True)
        imaginary.index_put_((v_idx,u_idx),visibility.imag.float(),accumulate=True)
        counts.index_put_((v_idx, u_idx), torch.ones_like(v_idx, dtype=torch.float32), accumulate=True)

        input_grid = torch.zeros((3, self.grid_size, self.grid_size), dtype=torch.float32)
        
        mask = (counts > 0)
     

        real[mask] /= counts[mask]
        imaginary[mask] /= counts[mask]
        target_grid=torch.stack((real,imaginary),dim=0)

        complex_grid = torch.complex(real, imaginary)
        #Amplitude and Phase tensors
        amplitude=torch.abs(complex_grid)
        phase=torch.atan2(imaginary,real)
        amplitude_normalization=torch.log1p((amplitude/self.amplitude_scale))

        input_grid[0][mask]=amplitude_normalization[mask]
        input_grid[1][mask]=phase[mask]
        input_grid[2][mask] = 1.0  


        return {
            "input_grid": input_grid,
            "target_grid":target_grid,
        }