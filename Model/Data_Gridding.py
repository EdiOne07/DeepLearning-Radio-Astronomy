import torch

class UVDataGridding:
    def __init__(self, grid_size: int = 256, max_uv: float = 70000.0):
        self.grid_size = grid_size
        self.max_uv = max_uv

    def __call__(self, item: dict) -> dict:
        uvw = item["uvw"]
        vis = item["vis"]

        u = uvw[:, :, 0].flatten()
        v = uvw[:, :, 1].flatten()
        visibility = vis[:, 0, :, 0].flatten()

        u_idx = ((u + self.max_uv) / (2 * self.max_uv) * (self.grid_size - 1)).long()
        v_idx = ((v + self.max_uv) / (2 * self.max_uv) * (self.grid_size - 1)).long()

        valid_points = (u_idx >= 0) & (u_idx < self.grid_size) & (v_idx >= 0) & (v_idx < self.grid_size)
        u_idx, v_idx = u_idx[valid_points], v_idx[valid_points]
        visibility = visibility[valid_points]

        grid = torch.zeros((3, self.grid_size, self.grid_size), dtype=torch.float32)

        grid[0].index_put_((v_idx, u_idx), visibility.real.float(), accumulate=True)
        grid[1].index_put_((v_idx, u_idx), visibility.imag.float(), accumulate=True)
        
        counts = torch.zeros((self.grid_size, self.grid_size), dtype=torch.float32)
        counts.index_put_((v_idx, u_idx), torch.ones_like(v_idx, dtype=torch.float32), accumulate=True)

        mask = (counts > 0)
        grid[0][mask] /= counts[mask]
        grid[1][mask] /= counts[mask]
        grid[2][mask] = 1.0  

        if mask.any():
            real_std = grid[0][mask].std() + 1e-8
            imag_std = grid[1][mask].std() + 1e-8
            grid[0][mask] /= real_std
            grid[1][mask] /= imag_std

        return {
            "input_grid": grid,
        }