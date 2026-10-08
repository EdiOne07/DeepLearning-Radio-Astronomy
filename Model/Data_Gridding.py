import torch
from torch.utils.data import Dataset, DataLoader

class UVDataGridding:
    def __init__(self,grid_size:int=256,max_uv:float=70000.0):
        self.grid_size=grid_size
        self.max_uv=max_uv
        pass
    def  __call__(self, item:dict)->dict:
        uvw=item["uvw"]
        vis=item["vis"]

        u=uvw[:,:,0].flatten()
        v=uvw[:,:,1].flatten()
        visibility=vis[:,0,:,0].flatten()

        grid=torch.zeros((3,self.grid_size,self.grid_size),dtype=torch.float32)

        u_idx = ((u + self.max_uv) / (2 * self.max_uv) * (self.grid_size - 1)).long()
        v_idx = ((v + self.max_uv) / (2 * self.max_uv) * (self.grid_size - 1)).long()

        valid_points=((u_idx>=0) & (u_idx<self.grid_size) & (v_idx>=0) & (v_idx<self.grid_size))

        u_idx,v_idx,visibility=u_idx[valid_points],v_idx[valid_points],visibility[valid_points]
        grid[0,u_idx,v_idx]=visibility.real.float()
        grid[1,u_idx,v_idx]=visibility.imag.float()
        grid[2,u_idx,v_idx]=1.0

        return {
            "input_grid": grid,
        }
