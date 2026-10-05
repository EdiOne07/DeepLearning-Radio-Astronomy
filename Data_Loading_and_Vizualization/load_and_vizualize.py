import sys
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from rat_stack import TimeSlicedVisDataset


def inspect_and_plot(paths: list[str]) -> None:
    use_cuda=torch.cuda.is_available()
    dataset = TimeSlicedVisDataset(
        paths,
        window_size=8,
        target_size=1,
        stride=1,
        pin_memory=use_cuda,
        read_uvw=True,
    )
    indx=[0,1000,2000]
    u_coordinates_combined=[]
    v_coordinates_combined=[]
    vis_values_combined=[]

    for idx in range(0,len(dataset),50):
        item=dataset[idx]
        vis=item["vis"]
        uvw=item["uvw"]

        u_coordinates=uvw[:,:,0]
        v_coordinates=uvw[:,:,1]
        vis_values=vis[:,0,:,0]

        u_coordinates_combined.append(u_coordinates)
        v_coordinates_combined.append(v_coordinates)
        vis_values_combined.append(vis_values)

    u_all=torch.cat(u_coordinates_combined,dim=0)
    v_all=torch.cat(v_coordinates_combined,dim=0)
    vis_all=torch.cat(vis_values_combined,dim=0)
        
    print(dataset)
    print(f"Total time-sliced windows across {len(paths)} file(s): {len(dataset)}")
    print(f"Item size: {dataset.bytes_per_item / 2**20:.1f} MiB")

    item = dataset[0]
    vis = item["vis"]        
    uvw=item["uvw"]
    target = item["target"]  

    print("\n--- Tensor Shapes ---")
    print(f"vis shape:    {tuple(vis.shape)} | dtype: {vis.dtype}")
    print(f"target shape: {tuple(target.shape)} | dtype: {target.dtype}")
    print(f"uvw shape: {tuple(uvw.shape)} | dtype: {uvw.dtype}")

    #u_coordinates=uvw[:,:,0].flatten()
    #v_coordinates=uvw[:,:,1].flatten()

    #vis_values=vis[:,0,:,0].flatten()
    #amplitude=torch.abs(vis_values)
    #print(f"U_coordinates:",u_coordinates)
    #print(f"V_coordinates:",v_coordinates)
    #print(f"Amplitude:",amplitude)
    plot_fourier_space(u_all,v_all,vis_all)



def plot_fourier_space(u, v, visibilities, title="Fourier Space Sampling (UV Plane)"):
    """
    Plots sampled visibilities in 2D Fourier space (UV plane).
    
    Parameters:
        u: 1D array/tensor of u coordinates
        v: 1D array/tensor of v coordinates
        visibilities: 1D complex array/tensor of visibility measurements
    """
    if isinstance(u, torch.Tensor):
        u = u.cpu().numpy()
        v = v.cpu().numpy()
        visibilities = visibilities.cpu().numpy()

    u_full = np.concatenate([u, -u])
    v_full = np.concatenate([v, -v])
    vis_full = np.concatenate([visibilities, np.conj(visibilities)])

    amps = np.abs(vis_full)
    log_amps = np.log10(amps + 1e-6)

    plt.figure(figsize=(8, 8))
    
    scatter = plt.scatter(
        u_full, 
        v_full, 
        c=log_amps, 
        cmap="viridis", 
        s=12,          
        alpha=0.8,
        edgecolors="none"
    )
    
    plt.colorbar(scatter, label=r"Log10(|Visibility|) [Jy]")
    plt.title(title)
    plt.xlabel("u [wavelengths / meters]")
    plt.ylabel("v [wavelengths / meters]")
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.axis("equal")  # Keep aspect ratio square for spatial frequencies

    plt.tight_layout()
    plt.savefig("fourier_space_uv.png", dpi=300)
    plt.show()

   


if __name__ == "__main__":
    args = sys.argv[1:]
    paths = [p for arg in args for p in (glob.glob(arg) or [arg])]
    
    if not paths:
        sys.exit("Usage: python scripts/visualize_vis.py <file0.vis> [file1.vis ...]")
        
    inspect_and_plot(paths)