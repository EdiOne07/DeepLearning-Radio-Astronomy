import torch
from torch.utils.data import DataLoader
import Data_Gridding
from rat_stack import TimeSlicedVisDataset
import sys
from pathlib import Path

chuchichaestli_src = Path("/home/edione/Projects/chuchichaestli/src") 
if str(chuchichaestli_src) not in sys.path:
    sys.path.insert(0, str(chuchichaestli_src))

from chuchichaestli.models.autoencoder import Autoencoder
import sys
import glob

transform = Data_Gridding.UVDataGridding(grid_size=256, max_uv=70000.0)
class TransformerDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self.transform(item)
    
def train_model(paths:list[str])->None:    
    use_cuda=torch.cuda.is_available()
    dataset=TimeSlicedVisDataset(paths=paths,
                                window_size=8,
                                target_size=1,
                                stride=1,
                                pin_memory=use_cuda,
                                read_uvw=True,)

    train_dataset=TransformerDataset(dataset,transform)
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=2
    )

    device= torch.device("cuda" if use_cuda else "cpu")
    model = Autoencoder.build(
        dimensions=2,          
        in_channels=3,           
        out_channels=1,          
        latent_dim=16,           
        res_act_fn="silu",    
        res_norm_type="group",   
        encoder_args={"n_channels": 16,},
    ).to(device)
    criterion=torch.nn.MSELoss()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)

    model.train()

    for epoch in range(5):
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch["input_grid"].to(device)

            targets = inputs[
                :, 0:1, :, :
            ]

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch+1}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.6f}"
                )
if __name__ == "__main__":
    args = sys.argv[1:]
    paths = [p for arg in args for p in (glob.glob(arg) or [arg])]
    
    if not paths:
        sys.exit("Usage: python scripts/visualize_vis.py <file0.vis> [file1.vis ...]")
        
    train_model(paths)