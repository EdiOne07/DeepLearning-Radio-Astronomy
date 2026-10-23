import csv
import glob
import json
import math
import random
import sys
from enum import Enum
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rat_stack import TimeSlicedVisDataset
from Data_Gridding_Patches import UV_Patch_Transform
from MSE_Loss_Amplitude_Phase import MSELoss_Amplitude_Phase, polar_to_cartesian
from Data_Loading_and_Vizualization.ReconstructVisualization import visualize_reconstruction
from chuchichaestli.models.autoencoder import Autoencoder

WINDOW_SIZE = 8
TARGET_SIZE = 1
STRIDE = 1
BATCH_SIZE = 16 
PATCH_MICROBATCH = 128  # maximum patches passed through model at once
EPOCHS = 50
K_FOLDS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRID_SIZE = 8192
PATCH_SIZE = 16
MAX_UV = 70000
AMPLITUDE_SCALE = 0.1
HANDLE_COLLISION = 'mean'
LATENT_DIM = 64
SEED = 42

OUTPUT_DIR = PROJECT_ROOT / 'cross_validation_results'

transform = UV_Patch_Transform(GRID_SIZE, PATCH_SIZE, MAX_UV, AMPLITUDE_SCALE, HANDLE_COLLISION)


class Model_Type(Enum):
    AUTOENCODER_BASE = 'AUTOENCODER_BASE'


class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, patch_transform):
        self.dataset = dataset
        self.transform = patch_transform

    def __getitem__(self, index):
        return self.transform(self.dataset[index])

    def __len__(self):
        return len(self.dataset)


def makeDataset(paths, use_cuda=False):
    dataset = TimeSlicedVisDataset(
        paths=paths,
        window_size=WINDOW_SIZE,
        target_size=TARGET_SIZE,
        stride=STRIDE,
        pin_memory=False,
        read_uvw=True,
    )
    return TransformDataset(dataset, transform)


def collate_windows(batch):
    return batch


def makeModel(device, model_type=Model_Type.AUTOENCODER_BASE):
    if model_type != Model_Type.AUTOENCODER_BASE:
        raise ValueError(model_type)
    return Autoencoder.build(
        dimensions=2,
        in_channels=3,
        out_channels=2,
        latent_dim=LATENT_DIM,
        res_act_fn='silu',
        res_norm_type='group',
        encoder_args={'n_channels': 16},
    ).to(device)


def patch_chunks(sample):
    x = sample['input_patch']
    y = sample['target_patch']
    if x.ndim != 4 or x.shape[1:] != (3, PATCH_SIZE, PATCH_SIZE):
        raise ValueError(f'Unexpected input_patch shape: {tuple(x.shape)}')
    if y.shape != (x.shape[0], 2, PATCH_SIZE, PATCH_SIZE):
        raise ValueError(f'Unexpected target_patch shape: {tuple(y.shape)}')
    for start in range(0, x.shape[0], PATCH_MICROBATCH):
        yield x[start:start + PATCH_MICROBATCH], y[start:start + PATCH_MICROBATCH]


def observed_count(sample):
    return int(sample['input_patch'][:, 2].sum().item())


def batch_totals(model, windows, device, train=False, optimizer=None):
    """Loss is cell-weighted over 16 windows, with gradients accumulated by chunk."""
    n_observed = sum(observed_count(sample) for sample in windows)
    if n_observed <= 0:
        raise RuntimeError('Batch has no occupied UV cells.')
    error_sum = 0.0
    power_sum = 0.0
    for sample in windows:
        for x_cpu, y_cpu in patch_chunks(sample):
            x = x_cpu.to(device)
            target = y_cpu.to(device)
            mask = x[:, 2:3]
            if train:
                raw = model(x)
                pred = polar_to_cartesian(raw, amplitude_scale=AMPLITUDE_SCALE)
                squared = (pred - target).square().sum(dim=1, keepdim=True)
                error = (squared * mask).sum()
                if not torch.isfinite(error):
                    raise RuntimeError('Non-finite training reconstruction error')
                # Normalizes globally over observed cells across the entire batch.
                (error / n_observed).backward()
            else:
                with torch.no_grad():
                    raw = model(x)
                    pred = polar_to_cartesian(raw, amplitude_scale=AMPLITUDE_SCALE)
                    squared = (pred - target).square().sum(dim=1, keepdim=True)
                    error = (squared * mask).sum()
            error_sum += error.detach().item()
            if not train:
                power_sum += (target.square().sum(dim=1, keepdim=True) * mask).sum().item()
    if train:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        return error_sum, float(n_observed), float(grad_norm.item())
    return error_sum, power_sum, float(n_observed)


def validate_model(model, val_loader, device):
    model.eval()
    total_error = total_power = total_observed = 0.0
    for windows in val_loader:
        error, power, observed = batch_totals(model, windows, device, train=False)
        total_error += error
        total_power += power
        total_observed += observed
    if total_observed == 0 or total_power <= 0:
        raise RuntimeError('Invalid validation data: zero observed cells or signal power')
    nmse = total_error / total_power
    return total_error / total_observed, nmse, -10.0 * math.log10(nmse) if nmse > 0 else math.inf


def make_loader(dataset, shuffle):
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
        num_workers=0, pin_memory=False, collate_fn=collate_windows,
    )


def write_history(path, history):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss', 'val_nmse', 'val_snr_db'])
        writer.writeheader()
        writer.writerows(history)


def save_fold_graphs(path, history):
    epochs = [r['epoch'] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [r['train_loss'] for r in history], label='Train')
    axes[0].plot(epochs, [r['val_loss'] for r in history], label='Validation')
    axes[0].set_title('Masked complex MSE')
    axes[0].legend()
    axes[1].plot(epochs, [r['val_nmse'] for r in history])
    axes[1].axhline(1.0, color='gray', linestyle='--', linewidth=1)
    axes[1].set_title('Validation NMSE')
    axes[2].plot(epochs, [r['val_snr_db'] for r in history])
    axes[2].set_title('Validation SNR (dB)')
    for ax in axes:
        ax.set_xlabel('Epoch')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def train_one_fold(fold_number, train_paths, val_paths, output_dir, device):
    print('\n' + '=' * 68, flush=True)
    print(f'Fold {fold_number}/{K_FOLDS} | training={train_paths} | validation={val_paths}', flush=True)
    train_dataset = makeDataset(train_paths, device.type == 'cuda')
    val_dataset = makeDataset(val_paths, device.type == 'cuda')
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError('Empty dataset')
    print('Time windows:', len(train_dataset), 'train,', len(val_dataset), 'validation', flush=True)
    print('First window patches:', train_dataset[0]['input_patch'].shape,
          'occupied:', observed_count(train_dataset[0]), flush=True)

    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader = make_loader(val_dataset, shuffle=False)
    model = makeModel(device)
    criterion = MSELoss_Amplitude_Phase(amplitude_scale=AMPLITUDE_SCALE)
    assert criterion.amplitude_scale == AMPLITUDE_SCALE
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    fold_dir = output_dir / f'fold_{fold_number}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_nmse = math.inf
    best_epoch = None
    checkpoint = fold_dir / 'best_model.pt'
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_error = epoch_observed = 0.0
        for windows in train_loader:
            optimizer.zero_grad(set_to_none=True)
            error, observed, grad_norm = batch_totals(model, windows, device, train=True, optimizer=optimizer)
            epoch_error += error
            epoch_observed += observed
        train_loss = epoch_error / epoch_observed
        val_loss, val_nmse, val_snr = validate_model(model, val_loader, device)
        record = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                      val_nmse=val_nmse, val_snr_db=val_snr)
        history.append(record)
        print(f'Fold {fold_number}/{K_FOLDS} | Epoch {epoch}/{EPOCHS} | '
              f'Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f} | '
              f'Val NMSE: {val_nmse:.6f} | Val SNR: {val_snr:.2f} dB', flush=True)
        write_history(fold_dir / 'history.csv', history)
        if val_nmse < best_nmse:
            best_nmse, best_epoch = val_nmse, epoch
            torch.save({
                'fold': fold_number, 'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_nmse': best_nmse, 'best_val_snr': val_snr,
                'train_paths': train_paths, 'val_paths': val_paths,
                'config': run_config(),
            }, checkpoint)

    save_fold_graphs(fold_dir / 'training_metrics.png', history)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved['model_state_dict'])
    visualize_reconstruction(
        model=model, dataset=val_dataset, device=device,
        amplitude_scale=AMPLITUDE_SCALE, output_dir=fold_dir,
        fold_number=fold_number, sample_index=0,
        patch_microbatch=PATCH_MICROBATCH, grid_size=GRID_SIZE,
        patch_size=PATCH_SIZE,
    )
    print(f'Fold {fold_number} best NMSE: {best_nmse:.6f} at epoch {best_epoch}', flush=True)
    return history, best_nmse, best_epoch


def run_config():
    return dict(window_size=WINDOW_SIZE, target_size=TARGET_SIZE,
                stride=STRIDE, batch_size_windows=BATCH_SIZE,
                patch_microbatch=PATCH_MICROBATCH, epochs=EPOCHS,
                k_folds=K_FOLDS, latent_dim=LATENT_DIM, grid_size=GRID_SIZE,
                patch_size=PATCH_SIZE, max_uv=MAX_UV,
                amplitude_scale=AMPLITUDE_SCALE, collision=HANDLE_COLLISION,
                learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                seed=SEED, input_channels='log1p_amplitude,phase,mask',
                decoder_channels='softplus_amplitude,phase')


def train_model(paths):
    if len(paths) != 5:
        raise ValueError('Expected exactly five independent .vis observations')
    paths = sorted(paths)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'config.json').write_text(json.dumps(run_config(), indent=2))
    print('Device:', device, '| Output:', OUTPUT_DIR, flush=True)
    results = []
    for val_idx in range(K_FOLDS):
        val_paths = [paths[val_idx]]
        train_paths = [p for i, p in enumerate(paths) if i != val_idx]
        history, nmse, epoch = train_one_fold(val_idx + 1, train_paths, val_paths, OUTPUT_DIR, device)
        results.append(dict(fold=val_idx + 1, validation_path=val_paths[0],
                            best_nmse=nmse, best_epoch=epoch,
                            best_snr_db=-10 * math.log10(nmse)))
        with open(OUTPUT_DIR / 'fold_summary.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for i in range(1, K_FOLDS + 1):
        with open(OUTPUT_DIR / f'fold_{i}' / 'history.csv', newline='') as f:
            hist = list(csv.DictReader(f))
        ep = [int(r['epoch']) for r in hist]
        axes[0].plot(ep, [float(r['val_nmse']) for r in hist], label=f'Fold {i}')
        axes[1].plot(ep, [float(r['val_snr_db']) for r in hist], label=f'Fold {i}')

    axes[0].axhline(1.0, color='gray', linestyle='--', linewidth=1)
    axes[0].set_title('Validation NMSE')
    axes[1].set_title('Validation SNR (dB)')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'cross_validation_metrics.png', dpi=300)
    plt.close(fig)
    nmse_scores = np.array([r['best_nmse'] for r in results])
    print('\nFIVE-FOLD SUMMARY:', results, flush=True)
    print('Mean NMSE:', nmse_scores.mean(), 'Std NMSE:', nmse_scores.std(), flush=True)
    return results


if __name__ == '__main__':
    paths = sorted({str(Path(p).resolve())
                    for arg in sys.argv[1:]
                    for p in (glob.glob(arg) or [arg])
                    if Path(p).is_file()})
    train_model(paths)