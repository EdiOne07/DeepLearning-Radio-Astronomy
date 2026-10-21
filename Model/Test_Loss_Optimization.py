
import sys
from pathlib import Path

import torch

# Allow imports from the Model directory.
MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent

sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from Training_Amplitude_Phase import (
    makeDataset,
    transform,
    MSELoss_Amplitude_Phase,
    polar_to_cartesian,
)


def main():
    if len(sys.argv) != 2:
        raise ValueError(
            "Usage: python Model/Test_Loss_Optimization.py "
            "/path/to/observation.vis"
        )

    observation_path = sys.argv[1]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Observation:", observation_path)

    dataset = makeDataset(
        [observation_path],
        use_cuda=False,
    )

    sample = dataset[0]

    x = sample["input_grid"].unsqueeze(0).to(device)
    target = sample["target_grid"].unsqueeze(0).to(device)
    mask = x[:, 2:3]

    print("Input shape:", tuple(x.shape))
    print("Target shape:", tuple(target.shape))
    print("Occupied cells:", int(mask.sum().item()))

    criterion = MSELoss_Amplitude_Phase(
        amplitude_scale=transform.amplitude_scale
    )

    # Directly train the predicted polar parameters.
    # No autoencoder is involved.
    prediction_parameters = torch.nn.Parameter(
        torch.zeros_like(x[:, :2])
    )

    optimizer = torch.optim.Adam(
        [prediction_parameters],
        lr=0.1,
    )

    # Zero-visibility baseline.
    zero_error = (
        target.square().sum(dim=1, keepdim=True) * mask
    ).sum() / mask.sum().clamp_min(1)

    print("Zero-prediction loss:", zero_error.item())

    for step in range(301):
        optimizer.zero_grad()

        loss = criterion(
            prediction_parameters,
            target,
            mask,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step}"
            )

        loss.backward()

        if step == 0:
            grad = prediction_parameters.grad

            print(
                "Initial amplitude gradient norm:",
                grad[:, 0].norm().item(),
            )

            print(
                "Initial phase gradient norm:",
                grad[:, 1].norm().item(),
            )

        optimizer.step()

        if step % 25 == 0:
            with torch.no_grad():
                reconstructed = polar_to_cartesian(
                    prediction_parameters,
                    amplitude_scale=transform.amplitude_scale,
                )

                error = (
                    (reconstructed - target)
                    .square()
                    .sum(dim=1, keepdim=True)
                    * mask
                ).sum()

                power = (
                    target.square()
                    .sum(dim=1, keepdim=True)
                    * mask
                ).sum()

                nmse = (error / power).item()

            print(
                f"Step {step:03d} | "
                f"Loss: {loss.item():.9f} | "
                f"NMSE: {nmse:.9f}"
            )


if __name__ == "__main__":
    main()
