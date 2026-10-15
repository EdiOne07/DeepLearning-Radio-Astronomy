
import torch


class SpatialKfoldMask:
    def __init__(
        self,
        grid_size: int = 256,
        block_size: int = 32,
        split_ratio: float = 0.2,
    ):
        self.grid_size = grid_size
        self.block_size = block_size
        self.split_ratio = split_ratio

    def train_validation_split(
        self,
        mask: torch.Tensor,
        generator=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split observed UV cells into visible context and
        held-out observed blocks.

        Args:
            mask: [B, 1, H, W] binary occupancy mask.
            generator: Optional CPU torch.Generator for
                       reproducible block selection.

        Returns:
            train_mask: cells visible to the model.
            val_mask: observed cells hidden from the model.
        """
        observed = (
            torch.isfinite(mask) & (mask > 0)
        )

        train_mask = observed.clone()
        val_mask = torch.zeros_like(mask)

        _, _, height, width = mask.shape

        for i in range(mask.shape[0]):
            active_blocks = []

            for y in range(0, height, self.block_size):
                for x in range(0, width, self.block_size):
                    block = observed[
                        i, :, y:y + self.block_size,
                        x:x + self.block_size
                    ]

                    if block.any().item():
                        active_blocks.append((y, x))

            if not active_blocks:
                continue

            n_active = len(active_blocks)

            if n_active < 2:
                continue

            n_val = max(
                1,
                round(n_active * self.split_ratio),
            )
            n_val = min(n_val, n_active - 1)

            permutation = torch.randperm(
                n_active,
                generator=generator,
            )

            for block_idx in permutation[:n_val].tolist():
                y, x = active_blocks[block_idx]

                block = observed[
                    i, :, y:y + self.block_size,
                    x:x + self.block_size
                ]

                val_mask[
                    i, :, y:y + self.block_size,
                    x:x + self.block_size
                ] = block.to(mask.dtype)

                train_mask[
                    i, :, y:y + self.block_size,
                    x:x + self.block_size
                ] = 0

        return train_mask.to(mask.dtype), val_mask
