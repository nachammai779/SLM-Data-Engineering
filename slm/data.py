from pathlib import Path
import numpy as np
import torch

from slm.config import TrainConfig


def get_batch(split: str, cfg: TrainConfig, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    fname = "train.bin" if split == "train" else "val.bin"
    path = Path(cfg.data_dir) / fname
    data = np.memmap(path, dtype=np.uint16, mode="r")
    if len(data) <= cfg.block_size + 1:
        raise RuntimeError(
            f"{path} too small ({len(data)} tokens) for block_size {cfg.block_size}; "
            "tokenize a larger corpus."
        )
    ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + cfg.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + cfg.block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)