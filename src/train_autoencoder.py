import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.gatr_autoencoder import GATrAutoencoder


class HitsDataset(Dataset):
    """
    Expects a list of events.
    Each event is a dict with:
      - "xyz": (Ni, 3)
      - "depth": (Ni, 1)
      - "energy_onehot": (Ni, 3)
    """

    def __init__(self, events):
        self.events = events

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        evt = self.events[idx]
        xyz = torch.as_tensor(evt["xyz"], dtype=torch.float32)
        depth = torch.as_tensor(evt["depth"], dtype=torch.float32)
        energy_onehot = torch.as_tensor(evt["energy_onehot"], dtype=torch.float32)
        return xyz, depth, energy_onehot


def collate_hits(batch):
    """
    Flattens variable-length events into a single batch with a `batch` index.
    Returns:
      - mv_v_part: (N, 3)
      - mv_s_part: (N, 1)
      - scalars: (N, 3) one-hot energy threshold
      - batch_idx: (N,)
    """
    xyz_list, depth_list, energy_list, batch_idx_list = [], [], [], []
    for i, (xyz, depth, energy_onehot) in enumerate(batch):
        n = xyz.shape[0]
        xyz_list.append(xyz)
        depth_list.append(depth)
        energy_list.append(energy_onehot)
        batch_idx_list.append(torch.full((n,), i, dtype=torch.long))

    mv_v_part = torch.cat(xyz_list, dim=0)
    mv_s_part = torch.cat(depth_list, dim=0)
    scalars = torch.cat(energy_list, dim=0)
    batch_idx = torch.cat(batch_idx_list, dim=0)

    return mv_v_part, mv_s_part, scalars, batch_idx


def reconstruction_loss(outputs, mv_v_part, mv_s_part, scalars):
    """
    Simple MSE losses for coordinates, depth, and energy one-hot.
    """
    point_rec = outputs["point_rec"]
    scalar_rec = outputs["scalar_rec"]
    s_rec = outputs["s_rec"]

    loss_xyz = nn.functional.mse_loss(point_rec, mv_v_part)
    loss_depth = nn.functional.mse_loss(scalar_rec, mv_s_part)
    loss_energy = nn.functional.mse_loss(s_rec, scalars)

    return loss_xyz + loss_depth + loss_energy


def main():
    # TODO: Replace with real data loading
    dummy_events = [
        {
            "xyz": torch.randn(128, 3),
            "depth": torch.randn(128, 1),
            "energy_onehot": nn.functional.one_hot(
                torch.randint(0, 3, (128,)), num_classes=3
            ).float(),
        }
        for _ in range(10)
    ]

    dataset = HitsDataset(dummy_events)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_hits)

    cfg_enc = {
        "hidden_mv_channels": 32,
        "hidden_s_channels": 64,
        "num_blocks": 2,
        "in_s_channels": 3,
        "in_mv_channels": 1,
        "out_mv_channels": 1,
        "dropout": 0.1,
        "out_s_channels": 16,
    }

    cfg_dec = {
        "hidden_mv_channels": 32,
        "hidden_s_channels": 64,
        "num_blocks": 2,
        "in_s_channels": 3,
        "in_mv_channels": 1,
        "out_mv_channels": 1,
        "dropout": 0.1,
        "out_s_channels": 16,
    }

    model = GATrAutoencoder(cfg_enc=cfg_enc, cfg_dec=cfg_dec, latent_s_channels=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(5):
        total_loss = 0.0
        for mv_v_part, mv_s_part, scalars, batch_idx in dataloader:
            mv_v_part = mv_v_part.to(device)
            mv_s_part = mv_s_part.to(device)
            scalars = scalars.to(device)
            batch_idx = batch_idx.to(device)

            outputs = model(mv_v_part, mv_s_part, scalars, batch_idx)
            loss = reconstruction_loss(outputs, mv_v_part, mv_s_part, scalars)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dataloader))
        print(f"Epoch {epoch+1}: loss={avg_loss:.6f}")


if __name__ == "__main__":
    main()
