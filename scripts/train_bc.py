#!/usr/bin/env python
"""Train a simple MLP Behavioral Cloning baseline for TwoBlockPick.

This serves as a non-multimodal comparison: a deterministic MLP policy
trained with MSE regression cannot express multiple modes.

Usage:
    python scripts/train_bc.py --config configs/train.yaml
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

# ── Dataset ──────────────────────────────────────────────────────────

class BCDataset(Dataset):
    """Single-step (obs, action) pairs from demos — no action chunking."""

    def __init__(self, path: str) -> None:
        data = np.load(path, allow_pickle=True)
        obs = data["obs"]            # (N, T, obs_dim)
        actions = data["actions"]    # (N, T, act_dim)
        ep_lens = data["episode_lengths"]

        all_obs, all_act = [], []
        for i in range(obs.shape[0]):
            L = int(ep_lens[i])
            all_obs.append(obs[i, :L])
            all_act.append(actions[i, :L])

        self.obs = np.concatenate(all_obs, axis=0).astype(np.float32)
        self.act = np.concatenate(all_act, axis=0).astype(np.float32)

        # Obs normalisation (same approach as diffusion policy)
        self.obs_mean = self.obs.mean(0)
        self.obs_std = np.maximum(self.obs.std(0), 0.01)
        self.obs = (self.obs - self.obs_mean) / self.obs_std

        print(f"  BC dataset: {self.obs.shape[0]} transitions, "
              f"obs_dim={self.obs.shape[1]}, act_dim={self.act.shape[1]}")

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.from_numpy(self.obs[idx]),
                torch.from_numpy(self.act[idx]))


# ── MLP Policy ───────────────────────────────────────────────────────

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int,
                 hidden_dim: int = 256, n_layers: int = 4) -> None:
        super().__init__()
        layers = [nn.Linear(obs_dim, hidden_dim), nn.Mish()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Mish()]
        layers.append(nn.Linear(hidden_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ── Training ─────────────────────────────────────────────────────────

def train(cfg: Dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    ds = BCDataset(cfg["demo_path"])
    dl = DataLoader(ds, batch_size=cfg.get("batch_size", 256),
                    shuffle=True, drop_last=True, num_workers=0)

    model = MLPPolicy(
        obs_dim=cfg["obs_dim"],
        act_dim=cfg["act_dim"],
        hidden_dim=cfg.get("hidden_dim", 256),
        n_layers=4,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  MLP params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.get("lr", 1e-4),
                                  weight_decay=cfg.get("weight_decay", 1e-6))

    epochs = cfg.get("bc_epochs", 500)
    ckpt_dir = Path(cfg.get("ckpt_dir", "runs")) / "bc_latest"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for obs_b, act_b in dl:
            obs_b, act_b = obs_b.to(device), act_b.to(device)
            pred = model(obs_b)
            loss = nn.functional.mse_loss(pred, act_b)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        mean_loss = float(np.mean(losses))
        if epoch % 20 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{epochs}  loss={mean_loss:.6f}")
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save({
                "model": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "loss": mean_loss,
                "obs_mean": ds.obs_mean,
                "obs_std": ds.obs_std,
            }, ckpt_dir / "bc_ckpt.pt")

    print(f"\n  BC training done — best loss {best_loss:.6f}")
    print(f"  checkpoint: {ckpt_dir / 'bc_ckpt.pt'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()
