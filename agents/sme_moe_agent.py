"""
SMETimes Mixture of Experts (MoE) agent implementation for the multi-agent dropout prediction system
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from tqdm.auto import tqdm


class SMEMoETS(nn.Module):
    """
    SMETimes Mixture of Experts model for time series prediction
    """
    def __init__(self, input_dim):
        super().__init__()
        self.trend = nn.Linear(input_dim, 32)
        self.vol = nn.Linear(input_dim, 32)
        self.level = nn.Linear(input_dim, 32)
        self.temporal = nn.Linear(input_dim, 32)
        self.expert = nn.Linear(32, 1)
        self.gate = nn.Linear(32 * 4, 4)

    def forward(self, x):
        v0 = torch.relu(self.trend(x))
        v1 = torch.relu(self.vol(x))
        v2 = torch.relu(self.level(x))
        v3 = torch.relu(self.temporal(x))
        vs = [v0, v1, v2, v3]
        logits = torch.cat([self.expert(v) for v in vs], 1)
        w = torch.nn.functional.softmax(self.gate(torch.cat(vs, 1)), 1)
        return torch.sigmoid((w * logits).sum(1)), w


class SMEMoEAgent:
    """
    SMEMoETS agent with training and prediction capabilities
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        self.trained = False

    def train(self, X_train, y_train, epochs=None):
        """
        Train the SMEMoETS model
        """
        from utils.logger import log
        
        input_dim = X_train.shape[1]
        self.model = SMEMoETS(input_dim).to(self.device)
        
        # Calculate epochs based on dataset size
        if epochs is None:
            epochs = int(max(500, len(X_train) // 2000))  # ≈1 epoch / 4k rows, ≥400
        
        # Create dataset and dataloader
        ds = TensorDataset(
            torch.tensor(X_train.values, dtype=torch.float32, device=self.device),
            torch.tensor(y_train.values, dtype=torch.float32, device=self.device)
        )
        dl = DataLoader(ds, batch_size=4096, shuffle=True, drop_last=True)
        
        # Training setup
        optim = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        loss_fn = nn.BCELoss()

        if len(dl) > 0:
            for epoch in tqdm(range(epochs), desc="SME-MoE train", unit="epoch"):
                for xb, yb in dl:
                    optim.zero_grad()
                    p, _ = self.model(xb)
                    loss = loss_fn(p.squeeze(), yb)
                    loss.backward()
                    optim.step()

        self.model.eval()
        self.trained = True
        log("SMEMoETS agent training completed")

    def risk(self, row):
        if not self.trained or self.model is None:
            return 0.5
        
        x = torch.tensor(row.values, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            p, _ = self.model(x)
        return float(p.item())

    def rat(self, row):
        if not self.trained or self.model is None:
            return "SMEMoETS analysis unavailable."

        x = torch.tensor(row.values, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, w = self.model(x)
        names = ["trend", "volatility", "level", "temporal"]
        top = names[int(w.argmax())]
        return f"Gate {np.round(w.cpu().numpy(), 2).tolist()} → {top} expert dominates."