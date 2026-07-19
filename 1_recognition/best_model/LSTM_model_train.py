import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ============================================================
# Model
# ============================================================
class AssistLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_steps):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.step_head = nn.Linear(hidden_dim, num_steps)
        self.progress_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)

        h = h_n[-1]              # [B, hidden_dim]
        feat = self.shared(h)

        step_logits = self.step_head(feat)          # [B, num_steps]
        progress_pred = self.progress_head(feat)    # [B, 1]
        progress_pred = progress_pred.squeeze(-1)   # [B]

        return step_logits, progress_pred