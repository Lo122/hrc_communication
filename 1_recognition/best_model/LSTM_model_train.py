import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader



# ============================================================
# Model
# ============================================================
class AssistLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_steps, dropout=0.5, num_layers=1, num_mistakes=None):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.step_head = nn.Linear(hidden_dim, num_steps)
        self.progress_head = nn.Linear(hidden_dim, 1)
        self.mistake_head = (
            nn.Linear(hidden_dim, num_mistakes) if num_mistakes is not None else None
        )

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)

        h = h_n[-1]              # [B, hidden_dim]
        feat = self.shared(h)

        step_logits = self.step_head(feat)          # [B, num_steps]
        progress_pred = self.progress_head(feat)    # [B, 1]
        progress_pred = progress_pred.squeeze(-1)   # [B]

        if self.mistake_head is not None:
            mistake_logits = self.mistake_head(feat)  # [B, num_mistakes]
            return step_logits, progress_pred, mistake_logits
        return step_logits, progress_pred
