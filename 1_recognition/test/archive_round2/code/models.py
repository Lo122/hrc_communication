"""The models being compared. All share one interface: (B, T, D) -> task logits,
mistake logit, progress. Multi-task by design -- the thesis needs all three."""
import torch, torch.nn as nn


class Heads(nn.Module):
    """Four heads. `task` is 7 INDEPENDENT logits, not a softmax: 19.4% of frames
    have two tasks annotated at once, so the target is multi-label. `prog` is
    per-lane, which is what TRIGGER_RULES needs -- progress of a specific task,
    available even when that task is not the argmax winner. `bg` lets the model
    say "nobody is working" instead of defaulting to class 0."""

    def __init__(self, h, n_tasks=7):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.3))
        self.task = nn.Linear(h, n_tasks)
        self.mist = nn.Linear(h, 1)
        self.prog = nn.Linear(h, n_tasks)
        self.bg = nn.Linear(h, 1)

    def forward(self, f):
        f = self.shared(f)
        return {
            "task": self.task(f),
            "mistake": self.mist(f).squeeze(-1),
            "prog": self.prog(f),
            "bg": self.bg(f).squeeze(-1),
        }


class LSTMNet(nn.Module):
    """The existing AssistLSTM shape, plus dropout and optional bidirectionality
    off (causal -- must stay usable online)."""
    def __init__(self, d, h=128, layers=1, n_tasks=7):
        super().__init__()
        self.rnn = nn.LSTM(d, h, num_layers=layers, batch_first=True,
                           dropout=0.2 if layers > 1 else 0.0)
        self.heads = Heads(h, n_tasks)

    def forward(self, x):
        o, (hn, _) = self.rnn(x)
        return self.heads(hn[-1])


class GRUNet(nn.Module):
    def __init__(self, d, h=128, layers=1, n_tasks=7):
        super().__init__()
        self.rnn = nn.GRU(d, h, num_layers=layers, batch_first=True,
                          dropout=0.2 if layers > 1 else 0.0)
        self.heads = Heads(h, n_tasks)

    def forward(self, x):
        o, hn = self.rnn(x)
        return self.heads(hn[-1])


class TCNNet(nn.Module):
    """Dilated causal conv stack -- no recurrence, much faster, often stronger on
    fixed windows. Causal padding so it stays deployable."""
    def __init__(self, d, h=128, levels=4, k=5, n_tasks=7):
        super().__init__()
        layers, cin = [], d
        for i in range(levels):
            dil = 2 ** i
            layers += [nn.Conv1d(cin, h, k, padding=(k - 1) * dil, dilation=dil),
                       nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            cin = h
        self.net = nn.Sequential(*layers)
        self.k, self.levels = k, levels
        self.heads = Heads(h, n_tasks)

    def forward(self, x):
        z = self.net(x.transpose(1, 2))
        z = z[:, :, :x.shape[1]]          # strip the causal right-padding
        return self.heads(z[:, :, -1])    # feature at the LAST frame


def build(name, d, n_tasks=7):
    return {
        "lstm":   lambda: LSTMNet(d, 128, 1, n_tasks),
        "lstm2":  lambda: LSTMNet(d, 128, 2, n_tasks),
        "gru":    lambda: GRUNet(d, 128, 1, n_tasks),
        "tcn":    lambda: TCNNet(d, 128, 4, 5, n_tasks),
    }[name]()
