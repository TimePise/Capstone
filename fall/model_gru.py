import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(weights * x, dim=1)

class FallBiGRUAttentionNet(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=128, num_layers=2, fall_classes=2, part_classes=4):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.attn = Attention(hidden_dim)
        self.dropout = nn.Dropout(0.3)

        self.fall_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, fall_classes)
        )

        self.part_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, part_classes)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.attn(out)
        out = self.dropout(out)
        return self.fall_head(out), self.part_head(out)