import torch
import torch.nn as nn

class FallGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, fall_classes=2, part_classes=4):
        super(FallGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.3)

        self.fall_fc = nn.Linear(hidden_dim, fall_classes)
        self.part_fc = nn.Linear(hidden_dim, part_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.bn(out)
        out = self.dropout(out)

        fall_out = self.fall_fc(out)
        part_out = self.part_fc(out)

        return fall_out, part_out
