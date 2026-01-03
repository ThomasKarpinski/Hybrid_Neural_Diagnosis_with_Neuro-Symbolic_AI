import torch
import torch.nn as nn

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=21, hidden_dims=(32,16), dropout=0.0):
        super().__init__()
        h1, h2 = hidden_dims
        layers = [
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(h2, 1),
            # nn.Sigmoid(),  Removed for BCEWithLogitsLoss
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

