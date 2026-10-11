"""Inference-safe definition of the trained feature-fusion network.

Training orchestration remains in feature_fusion.py; this module deliberately has
no dataset, dataloader, or checkpoint side effects so Flask can load inference.
"""
import torch.nn as nn

from config import NUM_CLASSES


class FeatureAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 2048), nn.ReLU(),
            nn.Linear(2048, feature_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention(x)


class FusionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = FeatureAttention(17408)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(17408), nn.Linear(17408, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.attention(x))
