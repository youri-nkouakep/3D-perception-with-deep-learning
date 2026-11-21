"""PointNet (Qi et al., CVPR 2017) pour la classification de nuages de points.

Idée clé : un nuage de points est un *ensemble* non ordonné. Le réseau applique le même
petit MLP à chaque point indépendamment (convolutions 1x1), puis agrège tout avec un
max-pooling, une opération symétrique : l'ordre des points n'a donc aucune influence.
Les T-Nets apprennent une transformation qui aligne le nuage (et les caractéristiques)
dans une pose canonique avant l'extraction.

Entrée : (B, N, 3)  ->  sortie : logits (B, n_classes)
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _mlp1d(channels: list[int]) -> nn.Sequential:
    """MLP partagé entre tous les points (Conv1d noyau 1) + BatchNorm + ReLU."""
    layers: list[nn.Module] = []
    for c_in, c_out in zip(channels[:-1], channels[1:]):
        layers += [nn.Conv1d(c_in, c_out, 1), nn.BatchNorm1d(c_out), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class TNet(nn.Module):
    """Mini-réseau qui prédit une matrice k x k appliquée au nuage."""

    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.features = _mlp1d([k, 64, 128, 1024])
        self.head = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, k * k),
        )
        # initialisé à l'identité : au départ, la transformation ne fait rien
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x : (B, k, N)
        feat = self.features(x).max(dim=2).values
        mat = self.head(feat).view(-1, self.k, self.k)
        return mat + torch.eye(self.k, device=x.device)


class PointNet(nn.Module):
    def __init__(self, n_classes: int = 40, input_transform: bool = True,
                 feature_transform: bool = True, dropout: float = 0.3):
        super().__init__()
        self.input_tnet = TNet(3) if input_transform else None
        self.mlp1 = _mlp1d([3, 64, 64])
        self.feature_tnet = TNet(64) if feature_transform else None
        self.mlp2 = _mlp1d([64, 64, 128, 1024])
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, points: torch.Tensor):
        """Renvoie (logits, matrice de transformation des caractéristiques ou None)."""
        x = points.transpose(1, 2)  # (B, 3, N)
        if self.input_tnet is not None:
            x = torch.bmm(self.input_tnet(x), x)
        x = self.mlp1(x)
        feat_mat = None
        if self.feature_tnet is not None:
            feat_mat = self.feature_tnet(x)
            x = torch.bmm(feat_mat, x)
        x = self.mlp2(x)
        global_feat = x.max(dim=2).values  # max-pooling : invariant à l'ordre des points
        return self.classifier(global_feat), feat_mat


def orthogonality_loss(mat: torch.Tensor | None) -> torch.Tensor:
    """Pénalité ||I - A Aᵀ||² qui pousse la transformation 64x64 à rester orthogonale."""
    if mat is None:
        return torch.tensor(0.0)
    eye = torch.eye(mat.shape[1], device=mat.device)
    return ((eye - torch.bmm(mat, mat.transpose(1, 2))) ** 2).sum(dim=(1, 2)).mean()


def pointnet_loss(logits, labels, feat_mat, reg_weight: float = 0.001, label_smoothing: float = 0.0):
    loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    return loss + reg_weight * orthogonality_loss(feat_mat).to(loss.device)
