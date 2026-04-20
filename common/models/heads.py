from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class A1Head(nn.Module):

    def __init__(self, d_in: int, bias_init: list[float] | None = None) -> None:
        super().__init__()
        self.fc = nn.Linear(d_in, 3)
        # self.fc = nn.Sequential(
        #     nn.Linear(d_in, d_in // 2),
        #     nn.GELU(),
        #     nn.Linear(d_in // 2, 3),
        # )
        if bias_init is not None:
            with torch.no_grad():
                self.fc.bias.copy_(torch.tensor(bias_init, dtype=torch.float32))
                # self.fc[-1].bias.copy_(torch.tensor(bias_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    @staticmethod
    def predict_probs(logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)


class A2OrdinalHead(nn.Module):
    def __init__(self, d_in: int, n_items: int = 21, n_thresholds: int = 3) -> None:
        super().__init__()
        self.n_items = n_items
        self.n_thresholds = n_thresholds
        self.fc = nn.Linear(d_in, n_items * n_thresholds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        return self.fc(x).view(B, self.n_items, self.n_thresholds)

    @staticmethod
    def predict_int(logits: torch.Tensor) -> torch.Tensor:
        return (torch.sigmoid(logits) > 0.5).long().sum(dim=-1)

    @staticmethod
    def predict_int_monotonic(logits: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(logits)  

        p1 = s[..., 0]
        p2 = torch.min(s[..., 1], p1)
        p3 = torch.min(s[..., 2], p2)

        P0 = 1.0 - p1
        P1 = p1 - p2
        P2 = p2 - p3
        P3 = p3
        class_probs = torch.stack([P0, P1, P2, P3], dim=-1)  
        return class_probs.argmax(dim=-1) 

    @staticmethod
    def predict_expectation(logits: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(logits)
        p1 = s[..., 0]
        p2 = torch.min(s[..., 1], p1)
        p3 = torch.min(s[..., 2], p2)
        E = p1 + p2 + p3  
        return E.round().long().clamp(0, 3)

    @staticmethod
    def build_ordinal_targets(labels: torch.Tensor, n_thresholds: int = 3) -> torch.Tensor:
        B, I = labels.shape
        thresholds = torch.arange(1, n_thresholds + 1, device=labels.device).float()
        targets = (labels.unsqueeze(-1).float() >= thresholds.view(1, 1, -1)).float()
        return targets



def a1_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if label_smoothing > 0.0:
        targets = targets.float() * (1.0 - label_smoothing) + 0.5 * label_smoothing
    # logits = torch.clamp(logits, -1e2, 1e2)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def a2_ordinal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    targets = A2OrdinalHead.build_ordinal_targets(labels, n_thresholds=logits.size(-1))
    if label_smoothing > 0.0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
    # logits = torch.clamp(logits, -1e2, 1e2)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


class contrastive_loss():
    def __init__(self, N, device, temperature: float = 0.5):
        self.N = N
        row_group = torch.arange(N).unsqueeze(1) // 4
        col_group = torch.arange(N).unsqueeze(0) // 4
        self.mask = torch.eq(row_group, col_group).float().to(device)
        self.temperature = temperature

    def __call__(self, a_repr, v_repr) -> torch.Tensor:
        N, _ = a_repr.shape
        a_norm = F.normalize(a_repr, dim=-1)
        v_norm = F.normalize(v_repr, dim=-1)

        similarity_matrix = torch.matmul(a_norm, v_norm.T) / self.temperature

        logits = similarity_matrix - torch.logsumexp(similarity_matrix, dim=1, keepdim=True)
        loss = -torch.sum(self.mask[:N, :N] * logits) / self.mask[:N, :N].sum()

        return loss


class supcon_loss():
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.label_weights = [1.0, 1.0, 1.0]

    def __call__(self, a_repr: torch.Tensor, v_repr: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:        
        N, _ = a_repr.shape    
        labels = labels.repeat_interleave(4, dim=0)
        
        a_norm = F.normalize(a_repr, dim=-1)
        v_norm = F.normalize(v_repr, dim=-1)
        similarity_matrix = torch.matmul(a_norm, v_norm.T) / self.temperature

        total_loss = 0.0
        for dim_idx in range(labels.shape[1]):
            label_dim = labels[:, dim_idx]
            label_sim = torch.matmul(label_dim.unsqueeze(1), label_dim.unsqueeze(0)) 
            label_sim = (label_sim + 1) / 2 
            logits = similarity_matrix - torch.logsumexp(similarity_matrix, dim=1, keepdim=True)
            weighted_logits = logits * label_sim.float()
            pos_logits = torch.diag(weighted_logits)
            loss_dim = -pos_logits.mean()            
            total_loss += loss_dim * self.label_weights[dim_idx]
        
        return total_loss