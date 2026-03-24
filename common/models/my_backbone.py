from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mtcn_backbone import GroupAdapter, ModalityFusion, DilatedResidualBlock, ASP, TCN

@dataclass
class DualTCNBackboneConfig:
    audio_group_dims: dict[str, int] = field(default_factory=dict)
    audio_pooled_group_dims: dict[str, int] = field(default_factory=dict)
    video_group_dims: dict[str, int] = field(default_factory=dict)

    d_adapter: int = 64
    d_model: int = 256

    tcn_layers: int = 4
    tcn_kernel_size: int = 3
    n_heads: int = 4

    asp_alpha: float = 0.5
    asp_beta: float = 0.5

    dropout: float = 0.1

    n_sessions: int = 4
    d_session: int = 16
    d_shared: int = 256


class DualTCN(nn.Module):
    def __init__(
        self, d_model: int, tcn_layers: int, tcn_kernel_size: int, num_heads: int, dropout: float
    ) -> None:
        super().__init__()
        self.a2v_attn = nn.MultiheadAttention(d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.v2a_attn = nn.MultiheadAttention(d_model, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.audio_norm = nn.LayerNorm(d_model)
        self.video_norm = nn.LayerNorm(d_model)

        self.audio_tcn = TCN(d_model, tcn_layers, tcn_kernel_size, dropout)
        self.video_tcn = TCN(d_model, tcn_layers, tcn_kernel_size, dropout)

    def forward(self, audio: torch.Tensor, video: torch.Tensor,
                audio_mask: torch.Tensor, video_mask: torch.Tensor):
        enhanced_audio, _ = self.a2v_attn(audio, video, video, key_padding_mask=video_mask)
        enhanced_video, _ = self.v2a_attn(video, audio, audio, key_padding_mask=audio_mask)
        enhanced_audio = self.audio_norm(enhanced_audio + audio)
        enhanced_video = self.video_norm(enhanced_video + video)
        out_audio = self.audio_tcn(enhanced_audio, audio_mask)
        out_video = self.video_tcn(enhanced_video, video_mask)
        return out_audio, out_video


class DualTCNBackbone(nn.Module):
    """
    使用Transformer替换原有的TCN结果，实现早期融合策略
    """
    def __init__(self, cfg: DualTCNBackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.audio_adapters = nn.ModuleDict({
            name: GroupAdapter(d_in, cfg.d_adapter, cfg.dropout)
            for name, d_in in cfg.audio_group_dims.items()
        })
        self.audio_pooled_adapters = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(d_in),
                nn.Linear(d_in, cfg.d_adapter),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
            for name, d_in in cfg.audio_pooled_group_dims.items()
        })
        self.video_adapters = nn.ModuleDict({
            name: GroupAdapter(d_in, cfg.d_adapter, cfg.dropout)
            for name, d_in in cfg.video_group_dims.items()
        })
        self.audio_group_names = sorted(cfg.audio_group_dims.keys())
        self.audio_pooled_group_names = sorted(cfg.audio_pooled_group_dims.keys())
        self.video_group_names = sorted(cfg.video_group_dims.keys())

        self.audio_fusion = ModalityFusion(
            len(self.audio_group_names), cfg.d_adapter, cfg.d_model
        )
        self.video_fusion = ModalityFusion(
            len(self.video_group_names), cfg.d_adapter, cfg.d_model
        )

        # self.audio_tcn = TCN(cfg.d_model, cfg.tcn_layers, cfg.tcn_kernel_size, cfg.dropout)
        # self.video_tcn = TCN(cfg.d_model, cfg.tcn_layers, cfg.tcn_kernel_size, cfg.dropout)
        self.dual_tcn = DualTCN(cfg.d_model, cfg.tcn_layers, cfg.tcn_kernel_size, cfg.n_heads, dropout=cfg.dropout)

        self.audio_asp = ASP(cfg.d_model, cfg.asp_alpha, cfg.asp_beta)
        self.video_asp = ASP(cfg.d_model, cfg.asp_alpha, cfg.asp_beta)

        fusion_in = 2 * cfg.d_model * 2
        fusion_in += len(self.audio_pooled_group_names) * cfg.d_adapter
        fusion_in += cfg.d_session

        self.session_embed = nn.Embedding(cfg.n_sessions, cfg.d_session)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, cfg.d_shared),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_shared, cfg.d_shared),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        audio_adapted = [
            self.audio_adapters[n](batch["audio_groups"][n])
            for n in self.audio_group_names
        ]
        video_adapted = [
            self.video_adapters[n](batch["video_groups"][n])
            for n in self.video_group_names
        ]

        a = self.audio_fusion(audio_adapted)
        v = self.video_fusion(video_adapted)

        mask_a = batch["mask_audio"]
        mask_v = batch["mask_video"]
        a = a * mask_a.unsqueeze(-1).float()
        v = v * mask_v.unsqueeze(-1).float()

        # a = self.audio_tcn(a, mask_a)
        # v = self.video_tcn(v, mask_v)
        a, v = self.dual_tcn(a, v, mask_a, mask_v)

        vad = batch["vad_signal"]
        qc = batch["qc_quality"]
        z_a = self.audio_asp(a, mask_a, vad, qc)
        z_v = self.video_asp(v, mask_v, vad, qc)

        parts = [z_a, z_v]
        parts.extend(
            self.audio_pooled_adapters[name](batch["audio_pooled_groups"][name])
            for name in self.audio_pooled_group_names
        )
        parts.append(self.session_embed(batch["session_idx"]))

        z = torch.cat(parts, dim=-1)
        return self.fusion_mlp(z)