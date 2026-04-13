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
        self.a2v_lstm = nn.LSTM(d_model, d_model, num_layers=2, batch_first=True, dropout=dropout)
        self.v2a_lstm = nn.LSTM(d_model, d_model, num_layers=2, batch_first=True, dropout=dropout)

        # self.audio_norm = nn.LayerNorm(d_model)
        # self.video_norm = nn.LayerNorm(d_model)

        # self.audio_tcn = TCN(d_model, tcn_layers, tcn_kernel_size, dropout)
        # self.video_tcn = TCN(d_model, tcn_layers, tcn_kernel_size, dropout)

    def forward(self, audio: torch.Tensor, video: torch.Tensor,
                audio_mask: torch.Tensor, video_mask: torch.Tensor):
        # print(f"DEBUG: audio size : {audio.shape}, video size : {video.shape}")
        # enhanced_audio, _ = self.a2v_attn(audio, video, video, key_padding_mask=video_mask)
        # enhanced_video, _ = self.v2a_attn(video, audio, audio, key_padding_mask=audio_mask)
        # print(f"DEBUG: attn enhanced_audio size : {enhanced_audio.shape}, attn enhanced_video size : {enhanced_video.shape}")
        out_audio, _ = self.a2v_lstm(audio)
        out_video, _ = self.v2a_lstm(video)
        # print(f"DEBUG: lstm enhanced_audio size : {enhanced_audio.shape}, lstm enhanced_video size : {enhanced_video.shape}")

        # enhanced_audio = self.audio_norm(enhanced_audio)
        # enhanced_video = self.video_norm(enhanced_video)
        # out_audio = self.audio_tcn(enhanced_audio, audio_mask)
        # out_video = self.video_tcn(enhanced_video, video_mask)

        return out_audio, out_video


class InterModalityAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_in, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(d_in, d_out)

    def forward(self, feature_list) -> torch.Tensor:
        x = torch.stack(feature_list, dim=2)
        bt, t, n, d = x.shape
        x = x.view(bt * t, n, d) 
        attn_out, _ = self.attn(x, x, x)
        fused = attn_out.mean(dim=1)
        fused = fused.view(bt, t, -1)
        out = self.proj(fused)
        return out


class newASP(nn.Module):
    """Attentive Statistics Pooling with VAD and quality control signals."""
    def __init__(self, d_model: int, alpha: float = 0.5, beta: float = 0.5) -> None:
        super().__init__()
        self.attn = nn.Linear(d_model, 1)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=2, batch_first=True, dropout=0.2)  # , bidirectional=True)
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        vad: torch.Tensor,
        qc: torch.Tensor,
    ) -> torch.Tensor:
        """
        x    : (B, T, D)
        mask : (B, T) bool
        vad  : (B, T) float
        qc   : (B, T) float
        Returns: (B, 2*D)
        """
        e = self.attn(x).squeeze(-1) 
        e = e + self.alpha * vad + self.beta * qc

        # mask invalid positions
        e = e.masked_fill(~mask, float("-inf"))
        w = F.softmax(e, dim=-1)
        w = w.masked_fill(~mask, 0.0)   # to avoid NaN in mean/std when all masked
        w_unsq = w.unsqueeze(-1)
        x *= w_unsq
        # lstm_out, _ = self.lstm(x)
        return torch.mean(x, dim=1)


class LSTM(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(d_model, d_hidden, num_layers=n_layers, batch_first=True, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return lstm_out

class LateFusionTransformer(nn.Module):
    def __init__(self, d_in=256, d_out=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_in,
            nhead=nhead, 
            dim_feedforward=d_in*4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_in, d_out)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        output = self.transformer(x)
        output = self.proj(output)
        return output
    

class DualTCNBackbone(nn.Module):
    """
    使用Transformer替换原有的独立TCN，实现早期融合策略
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
                # nn.Linear(cfg.d_adapter, cfg.d_shared)  # align with the ASP representation
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

        self.inter_audio_attn = InterModalityAttention(cfg.d_adapter, cfg.d_model, num_heads=cfg.n_heads, dropout=cfg.dropout)
        self.inter_video_attn = InterModalityAttention(cfg.d_adapter, cfg.d_model, num_heads=cfg.n_heads, dropout=cfg.dropout)
        self.audio_fusion = ModalityFusion(
            len(self.audio_group_names), cfg.d_adapter, cfg.d_model
        )
        self.video_fusion = ModalityFusion(
            len(self.video_group_names), cfg.d_adapter, cfg.d_model
        )

        self.audio_tcn = TCN(cfg.d_model, cfg.tcn_layers, cfg.tcn_kernel_size, cfg.dropout)
        self.video_tcn = TCN(cfg.d_model, cfg.tcn_layers, cfg.tcn_kernel_size, cfg.dropout)
        self.shared_tcn = TCN(cfg.d_model, cfg.tcn_layers, cfg.tcn_kernel_size, cfg.dropout)
        self.dual_tcn = DualTCN(cfg.d_model, cfg.tcn_layers, cfg.tcn_kernel_size, cfg.n_heads, dropout=cfg.dropout)
        self.audio_lstm = LSTM(cfg.d_model, cfg.d_model, n_layers=1, dropout=cfg.dropout)
        self.video_lstm = LSTM(cfg.d_model, cfg.d_model, n_layers=1, dropout=cfg.dropout)

        self.audio_asp = ASP(cfg.d_model, cfg.asp_alpha, cfg.asp_beta)
        self.video_asp = ASP(cfg.d_model, cfg.asp_alpha, cfg.asp_beta)
        self.audio_newasp = newASP(cfg.d_model, cfg.asp_alpha, cfg.asp_beta)
        self.video_newasp = newASP(cfg.d_model, cfg.asp_alpha, cfg.asp_beta)
        self.audio_proj = nn.Linear(cfg.d_model * 2, cfg.d_adapter)
        self.video_proj = nn.Linear(cfg.d_model * 2, cfg.d_adapter)

        fusion_in = 2 * cfg.d_model * 2
        fusion_in += len(self.audio_pooled_group_names) * cfg.d_adapter
        fusion_in += cfg.d_session

        self.session_embed = nn.Embedding(cfg.n_sessions, cfg.d_session)
        self.session_proj = nn.Linear(cfg.d_session, cfg.d_adapter)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, cfg.d_shared),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_shared, cfg.d_shared),
        )
        self.late_fusion_transformer = LateFusionTransformer(d_in=cfg.d_adapter, d_out=cfg.d_shared, nhead=cfg.n_heads, num_layers=2, dropout=cfg.dropout)

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

        # TODO：在这里加更早期的attention，保证对于有语义的channel建模 --> ASP背景下弱于baseline
        # a = self.inter_audio_attn(audio_adapted)
        # v = self.inter_video_attn(video_adapted)
        # print(f"DEBUG: inter modality attention output size : a {a.shape}, v {v.shape}")  # B, T, 64
        a = self.audio_fusion(audio_adapted)
        v = self.video_fusion(video_adapted)
        # print(f"DEBUG: modality fusion output size : a {a.shape}, v {v.shape}")  # 256, T, 256

        mask_a = batch["mask_audio"]
        mask_v = batch["mask_video"]
        a = a * mask_a.unsqueeze(-1).float()
        v = v * mask_v.unsqueeze(-1).float()

        # TODO：attention作为分支，不替代，做添加
        # a = self.audio_tcn(a, mask_a)
        # v = self.video_tcn(v, mask_v)
        a = self.shared_tcn(a, mask_a)
        v = self.shared_tcn(v, mask_v)
        # a, v = self.dual_tcn(a, v, mask_a, mask_v)
        # print(f"DEBUG: a size : {a.shape} , v size : {v.shape}")  # B, T, 256
        a = self.audio_lstm(a)
        v = self.video_lstm(v)
        # print(f"DEBUG: lstm output size : a {a.shape} , v size : {v.shape}")

        vad = batch["vad_signal"]
        qc = batch["qc_quality"]
        z_a = self.audio_asp(a, mask_a, vad, qc)
        z_v = self.video_asp(v, mask_v, vad, qc)
        # print(f"DEBUG: ASP output size : z_a {z_a.shape}, z_v {z_v.shape}")  # B, 512
        # z_a = self.audio_newasp(a, mask_a, vad, qc)
        # z_v = self.video_newasp(v, mask_v, vad, qc)

        # parts = [z_a, z_v]
        # parts.extend(
        #     self.audio_pooled_adapters[name](batch["audio_pooled_groups"][name])
        #     for name in self.audio_pooled_group_names
        # )
        # parts.append(self.session_embed(batch["session_idx"]))        
        # z = torch.cat(parts, dim=-1)
        # # print(f"DEBUG: concatenated feature size : {z.shape}")
        # return self.fusion_mlp(z)

        z_a = self.audio_proj(z_a)
        z_v = self.video_proj(z_v)
        parts = [z_a, z_v]
        parts.extend(
            self.audio_pooled_adapters[name](batch["audio_pooled_groups"][name])
            for name in self.audio_pooled_group_names
        )
        parts.append(self.session_proj(self.session_embed(batch["session_idx"])))
        # print(f"DEBUG: parts size : {[part.shape for part in parts]}")

        parts = torch.stack(parts, dim=1)
        # print(f"DEBUG: stacked parts size : {parts.shape}")
        z = self.late_fusion_transformer(parts).mean(dim=1)
        # print(f"DEBUG: late fusion transformer output size : {z.shape}")
        return z