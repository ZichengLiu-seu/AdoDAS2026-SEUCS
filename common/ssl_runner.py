import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
# from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from common.runner import parse_args, load_config, seed_everything, build_run_name, setup_run_dirs, setup_logging, _fmt_duration, _to_device, _compute_bias_init_a1, _compute_pos_weight_a1, compute_a2_pos_weight, AdaptiveLossWeight, _build_scheduler, EarlyStopping, _flatten_valid_session_mask
from common.data.dataset import FeatureConfig, ITEM_COLS, A1_COLS
from common.data.grouped_dataset import GroupedParticipantDataset, grouped_collate_fn
from common.models.grouped_model import GroupedModel, SelfSupervisedModel, PostTrainModel, CORALHead
from common.models.my_backbone import DualTCNBackbone, DualTCNBackboneConfig, TwinTowerBackbone
from common.models.heads import A1Head, A2OrdinalHead, a1_loss, a2_ordinal_loss
from common.utils.run_metadata import RunMetadata


log = logging.getLogger("pretrain_grouped")

def train_one_epoch(
    ssl_model: SelfSupervisedModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task: str,
    epoch: int,
    epochs: int,
    scaler=None,
    use_amp: bool = True,
    grad_clip: float = 1.0,
    best_metric: float = -1.0,
    feature_noise_std: float = 0.01,
) -> float:
    ssl_model.train()
    total_loss = 0.0
    n_batches = 0

    desc = f"Train {epoch}/{epochs}"
    if best_metric >= 0:
        desc += f" [best={best_metric:.4f}]"
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)

    for batch in pbar:
        flat_batch = _to_device(batch["flat_batch"], device)
        session_valid = batch["session_valid"].to(device)
        session_types = batch["session_types"].to(device)
        B = batch["n_participants"]

        if feature_noise_std > 0.0:
            noise_mask = (~flat_batch["pad_mask"]).unsqueeze(-1).float()
            for key in ("audio_groups", "video_groups"):
                for name in flat_batch[key]:
                    flat_batch[key][name] = flat_batch[key][name] + torch.randn_like(
                        flat_batch[key][name]
                    ) * feature_noise_std * noise_mask

        if task == "ssl_pretrain":
            targets = batch["participant_y_a1"].to(device)
        else:
            targets = batch["participant_y_a2"].to(device).long()

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            out = ssl_model(flat_batch, B, session_valid)
            valid_session_mask = _flatten_valid_session_mask(session_valid)
            has_valid_sessions = bool(valid_session_mask.any().item())

            lowdim_logits = out["lowdim_logits"]

def main():
    args = parse_args()
    cfg = load_config(args)
    task = cfg["task"]

    seed_everything(cfg.get("seed", 42))
    device_str = cfg.get("device")
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    output_root = Path(cfg.get("output_dir", "/media/k3nwong/Data1/test/train/output"))
    manifest_dir = Path(cfg.get("manifest_dir", "/media/k3nwong/Data1/test/outputs/data"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = build_run_name(cfg, task, timestamp, training_mode="grouped_participant")
    run_dirs = setup_run_dirs(output_root, run_name)

    setup_logging(run_dirs["logs"], task)
    log.info(f"Device: {device}")
    log.info(f"Task: {task}")
    log.info(f"Run name: {run_name}")
    log.info(f"Config: {cfg}")

    meta = RunMetadata(run_dirs["root"], cfg, task, run_name)

    _defaults = FeatureConfig()
    feat_cfg = FeatureConfig(
        feature_root=cfg.get("feature_root", _defaults.feature_root),
        audio_features=cfg.get("audio_features", _defaults.audio_features),
        video_features=cfg.get("video_features", _defaults.video_features),
        audio_ssl_model_tag=cfg.get("audio_ssl_model_tag", _defaults.audio_ssl_model_tag),
        video_ssl_model_tag=cfg.get("video_ssl_model_tag", _defaults.video_ssl_model_tag),
        mask_policy=cfg.get("mask_policy", _defaults.mask_policy),
        core_audio=cfg.get("core_audio", _defaults.core_audio),
        core_video=cfg.get("core_video", _defaults.core_video),
    )
    log.info(f"Mask policy: {feat_cfg.mask_policy}")

    train_ds = GroupedParticipantDataset(
        manifest_dir / "train.csv", feat_cfg, split="train",
        session_drop_prob=cfg.get("session_drop_prob", 0.1),
    )
    val_ds = GroupedParticipantDataset(manifest_dir / "val.csv", feat_cfg, split="val")

    batch_size = cfg.get("batch_size", 64)
    num_workers = cfg.get("num_workers", 8)
    log.info(f"Train: {len(train_ds)} participants, Val: {len(val_ds)} participants")

    preload = bool(cfg.get("preload", True))
    if preload:
        log.info("Preloading data into RAM ...")
        t_pre = time.time()
        train_gb = train_ds.preload(desc="Preload train")
        val_gb = val_ds.preload(desc="Preload val")
        log.info(f"Preload done: {train_gb:.1f}G + {val_gb:.1f}G = {train_gb + val_gb:.1f}G, "
                 f"took {_fmt_duration(time.time() - t_pre)}")
        num_workers = 0

    log.info(f"batch_size={batch_size}, num_workers={num_workers}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=grouped_collate_fn,
        pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=grouped_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    dims = train_ds.feature_dims
    audio_group_dims = {n: dims[n] for n in feat_cfg.audio_sequence_features if n in dims}
    audio_pooled_group_dims = {n: dims[n] for n in feat_cfg.audio_pooled_features if n in dims}
    video_group_dims = {n: dims[n] for n in feat_cfg.video_features if n in dims}

    temporal_conv = cfg.get("temporal_conv", "default")
    if temporal_conv == "TwinTower":
        bb_cfg = DualTCNBackboneConfig(
            audio_group_dims=audio_group_dims,
            audio_pooled_group_dims=audio_pooled_group_dims,
            video_group_dims=video_group_dims,
            d_adapter=cfg.get("d_adapter", 64),
            d_model=cfg.get("d_model", 256),
            tcn_layers=cfg.get("tcn_layers", 6),
            tcn_kernel_size=cfg.get("tcn_kernel_size", 3),
            n_heads=cfg.get("n_heads", 4),
            asp_alpha=cfg.get("asp_alpha", 0.5),
            asp_beta=cfg.get("asp_beta", 0.5),
            dropout=cfg.get("dropout", 0.2),
            d_shared=cfg.get("d_shared", 256),
        )
        backbone = DualTCNBackbone(bb_cfg)
    else:
        log.error(f"pretrained do not needed in this model")
        

    grouped_model = SelfSupervisedModel(
        backbone=backbone,
        d_shared=bb_cfg.d_shared,
        aggregator_method=cfg.get("aggregator", "mlp"),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)

    use_coral = bool(cfg.get("use_coral", False))
    if task == "a1":
        bias_init = _compute_bias_init_a1(manifest_dir / "train.csv")
        task_head = A1Head(bb_cfg.d_shared, bias_init=bias_init).to(device)
    else:
        if use_coral:
            task_head = CORALHead(bb_cfg.d_shared).to(device)
            log.info("Using CORAL head for A2")
        else:
            task_head = A2OrdinalHead(bb_cfg.d_shared).to(device)

    n_params = sum(p.numel() for p in grouped_model.parameters()) + sum(p.numel() for p in task_head.parameters())
    log.info(f"Model params: {n_params:,}")

    use_amp = bool(cfg.get("amp", True))
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        log.info("AMP enabled (BF16)")

    grad_clip = cfg.get("grad_clip", 1.0)
    pos_weight_t = None
    if cfg.get("use_pos_weight", True):
        if task == "a1":
            pw = _compute_pos_weight_a1(manifest_dir / "train.csv")
            pos_weight_t = torch.tensor(pw, dtype=torch.float32, device=device)
            log.info(f"pos_weight [D/A/S]: {pw[0]:.2f} / {pw[1]:.2f} / {pw[2]:.2f}")
        else:
            pos_weight_t = compute_a2_pos_weight(manifest_dir / "train.csv").to(device)
            log.info(f"A2 pos_weight shape: {pos_weight_t.shape}")

    session_loss_weight = cfg.get("session_loss_weight", 0.2)
    session_type_loss_weight = cfg.get("session_type_loss_weight", 0.15)
    adaptive_loss_weight = AdaptiveLossWeight(initial_weights=[session_loss_weight, session_type_loss_weight], device=device)
    params = list(grouped_model.parameters()) + list(task_head.parameters()) + list(adaptive_loss_weight.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 1e-2)
    )
    epochs = cfg.get("epochs", 20)
    warmup_epochs = cfg.get("warmup_epochs", 3)
    multistep = cfg.get("multistep", False)
    scheduler = _build_scheduler(optimizer, warmup_epochs, epochs, multistep)
    if multistep:
        log.info(f"Scheduler: warmup={warmup_epochs} -> multistep at epochs, total={epochs}")
    else:
        log.info(f"Scheduler: warmup={warmup_epochs} -> cosine, total={epochs}")
    log.info(f"Grad clip: {grad_clip}")

    session_loss_weight = cfg.get("session_loss_weight", 0.5)
    session_type_loss_weight = cfg.get("session_type_loss_weight", 0.15)
    log.info(f"Session loss weight: {session_loss_weight}")
    log.info(f"Session type loss weight: {session_type_loss_weight}")

    patience = cfg.get("patience", 8)
    early_stop_metric = cfg.get("early_stop_metric", "val_loss")
    es_mode = "min" if early_stop_metric == "val_loss" else "max"
    early_stop = EarlyStopping(patience=patience, mode=es_mode)
    log.info(f"EarlyStopping: patience={patience}, metric={early_stop_metric}, mode={es_mode}")

    label_smoothing = cfg.get("label_smoothing", 0.05)
    feature_noise_std = cfg.get("feature_noise_std", 0.01)
    session_drop_prob = cfg.get("session_drop_prob", 0.1)
    log.info(f"Label smoothing: {label_smoothing}")
    log.info(f"Feature noise std: {feature_noise_std}")
    log.info(f"Session drop prob: {session_drop_prob}")

    # best_metric = -1.0
    # metric_name = "F1" if task == "a1" else "QWK"
    t_start = time.time()

    log.info("=" * 90)
    if task == "ssl_pretrain":
        log.info("  Epoch  |   LR    | Train Loss | Val Loss | Time")
    else:
        log.error(" FAULT ENTRY ")
    log.info("=" * 90)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            grouped_model, train_loader, optimizer, device,
            task, epoch, epochs, scaler, use_amp, 
            grad_clip=grad_clip,
            feature_noise_std=feature_noise_std,
        )


if __name__ == "__main__":
    main()

