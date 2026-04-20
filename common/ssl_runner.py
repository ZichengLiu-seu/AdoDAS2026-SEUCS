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

from .runner import parse_args, load_config, seed_everything, build_run_name, setup_run_dirs, setup_logging, \
    _fmt_duration, _to_device, _compute_bias_init_a1, _compute_pos_weight_a1, compute_a2_pos_weight, _build_scheduler, \
    AdaptiveLossWeight, EarlyStopping, _flatten_valid_session_mask, validate_grouped, \
    _normalize_decode_method, collect_val_logits_grouped_a1, calibrate_a1_bias,\
    generate_submission_grouped
from .data.dataset import FeatureConfig, ITEM_COLS, A1_COLS
from .data.grouped_dataset import GroupedParticipantDataset, grouped_collate_fn
from .models.grouped_model import GroupedModel, SelfSupervisedModel, PostTrainModel, CORALHead
from .models.mtcn_backbone import MTCNBackbone, BackboneConfig
from .models.my_backbone import DualTCNBackbone, DualTCNBackboneConfig, TwinTowerBackbone
from .models.heads import contrastive_loss, supcon_loss, a1_loss, a2_ordinal_loss, A1Head, A2OrdinalHead
from .utils.ckpt import save_checkpoint, load_checkpoint
from .utils.run_naming import build_run_name, setup_run_dirs
from .utils.run_metadata import RunMetadata


def pretrain_one_epoch(
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
    adaptive_loss_weight: AdaptiveLossWeight | None = None,
) -> float:
    ssl_model.train()
    total_loss = 0.0
    n_batches = 0

    desc = f"Train {epoch}/{epochs}"
    if best_metric >= 0:
        desc += f" [best={best_metric:.4f}]"
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)

    # contraLoss = contrastive_loss(N=loader.batch_size*4, device=device)
    contraLoss = supcon_loss()
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

        # targets = torch.tensor(batch["flat_pids"]).to(device)
        targets = batch["participant_y_a1"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            out = ssl_model(flat_batch, B, session_valid)
            valid_session_mask = _flatten_valid_session_mask(session_valid)
            has_valid_sessions = bool(valid_session_mask.any().item())

            # low_part_contrastive_loss = contraLoss(out["a_low_repr"], out["v_low_repr"])
            # high_part_contrastive_loss = contraLoss(out["a_high_repr"], out["v_high_repr"])            
            low_part_contrastive_loss = contraLoss(out["a_low_repr"], out["v_low_repr"], targets)
            high_part_contrastive_loss = contraLoss(out["a_high_repr"], out["v_high_repr"], targets)
            # print(f"[DEBUG] low_part_contrastive_loss: {low_part_contrastive_loss:.4f}")
            # print(f"[DEBUG] high_part_contrastive_loss: {high_part_contrastive_loss:.4f}")

            if adaptive_loss_weight is not None:
                weights = adaptive_loss_weight()
                loss = weights[0] * low_part_contrastive_loss + weights[1] * high_part_contrastive_loss
            else:
                loss = low_part_contrastive_loss + high_part_contrastive_loss

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(ssl_model.parameters()), # + list(adaptive_loss_weight.parameters()),
                max_norm=grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(ssl_model.parameters()), #  + list(adaptive_loss_weight.parameters()),    
                max_norm=grad_clip,
            )
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix_str(f"{loss.item():.4f}")
    
    pbar.close()
    return total_loss / max(n_batches, 1)


def posttrain_one_epoch(
    ssl_model: SelfSupervisedModel,
    task_head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task: str,
    epoch: int,
    epochs: int,
    scaler=None,
    use_amp: bool = True,
    pos_weight=None,
    grad_clip: float = 1.0,
    session_loss_weight: float = 0.5,
    session_type_loss_weight: float = 0.15,
    best_metric: float = -1.0,
    label_smoothing: float = 0.0,
    feature_noise_std: float = 0.01,
    adaptive_loss_weight: AdaptiveLossWeight | None = None,
)->float:    
    ssl_model.train()
    task_head.train()
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

        if task == "ssl_posttrain_a1":
            targets = batch["participant_y_a1"].to(device)
        elif task == "ssl_posttrain_a2":
            targets = batch["participant_y_a2"].to(device).long()

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            out = ssl_model(flat_batch, B, session_valid)
            valid_session_mask = _flatten_valid_session_mask(session_valid)
            has_valid_sessions = bool(valid_session_mask.any().item())

            p_logits = task_head(out["participant_repr"])
            if task == "ssl_posttrain_a1":
                main_loss = a1_loss(p_logits, targets, pos_weight=pos_weight, label_smoothing=label_smoothing)
            elif task == "ssl_posttrain_a2":
                main_loss = a2_ordinal_loss(p_logits, targets, pos_weight=pos_weight, label_smoothing=label_smoothing)

            if has_valid_sessions:
                s_logits = task_head(out["session_reprs"])[valid_session_mask]
                if task == "ssl_posttrain_a1":
                    s_targets = targets.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 3)[valid_session_mask]
                    sess_loss = a1_loss(s_logits, s_targets, pos_weight=pos_weight, label_smoothing=label_smoothing)
                elif task == "ssl_posttrain_a2":
                    s_targets = targets.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 21)[valid_session_mask]
                    sess_loss = a2_ordinal_loss(
                        s_logits, s_targets, pos_weight=pos_weight, label_smoothing=label_smoothing
                    )

                type_loss = F.cross_entropy(
                    out["session_type_logits"][valid_session_mask],
                    session_types[valid_session_mask],
                )
            else:
                sess_loss = p_logits.new_zeros(())
                type_loss = p_logits.new_zeros(())

            if adaptive_loss_weight is not None:
                weights = adaptive_loss_weight()
                loss = main_loss + weights[0] * sess_loss + weights[1] * type_loss
            else:
                loss = main_loss + session_loss_weight * sess_loss + session_type_loss_weight * type_loss

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(ssl_model.parameters()) + list(task_head.parameters()) + list(adaptive_loss_weight.parameters()),
                max_norm=grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(ssl_model.parameters()) + list(task_head.parameters()) + list(adaptive_loss_weight.parameters()),    
                max_norm=grad_clip,
            )
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix_str(f"{loss.item():.4f}")

    pbar.close()
    return total_loss / max(n_batches, 1)


def validate(
    ssl_model: SelfSupervisedModel,
    loader: DataLoader,
    device: torch.device,
    task: str,
    epoch: int,
    epochs: int,
    use_amp: bool = True,
    adaptive_loss_weight: AdaptiveLossWeight | None = None,
):
    ssl_model.eval()
    if adaptive_loss_weight is not None:
        adaptive_loss_weight.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []
    all_logits = []
    all_sess_preds = []
    # contraLoss = contrastive_loss(N=loader.batch_size*4, device=device)
    contraLoss = supcon_loss()

    for batch in tqdm(loader, desc=f"Val {epoch}/{epochs}", leave=False, dynamic_ncols=True):
        flat_batch = _to_device(batch["flat_batch"], device)
        session_valid = batch["session_valid"].to(device)
        B = batch["n_participants"]
        targets = batch["participant_y_a1"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            out = ssl_model(flat_batch, B, session_valid)
            valid_session_mask = _flatten_valid_session_mask(session_valid)

            # low_part_contrastive_loss = contraLoss(out["a_low_repr"], out["v_low_repr"])
            # high_part_contrastive_loss = contraLoss(out["a_high_repr"], out["v_high_repr"])
            low_part_contrastive_loss = contraLoss(out["a_low_repr"], out["v_low_repr"], targets)
            high_part_contrastive_loss = contraLoss(out["a_high_repr"], out["v_high_repr"], targets)

            if adaptive_loss_weight is not None:
                weights = adaptive_loss_weight.get_weights()
                loss = weights[0] * low_part_contrastive_loss + weights[1] * high_part_contrastive_loss
            else:
                loss = low_part_contrastive_loss + high_part_contrastive_loss

        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def preTrain():
    log = logging.getLogger("pretrain_grouped")
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
        backbone = TwinTowerBackbone(bb_cfg)
    else:
        log.error(f"pretrained do not needed in this model")
        

    ssl_model = SelfSupervisedModel(
        backbone=backbone,
        d_shared=bb_cfg.d_shared,
        aggregator_method=cfg.get("aggregator", "mlp"),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)

    n_params = sum(p.numel() for p in ssl_model.parameters())
    log.info(f"Model params: {n_params:,}")

    use_amp = bool(cfg.get("amp", True))
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        log.info("AMP enabled (BF16)")

    grad_clip = cfg.get("grad_clip", 1.0)

    adaptive_loss_weight = AdaptiveLossWeight(initial_weights=[1, 1], device=device)
    params = list(ssl_model.parameters())
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

    patience = cfg.get("patience", 8)
    early_stop_metric = cfg.get("early_stop_metric", "val_loss")
    es_mode = "min" if early_stop_metric == "val_loss" else "max"
    early_stop = EarlyStopping(patience=patience, mode=es_mode)
    log.info(f"EarlyStopping: patience={patience}, metric={early_stop_metric}, mode={es_mode}")

    # label_smoothing = cfg.get("label_smoothing", 0.05)
    feature_noise_std = cfg.get("feature_noise_std", 0.01)
    session_drop_prob = cfg.get("session_drop_prob", 0.1)
    # log.info(f"Label smoothing: {label_smoothing}")
    log.info(f"Feature noise std: {feature_noise_std}")
    log.info(f"Session drop prob: {session_drop_prob}")

    best_metric = 1e10
    metric_name = "Contrastive Loss"
    t_start = time.time()

    log.info("=" * 90)
    if task == "ssl_pretrain":
        log.info("   Epoch  |    LR    | Train Loss | Val Loss | Time")
    else:
        log.error(" FAULT ENTRY ")
    log.info("=" * 90)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = pretrain_one_epoch(
            ssl_model, train_loader, optimizer, device,
            task, epoch, epochs, scaler, use_amp, 
            grad_clip=grad_clip,
            best_metric=best_metric,
            feature_noise_std=feature_noise_std,
            adaptive_loss_weight=None,
        )
        val_loss = validate(
            ssl_model, val_loader, device,
            task, epoch, epochs, use_amp,
            adaptive_loss_weight=None,
        )
        scheduler.step()

        elapsed = time.time() - t0
        total_elapsed = time.time() - t_start
        eta = (total_elapsed / epoch) * (epochs - epoch)
        lr_now = optimizer.param_groups[0]["lr"]
        vram_gb = torch.cuda.max_memory_allocated() / 1024**3

        primary = val_loss
        is_best = primary < best_metric
        marker = " *" if is_best else ""

        if task == "ssl_pretrain":
            log.info(
                f"  {epoch:3d}/{epochs:3d} | {lr_now:.2e} |   {train_loss:.4f}   |  {val_loss:.4f}  | "
                f"{_fmt_duration(elapsed)} ETA {_fmt_duration(eta)} VRAM {vram_gb:.1f}G{marker}"
            )

        if is_best:
            best_metric = primary
            save_checkpoint(
                run_dirs["checkpoints"] / "best.pt",
                ssl_model, optimizer, epoch, best_metric,
            )
            log.info(f"  >>> New best {metric_name}={best_metric:.4f} saved at epoch {epoch}.")
            meta.update_best(epoch, val_loss)
        
        es_value = val_loss if early_stop_metric == "val_loss" else primary
        if early_stop.step(es_value):
            log.info(f"  EarlyStopping triggered at epoch {epoch} (patience={patience}, metric={early_stop_metric})")
            break

    log.info("=" * 90)
    total_time = time.time() - t_start
    log.info(f"PreTrain complete. Best {metric_name}={best_metric:.4f}, time={_fmt_duration(total_time)}")

    meta.finish("completed")
    log.info(f"Run complete: {run_name}")
    log.info(f"Output dir: {run_dirs['root']}")


def postTrain():
    log = logging.getLogger("posttrain_grouped")
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
    pt_path = Path(cfg.get("pt_path", None))
    
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
    print(f"[DEBUG] batch_size type: {type(batch_size)}")
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
    backbone = TwinTowerBackbone(bb_cfg)
    backbone.load_pretrained(pt_path, device)

    d_low=cfg.get("d_low", 32)
    d_high=cfg.get("d_high", 128)
    d_backbone_out = (d_low + d_high) * 2
    ssl_model = PostTrainModel(
        backbone=backbone,
        d_shared=bb_cfg.d_shared,
        d_backbone_out=d_backbone_out,
        aggregator_method=cfg.get("aggregator", "mlp"),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)

    use_coral = bool(cfg.get("use_coral", False))
    if task == "ssl_posttrain_a1":
        bias_init = _compute_bias_init_a1(manifest_dir / "train.csv")
        # task_head = A1Head(d_backbone_out, bias_init=bias_init).to(device)
        task_head = A1Head(bb_cfg.d_shared, bias_init=bias_init).to(device)
    elif task == "ssl_posttrain_a2":
        if use_coral:
            task_head = CORALHead(d_backbone_out).to(device)
            log.info("Using CORAL head for A2")
        else:
            task_head = A2OrdinalHead(d_backbone_out).to(device)

    n_params = sum(p.numel() for p in ssl_model.parameters()) + sum(p.numel() for p in task_head.parameters())
    log.info(f"Model params: {n_params:,}")

    use_amp = bool(cfg.get("amp", True))
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        log.info("AMP enabled (BF16)")

    grad_clip = cfg.get("grad_clip", 1.0)
    pos_weight_t = None
    if cfg.get("use_pos_weight", True):
        if task == "ssl_posttrain_a1":
            pw = _compute_pos_weight_a1(manifest_dir / "train.csv")
            pos_weight_t = torch.tensor(pw, dtype=torch.float32, device=device)
            log.info(f"pos_weight [D/A/S]: {pw[0]:.2f} / {pw[1]:.2f} / {pw[2]:.2f}")
        elif task == "ssl_posttrain_a2":
            pos_weight_t = compute_a2_pos_weight(manifest_dir / "train.csv").to(device)
            log.info(f"A2 pos_weight shape: {pos_weight_t.shape}")

    session_loss_weight = cfg.get("session_loss_weight", 0.2)
    session_type_loss_weight = cfg.get("session_type_loss_weight", 0.15)
    adaptive_loss_weight = AdaptiveLossWeight(initial_weights=[session_loss_weight, session_type_loss_weight], device=device)
    params = list(ssl_model.parameters()) + list(task_head.parameters()) + list(adaptive_loss_weight.parameters())
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

    best_metric = -1.0
    metric_name = "F1" if task == "ssl_posttrain_a1" else "QWK"
    t_start = time.time()

    log.info("=" * 90)
    if task == "ssl_posttrain_a1":
        log.info("  Epoch  |    LR     | Train Loss | Val Loss | F1 raw | F1 sel |  AUROC | F1[D/A/S]       | Time")
    elif task == "ssl_posttrain_a2":
        log.info("  Epoch  |    LR     | Train Loss | Val Loss | mean QWK | mean MAE | Time")
    log.info("=" * 90)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = posttrain_one_epoch(
            ssl_model, task_head, train_loader, optimizer, device,
            task, epoch, epochs, scaler, use_amp,
            pos_weight=pos_weight_t, grad_clip=grad_clip,
            session_loss_weight=session_loss_weight,
            session_type_loss_weight=session_type_loss_weight,
            best_metric=best_metric,
            label_smoothing=label_smoothing,
            feature_noise_std=feature_noise_std,
            adaptive_loss_weight=adaptive_loss_weight,
        )

        val_metrics = validate_grouped(
            ssl_model, task_head, val_loader, device,
            task, epoch, epochs, use_amp, pos_weight=pos_weight_t,
            decode_method=cfg.get("decode_method", "expectation"),
        )
        scheduler.step()

        elapsed = time.time() - t0
        total_elapsed = time.time() - t_start
        eta = (total_elapsed / epoch) * (epochs - epoch)
        lr_now = optimizer.param_groups[0]["lr"]
        vram_gb = torch.cuda.max_memory_allocated() / 1024**3

        primary = val_metrics["primary_metric"]
        is_best = primary > best_metric
        marker = " *" if is_best else ""

        if task == "ssl_posttrain_a1":
            pcf1 = val_metrics.get("pcf1", [0, 0, 0])
            selected_f1 = val_metrics["primary_metric"]
            log.info(
                f"  {epoch:3d}/{epochs:3d} | {lr_now:.2e} |   {train_loss:.4f}   |  {val_metrics['loss']:.4f}  | "
                f"{val_metrics['mean_f1']:.4f} | {selected_f1:.4f} | {val_metrics['auroc']:.4f} | "
                f"{pcf1[0]:.3f}/{pcf1[1]:.3f}/{pcf1[2]:.3f} | "
                f"{_fmt_duration(elapsed)} ETA {_fmt_duration(eta)} VRAM {vram_gb:.1f}G{marker}"
            )
        elif task == "ssl_posttrain_a2":
            log.info(
                f"  {epoch:3d}/{epochs:3d} | {lr_now:.2e} |   {train_loss:.4f}   |  {val_metrics['loss']:.4f}  | "
                f" {val_metrics['mean_qwk']:.4f}  |  {val_metrics['mean_mae']:.4f}  | "
                f"{_fmt_duration(elapsed)} ETA {_fmt_duration(eta)} VRAM {vram_gb:.1f}G{marker}"
            )

        if is_best:
            best_metric = primary
            save_checkpoint(
                run_dirs["checkpoints"] / "best.pt",
                ssl_model, optimizer, epoch, best_metric,
                extra={"head_state_dict": task_head.state_dict()},
            )
            log.info(f"  >>> New best {metric_name}={best_metric:.4f} saved at epoch {epoch}.")
            meta.update_best(epoch, val_metrics)

        es_value = val_metrics["loss"] if early_stop_metric == "val_loss" else primary
        if early_stop.step(es_value):
            log.info(f"  EarlyStopping triggered at epoch {epoch} (patience={patience}, metric={early_stop_metric})")
            break

    log.info("=" * 90)
    total_time = time.time() - t_start
    log.info(f"Training complete. Best {metric_name}={best_metric:.4f}, time={_fmt_duration(total_time)}")

    log.info("Loading best checkpoint for submission generation ...")
    state = load_checkpoint(run_dirs["checkpoints"] / "best.pt", ssl_model, optimizer=None)
    task_head.load_state_dict(state["head_state_dict"])
    ssl_model.to(device)
    task_head.to(device)

    submission_level = cfg.get("submission_level", "participant")
    decode_method = _normalize_decode_method(cfg.get("decode_method", "expectation"))
    log.info(f"Submission level: {submission_level}")
    log.info(f"Decode method: {decode_method}")

    a1_biases = None
    a2_offsets = None
    selected_decode_method = decode_method

    if task == "ssl_posttrain_a1":
        log.info("Calibrating per-task bias offsets on val ...")
        val_logits, val_labels = collect_val_logits_grouped_a1(
            ssl_model, task_head, val_loader, device, use_amp,
            submission_level=submission_level,
        )
        biases, cal_f1s = calibrate_a1_bias(val_logits, val_labels)
        for t, name in enumerate(["D", "A", "S"]):
            log.info(f"  {name}: bias={biases[t]:+.2f}  F1_cal={cal_f1s[t]:.4f}")
        cal_mean_f1 = float(np.mean(cal_f1s))
        best_raw_f1 = float(meta.meta.get("best_metrics", {}).get("mean_f1", best_metric))
        best_selected_f1 = float(meta.meta.get("best_metrics", {}).get("primary_metric", best_metric))
        log.info(
            f"  Mean calibrated F1: {cal_mean_f1:.4f} "
            f"(vs selected best: {best_selected_f1:.4f}, raw best: {best_raw_f1:.4f})"
        )
        a1_biases = biases
        final_a1_metric = max(best_raw_f1, cal_mean_f1)
        final_a1_strategy = "bias_calibrated" if cal_mean_f1 >= best_raw_f1 else "raw"
        meta.set_extra("final_selected_strategy", final_a1_strategy)
        meta.set_extra("final_selected_metrics", {
            "mean_f1": final_a1_metric,
            "mean_f1_raw": best_raw_f1,
            "mean_f1_calibrated": cal_mean_f1,
            "auroc": meta.meta.get("best_metrics", {}).get("auroc"),
        })

        cal_data = {"biases": biases.tolist(), "cal_f1": cal_f1s, "mean_cal_f1": cal_mean_f1}
        with open(run_dirs["calibration"] / "a1_bias_grouped.json", "w") as f:
            json.dump(cal_data, f, indent=2)
    elif task == "ssl_posttrain_a2":
        log.info("Calibrating and selecting A2 decode strategy on val ...")
        val_logits, val_labels = collect_val_logits_grouped_a2(
            ssl_model, task_head, val_loader, device, use_amp,
            submission_level=submission_level,
        )
        val_labels_int = val_labels.astype(int)
        raw_results = _evaluate_a2_decode_candidates(
            task_head,
            torch.from_numpy(val_logits).float(),
            val_labels_int,
            decode_methods=["argmax", "monotonic", "expectation"],
        )
        calibrated_results = {}
        for method in ("argmax", "monotonic", "expectation"):
            offsets, item_qwks = calibrate_a2_thresholds(
                val_logits,
                val_labels_int,
                decode_method=method,
            )
            preds = _decode_a2_logits(
                task_head,
                torch.from_numpy(val_logits).float() + torch.as_tensor(offsets, dtype=torch.float32),
                decode_method=method,
            ).cpu().numpy()
            calibrated_results[f"calibrated_{method}"] = {
                "preds": preds,
                "qwk": mean_qwk(preds, val_labels_int),
                "mae": mean_mae(preds, val_labels_int),
                "decode_method": method,
                "offsets": offsets,
                "item_qwks": item_qwks,
            }

        strategy_results = {**raw_results, **calibrated_results}
        best_strategy, best_result = _select_best_a2_result(strategy_results)
        selected_decode_method = str(best_result["decode_method"])
        a2_offsets = best_result.get("offsets")

        log.info("  A2 decode comparison on val:")
        for name in ("argmax", "monotonic", "expectation", "calibrated_argmax", "calibrated_monotonic", "calibrated_expectation"):
            result = strategy_results[name]
            preds = result["preds"]
            total = preds.size
            dist = [np.sum(preds == v) / total * 100 for v in range(4)]
            log.info(
                f"    {name:<22} QWK={float(result['qwk']):.4f} MAE={float(result['mae']):.4f} "
                f"| 0={dist[0]:.1f}% 1={dist[1]:.1f}% 2={dist[2]:.1f}% 3={dist[3]:.1f}%"
            )

        log.info(
            f"  Selected A2 strategy: {best_strategy} "
            f"(decode={selected_decode_method}, QWK={float(best_result['qwk']):.4f}, MAE={float(best_result['mae']):.4f})"
        )

        meta.set_extra("final_selected_strategy", best_strategy)
        meta.set_extra("final_selected_metrics", {
            "mean_qwk": float(best_result["qwk"]),
            "mean_mae": float(best_result["mae"]),
            "decode_method": selected_decode_method,
        })

        cal_data = {
            "selected_strategy": best_strategy,
            "selected_decode_method": selected_decode_method,
            "selected_qwk": float(best_result["qwk"]),
            "selected_mae": float(best_result["mae"]),
            "strategies": {
                name: {
                    "decode_method": str(result["decode_method"]),
                    "qwk": float(result["qwk"]),
                    "mae": float(result["mae"]),
                    **({"offsets": result["offsets"].tolist()} if "offsets" in result else {}),
                    **({"item_qwks": result["item_qwks"]} if "item_qwks" in result else {}),
                }
                for name, result in strategy_results.items()
            },
        }
        with open(run_dirs["calibration"] / "a2_threshold_offsets_grouped.json", "w") as f:
            json.dump(cal_data, f, indent=2)

    if bool(cfg.get("run_inference_after_train", False)):
        run_dirs["submissions"].mkdir(parents=True, exist_ok=True)
        for split_name in ("val", "test_hidden"):
            manifest_path = manifest_dir / f"{split_name}.csv"
            if not manifest_path.exists():
                continue
            ds = GroupedParticipantDataset(manifest_path, feat_cfg, split=split_name)
            loader = DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, collate_fn=grouped_collate_fn,
            )

            pids, sessions, preds = generate_submission_grouped(
                ssl_model, task_head, loader, device, task, use_amp,
                desc=f"Submit {split_name}",
                submission_level=submission_level,
                a1_biases=a1_biases,
                decode_method=selected_decode_method,
                a2_threshold_offsets=a2_offsets,
            )

            manifest_df = pd.read_csv(manifest_path)
            file_ids = []
            filtered_preds = []
            if submission_level == "participant":
                pid_to_info = {}
                for _, row in manifest_df.iterrows():
                    pid = str(row["anon_pid"])
                    pid_to_info.setdefault(pid, (str(row["anon_school"]), str(row["anon_class"])))

                for pid, pred in zip(pids, preds):
                    pid_str = str(pid)
                    info = pid_to_info.get(pid_str)
                    if info is None:
                        continue
                    school, cls = info
                    file_ids.append(f"{school}_{cls}_{pid_str}")
                    filtered_preds.append(pred)
                expected_rows = int(manifest_df["anon_pid"].astype(str).nunique())
            else:
                pid_to_info = {}
                for _, row in manifest_df.iterrows():
                    pid_to_info[(str(row["anon_pid"]), str(row["session"]))] = (
                        str(row["anon_school"]), str(row["anon_class"])
                    )

                for pid, sess, pred in zip(pids, sessions, preds):
                    key = (str(pid), str(sess))
                    info = pid_to_info.get(key)
                    if info is None:
                        continue
                    school, cls = info
                    file_ids.append(f"{school}_{cls}_{key[0]}_{key[1]}")
                    filtered_preds.append(pred)
                expected_rows = len(manifest_df)

            if filtered_preds:
                preds = np.asarray(filtered_preds)
            elif task == "ssl_posttrain_a1":
                preds = np.zeros((0, 3), dtype=np.float32)
            elif task == "ssl_posttrain_a2":
                preds = np.zeros((0, 21), dtype=np.int64)
            if len(file_ids) != expected_rows:
                log.warning(
                    f"Submission row count mismatch for {split_name}: expected={expected_rows} generated={len(file_ids)}"
                )

            if task == "ssl_posttrain_a1":
                sub = pd.DataFrame({
                    "file_id": file_ids,
                    "p_D": preds[:, 0],
                    "p_A": preds[:, 1],
                    "p_S": preds[:, 2],
                })
            elif task == "ssl_posttrain_a2":
                item_cols = [f"d{i:02d}" for i in range(1, 22)]
                sub = pd.DataFrame({"file_id": file_ids})
                for j, col in enumerate(item_cols):
                    sub[col] = preds[:, j]

            out_path = run_dirs["submissions"] / f"submission_{task}_{split_name}.csv"
            sub.to_csv(out_path, index=False)
            log.info(f"Wrote {len(sub)} rows to {out_path}")
    else:
        log.info("Skipping submission generation after training; use infer.py for release inference.")

    meta.finish("completed")
    log.info(f"Run complete: {run_name}")
    log.info(f"Output dir: {run_dirs['root']}")
    