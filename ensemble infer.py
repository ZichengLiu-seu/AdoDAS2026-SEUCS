#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from common.data.dataset import FeatureConfig
from common.data.grouped_dataset import GroupedParticipantDataset, grouped_collate_fn
from common.models.grouped_model import CORALHead, GroupedModel, PostTrainModel, PreTrainModel
from common.models.heads import A1Head, A2OrdinalHead, A1SpecificHead
from common.models.mtcn_backbone import BackboneConfig, MTCNBackbone
from common.models.my_backbone import DualTCNBackboneConfig, DualTCNBackbone, TwinTowerBackbone
from common.runner import (
    _normalize_decode_method,
    generate_submission_grouped,
    setup_logging, _compute_bias_init_a1, _compute_pos_weight_a1, compute_a2_pos_weight, _build_scheduler, \
    AdaptiveLossWeight, EarlyStopping, _flatten_valid_session_mask, validate_grouped, \
    collect_val_logits_grouped_a1, collect_val_logits_grouped_a2, calibrate_a1_bias, _evaluate_a2_decode_candidates, calibrate_a2_thresholds, \
    _decode_a2_logits, _select_best_a2_result,
)
from common.utils.ckpt import load_checkpoint, load_taskhead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["a1", "a2"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_config(config_path: str | None, checkpoint_path: Path=None) -> dict:
    if config_path is None:
        candidate = checkpoint_path.parent.parent / "config_used.yaml"
        config_path = str(candidate)
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    feature_selection = cfg.pop("feature_selection", {}) or {}
    if not isinstance(feature_selection, dict):
        raise TypeError("feature_selection must be a mapping in the config YAML")
    cfg.update(feature_selection)
    return cfg


def load_calibration(run_dir: Path, task: str) -> tuple[torch.Tensor | None, torch.Tensor | None, str]:
    calibration_dir = run_dir / "calibration"
    if task == "a1":
        path = calibration_dir / "a1_bias_grouped.json"
        if not path.exists():
            return None, None, None
        with open(path) as f:
            data = json.load(f)
        biases = torch.tensor(data.get("biases", []), dtype=torch.float32) if data.get("biases") else None
        return biases, None, "expectation"

    path = calibration_dir / "a2_threshold_offsets_grouped.json"
    if not path.exists():
        return None, None, _normalize_decode_method("expectation")
    with open(path) as f:
        data = json.load(f)
    selected_method = _normalize_decode_method(data.get("selected_decode_method", "expectation"))
    strategies = data.get("strategies", {})
    selected_strategy = data.get("selected_strategy", "")
    offsets = None
    if selected_strategy in strategies and "offsets" in strategies[selected_strategy]:
        offsets = torch.tensor(strategies[selected_strategy]["offsets"], dtype=torch.float32)
    # print(f"[DEBUG]: selected_method: {data.get('selected_method', '')}")
    return None, offsets, selected_method


def hard_voting_ensemble(fold_preds, fold_logits, task):
    fold_preds = np.array(fold_preds)
    fold_logits = np.array(fold_logits)
    n_folds, num = fold_preds.shape[0], fold_preds.shape[1]
    threshold = n_folds - 1
    print(f"[DEBUG] logits's scale: max: {max(fold_logits.flatten()):.4f}, min: {min(fold_logits.flatten()):.4f})")

    final_preds = []
    for n in range(num):
        sample_preds = []
        if task == "a1":
            for t in range(3):
                if task == "a1":
                    a = fold_preds[:, n, t]
                    votes_1 = int((fold_preds[:, n, t] == 1).sum())
                    majority_votes = max(votes_1, n_folds - votes_1)
                    
                    if majority_votes >= threshold:
                        sample_preds.append(int(1) if majority_votes == votes_1 else int(0))
                    else:
                        avg_logit = np.mean([fold_logits[f][n, t] for f in range(n_folds)])
                        sample_preds.append(int(1) if avg_logit > 0 else int(0))
        else:
            # For A2, we can do majority voting on the predicted classes
            class_votes = np.sum(fold_preds[:, n, :], axis=0)
            final_pred = np.argmax(class_votes)
        final_preds.append(sample_preds)
    
    return np.array(final_preds, dtype=np.int32)


def main() -> None:
    args = parse_args()
    # checkpoint_path = Path(args.checkpoint).resolve()
    # cfg = load_config(args.config, checkpoint_path)'
    cfg = load_config(args.config)
    checkpoint_path = Path(cfg.get("checkpoint", None)).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = checkpoint_path.parent.parent
    setup_logging(run_dir / "logs", f"infer_{args.task}")

    manifest_dir = Path(cfg.get("manifest_dir", "/media/k3nwong/Data1/test/outputs/data"))
    manifest_path = Path(args.manifest) if args.manifest else manifest_dir / f"{args.split}.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    defaults = FeatureConfig()
    feat_cfg = FeatureConfig(
        feature_root=cfg.get("feature_root", defaults.feature_root),
        audio_features=cfg.get("audio_features", defaults.audio_features),
        video_features=cfg.get("video_features", defaults.video_features),
        audio_ssl_model_tag=cfg.get("audio_ssl_model_tag", defaults.audio_ssl_model_tag),
        video_ssl_model_tag=cfg.get("video_ssl_model_tag", defaults.video_ssl_model_tag),
        mask_policy=cfg.get("mask_policy", defaults.mask_policy),
        core_audio=cfg.get("core_audio", defaults.core_audio),
        core_video=cfg.get("core_video", defaults.core_video),
    )

    ds = GroupedParticipantDataset(manifest_path, feat_cfg, split=args.split)
    preload = bool(cfg.get("preload", True))
    num_workers = int(cfg.get("num_workers", 8))
    if preload:
        ds.preload(desc=f"Preload {args.split}")
        num_workers = 0

    loader = DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=grouped_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    dims = ds.feature_dims
    audio_group_dims = {n: dims[n] for n in feat_cfg.audio_sequence_features if n in dims}
    audio_pooled_group_dims = {n: dims[n] for n in feat_cfg.audio_pooled_features if n in dims}
    video_group_dims = {n: dims[n] for n in feat_cfg.video_features if n in dims}
    temporal_conv = cfg.get("temporal_conv", "default")
    if temporal_conv == "default":
        bb_cfg = BackboneConfig(
            audio_group_dims=audio_group_dims,
            audio_pooled_group_dims=audio_pooled_group_dims,
            video_group_dims=video_group_dims,
            d_adapter=cfg.get("d_adapter", 64),
            d_model=cfg.get("d_model", 256),
            tcn_layers=cfg.get("tcn_layers", 6),
            tcn_kernel_size=cfg.get("tcn_kernel_size", 3),
            asp_alpha=cfg.get("asp_alpha", 0.5),
            asp_beta=cfg.get("asp_beta", 0.5),
            dropout=cfg.get("dropout", 0.2),
            d_shared=cfg.get("d_shared", 256),
        ) 
        backbone = MTCNBackbone(bb_cfg)
    elif temporal_conv == "DualTCN":
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
    elif temporal_conv == "TwinTower":
        bb_cfg = DualTCNBackboneConfig(
            audio_group_dims=audio_group_dims,
            audio_pooled_group_dims=audio_pooled_group_dims,
            video_group_dims=video_group_dims,
            d_adapter=cfg.get("d_adapter", 64),
            d_model=cfg.get("d_model", 256),
            d_low=cfg.get("d_low", 32),
            d_high=cfg.get("d_high", 128),
            tcn_layers=cfg.get("tcn_layers", 6),
            tcn_kernel_size=cfg.get("tcn_kernel_size", 3),
            n_heads=cfg.get("n_heads", 4),
            asp_alpha=cfg.get("asp_alpha", 0.5),
            asp_beta=cfg.get("asp_beta", 0.5),
            dropout=cfg.get("dropout", 0.2),
            d_shared=cfg.get("d_shared", 256),
        )
        backbone = TwinTowerBackbone(bb_cfg)    

    d_backbone_out = None
    fold = cfg.get("fold", 4)
    if temporal_conv == "TwinTower":
        d_low=cfg.get("d_low", 32)
        d_high=cfg.get("d_high", 128)
        d_backbone_out = (d_low + d_high) * 4
        # print(f"[DEBUG] d_backbone_out: {d_backbone_out}")
        grouped_model_list = [PostTrainModel(
            backbone=backbone,
            d_backbone_out=d_backbone_out,
            aggregator_method=cfg.get("aggregator", "mlp"),
            dropout=cfg.get("dropout", 0.2),
        ).to(device) for _ in range(fold)]
    else:
        grouped_model_list = [GroupedModel(
            backbone=backbone,
            d_shared=bb_cfg.d_shared,
            aggregator_method=cfg.get("aggregator", "mlp"),
            dropout=cfg.get("dropout", 0.2),
        ).to(device) for _ in range(fold)]

    if args.task == "a1":
        # task_head = A1Head(bb_cfg.d_shared).to(device)
        bias_init = _compute_bias_init_a1(manifest_dir / "train.csv")
        task_head_list = [A1SpecificHead(d_backbone_out, bias_init=bias_init).to(device) for _ in range(fold)]
    else:
        if bool(cfg.get("use_coral", False)):
            if temporal_conv == "TwinTower":
                task_head_list = [CORALHead(d_backbone_out).to(device) for _ in range(fold)]
            else:
                task_head_list = [CORALHead(bb_cfg.d_shared).to(device) for _ in range(fold)]
        else:
            if temporal_conv == "TwinTower":
                task_head_list = [A2OrdinalHead(d_backbone_out).to(device) for _ in range(fold)]
            else:
                task_head_list = [A2OrdinalHead(bb_cfg.d_shared).to(device) for _ in range(fold)]

    all_preds = []
    all_logits = []
    for i, (grouped_model, task_head) in enumerate(zip(grouped_model_list, task_head_list)):
        fold_checkpoint_path = checkpoint_path.parent / f"best-{i}.pt"
        state = load_checkpoint(fold_checkpoint_path, grouped_model, optimizer=None)
        task_head.load_state_dict(state["head_state_dict"])
        grouped_model.eval()
        task_head.eval()

        a1_biases, a2_offsets, selected_decode_method = load_calibration(run_dir, args.task)
        use_amp = bool(cfg.get("amp", True))
        submission_level = cfg.get("submission_level", "participant")

        pids, sessions, preds, logits = generate_submission_grouped(
            grouped_model=grouped_model,
            task_head=task_head,
            loader=loader,
            device=device,
            task=args.task,
            use_amp=use_amp,
            desc=f"Infer Fold {i+1}/{fold}",
            submission_level=submission_level,
            a1_biases=None if a1_biases is None else a1_biases.to(device),
            decode_method=selected_decode_method,
            a2_threshold_offsets=None if a2_offsets is None else a2_offsets.to(device),
        )
        all_preds.append(preds)
        all_logits.append(logits)

    final_preds = hard_voting_ensemble(all_preds, all_logits, args.task)

    manifest_df = pd.read_csv(manifest_path)
    file_ids = []
    filtered_preds = []
    if submission_level == "participant":
        pid_to_info = {}
        for _, row in manifest_df.iterrows():
            pid = str(row["anon_pid"])
            pid_to_info.setdefault(pid, (str(row["anon_school"]), str(row["anon_class"])))
        for pid, pred in zip(pids, final_preds):
            pid_str = str(pid)
            info = pid_to_info.get(pid_str)
            if info is None:
                continue
            school, cls = info
            file_ids.append({"anon_school" : school, "anon_class" : cls, "anon_pid" : pid_str})
            filtered_preds.append(pred)
    else:
        pid_to_info = {
            (str(row["anon_pid"]), str(row["session"])): (
                str(row["anon_school"]),
                str(row["anon_class"]),
            )
            for _, row in manifest_df.iterrows()
        }
        for pid, sess, pred in zip(pids, sessions, final_preds):
            key = (str(pid), str(sess))
            info = pid_to_info.get(key)
            if info is None:
                continue
            school, cls = info
            file_ids.append(f"{school}_{cls}_{key[0]}_{key[1]}")
            filtered_preds.append(pred)

    if args.task == "a1":
        sub = pd.DataFrame(
            {
                "anon_school": [f["anon_school"] for f in file_ids],
                "anon_class": [f["anon_class"] for f in file_ids],
                "anon_pid": [f["anon_pid"] for f in file_ids],
                "p_D": [float(pred[0]) for pred in filtered_preds],
                "p_A": [float(pred[1]) for pred in filtered_preds],
                "p_S": [float(pred[2]) for pred in filtered_preds],
            }
        )
    else:
        sub = pd.DataFrame(
            {
                "anon_school": [f["anon_school"] for f in file_ids],
                "anon_class": [f["anon_class"] for f in file_ids],
                "anon_pid": [f["anon_pid"] for f in file_ids],
            }
        )
        for idx, col in enumerate([f"d{i:02d}" for i in range(1, 22)]):
            sub[col] = [int(pred[idx]) for pred in filtered_preds]

    output_path = Path(args.output) if args.output else run_dir / "submissions" / f"submission_{args.task}_{args.split}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
