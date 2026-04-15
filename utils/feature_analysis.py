import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from common.runner import parse_args, load_config, _fmt_duration
from common.data.dataset import FeatureConfig, ITEM_COLS, A1_COLS
from common.data.grouped_dataset import GroupedParticipantDataset, grouped_collate_fn


def extract_features(dataset, name_group, target_session, mode="audio"):
    """提取数据集中的特征并返回特征矩阵和标签"""
    features_group = {}
    labels = []
    bar = tqdm(dataset, desc=f"Loading feature npz", leave=False, dynamic_ncols=True)
    for sample in bar:
        # if len(labels) >= 50:
        #     break
        sess = sample["sessions"][target_session]
        if sess is not None:
            for target_feature in name_group:
                if target_feature not in features_group:
                    features_group[target_feature] = []
                features_group[target_feature].append(sess["audio_groups"][target_feature] if mode == "audio" else sess["video_groups"][target_feature])
            # if mode == "audio":
            #     for target_feature in name_group:
            #         if target_feature not in features_group:
            #             features_group[target_feature] = []
            #         features_group[target_feature].append(sess["audio_groups"][target_feature])
            # elif mode == "video":
            #     for target_feature in name_group:
            #         if target_feature not in features_group:
            #             features_group[target_feature] = []
            #         features_group[target_feature].append(sess["video_groups"][target_feature])
            labels.append(sample["y_a1"].numpy())

    padded_features_group = {}
    for target_feature, feat_list in features_group.items():
        max_length = max(feat.shape[0] for feat in feat_list)
        feature_dim = feat_list[0].shape[1]
        padded_features = np.zeros((len(feat_list), max_length, feature_dim), dtype=np.float32)
        padded_features_group[target_feature] = padded_features

    for target_feature, padded_features in padded_features_group.items():
        for i, feat in enumerate(features_group[target_feature]):
            padded_features[i, :feat.shape[0]] = feat

    return padded_features_group, np.array(labels)


def visualize_features(features_group, labels, method="pca", modality_name="audio", session_name=1):
    """降维并可视化特征"""
    for feature_name, features in features_group.items():
        n_samples, time_steps, feature_dim = features.shape
        # processed_features = features.reshape(n_samples, time_steps * feature_dim)
        processed_features = features.mean(axis=-1)

        for label_type in range(3):
            target_label = labels[:, label_type]

            if method == "pca":
                reducer = PCA(n_components=2)
            elif method == "tsne":
                reducer = TSNE(n_components=2, random_state=42)
            # elif method == "fpca":
            #     reducer = FPCA(n_components=2) 
            else:
                raise ValueError("Unsupported method: choose 'pca' or 'tsne'")
            
            reduced_features = reducer.fit_transform(processed_features)
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=target_label, cmap="viridis", alpha=0.7)
            plt.colorbar(scatter, label=f"Target Labels:{label_type}")
            plt.title(f"Feature Visualization using {method.upper()}")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.savefig(f"utils/results/{modality_name}_{feature_name}_S{session_name}_{label_type}.png")
            plt.close()


def visualize_features_fpca(features_group, labels, modality_name="audio", session_name=1):
    """使用 FPCA 可视化功能特征"""
    for feature_name, features in features_group.items():
            n_samples, time_steps, feature_dim = features.shape  # [N, T, D]

            for label_type in range(3):
                target_label = labels[:, label_type]

                t = np.linspace(0, 1, time_steps)
                fd = FDataGrid(
                    data_matrix=features,
                    grid_points=t,
                    dim_codomain=feature_dim
                )
                fpca = FPCA(n_components=2)
                fpca.fit(fd)
                reduced_features = fpca.transform(fd)

                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=target_label, cmap="viridis", alpha=0.7)
                plt.colorbar(scatter, label=f"Target Labels: {label_type}")
                plt.title(f"Feature Visualization using FPCA")
                plt.xlabel("Component 1")
                plt.ylabel("Component 2")
                plt.savefig(f"utils/results/{modality_name}_{feature_name}_S{session_name}_{label_type}.png")
                plt.close()


def main():
    args = parse_args()
    cfg = load_config(args)
    device_str = cfg.get("device")
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    manifest_dir = Path(cfg.get("manifest_dir", "/media/k3nwong/Data1/test/outputs/data"))

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

    print("Loading datasets...")
    train_ds = GroupedParticipantDataset(
        manifest_dir / "train.csv", feat_cfg, split="train",
        session_drop_prob=cfg.get("session_drop_prob", 0.1),
    )
    # val_ds = GroupedParticipantDataset(manifest_dir / "val.csv", feat_cfg, split="val")
    dims = train_ds.feature_dims
    audio_group_dims = {n: dims[n] for n in feat_cfg.audio_sequence_features if n in dims}
    audio_pooled_group_dims = {n: dims[n] for n in feat_cfg.audio_pooled_features if n in dims}
    video_group_dims = {n: dims[n] for n in feat_cfg.video_features if n in dims}

    # 提取并可视化特征
    target_session = 1
    print("Extracting features...")
    audio_features, labels = extract_features(train_ds, audio_group_dims, target_session, mode="audio")
    visualize_features(audio_features, labels, modality_name="audio", session_name=target_session)    
    video_features, labels = extract_features(train_ds, video_group_dims, target_session, mode="video")
    visualize_features(video_features, labels, modality_name="video", session_name=target_session)


if __name__ == "__main__":
    main()
