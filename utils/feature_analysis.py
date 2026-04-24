from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, SparsePCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    import umap
except ImportError:
    umap = None

from common.runner import load_config, parse_args
from common.data.dataset import A1_COLS, FeatureConfig
from common.data.grouped_dataset import GroupedParticipantDataset


RESULT_DIR = Path("utils/results/feature_analysis")
STAGE1_DIR = RESULT_DIR / "stage1"
LDA_DIR = RESULT_DIR / "stage1_lda"
UMAP_DIR = RESULT_DIR / "stage1_umap"
STAGE2_DIR = RESULT_DIR / "stage2"
LABEL_NAMES = A1_COLS
SUMMARY_ROWS: list[dict[str, Any]] = []


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y) & (y >= 0)
    if valid.sum() < 3:
        return np.nan
    x = x[valid]
    y = y[valid]
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def migrate_legacy_results() -> None:
    """Move legacy flat outputs into stage-specific folders.

    Older runs wrote every file directly into RESULT_DIR. This migration keeps new runs tidy by
    routing historical files into stage1/stage2 based on filename patterns.
    """
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    STAGE1_DIR.mkdir(parents=True, exist_ok=True)
    LDA_DIR.mkdir(parents=True, exist_ok=True)
    UMAP_DIR.mkdir(parents=True, exist_ok=True)
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)

    stage2_tokens = ("_fpca", "_robust_fpca", "_mfpca")
    lda_tokens = ("_lda_",)
    umap_tokens = ("_umap_", "_supervised_umap_")
    for item in RESULT_DIR.iterdir():
        if item.name in {STAGE1_DIR.name, LDA_DIR.name, UMAP_DIR.name, STAGE2_DIR.name}:
            continue
        if item.is_dir():
            if any(tok in item.name for tok in stage2_tokens):
                target = STAGE2_DIR / item.name
            elif any(tok in item.name for tok in lda_tokens):
                target = LDA_DIR / item.name
            elif any(tok in item.name for tok in umap_tokens):
                target = UMAP_DIR / item.name
            else:
                target = STAGE1_DIR / item.name
        else:
            if any(tok in item.name for tok in stage2_tokens):
                target = STAGE2_DIR / item.name
            elif any(tok in item.name for tok in lda_tokens):
                target = LDA_DIR / item.name
            elif any(tok in item.name for tok in umap_tokens):
                target = UMAP_DIR / item.name
            else:
                target = STAGE1_DIR / item.name
        if target.exists():
            continue
        shutil.move(str(item), str(target))


def _sanitize_for_label(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop invalid labels so downstream stats and probes use the same sample subset."""
    valid = np.isfinite(y) & (y >= 0)
    if x.ndim == 1:
        return x[valid], y[valid]
    return x[valid], y[valid]


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return NaN instead of raising when a fold contains a single class."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return np.nan


def _safe_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return NaN instead of producing a misleading F1 on degenerate folds."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(f1_score(y_true, y_pred))


def _pad_sequences(seq_list: list[np.ndarray], mask_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pad variable-length sequences into a common [N, T, D] / [N, T] analysis tensor."""
    max_length = max(feat.shape[0] for feat in seq_list)
    feature_dim = seq_list[0].shape[1]
    features = np.zeros((len(seq_list), max_length, feature_dim), dtype=np.float32)
    masks = np.zeros((len(seq_list), max_length), dtype=bool)
    for i, (feat, mask) in enumerate(zip(seq_list, mask_list)):
        length = feat.shape[0]
        features[i, :length] = feat
        masks[i, :length] = mask[:length]
    return features, masks


def extract_features(
    dataset: GroupedParticipantDataset,
    name_group: dict[str, int] | list[str],
    target_session: int,
    mode: str = "audio",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, pd.DataFrame]:
    """Extract variable-length temporal features, masks, labels, and sample metadata."""
    feature_names = list(name_group.keys()) if isinstance(name_group, dict) else list(name_group)
    group_key = "audio_groups" if mode == "audio" else "video_groups"
    mask_key = "mask_audio" if mode == "audio" else "mask_video"

    features_group: dict[str, list[np.ndarray]] = {name: [] for name in feature_names}
    masks_group: dict[str, list[np.ndarray]] = {name: [] for name in feature_names}
    labels: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    bar = tqdm(dataset, desc=f"Loading {mode} features", leave=False, dynamic_ncols=True)
    for sample_idx, sample in enumerate(bar):
        if target_session >= len(sample["sessions"]):
            continue
        sess = sample["sessions"][target_session]
        if sess is None:
            continue

        labels.append(_as_numpy(sample["y_a1"]).astype(np.float32))
        rows.append(
            {
                "sample_idx": sample_idx,
                "anon_pid": sample.get("anon_pid", ""),
                "anon_school": sample.get("anon_school", ""),
                "anon_class": sample.get("anon_class", ""),
                "session": sess.get("session", target_session),
                "seq_len": int(sess.get("seq_len", 0)),
                "valid_ratio": float(_as_numpy(sess[mask_key]).mean()) if mask_key in sess else 0.0,
            }
        )

        base_mask = _as_numpy(sess[mask_key]).astype(bool)
        for feature_name in feature_names:
            feat = _as_numpy(sess[group_key][feature_name]).astype(np.float32)
            mask = base_mask[: feat.shape[0]]
            features_group[feature_name].append(feat)
            masks_group[feature_name].append(mask)

    padded_features: dict[str, np.ndarray] = {}
    padded_masks: dict[str, np.ndarray] = {}
    for feature_name, seq_list in features_group.items():
        if not seq_list:
            continue
        padded_features[feature_name], padded_masks[feature_name] = _pad_sequences(
            seq_list, masks_group[feature_name]
        )

    return padded_features, padded_masks, np.asarray(labels, dtype=np.float32), pd.DataFrame(rows)


def _frame_pca_projection(
    features: np.ndarray,
    masks: np.ndarray,
    max_components: int = 8,
    max_fit_frames: int = 20000,
) -> tuple[np.ndarray, int]:
    """Reduce per-frame D-dimensional features to a few temporal channels with PCA.

    PCA is fit on valid frames only so missing or padded positions do not dominate the axes.
    """
    n_samples, time_steps, feature_dim = features.shape
    valid_frames = features[masks]
    if valid_frames.shape[0] < 3:
        channel = np.zeros((n_samples, time_steps, 1), dtype=np.float32)
        for i in range(n_samples):
            if masks[i].any():
                channel[i, masks[i], 0] = features[i, masks[i]].mean(axis=1)
        return channel, 1

    if valid_frames.shape[0] > max_fit_frames:
        rng = np.random.default_rng(42)
        keep = rng.choice(valid_frames.shape[0], size=max_fit_frames, replace=False)
        fit_frames = valid_frames[keep]
    else:
        fit_frames = valid_frames

    n_components = min(max_components, feature_dim, fit_frames.shape[0] - 1)
    scaler = StandardScaler()
    pca = PCA(n_components=n_components, random_state=42)
    scaler.fit(fit_frames)
    pca.fit(scaler.transform(fit_frames))

    projected = np.zeros((n_samples, time_steps, n_components), dtype=np.float32)
    for i in range(n_samples):
        projected[i] = pca.transform(scaler.transform(features[i])).astype(np.float32)
    projected[~masks] = 0.0
    return projected, n_components


def _resample_valid_series(series: np.ndarray, valid: np.ndarray, n_bins: int) -> np.ndarray:
    """Resample a variable-length valid trajectory to a fixed-length curve."""
    if valid.sum() == 0:
        return np.zeros((n_bins, series.shape[1]), dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, int(valid.sum()))
    x_new = np.linspace(0.0, 1.0, n_bins)
    valid_series = series[valid]
    out = np.zeros((n_bins, series.shape[1]), dtype=np.float32)
    for dim in range(series.shape[1]):
        out[:, dim] = np.interp(x_new, x_old, valid_series[:, dim])
    return out


def _robust_winsorize_and_scale(
    features: np.ndarray,
    masks: np.ndarray,
    clip_percentiles: tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """Robustly clip and scale valid frames to reduce outlier dominance."""
    x = features.astype(np.float32).copy()
    feature_dim = x.shape[-1]
    for dim in range(feature_dim):
        vals = x[..., dim][masks]
        if vals.size < 5:
            continue
        lo, hi = np.percentile(vals, clip_percentiles)
        clipped = np.clip(vals, lo, hi)
        med = float(np.median(clipped))
        q25, q75 = np.percentile(clipped, [25, 75])
        iqr = max(float(q75 - q25), 1e-6)
        x[..., dim] = (np.clip(x[..., dim], lo, hi) - med) / iqr
    x[~masks] = 0.0
    return x


def build_functional_matrix(
    features: np.ndarray,
    masks: np.ndarray,
    n_bins: int = 64,
    frame_pca_components: int = 8,
    robust: bool = False,
    multivariate: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Build functional curves for FPCA/MFPCA-style analysis.

    FPCA uses the first projected temporal channel.
    MFPCA-style uses multiple projected channels concatenated on a common time grid.
    """
    base = _robust_winsorize_and_scale(features, masks) if robust else features
    projected, n_components = _frame_pca_projection(base, masks, max_components=frame_pca_components)
    if not multivariate:
        projected = projected[..., :1]
        n_components = 1

    rows: list[np.ndarray] = []
    names: list[str] = []
    for comp in range(n_components):
        names.extend([f"pc{comp + 1}_t{idx:02d}" for idx in range(n_bins)])
    for i in range(projected.shape[0]):
        resampled = _resample_valid_series(projected[i], masks[i], n_bins)
        rows.append(resampled[:, :n_components].reshape(-1))
    return np.asarray(rows, dtype=np.float32), names


def build_temporal_embedding(
    features: np.ndarray,
    masks: np.ndarray,
    n_bins: int = 32,
    frame_pca_components: int = 8,
) -> np.ndarray:
    """Build a fixed-length trajectory embedding for visualization and coarse comparison."""
    projected, _ = _frame_pca_projection(features, masks, max_components=frame_pca_components)
    embeddings = [
        _resample_valid_series(projected[i], masks[i], n_bins).reshape(-1)
        for i in range(projected.shape[0])
    ]
    return np.asarray(embeddings, dtype=np.float32)


def build_frame_summary_embedding(
    features: np.ndarray,
    masks: np.ndarray,
    frame_pca_components: int = 8,
) -> tuple[np.ndarray, list[str]]:
    """Build compact summary features for supervised ranking and sparse loadings.

    These summaries intentionally mix global level and early/late dynamics so PLS can score
    a feature group without consuming the full resampled trajectory.
    """
    projected, n_components = _frame_pca_projection(features, masks, max_components=frame_pca_components)
    rows: list[list[float]] = []
    names: list[str] = []
    for comp in range(n_components):
        names.extend(
            [
                f"pc{comp + 1}_mean",
                f"pc{comp + 1}_std",
                f"pc{comp + 1}_min",
                f"pc{comp + 1}_max",
                f"pc{comp + 1}_early",
                f"pc{comp + 1}_late",
                f"pc{comp + 1}_late_minus_early",
            ]
        )

    for i in range(projected.shape[0]):
        valid = masks[i]
        x = projected[i, valid]
        if x.shape[0] == 0:
            x = np.zeros((1, n_components), dtype=np.float32)
        first = x[: max(1, x.shape[0] // 3)]
        last = x[-max(1, x.shape[0] // 3) :]
        row: list[float] = []
        for comp in range(n_components):
            s = x[:, comp]
            early = float(first[:, comp].mean())
            late = float(last[:, comp].mean())
            row.extend(
                [
                    float(s.mean()),
                    float(s.std()),
                    float(s.min()),
                    float(s.max()),
                    early,
                    late,
                    late - early,
                ]
            )
        rows.append(row)
    return np.asarray(rows, dtype=np.float32), names


def build_temporal_descriptors(
    features: np.ndarray,
    masks: np.ndarray,
    frame_pca_components: int = 8,
) -> tuple[np.ndarray, list[str]]:
    """Build interpretable descriptors: level, spread, trend, local dynamics, and spectrum."""
    projected, n_components = _frame_pca_projection(features, masks, max_components=frame_pca_components)
    names: list[str] = []
    values: list[list[float]] = []

    for i in range(projected.shape[0]):
        valid = masks[i]
        x = projected[i, valid]
        row: list[float] = []
        if x.shape[0] == 0:
            x = np.zeros((1, n_components), dtype=np.float32)

        t = np.linspace(0.0, 1.0, x.shape[0], dtype=np.float32)
        centered_t = t - t.mean()
        denom = float(np.sum(centered_t**2)) + 1e-8
        diff = np.diff(x, axis=0) if x.shape[0] > 1 else np.zeros((1, n_components), dtype=np.float32)

        spectrum = np.abs(np.fft.rfft(x - x.mean(axis=0, keepdims=True), axis=0)) ** 2
        if spectrum.shape[0] > 1:
            spectrum = spectrum[1:]
        third = max(1, spectrum.shape[0] // 3)
        bands = [
            spectrum[:third],
            spectrum[third : 2 * third],
            spectrum[2 * third :],
        ]

        for comp in range(n_components):
            series = x[:, comp]
            q25, q50, q75 = np.percentile(series, [25, 50, 75])
            slope = float(np.sum(centered_t * (series - series.mean())) / denom)
            comp_values = [
                float(series.mean()),
                float(series.std()),
                float(series.min()),
                float(series.max()),
                float(q25),
                float(q50),
                float(q75),
                slope,
                float(np.mean(np.abs(diff[:, comp]))),
                float(np.std(diff[:, comp])),
            ]
            for band in bands:
                comp_values.append(float(np.mean(band[:, comp])) if band.size else 0.0)
            row.extend(comp_values)

        values.append(row)

    per_component = [
        "mean",
        "std",
        "min",
        "max",
        "q25",
        "median",
        "q75",
        "slope",
        "abs_delta_mean",
        "delta_std",
        "fft_low",
        "fft_mid",
        "fft_high",
    ]
    for comp in range(n_components):
        names.extend([f"pc{comp + 1}_{name}" for name in per_component])
    return np.asarray(values, dtype=np.float32), names


def save_descriptor_correlations(
    descriptors: np.ndarray,
    descriptor_names: list[str],
    labels: np.ndarray,
    output_path: Path,
    top_k: int = 30,
) -> pd.DataFrame:
    """Save per-descriptor Pearson correlation tables for quick manual inspection."""
    rows: list[dict[str, Any]] = []
    for label_idx, label_name in enumerate(LABEL_NAMES):
        y = labels[:, label_idx]
        for feat_idx, descriptor_name in enumerate(descriptor_names):
            corr = _safe_corr(descriptors[:, feat_idx], y)
            rows.append(
                {
                    "label": label_name,
                    "descriptor": descriptor_name,
                    "pearson_r": corr,
                    "abs_pearson_r": abs(corr) if np.isfinite(corr) else np.nan,
                }
            )
    df = pd.DataFrame(rows).sort_values(["label", "abs_pearson_r"], ascending=[True, False])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    df.groupby("label").head(top_k).to_csv(output_path.with_name(output_path.stem + "_top.csv"), index=False)
    return df


def run_pls_probe(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    n_splits: int = 5,
    max_components: int = 3,
) -> pd.DataFrame:
    """Score each feature group with a lightweight supervised probe.

    Workflow:
    1. Learn low-dimensional supervised factors with PLS.
    2. Fit a logistic regression on the PLS scores in cross-validation.
    3. Report AUROC/F1 so feature ranking is tied to predictive utility, not only correlation.
    """
    rows: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for label_idx, label_name in enumerate(LABEL_NAMES):
        x, y = _sanitize_for_label(feature_matrix, labels[:, label_idx])
        if len(y) < max(20, n_splits) or len(np.unique(y)) < 2:
            rows.append(
                {
                    "label": label_name,
                    "n_samples": len(y),
                    "positive_ratio": float(y.mean()) if len(y) else np.nan,
                    "pls_score_corr": np.nan,
                    "cv_auroc": np.nan,
                    "cv_f1": np.nan,
                    "n_components": np.nan,
                }
            )
            continue

        n_components = min(max_components, x.shape[1], len(y) - 1)
        pls = PLSRegression(n_components=n_components, scale=True)
        pls.fit(x, y)
        score = pls.transform(x)[:, 0]
        corr = _safe_corr(score, y)

        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_aurocs: list[float] = []
        fold_f1s: list[float] = []
        for train_idx, test_idx in splitter.split(x, y.astype(int)):
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            inner_components = min(n_components, x_train.shape[1], len(y_train) - 1)
            inner_pls = PLSRegression(n_components=inner_components, scale=True)
            inner_pls.fit(x_train, y_train)
            train_score = inner_pls.transform(x_train)
            test_score = inner_pls.transform(x_test)
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
            clf.fit(train_score, y_train.astype(int))
            prob = clf.predict_proba(test_score)[:, 1]
            pred = (prob >= 0.5).astype(int)
            fold_aurocs.append(_safe_auroc(y_test, prob))
            fold_f1s.append(_safe_f1(y_test, pred))

        rows.append(
            {
                "label": label_name,
                "n_samples": len(y),
                "positive_ratio": float(y.mean()),
                "pls_score_corr": corr,
                "cv_auroc": float(np.nanmean(fold_aurocs)),
                "cv_f1": float(np.nanmean(fold_f1s)),
                "n_components": n_components,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def save_sparse_pca_contributions(
    feature_matrix: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    n_components: int = 3,
    top_k: int = 20,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Expose which summary dimensions dominate the strongest feature groups.

    SparsePCA is used here as an interpretable loading extractor rather than a classifier.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if feature_matrix.shape[0] < 5 or feature_matrix.shape[1] < 2:
        df = pd.DataFrame(columns=["component", "feature", "loading", "abs_loading"])
        df.to_csv(output_path, index=False)
        return df

    scaler = StandardScaler()
    x = scaler.fit_transform(feature_matrix)
    n_components = min(n_components, x.shape[1], max(1, x.shape[0] - 1))
    model = SparsePCA(n_components=n_components, alpha=alpha, random_state=42, max_iter=1000)
    model.fit(x)

    rows: list[dict[str, Any]] = []
    for comp_idx, comp in enumerate(model.components_, start=1):
        order = np.argsort(np.abs(comp))[::-1][:top_k]
        for idx in order:
            rows.append(
                {
                    "component": comp_idx,
                    "feature": feature_names[idx],
                    "loading": float(comp[idx]),
                    "abs_loading": float(abs(comp[idx])),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def run_supervised_pca(
    feature_matrix: np.ndarray,
    feature_names: list[str],
    labels: np.ndarray,
    output_prefix: Path,
    top_k_features: int = 20,
    n_components: int = 2,
) -> pd.DataFrame:
    """Select label-related features first, then run PCA on the selected subset.

    This is a practical supervised PCA variant for feature-group inspection:
    1. rank summary dimensions by absolute Pearson correlation with the label
    2. keep the top-k ranked dimensions
    3. run PCA only on the retained dimensions
    """
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    for label_idx, label_name in enumerate(LABEL_NAMES):
        x, y = _sanitize_for_label(feature_matrix, labels[:, label_idx])
        if len(y) < 10 or len(np.unique(y)) < 2:
            summary_rows.append(
                {
                    "label": label_name,
                    "n_selected": 0,
                    "selected_features": "",
                    "pc1_var_ratio": np.nan,
                    "pc2_var_ratio": np.nan,
                    "pc1_label_corr": np.nan,
                    "pc2_label_corr": np.nan,
                }
            )
            continue

        corrs = np.array([abs(_safe_corr(x[:, j], y)) for j in range(x.shape[1])], dtype=np.float64)
        corrs = np.nan_to_num(corrs, nan=-1.0)
        order = np.argsort(corrs)[::-1]
        keep = order[: min(top_k_features, x.shape[1])]
        selected_feature_names = [feature_names[idx] for idx in keep]
        selected_scores = corrs[keep]

        selected_df = pd.DataFrame(
            {
                "feature": selected_feature_names,
                "selection_score_abs_pearson": selected_scores,
                "rank": np.arange(1, len(keep) + 1),
            }
        )
        selected_df.to_csv(output_prefix.with_name(f"{output_prefix.name}_{label_name}_selected_features.csv"), index=False)

        x_sel = x[:, keep]
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_sel)
        pca_dim = min(n_components, x_scaled.shape[1], x_scaled.shape[0] - 1)
        if pca_dim < 1:
            summary_rows.append(
                {
                    "label": label_name,
                    "n_selected": len(keep),
                    "selected_features": ",".join(selected_feature_names),
                    "pc1_var_ratio": np.nan,
                    "pc2_var_ratio": np.nan,
                    "pc1_label_corr": np.nan,
                    "pc2_label_corr": np.nan,
                }
            )
            continue

        pca = PCA(n_components=pca_dim, random_state=42)
        scores = pca.fit_transform(x_scaled)
        if scores.shape[1] == 1:
            scores = np.concatenate([scores, np.zeros((scores.shape[0], 1), dtype=scores.dtype)], axis=1)
            var_ratio = [float(pca.explained_variance_ratio_[0]), 0.0]
        else:
            var_ratio = [float(v) for v in pca.explained_variance_ratio_[:2]]
            if len(var_ratio) < 2:
                var_ratio.append(0.0)

        score_df = pd.DataFrame(
            {
                "spca1": scores[:, 0],
                "spca2": scores[:, 1],
                "label": y,
            }
        )
        score_df.to_csv(output_prefix.with_name(f"{output_prefix.name}_{label_name}_scores.csv"), index=False)

        plt.figure(figsize=(9, 7))
        scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y, cmap="viridis", alpha=0.75, s=24)
        plt.colorbar(scatter, label=label_name)
        plt.title(f"{output_prefix.name} - Supervised PCA colored by {label_name}")
        plt.xlabel(f"SPCA 1 ({var_ratio[0]:.2%} var)")
        plt.ylabel(f"SPCA 2 ({var_ratio[1]:.2%} var)")
        plt.tight_layout()
        plt.savefig(output_prefix.with_name(f"{output_prefix.name}_{label_name}.png"), dpi=180)
        plt.close()

        summary_rows.append(
            {
                "label": label_name,
                "n_selected": len(keep),
                "selected_features": ",".join(selected_feature_names),
                "pc1_var_ratio": var_ratio[0],
                "pc2_var_ratio": var_ratio[1],
                "pc1_label_corr": _safe_corr(scores[:, 0], y),
                "pc2_label_corr": _safe_corr(scores[:, 1], y),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_prefix.with_name(f"{output_prefix.name}_summary.csv"), index=False)
    return summary_df


def run_functional_pca(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    output_prefix: Path,
    title_prefix: str,
    n_components: int = 5,
) -> pd.DataFrame:
    """Run practical FPCA/MFPCA-style analysis on resampled functional curves."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    if feature_matrix.shape[0] < 10 or feature_matrix.shape[1] < 2:
        df = pd.DataFrame(columns=["label", "pc1_corr", "pc2_corr", "pc1_var_ratio", "pc2_var_ratio"])
        df.to_csv(output_prefix.with_name(f"{output_prefix.name}_summary.csv"), index=False)
        return df

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(feature_matrix)
    pca_dim = min(n_components, x_scaled.shape[1], x_scaled.shape[0] - 1)
    pca = PCA(n_components=pca_dim, random_state=42)
    scores = pca.fit_transform(x_scaled)
    if scores.shape[1] == 1:
        scores = np.concatenate([scores, np.zeros((scores.shape[0], 1), dtype=scores.dtype)], axis=1)
        var_ratio = [float(pca.explained_variance_ratio_[0]), 0.0]
    else:
        var_ratio = [float(v) for v in pca.explained_variance_ratio_[:2]]
        if len(var_ratio) < 2:
            var_ratio.append(0.0)

    pd.DataFrame({"fpc1": scores[:, 0], "fpc2": scores[:, 1]}).to_csv(
        output_prefix.with_name(f"{output_prefix.name}_scores.csv"),
        index=False,
    )

    rows: list[dict[str, Any]] = []
    for label_idx, label_name in enumerate(LABEL_NAMES):
        x_lab, y = _sanitize_for_label(scores, labels[:, label_idx])
        if len(y) < 10 or len(np.unique(y)) < 2:
            rows.append(
                {
                    "label": label_name,
                    "pc1_corr": np.nan,
                    "pc2_corr": np.nan,
                    "pc1_var_ratio": var_ratio[0],
                    "pc2_var_ratio": var_ratio[1],
                }
            )
            continue

        plt.figure(figsize=(9, 7))
        scatter = plt.scatter(x_lab[:, 0], x_lab[:, 1], c=y, cmap="viridis", alpha=0.75, s=24)
        plt.colorbar(scatter, label=label_name)
        plt.title(f"{title_prefix} colored by {label_name}")
        plt.xlabel(f"FPC 1 ({var_ratio[0]:.2%} var)")
        plt.ylabel(f"FPC 2 ({var_ratio[1]:.2%} var)")
        plt.tight_layout()
        plt.savefig(output_prefix.with_name(f"{output_prefix.name}_{label_name}.png"), dpi=180)
        plt.close()

        rows.append(
            {
                "label": label_name,
                "pc1_corr": _safe_corr(x_lab[:, 0], y),
                "pc2_corr": _safe_corr(x_lab[:, 1], y),
                "pc1_var_ratio": var_ratio[0],
                "pc2_var_ratio": var_ratio[1],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_prefix.with_name(f"{output_prefix.name}_summary.csv"), index=False)
    return df


def visualize_pls_embedding(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    output_prefix: Path,
    title_prefix: str,
    max_components: int = 2,
) -> None:
    """Project summary features to supervised PLS space and save 2D scatter plots."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for label_idx, label_name in enumerate(LABEL_NAMES):
        x, y = _sanitize_for_label(feature_matrix, labels[:, label_idx])
        if len(y) < 10 or len(np.unique(y)) < 2:
            continue
        n_components = min(max_components, x.shape[1], len(y) - 1)
        if n_components < 1:
            continue
        pls = PLSRegression(n_components=n_components, scale=True)
        pls.fit(x, y)
        coords = pls.transform(x)
        if coords.shape[1] == 1:
            coords = np.concatenate([coords, np.zeros((coords.shape[0], 1), dtype=coords.dtype)], axis=1)

        plt.figure(figsize=(9, 7))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=y, cmap="viridis", alpha=0.75, s=24)
        plt.colorbar(scatter, label=label_name)
        plt.title(f"{title_prefix} - PLS colored by {label_name}")
        plt.xlabel("PLS 1")
        plt.ylabel("PLS 2")
        plt.tight_layout()
        plt.savefig(output_prefix.with_name(f"{output_prefix.name}_pls_{label_name}.png"), dpi=180)
        plt.close()


def visualize_lda_embedding(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    output_prefix: Path,
    title_prefix: str,
) -> None:
    """Project summary features to label-aware LDA space and save plots in a separate folder."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for label_idx, label_name in enumerate(LABEL_NAMES):
        x, y = _sanitize_for_label(feature_matrix, labels[:, label_idx])
        y_int = y.astype(int)
        unique = np.unique(y_int)
        if len(y_int) < 10 or len(unique) < 2:
            continue

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        n_components = min(2, len(unique) - 1, x_scaled.shape[1])
        if n_components < 1:
            continue

        lda = LinearDiscriminantAnalysis(n_components=n_components)
        coords = lda.fit_transform(x_scaled, y_int)
        if coords.ndim == 1:
            coords = coords[:, None]
        if coords.shape[1] == 1:
            coords = np.concatenate([coords, np.zeros((coords.shape[0], 1), dtype=coords.dtype)], axis=1)

        pd.DataFrame(
            {
                "lda1": coords[:, 0],
                "lda2": coords[:, 1],
                "label": y_int,
            }
        ).to_csv(output_prefix.with_name(f"{output_prefix.name}_{label_name}_scores.csv"), index=False)

        plt.figure(figsize=(9, 7))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=y_int, cmap="viridis", alpha=0.75, s=24)
        plt.colorbar(scatter, label=label_name)
        plt.title(f"{title_prefix} - LDA colored by {label_name}")
        plt.xlabel("LDA 1")
        plt.ylabel("LDA 2")
        plt.tight_layout()
        plt.savefig(output_prefix.with_name(f"{output_prefix.name}_{label_name}.png"), dpi=180)
        plt.close()


def visualize_umap_embedding(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    output_prefix: Path,
    title_prefix: str,
    supervised: bool = False,
) -> None:
    """Project summary features with UMAP or supervised UMAP into 2D."""
    if umap is None:
        return

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for label_idx, label_name in enumerate(LABEL_NAMES):
        x, y = _sanitize_for_label(feature_matrix, labels[:, label_idx])
        y_int = y.astype(int)
        unique = np.unique(y_int)
        if len(y_int) < 10:
            continue

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        perplexity_like = max(5, min(30, len(y_int) // 20))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=perplexity_like,
            min_dist=0.15,
            metric="euclidean",
            random_state=42,
        )

        if supervised and len(unique) >= 2:
            coords = reducer.fit_transform(x_scaled, y_int)
            mode_name = "Supervised UMAP"
            stem = f"{output_prefix.name}_supervised"
        else:
            coords = reducer.fit_transform(x_scaled)
            mode_name = "UMAP"
            stem = output_prefix.name

        pd.DataFrame(
            {
                "umap1": coords[:, 0],
                "umap2": coords[:, 1],
                "label": y_int,
            }
        ).to_csv(output_prefix.with_name(f"{stem}_{label_name}_scores.csv"), index=False)

        plt.figure(figsize=(9, 7))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=y_int, cmap="viridis", alpha=0.75, s=24)
        plt.colorbar(scatter, label=label_name)
        plt.title(f"{title_prefix} - {mode_name} colored by {label_name}")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.tight_layout()
        plt.savefig(output_prefix.with_name(f"{stem}_{label_name}.png"), dpi=180)
        plt.close()


def plot_sparse_pca_contributions(contrib_df: pd.DataFrame, output_prefix: Path, title_prefix: str) -> None:
    """Visualize SparsePCA loadings with per-component bars and a component-feature heatmap."""
    if contrib_df.empty:
        return
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    for component, group in contrib_df.groupby("component"):
        ordered = group.sort_values("abs_loading", ascending=False).head(15)
        plt.figure(figsize=(10, 6))
        colors = ["#c44e52" if v < 0 else "#4c72b0" for v in ordered["loading"]]
        plt.barh(ordered["feature"], ordered["loading"], color=colors)
        plt.gca().invert_yaxis()
        plt.title(f"{title_prefix} SparsePCA Component {component} top loadings")
        plt.xlabel("Loading")
        plt.tight_layout()
        plt.savefig(output_prefix.with_name(f"{output_prefix.name}_component{component}_bar.png"), dpi=180)
        plt.close()

    heatmap = (
        contrib_df.sort_values(["component", "abs_loading"], ascending=[True, False])
        .pivot_table(index="component", columns="feature", values="loading", fill_value=0.0)
    )
    if heatmap.shape[0] and heatmap.shape[1]:
        plt.figure(figsize=(max(10, heatmap.shape[1] * 0.35), 4 + heatmap.shape[0] * 0.6))
        im = plt.imshow(heatmap.values, aspect="auto", interpolation="nearest", cmap="coolwarm")
        plt.colorbar(im, label="SparsePCA loading")
        plt.yticks(range(len(heatmap.index)), [f"C{idx}" for idx in heatmap.index])
        plt.xticks(range(len(heatmap.columns)), heatmap.columns, rotation=75, ha="right")
        plt.title(f"{title_prefix} SparsePCA loading heatmap")
        plt.tight_layout()
        plt.savefig(output_prefix.with_name(f"{output_prefix.name}_heatmap.png"), dpi=180)
        plt.close()


def update_summary_rows(
    tag: str,
    descriptor_corr_df: pd.DataFrame,
    pls_probe_df: pd.DataFrame,
) -> None:
    """Append one summary row per label so all feature groups can be ranked together."""
    for label_name in LABEL_NAMES:
        corr_group = descriptor_corr_df[descriptor_corr_df["label"] == label_name].dropna(subset=["abs_pearson_r"])
        probe_group = pls_probe_df[pls_probe_df["label"] == label_name]
        best_corr = corr_group.iloc[0] if len(corr_group) else None
        probe = probe_group.iloc[0] if len(probe_group) else None
        SUMMARY_ROWS.append(
            {
                "feature": tag,
                "label": label_name,
                "best_descriptor": None if best_corr is None else best_corr["descriptor"],
                "best_descriptor_r": np.nan if best_corr is None else best_corr["pearson_r"],
                "best_descriptor_abs_r": np.nan if best_corr is None else best_corr["abs_pearson_r"],
                "pls_score_corr": np.nan if probe is None else probe["pls_score_corr"],
                "pls_cv_auroc": np.nan if probe is None else probe["cv_auroc"],
                "pls_cv_f1": np.nan if probe is None else probe["cv_f1"],
                "n_samples": np.nan if probe is None else probe["n_samples"],
            }
        )


def save_global_summary() -> None:
    """Save two ranking tables: per-label detail and per-feature averages."""
    if not SUMMARY_ROWS:
        return
    df = pd.DataFrame(SUMMARY_ROWS)
    df = df.sort_values(["label", "pls_cv_auroc", "best_descriptor_abs_r"], ascending=[True, False, False])
    STAGE1_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(STAGE1_DIR / "feature_group_summary.csv", index=False)

    avg = (
        df.groupby("feature", as_index=False)
        .agg(
            mean_pls_cv_auroc=("pls_cv_auroc", "mean"),
            mean_pls_cv_f1=("pls_cv_f1", "mean"),
            mean_best_descriptor_abs_r=("best_descriptor_abs_r", "mean"),
            max_best_descriptor_abs_r=("best_descriptor_abs_r", "max"),
        )
        .sort_values(["mean_pls_cv_auroc", "mean_best_descriptor_abs_r"], ascending=[False, False])
    )
    avg.to_csv(STAGE1_DIR / "feature_group_summary_by_feature.csv", index=False)

    for metric in ("mean_pls_cv_auroc", "mean_pls_cv_f1", "mean_best_descriptor_abs_r"):
        ordered = avg.sort_values(metric, ascending=False)
        plt.figure(figsize=(10, max(5, 0.55 * len(ordered))))
        plt.barh(ordered["feature"], ordered[metric], color="#4c72b0")
        plt.gca().invert_yaxis()
        plt.title(f"Feature Group Ranking by {metric}")
        plt.xlabel(metric)
        plt.tight_layout()
        plt.savefig(STAGE1_DIR / f"feature_group_ranking_{metric}.png", dpi=180)
        plt.close()

    for label_name, group in df.groupby("label"):
        ordered = group.sort_values("pls_cv_auroc", ascending=False)
        plt.figure(figsize=(10, max(5, 0.55 * len(ordered))))
        plt.barh(ordered["feature"], ordered["pls_cv_auroc"], color="#55a868")
        plt.gca().invert_yaxis()
        plt.title(f"Feature Group Ranking by PLS AUROC ({label_name})")
        plt.xlabel("pls_cv_auroc")
        plt.tight_layout()
        plt.savefig(STAGE1_DIR / f"feature_group_ranking_pls_cv_auroc_{label_name}.png", dpi=180)
        plt.close()


def visualize_embedding(
    embedding: np.ndarray,
    labels: np.ndarray,
    method: str,
    output_prefix: Path,
    title_prefix: str,
) -> None:
    """Project fixed-length temporal embeddings to 2D for qualitative inspection."""
    if embedding.shape[0] < 3:
        return
    scaled = StandardScaler().fit_transform(embedding)
    if method == "pca":
        reducer = PCA(n_components=2, random_state=42)
    elif method == "tsne":
        perplexity = max(2, min(30, embedding.shape[0] // 3))
        reducer = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", perplexity=perplexity)
    else:
        raise ValueError("Unsupported method: choose 'pca' or 'tsne'")

    reduced = reducer.fit_transform(scaled)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for label_idx, label_name in enumerate(LABEL_NAMES):
        target_label = labels[:, label_idx]
        plt.figure(figsize=(9, 7))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=target_label, cmap="viridis", alpha=0.75, s=24)
        plt.colorbar(scatter, label=label_name)
        plt.title(f"{title_prefix} - {method.upper()} colored by {label_name}")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.tight_layout()
        plt.savefig(output_prefix.with_name(f"{output_prefix.name}_{method}_{label_name}.png"), dpi=180)
        plt.close()


def plot_label_sorted_temporal_profile(
    features: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    output_prefix: Path,
    title_prefix: str,
    n_bins: int = 64,
    clip_percentiles: tuple[float, float] = (2.0, 98.0),
) -> None:
    """Plot time-normalized trajectories sorted by label to inspect coarse temporal structure."""
    projected, _ = _frame_pca_projection(features, masks, max_components=1)
    profiles = np.asarray(
        [_resample_valid_series(projected[i], masks[i], n_bins)[:, 0] for i in range(projected.shape[0])],
        dtype=np.float32,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for label_idx, label_name in enumerate(LABEL_NAMES):
        label_values = labels[:, label_idx]
        valid_label = np.isfinite(label_values) & (label_values >= 0)
        valid_order = np.where(valid_label)[0][np.argsort(label_values[valid_label])]
        invalid_order = np.where(~valid_label)[0]
        order = np.concatenate([valid_order, invalid_order])

        finite_values = profiles[np.isfinite(profiles)]
        if finite_values.size:
            vmin, vmax = np.percentile(finite_values, clip_percentiles)
            if np.isclose(vmin, vmax):
                vmin, vmax = float(finite_values.min()), float(finite_values.max())
        else:
            vmin, vmax = None, None

        fig, (ax_heat, ax_label) = plt.subplots(
            ncols=2,
            figsize=(11, 7),
            gridspec_kw={"width_ratios": [20, 1.6], "wspace": 0.08},
        )
        im = ax_heat.imshow(
            profiles[order],
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
        )
        ax_heat.set_title(f"{title_prefix} temporal profile sorted by {label_name}")
        ax_heat.set_xlabel("Normalized time")
        ax_heat.set_ylabel("Samples sorted by label")

        sorted_labels = label_values[order].reshape(-1, 1)
        label_unique = np.unique(label_values[valid_label])
        if len(label_unique) <= 2:
            label_cmap = ListedColormap(["#d9d9d9", "#2b6cb0"])
            label_norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5], ncolors=label_cmap.N)
            ax_label.imshow(
                sorted_labels,
                aspect="auto",
                interpolation="nearest",
                cmap=label_cmap,
                norm=label_norm,
            )
            ax_label.set_yticks([])
            ax_label.set_xticks([0])
            ax_label.set_xticklabels([label_name])
            ax_label.text(
                0.5,
                -0.02,
                "0 / 1",
                transform=ax_label.transAxes,
                ha="center",
                va="top",
                fontsize=9,
            )
        else:
            ax_label.imshow(
                sorted_labels,
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
            )
            ax_label.set_yticks([])
            ax_label.set_xticks([0])
            ax_label.set_xticklabels([label_name])
        ax_label.set_title(label_name)
        fig.colorbar(
            im,
            ax=ax_heat,
            label=f"Frame-PC1 value, p{clip_percentiles[0]:g}-p{clip_percentiles[1]:g} clipped",
        )
        fig.subplots_adjust(left=0.08, right=0.92, top=0.9, bottom=0.12, wspace=0.08)
        plt.savefig(output_prefix.with_name(f"{output_prefix.name}_profile_{label_name}.png"), dpi=180)
        plt.close(fig)

        plt.figure(figsize=(10, 7))
        plt.imshow(profiles[order], aspect="auto", interpolation="nearest", cmap="coolwarm")
        plt.colorbar(label="Frame-PC1 value")
        plt.title(f"{title_prefix} temporal profile sorted by {label_name} (raw scale)")
        plt.xlabel("Normalized time")
        plt.ylabel("Samples sorted by label")
        plt.subplots_adjust(left=0.08, right=0.92, top=0.9, bottom=0.12)
        plt.savefig(output_prefix.with_name(f"{output_prefix.name}_profile_{label_name}_raw_scale.png"), dpi=180)
        plt.close()


def analyze_feature_group(
    features_group: dict[str, np.ndarray],
    masks_group: dict[str, np.ndarray],
    labels: np.ndarray,
    metadata: pd.DataFrame,
    modality_name: str,
    session_name: int,
    sparse_pca_top_feature_count: int = 3,
    functional_top_feature_count: int = 3,
) -> None:
    """Run the full analysis stack for one modality/session.

    Per feature group this produces:
    - temporal embeddings for PCA/t-SNE plots
    - interpretable descriptors and descriptor-label correlations
    - supervised PLS probe scores
    - optional SparsePCA loadings for the strongest groups
    """
    # STAGE1_DIR.mkdir(parents=True, exist_ok=True)
    # LDA_DIR.mkdir(parents=True, exist_ok=True)
    UMAP_DIR.mkdir(parents=True, exist_ok=True)
    # STAGE2_DIR.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(STAGE1_DIR / f"{modality_name}_S{session_name}_metadata.csv", index=False)

    for feature_name, features in features_group.items():
        masks = masks_group[feature_name]
        if labels.shape[0] != features.shape[0]:
            raise RuntimeError(f"Label count mismatch for {modality_name}/{feature_name}")

        tag = f"{modality_name}_{feature_name}_S{session_name}"
        print(f"Analyzing {tag}: N={features.shape[0]}, T={features.shape[1]}, D={features.shape[2]}")

        # Three complementary views of the same feature group:
        # 1) dense trajectory embedding for plots,
        # 2) compact summary embedding for supervised probes,
        # 3) interpretable descriptors for correlation-based inspection,
        # 4) supervised PCA for label-aware prescreening.
        embedding = build_temporal_embedding(features, masks)
        summary_embedding, summary_feature_names = build_frame_summary_embedding(features, masks)
        descriptors, descriptor_names = build_temporal_descriptors(features, masks)
        # pd.DataFrame(embedding).to_csv(STAGE1_DIR / f"{tag}_temporal_embedding.csv", index=False)
        # pd.DataFrame(summary_embedding, columns=summary_feature_names).to_csv(
        #     STAGE1_DIR / f"{tag}_summary_embedding.csv", index=False
        # )
        # pd.DataFrame(descriptors, columns=descriptor_names).to_csv(STAGE1_DIR / f"{tag}_descriptors.csv", index=False)
        # descriptor_corr_df = save_descriptor_correlations(
        #     descriptors,
        #     descriptor_names,
        #     labels,
        #     STAGE1_DIR / f"{tag}_descriptor_label_correlations.csv",
        # )
        # pls_probe_df = run_pls_probe(
        #     summary_embedding,
        #     labels,
        #     STAGE1_DIR / f"{tag}_pls_probe.csv",
        # )
        # update_summary_rows(tag, descriptor_corr_df, pls_probe_df)

        # visualize_embedding(embedding, labels, "pca", STAGE1_DIR / tag, tag)
        # visualize_embedding(embedding, labels, "tsne", STAGE1_DIR / tag, tag)
        # visualize_pls_embedding(summary_embedding, labels, STAGE1_DIR / tag, tag)
        # visualize_lda_embedding(summary_embedding, labels, LDA_DIR / tag, tag)
        visualize_umap_embedding(summary_embedding, labels, UMAP_DIR / f"{tag}_umap", tag, supervised=False)
        visualize_umap_embedding(summary_embedding, labels, UMAP_DIR / f"{tag}_umap", tag, supervised=True)
        # run_supervised_pca(
        #     summary_embedding,
        #     summary_feature_names,
        #     labels,
        #     STAGE1_DIR / f"{tag}_supervised_pca",
        # )
        # plot_label_sorted_temporal_profile(features, masks, labels, STAGE1_DIR / tag, tag)

    current_rows = pd.DataFrame(
        [r for r in SUMMARY_ROWS if r["feature"].startswith(f"{modality_name}_") and f"_S{session_name}" in r["feature"]]
    )
    # if len(current_rows):
    #     # Second-stage functional analysis and SparsePCA are only run on the strongest groups.
    #     top_features = (
    #         current_rows.groupby("feature", as_index=False)["pls_cv_auroc"]
    #         .mean()
    #         .sort_values("pls_cv_auroc", ascending=False)
    #         .head(max(sparse_pca_top_feature_count, functional_top_feature_count))["feature"]
    #         .tolist()
    #     )
        # for rank_idx, tag in enumerate(top_features):
        #     feature_name = tag.split("_", 1)[1].rsplit(f"_S{session_name}", 1)[0]
        #     # if rank_idx < functional_top_feature_count:
        #         # fpca_matrix, _ = build_functional_matrix(
        #         #     features_group[feature_name],
        #         #     masks_group[feature_name],
        #         #     robust=False,
        #         #     multivariate=False,
        #         # )
        #         # run_functional_pca(
        #         #     fpca_matrix,
        #         #     labels,
        #         #     STAGE2_DIR / f"{tag}_fpca",
        #         #     f"{tag} FPCA",
        #         # )

        #         # robust_fpca_matrix, _ = build_functional_matrix(
        #         #     features_group[feature_name],
        #         #     masks_group[feature_name],
        #         #     robust=True,
        #         #     multivariate=False,
        #         # )
        #         # run_functional_pca(
        #         #     robust_fpca_matrix,
        #         #     labels,
        #         #     STAGE2_DIR / f"{tag}_robust_fpca",
        #         #     f"{tag} Robust FPCA",
        #         # )

        #         # mfpca_matrix, _ = build_functional_matrix(
        #         #     features_group[feature_name],
        #         #     masks_group[feature_name],
        #         #     robust=True,
        #         #     multivariate=True,
        #         # )
        #         # run_functional_pca(
        #         #     mfpca_matrix,
        #         #     labels,
        #         #     STAGE2_DIR / f"{tag}_mfpca",
        #         #     f"{tag} MFPCA-style",
        #         # )

        #     if rank_idx < sparse_pca_top_feature_count:
        #         feature_matrix, feature_names = build_frame_summary_embedding(
        #             features_group[feature_name],
        #             masks_group[feature_name],
        #         )
        #         # contrib_df = save_sparse_pca_contributions(
        #         #     feature_matrix,
        #         #     feature_names,
        #         #     STAGE1_DIR / f"{tag}_sparse_pca_contributions.csv",
        #         # )
        #         # plot_sparse_pca_contributions(contrib_df, STAGE1_DIR / f"{tag}_sparse_pca", tag)


def main() -> None:
    """Entry point for one-session grouped feature analysis."""
    args = parse_args()
    cfg = load_config(args)

    device_str = cfg.get("device")
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    manifest_dir = Path(cfg.get("manifest_dir", "/home/zhouyue/AdodasChallange/AdoDAS2026-SEUCS/outputs"))

    migrate_legacy_results()

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

    print("Loading datasets...")
    train_ds = GroupedParticipantDataset(
        manifest_dir / "train.csv",
        feat_cfg,
        split="train",
        session_drop_prob=0.1,
    )

    # val_ds = GroupedParticipantDataset(manifest_dir / "val.csv", feat_cfg, split="val")
    dims = train_ds.feature_dims
    audio_group_dims = {name: dims[name] for name in feat_cfg.audio_sequence_features if name in dims}
    audio_pooled_group_dims = {n: dims[n] for n in feat_cfg.audio_pooled_features if n in dims}
    video_group_dims = {name: dims[name] for name in feat_cfg.video_features if name in dims}

    # 提取并可视化特征
<<<<<<< Updated upstream
    target_session = 2
    print("Extracting features...")
    audio_features, labels = extract_features(train_ds, audio_group_dims, target_session, mode="audio")
    visualize_features(audio_features, labels, modality_name="audio", session_name=target_session)    
    video_features, labels = extract_features(train_ds, video_group_dims, target_session, mode="video")
    visualize_features(video_features, labels, modality_name="video", session_name=target_session)
=======
    target_session = int(cfg.get("analysis_session", 2))
    print(f"Extracting session index {target_session} features...")

    audio_features, audio_masks, labels, audio_meta = extract_features(
        train_ds, audio_group_dims, target_session, mode="audio"
    )
    analyze_feature_group(audio_features, audio_masks, labels, audio_meta, "audio", target_session)

    video_features, video_masks, labels, video_meta = extract_features(
        train_ds, video_group_dims, target_session, mode="video"
    )
    analyze_feature_group(video_features, video_masks, labels, video_meta, "video", target_session)

    save_global_summary()
    print(f"Saved feature analysis outputs to {RESULT_DIR}")
>>>>>>> Stashed changes


if __name__ == "__main__":
    main()
