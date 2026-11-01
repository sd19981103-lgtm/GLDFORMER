# -*- coding: utf-8 -*-
"""GLDformer revision based on the supplied manuscript and original script.

Inputs: data_x{1,4,6}.npy [N, windows, features] and data_y{1,4,6}.npy
[N] or [N,1]. The default protocol is the manuscript's 80/20 split within
each development cutter. This revision does NOT contain verified paper results.
Run ``python gldformer.py --help``; see README.md and PAPER_ALIGNMENT.md.
"""
from __future__ import annotations

import os

# Set thread/CUDA settings before importing NumPy or PyTorch. Do not suppress
# duplicate OpenMP library errors with KMP_DUPLICATE_LIB_OK.
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import hashlib
import json
import logging
import math
import platform
import random
import sys
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_VERSION = "2.0.0-review"
LOGGER = logging.getLogger("gldformer")
METRICS = ("MAE", "RMSE", "R2")
VARIANTS = ("full", "no_local", "no_global", "no_trend", "no_tokenizer")
SENSOR_CONFIGS = ("all", "no_force", "no_vibration", "no_ae")
EXPERIMENTS = (
    {"name": "Train_C1_4_Test_C6", "train_ids": ("1", "4"), "test_id": "6"},
    {"name": "Train_C1_6_Test_C4", "train_ids": ("1", "6"), "test_id": "4"},
    {"name": "Train_C4_6_Test_C1", "train_ids": ("4", "6"), "test_id": "1"},
)


def write_json(path, obj):
    """Reject NaN/Infinity so an invalid run cannot look like a valid JSON result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_array(value, name, ndim=None):
    value = np.asarray(value)
    if not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value):
        raise ValueError(f"{name}: expected a real numeric array, got {value.dtype}")
    if value.size == 0 or (ndim is not None and value.ndim != ndim):
        raise ValueError(f"{name}: invalid/empty shape {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name}: NaN or Inf found; correct the source data first")
    return value


def set_seed(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic)
    if deterministic:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # PyTorch 2.3 fused attention backward can be nondeterministic.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


class NoisyDataset(Dataset):
    def __init__(self, X, y, noise_std=0.01, training=True, feature_mask=None):
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
        if self.X.ndim != 3 or self.y.ndim != 1 or len(self.X) != len(self.y):
            raise ValueError("Expected X[N,T,C] and y[N]")
        if not math.isfinite(noise_std) or noise_std < 0:
            raise ValueError("noise_std must be finite and nonnegative")
        self.noise_std, self.training = noise_std, training
        self.mask = None if feature_mask is None else torch.as_tensor(feature_mask, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        if self.mask is not None:
            # Apply AFTER noise so ablated sensors remain identically zero.
            x = x * self.mask
        return x, self.y[index]


def load_cutter_data(data_dir, cutter_id, expected_seq_len=100,
                     expected_input_dim=21, expected_samples=315):
    xp = Path(data_dir) / f"data_x{cutter_id}.npy"
    yp = Path(data_dir) / f"data_y{cutter_id}.npy"
    if not xp.is_file() or not yp.is_file():
        raise FileNotFoundError(f"Missing cutter {cutter_id} data: {xp}, {yp}")
    X = finite_array(np.load(xp, allow_pickle=False), str(xp), 3)
    y = finite_array(np.load(yp, allow_pickle=False), str(yp))
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1 or len(X) != len(y):
        raise ValueError(f"C{cutter_id}: expected one label per pass, X={X.shape}, y={y.shape}; "
                         "use the mean of the three flute wear values, do not flatten [N,3]")
    for actual, expected, name in ((X.shape[0], expected_samples, "passes"),
                                   (X.shape[1], expected_seq_len, "windows"),
                                   (X.shape[2], expected_input_dim, "features")):
        if expected is not None and expected > 0 and actual != expected:
            raise ValueError(f"C{cutter_id}: {name}={actual}, expected {expected}; check N,T,C order")
    X, y = X.astype(np.float32), y.astype(np.float32)
    finite_array(X, f"C{cutter_id} float32 X", 3)
    finite_array(y, f"C{cutter_id} float32 y", 1)
    return X, y


class TrainMinMaxScaler:
    """Training-only min/max. Nonconstant denominator is range + eps (Eq. 5).

    Default policy aborts for constant/near-constant training features instead
    of amplifying unseen values by 1/eps. Explicit 'unit' policy uses scale=1
    for those columns, records them, and warns; it is a documented extension.
    """
    def __init__(self, epsilon=1e-8, constant_feature_policy="error"):
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be positive and finite")
        if constant_feature_policy not in ("error", "unit"):
            raise ValueError("constant_feature_policy must be error or unit")
        self.epsilon = float(epsilon)
        self.constant_feature_policy = constant_feature_policy

    def fit(self, X, y):
        X = finite_array(X, "training features", 3).astype(np.float64)
        y = finite_array(y, "training labels", 1).astype(np.float64)
        if len(X) != len(y):
            raise ValueError("X/y count mismatch")
        self.feat_min = X.min(axis=(0, 1))
        self.feat_max = X.max(axis=(0, 1))
        raw_range = self.feat_max - self.feat_min
        self.constant_columns = np.flatnonzero(raw_range <= self.epsilon).tolist()
        if self.constant_columns and self.constant_feature_policy == "error":
            raise ValueError(f"Constant/near-constant training columns: {self.constant_columns}. "
                             "Inspect preprocessing; explicitly use --constant-feature-policy unit "
                             "only if unit scaling is intended.")
        self.feat_scale = raw_range + self.epsilon
        if self.constant_columns:
            self.feat_scale[self.constant_columns] = 1.0
            warnings.warn(f"Unit scaling for constant training columns {self.constant_columns}")
        self.y_min, self.y_max = float(y.min()), float(y.max())
        if self.y_max - self.y_min <= self.epsilon:
            raise ValueError("Training wear labels are constant/near-constant; check the dataset")
        self.y_scale = self.y_max - self.y_min + self.epsilon
        return self

    def transform_X(self, X):
        X = finite_array(X, "features", 3)
        if X.shape[-1] != len(self.feat_min):
            raise ValueError("Feature count differs from fitted scaler")
        result = ((X.astype(np.float64) - self.feat_min) / self.feat_scale).astype(np.float32)
        return finite_array(result, "normalized features", 3)

    def transform_y(self, y):
        result = ((finite_array(y, "labels", 1).astype(np.float64) - self.y_min)
                  / self.y_scale).astype(np.float32)
        return finite_array(result, "normalized labels", 1)

    def inverse_y(self, y):
        # Exact same denominator as normalization, including epsilon.
        return finite_array(y, "normalized predictions", 1).astype(np.float64) * self.y_scale + self.y_min

    def to_dict(self):
        return {"epsilon": self.epsilon, "constant_feature_policy": self.constant_feature_policy,
                "feat_min": self.feat_min.tolist(), "feat_max": self.feat_max.tolist(),
                "feat_scale": self.feat_scale.tolist(), "constant_columns": self.constant_columns,
                "y_min": self.y_min, "y_max": self.y_max, "y_scale": self.y_scale}

    @classmethod
    def from_dict(cls, values):
        obj = cls(values["epsilon"], values["constant_feature_policy"])
        for key in ("feat_min", "feat_max", "feat_scale"):
            setattr(obj, key, np.asarray(values[key], dtype=np.float64))
        for key in ("constant_columns", "y_min", "y_max", "y_scale"):
            setattr(obj, key, values[key])
        return obj


def prepare_development(data_dir, train_ids, test_id, *, train_fraction=0.8,
                        expected_seq_len=100, expected_input_dim=21, expected_samples=315,
                        constant_feature_policy="error"):
    """Does not load, normalize, or inspect the held-out cutter."""
    ids = [str(x) for x in train_ids]
    if len(ids) != 2 or len(set(ids)) != 2 or str(test_id) in ids:
        raise ValueError("Need two distinct development cutters and a separate test cutter")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    train_x, train_y, val_x, val_y, splits = [], [], [], [], {}
    for cutter_id in ids:
        X, y = load_cutter_data(data_dir, cutter_id, expected_seq_len, expected_input_dim, expected_samples)
        n_train = int(len(X) * train_fraction)
        if n_train < 1 or n_train >= len(X):
            raise ValueError(f"C{cutter_id}: empty train/validation split")
        train_x.append(X[:n_train]); train_y.append(y[:n_train])
        val_x.append(X[n_train:]); val_y.append(y[n_train:])
        splits[cutter_id] = {"n_total": len(X), "train_indices_0based": list(range(n_train)),
                            "val_indices_0based": list(range(n_train, len(X))),
                            "X_sha256": file_sha256(Path(data_dir) / f"data_x{cutter_id}.npy"),
                            "y_sha256": file_sha256(Path(data_dir) / f"data_y{cutter_id}.npy")}
    shapes = {x.shape[1:] for x in train_x + val_x}
    if len(shapes) != 1:
        raise ValueError(f"Incompatible development shapes: {shapes}")
    X, y, VX, vy = map(np.concatenate, (train_x, train_y, val_x, val_y))
    scaler = TrainMinMaxScaler(constant_feature_policy=constant_feature_policy).fit(X, y)
    return scaler.transform_X(X), scaler.transform_y(y), scaler.transform_X(VX), scaler.transform_y(vy), scaler, splits


def sensor_mask(input_dim, sensor_config="all"):
    mask = np.ones(input_dim, dtype=np.float32)
    if sensor_config != "all":
        if input_dim != 21:
            raise ValueError("Sensor ablations require 21 channel-major [Max,Mean,Var] features")
        ranges = {"no_force": (0, 9), "no_vibration": (9, 18), "no_ae": (18, 21)}
        if sensor_config not in ranges:
            raise ValueError("Unknown sensor configuration")
        start, end = ranges[sensor_config]
        mask[start:end] = 0.0
    return mask


class MultiScaleConvTokenizer(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        d1 = d_model // 3
        d2 = d_model // 3
        d3 = d_model - d1 - d2
        self.branch_short = nn.Conv1d(input_dim, d1, 3, padding=1)
        self.branch_mid = nn.Conv1d(input_dim, d2, 7, padding=3)
        self.branch_long = nn.Conv1d(input_dim, d3, 15, padding=7)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        h = torch.cat([self.branch_short(x), self.branch_mid(x), self.branch_long(x)], dim=1)
        h = h.permute(0, 2, 1)
        return self.dropout(self.act(self.norm(h)))


class LocalConvBlock(nn.Module):
    def __init__(self, d_model, kernel_size=7, dropout=0.1):
        super().__init__()
        self.dw_conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model)
        self.pw_conv = nn.Conv1d(d_model, d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        h = x.permute(0, 2, 1)
        h = self.dw_conv(h)
        h = self.pw_conv(h)
        h = h.permute(0, 2, 1)
        return self.norm(self.dropout(self.act(h)) + residual)


class MultiScaleTrendDecomposition(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trend_conv5 = nn.Conv1d(d_model, d_model, 5, padding=2, groups=d_model, bias=False)
        self.trend_conv11 = nn.Conv1d(d_model, d_model, 11, padding=5, groups=d_model, bias=False)
        self.trend_conv21 = nn.Conv1d(d_model, d_model, 21, padding=10, groups=d_model, bias=False)

        nn.init.constant_(self.trend_conv5.weight, 1.0 / 5.0)
        nn.init.constant_(self.trend_conv11.weight, 1.0 / 11.0)
        nn.init.constant_(self.trend_conv21.weight, 1.0 / 21.0)

        self.fuse = nn.Conv1d(d_model * 3, d_model, 1)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        h = x.permute(0, 2, 1)
        t5, t11, t21 = self.trend_conv5(h), self.trend_conv11(h), self.trend_conv21(h)
        trend = self.fuse(torch.cat([t5, t11, t21], dim=1)).permute(0, 2, 1)
        detail = x - trend
        gate = self.gate(torch.cat([trend, detail], dim=-1))
        out = gate * trend + (1.0 - gate) * detail
        return self.norm(out + residual)


class DeepTemporalAttention(nn.Module):
    def __init__(self, d_model, hidden=64):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        score = self.score(x).squeeze(-1)
        weight = torch.softmax(score, dim=-1).unsqueeze(-1)
        return torch.sum(x * weight, dim=1)


class GLDformerModel(nn.Module):
    def __init__(self, input_dim, d_model=192, n_heads=8, n_layers=2, dropout=0.2, max_seq_len=100, variant="full"):
        super().__init__()
        if d_model < 3 or n_heads < 1 or d_model % n_heads or n_layers < 1:
            raise ValueError("Invalid embedding dimension/head count/layer count")
        if input_dim < 1 or max_seq_len < 1 or variant not in VARIANTS:
            raise ValueError("Invalid input dimensions or model variant")
        self.input_dim = input_dim
        self.variant = variant
        self.tokenizer = (
            nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model),
                          nn.GELU(), nn.Dropout(0.1))
            if variant == "no_tokenizer" else MultiScaleConvTokenizer(input_dim, d_model)
        )
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer_stream = (None if variant == "no_global" else
                                   nn.TransformerEncoder(encoder_layer, num_layers=n_layers))
        self.local_stream = (None if variant == "no_local" else nn.Sequential(
            LocalConvBlock(d_model, 7, 0.1),
            LocalConvBlock(d_model, 3, 0.1)
        ))
        self.fusion_gate = (None if variant in ("no_global", "no_local") else
                            nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid()))
        self.fusion_norm = nn.LayerNorm(d_model)
        self.trend_decomp = nn.Identity() if variant == "no_trend" else MultiScaleTrendDecomposition(d_model)
        self.attn_pool = DeepTemporalAttention(d_model, hidden=64)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        self.input_residual = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.input_dim or x.shape[1] < 1:
            raise ValueError("Expected model input [batch, windows, input_dim]")
        raw_x = x
        _, seq_len, _ = x.shape
        x = self.tokenizer(x)

        pos_embed = self.pos_embed
        if seq_len != self.pos_embed.size(1):
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=seq_len,
                mode="linear",
                align_corners=False
            ).transpose(1, 2)

        x = x + pos_embed
        if self.transformer_stream is None:
            h = self.local_stream(x)
        elif self.local_stream is None:
            h = self.transformer_stream(x)
        else:
            h_global = self.transformer_stream(x)
            h_local = self.local_stream(x)
            gate = self.fusion_gate(torch.cat([h_global, h_local], dim=-1))
            h = gate * h_global + (1.0 - gate) * h_local
        h = self.fusion_norm(h + x)
        h = self.trend_decomp(h)
        h = self.attn_pool(h)

        deep_output = self.head(h).squeeze(-1)
        residual_output = self.input_residual(raw_x.mean(dim=1)).squeeze(-1)
        return deep_output + 0.1 * residual_output


class WeightedSmoothL1Loss(nn.Module):
    def __init__(self, beta=0.05, high_value_weight=1.0):
        super().__init__()
        if beta <= 0 or high_value_weight < 0:
            raise ValueError("beta must be positive, high_value_weight nonnegative")
        self.beta, self.high_value_weight = beta, high_value_weight

    def forward(self, pred, target):
        if pred.shape != target.shape:
            raise ValueError(f"Prediction/target shape mismatch: {pred.shape}, {target.shape}")
        loss = F.smooth_l1_loss(pred, target, beta=self.beta, reduction="none")
        return (loss * (1.0 + self.high_value_weight * target)).mean()


def calculate_metrics(pred, target):
    pred = finite_array(pred, "predictions", 1).astype(np.float64)
    target = finite_array(target, "targets", 1).astype(np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape}, {target.shape}")
    error = pred - target
    ss_res = float(np.dot(error, error))
    centered = target - target.mean()
    ss_tot = float(np.dot(centered, centered))
    # R² is undefined for fewer than 2 observations or constant targets.
    r2 = None if len(target) < 2 or ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {"MAE": float(np.abs(error).mean()), "RMSE": float(np.sqrt(ss_res / len(target))), "R2": r2}


def make_loader(X, y, batch_size, device, seed, training=False, noise_std=0.0, mask=None):
    return DataLoader(NoisyDataset(X, y, noise_std, training, mask), batch_size=batch_size,
                      shuffle=training, num_workers=0, pin_memory=device.type == "cuda",
                      drop_last=False, generator=torch.Generator().manual_seed(seed))


def evaluate_mse(model, loader, device):
    """Sample-weighted MSE, independent of batch partition for a fixed model."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            if pred.shape != y.shape:
                raise ValueError("Evaluation prediction/target shape mismatch")
            if not torch.isfinite(pred).all() or not torch.isfinite(y).all():
                raise FloatingPointError("Nonfinite evaluation prediction/target")
            total += ((pred.double() - y.double()) ** 2).sum().item()
            count += y.numel()
    if count == 0:
        raise ValueError("Evaluation loader is empty")
    result = total / count
    if not math.isfinite(result):
        raise FloatingPointError("Nonfinite evaluation loss")
    return result


def train_model(model, train_loader, val_loader, clean_train_loader, device,
                epochs=500, lr=3e-4, weight_decay=5e-4, patience=80,
                high_value_weight=1.0, history_path=None):
    if epochs < 1 or patience < 1 or len(train_loader.dataset) == 0:
        raise ValueError("epochs/patience must be positive and training data nonempty")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6)
    criterion = WeightedSmoothL1Loss(beta=0.05, high_value_weight=high_value_weight)
    best_val, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(X)
            loss = criterion(pred, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * y.numel()
            count += y.numel()
        # Both MSE curves use clean inputs, eval mode and the same epoch weights.
        train_mse = evaluate_mse(model, clean_train_loader, device)
        val_mse = evaluate_mse(model, val_loader, device)
        scheduler.step(val_mse)
        row = {"epoch": epoch, "train_weighted_smooth_l1": total / count,
               "train_mse_clean": train_mse, "val_mse": val_mse,
               "lr_next_epoch": float(optimizer.param_groups[0]["lr"])}
        history.append(row)
        if history_path is not None:
            with open(history_path, "a", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                if epoch == 1:
                    writer.writeheader()
                writer.writerow(row)
        if val_mse < best_val:
            best_val, best_epoch, stale = val_mse, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 25 == 0:
            LOGGER.info("Epoch %d | weighted L1 %.6f | clean train MSE %.6f | val MSE %.6f",
                        epoch, total / count, train_mse, val_mse)
        if stale >= patience:
            LOGGER.info("Early stopping at %d; best epoch %d", epoch, best_epoch)
            break
    if best_state is None:
        raise RuntimeError("No valid validation checkpoint; refusing to save a best model")
    model.load_state_dict(best_state)
    return model, history, float(best_val), best_epoch


def predict(model, X, device, batch_size=32):
    X = finite_array(X, "prediction inputs", 3)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(X[start:start + batch_size], dtype=np.float32)).to(device)
            pred = model(batch)
            if pred.ndim != 1 or pred.shape[0] != len(batch) or not torch.isfinite(pred).all():
                raise FloatingPointError("Invalid prediction output")
            outputs.append(pred.detach().cpu().numpy())
    return np.concatenate(outputs)


def plot_loss(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [r["epoch"] for r in history]
    axes[0].plot(epochs, [r["train_mse_clean"] for r in history], label="Train (clean, eval)")
    axes[0].plot(epochs, [r["val_mse"] for r in history], label="Validation (clean, eval)")
    axes[0].set_title("Comparable MSE on normalized targets")
    axes[0].set_ylabel("MSE")
    axes[1].plot(epochs, [r["train_weighted_smooth_l1"] for r in history], label="Training objective")
    axes[1].set_title("Weighted Smooth L1 during training")
    axes[1].set_ylabel("Weighted Smooth L1")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)


def plot_predictions(true_y, pred_y, metrics, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    indices = np.arange(1, len(true_y) + 1)
    axes[0].plot(indices, true_y, label="Measured")
    axes[0].plot(indices, pred_y, label="Predicted")
    axes[0].set_xlabel("Cut index (1-based)"); axes[0].set_ylabel("Wear (um)")
    r2 = "undefined" if metrics["R2"] is None else f"{metrics['R2']:.4f}"
    axes[0].set_title(f"{title}\nMAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={r2}")
    axes[0].legend()
    axes[1].scatter(true_y, pred_y, s=12, alpha=0.6)
    low, high = min(true_y.min(), pred_y.min()), max(true_y.max(), pred_y.max())
    axes[1].plot([low, high], [low, high], "r--")
    axes[1].set_xlabel("Measured wear (um)"); axes[1].set_ylabel("Predicted wear (um)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)


def build_model(config, input_dim, seq_len, variant="full"):
    return GLDformerModel(input_dim, d_model=config["d_model"], n_heads=config["n_heads"],
                          n_layers=config["n_layers"], dropout=config["dropout"],
                          max_seq_len=seq_len, variant=variant)


def load_checkpoint(path, device="cpu"):
    """This revision stores only tensors and Python primitives; no NumPy pickle."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("code_version") != CODE_VERSION:
        raise ValueError("This loader expects a checkpoint from this revision")
    model = build_model(checkpoint["config"], checkpoint["input_dim"], checkpoint["seq_len"],
                        checkpoint["variant"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, TrainMinMaxScaler.from_dict(checkpoint["scaler"]), checkpoint


def predict_from_checkpoint(path, X_raw, device="cpu", batch_size=32):
    model, scaler, checkpoint = load_checkpoint(path, device)
    if X_raw.shape[1:] != (checkpoint["seq_len"], checkpoint["input_dim"]):
        raise ValueError("Prediction shape differs from checkpoint (T,C)")
    X = scaler.transform_X(X_raw) * sensor_mask(checkpoint["input_dim"], checkpoint["sensor_config"])
    return scaler.inverse_y(predict(model, X, torch.device(device), batch_size))


def summarize_results(records):
    """Fold mean and sample std (ddof=1); Average std = mean of fold stds.

    These are averages of per-run metrics, NOT metrics of ensemble predictions.
    A single seed has no standard-deviation estimate; JSON uses null, not zero.
    """
    groups = {}
    for row in records:
        key = (row["variant"], row["sensor_config"], row["test_id"])
        groups.setdefault(key, []).append(row)
    folds, averages = [], []
    for (variant, sensors, test_id), rows in sorted(groups.items()):
        seeds = [row["seed"] for row in rows]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate seeds in result group")
        item = {"variant": variant, "sensor_config": sensors, "test_id": test_id,
                "seeds": seeds, "n_runs": len(rows)}
        for metric in METRICS:
            values = [r["metrics"][metric] for r in rows]
            if any(v is None for v in values):
                item[metric] = {"mean": None, "std": None}
            else:
                values = np.asarray(values, dtype=np.float64)
                item[metric] = {"mean": float(values.mean()),
                                "std": float(values.std(ddof=1)) if len(values) > 1 else None}
        folds.append(item)
    for variant, sensors in sorted({(r["variant"], r["sensor_config"]) for r in folds}):
        rows = [r for r in folds if (r["variant"], r["sensor_config"]) == (variant, sensors)]
        seed_sets = {tuple(sorted(r["seeds"])) for r in rows}
        if len(seed_sets) != 1:
            raise ValueError("All folds must include the same predeclared seeds")
        average = {"variant": variant, "sensor_config": sensors, "n_folds": len(rows),
                   "complete_three_folds": {r["test_id"] for r in rows} == {"1", "4", "6"},
                   "std_definition": "mean of within-fold sample standard deviations"}
        for metric in METRICS:
            means, stds = [r[metric]["mean"] for r in rows], [r[metric]["std"] for r in rows]
            average[metric] = {"mean": None if any(v is None for v in means) else float(np.mean(means)),
                               "std": None if any(v is None for v in stds) else float(np.mean(stds))}
        averages.append(average)
    return {"folds": folds, "averages": averages, "std_ddof": 1}


def environment_info(device):
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "matplotlib": matplotlib.__version__,
            "platform": platform.platform(), "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "torch_threads": torch.get_num_threads(),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def train_fold(args, experiment, variant, sensors, run_root, device, environment):
    fold_dir = run_root / variant / sensors / experiment["name"]
    fold_dir.mkdir(parents=True, exist_ok=False)
    X, y, VX, vy, scaler, splits = prepare_development(
        args.data_dir, experiment["train_ids"], experiment["test_id"],
        expected_seq_len=args.seq_len, expected_input_dim=args.input_dim,
        expected_samples=args.expected_samples, constant_feature_policy=args.constant_feature_policy)
    write_json(fold_dir / "split_and_scaler.json", {"split": splits, "scaler": scaler.to_dict()})
    mask = sensor_mask(X.shape[-1], sensors)
    options = [(d, p) for d in (128, 160, 192) for p in (0.15, 0.20)] if args.tune else [(args.d_model, args.dropout)]
    candidates = []
    for d_model, dropout in options:
        config = {"d_model": d_model, "n_heads": 8, "n_layers": 2, "dropout": dropout,
                  "lr": 3e-4, "weight_decay": 5e-4, "high_value_weight": 1.0}
        candidate_dir = fold_dir / f"d{d_model}_drop{dropout:.2f}"
        candidate_dir.mkdir()
        checkpoints, val_scores = [], []
        for seed in args.seeds:
            set_seed(seed, deterministic=not args.allow_nondeterministic)
            seed_dir = candidate_dir / f"seed_{seed}"
            seed_dir.mkdir()
            LOGGER.info("%s | %s | %s | d=%d drop=%.2f seed=%d", experiment["name"], variant, sensors, d_model, dropout, seed)
            model = build_model(config, X.shape[-1], X.shape[1], variant).to(device)
            train_loader = make_loader(X, y, args.batch_size, device, seed, True, args.noise_std, mask)
            val_loader = make_loader(VX, vy, args.eval_batch_size, device, seed + 1, mask=mask)
            clean_loader = make_loader(X, y, args.eval_batch_size, device, seed + 2, mask=mask)
            model, history, best_val, best_epoch = train_model(
                model, train_loader, val_loader, clean_loader, device, epochs=args.epochs,
                patience=args.patience, history_path=seed_dir / "history.csv")
            checkpoint = {"code_version": CODE_VERSION,
                          "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                          "input_dim": X.shape[-1], "seq_len": X.shape[1], "config": config,
                          "variant": variant, "sensor_config": sensors, "seed": seed,
                          "scaler": scaler.to_dict(), "train_ids": list(experiment["train_ids"]),
                          "test_id": experiment["test_id"], "best_epoch": best_epoch,
                          "best_val_mse": best_val, "split": splits, "environment": environment,
                          "training": {"batch_size": args.batch_size, "max_epochs": args.epochs,
                                       "patience": args.patience, "noise_std": args.noise_std,
                                       "deterministic": not args.allow_nondeterministic}}
            path = seed_dir / "gldformer_best_model.pth"
            torch.save(checkpoint, path)
            write_json(seed_dir / "training_result.json", {k: v for k, v in checkpoint.items() if k != "model_state_dict"})
            plot_loss(history, seed_dir / "loss_curve.png")
            checkpoints.append(str(path))
            val_scores.append(best_val)
            del model
        candidates.append({"config": config, "validation_mse_by_seed": dict(zip(map(str, args.seeds), val_scores)),
                           "selection_score": float(np.mean(val_scores)), "checkpoints": checkpoints})
    selected = min(candidates, key=lambda c: c["selection_score"])
    selection = {"rule": "minimum mean best validation MSE across ALL predeclared seeds" if args.tune else "fixed configuration; no search",
                 "candidates": candidates, "selected": selected, "test_used_for_selection": False}
    write_json(fold_dir / "selection.json", selection)
    return {"experiment": experiment, "variant": variant, "sensor_config": sensors,
            "checkpoints": selected["checkpoints"], "fold_dir": str(fold_dir)}


def evaluate_fold(args, job, device):
    # Called only after every requested fold/configuration selection is finished.
    exp = job["experiment"]
    test_X, target = load_cutter_data(args.data_dir, exp["test_id"], args.seq_len, args.input_dim, args.expected_samples)
    test_files = {"X_sha256": file_sha256(Path(args.data_dir) / f"data_x{exp['test_id']}.npy"),
                  "y_sha256": file_sha256(Path(args.data_dir) / f"data_y{exp['test_id']}.npy")}
    results = []
    for path in job["checkpoints"]:
        model, scaler, checkpoint = load_checkpoint(path, device)
        if test_X.shape[1:] != (checkpoint["seq_len"], checkpoint["input_dim"]):
            raise ValueError("Held-out cutter (T,C) differs from development cutters")
        normalized = scaler.transform_X(test_X) * sensor_mask(test_X.shape[-1], job["sensor_config"])
        pred_norm = predict(model, normalized, device, args.eval_batch_size)
        pred_real = scaler.inverse_y(pred_norm)
        metrics = calculate_metrics(pred_real, target)
        record = {"variant": job["variant"], "sensor_config": job["sensor_config"],
                  "test_id": exp["test_id"], "seed": checkpoint["seed"], "metrics": metrics,
                  "config": checkpoint["config"], "best_epoch": checkpoint["best_epoch"],
                  "best_val_mse": checkpoint["best_val_mse"], "test_files": test_files,
                  "checkpoint": path, "n_test": len(target)}
        dest = Path(path).parent
        np.save(dest / "predictions_um.npy", pred_real)
        np.save(dest / "predictions_normalized.npy", pred_norm)
        np.save(dest / "targets_um.npy", target)
        with open(dest / "predictions.csv", "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["cut_index_1based", "target_um", "prediction_um", "prediction_normalized"])
            writer.writerows(zip(range(1, len(target) + 1), target, pred_real, pred_norm))
        write_json(dest / "metrics.json", record)
        plot_predictions(target, pred_real, metrics, f"C{exp['test_id']} / seed {checkpoint['seed']}", dest / "prediction.png")
        LOGGER.info("Test C%s seed=%d: %s", exp["test_id"], checkpoint["seed"], metrics)
        results.append(record)
        del model
    return results


def parse_args(argv=None):
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", "--data_dir", default=str(base / "processed_data"))
    p.add_argument("--output-dir", "--output_dir", default=str(base / "gldformer_results"))
    p.add_argument("--batch-size", "--batch_size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=80)
    p.add_argument("--noise-std", "--noise_std", type=float, default=0.01)
    seeds = p.add_mutually_exclusive_group()
    seeds.add_argument("--seeds", type=int, nargs="+", help="All predeclared seeds; supply the actual experiment seeds")
    seeds.add_argument("--seed", type=int, help="Backward-compatible single-seed run")
    p.add_argument("--experiment", choices=("all", "1", "2", "3"), default="all",
                   help="1: test C6; 2: test C4; 3: test C1")
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=["full"])
    p.add_argument("--sensor-configs", nargs="+", choices=SENSOR_CONFIGS, default=["all"])
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--tune", action="store_true", help="Validation-only search: dims 128/160/192 x dropout 0.15/0.20")
    p.add_argument("--seq-len", type=int, default=100, help="Expected windows per pass; sensitivity studies may use 50/150")
    p.add_argument("--input-dim", type=int, default=21)
    p.add_argument("--expected-samples", type=int, default=315, help="Expected passes per cutter; 0 disables count check")
    p.add_argument("--constant-feature-policy", choices=("error", "unit"), default="error")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--allow-nondeterministic", action="store_true")
    args = p.parse_args(argv)
    args.seeds = args.seeds if args.seeds is not None else [42 if args.seed is None else args.seed]
    del args.seed
    for name in ("batch_size", "eval_batch_size", "epochs", "patience", "seq_len", "input_dim", "num_threads"):
        if getattr(args, name) < 1:
            p.error(f"{name} must be positive")
    if args.d_model < 8 or args.d_model % 8 or not 0 <= args.dropout < 1:
        p.error("d_model must be a positive multiple of 8; dropout must be in [0,1)")
    if not math.isfinite(args.noise_std) or args.noise_std < 0 or args.expected_samples < 0:
        p.error("noise_std and expected_samples must be nonnegative (noise_std finite)")
    for name in ("seeds", "variants", "sensor_configs"):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            p.error(f"{name} must contain no duplicates")
    if args.tune and (args.variants != ["full"] or args.sensor_configs != ["all"]):
        p.error("Tune full/all first; use fixed configurations for ablations so other settings stay unchanged")
    if any(seed < 0 or seed > 2**32 - 3 for seed in args.seeds):
        p.error("seeds must be in [0, 2**32-3]")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(args.num_threads)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = Path(args.output_dir).resolve() / f"run_{timestamp}_{uuid.uuid4().hex[:8]}"
    run_root.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(run_root / "run.log", encoding="utf-8")], force=True)
    environment = environment_info(device)
    manifest = {"code_version": CODE_VERSION, "code_sha256": file_sha256(__file__),
                "created_utc": timestamp, "arguments": vars(args), "environment": environment,
                "protocol": "80/20 chronological split within each development cutter; no test selection",
                "status": "training", "verified_paper_reproduction": False}
    write_json(run_root / "manifest.json", manifest)
    LOGGER.info("Results: %s", run_root)
    if len(args.seeds) < 5:
        LOGGER.warning("Only %d seed(s): this is not the manuscript's five-run protocol; std is undefined for one run", len(args.seeds))
    selected = EXPERIMENTS if args.experiment == "all" else [EXPERIMENTS[int(args.experiment) - 1]]
    try:
        jobs = []
        for variant in args.variants:
            for sensors in args.sensor_configs:
                for experiment in selected:
                    jobs.append(train_fold(args, experiment, variant, sensors, run_root, device, environment))
        # Freeze ALL requested choices before evaluating any held-out fold.
        write_json(run_root / "frozen_selections.json", jobs)
        manifest["status"] = "evaluating"
        write_json(run_root / "manifest.json", manifest)
        results = []
        for job in jobs:
            results.extend(evaluate_fold(args, job, device))
        write_json(run_root / "all_runs.json", results)
        summary = summarize_results(results)
        write_json(run_root / "summary.json", summary)
        with open(run_root / "summary.csv", "w", encoding="utf-8", newline="") as stream:
            fields = ["variant", "sensor_config", "test_id", "n_runs"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in summary["folds"]:
                flat = {k: row[k] for k in fields[:4]}
                flat.update({f"{m}_{s}": row[m][s] for m in METRICS for s in ("mean", "std")})
                writer.writerow(flat)
            for row in summary["averages"]:
                flat = {"variant": row["variant"], "sensor_config": row["sensor_config"],
                        "test_id": "Average" if row["complete_three_folds"] else "PartialAverage", "n_runs": len(args.seeds)}
                flat.update({f"{m}_{s}": row[m][s] for m in METRICS for s in ("mean", "std")})
                writer.writerow(flat)
        manifest["status"] = "complete"
        manifest["n_evaluated_runs"] = len(results)
        write_json(run_root / "manifest.json", manifest)
        LOGGER.info("Complete. Summary: %s", run_root / "summary.csv")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_json(run_root / "manifest.json", manifest)
        LOGGER.exception("Run failed; no complete summary should be reported")
        raise
    return run_root


if __name__ == "__main__":
    main()
