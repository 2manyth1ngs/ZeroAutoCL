"""Dataset loading and preprocessing for ZeroAutoCL.

Supported datasets
------------------
Classification   : HAR, Epilepsy, SleepEEG, Gesture, NATOPS
Anomaly detection: Yahoo, KPI
Forecasting      : ETTh1, ETTh2, ETTm1, PEMS series, exchange rate, electricity

All returned data arrays have shape (N, T, C) where:
  N — number of samples / time-series instances
  T — sequence length
  C — number of channels (features)

Data is always z-score normalised using statistics from the training split.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Split ratios per task type
# ---------------------------------------------------------------------------
# Classification: follow original paper splits (loaded from files directly).
# Anomaly detection: 50 % train / 50 % test; search uses 90 % / 10 % of train.
# Forecasting: standard ETT splits (fixed time-step counts).
_ETT_HOURLY_SPLITS = {
    "train": slice(None, 12 * 30 * 24),
    "val": slice(12 * 30 * 24, 16 * 30 * 24),
    "test": slice(16 * 30 * 24, 20 * 30 * 24),
}
_ETT_MINUTE_SPLITS = {
    "train": slice(None, 12 * 30 * 24 * 4),
    "val": slice(12 * 30 * 24 * 4, 16 * 30 * 24 * 4),
    "test": slice(16 * 30 * 24 * 4, 20 * 30 * 24 * 4),
}

# Sliding-window length used to slice long forecasting series into training
# samples for contrastive pretraining.  Matches the TS2Vec default crop size.
# See Bug #001 in CLAUDE_DEBUG.md.
#
# CLAUDE_ADV.md §10.1 (revised after the 2026-04-09 A/B test): aligned to
# the *actual* TS2Vec default `max_train_length=3000`, verified against
# reference/AutoCTS_plusplus/exps/generate_task_feature.py:44.  Must be
# ≥ the 10-layer dilated CNN receptive field (2^11-1=2047), otherwise the
# deep dilation layers only ever see padding for inputs shorter than RF
# and the model cannot learn dependencies long enough for H=336/720
# forecasting.  3000 brought ETTh1 mean MSE from 0.41 → 0.15 in the
# protocol-fix A/B test (outputs/test_protocol_ab/diff.txt).
_FORECAST_WINDOW_LEN = 3000

# Univariate forecasting on ETT (TS2Vec / AutoCLS protocol).
# Both TS2Vec (Table 2 in the paper) and AutoCLS (Table 4) report the
# headline ETT MSE numbers (~0.08 average over 5 horizons) under the
# **univariate** setting where only the OT (Oil Temperature) channel — the
# last column of the CSV — is used as both encoder input and prediction
# target.  Set this flag to False to recover the multivariate setup.
# See Bug #002 in CLAUDE_DEBUG.md.
_ETT_UNIVARIATE = True

# Ratio-based splits for datasets without fixed time-step boundaries.
_RATIO_SPLITS = {
    # PEMS / Exchange Rate: 60 / 20 / 20
    "6-2-2": (0.6, 0.2, 0.2),
    # Electricity: 70 / 10 / 20
    "7-1-2": (0.7, 0.1, 0.2),
}


# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------
@dataclass
class DatasetInfo:
    """Metadata for a single dataset."""

    name: str
    task_type: str  # 'classification' | 'forecasting' | 'anomaly_detection'
    n_channels: int
    seq_len: int
    n_classes: Optional[int] = None  # classification only
    horizon: Optional[int] = None    # forecasting only


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapper
# ---------------------------------------------------------------------------
class TimeSeriesDataset(Dataset):
    """Generic time-series dataset wrapper.

    Args:
        data: Array of shape (N, T, C).
        labels: Optional array of shape (N,) for classification labels or
            binary anomaly flags.  ``None`` for unsupervised pre-training.
        task_type: One of 'classification', 'forecasting', 'anomaly_detection'.
        max_len: If provided, sequences longer than this are truncated to the
            last *max_len* time steps.
        window_len: If provided and ``data`` is a single long series of shape
            ``(1, T, C)`` with ``T > window_len``, enable lazy sliding-window
            access: ``__getitem__`` slices windows on the fly from
            ``self.data``.  The DataLoader iterates over these windows, while
            ``self.data`` still points at the full original series so that
            downstream evaluation (e.g. Ridge forecasting on continuous
            embeddings) keeps working.
        window_stride: Stride between window starts.  Defaults to 1.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        task_type: str = "classification",
        max_len: Optional[int] = 2000,
        window_len: Optional[int] = None,
        window_stride: int = 1,
        scaler: Optional[StandardScaler] = None,
        n_covariate_cols: int = 0,
    ) -> None:
        if data.ndim != 3:
            raise ValueError(f"Expected data with ndim=3, got {data.ndim}")

        if max_len is not None and data.shape[1] > max_len:
            data = data[:, -max_len:, :]

        self.data = torch.from_numpy(data.astype(np.float32))
        self.labels = (
            torch.from_numpy(labels.astype(np.int64))
            if labels is not None
            else None
        )
        self.task_type = task_type
        self.scaler = scaler
        self.n_covariate_cols = int(n_covariate_cols)

        # Optional sliding-window view for contrastive pretraining on long series.
        # Bug #001 fix: ETT-style datasets are stored as (1, T, C); without
        # windowing, ``len(self) == 1`` and the DataLoader (with drop_last=True
        # and a large batch size) yields zero batches per epoch — encoder is
        # never trained.
        #
        # Bug #005b fix: windows are computed lazily in ``__getitem__`` instead
        # of pre-materialised.  For PEMS07 (883ch, stride=1, T_train≈16k) the
        # old approach needed ~148 GB CPU RAM; lazy slicing uses O(1) extra.
        self._window_len: Optional[int] = None
        self._window_stride: int = 1
        self._n_windows: int = 0
        if (
            window_len is not None
            and self.data.shape[0] == 1
            and self.data.shape[1] > window_len
        ):
            T = self.data.shape[1]
            stride = max(1, int(window_stride))
            self._window_len = window_len
            self._window_stride = stride
            self._n_windows = (T - window_len) // stride + 1
            logger.info(
                "Lazy sliding windows: %d windows of length %d (stride=%d) "
                "from series of length %d",
                self._n_windows, window_len, stride, T,
            )

    def __len__(self) -> int:
        if self._n_windows > 0:
            return self._n_windows
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._n_windows > 0:
            start = idx * self._window_stride
            x = self.data[0, start : start + self._window_len]  # (window_len, C)
            return x, torch.tensor(-1)
        x = self.data[idx]
        y = self.labels[idx] if self.labels is not None else torch.tensor(-1)
        return x, y

    @property
    def n_channels(self) -> int:
        return self.data.shape[2]

    @property
    def seq_len(self) -> int:
        return self.data.shape[1]


# ---------------------------------------------------------------------------
# Loaders per dataset
# ---------------------------------------------------------------------------

def _load_pt_classification(
    data_dir: str,
    name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a classification dataset stored as ``{name}/{{train,val,test}}.pt``.

    Each ``.pt`` file is a dict with keys ``'samples'`` (N, C, T) and
    ``'labels'`` (N,).  This is the format produced by the TS-TCC / TS2Vec
    preprocessing scripts.

    Returns:
        train_x, train_y, val_x, val_y, test_x, test_y — arrays with
        x in **(N, T, C)** layout (transposed from the on-disk (N, C, T)).
    """
    base = os.path.join(data_dir, name)
    splits_out: List[Tuple[np.ndarray, np.ndarray]] = []
    for split in ("train", "val", "test"):
        d = torch.load(os.path.join(base, f"{split}.pt"), weights_only=False)
        x = d["samples"].numpy().astype(np.float32)  # (N, C, T)
        y = d["labels"].numpy().astype(np.int64)
        # Transpose to (N, T, C) — the project-wide convention.
        x = np.transpose(x, (0, 2, 1))
        splits_out.append((x, y))

    (train_x, train_y), (val_x, val_y), (test_x, test_y) = splits_out

    # z-score normalise per-channel using training statistics.
    N_tr, T, C = train_x.shape
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x.reshape(-1, C)).reshape(N_tr, T, C)
    val_x = scaler.transform(val_x.reshape(-1, C)).reshape(val_x.shape[0], T, C)
    test_x = scaler.transform(test_x.reshape(-1, C)).reshape(test_x.shape[0], T, C)

    # Remap labels to contiguous 0-based integers.
    unique_labels = np.sort(np.unique(np.concatenate([train_y, val_y, test_y])))
    label_map = {int(old): new for new, old in enumerate(unique_labels)}
    train_y = np.array([label_map[int(v)] for v in train_y], dtype=np.int64)
    val_y = np.array([label_map[int(v)] for v in val_y], dtype=np.int64)
    test_y = np.array([label_map[int(v)] for v in test_y], dtype=np.int64)

    return train_x, train_y, val_x, val_y, test_x, test_y


def _load_natops(
    data_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load NATOPS dataset from UEA ``.ts`` archive format.

    Returns:
        train_x (N_tr, T, C), train_y (N_tr,), test_x (N_te, T, C), test_y (N_te,)
    """

    def _parse_ts_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
        samples: List[np.ndarray] = []
        labels: List[int] = []
        in_data = False
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower() == "@data":
                    in_data = True
                    continue
                if line.startswith("@"):
                    continue
                if not in_data:
                    continue
                # Each line: ch1_vals:ch2_vals:...:chN_vals:label
                parts = line.split(":")
                label = float(parts[-1])
                channels = []
                for ch_str in parts[:-1]:
                    channels.append(
                        [float(v) for v in ch_str.split(",") if v.strip()]
                    )
                # channels list of C arrays → (C, T), transpose → (T, C)
                samples.append(np.array(channels, dtype=np.float32).T)
                labels.append(int(label))
        return np.stack(samples), np.array(labels, dtype=np.int64)

    base = os.path.join(data_dir, "NATOPS")
    train_x, train_y = _parse_ts_file(os.path.join(base, "NATOPS_TRAIN.ts"))
    test_x, test_y = _parse_ts_file(os.path.join(base, "NATOPS_TEST.ts"))

    # z-score normalise per-channel using training statistics.
    N_tr, T, C = train_x.shape
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x.reshape(-1, C)).reshape(N_tr, T, C)
    N_te = test_x.shape[0]
    test_x = scaler.transform(test_x.reshape(-1, C)).reshape(N_te, T, C)

    # Remap labels to 0-based.
    unique_labels = np.sort(np.unique(np.concatenate([train_y, test_y])))
    label_map = {int(old): new for new, old in enumerate(unique_labels)}
    train_y = np.array([label_map[int(v)] for v in train_y], dtype=np.int64)
    test_y = np.array([label_map[int(v)] for v in test_y], dtype=np.int64)

    return train_x, train_y, test_x, test_y


def _get_time_features(dt: pd.DatetimeIndex) -> np.ndarray:
    """Extract 7 calendar features from a DatetimeIndex.

    Mirrors ``reference/ts2vec/datautils.py::_get_time_features``.  These
    features help the encoder capture daily / weekly seasonality that is
    otherwise invisible after z-score normalisation of the target series.

    Args:
        dt: Pandas DatetimeIndex of length T.

    Returns:
        NumPy array of shape (T, 7) with columns
        [minute, hour, dayofweek, day, dayofyear, month, week].
    """
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.isocalendar().week.to_numpy(),
    ], axis=1).astype(np.float32)


def _load_ett(
    data_dir: str,
    name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, int]:
    """Load an ETT dataset (ETTh1, ETTh2, ETTm1) with time covariates.

    The returned series are packed as ``(1, T, 7 + C_raw)`` where the first 7
    channels are z-scored calendar features (minute/hour/dayofweek/day/
    dayofyear/month/week) and the remaining channels are the raw variables.
    This matches the TS2Vec / AutoCLS protocol (see
    ``reference/ts2vec/datautils.py::load_forecast_csv``).

    The returned scaler operates on the **raw variable columns only**, so it
    can be used directly to inverse-transform Ridge predictions for raw-space
    MSE/MAE reporting.

    Args:
        data_dir: Root data directory.
        name: One of 'ETTh1', 'ETTh2', 'ETTm1'.

    Returns:
        ``(train_data, val_data, test_data, raw_scaler, n_covariate_cols)``
        — the three arrays each have shape ``(1, T_split, 7 + C_raw)``,
        ``raw_scaler`` is a :class:`StandardScaler` fit on the raw training
        variables (shape ``(C_raw,)``), and ``n_covariate_cols == 7``.
    """
    csv_path = os.path.join(data_dir, f"{name}.csv")
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)

    # Time features come from the original DatetimeIndex, before any column
    # manipulation.  They are always 7-wide regardless of univariate mode.
    dt_feats = _get_time_features(df.index)             # (T_total, 7)
    n_covariate_cols = dt_feats.shape[1]

    data = df.values.astype(np.float32)                  # (T_total, C_raw_full)

    # Univariate protocol: keep only the OT column (last column of every ETT
    # CSV — the canonical target used by TS2Vec / AutoCLS for the headline
    # ETT MSE numbers).  See Bug #002 in CLAUDE_DEBUG.md.
    if _ETT_UNIVARIATE:
        data = data[:, -1:]  # (T_total, 1)
        logger.info(
            "ETT univariate mode: %s reduced to OT channel only, shape=%s",
            name, data.shape,
        )

    if "m1" in name.lower() or "m2" in name.lower():
        splits = _ETT_MINUTE_SPLITS
    else:
        splits = _ETT_HOURLY_SPLITS

    # Separate scalers: one for raw variables (used by inverse_transform in
    # eval_forecasting), one for time features (internal only).
    raw_scaler = StandardScaler().fit(data[splits["train"]])
    tf_scaler = StandardScaler().fit(dt_feats[splits["train"]])
    data_scaled = raw_scaler.transform(data)             # (T_total, C_raw)
    dt_scaled = tf_scaler.transform(dt_feats)            # (T_total, 7)

    # Concatenate: time features come FIRST, raw variables LAST — matches
    # TS2Vec's convention so downstream code can select raw cols via
    # ``data[:, :, n_covariate_cols:]``.
    full = np.concatenate([dt_scaled, data_scaled], axis=-1).astype(np.float32)

    train_data = full[splits["train"]][np.newaxis]       # (1, T_tr, 7+C_raw)
    val_data = full[splits["val"]][np.newaxis]
    test_data = full[splits["test"]][np.newaxis]

    return train_data, val_data, test_data, raw_scaler, n_covariate_cols


def _ratio_split(
    data: np.ndarray,
    ratios: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a temporal array by ratio along the first (time) axis.

    Args:
        data: Array of shape ``(T, C)`` or ``(T, N, C)``.
        ratios: ``(train_ratio, val_ratio, test_ratio)``.

    Returns:
        ``(train, val, test)`` arrays.
    """
    T = data.shape[0]
    t1 = int(T * ratios[0])
    t2 = int(T * (ratios[0] + ratios[1]))
    return data[:t1], data[t1:t2], data[t2:]


def _load_pems(
    data_dir: str,
    name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Load a PEMS traffic dataset (PEMS03, PEMS04, PEMS07, PEMS08).

    Expects ``{data_dir}/{name}.npz`` with a ``'data'`` key of shape
    ``(T, N, C)`` where N is the number of sensor nodes and C is typically 1
    (traffic flow).  The N nodes are treated as channels of a single
    multivariate time series, yielding shape ``(1, T, N)``.

    Returns:
        ``(train_data, val_data, test_data, scaler)`` — each array has shape
        ``(1, T_split, N)``.  ``scaler`` is fit on the training split and can
        be used to inverse-transform raw-space predictions.
    """
    npz_path = os.path.join(data_dir, f"{name}.npz")
    raw = np.load(npz_path)["data"].astype(np.float32)  # (T, N, C)
    # Use only the first feature channel (traffic flow).
    data = raw[:, :, 0]  # (T, N)

    scaler = StandardScaler()
    train_raw, val_raw, test_raw = _ratio_split(data, _RATIO_SPLITS["6-2-2"])

    # Fit scaler on training split (T_tr, N) → normalise per-node.
    train_data = scaler.fit_transform(train_raw)
    val_data = scaler.transform(val_raw)
    test_data = scaler.transform(test_raw)

    # Pack as (1, T, N) — single multivariate series.
    return (
        train_data[np.newaxis],
        val_data[np.newaxis],
        test_data[np.newaxis],
        scaler,
    )


def _load_pems_bay(
    data_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Load PEMS-BAY traffic speed dataset (325 sensors).

    Supports two common file formats:
      - HDF5: ``{data_dir}/pems-bay.h5`` with a ``'df'`` key containing a
        DataFrame of shape ``(T, 325)``.
      - NPZ:  ``{data_dir}/PEMS-BAY.npz`` with a ``'data'`` key of shape
        ``(T, N, C)``.

    Returns:
        ``(train_data, val_data, test_data, scaler)`` — each array has shape
        ``(1, T_split, 325)``.
    """
    h5_path  = os.path.join(data_dir, "pems-bay.h5")
    npz_path = os.path.join(data_dir, "PEMS-BAY.npz")

    if os.path.exists(h5_path):
        df = pd.read_hdf(h5_path)
        data = df.values.astype(np.float32)  # (T, 325)
    elif os.path.exists(npz_path):
        raw = np.load(npz_path)["data"].astype(np.float32)  # (T, N, C)
        data = raw[:, :, 0]  # (T, N)
    else:
        raise FileNotFoundError(
            f"PEMS-BAY data not found at {h5_path} or {npz_path}"
        )

    train_raw, val_raw, test_raw = _ratio_split(data, _RATIO_SPLITS["6-2-2"])

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_raw)
    val_data = scaler.transform(val_raw)
    test_data = scaler.transform(test_raw)

    return (
        train_data[np.newaxis],
        val_data[np.newaxis],
        test_data[np.newaxis],
        scaler,
    )


def _load_exchange_rate(
    data_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Load Exchange Rate dataset.

    Expects ``{data_dir}/exchange_rates.csv`` (CSV with date column + 8 feature
    columns) or ``{data_dir}/exchange_rates.txt`` (plain text, 8 columns,
    no header).

    Returns:
        train_data, val_data, test_data — each of shape ``(1, T_split, 8)``.
    """
    csv_path = os.path.join(data_dir, "exchange_rates.csv")
    txt_path = os.path.join(data_dir, "exchange_rates.txt")

    if os.path.exists(csv_path):
        # Detect format: if first cell contains descriptive text, this is the
        # Fed H.10 raw format with 5-6 metadata header rows.
        probe = pd.read_csv(csv_path, nrows=1, header=None)
        first_cell = str(probe.iloc[0, 0]).strip().lower()
        if first_cell.startswith("series") or first_cell.startswith("unit"):
            # Fed H.10 format — find the row that starts with a date-like value.
            raw = pd.read_csv(csv_path, header=None)
            # Locate the first data row (starts with a date like "1971-01-04").
            data_start = None
            for i in range(len(raw)):
                cell = str(raw.iloc[i, 0]).strip()
                if len(cell) >= 8 and cell[4] == "-":
                    data_start = i
                    break
            if data_start is None:
                raise ValueError("Cannot find data rows in exchange_rates.csv")
            df = raw.iloc[data_start:].copy()
            df = df.iloc[:, 1:]  # drop date column
            df = df.apply(pd.to_numeric, errors="coerce")
            df = df.ffill().bfill()  # fill missing values
            data = df.values.astype(np.float32)
        else:
            df = pd.read_csv(csv_path)
            date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
            df = df.drop(columns=date_cols)
            df = df.apply(pd.to_numeric, errors="coerce").ffill().bfill()
            data = df.values.astype(np.float32)
    elif os.path.exists(txt_path):
        data = np.loadtxt(txt_path, delimiter=",").astype(np.float32)
    else:
        raise FileNotFoundError(
            f"Exchange Rate data not found at {csv_path} or {txt_path}"
        )

    train_raw, val_raw, test_raw = _ratio_split(data, _RATIO_SPLITS["6-2-2"])

    scaler = StandardScaler().fit(train_raw)
    train_data = scaler.transform(train_raw)
    val_data = scaler.transform(val_raw)
    test_data = scaler.transform(test_raw)

    return (
        train_data[np.newaxis],
        val_data[np.newaxis],
        test_data[np.newaxis],
        scaler,
    )


def _load_electricity(
    data_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Load Electricity (ECL) dataset.

    Expects ``{data_dir}/electricity.csv`` with a date/timestamp column and
    321 electricity consumption columns.

    Returns:
        train_data, val_data, test_data — each of shape ``(1, T_split, 321)``.
    """
    csv_path = os.path.join(data_dir, "electricity.csv")
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    data = df.values.astype(np.float32)  # (T, 321)

    train_raw, val_raw, test_raw = _ratio_split(data, _RATIO_SPLITS["7-1-2"])

    scaler = StandardScaler().fit(train_raw)
    train_data = scaler.transform(train_raw)
    val_data = scaler.transform(val_raw)
    test_data = scaler.transform(test_raw)

    return (
        train_data[np.newaxis],
        val_data[np.newaxis],
        test_data[np.newaxis],
        scaler,
    )


def _load_anomaly(
    data_dir: str,
    name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load anomaly-detection dataset (Yahoo or KPI).

    Expects:
        {data_dir}/{name}/train_x.npy  — shape (N_tr, T, C)
        {data_dir}/{name}/test_x.npy   — shape (N_te, T, C)
        {data_dir}/{name}/test_y.npy   — shape (N_te, T) binary labels

    Returns:
        train_x, train_y (all-zeros placeholder), test_x, test_y
    """
    base = os.path.join(data_dir, name)
    train_x = np.load(os.path.join(base, "train_x.npy")).astype(np.float32)
    test_x = np.load(os.path.join(base, "test_x.npy")).astype(np.float32)
    test_y = np.load(os.path.join(base, "test_y.npy")).astype(np.int64)

    if train_x.ndim == 2:
        train_x = train_x[:, :, np.newaxis]
        test_x = test_x[:, :, np.newaxis]

    N, T, C = train_x.shape
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x.reshape(-1, C)).reshape(N, T, C)
    N_te, T_te = test_x.shape[0], test_x.shape[1]
    test_x = scaler.transform(test_x.reshape(-1, C)).reshape(N_te, T_te, C)

    train_y = np.zeros(len(train_x), dtype=np.int64)
    return train_x, train_y, test_x, test_y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_SUPPORTED_DATASETS: Dict[str, str] = {
    "HAR": "classification",
    "Epilepsy": "classification",
    "SleepEEG": "classification",
    "Gesture": "classification",
    "NATOPS": "classification",
    "Yahoo": "anomaly_detection",
    "KPI": "anomaly_detection",
    "ETTh1": "forecasting",
    "ETTh2": "forecasting",
    "ETTm1": "forecasting",
    "PEMS03": "forecasting",
    "PEMS04": "forecasting",
    "PEMS07": "forecasting",
    "PEMS08": "forecasting",
    "PEMS-BAY": "forecasting",
    "ExchangeRate": "forecasting",
    "Electricity": "forecasting",
}


def load_dataset(
    name: str,
    data_dir: str,
    max_len: int = 2000,
    window_len_override: Optional[int] = None,
) -> Dict[str, TimeSeriesDataset]:
    """Load a named dataset and return train/val/test splits.

    Args:
        name: Dataset name. Must be one of the supported datasets listed in
            ``_SUPPORTED_DATASETS``.
        data_dir: Root directory that contains per-dataset subdirectories.
        max_len: Maximum sequence length (longer sequences are truncated).
        window_len_override: If given, use this instead of
            ``_FORECAST_WINDOW_LEN`` for the sliding-window length of
            forecasting training splits.  Useful for seed generation where
            a shorter crop (e.g. 1024) speeds up training ~3× without
            affecting the relative ranking of CL strategies.

    Returns:
        Dict with keys 'train', 'val' (if available), 'test', each mapping to
        a :class:`TimeSeriesDataset`.

    Raises:
        ValueError: If *name* is not a supported dataset.
    """
    if name not in _SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset {name!r} not supported. "
            f"Choose from {list(_SUPPORTED_DATASETS)}"
        )

    task_type = _SUPPORTED_DATASETS[name]

    # Resolve the sliding-window length for forecasting training splits.
    forecast_wl = window_len_override if window_len_override is not None else _FORECAST_WINDOW_LEN

    # Forecasting tasks benefit from using the full series; only truncate
    # classification / anomaly-detection samples.
    if task_type == "forecasting":
        max_len = None

    logger.info("Loading dataset %s (task=%s) from %s", name, task_type, data_dir)

    splits: Dict[str, TimeSeriesDataset] = {}

    # ── Classification datasets with pre-split .pt files ────────────────
    if name in ("HAR", "Epilepsy", "SleepEEG", "Gesture"):
        train_x, train_y, val_x, val_y, test_x, test_y = (
            _load_pt_classification(data_dir, name)
        )
        splits["train"] = TimeSeriesDataset(train_x, train_y, task_type, max_len)
        splits["val"] = TimeSeriesDataset(val_x, val_y, task_type, max_len)
        splits["test"] = TimeSeriesDataset(test_x, test_y, task_type, max_len)

    elif name == "NATOPS":
        train_x, train_y, test_x, test_y = _load_natops(data_dir)
        # No pre-existing val split; carve 10% from train.
        n_val = max(1, int(0.1 * len(train_x)))
        splits["train"] = TimeSeriesDataset(train_x[:-n_val], train_y[:-n_val], task_type, max_len)
        splits["val"] = TimeSeriesDataset(train_x[-n_val:], train_y[-n_val:], task_type, max_len)
        splits["test"] = TimeSeriesDataset(test_x, test_y, task_type, max_len)

    elif name in ("Yahoo", "KPI"):
        train_x, train_y, test_x, test_y = _load_anomaly(data_dir, name)
        n_val = max(1, int(0.1 * len(train_x)))
        splits["train"] = TimeSeriesDataset(train_x[:-n_val], train_y[:-n_val], task_type, max_len)
        splits["val"] = TimeSeriesDataset(train_x[-n_val:], train_y[-n_val:], task_type, max_len)
        splits["test"] = TimeSeriesDataset(test_x, test_y, task_type, max_len)

    elif name in ("ETTh1", "ETTh2", "ETTm1"):
        train_data, val_data, test_data, scaler, n_cov = _load_ett(data_dir, name)
        splits["train"] = TimeSeriesDataset(
            train_data, None, task_type, max_len,
            window_len=forecast_wl, window_stride=1,
            scaler=scaler, n_covariate_cols=n_cov,
        )
        splits["val"] = TimeSeriesDataset(
            val_data, None, task_type, max_len,
            scaler=scaler, n_covariate_cols=n_cov,
        )
        splits["test"] = TimeSeriesDataset(
            test_data, None, task_type, max_len,
            scaler=scaler, n_covariate_cols=n_cov,
        )

    elif name in ("PEMS03", "PEMS04", "PEMS07", "PEMS08"):
        train_data, val_data, test_data, scaler = _load_pems(data_dir, name)
        splits["train"] = TimeSeriesDataset(
            train_data, None, task_type, max_len,
            window_len=forecast_wl, window_stride=1,
            scaler=scaler,
        )
        splits["val"] = TimeSeriesDataset(
            val_data, None, task_type, max_len, scaler=scaler,
        )
        splits["test"] = TimeSeriesDataset(
            test_data, None, task_type, max_len, scaler=scaler,
        )

    elif name == "PEMS-BAY":
        train_data, val_data, test_data, scaler = _load_pems_bay(data_dir)
        splits["train"] = TimeSeriesDataset(
            train_data, None, task_type, max_len,
            window_len=forecast_wl, window_stride=1,
            scaler=scaler,
        )
        splits["val"] = TimeSeriesDataset(
            val_data, None, task_type, max_len, scaler=scaler,
        )
        splits["test"] = TimeSeriesDataset(
            test_data, None, task_type, max_len, scaler=scaler,
        )

    elif name == "ExchangeRate":
        train_data, val_data, test_data, scaler = _load_exchange_rate(data_dir)
        splits["train"] = TimeSeriesDataset(
            train_data, None, task_type, max_len,
            window_len=forecast_wl, window_stride=1,
            scaler=scaler,
        )
        splits["val"] = TimeSeriesDataset(
            val_data, None, task_type, max_len, scaler=scaler,
        )
        splits["test"] = TimeSeriesDataset(
            test_data, None, task_type, max_len, scaler=scaler,
        )

    elif name == "Electricity":
        train_data, val_data, test_data, scaler = _load_electricity(data_dir)
        splits["train"] = TimeSeriesDataset(
            train_data, None, task_type, max_len,
            window_len=forecast_wl, window_stride=1,
            scaler=scaler,
        )
        splits["val"] = TimeSeriesDataset(
            val_data, None, task_type, max_len, scaler=scaler,
        )
        splits["test"] = TimeSeriesDataset(
            test_data, None, task_type, max_len, scaler=scaler,
        )

    logger.info(
        "Loaded %s — train=%d, val=%d, test=%d",
        name,
        len(splits["train"]),
        len(splits.get("val", [])),
        len(splits["test"]),
    )
    return splits
