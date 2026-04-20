"""Evaluate the pretrained TS2Vec encoder under the **official TS2Vec
forecasting protocol** (causal sliding encode + multi-step Ridge).

Differences from the previous (naive) version of this script:
  1. Encoding uses ``model.encode(..., causal=True, sliding_length=1,
     sliding_padding=200)`` — each timestep t sees only context [t-200, t].
  2. Labels are the **multi-step** sequence x[t+1..t+H] (flattened), not a
     single point x[t+H].
  3. Ridge ``alpha`` is selected on the val split from
     {0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000}
     (``reference/ts2vec/tasks/_eval_protocols.py::fit_ridge``).
  4. Training features drop the first ``padding`` steps (sliding_padding=200).
  5. Target uses only the OT column (``data[..., n_covariate_cols:]``); the
     7 time-feature channels serve only as covariate inputs.

Reuses ``reference/ts2vec/ts2vec.py::TS2Vec.encode`` and the helpers
``tasks.forecasting.generate_pred_samples`` + ``tasks._eval_protocols.fit_ridge``
directly, so the numbers here are comparable to
``reference/ts2vec/training/ETTh1__ts2vec_etth1_univar_.../eval_res.pkl``.
"""

from __future__ import annotations

import os
import sys

THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
AUTOCLS_ROOT = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(AUTOCLS_ROOT)
TS2VEC_DIR   = os.path.join(PROJECT_ROOT, "reference", "ts2vec")

sys.path.insert(0, TS2VEC_DIR)  # so that `ts2vec.py`, `models/`, `tasks/` import as top-level

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from ts2vec import TS2Vec                                   # noqa: E402
from tasks.forecasting import generate_pred_samples, cal_metrics  # noqa: E402
from tasks._eval_protocols import fit_ridge                 # noqa: E402


CKPT_PATH = os.path.join(
    TS2VEC_DIR, "training",
    "ETTh1__ts2vec_etth1_univar_20260415_233831", "model.pkl",
)
ETTH1_CSV = os.path.join(AUTOCLS_ROOT, "data", "datasets", "ETTh1.csv")

# ETT hourly splits (identical to TS2Vec's load_forecast_csv).
TRAIN_SLICE = slice(None, 12 * 30 * 24)                      # [0, 8640)
VALID_SLICE = slice(12 * 30 * 24, 16 * 30 * 24)              # [8640, 11520)
TEST_SLICE  = slice(16 * 30 * 24, 20 * 30 * 24)              # [11520, 14400)
PRED_LENS   = [24, 48, 168, 336, 720]                        # ETTh1 paper horizons
PADDING     = 200                                            # TS2Vec sliding_padding


def _time_features(dt: pd.DatetimeIndex) -> np.ndarray:
    """Mirror of reference/ts2vec/datautils.py::_get_time_features."""
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.isocalendar().week.to_numpy(),
    ], axis=1).astype(np.float64)


def _load_etth1_ts2vec_univar():
    """Reproduce load_forecast_csv('ETTh1', univar=True) exactly."""
    df = pd.read_csv(ETTH1_CSV, index_col="date", parse_dates=True)
    dt_embed = _time_features(df.index)                      # (T, 7)
    data = df[["OT"]].to_numpy().astype(np.float64)          # (T, 1)

    # Per-channel z-score using *train* statistics.
    scaler = StandardScaler().fit(data[TRAIN_SLICE])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)                           # (1, T, 1)

    dt_scaler = StandardScaler().fit(dt_embed[TRAIN_SLICE])
    dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)  # (1, T, 7)

    # Concatenate along the channel axis — time features first, then OT.
    data = np.concatenate([dt_embed, data], axis=-1)         # (1, T, 8)
    n_covariate_cols = 7
    return data.astype(np.float32), scaler, n_covariate_cols


def main() -> None:
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device_str}")
    print(f"[info] ckpt   = {CKPT_PATH}")

    data, scaler, n_cov = _load_etth1_ts2vec_univar()
    print(f"[info] data shape (B,T,C) = {data.shape}  (7 time feats + OT)")

    # Re-hydrate the pretrained model using its own class.
    model = TS2Vec(
        input_dims=data.shape[-1],                           # 8
        output_dims=320, hidden_dims=64, depth=10,
        device=device_str, batch_size=8,
    )
    model.load(CKPT_PATH)

    # Causal sliding encode — the correct forecasting feature for each timestep.
    print("[info] encoding with causal=True, sliding_length=1, sliding_padding=200 ...")
    all_repr = model.encode(
        data,
        causal=True,
        sliding_length=1,
        sliding_padding=PADDING,
        batch_size=256,
    )
    print(f"[info] all_repr shape = {all_repr.shape}")

    # Split reprs and targets.  Target = OT column only (skip 7 covariate cols).
    train_repr = all_repr[:, TRAIN_SLICE]
    valid_repr = all_repr[:, VALID_SLICE]
    test_repr  = all_repr[:, TEST_SLICE]
    train_data = data[:, TRAIN_SLICE, n_cov:]                # (1, T_tr, 1)
    valid_data = data[:, VALID_SLICE, n_cov:]
    test_data  = data[:, TEST_SLICE,  n_cov:]

    results = {}
    for H in PRED_LENS:
        tr_f, tr_y = generate_pred_samples(train_repr, train_data, H, drop=PADDING)
        va_f, va_y = generate_pred_samples(valid_repr, valid_data, H)
        te_f, te_y = generate_pred_samples(test_repr,  test_data,  H)

        lr = fit_ridge(tr_f, tr_y, va_f, va_y)
        te_pred = lr.predict(te_f)

        ori_shape = test_data.shape[0], -1, H, test_data.shape[2]
        te_pred_r = te_pred.reshape(ori_shape)
        te_y_r    = te_y.reshape(ori_shape)

        C = test_data.shape[2]
        te_pred_raw = scaler.inverse_transform(te_pred_r.reshape(-1, C)).reshape(te_pred_r.shape)
        te_y_raw    = scaler.inverse_transform(te_y_r.reshape(-1, C)).reshape(te_y_r.shape)

        m_norm = cal_metrics(te_pred_r, te_y_r)
        m_raw  = cal_metrics(te_pred_raw, te_y_raw)
        results[H] = {"norm": m_norm, "raw": m_raw}
        print(f"  H={H:>3d}  norm MSE={m_norm['MSE']:.6f}  norm MAE={m_norm['MAE']:.6f}  "
              f"|  raw MSE={m_raw['MSE']:.4f}  raw MAE={m_raw['MAE']:.4f}")

    print("=" * 78)
    print("ETTh1 (univariate, OT target) — Pretrained TS2Vec encoder")
    print("Official TS2Vec forecasting protocol (causal sliding encode, multi-step Ridge)")
    print("-" * 78)
    h24 = results[24]
    print(f"  H = 24   test  MSE (norm) = {h24['norm']['MSE']:.6f}")
    print(f"           test  MAE (norm) = {h24['norm']['MAE']:.6f}")
    print(f"           GGS-style perf (-MSE) = {-h24['norm']['MSE']:.6f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
