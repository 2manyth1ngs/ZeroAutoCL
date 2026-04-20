"""Verify ZeroAutoCL's new forecasting_eval primitives reproduce TS2Vec's number.

Reuses the pretrained TS2Vec encoder
(``reference/ts2vec/training/ETTh1__ts2vec_etth1_univar_.../model.pkl``) but
drives it through ZeroAutoCL's own ``train.forecasting_eval`` helpers
(``causal_sliding_encode`` + ``generate_pred_samples`` + ``fit_ridge``).

If ZeroAutoCL's implementation is protocol-faithful, the H=24 normalised MSE
should match ``eval_res.pkl`` (0.042028) exactly.

Note on imports: both ZeroAutoCL and TS2Vec have a top-level ``models``
package, so we cannot put both on sys.path.  We put AUTOCLS_ROOT on sys.path
(so ``train.forecasting_eval`` resolves to ZeroAutoCL's), and load TS2Vec's
TSEncoder via importlib under a renamed package ``_ts2vec_models``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
AUTOCLS_ROOT = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(AUTOCLS_ROOT)
TS2VEC_DIR   = os.path.join(PROJECT_ROOT, "reference", "ts2vec")

sys.path.insert(0, AUTOCLS_ROOT)  # ZeroAutoCL/train/forecasting_eval

import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402
import torch                                                        # noqa: E402
from sklearn.preprocessing import StandardScaler                    # noqa: E402

from train.forecasting_eval import (                                # noqa: E402
    DEFAULT_PADDING, causal_sliding_encode, fit_ridge, generate_pred_samples,
)


def _load_ts2vec_tsencoder_class():
    """Load TSEncoder from reference/ts2vec/models/ under a renamed package
    so it does not collide with ZeroAutoCL's own ``models`` package."""
    pkg_name = "_ts2vec_models"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [os.path.join(TS2VEC_DIR, "models")]
    sys.modules[pkg_name] = pkg

    def _load(sub, filename):
        full = f"{pkg_name}.{sub}"
        spec = importlib.util.spec_from_file_location(
            full, os.path.join(TS2VEC_DIR, "models", filename),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("dilated_conv", "dilated_conv.py")     # resolves the relative import
    enc = _load("encoder", "encoder.py")
    return enc.TSEncoder


TSEncoder = _load_ts2vec_tsencoder_class()


CKPT_PATH = os.path.join(
    TS2VEC_DIR, "training",
    "ETTh1__ts2vec_etth1_univar_20260415_233831", "model.pkl",
)
ETTH1_CSV = os.path.join(AUTOCLS_ROOT, "data", "datasets", "ETTh1.csv")
TRAIN_END, VAL_END, TEST_END = 12 * 30 * 24, 16 * 30 * 24, 20 * 30 * 24
PRED_LENS = [24, 48, 168, 336, 720]


def _time_features(dt: pd.DatetimeIndex) -> np.ndarray:
    return np.stack([
        dt.minute.to_numpy(),    dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(), dt.day.to_numpy(),
        dt.dayofyear.to_numpy(), dt.month.to_numpy(),
        dt.isocalendar().week.to_numpy(),
    ], axis=1).astype(np.float64)


def _load_ts2vec_encoder(device):
    base = TSEncoder(input_dims=8, output_dims=320, hidden_dims=64, depth=10).to(device)
    wrapped = torch.optim.swa_utils.AveragedModel(base)
    wrapped.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=False))
    wrapped.eval()
    return wrapped


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Same preprocessing as TS2Vec's load_forecast_csv('ETTh1', univar=True).
    df = pd.read_csv(ETTH1_CSV, index_col="date", parse_dates=True)
    dt = _time_features(df.index)
    ot = df[["OT"]].to_numpy().astype(np.float64)
    tr_slice = slice(None, TRAIN_END)

    ot_scaled = StandardScaler().fit(ot[tr_slice]).transform(ot)
    dt_scaled = StandardScaler().fit(dt[tr_slice]).transform(dt)
    data = np.concatenate(
        [np.expand_dims(dt_scaled, 0), np.expand_dims(ot_scaled, 0)], axis=-1
    ).astype(np.float32)                                    # (1, T, 8)
    print(f"[info] data {data.shape}")

    enc = _load_ts2vec_encoder(device)

    # ZeroAutoCL's causal sliding encode — should behave like TS2Vec's
    # encode(causal=True, sliding_length=1, sliding_padding=200).
    h_all = causal_sliding_encode(
        enc, torch.from_numpy(data), padding=DEFAULT_PADDING, device=device,
    )                                                        # (T, 320)
    print(f"[info] h_all {h_all.shape}")

    h_tr = h_all[:TRAIN_END]
    h_va = h_all[TRAIN_END:VAL_END]
    h_te = h_all[VAL_END:TEST_END]
    d_tr = data[0, :TRAIN_END, -1:]
    d_va = data[0, TRAIN_END:VAL_END, -1:]
    d_te = data[0, VAL_END:TEST_END, -1:]

    for H in PRED_LENS:
        tr_f, tr_y = generate_pred_samples(h_tr, d_tr, H, drop=DEFAULT_PADDING)
        va_f, va_y = generate_pred_samples(h_va, d_va, H, drop=0)
        te_f, te_y = generate_pred_samples(h_te, d_te, H, drop=0)

        lr = fit_ridge(tr_f, tr_y, va_f, va_y)
        pred = lr.predict(te_f)
        mse = float(((pred - te_y) ** 2).mean())
        mae = float(np.abs(pred - te_y).mean())
        print(f"  H={H:>3d}  test MSE={mse:.6f}  test MAE={mae:.6f}")

    print("\nReference (reference/ts2vec/training/.../eval_res.pkl):")
    print("  H=24 MSE=0.042028   H=48 MSE=0.062730   H=168 MSE=0.120825")
    print("  H=336 MSE=0.140754  H=720 MSE=0.162356")


if __name__ == "__main__":
    main()
