"""HAR GGS parameter tuning: test several configurations and compare."""
import sys, os, time, logging, copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from data.dataset import load_dataset
from models.encoder.dilated_cnn import DilatedCNNEncoder
from models.contrastive.cl_pipeline import CLPipeline
from models.search_space.cl_strategy_space import GGS_STRATEGY
from train.pretrain import contrastive_pretrain
from train.evaluate import eval_classification
from utils.reproducibility import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tuning")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "datasets")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_ENC = {"n_layers": 10, "hidden_dim": 64, "output_dim": 320}

# ── Configurations to test ───────────────────────────────────────────────
CONFIGS = {
    # baseline (上次跑的)
    "A_baseline": {
        "pretrain_epochs": 40, "pretrain_lr": 1e-3,
        "batch_size": 64, "eval_every": 10, "val_best": True,
    },
    # 更多 epoch，更频繁 eval
    "B_longer": {
        "pretrain_epochs": 200, "pretrain_lr": 1e-3,
        "batch_size": 64, "eval_every": 10, "val_best": True,
    },
    # 不用 val_best，直接用 last checkpoint（避免 val/test 分布偏差）
    "C_no_valbest": {
        "pretrain_epochs": 200, "pretrain_lr": 1e-3,
        "batch_size": 64, "eval_every": 0, "val_best": False,
    },
    # 小 batch + 低 lr + 长训练
    "D_small_batch": {
        "pretrain_epochs": 200, "pretrain_lr": 5e-4,
        "batch_size": 16, "eval_every": 0, "val_best": False,
    },
    # 小 lr + 长训练 + val_best
    "E_low_lr_valbest": {
        "pretrain_epochs": 200, "pretrain_lr": 5e-4,
        "batch_size": 64, "eval_every": 10, "val_best": True,
    },
}


def run_one(name: str, cfg: dict) -> dict:
    set_seed(42)
    encoder = DilatedCNNEncoder.from_config_dict(3, DEFAULT_ENC)
    pipeline = CLPipeline(encoder, GGS_STRATEGY)

    splits = load_dataset("HAR", DATA_DIR)
    train_ds, val_ds, test_ds = splits["train"], splits["val"], splits["test"]

    logger.info("─── %s: %s ───", name, cfg)
    t0 = time.time()
    contrastive_pretrain(
        encoder=encoder, cl_pipeline=pipeline,
        train_data=train_ds, config=cfg, device=DEVICE,
        val_data=val_ds, task_type="classification",
    )
    elapsed = time.time() - t0

    encoder.eval()
    test_m = eval_classification(encoder, train_ds, test_ds, device=DEVICE)
    val_m = eval_classification(encoder, train_ds, val_ds, device=DEVICE)

    return {
        "config": name,
        "test_acc": test_m["acc"], "test_f1": test_m["f1"],
        "val_acc": val_m["acc"], "val_f1": val_m["f1"],
        "time_s": round(elapsed, 1),
    }


results = []
for name, cfg in CONFIGS.items():
    r = run_one(name, cfg)
    results.append(r)
    logger.info(
        ">>> %s  test_acc=%.4f  test_f1=%.4f  val_acc=%.4f  time=%.0fs",
        r["config"], r["test_acc"], r["test_f1"], r["val_acc"], r["time_s"],
    )

# ── Summary table ────────────────────────────────────────────────────────
print("\n" + "=" * 85)
print(f"{'Config':<20s} {'Test Acc':>9s} {'Test F1':>9s} {'Val Acc':>9s} {'Val F1':>9s} {'Time':>7s}")
print("-" * 85)
for r in sorted(results, key=lambda x: -x["test_acc"]):
    print(
        f"{r['config']:<20s} {r['test_acc']:>9.4f} {r['test_f1']:>9.4f} "
        f"{r['val_acc']:>9.4f} {r['val_f1']:>9.4f} {r['time_s']:>6.0f}s"
    )
print("=" * 85)
