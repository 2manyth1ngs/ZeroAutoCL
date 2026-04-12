"""Quick test: GGS strategy + default encoder on HAR classification.

Verifies that the full CL pretrain → SVM eval pipeline works for
classification and reports baseline performance numbers.
"""
import sys, os, time, json, logging

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
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_har_ggs")

SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "datasets")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_ENCODER = {"n_layers": 10, "hidden_dim": 64, "output_dim": 320}

set_seed(SEED)

# ── 1. Load dataset ──────────────────────────────────────────────────────
logger.info("Loading HAR dataset...")
splits = load_dataset("HAR", DATA_DIR)
train_ds = splits["train"]
val_ds = splits["val"]
test_ds = splits["test"]
logger.info(
    "HAR loaded: train=%d  val=%d  test=%d  T=%d  C=%d  classes=%d",
    len(train_ds), len(val_ds), len(test_ds),
    train_ds.data.shape[1], train_ds.n_channels,
    len(torch.unique(train_ds.labels)),
)

# ── 2. Build encoder + CL pipeline ──────────────────────────────────────
encoder = DilatedCNNEncoder.from_config_dict(train_ds.n_channels, DEFAULT_ENCODER)
pipeline = CLPipeline(encoder, GGS_STRATEGY)
n_params = sum(p.numel() for p in pipeline.parameters())
logger.info("Encoder: %s  total params: %d", DEFAULT_ENCODER, n_params)
logger.info("Strategy: GGS_STRATEGY")
logger.info("Device: %s", DEVICE)

# ── 3. Contrastive pretrain ──────────────────────────────────────────────
cfg = {
    "pretrain_epochs": 40,
    "pretrain_lr": 0.001,
    "batch_size": 64,
    "eval_every": 10,
    "val_best": True,
}
logger.info("Pretrain config: %s", cfg)

t0 = time.time()
contrastive_pretrain(
    encoder=encoder,
    cl_pipeline=pipeline,
    train_data=train_ds,
    config=cfg,
    device=DEVICE,
    val_data=val_ds,
    task_type="classification",
)
train_time = time.time() - t0
logger.info("Pretrain done in %.1fs", train_time)

# ── 4. Evaluate ──────────────────────────────────────────────────────────
logger.info("Evaluating on test set...")
metrics = eval_classification(encoder, train_ds, test_ds, device=DEVICE)

logger.info("=" * 60)
logger.info("HAR GGS Baseline Results")
logger.info("=" * 60)
logger.info("  Accuracy : %.4f", metrics["acc"])
logger.info("  F1 (macro): %.4f", metrics["f1"])
logger.info("  Train time: %.1fs", train_time)
logger.info("  Encoder   : %s", DEFAULT_ENCODER)
logger.info("=" * 60)

# Also eval on val for reference
val_metrics = eval_classification(encoder, train_ds, val_ds, device=DEVICE)
logger.info("  Val Acc   : %.4f", val_metrics["acc"])
logger.info("  Val F1    : %.4f", val_metrics["f1"])
