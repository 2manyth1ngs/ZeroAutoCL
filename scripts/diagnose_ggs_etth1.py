"""Ablation: isolate which aspect of the pair_construction refactor
(hierarchical loop vs per-timestep instance contrast) caused the
GGS+ETTh1 regression.

Runs GGS with three variants under the same _quick_eval protocol:

  A: "refactor"  — current code, kernel_size=5 triggers hierarchical loop
  B: "no-hier"   — kernel_size forced to 0 (single-scale only);
                   still per-timestep 2B-2 instance contrast.
  C: "legacy"    — monkey-patch instance_loss back to mean-pool+B-1
                   (pre-refactor semantics); kernel_size=0 for fairness.

All three also report train-loss curve (first / mid / last iter) to
diagnose whether the loss is collapsing / saturating.
"""
from __future__ import annotations
import os, sys, time, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from data.dataset import load_dataset
from models.encoder.dilated_cnn import DilatedCNNEncoder
from models.contrastive.cl_pipeline import CLPipeline
from models.contrastive.pair_construction import ContrastivePairConstructor
from models.contrastive.losses import InfoNCELoss
from models.search_space.cl_strategy_space import GGS_STRATEGY, DEFAULT_ENCODER
from search.seed_generator import _quick_eval
from utils.reproducibility import set_seed


def legacy_instance_loss(self, h1, h2, loss_fn):
    """Pre-refactor instance contrast: mean-pool time, then B-1 negatives."""
    B = h1.shape[0]
    if B < 2:
        return h1.new_zeros(())
    anchor, positive = h1.mean(dim=1), h2.mean(dim=1)
    D = anchor.shape[-1]
    pos_exp = positive.unsqueeze(0).expand(B, B, D)
    mask = ~torch.eye(B, dtype=torch.bool, device=h1.device)
    negatives = pos_exp[mask].reshape(B, B - 1, D)
    return loss_fn(anchor, positive, negatives)


def run_one(tag: str, strategy: dict, legacy: bool, iters: int, train_ds, val_ds, device):
    set_seed(42)
    encoder = DilatedCNNEncoder.from_config_dict(train_ds.n_channels, DEFAULT_ENCODER).to(device)
    pipeline = CLPipeline(encoder, strategy).to(device)
    if legacy:
        pipeline.pair_constructor.instance_loss = legacy_instance_loss.__get__(pipeline.pair_constructor, ContrastivePairConstructor)

    opt = torch.optim.AdamW(encoder.parameters(), lr=1e-3)
    pipeline.train()

    # Lightweight training loop (iter-budget)
    from torch.utils.data import DataLoader
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    losses = []
    t0 = time.time()
    it = 0
    data_iter = iter(loader)
    while it < iters:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader); batch = next(data_iter)
        x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        opt.zero_grad()
        loss, _ = pipeline(x)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        it += 1
    train_time = time.time() - t0

    encoder.eval()
    perf = _quick_eval(encoder, train_ds, val_ds, train_ds.task_type, device)
    print(f"[{tag:10s}] loss first/mid/last = {losses[0]:.4f} / {losses[len(losses)//2]:.4f} / {losses[-1]:.4f}  "
          f"|  quick_eval(-MSE)={perf:.4f}  MSE={-perf:.4f}  ({train_time:.1f}s)")
    return perf


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open("configs/default.yaml", encoding="utf-8"))
    crop_len = cfg["seed_generation"].get("crop_len")
    splits = load_dataset("ETTh1", "data/datasets", window_len_override=int(crop_len) if crop_len else None)
    train_ds = splits["train"]
    val_ds = splits.get("val") or splits["test"]
    iters = 600

    # ── Three variants ────────────────────────────────────────────────
    strat_refactor = copy.deepcopy(GGS_STRATEGY)                    # kernel=5 hier loop active
    strat_no_hier  = copy.deepcopy(GGS_STRATEGY)                    # kernel=0 disables loop
    strat_no_hier["pair_construction"]["kernel_size"] = 0
    strat_legacy   = copy.deepcopy(GGS_STRATEGY)
    strat_legacy["pair_construction"]["kernel_size"] = 0            # legacy had no hier either

    print(f"device={device}  iters={iters}  crop_len={crop_len}")
    print("=" * 85)
    run_one("A refactor", strat_refactor, legacy=False, iters=iters, train_ds=train_ds, val_ds=val_ds, device=device)
    run_one("B no-hier",  strat_no_hier,  legacy=False, iters=iters, train_ds=train_ds, val_ds=val_ds, device=device)
    run_one("C legacy",   strat_legacy,   legacy=True,  iters=iters, train_ds=train_ds, val_ds=val_ds, device=device)
    print("=" * 85)


if __name__ == "__main__":
    main()
