# ZeroAutoCL

**Zero-shot Neural Architecture Search for Contrastive Learning on Time Series**

ZeroAutoCL automatically discovers high-quality contrastive-learning (CL) recipes
for time-series representation learning, *without* running any training on the
target dataset. Given a new dataset, it ranks billions of candidate
(encoder configuration, CL strategy) combinations using a pre-trained
comparator and returns the top-K configurations in seconds.

## Motivation

Designing a contrastive-learning pipeline for a new time-series task is
expensive and brittle:

- **Hand-crafted recipes don't transfer.** Augmentations and loss
  hyperparameters that work on traffic data fail on physiological signals.
- **Per-task search is unaffordable.** Searching tens of thousands of
  candidates with full CL pretraining costs hundreds of GPU-hours per dataset.
- **Existing AutoML for CL is task-specific.** Methods like AutoCLS search
  *within* one dataset and must repeat the entire procedure for every new task.

ZeroAutoCL solves this by amortising the search cost: a comparator is trained
**once** on a pool of source datasets, then ranks candidates for unseen target
datasets at near-zero inference cost.

## Key Innovations

1. **Zero-shot NAS for contrastive learning.** We adapt the AutoCTS++
   zero-shot ranking paradigm — originally proposed for time-series
   forecasting architectures — to the CL-strategy search problem framed by
   AutoCLS. To our knowledge this is the first zero-shot NAS framework
   targeting time-series contrastive learning.

2. **Decoupled search space.**
   - **Primary axis (≈ 3 × 10¹² configurations):** the CL strategy —
     six data augmentations with continuous probabilities, embedding
     transforms, instance / temporal / cross-scale pair construction, and
     loss family (InfoNCE / Triplet) with similarity and temperature.
   - **Auxiliary axis (36 configurations):** coarse encoder hyperparameters
     (depth, hidden dim, output dim) over a fixed dilated-CNN backbone.
   - This separation reflects the empirical observation that CL strategy
     dominates final performance, while encoder topology has secondary
     impact. It keeps the search space tractable and the comparator
     learnable.

3. **T-CLSC pairwise comparator.** A Set-Transformer-based task encoder
   produces a task feature from raw time-series statistics; a pairwise
   ranking head then predicts P(A ≻ B | task) for any two candidate
   configurations. The comparator is trained on `(config, task,
   performance)` seed records using a curriculum that orders pairs by
   performance gap (large gaps first, hard pairs last).

4. **Compact vector encoding of candidates.** Because both encoder
   hyperparameters and CL strategies are structured low-dimensional
   parameters, candidates are encoded as a ~31-dim concatenated vector
   rather than a graph. This is simpler and more sample-efficient than
   AutoCTS++'s graph encoding while preserving all ranking-relevant
   information.

5. **AutoCTS++-style task variants for dense supervision.** Each source
   dataset is fanned into multiple sub-tasks via time-window slicing,
   variable subsampling, and horizon-group variation. The same candidate is
   evaluated across many sub-tasks, training the comparator's task-feature
   head to recognise how a configuration's rank shifts across task
   conditions — the prerequisite for zero-shot transfer.

## Pipeline Overview

```
Phase 1 — Offline preparation (paid once)
  Source datasets ──► sample (encoder, CL strategy) candidates
                  ──► CL pretraining + downstream evaluation
                  ──► seed records (config, task, performance)
  seed records + task features ──► train T-CLSC comparator (curriculum)

Phase 2 — Zero-shot inference (paid per target task)
  Target dataset ──► extract task feature
                 ──► sample large candidate pool
                 ──► T-CLSC ranking ──► Top-K shortlist
                 ──► full evaluation on target ──► best (encoder, strategy)

Phase 3 — Final training
  Best configuration ──► full CL pretraining on target
                     ──► downstream task evaluation
```

Phase 1 is a one-off cost; Phases 2–3 produce a deployable model for any new
target dataset within the supported task type.

## Supported Tasks

The framework is task-agnostic in design but is currently configured for three
downstream task types, each running an **independent** end-to-end pipeline
with its own source / target dataset pools and its own comparator:

- **Forecasting** — sources include ETTh2, PEMS03/04/07/08,
  ExchangeRate, pems-bay; targets are ETTh1 and ETTm1.
- **Classification** — HAR, Epilepsy, etc. (dataset list to be finalised).
- **Anomaly Detection** — Yahoo, KPI, etc. (dataset list to be finalised).

Source / target splits are strict: a target dataset never appears in seed
generation for the same task.

## Repository Layout

```
ZeroAutoCL/
├── configs/             # default + search-space YAMLs
├── data/                # dataset loaders, augmentations, sub-task slicer
├── models/
│   ├── encoder/         # parameterised dilated-CNN encoder
│   ├── contrastive/     # augmentation pipeline, embedding transforms,
│   │                    # pair construction, losses, full CL pipeline
│   ├── search_space/    # CL strategy space + candidate vector encoder
│   └── comparator/      # Set-Transformer task encoder + T-CLSC ranker
├── search/              # candidate sampler, seed generator,
│                        # comparator pretraining, zero-shot search
├── train/               # CL pretraining, finetuning, evaluation
├── utils/               # metrics, logging, reproducibility
└── scripts/             # SLURM array submission + helpers
```

## Acknowledgements

ZeroAutoCL builds on two prior works:

- **AutoCLS** (CIKM '24) — defines the CL strategy search space and the
  contrastive-learning protocol used during seed generation.
- **AutoCTS++** (VLDB Journal '24) — supplies the zero-shot ranking
  paradigm, the curriculum training scheme, and the task-variant idea
  (time-window / variable-subset / horizon-group fan-out) that we adapt to
  the CL setting.
