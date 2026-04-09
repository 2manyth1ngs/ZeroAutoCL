"""CL strategy search space definition and random sampling.

The search space follows AutoCLS Table 1:
  - 6 data augmentations each with p ∈ {0.0, 0.1, …, 0.9, 0.95}
  - 5 augmentation orderings (AutoCLS Table 8)
  - Embedding transforms: jitter_p, mask_p (same p choices) + norm type
  - Pair construction: temporal, cross_scale, kernel_size, pool_op, adj_neighbor
  - Loss: type, sim_func, temperature

Combined with 36 encoder configs this gives ~10^14 total candidates.

Instance contrast is **always True** and is never sampled.
"""

import random
from typing import Dict, List, Tuple

from models.encoder.encoder_config import ENCODER_CHOICES

# ---------------------------------------------------------------------------
# Space constants
# ---------------------------------------------------------------------------

CL_STRATEGY_SPACE: Dict[str, object] = {
    "p_choices": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    "aug_types": ["resize", "rescale", "jitter", "point_mask", "freq_mask", "crop"],
    "aug_orders": 5,
    "emb_norm": ["none", "layer_norm", "l2"],
    "temporal": [True, False],
    "cross_scale": [True, False],
    "kernel_size": [0, 2, 3, 5],
    "pool_op": ["avg", "max"],
    "adj_neighbor": [True, False],
    "loss_type": ["infonce", "triplet"],
    "sim_func": ["dot", "cosine", "euclidean"],
    "temperature": [0.01, 0.1, 1.0, 10.0, 100.0],
}

# GGS default strategy (AutoCLS Table 5)
GGS_STRATEGY: Dict = {
    "augmentation": {
        "resize": 0.2, "rescale": 0.3, "jitter": 0.0,
        "point_mask": 0.2, "freq_mask": 0.0, "crop": 0.2, "order": 3,
    },
    "embedding_transform": {"jitter_p": 0.7, "mask_p": 0.1, "norm_type": "none"},
    "pair_construction": {
        "instance": True, "temporal": False, "cross_scale": False,
        "kernel_size": 5, "pool_op": "avg", "adj_neighbor": False,
    },
    "loss": {"type": "infonce", "sim_func": "euclidean", "temperature": 1.0},
}

DEFAULT_ENCODER: Dict[str, int] = {"n_layers": 10, "hidden_dim": 64, "output_dim": 320}


# ---------------------------------------------------------------------------
# Sampling functions
# ---------------------------------------------------------------------------

def sample_cl_strategy() -> Dict:
    """Uniformly sample one CL strategy configuration from the search space.

    Returns:
        A strategy config dict with keys ``'augmentation'``,
        ``'embedding_transform'``, ``'pair_construction'``, ``'loss'``,
        compatible with :class:`~models.contrastive.CLPipeline`.
    """
    p = CL_STRATEGY_SPACE["p_choices"]

    augmentation = {
        aug: random.choice(p)
        for aug in CL_STRATEGY_SPACE["aug_types"]
    }
    augmentation["order"] = random.randrange(CL_STRATEGY_SPACE["aug_orders"])

    embedding_transform = {
        "jitter_p": random.choice(p),
        "mask_p": random.choice(p),
        "norm_type": random.choice(CL_STRATEGY_SPACE["emb_norm"]),
    }

    pair_construction = {
        "instance": True,  # always True, not searched
        "temporal": random.choice(CL_STRATEGY_SPACE["temporal"]),
        "cross_scale": random.choice(CL_STRATEGY_SPACE["cross_scale"]),
        "kernel_size": random.choice(CL_STRATEGY_SPACE["kernel_size"]),
        "pool_op": random.choice(CL_STRATEGY_SPACE["pool_op"]),
        "adj_neighbor": random.choice(CL_STRATEGY_SPACE["adj_neighbor"]),
    }

    loss = {
        "type": random.choice(CL_STRATEGY_SPACE["loss_type"]),
        "sim_func": random.choice(CL_STRATEGY_SPACE["sim_func"]),
        "temperature": random.choice(CL_STRATEGY_SPACE["temperature"]),
    }

    return {
        "augmentation": augmentation,
        "embedding_transform": embedding_transform,
        "pair_construction": pair_construction,
        "loss": loss,
    }


def sample_cl_strategy_only() -> Dict:
    """Alias of :func:`sample_cl_strategy` for explicitness in Plan B code."""
    return sample_cl_strategy()


def sample_encoder_config() -> Dict[str, int]:
    """Uniformly sample one encoder configuration from the 36 candidates.

    Returns:
        Dict with keys ``'n_layers'``, ``'hidden_dim'``, ``'output_dim'``.
    """
    return {
        key: random.choice(choices)
        for key, choices in ENCODER_CHOICES.items()
    }


def sample_candidate() -> Tuple[Dict[str, int], Dict]:
    """Sample a complete (encoder_config, cl_strategy) pair.

    Returns:
        Tuple of ``(encoder_config_dict, strategy_config_dict)``.
    """
    return sample_encoder_config(), sample_cl_strategy()
