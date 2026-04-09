"""Batch sampling of (encoder_config, cl_strategy) candidates.

Thin wrapper around :func:`~models.search_space.cl_strategy_space.sample_candidate`
to generate large candidate pools for seed generation and zero-shot search.
"""

from typing import Dict, List, Tuple

import random

from models.search_space.cl_strategy_space import (
    sample_candidate,
    sample_cl_strategy_only,
)


def batch_sample_candidates(
    n: int,
) -> List[Tuple[Dict[str, int], Dict]]:
    """Sample *n* independent (encoder_config, cl_strategy) pairs.

    Instance contrast is guaranteed to be ``True`` in every sample (enforced
    by :func:`sample_candidate`).

    Args:
        n: Number of candidates to generate.

    Returns:
        List of ``(encoder_config_dict, strategy_config_dict)`` tuples.
    """
    return [sample_candidate() for _ in range(n)]


def batch_sample_strategies(
    n: int,
    encoders: List[Dict[str, int]],
) -> List[Tuple[Dict[str, int], Dict]]:
    """Sample *n* (encoder, cl_strategy) pairs with encoder restricted to a fixed pool.

    Used by Plan B Stage B: after Stage A has shortlisted ``encoders`` (the
    Top-K_enc grid winners), Stage B samples CL strategies uniformly from the
    full strategy space and pairs each one with a uniformly chosen encoder
    from the shortlist.

    Args:
        n: Number of (encoder, strategy) pairs to generate.
        encoders: Pool of encoder configurations to draw from. Must be non-empty.

    Returns:
        List of ``(encoder_config_dict, strategy_config_dict)`` tuples.
    """
    if not encoders:
        raise ValueError("batch_sample_strategies: 'encoders' must be non-empty")
    return [
        (random.choice(encoders), sample_cl_strategy_only())
        for _ in range(n)
    ]
