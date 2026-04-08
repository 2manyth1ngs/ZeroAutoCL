"""Batch sampling of (encoder_config, cl_strategy) candidates.

Thin wrapper around :func:`~models.search_space.cl_strategy_space.sample_candidate`
to generate large candidate pools for seed generation and zero-shot search.
"""

from typing import Dict, List, Tuple

from models.search_space.cl_strategy_space import sample_candidate


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
