"""Robustness module for Byzantine-resilient distributed learning."""

from .aggregators import (
    RobustAggregator,
    average,
    geometric_median,
    krum,
    mda,
    median,
    multi_krum,
    nneighbor_means,
    server_clip,
    trmean,
)
from .attacks import (
    ByzantineAttack,
    a_little_is_enough,
    auto_ALIE,
    auto_FOE,
    fall_of_empires,
    labelflipping,
    mimic,
    signflipping,
)

__all__ = [
    "ByzantineAttack",
    "RobustAggregator",
    "a_little_is_enough",
    "auto_ALIE",
    "auto_FOE",
    "average",
    "fall_of_empires",
    "geometric_median",
    "krum",
    "labelflipping",
    "mda",
    "median",
    "mimic",
    "multi_krum",
    "nneighbor_means",
    "server_clip",
    "signflipping",
    "trmean",
]
