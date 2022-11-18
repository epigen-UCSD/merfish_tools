"""Calculates and stores various statistics and quality metrics.

This module should generally not be interacted with directly, but rather through an instance of
the MerfishExperiment class (see experiment.py).
"""
from . import fileio

stats = {}
savefile = None


def set(key: str, value: float, silent: bool = False) -> None:
    stats[key] = value
    if not silent:
        print(f"{key}: {value}")
    if savefile:
        fileio.save_stats(stats, savefile)


def get(key: str) -> float:
    return stats[key]
