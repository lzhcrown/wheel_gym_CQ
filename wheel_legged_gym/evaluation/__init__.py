"""Evaluation utilities; package/config imports load neither torch nor Isaac Gym."""

from importlib import import_module

__all__ = ["CsvEvaluationWriter", "VectorEpisodeMetrics", "summarize_episodes"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(import_module(".metrics", __name__), name)
    globals()[name] = value
    return value
