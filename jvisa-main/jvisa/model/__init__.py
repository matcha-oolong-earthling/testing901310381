"""
jvisa.model – Sepsis prediction models.

Quick start
-----------
>>> from jvisa.model import SepsisRandomForest
>>> model = SepsisRandomForest()
>>> results = model.train_and_evaluate(df)
"""

from .random_forest import SepsisRandomForest

__all__ = ["SepsisRandomForest"]
__version__ = "0.1.0"
