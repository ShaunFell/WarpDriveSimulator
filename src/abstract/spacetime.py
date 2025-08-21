from abc import ABC, abstractmethod
from typing import Tuple
from dataclasses import dataclass
from functools import cached_property

from .metricvariables import Lapse, Shift, SpatialMetric
from .metric import Metric, Christoffel

import jax
import jax.numpy as jnp


class Spacetime(ABC):
    """Abstract base class for spacetime definitions"""

    def __init__(self, metric: Metric):
        self.metric = Metric
        self.christoffel = Christoffel(Metric)

    @abstractmethod
    def metric(
        self, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray, t: jnp.ndarray
    ) -> Metric:
        """Return the metric at given spacetime coordinates"""
        pass

    def christoffel(
        self,
        indices: tuple,
        t: jnp.ndarray,
        x: jnp.ndarray,
        y: jnp.ndarray,
        z: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return Christoffel symbol Γ^μ_νρ at given spacetime coordinates"""
        return self.christoffel.compute_symbol(indices, t, x, y, z)

    def dmetric(
        self,
        coord: str,
        t: jnp.ndarray,
        x: jnp.ndarray,
        y: jnp.ndarray,
        z: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute derivative of 4D metric with respect to given coordinate"""
        return self.metric._compute_4d_metric_derivative(coord, t, x, y, z)

    def __enter__(self):
        """Context manager entry - allows 'with spacetime:' syntax"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        pass
