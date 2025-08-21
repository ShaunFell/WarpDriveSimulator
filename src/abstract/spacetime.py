from abc import ABC, abstractmethod
from typing import Tuple
from dataclasses import dataclass
from functools import cached_property

from .metricvariables import Lapse, Shift, SpatialMetric

import jax
import jax.numpy as jnp


class Spacetime(ABC):
    """Abstract base class for spacetime definitions"""

    @abstractmethod
    def metric(
        self, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray, t: jnp.ndarray
    ) -> Metric:
        """Return the metric at given spacetime coordinates"""
        pass

    @abstractmethod
    def christoffel(
        self, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        """Return Christoffel symbols Γ^μ_νρ at given spacetime coordinates"""
        pass

    @abstractmethod
    def dmetric(
        self, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray, t: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return derivatives of metric (∂g_μν/∂x^α) at given spacetime coordinates

        Returns:
            Tuple of (dg/dx, dg/dy, dg/dz, dg/dt)
        """
        pass

    def __enter__(self):
        """Context manager entry - allows 'with spacetime:' syntax"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        pass
