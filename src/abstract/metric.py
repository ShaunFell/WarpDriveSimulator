from abc import ABC, abstractmethod
from typing import Tuple
from dataclasses import dataclass
from functools import cached_property, lru_cache

from .metricvariables import Lapse, Shift, SpatialMetric

import jax
import jax.numpy as jnp


class Metric(ABC):
    """Abstract base class for metric tensor in 3+1 decomposition"""

    def __init__(self, lapse: Lapse, shift: Shift, metric3: SpatialMetric):
        self.lapse = lapse
        self.shift = shift
        self.metric3 = metric3

    def metric4(
        self, t: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray
    ) -> jnp.ndarray:
        """Full 4D metric tensor g_μν in (-,+,+,+) signature"""
        # Build 4x4 metric tensor from 3+1 decomposition
        # g_00 = -α² + β_i β^i
        # g_0i = β_i
        # g_ij = γ_ij

        # Evaluate functions at coordinates
        alpha = self.lapse.value(t, x, y, z)
        beta_x = self.shift.x(t, x, y, z)
        beta_y = self.shift.y(t, x, y, z)
        beta_z = self.shift.z(t, x, y, z)

        # Convert spatial metric to matrix form
        gamma = jnp.array([
            [
                self.metric3.xx(t, x, y, z),
                self.metric3.xy(t, x, y, z),
                self.metric3.xz(t, x, y, z),
            ],
            [
                self.metric3.xy(t, x, y, z),
                self.metric3.yy(t, x, y, z),
                self.metric3.yz(t, x, y, z),
            ],
            [
                self.metric3.xz(t, x, y, z),
                self.metric3.yz(t, x, y, z),
                self.metric3.zz(t, x, y, z),
            ],
        ])

        # Shift vector
        beta = jnp.array([beta_x, beta_y, beta_z])

        # Compute β_i β^i = β_i γ^ij β_j
        gamma_inv = jnp.linalg.inv(gamma)
        beta_squared = jnp.einsum("i...,ij...,j...", beta, gamma_inv, beta)

        # Build full 4D metric
        g00 = -(alpha**2) + beta_squared
        g0i = beta

        metric4 = jnp.zeros((*g00.shape, 4, 4))
        metric4 = metric4.at[..., 0, 0].set(g00)
        metric4 = metric4.at[..., 0, 1:4].set(g0i)
        metric4 = metric4.at[..., 1:4, 0].set(g0i)
        metric4 = metric4.at[..., 1:4, 1:4].set(gamma)

        return metric4

    @lru_cache
    def dmetric4_dx(
        self, t: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray
    ) -> jnp.ndarray:
        """Derivative of 4D metric with respect to x"""
        return self._compute_4d_metric_derivative("dx", t, x, y, z)

    @lru_cache
    def dmetric4_dy(
        self, t: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray
    ) -> jnp.ndarray:
        """Derivative of 4D metric with respect to y"""
        return self._compute_4d_metric_derivative("dy", t, x, y, z)

    @lru_cache
    def dmetric4_dz(
        self, t: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray
    ) -> jnp.ndarray:
        """Derivative of 4D metric with respect to z"""
        return self._compute_4d_metric_derivative("dz", t, x, y, z)

    @lru_cache
    def dmetric4_dt(
        self, t: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray
    ) -> jnp.ndarray:
        """Derivative of 4D metric with respect to t"""
        return self._compute_4d_metric_derivative("dt", t, x, y, z)

    def _compute_4d_metric_derivative(
        self, coord: str, t: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute derivative of 4D metric with respect to given coordinate"""

        if coord not in ["dt", "dx", "dy", "dz"]:
            raise ValueError(
                f"coordinate direction {coord} is not a valid coordinate. Should be one of {['dt', 'dx', 'dy', 'dz']}"
            )

        # Get derivatives of 3+1 components by calling the derivative functions
        lapse_deriv = getattr(self.lapse, coord)(t, x, y, z)  # shape (N,)

        shift_derivs = jnp.array([
            getattr(self.shift, f"dx_{coord}")(t, x, y, z),
            getattr(self.shift, f"dy_{coord}")(t, x, y, z),
            getattr(self.shift, f"dz_{coord}")(t, x, y, z),
        ])  # shape: (3,N)

        gamma_derivs = jnp.array([
            [
                getattr(self.metric3, f"xx_{coord}")(t, x, y, z),
                getattr(self.metric3, f"xy_{coord}")(t, x, y, z),
                getattr(self.metric3, f"xz_{coord}")(t, x, y, z),
            ],
            [
                getattr(self.metric3, f"xy_{coord}")(t, x, y, z),
                getattr(self.metric3, f"yy_{coord}")(t, x, y, z),
                getattr(self.metric3, f"yz_{coord}")(t, x, y, z),
            ],
            [
                getattr(self.metric3, f"xz_{coord}")(t, x, y, z),
                getattr(self.metric3, f"yz_{coord}")(t, x, y, z),
                getattr(self.metric3, f"zz_{coord}")(t, x, y, z),
            ],
        ])  # shape: (3,3,N)

        # Current values for computing derivatives by calling the value functions
        alpha = self.lapse.value(t, x, y, z)  # shape (N,)
        gamma = jnp.array([
            [
                self.metric3.xx(t, x, y, z),
                self.metric3.xy(t, x, y, z),
                self.metric3.xz(t, x, y, z),
            ],
            [
                self.metric3.xy(t, x, y, z),
                self.metric3.yy(t, x, y, z),
                self.metric3.yz(t, x, y, z),
            ],
            [
                self.metric3.xz(t, x, y, z),
                self.metric3.yz(t, x, y, z),
                self.metric3.zz(t, x, y, z),
            ],
        ])  # shape (3,3,N)

        beta = jnp.array([
            self.shift.x(t, x, y, z),
            self.shift.y(t, x, y, z),
            self.shift.z(t, x, y, z),
        ])  # shape (3,N)

        # gamma has shape (3,3,N) but .inv compute inverse of (...,3,3) so need to swap axes
        gamma_inv = jnp.moveaxis(jnp.linalg.inv(jnp.moveaxis(gamma, -1, 0)), 0, -1)

        # Compute derivative of inverse
        gamma_inv_derivs = jnp.einsum(
            "lk...,ij...,kj...->li...", -gamma_inv, gamma_inv, gamma_derivs
        )

        # Compute d(β_i β^i)/dx
        dbeta_squared = 2 * jnp.einsum(
            "i...,ij...,j...", shift_derivs, gamma_inv, beta
        ) + jnp.einsum(
            "i...,ij...,j...",
            beta,
            gamma_inv_derivs,
            beta,
        )

        # Derivative of g_00
        dg00 = -2 * alpha * lapse_deriv + dbeta_squared

        # Derivative of g_0i (just the shift derivatives)
        dg0i = shift_derivs

        # Derivative of g_ij (spatial metric derivatives)
        dgij = gamma_derivs

        # Build derivative of full 4D metric
        shape = (*alpha.shape, 4, 4)
        dmetric4 = jnp.zeros(shape)
        dmetric4 = dmetric4.at[..., 0, 0].set(dg00)
        dmetric4 = dmetric4.at[..., 0, 1:4].set(dg0i)
        dmetric4 = dmetric4.at[..., 1:4, 0].set(dg0i)
        dmetric4 = dmetric4.at[..., 1:4, 1:4].set(dgij)

        return dmetric4

    def __call__(self, t, x, y, z):
        return self.metric4(t, x, y, z)


class Christoffel:
    """Class for computing Christoffel symbols"""

    def __init__(self, metric: Metric):
        self.metric = metric

    @lru_cache
    def compute_symbol(
        self,
        indices: tuple,
        t: jnp.ndarray,
        x: jnp.ndarray,
        y: jnp.ndarray,
        z: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute Christoffel symbol Γ^μ_νρ at given coordinates

        Args:
            indices: List of 3 integers [μ, ν, ρ] specifying the component
            t, x, y, z: Spacetime coordinates where to evaluate

        Returns:
            Christoffel symbol Γ^μ_νρ at the given coordinates
        """
        mu, nu, rho = indices
        if (mu > 3) or (nu > 3) or (rho > 3):
            raise ValueError(f"index must be either 0,1,2,3. Got {indices}")

        # Get the metric and its inverse at the point
        g = self.metric.metric4(t, x, y, z)
        g_inv = jnp.linalg.inv(g)

        # Get metric derivatives
        dg_dt = self.metric.dmetric4_dt(t, x, y, z)
        dg_dx = self.metric.dmetric4_dx(t, x, y, z)
        dg_dy = self.metric.dmetric4_dy(t, x, y, z)
        dg_dz = self.metric.dmetric4_dz(t, x, y, z)

        # Stack derivatives for easy indexing: [dt, dx, dy, dz]
        dg = jnp.array([dg_dt, dg_dx, dg_dy, dg_dz])

        # Christoffel symbol formula: Γ^μ_νρ = (1/2) g^μσ (∂g_σν/∂x^ρ + ∂g_σρ/∂x^ν - ∂g_νρ/∂x^σ)
        # Note: coordinate indices are [0=t, 1=x, 2=y, 3=z]
        coord_map = {0: 0, 1: 1, 2: 2, 3: 3}  # t, x, y, z -> 0, 1, 2, 3

        christoffel = 0.0
        for sigma in range(4):
            term1 = dg[coord_map[rho], ..., sigma, nu]  # ∂g_σν/∂x^ρ
            term2 = dg[coord_map[nu], ..., sigma, rho]  # ∂g_σρ/∂x^ν
            term3 = dg[sigma, ..., nu, rho]  # ∂g_νρ/∂x^σ

            christoffel += 0.5 * g_inv[..., mu, sigma] * (term1 + term2 - term3)

        return christoffel

    def __call__(
        self,
        indices: tuple,
        t: jnp.ndarray,
        x: jnp.ndarray,
        y: jnp.ndarray,
        z: jnp.ndarray,
    ):
        return self.compute_symbol(indices, t, x, y, z)
