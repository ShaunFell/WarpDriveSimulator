from abc import ABC, abstractmethod
from typing import Tuple
from dataclasses import dataclass
from functools import cached_property

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

    @cached_property
    def dmetric4_dx(self) -> jnp.ndarray:
        """Derivative of 4D metric with respect to x"""
        return self._compute_4d_metric_derivative("dx")

    @cached_property
    def dmetric4_dy(self) -> jnp.ndarray:
        """Derivative of 4D metric with respect to y"""
        return self._compute_4d_metric_derivative("dy")

    @cached_property
    def dmetric4_dz(self) -> jnp.ndarray:
        """Derivative of 4D metric with respect to z"""
        return self._compute_4d_metric_derivative("dz")

    @cached_property
    def dmetric4_dt(self) -> jnp.ndarray:
        """Derivative of 4D metric with respect to t"""
        return self._compute_4d_metric_derivative("dt")

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
            getattr(self.shift, f"x_{coord}")(t, x, y, z),
            getattr(self.shift, f"y_{coord}")(t, x, y, z),
            getattr(self.shift, f"z_{coord}")(t, x, y, z),
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


import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Tuple
from dataclasses import dataclass


from typing import Callable


@dataclass
class Lapse:
    """Lapse function α in 3+1 decomposition"""

    value: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ]  # α(t,x,y,z)
    dx: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray] = (
        None
    )
    dy: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray] = (
        None
    )
    dz: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray] = (
        None
    )
    dt: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray] = (
        None
    )


@dataclass
class Shift:
    """Shift vector β^i in 3+1 decomposition"""

    x: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ]  # β^x(t,x,y,z)
    y: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ]  # β^y(t,x,y,z)
    z: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ]  # β^z(t,x,y,z)
    # Derivatives of each component
    dx_dx: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    dx_dy: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    dx_dz: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    dx_dt: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    dy_dx: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    dy_dy: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    dy_dz: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    dy_dt: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    dz_dx: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    dz_dy: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    dz_dz: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    dz_dt: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None


@dataclass
class SpatialMetric:
    """3D spatial metric γ_ij"""

    xx: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ]  # γ_xx(t,x,y,z)
    xy: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ]  # γ_xy(t,x,y,z)
    xz: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ]  # γ_xz(t,x,y,z)
    yy: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ]  # γ_yy(t,x,y,z)
    yz: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ]  # γ_yz(t,x,y,z)
    zz: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ]  # γ_zz(t,x,y,z)
    # Derivatives of each component
    xx_dx: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    xx_dy: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    xx_dz: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    xx_dt: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    xy_dx: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    xy_dy: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    xy_dz: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    xy_dt: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    xz_dx: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    xz_dy: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    xz_dz: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    xz_dt: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    yy_dx: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    yy_dy: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    yy_dz: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    yy_dt: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    yz_dx: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    yz_dy: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    yz_dz: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    yz_dt: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    zz_dx: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    zz_dy: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    zz_dz: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None
    zz_dt: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = None


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

    @property
    def dmetric4_dx(self) -> jnp.ndarray:
        """Derivative of 4D metric with respect to x"""
        return self._compute_4d_metric_derivative("dx")

    @property
    def dmetric4_dy(self) -> jnp.ndarray:
        """Derivative of 4D metric with respect to y"""
        return self._compute_4d_metric_derivative("dy")

    @property
    def dmetric4_dz(self) -> jnp.ndarray:
        """Derivative of 4D metric with respect to z"""
        return self._compute_4d_metric_derivative("dz")

    @property
    def dmetric4_dt(self) -> jnp.ndarray:
        """Derivative of 4D metric with respect to t"""
        return self._compute_4d_metric_derivative("dt")

    def _compute_4d_metric_derivative(self, coord: str) -> jnp.ndarray:
        """Compute derivative of 4D metric with respect to given coordinate"""
        # Get derivatives of 3+1 components
        lapse_deriv = getattr(self.lapse, coord)

        shift_derivs = jnp.array([
            getattr(self.shift, f"x_{coord}"),
            getattr(self.shift, f"y_{coord}"),
            getattr(self.shift, f"z_{coord}"),
        ])

        gamma_derivs = jnp.array([
            [
                getattr(self.metric3, f"xx_{coord}"),
                getattr(self.metric3, f"xy_{coord}"),
                getattr(self.metric3, f"xz_{coord}"),
            ],
            [
                getattr(self.metric3, f"xy_{coord}"),
                getattr(self.metric3, f"yy_{coord}"),
                getattr(self.metric3, f"yz_{coord}"),
            ],
            [
                getattr(self.metric3, f"xz_{coord}"),
                getattr(self.metric3, f"yz_{coord}"),
                getattr(self.metric3, f"zz_{coord}"),
            ],
        ])

        # Current values for computing derivatives
        gamma = jnp.array([
            [self.metric3.xx, self.metric3.xy, self.metric3.xz],
            [self.metric3.xy, self.metric3.yy, self.metric3.yz],
            [self.metric3.xz, self.metric3.yz, self.metric3.zz],
        ])

        beta = jnp.array([self.shift.x, self.shift.y, self.shift.z])
        gamma_inv = jnp.linalg.inv(gamma)

        # Derivative of g_00 = -α² + β_i β^i
        # d/dx(-α²) = -2α dα/dx
        # d/dx(β_i β^i) = d/dx(β_i γ^ij β_j) = dβ_i/dx γ^ij β_j + β_i d(γ^ij)/dx β_j + β_i γ^ij dβ_j/dx

        # Compute d(β_i β^i)/dx
        dbeta_squared = 2 * jnp.einsum(
            "i,ij,j", shift_derivs, gamma_inv, beta
        ) + jnp.einsum(
            "i,ij,j", beta, jnp.linalg.solve(gamma, -gamma_derivs @ gamma_inv), beta
        )

        # Derivative of g_00
        dg00 = -2 * self.lapse.value * lapse_deriv + dbeta_squared

        # Derivative of g_0i (just the shift derivatives)
        dg0i = shift_derivs

        # Derivative of g_ij (spatial metric derivatives)
        dgij = gamma_derivs

        # Build derivative of full 4D metric
        shape = (*self.lapse.value.shape, 4, 4)
        dmetric4 = jnp.zeros(shape)
        dmetric4 = dmetric4.at[..., 0, 0].set(dg00)
        dmetric4 = dmetric4.at[..., 0, 1:4].set(dg0i.T)
        dmetric4 = dmetric4.at[..., 1:4, 0].set(dg0i.T)
        dmetric4 = dmetric4.at[..., 1:4, 1:4].set(dgij.transpose((2, 3, 0, 1)))

        return dmetric4


class Christoffel:
    """Class for computing Christoffel symbols"""

    def __init__(self, metric: Metric):
        self.metric = metric

    def compute_symbol(
        self,
        indices: list,
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
