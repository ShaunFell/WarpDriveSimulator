from abc import ABC, abstractmethod
from typing import Tuple
from dataclasses import dataclass
from functools import partial
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
        gamma = jnp.moveaxis(gamma, -1, 0)  # Shape (N, 3,3)

        # Shift vector
        beta = jnp.moveaxis(jnp.array([beta_x, beta_y, beta_z]), -1, 0)  # Shape (N,3)

        # Compute β_i β^i = β_i γ^ij β_j
        gamma_inv = jnp.linalg.inv(gamma)
        beta_squared = jnp.einsum(
            "...i,...ij,...j", beta, gamma_inv, beta
        )  # Shape (N,)

        # Build full 4D metric
        g00 = -(alpha**2) + beta_squared  # Shape (N,)
        g0i = beta  # Shape (N,3)

        metric4 = jnp.zeros((*g00.shape, 4, 4))
        metric4 = metric4.at[..., 0, 0].set(g00)
        metric4 = metric4.at[..., 0, 1:4].set(g0i)
        metric4 = metric4.at[..., 1:4, 0].set(g0i)
        metric4 = metric4.at[..., 1:4, 1:4].set(gamma)

        return metric4

    def dmetric4_dx(
        self, t: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray
    ) -> jnp.ndarray:
        """Derivative of 4D metric with respect to x"""
        return self._compute_4d_metric_derivative("dx", t, x, y, z)

    def dmetric4_dy(
        self, t: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray
    ) -> jnp.ndarray:
        """Derivative of 4D metric with respect to y"""
        return self._compute_4d_metric_derivative("dy", t, x, y, z)

    def dmetric4_dz(
        self, t: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray
    ) -> jnp.ndarray:
        """Derivative of 4D metric with respect to z"""
        return self._compute_4d_metric_derivative("dz", t, x, y, z)

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
        dg00 = jnp.moveaxis(-2 * alpha * lapse_deriv + dbeta_squared, -1, 0)

        # Derivative of g_0i (just the shift derivatives)
        dg0i = jnp.moveaxis(shift_derivs, -1, 0)

        # Derivative of g_ij (spatial metric derivatives)
        dgij = jnp.moveaxis(gamma_derivs, -1, 0)

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
    """Class for computing Christoffel symbols with JAX JIT-compatible memoization"""

    def __init__(self, metric: Metric):
        self.metric = metric
        # Pre-allocate cache arrays that JAX can handle
        self.cache_size = 100  # Adjust based on memory constraints

        # Cache storage as JAX arrays (not Python dicts)
        self._cache_coords = jnp.full(
            (self.cache_size, 4), jnp.nan
        )  # [t, x, y, z] coordinates
        self._cache_derivatives = {
            "dt": jnp.full((self.cache_size, 4, 4), jnp.nan),
            "dx": jnp.full((self.cache_size, 4, 4), jnp.nan),
            "dy": jnp.full((self.cache_size, 4, 4), jnp.nan),
            "dz": jnp.full((self.cache_size, 4, 4), jnp.nan),
        }
        self._cache_valid = jnp.zeros(
            self.cache_size, dtype=bool
        )  # Which cache entries are valid
        self._cache_index = 0  # Current cache insertion index

    @partial(jax.jit, static_argnums=0)
    def compute_symbol(
        self,
        indices: tuple,
        t: jnp.ndarray,
        x: jnp.ndarray,
        y: jnp.ndarray,
        z: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute Christoffel symbol Γ^μ_νρ at given coordinates with JAX-compatible caching

        Args:
            indices: List of 3 integers [μ, ν, ρ] specifying the component
            t, x, y, z: Spacetime coordinates where to evaluate

        Returns:
            Christoffel symbol Γ^μ_νρ at the given coordinates
        """
        mu, nu, rho = indices

        # Input validation
        mu = jnp.asarray(mu, dtype=int)
        nu = jnp.asarray(nu, dtype=int)
        rho = jnp.asarray(rho, dtype=int)

        # Convert scalars to arrays for consistent handling
        t_arr = jnp.atleast_1d(t)
        x_arr = jnp.atleast_1d(x)
        y_arr = jnp.atleast_1d(y)
        z_arr = jnp.atleast_1d(z)

        # For simplicity in JIT, work with first element if arrays
        # (full vectorization would require more complex caching)
        t_val = t_arr[0]
        x_val = x_arr[0]
        y_val = y_arr[0]
        z_val = z_arr[0]

        # Look for cached derivatives
        current_coords = jnp.array([t_val, x_val, y_val, z_val])

        # Check cache for matching coordinates (within tolerance)
        tolerance = 1e-12
        coord_diffs = jnp.linalg.norm(
            self._cache_coords - current_coords[None, :], axis=1
        )
        cache_matches = (coord_diffs < tolerance) & self._cache_valid
        cache_hit_idx = jnp.argmax(cache_matches)  # First matching index
        cache_hit = jnp.any(cache_matches)

        # Define function to compute derivatives (expensive operation)
        def compute_derivatives():
            dg_dt = self.metric.dmetric4_dt(t_arr, x_arr, y_arr, z_arr)[
                0
            ]  # Take first element
            dg_dx = self.metric.dmetric4_dx(t_arr, x_arr, y_arr, z_arr)[0]
            dg_dy = self.metric.dmetric4_dy(t_arr, x_arr, y_arr, z_arr)[0]
            dg_dz = self.metric.dmetric4_dz(t_arr, x_arr, y_arr, z_arr)[0]
            return dg_dt, dg_dx, dg_dy, dg_dz

        # Define function to get cached derivatives
        def get_cached_derivatives():
            return (
                self._cache_derivatives["dt"][cache_hit_idx],
                self._cache_derivatives["dx"][cache_hit_idx],
                self._cache_derivatives["dy"][cache_hit_idx],
                self._cache_derivatives["dz"][cache_hit_idx],
            )

        # Use JAX conditional to choose between cached and computed derivatives
        dg_dt, dg_dx, dg_dy, dg_dz = jax.lax.cond(
            cache_hit,
            lambda _: get_cached_derivatives(),
            lambda _: compute_derivatives(),
            operand=None,
        )

        # Update cache with new computation (only if not a cache hit)
        def update_cache():
            # Find insertion index (circular buffer)
            insert_idx = self._cache_index % self.cache_size

            # Update cache arrays
            new_cache_coords = self._cache_coords.at[insert_idx].set(current_coords)
            new_cache_dt = self._cache_derivatives["dt"].at[insert_idx].set(dg_dt)
            new_cache_dx = self._cache_derivatives["dx"].at[insert_idx].set(dg_dx)
            new_cache_dy = self._cache_derivatives["dy"].at[insert_idx].set(dg_dy)
            new_cache_dz = self._cache_derivatives["dz"].at[insert_idx].set(dg_dz)
            new_cache_valid = self._cache_valid.at[insert_idx].set(True)

            # Update instance variables (this works in JAX JIT)
            self._cache_coords = new_cache_coords
            self._cache_derivatives["dt"] = new_cache_dt
            self._cache_derivatives["dx"] = new_cache_dx
            self._cache_derivatives["dy"] = new_cache_dy
            self._cache_derivatives["dz"] = new_cache_dz
            self._cache_valid = new_cache_valid
            self._cache_index = insert_idx + 1

            return None

        def no_update():
            return None

        # Update cache only if we computed new derivatives
        jax.lax.cond(
            ~cache_hit, lambda _: update_cache(), lambda _: no_update(), operand=None
        )

        # Get the metric and its inverse at the point
        g = self.metric(t_arr, x_arr, y_arr, z_arr)[0]  # Take first element
        g_inv = jnp.linalg.inv(g)

        # Stack derivatives for easy indexing: [dt, dx, dy, dz]
        dg = jnp.array([dg_dt, dg_dx, dg_dy, dg_dz])

        # Christoffel symbol formula: Γ^μ_νρ = (1/2) g^μσ (∂g_σν/∂x^ρ + ∂g_σρ/∂x^ν - ∂g_νρ/∂x^σ)
        christoffel = 0.0
        for sigma in range(4):
            term1 = dg[rho, sigma, nu]  # ∂g_σν/∂x^ρ
            term2 = dg[nu, sigma, rho]  # ∂g_σρ/∂x^ν
            term3 = dg[sigma, nu, rho]  # ∂g_νρ/∂x^σ

            christoffel += 0.5 * g_inv[mu, sigma] * (term1 + term2 - term3)

        # Return scalar result that matches input type
        return jnp.where(jnp.isscalar(t), christoffel, jnp.array([christoffel]))[()]

    def clear_cache(self):
        """Clear the cache (call outside of JIT)"""
        self._cache_coords = jnp.full((self.cache_size, 4), jnp.nan)
        self._cache_derivatives = {
            "dt": jnp.full((self.cache_size, 4, 4), jnp.nan),
            "dx": jnp.full((self.cache_size, 4, 4), jnp.nan),
            "dy": jnp.full((self.cache_size, 4, 4), jnp.nan),
            "dz": jnp.full((self.cache_size, 4, 4), jnp.nan),
        }
        self._cache_valid = jnp.zeros(self.cache_size, dtype=bool)
        self._cache_index = 0

    def __call__(
        self,
        indices: tuple,
        t: jnp.ndarray,
        x: jnp.ndarray,
        y: jnp.ndarray,
        z: jnp.ndarray,
    ):
        return self.compute_symbol(indices, t, x, y, z)


class OLDChristoffel:
    """Class for computing Christoffel symbols"""

    def __init__(self, metric: Metric):
        self.metric = metric

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
        g = self.metric(t, x, y, z)
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
