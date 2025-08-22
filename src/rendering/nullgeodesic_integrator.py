import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
import time
from functools import partial
from src.abstract.spacetime import Spacetime


@dataclass
class RayState:
    """State of a light ray at a given point"""

    position: jnp.ndarray  # [t, x, y, z] coordinates, shape (4,)
    direction: jnp.ndarray  # [dt/dλ, dx/dλ, dy/dλ, dz/dλ] 4-velocity, shape (4,)
    affine_parameter: float  # Affine parameter λ


@dataclass
class IntegrationConfig:
    """Configuration for geodesic integration"""

    step_size: float = 0.01
    max_steps: int = 10000
    tolerance: float = 1e-8
    adaptive_stepping: bool = True
    min_step_size: float = 1e-6
    max_step_size: float = 0.1


class LightRayIntegrator:
    """
    Light ray integrator for general relativistic spacetimes.

    Integrates null geodesics (light rays) using the geodesic equation:
    d²x^μ/dλ² + Γ^μ_νρ (dx^ν/dλ)(dx^ρ/dλ) = 0

    where λ is an affine parameter and the rays satisfy the null condition:
    g_μν dx^μ/dλ dx^ν/dλ = 0
    """

    def __init__(self, spacetime: Spacetime, config: IntegrationConfig = None):
        """
        Initialize the light ray integrator.

        Args:
            spacetime: The spacetime geometry to integrate in
            config: Integration configuration parameters
        """
        self.spacetime = spacetime
        self.config = config or IntegrationConfig()

        # JIT compile integration functions for performance
        self._rk4_step = self._rk4_step_impl  # jax.jit(self._rk4_step_impl)
        self._compute_acceleration = jax.jit(self._compute_acceleration_impl)

    @partial(jax.jit, static_argnums=(0, 3))
    def integrate_ray(
        self,
        initial_position: jnp.ndarray,
        initial_direction: jnp.ndarray,
        max_affine_parameter: float = 100.0,
        termination_condition: Optional[Callable] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        normalized_direction = self._normalize_null_vector(
            initial_position, initial_direction
        )

        max_steps = int(max_affine_parameter / self.config.step_size + 1)

        positions = jnp.zeros((max_steps, 4))
        directions = jnp.zeros((max_steps, 4))
        affine_params = jnp.zeros((max_steps,))

        positions = positions.at[0].set(initial_position)
        directions = directions.at[0].set(normalized_direction)
        affine_params = affine_params.at[0].set(0.0)

        init_state = {
            "step": jnp.array(0, dtype=jnp.int32),
            "pos": initial_position,
            "dir": normalized_direction,
            "lam": jnp.array(0.0, dtype=jnp.float32),
            "step_size": self.config.step_size,
            "positions": positions,
            "directions": directions,
            "affine": affine_params,
        }

        def cond_fn(state):
            not_done = state["step"] < max_steps - 1
            not_exceeded = state["lam"] < max_affine_parameter
            return jnp.logical_and(not_done, not_exceeded)

        def body_fn(state):
            pos, dir, lam, step_size = (
                state["pos"],
                state["dir"],
                state["lam"],
                state["step_size"],
            )

            # step
            def adaptive(_):
                new_pos, new_dir, new_step_size = self._adaptive_step(
                    pos, dir, step_size
                )
                return new_pos, new_dir, new_step_size

            def fixed(_):
                new_pos, new_dir = self._rk4_step(pos, dir, step_size)
                return new_pos, new_dir, step_size

            new_pos, new_dir, new_step_size = jax.lax.cond(
                self.config.adaptive_stepping,
                adaptive,
                fixed,
                operand=None,
            )

            new_step = state["step"] + 1
            new_lam = lam + new_step_size

            positions = state["positions"].at[new_step].set(new_pos)
            directions = state["directions"].at[new_step].set(new_dir)
            affine = state["affine"].at[new_step].set(new_lam)

            return {
                "step": new_step,
                "pos": new_pos,
                "dir": new_dir,
                "lam": new_lam,
                "step_size": new_step_size,
                "positions": positions,
                "directions": directions,
                "affine": affine,
            }

        final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)

        # Optional mask of valid entries
        mask = jnp.arange(max_steps) <= final_state["step"]
        return (
            final_state["positions"],
            final_state["directions"],
            final_state["affine"],
            mask,
        )

    def _rk4_step_impl(
        self, position: jnp.ndarray, direction: jnp.ndarray, step_size: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Fourth-order Runge-Kutta integration step.

        Args:
            position: Current 4-position [t, x, y, z]
            direction: Current 4-direction [dt/dλ, dx/dλ, dy/dλ, dz/dλ]
            step_size: Integration step size

        Returns:
            Tuple of (new_position, new_direction)
        """

        # RK4 coefficients for position (dx/dλ = v)
        k1_pos = direction
        k1_dir = self._compute_acceleration_impl(position, direction)

        k2_pos = direction + 0.5 * step_size * k1_dir
        k2_dir = self._compute_acceleration_impl(
            position + 0.5 * step_size * k1_pos, direction + 0.5 * step_size * k1_dir
        )
        k3_pos = direction + 0.5 * step_size * k2_dir
        k3_dir = self._compute_acceleration_impl(
            position + 0.5 * step_size * k2_pos, direction + 0.5 * step_size * k2_dir
        )
        k4_pos = direction + step_size * k3_dir
        k4_dir = self._compute_acceleration_impl(
            position + step_size * k3_pos, direction + step_size * k3_dir
        )
        # Combine RK4 terms
        new_position = position + (step_size / 6.0) * (
            k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos
        )
        new_direction = direction + (step_size / 6.0) * (
            k1_dir + 2 * k2_dir + 2 * k3_dir + k4_dir
        )

        return new_position, new_direction

    def _compute_acceleration_impl(
        self, position: jnp.ndarray, direction: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute the acceleration d²x^μ/dλ² from the geodesic equation.

        Args:
            position: 4-position [t, x, y, z]
            direction: 4-direction [dt/dλ, dx/dλ, dy/dλ, dz/dλ]

        Returns:
            4-acceleration [d²t/dλ², d²x/dλ², d²y/dλ², d²z/dλ²]
        """
        t, x, y, z = position

        # Convert to JAX arrays with proper shape for spacetime methods
        t_arr = jnp.array([t])
        x_arr = jnp.array([x])
        y_arr = jnp.array([y])
        z_arr = jnp.array([z])

        acceleration = jnp.zeros(4)

        # Compute acceleration for each component μ
        for mu in range(4):
            accel_mu = 0.0

            # Sum over Christoffel symbols: -Γ^μ_νρ v^ν v^ρ
            for nu in range(4):
                for rho in range(4):
                    gamma = self.spacetime.christoffel(
                        (mu, nu, rho), t_arr, x_arr, y_arr, z_arr
                    )
                    accel_mu -= gamma[0] * direction[nu] * direction[rho]

            acceleration = acceleration.at[mu].set(accel_mu)

        return acceleration

    def _adaptive_step(
        self, position: jnp.ndarray, direction: jnp.ndarray, current_step_size: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """
        Adaptive step size control using embedded Runge-Kutta method.

        Args:
            position: Current position
            direction: Current direction
            current_step_size: Current step size

        Returns:
            Tuple of (new_position, new_direction, actual_step_size)
        """
        # Take one step with current step size
        pos1, dir1 = self._rk4_step(position, direction, current_step_size)

        # Take two steps with half step size
        pos_half, dir_half = self._rk4_step(position, direction, current_step_size / 2)
        pos2, dir2 = self._rk4_step(pos_half, dir_half, current_step_size / 2)

        # Estimate error
        pos_error = jnp.linalg.norm(pos2 - pos1)
        dir_error = jnp.linalg.norm(dir2 - dir1)
        total_error = pos_error + dir_error

        # branch 1: error too large
        def too_large(_):
            new_step = jnp.maximum(current_step_size * 0.5, self.config.min_step_size)
            return pos2, dir2, new_step

        # branch 2: error very small
        def too_small(_):
            new_step = jnp.minimum(current_step_size * 1.5, self.config.max_step_size)
            return pos1, dir1, new_step

        # branch 3: acceptable
        def acceptable(_):
            return pos1, dir1, current_step_size

        # Adjust step size based on error
        # First conditional: is error > tol?
        return jax.lax.cond(
            total_error > self.config.tolerance,
            too_large,
            lambda _: jax.lax.cond(
                total_error < self.config.tolerance / 10,
                too_small,
                acceptable,
                operand=None,
            ),
            operand=None,
        )

    def _normalize_null_vector(
        self, position: jnp.ndarray, direction: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Normalize a 4-vector to satisfy the null condition g_μν v^μ v^ν = 0.

        This adjusts the time component to ensure the ray is null.

        Args:
            position: 4-position where to evaluate the metric
            direction: 4-direction to normalize

        Returns:
            Normalized null 4-direction
        """
        t, x, y, z = position

        # Get metric at this position
        t_arr = jnp.array([t])
        x_arr = jnp.array([x])
        y_arr = jnp.array([y])
        z_arr = jnp.array([z])

        g = self.spacetime.metric(t_arr, x_arr, y_arr, z_arr)[0]  # Shape (4,4)

        # Extract spatial components of direction
        spatial_dir = direction[1:4]  # [dx/dλ, dy/dλ, dz/dλ]

        # For null geodesics: g_μν v^μ v^ν = 0
        # g_00 (dt/dλ)² + 2 g_0i (dt/dλ)(dx^i/dλ) + g_ij (dx^i/dλ)(dx^j/dλ) = 0

        # Quadratic equation coefficients for dt/dλ
        a = g[0, 0]
        b = 2 * jnp.sum(g[0, 1:4] * spatial_dir)
        c = jnp.sum(
            spatial_dir[jnp.newaxis, :] * g[1:4, 1:4] * spatial_dir[:, jnp.newaxis]
        )

        # Solve quadratic equation (take positive root for forward time)
        discriminant = b**2 - 4 * a * c

        # Handle edge cases
        dt_dlambda = jnp.where(
            jnp.abs(a) > 1e-12,
            (-b + jnp.sqrt(jnp.maximum(0, discriminant))) / (2 * a),
            -c / b,  # Linear case when a ≈ 0
        )

        # Construct normalized direction
        normalized_direction = jnp.array([dt_dlambda, *spatial_dir])

        return normalized_direction

    def verify_null_condition(
        self, position: jnp.ndarray, direction: jnp.ndarray
    ) -> float:
        """
        Verify that a 4-vector satisfies the null condition.

        Args:
            position: 4-position
            direction: 4-direction to check

        Returns:
            Value of g_μν v^μ v^ν (should be 0 for null vectors)
        """
        t, x, y, z = position

        t_arr = jnp.array([t])
        x_arr = jnp.array([x])
        y_arr = jnp.array([y])
        z_arr = jnp.array([z])

        g = self.spacetime.metric(t_arr, x_arr, y_arr, z_arr)[0]

        return jnp.sum(direction[jnp.newaxis, :] * g * direction[:, jnp.newaxis])


# Helper functions for common ray tracing scenarios


def create_camera_ray(
    camera_position: jnp.ndarray, pixel_direction: jnp.ndarray, spacetime: Spacetime
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create initial conditions for a camera ray.

    Args:
        camera_position: Camera position [t, x, y, z]
        pixel_direction: Spatial direction [dx, dy, dz] (will be normalized)
        spacetime: Spacetime geometry

    Returns:
        Tuple of (initial_position, initial_direction) for ray integration
    """
    # Normalize spatial direction
    spatial_norm = jnp.linalg.norm(pixel_direction)
    normalized_spatial = pixel_direction / spatial_norm

    # Create initial 4-direction (time component will be set by null condition)
    initial_direction = jnp.array([1.0, *normalized_spatial])

    # Create integrator to normalize the direction
    integrator = LightRayIntegrator(spacetime)
    normalized_direction = integrator._normalize_null_vector(
        camera_position, initial_direction
    )

    return camera_position, normalized_direction


def create_termination_at_surface(surface_function: Callable) -> Callable:
    """
    Create a termination condition that stops when hitting a surface.

    Args:
        surface_function: Function f(position) that returns 0 at the surface

    Returns:
        Termination condition function
    """

    def termination_condition(position, direction, affine_parameter):
        return surface_function(position) <= 0

    return termination_condition
