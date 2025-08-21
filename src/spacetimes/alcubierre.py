import jax.numpy as jnp
import jax

from src.abstract.metric import Metric, Christoffel
from src.abstract.spacetime import Spacetime
from src.abstract.metricvariables import Lapse, Shift, SpatialMetric


class AlcubierreMetric(Metric):
    """
    Alcubierre warp drive metric from the 1994 paper.

    The metric in 3+1 decomposition has:
    - Lapse function: α = 1 (unit lapse)
    - Shift vector: β^x = -v_s f(r_s), β^y = β^z = 0
    - Spatial metric: γ_ij = δ_ij (flat spatial geometry)

    Where:
    - v_s is the velocity of the warp bubble
    - f(r_s) is the warp function depending on r_s = sqrt((x - x_s(t))^2 + y^2 + z^2)
    - x_s(t) = v_s * t is the trajectory of the bubble center

    The original Alcubierre warp function is:
    f(r_s) = (r_s^2 - R^2) / (r_s^2 + R^2) for a sharp cutoff at radius R
    """

    def __init__(
        self,
        velocity: float = 0.8,
        bubble_radius: float = 10.0,
        bubble_thickness: float = 0.5,
    ):
        """
        Initialize Alcubierre metric.

        Args:
            velocity: Velocity of the warp bubble (v_s in the paper)
            bubble_radius: Characteristic radius of the warp bubble (R in the paper)
        """
        self.velocity = velocity
        self.bubble_radius = bubble_radius
        self.sigma = bubble_thickness

        # Lapse function: α = 1 (unit lapse)
        lapse = Lapse(value=lambda t, x, y, z: jnp.ones_like(x))

        shift = Shift(
            x=self._beta_x,
            y=lambda t, x, y, z: jnp.zeros_like(x),
            z=lambda t, x, y, z: jnp.zeros_like(x),
            # Only set non-zero derivatives
            dx_dx=lambda t, x, y, z: -self.velocity
            * self._dwarp_dr(self._r_s_function(t, x, y, z))
            * self._dr_s_dx(t, x, y, z),
            dx_dy=lambda t, x, y, z: -self.velocity
            * self._dwarp_dr(self._r_s_function(t, x, y, z))
            * self._dr_s_dy(t, x, y, z),
            dx_dz=lambda t, x, y, z: -self.velocity
            * self._dwarp_dr(self._r_s_function(t, x, y, z))
            * self._dr_s_dz(t, x, y, z),
            dx_dt=lambda t, x, y, z: -self.velocity
            * self._dwarp_dr(self._r_s_function(t, x, y, z))
            * self._dr_s_dt(t, x, y, z),
        )

        # Spatial metric: γ_ij = δ_ij (flat spatial geometry)
        metric3 = SpatialMetric(
            xx=lambda t, x, y, z: jnp.ones_like(x),
            xy=lambda t, x, y, z: jnp.zeros_like(x),
            xz=lambda t, x, y, z: jnp.zeros_like(x),
            yy=lambda t, x, y, z: jnp.ones_like(x),
            yz=lambda t, x, y, z: jnp.zeros_like(x),
            zz=lambda t, x, y, z: jnp.ones_like(x),
        )

        super().__init__(lapse, shift, metric3)

    # Shift vector: β^i = (-v_s f(r_s), 0, 0)
    def _beta_x(self, t, x, y, z):
        """x-component of shift vector: β^x = -v_s f(r_s)"""
        r_s = self._r_s_function(t, x, y, z)
        return -self.velocity * self._warp_function(r_s)

    # Define warp function and its derivatives
    def _warp_function(self, r_s):
        """Original Alcubierre warp function f(r_s)"""
        # R = self.bubble_radius
        # return -1 / 2 * (r_s**2 - R**2) / (r_s**2 + R**2) + 1 / 2
        sig = self.sigma
        R = self.bubble_radius
        func = (jnp.tanh(sig * (r_s + R)) - jnp.tanh(sig * (r_s - R))) / (
            2 * jnp.tanh(sig * R)
        )
        return func

    def _dwarp_dr(self, r_s):
        """Derivative of warp function: df/dr_s"""
        R = self.bubble_radius
        sig = self.sigma
        sechplus = 1 / jnp.cosh(sig * (r_s + R))
        sechminus = 1 / jnp.cosh(sig * (r_s - R))
        coth = jnp.cosh(sig * R) / jnp.sinh(sig * R)

        return (1 / 2) * sig * coth * (sechplus**2 - sechminus**2)

    # Distance from bubble center functions
    def _r_s_function(self, t, x, y, z):
        """Distance from bubble center: r_s = sqrt((x - v_s*t)^2 + y^2 + z^2)"""
        x_s = self.velocity * t  # Bubble center position
        return jnp.sqrt((x - x_s) ** 2 + y**2 + z**2)

    def _dr_s_dx(self, t, x, y, z):
        """∂r_s/∂x"""
        x_s = self.velocity * t
        r_s = self._r_s_function(t, x, y, z)
        return (x - x_s) / r_s

    def _dr_s_dy(self, t, x, y, z):
        """∂r_s/∂y"""
        r_s = self._r_s_function(t, x, y, z)
        return y / r_s

    def _dr_s_dz(self, t, x, y, z):
        """∂r_s/∂z"""
        r_s = self._r_s_function(t, x, y, z)
        return z / r_s

    def _dr_s_dt(self, t, x, y, z):
        """∂r_s/∂t"""
        x_s = self.velocity * t
        r_s = self._r_s_function(t, x, y, z)
        return -self.velocity * (x - x_s) / r_s


class AlcubierreSpacetime(Spacetime):
    """
    Alcubierre warp drive spacetime implementation.

    This represents the spacetime geometry around a warp drive bubble
    as described in Alcubierre's 1994 paper. The spacetime allows for
    faster-than-light travel by contracting space in front of the bubble
    and expanding space behind it, while keeping the space inside the
    bubble flat.
    """

    def __init__(
        self, velocity: float = 1.0, bubble_radius: float = 10.0, bubble_thickness=0.5
    ):
        """
        Initialize Alcubierre spacetime.

        Args:
            velocity: Velocity of the warp bubble (can exceed c = 1 in natural units)
            bubble_radius: Characteristic radius of the warp bubble
        """
        super().__init__(AlcubierreMetric(velocity, bubble_radius, bubble_thickness))
        self.velocity = velocity
        self.bubble_radius = bubble_radius

    def bubble_center_position(self, t: float) -> float:
        """
        Get the x-coordinate of the bubble center at time t.

        Args:
            t: Time coordinate

        Returns:
            x-coordinate of bubble center: x_s(t) = v_s * t
        """
        return self.metric._r_s_function(t, 0, 0, 0)

    def warp_function_value(self, x: float, y: float, z: float) -> float:
        """
        Evaluate the warp function f(r_s) at given coordinates.

        Args:
           x, y, z: Spacetime coordinates

        Returns:
            Value of warp function f(r_s)
        """
        return self.metric._warp_function(jnp.sqrt(x**2 + y**2 + z**2))
