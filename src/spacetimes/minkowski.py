import jax.numpy as jnp

from src.abstract.metric import Metric, Christoffel
from src.abstract.spacetime import Spacetime
from src.abstract.metricvariables import Lapse, Shift, SpatialMetric


class MinkowskiMetric(Metric):
    """Minkowski metric in 3+1 decomposition"""

    def __init__(self):
        # Minkowski spacetime: flat spacetime with no curvature
        # α = 1 (unit lapse)
        lapse = Lapse(
            value=lambda t, x, y, z: jnp.ones_like(x),
        )

        # β^i = 0 (no shift)
        shift = Shift(
            x=lambda t, x, y, z: jnp.zeros_like(x),
            y=lambda t, x, y, z: jnp.zeros_like(x),
            z=lambda t, x, y, z: jnp.zeros_like(x),
        )

        # γ_ij = δ_ij (flat spatial metric)
        metric3 = SpatialMetric(
            xx=lambda t, x, y, z: jnp.ones_like(x),
            xy=lambda t, x, y, z: jnp.zeros_like(x),
            xz=lambda t, x, y, z: jnp.zeros_like(x),
            yy=lambda t, x, y, z: jnp.ones_like(x),
            yz=lambda t, x, y, z: jnp.zeros_like(x),
            zz=lambda t, x, y, z: jnp.ones_like(x),
        )

        super().__init__(lapse, shift, metric3)


class MinkowskiSpacetime(Spacetime):
    """Minkowski spacetime implementation"""

    def __init__(self):
        super().__init__(MinkowskiMetric())
