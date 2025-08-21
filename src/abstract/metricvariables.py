import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable


@dataclass
class Lapse:
    """Lapse function α in 3+1 decomposition"""

    value: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ]  # α(t,x,y,z)
    dx: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray] = (
        lambda t, x, y, z: jnp.zeros_like(x)
    )
    dy: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray] = (
        lambda t, x, y, z: jnp.zeros_like(x)
    )
    dz: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray] = (
        lambda t, x, y, z: jnp.zeros_like(x)
    )
    dt: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray] = (
        lambda t, x, y, z: jnp.zeros_like(x)
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
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    dx_dy: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    dx_dz: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    dx_dt: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    dy_dx: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    dy_dy: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    dy_dz: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    dy_dt: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    dz_dx: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    dz_dy: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    dz_dz: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    dz_dt: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)


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
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    xx_dy: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    xx_dz: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    xx_dt: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    xy_dx: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    xy_dy: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    xy_dz: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    xy_dt: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    xz_dx: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    xz_dy: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    xz_dz: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    xz_dt: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    yy_dx: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    yy_dy: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    yy_dz: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    yy_dt: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    yz_dx: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    yz_dy: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    yz_dz: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    yz_dt: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    zz_dx: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    zz_dy: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    zz_dz: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
    zz_dt: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = lambda t, x, y, z: jnp.zeros_like(x)
