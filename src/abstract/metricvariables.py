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
