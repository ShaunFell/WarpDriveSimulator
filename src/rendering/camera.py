import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

from src.abstract.spacetime import Spacetime
from src.rendering.nullgeodesic_integrator import LightRayIntegrator, create_camera_ray


class ProjectionType(Enum):
    """Type of camera projection"""

    PERSPECTIVE = "perspective"
    ORTHOGRAPHIC = "orthographic"
    FISHEYE = "fisheye"
    SPHERICAL = "spherical"


@dataclass
class CameraParams:
    """Camera parameters for rendering"""

    # Image dimensions
    width: int = 1920
    height: int = 1080

    # Field of view (in radians)
    fov_horizontal: float = jnp.pi / 3  # 60 degrees
    fov_vertical: Optional[float] = None  # Auto-calculated from aspect ratio

    # Camera projection
    projection: ProjectionType = ProjectionType.PERSPECTIVE

    # Near/far clipping (in affine parameter units)
    near_clip: float = 0.01
    far_clip: float = 1000.0

    # Depth of field (for future extensions)
    aperture: float = 0.0  # 0 = pinhole camera
    focus_distance: float = 10.0


@dataclass
class CameraState:
    """Camera position and orientation in spacetime"""

    position: jnp.ndarray  # [t, x, y, z] camera position

    # Camera orientation vectors (in local spatial frame)
    forward: jnp.ndarray  # Camera forward direction (z-axis)
    up: jnp.ndarray  # Camera up direction (y-axis)
    right: jnp.ndarray  # Camera right direction (x-axis)

    # Velocity of camera (for moving cameras)
    velocity: Optional[jnp.ndarray] = None  # [dt/dτ, dx/dτ, dy/dτ, dz/dτ]


class SpacetimeCamera:
    """
    Camera system for rendering in curved spacetime.

    Handles ray generation for each pixel, taking into account the
    local geometry of spacetime at the camera position.
    """

    def __init__(
        self,
        spacetime: Spacetime,
        camera_params: CameraParams = None,
        integrator_config=None,
    ):
        """
        Initialize the spacetime camera.

        Args:
            spacetime: The spacetime geometry to render in
            camera_params: Camera configuration parameters
            integrator_config: Configuration for the ray integrator
        """
        self.spacetime = spacetime
        self.params = camera_params or CameraParams()

        # Initialize ray integrator
        from src.rendering.geodesic_integrator import IntegrationConfig

        integrator_config = integrator_config or IntegrationConfig()
        self.ray_integrator = LightRayIntegrator(spacetime, integrator_config)

        # Calculate vertical FOV if not provided
        if self.params.fov_vertical is None:
            aspect_ratio = self.params.height / self.params.width
            self.params.fov_vertical = 2 * jnp.arctan(
                jnp.tan(self.params.fov_horizontal / 2) * aspect_ratio
            )

        # JIT compile ray generation for performance
        self._generate_pixel_ray = jax.jit(self._generate_pixel_ray_impl)

    def render_image(
        self,
        camera_state: CameraState,
        scene_function: Callable,
        background_function: Optional[Callable] = None,
    ) -> jnp.ndarray:
        """
        Render a full image from the camera viewpoint.

        Args:
            camera_state: Camera position and orientation
            scene_function: Function that determines pixel color from ray intersection
            background_function: Function for background color (default: black)

        Returns:
            Rendered image as array of shape (height, width, 3) with RGB values
        """
        if background_function is None:
            background_function = lambda ray_pos, ray_dir: jnp.array([0.0, 0.0, 0.0])

        # Initialize image array
        image = jnp.zeros((self.params.height, self.params.width, 3))

        # Ensure camera basis is orthonormal
        camera_state = self._normalize_camera_basis(camera_state)

        # Render each pixel
        for y in range(self.params.height):
            for x in range(self.params.width):
                # Generate ray for this pixel
                ray_pos, ray_dir = self.generate_pixel_ray(camera_state, x, y)

                # Integrate the ray through spacetime
                try:
                    positions, directions, affine_params = (
                        self.ray_integrator.integrate_ray(
                            ray_pos, ray_dir, max_affine_parameter=self.params.far_clip
                        )
                    )

                    # Determine pixel color from scene
                    pixel_color = scene_function(positions, directions, affine_params)

                except Exception:
                    # Ray integration failed, use background
                    pixel_color = background_function(ray_pos, ray_dir)

                # Set pixel color
                image = image.at[y, x].set(pixel_color)

        return image

    def render_image_vectorized(
        self,
        camera_state: CameraState,
        scene_function: Callable,
        background_function: Optional[Callable] = None,
        batch_size: int = 1000,
    ) -> jnp.ndarray:
        """
        Vectorized version of image rendering for better performance.

        Processes pixels in batches to balance memory usage and speed.
        """
        if background_function is None:
            background_function = lambda ray_pos, ray_dir: jnp.array([0.0, 0.0, 0.0])

        # Initialize image
        image = jnp.zeros((self.params.height, self.params.width, 3))
        camera_state = self._normalize_camera_basis(camera_state)

        # Create pixel coordinate arrays
        total_pixels = self.params.height * self.params.width
        pixel_coords = []

        for y in range(self.params.height):
            for x in range(self.params.width):
                pixel_coords.append((x, y))

        # Process in batches
        for batch_start in range(0, total_pixels, batch_size):
            batch_end = min(batch_start + batch_size, total_pixels)
            batch_coords = pixel_coords[batch_start:batch_end]

            # Generate rays for this batch
            batch_colors = []
            for x, y in batch_coords:
                ray_pos, ray_dir = self.generate_pixel_ray(camera_state, x, y)

                try:
                    positions, directions, affine_params = (
                        self.ray_integrator.integrate_ray(
                            ray_pos, ray_dir, max_affine_parameter=self.params.far_clip
                        )
                    )
                    pixel_color = scene_function(positions, directions, affine_params)
                except Exception:
                    pixel_color = background_function(ray_pos, ray_dir)

                batch_colors.append(pixel_color)

            # Update image with batch results
            for i, (x, y) in enumerate(batch_coords):
                image = image.at[y, x].set(batch_colors[i])

        return image

    def generate_pixel_ray(
        self, camera_state: CameraState, pixel_x: int, pixel_y: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate a ray for a specific pixel.

        Args:
            camera_state: Camera position and orientation
            pixel_x, pixel_y: Pixel coordinates (0-indexed)

        Returns:
            Tuple of (ray_position, ray_direction) in spacetime coordinates
        """
        return self._generate_pixel_ray_impl(camera_state, pixel_x, pixel_y)

    def _generate_pixel_ray_impl(
        self, camera_state: CameraState, pixel_x: int, pixel_y: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """JIT-compiled implementation of pixel ray generation."""

        if self.params.projection == ProjectionType.PERSPECTIVE:
            return self._perspective_ray(camera_state, pixel_x, pixel_y)
        elif self.params.projection == ProjectionType.ORTHOGRAPHIC:
            return self._orthographic_ray(camera_state, pixel_x, pixel_y)
        elif self.params.projection == ProjectionType.FISHEYE:
            return self._fisheye_ray(camera_state, pixel_x, pixel_y)
        elif self.params.projection == ProjectionType.SPHERICAL:
            return self._spherical_ray(camera_state, pixel_x, pixel_y)
        else:
            raise ValueError(f"Unsupported projection type: {self.params.projection}")

    def _perspective_ray(
        self, camera_state: CameraState, pixel_x: int, pixel_y: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate perspective projection ray."""

        # Convert pixel coordinates to normalized device coordinates [-1, 1]
        ndc_x = (2.0 * pixel_x / self.params.width) - 1.0
        ndc_y = 1.0 - (2.0 * pixel_y / self.params.height)  # Flip Y for standard coords

        # Convert to camera space coordinates
        aspect_ratio = self.params.width / self.params.height
        cam_x = ndc_x * jnp.tan(self.params.fov_horizontal / 2) * aspect_ratio
        cam_y = ndc_y * jnp.tan(self.params.fov_vertical / 2)
        cam_z = -1.0  # Forward direction (negative z in camera coords)

        # Convert camera space direction to world space
        camera_dir = jnp.array([cam_x, cam_y, cam_z])
        world_dir = self._camera_to_world_direction(camera_state, camera_dir)

        # Create ray using helper function
        return create_camera_ray(camera_state.position, world_dir, self.spacetime)

    def _orthographic_ray(
        self, camera_state: CameraState, pixel_x: int, pixel_y: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate orthographic projection ray."""

        # Convert pixel coordinates to normalized device coordinates
        ndc_x = (2.0 * pixel_x / self.params.width) - 1.0
        ndc_y = 1.0 - (2.0 * pixel_y / self.params.height)

        # In orthographic projection, all rays are parallel
        # Ray origin is offset in camera plane
        aspect_ratio = self.params.width / self.params.height
        offset_x = ndc_x * jnp.tan(self.params.fov_horizontal / 2) * aspect_ratio
        offset_y = ndc_y * jnp.tan(self.params.fov_vertical / 2)

        # Calculate ray origin offset in world coordinates
        offset_world = offset_x * camera_state.right + offset_y * camera_state.up

        ray_origin = camera_state.position + jnp.array([0.0, *offset_world])
        world_dir = camera_state.forward

        return create_camera_ray(ray_origin, world_dir, self.spacetime)

    def _fisheye_ray(
        self, camera_state: CameraState, pixel_x: int, pixel_y: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate fisheye projection ray."""

        # Convert to normalized coordinates centered at image center
        center_x = self.params.width / 2
        center_y = self.params.height / 2

        dx = (pixel_x - center_x) / center_x
        dy = (pixel_y - center_y) / center_y

        # Calculate radius and angle
        r = jnp.sqrt(dx**2 + dy**2)

        if r > 1.0:
            # Outside fisheye circle, return background ray
            world_dir = camera_state.forward
        else:
            # Map radius to polar angle (0 to π/2 for hemisphere)
            theta = r * jnp.pi / 2
            phi = jnp.arctan2(dy, dx)

            # Convert spherical to cartesian in camera space
            cam_x = jnp.sin(theta) * jnp.cos(phi)
            cam_y = jnp.sin(theta) * jnp.sin(phi)
            cam_z = -jnp.cos(theta)  # Forward is -z

            camera_dir = jnp.array([cam_x, cam_y, cam_z])
            world_dir = self._camera_to_world_direction(camera_state, camera_dir)

        return create_camera_ray(camera_state.position, world_dir, self.spacetime)

    def _spherical_ray(
        self, camera_state: CameraState, pixel_x: int, pixel_y: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate spherical (360°) projection ray."""

        # Map pixels to spherical coordinates
        phi = (pixel_x / self.params.width) * 2 * jnp.pi  # Azimuth: 0 to 2π
        theta = (pixel_y / self.params.height) * jnp.pi  # Polar: 0 to π

        # Convert to cartesian in camera space
        cam_x = jnp.sin(theta) * jnp.cos(phi)
        cam_y = jnp.sin(theta) * jnp.sin(phi)
        cam_z = -jnp.cos(theta)

        camera_dir = jnp.array([cam_x, cam_y, cam_z])
        world_dir = self._camera_to_world_direction(camera_state, camera_dir)

        return create_camera_ray(camera_state.position, world_dir, self.spacetime)

    def _camera_to_world_direction(
        self, camera_state: CameraState, camera_direction: jnp.ndarray
    ) -> jnp.ndarray:
        """Convert camera space direction to world space direction."""

        # Camera basis vectors (right, up, forward)
        right = camera_state.right
        up = camera_state.up
        forward = camera_state.forward

        # Transform camera space [x, y, z] to world space
        world_dir = (
            camera_direction[0] * right
            + camera_direction[1] * up
            + camera_direction[2] * forward
        )

        return world_dir

    def _normalize_camera_basis(self, camera_state: CameraState) -> CameraState:
        """Ensure camera basis vectors are orthonormal."""

        # Normalize forward vector
        forward = camera_state.forward / jnp.linalg.norm(camera_state.forward)

        # Make up vector orthogonal to forward and normalize
        up = camera_state.up - jnp.dot(camera_state.up, forward) * forward
        up = up / jnp.linalg.norm(up)

        # Right vector is cross product of forward and up
        right = jnp.cross(forward, up)
        right = right / jnp.linalg.norm(right)

        return CameraState(
            position=camera_state.position,
            forward=forward,
            up=up,
            right=right,
            velocity=camera_state.velocity,
        )

    def look_at(
        self,
        camera_position: jnp.ndarray,
        target_position: jnp.ndarray,
        up_vector: jnp.ndarray = jnp.array([0.0, 0.0, 1.0]),
    ) -> CameraState:
        """
        Create camera state looking at a target position.

        Args:
            camera_position: [t, x, y, z] camera position
            target_position: [t, x, y, z] target to look at
            up_vector: World up direction (spatial coordinates only)

        Returns:
            CameraState with proper orientation
        """
        # Calculate forward direction (from camera to target)
        spatial_camera = camera_position[1:4]  # [x, y, z]
        spatial_target = target_position[1:4]  # [x, y, z]

        forward = spatial_target - spatial_camera
        forward = forward / jnp.linalg.norm(forward)

        # Calculate right vector (forward × up)
        right = jnp.cross(forward, up_vector)
        right = right / jnp.linalg.norm(right)

        # Recalculate up vector (right × forward)
        up = jnp.cross(right, forward)
        up = up / jnp.linalg.norm(up)

        return CameraState(
            position=camera_position, forward=forward, up=up, right=right
        )


# Helper functions for common camera setups


def create_orbital_camera(
    center_position: jnp.ndarray,
    radius: float,
    azimuth: float,
    elevation: float,
    time: float = 0.0,
) -> CameraState:
    """
    Create a camera orbiting around a central point.

    Args:
        center_position: [x, y, z] center of orbit
        radius: Distance from center
        azimuth: Horizontal angle (radians)
        elevation: Vertical angle (radians)
        time: Time coordinate

    Returns:
        CameraState positioned on orbit
    """
    # Calculate camera position in spherical coordinates
    x = center_position[0] + radius * jnp.cos(elevation) * jnp.cos(azimuth)
    y = center_position[1] + radius * jnp.cos(elevation) * jnp.sin(azimuth)
    z = center_position[2] + radius * jnp.sin(elevation)

    camera_pos = jnp.array([time, x, y, z])
    target_pos = jnp.array([time, *center_position])

    # Use look_at to determine orientation
    camera = SpacetimeCamera(None)  # Temporary for look_at method
    return camera.look_at(camera_pos, target_pos)


def create_tracking_camera(
    target_trajectory: Callable[[float], jnp.ndarray],
    camera_offset: jnp.ndarray,
    time: float,
) -> CameraState:
    """
    Create a camera that follows a moving target.

    Args:
        target_trajectory: Function that returns target position [x,y,z] at given time
        camera_offset: Offset from target [dx, dy, dz]
        time: Current time

    Returns:
        CameraState following the target
    """
    target_pos = target_trajectory(time)
    camera_spatial = target_pos + camera_offset

    camera_pos = jnp.array([time, *camera_spatial])
    target_4d = jnp.array([time, *target_pos])

    camera = SpacetimeCamera(None)
    return camera.look_at(camera_pos, target_4d)
