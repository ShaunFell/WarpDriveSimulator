import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

from src.abstract.spacetime import Spacetime
from src.rendering.nullgeodesic_integrator import LightRayIntegrator, create_camera_ray
from src.rendering.nullgeodesic_integrator import IntegrationConfig


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
        use_pmap: bool = True,
    ) -> jnp.ndarray:
        """
        Vectorized version of image rendering for better performance.

        Processes pixels in batches to balance memory usage and speed.
        Uses pmap for parallel processing across devices/cores.
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

        if use_pmap and jax.device_count() > 1:
            return self._render_with_pmap(
                camera_state,
                scene_function,
                background_function,
                pixel_coords,
                batch_size,
            )
        else:
            return self._render_sequential_batches(
                camera_state,
                scene_function,
                background_function,
                pixel_coords,
                batch_size,
                image,
            )

    def _render_sequential_batches(
        self,
        camera_state: CameraState,
        scene_function: Callable,
        background_function: Callable,
        pixel_coords: list,
        batch_size: int,
        image: jnp.ndarray,
    ) -> jnp.ndarray:
        """Sequential batch processing (fallback when pmap not available)."""

        total_pixels = len(pixel_coords)

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

    def _render_with_pmap(
        self,
        camera_state: CameraState,
        scene_function: Callable,
        background_function: Callable,
        pixel_coords: list,
        batch_size: int,
    ) -> jnp.ndarray:
        """Parallel batch processing using pmap."""

        # Get number of devices
        num_devices = jax.device_count()
        print(f"Using pmap with {num_devices} devices for parallel rendering")

        # Adjust batch size for parallel processing
        parallel_batch_size = max(batch_size // num_devices, 1)
        total_pixels = len(pixel_coords)

        # Pad pixel coordinates to make them divisible by num_devices
        remainder = total_pixels % num_devices
        if remainder != 0:
            padding_needed = num_devices - remainder
            # Pad with dummy coordinates (will be masked out)
            pixel_coords.extend([(-1, -1)] * padding_needed)

        # Reshape into device batches
        pixels_per_device = len(pixel_coords) // num_devices
        pixel_batches = []

        for device_idx in range(num_devices):
            start_idx = device_idx * pixels_per_device
            end_idx = start_idx + pixels_per_device
            device_pixels = pixel_coords[start_idx:end_idx]
            pixel_batches.append(device_pixels)

        pixel_batches = jnp.array(pixel_batches)

        # Define the parallel batch processing function
        @jax.pmap
        def process_device_batch(device_pixel_batch):
            """Process a batch of pixels on one device."""

            def process_pixel(pixel_pair):
                x = pixel_pair[0].astype(int)
                y = pixel_pair[1].astype(int)

                # Branch: dummy pixels vs. real pixels
                def dummy_branch(_):
                    return jnp.array([0.0, 0.0, 0.0])

                def real_branch(_):
                    ray_pos, ray_dir = self.generate_pixel_ray(camera_state, x, y)

                    positions, directions, affine_params = (
                        self.ray_integrator.integrate_ray(
                            ray_pos, ray_dir, max_affine_parameter=self.params.far_clip
                        )
                    )

                    # You can’t "catch exceptions", so instead you need to make
                    # scene_function & background_function *total* (always return something).
                    # A common trick: use NaN checks or masks to decide.
                    pixel_color = scene_function(positions, directions, affine_params)

                    # Example safeguard: if integration produced NaNs, fall back to background
                    invalid = jnp.any(jnp.isnan(positions))
                    pixel_color = jax.lax.cond(
                        invalid,
                        lambda _: background_function(ray_pos, ray_dir),
                        lambda _: pixel_color,
                        operand=None,
                    )
                    return pixel_color

                return jax.lax.cond(
                    (x < 0) | (y < 0), dummy_branch, real_branch, operand=None
                )

            device_colors = jax.vmap(process_pixel)(device_pixel_batch)
            return device_colors

        # Process all device batches in parallel
        all_colors = process_device_batch(pixel_batches)

        # Reshape results back to image
        image = jnp.zeros((self.params.height, self.params.width, 3))

        # Flatten the parallel results
        flat_colors = all_colors.reshape(-1, 3)
        original_pixel_coords = pixel_coords[:total_pixels]  # Remove padding

        # Fill in the image
        for i, (x, y) in enumerate(original_pixel_coords):
            if x >= 0 and y >= 0:  # Skip dummy pixels
                image = image.at[y, x].set(flat_colors[i])

        return image

    def render_image_chunked_pmap(
        self,
        camera_state: CameraState,
        scene_function: Callable,
        background_function: Optional[Callable] = None,
        chunk_size: int = 64,
    ) -> jnp.ndarray:
        """
        Alternative pmap implementation that processes rectangular chunks.

        This can be more efficient for larger images as it processes
        spatial chunks rather than arbitrary pixel batches.
        """
        if background_function is None:
            background_function = lambda ray_pos, ray_dir: jnp.array([0.0, 0.0, 0.0])

        camera_state = self._normalize_camera_basis(camera_state)

        # Divide image into square chunks
        height, width = self.params.height, self.params.width

        # Calculate chunk grid
        chunks_y = (height + chunk_size - 1) // chunk_size
        chunks_x = (width + chunk_size - 1) // chunk_size

        # Pad to make divisible by number of devices
        num_devices = jax.device_count()
        total_chunks = chunks_y * chunks_x

        print(f"Processing {total_chunks} chunks of size {chunk_size}x{chunk_size}")
        print(f"Using {num_devices} devices")

        # Create chunk coordinates
        chunk_coords = []
        for cy in range(chunks_y):
            for cx in range(chunks_x):
                start_y = cy * chunk_size
                end_y = min(start_y + chunk_size, height)
                start_x = cx * chunk_size
                end_x = min(start_x + chunk_size, width)
                chunk_coords.append((start_y, end_y, start_x, end_x))

        # Pad chunk coordinates
        remainder = len(chunk_coords) % num_devices
        if remainder != 0:
            padding_needed = num_devices - remainder
            chunk_coords.extend([(-1, -1, -1, -1)] * padding_needed)

        # Reshape for pmap
        chunks_per_device = len(chunk_coords) // num_devices
        chunk_batches = []

        for device_idx in range(num_devices):
            start_idx = device_idx * chunks_per_device
            end_idx = start_idx + chunks_per_device
            device_chunks = chunk_coords[start_idx:end_idx]
            chunk_batches.append(device_chunks)

        chunk_batches = jnp.array(chunk_batches)

        @jax.pmap
        def process_chunk_batch(device_chunk_batch):
            """Process a batch of chunks on one device."""
            batch_results = []

            for chunk_coords_array in device_chunk_batch:
                start_y, end_y, start_x, end_x = chunk_coords_array

                # Skip dummy chunks
                if start_y < 0:
                    # Return empty chunk
                    batch_results.append(jnp.zeros((chunk_size, chunk_size, 3)))
                    continue

                # Process this chunk
                chunk_height = int(end_y - start_y)
                chunk_width = int(end_x - start_x)
                chunk_image = jnp.zeros((chunk_height, chunk_width, 3))

                for local_y in range(chunk_height):
                    for local_x in range(chunk_width):
                        global_y = int(start_y + local_y)
                        global_x = int(start_x + local_x)

                        try:
                            ray_pos, ray_dir = self.generate_pixel_ray(
                                camera_state, global_x, global_y
                            )
                            positions, directions, affine_params = (
                                self.ray_integrator.integrate_ray(
                                    ray_pos,
                                    ray_dir,
                                    max_affine_parameter=self.params.far_clip,
                                )
                            )
                            pixel_color = scene_function(
                                positions, directions, affine_params
                            )
                        except Exception:
                            pixel_color = background_function(ray_pos, ray_dir)

                        chunk_image = chunk_image.at[local_y, local_x].set(pixel_color)

                # Pad chunk to standard size for consistent array shapes
                padded_chunk = jnp.zeros((chunk_size, chunk_size, 3))
                padded_chunk = padded_chunk.at[:chunk_height, :chunk_width].set(
                    chunk_image
                )
                batch_results.append(padded_chunk)

            return jnp.array(batch_results)

        # Process all chunks in parallel
        all_chunk_results = process_chunk_batch(chunk_batches)

        # Reconstruct full image
        image = jnp.zeros((height, width, 3))

        # Flatten results and place back into image
        flat_results = all_chunk_results.reshape(-1, chunk_size, chunk_size, 3)
        original_chunks = chunk_coords[:total_chunks]

        for i, (start_y, end_y, start_x, end_x) in enumerate(original_chunks):
            if start_y >= 0:  # Skip dummy chunks
                chunk_height = int(end_y - start_y)
                chunk_width = int(end_x - start_x)
                chunk_data = flat_results[i][:chunk_height, :chunk_width]

                image = image.at[start_y:end_y, start_x:end_x].set(chunk_data)

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
