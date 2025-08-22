#!/usr/bin/env python3
"""
Alcubierre Warp Drive Spacetime Simulation

This simulation demonstrates the visual effects of an Alcubierre warp drive
on light propagation. It renders the view from a camera positioned behind
the warp bubble, looking at a starfield background.

Simulation Parameters:
- Warp bubble at origin with radius R=10, velocity=0.8c, thickness=0.5
- Camera at x=100 (configurable)
- Starfield background at x=-100
"""

from pathlib import Path
import os, sys

project_root = str(Path(os.getcwd()).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

print(project_root)

from src.utils.device_setup import *

setup_device()

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import time


# Import our spacetime framework
from src.spacetimes.alcubierre import AlcubierreSpacetime
from src.spacetimes.minkowski import MinkowskiSpacetime
from src.rendering.nullgeodesic_integrator import LightRayIntegrator, IntegrationConfig
from src.rendering.camera import (
    SpacetimeCamera,
    CameraParams,
    CameraState,
    ProjectionType,
)


class AlcubierreSimulation:
    """
    Main simulation class for Alcubierre warp drive visualization.
    """

    def __init__(
        self,
        camera_x_position: float = 100.0,
        warp_velocity: float = 0.8,
        warp_radius: float = 10.0,
        warp_thickness: float = 0.5,
        starfield_x_position: float = -100.0,
        image_width: int = 512,
        image_height: int = 512,
    ):
        """
        Initialize the Alcubierre simulation.

        Args:
            camera_x_position: X position of camera (looking toward origin)
            warp_velocity: Velocity of warp bubble (in units of c)
            warp_radius: Radius of warp bubble
            warp_thickness: Thickness parameter of warp bubble transition
            starfield_x_position: X position of background starfield plane
            image_width: Width of rendered image in pixels
            image_height: Height of rendered image in pixels
        """
        self.camera_x = camera_x_position
        self.warp_velocity = warp_velocity
        self.warp_radius = warp_radius
        self.warp_thickness = warp_thickness
        self.starfield_x = starfield_x_position

        # Create spacetimes
        self.alcubierre_spacetime = AlcubierreSpacetime(
            velocity=warp_velocity,
            bubble_radius=warp_radius,
            bubble_thickness=warp_thickness,
        )

        self.minkowski_spacetime = MinkowskiSpacetime()  # For comparison

        # Set up camera parameters
        self.camera_params = CameraParams(
            width=image_width,
            height=image_height,
            fov_horizontal=jnp.pi / 3,  # 60 degrees
            projection=ProjectionType.PERSPECTIVE,
            far_clip=200.0,  # Ensure we can see the starfield
        )

        # Integration configuration optimized for curved spacetime
        self.integration_config = IntegrationConfig(
            step_size=0.1,
            max_steps=2000,
            adaptive_stepping=True,
            tolerance=1e-6,
            min_step_size=1e-3,
            max_step_size=1.0,
        )

        # Create cameras
        self.alcubierre_camera = SpacetimeCamera(
            self.alcubierre_spacetime, self.camera_params, self.integration_config
        )

        self.minkowski_camera = SpacetimeCamera(
            self.minkowski_spacetime, self.camera_params, self.integration_config
        )

        print(f"Alcubierre Simulation initialized:")
        print(f"  Warp velocity: {warp_velocity}c")
        print(f"  Warp radius: {warp_radius}")
        print(f"  Camera position: x={camera_x_position}")
        print(f"  Starfield position: x={starfield_x_position}")
        print(f"  Image size: {image_width}x{image_height}")

    def create_camera_state(self, time: float = 0.0) -> CameraState:
        """
        Create camera state looking from behind the warp bubble toward the starfield.

        Args:
            time: Simulation time

        Returns:
            CameraState positioned appropriately
        """
        # Camera position (behind the warp bubble)
        camera_position = jnp.array([time, self.camera_x, 0.0, 0.0])

        # Look toward the origin (where warp bubble is)
        target_position = jnp.array([time, 0.0, 0.0, 0.0])

        # Use the camera's look_at method
        return self.alcubierre_camera.look_at(
            camera_position,
            target_position,
            up_vector=jnp.array([0.0, 0.0, 1.0]),  # Z is up
        )

    def create_starfield_scene(self, star_density: float = 0.001) -> callable:
        """
        Create a starfield background at the specified x position.

        Args:
            star_density: Probability of star at each ray endpoint

        Returns:
            Scene function for starfield rendering
        """

        def starfield_scene_function(positions, directions, affine_params):
            """
            Determine color based on where ray ends up.
            If ray reaches the starfield plane, show stars.
            """
            if len(positions) == 0:
                return jnp.array([0.0, 0.0, 0.0])  # Black space

            # Check if ray reached the starfield plane
            final_position = positions[-1]
            final_x = final_position[1]

            # If ray reached approximately the starfield position
            if final_x <= self.starfield_x + 1.0:  # Small tolerance
                # Use final spatial position to generate stars
                y, z = final_position[2], final_position[3]

                # Create pseudo-random star pattern
                star_seed = jnp.abs(jnp.sin(y * 12.9898 + z * 78.233) * 43758.5453)
                star_seed = star_seed - jnp.floor(star_seed)  # Get fractional part

                if star_seed < star_density:
                    # Star brightness varies
                    brightness = 0.5 + 0.5 * star_seed / star_density
                    return jnp.array([brightness, brightness, brightness])
                else:
                    return jnp.array([0.05, 0.05, 0.1])  # Dark blue space
            else:
                # Ray didn't reach starfield - empty space
                return jnp.array([0.0, 0.0, 0.0])

        return starfield_scene_function

    def create_warp_field_visualization_scene(self, time: float = 0.0) -> callable:
        """
        Create a scene that visualizes the warp field strength.

        Args:
            time: Current simulation time

        Returns:
            Scene function that colors based on warp field
        """

        def warp_visualization_function(positions, directions, affine_params):
            """Color based on warp field strength along the ray path."""

            if len(positions) == 0:
                return jnp.array([0.0, 0.0, 0.0])

            # Sample warp field strength along the ray
            max_warp = 0.0
            warp_sign = 0.0

            # Check multiple points along ray
            sample_indices = jnp.linspace(
                0, len(positions) - 1, min(10, len(positions))
            ).astype(int)

            for i in sample_indices:
                pos = positions[i]
                t, x, y, z = pos

                # Calculate distance from warp bubble center
                bubble_center_x = self.warp_velocity * t
                r_s = jnp.sqrt((x - bubble_center_x) ** 2 + y**2 + z**2)

                # Get warp function value
                warp_value = self.alcubierre_spacetime.metric._warp_function(r_s)

                if jnp.abs(warp_value) > jnp.abs(max_warp):
                    max_warp = jnp.abs(warp_value)
                    warp_sign = jnp.sign(warp_value)

            # Color mapping based on warp field
            if max_warp > 0.01:  # Significant warp field
                intensity = jnp.minimum(1.0, max_warp * 5.0)

                if warp_sign > 0:
                    # Positive warp (space expansion) - red
                    return jnp.array([intensity, 0.0, 0.0])
                else:
                    # Negative warp (space contraction) - blue
                    return jnp.array([0.0, 0.0, intensity])
            else:
                # No significant warp - show starfield
                return self.create_starfield_scene()(
                    positions, directions, affine_params
                )

        return warp_visualization_function

    def render_comparison(self, time: float = 0.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Render both Alcubierre and Minkowski views for comparison.

        Args:
            time: Simulation time

        Returns:
            Tuple of (alcubierre_image, minkowski_image)
        """
        print(f"Rendering at time t={time:.2f}...")

        # Camera state
        camera_state = self.create_camera_state(time)

        # Create scene
        starfield_scene = self.create_starfield_scene(star_density=0.002)

        start_time = time

        # Render Alcubierre view
        print("  Rendering Alcubierre spacetime...")
        alcubierre_image = self.alcubierre_camera.render_image_vectorized(
            camera_state, starfield_scene, batch_size=500
        )

        # For Minkowski comparison, use same camera but different spacetime
        print("  Rendering Minkowski spacetime...")
        minkowski_camera_state = self.minkowski_camera.look_at(
            camera_state.position,
            jnp.array([time, 0.0, 0.0, 0.0]),
            up_vector=jnp.array([0.0, 0.0, 1.0]),
        )

        minkowski_image = self.minkowski_camera.render_image_vectorized(
            minkowski_camera_state, starfield_scene, batch_size=500
        )

        end_time = time
        print(f"  Rendering completed in {end_time - start_time:.2f}s")

        return alcubierre_image, minkowski_image

    def render_warp_field_visualization(self, time: float = 0.0) -> jnp.ndarray:
        """
        Render visualization showing the warp field strength.

        Args:
            time: Simulation time

        Returns:
            Rendered image showing warp field
        """
        print(f"Rendering warp field visualization at t={time:.2f}...")

        camera_state = self.create_camera_state(time)
        warp_scene = self.create_warp_field_visualization_scene(time)

        return self.alcubierre_camera.render_image_vectorized(
            camera_state, warp_scene, batch_size=500
        )

    def create_animation_frames(
        self, time_range: Tuple[float, float] = (0.0, 10.0), num_frames: int = 50
    ) -> list:
        """
        Create frames for animation showing warp bubble motion.

        Args:
            time_range: (start_time, end_time) for animation
            num_frames: Number of frames to generate

        Returns:
            List of rendered frames
        """
        times = jnp.linspace(time_range[0], time_range[1], num_frames)
        frames = []

        print(f"Creating {num_frames} animation frames...")

        for i, t in enumerate(times):
            print(f"  Frame {i + 1}/{num_frames} (t={t:.2f})")

            # Render both comparison and warp field
            alc_img, mink_img = self.render_comparison(t)
            warp_img = self.render_warp_field_visualization(t)

            frames.append({
                "time": t,
                "alcubierre": alc_img,
                "minkowski": mink_img,
                "warp_field": warp_img,
            })

        return frames

    def save_results(
        self,
        alcubierre_image: jnp.ndarray,
        minkowski_image: jnp.ndarray,
        warp_field_image: jnp.ndarray,
        filename_prefix: str = "alcubierre_simulation",
    ):
        """Save simulation results as images."""

        try:
            # Create comparison plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Alcubierre view
            axes[0].imshow(alcubierre_image, origin="upper")
            axes[0].set_title(
                f"Alcubierre Spacetime\n(v={self.warp_velocity}c, R={self.warp_radius})"
            )
            axes[0].set_xlabel("X pixel")
            axes[0].set_ylabel("Y pixel")

            # Minkowski view
            axes[1].imshow(minkowski_image, origin="upper")
            axes[1].set_title("Minkowski Spacetime\n(flat space comparison)")
            axes[1].set_xlabel("X pixel")
            axes[1].set_ylabel("Y pixel")

            # Warp field visualization
            axes[2].imshow(warp_field_image, origin="upper")
            axes[2].set_title(
                "Warp Field Visualization\n(red=expansion, blue=contraction)"
            )
            axes[2].set_xlabel("X pixel")
            axes[2].set_ylabel("Y pixel")

            plt.tight_layout()
            plt.savefig(
                f"{filename_prefix}_comparison.png", dpi=150, bbox_inches="tight"
            )
            plt.show()

            print(f"Results saved as '{filename_prefix}_comparison.png'")

        except ImportError:
            print("Matplotlib not available, saving as individual numpy arrays")
            np.save(f"{filename_prefix}_alcubierre.npy", alcubierre_image)
            np.save(f"{filename_prefix}_minkowski.npy", minkowski_image)
            np.save(f"{filename_prefix}_warp_field.npy", warp_field_image)


def run_basic_simulation():
    """Run a basic Alcubierre simulation with default parameters."""
    print("=" * 60)
    print("ALCUBIERRE WARP DRIVE SIMULATION")
    print("=" * 60)

    # Create simulation
    sim = AlcubierreSimulation(
        camera_x_position=100.0,
        warp_velocity=0.8,
        warp_radius=10.0,
        warp_thickness=0.5,
        starfield_x_position=-100.0,
        image_width=256,  # Smaller for testing
        image_height=256,
    )

    # Render at t=0 (warp bubble at origin)
    alcubierre_img, minkowski_img = sim.render_comparison(time=0.0)
    warp_field_img = sim.render_warp_field_visualization(time=0.0)

    # Save results
    sim.save_results(alcubierre_img, minkowski_img, warp_field_img)

    # Print analysis
    print("\nSimulation Analysis:")
    print(
        f"Alcubierre image - non-black pixels: {jnp.sum(jnp.any(alcubierre_img > 0.01, axis=2))}"
    )
    print(
        f"Minkowski image - non-black pixels: {jnp.sum(jnp.any(minkowski_img > 0.01, axis=2))}"
    )
    print(
        f"Warp field image - colored pixels: {jnp.sum(jnp.any(warp_field_img > 0.01, axis=2))}"
    )

    return sim, alcubierre_img, minkowski_img, warp_field_img


def run_time_series_simulation():
    """Run simulation at different times to show warp bubble motion."""
    print("\n" + "=" * 60)
    print("TIME SERIES SIMULATION")
    print("=" * 60)

    sim = AlcubierreSimulation(
        camera_x_position=100.0,
        warp_velocity=0.8,
        warp_radius=10.0,
        warp_thickness=0.5,
        image_width=128,  # Smaller for speed
        image_height=128,
    )

    # Render at different times
    times = [0.0, 5.0, 10.0, 15.0]

    for t in times:
        print(f"\nTime t={t:.1f}:")
        print(f"  Warp bubble center at x = {sim.warp_velocity * t:.1f}")

        alc_img, mink_img = sim.render_comparison(t)

        # Quick analysis
        alc_stars = jnp.sum(jnp.any(alc_img > 0.01, axis=2))
        mink_stars = jnp.sum(jnp.any(mink_img > 0.01, axis=2))

        print(f"  Alcubierre stars visible: {alc_stars}")
        print(f"  Minkowski stars visible: {mink_stars}")
        print(f"  Difference: {alc_stars - mink_stars}")


if __name__ == "__main__":
    # Run the simulations
    try:
        # Basic simulation
        simulation, alc_img, mink_img, warp_img = run_basic_simulation()

        # Time series
        run_time_series_simulation()

        print("\n" + "=" * 60)
        print("SIMULATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nThe simulation demonstrates:")
        print("1. Gravitational lensing effects of the Alcubierre warp drive")
        print("2. Comparison with flat Minkowski spacetime")
        print("3. Visualization of the warp field structure")
        print("4. Time evolution of the warp bubble")

    except Exception as e:
        print(f"\nSimulation failed with error: {e}")
        import traceback

        traceback.print_exc()
