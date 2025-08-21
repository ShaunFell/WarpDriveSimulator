import unittest
import jax.numpy as jnp
import numpy as np

from src.abstract.metric import Metric, Christoffel
from src.abstract.metricvariables import Lapse, Shift, SpatialMetric
from src.spacetimes.minkowski import MinkowskiSpacetime


class TestMetric(unittest.TestCase):
    """Test suite for the Metric class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a simple test metric (flat space)
        self.lapse = Lapse(
            value=lambda t, x, y, z: jnp.ones_like(x),
            dx=lambda t, x, y, z: jnp.zeros_like(x),
            dy=lambda t, x, y, z: jnp.zeros_like(x),
            dz=lambda t, x, y, z: jnp.zeros_like(x),
            dt=lambda t, x, y, z: jnp.zeros_like(x),
        )

        self.shift = Shift(
            x=lambda t, x, y, z: jnp.zeros_like(x),
            y=lambda t, x, y, z: jnp.zeros_like(x),
            z=lambda t, x, y, z: jnp.zeros_like(x),
            dx_dx=lambda t, x, y, z: jnp.zeros_like(x),
            dx_dy=lambda t, x, y, z: jnp.zeros_like(x),
            dx_dz=lambda t, x, y, z: jnp.zeros_like(x),
            dx_dt=lambda t, x, y, z: jnp.zeros_like(x),
            dy_dx=lambda t, x, y, z: jnp.zeros_like(x),
            dy_dy=lambda t, x, y, z: jnp.zeros_like(x),
            dy_dz=lambda t, x, y, z: jnp.zeros_like(x),
            dy_dt=lambda t, x, y, z: jnp.zeros_like(x),
            dz_dx=lambda t, x, y, z: jnp.zeros_like(x),
            dz_dy=lambda t, x, y, z: jnp.zeros_like(x),
            dz_dz=lambda t, x, y, z: jnp.zeros_like(x),
            dz_dt=lambda t, x, y, z: jnp.zeros_like(x),
        )

        self.metric3 = SpatialMetric(
            xx=lambda t, x, y, z: jnp.ones_like(x),
            xy=lambda t, x, y, z: jnp.zeros_like(x),
            xz=lambda t, x, y, z: jnp.zeros_like(x),
            yy=lambda t, x, y, z: jnp.ones_like(x),
            yz=lambda t, x, y, z: jnp.zeros_like(x),
            zz=lambda t, x, y, z: jnp.ones_like(x),
            xx_dx=lambda t, x, y, z: jnp.zeros_like(x),
            xx_dy=lambda t, x, y, z: jnp.zeros_like(x),
            xx_dz=lambda t, x, y, z: jnp.zeros_like(x),
            xx_dt=lambda t, x, y, z: jnp.zeros_like(x),
            xy_dx=lambda t, x, y, z: jnp.zeros_like(x),
            xy_dy=lambda t, x, y, z: jnp.zeros_like(x),
            xy_dz=lambda t, x, y, z: jnp.zeros_like(x),
            xy_dt=lambda t, x, y, z: jnp.zeros_like(x),
            xz_dx=lambda t, x, y, z: jnp.zeros_like(x),
            xz_dy=lambda t, x, y, z: jnp.zeros_like(x),
            xz_dz=lambda t, x, y, z: jnp.zeros_like(x),
            xz_dt=lambda t, x, y, z: jnp.zeros_like(x),
            yy_dx=lambda t, x, y, z: jnp.zeros_like(x),
            yy_dy=lambda t, x, y, z: jnp.zeros_like(x),
            yy_dz=lambda t, x, y, z: jnp.zeros_like(x),
            yy_dt=lambda t, x, y, z: jnp.zeros_like(x),
            yz_dx=lambda t, x, y, z: jnp.zeros_like(x),
            yz_dy=lambda t, x, y, z: jnp.zeros_like(x),
            yz_dz=lambda t, x, y, z: jnp.zeros_like(x),
            yz_dt=lambda t, x, y, z: jnp.zeros_like(x),
            zz_dx=lambda t, x, y, z: jnp.zeros_like(x),
            zz_dy=lambda t, x, y, z: jnp.zeros_like(x),
            zz_dz=lambda t, x, y, z: jnp.zeros_like(x),
            zz_dt=lambda t, x, y, z: jnp.zeros_like(x),
        )

        # Create a concrete metric for testing
        class TestMetric(Metric):
            def __init__(self, lapse, shift, metric3):
                super().__init__(lapse, shift, metric3)

        self.metric = TestMetric(self.lapse, self.shift, self.metric3)

        # Test coordinates
        self.t = jnp.array([0.0])
        self.x = jnp.array([1.0])
        self.y = jnp.array([2.0])
        self.z = jnp.array([3.0])

    def test_metric4_shape(self):
        """Test that metric4 returns correct shape"""
        g = self.metric.metric4(self.t, self.x, self.y, self.z)
        expected_shape = (*self.t.shape, 4, 4)
        self.assertEqual(g.shape, expected_shape)

    def test_metric4_signature(self):
        """Test that metric has correct signature (-,+,+,+)"""
        g = self.metric.metric4(self.t, self.x, self.y, self.z)
        # For Minkowski metric, should be diag(-1, 1, 1, 1)
        expected = jnp.array(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(g[0], expected, rtol=1e-10)

    def test_metric4_symmetry(self):
        """Test that metric is symmetric"""
        g = self.metric.metric4(self.t, self.x, self.y, self.z)
        np.testing.assert_allclose(g, jnp.transpose(g, axes=(0, 2, 1)), rtol=1e-10)

    def test_dmetric4_derivatives(self):
        """Test that metric derivatives have correct shape and are zero for flat space"""
        dg_dt = self.metric.dmetric4_dt(self.t, self.x, self.y, self.z)
        dg_dx = self.metric.dmetric4_dx(self.t, self.x, self.y, self.z)
        dg_dy = self.metric.dmetric4_dy(self.t, self.x, self.y, self.z)
        dg_dz = self.metric.dmetric4_dz(self.t, self.x, self.y, self.z)

        expected_shape = (*self.t.shape, 4, 4)
        self.assertEqual(dg_dt.shape, expected_shape)
        self.assertEqual(dg_dx.shape, expected_shape)
        self.assertEqual(dg_dy.shape, expected_shape)
        self.assertEqual(dg_dz.shape, expected_shape)

        # For flat space, all derivatives should be zero
        np.testing.assert_allclose(dg_dt, 0.0, atol=1e-10)
        np.testing.assert_allclose(dg_dx, 0.0, atol=1e-10)
        np.testing.assert_allclose(dg_dy, 0.0, atol=1e-10)
        np.testing.assert_allclose(dg_dz, 0.0, atol=1e-10)

    def test_invalid_coordinate_derivative(self):
        """Test that invalid coordinate direction raises ValueError"""
        with self.assertRaises(ValueError):
            self.metric._compute_4d_metric_derivative(
                "invalid", self.t, self.x, self.y, self.z
            )


class TestChristoffel(unittest.TestCase):
    """Test suite for the Christoffel class"""

    def setUp(self):
        """Set up test fixtures using Minkowski metric"""
        minkowski_spacetime = MinkowskiSpacetime()
        self.t = jnp.array([0.0])
        self.x = jnp.array([1.0])
        self.y = jnp.array([2.0])
        self.z = jnp.array([3.0])

        self.metric = minkowski_spacetime.metric
        self.christoffel = Christoffel(self.metric)

    def test_christoffel_flat_space(self):
        """Test that all Christoffel symbols are zero in flat spacetime"""
        # Test several components
        indices_to_test = [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
            (1, 2, 3),
            (2, 0, 3),
            (3, 1, 2),
            (0, 1, 2),
        ]

        for indices in indices_to_test:
            gamma = self.christoffel.compute_symbol(
                indices, self.t, self.x, self.y, self.z
            )
            np.testing.assert_allclose(
                gamma,
                0.0,
                atol=1e-10,
                err_msg=f"Christoffel symbol {indices} should be zero in flat space",
            )

    def test_christoffel_symmetry(self):
        """Test symmetry property: Γ^μ_νρ = Γ^μ_ρν"""
        # Test some components for symmetry in lower indices
        symmetric_pairs = [
            ([0, 1, 2], [0, 2, 1]),
            ([1, 0, 3], [1, 3, 0]),
            ([2, 1, 3], [2, 3, 1]),
        ]

        for indices1, indices2 in symmetric_pairs:
            gamma1 = self.christoffel.compute_symbol(
                indices1, self.t, self.x, self.y, self.z
            )
            gamma2 = self.christoffel.compute_symbol(
                indices2, self.t, self.x, self.y, self.z
            )
            np.testing.assert_allclose(
                gamma1,
                gamma2,
                rtol=1e-10,
                err_msg=f"Christoffel symbols {indices1} and {indices2} should be equal",
            )

    def test_christoffel_index_bounds(self):
        """Test that valid indices work"""
        # Test boundary indices
        boundary_indices = [[0, 0, 0], [3, 3, 3], [0, 3, 1], [2, 0, 3]]

        for indices in boundary_indices:
            try:
                gamma = self.christoffel.compute_symbol(
                    indices, self.t, self.x, self.y, self.z
                )
                self.assertIsInstance(gamma, jnp.ndarray)
            except Exception as e:
                self.fail(f"Valid indices {indices} raised exception: {e}")


class TestSpacetime(unittest.TestCase):
    """Test suite for the Spacetime abstract class using MinkowskiSpacetime"""

    def setUp(self):
        """Set up test fixtures"""
        self.spacetime = MinkowskiSpacetime()
        self.t = jnp.array([0.0])
        self.x = jnp.array([1.0])
        self.y = jnp.array([2.0])
        self.z = jnp.array([3.0])

    def test_metric_method(self):
        """Test that metric method returns a Metric instance"""
        metric = self.spacetime.metric(self.t, self.x, self.y, self.z)
        self.assertIsInstance(metric, jnp.ndarray)
        self.assertEqual(metric.shape, (1, 4, 4))

    def test_christoffel_method(self):
        """Test that christoffel method returns scalar values"""
        gamma = self.spacetime.christoffel((0, 1, 2), self.t, self.x, self.y, self.z)
        self.assertIsInstance(gamma, jnp.ndarray)
        # For Minkowski, should be zero
        np.testing.assert_allclose(gamma, 0.0, atol=1e-10)

    def test_dmetric_method(self):
        """Test that dmetric method returns tuple of derivatives"""
        derivatives = tuple(
            self.spacetime.dmetric(coord, self.t, self.x, self.y, self.z)
            for coord in ["dt", "dx", "dy", "dz"]
        )
        self.assertIsInstance(derivatives, tuple)
        self.assertEqual(len(derivatives), 4)

        dg_dt, dg_dx, dg_dy, dg_dz = derivatives
        expected_shape = (*self.t.shape, 4, 4)

        self.assertEqual(dg_dt.shape, expected_shape)
        self.assertEqual(dg_dx.shape, expected_shape)
        self.assertEqual(dg_dy.shape, expected_shape)
        self.assertEqual(dg_dz.shape, expected_shape)

        # For Minkowski, all derivatives should be zero
        np.testing.assert_allclose(dg_dt, 0.0, atol=1e-10)
        np.testing.assert_allclose(dg_dx, 0.0, atol=1e-10)
        np.testing.assert_allclose(dg_dy, 0.0, atol=1e-10)
        np.testing.assert_allclose(dg_dz, 0.0, atol=1e-10)

    def test_context_manager(self):
        """Test context manager functionality"""
        with self.spacetime as spacetime:
            self.assertIs(spacetime, self.spacetime)


class TestMinkowskiSpacetime(unittest.TestCase):
    """Test suite specifically for MinkowskiSpacetime"""

    def setUp(self):
        """Set up test fixtures"""
        self.spacetime = MinkowskiSpacetime()

        # Test with multiple points
        self.t = jnp.array([0.0, 1.0, 2.0])
        self.x = jnp.array([1.0, 2.0, 3.0])
        self.y = jnp.array([2.0, 3.0, 4.0])
        self.z = jnp.array([3.0, 4.0, 5.0])

    def test_metric_independence_of_coordinates(self):
        """Test that Minkowski metric is independent of coordinates"""
        metric1 = self.spacetime.metric(
            jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0])
        )
        metric2 = self.spacetime.metric(
            jnp.array([10.0]), jnp.array([5.0]), jnp.array([-3.0]), jnp.array([7.0])
        )

        # Metrics should be the same object (since Minkowski is homogeneous)
        np.testing.assert_array_equal(metric1, metric2)

    def test_minkowski_metric_values(self):
        """Test specific values of Minkowski metric"""
        g = self.spacetime.metric(self.t, self.x, self.y, self.z)

        # Check each point
        for i in range(len(self.t)):
            expected = jnp.array(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            np.testing.assert_allclose(g[i], expected, rtol=1e-10)

    def test_all_christoffel_symbols_zero(self):
        """Test that all Christoffel symbols are zero"""
        # Test all 64 components systematically
        for mu in range(4):
            for nu in range(4):
                for rho in range(4):
                    gamma = self.spacetime.christoffel(
                        (mu, nu, rho), self.t[:1], self.x[:1], self.y[:1], self.z[:1]
                    )
                    np.testing.assert_allclose(
                        gamma,
                        0.0,
                        atol=1e-10,
                        err_msg=f"Γ^{mu}_{nu}{rho} should be zero",
                    )

    def test_vectorized_operations(self):
        """Test that operations work with vectorized inputs"""
        # Test with multiple points simultaneously
        g = self.spacetime.metric(self.t, self.x, self.y, self.z)

        # Should work with all points at once
        self.assertEqual(g.shape, (3, 4, 4))

        # Test derivatives
        dg_dx = self.spacetime.metric.dmetric4_dx(self.t, self.x, self.y, self.z)
        self.assertEqual(dg_dx.shape, (3, 4, 4))
        np.testing.assert_allclose(dg_dx, 0.0, atol=1e-10)
