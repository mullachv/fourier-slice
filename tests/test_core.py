"""Tests for core Fourier-slice and projection logic."""
import numpy as np
import pytest

from core import (
    SCIPY_OK,
    make_test_image,
    fft1_centered,
    fft2_centered,
    projection_via_rotation,
    bilinear_sample,
    find_most_different_theta,
    slice_from_F,
)


@pytest.fixture
def small_image():
    """Deterministic small image for fast tests."""
    f, x, y, X, Y = make_test_image(N=32, seed=42)
    return f, x, y, X, Y


class TestMakeTestImage:
    def test_shape_and_range(self, small_image):
        f, x, y, X, Y = small_image
        assert f.shape == (32, 32)
        assert x.shape == (32,)
        assert y.shape == (32,)
        assert X.shape == Y.shape == (32, 32)

    def test_deterministic_for_same_seed(self):
        out1 = make_test_image(N=24, seed=7)
        out2 = make_test_image(N=24, seed=7)
        np.testing.assert_array_almost_equal(out1[0], out2[0])

    def test_different_for_different_seed(self):
        out1 = make_test_image(N=24, seed=1)
        out2 = make_test_image(N=24, seed=2)
        assert not np.allclose(out1[0], out2[0])


class TestFftCentered:
    def test_fft1_centered_real_signal(self):
        x = np.random.randn(16)
        out = fft1_centered(x)
        assert out.shape == (16,)
        assert out.dtype == np.dtype("complex128")

    def test_fft2_centered_real_image(self, small_image):
        f = small_image[0]
        out = fft2_centered(f)
        assert out.shape == f.shape
        assert out.dtype == np.dtype("complex128")

    def test_fft2_roundtrip(self, small_image):
        f = small_image[0]
        F = fft2_centered(f)
        # ifft2(ifftshift(fftshift(fft2(ifftshift(f))))) -> need inverse
        from numpy.fft import ifft2, fftshift, ifftshift
        recovered = np.real(ifftshift(ifft2(ifftshift(F))))
        np.testing.assert_allclose(recovered, f, atol=1e-10)


class TestBilinearSample:
    def test_identity_at_integer_indices(self):
        img = np.random.randn(8, 8)
        xs = np.array([0.0, 2.0, 5.0])
        ys = np.array([0.0, 3.0, 4.0])
        out, valid = bilinear_sample(img, xs, ys)
        assert valid.all()
        np.testing.assert_allclose(out, img[[0, 3, 4], [0, 2, 5]])

    def test_valid_mask_excludes_out_of_bounds(self):
        img = np.ones((4, 4))
        xs = np.array([-0.5, 1.0, 3.5])  # -0.5 and 3.5 out of [0,3]
        ys = np.array([1.0, 1.0, 1.0])
        out, valid = bilinear_sample(img, xs, ys)
        assert valid.sum() == 1
        assert out[valid][0] == 1.0


@pytest.mark.skipif(not SCIPY_OK, reason="SciPy required for projection_via_rotation")
class TestProjectionViaRotation:
    def test_projection_shape(self, small_image):
        f = small_image[0]
        p = projection_via_rotation(f, 0.0)
        assert p.shape == (32,)

    def test_projection_90_orthogonal_to_0(self, small_image):
        f = small_image[0]
        p0 = projection_via_rotation(f, 0.0)
        p90 = projection_via_rotation(f, 90.0)
        assert p0.shape == p90.shape
        # For this test image they should differ
        assert not np.allclose(p0, p90)


@pytest.mark.skipif(not SCIPY_OK, reason="SciPy required for find_most_different_theta")
class TestFindMostDifferentTheta:
    def test_returns_angle_and_correlation(self, small_image):
        f = small_image[0]
        angles = np.arange(0.0, 180.0, 15.0)
        best_theta, best_corr = find_most_different_theta(f, 0.0, angles)
        assert best_theta is not None
        assert -1.0 <= best_corr <= 1.0

    def test_most_different_not_near_reference(self, small_image):
        f = small_image[0]
        angles = np.arange(0.0, 180.0, 5.0)
        best_theta, _ = find_most_different_theta(f, 0.0, angles)
        # Should pick an angle that is not 0° (min correlation => roughly orthogonal)
        assert best_theta is not None
        assert abs(best_theta - 0.0) > 10 and abs(best_theta - 180.0) > 10


class TestSliceFromF:
    def test_slice_shapes(self, small_image):
        f = small_image[0]
        F = fft2_centered(f)
        F_log = np.log1p(np.abs(F))
        freq = np.fft.fftshift(np.fft.fftfreq(32, d=1.0))
        kx, ky, rhs, valid = slice_from_F(F_log, 0.0, freq)
        assert kx.shape == ky.shape == rhs.shape == valid.shape == (32,)

    def test_slice_theta_0_aligned_with_kx(self, small_image):
        f = small_image[0]
        F = fft2_centered(f)
        F_log = np.log1p(np.abs(F))
        freq = np.fft.fftshift(np.fft.fftfreq(32, d=1.0))
        kx, ky, rhs, valid = slice_from_F(F_log, 0.0, freq)
        # θ=0 => ky should be 0
        np.testing.assert_allclose(ky, 0.0, atol=1e-14)
        np.testing.assert_allclose(kx, freq, atol=1e-14)
