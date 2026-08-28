import numpy as np
import pytest
import astropy.units as u

from starships.spectrum import SolidRotationKernel


class TestSolidRotationKernel:
    """`SolidRotationKernel` (Chantier A Phase 3 follow-up, 2026-08-28) -- the
    phase-independent, uniform-disk rotational-broadening profile used as the
    default 'emission' rotation kernel (`model_sequence.py::_build_default_rotation_kernel`).
    Replaces reusing `CitrusRotationKernel` with a single citrus boundary, which
    turned out to give a kernel that was not symmetric around v=0 and was entirely
    zero for any phase other than exactly 0.0 (found while investigating a request
    for a dedicated class -- see `_build_default_rotation_kernel`'s docstring/comments
    for the full story)."""

    # Realistic WASP-33b-like scale (R_pl in meters, omega for a ~1.22 d tidally
    # locked rotation) -- large enough that v_eq is comparable to/bigger than a
    # typical resolution element, so the kernel exercises the real profile shape
    # rather than CitrusRotationKernel-style fallbacks for degenerate/tiny cases.
    R_PL = 1.2e8 * u.m
    OMEGA = 2 * np.pi / (1.22 * 24 * 3600.0) / u.s
    RESOLUTION = 250_000

    def test_symmetric_around_zero_velocity(self):
        ker_obj = SolidRotationKernel(self.R_PL, self.OMEGA, self.RESOLUTION)
        v_grid, kernel = ker_obj.get_ker(n_os=50, pad=7, norm=False)
        np.testing.assert_allclose(kernel, kernel[::-1], atol=1e-12)
        np.testing.assert_allclose(v_grid, -v_grid[::-1], atol=1e-6)

    def test_matches_closed_form_uniform_disk_profile(self):
        ker_obj = SolidRotationKernel(self.R_PL, self.OMEGA, self.RESOLUTION)
        v_grid, kernel = ker_obj.get_ker(n_os=50, pad=7, norm=False)

        v_eq = self.R_PL.to('m').value * self.OMEGA.to('1/s').value
        expected = np.zeros_like(v_grid)
        visible = np.abs(v_grid) <= v_eq
        expected[visible] = np.sqrt(1 - (v_grid[visible] / v_eq) ** 2)

        np.testing.assert_allclose(kernel, expected)

    def test_zero_beyond_equatorial_velocity(self):
        ker_obj = SolidRotationKernel(self.R_PL, self.OMEGA, self.RESOLUTION)
        v_grid, kernel = ker_obj.get_ker(n_os=50, pad=7, norm=False)

        v_eq = self.R_PL.to('m').value * self.OMEGA.to('1/s').value
        assert np.all(kernel[np.abs(v_grid) > v_eq] == 0.0)
        # And the profile must actually be nonzero somewhere inside v_eq -- an
        # all-zero kernel would trivially (and wrongly) pass the check above.
        assert np.any(kernel[np.abs(v_grid) <= v_eq] > 0.0)

    def test_normalized_sums_to_one(self):
        ker_obj = SolidRotationKernel(self.R_PL, self.OMEGA, self.RESOLUTION)
        _, kernel = ker_obj.get_ker(n_os=50, pad=7, norm=True)
        assert kernel.sum() == pytest.approx(1.0, rel=1e-9)
