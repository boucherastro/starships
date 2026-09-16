import numpy as np
import pytest
import astropy.units as u

from starships.spectrum import (SolidRotationKernel, HotspotWindRotationKernel,
                                 project_to_sky, phase_to_lam_obs)


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


class TestHotspotWindRotationKernel:
    """`HotspotWindRotationKernel` (Chantier A) -- numerical multi-region kernel for a
    planet with a displaced hotspot and a 3-component wind field (solid rotation, an
    equatorial jet, a day-to-night flow). Migrated from
    `starships_analysis/hotspot_wind_kernel/` once validated there; see
    `tutorials/rotation_kernel_examples/` for the full theoretical derivation.

    Two of these tests (`test_hotspot_drifts_east_past_eclipse`,
    `test_day_night_flow_is_redshifted_at_eclipse`) exist specifically because this
    class went through two sign-convention bugs during development (`phase_to_lam_obs`'s
    drift direction, and an overall redshift/blueshift sign flip) that a plain
    symmetric-kernel check like `test_matches_solid_rotation_in_trivial_limit` cannot
    catch -- both bugs left that check passing throughout, since a uniformly-bright,
    symmetric disk has no way to reveal a sign error. Only an asymmetric feature (a
    displaced hotspot, or a phase away from exact eclipse) makes the sign checkable at
    all, which is what these two tests exercise directly against independently-derived
    physical expectations (matching `starships.orbite.rv_theo_t`'s sign convention).
    """

    # Realistic WASP-33b-like scale, matching TestSolidRotationKernel above.
    R_PL = 1.2e8 * u.m
    OMEGA = 2 * np.pi / (1.22 * 24 * 3600.0) / u.s
    RESOLUTION = 70_000

    def test_matches_solid_rotation_in_trivial_limit(self):
        # A negligibly small, centered hotspot with no wind should reduce to plain
        # solid-body rotation -- the combined (hot + cold) kernel should match
        # SolidRotationKernel's closed-form profile at any phase.
        trivial = HotspotWindRotationKernel(
            self.R_PL, self.OMEGA, self.RESOLUTION,
            hotspot_lon=0 * u.deg, hotspot_half_lon=1 * u.deg, hotspot_half_lat=1 * u.deg,
            n_lat=241, n_lon=481,
        )
        solid = SolidRotationKernel(self.R_PL, self.OMEGA, self.RESOLUTION)
        v_grid_solid, ker_solid = solid.get_ker(n_os=5)

        for phase in [0.0, 0.3, 0.5, 0.7]:
            v_grid, (ker_hot, ker_cold) = trivial.get_ker(phase=phase, n_os=5)
            combined = ker_hot + ker_cold
            ker_solid_interp = np.interp(v_grid, v_grid_solid, ker_solid, left=0, right=0)
            ker_solid_interp /= ker_solid_interp.sum()
            # A few % near the disk edge is expected grid/histogram discretization
            # noise (the analytic profile has a vertical tangent there).
            np.testing.assert_allclose(combined, ker_solid_interp, atol=5e-3)

    def test_kernel_sums_to_one(self):
        ker_obj = HotspotWindRotationKernel(
            self.R_PL, self.OMEGA, self.RESOLUTION,
            hotspot_lon=20 * u.deg, hotspot_half_lon=35 * u.deg, hotspot_half_lat=25 * u.deg,
            jet_delta_omega=2 * np.pi / (2.5 * u.day).to('s'), jet_half_lat=15 * u.deg,
            day_night_speed=2000 * u.m / u.s, day_night_lat_onset=40 * u.deg,
            n_lat=81, n_lon=161,
        )
        for phase in [0.0, 0.25, 0.5, 0.75]:
            _, (ker_hot, ker_cold) = ker_obj.get_ker(phase=phase, n_os=5)
            assert ker_hot.sum() + ker_cold.sum() == pytest.approx(1.0, rel=1e-9)

    def test_hotspot_drifts_east_past_eclipse(self):
        # For a tidally-locked, prograde-rotating planet, a fixed surface feature
        # (e.g. the substellar point) must drift toward, and eventually disappear
        # over, the *eastern* limb (sky x -> +1) as phase increases past secondary
        # eclipse (phase=0.5) -- matches starships.orbite.rv_theo_t's sign convention
        # (RV_planet(phase) = +K*sin(2*pi*phase), redshift positive).
        substellar_phi = np.array([0.0])
        substellar_lam = np.array([0.0])
        x_positions = []
        for phase in [0.50, 0.55, 0.60, 0.65, 0.70]:
            lam_obs = phase_to_lam_obs(phase)
            x, _, _, _, visible = project_to_sky(substellar_phi, substellar_lam, lam_obs)
            assert visible[0]
            x_positions.append(x[0])
        assert np.all(np.diff(x_positions) > 0), \
            "substellar point should drift monotonically toward +x (east) past eclipse"

    def test_day_night_flow_is_redshifted_at_eclipse(self):
        # Gas flowing from the visible dayside into the hidden nightside must be
        # net redshifted (receding) when viewed face-on at secondary eclipse.
        no_flow = HotspotWindRotationKernel(
            self.R_PL, self.OMEGA, self.RESOLUTION,
            hotspot_lon=20 * u.deg, hotspot_half_lon=35 * u.deg, hotspot_half_lat=25 * u.deg,
            n_lat=101, n_lon=201,
        )
        with_flow = HotspotWindRotationKernel(
            self.R_PL, self.OMEGA, self.RESOLUTION,
            hotspot_lon=20 * u.deg, hotspot_half_lon=35 * u.deg, hotspot_half_lat=25 * u.deg,
            day_night_speed=3000 * u.m / u.s, day_night_lat_onset=30 * u.deg,
            n_lat=101, n_lon=201,
        )

        def mean_velocity(ker_obj):
            v_grid, (ker_hot, ker_cold) = ker_obj.get_ker(phase=0.5, n_os=5)
            combined = ker_hot + ker_cold
            return np.sum(combined * v_grid)

        assert mean_velocity(no_flow) == pytest.approx(0.0, abs=1.0)
        assert mean_velocity(with_flow) > 500.0  # m/s, well above noise level

    def test_disabled_components_contribute_nothing(self):
        # Default jet_delta_omega=0 and day_night_speed=0 should reduce
        # los_velocity_components to solid rotation alone, regardless of how the
        # (unused) latitude masks are configured.
        ker_obj = HotspotWindRotationKernel(
            self.R_PL, self.OMEGA, self.RESOLUTION,
            hotspot_lon=0 * u.deg, hotspot_half_lon=30 * u.deg, hotspot_half_lat=20 * u.deg,
            n_lat=41, n_lon=81,
        )
        components = ker_obj.los_velocity_components(phase=0.5)
        np.testing.assert_array_equal(components['jet'], 0.0)
        np.testing.assert_array_equal(components['day_night'], 0.0)
        np.testing.assert_allclose(components['total'], components['solid_rotation'])

    def test_velocity_components_sum_to_total(self):
        ker_obj = HotspotWindRotationKernel(
            self.R_PL, self.OMEGA, self.RESOLUTION,
            hotspot_lon=20 * u.deg, hotspot_half_lon=35 * u.deg, hotspot_half_lat=25 * u.deg,
            jet_delta_omega=2 * np.pi / (2.5 * u.day).to('s'), jet_half_lat=15 * u.deg,
            day_night_speed=2000 * u.m / u.s, day_night_lat_onset=40 * u.deg,
            n_lat=41, n_lon=81,
        )
        components = ker_obj.los_velocity_components(phase=0.53)
        np.testing.assert_allclose(
            components['total'],
            components['solid_rotation'] + components['jet'] + components['day_night'],
        )
