import numpy as np
import pytest

from starships import logl_grid


class TestNormalizePerExposure:

    def test_removes_median_and_scales_by_std(self):
        """Each exposure column should end up with median~0 and std~1 over
        the noise-region window, regardless of its original offset/scale."""
        vsys_axis = np.linspace(-20, 20, 41)
        is_out = (vsys_axis < -10) | (10 < vsys_axis)

        rng = np.random.default_rng(0)
        raw = rng.normal(loc=0., scale=1., size=(41, 3))
        # Give each exposure a different baseline offset and scale.
        offsets = np.array([5., -3., 100.])
        scales = np.array([1., 4., 20.])
        raw = raw * scales[None, :] + offsets[None, :]
        logl_map_ts = np.ma.masked_invalid(raw)

        normed = logl_grid.normalize_per_exposure(logl_map_ts, vsys_axis, (-10., 10.))

        np.testing.assert_allclose(
            np.ma.median(normed[is_out, :], axis=0), 0., atol=1e-10)
        np.testing.assert_allclose(
            np.ma.std(normed[is_out, :], axis=0), 1., rtol=1e-10)


class TestSelectInOutTrail:

    def test_in_and_out_windows(self):
        """Pixels within speed_limit of the trail count as in-trail, pixels
        beyond limit_out count as out-of-trail, the gap between is excluded."""
        vsys_axis = np.arange(-20., 20.01, 1.)  # -20 .. 20, step 1
        n_exp = 2
        # Column value = vsys index itself, so we can identify which pixels
        # were pooled by the values returned.
        logl_map_ts = np.ma.masked_invalid(
            np.tile(vsys_axis[:, None], (1, n_exp)))

        in_vals, out_vals = logl_grid.select_in_out_trail(
            logl_map_ts, vsys_axis, center_per_exposure=0.,
            speed_limit=3., limit_out=8., both_side=True)

        # In-trail: |v| <= 3, both exposures -> each value appears twice.
        expected_in = sorted(vsys_axis[np.abs(vsys_axis) <= 3.]) * n_exp
        assert sorted(np.ma.compressed(in_vals)) == pytest.approx(sorted(expected_in))

        # Out-of-trail: |v| > 8.
        assert np.all(np.abs(np.ma.compressed(out_vals)) > 8.)

    def test_both_side_false_keeps_only_high_velocity_side(self):
        vsys_axis = np.arange(-20., 20.01, 1.)
        logl_map_ts = np.ma.masked_invalid(np.tile(vsys_axis[:, None], (1, 1)))

        _, out_vals = logl_grid.select_in_out_trail(
            logl_map_ts, vsys_axis, center_per_exposure=0.,
            speed_limit=3., limit_out=8., both_side=False)

        assert np.all(np.ma.compressed(out_vals) > 8.)

    def test_per_exposure_center_array(self):
        """A per-exposure center array shifts the in-trail window independently
        for each exposure (the Kp-scan use case)."""
        vsys_axis = np.arange(-20., 20.01, 1.)
        n_exp = 2
        logl_map_ts = np.ma.masked_invalid(np.tile(vsys_axis[:, None], (1, n_exp)))

        centers = np.array([0., 10.])
        in_vals, _ = logl_grid.select_in_out_trail(
            logl_map_ts, vsys_axis, center_per_exposure=centers,
            speed_limit=1., limit_out=15.)

        # Exposure 0: values near 0 (-1, 0, 1); exposure 1: values near 10 (9, 10, 11).
        assert sorted(np.ma.compressed(in_vals)) == pytest.approx([-1., 0., 1., 9., 10., 11.])

    def test_min_sample_spacing_decimates(self):
        """A coarser min_sample_spacing should strictly reduce the pooled
        sample count relative to no decimation."""
        vsys_axis = np.linspace(-20, 20, 401)  # step 0.1 km/s
        logl_map_ts = np.ma.masked_invalid(np.zeros((401, 1)))

        in_full, out_full = logl_grid.select_in_out_trail(
            logl_map_ts, vsys_axis, 0., speed_limit=5., limit_out=10.)
        in_dec, out_dec = logl_grid.select_in_out_trail(
            logl_map_ts, vsys_axis, 0., speed_limit=5., limit_out=10.,
            min_sample_spacing=2.)

        assert in_dec.size < in_full.size
        assert out_dec.size < out_full.size


class TestTtestAtPoint:

    def test_clear_signal_gives_large_positive_sigma(self):
        rng = np.random.default_rng(1)
        in_vals = rng.normal(loc=5., scale=1., size=200)
        out_vals = rng.normal(loc=0., scale=1., size=2000)

        result = logl_grid.ttest_at_point(in_vals, out_vals)

        assert result['t'] > 0
        assert result['sigma'] > 5.
        assert result['n_in'] == 200
        assert result['n_out'] == 2000

    def test_no_signal_gives_small_sigma(self):
        rng = np.random.default_rng(2)
        in_vals = rng.normal(loc=0., scale=1., size=200)
        out_vals = rng.normal(loc=0., scale=1., size=2000)

        result = logl_grid.ttest_at_point(in_vals, out_vals)

        assert abs(result['sigma']) < 4.  # not a hard bound, just "no huge false positive"

    def test_too_few_samples_returns_nan(self):
        result = logl_grid.ttest_at_point([1.0], [1.0, 2.0, 3.0])
        assert np.isnan(result['sigma'])
        assert result['n_in'] == 1

    def test_equal_var_true_vs_false_differ_on_unequal_variance(self):
        rng = np.random.default_rng(3)
        in_vals = rng.normal(loc=0., scale=1., size=50)
        out_vals = rng.normal(loc=0., scale=10., size=2000)

        student = logl_grid.ttest_at_point(in_vals, out_vals, equal_var=True)
        welch = logl_grid.ttest_at_point(in_vals, out_vals, equal_var=False)

        assert student['p'] != pytest.approx(welch['p'])


class TestTtestMapQuicklookAndFullGrid:
    """Injects a known, non-tilted signal (see logl_grid.py's module note on
    why a correctly Kp-corrected trailing map shows a constant-vsys column)
    and checks both map variants recover it -- same construction as the
    dev-folder demo notebook this was ported from."""

    @pytest.fixture
    def synthetic_grid(self):
        rng = np.random.default_rng(42)
        n_vsys, n_kp, n_exp = 81, 11, 20
        vsys_axis = np.linspace(-40, 40, n_vsys)
        kp_array = np.linspace(150, 250, n_kp)
        kp_true_idx = 5
        vsys_true = 5.0
        phase = np.linspace(0.48, 0.53, n_exp)

        logl_cube = rng.normal(0.0, 1.0, size=(n_vsys, n_kp, n_exp))
        bump = 3.0 * np.exp(-0.5 * ((vsys_axis - vsys_true) / 2.0) ** 2)
        logl_cube[:, kp_true_idx, :] += bump[:, None]
        logl_cube = np.ma.masked_invalid(logl_cube)

        return dict(vsys_axis=vsys_axis, kp_array=kp_array, kp_true_idx=kp_true_idx,
                    kp_true=kp_array[kp_true_idx], vsys_true=vsys_true, phase=phase,
                    logl_cube=logl_cube, noise_rv_limits=(vsys_true - 15., vsys_true + 15.))

    def test_quicklook_recovers_injected_point(self, synthetic_grid):
        g = synthetic_grid
        logl_map_ts = g['logl_cube'][:, g['kp_true_idx'], :]

        t_map, p_map, sigma_map = logl_grid.ttest_map_quicklook(
            g['vsys_axis'], logl_map_ts, g['kp_true'], g['kp_array'], g['phase'],
            g['noise_rv_limits'], speed_limit=3., limit_out=10.)

        i_peak, j_peak = np.unravel_index(np.nanargmax(sigma_map), sigma_map.shape)
        assert g['kp_array'][i_peak] == pytest.approx(g['kp_true'])
        assert g['vsys_axis'][j_peak] == pytest.approx(g['vsys_true'], abs=1.0)

    def test_full_grid_recovers_injected_point(self, synthetic_grid):
        g = synthetic_grid

        t_map, p_map, sigma_map = logl_grid.ttest_map_full_grid(
            g['vsys_axis'], g['kp_array'], g['logl_cube'], g['noise_rv_limits'],
            speed_limit=3., limit_out=10.)

        i_peak, j_peak = np.unravel_index(np.nanargmax(sigma_map), sigma_map.shape)
        assert g['kp_array'][i_peak] == pytest.approx(g['kp_true'])
        assert g['vsys_axis'][j_peak] == pytest.approx(g['vsys_true'], abs=1.0)

    def test_min_sample_spacing_reduces_pixel_count_and_peak_significance(self, synthetic_grid):
        """Pseudo-replication guard: fewer (less inflated) pooled samples at
        the true point, and a lower sigma there.

        Note: with fewer samples per grid point, the *global* argmax over
        the whole map can land one Kp step away from the true value by pure
        noise -- this is the real look-elsewhere effect documented in
        `ttest_map_quicklook`'s "Warns" section (confirmed on real KELT-20b
        data), not a bug, so this test checks the point-test directly at
        the known true location rather than the map's own argmax."""
        g = synthetic_grid
        logl_map_ts = g['logl_cube'][:, g['kp_true_idx'], :]
        normed = logl_grid.normalize_per_exposure(logl_map_ts, g['vsys_axis'], g['noise_rv_limits'])

        in_full, out_full = logl_grid.select_in_out_trail(
            normed, g['vsys_axis'], g['vsys_true'], speed_limit=3., limit_out=10.)
        in_dec, out_dec = logl_grid.select_in_out_trail(
            normed, g['vsys_axis'], g['vsys_true'], speed_limit=3., limit_out=10.,
            min_sample_spacing=5.)

        assert in_dec.size < in_full.size
        assert out_dec.size < out_full.size

        sigma_full = logl_grid.ttest_at_point(in_full, out_full)['sigma']
        sigma_dec = logl_grid.ttest_at_point(in_dec, out_dec)['sigma']
        assert sigma_dec < sigma_full


class TestTtestAtPointAveraged:
    """`ttest_at_point_averaged` is documented as NOT calibrated -- these
    tests lock in that finding (a Monte Carlo null-hypothesis check) rather
    than pretending the function works. If someone improves the combination
    method (see the function's docstring for what wasn't tried), this test
    should start failing and needs updating alongside the fix -- that's the
    point of keeping it, not a test to silence."""

    def test_smoke_runs_and_returns_expected_shape(self):
        rng = np.random.default_rng(7)
        n_vsys, n_exp = 81, 10
        vsys_axis = np.linspace(-20, 20, n_vsys)
        logl_map_ts = np.ma.masked_invalid(rng.normal(0., 1., size=(n_vsys, n_exp)))

        res = logl_grid.ttest_at_point_averaged(
            logl_map_ts, vsys_axis, 0., min_sample_spacing=3.,
            speed_limit=3., limit_out=8.)

        assert np.isfinite(res['sigma'])
        assert res['n_offsets'] >= 1
        assert len(res['sigma_per_offset']) == res['n_offsets']

    def test_miscalibrated_on_correlated_noise(self):
        """Known limitation, not a bug to silently fix here: Stouffer-
        combining the per-offset sigmas assumes they are independent, but
        neighbouring offset groups of *correlated* noise are not -- the
        combination reproduces the same pseudo-replication overconfidence
        `min_sample_spacing` exists to prevent. Single-offset
        `ttest_at_point` stays properly calibrated (std ~ 1) on the exact
        same correlated noise and spacing -- see
        `test_min_sample_spacing_reduces_pixel_count_and_peak_significance`-
        style checks above for that side of the comparison."""
        from scipy.ndimage import gaussian_filter1d

        rng = np.random.default_rng(3)
        n_vsys, n_exp = 401, 30
        vsys_axis = np.linspace(-40, 40, n_vsys)
        corr_len = 3.0  # km/s -- makes the single-offset version well-calibrated

        n_trials = 100
        sigmas = []
        for _ in range(n_trials):
            raw = rng.normal(0., 1., size=(n_vsys, n_exp))
            noise = gaussian_filter1d(raw, sigma=5, axis=0)
            noise /= noise.std()
            noise = np.ma.masked_invalid(noise)
            res = logl_grid.ttest_at_point_averaged(
                noise, vsys_axis, 0., corr_len, speed_limit=3., limit_out=10.)
            sigmas.append(res['sigma'])

        sigmas = np.array(sigmas)
        # A well-calibrated sigma under the null has std ~ 1. This locks in
        # that the averaged version is currently well above that (known,
        # documented limitation) -- not an assertion that >2 is "correct".
        assert sigmas.std() > 2.
