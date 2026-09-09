import numpy as np
import pytest
from starships.correlation import calc_logl_BL_ord
from starships import logl_grid


class TestCalcLogLBLOrd:

    def test_known_value(self):
        """Valeur exacte vérifiable à la main.

        flux=[1,2,3], model=[1.5,1.5,1.5], alpha=1, N=3
        R    = 1*1.5 + 2*1.5 + 3*1.5 = 9.0
        s2f  = 1 + 4 + 9             = 14.0
        s2g  = 2.25 * 3              = 6.75
        chi2 = 14 - 2*9 + 6.75      = 2.75
        logL = -3/2 * log(2.75/3)
        """
        flux  = np.array([1.0, 2.0, 3.0])
        model = np.array([1.5, 1.5, 1.5])
        N     = 3
        expected = -N / 2 * np.log(2.75 / N)
        result   = calc_logl_BL_ord(flux, model, N, alpha=1.0)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_nolog_returns_chi2(self):
        """nolog=True doit retourner chi2 directement, pas la logL."""
        flux  = np.array([1.0, 2.0, 3.0])
        model = np.array([1.5, 1.5, 1.5])
        N     = 3
        chi2 = calc_logl_BL_ord(flux, model, N, alpha=1.0, nolog=True)
        np.testing.assert_allclose(chi2, 2.75, rtol=1e-12)

    def test_alpha_optimum_maximizes_logl(self):
        """La logL est maximale à alpha_opt = R/s2g (Brogi & Line eq. 6).

        Pour des données bruitées, le alpha qui maximise la logL est celui
        qui minimise le chi2, soit alpha_opt = sum(f*g) / sum(g²).
        """
        rng   = np.random.default_rng(42)
        flux  = rng.normal(0, 1, 300)
        model = rng.normal(0, 0.7, 300)
        N     = 300
        R     = np.sum(flux * model)
        s2g   = np.sum(model ** 2)
        alpha_opt = R / s2g

        logl_opt = calc_logl_BL_ord(flux, model, N, alpha=alpha_opt)
        for delta in [-0.5, -0.2, 0.2, 0.5]:
            logl_other = calc_logl_BL_ord(flux, model, N, alpha=alpha_opt + delta)
            assert logl_opt > logl_other, (
                f"logL non-maximale à alpha_opt (delta={delta:+.1f}): "
                f"{logl_opt:.4f} <= {logl_other:.4f}"
            )

    def test_consistency_with_chi2map_formulation(self):
        """calc_logl_BL_ord doit donner le même résultat que la formulation get_logl de chi2_map.

        Dans chi2_map.py, get_logl() reçoit des termes pré-calculés :
          f_x_g = sum(f * g)   (cross_terms)
          s2g   = sum(g²)      (squared_terms)
          s2f   = sum(f²)
          N
        La logL BL est : -N/2 * log(chi2/N) où chi2 = s2f - 2α*f_x_g + α²*s2g
        """
        rng   = np.random.default_rng(0)
        flux  = rng.normal(0, 1, 50)
        model = rng.normal(0, 0.8, 50)
        N     = 50
        alpha = 1.0

        logl_correlation = calc_logl_BL_ord(flux, model, N, alpha=alpha)

        f_x_g = np.sum(flux * model)
        s2g   = np.sum(model ** 2)
        s2f   = np.sum(flux ** 2)
        chi2  = s2f - 2 * alpha * f_x_g + alpha ** 2 * s2g
        logl_chi2map = -N / 2 * np.log(chi2 / N)

        np.testing.assert_allclose(logl_correlation, logl_chi2map, rtol=1e-12)

    def test_output_shape_2d(self):
        """Pour un tableau (n_spec, n_pix), la sortie doit avoir la forme (n_spec,)."""
        rng   = np.random.default_rng(1)
        flux  = rng.normal(0, 1, (7, 100))
        model = rng.normal(0, 0.5, (7, 100))
        N     = np.full(7, 100)
        result = calc_logl_BL_ord(flux, model, N, axis=-1)
        assert result.shape == (7,)

    def test_s2f_precomputed_vs_computed(self):
        """Passer s2f pré-calculé doit donner le même résultat que le calcul interne."""
        rng   = np.random.default_rng(5)
        flux  = rng.normal(0, 1, 80)
        model = rng.normal(0, 1, 80)
        N     = 80
        s2f_precomputed = np.sum(flux ** 2)
        logl_auto = calc_logl_BL_ord(flux, model, N)
        logl_pre  = calc_logl_BL_ord(flux, model, N, s2f=s2f_precomputed)
        np.testing.assert_allclose(logl_auto, logl_pre, rtol=1e-12)


class TestLoglFromChi2Terms:
    """`logl_grid.py::_logl_from_chi2_terms` (Chantier A Phase 2) -- coeur logL
    extrait de `get_logl`, sans dépendance aux globals du module, pour que
    `retrieval.py::lnprob` puisse le réutiliser tel quel plutôt que de dépendre de
    l'ancien `correlation.py::calc_logl_BL_ord`. Les deux formules sont
    mathématiquement identiques (voir `TestCalcLogLBLOrd.test_consistency_with_chi2map_formulation`
    ci-dessus) -- ce test vérifie seulement que l'extraction elle-même n'a rien cassé."""

    def test_matches_calc_logl_bl_ord(self):
        """Même exemple à la main que TestCalcLogLBLOrd.test_known_value."""
        flux  = np.array([1.0, 2.0, 3.0])
        model = np.array([1.5, 1.5, 1.5])
        N = 3
        ct = np.sum(flux * model)
        st = np.sum(model ** 2)
        sf = np.sum(flux ** 2)

        result = logl_grid._logl_from_chi2_terms(ct, st, sf, N, alpha=1.0, kind='BL')
        expected = calc_logl_BL_ord(flux, model, N, alpha=1.0)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError):
            logl_grid._logl_from_chi2_terms(1.0, 1.0, 1.0, 3, kind='not-a-kind')


class TestGetLoglRefactor:
    """`get_logl` (Chantier A Phase 2 refactor: now delegates to
    `_logl_from_chi2_terms` instead of duplicating the chi2->logL formula inline) --
    verifies the refactor didn't change its output, for both the scalar-alpha and the
    vectorised-alpha-array code paths."""

    def _set_grid_globals(self, monkeypatch, ct, st, sf, N, uncert_sum=None):
        # get_logl() reads its terms from module globals (fork-multiprocessing
        # pattern, see module docstring) -- monkeypatch them directly rather than
        # going through the full setup_logl_grid()/compute_logl_grid() workflow.
        # `uncert_sum` is indexed unconditionally (regardless of `kind`), so it must
        # be a valid array even for a 'BL' test that never actually uses its values.
        if uncert_sum is None:
            uncert_sum = np.ma.zeros(np.shape(N))
        monkeypatch.setattr(logl_grid, 'cross_terms', ct, raising=False)
        monkeypatch.setattr(logl_grid, 'squared_terms', st, raising=False)
        monkeypatch.setattr(logl_grid, 's2f', sf, raising=False)
        monkeypatch.setattr(logl_grid, 'N', N, raising=False)
        monkeypatch.setattr(logl_grid, 'uncert_sum', uncert_sum, raising=False)

    def test_scalar_alpha_matches_hand_computed_value(self, monkeypatch):
        """Même exemple à la main que TestCalcLogLBLOrd.test_known_value, mais à
        travers get_logl() (termes déjà "sommés sur les pixels", shape (1, 1) pour
        1 exposition/1 ordre)."""
        flux  = np.array([1.0, 2.0, 3.0])
        model = np.array([1.5, 1.5, 1.5])
        ct = np.array([[np.sum(flux * model)]])
        st = np.array([[np.sum(model ** 2)]])
        sf = np.array([[np.sum(flux ** 2)]])
        N = np.array([[3]])
        self._set_grid_globals(monkeypatch, ct, st, sf, N)

        result = logl_grid.get_logl(alpha=1.0, kind='BL')
        expected = -3 / 2 * np.log(2.75 / 3)
        np.testing.assert_allclose(result[0, 0], expected, rtol=1e-12)

    def test_vectorised_alpha_matches_scalar_loop(self, monkeypatch):
        """Le chemin vectorisé (alpha = tableau + sum_axis) doit reproduire, pour
        chaque valeur d'alpha, le même résultat que l'appel scalaire équivalent."""
        rng = np.random.default_rng(3)
        n_exp, n_ord = 4, 2
        ct = np.ma.array(rng.normal(size=(n_exp, n_ord)))
        st = np.ma.array(np.abs(rng.normal(size=(n_exp, n_ord))) + 0.1)
        sf = np.ma.array(np.abs(rng.normal(size=(n_exp, n_ord))) + 0.1)
        N = np.ma.array(np.full((n_exp, n_ord), 100))
        self._set_grid_globals(monkeypatch, ct, st, sf, N)

        alpha_array = np.array([0.5, 1.0, 1.5])
        result_vectorised = logl_grid.get_logl(alpha=alpha_array, kind='BL', sum_axis=(-2, -1))

        for i, a in enumerate(alpha_array):
            result_scalar = logl_grid.get_logl(alpha=float(a), kind='BL', sum_axis=(-2, -1))
            np.testing.assert_allclose(result_vectorised[i], result_scalar, rtol=1e-10)


def _set_grid_globals(monkeypatch, ct, st, sf, N, vsys_axis, kp_axis):
    """Monkeypatch the logl_grid module globals a synthetic (vsys, Kp) grid
    needs -- shared by TestDetectionStatistics and TestPosteriorFromLogMap."""
    uncert_sum = np.ma.zeros(np.shape(N))
    monkeypatch.setattr(logl_grid, 'cross_terms', ct, raising=False)
    monkeypatch.setattr(logl_grid, 'squared_terms', st, raising=False)
    monkeypatch.setattr(logl_grid, 's2f', sf, raising=False)
    monkeypatch.setattr(logl_grid, 'N', N, raising=False)
    monkeypatch.setattr(logl_grid, 'uncert_sum', uncert_sum, raising=False)
    monkeypatch.setattr(logl_grid, 'vsys_axis', vsys_axis, raising=False)
    monkeypatch.setattr(logl_grid, 'kp_axis', kp_axis, raising=False)


def _make_signal_grid(alpha_true=2.0, sf_val=100.0, n_vsys=5, n_kp=4):
    """(n_vsys, n_kp, 1, 1) grid, ct=0 (noise) everywhere except one grid
    point where ct = alpha_true * st (so alpha_opt = ct/st = alpha_true there)."""
    vsys_axis = np.linspace(-50., 50., n_vsys)
    kp_axis = np.linspace(100., 250., n_kp)
    i_v0, i_k0 = 3, 1  # arbitrary "true" location, not a grid edge/corner

    st = np.ones((n_vsys, n_kp, 1, 1))
    ct = np.zeros((n_vsys, n_kp, 1, 1))
    ct[i_v0, i_k0] = alpha_true * st[i_v0, i_k0]
    sf = np.full((n_vsys, n_kp, 1, 1), sf_val)
    N = np.full((n_vsys, n_kp, 1, 1), 100)

    return ct, st, sf, N, vsys_axis, kp_axis, i_v0, i_k0


class TestDetectionStatistics:
    """`compute_alpha_significance` / `compute_kpvsys_bayes_factor` (alpha
    fixed vs. marginalized detection statistics, see Notes/plan_revision_starships.md
    Chantier C). Synthetic (vsys, Kp) grids with a known injected signal at a
    single grid point, everywhere else pure noise (ct=0)."""

    def test_alpha_significance_recovers_injected_signal(self, monkeypatch):
        """`compute_alpha_significance` should locate the injected (vsys, Kp)
        and recover alpha_best close to the injected alpha_true, with a
        moderate (finite, non-trivial) detection significance."""
        alpha_true = 2.0
        ct, st, sf, N, vsys_axis, kp_axis, i_v0, i_k0 = _make_signal_grid(alpha_true)
        _set_grid_globals(monkeypatch, ct, st, sf, N, vsys_axis, kp_axis)

        result = logl_grid.compute_alpha_significance(idx_signal=np.array([0]))

        assert result['vsys_best'] == pytest.approx(vsys_axis[i_v0])
        assert result['kp_best'] == pytest.approx(kp_axis[i_k0])
        assert result['alpha_best'] == pytest.approx(alpha_true, abs=0.1)
        assert result['D'] > 0.
        assert 0. < result['p_value'] < 1.
        assert result['sigma'] > 0.

    def test_alpha_significance_no_signal_gives_zero_statistic(self, monkeypatch):
        """Pure noise (ct=0 everywhere) should give D=0 (alpha_best pinned at
        the lower bound) and p_value=0.5 (the Chernoff mixture's point mass),
        i.e. no detection."""
        ct, st, sf, N, vsys_axis, kp_axis, _, _ = _make_signal_grid(alpha_true=0.)
        _set_grid_globals(monkeypatch, ct, st, sf, N, vsys_axis, kp_axis)

        result = logl_grid.compute_alpha_significance(idx_signal=np.array([0]))

        assert result['D'] == pytest.approx(0., abs=1e-6)
        assert result['p_value'] == pytest.approx(0.5, abs=1e-6)

    def test_alpha_significance_map_matches_scalar_at_peak(self, monkeypatch):
        """`compute_alpha_significance_map`'s peak must agree exactly with
        `compute_alpha_significance` -- both are thin extractions of the
        same shared `_alpha_significance_full` core."""
        ct, st, sf, N, vsys_axis, kp_axis, i_v0, i_k0 = _make_signal_grid(alpha_true=2.0)
        _set_grid_globals(monkeypatch, ct, st, sf, N, vsys_axis, kp_axis)

        scalar_result = logl_grid.compute_alpha_significance(idx_signal=np.array([0]))
        map_result = logl_grid.compute_alpha_significance_map(idx_signal=np.array([0]))

        assert map_result['sigma_map'].shape == (len(vsys_axis), len(kp_axis))
        assert map_result['alpha_map'].shape == (len(vsys_axis), len(kp_axis))
        assert map_result['vsys_best'] == pytest.approx(scalar_result['vsys_best'])
        assert map_result['kp_best'] == pytest.approx(scalar_result['kp_best'])
        assert map_result['alpha_best'] == pytest.approx(scalar_result['alpha_best'])
        assert map_result['peak_sigma'] == pytest.approx(scalar_result['sigma'])
        assert map_result['logl_null'] == pytest.approx(scalar_result['logl_null'])

        i_v, i_k = i_v0, i_k0
        assert map_result['sigma_map'][i_v, i_k] == pytest.approx(map_result['peak_sigma'])
        assert map_result['alpha_map'][i_v, i_k] == pytest.approx(2.0, abs=1e-8)

    def test_alpha_significance_map_no_signal_is_flat_zero(self, monkeypatch):
        """Pure noise everywhere (ct=0) should give sigma ~= 0 at EVERY grid
        point, not just the reported peak."""
        ct, st, sf, N, vsys_axis, kp_axis, _, _ = _make_signal_grid(alpha_true=0.)
        _set_grid_globals(monkeypatch, ct, st, sf, N, vsys_axis, kp_axis)

        map_result = logl_grid.compute_alpha_significance_map(idx_signal=np.array([0]))

        np.testing.assert_allclose(map_result['sigma_map'], 0., atol=1e-6)

    def test_bayes_factor_occam_correction_is_range_invariant_for_flat_likelihood(
        self, monkeypatch
    ):
        """When the likelihood does not depend on alpha at all (ct=st=0, so
        chi2=sf regardless of alpha), the alpha-marginalized evidence must
        equal the (alpha-independent) logL exactly, for ANY alpha_array range.

        This directly tests the Occam-factor normalization in
        `compute_kpvsys_bayes_factor` (dividing by `alpha_range`): without it,
        log_bf would scale with log(alpha_range) instead of being invariant.
        """
        n_vsys, n_kp = 3, 3
        vsys_axis = np.linspace(-10., 10., n_vsys)
        kp_axis = np.linspace(100., 200., n_kp)
        ct = np.zeros((n_vsys, n_kp, 1, 1))
        st = np.zeros((n_vsys, n_kp, 1, 1))
        sf = np.full((n_vsys, n_kp, 1, 1), 100.)
        N = np.full((n_vsys, n_kp, 1, 1), 100)
        _set_grid_globals(monkeypatch, ct, st, sf, N, vsys_axis, kp_axis)

        bf_narrow = logl_grid.compute_kpvsys_bayes_factor(
            alpha_array=np.linspace(0.01, 2., 31), idx_signal=np.array([0]),
        )
        bf_wide = logl_grid.compute_kpvsys_bayes_factor(
            alpha_array=np.linspace(0.01, 5., 31), idx_signal=np.array([0]),
        )

        # No signal anywhere -> essentially no evidence either way, and (crucially)
        # the SAME regardless of the alpha prior's range.
        assert bf_narrow['log_bf'] == pytest.approx(0., abs=1e-6)
        assert bf_wide['log_bf'] == pytest.approx(0., abs=1e-6)
        assert bf_narrow['log_bf'] == pytest.approx(bf_wide['log_bf'], abs=1e-6)

    def test_bayes_factor_widening_prior_reduces_evidence_for_real_signal(self, monkeypatch):
        """With a genuine, well-localized signal, widening the alpha prior's
        range should REDUCE log(BF) (Occam penalty for a vaguer alternative),
        not leave it unchanged -- the opposite of the flat-likelihood case
        above, and the whole point of reporting a sensitivity check in the
        tutorial notebook rather than a single log_bf number."""
        ct, st, sf, N, vsys_axis, kp_axis, _, _ = _make_signal_grid(alpha_true=1.0)
        _set_grid_globals(monkeypatch, ct, st, sf, N, vsys_axis, kp_axis)

        bf_narrow = logl_grid.compute_kpvsys_bayes_factor(
            alpha_array=np.linspace(0.01, 2., 31), idx_signal=np.array([0]),
        )
        bf_wide = logl_grid.compute_kpvsys_bayes_factor(
            alpha_array=np.linspace(0.01, 20., 31), idx_signal=np.array([0]),
        )

        assert bf_wide['log_bf'] < bf_narrow['log_bf']

    def test_bayes_factor_vsys_kp_bounds_covering_peak_increase_evidence(self, monkeypatch):
        """Restricting vsys/Kp to a smaller sub-range that still fully covers
        the injected signal should INCREASE log(BF) relative to the full grid
        (same peak, smaller search volume -> less Occam dilution) -- the same
        mechanism as narrowing alpha_array, just applied to vsys/Kp."""
        ct, st, sf, N, vsys_axis, kp_axis, i_v0, i_k0 = _make_signal_grid(alpha_true=1.0)
        _set_grid_globals(monkeypatch, ct, st, sf, N, vsys_axis, kp_axis)
        # vsys_axis = [-50, -25, 0, 25, 50], kp_axis = [100, 150, 200, 250];
        # i_v0=3 (vsys_true=25), i_k0=1 (kp_true=150). Bounds below keep >= 3
        # grid points on each axis (Simpson needs more than a single point)
        # while still covering the injected peak and shrinking the range.

        bf_full = logl_grid.compute_kpvsys_bayes_factor(idx_signal=np.array([0]))
        bf_restricted = logl_grid.compute_kpvsys_bayes_factor(
            idx_signal=np.array([0]),
            vsys_bounds=(-1., 50.),   # keeps [0, 25, 50], range 50 vs full 100
            kp_bounds=(100., 200.),   # keeps [100, 150, 200], range 100 vs full 150
        )

        assert bf_restricted['vsys_range'] < bf_full['vsys_range']
        assert bf_restricted['kp_range'] < bf_full['kp_range']
        assert bf_restricted['log_bf'] > bf_full['log_bf']

    def test_bayes_factor_vsys_bounds_excluding_peak_is_truncation_not_evidence(
        self, monkeypatch
    ):
        """Restricting vsys to a sub-range that EXCLUDES the injected signal
        should give a much lower log(BF) than the full grid -- not because
        the signal is weaker, but because the integral no longer sees it at
        all (truncation, not a genuine Occam effect -- see the docstring)."""
        ct, st, sf, N, vsys_axis, kp_axis, i_v0, i_k0 = _make_signal_grid(alpha_true=1.0)
        _set_grid_globals(monkeypatch, ct, st, sf, N, vsys_axis, kp_axis)
        vsys_true = vsys_axis[i_v0]

        bf_full = logl_grid.compute_kpvsys_bayes_factor(idx_signal=np.array([0]))
        # vsys_axis = [-50, -25, 0, 25, 50], i_v0=3 -> vsys_true=25; excise it.
        bf_excluding_peak = logl_grid.compute_kpvsys_bayes_factor(
            idx_signal=np.array([0]), vsys_bounds=(vsys_axis[0], vsys_true - 1.),
        )

        assert bf_excluding_peak['log_bf'] < bf_full['log_bf']


class TestEmpiricalSigmaMap:
    """`compute_empirical_sigma_map` (box / sigma-clip noise-floor estimate,
    ported from `plotting_fcts.calculate_KpVsys_map` on the `mathis` branch --
    Chantier F review). Only needs `vsys_axis`/`kp_axis` monkeypatched, not
    the full cross_terms/squared_terms grid globals."""

    def _grid_with_corner_peak(self):
        """(5, 4) map: a deterministic "quiet" region in the top-left 4x3
        block, zeros elsewhere, and one large peak at (vsys=4, kp=130)."""
        vsys_axis = np.array([0., 1., 2., 3., 4.])
        kp_axis = np.array([100., 110., 120., 130.])
        map_2d = np.array([
            [1., 2., 3., 0.],
            [2., 3., 4., 0.],
            [3., 4., 5., 0.],
            [4., 5., 6., 0.],
            [0., 0., 0., 50.],
        ])
        return map_2d, vsys_axis, kp_axis

    def test_box_method_matches_manual_calculation(self, monkeypatch):
        map_2d, vsys_axis, kp_axis = self._grid_with_corner_peak()
        monkeypatch.setattr(logl_grid, 'vsys_axis', vsys_axis, raising=False)
        monkeypatch.setattr(logl_grid, 'kp_axis', kp_axis, raising=False)

        result = logl_grid.compute_empirical_sigma_map(
            map_2d, method='box', box_vsys=(0., 3.), box_kp=(100., 120.),
        )

        expected_submap = map_2d[:4, :3]   # the quiet region the box selects
        expected_noise_std = np.std(expected_submap)
        expected_median = np.median(map_2d)
        expected_sigma_map = (map_2d - expected_median) / expected_noise_std

        assert result['noise_std'] == pytest.approx(expected_noise_std)
        np.testing.assert_allclose(result['sigma_map'], expected_sigma_map)
        assert result['peak_sigma'] == pytest.approx(expected_sigma_map[4, 3])
        assert result['vsys_best'] == pytest.approx(4.)
        assert result['kp_best'] == pytest.approx(130.)

    def test_box_method_requires_bounds(self, monkeypatch):
        map_2d, vsys_axis, kp_axis = self._grid_with_corner_peak()
        monkeypatch.setattr(logl_grid, 'vsys_axis', vsys_axis, raising=False)
        monkeypatch.setattr(logl_grid, 'kp_axis', kp_axis, raising=False)

        with pytest.raises(ValueError):
            logl_grid.compute_empirical_sigma_map(map_2d, method='box')

    def test_invalid_method_raises(self, monkeypatch):
        map_2d, vsys_axis, kp_axis = self._grid_with_corner_peak()
        monkeypatch.setattr(logl_grid, 'vsys_axis', vsys_axis, raising=False)
        monkeypatch.setattr(logl_grid, 'kp_axis', kp_axis, raising=False)

        with pytest.raises(ValueError):
            logl_grid.compute_empirical_sigma_map(map_2d, method='not-a-method')

    def test_clip_method_excludes_peak_from_noise_estimate(self, monkeypatch):
        """Sigma-clipping should remove the lone peak from the noise-floor
        estimate, so the clipped std is much smaller than the raw std of the
        whole map -- and the peak should stand out with a large peak_sigma
        at the correct (vsys, Kp) location."""
        map_2d, vsys_axis, kp_axis = self._grid_with_corner_peak()
        monkeypatch.setattr(logl_grid, 'vsys_axis', vsys_axis, raising=False)
        monkeypatch.setattr(logl_grid, 'kp_axis', kp_axis, raising=False)

        result = logl_grid.compute_empirical_sigma_map(map_2d, method='clip')

        assert result['noise_std'] < np.std(map_2d)
        assert result['peak_sigma'] > 5.
        assert result['vsys_best'] == pytest.approx(4.)
        assert result['kp_best'] == pytest.approx(130.)

    def test_mismatched_map_shape_raises_clear_error(self, monkeypatch):
        """A map on a different grid than the module's native vsys_axis/kp_axis
        (e.g. an oversampled posterior from compute_kpvsys_posterior) must not
        silently index out of bounds -- it should raise a clear ValueError
        pointing at vsys_coords/kp_coords instead."""
        map_2d, vsys_axis, kp_axis = self._grid_with_corner_peak()
        monkeypatch.setattr(logl_grid, 'vsys_axis', vsys_axis, raising=False)
        monkeypatch.setattr(logl_grid, 'kp_axis', kp_axis, raising=False)

        oversampled_map = np.zeros((10, 8))   # different shape than the (5, 4) native grid
        with pytest.raises(ValueError):
            logl_grid.compute_empirical_sigma_map(oversampled_map, method='clip')

    def test_explicit_vsys_kp_coords_override_module_globals(self, monkeypatch):
        """Passing vsys_coords/kp_coords explicitly (as one would with an
        oversampled compute_kpvsys_posterior output) must be used instead of
        the module's native vsys_axis/kp_axis, including for peak location."""
        map_2d, native_vsys_axis, native_kp_axis = self._grid_with_corner_peak()
        # Native globals deliberately different/wrong-shaped, to prove they're unused.
        monkeypatch.setattr(logl_grid, 'vsys_axis', np.array([0., 1.]), raising=False)
        monkeypatch.setattr(logl_grid, 'kp_axis', np.array([0., 1.]), raising=False)

        result = logl_grid.compute_empirical_sigma_map(
            map_2d, method='clip', vsys_coords=native_vsys_axis, kp_coords=native_kp_axis,
        )

        assert result['vsys_best'] == pytest.approx(4.)
        assert result['kp_best'] == pytest.approx(130.)

    def test_non_finite_values_excluded_from_noise_and_peak(self, monkeypatch):
        """-inf pixels (e.g. from np.log(posterior) underflow) must not
        poison the noise-floor estimate or be reportable as the peak."""
        map_2d, vsys_axis, kp_axis = self._grid_with_corner_peak()
        map_2d = map_2d.copy()
        map_2d[0, 0] = -np.inf   # a single underflowed pixel, far from the real peak
        monkeypatch.setattr(logl_grid, 'vsys_axis', vsys_axis, raising=False)
        monkeypatch.setattr(logl_grid, 'kp_axis', kp_axis, raising=False)

        result_box = logl_grid.compute_empirical_sigma_map(
            map_2d, method='box', box_vsys=(0., 3.), box_kp=(100., 120.),
        )
        result_clip = logl_grid.compute_empirical_sigma_map(map_2d, method='clip')

        for result in (result_box, result_clip):
            assert np.isfinite(result['noise_std'])
            assert np.isfinite(result['peak_sigma'])
            assert result['vsys_best'] == pytest.approx(4.)
            assert result['kp_best'] == pytest.approx(130.)


class TestPosteriorFromLogMap:
    """`_posterior_from_log_map` (Chantier C: shared post-processing factored
    out of `compute_kpvsys_posterior`/`compute_alpha_kp_posterior`/
    `compute_alpha_vsys_posterior`) and `compute_kpvsys_posterior_fixed_alpha`."""

    def test_output_is_normalised_to_max_one(self):
        log_map = np.array([[-5., -2., -10.], [-1., 0., -3.]])
        axis0 = np.array([0., 1.])
        axis1 = np.array([10., 20., 30.])

        posterior, axis0_out, axis1_out, margin0, margin1 = logl_grid._posterior_from_log_map(
            log_map, axis0, axis1, oversample=1,
        )

        assert posterior.max() == pytest.approx(1.0)
        np.testing.assert_allclose(axis0_out, axis0)
        np.testing.assert_allclose(axis1_out, axis1)
        assert margin0.shape == (2,)
        assert margin1.shape == (3,)

    def test_fixed_alpha_posterior_matches_manual_get_logl(self, monkeypatch):
        """`compute_kpvsys_posterior_fixed_alpha` at alpha=1 should match a
        manual `get_logl(alpha=1., ...)` call fed through the same
        normalisation, with oversampling disabled for an exact comparison."""
        ct, st, sf, N, vsys_axis, kp_axis, _, _ = _make_signal_grid(alpha_true=1.5)
        _set_grid_globals(monkeypatch, ct, st, sf, N, vsys_axis, kp_axis)

        posterior, vsys_out, kp_out, _, _ = logl_grid.compute_kpvsys_posterior_fixed_alpha(
            alpha=1.0, idx_signal=np.array([0]), oversample=1,
        )

        log_map = logl_grid.get_logl(alpha=1.0, idx_exposure=np.array([0]), sum_axis=(-2, -1))
        expected = np.exp(log_map - log_map.max())

        np.testing.assert_allclose(posterior, expected, rtol=1e-10)
        np.testing.assert_allclose(vsys_out, vsys_axis)
        np.testing.assert_allclose(kp_out, kp_axis)
