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
