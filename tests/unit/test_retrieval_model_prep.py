import numpy as np
import pytest
from scipy.stats import norm

import starships.retrieval as retrieval
from starships.convolution import SIGMA_TO_FWHM


def _gaussian_line(wv, wv0, fwhm, amp=0.3):
    sigma = fwhm / SIGMA_TO_FWHM
    return 1.0 - amp * np.exp(-0.5 * ((wv - wv0) / sigma) ** 2)


class TestInitStellarSpectrum:
    """`init_stellar_spectrum` (Chantier A Phase 1) dépend de globals du module
    retrieval.py -- on les monkeypatch directement plutôt que de passer par
    tout le setup YAML du retrieval."""

    def test_degrades_stellar_spectrum_and_returns_interpolator(self, monkeypatch):
        R_true = 500_000
        R_prt = 100_000
        wv = np.linspace(1.99, 2.01, 20_000)
        flux = _gaussian_line(wv, 2.0, 2.0 / R_true)

        monkeypatch.setattr(retrieval, 'star_wv', wv, raising=False)
        monkeypatch.setattr(retrieval, 'star_flux', flux, raising=False)
        monkeypatch.setattr(retrieval, 'star_res', R_true, raising=False)
        monkeypatch.setattr(retrieval, 'kind_trans', 'emission', raising=False)
        monkeypatch.setattr(retrieval, 'prt_res', {'high': R_prt}, raising=False)
        monkeypatch.setattr(retrieval, 'wv_range_high', [[1.995, 2.005]], raising=False)

        fct_star = retrieval.init_stellar_spectrum(mode='high')

        assert callable(fct_star)
        out = fct_star(np.array([2.0, 2.001]))
        assert np.isfinite(np.ma.filled(out, np.nan)).all()

    def test_drops_nan_edge_points_instead_of_masking_them(self, monkeypatch):
        """`degrade_and_resample` renvoie du NaN près des bords de `sample` (pas assez
        de marge pour un noyau de convolution complet). `np.ma.masked_invalid` ne
        protège pas `interp1d`, qui lit les données brutes sans tenir compte du
        masque -- les points NaN doivent être réellement retirés (Chantier A,
        bug Spitzer/étoile, 2026-09-16)."""
        R_true = 500_000
        R_prt = 100_000
        wv = np.linspace(1.99, 2.01, 20_000)
        flux = _gaussian_line(wv, 2.0, 2.0 / R_true)

        monkeypatch.setattr(retrieval, 'star_wv', wv, raising=False)
        monkeypatch.setattr(retrieval, 'star_flux', flux, raising=False)
        monkeypatch.setattr(retrieval, 'star_res', R_true, raising=False)
        monkeypatch.setattr(retrieval, 'kind_trans', 'emission', raising=False)
        monkeypatch.setattr(retrieval, 'prt_res', {'high': R_prt}, raising=False)
        # Full sample range, so the very edges of `sample` inside init_stellar_spectrum
        # lack a full convolution margin and degrade_and_resample returns NaN there.
        monkeypatch.setattr(retrieval, 'wv_range_high', [[1.99, 2.01]], raising=False)

        fct_star = retrieval.init_stellar_spectrum(mode='high')

        assert callable(fct_star)
        # The interpolator's own underlying data must be NaN-free -- otherwise
        # querying near those (silently masked but not removed) points would return
        # NaN, exactly the bug this fix addresses.
        assert not np.isnan(fct_star.y).any()

    def test_no_stellar_spectrum_returns_blackbody_sentinel(self, monkeypatch):
        monkeypatch.setattr(retrieval, 'kind_trans', 'emission', raising=False)
        monkeypatch.setattr(retrieval, 'star_wv', None, raising=False)
        monkeypatch.setattr(retrieval, 'wv_range_high', [[1.995, 2.005]], raising=False)
        monkeypatch.setattr(retrieval, 'prt_res', {'high': 100_000}, raising=False)

        result = retrieval.init_stellar_spectrum(mode='high')
        assert result == 'blackbody'


class TestPreparePhotometry:
    """`prepare_photometry`/`prepare_spectrophotometry` (Chantier A Phase 1) sont
    des fonctions pures -- pas de dépendance à des globals du module."""

    def test_integrates_flat_response_close_to_continuum(self):
        model_res = 200_000
        instru_res = 5_000
        wv_mod = np.linspace(1.5, 2.5, 200_000)
        spec_mod = np.ones_like(wv_mod)  # spectre plat -> l'intégration doit rester à 1

        def response(wv):
            return norm.pdf(wv, loc=2.0, scale=0.02)

        data_info = {
            'wv_coverages': [[1.9, 2.1]],
            'res': instru_res,
            'wave': np.array([2.0]),
            'response_fcts': [response],
        }

        wv_band, mod_out = retrieval.prepare_photometry(wv_mod, spec_mod, model_res, data_info)

        np.testing.assert_array_equal(wv_band, [2.0])
        assert mod_out[0] == pytest.approx(1.0, rel=1e-3)


class TestPrepareSpectrophotometry:

    def test_projects_flat_spectrum_onto_instrument_grid(self):
        model_res = 200_000
        instru_res = 5_000
        wv_mod = np.linspace(1.5, 2.5, 200_000)
        spec_mod = np.ones_like(wv_mod)

        data_info = {
            'wv_range': [1.9, 2.1],
            'res': instru_res,
            'wave': np.linspace(1.95, 2.05, 20),
        }

        wv_grid, mod = retrieval.prepare_spectrophotometry(wv_mod, spec_mod, model_res, data_info)

        np.testing.assert_allclose(mod, 1.0, rtol=1e-3)
        np.testing.assert_array_equal(wv_grid, data_info['wave'])

    def test_no_nan_when_model_extent_exactly_matches_padded_wv_range(self):
        """Reproduces the real bug found on KELT-20b G395H NRS2 data (Chantier A,
        2026-09-16): `wv_mod` (built from `wv_range_low`, itself just the union of
        every instrument's own already-`required_margin`-padded `wv_range`) has
        zero margin left *beyond* `data_info['wv_range']`'s own edges. `cond` must
        therefore stay well inside `wv_range`, bracketing the real data (`wave`)
        rather than the padded range, or `degrade_and_resample`/`box_binning` run
        out of room and the projected spectrum comes back NaN."""
        model_res = 200_000
        instru_res = 5_000
        wave = np.linspace(1.95, 2.05, 20)
        wv_range = [wave.min() - retrieval.required_margin(wave.min(), instru_res),
                    wave.max() + retrieval.required_margin(wave.max(), instru_res)]
        # wv_mod spans *exactly* wv_range -- no extra margin beyond it, as would
        # happen when this is the only (or outermost) low-res instrument.
        wv_mod = np.linspace(wv_range[0], wv_range[1], 200_000)
        spec_mod = np.ones_like(wv_mod)

        data_info = {'wv_range': wv_range, 'res': instru_res, 'wave': wave}

        wv_grid, mod = retrieval.prepare_spectrophotometry(wv_mod, spec_mod, model_res, data_info)

        assert not np.isnan(mod).any()
        np.testing.assert_allclose(mod, 1.0, rtol=1e-3)
