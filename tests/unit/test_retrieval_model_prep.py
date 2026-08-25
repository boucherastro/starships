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
            'wv_range': [1.9, 2.1],
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
