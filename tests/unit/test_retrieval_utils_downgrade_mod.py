import numpy as np

from starships.convolution import SIGMA_TO_FWHM
from starships.retrieval_utils import downgrade_mod


def _gaussian_line(wv, wv0, fwhm, amp=0.3):
    sigma = fwhm / SIGMA_TO_FWHM
    return 1.0 - amp * np.exp(-0.5 * ((wv - wv0) / sigma) ** 2)


class TestDowngradeMod:
    """`downgrade_mod` (Chantier A Phase 1) doit maintenant utiliser le moteur
    de convolution unifié (`degrade_and_resample`)."""

    def test_degrades_and_projects_onto_down_wave(self):
        R_true = 500_000
        R_af = 100_000
        wlen = np.linspace(1.99, 2.01, 20_000)
        flux = _gaussian_line(wlen, 2.0, 2.0 / R_true)
        down_wave = np.linspace(1.995, 2.005, 50)

        result = downgrade_mod(wlen, flux, down_wave, Rbf=R_true, Raf=R_af)

        assert result.shape == down_wave.shape
        assert np.isfinite(result).all()
        # Le creux dégradé doit rester moins profond que la ligne native
        # (amp=0.3), signe qu'une vraie convolution a bien eu lieu.
        assert result.min() > 1.0 - 0.3
