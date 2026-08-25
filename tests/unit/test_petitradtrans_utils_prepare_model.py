import numpy as np

from starships.convolution import SIGMA_TO_FWHM, degrade_and_resample
from starships.petitradtrans_utils import prepare_model


def _gaussian_line(wv, wv0, fwhm, amp=0.3):
    sigma = fwhm / SIGMA_TO_FWHM
    return 1.0 - amp * np.exp(-0.5 * ((wv - wv0) / sigma) ** 2)


class TestPrepareModel:
    """`prepare_model` (Chantier A Phase 1) doit utiliser le moteur unifié
    (`degrade_and_resample`) quand aucun noyau de rotation n'est actif, et
    préserver tel quel l'ancien chemin (`resamp_model` + noyau explicite)
    dans le cas contraire (Phase 3, pas touché ici)."""

    R_true = 500_000
    R_af = 100_000
    wv = np.linspace(1.99, 2.01, 20_000)

    def test_no_rot_ker_matches_degrade_and_resample(self):
        flux = _gaussian_line(self.wv, 2.0, 2.0 / self.R_true)

        wv_out, model_out = prepare_model(self.wv, flux, self.R_true, Raf=self.R_af)

        # Reproduit exactement la logique interne : wv_trim = wv[:-1][15:-15]
        wv_trim = self.wv[:-1]
        expected = degrade_and_resample(wv_trim, flux[:-1], resolution=self.R_af,
                                         input_resolution=self.R_true, sample=wv_trim)
        np.testing.assert_allclose(model_out, np.ma.masked_invalid(expected)[15:-15])
        np.testing.assert_array_equal(wv_out, wv_trim[15:-15])

    def test_explicit_rot_ker_array_still_uses_legacy_path(self):
        """Avec un noyau de rotation explicite (tableau simple), l'ancienne
        méthode `resamp_model`/`resampling` avec noyau doit toujours être
        utilisée sans planter -- ce chemin n'est pas modifié par la Phase 1."""
        flux = _gaussian_line(self.wv, 2.0, 2.0 / self.R_true)
        simple_kernel = np.ones(21) / 21  # noyau boxcar trivial, pas physique

        wv_out, model_out = prepare_model(self.wv, flux, self.R_true, Raf=self.R_af,
                                          rot_ker=simple_kernel)

        assert np.isfinite(model_out[len(model_out) // 2])
        assert wv_out.shape == model_out.shape
