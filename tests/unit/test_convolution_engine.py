import numpy as np
import pytest
from scipy.optimize import curve_fit

from starships.analysis import resamp_model
from starships.convolution import SIGMA_TO_FWHM, degrade_and_resample


def _gaussian_line(wv, wv0, fwhm, amp=0.5):
    """Ligne d'absorption gaussienne : continuum = 1, creux d'amplitude `amp`."""
    sigma = fwhm / SIGMA_TO_FWHM
    return 1.0 - amp * np.exp(-0.5 * ((wv - wv0) / sigma) ** 2)


def _measure_fwhm(wv, flux, wv0_guess, fwhm_guess):
    """Récupère la FWHM d'une ligne d'absorption par fit gaussien (peu sensible au bruit
    d'échantillonnage, contrairement à une simple mesure par croisement de mi-hauteur)."""
    def model(x, amp, x0, sigma):
        return 1.0 - amp * np.exp(-0.5 * ((x - x0) / sigma) ** 2)

    finite = np.isfinite(flux)
    p0 = [0.5, wv0_guess, fwhm_guess / SIGMA_TO_FWHM]
    popt, _ = curve_fit(model, wv[finite], flux[finite], p0=p0)
    return abs(popt[2]) * SIGMA_TO_FWHM


class TestDegradeAndResample:
    """Vérifie que `degrade_and_resample` corrige le bug de conflation
    densité-d'échantillonnage / résolution physique de l'ancien moteur
    (`spectrum.py::resampling` / `analysis.py::resamp_model`, Chantier A bug #1)."""

    WV0 = 2.0  # µm

    def test_fixes_oversampled_input_conflation_bug(self):
        """Cas où le bug se manifeste : le tableau d'entrée est échantillonné à une
        densité (R_grid) très différente de sa résolution physique réelle (R_true).
        L'ancien moteur traite R_true comme si c'était la densité de la grille et
        sous-estime donc largement la largeur du noyau de convolution ; le nouveau
        moteur mesure la vraie densité de la grille et ne fait pas cette erreur.
        """
        R_true = 50_000   # résolution physique réelle de l'entrée (Rbf)
        R_grid = 1_000_000  # densité d'échantillonnage du tableau, très différente de R_true
        R_af = 25_000     # résolution cible, du même ordre que R_true (régime du bug)

        fwhm_true = self.WV0 / R_true
        fwhm_target = self.WV0 / R_af

        # Grille suréchantillonnée à R_grid, mais la ligne a une largeur physique
        # fixée par R_true, pas par R_grid.
        n_sigma_span = 400
        half_span = n_sigma_span * fwhm_true / SIGMA_TO_FWHM
        dlog_wv = np.log(1 + 1.0 / R_grid)
        log_wv = np.arange(np.log(self.WV0 - half_span), np.log(self.WV0 + half_span), dlog_wv)
        wv = np.exp(log_wv)
        flux = _gaussian_line(wv, self.WV0, fwhm_true)

        new_flux = degrade_and_resample(wv, flux, resolution=R_af, input_resolution=R_true, sample=wv)
        old_flux = resamp_model(wv, flux, R_true, Raf=R_af, sample=wv)

        new_fwhm = _measure_fwhm(wv, new_flux, self.WV0, fwhm_target)
        old_fwhm = _measure_fwhm(wv, np.ma.filled(old_flux, np.nan), self.WV0, fwhm_target)

        # Le nouveau moteur doit reproduire fidèlement la résolution cible.
        assert new_fwhm == pytest.approx(fwhm_target, rel=0.1)
        # L'ancien moteur, lui, sous-blurre largement (bug de conflation) : sa FWHM de
        # sortie reste beaucoup plus proche de la largeur physique d'entrée (fwhm_true)
        # que de la cible (fwhm_target = 2 * fwhm_true ici) -- loin d'être une simple
        # imprécision.
        assert old_fwhm == pytest.approx(fwhm_true, rel=0.1)
        assert old_fwhm < 0.7 * fwhm_target

    def test_agrees_with_old_engine_when_grid_matches_native_resolution(self):
        """Cas 'bien élevé', déjà correctement traité par l'ancien moteur : la grille
        d'entrée est échantillonnée exactement à sa résolution physique réelle. Les
        deux moteurs doivent alors s'accorder, pour ne pas introduire de régression
        sur l'usage historique."""
        R_true = 50_000
        R_af = 25_000
        fwhm_true = self.WV0 / R_true
        fwhm_target = self.WV0 / R_af

        n_sigma_span = 400
        half_span = n_sigma_span * fwhm_true / SIGMA_TO_FWHM
        dlog_wv = np.log(1 + 1.0 / R_true)
        log_wv = np.arange(np.log(self.WV0 - half_span), np.log(self.WV0 + half_span), dlog_wv)
        wv = np.exp(log_wv)
        flux = _gaussian_line(wv, self.WV0, fwhm_true)

        new_flux = degrade_and_resample(wv, flux, resolution=R_af, input_resolution=R_true, sample=wv)
        old_flux = resamp_model(wv, flux, R_true, Raf=R_af, sample=wv)

        new_fwhm = _measure_fwhm(wv, new_flux, self.WV0, fwhm_target)
        old_fwhm = _measure_fwhm(wv, np.ma.filled(old_flux, np.nan), self.WV0, fwhm_target)

        assert new_fwhm == pytest.approx(old_fwhm, rel=0.1)
        assert new_fwhm == pytest.approx(fwhm_target, rel=0.1)

    def test_edge_points_beyond_available_margin_are_nan(self):
        """Les points de `sample` trop proches du bord du tableau disponible (moins
        d'un noyau de convolution complet de marge) doivent revenir `NaN`, plutôt que
        l'approximation silencieuse à bord tronqué de l'ancien moteur."""
        R_true = 50_000
        R_af = 25_000
        wv = np.linspace(1.999, 2.001, 2000)
        flux = _gaussian_line(wv, self.WV0, self.WV0 / R_true)

        result = degrade_and_resample(wv, flux, resolution=R_af, input_resolution=R_true, sample=wv)
        assert np.isnan(result[0]) or np.isnan(result[-1])
        # Le centre, lui, doit rester fini.
        assert np.isfinite(result[len(result) // 2])
