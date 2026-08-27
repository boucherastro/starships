import numpy as np
import pytest
from sklearn.decomposition import PCA

import starships.petitradtrans_utils as prt
import starships.planet_obs as planet_obs
from starships.convolution import SIGMA_TO_FWHM
from starships.mask_tools import interp1d_masked
from starships.model_sequence import build_model_sequence, combine_regions, precompute_theta_model
from starships.spectrum import quick_inject_clean
from starships import homemade as hm


def _gaussian_line(wv, wv0, fwhm, amp=0.3):
    sigma = fwhm / SIGMA_TO_FWHM
    return 1.0 - amp * np.exp(-0.5 * ((wv - wv0) / sigma) ** 2)


class TestCombineRegions:
    """`combine_regions` (Chantier A Phase 2) remplace deux branchements spéciaux
    (somme des régions citrus, mélange nuageux/clair) par une seule somme pondérée."""

    def test_weighted_sum(self):
        m1 = np.array([1.0, 2.0, 3.0])
        m2 = np.array([4.0, 5.0, 6.0])
        out = combine_regions([m1, m2], [0.3, 0.7])
        np.testing.assert_allclose(out, 0.3 * m1 + 0.7 * m2)

    def test_reproduces_old_cloud_clear_blend(self):
        """`cloud_f*model_cloudy + clear_f*model_clear` (ancien branchement spécial de
        `prepare_model_high_or_low`) est un cas particulier à 2 régions."""
        cloudy = np.array([0.9, 0.8, 0.7])
        clear = np.array([1.0, 1.0, 1.0])
        cloud_f = 0.4
        clear_f = 1 - cloud_f
        expected = cloud_f * cloudy + clear_f * clear
        np.testing.assert_allclose(combine_regions([cloudy, clear], [cloud_f, clear_f]), expected)

    def test_per_exposure_weights_broadcast(self):
        """Les poids peuvent varier par exposition (ex. futur noyau dépendant de la phase)."""
        m1 = np.ones((3, 5))
        m2 = 2 * np.ones((3, 5))
        w1 = np.array([1.0, 0.5, 0.0])[:, None]
        w2 = 1 - w1
        out = combine_regions([m1, m2], [w1, w2])
        np.testing.assert_allclose(out[:, 0], [1.0, 1.5, 2.0])


def _make_data_wave(wv0, half_width, n_exp, n_ord, n_pix):
    """Grille de longueur d'onde factice, identique pour toutes les expositions/ordres."""
    wv_1d = np.linspace(wv0 - half_width, wv0 + half_width, n_pix)
    return np.tile(wv_1d, (n_exp, n_ord, 1))


class TestBuildModelSequence:
    """`build_model_sequence` (Chantier A Phase 2) remplace
    `gen_model_sequence_noinj`/`quick_inject_clean` pour les appelants qui gardent
    Fp/Fstar séparés -- corrige le bug #2 (RV stellaire figée sur la planète)."""

    WV0 = 2.0
    N_EXP, N_ORD, N_PIX = 5, 1, 4000

    def _synthetic_wave_fp_fstar(self):
        wave = np.linspace(self.WV0 - 0.01, self.WV0 + 0.01, 20_000)
        # Planet line at WV0 - 0.002, stellar line at WV0 + 0.003 -- far enough apart
        # that following the wrong velocity is easy to detect.
        Fp = _gaussian_line(wave, self.WV0 - 0.002, 2.0e-4, amp=0.3)
        Fstar = _gaussian_line(wave, self.WV0 + 0.003, 2.0e-4, amp=0.2)
        return wave, Fp, Fstar

    def test_matches_old_quick_inject_clean_when_star_shifted_like_planet(self):
        """Cas dégénéré reproduisant l'ancien comportement (bugué) : si `vr_orb` est mis
        égal à `vrp_orb` (l'étoile décalée comme la planète, comme le faisait l'ancien
        code), le résultat doit être proche de l'ancien `quick_inject_clean` appliqué au
        ratio Fp/Fstar déjà combiné. Une égalité bit-à-bit n'est pas attendue : interpoler
        Fp et Fstar séparément puis diviser n'est pas rigoureusement identique à
        interpoler le ratio Fp/Fstar déjà combiné (spline cubique non linéaire) -- mais
        les deux doivent converger numériquement (tolérance serrée)."""
        wave, Fp, Fstar = self._synthetic_wave_fp_fstar()
        ratio = Fp / Fstar

        data_wave = _make_data_wave(self.WV0, 0.008, self.N_EXP, self.N_ORD, self.N_PIX)
        vrp_orb = np.linspace(-50.0, 50.0, self.N_EXP)  # km/s, typical orbital excursion
        alpha = np.ones(self.N_EXP)

        # Old engine: single rigid shift of the already-combined ratio.
        flux_ones = np.ones((self.N_EXP, self.N_ORD, self.N_PIX))
        old_seq, _ = quick_inject_clean(data_wave, flux_ones, wave, ratio, vrp_orb,
                                        sep=None, R_star=None, A_star=None,
                                        alpha=alpha, kind_trans='emission')

        # New engine, star shifted at the same velocity as the planet (degenerate case).
        new_seq = build_model_sequence(wave, Fp, data_wave, vrp_orb, Fstar=Fstar,
                                       vr_orb=vrp_orb, alpha=alpha, kind_trans='emission')

        np.testing.assert_allclose(np.ma.filled(new_seq, np.nan),
                                   np.ma.filled(old_seq, np.nan),
                                   rtol=2e-3, atol=2e-3)

    def test_fixed_star_keeps_stellar_line_static_across_exposures(self):
        """Coeur du correctif : avec `vr_orb=0` (défaut), la raie stellaire reste à sa
        position d'origine dans le modèle recombiné, peu importe l'excursion orbitale de
        la planète -- contrairement à l'ancien moteur, qui la traînerait à `vrp_orb`."""
        wave, Fp, Fstar = self._synthetic_wave_fp_fstar()
        data_wave = _make_data_wave(self.WV0, 0.008, self.N_EXP, self.N_ORD, self.N_PIX)
        vrp_orb = np.linspace(-100.0, 100.0, self.N_EXP)  # large excursion

        new_seq = build_model_sequence(wave, Fp, data_wave, vrp_orb, Fstar=Fstar,
                                       kind_trans='emission')  # vr_orb defaults to 0.0

        # The stellar absorption line, once it appears in the Fp/Fstar ratio, turns into
        # a *peak* (dividing by a dip amplifies that wavelength) -- the planet's own line
        # is deeper (amp 0.3 vs 0.2) and stays a dip, so the two do not get confused.
        # Track the global maximum: it must stay at the star's true (unshifted)
        # wavelength for every exposure, regardless of the planet's excursion.
        wv_1d = data_wave[0, 0]
        for i_exp in range(self.N_EXP):
            spec = new_seq[i_exp, 0]
            idx_max = np.argmax(spec)
            assert abs(wv_1d[idx_max] - (self.WV0 + 0.003)) < 2 * np.diff(wv_1d)[0]

    def test_real_stellar_rv_option_follows_vr_not_vrp(self):
        """Option "vraie RV stellaire" : si `vr_orb` est passé explicitement et distinct
        de `vrp_orb`, la raie stellaire doit suivre `vr_orb`, pas `vrp_orb`."""
        wave, Fp, Fstar = self._synthetic_wave_fp_fstar()
        data_wave = _make_data_wave(self.WV0, 0.008, 3, self.N_ORD, self.N_PIX)
        vrp_orb = np.array([-100.0, 0.0, 100.0])
        vr_orb = np.array([-5.0, 0.0, 5.0])  # small, but not degenerate with vrp_orb

        new_seq = build_model_sequence(wave, Fp, data_wave, vrp_orb, Fstar=Fstar,
                                       vr_orb=vr_orb, kind_trans='emission')

        wv_1d = data_wave[0, 0]
        shifts_star = hm.calc_shift(vr_orb, kind='rel')
        for i_exp in range(3):
            spec = new_seq[i_exp, 0]
            # See test_fixed_star_keeps_stellar_line_static_across_exposures: the stellar
            # line is a peak in the Fp/Fstar ratio, not a dip.
            idx_max = np.argmax(spec)
            expected_wv0 = (self.WV0 + 0.003) * shifts_star[i_exp]
            assert abs(wv_1d[idx_max] - expected_wv0) < 2 * np.diff(wv_1d)[0]

    def test_transmission_mode_uses_fp_alone(self):
        """En transmission, pas de Fstar à séparer -- `build_model_sequence(Fstar=None)`
        doit reproduire exactement l'ancien `quick_inject_clean` (une seule interpolation)."""
        wave, Fp, _ = self._synthetic_wave_fp_fstar()
        data_wave = _make_data_wave(self.WV0, 0.008, self.N_EXP, self.N_ORD, self.N_PIX)
        vrp_orb = np.linspace(-50.0, 50.0, self.N_EXP)
        alpha = np.full(self.N_EXP, 0.1)

        flux_ones = np.ones((self.N_EXP, self.N_ORD, self.N_PIX))
        old_seq, _ = quick_inject_clean(data_wave, flux_ones, wave, Fp, vrp_orb,
                                        sep=None, R_star=None, A_star=None,
                                        alpha=alpha, kind_trans='transmission')

        new_seq = build_model_sequence(wave, Fp, data_wave, vrp_orb, Fstar=None,
                                       alpha=alpha, kind_trans='transmission')

        np.testing.assert_allclose(np.ma.filled(new_seq, np.nan),
                                   np.ma.filled(old_seq, np.nan), rtol=1e-10)

    def test_emission_requires_fstar(self):
        wave, Fp, _ = self._synthetic_wave_fp_fstar()
        data_wave = _make_data_wave(self.WV0, 0.008, 2, 1, 10)
        with pytest.raises(ValueError):
            build_model_sequence(wave, Fp, data_wave, vrp_orb=0.0, kind_trans='emission')


class TestSaveLoadVr:
    """`vr` (excursion RV stellaire par exposition, Chantier A Phase 2) suit le même
    schéma que `RV_sys`/`mid_berv`/`mid_vr` (Phase 0) : sauvé/chargé, avec repli sur
    `None` pour les anciens fichiers .npz qui ne l'ont pas."""

    def _minimal_npz_kwargs(self, n_exp=4, n_ord=2, n_pix=6, n_feat=3):
        rng = np.random.default_rng(0)
        pca = PCA(n_components=n_feat)
        pca.fit(rng.normal(size=(n_feat + 2, n_feat)))

        flux = np.ma.array(rng.normal(size=(n_exp, n_ord, n_pix)),
                           mask=np.zeros((n_exp, n_ord, n_pix), dtype=bool))
        noise = np.ma.array(np.ones((n_exp, n_ord, n_pix)), mask=flux.mask)
        N = np.ma.array(np.full((n_ord, n_pix), n_exp), mask=np.zeros((n_ord, n_pix), dtype=bool))

        return dict(
            components_=pca.components_, explained_variance_=pca.explained_variance_,
            explained_variance_ratio_=pca.explained_variance_ratio_,
            singular_values_=pca.singular_values_, mean_=pca.mean_,
            n_components_=pca.n_components_, n_samples_=pca.n_samples_,
            noise_variance_=pca.noise_variance_, n_features_in_=pca.n_features_in_,
            RV_const=12.3, RV_sys=10.0, mid_berv=2.0, mid_vr=0.3,
            params=[0, 0, 0, 0, 0, n_feat], wave=np.ones((n_ord, n_pix)),
            vrp=np.linspace(-50, 50, n_exp),
            sep=np.ones(n_exp), noise=np.ma.getdata(noise), mask_noise=np.ma.getmaskarray(noise),
            N=np.ma.getdata(N), mask_N=np.ma.getmaskarray(N),
            t_start=np.linspace(0, 1, n_exp),
            flux=np.ma.getdata(flux), mask_flux=np.ma.getmaskarray(flux),
            s2f=np.ma.getdata(flux)[:, :, 0], mask_s2f=np.ma.getmaskarray(flux)[:, :, 0],
            ratio=np.ma.getdata(flux), mask_ratio=np.ma.getmaskarray(flux),
            reconstructed=np.ma.getdata(flux), mask_reconstructed=np.ma.getmaskarray(flux),
            mast_out=np.ma.getdata(flux), mask_mast_out=np.ma.getmaskarray(flux),
            alpha_frac=np.linspace(0, 1, n_exp), icorr=np.arange(n_exp), bad_indexs=[],
        )

    def test_vr_round_trips(self, tmp_path):
        kwargs = self._minimal_npz_kwargs()
        vr = np.linspace(-0.2, 0.2, 4)
        np.savez(tmp_path / 'test_data_trs_0.npz', vr=vr, **kwargs)

        _, data_trs = planet_obs.load_sequences('test', do_tr=[0], path=tmp_path)

        np.testing.assert_allclose(data_trs['0']['vr'].to('km/s').value, vr)

    def test_vr_falls_back_to_none_for_legacy_file(self, tmp_path):
        """Fichier .npz d'avant l'ajout de `vr` (Chantier A Phase 2) : pas de KeyError,
        `vr` retombe sur `None` (même schéma que RV_sys/mid_berv/mid_vr en Phase 0)."""
        kwargs = self._minimal_npz_kwargs()
        np.savez(tmp_path / 'test_data_trs_0.npz', **kwargs)  # no `vr` key

        _, data_trs = planet_obs.load_sequences('test', do_tr=[0], path=tmp_path)

        assert data_trs['0']['vr'] is None


class _FakeAtmoObject:
    """Stand-in for a petitRADTRANS `Radtrans` object -- petitRADTRANS itself is not
    installed locally (CLAUDE.md: nothing depending on it runs outside Narval), so
    `retrieval_model_plain`'s RT physics cannot be exercised here. This fake only lets
    the *return_fp_fstar plumbing/algebra* added in Chantier A Phase 2 be checked
    locally (fp_out/fstar_out reproduce the combined ratio, shapes line up); the actual
    RT output still needs validating on Narval (see plan_revision_starships.md)."""

    def __init__(self, freq, flux=None, transm_rad=None):
        self.freq = freq
        self._flux = flux
        self._transm_rad = transm_rad

    def calc_transm(self, *args, **kwargs):
        self.transm_rad = self._transm_rad

    def calc_flux(self, *args, **kwargs):
        self.flux = self._flux


class TestRetrievalModelPlainFpFstar:
    """`return_fp_fstar` (Chantier A Phase 2 addition to `retrieval_model_plain`) --
    see `_FakeAtmoObject` docstring for what this test does and does not cover."""

    # cm/s, only used to fake petitRADTRANS.nat_cst.c locally -- confirmed by Antoine to
    # match the real petitRADTRANS constant (nc.c = 29979245800.0) bit for bit.
    C_CGS = 29979245800.0
    WV0 = 2.0  # micron

    def _fake_native_grid(self):
        wave_native = np.linspace(self.WV0 - 0.002, self.WV0 + 0.002, 40_000)
        freq = self.C_CGS / (wave_native * 1e-4)  # round-trips exactly back to wave_native
        return wave_native, freq

    def _theta_dict(self):
        return dict(pressures=np.logspace(-6, 2, 10), temperatures=np.full(10, 1500.0),
                   gravity=1000.0, P0=1e-2, cloud=None, R_pl=1.0, R_star=1.0)

    def test_emission_returns_fp_and_fstar_separately(self, monkeypatch):
        wave_native, freq = self._fake_native_grid()
        Fp_native = _gaussian_line(wave_native, self.WV0 - 0.0005, 5e-6, amp=0.3)
        Fstar_native = _gaussian_line(wave_native, self.WV0 + 0.0007, 5e-6, amp=0.2)

        fake_atmo = _FakeAtmoObject(freq=freq, flux=Fp_native)
        monkeypatch.setattr(prt, 'nc', type('FakeNc', (), {'c': self.C_CGS})(), raising=False)

        wave_out, Fp, Fstar = prt.retrieval_model_plain(
            fake_atmo, species={}, planet=None, kind_trans='emission',
            fct_star=lambda wv: Fstar_native, return_fp_fstar=True,
            abundances={}, MMW=2.3, VMR={}, **self._theta_dict(),
        )

        assert Fstar is not None
        assert wave_out.shape == Fp.shape == Fstar.shape == wave_native.shape
        # Fp/Fstar must reproduce exactly what the default (combined-ratio) path returns.
        _, ratio = prt.retrieval_model_plain(
            fake_atmo, species={}, planet=None, kind_trans='emission',
            fct_star=lambda wv: Fstar_native, return_fp_fstar=False,
            abundances={}, MMW=2.3, VMR={}, **self._theta_dict(),
        )
        np.testing.assert_allclose((Fp / Fstar).decompose().value, ratio.decompose().value, rtol=1e-12)

    def test_transmission_returns_none_for_fstar(self, monkeypatch):
        wave_native, freq = self._fake_native_grid()
        depth_native = _gaussian_line(wave_native, self.WV0, 5e-6, amp=0.01)

        fake_atmo = _FakeAtmoObject(freq=freq, transm_rad=np.sqrt(depth_native))
        monkeypatch.setattr(prt, 'nc', type('FakeNc', (), {'c': self.C_CGS})(), raising=False)

        wave_out, depth, Fstar = prt.retrieval_model_plain(
            fake_atmo, species={}, planet=None, kind_trans='transmission',
            return_fp_fstar=True, abundances={}, MMW=2.3, VMR={}, **self._theta_dict(),
        )

        assert Fstar is None
        np.testing.assert_allclose(depth, depth_native)

    def test_default_behaviour_unchanged(self, monkeypatch):
        """`return_fp_fstar=False` (the default) must still return a single combined
        value, exactly as before this Phase 2 addition."""
        wave_native, freq = self._fake_native_grid()
        depth_native = _gaussian_line(wave_native, self.WV0, 5e-6, amp=0.01)
        fake_atmo = _FakeAtmoObject(freq=freq, transm_rad=np.sqrt(depth_native))
        monkeypatch.setattr(prt, 'nc', type('FakeNc', (), {'c': self.C_CGS})(), raising=False)

        result = prt.retrieval_model_plain(
            fake_atmo, species={}, planet=None, kind_trans='transmission',
            abundances={}, MMW=2.3, VMR={}, **self._theta_dict(),
        )

        assert len(result) == 2


class TestPrecomputeThetaModel:
    """`precompute_theta_model` (Chantier A Phase 2) -- generation + single
    pre-convolution pass, before any Doppler shift. Same caveat as
    `TestRetrievalModelPlainFpFstar`: exercises the plumbing with a fake RT backend,
    not the real petitRADTRANS physics (Narval-only, see CLAUDE.md)."""

    C_CGS = 29979245800.0  # cm/s, confirmed by Antoine to match petitRADTRANS.nat_cst.c
    WV0 = 2.0

    def test_degrades_fp_and_fstar_to_the_same_resolution(self, monkeypatch):
        wave_native = np.linspace(self.WV0 - 0.002, self.WV0 + 0.002, 40_000)
        freq = self.C_CGS / (wave_native * 1e-4)
        Fp_native = _gaussian_line(wave_native, self.WV0 - 0.0005, 5e-6, amp=0.3)
        Fstar_native = _gaussian_line(wave_native, self.WV0 + 0.0007, 5e-6, amp=0.2)

        fake_atmo = _FakeAtmoObject(freq=freq, flux=Fp_native)
        monkeypatch.setattr(prt, 'nc', type('FakeNc', (), {'c': self.C_CGS})(), raising=False)

        theta = dict(pressures=np.logspace(-6, 2, 10), temperatures=np.full(10, 1500.0),
                    gravity=1000.0, P0=1e-2, p_cloud=None, R_pl=1.0, R_star=1.0)

        wave_out, Fp_out, Fstar_out = precompute_theta_model(
            fake_atmo, species={}, planet=None, theta_dict=theta, kind_trans='emission',
            resolution=100_000, native_resolution=200_000,
            fct_star=lambda wv: Fstar_native, abundances={}, MMW=2.3, VMR={},
        )

        assert Fstar_out is not None
        assert wave_out.shape == Fp_out.shape == Fstar_out.shape
        assert np.isfinite(np.ma.filled(Fp_out, np.nan)).any()
        assert np.isfinite(np.ma.filled(Fstar_out, np.nan)).any()
