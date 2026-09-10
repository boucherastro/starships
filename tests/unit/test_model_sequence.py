import numpy as np
import pytest

import starships.petitradtrans_utils as prt
from starships.convolution import SIGMA_TO_FWHM
from starships.mask_tools import interp1d_masked
from starships.model_sequence import (
    _build_default_rotation_kernel, apply_pca_to_model, build_model_sequence, combine_regions,
    combine_regions_with_kernel, generate_native_fp_fstar, precompute_theta_model,
)
from starships.spectrum import quick_inject_clean
from starships import homemade as hm


def _gaussian_line(wv, wv0, fwhm, amp=0.3):
    sigma = fwhm / SIGMA_TO_FWHM
    return 1.0 - amp * np.exp(-0.5 * ((wv - wv0) / sigma) ** 2)


class TestApplyPcaToModel:
    """`apply_pca_to_model` (Chantier C1) merges what used to be two near-duplicate
    functions in `transpec.py`, `build_trans_spectrum_mod2`/`build_trans_spectrum_mod_fast`
    -- the latter was exactly the former with `reference_spec`/`ratio` left out. `n_pca=0`
    is used throughout to isolate the normalization/division logic from
    `transpec.remove_dem_pca_all`'s own PCA math (covered elsewhere)."""

    def _make_flux(self):
        rng = np.random.default_rng(0)
        return np.ma.array(1.0 + 0.1 * rng.standard_normal((2, 3, 5)))

    def test_default_matches_median_normalize_then_mean_subtract(self):
        """No `ratio`/`reference_spec` (the old "_fast" case): just median-normalize,
        then subtract the mean (`norm=True`, `somme=False` defaults)."""
        flux = self._make_flux()
        result = apply_pca_to_model(flux, pca=None, n_pca=0)

        expected = flux / np.ma.median(flux, axis=-1)[:, :, None]
        expected = expected - np.ma.mean(expected, axis=-1)[:, :, None]
        np.testing.assert_allclose(result, expected)

    def test_ratio_applied_before_reference_spec(self):
        """The old `build_trans_spectrum_mod2` divided by `ratio` first, then by
        `reference_spec` -- order matters since division isn't commutative on masked
        arrays with different broadcast shapes."""
        flux = self._make_flux()
        ratio = np.ma.array(1.0 + 0.05 * np.arange(flux.shape[-1]))
        reference_spec = np.ma.array(0.9 + 0.02 * np.arange(flux.shape[-1]))

        result = apply_pca_to_model(flux, pca=None, n_pca=0, ratio=ratio,
                                     reference_spec=reference_spec, norm=False)

        expected = flux / np.ma.median(flux, axis=-1)[:, :, None]
        expected = expected / ratio
        expected = expected / reference_spec
        expected = expected / np.ma.mean(expected, axis=-1)[:, :, None]
        np.testing.assert_allclose(result, expected)

    def test_norm_false_divides_by_mean_instead_of_subtracting(self):
        flux = self._make_flux()
        result = apply_pca_to_model(flux, pca=None, n_pca=0, norm=False)

        expected = flux / np.ma.median(flux, axis=-1)[:, :, None]
        expected = expected / np.ma.mean(expected, axis=-1)[:, :, None]
        np.testing.assert_allclose(result, expected)


class TestCombineRegions:
    """`combine_regions` (Chantier A Phase 2) remplace deux branchements spéciaux
    (somme des régions -- ex. citrus/longitude, mélange nuageux/clair) par une seule
    somme pondérée."""

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


# `TestSaveLoadVr` (tested `vr` round-tripping through the saved .npz, with a fallback to
# `None` for older files missing it) was removed in Chantier B / B3: `vr`/`vrp` are no longer
# saved/loaded at all -- they are purely a deterministic function of the planet's ephemeris
# and exposure timestamps (`gen_rv_sequence`, `K=None`), recomputed identically every time by
# `load_reduced_sequence`, so there is nothing left to round-trip or fall back on (see
# `save_reduced_sequence`'s docstring). The synthetic minimal-.npz fixture this test used also
# predates B3's unified file format (no `spec_trans`/`fl_norm`/`fl_masked`/`fl_Sref`/
# `noise_npc`, all now required by `load_reduced_sequence`) and would need a realistic
# reduction round trip to reconstruct meaningfully -- covered instead by
# `tests/regression/test_regression_reduction.py::TestReadTimeNPCConsistency` against real
# WASP-33 data.


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


class _DummyQuantity:
    """Minimal stand-in for an astropy Quantity: only `.value` and `.to(unit)`."""

    def __init__(self, value):
        self.value = value

    def to(self, unit):
        return self


class _DummyPlanet:
    """Just enough of `starships.planet_obs.Planet` for `_build_default_rotation_kernel`'s
    `'emission'` branch (`planet.period[0].to('s').value`)."""

    def __init__(self, period_seconds):
        self.period = [_DummyQuantity(period_seconds)]


class TestDefaultRotationKernel:
    """`_build_default_rotation_kernel` (Chantier A Phase 3) -- the phase-*independent*
    default kernel (vsini-style in emission, wind broadening in transmission), computed
    once per theta in `precompute_theta_model`. Contrast with the per-exposure
    multi-region kernel covered by `TestBuildModelSequenceRegionKernel` below."""

    def test_none_returns_none(self):
        assert _build_default_rotation_kernel(None, theta_dict={}, planet=None,
                                              sampling_resolution=100_000) is None

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            _build_default_rotation_kernel('bogus', theta_dict={}, planet=None,
                                          sampling_resolution=100_000)

    # A WASP-33b-like R_pl, in centimeters -- matching the convention `theta_dict`
    # actually carries in the real pipeline (`retrieval.py::unpack_theta` converts
    # R_pl to cgs *before* theta_dict reaches this function; unlike M_pl, which stays
    # a bare Mjup float, see the M_pl note in `_build_default_rotation_kernel`).
    # Using a dummy value like `R_pl=1.0` (as this test used to) doesn't exercise the
    # real R_pl-unit convention at all -- that gap is exactly what let a genuine units
    # bug (`theta_dict['R_pl'] * const.R_jup`, silently inflating R_pl by ~7e9x) ship
    # unnoticed (found 2026-08-28). Also large enough that v_eq/wind-broadening is
    # much bigger than the resolution element (v_eq ~ 7 km/s vs res_elem ~ 1.2 km/s at
    # R=250_000) -- a dummy R_pl of order 1 gives v_eq << res_elem, which falls back to
    # CitrusRotationKernel's "kernel is zero everywhere -> delta function" safety net
    # and would silently mask a broken kernel shape (this is exactly how the old
    # phase-independence test below passed despite the kernel being broken).
    R_PL_CM = 1.2e10

    def test_transmission_kernel_is_finite_and_normalized(self):
        # M_pl: plain float in Mjup, matching how retrieval.py::setup_retrieval
        # actually stores it in theta_dict (fixed_params['M_pl'] = planet.M_pl.to
        # ('Mjup').value) -- not an astropy Quantity.
        theta = dict(R_pl=self.R_PL_CM, M_pl=1.2, T_eq=1500.0, wind=2.0)
        kernel = _build_default_rotation_kernel('transmission', theta, planet=None,
                                               sampling_resolution=250_000)
        assert np.isfinite(kernel).all()
        # BaseKer.resample's dv_new/dv_old rescale keeps the kernel's discrete sum
        # only approximately 1 (it conserves the *area* under the kernel, sum*dv, not
        # the discrete sum exactly -- see BaseKer.resample's docstring) -- a loose
        # tolerance is the right check here, not exact equality.
        assert kernel.sum() == pytest.approx(1.0, rel=0.05)

    def test_emission_kernel_is_finite_and_normalized(self):
        theta = dict(R_pl=self.R_PL_CM, rot_factor=1.0)
        planet = _DummyPlanet(period_seconds=1.22 * 24 * 3600.0)
        kernel = _build_default_rotation_kernel('emission', theta, planet=planet,
                                               sampling_resolution=250_000)
        assert np.isfinite(kernel).all()
        # BaseKer.resample's dv_new/dv_old rescale keeps the kernel's discrete sum
        # only approximately 1 (it conserves the *area* under the kernel, sum*dv, not
        # the discrete sum exactly -- see BaseKer.resample's docstring) -- a loose
        # tolerance is the right check here, not exact equality.
        assert kernel.sum() == pytest.approx(1.0, rel=0.05)

    def test_emission_kernel_is_symmetric_around_zero_velocity(self):
        """Regression test for a real bug found 2026-08-28: the 'emission' default
        kernel used to be built by reusing `CitrusRotationKernel` with a single citrus
        boundary ([0.0]) as a "degenerate, phase-independent" case. That geometry is
        built for >= 2 boundaries; with only one, self-referencing boundary, the
        resulting kernel was *not* symmetric around v=0 (centroid off by ~3 km/s for a
        WASP-33b-like case) and was entirely zero for any phase other than exactly
        0.0 -- masked by the previous version of this test using a dummy R_pl so small
        that CitrusRotationKernel's own "kernel is zero everywhere" fallback produced
        an (accidentally phase-independent) delta function regardless. Now built
        directly from `spectrum.SolidRotationKernel`'s closed-form profile, which is
        symmetric by construction -- checked here against realistic (R_pl >> the
        fallback regime) values, via `_build_default_rotation_kernel`'s real call
        path (not the kernel class in isolation), so a regression in the wiring would
        be caught too."""
        theta = dict(R_pl=self.R_PL_CM, rot_factor=1.0)
        planet = _DummyPlanet(period_seconds=1.22 * 24 * 3600.0)
        kernel = _build_default_rotation_kernel('emission', theta, planet=planet,
                                               sampling_resolution=250_000)
        # Checked via the left-half/right-half mass balance, not exact bin-by-bin
        # reversal: BaseKer.resample()'s final np.arange(v_grid.min(), v_grid.max(),
        # dv_new) grid isn't perfectly re-centered after resampling (a ~1-pixel
        # artifact shared by every BaseKer subclass, unrelated to this bug, and not
        # worth chasing exact-index symmetry over) -- the *real* bug this guards
        # against put essentially all the kernel's mass on one side (see docstring),
        # a much coarser asymmetry that a loose left/right balance check still catches.
        n = len(kernel)
        left, right = kernel[:n // 2].sum(), kernel[n // 2:].sum()
        assert min(left, right) / max(left, right) > 0.8


class TestBuildModelSequenceRegionKernel:
    """`build_model_sequence`'s `region_kernel` hook (Chantier A Phase 3) -- the
    phase-*dependent* case (multi-region, e.g. citrus/longitude slices), applied per
    exposure (one spline per exposure) instead of the single shared spline used when
    `region_kernel` is None."""

    WV0 = TestBuildModelSequence.WV0
    N_EXP, N_ORD, N_PIX = TestBuildModelSequence.N_EXP, 1, TestBuildModelSequence.N_PIX

    def test_requires_phase(self):
        wave = np.linspace(self.WV0 - 0.01, self.WV0 + 0.01, 2000)
        Fp = np.ones_like(wave)
        data_wave = _make_data_wave(self.WV0, 0.008, self.N_EXP, self.N_ORD, self.N_PIX)
        with pytest.raises(ValueError):
            build_model_sequence(wave, Fp, data_wave, vrp_orb=0.0, kind_trans='transmission',
                                 region_kernel=lambda wave, Fp, phase: Fp)

    def test_region_kernel_called_once_per_exposure_with_its_own_phase(self):
        wave = np.linspace(self.WV0 - 0.01, self.WV0 + 0.01, 2000)
        Fp = np.ones_like(wave)
        data_wave = _make_data_wave(self.WV0, 0.008, self.N_EXP, self.N_ORD, self.N_PIX)
        phase = np.linspace(0.1, 0.9, self.N_EXP)

        calls = []

        def region_kernel(wave, Fp, phase_i):
            calls.append(phase_i)
            return Fp

        build_model_sequence(wave, Fp, data_wave, vrp_orb=0.0, kind_trans='transmission',
                            region_kernel=region_kernel, phase=phase)

        assert len(calls) == self.N_EXP
        np.testing.assert_allclose(sorted(calls), sorted(phase))

    def test_different_kernel_per_exposure_changes_the_model_per_exposure(self):
        """The whole point of the per-exposure path: a kernel that actually varies with
        phase must produce a genuinely different spectrum per exposure, not just a
        different Doppler shift."""
        wave, Fp, _ = TestBuildModelSequence()._synthetic_wave_fp_fstar()
        data_wave = _make_data_wave(self.WV0, 0.008, 2, self.N_ORD, self.N_PIX)

        # Exposure 0 sees the planet line unchanged; exposure 1 gets it flattened out.
        def region_kernel(wave, Fp, phase_i):
            return Fp if phase_i == 0.0 else np.ones_like(Fp)

        seq = build_model_sequence(wave, Fp, data_wave, vrp_orb=0.0, kind_trans='transmission',
                                   region_kernel=region_kernel, phase=np.array([0.0, 1.0]))

        assert not np.ma.allclose(seq[0, 0], seq[1, 0])


class TestGenerateNativeFpFstar:
    """`generate_native_fp_fstar` (Chantier A Phase 3) -- `precompute_theta_model`'s
    Step 1 (petitRADTRANS call + trim the last native-grid point) factored out, with
    no degradation step, for reuse by the true multi-region path
    (`combine_regions_with_kernel`): the per-region kernel from `retrieval.py`'s
    `get_ker` already bakes in the resolution degradation, so degrading here too
    would degrade twice. Same caveat as `TestRetrievalModelPlainFpFstar`: exercises
    the plumbing with a fake RT backend, not the real petitRADTRANS physics
    (Narval-only, see CLAUDE.md)."""

    C_CGS = 29979245800.0  # cm/s, confirmed by Antoine to match petitRADTRANS.nat_cst.c
    WV0 = 2.0

    def _theta_dict(self):
        return dict(pressures=np.logspace(-6, 2, 10), temperatures=np.full(10, 1500.0),
                   gravity=1000.0, P0=1e-2, p_cloud=None, R_pl=1.0, R_star=1.0)

    def test_no_degradation_only_the_native_edge_point_is_trimmed(self, monkeypatch):
        wave_native = np.linspace(self.WV0 - 0.002, self.WV0 + 0.002, 40_000)
        freq = self.C_CGS / (wave_native * 1e-4)
        Fp_native = _gaussian_line(wave_native, self.WV0 - 0.0005, 5e-6, amp=0.3)
        Fstar_native = _gaussian_line(wave_native, self.WV0 + 0.0007, 5e-6, amp=0.2)

        fake_atmo = _FakeAtmoObject(freq=freq, flux=Fp_native)
        monkeypatch.setattr(prt, 'nc', type('FakeNc', (), {'c': self.C_CGS})(), raising=False)

        wave_out, Fp_out, Fstar_out = generate_native_fp_fstar(
            fake_atmo, species={}, planet=None, theta_dict=self._theta_dict(),
            kind_trans='emission', fct_star=lambda wv: Fstar_native,
            abundances={}, MMW=2.3, VMR={},
        )

        # Unlike precompute_theta_model, only the single native-grid-edge point is
        # dropped here -- no resolution degradation, no 15-point boundary trim.
        assert wave_out.shape == Fp_out.shape == Fstar_out.shape == (wave_native.size - 1,)
        # Fp is scaled by retrieval_model_plain (R_pl**2/R_star**2, units) -- same
        # caveat as TestRetrievalModelPlainFpFstar, check the ratio instead of the
        # raw value, against the un-degraded combined-ratio path (return_fp_fstar=False).
        # retrieval_model_plain's own positional args use 'cloud', not 'p_cloud'
        # (unpack_theta's naming) -- pass them explicitly rather than **theta_dict.
        theta = self._theta_dict()
        args = [theta[key] for key in ('pressures', 'temperatures', 'gravity', 'P0',
                                       'p_cloud', 'R_pl', 'R_star')]
        _, ratio_native = prt.retrieval_model_plain(
            fake_atmo, {}, None, *args, kind_trans='emission',
            fct_star=lambda wv: Fstar_native, return_fp_fstar=False,
            abundances={}, MMW=2.3, VMR={},
        )
        np.testing.assert_allclose((Fp_out / Fstar_out), ratio_native[:-1].decompose().value, rtol=1e-12)

    def test_transmission_returns_none_for_fstar(self, monkeypatch):
        wave_native = np.linspace(self.WV0 - 0.002, self.WV0 + 0.002, 40_000)
        freq = self.C_CGS / (wave_native * 1e-4)
        depth_native = _gaussian_line(wave_native, self.WV0, 5e-6, amp=0.01)

        fake_atmo = _FakeAtmoObject(freq=freq, transm_rad=np.sqrt(depth_native))
        monkeypatch.setattr(prt, 'nc', type('FakeNc', (), {'c': self.C_CGS})(), raising=False)

        _, depth_out, Fstar_out = generate_native_fp_fstar(
            fake_atmo, species={}, planet=None, theta_dict=self._theta_dict(),
            kind_trans='transmission', abundances={}, MMW=2.3, VMR={},
        )

        assert Fstar_out is None
        np.testing.assert_allclose(depth_out, depth_native[:-1])


class TestCombineRegionsWithKernel:
    """`combine_regions_with_kernel` (Chantier A Phase 3) -- the per-exposure
    convolve-then-weighted-sum step for the true multi-region case (any user-defined
    region split, e.g. citrus/longitude slices -- nothing here assumes that specific
    geometry), meant to be called from inside a `region_kernel` closure passed to
    `build_model_sequence` (see `TestBuildModelSequenceRegionKernel`)."""

    def test_single_region_reproduces_plain_kernel_convolution(self):
        wave = np.arange(50, dtype=float)
        Fp = np.zeros(50)
        Fp[25] = 1.0
        kernel = np.array([0.25, 0.5, 0.25])

        out = combine_regions_with_kernel(wave[15:-15], [Fp], [kernel], [1.0])

        expected = np.convolve(Fp, kernel, mode='same')[15:-15]
        np.testing.assert_allclose(out, expected)

    def test_two_regions_are_weighted_and_summed(self):
        # Delta kernels (no smoothing) isolate the weighting/summation step from any
        # convolution effect.
        wave = np.arange(60, dtype=float)
        Fp1 = np.full(60, 2.0)
        Fp2 = np.full(60, 5.0)
        kernel = np.array([1.0])

        out = combine_regions_with_kernel(wave[15:-15], [Fp1, Fp2], [kernel, kernel], [0.3, 0.7])

        expected = 0.3 * Fp1[15:-15] + 0.7 * Fp2[15:-15]
        np.testing.assert_allclose(out, expected)

    def test_output_length_matches_wave_after_edge_trim(self):
        wave_native = np.arange(100, dtype=float)
        Fp = np.random.default_rng(0).normal(size=100)
        kernel = np.array([0.2, 0.6, 0.2])

        out = combine_regions_with_kernel(wave_native[15:-15], [Fp], [kernel], [1.0])

        assert out.shape == wave_native[15:-15].shape
