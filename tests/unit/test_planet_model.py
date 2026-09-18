"""Chantier C: `model_sequence.PlanetModel` (saved model forms), the shared multi-region
per-visit helpers (`build_region_kernel`/`prepare_multi_region_visit`) and the
precomputed-spline path of `build_model_sequence` (`build_model_interpolators`).

Pure numpy / synthetic spectra -- no petitRADTRANS needed. The physics of the
underlying pieces (`degrade_fp_fstar`, `combine_regions_with_kernel`,
`_build_default_rotation_kernel`) is covered by `test_model_sequence.py`; what is
tested here is that the new, form-aware/globals-free entry points reproduce those
exact same computations bit for bit, and that a model survives a save/load round trip.
"""
import astropy.units as u
from dataclasses import replace
import numpy as np
import pytest

from starships import model_sequence as model_seq
from starships.convolution import SIGMA_TO_FWHM
from starships.model_sequence import (
    PlanetModel, apply_default_rotation_kernel, build_model_interpolators, build_model_sequence,
    build_region_kernel, combine_at_phase, combine_regions_with_kernel, degrade, degrade_fp_fstar,
    load_planet_model, prepare_multi_region_visit, save_planet_model,
)


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

# Native grid at a petitRADTRANS-like resolving power (lbl, opacity_sampling ~4), so
# the degradation margins stay small (as in the real pipeline) and the [20:-20]/
# [100:-100] conventions of lnprob/logl_grid apply unchanged.
NATIVE_RES = 250_000.0
INSTRU_RES = 70_000.0
WV_MIN, WV_MAX = 1.50, 1.52


def _gaussian_line(wv, wv0, fwhm, amp):
    sigma = fwhm / SIGMA_TO_FWHM
    return amp * np.exp(-0.5 * ((wv - wv0) / sigma) ** 2)


def _native_wave():
    n = int(np.log(WV_MAX / WV_MIN) * NATIVE_RES)
    return np.geomspace(WV_MIN, WV_MAX, n)


def _theta_dict(tag, line_depth):
    """A theta_dict-like dict with the value types `unpack_theta` really produces."""
    return dict(
        R_pl=1.2e10, M_pl=1.2, T_eq=2500.0, rot_factor=1.0, spec_scale=1.0, wind=None,
        pressures=np.logspace(-6, 2, 5), temperatures=np.full(5, 2000.0),
        CO=np.float64(1e-3), line_depth=line_depth, tag=tag,
    )


def synthetic_native_model(n_regions=1, kind_trans='emission'):
    """Native single- or multi-region PlanetModel with resolved synthetic lines."""
    wave = _native_wave()
    Fp_by_region, theta_regions = [], []
    for i_reg in range(n_regions):
        depth = 0.3 + 0.3 * i_reg
        Fp = 1e-3 * (1.0 - _gaussian_line(wave, 1.506 + 0.002 * i_reg, 1e-4, depth))
        Fp_by_region.append(Fp)
        theta_regions.append(_theta_dict(f'region_{i_reg + 1}', depth))
    Fstar = 1.0 - _gaussian_line(wave, 1.513, 1e-4, 0.2) if kind_trans == 'emission' else None
    return PlanetModel(wave=wave, Fp_by_region=Fp_by_region, Fstar=Fstar,
                       theta_regions=theta_regions, native_resolution=NATIVE_RES,
                       kind_trans=kind_trans, region_id=list(range(1, n_regions + 1)))


def synthetic_get_ker(n_ker=21):
    """A `get_ker`-like function: same gaussian kernel per region, phase-dependent
    visibility weights (regions rotating into/out of view), summing to 1 across
    regions at any phase like the real citrus kernel."""
    x = np.linspace(-3, 3, n_ker)
    gauss = np.exp(-0.5 * x ** 2)
    gauss /= gauss.sum()

    def get_ker(theta_regions, phase=None, planet=None, instrum=None, model_resolution=None):
        w1 = 0.5 + 0.4 * np.cos(2 * np.pi * phase)
        weights = [w1, 1.0 - w1] if len(theta_regions) == 2 else [1.0]
        return [w * gauss for w in weights]

    return get_ker


def noop_get_ker(theta_regions, phase=None, planet=None, instrum=None, model_resolution=None):
    return [None for _ in theta_regions]


class _DummyQuantity:
    def __init__(self, value):
        self.value = value

    def to(self, unit):
        return self


class _DummyPlanet:
    """Enough of `Planet` for `_build_default_rotation_kernel`'s 'emission' branch."""

    def __init__(self, period_seconds=1.22 * 24 * 3600.0):
        self.period = [_DummyQuantity(period_seconds)]


INSTRUM = dict(name='SPIRou-APERO', resol=INSTRU_RES)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

class TestPlanetModelSaveLoad:

    def test_native_single_region_round_trip(self, tmp_path):
        model = synthetic_native_model()
        path = tmp_path / 'model.npz'
        save_planet_model(path, model)
        loaded = load_planet_model(path)

        np.testing.assert_array_equal(loaded.wave, model.wave)
        assert loaded.n_regions == 1 and not loaded.is_multi_region
        np.testing.assert_array_equal(loaded.Fp_by_region[0], model.Fp_by_region[0])
        np.testing.assert_array_equal(loaded.Fstar, model.Fstar)
        assert not isinstance(loaded.Fp_by_region[0], np.ma.MaskedArray)  # native: plain arrays
        assert loaded.native_resolution == NATIVE_RES
        assert loaded.kind_trans == 'emission'
        assert loaded.region_id == [1]
        assert loaded.is_native and loaded.resolution is None
        assert loaded.rotation_kernel_applied is None and loaded.phase is None

    def test_theta_dict_values_survive_the_json_round_trip(self, tmp_path):
        model = synthetic_native_model()
        path = tmp_path / 'model.npz'
        save_planet_model(path, model)
        theta = load_planet_model(path).theta_regions[0]
        ref = model.theta_regions[0]

        assert set(theta) == set(ref)
        # numpy arrays come back as arrays (tagged in the JSON), not as plain lists
        assert isinstance(theta['pressures'], np.ndarray)
        np.testing.assert_array_equal(theta['pressures'], ref['pressures'])
        np.testing.assert_array_equal(theta['temperatures'], ref['temperatures'])
        # numpy scalars -> Python floats, None/str untouched
        assert theta['CO'] == pytest.approx(float(ref['CO']))
        assert theta['wind'] is None
        assert theta['tag'] == 'region_1'
        assert theta['R_pl'] == ref['R_pl']

    def test_degraded_multi_region_round_trip_restores_masks_and_form(self, tmp_path):
        model = degrade(synthetic_native_model(n_regions=2), INSTRU_RES)
        model = replace(model, rotation_kernel_applied='multi_region', phase=0.55)
        path = tmp_path / 'model.npz'
        save_planet_model(path, model)
        loaded = load_planet_model(path)

        assert loaded.is_multi_region and loaded.n_regions == 2
        assert loaded.resolution == INSTRU_RES and not loaded.is_native
        assert loaded.rotation_kernel_applied == 'multi_region'
        assert loaded.phase == 0.55
        assert loaded.region_id == [1, 2]
        for Fp_loaded, Fp in zip(loaded.Fp_by_region, model.Fp_by_region):
            assert isinstance(Fp_loaded, np.ma.MaskedArray)
            np.testing.assert_array_equal(np.ma.getmaskarray(Fp_loaded), np.ma.getmaskarray(Fp))
            np.testing.assert_array_equal(Fp_loaded.compressed(), Fp.compressed())
        np.testing.assert_array_equal(loaded.Fstar.compressed(), model.Fstar.compressed())

    def test_transmission_model_has_no_fstar(self, tmp_path):
        model = synthetic_native_model(kind_trans='transmission')
        path = tmp_path / 'model.npz'
        save_planet_model(path, model)
        loaded = load_planet_model(path)
        assert loaded.Fstar is None and loaded.kind_trans == 'transmission'

    def test_unserializable_theta_value_raises_clearly(self, tmp_path):
        model = synthetic_native_model()
        model.theta_regions[0]['bad'] = 1.0 * u.km / u.s  # a Quantity, not a plain float
        with pytest.raises(TypeError, match='bad|Quantity'):
            save_planet_model(tmp_path / 'model.npz', model)

    def test_describe_reports_the_form(self):
        model = synthetic_native_model(n_regions=2)
        assert '2 region(s)' in model.describe() and 'native sampling' in model.describe()
        frozen = replace(degrade(model, INSTRU_RES), phase=0.5,
                                  rotation_kernel_applied='multi_region')
        text = frozen.describe()
        assert 'degraded to R=70000' in text and 'multi_region' in text and 'phase 0.5000' in text


# ---------------------------------------------------------------------------
# Forms / transformations
# ---------------------------------------------------------------------------

class TestPlanetModelTransformations:

    def test_default_kernel_then_degrade_matches_degrade_fp_fstar(self):
        """The two-step form path must reproduce `degrade_fp_fstar(rotation_kernel=...)`
        (what lnprob's single-region branch does) bit for bit."""
        model = synthetic_native_model()
        planet = _DummyPlanet()

        with_kernel = apply_default_rotation_kernel(model, 'emission', planet)
        assert with_kernel.rotation_kernel_applied == 'emission'
        assert with_kernel.is_native
        degraded = degrade(with_kernel, INSTRU_RES)
        assert degraded.resolution == INSTRU_RES

        wave_ref, Fp_ref, Fstar_ref = degrade_fp_fstar(
            model.wave, model.Fp_by_region[0], model.Fstar, resolution=INSTRU_RES,
            native_resolution=NATIVE_RES, theta_dict=model.theta_regions[0], planet=planet,
            rotation_kernel='emission')

        np.testing.assert_array_equal(degraded.wave, wave_ref)
        np.testing.assert_array_equal(np.ma.filled(degraded.Fp_by_region[0], np.nan),
                                      np.ma.filled(Fp_ref, np.nan))
        np.testing.assert_array_equal(np.ma.filled(degraded.Fstar, np.nan),
                                      np.ma.filled(Fstar_ref, np.nan))

    def test_degrade_without_kernel_matches_degrade_fp_fstar(self):
        model = synthetic_native_model()
        degraded = degrade(model, INSTRU_RES)
        wave_ref, Fp_ref, Fstar_ref = degrade_fp_fstar(
            model.wave, model.Fp_by_region[0], model.Fstar, resolution=INSTRU_RES,
            native_resolution=NATIVE_RES)
        np.testing.assert_array_equal(degraded.wave, wave_ref)
        np.testing.assert_array_equal(np.ma.filled(degraded.Fp_by_region[0], np.nan),
                                      np.ma.filled(Fp_ref, np.nan))
        np.testing.assert_array_equal(np.ma.filled(degraded.Fstar, np.nan),
                                      np.ma.filled(Fstar_ref, np.nan))
        # Every region and Fstar share the trimmed grid
        assert degraded.wave.size == degraded.Fp_by_region[0].size == degraded.Fstar.size

    def test_no_default_kernel_returns_the_model_unchanged(self):
        model = synthetic_native_model()
        assert apply_default_rotation_kernel(model, None, _DummyPlanet()) is model

    def test_default_kernel_refuses_multi_region_degraded_or_already_applied(self):
        planet = _DummyPlanet()
        with pytest.raises(ValueError, match='single-region'):
            apply_default_rotation_kernel(synthetic_native_model(n_regions=2), 'emission', planet)
        with pytest.raises(ValueError, match='native sampling'):
            apply_default_rotation_kernel(degrade(synthetic_native_model(), INSTRU_RES),
                                          'emission', planet)
        once = apply_default_rotation_kernel(synthetic_native_model(), 'emission', planet)
        with pytest.raises(ValueError, match='already applied'):
            apply_default_rotation_kernel(once, 'emission', planet)

    def test_degrade_twice_raises(self):
        with pytest.raises(ValueError, match='already degraded'):
            degrade(degrade(synthetic_native_model(), INSTRU_RES), INSTRU_RES)

    def test_combine_at_phase_matches_the_per_exposure_region_kernel(self):
        model = synthetic_native_model(n_regions=2)
        get_ker = synthetic_get_ker()
        phase = 0.37

        frozen = combine_at_phase(model, get_ker, planet=None, instrum=INSTRUM, phase=phase)

        region_kernel, trim = build_region_kernel(model.theta_regions, get_ker, None, INSTRUM,
                                                  NATIVE_RES)
        wave_ref, Fp_ref, Fstar_ref = prepare_multi_region_visit(
            model.wave, model.Fp_by_region, model.Fstar, trim, INSTRU_RES, NATIVE_RES)
        Fp_combined_ref = region_kernel(wave_ref, Fp_ref, phase)

        assert frozen.n_regions == 1 and not frozen.is_multi_region
        assert frozen.resolution == INSTRU_RES
        assert frozen.rotation_kernel_applied == 'multi_region'
        assert frozen.phase == phase
        assert len(frozen.theta_regions) == 2  # provenance kept
        np.testing.assert_array_equal(frozen.wave, wave_ref)
        np.testing.assert_array_equal(frozen.Fp_by_region[0], Fp_combined_ref)
        np.testing.assert_array_equal(np.ma.filled(frozen.Fstar, np.nan),
                                      np.ma.filled(Fstar_ref, np.nan))

    def test_combine_at_phase_needs_a_native_multi_region_model(self):
        get_ker = synthetic_get_ker()
        with pytest.raises(ValueError, match='multi-region'):
            combine_at_phase(synthetic_native_model(), get_ker, None, INSTRUM, 0.5)
        with pytest.raises(ValueError, match='native'):
            combine_at_phase(degrade(synthetic_native_model(n_regions=2), INSTRU_RES),
                             get_ker, None, INSTRUM, 0.5)


# ---------------------------------------------------------------------------
# Shared multi-region helpers (moved out of retrieval.py's globals)
# ---------------------------------------------------------------------------

class TestBuildRegionKernel:

    def test_noop_get_ker_raises(self):
        model = synthetic_native_model(n_regions=2)
        with pytest.raises(ValueError, match='get_ker_file'):
            build_region_kernel(model.theta_regions, noop_get_ker, None, INSTRUM, NATIVE_RES)

    def test_trim_is_the_probed_kernel_half_width(self):
        model = synthetic_native_model(n_regions=2)
        _, trim = build_region_kernel(model.theta_regions, synthetic_get_ker(n_ker=41), None,
                                      INSTRUM, NATIVE_RES)
        assert trim == 20

    def test_closure_combines_with_spec_scale_weights(self):
        model = synthetic_native_model(n_regions=2)
        model.theta_regions[1]['spec_scale'] = 0.5
        get_ker = synthetic_get_ker()
        region_kernel, trim = build_region_kernel(model.theta_regions, get_ker, None, INSTRUM,
                                                  NATIVE_RES)
        wave = model.wave[trim:-trim]
        phase = 0.2

        out = region_kernel(wave, model.Fp_by_region, phase)

        kernels = get_ker(model.theta_regions, phase=phase, instrum=INSTRUM,
                          model_resolution=NATIVE_RES)
        expected = combine_regions_with_kernel(wave, model.Fp_by_region, kernels, [1.0, 0.5])
        np.testing.assert_array_equal(out, expected)
        assert out.size == wave.size

    def test_get_ker_receives_the_explicit_arguments(self):
        """No module globals involved: planet/instrum/model_resolution reach get_ker
        exactly as passed (this is what lets logl_grid.py reuse the function)."""
        seen = {}

        def spy_get_ker(theta_regions, phase=None, planet=None, instrum=None,
                        model_resolution=None):
            seen.update(planet=planet, instrum=instrum, model_resolution=model_resolution,
                        phase=phase)
            return synthetic_get_ker()(theta_regions, phase=phase)

        model = synthetic_native_model(n_regions=2)
        sentinel_planet = object()
        region_kernel, _ = build_region_kernel(model.theta_regions, spy_get_ker, sentinel_planet,
                                               INSTRUM, NATIVE_RES)
        assert seen['phase'] == 0.0  # the probe
        region_kernel(model.wave[10:-10], model.Fp_by_region, 0.42)
        assert seen == dict(planet=sentinel_planet, instrum=INSTRUM,
                            model_resolution=NATIVE_RES, phase=0.42)


class TestPrepareMultiRegionVisit:

    def test_trims_wave_degrades_fstar_and_leaves_fp_native(self):
        model = synthetic_native_model(n_regions=2)
        trim = 10
        wave, Fp_by_region, Fstar = prepare_multi_region_visit(
            model.wave, model.Fp_by_region, model.Fstar, trim, INSTRU_RES, NATIVE_RES)

        np.testing.assert_array_equal(wave, model.wave[trim:-trim])
        # Fp are passed through untouched (native, un-trimmed) -- same objects
        assert all(Fp is Fp_ref for Fp, Fp_ref in zip(Fp_by_region, model.Fp_by_region))
        assert isinstance(Fstar, np.ma.MaskedArray) and Fstar.size == wave.size
        # The star really was degraded: its line got shallower (resolved -> blurred)
        assert Fstar.min() > model.Fstar[trim:-trim].min()

    def test_zero_trim_and_no_fstar(self):
        model = synthetic_native_model(n_regions=2, kind_trans='transmission')
        wave, _, Fstar = prepare_multi_region_visit(
            model.wave, model.Fp_by_region, None, 0, INSTRU_RES, NATIVE_RES)
        assert wave.size == model.wave.size and Fstar is None


# ---------------------------------------------------------------------------
# Precomputed splines
# ---------------------------------------------------------------------------

def _data_wave(n_exp=4, n_ord=2, n_pix=300):
    orders = [np.linspace(1.504, 1.508, n_pix), np.linspace(1.511, 1.515, n_pix)][:n_ord]
    return np.broadcast_to(np.array(orders), (n_exp, n_ord, n_pix)).copy()


class TestBuildModelInterpolators:

    def _degraded_single(self):
        model = degrade(synthetic_native_model(), INSTRU_RES)
        return model.wave[20:-20], model.Fp_by_region[0][20:-20], model.Fstar[20:-20]

    def test_single_spline_output_is_identical_with_or_without_precomputation(self):
        wave, Fp, Fstar = self._degraded_single()
        data_wave = _data_wave()
        vrp_orb = np.array([-40.0, -10.0, 10.0, 40.0])
        kwargs = dict(vr_orb=0.0, alpha=np.array([0.6, 0.8, 1.0, 0.9]), kind_trans='emission',
                      RV=15.0)

        direct = build_model_sequence(wave, Fp, data_wave, vrp_orb, Fstar=Fstar, **kwargs)
        interp = build_model_interpolators(wave, Fp, Fstar, 'emission')
        assert not interp.per_exposure and interp.fct_star is not None
        cached = build_model_sequence(None, None, data_wave, vrp_orb, interpolators=interp,
                                      **kwargs)

        np.testing.assert_array_equal(np.ma.filled(cached, np.nan), np.ma.filled(direct, np.nan))

    def test_per_exposure_splines_output_is_identical_with_or_without_precomputation(self):
        model = synthetic_native_model(n_regions=2)
        get_ker = synthetic_get_ker()
        region_kernel, trim = build_region_kernel(model.theta_regions, get_ker, None, INSTRUM,
                                                  NATIVE_RES)
        wave, Fp_by_region, Fstar = prepare_multi_region_visit(
            model.wave, model.Fp_by_region, model.Fstar, trim, INSTRU_RES, NATIVE_RES)
        data_wave = _data_wave()
        vrp_orb = np.array([-40.0, -10.0, 10.0, 40.0])
        phase = np.array([0.45, 0.48, 0.52, 0.55])
        kwargs = dict(vr_orb=0.0, alpha=1.0, kind_trans='emission', RV=-5.0)

        direct = build_model_sequence(wave, Fp_by_region, data_wave, vrp_orb, Fstar=Fstar,
                                      region_kernel=region_kernel, phase=phase, **kwargs)
        interp = build_model_interpolators(wave, Fp_by_region, Fstar, 'emission',
                                           region_kernel=region_kernel, phase=phase)
        assert interp.per_exposure and len(interp.fct_p_per_exp) == 4
        cached = build_model_sequence(None, None, data_wave, vrp_orb, interpolators=interp,
                                      **kwargs)

        np.testing.assert_array_equal(np.ma.filled(cached, np.nan), np.ma.filled(direct, np.nan))

    def test_scalar_phase_needs_n_exp(self):
        model = synthetic_native_model(n_regions=2)
        region_kernel, trim = build_region_kernel(model.theta_regions, synthetic_get_ker(), None,
                                                  INSTRUM, NATIVE_RES)
        wave = model.wave[trim:-trim]
        with pytest.raises(ValueError, match='n_exp'):
            build_model_interpolators(wave, model.Fp_by_region, None, 'transmission',
                                      region_kernel=region_kernel, phase=0.5)
        interp = build_model_interpolators(wave, model.Fp_by_region, None, 'transmission',
                                           region_kernel=region_kernel, phase=0.5, n_exp=3)
        assert len(interp.fct_p_per_exp) == 3

    def test_exposure_count_mismatch_raises(self):
        model = synthetic_native_model(n_regions=2)
        region_kernel, trim = build_region_kernel(model.theta_regions, synthetic_get_ker(), None,
                                                  INSTRUM, NATIVE_RES)
        interp = build_model_interpolators(model.wave[trim:-trim], model.Fp_by_region, None,
                                           'transmission', region_kernel=region_kernel,
                                           phase=np.array([0.1, 0.2, 0.3]))
        with pytest.raises(ValueError, match='exposures'):
            build_model_sequence(None, None, _data_wave(n_exp=4), 0.0, kind_trans='transmission',
                                 interpolators=interp)

    def test_emission_without_stellar_spline_raises(self):
        wave, Fp, _ = self._degraded_single()
        interp = build_model_interpolators(wave, Fp, None, 'transmission')
        assert interp.fct_star is None
        with pytest.raises(ValueError, match='fct_star'):
            build_model_sequence(None, None, _data_wave(), 0.0, kind_trans='emission',
                                 interpolators=interp)

    def test_transmission_ignores_fstar(self):
        wave, Fp, Fstar = self._degraded_single()
        interp = build_model_interpolators(wave, Fp, Fstar, 'transmission')
        assert interp.fct_star is None


# ---------------------------------------------------------------------------
# Legacy combined-ratio models (old-engine .npz files)
# ---------------------------------------------------------------------------

class TestLegacyCombinedRatio:

    def _legacy_file(self, tmp_path, wave_key='wave', spec_key='spec', spec=None):
        wave = _native_wave()[::4]
        if spec is None:
            spec = 1e-3 * (1.0 - _gaussian_line(wave, 1.506, 3e-4, 0.3))
        path = tmp_path / 'legacy.npz'
        np.savez(path, **{wave_key: wave, spec_key: spec})
        return path, wave, spec

    @pytest.mark.parametrize('keys', [('wave', 'spec'), ('wave_mod', 'mod_spec'),
                                      ('wave_mod', 'model')])
    def test_loads_every_old_key_convention(self, tmp_path, keys):
        path, wave, spec = self._legacy_file(tmp_path, *keys)
        model = load_planet_model(path, kind_trans='emission')
        assert model.combined_ratio and not model.is_native and not model.is_multi_region
        assert model.Fstar is None and model.resolution is None and model.native_resolution is None
        assert model.kind_trans == 'emission' and model.theta_regions == [{}]
        np.testing.assert_array_equal(model.wave, wave)
        np.testing.assert_array_equal(model.Fp_by_region[0], spec)
        assert 'legacy' in model.describe()

    def test_two_dimensional_model_key_takes_the_first_row(self, tmp_path):
        wave = _native_wave()[::4]
        spec = np.vstack([np.ones_like(wave), 2 * np.ones_like(wave)])
        path, _, _ = self._legacy_file(tmp_path, 'wave_mod', 'model', spec=spec)
        model = load_planet_model(path, kind_trans='transmission')
        np.testing.assert_array_equal(model.Fp_by_region[0], np.ones_like(wave))

    def test_kind_trans_is_required(self, tmp_path):
        path, _, _ = self._legacy_file(tmp_path)
        with pytest.raises(ValueError, match='kind_trans'):
            load_planet_model(path)

    def test_unknown_keys_are_refused_with_the_recipe(self, tmp_path):
        path = tmp_path / 'weird.npz'
        np.savez(path, foo=np.ones(3))
        with pytest.raises(ValueError, match='save_planet_model'):
            load_planet_model(path, kind_trans='emission')

    def test_legacy_form_survives_a_save_load_round_trip(self, tmp_path):
        path, _, _ = self._legacy_file(tmp_path)
        model = load_planet_model(path, kind_trans='emission')
        out = tmp_path / 'resaved.npz'
        save_planet_model(out, model)
        loaded = load_planet_model(out)
        assert loaded.combined_ratio and loaded.native_resolution is None
        np.testing.assert_array_equal(loaded.Fp_by_region[0], model.Fp_by_region[0])

    def test_cannot_be_degraded_or_kernel_convolved(self, tmp_path):
        path, _, _ = self._legacy_file(tmp_path)
        model = load_planet_model(path, kind_trans='emission')
        with pytest.raises(ValueError, match='already degraded'):
            degrade(model, INSTRU_RES)
        with pytest.raises(ValueError, match='native sampling'):
            apply_default_rotation_kernel(model, 'emission', _DummyPlanet())

    def test_emission_sequence_shifts_the_ratio_rigidly_without_a_star(self):
        """The combined ratio in emission is 1 + alpha * ratio shifted at the planet's
        velocity -- exactly a transmission-style single shift, with the emission sign."""
        wave = _native_wave()[::4]
        ratio = 1e-3 * (1.0 - _gaussian_line(wave, 1.506, 3e-4, 0.3))
        data_wave = _data_wave()
        vrp_orb = np.array([-40.0, -10.0, 10.0, 40.0])

        with pytest.raises(ValueError, match='Fstar'):
            build_model_sequence(wave, ratio, data_wave, vrp_orb, kind_trans='emission')

        seq = build_model_sequence(wave, ratio, data_wave, vrp_orb, kind_trans='emission',
                                   combined_ratio=True, RV=3.0)
        as_transmission = build_model_sequence(wave, ratio, data_wave, vrp_orb,
                                               kind_trans='transmission', RV=3.0)
        # emission: 1 + depth ; transmission: 1 - depth -> same shifted depth
        np.testing.assert_allclose(seq - 1.0, 1.0 - as_transmission, atol=1e-15)

        interp = build_model_interpolators(wave, ratio, None, 'emission', combined_ratio=True)
        assert interp.combined_ratio and interp.fct_star is None
        cached = build_model_sequence(None, None, data_wave, vrp_orb, kind_trans='emission',
                                      interpolators=interp, RV=3.0)
        np.testing.assert_array_equal(np.ma.filled(cached, np.nan), np.ma.filled(seq, np.nan))
