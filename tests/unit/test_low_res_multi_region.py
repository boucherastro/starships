"""Chantier A Phase 3f: multi-region support for the LOW RES block of `lnprob`.

Pure Python / numpy tests -- no petitRADTRANS needed. `get_representative_low_res_phases`
is plain ephemeris math; `prepare_model_multi_reg_low`'s orchestration is exercised with
`_prepare_fp_native_by_region`/`_build_multi_region_kernel` mocked out (same level
`test_get_ker_loading.py::TestPrepareModelMultiRegGetKerWiring` mocks
`prepare_model_high_or_low`/`get_ker` for the HIGH RES counterpart) -- `build_model_sequence`
itself is real and already covered by `test_model_sequence.py`. Real petitRADTRANS physics
for this path still needs validating on Narval (see `Notes/plan_revision_starships.md`).
"""
import numpy as np
import pytest

import starships.retrieval as retrieval
from starships.homemade import calc_shift


class _DummyQuantity:
    """Minimal stand-in for an astropy Quantity: `.value`, `.to(unit)`, `.decompose()`
    and division (needed for `get_representative_low_res_phases`'s
    `trandur / 2 / period` ratio)."""

    def __init__(self, value):
        self.value = value

    def to(self, unit):
        return self

    def decompose(self):
        return self

    def __truediv__(self, other):
        other_value = other.value if isinstance(other, _DummyQuantity) else other
        return _DummyQuantity(self.value / other_value)


class _DummyPlanet:
    """Just enough of `starships.planet_obs.Planet` for the functions under test.

    `period`/`trandur` are wrapped as 1-element arrays, not plain scalars --
    matching real `Planet` instances (`planet_obs.py::Planet.__init__` stores them
    straight from the ExoFile table query, which comes back as 1-element-array
    Quantities, e.g. `planet.trandur = [10272.96] s`). A regression test on Narval
    caught this the hard way: `get_representative_low_res_phases`'s `half_width`
    silently inherited that shape, and `np.linspace(center - half_width, ...)`
    with array-valued bounds returned a `(n_phases, 1)` array instead of
    `(n_phases,)` -- using scalar dummy values here would have hidden that bug the
    same way `TestDefaultRotationKernel`'s old dummy `R_pl` hid a units bug (see
    that class's docstring in `test_model_sequence.py`).
    """

    def __init__(self, period_days=None, trandur_days=None, rv_sys_kms=None):
        if period_days is not None:
            self.period = _DummyQuantity(np.array([period_days]))
        if trandur_days is not None:
            self.trandur = _DummyQuantity(np.array([trandur_days]))
        if rv_sys_kms is not None:
            self.RV_sys = _DummyQuantity(rv_sys_kms)


class TestGetRepresentativeLowResPhases:
    """Ephemeris-only, circular-orbit approximation (Antoine: eccentricity-aware
    orbit code elsewhere in the package is unreliable, so deliberately not used
    here) -- see `get_representative_low_res_phases`'s docstring."""

    def test_transmission_defaults_to_four_phases_centered_on_zero(self):
        planet = _DummyPlanet(period_days=3.0, trandur_days=0.3)  # half_width = 0.05
        phases = retrieval.get_representative_low_res_phases(planet, 'transmission')

        # Explicit shape check, not just len() -- a regression where `half_width`
        # inherits period/trandur's array shape produces a (4, 1) array, which
        # still satisfies len(phases) == 4 but is not the flat (4,) contract
        # `build_model_sequence`'s `phase` argument expects.
        assert phases.shape == (4,)
        # linspace(-0.05, 0.05, 4) % 1 -- the two negative entries wrap around near 1.0.
        expected = np.linspace(-0.05, 0.05, 4) % 1
        np.testing.assert_allclose(sorted(phases), sorted(expected))

    def test_emission_defaults_to_two_phases_centered_on_half(self):
        planet = _DummyPlanet(period_days=3.0, trandur_days=0.3)  # half_width = 0.05
        phases = retrieval.get_representative_low_res_phases(planet, 'emission')

        assert phases.shape == (2,)
        np.testing.assert_allclose(sorted(phases), [0.45, 0.55])

    def test_n_phases_override(self):
        planet = _DummyPlanet(period_days=3.0, trandur_days=0.3)
        phases = retrieval.get_representative_low_res_phases(planet, 'transmission', n_phases=6)
        assert len(phases) == 6

    def test_single_phase_returns_the_center(self):
        planet = _DummyPlanet(period_days=3.0, trandur_days=0.3)
        phases = retrieval.get_representative_low_res_phases(planet, 'emission', n_phases=1)
        np.testing.assert_allclose(phases, [0.5])

    def test_phases_always_in_zero_one(self):
        # A deliberately wide trandur (relative to period) to push a phase past 1.0.
        planet = _DummyPlanet(period_days=1.0, trandur_days=0.9)
        phases = retrieval.get_representative_low_res_phases(planet, 'emission')
        assert np.all((phases >= 0.0) & (phases < 1.0))


class TestPrepareFpNativeByRegionMode:
    """`_prepare_fp_native_by_region`'s new `mode` parameter (Phase 3f) -- must
    select `linelist_names[mode]`/`prepare_abundances(theta_dict, mode)` instead of
    the hardcoded `'high'` it used before this change."""

    def test_mode_low_selects_the_low_res_linelist(self, monkeypatch):
        calls = []

        def fake_generate_native_fp_fstar(atmo_obj, species, planet, theta_dict, kind_trans,
                                          fct_star=None, **kwargs):
            calls.append(dict(species=species, specie_2_lnlst=kwargs.get('specie_2_lnlst')))
            wave = np.array([1.0, 2.0, 3.0])
            return wave, np.array([0.1, 0.2, 0.3]), None

        monkeypatch.setattr(retrieval.model_seq, 'generate_native_fp_fstar',
                            fake_generate_native_fp_fstar, raising=False)
        monkeypatch.setattr(retrieval, 'line_opacities', ['H2O'], raising=False)
        monkeypatch.setattr(retrieval, 'continuum_opacities', [], raising=False)
        monkeypatch.setattr(retrieval, 'other_species', [], raising=False)
        monkeypatch.setattr(retrieval, 'remove_mol_low', [], raising=False)
        monkeypatch.setattr(retrieval, 'remove_mol_high', [], raising=False)
        monkeypatch.setattr(retrieval, 'linelist_names',
                            {'low': {'H2O': 'H2O_lowres_lnlst'}, 'high': {'H2O': 'H2O_hires_lnlst'}},
                            raising=False)
        monkeypatch.setattr(retrieval, 'dissociation', False, raising=False)
        monkeypatch.setattr(retrieval, 'kind_trans', 'transmission', raising=False)
        monkeypatch.setattr(retrieval, 'planet', None, raising=False)

        theta_regions = [dict(H2O=1e-4, gamma_scat=1.0, scat_factor=1.0, **{'C/O': 0.5, 'Fe/H': 0.0},
                              cloud_fraction=None)]
        retrieval._prepare_fp_native_by_region(theta_regions, atmo_obj_list=[None],
                                               fct_star=None, mode='low')

        assert calls[0]['species'] == {'H2O_lowres_lnlst': 1e-4}
        assert calls[0]['specie_2_lnlst'] == {'H2O': 'H2O_lowres_lnlst'}


class TestBuildMultiRegionKernelModeAndInstrum:
    """`_build_multi_region_kernel`'s new `mode`/`instrum` parameters (Phase 3f) --
    `mode` selects `prt_res[mode]`; `instrum`, when given, overrides the default
    `instrum_param_list[tr_i]` lookup (LOW RES has no per-visit instrument list)."""

    def test_mode_selects_prt_res_and_explicit_instrum_is_forwarded(self, monkeypatch):
        calls = []

        def fake_get_ker(theta_regions, tr_i=0, phase=None, planet=None, instrum=None,
                         model_resolution=None):
            calls.append(dict(instrum=instrum, model_resolution=model_resolution))
            return [np.array([1.0]) for _ in theta_regions]

        monkeypatch.setattr(retrieval, 'get_ker', fake_get_ker, raising=False)
        monkeypatch.setattr(retrieval, 'prt_res', {'high': 250_000, 'low': 1000}, raising=False)
        monkeypatch.setattr(retrieval, 'planet', None, raising=False)
        monkeypatch.setattr(retrieval.model_seq, 'combine_regions_with_kernel',
                            lambda wave, Fp_list, rot_ker_list, weights: 'combined', raising=False)

        theta_regions = [{'spec_scale': 1.0}]
        region_kernel = retrieval._build_multi_region_kernel(
            theta_regions, tr_i=0, mode='low', instrum={'resol': 1000})
        region_kernel(wave=np.array([1.0]), Fp_by_region=[np.array([1.0])], phase_i=0.3)

        assert calls[0]['instrum'] == {'resol': 1000}
        assert calls[0]['model_resolution'] == 1000

    def test_default_instrum_falls_back_to_instrum_param_list(self, monkeypatch):
        calls = []

        def fake_get_ker(theta_regions, tr_i=0, phase=None, planet=None, instrum=None,
                         model_resolution=None):
            calls.append(dict(instrum=instrum, model_resolution=model_resolution))
            return [np.array([1.0]) for _ in theta_regions]

        monkeypatch.setattr(retrieval, 'get_ker', fake_get_ker, raising=False)
        monkeypatch.setattr(retrieval, 'prt_res', {'high': 250_000, 'low': 1000}, raising=False)
        monkeypatch.setattr(retrieval, 'planet', None, raising=False)
        monkeypatch.setattr(retrieval, 'instrum_param_list', [{'resol': 70_000}], raising=False)
        monkeypatch.setattr(retrieval.model_seq, 'combine_regions_with_kernel',
                            lambda wave, Fp_list, rot_ker_list, weights: 'combined', raising=False)

        theta_regions = [{'spec_scale': 1.0}]
        # mode defaults to 'high', instrum defaults to None -> instrum_param_list[tr_i].
        region_kernel = retrieval._build_multi_region_kernel(theta_regions, tr_i=0)
        region_kernel(wave=np.array([1.0]), Fp_by_region=[np.array([1.0])], phase_i=0.3)

        assert calls[0]['instrum'] == {'resol': 70_000}
        assert calls[0]['model_resolution'] == 250_000


class TestPrepareModelMultiRegLow:
    """`prepare_model_multi_reg_low` (Chantier A Phase 3f) -- build a fake sequence
    out of `representative_phases_low`, run it through the real
    `build_model_sequence`/`region_kernel` machinery, and average. Mocks
    `_prepare_fp_native_by_region`/`_build_multi_region_kernel` (already covered
    individually above) so this test only checks the new orchestration: fake-
    sequence construction, phase averaging, and the fixed `dv_shift` applied once
    at the end."""

    def test_averages_over_representative_phases_and_applies_fixed_shift(self, monkeypatch):
        wave_native = np.linspace(1.999, 2.001, 200)
        Fp_region_0 = np.full(wave_native.shape, 1.0)
        Fp_region_1 = np.full(wave_native.shape, 3.0)

        def fake_prepare_fp_native_by_region(theta_regions, atmo_obj_list, fct_star, mode='high'):
            assert mode == 'low'
            return wave_native, [Fp_region_0, Fp_region_1], None

        def fake_build_multi_region_kernel(theta_regions, tr_i=0, mode='high', instrum=None):
            assert mode == 'low'
            assert instrum == {'resol': 1000}

            def region_kernel(wave, Fp_by_region, phase_i):
                # Region 0 fully "visible" before phase 0.25, region 1 after --
                # matches combine_regions_with_kernel's own [15:-15] trim convention.
                w0 = 1.0 if phase_i < 0.25 else 0.0
                return w0 * Fp_by_region[0][15:-15] + (1 - w0) * Fp_by_region[1][15:-15]

            return region_kernel

        monkeypatch.setattr(retrieval, '_prepare_fp_native_by_region',
                            fake_prepare_fp_native_by_region, raising=False)
        monkeypatch.setattr(retrieval, '_build_multi_region_kernel',
                            fake_build_multi_region_kernel, raising=False)
        monkeypatch.setattr(retrieval, 'init_atmo_if_not_done', lambda mode: None, raising=False)
        monkeypatch.setattr(retrieval, 'init_stellar_spectrum_if_not_done', lambda mode: None,
                            raising=False)
        monkeypatch.setattr(retrieval, 'wv_range_low', [(1.9, 2.1)], raising=False)
        monkeypatch.setattr(retrieval, 'atmo_low_0', 'fake_atmo', raising=False)
        monkeypatch.setattr(retrieval, 'fct_star_low', None, raising=False)
        monkeypatch.setattr(retrieval, 'kind_trans', 'transmission', raising=False)
        monkeypatch.setattr(retrieval, 'prt_res', {'low': 1000}, raising=False)
        monkeypatch.setattr(retrieval, 'planet', _DummyPlanet(rv_sys_kms=5.0), raising=False)
        monkeypatch.setattr(retrieval, 'representative_phases_low', np.array([0.0, 0.5]),
                            raising=False)

        theta_regions = [dict(spec_scale=1.0, rv=1.0), dict(spec_scale=1.0)]
        wv_out, model_out = retrieval.prepare_model_multi_reg_low(theta_regions)

        # `build_model_sequence`'s injection formula (1 - alpha*depth, alpha=1.0)
        # is unwrapped back to the raw depth before returning -- regression test
        # for a real bug caught on Narval (2026-08-28): without the unwrap, this
        # function returned "1 - depth" (a HIGH RES time-series convention) instead
        # of the raw depth that prepare_photometry/prepare_spectrophotometry and
        # the single-region prepare_model_high_or_low('low') both expect.
        # phase 0.0 -> depth=1.0 (region 0 alone) ; phase 0.5 -> depth=3.0 (region 1
        # alone). Averaged raw depth over the two representative phases: 2.0.
        wave_trimmed = wave_native[15:-15]
        np.testing.assert_allclose(model_out, np.full(wave_trimmed.shape, 2.0))

        # dv_shift = planet.RV_sys (5.0) + theta_regions[0]['rv'] (1.0) = 6.0 km/s,
        # applied once, after averaging -- same convention as the single-region
        # `prepare_model_high_or_low`'s `mode == 'low'` branch.
        expected_shift = calc_shift(6.0, kind='rel')
        np.testing.assert_allclose(wv_out, wave_trimmed * expected_shift, rtol=1e-10)

    def test_emission_unwraps_the_injection_formula_correctly(self, monkeypatch):
        """Same regression as above, `kind_trans='emission'` branch (`1 + depth`,
        with a real `Fstar` so `depth = Fp/Fstar`)."""
        wave_native = np.linspace(1.999, 2.001, 200)
        Fp_region_0 = np.full(wave_native.shape, 1.0)
        Fp_region_1 = np.full(wave_native.shape, 3.0)
        Fstar_native = np.full(wave_native.shape, 2.0)

        def fake_prepare_fp_native_by_region(theta_regions, atmo_obj_list, fct_star, mode='high'):
            return wave_native, [Fp_region_0, Fp_region_1], Fstar_native

        def fake_build_multi_region_kernel(theta_regions, tr_i=0, mode='high', instrum=None):
            def region_kernel(wave, Fp_by_region, phase_i):
                w0 = 1.0 if phase_i < 0.25 else 0.0
                return w0 * Fp_by_region[0][15:-15] + (1 - w0) * Fp_by_region[1][15:-15]

            return region_kernel

        monkeypatch.setattr(retrieval, '_prepare_fp_native_by_region',
                            fake_prepare_fp_native_by_region, raising=False)
        monkeypatch.setattr(retrieval, '_build_multi_region_kernel',
                            fake_build_multi_region_kernel, raising=False)
        monkeypatch.setattr(retrieval, 'init_atmo_if_not_done', lambda mode: None, raising=False)
        monkeypatch.setattr(retrieval, 'init_stellar_spectrum_if_not_done', lambda mode: None,
                            raising=False)
        monkeypatch.setattr(retrieval, 'wv_range_low', [(1.9, 2.1)], raising=False)
        monkeypatch.setattr(retrieval, 'atmo_low_0', 'fake_atmo', raising=False)
        monkeypatch.setattr(retrieval, 'fct_star_low', None, raising=False)
        monkeypatch.setattr(retrieval, 'kind_trans', 'emission', raising=False)
        monkeypatch.setattr(retrieval, 'prt_res', {'low': 1000}, raising=False)
        monkeypatch.setattr(retrieval, 'planet', _DummyPlanet(rv_sys_kms=0.0), raising=False)
        monkeypatch.setattr(retrieval, 'representative_phases_low', np.array([0.0, 0.5]),
                            raising=False)

        theta_regions = [dict(spec_scale=1.0, rv=0.0), dict(spec_scale=1.0)]
        _, model_out = retrieval.prepare_model_multi_reg_low(theta_regions)

        # depth = Fp/Fstar: phase 0.0 -> 1.0/2.0=0.5 ; phase 0.5 -> 3.0/2.0=1.5.
        # Averaged raw depth: 1.0.
        wave_trimmed = wave_native[15:-15]
        np.testing.assert_allclose(model_out, np.full(wave_trimmed.shape, 1.0))
