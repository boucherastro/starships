"""Tests of the walker-reading / selection / merging / run-detection part of
`starships.retrieval_analysis`.

Pure Python: the walker files are real emcee HDF5 backends written by a tiny synthetic sampler
(no petitRADTRANS, no retrieval run), so they are read with the same code paths as real ones.
"""
from types import SimpleNamespace

import emcee
import numpy as np
import pytest

from starships import retrieval_analysis as ra
from starships import retrieval_utils as ru


def make_walker_file(path, n_walkers=8, n_params=3, n_steps=60, seed=0):
    """Write a small emcee walker file (Gaussian posterior) and return its path."""
    rng = np.random.default_rng(seed)
    backend = emcee.backends.HDFBackend(str(path))
    backend.reset(n_walkers, n_params)
    sampler = emcee.EnsembleSampler(n_walkers, n_params, lambda x: -0.5 * np.sum(x ** 2),
                                    backend=backend)
    sampler.run_mcmc(rng.normal(size=(n_walkers, n_params)), n_steps, progress=False)
    return path


class TestLoadPosteriorSample:

    def test_shape_and_sorted_best_last(self, tmp_path):
        f = make_walker_file(tmp_path / 'w.h5', n_walkers=8, n_params=3, n_steps=60)
        pos, logl, _ = ra.load_posterior_sample(f, tol=0)
        assert pos.shape == (60 * 8, 3)
        assert logl.shape == (60 * 8,)
        assert np.all(np.diff(logl) >= 0)          # worst first, best last
        # logl of the Gaussian posterior is a function of the parameters: rows stayed paired
        np.testing.assert_allclose(logl, -0.5 * np.sum(pos ** 2, axis=1))

    def test_discard_removes_first_steps(self, tmp_path):
        f = make_walker_file(tmp_path / 'w.h5', n_walkers=8, n_steps=60)
        pos, _, _ = ra.load_posterior_sample(f, discard=20, tol=0)
        assert pos.shape[0] == 40 * 8

    def test_discard_too_large_raises_clear_error(self, tmp_path):
        f = make_walker_file(tmp_path / 'w.h5', n_steps=60)
        with pytest.raises(ValueError, match='number of steps'):
            ra.load_posterior_sample(f, discard=60)

    def test_short_chain_falls_back_without_autocorr(self, tmp_path):
        f = make_walker_file(tmp_path / 'w.h5', n_steps=60)
        # A tolerance of 1000 can never be met with 60 steps: emcee would raise AutocorrError
        pos, logl, autocorr = ra.load_posterior_sample(f, tol=1000)
        assert autocorr is None
        assert pos.shape[0] == 60 * 8

    def test_tol_zero_gives_no_autocorr(self, tmp_path):
        f = make_walker_file(tmp_path / 'w.h5')
        assert ra.load_posterior_sample(f, tol=0)[2] is None


class TestSelectSamples:

    @pytest.fixture
    def sample(self):
        rng = np.random.default_rng(1)
        logl = np.sort(rng.normal(size=500))
        pos = rng.normal(size=(500, 3))
        return pos, logl

    def test_logl_min(self, sample):
        pos, logl = sample
        pos_sub, logl_sub = ra.select_samples(pos, logl, logl_min=0.0)
        assert np.all(logl_sub > 0.0)
        assert len(pos_sub) == len(logl_sub) == np.sum(logl > 0.0)
        assert np.all(np.diff(logl_sub) >= 0)      # still sorted

    def test_param_cuts_by_name_and_index_with_open_bounds(self, sample):
        pos, logl = sample
        names = ['kp', 'rv', 'T_eq']
        pos_sub, _ = ra.select_samples(pos, logl, param_names=names,
                                       param_cuts={'kp': (-0.5, 0.5), 2: (0.0, None)})
        assert np.all(np.abs(pos_sub[:, 0]) < 0.5)
        assert np.all(pos_sub[:, 2] > 0.0)

    def test_conditions_are_combined_with_and(self, sample):
        pos, logl = sample
        pos_sub, logl_sub = ra.select_samples(pos, logl, logl_min=0.0,
                                              param_names=['a', 'b', 'c'], param_cuts={'a': (None, 0.0)})
        assert np.all(logl_sub > 0.0) and np.all(pos_sub[:, 0] < 0.0)

    def test_unknown_parameter_name_lists_available_ones(self, sample):
        pos, logl = sample
        with pytest.raises(KeyError, match='rv'):
            ra.select_samples(pos, logl, param_names=['kp', 'rv', 'T_eq'], param_cuts={'oops': (0, 1)})

    def test_name_without_param_names_raises(self, sample):
        pos, logl = sample
        with pytest.raises(ValueError, match='param_names'):
            ra.select_samples(pos, logl, param_cuts={'kp': (0, 1)})

    def test_empty_selection_raises(self, sample):
        pos, logl = sample
        with pytest.raises(ValueError, match='No sample left'):
            ra.select_samples(pos, logl, logl_min=1e6)


class TestCombineWalkerFiles:

    def test_default_keeps_shortest_file_and_stacks_walkers(self, tmp_path):
        f1 = make_walker_file(tmp_path / 'a.h5', n_walkers=8, n_steps=60, seed=1)
        f2 = make_walker_file(tmp_path / 'b.h5', n_walkers=6, n_steps=40, seed=2)
        out = ra.combine_walker_files([f1, f2], tmp_path / 'merged.h5')

        chain = emcee.backends.HDFBackend(str(out), read_only=True).get_chain()
        assert chain.shape == (40, 8 + 6, 3)            # 40 = shortest file
        # Last 40 steps of each file, walkers of file 1 first then file 2
        np.testing.assert_array_equal(
            chain[:, :8], emcee.backends.HDFBackend(str(f1), read_only=True).get_chain()[-40:])
        np.testing.assert_array_equal(
            chain[:, 8:], emcee.backends.HDFBackend(str(f2), read_only=True).get_chain()[-40:])

    def test_merged_file_is_readable_by_load_posterior_sample(self, tmp_path):
        f1 = make_walker_file(tmp_path / 'a.h5', n_steps=30, seed=1)
        f2 = make_walker_file(tmp_path / 'b.h5', n_steps=30, seed=2)
        out = ra.combine_walker_files([f1, f2], tmp_path / 'merged.h5')
        pos, logl, _ = ra.load_posterior_sample(out, tol=0)
        assert pos.shape == (30 * 16, 3)
        np.testing.assert_allclose(logl, -0.5 * np.sum(pos ** 2, axis=1))

    def test_explicit_n_steps_and_walker_path(self, tmp_path):
        make_walker_file(tmp_path / 'a.h5', n_steps=60, seed=1)
        make_walker_file(tmp_path / 'b.h5', n_steps=60, seed=2)
        out = ra.combine_walker_files(['a.h5', 'b.h5'], 'merged.h5', walker_path=tmp_path, n_steps=20)
        assert emcee.backends.HDFBackend(str(out), read_only=True).get_chain().shape[0] == 20

    def test_n_steps_larger_than_shortest_file_raises(self, tmp_path):
        f1 = make_walker_file(tmp_path / 'a.h5', n_steps=60)
        f2 = make_walker_file(tmp_path / 'b.h5', n_steps=30)
        with pytest.raises(ValueError, match='shortest'):
            ra.combine_walker_files([f1, f2], tmp_path / 'm.h5', n_steps=40)

    def test_different_number_of_parameters_raises(self, tmp_path):
        f1 = make_walker_file(tmp_path / 'a.h5', n_params=3)
        f2 = make_walker_file(tmp_path / 'b.h5', n_params=4)
        with pytest.raises(ValueError, match='number of parameters'):
            ra.combine_walker_files([f1, f2], tmp_path / 'm.h5')

    def test_does_not_overwrite_unless_asked(self, tmp_path):
        f1 = make_walker_file(tmp_path / 'a.h5')
        out = ra.combine_walker_files([f1], tmp_path / 'm.h5')
        with pytest.raises(FileExistsError):
            ra.combine_walker_files([f1], out)
        ra.combine_walker_files([f1], out, overwrite=True)   # explicit opt-in works


def fake_retrieval(retrieval_type='JR', region_id=(1,), wv_range_high=((0.9, 2.5),),
                   wv_range_low=((1.0, 5.0),), spectrophotometric=None, photometric=None):
    """Stand-in for the `retrieval` module with just the globals `describe_run` reads."""
    return SimpleNamespace(retrieval_type=retrieval_type, kind_trans='emission',
                           region_id=list(region_id), wv_range_high=list(wv_range_high),
                           wv_range_low=list(wv_range_low),
                           spectrophotometric_data=spectrophotometric, photometric_data=photometric)


class TestDescribeRun:

    def test_lrr_with_instrum_can_still_model_high_res(self):
        # Like the KELT-20b LRR yaml: no high-res data fitted, but `instrum` gives a wv_range_high
        info = ra.describe_run(fake_retrieval('LRR',
                                              spectrophotometric={'wfc3': {}, 'g395H_1': {}},
                                              photometric={'spitzer': {}}))
        assert not info.has_high_res_data
        assert info.can_model_high and info.can_model_low
        assert info.low_res_instruments == {'wfc3': 'low', 'g395H_1': 'low', 'spitzer': 'low'}
        assert info.has_low_res_data and not info.is_multi_region

    def test_lrr_without_any_high_res_wavelength_range(self):
        info = ra.describe_run(fake_retrieval('LRR', wv_range_high=[], spectrophotometric={'x': {}}))
        assert not info.can_model_high
        assert info.can_model_low

    def test_hrr_without_low_res_data(self):
        info = ra.describe_run(fake_retrieval('HRR'))     # data dicts are empty/None
        assert info.has_high_res_data and info.can_model_high
        assert not info.has_low_res_data
        assert info.low_res_instruments == {}

    def test_model_type_assigned_by_the_retrieval_is_reported(self):
        info = ra.describe_run(fake_retrieval(
            'JR', spectrophotometric={'g395H_1': {'model_type': 'high'}, 'wfc3': {'model_type': 'low'}}))
        assert info.spectrophotometric == {'g395H_1': 'high', 'wfc3': 'low'}

    def test_multi_region(self):
        info = ra.describe_run(fake_retrieval(region_id=(1, 2)))
        assert info.is_multi_region and info.n_regions == 2

    def test_summary_mentions_the_key_facts(self):
        text = ra.describe_run(fake_retrieval('LRR', region_id=(1, 2),
                                              photometric={'spitzer': {'model_type': 'high'}})).summary()
        assert 'LRR' in text and 'multi-region' in text and 'spitzer' in text and 'lbl' in text


# =====================================================================================
# Profiles, spectra and kernels drawn from a sample
# =====================================================================================

PRESSURES = np.logspace(-6, 2, 30)
SPECIES = ['H2O', 'CO']


def fake_unpack_theta(params):
    """Two regions; `params` = [H2O_1, CO_1, T_1, H2O_2, CO_2, T_2] (one region if only 3 values)."""
    regions = []
    for i in range(0, len(params), 3):
        h2o, co, temp = params[i:i + 3]
        regions.append({'pressures': PRESSURES, 'temperatures': np.full(len(PRESSURES), temp),
                        'H2O': h2o, 'CO': co, 'H-': 1e-8, 'e-': 1e-6})
    return regions


def fake_static_model(theta_list, mode, Raf=None, atmo_obj=None):
    """Spectrum = wave * (sum of the species abundances of all the regions given)."""
    wave = np.linspace(1., 2., 50)
    spec = wave * sum(theta[sp] for theta in theta_list for sp in SPECIES)
    return wave, spec, 1000


def fake_multi_region_retrieval(**kwargs):
    attrs = dict(region_id=[1, 2], unpack_theta=fake_unpack_theta, line_opacities=list(SPECIES),
                 continuum_opacities=['H-'], other_species=['e-'], dissociation=False,
                 chemical_equilibrium=False, prepare_static_model=fake_static_model,
                 kind_trans='emission')
    attrs.update(kwargs)
    return SimpleNamespace(**attrs)


class TestDrawSampleIndices:

    def test_distinct_and_capped_by_sample_size(self):
        idx = ra.draw_sample_indices(50, 20)
        assert len(idx) == 20 and len(set(idx)) == 20
        assert len(ra.draw_sample_indices(5, 100)) == 5

    def test_seed_gives_the_same_draw(self):
        np.testing.assert_array_equal(ra.draw_sample_indices(10, 1000, seed=3),
                                      ra.draw_sample_indices(10, 1000, seed=3))


class TestRegionProfiles:

    def test_one_profile_per_region_with_its_own_temperature(self):
        pressures, profiles = ra.compute_region_profiles(fake_unpack_theta([1e-3, 1e-3, 2000., 1e-4, 1e-3, 1500.]),
                                                         fake_multi_region_retrieval())
        np.testing.assert_array_equal(pressures, PRESSURES)
        assert len(profiles) == 2
        assert np.all(profiles[0]['temperature'] == 2000.) and np.all(profiles[1]['temperature'] == 1500.)
        # Species of the model + the H2/He that `gen_abundances` adds
        assert {'H2O', 'CO', 'H-', 'e-', 'H2', 'He'} <= set(profiles[0])
        np.testing.assert_allclose(profiles[0]['H2O'], 1e-3)
        np.testing.assert_allclose(profiles[1]['H2O'], 1e-4)

    def test_chemical_equilibrium_is_refused(self):
        with pytest.raises(NotImplementedError):
            ra.compute_region_profiles(fake_unpack_theta([1e-3, 1e-3, 2000.]),
                                       fake_multi_region_retrieval(chemical_equilibrium=True))

    def test_draw_shapes_and_clipping(self):
        rng = np.random.default_rng(0)
        flat = np.column_stack([rng.uniform(1e-5, 1e-3, 40), rng.uniform(1e-5, 1e-3, 40),
                                rng.uniform(1000, 3000, 40)] * 2)
        flat[:, 3] = 0.                                  # H2O abundance of region 2 is zero
        pressures, samples = ra.draw_region_profiles(25, flat, fake_multi_region_retrieval(), seed=1)
        assert len(samples) == 2
        assert samples[0]['temperature'].shape == (25, len(PRESSURES))
        assert np.all(samples[1]['H2O'] == 1e-20)        # clipped, so log-space stats stay finite
        assert np.all(samples[0]['H2O'] > 1e-20)

    def test_same_seed_same_draws_for_all_regions(self):
        flat = np.column_stack([np.linspace(1e-5, 1e-3, 40)] * 2 + [np.linspace(1000, 3000, 40)]) 
        flat = np.column_stack([flat, flat])
        _, samples = ra.draw_region_profiles(10, flat, fake_multi_region_retrieval(), seed=4)
        # Both regions come from the same rows: their temperatures are identical here
        np.testing.assert_array_equal(samples[0]['temperature'], samples[1]['temperature'])

    def test_profile_stats_keys(self):
        _, samples = ra.draw_region_profiles(15, np.tile([1e-3, 1e-3, 2000., 1e-3, 1e-3, 1500.], (40, 1))
                                             * np.random.default_rng(0).uniform(0.9, 1.1, (40, 6)),
                                             fake_multi_region_retrieval(), seed=0)
        stats = ra.compute_profile_stats(samples[0])
        assert set(stats) == set(samples[0])
        assert {'median', '1-sig', '2-sig', '3-sig'} <= set(stats['temperature'])
        assert stats['H2O']['1-sig'].shape == (2, len(PRESSURES))


class TestDrawSpectra:

    @pytest.fixture
    def flat(self):
        rng = np.random.default_rng(2)
        return np.column_stack([rng.uniform(1e-4, 1e-3, 30), rng.uniform(1e-4, 1e-3, 30), np.full(30, 2000.),
                                rng.uniform(1e-4, 1e-3, 30), rng.uniform(1e-4, 1e-3, 30), np.full(30, 1500.)])

    def test_total_spectrum_only_by_default(self, flat):
        waves, spectra = ra.draw_spectra(8, flat, fake_multi_region_retrieval(), seed=0)
        assert list(spectra) == ['global'] and list(spectra['global']) == ['All']
        assert spectra['global']['All'].shape == (8, 50) and len(waves['global']) == 50

    def test_by_region_gives_each_region_alone_and_the_global_sum(self, flat):
        waves, spectra = ra.draw_spectra(6, flat, fake_multi_region_retrieval(), by_region=True, seed=0)
        assert set(spectra) == {'global', 'region_1', 'region_2'}
        # In the fake, the spectrum of the whole planet is the sum of the regions
        np.testing.assert_allclose(spectra['global']['All'],
                                   spectra['region_1']['All'] + spectra['region_2']['All'])

    def test_single_region_ignores_by_region(self, flat):
        one_region = flat[:, :3]
        _, spectra = ra.draw_spectra(5, one_region, fake_multi_region_retrieval(region_id=[1]),
                                     by_region=True, seed=0)
        assert list(spectra) == ['global']

    def test_species_contributions(self, flat):
        waves, spectra = ra.draw_spectra(4, flat, fake_multi_region_retrieval(), species_list=['H2O'], seed=0)
        s = spectra['global']
        assert {'All', 'No Species', 'H2O only', 'Without H2O'} <= set(s)
        np.testing.assert_allclose(s['H2O only'] + s['Without H2O'], s['All'], rtol=1e-6)
        assert np.all(s['No Species'] < 1e-90 * 1e3)      # abundances set to 1e-99


def fake_retrieval_with_low_res(spectrophotometric=None, photometric=None):
    def synth(wv, spec, res, infos, *extra):
        return None, np.array([spec.sum()])
    return SimpleNamespace(retrieval_type='JR', kind_trans='emission', region_id=[1],
                           wv_range_high=[[0.9, 2.5]], wv_range_low=[[1.0, 5.0]],
                           spectrophotometric_data=spectrophotometric or {},   # like the real module: {} not None
                           photometric_data=photometric or {},
                           prt_res={'high': 250000, 'low': 1000}, res_instru=70000,
                           prepare_spectrophotometry=synth, prepare_photometry=synth)


class TestSyntheticDataSamples:

    def test_each_instrument_uses_its_own_model(self):
        retrieval_obj = fake_retrieval_with_low_res({'nirspec': {'model_type': 'low'}},
                                                    {'spitzer': {'model_type': 'high'}})
        low = np.ones((3, 10))          # sums to 10
        high = 5 * np.ones((3, 20))     # sums to 100
        out = ra.synthetic_data_samples(low, retrieval_obj, wave_low=np.arange(10.),
                                        spec_samples_high=high, wave_high=np.arange(20.))
        assert out['nirspec'].shape == (3, 1) and np.all(out['nirspec'] == 10.)
        assert out['spitzer'].shape == (3, 1) and np.all(out['spitzer'] == 100.)

    def test_missing_spectra_for_an_instrument_raises(self):
        retrieval_obj = fake_retrieval_with_low_res(None, {'spitzer': {'model_type': 'high'}})
        with pytest.raises(ValueError, match='spec_samples_high'):
            ra.synthetic_data_samples(np.ones((2, 5)), retrieval_obj, wave_low=np.arange(5.))

    def test_high_only_needs_no_low_res_spectra(self):
        retrieval_obj = fake_retrieval_with_low_res({'g395H': {'model_type': 'high'}})
        out = ra.synthetic_data_samples(None, retrieval_obj, spec_samples_high=np.ones((4, 8)),
                                        wave_high=np.arange(8.))
        assert out['g395H'].shape == (4, 1)


class TestDrawRotationKernels:

    @staticmethod
    def make_retrieval(none_kernel=False, instrum_list=None):
        calls = []

        def get_ker(theta_regions, phase=None, planet=None, instrum=None, model_resolution=None):
            calls.append(dict(phase=phase, instrum=instrum, model_resolution=model_resolution))
            if none_kernel:
                return [None for _ in theta_regions]
            # Kernel size depends on the parameters (like a wind speed), as in the real ones
            n = 11 + 2 * int(theta_regions[0]['H2O'] * 1e4)
            x = np.arange(n) - (n - 1) / 2
            ker_hot = np.exp(-0.5 * (x / 2.) ** 2)
            ker_cold = np.exp(-0.5 * (x / 3.) ** 2)
            total = ker_hot.sum() + ker_cold.sum()
            return [ker_hot / total, ker_cold / total]

        retrieval_obj = fake_multi_region_retrieval(
            get_ker=get_ker, planet=object(), prt_res={'high': 250000},
            instrum_param_list=[{'resol': 70000}] if instrum_list is None else instrum_list)
        return retrieval_obj, calls

    @pytest.fixture
    def flat(self):
        return np.column_stack([np.linspace(1e-4, 5e-4, 20), np.full(20, 1e-3), np.full(20, 2000.),
                                np.full(20, 1e-4), np.full(20, 1e-3), np.full(20, 1500.)])

    def test_common_velocity_axis_and_shapes(self, flat):
        retrieval_obj, calls = self.make_retrieval()
        velocity, kernels = ra.draw_rotation_kernels(10, flat, retrieval_obj, seed=0)
        assert kernels.shape == (10, 2, len(velocity))
        # Symmetric axis, with the pixel size of the model resolution (c / R = 1.2 km/s here)
        np.testing.assert_allclose(velocity, -velocity[::-1])
        np.testing.assert_allclose(np.diff(velocity), 299792.458 / 250000, rtol=1e-6)
        # Every draw still sums to ~1 over the regions (padding with zeros must not lose weight)
        np.testing.assert_allclose(kernels.sum(axis=(1, 2)), 1., atol=1e-3)
        # The two regions have the same peak but different widths (the cold one is wider)
        assert kernels[0, 1].sum() > kernels[0, 0].sum()

    def test_default_phase_and_instrument(self, flat):
        retrieval_obj, calls = self.make_retrieval()
        ra.draw_rotation_kernels(3, flat, retrieval_obj, seed=0)
        assert calls[0]['phase'] == 0.5                     # emission
        assert calls[0]['instrum'] == {'resol': 70000}
        assert calls[0]['model_resolution'] == 250000
        retrieval_obj.kind_trans = 'transmission'
        ra.draw_rotation_kernels(3, flat, retrieval_obj, seed=0)
        assert calls[-1]['phase'] == 0.0
        ra.draw_rotation_kernels(3, flat, retrieval_obj, phase=0.62, seed=0)
        assert calls[-1]['phase'] == 0.62

    def test_run_without_get_ker_file_raises(self, flat):
        retrieval_obj, _ = self.make_retrieval(none_kernel=True)
        with pytest.raises(ValueError, match='get_ker_file'):
            ra.draw_rotation_kernels(3, flat, retrieval_obj)

    def test_no_instrument_raises_unless_given(self, flat):
        retrieval_obj, calls = self.make_retrieval(instrum_list=[])
        with pytest.raises(ValueError, match='instrum'):
            ra.draw_rotation_kernels(3, flat, retrieval_obj)
        ra.draw_rotation_kernels(3, flat, retrieval_obj, instrum={'resol': 50000})
        assert calls[0]['instrum'] == {'resol': 50000}


class TestEquilibriumProfiles:
    """Logic of `equilibrium_profiles` with a fake petitRADTRANS table (the real one needs petitRADTRANS,
    it is checked against the original notebook code on Narval)."""

    @pytest.fixture
    def fake_prt(self, monkeypatch):
        import sys
        import types
        calls = []

        def interpol_abundances(c_to_o, fe_to_h, temperature, pressures):
            calls.append(dict(c_to_o=np.array(c_to_o), fe_to_h=np.array(fe_to_h), temperature=np.array(temperature)))
            # Fake "mass fractions": the value encodes the inputs, so the test can check what was passed
            return {'H2O': fe_to_h * 0 + 0.1, 'CO': fe_to_h * 0 + 0.2, 'MMW': fe_to_h * 0 + 2.3}

        pkg = types.ModuleType('petitRADTRANS')
        mod = types.ModuleType('petitRADTRANS.poor_mans_nonequ_chem')
        mod.interpol_abundances = interpol_abundances
        monkeypatch.setitem(sys.modules, 'petitRADTRANS', pkg)
        monkeypatch.setitem(sys.modules, 'petitRADTRANS.poor_mans_nonequ_chem', mod)
        monkeypatch.setattr(ra, 'mass_frac_2_vmr',
                            lambda mass_frac_dict: {k: v for k, v in mass_frac_dict.items() if k != 'MMW'})
        return calls

    @staticmethod
    def make_samples(with_fe=True, n_draw=4):
        p = np.logspace(-6, 2, 10)
        samples = {'temperature': np.linspace(1500, 3000, n_draw)[:, None] * np.ones((1, len(p))),
                   'H2': np.full((n_draw, len(p)), 0.84), 'H2O': np.full((n_draw, len(p)), 1e-3)}
        if with_fe:
            samples['Fe'] = np.full((n_draw, len(p)), 1e-6)
        return p, samples

    def test_one_call_per_draw_and_output_shapes(self, fake_prt):
        p, samples = self.make_samples()
        out = ra.equilibrium_profiles(samples, p, c_to_o=0.7)
        assert len(fake_prt) == 4
        assert all(np.allclose(c['c_to_o'], 0.7) for c in fake_prt)
        assert set(out) == {'H2O', 'CO'} and out['H2O'].shape == (4, len(p))

    def test_metallicity_estimated_from_the_free_chemistry_fe_h_ratio(self, fake_prt):
        p, samples = self.make_samples(with_fe=True)
        ra.equilibrium_profiles(samples, p)
        # Fe = 1e-6, H = 2 * 0.84 + 2 * 1e-3 (H2 and H2O) -> log10(Fe/H) + 4.46
        expected = np.log10(1e-6 / (2 * 0.84 + 2 * 1e-3)) + 4.46
        np.testing.assert_allclose(fake_prt[0]['fe_to_h'], expected)

    def test_metallicity_default_is_solar_without_fe_and_can_be_forced(self, fake_prt):
        p, samples = self.make_samples(with_fe=False)
        ra.equilibrium_profiles(samples, p)
        np.testing.assert_allclose(fake_prt[0]['fe_to_h'], 0.)
        ra.equilibrium_profiles(samples, p, metallicity=0.5)
        np.testing.assert_allclose(fake_prt[-1]['fe_to_h'], 0.5)
        # An explicit value wins even if Fe is available
        p, samples = self.make_samples(with_fe=True)
        ra.equilibrium_profiles(samples, p, metallicity=-1.)
        np.testing.assert_allclose(fake_prt[-1]['fe_to_h'], -1.)


# =====================================================================================
# Abundance ratios
# =====================================================================================

class TestAbundanceRatioProfiles:

    @staticmethod
    def make_samples():
        # 2 draws, 3 pressure levels. H2O=1e-3, CO=2e-3, H2=0.8 everywhere (constant with pressure
        # for draw 0), draw 1 has different values to check draws don't get mixed up.
        return {
            'temperature': np.array([[2000., 2000., 2000.], [1500., 1500., 1500.]]),
            'H2O': np.array([[1e-3, 1e-3, 1e-3], [2e-3, 2e-3, 2e-3]]),
            'CO': np.array([[2e-3, 2e-3, 2e-3], [1e-3, 1e-3, 1e-3]]),
            'H2': np.array([[0.8, 0.8, 0.8], [0.8, 0.8, 0.8]]),
        }

    def test_c_to_o_ratio(self):
        samples = self.make_samples()
        ratio = ra.abundance_ratio_profiles(samples, 'C/O')
        # C = 2e-3 (from CO), O = 1e-3 (H2O) + 2e-3 (CO) = 3e-3 for draw 0
        np.testing.assert_allclose(ratio[0], 2e-3 / 3e-3)
        np.testing.assert_allclose(ratio[1], 1e-3 / 3e-3)

    def test_h_to_o_includes_h2_and_h2o(self):
        samples = self.make_samples()
        ratio = ra.abundance_ratio_profiles(samples, 'H/O')
        h_draw0 = 2 * 1e-3 + 2 * 0.8       # H2O has 2 H, H2 has 2 H
        o_draw0 = 1e-3 + 2e-3
        np.testing.assert_allclose(ratio[0], h_draw0 / o_draw0)

    def test_abund0_adds_a_baseline(self):
        samples = self.make_samples()
        ratio_default = ra.abundance_ratio_profiles(samples, 'Fe/H')
        np.testing.assert_allclose(ratio_default, 0.)      # no Fe, no abund0 -> 0/H = 0
        ratio_with_baseline = ra.abundance_ratio_profiles(samples, 'Fe/H', abund0={'Fe': 1e-5})
        h_draw0 = 2 * 1e-3 + 2 * 0.8
        np.testing.assert_allclose(ratio_with_baseline[0], 1e-5 / h_draw0)

    def test_shape_matches_input(self):
        samples = self.make_samples()
        assert ra.abundance_ratio_profiles(samples, 'C/O').shape == (2, 3)


class TestMetallicityProfiles:

    def test_matches_manual_dex_conversion(self):
        samples = TestAbundanceRatioProfiles.make_samples()
        reference = {'C': 8.0, 'H': 12.0}
        out = ra.metallicity_profiles(samples, 'C/H', reference_values=reference)
        ratio = ra.abundance_ratio_profiles(samples, 'C/H')
        np.testing.assert_allclose(out, np.log10(ratio) - (8.0 - 12.0))

    def test_default_reference_is_solar_asplund2009(self):
        samples = TestAbundanceRatioProfiles.make_samples()
        out = ra.metallicity_profiles(samples, 'C/H')
        ratio = ra.abundance_ratio_profiles(samples, 'C/H')
        expected_solar = ru.solar_asplund2009['C'] - ru.solar_asplund2009['H']
        np.testing.assert_allclose(out, np.log10(ratio) - expected_solar)

    def test_zero_when_ratio_equals_reference(self):
        # A ratio exactly matching the reference gives [X/H] = 0
        samples = {'temperature': np.zeros((1, 2)),
                  'C': np.array([[10 ** (8.0 - 12.0), 10 ** (8.0 - 12.0)]]),
                  'H': np.array([[1., 1.]])}
        out = ra.metallicity_profiles(samples, 'C/H', reference_values={'C': 8.0, 'H': 12.0})
        np.testing.assert_allclose(out, 0., atol=1e-12)


class TestProfileStatsAtPressures:

    def test_looks_up_by_searchsorted_insertion_index(self):
        # Same convention as `np.searchsorted` used elsewhere in the code: an insertion index,
        # not a true nearest-neighbor search. Values exactly on the grid remove the ambiguity.
        pressures = np.array([1e-4, 1e-2, 1e0, 1e2])
        rng = np.random.default_rng(0)
        profile = rng.normal(loc=[1., 2., 3., 4.], scale=0.1, size=(200, 4))
        stats = ra.profile_stats_at_pressures(profile, pressures, [1e-2, 1e0])
        assert set(stats) == {1e-2, 1e0}
        assert stats[1e-2]['median'] == pytest.approx(2., abs=0.05)
        assert stats[1e0]['median'] == pytest.approx(3., abs=0.05)

    def test_returned_values_are_scalars_not_arrays(self):
        pressures = np.array([1e-2, 1e0])
        profile = np.random.default_rng(1).normal(size=(50, 2))
        stats = ra.profile_stats_at_pressures(profile, pressures, [1e-2])
        s = stats[1e-2]
        assert np.isscalar(s['median'])
        assert len(s['1-sig']) == 2 and np.isscalar(s['1-sig'][0])

    def test_log_scale(self):
        pressures = np.array([1e-2, 1e0])
        profile = 10 ** np.random.default_rng(2).normal(loc=-3, size=(500, 2))
        stats_lin = ra.profile_stats_at_pressures(profile, pressures, [1e-2], log_scale=False)
        stats_log = ra.profile_stats_at_pressures(profile, pressures, [1e-2], log_scale=True)
        # log-space median of a log-normal-like sample differs from the linear-space median
        assert stats_lin[1e-2]['median'] != pytest.approx(stats_log[1e-2]['median'])

    def test_out_of_range_pressure_clips_to_the_edge(self):
        pressures = np.array([1e-2, 1e0])
        profile = np.random.default_rng(3).normal(loc=[1., 5.], size=(30, 2))
        stats = ra.profile_stats_at_pressures(profile, pressures, [1e3])
        assert stats[1e3]['median'] == pytest.approx(5., abs=0.5)   # clipped to the last column


class TestFitChemicalEquilibriumAtPressure:
    """`interpol_abundances` is faked (real petitRADTRANS is checked separately on Narval): the
    fake table is invertible, so the fit's ability to recover known (C/O, Fe/H) values is a real
    check of the optimisation logic, not just that it runs."""

    @pytest.fixture
    def fake_prt(self, monkeypatch):
        import sys
        import types

        def interpol_abundances(c_to_o, fe_to_h, temperature, pressure):
            scale = 10 ** fe_to_h
            h2o = 1e-3 * scale / (1 + c_to_o)
            co = 1e-3 * scale * c_to_o / (1 + c_to_o)
            return {'H2O': h2o, 'CO': co, 'MMW': 2.3}

        pkg = types.ModuleType('petitRADTRANS')
        mod = types.ModuleType('petitRADTRANS.poor_mans_nonequ_chem')
        mod.interpol_abundances = interpol_abundances
        monkeypatch.setitem(sys.modules, 'petitRADTRANS', pkg)
        monkeypatch.setitem(sys.modules, 'petitRADTRANS.poor_mans_nonequ_chem', mod)
        monkeypatch.setattr(ra, 'mass_frac_2_vmr',
                            lambda mass_frac_dict: {k: v for k, v in mass_frac_dict.items() if k != 'MMW'})

    @staticmethod
    def observed_samples(c_to_o_true, fe_to_h_true, n_draw=3):
        pressures = np.logspace(-4, 2, 5)
        scale = 10 ** fe_to_h_true
        h2o = 1e-3 * scale / (1 + c_to_o_true)
        co = 1e-3 * scale * c_to_o_true / (1 + c_to_o_true)
        samples = {'temperature': np.full((n_draw, len(pressures)), 2000.),
                  'H2O': np.full((n_draw, len(pressures)), h2o),
                  'CO': np.full((n_draw, len(pressures)), co)}
        return pressures, samples

    def test_free_fit_recovers_known_c_to_o_and_metallicity(self, fake_prt):
        pressures, samples = self.observed_samples(c_to_o_true=0.7, fe_to_h_true=0.4)
        out = ra.fit_chemical_equilibrium_at_pressure(samples, pressures, pressure=1e-2, metallicity=0.)
        assert len(out['C_to_O']) == 3
        np.testing.assert_allclose(out['C_to_O'], 0.7, atol=1e-4)
        np.testing.assert_allclose(out['metallicity'], 0.4, atol=1e-4)

    def test_fixed_metallicity_fit_recovers_c_to_o_when_metallicity_is_correct(self, fake_prt):
        pressures, samples = self.observed_samples(c_to_o_true=0.3, fe_to_h_true=0.6)
        out = ra.fit_chemical_equilibrium_at_pressure(samples, pressures, pressure=1e-2, metallicity=0.6)
        np.testing.assert_allclose(out['C_to_O_fixed_metallicity'], 0.3, atol=1e-4)

    def test_metallicity_from_free_chemistry_fe_h_ratio_when_not_given(self, fake_prt):
        pressures, samples = self.observed_samples(c_to_o_true=0.5, fe_to_h_true=0.)
        samples['Fe'] = np.full_like(samples['H2O'], 1e-6)
        out_auto = ra.fit_chemical_equilibrium_at_pressure(samples, pressures, pressure=1e-2)
        # Fe/H measured from the free-chemistry profile (H from H2O only here) should match a fit
        # using that same value explicitly as the metallicity
        species = ['H2O', 'CO', 'Fe']
        h = ru.get_atom_abundance('H', [samples[s] for s in species], species)
        fe = ru.get_atom_abundance('Fe', [samples[s] for s in species], species)
        metallicity_from_fe_h = (np.log10(fe / h) - (-4.46))[0, 0]
        out_manual = ra.fit_chemical_equilibrium_at_pressure(samples, pressures, pressure=1e-2,
                                                              metallicity=metallicity_from_fe_h)
        np.testing.assert_allclose(out_auto['C_to_O_fixed_metallicity'], out_manual['C_to_O_fixed_metallicity'])
