"""Tests of `starships.retrieval_figures`'s orchestration logic: which sections run or are
skipped for a given kind of run, that files land where expected, that one section failing does not
take the others down with it, and that the walker selection (discard/logl_min/best) is wired
correctly. The individual figure-making building blocks (`retrieval_analysis.*`) have their own
tests; this file fakes the `retrieval` module with just enough physics to exercise them cheaply.
"""
from types import SimpleNamespace

import emcee
import numpy as np
import pytest

from starships import retrieval_analysis as ra
from starships import retrieval_figures as rf

PRESSURES = np.logspace(-6, 2, 15)
SPECIES = ['H2O', 'CO']


def make_walker_file(path, n_walkers=8, n_params=3, n_steps=30, seed=0):
    """Write a small emcee walker file (Gaussian posterior)."""
    rng = np.random.default_rng(seed)
    backend = emcee.backends.HDFBackend(str(path))
    backend.reset(n_walkers, n_params)
    sampler = emcee.EnsembleSampler(n_walkers, n_params, lambda x: -0.5 * np.sum(x ** 2), backend=backend)
    sampler.run_mcmc(rng.normal(size=(n_walkers, n_params)), n_steps, progress=False)
    return path


def fake_unpack_theta(params):
    """One dict per region: `params` = [H2O, CO, T_eq] per region."""
    regions = []
    for i in range(0, len(params), 3):
        h2o, co, temp = params[i:i + 3]
        regions.append({'pressures': PRESSURES, 'temperatures': np.full(len(PRESSURES), temp),
                        'H2O': h2o, 'CO': co, 'H-': 1e-8, 'e-': 1e-6})
    return regions


def fake_static_model(theta_list, mode, Raf=None, atmo_obj=None):
    wave = np.linspace(1., 2., 15)
    spec = wave * 1e-3 * sum(theta[sp] for theta in theta_list for sp in SPECIES)
    return wave, spec, 1000


def fake_synt_data(wv, spec, res, infos, *extra):
    return None, np.full(len(infos['wave']), spec.sum())


@pytest.fixture(autouse=True)
def fake_petitradtrans_chemical_equilibrium(monkeypatch):
    """`fit_chemical_equilibrium_at_pressure` needs real petitRADTRANS (not installed in this
    venv). The orchestration tests here don't care about equilibrium-chemistry physics, only that
    the 'abundance_ratios' section runs and produces a file, so the fit itself is faked."""
    def fake_fit(profile_samples, pressures, pressure, mol_list=('H2O', 'CO'), metallicity=None):
        n = len(profile_samples['temperature'])
        return {'C_to_O': np.full(n, 0.5), 'metallicity': np.zeros(n),
               'C_to_O_fixed_metallicity': np.full(n, 0.5)}

    monkeypatch.setattr(ra, 'fit_chemical_equilibrium_at_pressure', fake_fit)


def make_fake_retrieval(n_regions=1, with_low_res=True, save_calls=None):
    """A `retrieval`-module stand-in with 3 real params per region (n_regions * 3 total)."""
    region_id = list(range(1, n_regions + 1))
    params_prior = {}
    for r in region_id:
        suffix = f'_{r}' if n_regions > 1 else ''
        params_prior.update({f'H2O{suffix}': ['log_uniform', -12, -0.5], f'CO{suffix}': ['log_uniform', -12, -0.5],
                             f'T_eq{suffix}': ['uniform', 500, 3000]})

    spectro = {'g395H': {'model_type': 'low', 'wave': np.linspace(1, 2, 5), 'data': np.full(5, 1e-3),
                        'err': np.full(5, 1e-5)}} if with_low_res else {}

    def get_ker(theta_regions, phase=None, planet=None, instrum=None, model_resolution=None):
        return [np.array([0.5, 0.5]) for _ in theta_regions]

    def save_planet_model(theta_regions, path):
        if save_calls is not None:
            save_calls.append(path)
        __import__('pathlib').Path(path).write_bytes(b'fake model')

    return SimpleNamespace(
        params_prior=params_prior, region_id=region_id, kind_trans='emission',
        retrieval_type='JR' if with_low_res else 'HRR',
        wv_range_high=[[0.9, 2.5]], wv_range_low=[[1.0, 5.0]],
        spectrophotometric_data=spectro, photometric_data={},
        line_opacities=list(SPECIES), continuum_opacities=['H-'], other_species=['e-'],
        dissociation=False, chemical_equilibrium=False,
        unpack_theta=fake_unpack_theta, prepare_static_model=fake_static_model,
        prepare_spectrophotometry=fake_synt_data, prepare_photometry=fake_synt_data,
        prt_res={'high': 250000, 'low': 1000}, res_instru=70000,
        instrum_param_list=[{'resol': 70000}], planet=object(), get_ker=get_ker,
        save_planet_model=save_planet_model,
    )


class TestSectionSelection:

    def test_single_region_run_gets_ratios_not_kernels(self, tmp_path):
        walker_file = make_walker_file(tmp_path / 'w.h5', n_params=3)
        retrieval_obj = make_fake_retrieval(n_regions=1)

        saved = rf.make_all_figures(retrieval_obj, walker_file, output_dir=tmp_path / 'out', n_draw=5)

        assert 'abundance_ratios' in saved
        assert 'rotation_kernels' not in saved
        assert all(p.exists() for paths in saved.values() for p in paths)

    def test_multi_region_run_gets_kernels_not_ratios(self, tmp_path):
        walker_file = make_walker_file(tmp_path / 'w.h5', n_params=6, n_walkers=16)
        retrieval_obj = make_fake_retrieval(n_regions=2)

        saved = rf.make_all_figures(retrieval_obj, walker_file, output_dir=tmp_path / 'out', n_draw=5)

        assert 'rotation_kernels' in saved
        assert 'abundance_ratios' not in saved
        # One contribution figure per region + one for the whole planet
        assert len(saved['contributions']) == 3

    def test_no_low_res_data_skips_that_section_but_keeps_others(self, tmp_path):
        walker_file = make_walker_file(tmp_path / 'w.h5', n_params=3)
        retrieval_obj = make_fake_retrieval(n_regions=1, with_low_res=False)

        saved = rf.make_all_figures(retrieval_obj, walker_file, output_dir=tmp_path / 'out', n_draw=5)

        assert 'low_res_data' not in saved
        assert 'corner' in saved and 'model_spectrum' in saved

    def test_model_saved_only_when_high_res_model_possible(self, tmp_path):
        walker_file = make_walker_file(tmp_path / 'w.h5', n_params=3)
        retrieval_obj = make_fake_retrieval(n_regions=1)
        save_calls = []
        retrieval_obj = make_fake_retrieval(n_regions=1, save_calls=save_calls)

        rf.make_all_figures(retrieval_obj, walker_file, output_dir=tmp_path / 'out', n_draw=5)
        assert len(save_calls) == 1

        retrieval_obj.wv_range_high = []   # no high-res model possible any more
        save_calls.clear()
        rf.make_all_figures(retrieval_obj, walker_file, output_dir=tmp_path / 'out2', n_draw=5)
        assert len(save_calls) == 0


class TestFailureIsolation:

    def test_one_failing_section_does_not_block_the_others(self, tmp_path, monkeypatch):
        walker_file = make_walker_file(tmp_path / 'w.h5', n_params=3)
        retrieval_obj = make_fake_retrieval(n_regions=1)

        def boom(*args, **kwargs):
            raise RuntimeError('synthetic failure')

        monkeypatch.setattr(ra, 'draw_region_profiles', boom)
        saved = rf.make_all_figures(retrieval_obj, walker_file, output_dir=tmp_path / 'out', n_draw=5)

        assert 'profiles' not in saved                 # the failing section...
        assert 'abundance_ratios' not in saved          # ...and anything depending on it
        assert 'corner' in saved and 'logl_scatter' in saved and 'model_spectrum' in saved


class TestWalkerSelection:

    def test_discard_and_logl_min_reduce_the_kept_sample(self, tmp_path):
        walker_file = make_walker_file(tmp_path / 'w.h5', n_params=3, n_steps=40, n_walkers=8)
        retrieval_obj = make_fake_retrieval(n_regions=1)

        pos_all, logl_all, _ = ra.load_posterior_sample(walker_file, discard=0)
        logl_min = np.median(logl_all)

        calls = []

        def spy_unpack(params):
            calls.append(np.array(params))
            return fake_unpack_theta(params)

        retrieval_obj.unpack_theta = spy_unpack
        rf.make_all_figures(retrieval_obj, walker_file, output_dir=tmp_path / 'out', discard=10,
                            logl_min=logl_min, n_draw=3)

        pos_sel, logl_sel, _ = ra.load_posterior_sample(walker_file, discard=10)
        pos_sel, logl_sel = ra.select_samples(pos_sel, logl_sel, logl_min=logl_min)
        # The first unpack_theta call is for `theta_regions` (the single best/median sample);
        # later ones (captured too, since it's the same spy) come from the profile/spectra draws.
        np.testing.assert_array_equal(calls[0], pos_sel[len(pos_sel) // 2])

    def test_best_uses_the_highest_logl_sample(self, tmp_path):
        walker_file = make_walker_file(tmp_path / 'w.h5', n_params=3)
        retrieval_obj = make_fake_retrieval(n_regions=1)

        calls = []

        def spy_unpack(params):
            calls.append(np.array(params))
            return fake_unpack_theta(params)

        retrieval_obj.unpack_theta = spy_unpack
        rf.make_all_figures(retrieval_obj, walker_file, output_dir=tmp_path / 'out', best='best', n_draw=3)

        pos, logl, _ = ra.load_posterior_sample(walker_file)
        np.testing.assert_array_equal(calls[0], pos[-1])

    def test_invalid_best_raises(self, tmp_path):
        walker_file = make_walker_file(tmp_path / 'w.h5', n_params=3)
        retrieval_obj = make_fake_retrieval(n_regions=1)
        with pytest.raises(ValueError, match='best'):
            rf.make_all_figures(retrieval_obj, walker_file, output_dir=tmp_path / 'out', best='worst')

    def test_walker_path_is_joined_with_walker_file(self, tmp_path):
        subdir = tmp_path / 'walkers'
        subdir.mkdir()
        make_walker_file(subdir / 'w.h5', n_params=3)
        retrieval_obj = make_fake_retrieval(n_regions=1)

        saved = rf.make_all_figures(retrieval_obj, 'w.h5', walker_path=subdir,
                                    output_dir=tmp_path / 'out', n_draw=3)
        assert 'corner' in saved


class TestFileOutput:

    def test_multiple_formats_are_all_written(self, tmp_path):
        walker_file = make_walker_file(tmp_path / 'w.h5', n_params=3)
        retrieval_obj = make_fake_retrieval(n_regions=1)

        saved = rf.make_all_figures(retrieval_obj, walker_file, output_dir=tmp_path / 'out',
                                    formats=['png', 'pdf'], n_draw=3)
        suffixes = {p.suffix for p in saved['corner']}
        assert suffixes == {'.png', '.pdf'}
