"""Chantier C: `logl_grid.py`'s model preparation must reproduce `retrieval.py::lnprob`.

`_prepare_visit_model` (once per visit, before the worker pool) + `_get_chi2_detailed`
(per grid point) together must give exactly the cross/squared terms `lnprob`'s HIGH RES
per-visit block computes for the same theta, visit and (Kp, vsys) -- single-region
(degrade to this visit's instrument resolution) and multi-region (per-exposure
`get_ker` kernel). The "expected" side is built here by hand from the same
`model_sequence` building blocks, in the exact order `lnprob` calls them.

Pure numpy / synthetic spectra -- no petitRADTRANS, no real data: `logl_grid`'s module
globals are injected with `monkeypatch`. Real-data validation (a grid point evaluated
at the theta of a saved model must match `lnprob(theta)`'s HIGH RES logL) is done on
Narval, see `Notes/plan_revision_starships.md`.
"""
import astropy.units as u
import numpy as np
import pytest

import starships.logl_grid as lg
from starships import model_sequence as model_seq
from starships.orbite import rv_theo_t

from tests.unit.test_planet_model import (
    INSTRU_RES, INSTRUM, NATIVE_RES, noop_get_ker, synthetic_get_ker, synthetic_native_model,
)


class _AstroPlanet:
    """Just enough of `Planet` for `rv_theo_t`/`_phase_per_exposure`/`_save_visit_result`:
    real astropy Quantities, `period` as a 1-element array like real instances."""

    def __init__(self):
        self.mid_tr = 0.0 * u.d
        self.period = np.array([1.0]) * u.d


N_EXP, N_ORD, N_PIX = 4, 2, 300


def _data_visit(n_pc=0, pca=None, seed=0):
    rng = np.random.default_rng(seed)
    orders = np.array([np.linspace(1.504, 1.508, N_PIX), np.linspace(1.511, 1.515, N_PIX)])
    shape = (N_EXP, N_ORD, N_PIX)
    params = [0.0] * 10
    params[5] = n_pc
    return dict(
        wave=np.broadcast_to(orders, shape).copy(),
        t_start=np.linspace(0.45, 0.55, N_EXP),   # phases around secondary eclipse
        RV_const=12.3,
        params=params,
        pca=pca,
        noise=rng.uniform(0.9, 1.1, shape),
        flux=np.ma.masked_array(rng.normal(size=shape), mask=np.zeros(shape, dtype=bool)),
    )


def _data_info():
    return dict(all_alpha_frac=np.linspace(0.6, 1.0, N_EXP), all_icorr=np.arange(N_EXP),
                all_N=np.full((N_EXP, N_ORD), N_PIX))


@pytest.fixture
def grid_env(monkeypatch):
    """Inject the module globals `setup_logl_grid`/`load_logl_grid_data` would set."""
    planet = _AstroPlanet()
    for name, value in dict(
        planet=planet, instrum_param_list=[INSTRUM], kind_trans='emission',
        use_real_stellar_rv=False, idx_orders=np.arange(N_ORD), axis_sum=-1, apply_alpha=True,
        get_ker=noop_get_ker, data_visits=[None],
    ).items():
        monkeypatch.setattr(lg, name, value, raising=False)
    monkeypatch.setattr(lg, '_current_apply_alpha', True)
    return planet


def _run_worker(monkeypatch, model, data_visit, v_sys, kp, get_ker=noop_get_ker):
    """What `_compute_visit_grid` does for one visit, then one worker call."""
    monkeypatch.setattr(lg, 'planet_model', model, raising=False)
    monkeypatch.setattr(lg, 'get_ker', get_ker, raising=False)
    interp = lg._prepare_visit_model(0, data_visit)
    monkeypatch.setattr(lg, '_current_interp', interp)
    monkeypatch.setattr(lg, '_current_data_visit', data_visit)
    monkeypatch.setattr(lg, '_current_alpha_frac', _data_info()['all_alpha_frac'])
    return lg._get_chi2_detailed((v_sys, kp))


def _chi2_terms_from_sequence(model_seq_arr, data_visit, n_pc=0):
    """lnprob's post-processing: PCA truncation, noise division, per-order sums."""
    model_norm = model_seq.apply_pca_to_model(
        model_seq_arr, data_visit['pca'], n_pca=n_pc) / data_visit['noise']
    ct = np.ma.sum(model_norm * data_visit['flux'], axis=-1)
    st = np.ma.sum(model_norm ** 2, axis=-1)
    return ct, st


class TestHarmonizationWithLnprob:

    V_SYS, KP = 3.0, 50.0

    def _vrp_orb(self, planet, data_visit):
        return rv_theo_t(self.KP, data_visit['t_start'] * u.d, planet.mid_tr, planet.period,
                         plnt=True).value

    def test_single_region_matches_lnprob_path(self, grid_env, monkeypatch):
        planet = grid_env
        model = synthetic_native_model()
        data_visit = _data_visit()

        ct, st = _run_worker(monkeypatch, model, data_visit, self.V_SYS, self.KP)

        # --- expected: lnprob's single-region branch, step by step ---
        wave, Fp, Fstar = model_seq.degrade_fp_fstar(
            model.wave, model.Fp_by_region[0], model.Fstar, resolution=INSTRU_RES,
            native_resolution=NATIVE_RES, rotation_kernel=None)
        seq = model_seq.build_model_sequence(
            wave[20:-20], Fp[20:-20], data_visit['wave'], self._vrp_orb(planet, data_visit),
            Fstar=Fstar[20:-20], vr_orb=0.0, alpha=_data_info()['all_alpha_frac'],
            kind_trans='emission', RV=self.V_SYS + data_visit['RV_const'])
        ct_ref, st_ref = _chi2_terms_from_sequence(seq, data_visit)

        assert ct.shape == (N_EXP, N_ORD)
        np.testing.assert_array_equal(np.ma.filled(ct, np.nan), np.ma.filled(ct_ref, np.nan))
        np.testing.assert_array_equal(np.ma.filled(st, np.nan), np.ma.filled(st_ref, np.nan))
        assert np.isfinite(ct).all() and (st > 0).all()

    def test_multi_region_matches_lnprob_path(self, grid_env, monkeypatch):
        planet = grid_env
        model = synthetic_native_model(n_regions=2)
        data_visit = _data_visit()
        get_ker = synthetic_get_ker()

        ct, st = _run_worker(monkeypatch, model, data_visit, self.V_SYS, self.KP, get_ker=get_ker)

        # --- expected: lnprob's multi-region branch, step by step ---
        region_kernel, trim = model_seq.build_region_kernel(
            model.theta_regions, get_ker, planet, INSTRUM, NATIVE_RES)
        wave, Fp_by_region, Fstar = model_seq.prepare_multi_region_visit(
            model.wave, model.Fp_by_region, model.Fstar, trim, INSTRU_RES, NATIVE_RES)
        phase = (data_visit['t_start'] - planet.mid_tr.value) / planet.period.to('d').value % 1
        seq = model_seq.build_model_sequence(
            wave, Fp_by_region, data_visit['wave'], self._vrp_orb(planet, data_visit),
            Fstar=Fstar, vr_orb=0.0, alpha=_data_info()['all_alpha_frac'],
            kind_trans='emission', RV=self.V_SYS + data_visit['RV_const'],
            region_kernel=region_kernel, phase=phase)
        ct_ref, st_ref = _chi2_terms_from_sequence(seq, data_visit)

        np.testing.assert_array_equal(np.ma.filled(ct, np.nan), np.ma.filled(ct_ref, np.nan))
        np.testing.assert_array_equal(np.ma.filled(st, np.nan), np.ma.filled(st_ref, np.nan))

    def test_multi_region_is_genuinely_phase_dependent(self, grid_env, monkeypatch):
        """Sanity check that the multi-region test above is not vacuous: freezing the
        kernel at one phase for every exposure gives a different answer."""
        model = synthetic_native_model(n_regions=2)
        data_visit = _data_visit()
        get_ker = synthetic_get_ker()
        ct, _ = _run_worker(monkeypatch, model, data_visit, self.V_SYS, self.KP, get_ker=get_ker)

        frozen = model_seq.combine_at_phase(model, get_ker, grid_env, INSTRUM, phase=0.5)
        ct_frozen, _ = _run_worker(monkeypatch, frozen, data_visit, self.V_SYS, self.KP)
        assert not np.allclose(ct, ct_frozen)

    def test_pca_truncation_uses_the_data_n_pc(self, grid_env, monkeypatch):
        """Regression test: the Phase 2 Fp/Fstar path of logl_grid skipped the PCA
        truncation of the model (the old combined-ratio path did it internally)."""
        calls = []

        def spy_apply_pca(flux, pca, n_pca=2, **kwargs):
            calls.append((pca, n_pca))
            return flux

        monkeypatch.setattr(model_seq, 'apply_pca_to_model', spy_apply_pca)
        data_visit = _data_visit(n_pc=3, pca='the-fitted-pca')
        _run_worker(monkeypatch, synthetic_native_model(), data_visit, self.V_SYS, self.KP)
        assert calls == [('the-fitted-pca', 3)]


class TestPrepareVisitModelForms:

    def test_already_degraded_model_is_used_as_is_with_a_warning_on_mismatch(
            self, grid_env, monkeypatch, caplog):
        model = model_seq.degrade(synthetic_native_model(), 55_000.0)
        monkeypatch.setattr(lg, 'planet_model', model, raising=False)
        with caplog.at_level('WARNING', logger='starships.logl_grid'):
            interp = lg._prepare_visit_model(0, _data_visit())
        assert 'already degraded' in caplog.text and '55000' in caplog.text
        assert not interp.per_exposure

    def test_already_degraded_model_at_the_right_resolution_matches_native_path(
            self, grid_env, monkeypatch):
        data_visit = _data_visit()
        native = synthetic_native_model()
        ct_native, _ = _run_worker(monkeypatch, native, data_visit, 3.0, 50.0)
        ct_degraded, _ = _run_worker(monkeypatch, model_seq.degrade(native, INSTRU_RES),
                                     data_visit, 3.0, 50.0)
        np.testing.assert_array_equal(np.ma.filled(ct_native, np.nan),
                                      np.ma.filled(ct_degraded, np.nan))

    def test_multi_region_model_must_be_native(self, grid_env, monkeypatch):
        model = model_seq.degrade(synthetic_native_model(n_regions=2), INSTRU_RES)
        monkeypatch.setattr(lg, 'planet_model', model, raising=False)
        monkeypatch.setattr(lg, 'get_ker', synthetic_get_ker(), raising=False)
        with pytest.raises(ValueError, match='native'):
            lg._prepare_visit_model(0, _data_visit())

    def test_multi_region_without_get_ker_file_fails_loudly(self, grid_env, monkeypatch):
        monkeypatch.setattr(lg, 'planet_model', synthetic_native_model(n_regions=2), raising=False)
        with pytest.raises(ValueError, match='get_ker_file'):
            lg._prepare_visit_model(0, _data_visit())

    def test_uses_this_visits_own_instrument_resolution(self, grid_env, monkeypatch):
        """Chantier A Phase 4 mirrored: two visits of different instruments get the
        model degraded to their own resolution, not a shared one."""
        monkeypatch.setattr(lg, 'instrum_param_list',
                            [dict(name='A', resol=70_000.0), dict(name='B', resol=30_000.0)],
                            raising=False)
        monkeypatch.setattr(lg, 'planet_model', synthetic_native_model(), raising=False)
        data_visit = _data_visit()
        interp_a = lg._prepare_visit_model(0, data_visit)
        interp_b = lg._prepare_visit_model(1, data_visit)
        probe = data_visit['wave'][0, 0]
        assert not np.allclose(interp_a.fct_p(probe), interp_b.fct_p(probe))


class TestLoadModel:

    def test_kind_trans_mismatch_raises(self, grid_env, monkeypatch, tmp_path):
        path = tmp_path / 'model.npz'
        model_seq.save_planet_model(path, synthetic_native_model(kind_trans='transmission'))
        with pytest.raises(ValueError, match='kind_trans'):
            lg.load_model(path)

    def test_loads_into_the_global(self, grid_env, tmp_path):
        path = tmp_path / 'model.npz'
        model_seq.save_planet_model(path, synthetic_native_model(n_regions=2))
        lg.load_model(path)
        assert lg.planet_model.is_multi_region


class TestSaveVisitResult:

    def test_instrument_name_is_saved_per_visit(self, grid_env, tmp_path):
        data_visit = _data_visit()
        visit_result = dict(cross_terms=np.zeros((3, 2, N_EXP, N_ORD)),
                            squared_terms=np.zeros((3, 2, N_EXP, N_ORD)),
                            kp=np.zeros((3, 2)), vsys=np.zeros((3, 2)))
        lg._save_visit_result(visit_result, data_visit, _data_info(), tmp_path, 'stem',
                              '_grid', np.array([0.4, 0.45, 0.55, 0.6]), visit_idx=1)
        with np.load(tmp_path / 'stem_grid_visit1.npz') as f:
            assert f['instrument'].tolist() == ['SPIRou-APERO']
            assert f['kind_trans'].tolist() == ['emission']


class TestLegacyModelInLoglGrid:

    def test_legacy_combined_ratio_is_used_as_is(self, grid_env, monkeypatch, tmp_path, caplog):
        """An old-engine model file runs through logl_grid unchanged (no degradation,
        rigid shift of the ratio) -- matches build_model_sequence's combined_ratio path."""
        planet = grid_env
        native = synthetic_native_model()
        ratio = native.Fp_by_region[0] / native.Fstar
        path = tmp_path / 'old_model.npz'
        np.savez(path, wave=native.wave, spec=ratio)
        with caplog.at_level('WARNING', logger='starships.logl_grid'):
            lg.load_model(path)
        assert lg.planet_model.combined_ratio

        data_visit = _data_visit()
        v_sys, kp = 3.0, 50.0
        interp = lg._prepare_visit_model(0, data_visit)
        assert 'legacy' in caplog.text
        monkeypatch.setattr(lg, '_current_interp', interp)
        monkeypatch.setattr(lg, '_current_data_visit', data_visit)
        monkeypatch.setattr(lg, '_current_alpha_frac', _data_info()['all_alpha_frac'])
        ct, st = lg._get_chi2_detailed((v_sys, kp))

        vrp_orb = rv_theo_t(kp, data_visit['t_start'] * u.d, planet.mid_tr, planet.period,
                            plnt=True).value
        seq = model_seq.build_model_sequence(
            native.wave[20:-20], ratio[20:-20], data_visit['wave'], vrp_orb, vr_orb=0.0,
            alpha=_data_info()['all_alpha_frac'], kind_trans='emission',
            RV=v_sys + data_visit['RV_const'], combined_ratio=True)
        ct_ref, st_ref = _chi2_terms_from_sequence(seq, data_visit)
        np.testing.assert_array_equal(np.ma.filled(ct, np.nan), np.ma.filled(ct_ref, np.nan))
        np.testing.assert_array_equal(np.ma.filled(st, np.nan), np.ma.filled(st_ref, np.nan))


class TestComputeGridInProcess:

    def test_n_process_one_runs_without_a_pool(self, grid_env, monkeypatch):
        """n_process=1 maps in-process (usable under a 'spawn' start method, e.g. macOS,
        and by the regression test) and returns the (n_vsys, n_kp, n_exp, n_ord) terms."""
        monkeypatch.setattr(lg, 'planet_model', synthetic_native_model(), raising=False)
        monkeypatch.setattr(lg, 'rv_grid', dict(min=-2.0, max=2.0, step=2.0), raising=False)
        monkeypatch.setattr(lg, 'kp_grid', dict(min=50.0, max=50.0, step=5.0), raising=False)
        monkeypatch.setattr(lg, 'n_processes_per_cpu', 1, raising=False)
        data_visit = _data_visit()
        monkeypatch.setattr(lg, 'data_visits', [data_visit], raising=False)
        monkeypatch.setattr(lg, 'data_info_list', [_data_info()], raising=False)

        results = lg.compute_logl_grid(n_process=1)

        assert len(results) == 1
        assert results[0]['cross_terms'].shape == (3, 1, N_EXP, N_ORD)
        assert results[0]['vsys'].shape == (3, 1)
        assert lg._current_interp is None  # freed after the visit
