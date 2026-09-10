"""Chantier C1: `planet_obs.load_sequences()` bridge to the B3/B5 reduced-data format.

Pure Python / numpy / monkeypatch tests -- no petitRADTRANS, no real data file needed.
`load_sequences` is now a thin wrapper around `load_reduced_sequence` (single visit,
no more `do_tr` multi-transit fanout, see Notes/plan_revision_starships.md's Chantier C1
section for the full "poids mort" story) -- these tests isolate that wrapper and the
`_visit_to_data_dict` conversion from `load_reduced_sequence`/`Observations` themselves,
which are already covered elsewhere.
"""
from types import SimpleNamespace

import numpy as np
import astropy.units as u
import pytest

import starships.planet_obs as pl_obs


def _make_fake_visit():
    """Minimal stand-in for the `Observations` object `load_reduced_sequence` returns --
    just enough attributes for `_visit_to_data_dict`/`load_sequences` to read."""
    n_exp, n_ord, n_pix = 3, 2, 5
    zeros_3d = np.ma.zeros((n_exp, n_ord, n_pix))
    zeros_2d = np.ma.zeros((n_exp, n_ord))
    return SimpleNamespace(
        pca="dummy_pca",
        RV_const=12.3,
        RV_sys=0.0,
        mid_berv=0.1,
        mid_vr=0.2,
        params=[0, 0, 0, 0, 0, 5],
        wave=np.ones((n_exp, n_ord, n_pix)),
        vrp=np.array([1.0, 2.0, 3.0]),
        vr=np.array([0.01, 0.02, 0.03]),
        sep=np.array([1.0, 1.0, 1.0]),
        noise=np.ma.ones((n_exp, n_ord, n_pix)),
        N=zeros_2d,
        t_start=np.array([0.1, 0.2, 0.3]),
        final=np.ma.ones((n_exp, n_ord, n_pix)),
        ratio=zeros_3d,
        reconstructed=zeros_3d,
        reference_spec=zeros_3d,
        alpha_frac=np.array([0.5, 0.6, 0.7]),
        spec_trans=zeros_3d,
        icorr=np.array([0, 1]),
        clip_ts=5.0,
        scaling=1.0,
        fl_norm=zeros_3d,
        fl_norm_mo=zeros_3d,
        full_ts=zeros_3d,
        ts_norm=zeros_3d,
        rebuilt=zeros_3d,
        fl_Sref=zeros_3d,
        fl_masked=zeros_3d,
        recon_time=zeros_3d,
        bad=np.array([], dtype=int),
    )


class TestVisitToDataDict:
    """`_visit_to_data_dict` -- extracted out of `load_sequences` in Chantier C1."""

    def test_reattaches_units_to_vrp_and_vr(self):
        visit = _make_fake_visit()
        data_visit = pl_obs._visit_to_data_dict(visit)

        assert data_visit['vrp'].unit == u.km / u.s
        assert data_visit['vr'].unit == u.km / u.s
        np.testing.assert_array_equal(data_visit['vrp'].value, visit.vrp)

    def test_flux_and_s2f_are_derived_from_final_over_noise(self):
        visit = _make_fake_visit()
        data_visit = pl_obs._visit_to_data_dict(visit)

        expected_flux = visit.final / visit.noise
        np.testing.assert_array_equal(data_visit['flux'], expected_flux)
        np.testing.assert_array_equal(
            data_visit['s2f'], np.ma.sum(expected_flux ** 2, axis=-1))

    def test_no_leftover_transit_index_wrapping(self):
        """Chantier C1: the old dict-of-dicts keyed by transit index ('0', '1', ...) is
        gone -- `_visit_to_data_dict` returns one flat dict for the one visit given."""
        data_visit = pl_obs._visit_to_data_dict(_make_fake_visit())
        assert '0' not in data_visit
        assert isinstance(data_visit['pca'], str)  # the dummy sentinel, not re-wrapped


class TestLoadSequences:
    """`load_sequences` -- thin single-visit wrapper around `load_reduced_sequence`."""

    def test_calls_load_reduced_sequence_without_do_tr(self, monkeypatch):
        """Regression: before Chantier C1, this built a filename with a numeric transit
        suffix (`{filename}_data_trs_{i}.npz`) that no longer matches what the B3/B5
        pipeline saves (`{filename}_data_trs_.npz`, no suffix) -- see the "Bloquant
        découvert" note in Notes/plan_revision_starships.md's Chantier C section. The
        fix is to stop building that filename at all and let `load_reduced_sequence`'s
        own (already-correct) default handle it.
        """
        calls = []

        def fake_load_reduced_sequence(filename, n_pc, path='', **kwargs):
            calls.append(dict(filename=filename, n_pc=n_pc, path=path, kwargs=kwargs))
            return _make_fake_visit()

        monkeypatch.setattr(pl_obs, 'load_reduced_sequence', fake_load_reduced_sequence)

        data_info, data_visit = pl_obs.load_sequences(
            'retrieval_input_visit1', 5, path='/some/dir', planet='dummy_planet')

        assert len(calls) == 1
        assert calls[0]['filename'] == 'retrieval_input_visit1'
        assert calls[0]['n_pc'] == 5
        assert calls[0]['path'] == '/some/dir'
        assert calls[0]['kwargs'] == {'planet': 'dummy_planet'}
        assert 'wave' in data_visit

    def test_falls_back_to_old_numeric_suffix_when_new_format_file_is_missing(self, monkeypatch):
        """Regression: real pre-B3 production data (e.g. Antoine's WASP-33b_v07232, May
        2025) is saved as `{filename}_data_trs_0.npz`, not the B3/B5 `{filename}_data_trs_.npz`
        -- caught by `test_regression_lnprob.py::wasp33b_emission` on Narval raising a real
        `FileNotFoundError` against that exact dataset. `load_sequences` must retry with the
        old `_0` suffix rather than giving up after the first (B3/B5-format) attempt.
        """
        calls = []

        def fake_load_reduced_sequence(filename, n_pc, path='', **kwargs):
            calls.append(dict(kwargs=kwargs))
            if 'filename_end' not in kwargs:
                raise FileNotFoundError('no B3/B5-format file')
            assert kwargs['filename_end'] == '0'
            return _make_fake_visit()

        monkeypatch.setattr(pl_obs, 'load_reduced_sequence', fake_load_reduced_sequence)

        data_info, data_visit = pl_obs.load_sequences('retrieval_input_2-pc_mask_wings90_day1', 2)

        assert len(calls) == 2
        assert 'wave' in data_visit

    def test_data_info_uses_all_prefixed_keys_not_trall(self, monkeypatch):
        """Chantier C1: `trall_*` (transit-centric) -> `all_*` (visit-centric, works for
        emission too, not just transits) -- see Notes/plan_revision_starships.md."""
        monkeypatch.setattr(pl_obs, 'load_reduced_sequence',
                             lambda filename, n_pc, path='', **kwargs: _make_fake_visit())

        data_info, data_visit = pl_obs.load_sequences('stem', 5)

        assert set(data_info.keys()) == {'all_alpha_frac', 'all_icorr', 'all_N', 'bad_indexs'}
        np.testing.assert_array_equal(data_info['all_alpha_frac'], np.array([0.5, 0.6, 0.7]))
