import numpy as np
import pytest

from starships.transpec import ReductionParams


class TestReductionParams:
    """`ReductionParams` (Chantier B, B1) replaces the old convention of stacking
    reduction parameters into a raw 10-element positional list (e.g.
    `[0.2, 0.97, 51, 41, 5, 2, 5.0, 5.0, 5.0, 5.0]`). Named fields make each value
    documented and easy to change; list-like access (`__getitem__`/`__setitem__`/
    `len`/`.copy()`) keeps it a drop-in replacement for code that still reads/writes
    reduction parameters positionally (e.g. `visit.params[5]`, `params_all.copy()`).
    """

    def test_defaults_match_legacy_positional_list(self):
        # The exact 10 values that used to be hardcoded at pipeline/reduction.py:172
        # (mask_tellu and mask_wings excluded here since they never had shared
        # defaults -- they're always passed explicitly per run).
        params = ReductionParams(mask_tellu=0.2, mask_wings=0.97)
        legacy = [0.2, 0.97, 51, 41, 5, 2, 5.0, 5.0, 5.0, 5.0]
        assert list(params) == legacy

    def test_named_field_access(self):
        params = ReductionParams(mask_tellu=0.3, mask_wings=0.9, n_pc=5)
        assert params.mask_tellu == 0.3
        assert params.mask_wings == 0.9
        assert params.n_pc == 5

    def test_positional_getitem_matches_named_fields(self):
        params = ReductionParams(mask_tellu=0.3, mask_wings=0.9, n_pc=5)
        assert params[0] == params.mask_tellu
        assert params[1] == params.mask_wings
        assert params[5] == params.n_pc

    def test_setitem_mutates_named_field(self):
        params = ReductionParams()
        params[5] = 7
        assert params.n_pc == 7

    def test_len_is_ten(self):
        assert len(ReductionParams()) == 10

    def test_copy_is_independent(self):
        params = ReductionParams(n_pc=3)
        params_copy = params.copy()
        params_copy[5] = 9
        assert params.n_pc == 3
        assert params_copy.n_pc == 9

    def test_converts_to_numpy_array(self):
        params = ReductionParams(mask_tellu=0.2, mask_wings=0.97)
        arr = np.asarray(params)
        np.testing.assert_allclose(arr, [0.2, 0.97, 51, 41, 5, 2, 5.0, 5.0, 5.0, 5.0])

    def test_survives_npz_roundtrip(self, tmp_path):
        params = ReductionParams(mask_tellu=0.2, mask_wings=0.97, n_pc=4)
        path = tmp_path / 'params.npz'
        np.savez(path, params=params)
        loaded = np.load(path)['params']
        np.testing.assert_allclose(loaded, [0.2, 0.97, 51, 41, 5, 4, 5.0, 5.0, 5.0, 5.0])
