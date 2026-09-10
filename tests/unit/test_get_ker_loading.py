"""Chantier A Phase 3: real `get_ker` wiring (retrieval.py + retrieval_utils.py).

Pure Python / numpy tests -- no petitRADTRANS needed, `get_ker` never touches the
model itself, only the rotation kernel plumbing around it.
"""
import numpy as np
import pytest

import starships.retrieval as retrieval
import starships.retrieval_utils as ru


GET_KER_FILE_CONTENT = '''
def get_ker(theta_regions, phase=None, planet=None, instrum=None, model_resolution=None):
    return {
        "phase": phase,
        "planet": planet,
        "instrum": instrum,
        "model_resolution": model_resolution,
        "n_regions": len(theta_regions),
    }
'''


class TestLoadCustomGetKer:
    """`ru.load_custom_get_ker` mirrors `ru.load_custom_prior` -- loads a user's
    Python file as its own module and returns its module-level `get_ker`."""

    def test_loads_module_level_get_ker(self, tmp_path):
        get_ker_file = tmp_path / "custom_get_ker.py"
        get_ker_file.write_text(GET_KER_FILE_CONTENT)

        get_ker = ru.load_custom_get_ker(str(get_ker_file))

        assert callable(get_ker)
        result = get_ker([{}, {}], phase=0.5, planet="dummy_planet",
                          instrum={"resol": 70_000}, model_resolution=250_000)
        assert result == {
            "phase": 0.5,
            "planet": "dummy_planet",
            "instrum": {"resol": 70_000},
            "model_resolution": 250_000,
            "n_regions": 2,
        }

    def test_loaded_function_does_not_need_retrieval_globals(self, tmp_path):
        """Regression test for the bug Antoine flagged in the documented example:
        the get_ker file used to reference `data_visits`/`planet`/`instrum`/`lbl_res`
        as bare names, which raises NameError since the file is loaded as its own
        module and never sees retrieval.py's globals. The new contract passes
        everything needed as explicit arguments instead, so a minimal get_ker file
        that only uses its arguments must work with no `starships` imports at all.
        """
        get_ker_file = tmp_path / "custom_get_ker.py"
        get_ker_file.write_text(GET_KER_FILE_CONTENT)
        get_ker = ru.load_custom_get_ker(str(get_ker_file))
        # Would raise NameError if the function tried to reach for retrieval.py's
        # globals directly instead of using its arguments.
        get_ker([{}], phase=0.1, planet=None, instrum=None, model_resolution=None)


class _DummyQuantity:
    """Minimal stand-in for an astropy Quantity: only `.value` and `.to(unit)`."""

    def __init__(self, value):
        self.value = value

    def to(self, unit):
        return self


class _DummyPlanet:
    def __init__(self, mid_tr_value, period_days):
        self.mid_tr = _DummyQuantity(mid_tr_value)
        self.period = _DummyQuantity(period_days)


class TestPrepareModelMultiRegGetKerWiring:
    """`prepare_model_multi_reg` computes the mean orbital phase itself (from
    `data_visits`/`planet`) and passes it -- along with `planet`, the visit's
    instrument, and the model sampling resolution -- explicitly to `get_ker`,
    instead of relying on `get_ker` reaching into retrieval.py's globals."""

    def test_computes_phase_and_forwards_context_to_get_ker(self, monkeypatch):
        calls = []

        def fake_get_ker(theta_regions, phase=None, planet=None, instrum=None,
                         model_resolution=None):
            calls.append(dict(phase=phase, planet=planet, instrum=instrum,
                              model_resolution=model_resolution))
            return [None for _ in theta_regions]

        def fake_prepare_model_high_or_low(theta_dict, mode, rot_ker=None, atmo_obj=None, Raf=None):
            return np.array([1.0, 2.0]), np.array([0.1, 0.2])

        # Mid-transit at phase 0, period of 3 days -- t_start in days gives a phase
        # directly (all_phases = t_start / period % 1).
        data_visits_fake = {
            0: {
                't_start': np.array([0.3, 0.6, 2.7]),  # -> phases 0.1, 0.2, 0.9
                'i_pl_signal': np.array([True, True, False]),  # only the first two count
            },
        }

        monkeypatch.setattr(retrieval, 'get_ker', fake_get_ker, raising=False)
        monkeypatch.setattr(retrieval, 'prepare_model_high_or_low', fake_prepare_model_high_or_low,
                            raising=False)
        monkeypatch.setattr(retrieval, 'data_visits', data_visits_fake, raising=False)
        monkeypatch.setattr(retrieval, 'planet', _DummyPlanet(mid_tr_value=0.0, period_days=3.0),
                            raising=False)
        monkeypatch.setattr(retrieval, 'instrum_param_list', [{'resol': 70_000}], raising=False)
        monkeypatch.setattr(retrieval, 'prt_res', {'high': 250_000}, raising=False)
        monkeypatch.setattr(retrieval, 'region_id', [1], raising=False)

        theta_regions = [{'spec_scale': 1.0}]
        retrieval.prepare_model_multi_reg(theta_regions, 'high', visit_i=0)

        assert len(calls) == 1
        call = calls[0]
        # `visit_i` (used above to select data_visits[0]/instrum_param_list[0]) is not
        # itself forwarded to get_ker -- fake_get_ker's signature above has no `visit_i`
        # parameter, so this would already TypeError if prepare_model_multi_reg tried to
        # pass it.
        assert call['instrum'] == {'resol': 70_000}
        assert call['model_resolution'] == 250_000
        assert call['planet'] is not None
        # Only the two exposures where i_pl_signal is True enter the mean.
        assert call['phase'] == pytest.approx(np.mean([0.1, 0.2]))
