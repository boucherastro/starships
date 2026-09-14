"""Chantier A Phase 4: c-k vs lbl opacity mode, per spectrophotometric instrument.

Pure Python / numpy tests -- no petitRADTRANS needed. `get_lbl_spectrophotometric_ranges`
is a pure helper factored out of `setup_retrieval` specifically so this decision (which
instruments opt into 'lbl', and therefore get folded into `wv_range_high` instead of
`wv_range_low`) is testable without running the whole, heavily side-effectful
`setup_retrieval`. `assign_model_type` is exercised end-to-end with a `wv_range_high`
built the same way `setup_retrieval` now builds it (real-instrument ranges + folded-in
lbl ranges, merged with `get_wv_range`), to confirm the lbl instrument is always
assigned 'high' and reuses the existing 'model_type == high' path -- no new pseudo-mode,
no new atmo objects. `get_res_instru` covers the other half of the same "no real
high-res instrument" scenario (`instrum_param_list` empty -- a retrieval running
entirely on lbl-flagged low-res data). Real petitRADTRANS physics for this path still
needs validating on Narval (see `Notes/plan_revision_starships.md`).
"""
import starships.retrieval as retrieval


class TestGetLblSpectrophotometricRanges:

    def test_no_instrument_has_opacity_mode_key(self):
        """Default behaviour (every existing config today): no `opacity_mode` key
        anywhere -- nothing is treated as lbl."""
        spectrophotometric_data = {
            'wfc3': {'wv_range': [1.1, 1.7]},
            'spitzer': {'wv_range': [3.0, 5.0]},
        }
        assert retrieval.get_lbl_spectrophotometric_ranges(spectrophotometric_data) == []

    def test_explicit_c_k_is_excluded(self):
        spectrophotometric_data = {'wfc3': {'wv_range': [1.1, 1.7], 'opacity_mode': 'c-k'}}
        assert retrieval.get_lbl_spectrophotometric_ranges(spectrophotometric_data) == []

    def test_lbl_instrument_is_included(self):
        spectrophotometric_data = {
            'wfc3': {'wv_range': [1.1, 1.7], 'opacity_mode': 'c-k'},
            'jwst_niriss': {'wv_range': [0.6, 2.8], 'opacity_mode': 'lbl'},
        }
        assert retrieval.get_lbl_spectrophotometric_ranges(spectrophotometric_data) == [[0.6, 2.8]]

    def test_mixed_instruments_keep_only_lbl_ones(self):
        spectrophotometric_data = {
            'a': {'wv_range': [1.0, 2.0], 'opacity_mode': 'lbl'},
            'b': {'wv_range': [2.0, 3.0]},  # defaults to c-k
            'c': {'wv_range': [3.0, 4.0], 'opacity_mode': 'lbl'},
        }
        assert retrieval.get_lbl_spectrophotometric_ranges(spectrophotometric_data) == [
            [1.0, 2.0], [3.0, 4.0]]


class TestAssignModelTypeWithLblFolding:
    """End-to-end (minus setup_retrieval's I/O) check of the actual design payoff:
    an lbl spectrophotometric instrument's own range, once folded into
    wv_range_high the way setup_retrieval now does it, always makes it back as
    `model_type == 'high'` from assign_model_type -- with no dedicated code path
    beyond what Phase 4 already built for real high-res instruments."""

    def test_lbl_instrument_is_assigned_high(self, monkeypatch):
        # A single real high-res spectrograph window, far from the lbl instrument.
        real_instrument_ranges = [[2.0, 2.5]]
        spectrophotometric_data = {
            'jwst_niriss': {'wv_range': [0.6, 2.8], 'opacity_mode': 'lbl'},
        }
        monkeypatch.setattr(retrieval, 'spectrophotometric_data', spectrophotometric_data,
                             raising=False)
        monkeypatch.setattr(retrieval, 'photometric_data', {}, raising=False)
        lbl_ranges = retrieval.get_lbl_spectrophotometric_ranges(spectrophotometric_data)
        wv_range_high = retrieval.get_wv_range(real_instrument_ranges + lbl_ranges)

        retrieval.assign_model_type(wv_range_high)

        assert spectrophotometric_data['jwst_niriss']['model_type'] == 'high'

    def test_c_k_instrument_outside_high_res_coverage_stays_low(self, monkeypatch):
        real_instrument_ranges = [[2.0, 2.5]]
        spectrophotometric_data = {
            'wfc3': {'wv_range': [1.1, 1.7]},  # opacity_mode defaults to 'c-k'
        }
        monkeypatch.setattr(retrieval, 'spectrophotometric_data', spectrophotometric_data,
                             raising=False)
        monkeypatch.setattr(retrieval, 'photometric_data', {}, raising=False)
        wv_range_high = retrieval.get_wv_range(real_instrument_ranges)

        retrieval.assign_model_type(wv_range_high)

        assert spectrophotometric_data['wfc3']['model_type'] == 'low'

    def test_lbl_and_c_k_instruments_coexist_with_correct_assignment(self, monkeypatch):
        real_instrument_ranges = [[2.0, 2.5]]
        spectrophotometric_data = {
            'jwst_niriss': {'wv_range': [0.6, 2.8], 'opacity_mode': 'lbl'},
            'spitzer': {'wv_range': [3.0, 5.0]},  # far from everything -> stays 'low'
        }
        monkeypatch.setattr(retrieval, 'spectrophotometric_data', spectrophotometric_data,
                             raising=False)
        monkeypatch.setattr(retrieval, 'photometric_data', {}, raising=False)
        lbl_ranges = retrieval.get_lbl_spectrophotometric_ranges(spectrophotometric_data)
        wv_range_high = retrieval.get_wv_range(real_instrument_ranges + lbl_ranges)

        retrieval.assign_model_type(wv_range_high)

        assert spectrophotometric_data['jwst_niriss']['model_type'] == 'high'
        assert spectrophotometric_data['spitzer']['model_type'] == 'low'


class TestGetResInstru:
    """`instrum_param_list` empty means no real high-res instrument at all -- the
    scenario Antoine wants for a retrieval running purely on lbl-flagged low-res
    data (e.g. JWST/NIRSpec G395H alone, no SPIRou/NIRPS). No real dataset used
    so far exercises this (KELT-20b's test config always has SPIRou in `instrum`),
    so this is currently the only automated coverage of it."""

    def test_takes_the_max_resolution_when_instruments_are_present(self):
        instrum_param_list = [{'resol': 70_000}, {'resol': 64_000}]
        assert retrieval.get_res_instru(instrum_param_list, 500_000) == 70_000

    def test_falls_back_to_native_resolution_when_no_instrument(self):
        assert retrieval.get_res_instru([], 500_000) == 500_000


class TestGetWvRangeLow:
    """`wv_range_all_low` can be genuinely empty (every low-res instrument
    lbl-flagged, no photometric data, pure LRR run with no high-res-range padding)
    -- found as a real crash (`np.min` on an empty array) while testing exactly
    this scenario on real KELT-20b data (g395H_1/g395H_2 both lbl, `instrum: []`,
    `retrieval_type: 'LRR'`)."""

    def test_spans_min_max_of_every_input_range(self):
        assert retrieval.get_wv_range_low([[1.0, 2.0], [3.0, 5.0]]) == [[1.0, 5.0]]

    def test_empty_input_gives_empty_output_not_a_crash(self):
        assert retrieval.get_wv_range_low([]) == []
