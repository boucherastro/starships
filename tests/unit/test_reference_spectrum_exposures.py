import numpy as np

from starships.transpec import resolve_reference_spectrum_exposures


class TestResolveReferenceSpectrumExposures:
    """`resolve_reference_spectrum_exposures` (Chantier B, B1) makes the reference
    spectrum's ("master-out") exposure selection a real, explicit choice. Before this
    was factored out of `build_trans_spectrum4`, the equivalent inline code silently
    overwrote `iOut_temp` to 'all' no matter what was passed in -- restoring the
    `iOut_temp is None` branch here makes the out-of-transit/out-of-eclipse option
    real again, without changing the default ('all') behaviour that Antoine confirmed
    is intentional (planetary signal negligible + diluted by its own motion, so using
    every exposure improves the reference spectrum's S/N).
    """

    def test_all_string_uses_every_exposure(self):
        result = resolve_reference_spectrum_exposures('all', iOut=np.array([1, 2]), n_exposures=5)
        np.testing.assert_array_equal(result, np.arange(5))

    def test_none_uses_true_out_of_transit_exposures(self):
        iOut = np.array([0, 1, 4])
        result = resolve_reference_spectrum_exposures(None, iOut=iOut, n_exposures=5)
        np.testing.assert_array_equal(result, iOut)

    def test_explicit_array_passed_through(self):
        explicit = np.array([2, 3])
        result = resolve_reference_spectrum_exposures(explicit, iOut=np.array([0, 1]), n_exposures=5)
        np.testing.assert_array_equal(result, explicit)

    def test_oversized_iout_temp_falls_back_to_all(self):
        # Guards a pre-existing edge case (a stale/mismatched iOut longer than the actual
        # number of exposures) -- falls back to 'all' rather than indexing out of bounds.
        oversized = np.arange(10)
        result = resolve_reference_spectrum_exposures(oversized, iOut=np.array([0]), n_exposures=5)
        np.testing.assert_array_equal(result, np.arange(5))
