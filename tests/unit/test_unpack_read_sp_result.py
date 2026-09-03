from starships.planet_obs import _unpack_read_sp_result


class TestUnpackReadSpResult:
    """`_unpack_read_sp_result` (Chantier B, B2 follow-up) lets `Observations.fetch_data` call
    any registered `read_all_sp_*` reader uniformly, whether it returns the standard 5-tuple
    `(headers, wave, count, blaze, filenames)` or the extended 6-tuple `(..., recon)` used by
    readers that can report an embedded telluric reconstruction spectrum.
    """

    def test_passes_through_six_tuple(self):
        result = (1, 2, 3, 4, 5, 6)
        assert _unpack_read_sp_result(result) == (1, 2, 3, 4, 5, 6)

    def test_pads_five_tuple_with_none(self):
        result = (1, 2, 3, 4, 5)
        assert _unpack_read_sp_result(result) == (1, 2, 3, 4, 5, None)
