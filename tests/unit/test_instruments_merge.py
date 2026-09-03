from starships.instruments import instruments_drs, load_instrum


class TestInstrumentsMerge:
    """`starships.instruments` used to hold a *separate* `instruments_drs` dict (resolution,
    wavelength coverage, nord/npix -- used by retrieval.py/logl_grid.py/make_model.py) with 9
    fields duplicated from `planet_obs.py`'s own dict (header keywords, file reading -- used
    by the reduction pipeline). Merged into a single dict, owned by `instruments.py` (which
    now also holds the `read_all_sp_*` reader functions themselves, one per instrument/DRS
    format, attached right next to each entry); `planet_obs.py` imports `instruments_drs`
    from there instead of defining its own (Chantier B, B2 follow-up).
    """

    def test_load_instrum_still_works_for_built_ins(self):
        info = load_instrum('NIRPS-APERO')
        assert info['resol'] == 80000
        assert info['high_res_wv_lim'] == [0.9, 1.98]
        assert info['bjd'] == 'BJD'  # reduction-side field, also present

    def test_lowercase_aliases_preserved(self):
        assert load_instrum('spirou') is instruments_drs['SPIRou-APERO']
        assert load_instrum('nirps_he') is instruments_drs['NIRPS-APERO']
        assert load_instrum('nirps_hr') is instruments_drs['NIRPS-GENEVA']
        assert load_instrum('igrins') is instruments_drs['IGRINS']

    def test_cadc_format_variant_inherits_physical_properties(self):
        """A retrieval config using the plain 'spirou'/'nirps_he' alias must see the same
        resol/high_res_wv_lim regardless of which DRS-format variant actually produced the
        reduced data -- retrieval shouldn't need to change when the raw-file format does."""
        spirou = load_instrum('spirou')
        spirou_cadc = load_instrum('SPIRou-APERO-CADC')
        assert spirou_cadc['resol'] == spirou['resol']
        assert spirou_cadc['high_res_wv_lim'] == spirou['high_res_wv_lim']

        nirps = load_instrum('nirps_he')
        nirps_cadc = load_instrum('NIRPS-APERO-CADC')
        assert nirps_cadc['resol'] == nirps['resol']
        assert nirps_cadc['high_res_wv_lim'] == nirps['high_res_wv_lim']

    def test_unknown_instrument_raises_with_helpful_message(self):
        try:
            load_instrum('NotAnInstrument')
            assert False, "should have raised"
        except KeyError as e:
            assert 'NotAnInstrument' in str(e)
