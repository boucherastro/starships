from starships.instruments import register_instrument, instruments_drs


class TestRegisterInstrument:
    """`register_instrument` (Chantier B, B2) is the supported way to add a custom
    instrument/DRS to `instruments_drs` without editing `instruments.py` -- either directly
    (e.g. from a notebook) or via `pipeline.reduction.load_custom_instrument` (config-driven,
    following the same pattern as `retrieval_utils.load_custom_get_ker`). Also reachable as
    `starships.planet_obs.register_instrument` (re-exported there for the reduction pipeline).
    """

    def test_registers_from_scratch(self):
        entry = register_instrument('TestScope-DRS', airmass='MY_AIRMASS', mjd='MY_MJD')
        try:
            assert instruments_drs['TestScope-DRS'] is entry
            assert entry['name'] == 'TestScope-DRS'
            assert entry['airmass'] == 'MY_AIRMASS'
            assert entry['mjd'] == 'MY_MJD'
        finally:
            del instruments_drs['TestScope-DRS']

    def test_base_on_copies_and_overrides(self):
        entry = register_instrument('TestScope2-DRS', base_on='SPIRou-APERO', mjd='MY_MJD')
        try:
            spirou = instruments_drs['SPIRou-APERO']
            # Copied from SPIRou-APERO...
            assert entry['airmass'] == spirou['airmass']
            assert entry['list_file_patterns'] == spirou['list_file_patterns']
            # ...except the explicit override
            assert entry['mjd'] == 'MY_MJD'
            assert spirou['mjd'] != 'MY_MJD'  # base entry itself untouched
        finally:
            del instruments_drs['TestScope2-DRS']

    def test_custom_read_all_sp_and_list_file_patterns(self):
        def fake_reader(path, file_list, **kwargs):
            return None

        patterns = {'e2ds_glob': '*.fits', 'e2ds_suffix': '.fits',
                    'tcorr_suffix': '_tcorr.fits', 'recon_suffix': '_recon.fits'}
        entry = register_instrument('TestScope3-DRS', read_all_sp=fake_reader,
                                     list_file_patterns=patterns)
        try:
            assert instruments_drs['TestScope3-DRS']['read_all_sp'] is fake_reader
            assert instruments_drs['TestScope3-DRS']['list_file_patterns'] == patterns
        finally:
            del instruments_drs['TestScope3-DRS']
