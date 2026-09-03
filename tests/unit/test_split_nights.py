from pipeline.split_nights import _sibling_filenames


class TestSiblingFilenames:
    """`_sibling_filenames` (Chantier B, B2) derives the tcorr/recon sibling filenames of an
    e2ds file from shared instrument-specific suffixes, replacing the old APERO-only
    `FILE_FRAMES`/`ARCFILE`-header-based logic (which crashed on real SPIRou-APERO files —
    they don't carry an `ARCFILE` keyword) with something that works for any instrument/DRS
    registered in `starships.planet_obs.instruments_drs`.
    """

    SPIROU_APERO_PATTERNS = {
        'e2ds_glob': '*_pp_e2dsff_AB.fits',
        'e2ds_suffix': '_pp_e2dsff_AB.fits',
        'tcorr_suffix': '_pp_e2dsff_tcorr_AB.fits',
        'recon_suffix': '_pp_e2dsff_recon_AB.fits',
    }

    def test_derives_tcorr_and_recon_from_e2ds(self):
        siblings = _sibling_filenames('2446828o_pp_e2dsff_AB.fits', self.SPIROU_APERO_PATTERNS)
        assert siblings == {
            'tcorr': '2446828o_pp_e2dsff_tcorr_AB.fits',
            'recon': '2446828o_pp_e2dsff_recon_AB.fits',
        }
