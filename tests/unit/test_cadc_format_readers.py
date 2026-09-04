import numpy as np
from astropy.io import fits

from starships.instruments import (
    read_all_sp_spirou_cadc_format,
    read_all_sp_nirps_apero_cadc_format,
    instruments_drs,
)


def _write_multi_ext_fits(path, n_ext, data_by_ext, header_by_ext=None):
    """Build a FITS file with `n_ext` image extensions (indices 1..n_ext), each filled with
    a distinct constant value from `data_by_ext` (dict of {ext_index: value}), matching the
    positional-extension convention `read_all_sp_*_CADC` expects (Chantier B, B2 follow-up:
    removing the CADC=True/False flag in favour of per-instrument read functions)."""
    hdus = [fits.PrimaryHDU()]
    for i in range(1, n_ext + 1):
        value = data_by_ext.get(i, 0.0)
        hdu = fits.ImageHDU(data=np.full((2, 3), value, dtype=float))
        if header_by_ext and i in header_by_ext:
            for k, v in header_by_ext[i].items():
                hdu.header[k] = v
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(path, overwrite=True)


class TestCadcFormatReaders:
    """`read_all_sp_spirou_cadc_format`/`read_all_sp_nirps_apero_cadc_format` (Chantier B, B2
    follow-up) adapt the pre-existing `read_all_sp_spirou_CADC`/`read_all_sp_nirps_apero_CADC`
    readers (previously reachable only via `Observations.fetch_data(CADC=True)`, with its own
    hardcoded per-instrument-name dispatch) to the standard `read_all_sp_*(path, file_list,
    **kwargs) -> (headers, wave, count, blaze, filenames, recon)` signature, so they can be
    registered as ordinary instrument/DRS profiles instead.
    """

    def test_spirou_cadc_format_reads_e2ds_by_fixed_indices(self, tmp_path):
        # read_all_sp_spirou_CADC uses ext=[1,5,9] for 'list_e2ds' -- count, wave, blaze.
        fits_path = tmp_path / 'spirou_e2ds.fits'
        _write_multi_ext_fits(fits_path, 9, {1: 100.0, 5: 2000.0, 9: 0.5})

        list_file = tmp_path / 'list_e2ds_test'
        list_file.write_text(fits_path.name + '\n')

        headers, wave, count, blaze, filenames, recon = read_all_sp_spirou_cadc_format(
            tmp_path, list_file.name)

        assert count[0][0, 0] == 100.0
        assert wave[0][0, 0] == 2000.0 / 1000  # reader divides wave by 1000
        assert blaze[0][0, 0] == 0.5
        assert filenames == [fits_path.name]
        assert recon is None

    def test_spirou_cadc_format_reads_tcorr_with_embedded_recon(self, tmp_path):
        # ext=[1,2,3] for 'list_tellu_corrected', recon at ext 4.
        fits_path = tmp_path / 'spirou_tcorr.fits'
        _write_multi_ext_fits(fits_path, 4, {1: 100.0, 2: 2000.0, 3: 0.5, 4: 0.9})

        list_file = tmp_path / 'list_tcorr_test'
        list_file.write_text(fits_path.name + '\n')

        headers, wave, count, blaze, filenames, recon = read_all_sp_spirou_cadc_format(
            tmp_path, list_file.name)

        assert count[0][0, 0] == 100.0
        assert recon is not None
        assert recon[0][0, 0] == 0.9

    def test_nirps_apero_cadc_format_reads_e2ds_by_fixed_indices(self, tmp_path):
        # read_all_sp_nirps_apero_CADC uses ext=[1,3,5] for 'list_e2ds'.
        fits_path = tmp_path / 'nirps_e2ds.fits'
        _write_multi_ext_fits(fits_path, 5, {1: 100.0, 3: 2000.0, 5: 0.5})

        list_file = tmp_path / 'list_e2ds_test'
        list_file.write_text(fits_path.name + '\n')

        headers, wave, count, blaze, filenames, recon = read_all_sp_nirps_apero_cadc_format(
            tmp_path, list_file.name)

        assert count[0][0, 0] == 100.0
        assert wave[0][0, 0] == 2000.0 / 1000
        assert blaze[0][0, 0] == 0.5
        assert recon is None

    def test_list_file_can_live_outside_the_data_directory(self, tmp_path):
        """The list file itself can live anywhere (e.g. a scratch dir with write access,
        separate from a read-only shared data directory) as long as `file_list` is passed as
        an absolute path -- `pathlib`'s `/` operator returns the right-hand side unchanged
        when it is already absolute, so `Path(data_dir) / Path(absolute_list_path)` resolves
        to the list file, while the FITS filenames *inside* it still resolve relative to
        `data_dir`. This used to only work for the non-CADC readers (`read_all_sp_spirou_apero`
        etc., already `pathlib`-based); `read_all_sp_*_CADC` used string concatenation
        (`path + '/' + filename`), which breaks for an absolute `filename` (produces a
        malformed double path) -- fixed to use `pathlib` too, for consistency."""
        data_dir = tmp_path / 'data'
        data_dir.mkdir()
        fits_path = data_dir / 'spirou_e2ds.fits'
        _write_multi_ext_fits(fits_path, 9, {1: 100.0, 5: 2000.0, 9: 0.5})

        list_dir = tmp_path / 'elsewhere'
        list_dir.mkdir()
        list_file = list_dir / 'list_e2ds_test'
        list_file.write_text(fits_path.name + '\n')

        headers, wave, count, blaze, filenames, recon = read_all_sp_spirou_cadc_format(
            data_dir, list_file)  # list_file is an absolute Path, not just a filename

        assert count[0][0, 0] == 100.0
        assert filenames == [fits_path.name]

    def test_instruments_are_registered(self):
        assert instruments_drs['SPIRou-APERO-CADC']['read_all_sp'] is read_all_sp_spirou_cadc_format
        assert instruments_drs['NIRPS-APERO-CADC']['read_all_sp'] is read_all_sp_nirps_apero_cadc_format
        # base_on='SPIRou-APERO'/'NIRPS-APERO' should have copied the header-keyword fields.
        assert instruments_drs['SPIRou-APERO-CADC']['mjd'] == instruments_drs['SPIRou-APERO']['mjd']
        assert instruments_drs['NIRPS-APERO-CADC']['mjd'] == instruments_drs['NIRPS-APERO']['mjd']
