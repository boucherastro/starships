import numpy as np
from astropy.io import fits

from starships.instruments import read_all_sp_nirps_apero


def _write_synthetic_nirps_file(path, flux, wave, blaze, recon=None, extra_header=None):
    """Build a minimal NIRPS-APERO-style multi-extension FITS file (Chantier B, B2
    follow-up): flux/wave/blaze (and optionally the telluric reconstruction spectrum) for
    fiber A bundled as extensions of the same file, instead of external wave-solution/blaze
    calibration files referenced by header keyword -- confirmed real on a NIRPS-APERO dataset
    with no CDBBLAZE/WAVEFILE/BJD/BERV in its primary header at all, and recon (when present)
    embedded in the tcorr file as a 'Recon' extension rather than a separate file."""
    # Set EXTNAME directly on the header (not via ImageHDU's `name=` kwarg) to preserve
    # mixed case on write -- astropy's `name=` kwarg uppercases it, but the real files this
    # mimics genuinely have mixed-case EXTNAME cards ('FluxA', not 'FLUXA').
    primary = fits.PrimaryHDU()
    primary.header['MJD-OBS'] = 60014.0
    flux_hdu = fits.ImageHDU(data=flux)
    flux_hdu.header['EXTNAME'] = 'FluxA'
    # The real dataset has BJD/BERV (and everything else) only on the flux extension's
    # header, not the primary one -- reproduce that here.
    flux_hdu.header['BJD'] = 2460014.5
    flux_hdu.header['BERV'] = -2.43
    if extra_header:
        for k, v in extra_header.items():
            flux_hdu.header[k] = v
    wave_hdu = fits.ImageHDU(data=wave)
    wave_hdu.header['EXTNAME'] = 'WaveA'
    blaze_hdu = fits.ImageHDU(data=blaze)
    blaze_hdu.header['EXTNAME'] = 'BlazeA'
    hdus = [primary, flux_hdu, wave_hdu, blaze_hdu]
    if recon is not None:
        recon_hdu = fits.ImageHDU(data=recon)
        recon_hdu.header['EXTNAME'] = 'Recon'
        hdus.append(recon_hdu)
    fits.HDUList(hdus).writeto(path, overwrite=True)


class TestReadAllSpNirpsAperoEmbeddedCalib:
    """Regression test for the embedded Wave*/Blaze*/BJD/BERV/Recon fallback added to
    `read_all_sp_nirps_apero` after finding a real NIRPS-APERO dataset (WASP-127b, Chantier B
    B2 Narval validation) with none of `CDBBLAZE`/`WAVEFILE`/`BJD`/`BERV` in its primary
    header -- before this fix, the reader raised `KeyError` on 'CDBBLAZE' for such files even
    though everything needed was already embedded as extensions of the same FITS file. The
    telluric reconstruction spectrum turned out to be embedded the same way (in the tcorr
    file, as a 'Recon' extension) rather than genuinely absent, as first assumed.
    """

    def test_reads_wave_and_blaze_from_embedded_extensions(self, tmp_path):
        flux = np.ones((3, 5)) * 10.0
        wave = np.linspace(1000, 1010, 15).reshape(3, 5)
        blaze = np.ones((3, 5)) * 0.9

        fits_path = tmp_path / 'NIRPS.synthetic.e.fits'
        _write_synthetic_nirps_file(fits_path, flux, wave, blaze)

        list_file = tmp_path / 'list_e2ds_test'
        list_file.write_text(fits_path.name + '\n')

        headers, wv, count, blaze_out, filenames, recon = read_all_sp_nirps_apero(tmp_path, list_file.name)

        np.testing.assert_allclose(count[0], flux)
        np.testing.assert_allclose(wv[0], wave / 1000)  # function divides by 1000
        np.testing.assert_allclose(blaze_out[0], blaze)
        assert filenames == [fits_path.name]
        assert recon is None  # no Recon extension in this file

    def test_bjd_and_berv_read_from_flux_extension_header(self, tmp_path):
        flux = np.ones((3, 5))
        wave = np.ones((3, 5))
        blaze = np.ones((3, 5))

        fits_path = tmp_path / 'NIRPS.synthetic2.e.fits'
        _write_synthetic_nirps_file(fits_path, flux, wave, blaze)

        list_file = tmp_path / 'list_e2ds_test2'
        list_file.write_text(fits_path.name + '\n')

        headers, *_ = read_all_sp_nirps_apero(tmp_path, list_file.name)

        assert headers[0]['BJD'] == 2460014.5
        assert headers[0]['BERV'] == -2.43

    def test_reads_embedded_recon_when_present(self, tmp_path):
        flux = np.ones((3, 5))
        wave = np.ones((3, 5))
        blaze = np.ones((3, 5))
        recon = np.ones((3, 5)) * 0.5

        fits_path = tmp_path / 'NIRPS.synthetic3.t.fits'
        _write_synthetic_nirps_file(fits_path, flux, wave, blaze, recon=recon)

        list_file = tmp_path / 'list_tcorr_test3'
        list_file.write_text(fits_path.name + '\n')

        *_, recon_out = read_all_sp_nirps_apero(tmp_path, list_file.name)

        assert recon_out is not None
        np.testing.assert_allclose(recon_out[0], recon)
