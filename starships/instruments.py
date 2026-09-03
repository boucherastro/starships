"""Everything about "what is instrument/DRS X and how do I get data from it": header
keywords, raw-file naming patterns, physical/observational properties (resolution,
wavelength coverage, orders/pixels), and the actual `read_all_sp_*` functions that read raw
FITS files for each one -- all in one place (Chantier B, B2 follow-up), instead of split
across a `planet_obs.py`-embedded dict (reduction-only fields) and a separate,
partly-duplicated `instruments_drs` dict here (retrieval-only fields, 9 fields kept in sync
by hand between the two).

`planet_obs.py` (the reduction pipeline, `Observations` class) imports `instruments_drs`
from here and uses `self.instrument['read_all_sp']` -- it does *not* need to attach anything
itself anymore, since each entry already has its own reader function attached below, right
next to its own dict definition. `load_instrum` (used by `retrieval.py`/`logl_grid.py`/
`pipeline/make_model.py`, none of which need `read_all_sp` or import `planet_obs.py` at all)
only ever reads the physical/observational fields.

This module does depend on `astropy.io.fits`/`pathlib`/`.homemade` (for the actual FITS
reading) but *not* on `planet_obs.py` itself, `exofile`, or `sklearn`/`PyAstronomy` -- so
importing it (e.g. from `pipeline/make_model.py`, which only ever needs `load_instrum`) stays
much lighter than importing all of `planet_obs.py`.
"""

import numpy as np
from pathlib import Path
from astropy.io import fits

from .list_of_dict import list_of_dict
from . import homemade as hm

# Raw-observation file-naming patterns, per instrument/DRS, used by split_nights.py to discover
# and group raw FITS files into visits/nights (Chantier B, B2). `e2ds_glob` finds the e2ds files
# in a directory; `e2ds_suffix`/`tcorr_suffix`/`recon_suffix` are the trailing part of the
# filename for each of the three file kinds STARSHIPS can use, sharing a common prefix (the
# reduction/observation ID). `recon_suffix` may be `None` for a dataset/DRS that doesn't
# produce a telluric reconstruction spectrum as a separate file (confirmed real for at least
# one NIRPS-APERO dataset below) — split_nights.py then skips writing a recon list entirely,
# matching pipeline.reduction.load_planet's own graceful fallback when list_recon is absent.
#
# Confirmed against real data on 2026-09-01 for two different real datasets:
# - SPIRou-APERO (local WASP-33b dataset): `{obs_id}_pp_e2dsff_AB.fits` /
#   `{obs_id}_pp_e2dsff_tcorr_AB.fits` / `{obs_id}_pp_e2dsff_recon_AB.fits`.
# - NIRPS-APERO (Narval WASP-127b dataset): `{obs_id}e.fits` / `{obs_id}t.fits`, no recon files.
#   (This dataset does carry an ARCFILE header matching {obs_id}.fits, like the ESO archive
#   convention the old split_nights.py assumed -- but reading it isn't actually necessary: the
#   local e2ds filename's own suffix already gives the same prefix, so no header read needed.)
# NIRPS-GENEVA/IGRINS entries are left unset until confirmed against real data too (see
# split_nights.py — raises a clear error rather than guessing).
_spirou_apero_file_patterns = {
    'e2ds_glob': '*_pp_e2dsff_AB.fits',
    'e2ds_suffix': '_pp_e2dsff_AB.fits',
    'tcorr_suffix': '_pp_e2dsff_tcorr_AB.fits',
    'recon_suffix': '_pp_e2dsff_recon_AB.fits',
}

_nirps_apero_file_patterns = {
    'e2ds_glob': '*e.fits',
    'e2ds_suffix': 'e.fits',
    'tcorr_suffix': 't.fits',
    'recon_suffix': None,
}

# spirou (apero)
spirou = dict()
spirou['name'] = 'SPIRou-APERO'
spirou['airmass'] = 'AIRMASS'
spirou['telaz'] = 'TELAZ'
spirou['adc1'] = 'SBADC1_P'
spirou['adc2'] = 'SBADC2_P'
spirou['mjd'] = 'MJD-OBS'
spirou['bjd'] = 'BJD'
spirou['exptime'] = 'EXPTIME'
spirou['berv'] = 'BERV'
spirou['list_file_patterns'] = _spirou_apero_file_patterns
# Physical/observational properties: resolution, high-res wavelength coverage, spectral
# order count, pixels per order.
spirou['resol'] = 64000
spirou['high_res_wv_lim'] = [0.9, 2.55]
spirou['nord'] = 49
spirou['npix'] = 4088

# nirps, apero DRS
nirps_apero = dict()
nirps_apero['name'] = 'NIRPS-APERO'
nirps_apero['airmass'] = 'HIERARCH ESO TEL AIRM START'
nirps_apero['telaz'] = 'HIERARCH ESO TEL AZ'
nirps_apero['adc1'] = 'HIERARCH ESO INS ADC1 START'
nirps_apero['adc2'] = 'HIERARCH ESO INS ADC2 START'
nirps_apero['mjd'] = 'MJD-OBS'
nirps_apero['bjd'] = 'BJD'
nirps_apero['exptime'] = 'EXPTIME'
nirps_apero['berv'] = 'BERV'
nirps_apero['list_file_patterns'] = _nirps_apero_file_patterns
nirps_apero['resol'] = 80000
nirps_apero['high_res_wv_lim'] = [0.9, 1.98]
# nirps, geneva/ESPRESSO DRS
# implementing
nirps_geneva = dict()
nirps_geneva['name'] = 'NIRPS-GENEVA'
nirps_geneva['airmass'] = 'HIERARCH ESO TEL AIRM START'
nirps_geneva['telaz'] = 'HIERARCH ESO TEL AZ'
nirps_geneva['adc1'] = 'HIERARCH ESO INS ADC1 START'
nirps_geneva['adc2'] = 'HIERARCH ESO INS ADC2 START'
nirps_geneva['mjd'] = 'MJD-OBS'
nirps_geneva['bjd'] = 'HIERARCH ESO QC BJD'
nirps_geneva['exptime'] = 'EXPTIME'
nirps_geneva['berv'] = 'HIERARCH ESO QC BERV'
nirps_geneva['list_file_patterns'] = None  # TODO: confirm against real NIRPS-GENEVA filenames
# No resol/high_res_wv_lim/nord/npix defined for NIRPS-GENEVA in the pre-merge
# starships.instruments module either -- left unset here too, not guessed.

igrins_zoe = dict()
igrins_zoe['name'] = 'IGRINS'
igrins_zoe['airmass'] = 'AMSTART'
# igrins_zoe['telaz'] = 'TELRA'
igrins_zoe['adc1'] = 'NADCS'
igrins_zoe['adc2'] = 'NADCS'
igrins_zoe['bjd'] = 'JD-OBS'
igrins_zoe['mjd'] = 'MJD-OBS'
igrins_zoe['exptime'] = 'EXPTIMET'
igrins_zoe['list_file_patterns'] = None  # TODO: confirm against real IGRINS filenames
igrins_zoe['resol'] = 45000
igrins_zoe['high_res_wv_lim'] = [1.45, 2.45]
igrins_zoe['nord'] = 54
igrins_zoe['npix'] = 2048

# dictionary with instrument-DRS names
instruments_drs = {
    'SPIRou-APERO': spirou,
    'NIRPS-APERO': nirps_apero,
    'NIRPS-GENEVA': nirps_geneva,
    'IGRINS': igrins_zoe
}

# Lowercase, DRS-format-agnostic aliases -- point at the physical instrument's base entry, so
# retrieval configs (`instrum: [spirou]`) never need to change when the *reduction* format
# does (e.g. switching from 'SPIRou-APERO' to 'SPIRou-APERO-CADC' raw data): both share the
# same resol/high_res_wv_lim (see `_derive_instrument`'s `base_on`), only 'spirou' is what
# retrieval.py/logl_grid.py/pipeline/make_model.py ever need to reference.
instruments_drs['spirou'] = spirou
instruments_drs['nirps_he'] = nirps_apero
instruments_drs['nirps_hr'] = nirps_geneva
instruments_drs['igrins'] = igrins_zoe


def _derive_instrument(name, base_on=None, **overrides):
    """Build one `instruments_drs` entry by copying `base_on`'s fields and applying
    `overrides` -- shared by the built-in DRS-format variants defined below (e.g.
    'SPIRou-APERO-CADC') and the public `register_instrument` (for genuinely custom,
    user-supplied instruments). Does *not* register the result in `instruments_drs` itself --
    callers decide that.
    """
    entry = dict(instruments_drs[base_on]) if base_on is not None else {}
    entry['name'] = name
    entry.update(overrides)
    return entry


def register_instrument(name, base_on=None, **overrides):
    """Register a new instrument/DRS in `instruments_drs`, for a **custom** instrument that
    isn't one of the built-in ones (Chantier B, B2) -- e.g. from a notebook, or from
    `pipeline/reduction.py::load_custom_instrument` when driven by a config file. Not used by
    this module's own built-in profiles (including the 'SPIRou-APERO-CADC'/'NIRPS-APERO-CADC'
    DRS-format variants below), which are defined directly as static entries instead, to keep
    this function's contract strictly about user-added instruments.

    Call this once, then use `Observations(instrument=name)` as usual. A custom read function
    can be supplied either here (`read_all_sp=...`) or per-call via
    `Observations.fetch_data(..., read_sp=...)`.

    Parameters
    ----------
    name : str
        Name to register the instrument/DRS under (used as the `instrument=` argument to
        `Observations`).
    base_on : str, optional
        Name of an existing instrument/DRS (e.g. `'SPIRou-APERO'`) to copy defaults from,
        overridden by `overrides`. Leave `None` to start from an empty dict (every field must
        then be given explicitly).
    **overrides
        Any of: `airmass`, `telaz`, `adc1`, `adc2`, `mjd`, `bjd`, `exptime`, `berv` (FITS
        header keywords), `read_all_sp` (callable, see `read_all_sp_spirou_apero` below for
        the expected signature/return), `list_file_patterns` (dict with `e2ds_glob`,
        `e2ds_suffix`, `tcorr_suffix`, `recon_suffix` — see the module-level comment above
        `instruments_drs` — only needed to use `split_nights.split_night` for this
        instrument), and the physical/observational properties used on the retrieval side
        (`resol`, `high_res_wv_lim`, `nord`, `npix`). `base_on` copies whichever of these the
        base entry already has, so a new DRS-format variant of an existing instrument (e.g.
        `base_on='NIRPS-APERO'` for a different reduction/file format of the same physical
        instrument) inherits its resolution/wavelength coverage for free without needing to
        repeat them.

    Returns
    -------
    dict
        The newly registered instrument/DRS dictionary (also stored in `instruments_drs[name]`).
    """
    entry = _derive_instrument(name, base_on=base_on, **overrides)
    instruments_drs[name] = entry
    return entry


def load_instrum(instrum_name):
    """Look up one `instruments_drs` entry by name (built-in name, lowercase alias, or a
    custom one registered via `register_instrument`). Used by the retrieval side
    (`retrieval.py`/`logl_grid.py`/`pipeline/make_model.py`) to get `resol`/
    `high_res_wv_lim`/etc. -- never needs `read_all_sp` (a reduction-only field)."""
    try:
        infos = instruments_drs[instrum_name]
    except KeyError:
        # Show the possible instruments in the error message
        raise KeyError(f"Invalid instrument name: {instrum_name}. Possible instruments are: {list(instruments_drs.keys())}")

    return infos


# =============================================================================
# Reading raw files: one read_all_sp_* function per instrument/DRS format, each attached to
# its own instruments_drs entry right below its definition. Observations.fetch_data (in
# planet_obs.py) calls whichever one is registered for the instrument in use via
# self.instrument['read_all_sp'] -- it never needs to know which reduction/file-format
# variant it's actually talking to.
# =============================================================================

def fits2wave(image, header):
    """
    Get the wave solution from the header using a filename
    """
    # size of the image
    nbypix, nbxpix = image.shape
    # get the keys with the wavelength polynomials
    wave_hdr = header['WAVE0*']
    # concatenate into a numpy array
    wave_poly = np.array([wave_hdr[i] for i in range(len(wave_hdr))])
    # get the per-order wavelength solution
    wave_poly = wave_poly.reshape(nbypix, len(wave_poly) // nbypix)
    # project polynomial coefficiels
    wavesol = np.zeros_like(image)
    # get the pixel range
    xpix = np.arange(nbxpix)
    # loop around orders
    for order_num in range(nbypix):
        wavesol[order_num] = np.polyval(wave_poly[order_num][::-1], xpix)
    # return wave grid
    return wavesol


def fits2wavenew(image, hdr):
    """
    Get the wave solution from the header using a filename
    """
    # size of the image
    nbypix, nbxpix = image.shape
    # get the keys with the wavelength polynomials
    wave_hdr = hdr['WAVE0*']
    # concatenate into a numpy array
    wave_poly = np.array([wave_hdr[i] for i in range(len(wave_hdr))])
    # get the number of orders
    nord = hdr['WAVEORDN']
    # get the per-order wavelength solution
    wave_poly = wave_poly.reshape(nord, len(wave_poly) // nord)
    # project polynomial coefficiels
    wavesol = np.zeros_like(image)
    # xpixel grid
    xpix = np.arange(nbxpix)
    # loop around orders
    for order_num in range(nord):
        # calculate wave solution for this order
        owave = val_cheby(wave_poly[order_num], xpix, domain=[0, nbxpix])
        # push into wave map
        wavesol[order_num] = owave
    # return wave grid
    return wavesol

def val_cheby(coeffs, xvector,  domain):
    """
    Using the output of fit_cheby calculate the fit to x  (i.e. y(x))
    where y(x) = T0(x) + T1(x) + ... Tn(x)

    :param coeffs: output from fit_cheby
    :param xvector: x value for the y values with fit
    :param domain: domain to be transformed to -1 -- 1. This is important to
    keep the components orthogonal. For SPIRou orders, the default is 0--4088.
    You *must* use the same domain when getting values with fit_cheby
    :return: corresponding y values to the x inputs
    """
    # transform to a -1 to 1 domain
    domain_cheby = 2 * (xvector - domain[0]) / (domain[1] - domain[0]) - 1
    # fit values using the domain and coefficients
    yvector = np.polynomial.chebyshev.chebval(domain_cheby, coeffs)
    # return y vector
    return yvector


def read_all_sp_spirou_apero(path, file_list, wv_default=None, blaze_default=None,
                blaze_path=None, debug=False, cheby=False):

    """
    Read all spectra
    Must have a list with all filename to read
    """

    headers, count, wv, blaze = list_of_dict([]), [], [], []
    blaze_path = blaze_path or path

    headers_princ = list_of_dict([])
    filenames = []
    blaze0 = None

    path = Path(path)
    blaze_path = Path(blaze_path)
    file_list = Path(file_list)

    with open(path / file_list) as f:

        for file in f:
            filename = file.split('\n')[0]

            if debug:
                print(filename)

            filenames.append(filename)
            hdul = fits.open(path / Path(filename))

            header = hdul[0].header
            image = hdul[1].data

            headers.append(header)
            count.append(image)

            try:
                wv_file = wv_default or hdul[0].header['WAVEFILE']
                with fits.open(path / Path(wv_file)) as f:
                    wvsol = f[0].data
            except (KeyError,FileNotFoundError) as e:
                use_cheby = cheby or (header.get('WAVEPOLY', '') == 'Chebyshev')
                if use_cheby:
                    wvsol = fits2wavenew(image, header)
                else:
                    wvsol = fits2wave(image, header)

            if blaze_default:
                blaze_file = blaze_default
            elif 'CDBBLAZE' in header:
                blaze_file = header['CDBBLAZE']
            else:
                raise KeyError(
                    f"Cannot find blaze file: 'CDBBLAZE' keyword missing from header of {filename}. "
                    "Pass blaze_default=<filename> to read_all_sp_spirou_apero or fetch_data."
                )

            blaze0 = fits.getdata(blaze_path / Path(blaze_file), ext=1)
            blaze.append(blaze0)

            wv.append(wvsol/1000)

    return headers, np.array(wv), np.array(count), np.array(blaze), filenames


spirou['read_all_sp'] = read_all_sp_spirou_apero


def read_all_sp_spirou_CADC(path, filename, file_list):
    '''
    Read all CADC-type spectra ('list_e2ds' or 'list_tellu_corrected' -- see
    `_read_all_sp_cadc_format`, the only caller since Chantier B B2 removed the standalone
    `fetch_data(CADC=True)` path).
    Must have a list with all filenames to read
    Note : Probably old-----updated by georgia on May 22, 2024
    '''
    headers_princ, headers_image, headers_tellu = list_of_dict([]), list_of_dict([]), list_of_dict([])
    count, wv, blaze, recon = [], [], [], []
    filenames = []

    with open(path + '/' + filename) as f:
        for file in f:
            filenames.append(file.split('\n')[0])

            hdul = fits.open(path + '/' + file.split('\n')[0])

            headers_princ.append(hdul[0].header)
            headers_image.append(hdul[1].header)
            if file_list == 'list_tellu_corrected':
                headers_tellu.append(hdul[4].header)
                recon.append(hdul[4].data)
                ext = [1,2,3]
            else:  # 'list_e2ds'
                ext = [1,5,9]
            count.append(hdul[ext[0]].data)
            wv.append(hdul[ext[1]].data / 1000)
            blaze.append(hdul[ext[2]].data)
    return headers_princ, headers_image, headers_tellu, np.array(wv), \
            np.array(count), np.array(blaze), np.array(recon), filenames


def read_all_sp_nirps_apero_CADC(path,filename,file_list):

    """
    Read all CADC-type spectra ('list_e2ds' or 'list_tellu_corrected' -- see
    `_read_all_sp_cadc_format`, the only caller since Chantier B B2 removed the standalone
    `fetch_data(CADC=True)` path).
    Must have a list with all filename to read
    """

    headers_princ, headers_image, headers_tellu = list_of_dict([]), list_of_dict([]), list_of_dict([])
    count, wv, blaze, recon = [], [], [], []
    filenames = []
    with open(str(path)+'/'+filename) as f:

        for file in f:
            filenames.append(file.split('\n')[0])
            hdul = fits.open(str(path)+'/'+file.split('\n')[0])

            headers_princ.append(hdul[0].header)
            headers_image.append(hdul[1].header)

            if file_list == 'list_tellu_corrected':
                headers_tellu.append(hdul[4].header)
                recon.append(hdul[4].data)
                ext = [1,2,3]
            else:  # 'list_e2ds'
                ext = [1,3,5]

            count.append(hdul[ext[0]].data)
            wv.append(hdul[ext[1]].data / 1000)
            blaze.append(hdul[ext[2]].data)

    return headers_princ, headers_image, headers_tellu, np.array(wv), \
            np.array(count), np.array(blaze), np.array(recon), filenames


def _read_all_sp_cadc_format(path, file_list, cadc_reader):
    """Adapts a `read_all_sp_*_CADC` reader (Chantier B, B2 follow-up) to the standard
    `read_all_sp_*(path, file_list, **kwargs) -> (headers, wave, count, blaze, filenames,
    recon)` signature, so a "bundled/embedded-extension" DRS format can be registered as its
    own instrument/DRS profile and used like any other instrument -- no more special-cased
    `CADC=True` boolean/dispatch in `Observations.fetch_data`.

    `cadc_reader` is `read_all_sp_spirou_CADC` or `read_all_sp_nirps_apero_CADC` -- both take
    `(path, filename, file_list)` where the 3rd argument is a fixed discriminator string
    (`'list_e2ds'` or `'list_tellu_corrected'`) picking which hardcoded extension indices to
    read, not a real file list name. Inferred here from whether 'tcorr' appears in the actual
    file-list name, matching the `list_e2ds_*`/`list_tcorr_*`/`list_recon_*` convention used
    everywhere else in the pipeline.

    Returns the flux extension's header (`hdul[1].header`, i.e. what the old code called
    `headers_image`) as `headers`, not the primary header (`hdul[0].header`,
    `headers_princ`) -- confirmed for NIRPS-APERO's embedded-extension format that BJD/BERV
    and everything else this package needs live only on the flux extension's header (see
    `read_all_sp_nirps_apero`'s own embedded-format handling), and
    `Observations.calc_sequence`'s old `CADC`-gated code path read BJD/BERV the same way
    (from `headers_image`) for both instruments, so this should carry over identically.
    """
    discriminator = 'list_tellu_corrected' if 'tcorr' in str(file_list).lower() else 'list_e2ds'
    _, headers_image, _, wave, count, blaze, recon, filenames = cadc_reader(
        str(path), str(file_list), discriminator)

    recon_out = recon if (discriminator == 'list_tellu_corrected' and len(recon)) else None
    return headers_image, wave, count, blaze, filenames, recon_out


def read_all_sp_spirou_cadc_format(path, file_list, **kwargs):
    """SPIRou-APERO, bundled/embedded-extension DRS format (formerly reached only via
    `fetch_data(CADC=True)`) -- see `_read_all_sp_cadc_format`. Not validated against real
    data this session (no such SPIRou dataset was available) -- this wraps the existing,
    previously-used `read_all_sp_spirou_CADC` as-is (same hardcoded extension indices), only
    the calling convention changed."""
    return _read_all_sp_cadc_format(path, file_list, read_all_sp_spirou_CADC)


def read_all_sp_nirps_apero_cadc_format(path, file_list, **kwargs):
    """NIRPS-APERO, bundled/embedded-extension DRS format (formerly reached only via
    `fetch_data(CADC=True)`) -- see `_read_all_sp_cadc_format`. Note: `read_all_sp_nirps_apero`
    (the default 'NIRPS-APERO' profile) already auto-detects and handles this same kind of
    embedded-extension format on its own (confirmed real, Chantier B B2) -- this separate
    profile is kept for the legacy hardcoded-index approach specifically, e.g. if a dataset's
    extension names ever don't match what the auto-detecting reader expects.
    """
    return _read_all_sp_cadc_format(path, file_list, read_all_sp_nirps_apero_CADC)


# Built-in "bundled/embedded-extension" DRS-format variants (Chantier B, B2 follow-up) --
# same physical instruments as 'SPIRou-APERO'/'NIRPS-APERO' above, different raw-file format.
# Built via `_derive_instrument` (not the public `register_instrument`) since these are
# officially-supported built-ins, not a user's custom instrument.
spirou_cadc = _derive_instrument('SPIRou-APERO-CADC', base_on='SPIRou-APERO',
                                  read_all_sp=read_all_sp_spirou_cadc_format)
nirps_apero_cadc = _derive_instrument('NIRPS-APERO-CADC', base_on='NIRPS-APERO',
                                       read_all_sp=read_all_sp_nirps_apero_cadc_format)
instruments_drs['SPIRou-APERO-CADC'] = spirou_cadc
instruments_drs['NIRPS-APERO-CADC'] = nirps_apero_cadc


def read_all_sp_igrins(path, file_list, blaze_path=None, input_type='data'):

    """
    Read all spectra
    Must have a list with all filename to read

    input_type: 'data'-observation data, 'recon'-telluric reconstruction
    """

    # create some empty list and append later

    file_list = path/Path(file_list)
    with open(file_list, 'r') as file:
        file_paths = file.readlines()
    file_paths = [path.strip() for path in file_paths]

    if input_type == 'data':

        headers, count, wv, blaze = list_of_dict([]), [], [], []
        filenames = []

        blaze_path = Path(blaze_path)

        # Iterate over the file paths and open each FITS file
        for file in file_paths:
            try:
                filenames.append(file)

                hdul = fits.open(file)

                header = hdul[0].header
                image = hdul[0].data
                wvsol = hdul[1].data

                headers.append(header)
                count.append(image)
                wv.append(wvsol)

                hdul.close()  # Close the FITS file after processing

            except IOError:
                print(f"Error opening FITS file: {file}")

        with fits.open(blaze_path) as hdul:
            b = hdul[0].data
            blaze.append(b)

        return headers, np.array(wv), np.array(count), np.array(blaze), filenames

    elif input_type == 'recon': # file_list is telluric_recon

        tellu_recon = []

        for file in file_paths:
            try:
                hdul = fits.open(file)

                tellu = hdul[0].data
                tellu_recon.append(tellu)

                hdul.close()

            except IOError:
                print(f"Error opening FITS file: {file}")

        return np.array(tellu_recon)


igrins_zoe['read_all_sp'] = read_all_sp_igrins


# a very slight modification of the spirou function: the wave solution is now in the second extension of the wave file
def read_all_sp_nirps_apero(path, file_list, wv_default=None, blaze_default=None,
                            blaze_path=None, debug=False, cheby=False):
    """
    Read all spectra
    Must have a list with all filename to read

    Returns
    -------
    headers, wv, count, blaze, filenames, recon
        `recon` (Chantier B, B2 follow-up) is the embedded telluric reconstruction spectrum
        when this file format bundles one as a `Recon` extension alongside `Flux{fiber}`/
        `Wave{fiber}`/`Blaze{fiber}` (confirmed real for at least one NIRPS-APERO dataset,
        where reading `list_tcorr` this way makes a separate `list_recon` unnecessary — see
        `Observations.fetch_data`), or `None` when this file doesn't have one (the normal
        case — recon then comes from a genuinely separate `list_recon` file list, as before).
    """

    headers, count, wv, blaze, recon = list_of_dict([]), [], [], [], []
    blaze_path = blaze_path or path

    headers_princ = list_of_dict([])
    filenames = []
    blaze0 = None

    path = Path(path)
    blaze_path = Path(blaze_path)
    file_list = Path(file_list)

    with open(path / file_list) as f:

        for file in f:
            filename = file.split('\n')[0]

            if debug:
                print(filename)

            filenames.append(filename)
            hdul = fits.open(path / Path(filename))

            # Newer NIRPS-APERO DRS versions bundle flux/wave/blaze for each fiber as
            # extensions of the *same* file (e.g. `FluxA`/`WaveA`/`BlazeA`, `FluxB`/`WaveB`/
            # `BlazeB`) instead of referencing separate wave-solution/blaze calibration files
            # by name (`WAVEFILE`/`CDBBLAZE` header keywords, looked up below), AND move
            # per-exposure keywords this reader needs (`BJD`/`BERV`/...) from the primary
            # header into the flux extension's header rather than keeping them in the primary
            # HDU. Confirmed real on a NIRPS-APERO dataset with none of `CDBBLAZE`/`WAVEFILE`/
            # `BJD`/`BERV` in its primary header at all (Chantier B, B2 follow-up) — the flux
            # extension's header turned out to be a strict superset of the primary header's
            # cards plus these extras, so using it as `header` covers both old and new formats.
            ext_names = [hdu.name for hdu in hdul]
            fiber = hdul[1].name[-1] if hdul[1].name.startswith('Flux') else None
            wave_ext, blaze_ext = f'Wave{fiber}', f'Blaze{fiber}'
            embedded_calib = fiber is not None and wave_ext in ext_names and blaze_ext in ext_names

            header = hdul[1].header if embedded_calib else hdul[0].header
            image = hdul[1].data

            headers.append(header)
            count.append(image)

            if embedded_calib:
                wvsol = hdul[wave_ext].data
                blaze0 = hdul[blaze_ext].data
                # Some formats also bundle the telluric reconstruction spectrum this way
                # (confirmed real: present in the tcorr file, absent from the e2ds file for
                # the same visit) -- collect it here instead of expecting a separate
                # list_recon file list.
                if 'Recon' in ext_names:
                    recon.append(hdul['Recon'].data)
            else:
                try:
                    wv_file = wv_default or hdul[0].header['WAVEFILE']
                    with fits.open(path / Path(wv_file)) as f:
                        wvsol = f[1].data
                except (KeyError, FileNotFoundError) as e:
                    if cheby is False:
                        wvsol = fits2wave(image, header)
                    else:
                        wvsol = fits2wavenew(image, header)

                if blaze_default:
                    blaze_file = blaze_default
                elif 'CDBBLAZE' in header:
                    blaze_file = header['CDBBLAZE']
                else:
                    raise KeyError(
                        f"Cannot find blaze file: 'CDBBLAZE' keyword missing from header of {filename}, "
                        "and no embedded Wave*/Blaze* extensions found either. "
                        "Pass blaze_default=<filename> to read_all_sp_nirps_apero or fetch_data."
                    )

                blaze0 = fits.getdata(blaze_path / Path(blaze_file), ext=1)

            blaze.append(blaze0)
            wv.append(wvsol / 1000)
            hdul.close()

    # Only treat recon as available if every exposure actually had an embedded Recon
    # extension -- a per-format property, not something that should vary per-exposure.
    recon_out = np.array(recon) if len(recon) == len(filenames) and recon else None

    return headers, np.array(wv), np.array(count), np.array(blaze), filenames, recon_out


nirps_apero['read_all_sp'] = read_all_sp_nirps_apero


def read_all_sp_nirps_geneva(path, file_list, wv_default=None, blaze_default=None,
                             blaze_path=None, debug=False, cheby=False):
    """
    Read all spectra
    Must have a list with all filename to read
    Include 'recon' in the name of the file list for the recon files
    """

    headers, count, wv, blaze = list_of_dict([]), [], [], []
    blaze_path = blaze_path or path

    filenames = []
    blaze0 = None

    recon = 'recon' in file_list

    path = Path(path)
    blaze_path = Path(blaze_path)
    file_list = Path(file_list)

    with open(path / file_list) as f:

        for file in f:
            filename = file.split('\n')[0]

            if debug:
                print(filename)

            filenames.append(filename)
            hdul = fits.open(path / Path(filename))

            header = hdul[0].header
            if recon:
                image = hdul[6].data
            else:
                image = hdul[1].data

            headers.append(header)
            count.append(image)

            # vacuum wavelengths
            if recon:
                wvsol = hdul[2].data
            else:
                wvsol = hdul[4].data

            # remove berv correction (Geneva data is already berv corrected)
            # barycentric correction (km/s)
            berv = header['HIERARCH ESO QC BERV']
            shift = hm.calc_shift(berv, kind='rel')
            wvsol = wvsol/shift

            try:
                blaze_file = blaze_default or header['HIERARCH ESO PRO REC1 CAL24 NAME']
            except KeyError:
                blaze_file = header['HIERARCH ESO PRO REC1 CAL24 NAME']

            blaze0 = fits.getdata(blaze_path / Path(blaze_file), ext=1)

            blaze.append(blaze0)

            wv.append(wvsol / 10000)

    return headers, np.array(wv), np.array(count), np.array(blaze), filenames


nirps_geneva['read_all_sp'] = read_all_sp_nirps_geneva
