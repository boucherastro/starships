
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

from pathlib import Path
import astropy.units as u
from astropy.time import Time
from astropy.io import fits
import astropy.constants as const
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.table import Table

from .list_of_dict import *
from . import orbite as o
from . import transpec as ts
from . import homemade as hm
from . import spectrum as spectrum
from .mask_tools import interp1d_masked
from .plotting_fcts import plot_all_orders_correl
from .analysis import calc_snr_1d
from .extract import get_mask_tell, get_mask_noise
from .correlation import calc_logl_BL_ord, quick_correl_3dmod # calc_logl_OG_cst, calc_logl_OG_ord
# from .analysis import make_quick_model
# from .extract import quick_norm
# from transit_prediction.masterfile import MasterFile 
# from masterfile.archive import MasterFile
from exofile.archive import ExoFile
from scipy.interpolate import interp1d

# from fits2wave import fits2wave
# from scipy.interpolate import InterpolatedUnivariateSpline
# from scipy.signal import medfilt
# from tqdm import tqdm
# import os
from sklearn.decomposition import PCA
from collections import OrderedDict
import gc
import logging
from PyAstronomy import pyasl

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# Constants
DEFAULT_LISTS_FILENAMES = {False: {'file_list': 'list_e2ds',
                                   'file_list_tcorr': 'list_tellu_corrected',
                                   'file_list_recon': 'list_tellu_recon'},
                           True: {'file_list': 'list_s1d',
                                  'file_list_tcorr': 'list_tellu_corrected_1d',
                                  'file_list_recon': 'list_tellu_recon_1d'}}

# Dictionaries for different instruments and/or DRS

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

igrins_zoe = dict()
igrins_zoe['name'] = 'IGRINS'
igrins_zoe['airmass'] = 'AMSTART'
# igrins_zoe['telaz'] = 'TELRA'
igrins_zoe['adc1'] = 'NADCS'
igrins_zoe['adc2'] = 'NADCS'
igrins_zoe['bjd'] = 'JD-OBS'
igrins_zoe['mjd'] = 'MJD-OBS'
igrins_zoe['exptime'] = 'EXPTIMET'

# dictionary with instrument-DRS names
instruments_drs = {
    'SPIRou-APERO': spirou,
    'NIRPS-APERO': nirps_apero,
    'NIRPS-GENEVA': nirps_geneva,
    'IGRINS': igrins_zoe
}

# def fits2wave(file_or_header):
#     info = """
#         Provide a fits header or a fits file
#         and get the corresponding wavelength
#         grid from the header.
        
#         Usage :
#           wave = fits2wave(hdr)
#                   or
#           wave = fits2wave('my_e2ds.fits')
        
#         Output has the same size as the input
#         grid. This is derived from NAXIS 
#         values in the header
#     """


#     # check that we have either a fits file or an astropy header
#     if type(file_or_header) == str:
#         hdr = fits.getheader(file_or_header)
#     elif str(type(file_or_header)) == "<class 'astropy.io.fits.header.Header'>":
#         hdr = file_or_header
#     else:
#         print()
#         print('~~~~ wrong type of input ~~~~')
#         print()

#         print(info)
#         return []

#     # get the keys with the wavelength polynomials
#     wave_hdr = hdr['WAVE0*']
#     # concatenate into a numpy array
#     wave_poly = np.array([wave_hdr[i] for i in range(len(wave_hdr))])

#     # get the number of orders
#     nord = hdr['WAVEORDN']

#     # get the per-order wavelength solution
#     wave_poly = wave_poly.reshape(nord, len(wave_poly) // nord)

#     # get the length of each order (normally that's 4088 pix)
#     npix = 4088 #hdr['NAXIS1']

#     # project polynomial coefficiels
#     wavesol = [np.polyval(wave_poly[i][::-1],np.arange(npix)) for i in range(nord) ]

#     # return wave grid
#     return np.array(wavesol)



def fits2wave(image, header):
    """
    Get the wave solution from the header using a filename
    """
#     header = fits.getheader(filename, ext=0)
#     image = fits.getdata(filename, ext=1)
# def fits2wave(filename):
#     """
#     Get the wave solution from the header using a filename
#     """
#     header = fits.getheader(filename, ext=0)
#     image = fits.getdata(filename, ext=1)
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
                blaze_path=None, debug=False, ver06=False, cheby=False):

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

            if ver06 is False: # --- for V0.6 data ---
                header = hdul[0].header
                image = hdul[1].data
            else:
                header = hdul[0].header
                image = hdul[0].data

            headers.append(header)
            count.append(image)

            try:
                wv_file = wv_default or hdul[0].header['WAVEFILE']
                with fits.open(path / Path(wv_file)) as f:
                      wvsol = f[1].data
            except (KeyError,FileNotFoundError) as e:
                if cheby is False:
                    wvsol = fits2wave(image, header)
                else:
                    wvsol = fits2wavenew(image, header)
#                 if debug:
#                     print(wvsol)

#                 if blaze0 is None:
            try:
                blaze_file = blaze_default or header['CDBBLAZE']
            except KeyError:
                blaze_file = header['CDBBLAZE']

            if ver06 is False:
                blaze0 = fits.getdata(blaze_path / Path(blaze_file), ext=1)
            else:
                with fits.open(blaze_path / Path(blaze_file)) as f:
#                         header = fits.getheader(filename, ext=0) 
                    blaze0 = f[0].data
#                         print(blaze)
            blaze.append(blaze0)
                
            wv.append(wvsol/1000)

    return headers, np.array(wv), np.array(count), np.array(blaze), filenames


def read_all_sp_spirou_CADC(path, filename, file_list):
    '''
    Read all CADC-type spectra
    Must have a list with all filenames to read
    Note : Probably old-----updated by georgia on May 22, 2024
    '''
    headers_princ, headers_image, headers_tellu = list_of_dict([]), list_of_dict([]), list_of_dict([])
    count, wv, blaze, recon = [], [], [], []
    filenames = []
    # print(path)

    with open(path + '/' + filename) as f:
        for file in f:
            # print(file)
            
            filenames.append(file.split('\n')[0])
            # print(filenames)
            # print(file.split(‘\n’)[0])
            # print(path+‘/’+file.split(‘\n’)[0])
            
            hdul = fits.open(path + '/' + file.split('\n')[0])
            # print(hdul)

            headers_princ.append(hdul[0].header)
            headers_image.append(hdul[1].header)
            if file_list == 'list_v':
                count.append(hdul[1].data)
            else:
                if file_list == 'list_tellu_corrected':
                    headers_tellu.append(hdul[4].header)
                    recon.append(hdul[4].data)
                    ext = [1,2,3]
                if file_list == 'list_e2ds':
                    ext = [1,5,9]
                count.append(hdul[ext[0]].data)
                wv.append(hdul[ext[1]].data / 1000)
                blaze.append(hdul[ext[2]].data)
    return headers_princ, headers_image, headers_tellu, np.array(wv), \
            np.array(count), np.array(blaze), np.array(recon), filenames

def read_all_sp_nirps_apero_CADC(path,filename,file_list):
    
    """
    Read all CADC-type spectra
    Must have a list with all filename to read 
    """
    
    headers_princ, headers_image, headers_tellu = list_of_dict([]), list_of_dict([]), list_of_dict([])
    count, wv, blaze, recon = [], [], [], []
    filenames = []
    # print(path)
    with open(str(path)+'/'+filename) as f:

        for file in f:
            # print(file)
            filenames.append(file.split('\n')[0])
            # print(filenames)
            # print(file.split('\n')[0])
            # print(path+'/'+file.split('\n')[0])
            hdul = fits.open(str(path)+'/'+file.split('\n')[0])
            # print(hdul)
            
            
            headers_princ.append(hdul[0].header)
            headers_image.append(hdul[1].header)
            
            if file_list == 'list_v':
                count.append(hdul[1].data)
            else:
                if file_list == 'list_tellu_corrected':
                    headers_tellu.append(hdul[4].header)
                    recon.append(hdul[4].data)
                    ext = [1,2,3]
                if file_list == 'list_e2ds':
                    ext = [1,3,5]

                count.append(hdul[ext[0]].data)
                wv.append(hdul[ext[1]].data / 1000)
                blaze.append(hdul[ext[2]].data)
    
    return headers_princ, headers_image, headers_tellu, np.array(wv), \
            np.array(count), np.array(blaze), np.array(recon), filenames

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
        # headers, count, wv, blaze = [], [], [], []
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


# a very slight modification of the spirou function: the wave solution is now in the second extension of the wave file
def read_all_sp_nirps_apero(path, file_list, wv_default=None, blaze_default=None,
                            blaze_path=None, debug=False, ver06=False, cheby=False):
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

            if ver06 is False:  # --- for V0.6 data ---
                header = hdul[0].header
                image = hdul[1].data
            else:
                header = hdul[0].header
                image = hdul[0].data

            headers.append(header)
            count.append(image)

            try:
                wv_file = wv_default or hdul[0].header['WAVEFILE']
                with fits.open(path / Path(wv_file)) as f:
                    wvsol = f[1].data
            except (KeyError, FileNotFoundError) as e:
                if cheby is False:
                    wvsol = fits2wave(image, header)
                else:
                    wvsol = fits2wavenew(image, header)
            #                 if debug:
            #                     print(wvsol)

            #                 if blaze0 is None:
            try:
                blaze_file = blaze_default or header['CDBBLAZE']
            except KeyError:
                blaze_file = header['CDBBLAZE']

            if ver06 is False:
                blaze0 = fits.getdata(blaze_path / Path(blaze_file), ext=1)
            else:
                with fits.open(blaze_path / Path(blaze_file)) as f:
                    #                         header = fits.getheader(filename, ext=0)
                    blaze0 = f[0].data
            #                         print(blaze)
            blaze.append(blaze0)

            wv.append(wvsol / 1000)

    return headers, np.array(wv), np.array(count), np.array(blaze), filenames

def read_all_sp_nirps_geneva(path, file_list, wv_default=None, blaze_default=None,
                             blaze_path=None, debug=False, cheby=False):
    """
    Read all spectra
    Must have a list with all filename to read
    Include 'recon' in the name of the file list for the recon files
    """

    headers, count, wv, blaze = list_of_dict([]), [], [], []
    blaze_path = blaze_path or path

    # headers_princ = list_of_dict([])
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


# give the appropriate functions to read spectra to all the instrument/DRS dictionaries
spirou['read_all_sp'] = read_all_sp_spirou_apero
nirps_apero['read_all_sp'] = read_all_sp_nirps_apero
nirps_geneva['read_all_sp'] = read_all_sp_nirps_geneva
igrins_zoe['read_all_sp'] = read_all_sp_igrins


def fake_noise(flux, gwidth=1):
    # Generate white noise
    mean = 0.
    std = np.ma.std(flux, axis=0)
    noise = np.random.normal(mean, std, flux.shape)
    noise = np.ma.array(noise, mask=flux.mask)
    # Convolve noise with gaussian kernel (correlated noise)
    fct = lambda f:convolve(f ,
                            Gaussian1DKernel(gwidth),
                            boundary='extend',
                            mask=f.mask,
                            preserve_nan=True)
    noise = np.apply_along_axis(fct, -1, noise)
    noise = np.ma.masked_invalid(noise)
    # Renormalize since the convolution reduces the noise
    new_std = np.ma.std(noise, axis=0)
    factor = np.ma.median(new_std / std)
    return noise / factor


def gen_rv_sequence(self, p, plot=False, K=None):
    if K is None:
        K, vr, Kp, vrp = o.rv(self.nu, p.period, e=p.excent, i=p.incl, w=p.w, Mp=p.M_pl, Mstar=p.M_star)
    else:
        if isinstance(K, u.Quantity):
            K = K.to(u.km / u.s)
        else:
            K = K * u.km / u.s
        Kp = o.Kp_theo(K, p.M_star, p.M_pl)
        vr = o.rv_theo_t(K, self.t, p.mid_tr, p.period, plnt=False)
        vrp = o.rv_theo_t(Kp, self.t, p.mid_tr, p.period, plnt=True)

    self.vrp = vrp.to('km/s').squeeze()  # km/s
    self.vr = vr.to('km/s').squeeze()  # km/s   # np.zeros_like(vrp)  # ********
    self.K, self.Kp = K.to('km/s').squeeze(), Kp.to('km/s').squeeze()

    v_star = (vr + p.RV_sys).to('km/s').value
    v_pl = (vrp + p.RV_sys).to('km/s').value
    self.dv_pl = v_pl - self.berv  # +berv
    self.dv_star = v_star - self.berv  # +berv

    if plot is True:
        full_seq = np.arange(self.t_start[0] - 1, self.t_start[-1] + 1, 0.05)
        full_t = full_seq * u.d

        full_nu = o.t2trueanom(p.period, full_t.to(u.d), t0=p.mid_tr, e=p.excent)

        K_full, vr_full, Kp_full, vrp_full = o.rv(full_nu, p.period, e=p.excent, i=p.incl, w=p.w,
                                                  Mp=p.M_pl, Mstar=p.M_star)
        vrp_full = vrp_full.to('km/s')  # km/s
        vr_full = vr_full.to('km/s')  # km/s   # np.zeros_like(vrp)  # ********

        plt.figure()
        plt.plot(full_t, vr_full + p.RV_sys)
        plt.plot(self.t, self.vr + p.RV_sys, 'ro')


def gen_transit_model(self, p, kind_trans, coeffs, ld_model, iin=False, plot=False):
    self.nu = o.t2trueanom(p.period, self.t.to(u.d), t0=p.t_peri, e=p.excent)

    rp, x, y, z, self.sep, p.bRstar = o.position(self.nu, e=p.excent, i=p.incl, w=p.w, omega=p.omega,
                                                 Rstar=p.R_star, P=p.period, ap=p.ap, Mp=p.M_pl, Mstar=p.M_star)


    self.phase = ((self.t - p.mid_tr) / p.period).decompose().value
    self.phase -= np.round(self.phase.mean())
    if kind_trans == 'emission':
        if (self.phase < 0).all():
            self.phase += 1.0

    tag = ['primary', 'secondary'][np.argmin([np.abs(np.mean(self.phase) - 0), np.abs(np.mean(self.phase) - 0.5)])]
    # if tag == 'primary':
    T0 = p.mid_tr
    if tag == 'secondary':
        T0 = p.mid_tr + 0.5 * p.period.to(u.d)
        z = None
    
    print(p.mid_tr)
    
    i_peri = np.searchsorted(self.t, p.mid_tr)

    p.b = (p.bRstar / p.R_star).decompose()
    out, part, total = o.transit(p.R_star, p.R_pl + p.H, self.sep,
                                 z=z, nu=self.nu, r=np.array(rp.decompose()), i_tperi=i_peri, w=p.w)
    #         print(out,part,total)

    
        
    if kind_trans == 'transmission':
        print('Transmission')
        self.iOut = out
        self.part = part
        self.total = total
        self.iIn = np.sort(np.concatenate([part, total]))
    #             print(self.iIn.size, self.iOut.size)

    elif kind_trans == 'emission':
        print('Emission')
        self.iOut = total
        self.part = part
        self.total = out
        self.iIn = np.sort(np.concatenate([out]))
    #             print(self.iIn.size, self.iOut.size)

    self.iin, self.iout = o.where_is_the_transit(self.t, p.mid_tr, p.period, p.trandur)
    self.iout_e, self.iin_e = o.where_is_the_transit(self.t, p.mid_tr + 0.5 * p.period, p.period, p.trandur)

    
    if (self.part.size == 0) and (iin is True):
        print('Taking iin and iout')
        if kind_trans == 'transmission':
            self.iIn = self.iin
            self.iOut = self.iout
        elif kind_trans == 'emission':
            self.iIn = self.iin_e
            self.iOut = self.iout_e

    self.icorr = self.iIn

    if plot is True:
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[1].plot(self.t, np.nanmean(self.SNR[:, :], axis=-1), 'o-')
        if out.size > 0:
            ax[0].plot(self.t[self.iOut], self.AM[self.iOut], 'ro', label="Out of transit")
            #             ax[1].plot(self.t[self.iOut], self.adc1[self.iOut],'ro')
            ax[1].plot(self.t[self.iOut], np.nanmean(self.SNR[self.iOut, :], axis=-1), 'ro', label="Out of transit")
        if part.size > 0:
            ax[0].plot(self.t[self.part], self.AM[self.part], 'go', label="Ingress/Egress")
            #             ax[1].plot(self.t[self.part], self.adc1[self.part],'go')
            ax[1].plot(self.t[self.part], np.nanmean(self.SNR[self.part, :], axis=-1), 'go', label="Ingress/Egress")
        if total.size > 0:
            ax[0].plot(self.t[self.total], self.AM[self.total], 'bo', label="In transit")
            #             ax[1].plot(self.t[self.total], self.adc1[self.total],'bo')
            ax[1].plot(self.t[self.total], np.nanmean(self.SNR[self.total, :], axis=-1), 'bo', label="In transit")
        if self.iin.size > 0:
            ax[0].plot(self.t[self.iIn], self.AM[self.iIn], 'g.')
            #             ax[1].plot(self.t[self.iIn], self.adc1[self.iIn],'g.')
            ax[1].plot(self.t[self.iIn], np.nanmean(self.SNR[self.iIn, :], axis=-1), 'g.')
        if self.iout.size > 0:
            ax[0].plot(self.t[self.iOut], self.AM[self.iOut], 'r.')
            #             ax[1].plot(self.t[self.iOut], self.adc1[self.iOut],'r.')
            ax[1].plot(self.t[self.iOut], np.nanmean(self.SNR[self.iOut, :], axis=-1), 'r.')
        
        ax[0].set_ylabel('Airmass')
        # ax[1].set_ylabel('ADC1 angle')
        ax[1].set_xticks(np.array(self.t), np.arange(1, np.shape(self.t)[0] + 1))
        ax[1].set_xlabel("Exposition")
        ax[1].set_ylabel('Mean SNR')
        ax[0].legend()
        ax[1].legend()

    self.alpha = hm.calc_tr_lightcurve(p, coeffs, self.t, T0, ld_model=ld_model, kind_trans=kind_trans)
    #         self.alpha = np.array([(hm.circle_overlap(p.R_star.to(u.m), p.R_pl.to(u.m), sep) / \
    #                         p.A_star).value for sep in self.sep]).squeeze()
    #         self.alpha = np.array([hm.circle_overlap(p.R_star, p.R_pl, sep).value for sep in self.sep])
    
    
    self.alpha_frac = self.alpha / self.alpha.max()

    self.kind_trans, self.coeffs, self.ld_model = kind_trans, coeffs, ld_model


#######################
### Observation class
#######################

class Observations():
    
    """
    Observations class object
    Will contain all data and parameters
    Note : Probably could be optimized
    """

    # added an instrument argument to pass the appropriate dictionary. Default=spirou
    # to properly pass a dictionary from outside (e.g. a jupyter notebook),
    # need to write e.g. instrument=planet_obs.nirps_apero
    def __init__(self, wave=np.array([]), count=np.array([]), blaze=np.array([]),
                 headers = list_of_dict([]), headers_image = list_of_dict([]), headers_tellu = list_of_dict([]), 
                 tellu=np.array([]), uncorr=np.array([]), 
                 name='', path='',filenames=[], planet=None, CADC=False, pl_kwargs=None, instrument='SPIRou-APERO'):
        
        self.name = name
        self.path = Path(path)
        
        # --- Get the system parameters from the ExoFile
        if planet is None:
            if pl_kwargs is not None:
                self.planet = Planet(name, **pl_kwargs)
            else:
                self.planet = Planet(name)
        else:
            self.planet=planet
        
        self.wave=wave
        self.count=count
        self.blaze=blaze
        self.headers=headers
        self.headers_image=headers_image
        self.headers_tellu=headers_tellu
        self.filenames=filenames
        self.n_spec = len(self.filenames)
        
        self.uncorr=uncorr
        self.tellu=tellu
        self.CADC = CADC
        # get the instrument dictionary from the dict of instruments
        # the string/name
        self.instrument_name = instrument
        # the dictionary
        self.instrument = instruments_drs[instrument]

                     
    def fetch_data(self, path, CADC=False, list_e2ds='list_e2ds',
                    list_tcorr='list_tellu_corrected', list_recon='list_tellu_recon',
                    read_sp=None, **kwargs):
        """
        Retrieve all the relevent data in path 
        (tellu corrected, tellu recon and uncorrected spectra from lists of files)
        Georgia Mraz--debugged CADC function on May 22nd 2024
        """

            # TODO Remove CADC references -> Use an instrument/reduction configuration instead
            self.CADC = CADC

            # get the appropriate function to read spectra from the instrument's dictionary
            # if read function is not specified as an argument
            if not read_sp:
                read_sp = self.instrument['read_all_sp']

            if CADC:
                log.info('Fetching data')
                
                if self.instrument_name == 'SPIRou-APERO':
                    headers, headers_image, headers_tellu, \
                    wave, count, blaze, tellu, filenames = read_all_sp_spirou_CADC(path, list_tcorr, 'list_tellu_corrected')

                    self.headers_image, self.headers_tellu = headers_image, headers_tellu
                    log.info("Fetching the uncorrected spectra")
                    _, _, _, _, count_uncorr, blaze_uncorr, _, filenames_uncorr = read_all_sp_spirou_CADC(path, list_e2ds, 'list_e2ds')

                if self.instrument_name == 'NIRPS-APERO':
                    headers, headers_image, headers_tellu, \
                    wave, count, blaze, tellu, filenames = read_all_sp_nirps_apero_CADC(path, list_tcorr, 'list_tellu_corrected')
                
                    self.headers_image, self.headers_tellu = headers_image, headers_tellu
                    log.info("Fetching the uncorrected spectra")
                    _, _, _, _, count_uncorr, blaze_uncorr, _, filenames_uncorr = read_all_sp_nirps_apero_CADC(path, list_e2ds, 'list_e2ds')

                else:
                    log.info('Fetching data')
                    headers, headers_image, headers_tellu, \
                    wave, count, blaze, tellu, filenames = read_all_sp_spirou_CADC(path, list_tcorr, 'list_tellu_corrected')
                
                    self.headers_image, self.headers_tellu = headers_image, headers_tellu
                    log.info("Fetching the uncorrected spectra")
                    _, _, _, _, count_uncorr, blaze_uncorr, _, filenames_uncorr = read_all_sp_spirou_CADC(path, list_e2ds,'list_e2ds')

            else:
                log.info("Fetching the uncorrected spectra")
                log.info(f"File: {list_e2ds}")

                headers, wave, count_uncorr, blaze_uncorr, filenames_uncorr = read_sp(path, list_e2ds, **kwargs)

                if list_tcorr is None:
                    log.info('No telluric correction available')
                    count = count_uncorr.copy()
                    blaze = blaze_uncorr.copy()
                    filenames = filenames_uncorr

                else:
                    log.info('Fetching data')
                    log.info(f"File: {list_tcorr}")
                    headers, wave, count, blaze, filenames = read_sp(path, list_tcorr, **kwargs)

                #             self.headers = headers
                #             self.wave = np.array(wv)
                #             self.count = np.ma.masked_invalid(count)
                #             self.blaze = np.ma.masked_array(blaze)
                #             self.filenames  = filenames


                if list_recon is None:
                    log.info('No reconstruction available')
                    tellu = np.ones_like(count)

                else:
                    log.info("Fetching the tellurics")
                    log.info(f"File: {list_recon}")
                    _, _, tellu, _, _ = read_sp(path, list_recon, **kwargs)
                    # tellu = read_sp(path, list_recon, input_type='recon', **kwargs)

            self.headers = headers
            self.wave = np.array(wave)
            self.count = np.ma.masked_invalid(count)
            self.blaze = np.ma.masked_invalid(blaze)
            self.filenames = filenames
            self.filenames_uncorr = filenames_uncorr

            self.tellu = np.ma.masked_invalid(tellu)
            if np.mean(count_uncorr) < 0:
                print('Mean below 0 = {}, flipping sign'.format(np.mean(count_uncorr))) 
                count_uncorr = -count_uncorr
            count_uncorr = np.ma.masked_invalid(np.clip(count_uncorr, 0,None))

            self.uncorr = count_uncorr

            self.uncorr_fl = self.uncorr/(blaze_uncorr/np.nanmax(blaze_uncorr, axis=-1)[:,:,None])
                
            self.path = Path(path)
            
        
    def select_transit(self, transit_tag, bloc=None):
        """
        To split down all the data into singular observing blocks/nights
        """
        
        if bloc is not None:
            transit_tag = transit_tag[bloc]
        
        new_headers = list_of_dict([])
        for tag in transit_tag:
            new_headers.append(self.headers[tag])
            
        
        new_headers_im = list_of_dict([])
        new_headers_tl = list_of_dict([])
        if self.CADC is True:
            for tag in transit_tag:
                new_headers_im.append(self.headers_image[tag])
                new_headers_tl.append(self.headers_tellu[tag])
        
        
#         sub_obs = Observations(headers=new_headers, wave=self.wave[transit_tag],
#                             count=self.count[transit_tag], blaze=self.blaze[transit_tag], 
#                             tellu=self.tellu[transit_tag], 
#                             uncorr=self.uncorr[transit_tag],
#                             name=self.name, planet=self.planet , 
#                             path=self.path, filenames=np.array(self.filenames)[transit_tag],
# #                             filenames_uncorr=np.array(self.filenames_uncorr)[transit_tag], 
#                             CADC=self.CADC, headers_image=new_headers_im, headers_tellu=new_headers_tl)
        
#         try:
#             sub_obs.filenames_uncorr = np.array(self.filenames_uncorr)[transit_tag]
#         except AttributeError:
#             sub_obs.filenames_uncorr = np.array(self.filenames)[transit_tag]
            
        
#         return sub_obs

        # add instrument argument
        return Observations(headers=new_headers,
                            wave=self.wave[transit_tag],
                            count=self.count[transit_tag], blaze=self.blaze[transit_tag], 
                            tellu=self.tellu[transit_tag], 
                            uncorr=self.uncorr[transit_tag],
                            name=self.name, planet=self.planet , 
                            path=self.path, filenames=np.array(self.filenames)[transit_tag],
                            # filenames_uncorr=np.array(self.filenames_uncorr)[transit_tag],
                            CADC=self.CADC, headers_image=new_headers_im, headers_tellu=new_headers_tl,
                            instrument=self.instrument_name) #, n_spec = len(self.filenames))
    
    # switched hard '49' value to self.nord
    # call instrument dictionary for problematic header keys         
    def calc_sequence(self, plot=True, sequence=None, K=None, uncorr=False, iin=False,
                      coeffs=[0.4], ld_model='linear', time_type='BJD', kind_trans='transmission'):
        
        """
        Compute the sequence time series stuff 
        (time, airmass(t), RV(t), when and where it is in transit/eclipse, etc.)
        """
        
        p = self.planet
        # self.headers[0]['VERSION']
        self.n_spec, self.nord, self.npix = self.count.shape
        
        if sequence is None:
            
            if self.CADC is False:

                if time_type == 'BJD':
                    self.t_start = Time(np.array(self.headers.get_all(self.instrument['bjd'])[0], dtype='float'),
                                format='jd').jd.squeeze()# * u.d
                # TODO check for start, end or mid mjd keys for instruments
                # or take mjd + exptime / 2
                elif time_type == 'MJD':
                    self.t_start = Time((np.array(self.headers.get_all('MJDATE')[0], dtype='float') + \
                                        np.array(self.headers.get_all('MJDEND')[0], dtype='float')) / 2, 
                                format='jd').jd.squeeze()# * u.d
                    
                try:
                    self.SNR = np.ma.masked_invalid([np.array(self.headers.get_all('EXTSN'+'{:03}'.format(order))[0],
                                dtype='float') for order in range(self.nord)]).T
                except KeyError:
                    self.SNR = np.sqrt(np.ma.median(self.count,axis=-1))

                try:
                    self.berv0 = np.array(self.headers.get_all(self.instrument['berv'])[0], dtype='float').squeeze()
                except KeyError:
                    ra = self.headers[0]['OBJRA']
                    dec = self.headers[0]['OBJDEC']
                    bjds = [hdr['JD-OBS'] for hdr in self.headers]

                    # Cerro Pachon, Chile
                    lat = -70.73669
                    lon = -30.24075
                    alt = 2722.0
                    berv = np.array([pyasl.helcorr(lat, lon, alt, ra, dec, bjd)[0] for bjd in bjds])
                    # berv = np.zeros_like(berv)
                    self.berv0 = berv

            else:
                # print('CADC correct')
                
                print((self.headers_image.get_all('EXTSN002')))
                
                
#                 obs_date = [date+' '+hour for date,hour in zip(self.headers_image.get_all('DATE-OBS')[0], \
#                                                self.headers.get_all('UTIME')[0])]
#                 self.t_start = Time(obs_date).jd * u.d
#                 self.t_start = Time(np.array(self.headers_image.get_all('BJD')[0], dtype='float'), 
#                                 format='jd').jd.squeeze() * u.d
                if time_type == 'BJD':
                    self.t_start = Time(np.array(self.headers_image.get_all(self.instrument['bjd'])[0], dtype='float'),
                                format='jd').jd.squeeze() #* u.d
                elif time_type == 'MJD':
                    self.t_start = Time((np.array(self.headers_image.get_all('MJDATE')[0], dtype='float') + \
                                        np.array(self.headers_image.get_all('MJDEND')[0], dtype='float')) / 2,
                                format='jd').jd.squeeze() #* u.d
                    
            #note to self---> this was commented out (lines 976-982 of the SNR and BERV) and i put it back to make the CADC reduction run
            #also the EXTSN+{003} was changed from 03 to 003 - Georgia Mraz(May 23rd 2024) also the EXTSN+{003} was changed from 03 to 003

                try:
                    self.SNR = np.ma.masked_invalid([np.array(self.headers_image.get_all('SNR'+'{}'.format(order))[0], \
                                             dtype='float') for order in range(self.nord)]).T
                except KeyError:
                    self.SNR = np.ma.masked_invalid([np.array(self.headers_image.get_all('EXTSN'+'{:003}'.format(order))[0], \
                                         dtype='float') for order in range(self.nord)]).T
                self.berv0 = np.array(self.headers_image.get_all('BERV')[0], dtype='float').squeeze()
            
            self.dt = np.array(np.array(self.headers.get_all(self.instrument['exptime'])[0], dtype='float') ).squeeze() * u.s
            self.AM = np.array(self.headers.get_all(self.instrument['airmass'])[0], dtype='float').squeeze()

            try:
                self.telaz = np.array(self.headers.get_all(self.instrument['telaz'])[0], dtype='float').squeeze()
            except KeyError:
                self.telaz = None

            self.adc1 = np.array(self.headers.get_all(self.instrument['adc1'])[0], dtype='float').squeeze()
            self.adc2 = np.array(self.headers.get_all(self.instrument['adc2'])[0], dtype='float').squeeze()
            
            self.SNR = np.clip(self.SNR, 0,None)
            self.flux = self.count/(self.blaze/np.nanmax(self.blaze, axis=-1)[:,:,None])
            # print(self.SNR)
        else :
            self.t_start = sequence[0] #* u.d
            self.SNR = sequence[1]
            self.berv0 = sequence[2]
            self.dt = sequence[3] * u.s

            self.AM = sequence[4]
            self.telaz = np.empty_like(self.AM)
            self.adc1 = np.empty_like(self.AM)
            self.adc2 = np.empty_like(self.AM)
        
            self.flux = self.count/(self.blaze/np.nanmax(self.blaze, axis=-1)[:,:,None])

        self.berv= self.berv0.copy()
        light_curve = np.ma.sum(np.ma.sum(self.count, axis=-1), axis=-1)
        self.light_curve = light_curve / np.nanmax(light_curve)
        self.t = self.t_start.copy() * u.d #+ self.dt/2
        
        self.N0f= (~np.isnan(self.flux)).sum(axis=-1)
        self.N0= (~np.isnan(self.uncorr)).sum(axis=-1)
        
        if uncorr is False:
            # - Noise
            medians_rel_noise = np.ma.median(np.sqrt(self.uncorr)/self.flux, axis=-1)
        else:
            # - Noise
            medians_rel_noise = np.ma.median(np.sqrt(self.uncorr)/self.uncorr, axis=-1)
#         if np.mean(self.uncorr) < 0:
#             medians_rel_noise = np.ma.median(np.sqrt(self.uncorr)/self.uncorr, axis=-1)
#             medians_rel_noise = np.ma.median(np.sqrt(self.flux)/self.flux, axis=-1)
            
        median0 = np.ma.median(medians_rel_noise, axis=0)
        self.scaling = (medians_rel_noise/median0[None,:])[:,:,None]
        if not hasattr(self, 'noise'):
            self.noise = None
            
        # ---- Transit model
#
#         self.nu = o.t2trueanom(p.period, self.t.to(u.d), t0=p.mid_tr, e=p.excent)
#
#         rp, x, y, z, self.sep, p.bRstar = o.position(self.nu, e=p.excent, i=p.incl, w=p.w, omega=p.omega,
#                                                 Rstar=p.R_star, P=p.period, ap=p.ap, Mp=p.M_pl, Mstar=p.M_star)
#
#         i_peri = np.searchsorted(self.t, p.mid_tr)
#
#         p.b = (p.bRstar / p.R_star).decompose()
#         out, part, total = o.transit(p.R_star, p.R_pl + p.H, self.sep,
#                                      z=z, nu=self.nu, r=np.array(rp.decompose()), i_tperi=i_peri, w=p.w)
#         #         print(out,part,total)
#
#         if kind_trans == 'transmission':
#             print('Transmission')
#             self.iOut = out
#             self.part = part
#             self.total = total
#             self.iIn = np.sort(np.concatenate([part,total]))
#         #             print(self.iIn.size, self.iOut.size)
#
#
#
#         elif kind_trans == 'emission':
#             print('Emission')
#             self.iOut = total
#             self.part = part
#             self.total = out
#             self.iIn = np.sort(np.concatenate([out]))
#
#         #             print(self.iIn.size, self.iOut.size)
#
#         self.iin, self.iout = o.where_is_the_transit(self.t, p.mid_tr, p.period, p.trandur)
#         self.iout_e, self.iin_e = o.where_is_the_transit(self.t, p.mid_tr+0.5*p.period, p.period, p.trandur)
#
#         if (self.part.size == 0) and (iin is True) :
#             print('Taking iin and iout')
#             if kind_trans == 'transmission':
#                 self.iIn = self.iin
#                 self.iOut = self.iout
#             elif kind_trans == 'emission':
#                 self.iIn = self.iin_e
#                 self.iOut = self.iout_e
#
#         self.icorr = self.iIn
#
#         if plot is True:
#             fig, ax = plt.subplots(3,1, sharex=True)
#             ax[2].plot(self.t, np.nanmean(self.SNR[:, :],axis=-1),'o-')
#             if out.size > 0 :
#                 ax[0].plot(self.t[self.iOut],self.AM[self.iOut],'ro')
#                 ax[1].plot(self.t[self.iOut], self.adc1[self.iOut],'ro')
#                 ax[2].plot(self.t[self.iOut], np.nanmean(self.SNR[self.iOut, :],axis=-1),'ro')
#             if part.size > 0 :
#                 ax[0].plot(self.t[self.part],self.AM[self.part],'go')
#                 ax[1].plot(self.t[self.part], self.adc1[self.part],'go')
#                 ax[2].plot(self.t[self.part], np.nanmean(self.SNR[self.part, :],axis=-1),'go')
#             if total.size > 0 :
#                 ax[0].plot(self.t[self.total],self.AM[self.total],'bo')
#                 ax[1].plot(self.t[self.total], self.adc1[self.total],'bo')
#                 ax[2].plot(self.t[self.total], np.nanmean(self.SNR[self.total, :],axis=-1),'bo')
#             if self.iin.size > 0 :
#                 ax[0].plot(self.t[self.iIn], self.AM[self.iIn],'g.')
#                 ax[1].plot(self.t[self.iIn], self.adc1[self.iIn],'g.')
#                 ax[2].plot(self.t[self.iIn], np.nanmean(self.SNR[self.iIn, :],axis=-1),'g.')
#             if self.iout.size > 0 :
#                 ax[0].plot(self.t[self.iOut], self.AM[self.iOut],'r.')
#                 ax[1].plot(self.t[self.iOut], self.adc1[self.iOut],'r.')
#                 ax[2].plot(self.t[self.iOut], np.nanmean(self.SNR[self.iOut, :],axis=-1),'r.')
#
#
#             ax[0].set_ylabel('Airmass')
#             ax[1].set_ylabel('ADC1 angle')
#             ax[2].set_ylabel('Mean SNR')
#
#
#
#         self.alpha = hm.calc_tr_lightcurve(p, coeffs, self.t.value, ld_model=ld_model, kind_trans=kind_trans)
# #         self.alpha = np.array([(hm.circle_overlap(p.R_star.to(u.m), p.R_pl.to(u.m), sep) / \
# #                         p.A_star).value for sep in self.sep]).squeeze()
# #         self.alpha = np.array([hm.circle_overlap(p.R_star, p.R_pl, sep).value for sep in self.sep])
#         self.alpha_frac = self.alpha/self.alpha.max()
#
        gen_transit_model(self, p, kind_trans, coeffs, ld_model, plot=plot)
        # --- Radial velocities

        gen_rv_sequence(self, p, plot=False)
        #
        # if K is None:
        #     K, vr, Kp, vrp = o.rv(self.nu, p.period, e=p.excent, i=p.incl, w=p.w, Mp=p.M_pl, Mstar=p.M_star)
        # else:
        #     if isinstance(K, u.Quantity):
        #          K= K.to(u.km/u.s)
        #     else:
        #          K= K*u.km/u.s
        #     Kp = o.Kp_theo(K, p.M_star, p.M_pl)
        #     vr = o.rv_theo_t(K, self.t, p.mid_tr, p.period, plnt=False)
        #     vrp = o.rv_theo_t(Kp, self.t, p.mid_tr, p.period, plnt=True)
        #
        # self.vrp = vrp.to('km/s').squeeze()  # km/s
        # self.vr = vr.to('km/s').squeeze()  # km/s   # np.zeros_like(vrp)  # ********
        # self.K, self.Kp = K.to('km/s').squeeze(), Kp.to('km/s').squeeze()
        #
        # v_star = (vr + p.RV_sys).to('km/s').value
        # v_pl = (vrp + p.RV_sys).to('km/s').value
        # self.dv_pl = v_pl - self.berv  #+berv
        # self.dv_star = v_star - self.berv  #+berv
    
#         if plot is True:
#             full_seq = np.arange(self.t_start[0].value-1, self.t_start[-1].value+1, 0.05)
#             full_t = full_seq * u.d

#             full_nu = o.t2trueanom(p.period, full_t.to(u.d), t0=p.mid_tr, e=p.excent)

#             K_full, vr_full, Kp_full, vrp_full = o.rv(full_nu, p.period, e=p.excent, i=p.incl, w=p.w, 
#                                                       Mp=p.M_pl, Mstar=p.M_star)
#             vrp_full = vrp_full.to('km/s')  # km/s
#             vr_full = vr_full.to('km/s')  # km/s   # np.zeros_like(vrp)  # ********

#             plt.figure()
#             plt.plot(full_t, vr_full+p.RV_sys)
#             plt.plot(self.t, self.vr+p.RV_sys, 'ro')

#         self.phase = (((self.t_start - p.mid_tr - p.period/2) % p.period)/p.period) - 0.5

#         self.phase = ((self.t-p.mid_tr)/p.period).decompose().value
#         self.phase -= np.round(self.phase.mean())
#         if kind_trans == 'emission':
#             if (self.phase < 0).all():
#                 self.phase += 1.0

        self.wv = np.mean(self.wave, axis=0)     
        

    def get_plot_cst(self) :   
        return  [(self.vrp-self.vr), self.berv, self.Kp, self.planet.RV_sys, \
                 self.nu, self.planet.w]
    
    def build_trans_spec(self, flux=None, params=None, n_comps=None, 
                         change_ratio=False, change_noise=False, ratio_recon=False, 
                         clip_ts=None, clip_ratio=None, fast=False, poly_time=None, counting = True, **kwargs):
        
        """
        Compute the transmission/emission spectrum of the planet
        """
        
        if params is None:
            params=[0.2, 0.97, 51, 41, 5, 2, 5.0, 5.0, 5.0, 5.0]
        if flux is None:
            flux=self.flux
        if n_comps is None:
            self.n_comps = self.n_spec-2
        else:
            self.n_comps = n_comps
 
        noise = self.noise
        self.fl_norm, self.fl_norm_mo, self.mast_out, \
        self.spec_trans, self.full_ts, self.ts_norm, \
        self.final, self.rebuilt, \
        self.pca, self.fl_Sref, self.fl_masked, \
        ratio, last_mask, self.recon_time = ts.build_trans_spectrum4(self.wave, flux,
                                     self.berv, self.planet.RV_sys, self.vr, self.iOut,
                                     path=self.path, tellu=self.tellu, noise=noise,
                                    lim_mask=params[0], lim_buffer=params[1],
                                    mo_box=params[2], mo_gauss_box=params[4],
                                    n_pca=params[5],
                                    tresh=params[6], tresh_lim=params[7],
                                    last_tresh=params[8], last_tresh_lim=params[9],
                                    n_comps=self.n_comps,
                                    clip_ts=clip_ts, clip_ratio=clip_ratio,
                                    poly_time=poly_time, counting = counting, **kwargs)
        
#         self.n_comps = n_comps
#         self.reconstructed = (self.blaze/np.nanmax(self.blaze, axis=-1)[:,:,None] * \
#                               np.ma.median(self.fl_masked,axis=-1)[:,:,None] * \
#                               self.mast_out[None, :, :] * self.ratio * self.rebuilt).squeeze()
        if (not hasattr(self, 'ratio')) or (change_ratio is True):
            self.ratio = ratio
        if not hasattr(self, 'last_mask'):
            self.last_mask = last_mask
    
        self.ratio_recon = ratio_recon
        if fast is False:
            self.reconstructed = (np.ma.median(flux,axis=-1)[:,:,None] * \
                              self.mast_out[None, :, :] * self.rebuilt).squeeze()
        else:
            self.reconstructed = self.rebuilt
            self.ratio_recon = False
        if ratio_recon is True:
            self.reconstructed *= self.ratio
        if poly_time is not None:
#             self.recon_time = recon_time
            self.reconstructed *= self.recon_time
            self.ratio *= self.recon_time

        self.params = params
        self.clip_ts = clip_ts
        self.clip_ratio = clip_ratio
        self.N = (~np.isnan(self.final)).sum(axis=-1)
        
        self.N_frac = np.nanmean(self.N/self.N0, axis=0).data #4088
        self.N_frac[np.isnan(self.N_frac)] = 0
        
        self.N_frac_f = np.nanmean(self.N/self.N0f, axis=0).data #4088
        self.N_frac_f[np.isnan(self.N_frac_f)] = 0
        

        if (self.noise is None) or (change_noise is True):
            print('Calculating noise with {} PCs'.format(params[5]))
            self.sig_col = np.ma.std(self.final, axis=0)[None,:,:]  #self.final  # self.spec_trans
            self.noise = self.sig_col*self.scaling
        
        
    def norv_sequence(self, RV=None):
        
        if RV is None:
            self.RV_sys = self.planet.RV_sys.value.copy()
        else:
            self.RV_sys = RV
            
        self.berv = -self.berv0
        self.mid_id = int(np.ceil(self.n_spec/2)-1)
        self.mid_berv = self.berv[self.mid_id]
        self.mid_vr = self.vr[self.mid_id].value
        self.mid_vrp = self.vrp[self.mid_id].value

        self.berv = (self.berv-self.berv[self.mid_id])
        self.vr = (self.vr-self.vr[self.mid_id]).to(u.km / u.s).value
        self.vrp = (self.vrp-self.vrp[self.mid_id]).to(u.km / u.s).value
        self.planet.RV_sys=0*u.km/u.s

        self.RV_const = self.mid_berv+self.mid_vr+self.RV_sys


#         self.build_trans_spec(**kwargs)
    
    
    def norv_split_sequence(self, tb1, tb2, RV=None):
#         t1 = obs.select_transit(transit_tag1)
#         t1.calc_sequence(K=K, coeffs=[0.5802,-0.1496],ld_model='quadratic')
        if RV is None:
            self.RV_sys = self.planet.RV_sys.value.copy()
            tb1.RV_sys = tb1.planet.RV_sys.value.copy()
            tb2.RV_sys = tb2.planet.RV_sys.value.copy()
        else:
            self.RV_sys = RV
            tb1.RV_sys = RV
            tb2.RV_sys = RV
            
        self.berv = -self.berv
        self.mid_id = int(np.ceil(self.n_spec/2)-1)
        self.mid_berv = self.berv[self.mid_id]
        self.planet.RV_sys=0*u.km/u.s
        self.mid_vr = self.vr[self.mid_id].value
        self.mid_vrp = self.vrp[self.mid_id].value
        self.RV_const = self.mid_berv + self.mid_vr + self.RV_sys

        tb1.berv = -tb1.berv
        tb1.berv = (tb1.berv-self.mid_berv)
        tb1.vr = (tb1.vr-self.mid_vr).to(u.km / u.s).value
        tb1.vrp = (tb1.vrp-self.mid_vrp).to(u.km / u.s).value
        tb1.planet.RV_sys=0*u.km/u.s
        tb1.RV_const = self.mid_berv + self.mid_vr + self.RV_sys
        tb1.mid_berv = self.mid_berv
        tb1.mid_vrp = self.mid_vrp
        tb1.mid_vr = self.mid_vr
        
#         tb1.build_trans_spec(**kwargs1, **kwargs)

        tb2.berv = -tb2.berv
        tb2.berv = (tb2.berv-self.mid_berv)
        tb2.vr = (tb2.vr-self.mid_vr).to(u.km / u.s).value
        tb2.vrp = (tb2.vrp-self.mid_vrp).to(u.km / u.s).value
        tb2.planet.RV_sys=0*u.km/u.s
        tb2.RV_const = self.mid_berv + self.mid_vr + self.RV_sys
        tb2.mid_berv = self.mid_berv
        tb2.mid_vrp = self.mid_vrp
        tb2.mid_vr = self.mid_vr
        
#         tb2.build_trans_spec(**kwargs2, **kwargs)

        self.berv = (self.berv-self.berv[self.mid_id])
        self.vr = (self.vr-self.vr[self.mid_id]).to(u.km / u.s).value
        self.vrp = (self.vrp-self.vrp[self.mid_id]).to(u.km / u.s).value


        
    def inject_signal(self, mod_x, mod_y, dv_pl=None, dv_star=0, RV=0, flux=None, noise=False, alpha=None, **kwargs):
        if flux is None:
            flux = self.rebuilt
        if dv_pl is None:
            dv_pl = self.vrp
#         if dv_star is None:
#             dv_star = self.berv + self.vr + self.RV_sys
        if alpha is None:
            alpha = self.alpha

#         self.flux_inj, self.inj_mod = spectrum.quick_inject(self.wave, flux, mod_x, mod_y, 
#                                                  dv_pl+RV, self.sep, 
#                                                  self.planet.R_star, self.planet.A_star, 
#                                                  R0 = self.planet.R_pl, alpha=alpha, **kwargs)

        self.flux_inj, self.inj_mod = spectrum.quick_inject_clean(self.wave, flux, mod_x, mod_y, 
                                                 dv_pl, self.sep, self.planet.R_star, self.planet.A_star, 
                                                                  RV=RV, dv_star=dv_star, 
                                                 R0 = self.planet.R_pl, alpha=alpha, **kwargs)

        if noise is True:
            self.flux_inj += fake_noise(self.spec_trans)
            self.flux_inj = np.ma.masked_invalid(self.flux_inj)

    
    def calc_correl(self, corrRV, mod_x, mod_y, get_corr=True, get_logl=True, 
                    kind='BL', somme=False, sfsg=False, binning=False, counting = True):
        print("Trans spec reduction params :  ", self.params) 

        correl = np.ma.zeros((self.n_spec, self.nord, corrRV.size))
#         correl0 = np.ma.zeros((self.n_spec, self.nord, corrRV.size))
        logl = np.ma.zeros((self.n_spec, self.nord, corrRV.size))

        # - Calculate the shinft -
        shifts = hm.calc_shift(corrRV, kind='rel')

        # - Interpolate over the orginal data
#         if binning is False:
        fct = interp1d_masked(mod_x, mod_y, kind='cubic', fill_value='extrapolate')

#         if (get_logl is True) and (kind == 'OG'):
#             sig, flux_norm, s2f, cst = calc_logl_OG_cst(self.final[:, :, :, None], axis=2)

        for iOrd in range(self.nord):
            if counting:
                hm.print_static('{} / {}'.format(iOrd+1, self.nord))

            if self.final[:,iOrd].mask.all():
                continue
#             if binning is True:
#                 wv_sh_lim = np.concatenate((self.wv_ext[iOrd][0]/shifts[[0,-1]], \
#                                             self.wv_ext[iOrd][-1]/shifts[[0,-1]]))
#                 cond = (mod_x >= wv_sh_lim.min()) & (mod_x <= wv_sh_lim.max())
                
# #                 binned = binning_model(P_x[cond], P_y[cond], wv_bins[iOrd])
#                 binned, _, _ = stats.binned_statistic(mod_x[cond], mod_y[cond], 'mean', bins=self.wv_bins[iOrd])
                
#                 # - Interpolating the spirou grid to shift
#                 fct = interp1d_masked(self.wv_ext[iOrd], np.ma.masked_invalid(binned),\
#                                       kind='cubic', fill_value='extrapolate')
#     #           # - Shifting it
#                 model = fct(self.wv[iOrd][:, None] / shifts[None,:])[None,:,:] 
#             else:
            # - Evaluate it at the shifted grid
            model = fct(self.wv[iOrd][:, None] / shifts[None,:])[None,:,:]  #/ shifts[None,:]
    #             model = quick_norm(model, somme=somme, take_all=False)
            model -= model.mean(axis=1)
            if somme is True:
                model /= np.sqrt(np.ma.sum(model**2, axis=1))#[:,None,:]
            
            if get_logl is True:
                if kind == 'BL':
                    logl[:, iOrd, :] = calc_logl_BL_ord(self.final[:, iOrd, :, None], model, self.N[:,iOrd, None],axis=1)

#                 if kind == 'OG':
#                     logl[:, iOrd, :] = calc_logl_OG_ord(flux_norm[:,iOrd], model, sig[:,iOrd],
#                                                           cst[:,iOrd], s2f[:,iOrd], axis=1)
            if get_corr is True:
                if sfsg is False:
                    correl[:, iOrd, :] = np.ma.sum(self.final_std[:, iOrd, :, None] * model, axis=1)
                else:
                    R = np.ma.sum(self.final[:, iOrd, :, None] * model, axis=1) 
                    s2f = np.ma.sum(self.final[:, iOrd, :, None]**2, axis=1)
                    s2g = np.ma.sum(model**2, axis=1)

                    correl[:, iOrd, :] =  R/np.sqrt(s2f*s2g)
        
        if get_corr is True:
            self.correl = np.ma.masked_invalid(correl)
        if get_logl is True:
            self.logl = np.ma.masked_invalid(logl)
           
        
    def get_template(self, file):

        data = Table.read(self.path / Path(file))
        self.wvsol = np.ma.masked_invalid(data['wavelength']/1e3)  # [None,:]
        self.template = np.ma.masked_invalid(data['flux'])   # [None,:]

            
#     def fct_quick_correl(self, corrRV, mod_x, mod_y,  
#                      get_logl=False, flux=None, kind='BL', **kwargs):
#         wave = self.wave
        
#         if get_logl is False:
#             if flux is None:
#                 flux = self.final_std
#             self.correl = corr.quick_correl(wave, flux, corrRV, mod_x, mod_y, wave_ref=self.wv, 
#                      get_logl=False, **kwargs)
#         else:
#             if flux is None:
#                 flux = self.final
#             self.logl = corr.quick_correl(wave, flux, corrRV, mod_x, mod_y, wave_ref=self.wv, 
#                      get_logl=True, kind=kind, **kwargs)
        
            
#     def combine_spec_trans():
#         self.spec_fin, _ = ts.build_stacked_st(wave_temp, spec_trans[iIn_tag], vr[tag[iIn_tag]], vrp[tag[iIn_tag]], 
#                                   light_curve[tag[iIn_tag]])

#         self.spec_fin_out, _ = ts.build_stacked_st(wave_temp, spec_trans[iOut_tag],vr[tag[iOut_tag]],vrp[tag[iOut_tag]],
#                                               light_curve[tag[iOut_tag]])

#         self.spec_fin_Sref = np.ma.average(spec_trans[iIn_tag], axis=0, weights=light_curve[iIn_tag])
        
        
#     def calc_logl_injred(self, Kp_array, corrRV, n_pcas, modelWave0, modelTD0=None, 
#                          filenames=None, R_mod=125000, path=None):  #, div_sig=True
    
#         if filenames is None:
#             filenames = np.array([''])
        
#         if path is None:
#             path_grid_mod = "/home/boucher/spirou/planetModels/"+hm.replace_with_check(self.name, ' ', '_')+'/'
            
# #         if div_sig is True:
#         sig = np.ma.std(self.final, axis=0)[None,:,:]

#         logl_BL = np.ma.zeros((self.n_spec, self.nord, Kp_array.size, corrRV.size, len(n_pcas), filenames.size))
            
#         for n,n_pc in enumerate(n_pcas):
#             # -- Built the star+tell sequence from PCAs
#             rebuilt = ts.remove_dem_pca(self.spec_trans, n_pcs=n_pc, n_comps=10, plot=False)[1]

#             for f,file in enumerate(filenames):

#                 if filenames.size > 1: 
#                     modelTD0 = np.load(path_grid_mod + file.replace('thermal','dppm'))
#                     specMod = make_quick_model(modelWave0, modelTD0, somme=False, Rbf=R_mod,
#                                                  box=self.params[2], gauss_box=5)
#                 else:
#                     specMod = modelTD0
            
#                 for i,Kpi in enumerate(Kp_array):

#                     vrp_orb = o.rv_theo_nu(Kpi, self.nu*u.rad, self.planet.w, plnt=True).value

#                     for v,rv in enumerate(corrRV):
#                         hm.print_static('            N_pca = {}, Kp = {}, File = {}/{}, RV = {}/{}'.format(\
#                                                  n_pc, Kpi, f+1,filenames.size, v+1,corrRV.size))

#                         # -- Use that to inject the signal
#                         self.inject_signal(modelWave0,-specMod, RV=rv, dv_pl=vrp_orb+self.planet.RV_sys.value, 
#                                            flux=rebuilt, resol=70000)
#                         # -- Remove the same number of pcas that were used to inject
#                         model_seq, _ = ts.remove_dem_pca(self.flux_inj, n_pcs=n_pc, n_comps=10, plot=False)
#                         # -- calculate the correlation with the observed sequence
#                         model_seq -= np.nanmean(model_seq, axis=-1)[:,:,None]

#                         for iOrd in range(self.nord):

#                             if self.final[:,iOrd].mask.all():
#                                 continue
# #                             if div_sig is True:
# #                                 flux = self.final[:,iOrd]/sig[:,iOrd]
# #                                 mod = model_seq[:,iOrd]/sig[:,iOrd]
# #                             else:
# #                                 flux = self.final[:,iOrd]
# #                                 mod = model_seq[:,iOrd]
# #                             logl_BL[:, iOrd, i, v, n, f] = calc_logl_BL_ord(flux, mod, self.N[:,iOrd])
#                             logl_BL[:, iOrd, i, v, n, f] = calc_logl_BL_ord(self.final[:,iOrd]/sig[:,iOrd], \
#                                                                             model_seq[:,iOrd]/sig[:,iOrd], \
#                                                                             self.N[:,iOrd])
#         return logl_BL
    
    
        
#     def plot(self, *args, fig=None, ax=None, **kwargs):

#         if ax is None and fig is None:
#             fig, ax = plt.subplots(figsize=(9, 3))
#         ax.plot(self["wave"], self[self.y], *args, **kwargs)
#         return fig, ax


class Planet():
    def __init__(self, name, parametres=None, observatory='cfht', **kwargs):
        self.name = name
        
        if parametres is None:
            log.info('Getting {} from ExoFile'.format(name))
            # Try locally, if not available, try to query the exofile
            try:
                parametres = ExoFile.load(query=False, use_alt_file=True).by_pl_name(name)
            except FileNotFoundError:
                parametres = ExoFile.load(use_alt_file=True).by_pl_name(name)

        #  --- Propriétés du système
        self.R_star = parametres['st_rad'].to(u.m)
        self.M_star = parametres['st_mass'].to(u.kg)
        self.RV_sys = parametres['st_radv'].to(u.km/u.s)
        self.Teff = parametres['st_teff'].to(u.K)

        try:
            self.vsini = parametres['st_vsin'].data.data * u.m / u.s
        except AttributeError:
            self.vsini = parametres['st_vsin'].to(u.m / u.s)

        # --- Propriétés de la planète
        try:
            self.R_pl = (parametres['pl_radj'].data * const.R_jup).data * u.m
            self.M_pl = (parametres['pl_bmassj'].data * const.M_jup).data * u.kg
            self.ap = (parametres['pl_orbsmax'].data * const.au).data * u.m
        except (AttributeError, TypeError) as e:
            self.R_pl = parametres['pl_radj'].to(u.m)
            self.M_pl = parametres['pl_bmassj'].to(u.kg)
            self.ap = parametres['pl_orbsmax'].to(u.m)
        self.rho = parametres['pl_dens']  # 5.5 *u.g/u.cm**3 # --- Jupiter : 1.33 g cm-3  /// Terre : 5.5 g cm-3
        self.Tp = np.asarray(parametres['pl_eqt'], dtype=np.float64) * u.K

        # --- Paramètres d'observations
        self.observatoire = observatory
        try:
            self.radec = [parametres['ra'].data.data * u.deg,
                          parametres['dec'].data.data * u.deg]  # parametres['radec']
        except AttributeError:
            self.radec = [parametres['ra'] * u.deg, parametres['dec'] * u.deg]
        # --- Paramètres transit
        self.period = parametres['pl_orbper'].to(u.s)
        try:
            self.mid_tr = parametres['pl_tranmid'].data * u.d  # from nasa exo archive
        except (AttributeError, TypeError) as e:
            self.mid_tr = parametres['pl_tranmid'].to(u.d)
        self.trandur = (parametres['pl_trandur'] / u.d * u.h).to(u.s)

        # --- Paramètres Orbitaux
        self.excent = parametres['pl_orbeccen']
        if self.excent.mask:
            self.excent = 0.0
        self.incl = parametres['pl_orbincl'].to(u.rad)
        self.w = parametres['pl_orblper'].to(u.rad) + (3 * np.pi / 2) * u.rad
        self.omega = np.radians(0) * u.rad
        # time of periastron passage; if not available, set it to the same value as the mid transit time
        self.t_peri = parametres['pl_orbtper'].data
        if not self.t_peri:
            self.t_peri = self.mid_tr
        else:
            self.t_peri = self.t_peri * u.d

        for key in list(kwargs.keys()):
            new_value = kwargs[key]
            old_value = getattr(self,key)
            log.info('Changing {} from {} to {}'.format(key, old_value, new_value))
            if isinstance(new_value, u.Quantity) & isinstance(old_value, u.Quantity):
                new_value = new_value.to(old_value.unit)
                new_value = np.array([new_value.value])*new_value.unit
            elif isinstance(old_value, u.Quantity):
                new_value = new_value * old_value.unit
                new_value = np.array([new_value.value])*new_value.unit

            log.info('It became {}'.format(new_value))
            setattr(self, key, new_value)

        surf_grav_pl = (const.G * self.M_pl / self.R_pl**2).cgs
        self.logg_pl = np.log10(surf_grav_pl.value)
        
        # --- Paramètres de l'étoile
        self.A_star = np.pi * self.R_star**2
        surf_grav = (const.G * self.M_star / self.R_star**2).cgs
        self.logg = np.log10(surf_grav.value)
        self.gp = const.G * self.M_pl / self.R_pl**2


        # # - Paramètres atmosphériques approximatifs
        self.mu = 2.3 * const.u
        self.H = (const.k_B * self.Tp / (self.mu * self.gp)).decompose()
        self.all_params = parametres
        self.sync_equat_rot_speed = (2*np.pi*self.R_pl/self.period).to(u.km/u.s)

        
        
from astropy.io import ascii


def get_blaze_file(path, file_list='list_tellu_corrected', blaze_default=None,
                blaze_path=None, debug=False, folder='cfht_sept1'):
    blaze_path = blaze_path or path

    blaze_file_list = []
    with open(path + file_list) as f:

        for file in f:
            filename = file.split('\n')[0]
            
            if debug:
                print(filename)

            hdul = fits.open(path + filename)

            try:
                blaze_file = blaze_default or hdul[1].header['CDBBLAZE']
            except KeyError:
                blaze_file = hdul[1].header['CDBBLAZE']

            date = hdul[0].header['DATE-OBS']
            blaze_file_list.append(date+'/'+blaze_file)

    x = []
 
    for file in np.unique(blaze_file_list):
        blz = '{}'.format(folder, file)
        print(blz)
        x.append(blz)
        

    data = Table()
    data[''] = x

    ascii.write(data, path+'blaze_files', overwrite=True,comment=False)          
    print('Dont forget to remove "col0" from file')
                
    return np.unique(blaze_file_list)



 ##############################################################################   

def merge_tr(tr_merge, list_tr, merge_tr_idx, params=None, light=False):
    

    icorr_list = []
    iIn_list = []
    iOut_list = []
    
    add_n_spec = 0
    for idx, tr_i in enumerate(merge_tr_idx):
        if idx == 0:
            icorr_list.append(list_tr[str(tr_i)].icorr)
            iIn_list.append(list_tr[str(tr_i)].iIn)
            iOut_list.append(list_tr[str(tr_i)].iOut)
        else:
            add_n_spec += list_tr[str(tr_i-1)].n_spec
            icorr_list.append(list_tr[str(tr_i)].icorr + add_n_spec)
            iIn_list.append(list_tr[str(tr_i)].iIn + add_n_spec)
            iOut_list.append(list_tr[str(tr_i)].iOut + add_n_spec)
    tr_merge.icorr = np.concatenate(icorr_list)
    tr_merge.iIn = np.concatenate(iIn_list)
    tr_merge.iOut = np.concatenate(iOut_list)
    tr_merge.n_spec = np.sum([list_tr[str(tr_i)].n_spec for tr_i in merge_tr_idx])
    
    tr_merge.alpha_frac = np.concatenate([list_tr[str(tr_i)].alpha_frac for tr_i in merge_tr_idx])
    tr_merge.t_start = np.concatenate([list_tr[str(tr_i)].t_start for tr_i in merge_tr_idx])
    tr_merge.dt = np.concatenate([list_tr[str(tr_i)].dt for tr_i in merge_tr_idx])
    tr_merge.t = tr_merge.t_start*u.d
    tr_merge.phase = np.concatenate([list_tr[str(tr_i)].phase for tr_i in merge_tr_idx]) #.value
    tr_merge.noise = np.ma.concatenate([list_tr[str(tr_i)].noise for tr_i in merge_tr_idx], axis=0)

    if light is False:
        tr_merge.fl_norm = np.ma.concatenate([list_tr[str(tr_i)].fl_norm for tr_i in merge_tr_idx], axis=0)
        tr_merge.fl_Sref = np.ma.concatenate([list_tr[str(tr_i)].fl_Sref for tr_i in merge_tr_idx], axis=0)
        tr_merge.fl_masked = np.ma.concatenate([list_tr[str(tr_i)].fl_masked for tr_i in merge_tr_idx], axis=0)
        tr_merge.fl_norm_mo = np.ma.concatenate([list_tr[str(tr_i)].fl_norm_mo for tr_i in merge_tr_idx], axis=0)
        tr_merge.full_ts = np.ma.concatenate([list_tr[str(tr_i)].full_ts for tr_i in merge_tr_idx], axis=0)
        tr_merge.rebuilt = np.ma.concatenate([list_tr[str(tr_i)].rebuilt for tr_i in merge_tr_idx], axis=0)

    if list_tr[str(merge_tr_idx[0])].mast_out.ndim == 2:
        tr_merge.mast_out = np.ma.mean([np.ma.masked_invalid(list_tr[str(tr_i)].mast_out) \
                                                          for tr_i in merge_tr_idx], axis=0)
    elif list_tr[str(merge_tr_idx[0])].mast_out.ndim == 3:
        tr_merge.mast_out = np.ma.concatenate([list_tr[str(tr_i)].mast_out for tr_i in merge_tr_idx], axis=0)

    tr_merge.spec_trans = np.ma.concatenate([list_tr[str(tr_i)].spec_trans for tr_i in merge_tr_idx], axis=0)
    tr_merge.final = np.ma.concatenate([list_tr[str(tr_i)].final for tr_i in merge_tr_idx], axis=0)
    tr_merge.N = np.ma.concatenate([list_tr[str(tr_i)].N for tr_i in merge_tr_idx], axis=0)

    try:
        tr_merge.uncorr = np.ma.concatenate([list_tr[str(tr_i)].uncorr for tr_i in merge_tr_idx], axis=0)
        tr_merge.N0 = (~np.isnan(tr_merge.uncorr)).sum(axis=-1)
        tr_merge.N_frac = np.nanmean(tr_merge.N / tr_merge.N0, axis=0).data  # 4088
        tr_merge.N_frac[np.isnan(tr_merge.N_frac)] = 0
    except KeyError:
        print('Did not find Uncorr key.')
        print('Not computing N0 and N_frac.')

        # tr_merge.N_frac = np.min(np.array([list_tr[str(tr_i)].N_frac for tr_i in merge_tr_idx]),axis=0)

    tr_merge.reconstructed = np.ma.concatenate([list_tr[str(tr_i)].reconstructed for tr_i in merge_tr_idx], axis=0)
    tr_merge.ratio = np.ma.concatenate([list_tr[str(tr_i)].ratio for tr_i in merge_tr_idx], axis=0)
    if params is None:
        tr_merge.params = list_tr[str(merge_tr_idx[0])].params
    
#     return tr_merge

def merge_velocity(tr_merge, list_tr, merge_tr_idx):
    
    tr_merge.mid_vrp = np.concatenate([list_tr[str(tr_i)].mid_vrp* \
                                       np.ones((list_tr[str(tr_i)].n_spec)) for tr_i in merge_tr_idx])
    tr_merge.RV_sys = np.concatenate([list_tr[str(tr_i)].RV_sys* \
                                       np.ones((list_tr[str(tr_i)].n_spec)) for tr_i in merge_tr_idx])
    tr_merge.mid_berv = np.concatenate([list_tr[str(tr_i)].mid_berv* \
                                       np.ones((list_tr[str(tr_i)].n_spec)) for tr_i in merge_tr_idx])
    tr_merge.mid_vr = np.concatenate([list_tr[str(tr_i)].mid_vr* \
                                       np.ones((list_tr[str(tr_i)].n_spec)) for tr_i in merge_tr_idx])
    tr_merge.berv = np.concatenate([list_tr[str(tr_i)].berv for tr_i in merge_tr_idx])
    tr_merge.vrp = np.concatenate([list_tr[str(tr_i)].vrp for tr_i in merge_tr_idx])
    tr_merge.vr = np.concatenate([list_tr[str(tr_i)].vr for tr_i in merge_tr_idx])
    tr_merge.RV_const = np.concatenate([list_tr[str(tr_i)].RV_const* \
                                       np.ones((list_tr[str(tr_i)].n_spec)) for tr_i in merge_tr_idx])
    tr_merge.Kp = list_tr[str(merge_tr_idx[0])].Kp


def split_transits(obs_obj, transit_tag, mid_idx, 
                   params0=[0.85, 0.97, 51, 41, 3, 1, 2.0, 1.0, 3.0, 1.0],
                   params=None, K=None, plot=False, tr=None, fix_master_out=None, 
                   kwargs1 = {}, kwargs2 = {}, **kwargs):
    
#     if tr is None:
#         tr = obs_obj.select_transit(transit_tag)
#         tr.calc_sequence(plot=plot, K=K)
#         tr.build_trans_spec(params=params0, **kwargs)
#         tr.build_trans_spec(params=params, flux_masked=tr.fl_norm, flux_Sref=tr.fl_norm, 
#                                   flux_norm=tr.fl_norm, flux_norm_mo=tr.fl_norm_mo, master_out=tr.mast_out, 
#                                   spec_trans=tr.spec_trans, mask_var=False, **kwargs)
        
    # --- bloc1 ---
    trb1 = obs_obj.select_transit(transit_tag, bloc = np.arange(0, mid_idx))
    trb1.calc_sequence(plot=plot, K=K)
    # --- bloc2 ---
    trb2 = obs_obj.select_transit(transit_tag, bloc = np.arange(mid_idx, transit_tag.size))
    trb2.calc_sequence(plot=plot, K=K)
    
    if fix_master_out is not None:
        trb1.build_trans_spec(params=params0, master_out=fix_master_out, **kwargs, **kwargs1)
        trb2.build_trans_spec(params=params0, master_out=fix_master_out, **kwargs, **kwargs2) 
    else:
        if ((trb1.iOut.size > 0) and (trb2.iOut.size > 0)) or (kwargs.get('iOut_temp') == 'all'):
            trb1.build_trans_spec(params=params0, **kwargs1, **kwargs)
            trb2.build_trans_spec(params=params0, **kwargs2, **kwargs)
        else:
            if (trb1.iOut.size == 0) and (trb2.iOut.size > 0):
                trb2.build_trans_spec(params=params0, **kwargs, **kwargs2)
                if (kwargs1.get('iOut_temp') == 'all'):
                    trb1.build_trans_spec(params=params0, **kwargs, **kwargs1)
                else:
                    trb1.build_trans_spec(params=params0, master_out=trb2.mast_out, **kwargs, **kwargs1)
            elif (trb2.iOut.size == 0) and (trb1.iOut.size > 0):
                trb1.build_trans_spec(params=params0, **kwargs, **kwargs1)
                if (kwargs2.get('iOut_temp') == 'all'):
                    trb2.build_trans_spec(params=params0, **kwargs, **kwargs2)
                else:
                    trb2.build_trans_spec(params=params0, master_out=trb1.mast_out, **kwargs, **kwargs2)

    tr_new = obs_obj.select_transit(transit_tag)
    tr_new.calc_sequence(plot=plot, K=K)
    tr_new.build_trans_spec(params=params0, **kwargs)
    
    if params is not None:
#         if (trb1.iOut.size > 0) and (trb2.iOut.size > 0):
        trb1.build_trans_spec(params=params, flux_masked=trb1.fl_norm, flux_Sref=trb1.fl_norm, 
                              flux_norm=trb1.fl_norm, flux_norm_mo=trb1.fl_norm_mo, master_out=trb1.mast_out, 
                              spec_trans=trb1.spec_trans, mask_var=False, **kwargs, **kwargs1)
        trb2.build_trans_spec(params=params, flux_masked=trb2.fl_norm, flux_Sref=trb2.fl_norm, 
                              flux_norm=trb2.fl_norm, flux_norm_mo=trb2.fl_norm_mo, master_out=trb2.mast_out, 
                              spec_trans=trb2.spec_trans, mask_var=False, **kwargs, **kwargs2)
#         elif (trb1.iOut.size == 0) and (trb2.iOut.size > 0):
#             trb2.build_trans_spec(params=params, flux_masked=trb2.fl_norm, flux_Sref=trb2.fl_norm, 
#                                   flux_norm=trb2.fl_norm, flux_norm_mo=trb2.fl_norm_mo, master_out=trb2.mast_out, 
#                                   spec_trans=trb2.spec_trans, mask_var=False, **kwargs, **kwargs2)
#             trb1.build_trans_spec(params=params, flux_masked=trb1.fl_norm, flux_Sref=trb1.fl_norm, 
#                                   flux_norm=trb1.fl_norm, flux_norm_mo=trb1.fl_norm_mo, master_out=trb2.mast_out, 
#                                   spec_trans=trb1.spec_trans, mask_var=False,**kwargs, **kwargs1)
#         elif (trb2.iOut.size == 0) and (trb1.iOut.size > 0):
#             trb1.build_trans_spec(params=params, flux_masked=trb1.fl_norm, flux_Sref=trb1.fl_norm, 
#                                   flux_norm=trb1.fl_norm, flux_norm_mo=trb1.fl_norm_mo, master_out=trb1.mast_out, 
#                                   spec_trans=trb1.spec_trans, mask_var=False, **kwargs, **kwargs1)
#             trb2.build_trans_spec(params=params, flux_masked=trb2.fl_norm, flux_Sref=trb2.fl_norm, 
#                                   flux_norm=trb2.fl_norm, flux_norm_mo=trb2.fl_norm_mo, master_out=trb1.mast_out, 
#                                   spec_trans=trb2.spec_trans, mask_var=False,**kwargs, **kwargs2)

    merge_tr(trb1,trb2, tr_new, params=params)
    
    return tr, trb1, trb2, tr_new



def save_single_sequences(filename, tr, path='',
                          save_all=False, filename_end='', bad_indexs=None):

    filename = Path(filename)
    path = Path(path)
    out_filename = Path(f'{filename.name}_data_trs_{filename_end}.npz')
    out_filename = path / out_filename

    if bad_indexs is None:
        bad_indexs = []
    print(out_filename)
    if save_all is False:
        np.savez(out_filename,
             components_ = tr.pca.components_,
             explained_variance_ = tr.pca.explained_variance_,
             explained_variance_ratio_ = tr.pca.explained_variance_ratio_,
             singular_values_ = tr.pca.singular_values_,
             mean_ = tr.pca.mean_,
             n_components_ = tr.pca.n_components_,
             n_features_ = tr.pca.n_features_,
             n_samples_ = tr.pca.n_samples_,
             noise_variance_ = tr.pca.noise_variance_,
             n_features_in_ = tr.pca.n_features_in_,
             RV_const = tr.RV_const,
             params = tr.params,
             wave = tr.wave,
             # vrp = tr.vrp,
             sep = tr.sep,
             noise = tr.noise,
             N = tr.N,
             t_start = tr.t_start,
             dt=tr.dt.value,
             flux = tr.flux,
             uncorr = tr.uncorr,
             blaze = tr.blaze,
             tellu = tr.tellu,
             # s2f = np.ma.sum((tr.final/tr.noise)**2, axis=-1),
             mask_flux = (tr.flux).mask,
             mask_uncorr = (tr.uncorr).mask,
             mask_blaze = (tr.blaze).mask,
             mask_tellu = (tr.tellu).mask,
             mask_noise = (tr.noise).mask,
             # mask_s2f = (np.ma.sum((tr.final/tr.noise)**2, axis=-1)).mask,
             mask_N = (tr.N).mask, 
             ratio = tr.ratio,
             reconstructed = tr.reconstructed,
             mast_out = tr.mast_out,
             mask_ratio = (tr.ratio).mask,
             mask_reconstructed = (tr.reconstructed).mask,
             mask_mast_out = (tr.mast_out).mask,
             spec_trans = tr.spec_trans,
             final = tr.final,
             mask_spec_trans = tr.spec_trans.mask,
             mask_final = tr.final.mask,
                 # alpha_frac = tr.alpha_frac,
             filenames=tr.filenames,
             icorr = tr.icorr,
             bad_indexs = bad_indexs,
             clip_ts = tr.clip_ts,
             scaling = tr.scaling,
             phase = tr.phase,
             SNR = tr.SNR,
             nu = tr.nu,
             berv0=tr.berv0,
             AM=tr.AM,
             RV_sys = tr.RV_sys,
             kind_trans = tr.kind_trans,
             coeffs = tr.coeffs,
             ld_model = tr.ld_model,
                 # iIn = tr.iIn,
                 # iOut = tr.iOut,
             )
    else:
        np.savez(out_filename,
             components_ = tr.pca.components_,
             explained_variance_ = tr.pca.explained_variance_,
             explained_variance_ratio_ = tr.pca.explained_variance_ratio_,
             singular_values_ = tr.pca.singular_values_,
             mean_ = tr.pca.mean_,
             n_components_ = tr.pca.n_components_,
             n_features_ = tr.pca.n_features_,
             n_samples_ = tr.pca.n_samples_,
             noise_variance_ = tr.pca.noise_variance_,
             n_features_in_ = tr.pca.n_features_in_,
             RV_const = tr.RV_const,
             params = tr.params,
             wave = tr.wave,
             # vrp = tr.vrp,
             sep = tr.sep,
             noise = tr.noise,
             N = tr.N,
             t_start = tr.t_start,
             dt=tr.dt.value,
             flux = tr.flux,
             uncorr = tr.uncorr,
             blaze = tr.blaze,
             tellu = tr.tellu,
             # s2f = np.ma.sum((tr.final/tr.noise)**2, axis=-1),
             mask_flux = (tr.flux).mask,
             mask_uncorr = (tr.uncorr).mask,
             mask_blaze = (tr.blaze).mask,
             mask_tellu = (tr.tellu).mask,
             mask_noise = (tr.noise).mask,
             # mask_s2f = (np.ma.sum((tr.final/tr.noise)**2, axis=-1)).mask,
             mask_N = (tr.N).mask, 
             ratio = tr.ratio,
             reconstructed = tr.reconstructed,
             mast_out = tr.mast_out,
             mask_ratio = (tr.ratio).mask,
             mask_reconstructed = (tr.reconstructed).mask,
             mask_mast_out = (tr.mast_out).mask, 
             spec_trans = tr.spec_trans,
             final = tr.final,
             mask_spec_trans = tr.spec_trans.mask,
             mask_final = tr.final.mask,
                 # alpha_frac = tr.alpha_frac,
             filenames = tr.filenames,
             icorr = tr.icorr,
             bad_indexs = bad_indexs,
             clip_ts = tr.clip_ts,
             scaling = tr.scaling,
             phase = tr.phase,
             berv0=tr.berv0,
             AM=tr.AM,
             RV_sys=tr.RV_sys,
             kind_trans = tr.kind_trans,
             coeffs = tr.coeffs,
             ld_model = tr.ld_model,
                 # iIn = tr.iIn,
                 # iOut = tr.iOut,
             SNR = tr.SNR,
             nu = tr.nu,
             fl_norm = tr.fl_norm,
             fl_norm_mo = tr.fl_norm_mo,
             full_ts = tr.full_ts,
             ts_norm = tr.ts_norm,
             rebuilt = tr.rebuilt,
             fl_Sref = tr.fl_Sref,
             fl_masked = tr.fl_masked,
             recon_time = tr.recon_time,
             mask_fl_norm = tr.fl_norm.mask,
             mask_fl_norm_mo = tr.fl_norm_mo.mask,
             mask_full_ts = tr.full_ts.mask,
             mask_ts_norm = tr.ts_norm.mask,
             mask_rebuilt = tr.rebuilt.mask,
             mask_fl_Sref = tr.fl_Sref.mask,
             mask_fl_masked = tr.fl_masked.mask,
             mask_recon_time = tr.recon_time.mask,
             )



        
# def load_fast_correl(path, filename, load_all=False, filename_end='', data_trs=None):
#
#     if data_trs is None:
#         data_trs = {}
#     #     flux = []
#
#     data_trs[filename_end] = {}
#
#     data_tr = np.load(path+filename+'_data_trs_'+filename_end+'.npz')
#
#     data_trs[filename_end]['RV_const'] = data_tr['RV_const']
#     data_trs[filename_end]['params'] = data_tr['params']
#     data_trs[filename_end]['vrp'] = data_tr['vrp']*u.km/u.s
#     data_trs[filename_end]['sep'] = data_tr['sep']*u.m
#     data_trs[filename_end]['noise'] = np.ma.array(data_tr['noise'], mask=data_tr['mask_noise'])
#     data_trs[filename_end]['N'] = np.ma.array(data_tr['N'], mask=data_tr['mask_N'])
#     data_trs[filename_end]['t_start'] = data_tr['t_start']
#     data_trs[filename_end]['alpha_frac'] = data_tr['alpha_frac']
#     data_trs[filename_end]['icorr'] = data_tr['icorr']
#     data_trs[filename_end]['clip_ts'] = data_tr['clip_ts']
#
#
#     return data_trs


def load_single_sequences(filename, name, path='',
                          load_all=False, filename_end='', plot=True, **kwargs):
    #     data_trs[filename_end] = {}

    filename = Path(filename)
    path = Path(path)

    try:
        data_tr = np.load(path / filename)
    except FileNotFoundError:
        input_filename = Path(f'{filename.name}_data_trs_{filename_end}.npz')
        input_filename = path / input_filename
        data_tr = np.load(input_filename)

    pca = PCA(data_tr['n_components_'])
    pca.components_ = data_tr['components_']
    pca.explained_variance_ = data_tr['explained_variance_']
    pca.explained_variance_ratio_ = data_tr['explained_variance_ratio_']
    pca.singular_values_ = data_tr['singular_values_']
    pca.mean_ = data_tr['mean_']
    pca.n_components_ = data_tr['n_components_']
    # pca.n_features_ = data_tr['n_features_']
    pca.n_samples_ = data_tr['n_samples_']
    pca.noise_variance_ = data_tr['noise_variance_']
    pca.n_features_in_ = data_tr['n_features_in_']

    tr = Observations(
        wave=data_tr['wave'],
        #                             count=self.count[transit_tag],
        #                             blaze=self.blaze[transit_tag],
        #                             tellu=self.tellu[transit_tag],
        #                             uncorr=self.uncorr[transit_tag],
        name=name,
        **kwargs
        # planet=planet,
        #                             path=self.path,
        #         filenames=np.array(self.filenames)[transit_tag],
        #                             filenames_uncorr=np.array(self.filenames_uncorr)[transit_tag],
        #                             CADC=self.CADC,
        #         headers_image=new_headers_im, headers_tellu=new_headers_tl
    )

    tr.wv = np.mean(tr.wave, axis=0)
    tr.pca = pca
    tr.RV_const = data_tr['RV_const']
    tr.params = list(data_tr['params'])
    for i_param in range(2,6):
        tr.params[i_param] = int(tr.params[i_param])
    #     tr.vrp = data_tr['vrp']
    tr.sep = data_tr['sep'] * u.m
    tr.noise = np.ma.array(data_tr['noise'], mask=data_tr['mask_noise'])
    tr.N = np.ma.array(data_tr['N'], mask=data_tr['mask_N'])

    tr.t_start = data_tr['t_start']
    tr.t = data_tr['t_start'] * u.d
    try:
        tr.dt = data_tr['dt']*u.s
    except KeyError:
        print('Did not have dt, using the delta_time instead.')
        tr.dt = (np.diff(tr.t_start)*u.d).to(u.s)-28*u.s
    tr.bad = data_tr['bad_indexs']
    tr.flux = np.ma.array(data_tr['flux'],
                     mask=data_tr['mask_flux'])
    #     tr.s2f = np.ma.array(data_tr['s2f'],
    #                     mask=data_tr['mask_s2f'])

    tr.ratio = np.ma.array(data_tr['ratio'],
                           mask=data_tr['mask_ratio'])
    tr.ratio_recon = True
    tr.reconstructed = np.ma.array(data_tr['reconstructed'],
                                   mask=data_tr['mask_reconstructed'])
    try:
        tr.uncorr = np.ma.array(data_tr['uncorr'],
                                   mask=data_tr['mask_uncorr'])
        tr.N0 = (~np.isnan(tr.uncorr)).sum(axis=-1)
        tr.N_frac = np.nanmean(tr.N / tr.N0, axis=0).data  # 4088
        tr.N_frac[np.isnan(tr.N_frac)] = 0
    except KeyError:
        print('Did not find Uncorr key.')
        print('Not computing N0 and N_frac.')
    try:
        tr.blaze = np.ma.array(data_tr['blaze'],
                               mask=data_tr['mask_blaze'])
    except KeyError:
        print('Did not find Blaze key.')
    tr.mast_out = np.ma.array(data_tr['mast_out'],
                              mask=data_tr['mask_mast_out'])
    tr.final = np.ma.array(data_tr['final'],
                           mask=data_tr['mask_final'])
    tr.spec_trans = np.ma.array(data_tr['spec_trans'],
                                mask=data_tr['mask_spec_trans'])
    tr.tellu = np.ma.array(data_tr['tellu'],
                                mask=data_tr['mask_tellu'])
        # tr.alpha_frac = data_tr['alpha_frac']
    try:
        tr.filenames = data_tr['filenames']
    except KeyError:
        print('Did not find Filenames key.')

    tr.clip_ts = data_tr['clip_ts']
    tr.scaling = data_tr['scaling']

    tr.n_spec, tr.nord, tr.npix = tr.final.shape
    tr.phase = data_tr['phase']

    tr.icorr = data_tr['icorr']
    
        # tr.iIn = data_tr['iIn']
        # tr.iOut = data_tr['iOut']

    tr.AM = data_tr['AM']
    tr.berv0 = data_tr['berv0']
    tr.berv = data_tr['berv0']
    tr.SNR = data_tr['SNR']
    tr.nu = data_tr['nu']

    if load_all:
        tr.fl_norm = np.ma.array(data_tr['fl_norm'],
                                 mask=data_tr['mask_fl_norm'])
        tr.fl_norm_mo = np.ma.array(data_tr['fl_norm_mo'],
                                    mask=data_tr['mask_fl_norm_mo'])
        tr.full_ts = np.ma.array(data_tr['full_ts'],
                                 mask=data_tr['mask_full_ts'])
        tr.ts_norm = np.ma.array(data_tr['ts_norm'],
                                 mask=data_tr['mask_ts_norm'])
        tr.rebuilt = np.ma.array(data_tr['rebuilt'],
                                 mask=data_tr['mask_rebuilt'])
        tr.fl_Sref = np.ma.array(data_tr['fl_Sref'],
                                 mask=data_tr['mask_fl_Sref'])
        tr.fl_masked = np.ma.array(data_tr['fl_masked'],
                                   mask=data_tr['mask_fl_masked'])
        tr.recon_time = np.ma.array(data_tr['recon_time'],
                                    mask=data_tr['mask_recon_time'])

    # ---- Transit model
    gen_transit_model(tr, tr.planet, data_tr['kind_trans'], data_tr['coeffs'], data_tr['ld_model'], plot=plot)

    # --- Radial velocities
    gen_rv_sequence(tr, tr.planet, plot=False)

    tr.norv_sequence(RV=data_tr['RV_sys'])

    return tr



def load_single_data_dict(path, filename, load_all=False, filename_end='', data_trs=None):
    
    if data_trs is None:
        data_trs = {}
    #     flux = []

    data_trs[filename_end] = {}

    data_tr = np.load(path+filename+'_data_trs_'+filename_end+'.npz')

    pca=PCA(data_tr['n_components_'])
    pca.components_ = data_tr['components_']
    pca.explained_variance_ = data_tr['explained_variance_']
    pca.explained_variance_ratio_ = data_tr['explained_variance_ratio_']
    pca.singular_values_ = data_tr['singular_values_']
    pca.mean_ = data_tr['mean_']
    pca.n_components_ = data_tr['n_components_']
    pca.n_features_ = data_tr['n_features_']
    pca.n_samples_ = data_tr['n_samples_']
    pca.noise_variance_ = data_tr['noise_variance_']
    pca.n_features_in_ = data_tr['n_features_in_']

    data_trs[filename_end]['pca'] = pca
    data_trs[filename_end]['RV_const'] = data_tr['RV_const']
    data_trs[filename_end]['params'] = data_tr['params']
    data_trs[filename_end]['wave'] = data_tr['wave']
    data_trs[filename_end]['vrp'] = data_tr['vrp']*u.km/u.s
    data_trs[filename_end]['sep'] = data_tr['sep']*u.m
    data_trs[filename_end]['noise'] = np.ma.array(data_tr['noise'], mask=data_tr['mask_noise'])
    data_trs[filename_end]['N'] = np.ma.array(data_tr['N'], mask=data_tr['mask_N'])
    data_trs[filename_end]['t_start'] = data_tr['t_start']
    data_trs[filename_end]['flux'] = np.ma.array(data_tr['flux'], 
                                              mask=data_tr['mask_flux'])
    data_trs[filename_end]['s2f'] = np.ma.array(data_tr['s2f'], 
                                             mask=data_tr['mask_s2f'])
    data_trs[filename_end]['ratio'] = np.ma.array(data_tr['ratio'], 
                                               mask=data_tr['mask_ratio'])
    data_trs[filename_end]['reconstructed'] = np.ma.array(data_tr['reconstructed'], 
                                                       mask=data_tr['mask_reconstructed'])
    data_trs[filename_end]['mast_out'] = np.ma.array(data_tr['mast_out'], 
                                                  mask=data_tr['mask_mast_out'])
    data_trs[filename_end]['final'] = np.ma.array(data_tr['final'], 
                                                  mask=data_tr['mask_final'])
    data_trs[filename_end]['spec_trans'] = np.ma.array(data_tr['spec_trans'], 
                                                  mask=data_tr['mask_spec_trans'])
    data_trs[filename_end]['tellu'] = np.ma.array(data_tr['tellu'],
                                                       mask=data_tr['mask_tellu'])
    data_trs[filename_end]['alpha_frac'] = data_tr['alpha_frac']
    data_trs[filename_end]['icorr'] = data_tr['icorr']
    data_trs[filename_end]['clip_ts'] = data_tr['clip_ts']
    data_trs[filename_end]['scaling'] = data_tr['scaling']

    data_trs[filename_end]['phase'] = data_tr['phase']
    data_trs[filename_end]['iIn'] = data_tr['iIn']
    data_trs[filename_end]['iOut'] = data_tr['iOut']

    if load_all:
        data_trs[filename_end]['fl_norm'] = np.ma.array(data_tr['fl_norm'], 
                                                  mask=data_tr['mask_fl_norm'])
        data_trs[filename_end]['fl_norm_mo'] = np.ma.array(data_tr['fl_norm_mo'], 
                                                  mask=data_tr['mask_fl_norm_mo'])
        data_trs[filename_end]['full_ts'] = np.ma.array(data_tr['full_ts'], 
                                                  mask=data_tr['mask_full_ts'])
        data_trs[filename_end]['ts_norm'] = np.ma.array(data_tr['ts_norm'], 
                                                  mask=data_tr['mask_ts_norm'])
        data_trs[filename_end]['rebuilt'] = np.ma.array(data_tr['rebuilt'], 
                                                  mask=data_tr['mask_rebuilt'])
        data_trs[filename_end]['fl_Sref'] = np.ma.array(data_tr['fl_Sref'], 
                                                  mask=data_tr['mask_fl_Sref'])
        data_trs[filename_end]['fl_masked'] = np.ma.array(data_tr['fl_masked'], 
                                                  mask=data_tr['mask_fl_masked'])
        data_trs[filename_end]['recon_time'] = np.ma.array(data_tr['recon_time'], 
                                                  mask=data_tr['mask_recon_time'])
        
    return data_trs
    

def save_sequences(filename, list_tr, do_tr, path='', bad_indexs=None, save_all=False):

    filename = Path(filename)
    path = Path(path)

    if bad_indexs is None:
        bad_indexs = []

    out_filename = Path(f'{filename.name}_data_info.npz')
    print(path / out_filename)
    np.savez(path / out_filename,
             trall_alpha_frac = list_tr[str(do_tr[-1])].alpha_frac,
             trall_icorr = list_tr[str(do_tr[-1])].icorr,
             trall_N = list_tr[str(do_tr[-1])].N  ,
             bad_indexs = bad_indexs
             )

    for i_tr, tr_key in enumerate(list(list_tr.keys())[:np.nonzero(np.array(do_tr) < 10)[0].size]):
        out_filename = Path(f'{filename.name}_data_trs_{i_tr}.npz')
        print(path / out_filename)
        if save_all is False:
            np.savez(path / out_filename,
                 components_ = list_tr[tr_key].pca.components_,
                 explained_variance_ = list_tr[tr_key].pca.explained_variance_,
                 explained_variance_ratio_ = list_tr[tr_key].pca.explained_variance_ratio_,
                 singular_values_ = list_tr[tr_key].pca.singular_values_,
                 mean_ = list_tr[tr_key].pca.mean_,
                 n_components_ = list_tr[tr_key].pca.n_components_,
                 n_features_ = list_tr[tr_key].pca.n_features_,
                 n_samples_ = list_tr[tr_key].pca.n_samples_,
                 noise_variance_ = list_tr[tr_key].pca.noise_variance_,
                 n_features_in_ = list_tr[tr_key].pca.n_features_in_,
                 RV_const = list_tr[tr_key].RV_const,
                 params = list_tr[tr_key].params,
                 wave = list_tr[tr_key].wave,
                 vrp = list_tr[tr_key].vrp,
                 sep = list_tr[tr_key].sep,
                 noise = list_tr[tr_key].noise,
                 N = list_tr[tr_key].N,
                 t_start = list_tr[tr_key].t_start, #.value,
                 flux = list_tr[tr_key].final/list_tr[tr_key].noise,
                 s2f = np.ma.sum((list_tr[tr_key].final/list_tr[tr_key].noise)**2, axis=-1),
                 mask_flux = (list_tr[tr_key].final/list_tr[tr_key].noise).mask, 
                 mask_noise = (list_tr[tr_key].noise).mask,
                 mask_s2f = (np.ma.sum((list_tr[tr_key].final/list_tr[tr_key].noise)**2, axis=-1)).mask,
                 mask_N = (list_tr[tr_key].N).mask, 
                 ratio = list_tr[tr_key].ratio,
                 reconstructed = list_tr[tr_key].reconstructed,
                 mast_out = list_tr[tr_key].mast_out,
                 mask_ratio = (list_tr[tr_key].ratio).mask,
                 mask_reconstructed = (list_tr[tr_key].reconstructed).mask,
                 mask_mast_out = (list_tr[tr_key].mast_out).mask,
                 # spec_trans = list_tr[tr_key].spec_trans,
                 # final=list_tr[tr_key].final,
                 # mask_spec_trans=list_tr[tr_key].spec_trans.mask,
                 # mask_final=list_tr[tr_key].final.mask,
                 alpha_frac=list_tr[tr_key].alpha_frac,
                 icorr=list_tr[tr_key].icorr,
                 bad_indexs=bad_indexs,
                 final = list_tr[tr_key].final,
                 # clip_ts=list_tr[tr_key].clip_ts,
                 # scaling=list_tr[tr_key].scaling,
                 )
        else:
            np.savez(path / out_filename,
                 components_ = list_tr[tr_key].pca.components_,
                 explained_variance_ = list_tr[tr_key].pca.explained_variance_,
                 explained_variance_ratio_ = list_tr[tr_key].pca.explained_variance_ratio_,
                 singular_values_ = list_tr[tr_key].pca.singular_values_,
                 mean_ = list_tr[tr_key].pca.mean_,
                 n_components_ = list_tr[tr_key].pca.n_components_,
                 n_features_ = list_tr[tr_key].pca.n_features_,
                 n_samples_ = list_tr[tr_key].pca.n_samples_,
                 noise_variance_ = list_tr[tr_key].pca.noise_variance_,
                 n_features_in_ = list_tr[tr_key].pca.n_features_in_,
                 RV_const = list_tr[tr_key].RV_const,
                 params = list_tr[tr_key].params,
                 wave = list_tr[tr_key].wave,
                 vrp = list_tr[tr_key].vrp,
                 sep = list_tr[tr_key].sep,
                 noise = list_tr[tr_key].noise,
                 N = list_tr[tr_key].N,
                 t_start = list_tr[tr_key].t_start, #.value,
                 flux = list_tr[tr_key].final/list_tr[tr_key].noise,
                 s2f = np.ma.sum((list_tr[tr_key].final/list_tr[tr_key].noise)**2, axis=-1),
                 mask_flux = (list_tr[tr_key].final/list_tr[tr_key].noise).mask, 
                 mask_noise = (list_tr[tr_key].noise).mask,
                 mask_s2f = (np.ma.sum((list_tr[tr_key].final/list_tr[tr_key].noise)**2, axis=-1)).mask,
                 mask_N = (list_tr[tr_key].N).mask, 
                 ratio = list_tr[tr_key].ratio,
                 reconstructed = list_tr[tr_key].reconstructed,
                 mast_out = list_tr[tr_key].mast_out,
                 mask_ratio = (list_tr[tr_key].ratio).mask,
                 mask_reconstructed = (list_tr[tr_key].reconstructed).mask,
                 mask_mast_out = (list_tr[tr_key].mast_out).mask, 
                 spec_trans = list_tr[tr_key].spec_trans,
                 final = list_tr[tr_key].final,
                 mask_spec_trans = list_tr[tr_key].spec_trans.mask,
                 mask_final = list_tr[tr_key].final.mask,
             alpha_frac = list_tr[tr_key].alpha_frac,
             icorr = list_tr[tr_key].icorr,
             clip_ts = list_tr[tr_key].clip_ts,
             scaling = list_tr[tr_key].scaling,
                 fl_norm = list_tr[tr_key].fl_norm,
                 fl_norm_mo = list_tr[tr_key].fl_norm_mo,
                 full_ts = list_tr[tr_key].full_ts,
                 ts_norm = list_tr[tr_key].ts_norm,
                 rebuilt = list_tr[tr_key].rebuilt,
                 fl_Sref = list_tr[tr_key].fl_Sref,
                 fl_masked = list_tr[tr_key].fl_masked,
                 recon_time = list_tr[tr_key].recon_time,
                 mask_fl_norm = list_tr[tr_key].fl_norm.mask,
                 mask_fl_norm_mo = list_tr[tr_key].fl_norm_mo.mask,
                 mask_full_ts = list_tr[tr_key].full_ts.mask,
                 mask_ts_norm = list_tr[tr_key].ts_norm.mask,
                 mask_rebuilt = list_tr[tr_key].rebuilt.mask,
                 mask_fl_Sref = list_tr[tr_key].fl_Sref.mask,
                 mask_fl_masked = list_tr[tr_key].fl_masked.mask,
                 mask_recon_time = list_tr[tr_key].recon_time.mask,
                 bad_indexs=bad_indexs
                 )

        
def load_sequences(filename, do_tr, path='', load_all=False):

    filename = Path(filename)
    path = Path(path)

    if len(do_tr) > 1 :
        out_filename = Path(f'{filename.name}_data_info.npz')
        log.info(f'Reading: {path / out_filename}')
        data_info_file = np.load(path / out_filename)
        data_info = {}

        data_info['trall_alpha_frac'] = data_info_file['trall_alpha_frac']
        data_info['trall_icorr'] = data_info_file['trall_icorr']
        data_info['trall_N'] = data_info_file['trall_N']
        data_info['bad_indexs'] = data_info_file['bad_indexs']


    data_trs = {}
    #     flux = []

    for i_tr, tr_key in enumerate(do_tr[:np.nonzero(np.array(do_tr) < 10)[0].size]):
        data_trs[str(i_tr)] = {}

        out_filename = Path(f'{filename.name}_data_trs_{i_tr}.npz')
        log.info(f'Reading: {path / out_filename}')
        data_tr = np.load(path / out_filename)

        if len(do_tr) <= 1:
            data_info = {}
            try:
                data_info['trall_alpha_frac'] = data_tr['alpha_frac']
                data_info['trall_icorr'] = data_tr['icorr']
                data_info['trall_N'] = data_tr['N']
                data_info['bad_indexs'] = data_tr['bad_indexs']
            except KeyError:
                out_filename = Path(f'{filename.name}_data_info.npz')
                log.info(f'Reading: {path / out_filename}')
                data_info_file = np.load(path / out_filename)
                data_info = {}

                data_info['trall_alpha_frac'] = data_info_file['trall_alpha_frac']
                data_info['trall_icorr'] = data_info_file['trall_icorr']
                data_info['trall_N'] = data_info_file['trall_N']
                data_info['bad_indexs'] = data_info_file['bad_indexs']

        pca=PCA(data_tr['n_components_'])
        pca.components_ = data_tr['components_']
        pca.explained_variance_ = data_tr['explained_variance_']
        pca.explained_variance_ratio_ = data_tr['explained_variance_ratio_']
        pca.singular_values_ = data_tr['singular_values_']
        pca.mean_ = data_tr['mean_']
        pca.n_components_ = data_tr['n_components_']
        # pca.n_features_ = data_tr['n_features_']
        pca.n_samples_ = data_tr['n_samples_']
        pca.noise_variance_ = data_tr['noise_variance_']
        pca.n_features_in_ = data_tr['n_features_in_']

        data_trs[str(i_tr)]['pca'] = pca
        data_trs[str(i_tr)]['RV_const'] = data_tr['RV_const']
        data_trs[str(i_tr)]['params'] = data_tr['params']
        data_trs[str(i_tr)]['wave'] = data_tr['wave']
        data_trs[str(i_tr)]['vrp'] = data_tr['vrp']*u.km/u.s
        data_trs[str(i_tr)]['sep'] = data_tr['sep']*u.m
        data_trs[str(i_tr)]['noise'] = np.ma.array(data_tr['noise'], mask=data_tr['mask_noise'])
        data_trs[str(i_tr)]['N'] = np.ma.array(data_tr['N'], mask=data_tr['mask_N'])
        data_trs[str(i_tr)]['t_start'] = data_tr['t_start']
        data_trs[str(i_tr)]['flux'] = np.ma.array(data_tr['flux'], 
                                                  mask=data_tr['mask_flux'])
        data_trs[str(i_tr)]['s2f'] = np.ma.array(data_tr['s2f'], 
                                                 mask=data_tr['mask_s2f'])
        data_trs[str(i_tr)]['ratio'] = np.ma.array(data_tr['ratio'], 
                                                   mask=data_tr['mask_ratio'])
        data_trs[str(i_tr)]['reconstructed'] = np.ma.array(data_tr['reconstructed'], 
                                                           mask=data_tr['mask_reconstructed'])
        data_trs[str(i_tr)]['mast_out'] = np.ma.array(data_tr['mast_out'], 
                                                      mask=data_tr['mask_mast_out'])

        if load_all:
            data_trs[str(i_tr)]['final'] = np.ma.array(data_tr['final'],
                                                       mask=data_tr['mask_final'])
            data_trs[str(i_tr)]['spec_trans'] = np.ma.array(data_tr['spec_trans'],
                                                            mask=data_tr['mask_spec_trans'])
            data_trs[str(i_tr)]['alpha_frac'] = data_tr['alpha_frac']
            data_trs[str(i_tr)]['icorr'] = data_tr['icorr']
            data_trs[str(i_tr)]['clip_ts'] = data_tr['clip_ts']
            data_trs[str(i_tr)]['scaling'] = data_tr['scaling']

            data_trs[str(i_tr)]['fl_norm'] = np.ma.array(data_tr['fl_norm'], 
                                                      mask=data_tr['mask_fl_norm'])
            data_trs[str(i_tr)]['fl_norm_mo'] = np.ma.array(data_tr['fl_norm_mo'], 
                                                      mask=data_tr['mask_fl_norm_mo'])
            data_trs[str(i_tr)]['full_ts'] = np.ma.array(data_tr['full_ts'], 
                                                      mask=data_tr['mask_full_ts'])
            data_trs[str(i_tr)]['ts_norm'] = np.ma.array(data_tr['ts_norm'], 
                                                      mask=data_tr['mask_ts_norm'])
            data_trs[str(i_tr)]['rebuilt'] = np.ma.array(data_tr['rebuilt'], 
                                                      mask=data_tr['mask_rebuilt'])
            data_trs[str(i_tr)]['fl_Sref'] = np.ma.array(data_tr['fl_Sref'], 
                                                      mask=data_tr['mask_fl_Sref'])
            data_trs[str(i_tr)]['fl_masked'] = np.ma.array(data_tr['fl_masked'], 
                                                      mask=data_tr['mask_fl_masked'])
            data_trs[str(i_tr)]['recon_time'] = np.ma.array(data_tr['recon_time'], 
                                                      mask=data_tr['mask_recon_time'])
        
    return data_info, data_trs



def gen_obs_sequence(obs, transit_tag, params_all, iOut_temp,
                     coeffs, ld_model, kind_trans, RV_sys, polynome=None, 
                     ratio_recon=False, cont=False, cbp=True, noise_npc=None, counting = True, **kwargs_build_ts):
    if transit_tag is not None:
        tr = obs.select_transit(transit_tag)
    else:
        tr = obs
    tr.calc_sequence(plot=False,  coeffs=coeffs, ld_model=ld_model, kind_trans=kind_trans)
    tr.norv_sequence(RV=RV_sys)

    if polynome is not None:
    #                 print("P(O-2) = ", polynome[tag-1])
        if polynome:
            poly_time = tr.t_start  # .value  # tr.AM
        else:
            poly_time = None
    else:
        poly_time = None
        
    if noise_npc is None:
        tr.build_trans_spec(params= params_all, \
                    iOut_temp=iOut_temp, ratio_recon=ratio_recon, cont=cont, 
                        cbp=cbp, poly_time=poly_time, counting = counting, **kwargs_build_ts)
    else:
       
        params_copy = params_all.copy()
        params_copy[5] = noise_npc
#         tr.build_trans_spec(params= params_all, \
#                     iOut_temp=iOut_temp, ratio_recon=ratio_recon, cont=cont, 
#                         cbp=cbp, poly_time=poly_time, **kwargs_build_ts)
        tr.build_trans_spec(params= params_copy, \
                        iOut_temp=iOut_temp, ratio_recon=ratio_recon, cont=cont, 
                        cbp=cbp, poly_time=poly_time, **kwargs_build_ts)
        tr.build_trans_spec(params= params_all, \
                         iOut_temp=iOut_temp, ratio_recon=ratio_recon, cont=cont, 
                        cbp=False, poly_time=poly_time, 
                       flux_masked=tr.fl_masked, flux_Sref=tr.fl_Sref, flux_norm=tr.fl_norm, 
                        flux_norm_mo=tr.fl_norm_mo, master_out=tr.mast_out, spec_trans=tr.spec_trans, 
                            mask_var=False, **kwargs_build_ts)
        
        
    return tr

def gen_merge_obs_sequence(obs, list_tr, merge_tr_idx, transit_tags, coeffs, ld_model, kind_trans, light=False):

    if transit_tags is not None:
        tr_merge = obs.select_transit(np.concatenate([transit_tags[tr_i-1] for tr_i in merge_tr_idx]))
    else:
        tr_merge = deepcopy(obs)

    if light is False:
        tr_merge.calc_sequence(plot=False,  coeffs=coeffs, ld_model=ld_model, kind_trans=kind_trans)
    # else:
    #     tr_merge.dt =


    merge_tr(tr_merge, list_tr, merge_tr_idx, light=light)
    merge_velocity(tr_merge, list_tr, merge_tr_idx)
    
    return tr_merge


def generate_all_transits(obs, transit_tags, RV_sys, params_all, iOut_temp,
                          do_tr=[1,2,3,12,123], cbp=True, 
                           kind_trans='transmission', flux_all=None,
                          ld_model = 'linear', coeffs=[0.53],
                           polynome=None, noise_npc=None, counting = True, **kwargs_build_ts):
    #                           
    
    ratio_recon=True
    cont=False

    list_tr = OrderedDict({})
    
    for tag in do_tr:
        name_tag = str(tag)
        if len(name_tag) < 2:
            if flux_all is not None:
                kwargs_build_ts['flux'] = flux_all[tag-1]

            list_tr[name_tag] = gen_obs_sequence(obs, transit_tags[tag-1], params_all[tag-1], 
                                                 iOut_temp[tag-1],
                                                 coeffs, ld_model, kind_trans, RV_sys[tag-1], 
                                                 polynome=polynome[tag-1], noise_npc=noise_npc, 
                                 ratio_recon=ratio_recon, cont=cont, cbp=cbp, counting = counting, **kwargs_build_ts)
        
        else :  

            merge_tr_idx = [int(tag_i) for tag_i in name_tag]

            list_tr[name_tag] = gen_merge_obs_sequence(obs, list_tr, merge_tr_idx, transit_tags,
                                               coeffs, ld_model, kind_trans)

    return list_tr



### --- Telluric custom masking


def mask_custom_pclean_ord(tr, flux, pclean, ccf_pclean, corrRV0,
                           thresh=None, plot=False, pad_to=None,
                           snr_floor=None,
                           masking_spectra=None, correl_spectra=None,
                           kind='tellu', counting = True):
    new_mask_pclean = np.empty_like(pclean)
    ccf_pclean_new = ccf_pclean.copy()  # np.empty_like(ccf_pclean)
    if kind == 'tellu':
        add_to_mask = 0.025
    elif kind == 'sky':
        add_to_mask = 0.05
    print('add_to_mask', add_to_mask)
    if pad_to is None:
        if kind == 'tellu':
            pad_to = 0.97
        elif kind == 'sky':
            pad_to = 0.999
    print('pad_to', pad_to)
    if thresh is None:
        if kind == 'tellu':
            thresh = 1.9
        elif kind == 'sky':
            thresh = 2.0
    print('thresh', thresh)

    if snr_floor is None:
        if kind == 'tellu':
            snr_floor = 0.9
        elif kind == 'sky':
            snr_floor = 1.0
    print('snr_floor', snr_floor)

    if masking_spectra is None:
        masking_spectra = pclean
    if correl_spectra is None:
        correl_spectra = pclean

    for iOrd in range(tr.nord):
        #     print(iOrd)
        limit_mask = tr.params[0]
        #     flux_ord = flux[:,iOrd,None,:]

        _, rv_snr, _, snr_i, _ = calc_snr_1d(np.abs(ccf_pclean[:, iOrd]),
                                             corrRV0, np.zeros_like(tr.vrp), RV_sys=0.0)

        #         param, pcov = curve_fit(gauss, ydata=snr_i, xdata=rv_snr, p0=[5,3,0])
        #         print('sig = {}, amp = {}, x0 = {}'.format(*param))

        fct_snr = interp1d(rv_snr, snr_i)
        snr_i_0 = np.max(snr_i[100 - 3:100 + 3 + 1])  # fct_snr(0.0)

        if counting:
            hm.print_static(iOrd, snr_i_0, np.ma.max(snr_i), limit_mask)

        if plot:
            fig, axs = plt.subplots(2, 2, figsize=(6, 7), sharex=True)
            axs[0, 0].pcolormesh(corrRV0, tr.phase, np.abs(ccf_pclean[:, iOrd]))
            axs[0, 0].set_title(str(iOrd))
            axs[1, 0].plot(rv_snr, snr_i)
            axs[1, 0].axvline(0.0)
            axs[1, 0].axhline(snr_i_0)
        #             axs[1,0].plot(rv_snr, gauss(rv_snr,*param))

        last_snr_i = snr_i_0.copy()
        #         print("last",last_snr_i)

        new_mask = flux[:, iOrd].mask

        if (snr_i_0 > thresh):
            print(snr_i_0, snr_floor, limit_mask, pad_to)
            while (snr_i_0 > snr_floor) and (limit_mask < pad_to):

                limit_mask += add_to_mask
                if counting:
                    hm.print_static(iOrd, snr_i_0, np.ma.max(snr_i), limit_mask)

                new_mask = [get_mask_tell(np.ma.masked_invalid(tell),
                                          limit_mask, pad_to) for tell in masking_spectra[:, iOrd, :]]
                new_mask = new_mask | flux[:, iOrd].mask
                flux_ord = np.ma.array(flux[:, iOrd], mask=new_mask)[:, None]

                ccf_pclean_ord = quick_correl_3dmod(tr.wave[:, iOrd, None],
                                                         flux_ord,
                                                         corrRV0,
                                                         tr.wave[:, iOrd, None],
                                                         correl_spectra[:, iOrd, None])

                _, rv_snr, _, snr_i, _ = calc_snr_1d(np.abs(ccf_pclean_ord).squeeze(),
                                                     corrRV0, np.zeros_like(tr.vrp), RV_sys=0.0)
                ccf_pclean_new[:, iOrd] = ccf_pclean_ord.squeeze()

                fct_snr = interp1d(rv_snr, snr_i)
                snr_i_0 = np.max(snr_i[100 - 3:100 + 3 + 1])  # fct_snr(0.0)

                if kind == 'tellu':
                    if snr_i_0 == last_snr_i:
                        #                     print("new_snr == last",snr_i_0, last_snr_i, 'add 0.05')
                        if snr_i_0 >= 5:
                            add_to_mask = 0.1
                        else:
                            add_to_mask = 0.05
                    else:
                        #                     print("new_snr != last",snr_i_0, last_snr_i, 'add 0.025')
                        if snr_i_0 >= 5:
                            add_to_mask = 0.05
                        else:
                            add_to_mask = 0.025
                elif kind == 'sky':
                    if snr_i_0 == last_snr_i:
                        #                     print("new_snr == last",snr_i_0, last_snr_i, 'add 0.05')
                        if limit_mask + 0.1 < 0.97:
                            add_to_mask = 0.1
                        else:
                            add_to_mask = 0.05
                        if limit_mask >= 0.95:
                            add_to_mask = 0.002
                        if limit_mask >= 0.98:
                            add_to_mask = 0.001
                    else:
                        #                     print("new_snr != last",snr_i_0, last_snr_i, 'add 0.025')
                        if snr_i_0 >= 3:
                            add_to_mask = 0.05
                        else:
                            add_to_mask = 0.025
                        if limit_mask >= 0.95:
                            add_to_mask = 0.002
                        if limit_mask >= 0.98:
                            add_to_mask = 0.001

                last_snr_i = snr_i_0

        if kind == 'tellu':
            new_mask = [get_mask_tell(tell, limit_mask + 0.025, pad_to) for tell in tr.pclean[:, iOrd, :]]
            new_mask = new_mask | flux[:, iOrd].mask
        #         flux_ord = np.ma.array(flux[:,iOrd], mask=new_mask)[:,None]
        print(iOrd, snr_i_0, np.ma.max(snr_i), limit_mask)

        if plot:
            axs[0, 1].pcolormesh(corrRV0, tr.phase, np.abs(ccf_pclean_new[:, iOrd]))
            axs[1, 1].plot(rv_snr, snr_i)
            axs[1, 1].axvline(0.0)
            axs[1, 1].axhline(snr_i_0)

        #         pltr.figure()
        #         pltr.pcolormesh(corrRV0, tr.phase, np.abs(ccf_pclean_ord).squeeze())

        new_mask_pclean[:, iOrd, :] = new_mask | flux[:, iOrd].mask

    new_flux = np.ma.array(flux, mask=new_mask_pclean)

    return new_mask_pclean, new_flux, ccf_pclean_new


# for tr in [t1,t2,t3]:
def mask_tellu_sky(tr, corrRV0, pad_to=0.99, plot_clean=False, fig_output_file=None, counting = True):
    if not (hasattr(tr, 'pclean') | hasattr(tr, 'sky')):
        sky = []
        tellu = []
        #     with open(path + file_list) as f:

        for file in tr.filenames:
            blocks = file.split('_')
            filename = '_'.join(blocks[0:2]) + '_tellu_pclean_' + blocks[-1]
            filename = Path(filename)

            print(filename)

            sky_model = fits.getdata(tr.path / filename, ext=4)
            tell_model = fits.getdata(tr.path / filename, ext=3)

            sky.append(np.ma.masked_invalid(sky_model))
            tellu.append(np.ma.masked_invalid(tell_model))

        tr.sky = np.ma.masked_invalid(sky)
        tr.pclean = np.ma.masked_invalid(tellu)

    sky_t = np.ma.masked_invalid(tr.sky)
    skynorm = sky_t / np.ma.max(sky_t)
    skydown = 1 - skynorm
    plt.figure()
    plt.plot(skydown[0, 34])

    spec_trans_tr = (tr.uncorr / tr.pclean) / tr.blaze / tr.reconstructed

    sky_tr = spec_trans_tr - np.ma.median(spec_trans_tr, axis=-1)[:, :, None]
    sky_tr = sky_tr / np.ma.max(sky_tr)
    sky_tr = 1 - sky_tr
    tile_sky_tr = np.tile(np.ma.mean(sky_tr, axis=0), (tr.n_spec, 1, 1))

    params = tr.params
    flux_mask = ts.build_trans_spectrum4(tr.wave, spec_trans_tr,
                                         tr.berv, tr.planet.RV_sys, tr.vr, tr.iOut,
                                         tellu=tr.tellu, noise=tr.noise,
                                         lim_mask=params[0], lim_buffer=params[1],
                                         mo_box=params[2], mo_gauss_box=params[4],
                                         n_pca=params[5],
                                         tresh=params[6], tresh_lim=params[7],
                                         last_tresh=params[8], last_tresh_lim=params[9],
                                         n_comps=tr.n_spec-2,
                                         clip_ts=None, clip_ratio=None,
                                         iOut_temp='all', cont=False,
                                         cbp=True, poly_time=None,
                                         flux_masked=spec_trans_tr,
                                         flux_Sref=spec_trans_tr, flux_norm=spec_trans_tr,
                                         flux_norm_mo=spec_trans_tr, master_out=spec_trans_tr,
                                         spec_trans=spec_trans_tr,
                                         mask_var=False)[6]

    #     flux_mask = tr.final.copy()
    new_mask = [get_mask_noise(f, 4, 3, gwidth=0.01, poly_ord=5) for f in flux_mask.swapaxes(0, 1)]
    new_mask = new_mask | flux_mask.mask
    flux_mask = np.ma.array(flux_mask, mask=new_mask)

    # --- Tellu masking ---

    ccf_tellu_tr = quick_correl_3dmod(tr.wave, flux_mask, corrRV0, tr.wave, tr.pclean)

    tellu_mask, cmasked_flux_tr, ccf_tellu_clean = mask_custom_pclean_ord(tr, flux_mask, tr.pclean,
                                                                          ccf_tellu_tr, corrRV0, plot=False, counting = counting)

    if plot_clean:
        ccf_tellu_clean = quick_correl_3dmod(tr.wave, cmasked_flux_tr,
                                                  corrRV0, tr.wave, tr.pclean)
        _ = plot_all_orders_correl(corrRV0, np.abs(ccf_tellu_clean), tr,
                                      icorr=None, logl=False, sharey=True,
                                      vrp=np.zeros_like(tr.vrp), RV_sys=-7.0, vmin=None, vmax=None,
                                      vline=None, hline=2, kind='snr', return_snr=True, output_file=fig_output_file)

        # --- Sky masking ---

    ccf_sky_tr = quick_correl_3dmod(tr.wave, cmasked_flux_tr, corrRV0,
                                         tr.wave, skydown)

    sky_mask, cmasked_flux_sky, ccf_sky_clean = mask_custom_pclean_ord(tr, cmasked_flux_tr, sky_tr,
                                                                       ccf_sky_tr, corrRV0, kind='sky',
                                                                       thresh=2.0, plot=False, pad_to=pad_to,
                                                                       masking_spectra=skydown, correl_spectra=skydown)

    if plot_clean:
        ccf_sky_clean = quick_correl_3dmod(tr.wave, cmasked_flux_sky,
                                                corrRV0, tr.wave, skydown)
        _ = plot_all_orders_correl(corrRV0, np.abs(ccf_sky_clean), tr,
                                      icorr=None, logl=False, sharey=True,
                                      vrp=np.zeros_like(tr.vrp), RV_sys=-7.0, vmin=None, vmax=None,
                                      vline=None, hline=2, kind='snr', return_snr=True, output_file=fig_output_file)

    if not hasattr(tr,'original_mask'):
        tr.original_mask = tr.spec_trans.mask.copy()

    tr.OG_final_mask = tr.final.mask.copy()
    tr.OG_spec_trans_mask = tr.spec_trans.mask.copy()
    tr.final = cmasked_flux_sky
    tr.custom_mask = cmasked_flux_sky.mask
    tr.spec_trans.mask = cmasked_flux_sky.mask

    del sky_mask, ccf_sky_clean, ccf_sky_tr, ccf_tellu_tr, tellu_mask, cmasked_flux_tr, ccf_tellu_clean, cmasked_flux_sky
    del flux_mask, new_mask, sky_t, skynorm, spec_trans_tr, sky_tr, tile_sky_tr
    gc.collect()
