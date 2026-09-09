
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


# Everything about instrument/DRS identity and how to read raw files for each one lives in
# instruments.py (Chantier B, B2 follow-up) -- header keywords, file patterns, physical
# properties, and the read_all_sp_* functions themselves, each attached to its own
# instruments_drs entry there. This module only ever needs the populated dict.
from .instruments import instruments_drs, register_instrument

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


def _unpack_read_sp_result(result):
    """Backward/forward-compatible unpacking of a `read_all_sp_*` reader's return value
    (Chantier B, B2 follow-up): either the standard `(headers, wave, count, blaze,
    filenames)`, or `(headers, wave, count, blaze, filenames, recon)` for a reader that can
    also report an embedded telluric reconstruction spectrum (`recon`, `None` when this
    particular read didn't have one) -- see `read_all_sp_nirps_apero`. Always returns the
    6-tuple form so callers (`Observations.fetch_data`) don't need to care which kind of
    reader they're calling."""
    if len(result) == 6:
        return result
    return (*result, None)


#######################
### Observation class
#######################

class Observations():
    
    """
    Observations class object
    Will contain all data and parameters
    Note : Probably could be optimized
    """

    # instrument=<name> selects the header-keyword/reader dictionary to use (see
    # starships.instruments.instruments_drs) -- pass the string name (e.g. 'NIRPS-APERO'),
    # not the dict itself.
    def __init__(self, wave=np.array([]), count=np.array([]), blaze=np.array([]),
                 headers = list_of_dict([]),
                 tellu=np.array([]), uncorr=np.array([]),
                 name='', path='',filenames=[], planet=None, pl_kwargs=None, instrument='SPIRou-APERO'):

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
        self.filenames=filenames
        self.n_spec = len(self.filenames)

        self.uncorr=uncorr
        self.tellu=tellu
        # get the instrument dictionary from the dict of instruments
        # the string/name
        self.instrument_name = instrument
        # the dictionary
        self.instrument = instruments_drs[instrument]


    def fetch_data(self, path, list_e2ds='list_e2ds',
                    list_tcorr='list_tellu_corrected', list_recon='list_tellu_recon',
                    read_sp=None, **kwargs):
        """
        Retrieve all the relevent data in path
        (tellu corrected, tellu recon and uncorrected spectra from lists of files)

        Which raw-file format to expect (external blaze/wave calibration files vs.
        bundled/embedded extensions, "CADC" in the old naming) is entirely determined by
        `instrument` (see `starships.planet_obs.instruments_drs`/`register_instrument`) --
        there used to be a separate `CADC=True/False` flag here with its own hardcoded
        per-instrument-name dispatch, removed once every format variant this package
        supports had its own instrument/DRS profile (Chantier B, B2 follow-up). Use
        `instrument='SPIRou-APERO-CADC'`/`'NIRPS-APERO-CADC'` for the bundled/embedded
        format instead of a boolean.
        """

        # get the appropriate function to read spectra from the instrument's dictionary
        # if read function is not specified as an argument
        if not read_sp:
            read_sp = self.instrument['read_all_sp']

        log.info("Fetching the uncorrected spectra")
        log.info(f"File: {list_e2ds}")

        headers, wave, count_uncorr, blaze_uncorr, filenames_uncorr, _ = \
            _unpack_read_sp_result(read_sp(path, list_e2ds, **kwargs))

        embedded_recon = None
        if list_tcorr is None:
            log.info('No telluric correction available')
            count = count_uncorr.copy()
            blaze = blaze_uncorr.copy()
            filenames = filenames_uncorr

        else:
            log.info('Fetching data')
            log.info(f"File: {list_tcorr}")
            headers, wave, count, blaze, filenames, embedded_recon = \
                _unpack_read_sp_result(read_sp(path, list_tcorr, **kwargs))
            #             self.filenames  = filenames

        if embedded_recon is not None:
            # Some formats bundle the telluric reconstruction spectrum as an extension of
            # the tcorr file itself (Chantier B, B2 follow-up) -- no separate list_recon
            # file needed/expected in that case, use what the tcorr read already gave us.
            log.info('Using the telluric reconstruction spectrum embedded in the tcorr file')
            tellu = embedded_recon

        elif list_recon is None:
            log.info('No reconstruction available')
            tellu = np.ones_like(count)

        else:
            log.info("Fetching the tellurics")
            log.info(f"File: {list_recon}")
            _, _, tellu, _, _, _ = _unpack_read_sp_result(read_sp(path, list_recon, **kwargs))
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

        # add instrument argument
        return Observations(headers=new_headers,
                            wave=self.wave[transit_tag],
                            count=self.count[transit_tag], blaze=self.blaze[transit_tag],
                            tellu=self.tellu[transit_tag],
                            uncorr=self.uncorr[transit_tag],
                            name=self.name, planet=self.planet ,
                            path=self.path, filenames=np.array(self.filenames)[transit_tag],
                            # filenames_uncorr=np.array(self.filenames_uncorr)[transit_tag],
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
                try:
                    # Some formats (formerly reached only via fetch_data(CADC=True), see
                    # Chantier B B2 follow-up) key the per-order SNR/EXTSN header cards
                    # differently -- try the other numbering before falling back further.
                    self.SNR = np.ma.masked_invalid([np.array(self.headers.get_all('SNR'+'{}'.format(order))[0],
                                dtype='float') for order in range(self.nord)]).T
                except KeyError:
                    try:
                        self.SNR = np.ma.masked_invalid([np.array(self.headers.get_all('EXTSN'+'{:003}'.format(order))[0],
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
        self.fl_norm, self.fl_norm_mo, self.reference_spec, \
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
#                               self.reference_spec[None, :, :] * self.ratio * self.rebuilt).squeeze()
        if (not hasattr(self, 'ratio')) or (change_ratio is True):
            self.ratio = ratio
        if not hasattr(self, 'last_mask'):
            self.last_mask = last_mask
    
        self.ratio_recon = ratio_recon
        if fast is False:
            self.reconstructed = (np.ma.median(flux,axis=-1)[:,:,None] * \
                              self.reference_spec[None, :, :] * self.rebuilt).squeeze()
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

        self.all_params = parametres
        self.apply_overrides(**kwargs)

    def apply_overrides(self, **kwargs):
        """Override planet attributes in place (e.g. `mid_tr=...`) and recompute every
        derived quantity that depends on them.

        Used both by `__init__` (config-wide `pl_kwargs`, e.g. `retrieval.py`'s `pl_params`)
        and, since B3 (Chantier B), by `load_reduced_sequence` to restore a *per-visit*
        override saved at reduction time (e.g. `mid_tr` for TTV/resonant systems, where the
        transit epoch genuinely differs from one visit to the next) on top of a shared base
        `Planet` -- see `save_reduced_sequence`'s `planet_overrides`.
        """
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

def merge_tr(tr_merge, visits, merge_tr_idx, params=None, light=False):
    

    icorr_list = []
    iIn_list = []
    iOut_list = []
    
    add_n_spec = 0
    for idx, tr_i in enumerate(merge_tr_idx):
        if idx == 0:
            icorr_list.append(visits[str(tr_i)].icorr)
            iIn_list.append(visits[str(tr_i)].iIn)
            iOut_list.append(visits[str(tr_i)].iOut)
        else:
            add_n_spec += visits[str(tr_i-1)].n_spec
            icorr_list.append(visits[str(tr_i)].icorr + add_n_spec)
            iIn_list.append(visits[str(tr_i)].iIn + add_n_spec)
            iOut_list.append(visits[str(tr_i)].iOut + add_n_spec)
    tr_merge.icorr = np.concatenate(icorr_list)
    tr_merge.iIn = np.concatenate(iIn_list)
    tr_merge.iOut = np.concatenate(iOut_list)
    tr_merge.n_spec = np.sum([visits[str(tr_i)].n_spec for tr_i in merge_tr_idx])
    
    tr_merge.alpha_frac = np.concatenate([visits[str(tr_i)].alpha_frac for tr_i in merge_tr_idx])
    tr_merge.t_start = np.concatenate([visits[str(tr_i)].t_start for tr_i in merge_tr_idx])
    tr_merge.dt = np.concatenate([visits[str(tr_i)].dt for tr_i in merge_tr_idx])
    tr_merge.t = tr_merge.t_start*u.d
    tr_merge.phase = np.concatenate([visits[str(tr_i)].phase for tr_i in merge_tr_idx]) #.value
    tr_merge.noise = np.ma.concatenate([visits[str(tr_i)].noise for tr_i in merge_tr_idx], axis=0)

    if light is False:
        tr_merge.fl_norm = np.ma.concatenate([visits[str(tr_i)].fl_norm for tr_i in merge_tr_idx], axis=0)
        tr_merge.fl_Sref = np.ma.concatenate([visits[str(tr_i)].fl_Sref for tr_i in merge_tr_idx], axis=0)
        tr_merge.fl_masked = np.ma.concatenate([visits[str(tr_i)].fl_masked for tr_i in merge_tr_idx], axis=0)
        tr_merge.fl_norm_mo = np.ma.concatenate([visits[str(tr_i)].fl_norm_mo for tr_i in merge_tr_idx], axis=0)
        tr_merge.full_ts = np.ma.concatenate([visits[str(tr_i)].full_ts for tr_i in merge_tr_idx], axis=0)
        tr_merge.rebuilt = np.ma.concatenate([visits[str(tr_i)].rebuilt for tr_i in merge_tr_idx], axis=0)

    if visits[str(merge_tr_idx[0])].reference_spec.ndim == 2:
        tr_merge.reference_spec = np.ma.mean([np.ma.masked_invalid(visits[str(tr_i)].reference_spec) \
                                                          for tr_i in merge_tr_idx], axis=0)
    elif visits[str(merge_tr_idx[0])].reference_spec.ndim == 3:
        tr_merge.reference_spec = np.ma.concatenate([visits[str(tr_i)].reference_spec for tr_i in merge_tr_idx], axis=0)

    tr_merge.spec_trans = np.ma.concatenate([visits[str(tr_i)].spec_trans for tr_i in merge_tr_idx], axis=0)
    tr_merge.final = np.ma.concatenate([visits[str(tr_i)].final for tr_i in merge_tr_idx], axis=0)
    tr_merge.N = np.ma.concatenate([visits[str(tr_i)].N for tr_i in merge_tr_idx], axis=0)

    try:
        tr_merge.uncorr = np.ma.concatenate([visits[str(tr_i)].uncorr for tr_i in merge_tr_idx], axis=0)
        tr_merge.N0 = (~np.isnan(tr_merge.uncorr)).sum(axis=-1)
        tr_merge.N_frac = np.nanmean(tr_merge.N / tr_merge.N0, axis=0).data  # 4088
        tr_merge.N_frac[np.isnan(tr_merge.N_frac)] = 0
    except KeyError:
        print('Did not find Uncorr key.')
        print('Not computing N0 and N_frac.')

        # tr_merge.N_frac = np.min(np.array([visits[str(tr_i)].N_frac for tr_i in merge_tr_idx]),axis=0)

    tr_merge.reconstructed = np.ma.concatenate([visits[str(tr_i)].reconstructed for tr_i in merge_tr_idx], axis=0)
    tr_merge.ratio = np.ma.concatenate([visits[str(tr_i)].ratio for tr_i in merge_tr_idx], axis=0)
    if params is None:
        tr_merge.params = visits[str(merge_tr_idx[0])].params
    
#     return tr_merge

def merge_velocity(tr_merge, visits, merge_tr_idx):
    
    tr_merge.mid_vrp = np.concatenate([visits[str(tr_i)].mid_vrp* \
                                       np.ones((visits[str(tr_i)].n_spec)) for tr_i in merge_tr_idx])
    tr_merge.RV_sys = np.concatenate([visits[str(tr_i)].RV_sys* \
                                       np.ones((visits[str(tr_i)].n_spec)) for tr_i in merge_tr_idx])
    tr_merge.mid_berv = np.concatenate([visits[str(tr_i)].mid_berv* \
                                       np.ones((visits[str(tr_i)].n_spec)) for tr_i in merge_tr_idx])
    tr_merge.mid_vr = np.concatenate([visits[str(tr_i)].mid_vr* \
                                       np.ones((visits[str(tr_i)].n_spec)) for tr_i in merge_tr_idx])
    tr_merge.berv = np.concatenate([visits[str(tr_i)].berv for tr_i in merge_tr_idx])
    tr_merge.vrp = np.concatenate([visits[str(tr_i)].vrp for tr_i in merge_tr_idx])
    tr_merge.vr = np.concatenate([visits[str(tr_i)].vr for tr_i in merge_tr_idx])
    tr_merge.RV_const = np.concatenate([visits[str(tr_i)].RV_const* \
                                       np.ones((visits[str(tr_i)].n_spec)) for tr_i in merge_tr_idx])
    tr_merge.Kp = visits[str(merge_tr_idx[0])].Kp


def split_transits(obs_obj, transit_tag, mid_idx, 
                   params0=[0.85, 0.97, 51, 41, 3, 1, 2.0, 1.0, 3.0, 1.0],
                   params=None, K=None, plot=False, visit=None, fix_reference_spec=None, 
                   kwargs1 = {}, kwargs2 = {}, **kwargs):
    
#     if visit is None:
#         visit = obs_obj.select_transit(transit_tag)
#         visit.calc_sequence(plot=plot, K=K)
#         visit.build_trans_spec(params=params0, **kwargs)
#         visit.build_trans_spec(params=params, flux_masked=visit.fl_norm, flux_Sref=visit.fl_norm, 
#                                   flux_norm=visit.fl_norm, flux_norm_mo=visit.fl_norm_mo, reference_spec=visit.reference_spec, 
#                                   spec_trans=visit.spec_trans, mask_var=False, **kwargs)
        
    # --- bloc1 ---
    trb1 = obs_obj.select_transit(transit_tag, bloc = np.arange(0, mid_idx))
    trb1.calc_sequence(plot=plot, K=K)
    # --- bloc2 ---
    trb2 = obs_obj.select_transit(transit_tag, bloc = np.arange(mid_idx, transit_tag.size))
    trb2.calc_sequence(plot=plot, K=K)
    
    if fix_reference_spec is not None:
        trb1.build_trans_spec(params=params0, reference_spec=fix_reference_spec, **kwargs, **kwargs1)
        trb2.build_trans_spec(params=params0, reference_spec=fix_reference_spec, **kwargs, **kwargs2) 
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
                    trb1.build_trans_spec(params=params0, reference_spec=trb2.reference_spec, **kwargs, **kwargs1)
            elif (trb2.iOut.size == 0) and (trb1.iOut.size > 0):
                trb1.build_trans_spec(params=params0, **kwargs, **kwargs1)
                if (kwargs2.get('iOut_temp') == 'all'):
                    trb2.build_trans_spec(params=params0, **kwargs, **kwargs2)
                else:
                    trb2.build_trans_spec(params=params0, reference_spec=trb1.reference_spec, **kwargs, **kwargs2)

    tr_new = obs_obj.select_transit(transit_tag)
    tr_new.calc_sequence(plot=plot, K=K)
    tr_new.build_trans_spec(params=params0, **kwargs)
    
    if params is not None:
#         if (trb1.iOut.size > 0) and (trb2.iOut.size > 0):
        trb1.build_trans_spec(params=params, flux_masked=trb1.fl_norm, flux_Sref=trb1.fl_norm, 
                              flux_norm=trb1.fl_norm, flux_norm_mo=trb1.fl_norm_mo, reference_spec=trb1.reference_spec, 
                              spec_trans=trb1.spec_trans, mask_var=False, **kwargs, **kwargs1)
        trb2.build_trans_spec(params=params, flux_masked=trb2.fl_norm, flux_Sref=trb2.fl_norm, 
                              flux_norm=trb2.fl_norm, flux_norm_mo=trb2.fl_norm_mo, reference_spec=trb2.reference_spec, 
                              spec_trans=trb2.spec_trans, mask_var=False, **kwargs, **kwargs2)
#         elif (trb1.iOut.size == 0) and (trb2.iOut.size > 0):
#             trb2.build_trans_spec(params=params, flux_masked=trb2.fl_norm, flux_Sref=trb2.fl_norm, 
#                                   flux_norm=trb2.fl_norm, flux_norm_mo=trb2.fl_norm_mo, reference_spec=trb2.reference_spec, 
#                                   spec_trans=trb2.spec_trans, mask_var=False, **kwargs, **kwargs2)
#             trb1.build_trans_spec(params=params, flux_masked=trb1.fl_norm, flux_Sref=trb1.fl_norm, 
#                                   flux_norm=trb1.fl_norm, flux_norm_mo=trb1.fl_norm_mo, reference_spec=trb2.reference_spec, 
#                                   spec_trans=trb1.spec_trans, mask_var=False,**kwargs, **kwargs1)
#         elif (trb2.iOut.size == 0) and (trb1.iOut.size > 0):
#             trb1.build_trans_spec(params=params, flux_masked=trb1.fl_norm, flux_Sref=trb1.fl_norm, 
#                                   flux_norm=trb1.fl_norm, flux_norm_mo=trb1.fl_norm_mo, reference_spec=trb1.reference_spec, 
#                                   spec_trans=trb1.spec_trans, mask_var=False, **kwargs, **kwargs1)
#             trb2.build_trans_spec(params=params, flux_masked=trb2.fl_norm, flux_Sref=trb2.fl_norm, 
#                                   flux_norm=trb2.fl_norm, flux_norm_mo=trb2.fl_norm_mo, reference_spec=trb1.reference_spec, 
#                                   spec_trans=trb2.spec_trans, mask_var=False,**kwargs, **kwargs2)

    merge_tr(trb1,trb2, tr_new, params=params)
    
    return visit, trb1, trb2, tr_new



def save_reduced_sequence(filename, visit, path='', filename_end='', bad_indexs=None):
    """Save one reduced visit to a single ``.npz`` file (B3, Chantier B).

    Only saves what is independent of `n_pc` -- the fitted PCA (`visit.pca`, fit once on
    `spec_trans` at reduction time) and every product upstream of the PCA truncation step
    (`spec_trans`, `fl_Sref`, `fl_masked`, `fl_norm`, `fl_norm_mo`, `reference_spec`/reference
    spectrum, `recon_time`) -- plus `noise`, itself now fixed at a `noise_npc` independent of
    the science `n_pc` (see `gen_obs_sequence`). Everything that depends on `n_pc`
    (`final`/`clean_ts`/`ts_norm`/`rebuilt`/`reconstructed`/`N`) is deliberately *not* saved:
    `load_reduced_sequence` recomputes it cheaply for whatever `n_pc` is requested at read
    time, from `spec_trans` and the already-fitted `pca` (`apply_pca_truncation`, no refit).

    This replaces the old `save_single_sequences`/`save_sequences` pair (one "diagnostic"
    file with every intermediate + one "light retrieval" file without them) -- since nothing
    n_pc-dependent is saved anymore, that distinction no longer applies: there is only one
    file per visit now.

    Also saves any per-visit planet parameter override actually used at reduction time
    (`visit.planet.reduction_overrides`, set by `pipeline.reduction.load_planet` -- e.g. `mid_tr`
    for a TTV/resonant system where the transit epoch genuinely differs per visit), so
    `load_reduced_sequence` can restore the same per-visit ephemeris later instead of losing
    it as soon as the file is loaded again.

    Parameters
    ----------
    filename : str or Path
        Base name for the output file (`{filename}_data_trs_{filename_end}.npz`).
    visit : Observations
        The reduced visit, after `Observations.build_trans_spec` has run.
    path : str or Path, optional
        Output directory.
    filename_end : str, optional
        Suffix inserted before `.npz`, e.g. a transit index when saving several visits
        (see `save_sequences`).
    bad_indexs : list, optional
        Exposure indices flagged as excluded for this visit. Defaults to an empty list.
    """
    filename = Path(filename)
    path = Path(path)
    out_filename = path / Path(f'{filename.name}_data_trs_{filename_end}.npz')

    if bad_indexs is None:
        bad_indexs = []

    # Per-visit planet parameter overrides actually used at reduction time (e.g. `mid_tr` for
    # a TTV/resonant system, see `pipeline.reduction.load_planet`), if any -- saved so
    # `load_reduced_sequence` can restore the *same* per-visit ephemeris later, instead of
    # this visit-specific choice being silently lost as soon as the file is loaded again.
    overrides = getattr(visit.planet, 'reduction_overrides', {}) or {}
    override_keys = np.array(list(overrides.keys()), dtype=str)
    # Not every override is a Quantity (e.g. `pl_param_units`/`convert_to_quantity` gives a
    # bare float for `unit: null` in the config, like `excent`) -- an empty unit string is the
    # sentinel for "this one was a bare scalar, re-apply it as one" (see load_reduced_sequence).
    override_values = np.array([
        float(np.asarray(q.value if isinstance(q, u.Quantity) else q).ravel()[0])
        for q in overrides.values()
    ])
    override_units = np.array(
        [str(q.unit) if isinstance(q, u.Quantity) else '' for q in overrides.values()], dtype=str)

    print(out_filename)
    np.savez(out_filename,
         planet_override_keys = override_keys,
         planet_override_values = override_values,
         planet_override_units = override_units,
         components_ = visit.pca.components_,
         explained_variance_ = visit.pca.explained_variance_,
         explained_variance_ratio_ = visit.pca.explained_variance_ratio_,
         singular_values_ = visit.pca.singular_values_,
         mean_ = visit.pca.mean_,
         n_components_ = visit.pca.n_components_,
         n_samples_ = visit.pca.n_samples_,
         noise_variance_ = visit.pca.noise_variance_,
         n_features_in_ = visit.pca.n_features_in_,
         RV_const = visit.RV_const,
         # Individual components of RV_const, kept separately for traceability
         # (RV_const = mid_berv + mid_vr + RV_sys, see Transit.norv_sequence()).
         RV_sys = visit.RV_sys,
         mid_berv = visit.mid_berv,
         mid_vr = visit.mid_vr,
         params = visit.params,
         # Fixed n_pc used to estimate `noise` (B3, `noise_npc` in `gen_obs_sequence`) --
         # saved explicitly so a reduced file is self-documenting about what its `noise`
         # corresponds to, independently of whatever `n_pc` is requested at read time.
         noise_npc = visit.noise_npc,
         wave = visit.wave,
         # `vrp`/`vr` are NOT saved here: they are purely a deterministic function of the
         # planet's ephemeris + exposure timestamps (`gen_rv_sequence`, `K=None`), recomputed
         # identically by `load_reduced_sequence` from `planet`/`t_start` -- saving them would
         # just be dead weight in the file (confirmed: the old loader never read them back
         # either, `Observations.norv_sequence` overwrites whatever was set beforehand).
         sep = visit.sep,
         noise = visit.noise,
         t_start = visit.t_start,
         dt = visit.dt.value,
         flux = visit.flux,
         uncorr = visit.uncorr,
         blaze = visit.blaze,
         tellu = visit.tellu,
         mask_flux = (visit.flux).mask,
         mask_uncorr = (visit.uncorr).mask,
         mask_blaze = (visit.blaze).mask,
         mask_tellu = (visit.tellu).mask,
         mask_noise = (visit.noise).mask,
         ratio = visit.ratio,
         reference_spec = visit.reference_spec,
         mask_ratio = (visit.ratio).mask,
         mask_reference_spec = (visit.reference_spec).mask,
         spec_trans = visit.spec_trans,
         mask_spec_trans = visit.spec_trans.mask,
         alpha_frac = visit.alpha_frac,
         filenames = visit.filenames,
         icorr = visit.icorr,
         bad_indexs = bad_indexs,
         clip_ts = visit.clip_ts,
         scaling = visit.scaling,
         phase = visit.phase,
         SNR = visit.SNR,
         nu = visit.nu,
         berv0 = visit.berv0,
         AM = visit.AM,
         kind_trans = visit.kind_trans,
         coeffs = visit.coeffs,
         ld_model = visit.ld_model,
         fl_norm = visit.fl_norm,
         fl_norm_mo = visit.fl_norm_mo,
         fl_Sref = visit.fl_Sref,
         fl_masked = visit.fl_masked,
         recon_time = visit.recon_time,
         mask_fl_norm = visit.fl_norm.mask,
         mask_fl_norm_mo = visit.fl_norm_mo.mask,
         mask_fl_Sref = visit.fl_Sref.mask,
         mask_fl_masked = visit.fl_masked.mask,
         mask_recon_time = visit.recon_time.mask,
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


def load_reduced_sequence(filename, n_pc, name='', path='', filename_end='', plot=False, **kwargs):
    """Load one visit saved by `save_reduced_sequence` and apply `n_pc` at read time (B3).

    Every n_pc-*independent* product is read straight from disk (`spec_trans`, `fl_Sref`,
    `fl_masked`, `fl_norm`, `reference_spec`/reference spectrum, the fitted `pca`, `noise` fixed at
    the file's own `noise_npc`). Everything that depends on `n_pc` (`final`, `clean_ts`,
    `ts_norm`, `rebuilt`, `N`, `reconstructed`) is then recomputed for the requested `n_pc` by
    truncating the already-fitted PCA (`Observations.build_trans_spec` reusing `pca=visit.pca`,
    same reuse pattern as `gen_obs_sequence`'s `noise_npc` branch) -- no refit, and
    numerically identical to the old behaviour of loading a separate file saved per `n_pc`
    (see `apply_pca_truncation` / `tests/unit/test_pca_truncation.py`).

    Also restores any per-visit planet parameter override saved by `save_reduced_sequence`
    (`planet_override_*`, e.g. `mid_tr` for a TTV/resonant system) on top of whichever base
    planet is used -- applied to a private copy if a shared `planet=` was passed in (so
    loading several visits with a shared planet, as `retrieval.py` does, can't leak one
    visit's override into another's), or merged into `pl_kwargs` if building a fresh one.

    Parameters
    ----------
    filename : str or Path
        Base name of the file to load (as passed to `save_reduced_sequence`).
    n_pc : int
        Number of PCA components to remove, applied at read time.
    name : str, optional
        Name for the resulting `Observations` (e.g. the planet name).
    path : str or Path, optional
        Input directory.
    filename_end : str, optional
        Suffix inserted before `.npz` (see `save_reduced_sequence`).
    plot : bool, optional
        Passed through to `gen_transit_model` (light-curve model diagnostic plot).
    **kwargs
        Passed through to the `Observations` constructor (e.g. `planet=`, `pl_kwargs=`,
        `instrument=`) -- `planet`/`pl_kwargs` are intercepted first to merge in this file's
        own saved per-visit override, if any (see above).

    Returns
    -------
    Observations
    """
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
    pca.n_samples_ = data_tr['n_samples_']
    pca.noise_variance_ = data_tr['noise_variance_']
    pca.n_features_in_ = data_tr['n_features_in_']

    # Restore any per-visit planet parameter override saved at reduction time (B3 -- TTV/
    # resonant systems, e.g. Mathis's TRAPPIST-1 retrieval, where `mid_tr` genuinely differs
    # per visit; see `save_reduced_sequence`/`pipeline.reduction.load_planet`).
    override_keys = data_tr['planet_override_keys']
    if override_keys.size:
        planet_overrides = {}
        for key, value, unit in zip(
                override_keys, data_tr['planet_override_values'], data_tr['planet_override_units']):
            unit = str(unit)
            # Empty unit = this override was a bare scalar at reduction time, not a Quantity
            # (see save_reduced_sequence) -- restore it the same way.
            planet_overrides[str(key)] = float(value) if unit == '' else float(value) * u.Unit(unit)
    else:
        planet_overrides = {}

    base_planet = kwargs.pop('planet', None)
    pl_kwargs_ctor = kwargs.pop('pl_kwargs', None)
    if base_planet is not None:
        if planet_overrides:
            # Don't mutate a shared planet object (e.g. retrieval.py's single `planet` reused
            # across every visit) -- apply this visit's override to a private copy instead.
            base_planet = deepcopy(base_planet)
            base_planet.apply_overrides(**planet_overrides)
        visit = Observations(wave=data_tr['wave'], name=name, planet=base_planet, **kwargs)
    else:
        merged_pl_kwargs = dict(pl_kwargs_ctor or {})
        merged_pl_kwargs.update(planet_overrides)
        visit = Observations(wave=data_tr['wave'], name=name,
                           pl_kwargs=merged_pl_kwargs or None, **kwargs)

    visit.wv = np.mean(visit.wave, axis=0)
    visit.pca = pca
    visit.RV_const = data_tr['RV_const']
    visit.mid_berv = data_tr['mid_berv']
    visit.mid_vr = data_tr['mid_vr']
    visit.params = list(data_tr['params'])
    for i_param in range(2, 6):
        visit.params[i_param] = int(visit.params[i_param])
    visit.params[5] = n_pc  # the read-time n_pc requested here, may differ from noise_npc below
    visit.noise_npc = int(data_tr['noise_npc'])
    # `vrp`/`vr` are not saved (see `save_reduced_sequence`) -- `gen_rv_sequence` below
    # recomputes them deterministically from `planet`/`t_start`.
    visit.sep = data_tr['sep'] * u.m
    visit.noise = np.ma.array(data_tr['noise'], mask=data_tr['mask_noise'])

    visit.t_start = data_tr['t_start']
    visit.t = data_tr['t_start'] * u.d
    visit.dt = data_tr['dt'] * u.s
    visit.bad = data_tr['bad_indexs']
    visit.flux = np.ma.array(data_tr['flux'], mask=data_tr['mask_flux'])
    visit.ratio = np.ma.array(data_tr['ratio'], mask=data_tr['mask_ratio'])
    visit.ratio_recon = True
    visit.uncorr = np.ma.array(data_tr['uncorr'], mask=data_tr['mask_uncorr'])
    visit.N0 = (~np.isnan(visit.uncorr)).sum(axis=-1)
    visit.N0f = (~np.isnan(visit.flux)).sum(axis=-1)
    visit.blaze = np.ma.array(data_tr['blaze'], mask=data_tr['mask_blaze'])
    visit.reference_spec = np.ma.array(data_tr['reference_spec'], mask=data_tr['mask_reference_spec'])
    visit.spec_trans = np.ma.array(data_tr['spec_trans'], mask=data_tr['mask_spec_trans'])
    visit.tellu = np.ma.array(data_tr['tellu'], mask=data_tr['mask_tellu'])
    visit.filenames = data_tr['filenames']

    visit.clip_ts = data_tr['clip_ts']
    visit.scaling = data_tr['scaling']

    visit.n_spec, visit.nord, visit.npix = visit.spec_trans.shape
    visit.phase = data_tr['phase']

    visit.icorr = data_tr['icorr']

    visit.AM = data_tr['AM']
    visit.berv0 = data_tr['berv0']
    visit.berv = data_tr['berv0']
    visit.SNR = data_tr['SNR']
    visit.nu = data_tr['nu']
    visit.alpha_frac = data_tr['alpha_frac']

    visit.fl_norm = np.ma.array(data_tr['fl_norm'], mask=data_tr['mask_fl_norm'])
    visit.fl_norm_mo = np.ma.array(data_tr['fl_norm_mo'], mask=data_tr['mask_fl_norm_mo'])
    visit.fl_Sref = np.ma.array(data_tr['fl_Sref'], mask=data_tr['mask_fl_Sref'])
    visit.fl_masked = np.ma.array(data_tr['fl_masked'], mask=data_tr['mask_fl_masked'])
    visit.recon_time = np.ma.array(data_tr['recon_time'], mask=data_tr['mask_recon_time'])

    # ---- Transit model
    gen_transit_model(visit, visit.planet, data_tr['kind_trans'], data_tr['coeffs'], data_tr['ld_model'], plot=plot)

    # --- Radial velocities
    gen_rv_sequence(visit, visit.planet, plot=False)

    visit.norv_sequence(RV=data_tr['RV_sys'])

    # ---- Apply the requested n_pc: cheap PCA truncation only (reuses `visit.pca`, no refit),
    # fills in final/clean_ts/ts_norm/rebuilt/N/reconstructed for this n_pc. `visit.noise` (set
    # above, fixed at `noise_npc`) is left untouched (`change_noise` defaults to False).
    # `clip_ts` must be passed explicitly here (not just stored as `visit.clip_ts`) -- it gates
    # a sigma-clip of `spec_trans` applied right before the PCA truncation
    # (`apply_pca_truncation`), so omitting it would silently skip that clip at read time.
    visit.build_trans_spec(params=visit.params, flux_masked=visit.fl_masked, flux_Sref=visit.fl_Sref,
                         flux_norm=visit.fl_norm, flux_norm_mo=visit.fl_norm_mo, reference_spec=visit.reference_spec,
                         spec_trans=visit.spec_trans, pca=visit.pca, mask_var=False, ratio_recon=True,
                         cont=False, clip_ts=float(visit.clip_ts))

    return visit




def save_sequences(filename, visits, do_tr, path='', bad_indexs=None):
    """Save one ``.npz`` file per transit in `visits` (B3: `save_reduced_sequence`).

    Companion function to `load_sequences`, which reads back the files written here.
    Multi-transit orchestration only -- what actually goes into each per-transit file is
    entirely delegated to `save_reduced_sequence` (see its docstring for what's saved: no
    n_pc-dependent product is saved anymore, so there is only one file per transit, not a
    "diagnostic" vs "light retrieval" pair as there used to be -- the separate shared
    `_data_info.npz` this function used to also write is gone too, for the same reason
    (it duplicated the last transit's `alpha_frac`/`icorr`/`N`, and `N` is n_pc-dependent
    now; `load_sequences` derives the equivalent `data_info` from the last loaded transit
    directly instead).

    Parameters
    ----------
    filename : str or Path
        Base name used to build the output file names (`{filename}_data_trs_{i}.npz`).
    visits : dict
        Transit objects to save, keyed by transit index (as a string).
    do_tr : list or array
        Transit indices to include, in the same order as `visits`. Indices >= 10
        are excluded (reserved for another use elsewhere in the pipeline).
    path : str or Path, optional
        Output directory.
    bad_indexs : list, optional
        Exposure indices to flag as bad. Defaults to an empty list.
    """
    filename = Path(filename)

    for i_tr, tr_key in enumerate(list(visits.keys())[:np.nonzero(np.array(do_tr) < 10)[0].size]):
        save_reduced_sequence(filename, visits[tr_key], path=path, filename_end=str(i_tr),
                               bad_indexs=bad_indexs)

        
def load_sequences(filename, do_tr, n_pc, path='', **kwargs):
    """Load the `.npz` files written by `save_sequences` back into plain dicts, applying
    `n_pc` at read time (B3, see `load_reduced_sequence`).

    Parameters
    ----------
    filename : str or Path
        Base name used to build the input file names, must match what was passed
        to `save_sequences`.
    do_tr : list or array
        Transit indices to load. Indices >= 10 are excluded (see `save_sequences`).
    n_pc : int
        Number of PCA components to remove, applied at read time for every transit.
    path : str or Path, optional
        Input directory.
    **kwargs
        Passed through to `load_reduced_sequence` for every transit -- in particular
        `planet=` to reuse an already-built `Planet` (e.g. with config `pl_kwargs`
        overrides, as `retrieval.py` does) instead of a fresh ExoFile lookup by name for
        every single transit loaded.

    Returns
    -------
    data_info : dict
        alpha_frac/icorr/N/bad_indexs of the *last* loaded transit -- this used to come from
        a separate shared `_data_info.npz` file (which just duplicated the last transit's own
        values); B3 derives it directly instead (see `save_sequences`).
    data_trs : dict
        One entry per transit index (as a string), each a dict of arrays/PCA object, built
        from the corresponding `Observations` returned by `load_reduced_sequence`.
    """
    filename = Path(filename)

    data_trs = {}
    data_info = {}

    for i_tr, tr_key in enumerate(do_tr[:np.nonzero(np.array(do_tr) < 10)[0].size]):
        out_filename = Path(f'{filename.name}_data_trs_{i_tr}.npz')
        log.info(f'Reading: {Path(path) / out_filename}')
        visit = load_reduced_sequence(out_filename, n_pc, path=path, **kwargs)

        data_trs[str(i_tr)] = {
            'pca': visit.pca,
            'RV_const': visit.RV_const,
            'RV_sys': visit.RV_sys,
            'mid_berv': visit.mid_berv,
            'mid_vr': visit.mid_vr,
            'params': visit.params,
            'wave': visit.wave,
            # `visit.vrp`/`visit.vr` are bare floats (km/s) after `Observations.norv_sequence` --
            # re-attach units here to match what consumers expect (e.g. `data_tr['vr'].to(...)`
            # in retrieval.py/logl_grid.py), same contract as the pre-B3 dict.
            'vrp': visit.vrp * u.km / u.s,
            # Per-exposure stellar reflex-motion excursion (same recentering convention as
            # vrp) -- needed to Doppler-shift Fstar independently from Fp (Chantier A Phase 2).
            'vr': visit.vr * u.km / u.s,
            'sep': visit.sep,
            'noise': visit.noise,
            'N': visit.N,
            't_start': visit.t_start,
            'flux': visit.final / visit.noise,
            's2f': np.ma.sum((visit.final / visit.noise) ** 2, axis=-1),
            'ratio': visit.ratio,
            'reconstructed': visit.reconstructed,
            'reference_spec': visit.reference_spec,
            'alpha_frac': visit.alpha_frac,
            'final': visit.final,
            'spec_trans': visit.spec_trans,
            'icorr': visit.icorr,
            'clip_ts': visit.clip_ts,
            'scaling': visit.scaling,
            'fl_norm': visit.fl_norm,
            'fl_norm_mo': visit.fl_norm_mo,
            'full_ts': visit.full_ts,
            'ts_norm': visit.ts_norm,
            'rebuilt': visit.rebuilt,
            'fl_Sref': visit.fl_Sref,
            'fl_masked': visit.fl_masked,
            'recon_time': visit.recon_time,
        }

        data_info = {
            'trall_alpha_frac': visit.alpha_frac,
            'trall_icorr': visit.icorr,
            'trall_N': visit.N,
            'bad_indexs': visit.bad,
        }

    return data_info, data_trs



def gen_obs_sequence(obs, transit_tag, params_all, iOut_temp,
                     coeffs, ld_model, kind_trans, RV_sys, polynome=None,
                     ratio_recon=False, cont=False, cbp=True, noise_npc=2, counting = True, **kwargs_build_ts):
    """Build one visit's transmission spectrum sequence, with `noise` decoupled from `n_pc`.

    B3 (Chantier B): `noise` is estimated once from the PCA-cleaned spectrum at a *fixed*
    `noise_npc` (independent of whatever `n_pc` is used for science), instead of being
    recomputed at whatever `n_pc` happens to be requested. This branch already existed
    (dormant, never called with `noise_npc != None` before B3) -- see the PCA cleanup notes
    for why: `noise_npc=2` matches `ReductionParams.n_pc`'s own historical default, which is
    what most non-swept reductions used for both science and noise in practice already.

    Two `build_trans_spec` calls are made when `noise_npc` is not None: the first, at
    `noise_npc` components, fixes `visit.noise` (and caches the n_pc-independent intermediates
    `fl_masked`/`fl_Sref`/`fl_norm`/`fl_norm_mo`/`reference_spec`/`spec_trans`/`pca` on `visit`); the
    second, at the real science `n_pc` (`params_all[5]`), reuses all of those (including the
    already-fitted `pca`, so it only redoes the cheap PCA truncation) and does not touch
    `visit.noise` again (`change_noise` defaults to False in `Observations.build_trans_spec`).

    Parameters
    ----------
    noise_npc : int or None
        Fixed number of PCA components used to estimate `noise`. `None` restores the old
        (pre-B3) behaviour of estimating `noise` at the same `n_pc` as the science spectrum.
    obs, transit_tag, params_all, iOut_temp, coeffs, ld_model, kind_trans, RV_sys, polynome,
    ratio_recon, cont, cbp, counting, **kwargs_build_ts :
        See `Observations.calc_sequence`/`Observations.build_trans_spec`.

    Returns
    -------
    Observations
        The visit (or merged/selected transit), with the reduction results attached.
    """
    if transit_tag is not None:
        visit = obs.select_transit(transit_tag)
    else:
        visit = obs
    visit.calc_sequence(plot=False,  coeffs=coeffs, ld_model=ld_model, kind_trans=kind_trans)
    visit.norv_sequence(RV=RV_sys)

    if polynome is not None:
    #                 print("P(O-2) = ", polynome[tag-1])
        if polynome:
            poly_time = visit.t_start  # .value  # visit.AM
        else:
            poly_time = None
    else:
        poly_time = None
        
    if noise_npc is None:
        visit.build_trans_spec(params= params_all, \
                    iOut_temp=iOut_temp, ratio_recon=ratio_recon, cont=cont,
                        cbp=cbp, poly_time=poly_time, counting = counting, **kwargs_build_ts)
        # Pre-B3 behaviour: `noise` was estimated at the same n_pc as the science spectrum.
        visit.noise_npc = params_all[5]
    else:

        params_copy = params_all.copy()
        params_copy[5] = noise_npc
        visit.build_trans_spec(params= params_copy, \
                        iOut_temp=iOut_temp, ratio_recon=ratio_recon, cont=cont,
                        cbp=cbp, poly_time=poly_time, **kwargs_build_ts)
        # Reuse everything computed above (including the fitted `pca`, B3) -- only the cheap
        # PCA truncation to the real science `n_pc` (and final normalization/masking) reruns.
        visit.build_trans_spec(params= params_all, \
                         iOut_temp=iOut_temp, ratio_recon=ratio_recon, cont=cont,
                        cbp=False, poly_time=poly_time,
                       flux_masked=visit.fl_masked, flux_Sref=visit.fl_Sref, flux_norm=visit.fl_norm,
                        flux_norm_mo=visit.fl_norm_mo, reference_spec=visit.reference_spec, spec_trans=visit.spec_trans,
                            pca=visit.pca, mask_var=False, **kwargs_build_ts)
        # `noise` was fixed by the first call above, at `noise_npc` components -- record it so
        # a saved file is self-documenting (`save_reduced_sequence`), independently of `params_all[5]`.
        visit.noise_npc = noise_npc

    return visit

def gen_merge_obs_sequence(obs, visits, merge_tr_idx, transit_tags, coeffs, ld_model, kind_trans, light=False):

    if transit_tags is not None:
        tr_merge = obs.select_transit(np.concatenate([transit_tags[tr_i-1] for tr_i in merge_tr_idx]))
    else:
        tr_merge = deepcopy(obs)

    if light is False:
        tr_merge.calc_sequence(plot=False,  coeffs=coeffs, ld_model=ld_model, kind_trans=kind_trans)
    # else:
    #     tr_merge.dt =


    merge_tr(tr_merge, visits, merge_tr_idx, light=light)
    merge_velocity(tr_merge, visits, merge_tr_idx)
    
    return tr_merge


def generate_all_transits(obs, transit_tags, RV_sys, params_all, iOut_temp,
                          do_tr=[1,2,3,12,123], cbp=True,
                           kind_trans='transmission', flux_all=None,
                          ld_model = 'linear', coeffs=[0.53],
                           polynome=None, noise_npc=2, counting = True, **kwargs_build_ts):
    #                           
    
    ratio_recon=True
    cont=False

    visits = OrderedDict({})
    
    for tag in do_tr:
        name_tag = str(tag)
        if len(name_tag) < 2:
            if flux_all is not None:
                kwargs_build_ts['flux'] = flux_all[tag-1]

            visits[name_tag] = gen_obs_sequence(obs, transit_tags[tag-1], params_all[tag-1], 
                                                 iOut_temp[tag-1],
                                                 coeffs, ld_model, kind_trans, RV_sys[tag-1], 
                                                 polynome=polynome[tag-1], noise_npc=noise_npc, 
                                 ratio_recon=ratio_recon, cont=cont, cbp=cbp, counting = counting, **kwargs_build_ts)
        
        else :  

            merge_tr_idx = [int(tag_i) for tag_i in name_tag]

            visits[name_tag] = gen_merge_obs_sequence(obs, visits, merge_tr_idx, transit_tags,
                                               coeffs, ld_model, kind_trans)

    return visits



### --- Telluric custom masking


def mask_custom_pclean_ord(visit, flux, pclean, ccf_pclean, corrRV0,
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

    for iOrd in range(visit.nord):
        #     print(iOrd)
        limit_mask = visit.params[0]
        #     flux_ord = flux[:,iOrd,None,:]

        _, rv_snr, _, snr_i, _ = calc_snr_1d(np.abs(ccf_pclean[:, iOrd]),
                                             corrRV0, np.zeros_like(visit.vrp), RV_sys=0.0)

        #         param, pcov = curve_fit(gauss, ydata=snr_i, xdata=rv_snr, p0=[5,3,0])
        #         print('sig = {}, amp = {}, x0 = {}'.format(*param))

        fct_snr = interp1d(rv_snr, snr_i)
        snr_i_0 = np.max(snr_i[100 - 3:100 + 3 + 1])  # fct_snr(0.0)

        if counting:
            hm.print_static(iOrd, snr_i_0, np.ma.max(snr_i), limit_mask)

        if plot:
            fig, axs = plt.subplots(2, 2, figsize=(6, 7), sharex=True)
            axs[0, 0].pcolormesh(corrRV0, visit.phase, np.abs(ccf_pclean[:, iOrd]))
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

                ccf_pclean_ord = quick_correl_3dmod(visit.wave[:, iOrd, None],
                                                         flux_ord,
                                                         corrRV0,
                                                         visit.wave[:, iOrd, None],
                                                         correl_spectra[:, iOrd, None])

                _, rv_snr, _, snr_i, _ = calc_snr_1d(np.abs(ccf_pclean_ord).squeeze(),
                                                     corrRV0, np.zeros_like(visit.vrp), RV_sys=0.0)
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
            new_mask = [get_mask_tell(tell, limit_mask + 0.025, pad_to) for tell in visit.pclean[:, iOrd, :]]
            new_mask = new_mask | flux[:, iOrd].mask
        #         flux_ord = np.ma.array(flux[:,iOrd], mask=new_mask)[:,None]
        print(iOrd, snr_i_0, np.ma.max(snr_i), limit_mask)

        if plot:
            axs[0, 1].pcolormesh(corrRV0, visit.phase, np.abs(ccf_pclean_new[:, iOrd]))
            axs[1, 1].plot(rv_snr, snr_i)
            axs[1, 1].axvline(0.0)
            axs[1, 1].axhline(snr_i_0)

        #         pltr.figure()
        #         pltr.pcolormesh(corrRV0, visit.phase, np.abs(ccf_pclean_ord).squeeze())

        new_mask_pclean[:, iOrd, :] = new_mask | flux[:, iOrd].mask

    new_flux = np.ma.array(flux, mask=new_mask_pclean)

    return new_mask_pclean, new_flux, ccf_pclean_new


# for visit in [t1,t2,t3]:
def mask_tellu_sky(visit, corrRV0, pad_to=0.99, plot_clean=False, fig_output_file=None, counting = True):
    if not (hasattr(visit, 'pclean') | hasattr(visit, 'sky')):
        sky = []
        tellu = []
        #     with open(path + file_list) as f:

        for file in visit.filenames:
            blocks = file.split('_')
            filename = '_'.join(blocks[0:2]) + '_tellu_pclean_' + blocks[-1]
            filename = Path(filename)

            print(filename)

            sky_model = fits.getdata(visit.path / filename, ext=4)
            tell_model = fits.getdata(visit.path / filename, ext=3)

            sky.append(np.ma.masked_invalid(sky_model))
            tellu.append(np.ma.masked_invalid(tell_model))

        visit.sky = np.ma.masked_invalid(sky)
        visit.pclean = np.ma.masked_invalid(tellu)

    sky_t = np.ma.masked_invalid(visit.sky)
    skynorm = sky_t / np.ma.max(sky_t)
    skydown = 1 - skynorm
    plt.figure()
    plt.plot(skydown[0, 34])

    spec_trans_tr = (visit.uncorr / visit.pclean) / visit.blaze / visit.reconstructed

    sky_tr = spec_trans_tr - np.ma.median(spec_trans_tr, axis=-1)[:, :, None]
    sky_tr = sky_tr / np.ma.max(sky_tr)
    sky_tr = 1 - sky_tr
    tile_sky_tr = np.tile(np.ma.mean(sky_tr, axis=0), (visit.n_spec, 1, 1))

    params = visit.params
    flux_mask = ts.build_trans_spectrum4(visit.wave, spec_trans_tr,
                                         visit.berv, visit.planet.RV_sys, visit.vr, visit.iOut,
                                         tellu=visit.tellu, noise=visit.noise,
                                         lim_mask=params[0], lim_buffer=params[1],
                                         mo_box=params[2], mo_gauss_box=params[4],
                                         n_pca=params[5],
                                         tresh=params[6], tresh_lim=params[7],
                                         last_tresh=params[8], last_tresh_lim=params[9],
                                         n_comps=visit.n_spec-2,
                                         clip_ts=None, clip_ratio=None,
                                         iOut_temp='all', cont=False,
                                         cbp=True, poly_time=None,
                                         flux_masked=spec_trans_tr,
                                         flux_Sref=spec_trans_tr, flux_norm=spec_trans_tr,
                                         flux_norm_mo=spec_trans_tr, reference_spec=spec_trans_tr,
                                         spec_trans=spec_trans_tr,
                                         mask_var=False)[6]

    #     flux_mask = visit.final.copy()
    new_mask = [get_mask_noise(f, 4, 3, gwidth=0.01, poly_ord=5) for f in flux_mask.swapaxes(0, 1)]
    new_mask = new_mask | flux_mask.mask
    flux_mask = np.ma.array(flux_mask, mask=new_mask)

    # --- Tellu masking ---

    ccf_tellu_tr = quick_correl_3dmod(visit.wave, flux_mask, corrRV0, visit.wave, visit.pclean)

    tellu_mask, cmasked_flux_tr, ccf_tellu_clean = mask_custom_pclean_ord(visit, flux_mask, visit.pclean,
                                                                          ccf_tellu_tr, corrRV0, plot=False, counting = counting)

    if plot_clean:
        ccf_tellu_clean = quick_correl_3dmod(visit.wave, cmasked_flux_tr,
                                                  corrRV0, visit.wave, visit.pclean)
        _ = plot_all_orders_correl(corrRV0, np.abs(ccf_tellu_clean), visit,
                                      icorr=None, logl=False, sharey=True,
                                      vrp=np.zeros_like(visit.vrp), RV_sys=-7.0, vmin=None, vmax=None,
                                      vline=None, hline=2, kind='snr', return_snr=True, output_file=fig_output_file)

        # --- Sky masking ---

    ccf_sky_tr = quick_correl_3dmod(visit.wave, cmasked_flux_tr, corrRV0,
                                         visit.wave, skydown)

    sky_mask, cmasked_flux_sky, ccf_sky_clean = mask_custom_pclean_ord(visit, cmasked_flux_tr, sky_tr,
                                                                       ccf_sky_tr, corrRV0, kind='sky',
                                                                       thresh=2.0, plot=False, pad_to=pad_to,
                                                                       masking_spectra=skydown, correl_spectra=skydown)

    if plot_clean:
        ccf_sky_clean = quick_correl_3dmod(visit.wave, cmasked_flux_sky,
                                                corrRV0, visit.wave, skydown)
        _ = plot_all_orders_correl(corrRV0, np.abs(ccf_sky_clean), visit,
                                      icorr=None, logl=False, sharey=True,
                                      vrp=np.zeros_like(visit.vrp), RV_sys=-7.0, vmin=None, vmax=None,
                                      vline=None, hline=2, kind='snr', return_snr=True, output_file=fig_output_file)

    if not hasattr(visit,'original_mask'):
        visit.original_mask = visit.spec_trans.mask.copy()

    visit.OG_final_mask = visit.final.mask.copy()
    visit.OG_spec_trans_mask = visit.spec_trans.mask.copy()
    visit.final = cmasked_flux_sky
    visit.custom_mask = cmasked_flux_sky.mask
    visit.spec_trans.mask = cmasked_flux_sky.mask

    del sky_mask, ccf_sky_clean, ccf_sky_tr, ccf_tellu_tr, tellu_mask, cmasked_flux_tr, ccf_tellu_clean, cmasked_flux_sky
    del flux_mask, new_mask, sky_t, skynorm, spec_trans_tr, sky_tr, tile_sky_tr
    gc.collect()
