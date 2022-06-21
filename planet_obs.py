
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.time import Time
from astropy.io import fits
import astropy.constants as const
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.table import Table

from starships.list_of_dict import *
import starships.orbite as o
from starships import transpec as ts
from starships import homemade as hm
from starships import spectrum as spectrum
from starships.mask_tools import interp1d_masked
from starships.correlation import calc_logl_BL_ord # calc_logl_OG_cst, calc_logl_OG_ord
# from starships.analysis import make_quick_model
# from starships.extract import quick_norm
# from transit_prediction.masterfile import MasterFile 
# from masterfile.archive import MasterFile
from exofile.archive import ExoFile

# from fits2wave import fits2wave
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import medfilt
# from tqdm import tqdm
import os


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

def read_all_sp(path, file_list, onedim=False, wv_default=None, blaze_default=None,
                blaze_path=None, debug=False, ver06=False):
    
    """
    Read all spectra
    Must have a list with all filename to read 
    """

    headers, count, wv, blaze = list_of_dict([]), [], [], []
    blaze_path = blaze_path or path
    
    headers_princ = list_of_dict([])
    filenames = []
    blaze0 = None

    with open(path + file_list) as f:

        for file in f:
            filename = file.split('\n')[0]
            
            if debug:
                print(filename)

            filenames.append(filename)
            hdul = fits.open(path + filename)
        
            # --- If not 1D spectra (so if they are E2DS) ---
            if onedim is False: 
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
                    with fits.open(path + wv_file) as f:
                        wvsol = f[0].data
                except (KeyError,FileNotFoundError) as e:
                    wvsol = fits2wave(image, header)
    #                 if debug:
    #                     print(wvsol)
                
#                 if blaze0 is None:
                try:
                    blaze_file = blaze_default or header['CDBBLAZE']
                except KeyError:
                    blaze_file = header['CDBBLAZE']
                
                if ver06 is False:
                    blaze0 = fits.getdata(blaze_path + blaze_file, ext=1)
                else:
                    with fits.open(blaze_path + blaze_file) as f:
#                         header = fits.getheader(filename, ext=0) 
                        blaze0 = f[0].data
#                         print(blaze)
                blaze.append(blaze0)

            # --- For 1D spectra --- (probably old)
            else:
                headers.append(hdul[1].header)

                data = Table.read(path+filename)
                wvsol = data['wavelength'][None,:]
                count.append(data['flux'][None,:])
        #         eflux = data['eflux']
                blaze.append(data['weight'][None,:])
                
            wv.append(wvsol/1000)

    return headers, np.array(wv), np.array(count), np.array(blaze), filenames


def read_all_sp_CADC(path, file_list):
    
    """
    Read all CADC-type spectra
    Must have a list with all filename to read 
    Note : Probably old
    """
    
    headers_princ, headers_image, headers_tellu = list_of_dict([]), list_of_dict([]), list_of_dict([])
    count, wv, blaze, recon = [], [], [], []
    filenames = []
    with open(path + file_list) as f:

        for file in f:
           # print(file)
            filenames.append(file.split('\n')[0])
            hdul = fits.open(path + file.split('\n')[0])
            
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


class Observations():
    
    """
    Observations class object
    Will contain all data and parameters
    Note : Probably could be optimized
    """

    def __init__(self, wave=np.array([]), count=np.array([]), blaze=np.array([]),
                 headers = list_of_dict([]), headers_image = list_of_dict([]), headers_tellu = list_of_dict([]), 
                 tellu=np.array([]), uncorr=np.array([]), 
                 name='', path='',filenames=[], planet=None, CADC=False, **kwargs):
        
        self.name = name
        self.path=path
        
        # --- Get the system parameters from the ExoFile
        if planet is None:
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
        
        self.uncorr=uncorr
        self.tellu=tellu
        self.CADC = CADC

                     
    def fetch_data(self, path, onedim=False, CADC=False, sanit=False, **kwargs):
        
        """
        Retrieve all the relevent data in path 
        (tellu corrected, tellu recon and uncorrected spectra from lists of files)
        """
        
        self.CADC = CADC
        
        if onedim is False:
            file_list = 'list_e2ds'
            file_list_tcorr = 'list_tellu_corrected'
            file_list_recon = 'list_tellu_recon'
#             if sanit is True:
#                 file_list = 'list_e2ds_sanit'
#                 file_list_tcorr = 'list_sanit'
#                 file_list_recon = 'list_tellu_recon_sanit'
            
        else:
            file_list = 'list_s1d'
            file_list_tcorr = 'list_tellu_corrected_1d'
            file_list_recon = 'list_tellu_recon_1d'
        
        if CADC is False:

            print('Fetching data')
            headers, wv, count, blaze, filenames = read_all_sp(path, file_list_tcorr, onedim=onedim, **kwargs)

#             self.headers = headers
#             self.wave = np.array(wv)
#             self.count = np.ma.masked_invalid(count)
#             self.blaze = np.ma.masked_array(blaze)
#             self.filenames  = filenames

            print("Fetching the tellurics")
            _, _, tellu, _, _ = read_all_sp(path, file_list_recon, onedim=onedim, **kwargs)

            print("Fetching the uncorrected spectra")
            
            _, _, count_uncorr, blaze_uncorr, filenames_uncorr = read_all_sp(path, file_list, onedim=onedim, **kwargs)

        else:
            print('Fetching data')
            headers, headers_image, headers_tellu, \
            wv, count, blaze, tellu, filenames = read_all_sp_CADC(path, 'list_tellu_corrected')
            
            self.headers_image, self.headers_tellu = headers_image, headers_tellu
            
            print("Fetching the uncorrected spectra")
            _, _, _, _, count_uncorr, blaze_uncorr, _, filenames_uncorr = read_all_sp_CADC(path, 'list_e2ds')
           
            
        self.headers = headers
        self.wave = np.array(wv)
        self.count = np.ma.masked_invalid(count)
        self.blaze = np.ma.masked_invalid(blaze)
        self.filenames  = filenames
        self.filenames_uncorr = filenames_uncorr

        self.tellu = np.ma.masked_invalid(tellu)
        if np.mean(count_uncorr) < 0:
            print('Mean below 0 = {}, flipping sign'.format(np.mean(count_uncorr))) 
            count_uncorr = -count_uncorr
        count_uncorr = np.ma.masked_invalid(np.clip(count_uncorr, 0,None))

        self.uncorr = count_uncorr

        if onedim is False:
            self.uncorr_fl = self.uncorr/(blaze_uncorr/np.nanmax(blaze_uncorr, axis=-1)[:,:,None])
            
        self.path=path
        
        
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
        
        return Observations(headers=new_headers, wave=self.wave[transit_tag],
                            count=self.count[transit_tag], blaze=self.blaze[transit_tag], 
                            tellu=self.tellu[transit_tag], 
                            uncorr=self.uncorr[transit_tag],
                            name=self.name, planet=self.planet , 
                            path=self.path, filenames=np.array(self.filenames)[transit_tag],
                            filenames_uncorr=np.array(self.filenames_uncorr)[transit_tag], 
                            CADC=self.CADC, headers_image=new_headers_im, headers_tellu=new_headers_tl)
    
    def calc_sequence(self, plot=True, sequence=None, K=None, uncorr=False, iin=False, 
                      coeffs=[0.4], ld_model='linear', time_type='BJD', kind_trans='transmission'):
        
        """
        Compute the sequence time series stuff 
        (time, airmass(t), RV(t), when and where it is in transit/eclipse, etc.)
        """
        
        p = self.planet
        self.headers[0]['VERSION']
#         if self.count.ndim == 3:
#             self.n_spec, self.nord, self.npix = self.count.shape
#             self.onedim = False
#         elif self.count.ndim == 2:
#             self.n_spec, self.npix = self.count.shape
#             self.nord = 1
#             self.onedim = True
        self.n_spec, self.nord, self.npix = self.count.shape
        if self.nord == 1:
             self.onedim = True
        else:
             self.onedim = False
        
        if sequence is None:
            
            if self.CADC is False:
                if time_type == 'BJD':
                    self.t_start = Time(np.array(self.headers.get_all('BJD')[0], dtype='float'), 
                                format='jd').jd.squeeze() * u.d
                elif time_type == 'MJD':
                    self.t_start = Time((np.array(self.headers.get_all('MJDATE')[0], dtype='float') + \
                                        np.array(self.headers.get_all('MJDEND')[0], dtype='float')) / 2, 
                                format='jd').jd.squeeze() * u.d
                    
                self.SNR = np.ma.masked_invalid([np.array(self.headers.get_all('EXTSN'+'{:03}'.format(order))[0], 
                                 dtype='float') for order in range(49)]).T
                self.berv = np.array(self.headers.get_all('BERV')[0], dtype='float').squeeze()
            else:
#                 obs_date = [date+' '+hour for date,hour in zip(self.headers_image.get_all('DATE-OBS')[0], \
#                                                self.headers.get_all('UTIME')[0])]
#                 self.t_start = Time(obs_date).jd * u.d
#                 self.t_start = Time(np.array(self.headers_image.get_all('BJD')[0], dtype='float'), 
#                                 format='jd').jd.squeeze() * u.d
                if time_type == 'BJD':
                    self.t_start = Time(np.array(self.headers_image.get_all('BJD')[0], dtype='float'), 
                                format='jd').jd.squeeze() * u.d
                elif time_type == 'MJD':
                    self.t_start = Time((np.array(self.headers_image.get_all('MJDATE')[0], dtype='float') + \
                                        np.array(self.headers_image.get_all('MJDEND')[0], dtype='float')) / 2, 
                                format='jd').jd.squeeze() * u.d

                try:
                    self.SNR = np.ma.masked_invalid([np.array(self.headers_image.get_all('SNR'+'{}'.format(order))[0], \
                                             dtype='float') for order in range(49)]).T
                except KeyError:
                    self.SNR = np.ma.masked_invalid([np.array(self.headers_image.get_all('EXTSN'+'{:03}'.format(order))[0], \
                                         dtype='float') for order in range(49)]).T
                self.berv = np.array(self.headers_image.get_all('BERV')[0], dtype='float').squeeze()
            
            self.dt = np.array(np.array(self.headers.get_all('EXPTIME')[0], dtype='float') ).squeeze() * u.s
            self.AM = np.array(self.headers.get_all('AIRMASS')[0], dtype='float').squeeze()
            self.telaz = np.array(self.headers.get_all('TELAZ')[0], dtype='float').squeeze()
            self.adc1 = np.array(self.headers.get_all('SBADC1_P')[0], dtype='float').squeeze()
            self.adc2 = np.array(self.headers.get_all('SBADC2_P')[0], dtype='float').squeeze()
            self.SNR = np.clip(self.SNR, 0,None)
        else : 
            self.t_start = sequence[0] * u.d
            self.SNR = sequence[1]
            self.berv = sequence[2]
            self.dt = sequence[3] * u.s

            self.AM = sequence[4]
            self.telaz = np.empty_like(self.AM)
            self.adc1 = np.empty_like(self.AM)
            self.adc2 = np.empty_like(self.AM)
        
        if self.onedim is False:
            self.flux = self.count/(self.blaze/np.nanmax(self.blaze, axis=-1)[:,:,None])
        else:
            self.flux = self.count
        light_curve = np.ma.sum(np.ma.sum(self.count, axis=-1), axis=-1)
        self.light_curve = light_curve / np.nanmax(light_curve)
        self.t = self.t_start #+ self.dt/2
        
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
        
        self.nu = o.t2trueanom(p.period, self.t.to(u.d), t0=p.mid_tr, e=p.excent)

        rp, x, y, z, self.sep, p.bRstar = o.position(self.nu, e=p.excent, i=p.incl, w=p.w, omega=p.omega,
                                                Rstar=p.R_star, P=p.period, ap=p.ap, Mp=p.M_pl, Mstar=p.M_star)

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
            self.iIn = np.sort(np.concatenate([part,total]))
#             print(self.iIn.size, self.iOut.size)
            
            
        
        elif kind_trans == 'emission':
            print('Emission')
            self.iOut = total
            self.part = part
            self.total = out
            self.iIn = np.sort(np.concatenate([out]))
            
#             print(self.iIn.size, self.iOut.size)
            
        self.iin, self.iout = o.where_is_the_transit(self.t, p.mid_tr, p.period, p.trandur)
        self.iout_e, self.iin_e = o.where_is_the_transit(self.t, p.mid_tr+0.5*p.period, p.period, p.trandur)
        
        if (self.part.size == 0) and (iin is True) :
            print('Taking iin and iout')
            self.iIn = self.iin
            self.iOut = self.iout
        
        self.icorr = self.iIn
        
        if plot is True:
            fig, ax = plt.subplots(3,1, sharex=True)
            ax[2].plot(self.t, np.nanmean(self.SNR[:, :],axis=-1),'o-')
            if out.size > 0 :
                ax[0].plot(self.t[out],self.AM[out],'ro')
                ax[1].plot(self.t[out], self.adc1[out],'ro')
                ax[2].plot(self.t[out], np.nanmean(self.SNR[out, :],axis=-1),'ro')
            if part.size > 0 :
                ax[0].plot(self.t[part],self.AM[part],'go')
                ax[1].plot(self.t[part], self.adc1[part],'go')
                ax[2].plot(self.t[part], np.nanmean(self.SNR[part, :],axis=-1),'go')
            if total.size > 0 :
                ax[0].plot(self.t[total],self.AM[total],'bo')
                ax[1].plot(self.t[total], self.adc1[total],'bo')
                ax[2].plot(self.t[total], np.nanmean(self.SNR[total, :],axis=-1),'bo')
#             if self.iin.size > 0 :    
#                 ax[0].plot(self.t[self.iin], self.AM[self.iin],'g.')
#                 ax[1].plot(self.t[self.iin], self.adc1[self.iin],'g.')
#                 ax[2].plot(self.t[self.iin], np.nanmean(self.SNR[self.iin, :],axis=-1),'g.')
#             if self.iout.size > 0 :  
#                 ax[0].plot(self.t[self.iout], self.AM[self.iout],'r.')
#                 ax[1].plot(self.t[self.iout], self.adc1[self.iout],'r.')
#                 ax[2].plot(self.t[self.iout], np.nanmean(self.SNR[self.iout, :],axis=-1),'r.')
            

            ax[0].set_ylabel('Airmass')
            ax[1].set_ylabel('ADC1 angle')
            ax[2].set_ylabel('Mean SNR')
        
        
        
        self.alpha = hm.calc_tr_lightcurve(p, coeffs, self.t.value, ld_model=ld_model, kind_trans=kind_trans)
#         self.alpha = np.array([(hm.circle_overlap(p.R_star.to(u.m), p.R_pl.to(u.m), sep) / \
#                         p.A_star).value for sep in self.sep]).squeeze()
#         self.alpha = np.array([hm.circle_overlap(p.R_star, p.R_pl, sep).value for sep in self.sep])
        self.alpha_frac = self.alpha/self.alpha.max()
    
        # --- Radial velocities
        
        if K is None:
            K, vr, Kp, vrp = o.rv(self.nu, p.period, e=p.excent, i=p.incl, w=p.w, Mp=p.M_pl, Mstar=p.M_star)
        else:
            if isinstance(K, u.Quantity):
                 K= K.to(u.km/u.s)
            else:
                 K= K*u.km/u.s
            Kp = o.Kp_theo(K, p.M_star, p.M_pl)
            vr = o.rv_theo_t(K, self.t, p.mid_tr, p.period, plnt=False)
            vrp = o.rv_theo_t(Kp, self.t, p.mid_tr, p.period, plnt=True)

        self.vrp = vrp.to('km/s').squeeze()  # km/s
        self.vr = vr.to('km/s').squeeze()  # km/s   # np.zeros_like(vrp)  # ********
        self.K, self.Kp = K.to('km/s').squeeze(), Kp.to('km/s').squeeze()

        v_star = (vr + p.RV_sys).to('km/s').value
        v_pl = (vrp + p.RV_sys).to('km/s').value
        self.dv_pl = v_pl - self.berv  #+berv
        self.dv_star = v_star - self.berv  #+berv
    
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
        self.phase = ((self.t-p.mid_tr)/p.period).decompose()
        self.phase -= np.round(self.phase.mean())
        if kind_trans == 'emission':
            if (self.phase < 0).all():
                self.phase += 1.0

        self.wv = np.mean(self.wave, axis=0)     
        

    def get_plot_cst(self) :   
        return  [(self.vrp-self.vr), self.berv, self.Kp, self.planet.RV_sys, \
                 self.nu, self.planet.w]
    
    def build_trans_spec(self, flux=None, params=None, n_comps=None, 
                         change_ratio=False, change_noise=False, ratio_recon=False, 
                         clip_ts=None, clip_ratio=None, fast=False, **kwargs):
        
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
        self.spec_trans, self.full_ts, self.chose, \
        self.final, self.final_std, self.rebuilt, \
        self.pca, self.fl_Sref, self.fl_masked, \
        ratio, last_mask = ts.build_trans_spectrum4(self.wave, flux, 
                                     self.light_curve, self.berv, self.planet.RV_sys, 
                                     self.vr, self.vrp, self.iIn, self.iOut, 
                                     path=self.path, tellu=self.tellu, noise=noise, 
                                    lim_mask=params[0], lim_buffer=params[1],
                                    mo_box=params[2], mo_gauss_box=params[4],
                                    n_pca=params[5],
                                    tresh=params[6], tresh_lim=params[7],
                                    tresh2=params[8], tresh_lim2=params[9], 
                                    n_comps=self.n_comps,
                                    clip_ts=clip_ts, clip_ratio=clip_ratio, **kwargs)
        
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
            
        self.berv = -self.berv
        self.mid_id = int(np.ceil(self.n_spec/2)-1)
        self.mid_berv = self.berv[self.mid_id]
        self.mid_vr = self.vr[self.mid_id]
        self.mid_vrp = self.vrp[self.mid_id]

        self.berv = (self.berv-self.berv[self.mid_id])
        self.vr = (self.vr-self.vr[self.mid_id]).to(u.km / u.s)
        self.vrp = (self.vrp-self.vrp[self.mid_id]).to(u.km / u.s)
        self.planet.RV_sys=0*u.km/u.s

        self.RV_const = self.mid_berv+self.mid_vr.value+self.RV_sys


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
        self.mid_vr = self.vr[self.mid_id]
        self.mid_vrp = self.vrp[self.mid_id]
        self.RV_const = self.mid_berv + self.mid_vr.value + self.RV_sys

        tb1.berv = -tb1.berv
        tb1.berv = (tb1.berv-self.mid_berv)
        tb1.vr = (tb1.vr-self.mid_vr).to(u.km / u.s)
        tb1.vrp = (tb1.vrp-self.mid_vrp).to(u.km / u.s)
        tb1.planet.RV_sys=0*u.km/u.s
        tb1.RV_const = self.mid_berv + self.mid_vr.value + self.RV_sys
        tb1.mid_berv = self.mid_berv
        tb1.mid_vrp = self.mid_vrp
        tb1.mid_vr = self.mid_vr
        
#         tb1.build_trans_spec(**kwargs1, **kwargs)

        tb2.berv = -tb2.berv
        tb2.berv = (tb2.berv-self.mid_berv)
        tb2.vr = (tb2.vr-self.mid_vr).to(u.km / u.s)
        tb2.vrp = (tb2.vrp-self.mid_vrp).to(u.km / u.s)
        tb2.planet.RV_sys=0*u.km/u.s
        tb2.RV_const = self.mid_berv + self.mid_vr.value + self.RV_sys
        tb2.mid_berv = self.mid_berv
        tb2.mid_vrp = self.mid_vrp
        tb2.mid_vr = self.mid_vr
        
#         tb2.build_trans_spec(**kwargs2, **kwargs)

        self.berv = (self.berv-self.berv[self.mid_id])
        self.vr = (self.vr-self.vr[self.mid_id]).to(u.km / u.s)
        self.vrp = (self.vrp-self.vrp[self.mid_id]).to(u.km / u.s)


        
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
                    kind='BL', somme=False, sfsg=False, binning=False):
        print("Trans spec reduction params :  ", self.params) 

        correl = np.ma.zeros((self.n_spec, self.nord, corrRV.size))
#         correl0 = np.ma.zeros((self.n_spec, self.nord, corrRV.size))
        logl = np.ma.zeros((self.n_spec, self.nord, corrRV.size))

        # - Calculate the shift -
        shifts = hm.calc_shift(corrRV, kind='rel')

        # - Interpolate over the orginal data
#         if binning is False:
        fct = interp1d_masked(mod_x, mod_y, kind='cubic', fill_value='extrapolate')

#         if (get_logl is True) and (kind == 'OG'):
#             sig, flux_norm, s2f, cst = calc_logl_OG_cst(self.final[:, :, :, None], axis=2)

        for iOrd in range(self.nord):
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

        data = Table.read(self.path+file)
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
    def __init__(self, name, parametres=None):
        self.name = name
        
        if parametres is None:
            parametres = ExoFile.load(use_alt_file=True).by_pl_name(name)

        #  --- Propriétés du système
        self.R_star = parametres['st_rad'].to(u.m)
        self.M_star = parametres['st_mass'].to(u.kg)
        self.RV_sys = parametres['st_radv'].to(u.km/u.s)
        self.Teff = parametres['st_teff'].to(u.K)
        try :
            self.vsini= parametres['st_vsin'].data.data*u.m/u.s
        except AttributeError:
            self.vsini= parametres['st_vsin'].to(u.m/u.s)

        #--- Propriétés de la planète
        try :
            self.R_pl = (parametres['pl_radj'].data*const.R_jup).data*u.m
            self.M_pl = (parametres['pl_bmassj'].data*const.M_jup).data*u.kg
            self.ap = (parametres['pl_orbsmax'].data*const.au).data*u.m
        except (AttributeError, TypeError) as e:
            self.R_pl = parametres['pl_radj'].to(u.m)
            self.M_pl = parametres['pl_bmassj'].to(u.kg)
            self.ap = parametres['pl_orbsmax'].to(u.m)
        surf_grav_pl = (const.G * self.M_pl / self.R_pl**2).cgs
        self.logg_pl = np.log10(surf_grav_pl.value)
        
        # --- Paramètres de l'étoile
        self.A_star = np.pi * self.R_star**2
        surf_grav = (const.G * self.M_star / self.R_star**2).cgs
        self.logg = np.log10(surf_grav.value)
        self.gp = const.G * self.M_pl / self.R_pl**2

        # --- Paramètres d'observations
        self.observatoire = 'cfht'
        try:
            self.radec = [parametres['ra'].data.data*u.deg, parametres['dec'].data.data*u.deg]  # parametres['radec']
        except AttributeError:
            self.radec = [parametres['ra']*u.deg, parametres['dec']*u.deg]
        # --- Paramètres transit
        self.period = parametres['pl_orbper'].to(u.s) 
        try:
            self.mid_tr = parametres['pl_tranmid'].data * u.d  # from nasa exo archive
        except (AttributeError, TypeError) as e:
            self.mid_tr = parametres['pl_tranmid'].to(u.d)
        self.trandur = (parametres['pl_trandur']/u.d*u.h).to(u.s) 

        # --- Paramètres Orbitaux
        self.excent = parametres['pl_orbeccen']
        if self.excent.mask:
            self.excent = 0
        self.incl = parametres['pl_orbincl'].to(u.rad)
        self.w = parametres['pl_orblper'].to(u.rad)+ (3*np.pi / 2 )* u.rad 
        self.omega = np.radians(0)*u.rad

        # # - Paramètres atmosphériques approximatifs
        self.mu = 2.3 * const.u
        self.rho = parametres['pl_dens']  # 5.5 *u.g/u.cm**3 # --- Jupiter : 1.33 g cm-3  /// Terre : 5.5 g cm-3
        self.Tp = np.asarray(parametres['pl_eqt'],dtype=np.float64)*u.K
        self.H = (const.k_B * self.Tp / (self.mu * self.gp)).decompose()

        self.all_params = parametres

        self.sync_equat_rot_speed = (2*np.pi*self.R_pl/self.period).to(u.km/u.s)

        
        
from astropy.io import ascii


def get_blaze_file(path, file_list='list_tellu_corrected', onedim=False, blaze_default=None,
                blaze_path=None, debug=False, folder='cfht_sept1'):
    blaze_path = blaze_path or path

    blaze_file_list = []
    with open(path + file_list) as f:

        for file in f:
            filename = file.split('\n')[0]
            
            if debug:
                print(filename)

            hdul = fits.open(path + filename)

            if onedim is False:
                    
                try:
                    blaze_file = blaze_default or hdul[0].header['CDBBLAZE']
                except KeyError:
                    blaze_file = hdul[0].header['CDBBLAZE']
                
                date = hdul[0].header['DATE-OBS']
                blaze_file_list.append(date+'/'+blaze_file)

    x = []
 
    for file in np.unique(blaze_file_list):
        blz = '/spirou/cfht_nights/{}/reduced/{}'.format(folder, file)
        print(blz)
        x.append(blz)
        

    data = Table()
    data[''] = x

    ascii.write(data, path+'blaze_files', overwrite=True,comment=False)          
    print('Dont forget to remove "col0" from file')
                
    return np.unique(blaze_file_list)



 ##############################################################################   

def merge_tr(trb1,trb2,tr_merge, params=None):
    tr_merge.icorr = np.concatenate((trb1.icorr, trb1.n_spec + trb2.icorr))
    tr_merge.phase = np.concatenate((trb1.phase, trb2.phase)).value
#     tr_merge.phase = np.concatenate((trb1.phase, trb2.phase), axis=0).value
    
    tr_merge.noise = np.ma.concatenate((trb1.noise, trb2.noise), axis=0)
    tr_merge.fl_norm = np.ma.concatenate((trb1.fl_norm, trb2.fl_norm), axis=0)
    tr_merge.fl_Sref = np.ma.concatenate((trb1.fl_Sref, trb2.fl_Sref), axis=0)
    tr_merge.fl_masked = np.ma.concatenate((trb1.fl_masked, trb2.fl_masked), axis=0)
    tr_merge.fl_norm_mo = np.ma.concatenate((trb1.fl_norm_mo, trb2.fl_norm_mo), axis=0)
    
    if trb1.mast_out.ndim == 2:
        tr_merge.mast_out = np.ma.mean(np.ma.array([np.ma.masked_invalid(trb1.mast_out),
                                                np.ma.masked_invalid(trb2.mast_out)]), axis=0)
    elif trb1.mast_out.ndim == 3:
        tr_merge.mast_out = np.ma.concatenate((trb1.mast_out, trb2.mast_out), axis=0)

    tr_merge.spec_trans = np.ma.concatenate((trb1.spec_trans, trb2.spec_trans), axis=0)
    tr_merge.full_ts = np.ma.concatenate((trb1.full_ts, trb2.full_ts), axis=0)
#     tr_merge.chose = np.ma.concatenate((trb1.chose, trb2.chose), axis=0)
    tr_merge.final = np.ma.concatenate((trb1.final, trb2.final), axis=0)
    tr_merge.final_std = np.ma.concatenate((trb1.final_std, trb2.final_std), axis=0)
    tr_merge.rebuilt = np.ma.concatenate((trb1.rebuilt, trb2.rebuilt), axis=0)
#     tr_merge.pca = tr.pca  # np.concatenate((trb1.pca, trb2.pca), axis=0)
    tr_merge.N = np.ma.concatenate((trb1.N, trb2.N), axis=0)
    tr_merge.N_frac = np.min(np.array([trb1.N_frac,trb2.N_frac]),axis=0)
    tr_merge.reconstructed = np.ma.concatenate((trb1.reconstructed, trb2.reconstructed), axis=0)
    tr_merge.ratio = np.ma.concatenate((trb1.ratio, trb2.ratio), axis=0)
#     tr_merge.recon_all = np.ma.concatenate((trb1.recon_all, trb2.recon_all), axis=0)
    if params is None:
        tr_merge.params = trb1.params
        
        
def merge_velocity(trb1,trb2,tr_merge):
    tr_merge.mid_vrp = np.concatenate((trb1.mid_vrp*np.ones_like(trb1.vrp.value),
                                       trb2.mid_vrp*np.ones_like(trb2.vrp.value)))
    tr_merge.RV_sys = np.concatenate((trb1.RV_sys*np.ones_like(trb1.vrp.value),
                                      trb2.RV_sys*np.ones_like(trb2.vrp.value)))
    tr_merge.mid_berv = np.concatenate((trb1.mid_berv*np.ones_like(trb1.vrp.value),
                                        trb2.mid_berv*np.ones_like(trb2.vrp.value)))
    tr_merge.mid_vr = np.concatenate((trb1.mid_vr*np.ones_like(trb1.vrp.value),
                                      trb2.mid_vr*np.ones_like(trb2.vrp.value)))
    tr_merge.berv = np.concatenate((trb1.berv*np.ones_like(trb1.vrp.value),
                                    trb2.berv*np.ones_like(trb2.vrp.value)))
    tr_merge.vrp = np.concatenate((trb1.vrp*np.ones_like(trb1.vrp.value),
                                   trb2.vrp*np.ones_like(trb2.vrp.value)))
    tr_merge.vr = np.concatenate((trb1.vr*np.ones_like(trb1.vrp.value),
                                  trb2.vr*np.ones_like(trb2.vrp.value)))
    tr_merge.RV_const = np.concatenate((trb1.RV_const*np.ones_like(trb1.vrp.value),
                                        trb2.RV_const*np.ones_like(trb2.vrp.value)))
    # t12.phase = np.concatenate((t1.phase,t2.phase)).value

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

        

