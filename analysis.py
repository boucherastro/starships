
import matplotlib.pyplot as plt

from starships import homemade as hm
from starships.orbite import rv_theo, rv_theo_t, rv_theo_nu
from starships.spectrum import gen_mod,  resampling, box_binning
# from spirou_exo.transpec import #build_trans_spectrum3, remove_dem_pca, mask_deep
from starships.mask_tools import interp1d_masked
from starships.extract import median_filter, quick_norm 
# from scipy.ndimage import generic_filter
interp1d_masked.iprint = False


import numpy as np
# from scipy.interpolate import interp1d
# from scipy.interpolate import interp2d
import scipy as sp
from scipy.optimize import fminbound
# from scipy.ndimage.filters import generic_filter1d
from scipy.optimize import curve_fit
# from scipy.ndimage import convolve1d

# from astropy.time import Time
# from astropy.io import fits
# from astropy.io import ascii
from astropy.stats import sigma_clip
import astropy.units as u
import astropy.constants as const
from astropy.units import cds
cds.enable() 

# from datetime import date, datetime
# from .config import *


from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))

# from cycler import cycler
# from hapi import *
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from scipy.ndimage.filters import gaussian_filter1d #median_filter


### --- ANALYTIC FUNCTIONS ---#


def gauss_y0(x, sig=1, a=1, x0=0, y0=0):

    return a * np.exp(-0.5 * (x - x0)**2 / sig**2) + y0


def lorentz(x, a, scale, x0):
    #     1/(np.pi * a * (1 + ((x-x0)/a)**2 ))
    return scale * (a / ((x - x0)**2 + a**2)) / np.pi


def gauss(x, sig, a, x0):  # y0

    return a * np.exp(-0.5 * (x - x0)**2 / sig**2)  # + y0


def lnlikelihood(y, yerr, model, axis=0):
    # --- Likelihood function
    '''
    This likelihood function is simply a Gaussian where the variance is
    underestimated by some fractional amount: f. In Python, you would code
    this up as:
    '''
    # inv_sigma2 = 1.0/(yerr**2)  # + model**2*np.exp(2*lnf))
    return -(np.nansum(((y - model) / yerr)**2, axis=axis)) / 2  # - np.log(inv_sigma2))) # *inv_sigma2


### --- SHIFTING CORRELATION --- ###

def shift_correl(interpol_RV_grid, corr_map, previous_RV_grid, unshift_RV):
    
    shifted = np.empty((corr_map.shape[0], interpol_RV_grid.size))

    for i in range(corr_map.shape[0]):

        fct = interp1d_masked(previous_RV_grid - unshift_RV[i], corr_map[i], 
                              kind='linear', fill_value='extrapolate')
        shifted[i, :] = fct(interpol_RV_grid).squeeze()            

    return shifted


# --- find max with gaussian
def find_max_gaussian(x, y, kind='max', p0_estim=None, error_gauss=False, debug=False):
    # p0_estim=[3, 1, 0, 0]  #gauss_y0
    # p0_estim=[3, 1, 0]  # gauss : sig,a,x0
    
    if kind == 'min':
        y *= -1
    
    try:
        param, pcov = curve_fit(gauss, x, y, p0=p0_estim)  # y[n]
        perr = np.sqrt(np.diag(pcov))
        
    except (RuntimeError, ValueError) as e:
        param, perr = np.empty(3), np.empty(3)
        param[2] = find_max_spline(x, -y, kind=kind) or x[np.argmax(y)]  # y[n]
        perr[2] = np.diff(x)[0]
        param[0] = np.nan
    
    if debug is True:
        print(param)
        plt.figure()
        plt.plot(x,y)
        plt.axvline(param[2])
        plt.axvline(x[np.argmax(y)],color='b')
        print(find_max_spline(x, -y, kind=kind))
        plt.axvline(find_max_spline(x, y, kind=kind)[0], color='r')
        hm.stop()
    
    if error_gauss is False:
        return param
    else:
        return param, perr

# --- find max with spline


def find_max_spline(x, y, kind='max', binning=False ):

    if kind == 'min':
        y *= -1
    
    if binning is True:
        y = box_binning(y,2)
    s_fct = interp1d_masked(x, -y, kind='cubic')  # y[n]

    return fminbound(s_fct, *x[[0, -1]], xtol=1e-5, full_output=True)  # [n]

# --- find max with max


def find_max_max(x, y, kind='max', axis=-1, norm=False):

    if kind == 'min':
        y *= -1

    if norm is True:
        chose = y / np.expand_dims(np.max(y, axis=axis), axis=-1)
    else:
        chose = y

    x0 = x[np.argmax(chose, axis=axis)]

    return x0

def find_lorentz_peak(xx, yy, kind='max', plot=False):

    n_elem = yy.shape[0]
    x0 = np.empty(n_elem)
    x0_err = np.empty(n_elem)

    if kind == 'min':
        y *= -1

    for n in range(n_elem):
        if isinstance(yy[n], np.ma.MaskedArray):
            #             print('masked')
            y = yy[n][~yy[n].mask].copy()
            x = xx[n][~xx[n].mask].copy()
#             print(y.shape,x.shape)
        else:
            y = yy[n].copy()
            x = xx[n].copy()

        mean = np.sum(x * y) / x.size  # note this correction
        sigma = np.sum(y * (x - mean)**2) / x.size  # note this correction

        try:
            param, pcov = curve_fit(lorentz, x, y, p0=[sigma, mean, 0])
            erreur = np.sqrt(np.diag(pcov))[2]
#             param = curve_fit(gaussian, x, y, p0=[sigma,mean,0])[0]   # ,ftol=1e-14,bounds=([0.01,40,-0.5,0], [1, 80, 0, 0.5])
        except RuntimeError:
            param = np.empty(3)
            param[2] = x[np.argmax(y)]
            erreur = 0.
        x0[n] = param[2]
#         print(np.sqrt(np.diag(pcov)).shape)
        x0_err[n] = erreur
#         print(param)

        if plot is True:
            plt.figure()
            plt.plot(x, y, alpha=0.5)
#             plt.plot(x,gaussian(x,*param))
            plt.plot(x, lorentz(x, *param))
            plt.axvline(param[2], color='blue')

#     print(x0)

    return x0, x0_err


def find_maxes(filtered_data, n_bins=600, sigma=8, plot=False):

    # n_bins = 30
    n_spec = filtered_data.shape[0]

    clean_hist = np.ma.empty((n_spec, n_bins))
    clean_x_hist = np.ma.empty((n_spec, n_bins))

    # plt.figure()
    # plt.xlim(-2,2)

    for n in range(n_spec):

        histog = np.histogram(filtered_data[n, :, :].compressed(), bins=n_bins)

        x_hist = histog[1][:-1] + np.diff(histog[1]) / 2

        clean_hist[n, :] = sigma_clip(histog[0], sigma=sigma)

    #     plt.plot(x_hist,histog[0])
    #     plt.plot(x_hist,clean_hist[n,:],'r')
    #     hm.stop()

        if isinstance(clean_hist, np.ma.MaskedArray):
            clean_x_hist[n, :] = np.ma.array(x_hist, mask=clean_hist[n, :].mask)
        else:
            clean_x_hist[n, :] = x_hist
    # hm.stop()
    return find_lorentz_peak(clean_x_hist, clean_hist, kind='max', plot=plot)



#### ORDERS SELECTIONS ######


def bands(wave2d, band_list, cut=False):
    y_band = np.unique(np.where((wave2d >= 0.98) & (wave2d < 1.113))[0])
    j_band = np.unique(np.where((wave2d >= 1.15) & (wave2d < 1.40))[0])
    h_band = np.unique(np.where((wave2d >= 1.47) & (wave2d < 1.85))[0])
    k_band = np.unique(np.where((wave2d >= 1.91))[0])
    
    if cut is True:
        k_band = np.delete(k_band, np.array([1,2,5]))
    
    band_idx = []
    
    if ('y' in band_list) | ('Y' in band_list):
        band_idx.append(y_band)
    if ('j' in band_list) | ('J' in band_list):
        band_idx.append(j_band)
    if ('h' in band_list) | ('H' in band_list):
        band_idx.append(h_band)
    if ('k' in band_list) | ('K' in band_list):
        band_idx.append(k_band)
        
    flat_list = [item for sublist in band_idx for item in sublist]
    
    if band_list == 'all':
        flat_list = np.arange(0,49)
    
    return np.array(flat_list)


def remove_values_from_array(array, values_to_delete):
    index_to_delete = []
    for val in values_to_delete:
        if val in array:
            index_to_delete.append(np.where(val == np.array(array))[0][0])

    new_array = np.delete(array, index_to_delete)
    return new_array


def ord_frac_tresh(tr, frac, exp=1):
    return list(np.where(tr.N_frac < frac**exp)[0])    


def select_orders(tr, tresh, select_bands='yjhk', del_ord=[], add_ord=[], exp=1, verbose=True):
    
    if select_bands != 'co':
        orders = list(remove_values_from_array(bands(tr.wv, select_bands), \
                                         del_ord + ord_frac_tresh(tr, tresh, exp=exp))) + add_ord
    else:
        orders = list(remove_values_from_array([30,31,32,33,46,47,48], del_ord)) + add_ord
    if verbose:
        print('Using {:.2f}% of orders, i.e. {}/49'.format(100*len(orders)/49, len(orders)))
    return orders

def remove_means(xxx, choice):

    xxx_mean = xxx - np.mean(xxx, axis=0)
    yyy = xxx - np.ma.mean(xxx, axis=-1)[:, :, None]
    yyy_mean = yyy - np.mean(yyy, axis=0)[None, :, :]

    zzz = [xxx_mean, yyy, yyy_mean]

    return zzz[choice]


##################################    
    
from scipy import signal
def pseudo_cont(x, y, kWidth, wSize):
    # kWidth is the filter width for the inital smoothing
    win = signal.hann(kWidth)
    ysmooth = np.convolve(y, win, mode='same') / win.sum()
    #wSize is the width used to find the local maxima
    iMax,_ = sp.signal.find_peaks(ysmooth, distance=wSize)
    if iMax.size < 2:
        return np.ones_like(x) * np.nan
#     iMax = sp.signal.argrelextrema(ysmooth, np.greater, order=wSize)
    akimafunction = sp.interpolate.Akima1DInterpolator(x[iMax], ysmooth[iMax])

    return akimafunction(x)

def remove_pseudo_cont(wave_grid, flux, kWidth=5, wSize=31):

    flux_pc = np.ones_like(flux) * np.nan

    for iOrd in range(wave_grid.shape[0]):

        if flux[iOrd].mask.all():
            continue

        flux_pc[iOrd] = pseudo_cont(wave_grid[iOrd], flux[iOrd], kWidth, wSize)
    
    return np.ma.masked_invalid(flux_pc)


def make_model_pseudo_cont(modelWave0, modelTD0, wave_grid, spec, kind='pc', norm=False):

    model = gen_mod(modelWave0, 1-modelTD0, wave_grid)
    model = np.ma.array(model, mask=~np.isfinite(model))
    
    if kind == 'median':
        model_flat = model/median_filter(model[None,:,:])
    elif kind == 'pc':   
        model_flat = remove_pseudo_cont(wave_grid, model)  
    else:
        model_flat = model

    if norm is True:
        model_flat = quick_norm(model_flat, mean=False).squeeze()
        
    weight_ord = np.trapz(np.abs(np.ma.array(model_flat, mask=spec[0].mask)), axis=-1)
 
    return model_flat, weight_ord


def pseudo_cont_spectrum(x,y, prominence=None, plot=False,**kwargs):
    
    peaks,_ = sp.signal.find_peaks(y,prominence=(None, prominence))

    akima_fct = sp.interpolate.Akima1DInterpolator(x[peaks], y[peaks])

    akima_y = np.ma.masked_invalid(akima_fct(x))

    filter_peak = median_filter(akima_y[None,None,:], **kwargs).squeeze()
    
    if plot is True:
        plt.figure(figsize=(5,2))
        plt.plot(x, y)
        plt.plot(x[peaks], y[peaks],'r.')
        plt.plot(x, akima_y, alpha=0.5)
        plt.plot(x, filter_peak)
    
    return filter_peak


def resamp_model(modelWave0, modelTD0, Rbf, Raf=70000, pix_per_elem=2,rot_ker=None, sample=None, **kwargs): #, binning=False):
    
    if ~isinstance(modelTD0, np.ma.MaskedArray):
        modelTD0 = np.ma.masked_invalid(modelTD0)
        
    if sample is None:
        sample = modelWave0
        

#     R_more = Raf #np.ones_like(modelWave0) * Raf  # get_var_res(iOrd, x)
    modwave0, modelTD0_resamp = resampling(modelWave0, modelTD0, Raf=Raf, Rbf=Rbf, sample=sample,
                                           rot_ker=rot_ker, **kwargs)

    
#     new_lb_420k, modelTD0_oversamp = resampling(modelWave0, modelTD0, Raf=70000*2*3, Rbf=Rbf)
#     modelTD0_smooth = box_binning(modelTD0_oversamp, 3) #3 avec 420000

    try:
        modelTD0_smooth = box_binning(modelTD0_resamp, Rbf/Raf/pix_per_elem) #3 avec 420000
    except ValueError:
        modelTD0_smooth = modelTD0_resamp

#     if binning_wv is not None:
#         modelTD0_resamp = binning_model(binning_wv, modwave0, modelTD0_resamp)

    return np.ma.masked_invalid(modelTD0_smooth)
    
def normalize_model(modelWave0, modelTD0_resamp, norm=True, switch=False, somme=False, \
              plot=False, med_filt=None, **kwargs):
    
    if med_filt == 'filter':
        print(kwargs)
        filter_mod = median_filter(modelTD0_resamp[None,None,:], **kwargs).squeeze()
    elif med_filt == 'pseudo_cont':
        filter_mod = pseudo_cont_spectrum(modelWave0, modelTD0_resamp, plot=plot, **kwargs)
    else:
        filter_mod = np.nanmedian(modelTD0_resamp)*np.ones_like(modelWave0)
        if np.isnan(filter_mod).all():
            filter_mod = np.ma.median(modelTD0_resamp)*np.ones_like(modelWave0)
    modelTD0_norm = -modelTD0_resamp/filter_mod
    
    modelTD0_norm = np.ma.masked_invalid(modelTD0_norm)
    if plot is True:
        fig, ax = plt.subplots(2,1, sharex=True)
#         ax[0].plot(modelWave0, modelTD0)
        ax[0].plot(modelWave0, modelTD0_resamp)
        ax[0].plot(modelWave0, filter_mod)

    if switch is True:
        modelTD0_norm = 1 - modelTD0_norm

    if norm is True:
        modelTD0_norm = quick_norm(modelTD0_norm[None,None,:], somme=somme, take_all=False).squeeze()
        if plot is True:
            _ = ax[1].plot(modelWave0, modelTD0_norm, 'c', alpha=0.2)
        
    return modelTD0_norm   

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
def gaussian(amplitude, xo, yo, sigma_x, sigma_y, offset):
    """Returns a gaussian function with the given parameters"""
    sigma_x = float(sigma_x)
    sigma_y = float(sigma_y)
    return lambda x,y: offset+amplitude*np.exp(
                -(((xo-x)/sigma_x)**2+((yo-y)/sigma_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    try:
        col = data[:, int(y)]
    except:
        print('Could not calculate the y position')
        col = data[:, int(data.shape[1]-1)]    
    width_x = np.sqrt(np.ma.sum((np.arange(col.size)-y)**2*col)/col.sum())
    
    try:
        row = data[int(x), :]
    except:
        print('Could not calculate the x position')
        row = data[int(data.shape[0]-1), :]
    width_y = np.sqrt(np.ma.sum((np.arange(row.size)-x)**2*row)/row.sum())
    height = data.max()
    
    return height, x, y, width_x, width_y

def gaussian2(amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """Returns a gaussian function with the given parameters"""
    sigma_x = float(sigma_x)
    sigma_y = float(sigma_y)
    return lambda x,y: offset + amplitude*np.exp( - ((np.cos(theta)**2)/(2*sigma_x**2) + \
                                                     (np.sin(theta)**2)/(2*sigma_y**2)*((x-xo)**2) + \
                                                     2*(-(np.sin(2*theta))/(4*sigma_x**2) + \
                                                        (np.sin(2*theta))/(4*sigma_y**2))*(x-xo)*(y-yo) 
                            + ((np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2))*((y-yo)**2)))

def fitgaussian(data, get_theta=False):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)#+(-10,0)
    
    if get_theta is True:
        params += (-10,0)
        errorfunction = lambda p: np.ravel(gaussian2(*p)(*np.indices(data.shape)) - data)
    else:
        params += (0,)
        errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
    all_output = sp.optimize.leastsq(errorfunction, params, full_output=1)
#     print(all_output)
    popt, cov_x = all_output[0], all_output[1]
    res = np.ravel(data) - all_output[2]['fvec']
    pcov = np.var(res)*cov_x
    err_param = np.sqrt(np.diag(pcov))
    
    return popt, err_param


def calc_snr_1d(ccf, rv_grid, vrp, limit_bruit=8, limit_shift=60,
                interp_size=201, RV_sys=0):

    if isinstance(RV_sys, u.Quantity):
        RV_sys = (RV_sys.to(u.km / u.s)).value

    interp_grid = np.linspace(-limit_shift + RV_sys, limit_shift + RV_sys, interp_size).squeeze()
    shifted_test = np.ma.masked_invalid(shift_correl(interp_grid, ccf, rv_grid, vrp))
    
    courbe = np.ma.sum(shifted_test, axis=0)

    idx_bruit = (interp_grid < RV_sys - limit_bruit) | (interp_grid > RV_sys + limit_bruit)
    bruit = np.ma.std(courbe[idx_bruit])
    snr = (courbe - np.ma.mean(courbe[idx_bruit])) / np.ma.masked_invalid(bruit)

    return np.ma.masked_invalid(shifted_test), \
        interp_grid, \
        np.ma.masked_invalid(courbe), \
        np.ma.masked_invalid(snr), \
        idx_bruit


def calc_snr_2d(ccf, rv_grid, vrp, Kp, nu, w, limit_shift=60,
                interp_size=201, RV_sys=0, kp0=0, kp1=2, Kp_slice=None):
    if isinstance(RV_sys, u.Quantity):
        RV_sys = (RV_sys.to(u.km / u.s)).value
    if Kp_slice is None:
        Kp_slice = Kp

#     print(-limit_shift+2*vrp.value[0]+RV_sys, limit_shift+2*vrp.value[-1]+RV_sys)
    interp_grid = np.linspace(-limit_shift - 2 * vrp.value[0] + RV_sys,
                              limit_shift - 2 * vrp.value[-1] + RV_sys, interp_size).squeeze()

    Kp_array = np.arange(kp0, int(Kp.value * kp1))

    sum_ccf = np.ones((Kp_array.size, interp_grid.size))
    for i, Kpi in enumerate(Kp_array):
        hm.print_static(i)

        vrp_orb = rv_theo_nu(Kpi, nu * u.rad, w, plnt=True)

        shifted_ccf = np.ma.masked_invalid(shift_correl(interp_grid, ccf, rv_grid, vrp_orb.value))

        sum_ccf[i] = np.ma.sum(shifted_ccf, axis=0)

    idx_bruit = (interp_grid < RV_sys - 15) | (interp_grid > RV_sys + 15)
    idx_bruit2 = (Kp_array < Kp.to(u.km / u.s).value - 70) | \
                 (Kp_array > Kp.to(u.km / u.s).value + 70)
    bruit2 = np.ma.std(sum_ccf[idx_bruit2][:, idx_bruit])
    snr2 = (sum_ccf - np.ma.mean(sum_ccf[idx_bruit2][:, idx_bruit])) / np.ma.masked_invalid(bruit2)

    if kp0 < -Kp.value:
        idx_bruit_mKp = (Kp_array >= -Kp.to(u.km / u.s).value - 50) | \
            (Kp_array <= -Kp.to(u.km / u.s).value + 50)
#     idx_bruit_rv_mkp = (interp_grid > RV_sys-15) | (interp_grid < RV_sys+15)

        bruit_mKp = np.nanstd(sum_ccf[idx_bruit_mKp])  # [:,idx_bruit_rv_mkp])
        snr2_mKp = (sum_ccf - np.ma.mean(sum_ccf[idx_bruit_mKp])) / np.ma.masked_invalid(bruit_mKp)
        snr_mKp = snr2_mKp[hm.nearest(Kp_array, Kp.value)].squeeze()
        print('SNR à Kp, bruit moins Kp = ', snr_mKp[~idx_bruit].max())

#     sum_ccf_nonoise = sum_ccf[~idx_bruit2][:,~idx_bruit]
#     idx_max2 = np.where(sum_ccf_nonoise == sum_ccf_nonoise.max())
#     idx_min2 = np.where(sum_ccf_nonoise == sum_ccf_nonoise.min())

    snr_fct2d = interp2d(interp_grid, Kp_array, snr2)

    snr = snr_fct2d(interp_grid, Kp_slice.value)  # snr2[hm.nearest(Kp_array, Kp_slice.value)].squeeze()
    courbe = sum_ccf[hm.nearest(Kp_array, Kp_slice.value)].squeeze()

    return interp_grid, Kp_array, \
        np.ma.masked_invalid(sum_ccf), \
        np.ma.masked_invalid(snr2), \
        idx_bruit, idx_bruit2, \
        np.ma.masked_invalid(courbe), \
        np.ma.masked_invalid(snr), snr_fct2d



