from astropy.io import fits
from starships.list_of_dict import *
import numpy as np
from starships import homemade as hm
from scipy.ndimage.filters import gaussian_filter1d #, median_filter
# from spirou_exo.transpec import median_filter
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian1DKernel


def read_all_sp(path, file_list, wv_default=None, blaze_default=None,
                blaze_path=None, debug=False):
    #'20180805_2295520f_pp_blaze_AB.fits'

    headers, count, wv, blaze = list_of_dict([]), [], [], []
    blaze_path = blaze_path or path

    with open(path + file_list) as f:

        for file in f:

            hdul = fits.open(path + file.split('\n')[0])
            headers.append(hdul[0].header)
            count.append(hdul[0].data)

            wv_file = wv_default or hdul[0].header['WAVEFILE']
            try:
                blaze_file = blaze_default or build_blaze_file(hdul[0].header)
            except KeyError:
                blaze_file = hdul[0].header['CDBBLAZE']

            if debug:
                print(file.split('\n')[0], wv_file, blaze_file)

            with fits.open(path + wv_file) as f:
                wv.append(f[0].data / 1000)
            
            with fits.open(blaze_path + blaze_file) as f:
                blaze.append(f[0].data)

    return headers, np.array(wv), np.array(count), np.array(blaze)

def read_all_sp_CADC(path, file_list):
    
    headers_princ, headers_image, headers_tellu = list_of_dict([]), list_of_dict([]), list_of_dict([])
    count, wv, blaze, recon = [], [], [], []

    with open(path + file_list) as f:

        for file in f:
           # print(file)

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
            np.array(count), np.array(blaze), np.array(recon)



def build_blaze_file(header):

    frame = header['WAVEFILE'].split('.fits')[0].split('_')
    blaz = header['BLAZFILE'].split('.fits')[0].split('_')

    return '_'.join([frame[0], *blaz, 'blaze', frame[-1]]) + '.fits'


def norm_median(flux):

    shape = flux.shape
    flux = flux.reshape(-1, shape[-1])
    flux_norm = np.array([f / np.ma.median(f) for f in flux])

    return flux_norm.reshape(shape)


def filter_norm(flux, g_sigma, med_size, g_kwargs={}, med_kwargs={}):

    try:
        n, nOrd, _ = flux.shape
    except ValueError:
        flux = flux.copy()[None, :, :]
        n, nOrd, _ = flux.shape

    def f_g(x): return gaussian_filter1d(x, g_sigma, **g_kwargs)

    def f_med(x): return median_filter(x, med_size, **med_kwargs)

    def big_filter(x): return f_g(f_med(x))

    out = []
    for iOrd in range(nOrd):
        hm.print_static(iOrd)
        f_in = np.ma.array(flux[:, iOrd].copy())
        fill = np.ma.median(f_in, axis=-1)
        norm_vec = [big_filter(
                    f_in[i].filled(fill_value=fill[i])
                    ) for i in range(n)]
        out.append(np.nanmedian(norm_vec, axis=0))

    return np.ma.array(out, mask=np.isnan(out))


from scipy import ndimage

def median_filter(flux, box=201, gauss_box=5, gauss=True, sp=False):
    n_spec, nord,_ = flux.shape
    flux_filt = np.ones_like(flux) * np.nan
    for iOrd in range(nord):
        hm.print_static(iOrd)
        if sp is False:
            flux_filt[:,iOrd] = running_filter(flux[:,iOrd], np.nanmedian, box)
        else:
            flux_filt[:,iOrd] = ndimage.median_filter(flux[:,iOrd], box)
        if gauss is True:
            flux_filt[:,iOrd] = [convolve(y, Gaussian1DKernel(gauss_box), boundary='extend', mask=y.mask, 
                                      preserve_nan=True) for y in flux_filt[:,iOrd]]

    return np.ma.array(flux_filt, mask=~np.isfinite(flux_filt))


def quick_norm(chose0, mean=True, somme=True, take_all=True):

    if mean is True:
        chose1 = chose0-np.ma.mean(chose0, axis=-1)[:,:,None]
    else:
        chose1 = chose0

    if somme is True:
        if take_all is True:
            chose2 = chose1/np.sqrt(np.ma.sum(np.concatenate(chose1.T).T**2, axis=-1)[:,None,None])  #[:,:,None]
        else:
            chose2 = chose1/np.sqrt(np.ma.sum(chose1**2, axis=-1))[:,:,None]  #
    else:
        chose2 = chose1

    return chose2


################################

# ------ Running filter --------
def running_filter_1D(data, func, length, cval=np.nan, fargs=(), fkwargs={}, verbose=True):
    if isinstance(data, np.ma.MaskedArray):
        data = data.filled(fill_value=np.nan)
    if length % 2 != 1:
        raise ValueError("length should be odd.")
    n = len(data)
    sides = int((length - 1)/2)
    data_ext = np.pad(data, sides, "constant", constant_values=cval)
    # Generate index for each interval to apply filter
    # Same as [np.arange(i,i+length) for i in range(n)] but quicker
    index = np.arange(length)[None,:] + np.arange(n)[:,None]
    try:
        out = func(data_ext[index], *fargs, axis=1, **fkwargs)
    except TypeError as e:
        if verbose:
            print(e, "so may take longer.")
            print("Consider using numpy func with axis keyword.")
        out = [func(x, *fargs, **fkwargs) for x in data_ext[index]]
#     mask = ~np.isfinite(out) | np.isnan(data)
    return out

def running_filter_1D_by_part(data, func, length, part=int(3e6), **kwargs):
    n = len(data)
    step = int(part/length)
    out = [running_filter_1D(data[i:i+step], func, length, **kwargs)
           for i in range(0, n, step)]
    return np.concatenate(out)


def running_filter(data, func, length, cval=np.nan, **kwargs):
    if isinstance(data, np.ma.MaskedArray):
        data = data.filled(fill_value=np.nan)
    # Make sure we do not exceed memory
    if length * data.shape[-1] > 2e6:
        filter_1D = running_filter_1D_by_part
    else:
        filter_1D = running_filter_1D
    if data.ndim>1:
        out = [filter_1D(dat, func, length, cval=cval, **kwargs)
               for dat in data]
    else:
        out = filter_1D(data, func, length, cval=cval, **kwargs)
    mask = ~np.isfinite(out) | np.isnan(data)
    return np.ma.array(out, mask=mask)


def rm_star_median(wave, flux_t, dv_star, iOut, nOrd,
                   icorr=None, iWaveGood=slice(None)):

    specTr = np.ma.empty_like(flux_t)
    for iOrd in nOrd:
        hm.print_static('{}/{}'.format(iOrd + 1, nOrd.size))
        specTr[:, iOrd, :] = \
            _rm_star_median(wave[iOrd], flux_t[:, iOrd, :], dv_star, iOut,
                            icorr=icorr, iWaveGood=iWaveGood)

    return specTr


def _rm_star_median(wv, flux_t, dv_star, iOut, icorr=None, iWaveGood=slice(None)):

    if icorr is None:
        icorr = np.arange(dv_star.size)

    # loop through all spectra and divide out the star
    specTr = np.ma.empty((icorr.size, wv.size))  # the transit spectra
    starThisVarr = np.ma.empty((iOut.size, wv.size))

    for n, i in enumerate(icorr):
        #         hm.print_static('{}/{}'.format(n+1,icorr.size))

        # Build star spectrum at RV using out-of-transit spectra
        for k, j in enumerate(iOut):
            # Shift spectrum j to RV i
            starThisVarr[k, :], sh = \
                hm.doppler_shift(wv, wv, flux_t[j, :],
                                 (dv_star[i] - dv_star[j]))

        # Normalize each spectrum by its median ...
        starThisVarr /= np.ma.median(starThisVarr[:, iWaveGood], axis=1)[:, None]

        # ... then take the median of all spectra ...
        starThisV = np.ma.median(starThisVarr, axis=0)

        # ... and re-normalize
        starThisV /= np.ma.median(starThisV[iWaveGood])

        # divide out the star
        specTr[n, :] = flux_t[i, :] / np.ma.median(flux_t[i, iWaveGood]) \
            - starThisV

    return specTr


def get_res(iOrd):

    ires = iOrd // 10

    # -- First version / file from 3 oct 2018
#     res = np.array([[62962, 64506, 65249, 63351],
#                     [62100, 66876, 66845, 64218],
#                     [62236, 66969, 66994, 65270],
#                     [61443, 63776, 64192, 65239],
#                     [56956, 59075, 60687, 61438]])

    # -- HC-HC lamp on August 5th 2018 -- 
    res = np.array([[62683, 62862, 63224, 63209],
                    [62397, 63550, 64131, 64041],
                    [61721, 63594, 64461, 65012],
                    [61798, 63353, 64784, 65005],
                    [62135, 62570, 64656, 66393]])

    res = res[:, 1:3].mean(axis=1)

    return res[ires]


def get_var_res(iOrd, x, plot=False):
    ordre_num = np.arange(4)
    ordre_res = np.array([[62683, 62862, 63224, 63209],
                        [62397, 63550, 64131, 64041],
                        [61721, 63594, 64461, 65012],
                        [61798, 63353, 64784, 65005],
                        [62135, 62570, 64656, 66393]])

    fit_ord = poly_fct(ordre_num, np.ma.masked_invalid(ordre_res[iOrd // 10]), 2)
    x_more = np.linspace(0, 3, x.shape[0])
    
    if plot is True:
        plt.figure()
        plt.plot(ordre_num, ordre_res.T)
        plt.plot(x_more, fit_ord(x_more),'r')

    return fit_ord(x_more)


def get_w_fcts(wave, count, order=3):

    weight = count / count.sum(axis=0)

    n, nOrd, _ = count.shape

    w_fcts = []
    for i in range(n):
        fit_w = []

        for iOrd in range(nOrd):
            fit_w.append(poly_fct(wave[iOrd], weight[i][iOrd, :], order))

        w_fcts.append(fit_w)

    return arr_of_fcts(w_fcts)


class arr_of_fcts(np.ndarray):

    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # Finally, we must return the newly created object:
        return obj

    def __call__(self, x_vect):

        fcts = self.ravel()
        nfcts = fcts.size
        try:
            outshape = x_vect.shape
            x = x_vect.reshape(nfcts, -1)
        except ValueError:
            x = np.tile(x_vect, (self.shape[0], 1, 1))
            outshape = x.shape
            x = x.reshape(nfcts, -1)
        out = np.empty_like(x)
        for i in range(nfcts):
            out[i, :] = np.array(fcts[i](x[i]))

        return out.reshape(outshape)


def rm_star_sum(wave, flux_norm, dv_star, w_fcts, iOut, icorr=None, iWaveGood=slice(None), nOrd=None):
    '''
    Compute the stellar spectrum at each exposure
    by Doppler-shifting the out-of-transit spectra.
    The spectra are then weighted-summed.
    '''

    if icorr is None:
        icorr = np.arange(dv_star.size)

    if nOrd is None:
        nOrd = np.arange(wv.shape[0])

    # loop through all spectra and divide out the star
    specTr = []  # the corrected spectra

    for n, i in enumerate(icorr):
        hm.print_static('{}/{}'.format(n + 1, icorr.size))
        specTr_n = np.empty_like(flux_norm[i])

        # Iterate on each orders
        for iOrd in nOrd:
            wv = wave[iOrd]
            starThisVarr = np.empty((iOut.size, wv.size))
            # Build star spectrum at RV using out-transit spectra
            for k, j in enumerate(iOut):
                # Shift spectrum j to RV i
                starThisVarr[k, :], sh = \
                    hm.doppler_shift(wv, wv, flux_norm[j, iOrd, :],
                                     (dv_star[i] - dv_star[j]))
                starThisVarr[k] *= w_fcts[j, iOrd](wv)

            # ... then sum all spectra ...
            starThisV = np.ma.sum(starThisVarr, axis=0)

            # ... and re-normalize
            starThisV /= np.ma.array([fct(wv) for fct in w_fcts[iOut, iOrd]]).sum(axis=0)

            # divide out the star
            specTr_n[iOrd, :] = flux_norm[i, iOrd, :] / starThisV

        specTr.append(specTr_n)

    print('\r done')

    return np.ma.array(specTr)


from operator import methodcaller
import warnings


def poly_fct(x, y, order, w=None):
    if (y.mask.all()) or (y.mask.sum() > y.shape[0]*0.9):
        return f_nan
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try :
                
                coeff = np.ma.polyfit(x, y, order, w=w)
            except RankWarning:
                plt.figure()
                plt.plot(x,y)
                print(x.shape, y.shape, order)
                fct = np.poly1d(coeff)
                plt.plot(x,fct(x))
                hm.stop()  
#                 print('Bad fit, using a linear fit instead...')
                coeff = np.ma.polyfit(x, y, 1, w=w)
  
        return np.poly1d(coeff)
    except TypeError:
        if y.mask.all():
            return f_nan


def f_nan(x):
    return np.ones_like(x) * np.nan


def col_remove(flux):

    shape = flux.shape
    x = np.arange(shape[0])
    flux = flux.reshape(shape[0], -1).swapaxes(0, -1)
#     print(x.shape, flux.shape)
    # Define functions
    func = [poly_fct(x, f, 2) for f in flux]

    # Apply all functions to x
    poly = list(map(methodcaller('__call__', x), func))

    # Convert to array with good shape and substract from flux
    out = flux/np.ma.array(poly)

    # Return with good shape
    return out.swapaxes(0, 1).reshape(shape)

def get_mask_tell(tell, tresh, pad_tresh):
    
    if tell.mask.all():
        return tell.mask
    
    new_mask = (tell < tresh)
    
    while True:
        temp_mask = np.convolve(new_mask, [1,1,1], 'same').astype(bool)
        temp_mask = (temp_mask & (tell < pad_tresh))
        if (temp_mask == new_mask).all():
            break
        new_mask = temp_mask
        
    return new_mask


from scipy.optimize import least_squares
from astropy.convolution import convolve, Gaussian1DKernel

def polyfit_robust(y, order, x=None, loss='soft_l1', f_scale=0.01, **kwargs):
    
    if x is None:
        x = np.indices(y.shape)[0]
        
    guess = np.polyfit(x, y, order)
    f_res = lambda coeff:(y - np.poly1d(coeff)(x))/y
    coeff = least_squares(f_res, guess, loss=loss, f_scale=f_scale, **kwargs).x
    
    return coeff

def poly_out(y, order, x=None, ind_fit=slice(None), **kwargs):
    if x is None:
        x = np.indices(y.shape)[0]
    coeff = polyfit_robust(y[ind_fit], order, x=x[ind_fit], **kwargs)
    
    return np.poly1d(coeff)(x)



def get_mask_noise(flux_array, tresh, pad_tresh, gwidth=1, poly_ord=7, 
                   iplot=False, inverse=False, noise=None):
    if noise is None:
        # Noise is the std dev along time axis
        noise = np.ma.std(flux_array,axis=0)
    # Don’t compute anything if all masked
    if (noise.mask.all()) or (4088-noise.mask.sum() < 41):
        return noise.mask
    
#     if iplot:
#         fig, ax = plt.subplots(3,1, sharex=True)
#         ax[0].plot(np.ma.median(flux_array, axis=0))
#         ax[0].set_ylabel('Median flux')
#         ax[1].plot(noise,'k', label='Noise')
#         ax[2].pcolormesh(flux_array)
    # Remove the ‘continuum’ noise (increasing on order sides)
#     if noise.ndim > 1:
#         noise_floor = np.zeros_like(noise)
#         for n in range(noise.shape[0]):
#             noise_floor[n] = poly_out(noise[n], poly_ord, ind_fit=~noise[n].mask)
#     else:
    noise_floor = poly_out(noise, poly_ord, ind_fit=~noise.mask)
    # Plot noise
    if iplot:
        fig, ax = plt.subplots(3,1, sharex=True)
        ax[0].plot(np.ma.median(flux_array, axis=0))
        ax[0].set_ylabel('Median flux')
        ax[1].plot(noise,'k', label='Noise')
        ax[1].plot(np.ma.array(noise_floor, mask=noise.mask),'c', label="Noise floor")
        ax[1].legend()
        ax[1].set_ylabel('Noise [absolute units]')
    # Convert noise in std dev units
    sig = np.ma.std(noise - noise_floor)
    noise /= sig
    # Re-compute ‘continuum’ in std dev units
    noise_floor = poly_out(noise, poly_ord, ind_fit=~noise.mask)
    # Plot
    if iplot:
        ax[2].plot(noise-noise_floor, alpha=0.5, label='Noise - noise floor')
    # Smooth the noise with a Gaussian
    noise = convolve(noise-noise_floor,
                     Gaussian1DKernel(gwidth), boundary='extend',
                     mask=noise.mask,
                     preserve_nan=True)
    noise = np.ma.masked_invalid(noise)
    # Mask using the same technique as tellurics
    # Peaks and wings
    if inverse is False:
        mask = get_mask_tell(-1*noise, -1*tresh, -1*pad_tresh)
    else:
        mask = get_mask_tell(noise, tresh, pad_tresh)
    # Plot result
    if iplot:
        ax[2].plot(noise,':', label='Smoothed')
        ax[2].plot(np.ma.array(noise, mask=mask),'g', label='Masked')
        ax[2].hlines(tresh, *plt.gca().get_xlim(), linestyle='--')
        ax[2].hlines(pad_tresh, *plt.gca().get_xlim(),'g', linestyle='--')
        ax[2].legend()
        ax[2].set_xlabel('Pixel')
        ax[2].set_ylabel('Noise [$\sigma$]')
    return mask

# def get_mask_noise(flux_array, tresh, pad_tresh, gwidth=1, poly_ord=2, iplot=False):
 # --- Old version   
#     noise = np.ma.std(flux_array,axis=0)
    
#     if noise.mask.all():
#         return noise.mask
    
    
#     noise_floor = poly_out(noise, poly_ord, ind_fit=~noise.mask)
#     if iplot: 
#         plt.figure()
#         plt.plot(noise-noise_floor, alpha=0.5)
    
#     noise = convolve(noise-noise_floor,
#                      Gaussian1DKernel(gwidth), boundary='extend',
#                      mask=noise.mask,
#                      preserve_nan=True)
#     noise = np.ma.masked_invalid(noise)
    
#     mask = get_mask_tell(-1*noise, -1*tresh, -1*pad_tresh)
    
#     if iplot:
#         plt.plot(noise,':')
#         plt.plot(np.ma.array(noise, mask=mask),'g')
    
#     return mask
    
# Exemple
# new_mask = [get_mask_noise(f, 0.007, 0.002) for f in flux_norm.swapaxes(0,1)]
# new_mask = new_mask | flux_norm.mask
# flux_norm = np.ma.array(flux_norm, mask=new_mask)


# --- Robust Polyfit, from David Lafrenière
from scipy.optimize import least_squares
def poly(x,p):
    return np.poly1d(p)(x)
def fit_resFunc(p,x,y):
    return poly(x,p)-y
def robust_polyfit(x,y,p0):
    res = least_squares(fit_resFunc, p0, loss='soft_l1', f_scale=0.1, args=(x,y))
    return res.x
