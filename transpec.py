import starships.extract as ext
from starships import analysis as a
from starships import homemade as hm
from starships.extract import quick_norm, running_filter 

import numpy as np
import scipy.constants as cst
from astropy import units as u
from astropy import constants as const
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian1DKernel
from scipy.interpolate import interp1d

# from spirou_exo.plotting_fcts import plot_order
from numpy.polynomial.polynomial import polyval

from itertools import groupby
from operator import itemgetter



def mask_deep_tellu(flux, tellu=None, path=None, tellu_list='list_tellu_tr', 
                    limit_mask=0.5, limit_buffer=0.98, plot=False, new_mask_tellu=None):
    
    n_spec, nord, _ = flux.shape
    
    if new_mask_tellu is None:
        if tellu is None:
            _, wave_tell, tellu, _ = ext.read_all_sp(path, tellu_list, wv_default="MASTER_WAVE.fits")
        tellu = np.ma.array(tellu, mask = ~np.isfinite(tellu))

        # Find the strong tellurics with a pad around in the order
        new_mask_tellu = np.empty_like(tellu)
        for iOrd in range(nord):
            new_mask = [ext.get_mask_tell(tell, limit_mask, limit_buffer) for tell in tellu[:,iOrd,:]]
            new_mask = new_mask | flux[:,iOrd,:].mask
            new_mask_tellu[:,iOrd,:] = new_mask

    flux_masked = np.ma.array(flux.copy(), mask=new_mask_tellu)
    
    if plot is True:
        iord=35
        plt.figure()
        plt.plot(wave[0,iord,:], flux[0,iord,:]/np.nanmedian(flux[0,iord,:]), 'k')
        plt.plot(wave[0,iord,:], flux_masked[0,iord,:]/np.nanmedian(flux[0,iord,:]),'r')
        # plt.plot(wave[0,iord,:], flux_norm[0,iord,:],'orange')
        plt.plot(wave_tell[0,iord,:], tellu[0,iord,:],'b',alpha=0.5)
    
    return flux_masked

from spirou_exo.utils.mask_tools import interp1d_masked
def unberv(wave, flux_masked, berv, RV_sys, vr, clip=False):  #, norm=False
    
    n_spec, nord, _ = flux_masked.shape
    
    if isinstance(berv, u.Quantity):
        berv = berv.to(u.km/u.s).value
    if isinstance(RV_sys, u.Quantity):
        RV_sys = RV_sys.to(u.km/u.s).value
    if isinstance(vr, u.Quantity):
        vr = vr.to(u.km/u.s).value

#     if clip is True:
#         # -- Sigma clipping of the spectra
#         flux_masked = sigma_clip(flux_masked, axis=0, sigma=4)

    flux_Sref = np.ones_like(flux_masked) * np.nan
    
    
    shifts = hm.calc_shift(-(berv+RV_sys+vr), kind='rel')
    print('')
    for n in range(n_spec):
        for iOrd in range(nord):
            hm.print_static(' Unberv : {} - {}  '.format(iOrd, n))

            if flux_masked[n,iOrd].mask.all():
                continue
                
            # - Interpolate over the orginal data
            fct = interp1d_masked(wave[n,iOrd], np.ma.masked_invalid(flux_masked[n,iOrd]), \
                                  kind='cubic', fill_value='extrapolate')

            # - Evaluate it at the shifted grid
#             if dv.ndim > 0:
            flux_Sref[n,iOrd] = fct(wave[n,iOrd]/shifts[n])
#             else:
#                 yp = fct(x/shifts)

#             flux_Sref[n,iOrd] = hm.doppler_shift2(wave[n,iOrd], np.ma.masked_invalid(flux_masked[n,iOrd]), \
#                                                   -berv[n]-RV_sys-vr[n], scale=False)
    print('')
    return np.ma.masked_invalid(flux_Sref)


def build_master_out(wave, flux_Sref_norm, iOut, kind_lp='filter', 
                     box=201, gauss_box=5, master_out=None, kind_mo='median', 
                     clip_ratio=None, cont=False): #, light_curve
    
    nspec, nord, _ = flux_Sref_norm.shape
    
    if master_out is None:
        # --- Building the master out-of-transit spectrum
        if kind_mo == "median":
            master_out = np.ma.median(flux_Sref_norm[iOut],  axis=0)
        elif kind_mo == "mean":
            master_out = np.ma.mean(flux_Sref_norm[iOut],  axis=0)
    #     master_out = np.ma.average(flux_Sref_norm[iOut], weights=light_curve[iOut], axis=0)
        
    if cont is True:
        master_out_ratio = a.remove_pseudo_cont(wave[0], np.clip(master_out.copy(),0.95,None), 
                                                kWidth=30, wSize=55)
        master_out = master_out.copy()/master_out_ratio
#         master_out_ratio = a.remove_pseudo_cont(wave[0], master_out.copy(), 5, 31)
#         master_out = master_out/master_out_ratio
        
#     plt.figure()
#     plt.plot(wave[0],master_out.T)
    # plt.show()

    # -- Polynome fiting on the ratio of spec/master

    ratio  = np.ma.array(flux_Sref_norm/master_out)

#     fit = np.ma.array(np.ones_like(ratio)) * np.nan
    ratio_filt = np.ma.array(np.ones_like(ratio)) * np.nan
#     for n in range(n_spec):
    if kind_lp == 'poly' or kind_lp == 'filter':
        for iOrd in range(nord):
            hm.print_static(iOrd)

            # -- normalizing the spectra to get them at the same level as the master_out
            if kind_lp == 'poly':
                for n in range(nspec):           
                    fit_fct = ext.poly_fct(wave[n, iOrd], ratio[n, iOrd], 4)
                    ratio_filt[n, iOrd] = fit_fct(wave[n, iOrd])
                    ratio_filt[n, iOrd].mask = ~np.isfinite(ratio_filt[n, iOrd])
            elif kind_lp == 'filter':
    #         ratio_filt[:,iOrd] = running_filter(ratio[:,iOrd]/ratio[:,iOrd], np.nanmedian, 201)
                ratio_filt[:,iOrd] = running_filter(ratio[:,iOrd], np.nanmedian, box)
                if gauss_box is not None:
                    ratio_filt[:,iOrd] = [convolve(y, Gaussian1DKernel(gauss_box), boundary='extend', mask=y.mask, 
                                                  preserve_nan=True) for y in ratio_filt[:,iOrd]]

#                     if (ratio_filt[:,iOrd] <= 0.75).any() and clip_ratio is not None:
#                         print('ratio_filt has values <= 0.75!')
#                         ratio_filt[:,iOrd] = sigma_clip(ratio_filt[:,iOrd], clip_ratio)
                        
    if (ratio_filt <= 0.75).any() and clip_ratio is not None:
#         plt.plot(wave[17, [22,23,24]].T, ratio_filt[17, [22,23,24]].T,'k.')
        print('ratio_filt has values <= 0.75!')
        ratio_filt = sigma_clip(ratio_filt, clip_ratio)
        
#         plt.plot(wave[17, [22,23,24]].T, ratio_filt[17, [22,23,24]].T,'r.', alpha=0.5)
#         plt.plot(wave[17, [22,23,24]].T, np.ma.masked_invalid(ratio_filt[17, [22,23,24]]).T,'b.', alpha=0.3)
#         hm.stop()
    

    
    flux_norm_mo = flux_Sref_norm/ratio_filt
    
#     plt.figure()
#     plt.plot(wave[0,33], flux_Sref_norm[0,33].T/master_out[33])
#     plt.plot(wave[0,33], ratio_filt[0,33].T)
#     plt.plot(wave[0,33], sigma_clip(ratio_filt[0,33],5))
#     hm.stop()
    
    if master_out is None:
    #     new_master_out = np.ma.average(flux_norm_mo[iOut], weights=light_curve[iOut], axis=0)
        new_master_out = np.ma.median(flux_norm_mo[iOut], axis=0)
    else:
        new_master_out = master_out

    return flux_norm_mo, new_master_out, np.ma.masked_invalid(ratio_filt)


def build_master_out_pc(wave, flux_Sref_norm, iOut, plot=False, **kwargs):
    
    nspec, nord, npix = flux_Sref_norm.shape
    
    flux_norm = np.empty_like(flux_Sref_norm)
    
    for n in range(nspec):
        
        for iOrd in range(nord):
            hm.print_static('   MO_pc    ',n,' - ',iOrd)
            if  flux_Sref_norm[n,iOrd].mask.sum() != npix:
                filter_mod = a.pseudo_cont_spectrum(wave[n,iOrd], flux_Sref_norm[n,iOrd], 
                                                  plot=plot, **kwargs)
            else:
                filter_mod = np.ones_like(flux_Sref_norm[n,iOrd])
            flux_norm[n,iOrd] = flux_Sref_norm[n,iOrd]/filter_mod
            
    flux_norm = np.ma.masked_invalid(flux_norm)
    
    # --- Building the master out-of-transit spectrum

    master_out = np.ma.median(flux_norm[iOut],  axis=0)
    
    if plot is True:
        plt.figure()
        plt.plot(wave[0,35], flux_norm[0,35])
        plt.plot(wave[0,35], master_out[35])

    return flux_norm, master_out


# TRANSMISSION SPECTRUM

from sklearn.decomposition import PCA

# returns the 'n_components' number of principal components of a dataset.  matrix is N x M, returns what is common to the M axis.  
def PCA_decompose(matrix, n_components=None):
    if n_components == None:
        n_components = np.shape(matrix)[1]
    pca = PCA(n_components=n_components)
    coefficients = pca.fit_transform(matrix)
    pcs = pca.components_
    return pca, pcs, coefficients

# removes 'n_pcs' number of principal components from input
def PCA_remove(matrix, pcs, coefficients, n_pcs, kind='rebuilt'):

    comps = pcs[:n_pcs,:]
    rebuilt = 0.0
    for i in range(n_pcs):
        rebuilt += comps[i,:][None,:]*coefficients[:,i][:,None]
    if kind == 'rebuilt':
        return rebuilt #matrix - rebuilt
    if kind == 'remove':
        return matrix - rebuilt


def remove_dem_pca(flux, n_pcs=5, n_comps=10, plot=False, pca0=None):

    n_spec, nord, npix = flux.shape

    flux_cleaned = np.ones_like(flux) * np.nan
    flux_rebuilt = np.ones_like(flux) * np.nan
    if pca0 is None:
        pca_ord = []
    else:
        pca_ord = pca0
        
    for iOrd in range(nord):
#         hm.print_static(iOrd)
        
        if flux[0,iOrd].mask.all():
            pca_ord.append(None)
            continue
        
        flux_norm_ord = flux[:,iOrd]  # /medianes[:,iii, None]
        index = ~flux_norm_ord.mask.any(axis=0)
        if index.sum() == 0:
            continue
        
        matrix = np.log(flux_norm_ord[:,index]).T
        if pca0 is None:
            pca, npcs, coeffs = PCA_decompose(matrix, n_components=n_comps)
            pca_ord.append(pca)
        else:
            if pca_ord[iOrd] is None:
                continue
            npcs, coeffs = pca_ord[iOrd].components_, pca_ord[iOrd].transform(matrix)

        if plot is True:
#             print(np.array(pca_result[0]).shape)
            print(matrix.shape)
            print(' ')
#             print(pca_result[1])
            plt.figure()
            plt.plot(npcs[:,0],'r',alpha=0.9)
            plt.plot(npcs[:,1],'b',alpha=0.9)
#             plt.plot(pca_result[0][:,2],'g',alpha=0.9)
#             plt.plot(pca_result[0][:,3],'k',alpha=0.9)
#             plt.plot(pca_result[0][:,4],'c',alpha=0.9)
        
            hm.stop()
        if matrix.shape[0] == 1:
            continue
        
        rebuilt = PCA_remove(matrix, npcs, coeffs, n_pcs=n_pcs)
        cleaned = np.exp(matrix-rebuilt).T

        flux_new = np.ones_like(flux[:,iOrd]) * np.nan
        flux_new[:,index] = cleaned

        flux_cleaned[:, iOrd] = np.ma.masked_invalid(flux_new)
        flux_rebuilt[:, iOrd, index] = np.ma.masked_invalid(np.exp(rebuilt).T)
        
    return flux_cleaned, flux_rebuilt, pca_ord



def remove_dem_pca_all(flux, n_pcs=5, n_comps=10, plot=False, pca=None):

    n_spec, nord, npix = flux.shape

    flux_cleaned = np.ones_like(flux) * np.nan
    flux_rebuilt = np.ones_like(flux) * np.nan
    
    index = ~flux.mask.any(axis=0)
    flux_reshape = flux.copy()

    flux_reshape = flux_reshape.reshape((n_spec,nord*npix))
    index_reshape = ~flux_reshape.mask.any(axis=0)
    matrix = np.log(flux_reshape[:,index_reshape]).T
#     print(matrix.mask.all())
#     print(np.ma.mean(matrix, axis=-1))
#     print(np.nanmean(matrix, axis=-1))
    
    matrix_mean = np.ma.mean(matrix, axis=-1)
#     matrix_mean = np.nanmean(matrix, axis=-1)
    matrix -= matrix_mean[:, None]  #[None, :]#
    
    if np.isnan(matrix).all():
        print('Matrix given to PCA is all nan : {}, matrix_mean = {}'.format(np.isnan(matrix).all(), matrix_mean))

    if pca is None:
        pca, npcs, coeffs = PCA_decompose(matrix, n_components=n_comps)
    else:
        npcs, coeffs = pca.components_, pca.transform(matrix)

#     if plot is True:
# #             print(np.array(pca_result[0]).shape)
#         print(matrix.shape)
#         print(' ')
# #             print(pca_result[1])
#         plt.figure()
#         plt.plot(npcs[:,0],'r',alpha=0.9)
#         plt.plot(npcs[:,1],'b',alpha=0.9)

#         hm.stop()

    rebuilt = PCA_remove(matrix, npcs, coeffs, n_pcs=n_pcs)
    cleaned = np.exp(matrix-rebuilt).T

    flux_new = np.ones_like(flux_reshape) * np.nan
    flux_new[:,index_reshape] = cleaned
    flux_new = flux_new.reshape((n_spec, nord, npix))
#     flux_cleaned = np.ma.masked_invalid(flux_new)
    
    flux_rebuilt = np.ones_like(flux_reshape) * np.nan
    flux_rebuilt[:,index_reshape] = np.exp(rebuilt+matrix_mean[:, None]).T
    flux_rebuilt = flux_rebuilt.reshape((n_spec, nord, npix))
#     flux_rebuilt = np.ma.masked_invalid(flux_rebuilt)
        
    return np.ma.masked_invalid(flux_new), np.ma.masked_invalid(flux_rebuilt), pca



def build_stacked_st(wave, spec_trans, vr, vrp, weight, kind='average', 
                     iOrd0=None, RV=None, alpha=None):

    n_spec, nord,_ = spec_trans.shape
    spec_trans_Pref = np.ones_like(spec_trans) * np.nan
    
    if RV is None:
        RV=np.zeros_like(vrp)
    if alpha is None:
        alpha=np.ones_like(vrp.value)

    for n in range(n_spec):
        if iOrd0 is None:
            for iOrd in range(nord):

                hm.print_static('    Build_stacked_ts   {} - {}  '.format(n,iOrd))

                if spec_trans[n,iOrd].mask.all():
                    continue

                spec_trans_Pref[n,iOrd] = hm.doppler_shift2(wave[iOrd], 
                                                      spec_trans[n,iOrd], vr[n]-vrp[n]+RV[n],scale=False)
        else:
            hm.print_static('   Build_stacked_ts    {} - {}  '.format(n,iOrd))

            if spec_trans[n,iOrd0].mask.all():
                continue

            spec_trans_Pref[n,iOrd0] = hm.doppler_shift2(wave[iOrd0], 
                                                  spec_trans[n,iOrd0], vr[n]-vrp[n]+RV[n],scale=False)
        
    spec_trans_Pref = np.ma.masked_invalid(spec_trans_Pref)
    
    if kind=='average':
        spec_fin = np.ma.average(spec_trans_Pref, axis=0, weights=weight*alpha)
    elif kind=='mean':
        spec_fin = np.ma.mean(spec_trans_Pref, axis=0)
    elif kind=='median':
        spec_fin = np.ma.median(spec_trans_Pref, axis=0)

    return spec_fin, spec_trans_Pref



def clean_bad_pixels(wave, uncorr0, noise_lim=4, plot=False, t1=None, iOrd=34, tresh=4, tresh_lim=3):
    if plot is True:
        pf.plot_order(t1,iOrd,t1.uncorr)
        
    n_spec, nord, _ = uncorr0.shape

    median_wv = np.nanmedian(np.clip(uncorr0,0,None), axis=-1)[:,:,None]
    uncorr_mean_norm = uncorr0/median_wv
    if plot is True:
        pf.plot_order(t1,iOrd,uncorr_mean_norm)

    median_time = np.nanmedian(uncorr_mean_norm, axis=0)[None,:,:]
    uncorr_norm = uncorr_mean_norm/median_time
    if plot is True:
        pf.plot_order(t1,iOrd,uncorr_norm, cbar=True)

    noise_level = (uncorr_norm-np.nanmedian(uncorr_norm, axis=-1)[:,:,None])/np.nanstd(uncorr_norm, axis=-1)[:,:,None]

    if plot is True:
        pf.plot_order(t1,iOrd, noise_level, cbar=True)
        
    good = np.where((noise_level <= 4) & (noise_level >= -5), uncorr_norm, np.nan)
    master = np.nanmean(good , axis=0)


    clipped_noise = uncorr_norm.copy()
    for n in range(n_spec):
        for iord in range(nord):
            hm.print_static('{}, {}'.format(n, iord))
#             cond = (noise_level[n,iord] >= noise_lim) | (noise_level[n,iord] <= -3)
            cond = np.array(ext.get_mask_noise(noise_level, tresh, tresh_lim, gwidth=0.01, poly_ord=5, 
                                           noise=noise_level[n,iord]), dtype=bool)
            difference = np.diff(np.arange(wave[iord].size)[cond])

            single = np.arange(wave[iord].size)[cond][:-1][difference>1]
            multiple = np.unique([np.arange(wave[iord].size)[cond][:-1][difference==1], 
                                  np.arange(wave[iord].size)[cond][:-1][difference==1]+1])
            try:
                fct_sp=interp1d_masked(wave[iord], master[iord], 
#                     wave[iord][~cond], uncorr_norm[n,iord][~cond], 
                                   kind='cubic', fill_value='extrapolate')
                clipped_noise[n,iord][single] = np.ma.masked_array(fct_sp(wave[iord][single]))
            except ValueError:
                if plot is True:
                    print('chunks are overlapping at {}, {}'.format(n, iord))
#                     print(t1.wv[iord][~cond], uncorr_norm[n,iord][~cond])
                pass
    #         print(fct_sp(t1.wv[iord][single][0]))
            try:
                fct_lin=interp1d_masked(wave[iord], master[iord], 
#                     wave[iord][~cond], uncorr_norm[n,iord][~cond], 
                                    kind='linear', fill_value='extrapolate')
                clipped_noise[n,iord][multiple] = np.ma.masked_array(fct_lin(wave[iord][multiple]))
            except ValueError:
                pass
    
    if plot is True:
        pf.plot_order(t1,iOrd, clipped_noise, cbar=True)
    if plot is True:
        pf.plot_order(t1,iOrd, clipped_noise*median_time, cbar=True)
    if plot is True:
        pf.plot_order(t1,iOrd,t1.fl_norm, cbar=True)
        
    return np.ma.masked_invalid(clipped_noise*median_time)


# from spirou_exo import plotting_fcts as pf
def clean_bad_pixels_time(wave, uncorr0, tresh=3., plot=False, tr=None, iOrd=34, cmap=None, **kwargs):
#     if plot is True:
#         pf.plot_order(tr,iOrd,tr.uncorr, **kwargs)
    n_spec, nord,_ = uncorr0.shape
    
    median_wv = np.ma.median(uncorr0, axis=-1)[:,:,None]
    uncorr_mean_norm = uncorr0/median_wv
#     if plot is True:
#         pf.plot_order(tr,iOrd,uncorr_mean_norm, cmap=cmap, ylabel='uncorr_mean_norm', **kwargs)

    median_time = np.ma.median(uncorr_mean_norm, axis=0)[None,:,:]
    uncorr_norm = uncorr_mean_norm/median_time

#     if plot is True:
#         pf.plot_order(tr,iOrd, uncorr_norm, cbar=True, cmap=cmap, ylabel='uncorr_norm', **kwargs)

    noise_level = np.abs((uncorr_norm-np.ma.median(uncorr_norm, axis=-1)[:,:,None])/np.ma.std(uncorr_norm, axis=-1)[:,:,None])

#     if plot is True:
#         pf.plot_order(tr,iOrd, noise_level, cbar=True, cmap=cmap, ylabel='noise_level', **kwargs)

    good = np.where((noise_level <= 8) , uncorr_norm, np.nan)
    master = np.ma.masked_invalid(np.nanmean(good , axis=0))
    
    good_noise = np.where((noise_level <= 8), noise_level, np.nan)
    noise = np.ma.masked_invalid(np.nanmean(good_noise, axis=0))
    
    
    noise_floor = []
    for i in range(nord):
        poly_ord = 7
        fit = ext.poly_out(noise[i], poly_ord, ind_fit=~noise[i].mask)
        while (fit <=0).any():
            poly_ord -= 1
            fit = ext.poly_out(noise[i], poly_ord, ind_fit=~noise[i].mask)

        noise_floor.append(fit)
    noise_floor=np.ma.masked_invalid(noise_floor)
 
    if plot is True:
        plt.figure()
        plt.plot(wave[iOrd], noise[iOrd])
        plt.plot(wave[iOrd], noise_floor[iOrd])
    
    noise_level = noise_level-noise_floor
    
    clipped_noise = uncorr_norm.copy()
    for n in range(n_spec):
        for iord in range(nord):
            hm.print_static('{}, {}'.format(n, iord))
            cond = (noise_level[n,iord] >= tresh) #| (noise_level[n,iord] <= -5)

            difference = np.diff(np.arange(wave[iord].size)[cond])

            single = np.arange(wave[iord].size)[cond][:-1][difference>1]
            multiple = np.unique([np.arange(wave[iord].size)[cond][:-1][difference==1], 
                                  np.arange(wave[iord].size)[cond][:-1][difference==1]+1])
            
            
            try:
#                 fct_sp=interp1d_masked(tr.wv[iord][~cond], uncorr_norm[n,iord][~cond], 
#                                    kind='cubic', fill_value='extrapolate')
                fct_sp=interp1d_masked(wave[iord], master[iord], 
                                   kind='cubic', fill_value='extrapolate')
                clipped_noise[n,iord][single] = np.ma.masked_array(fct_sp(wave[iord][single]))
            except ValueError:
                print('chunks are overlapping at {}, {}'.format(n, iord))
    #             print(tr.wv[iord][~cond], uncorr_norm[n,iord][~cond])
                pass
    #         print(fct_sp(tr.wv[iord][single][0]))
#             clipped_noise[n,iord][multiple] = np.nan
            try:
#                 fct_lin=interp1d_masked(tr.wv[iord][~cond], uncorr_norm[n,iord][~cond], 
#                                     kind='linear', fill_value='extrapolate')
                fct_lin=interp1d_masked(wave[iord], master[iord], 
                                    kind='linear', fill_value='extrapolate')
                clipped_noise[n,iord][multiple] = np.ma.masked_array(fct_lin(wave[iord][multiple]))
            except ValueError:
    #             print('chunks are overlapping at {}, {}'.format(n, iord))
    #             print(tr.wv[iord][~cond], uncorr_norm[n,iord][~cond])
                pass
            
            for k, g in groupby(enumerate(multiple), lambda ix : ix[0] - ix[1]):
                if len(list(map(itemgetter(1), g))) >= 4:
                    clipped_noise[n,iord][list(map(itemgetter(1), g))] = np.nan

    if plot is True:
        pf.plot_order(tr,iOrd, clipped_noise, cbar=True, cmap=cmap, ylabel='clipped_noise', **kwargs)
    if plot is True:
        pf.plot_order(tr,iOrd, clipped_noise*median_time, cbar=True, cmap=cmap, ylabel='clipped*median_time', **kwargs)
    if plot is True:
        pf.plot_order(tr,iOrd,tr.fl_norm, cbar=True, cmap=cmap, ylabel='tr.fl_norm', **kwargs)
        
    return np.ma.masked_invalid(clipped_noise*median_time)  #*median_wv




def build_trans_spectrum4(wave, flux, light_curve, berv, RV_sys, vr, vrp, iIn, iOut, 
                         lim_mask=0.75, lim_buffer=0.97, tellu=None, path=None, mask_tellu=True, new_mask_tellu=None,
                          mask_var=True, last_mask=True, iOut_temp=None, plot=False, #kind_mo_lp='filter',
                          mo_box=51, mo_gauss_box=5, n_pca=1, n_comps=10, clip_ratio=None, clip_ts=None,
                          poly_time=None, kind_mo="median", cont=False, cbp=False, ##blaze=None,
                          tresh=3., tresh_lim=1., tresh2=3, tresh_lim2=1, noise=None, somme=False, norm=True,
                          flux_masked=None, flux_Sref=None, flux_norm=None, flux_norm_mo=None, master_out=None,
                          spec_trans=None, full_ts=None, unberv_it=True, wave_mo=None, template=None):
    
    rebuilt=np.ma.empty_like(flux)
    ratio=np.ma.empty_like(flux)
    if cbp is False:
        if flux_norm is None:
            hm.print_static('Normalizing by median. \n')
            flux_norm = flux/np.ma.median(np.clip(flux,0,None),axis=-1)[:,:,None]

        if mask_var is True:
            hm.print_static('Masking high variance pixels (quick fix for OH lines). \n')
            new_mask = [ext.get_mask_noise(f, tresh, tresh_lim, gwidth=0.01, poly_ord=5) for f in flux_norm.swapaxes(0,1)]
            new_mask = new_mask | flux_norm.mask
            flux_norm = np.ma.array(flux_norm, mask=new_mask)
    else:
        if flux_norm is None:
            flux_norm = clean_bad_pixels_time(np.mean(wave,axis=0), flux, tresh=tresh)#, plot=False, , tresh_lim=tresh_lim)
        
    print('flux_norm all nan : {}'.format(flux_norm.mask.all()))    
    if flux_Sref is None:
        hm.print_static('Shifting everything in the stellar ref. frame and normalizing by the median \n')
        if unberv_it is True:
            print('Spectra ', end="")
            flux_Sref = unberv(wave, flux_norm, berv, RV_sys, vr, clip=False)
            print('Telluriques ', end="")
            tellu_Sref = unberv(wave, tellu, berv, RV_sys, vr, clip=False)
        else:
            flux_Sref = flux_norm
            tellu_Sref = tellu
    print('flux_Sref all nan : {}'.format(flux_Sref.mask.all()))
    if flux_masked is None:
        if mask_tellu is True:
            hm.print_static('Masking deep tellurics. \n')
            flux_masked = mask_deep_tellu(flux_Sref, path=path, tellu=tellu_Sref, #tellu_list='list_tellu_recon',
                                          limit_mask=lim_mask, limit_buffer=lim_buffer, plot=False)
        else:
            flux_masked = flux_Sref.copy()            
    print('flux_masked all nan : {}'.format(flux_masked.mask.all()))
    # --- ***** CHANGED iOut FOR SOMETHING ELSE
    if (iOut_temp is None):  # or (iOut_temp == ''):
        iOut_temp = iOut #np.arange(0,36)
    elif iOut_temp == 'all':
        iOut_temp = np.arange(flux.shape[0])
#     else:
#         iOut_temp = iOut_temp

    if iOut_temp.size > flux.shape[0]:
        print('iOut size too big, flux size')
        iOut_temp = np.arange(flux.shape[0])
    
    if master_out is None:
        hm.print_static('Building the master out #1 \n')
        if flux_norm_mo is None:
            flux_norm_mo, master_out, ratio = build_master_out(wave, flux_masked, iOut_temp, 
                                            box=mo_box, gauss_box=mo_gauss_box, kind_mo=kind_mo, 
                                                               clip_ratio=clip_ratio, cont=cont)
        else:
            _, master_out, ratio = build_master_out(wave, flux_masked, iOut_temp,  
                                            box=mo_box, gauss_box=mo_gauss_box, kind_mo=kind_mo, 
                                                    clip_ratio=clip_ratio, cont=cont)
    else:
        if flux_norm_mo is None:
            flux_norm_mo, master_out, ratio = build_master_out(wave, flux_masked, iOut_temp, master_out=master_out,
                                            box=mo_box, gauss_box=mo_gauss_box, 
                                                               clip_ratio=clip_ratio, cont=cont)
    print('flux_norm_mo all nan : {}'.format(flux_norm_mo.mask.all()))
    print('master_out all nan : {}'.format(master_out.mask.all()))
    if spec_trans is None:
        hm.print_static('Building the transmission spectrum #1 \n')
        spec_trans = flux_norm_mo/master_out
        print('spec-trans all nan : {}'.format(spec_trans.mask.all()))
    if poly_time is not None:
#         for iOrd in range(49):
#             spec_trans[:,iOrd] = ext.col_remove(spec_trans[:,iOrd])
        # --- Polynomial fit on time --- #
        hm.print_static('Removing 2nd ord polynome in time \n')
        recon_time = np.ones_like(spec_trans)*np.nan
        x = poly_time
        z_t = np.zeros((49, 4088, 3))

        for iord in range(49):

            for col in range(4088):

                y = spec_trans[:,iord, col]

                if y.mask.all():
                    continue

                idx = np.isfinite(x) & np.isfinite(y)

                # Poly.fit(x[idx], y[idx], 2)
                z_t[iord,col] = np.polyfit(x[idx], y[idx], 2)

                recon_time[idx, iord, col] = np.poly1d(z_t[iord,col])(x[idx])

        recon_time = np.ma.masked_invalid(recon_time)
        spec_trans = spec_trans/recon_time

    if full_ts is None:
        hm.print_static('Removing the static noise with PCA and sigma cliping \n')
#         print(n_pca, n_comps)
        print(spec_trans.shape)
        if clip_ts is not None:
            spec_trans = sigma_clip(spec_trans, clip_ts)
        #*** mean normalize again??***
        full_ts, rebuilt, pca = remove_dem_pca_all(spec_trans, n_pcs=n_pca, n_comps=n_comps, plot=plot)

    if norm is True:
        hm.print_static('Removing the mean \n')
        chose = quick_norm(full_ts, somme=False, take_all=False)

        if last_mask is True:
            print('Removing the remaining high variance pixels. \n')
            if tresh2 != tresh_lim:
                new_mask = [ext.get_mask_noise(f, tresh2, tresh_lim2, gwidth=0.01) for f in chose.swapaxes(0,1)]
                new_mask = new_mask | chose.mask
                final_ts = np.ma.array(chose, mask=new_mask)
            else:
                final_ts = sigma_clip(chose, tresh2)
                new_mask = final_ts.mask
            
            hm.print_static('Removing the mean. \n')
            final_ts = quick_norm(final_ts, somme=somme, take_all=False)
        else:
            new_mask = chose.mask
            final_ts = chose

    else:
        final_ts = full_ts/np.ma.mean(full_ts, axis=-1)[:,:,None]
 
    if noise is None:
        final_ts_std = final_ts/np.std(final_ts, axis=0)**2
    else:
        final_ts_std = final_ts/noise**2
        
    return flux_norm, flux_norm_mo, master_out, spec_trans, full_ts, chose, \
           final_ts, final_ts_std, rebuilt, pca, flux_Sref, flux_masked, ratio, new_mask  #, flux_BARYref, flux_SYSref, flux_Sref



def build_trans_spectrum_mod2(wave, flux, master_out, pca, noise, iOut=None,
                              plot=False, n_pca=2, n_comps=10, somme=False, verbose=False,
                              mo_box=51, mo_gauss_box=3, norm=True, blaze=None, ratio=None, debug=False):
#     if debug is True:
#         print('wave', wave[10,34,2000], np.isnan(wave).all())
#         print('flux', flux[10,34,2000], np.isnan(flux).all())
#         print('master_out', master_out[34,2000], np.isnan(master_out).all())
#         print('pca', pca)
#         print('noise', noise[10,34,2000], np.isnan(noise).all())
#         print(n_pca)
#         print('blaze', blaze)
#         print('ratio', ratio[10,34,2000], np.isnan(ratio).all())
    
#     if blaze is not None:
#         flux = flux/(blaze/np.nanmax(blaze, axis=-1)[:,:,None])
        
#     if verbose:
#         hm.print_static('Normalizing by median \n')
    flux_norm_mo = flux/np.ma.median(flux,axis=-1)[:,:,None]

#     if iOut is not None:
#         flux_norm_mo, master_out, ratio = build_master_out(wave, flux_norm, iOut, master_out=master_out,
#                                                     box=mo_box, gauss_box=mo_gauss_box)
#     else:
#         flux_norm_mo = flux_norm
        
    if ratio is not None:
#         hm.print_static('Normalizing by the ratio \n')
        flux_norm_mo = flux_norm_mo/ratio

#     if verbose :
#         hm.print_static('Building the transmission spectrum #1 \n')
#     spec_trans = flux_norm_mo/master_out
        
#     if verbose :
#         hm.print_static('Removing the static noise with PCA and sigma cliping \n')

    if n_pca > 0:
#         if debug is True:
#             print(spec_trans, n_pca, pca)
#             print(np.isnan(spec_trans), np.isnan(spec_trans).all(), n_pca)
        full_ts, _, _ = remove_dem_pca_all(flux_norm_mo/master_out, pca=pca, n_pcs=n_pca, n_comps=n_comps, plot=plot)
    else:
        full_ts = flux_norm_mo/master_out
    
    if norm is True:
#         if verbose :   
#             hm.print_static('Removing the mean \n')
        final_ts = quick_norm(full_ts, somme=somme, take_all=False)
    else:
        final_ts = full_ts/np.ma.mean(full_ts, axis=-1)[:,:,None]
#         final_ts = full_ts
#     final_ts_std = final_ts/noise**2
    
    return final_ts#, final_ts_std


def build_trans_spectrum_mod_fast(wave, flux, master_out, pca, noise, iOut=None,
                              plot=False, n_pca=2, n_comps=10, somme=False, verbose=False,
                              mo_box=51, mo_gauss_box=3, norm=True, blaze=None, ratio=None, debug=False):

#     flux_norm_mo = flux/np.ma.median(flux,axis=-1)[:,:,None]

    if n_pca > 0:

        full_ts, _, _ = remove_dem_pca_all(flux/np.ma.median(flux,axis=-1)[:,:,None], 
                                           pca=pca, n_pcs=n_pca, n_comps=n_comps, plot=plot)
    else:
        full_ts = flux/np.ma.median(flux,axis=-1)[:,:,None]
    
    final_ts = quick_norm(full_ts, somme=somme, take_all=False)
    
    return final_ts#, final_ts_std


def build_trans_spectrum_mod_new(tr, flux, z=None, z_t=None, plot=False, 
                                 clip=3, npc=3, id_ord=34, xlim=[None,None]):
    
    if z is None:
        z = tr.z
    if z_t is None:
        z_t = tr.z_t    
        
    if plot is True:
        plot_order(tr, id_ord, flux)
        plt.colorbar()
        plt.xlim(*xlim)
#     uncorr_lp=flux
#     spec_trans=flux
    # --- Polynomial fit on master spectrum --- #
    hm.print_static('{}/10'.format(6))
#     recon_poly = np.ones_like(flux)*np.nan
    
#     x = np.nanmedian(flux, axis=0)#[iord]
#     idx_x = np.isfinite(x)

#     for iord in range(49):
#         hm.print_static('{}/10  - {}  '.format(6, iord))

#         for n in range(tr.n_spec):
#             idx = idx_x[iord] & np.isfinite(flux[n,iord])
#             recon_poly[n,iord, idx] = polyval(x[iord][idx], z[n, iord][::-1])
#         recon_poly[:,iord, idx[iord]] = polyval(x[iord][idx[iord]], z[:, iord][::-1].T, tensor=True)

    recon_poly = tr.recon_poly #np.ma.masked_invalid(recon_poly)

    if plot is True:
        plot_order(tr, id_ord, recon_poly)
#         pf.plot_order(tr, id_ord, recon_poly)
        plt.colorbar()
        plt.xlim(*xlim)
        plot_order(tr, id_ord, sigma_clip(recon_poly/tr.recon_poly, 3))
#         pf.plot_order(tr, id_ord, recon_poly)
        plt.colorbar()
        plt.xlim(*xlim)
        

    uncorr_nostar = flux/recon_poly
    
    if plot is True:
        plot_order(tr, id_ord, uncorr_nostar)
        plt.colorbar()
        plt.xlim(*xlim)

    # --- Polynomial fit on time --- #
    hm.print_static('{}/10'.format(7))
    recon_time = np.ones_like(uncorr_nostar)*np.nan
    x = tr.t.value
    idx = np.isfinite(x)
    for iord in range(49):
        hm.print_static('{}/10  - {}  '.format(7, iord))
#         for col in range(tr.npix):
#             idx &= np.isfinite(uncorr_nostar[:,iord,col])
#             recon_time[idx, iord, col] = polyval(x[idx], z_t[iord,col][::-1], tensor=True)
        recon_time[idx, iord, :] = polyval(x[idx], z_t[iord,:].T[::-1], tensor=True).T
#         print((recon_time[idx, iord, :] == 0).sum())
    recon_time[recon_time == 0] = np.nan
        
    
    recon_time = np.ma.masked_invalid(recon_time)


#     # --- Polynomial fit on master spectrum --- #
#     hm.print_static('{}/10'.format(6))
#     recon_poly = np.ones_like(uncorr_lp)*np.nan
#     z = np.zeros((tr.n_spec, tr.nord, 3))

#     x = np.nanmedian(uncorr_lp,axis=0)
#     idx_x = np.isfinite(x)
#     for iord in range(49):

#         if idx_x[iord].sum()==0:
#             continue
        
#         for n in range(tr.n_spec):

#             y = uncorr_lp[n,iord]

#             idx = idx_x[iord] & np.isfinite(y)

#             # Poly.fit(x[idx], y[idx], 2)
#             z[n,iord] = np.polyfit(x[iord][idx], y[idx], 2)
#             recon_poly[n,iord, idx] = np.poly1d(z[n,iord])(x[iord][idx])

#     recon_poly = np.ma.masked_invalid(recon_poly)

#     if plot is True:
#         pf.plot_order(tr, id_ord, recon_poly)
#         plt.colorbar()
#         plt.xlim(*xlim)

#     uncorr_nostar = uncorr_lp/recon_poly
#     if plot is True:
#         pf.plot_order(tr, id_ord, uncorr_nostar)
#         plt.colorbar()
#         plt.xlim(*xlim)

#     # --- Polynomial fit on time --- #
#     hm.print_static('{}/10'.format(7))
#     recon_time = np.ones_like(uncorr_nostar)*np.nan
#     x = tr.t.value
#     z_t = np.zeros((tr.nord, tr.npix, 3))

#     for iord in range(49):

#         for col in range(tr.npix):

#             y = uncorr_nostar[:,iord, col]

#             if y.mask.all():
#                 continue

#             idx = np.isfinite(x) & np.isfinite(y)

#             # Poly.fit(x[idx], y[idx], 2)
#             z_t[iord,col] = np.polyfit(x[idx], y[idx], 2)

#             recon_time[idx, iord, col] = np.poly1d(z_t[iord,col])(x[idx])

#     recon_time = np.ma.masked_invalid(recon_time)


    if plot is True:
        plot_order(tr, id_ord, sigma_clip(recon_time/tr.recon_time,3) )
        plt.colorbar()
        plt.xlim(*xlim)
        
        plot_order(tr, id_ord, uncorr_nostar/recon_time )
        plt.colorbar()
        plt.xlim(*xlim)

    spec_trans = uncorr_nostar/recon_time
    
    if plot is True:
        plot_order(tr, id_ord, sigma_clip(spec_trans, 3) )
        plt.colorbar()
        plt.xlim(*xlim)
    
#     tr.mast_out = recon_time*recon_poly
#     tr.spec_trans = spec_trans

    # --- PCA clean up --- #
    hm.print_static('{}/10'.format(8))
    if npc >0 :
        pca_clean_ts, rebuilt, pca = remove_dem_pca_all(spec_trans, n_pcs=npc)#, pca= tr.pca)
#         tr.pca = pca
#         tr.rebuilt = rebuilt 
    else:
        pca_clean_ts = spec_trans
#         tr.rebuilt = np.ones_like(spec_trans)

    if plot is True:
        plot_order(tr, id_ord, pca_clean_ts )
        plt.colorbar()
        plt.xlim(*xlim)
    
    # --- Cleaning remaining deviant pixels --- #
#     hm.print_static('{}/10'.format(9))
# #     new_mask = [ext.get_mask_noise(f, 3, 1., gwidth=0.01) for f in pca_clean_ts.swapaxes(0,1)]
# #     new_mask = new_mask | pca_clean_ts.mask
#     final_ts = np.ma.array(pca_clean_ts, mask=tr.final.mask)
    final_ts = pca_clean_ts

    # --- Mean removal --- #
    hm.print_static('{}/10'.format(10))
    final_ts = ext.quick_norm(final_ts, take_all=False)

    if plot is True:
        plot_order(tr, id_ord, final_ts )
        plt.colorbar()
        plt.xlim(*xlim)
    
#     tr.final = final_ts
#     tr.reconstructed = tr.mast_out * tr.rebuilt * tr.ratio * (tr.blaze/np.nanmax(tr.blaze, axis=-1)[:,:,None])
    
    
    return final_ts  #, final_ts_std



def calc_stacked_spectra(tr, flux=None, weight=None, pca_red=False, kind='average', 
                         iOrd0=None, iin=None, iout=None, RV=None, vr=None, vrp=None, alpha=None, RV_star=None):
    
    if flux is None:
        flux = tr.spec_trans
    if weight is None:
        weight = tr.light_curve
    if iin is None:
        iin = tr.iIn
    if iout is None:
        iout = tr.iOut
    if vr is None:
        vr = tr.vr
    if vrp is None:
        vrp = tr.vrp   
    if alpha is None:
        alpha = tr.alpha_frac
    
    spec_fin, _ = build_stacked_st(tr.wv, flux[iin], vr[iin], vrp[iin], 
                                   weight[iin], kind=kind, iOrd0=iOrd0, RV=RV, alpha=alpha[iin])

    spec_fin_out, _ = build_stacked_st(tr.wv, flux[iout], vr[iout], vrp[iout],
                                       weight[iout], kind=kind, iOrd0=iOrd0, RV=RV, alpha=alpha[iout])

    if RV_star is None:
        spec_fin_Sref = np.ma.average(flux[iin], axis=0, weights=weight[iin])
    else:
        spec_fin_Sref, _ = build_stacked_st(tr.wv, flux[iin], np.zeros_like(vr)[iin], np.zeros_like(vrp)[iin],
                                       weight[iin], kind=kind, iOrd0=iOrd0, RV=RV_star, alpha=alpha[iin])
    
    if pca_red is True:
        spec_fin_ts, _ = build_stacked_st(tr.wv, tr.final[iin], vr, vrp[iin], 
                                          weight[iin], kind=kind, iOrd0=iOrd0, RV=RV, alpha=alpha[iin])
        return spec_fin, spec_fin_out, spec_fin_Sref, spec_fin_ts
    else:
        return spec_fin, spec_fin_out, spec_fin_Sref

