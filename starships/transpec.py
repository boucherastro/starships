from starships import extract as ext
from starships import analysis as a
from starships import homemade as hm
from .extract import quick_norm, running_filter
from .mask_tools import interp1d_masked

import numpy as np
# import scipy.constants as cst
from astropy import units as u
# from astropy import constants as const
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian1DKernel
# from scipy.interpolate import interp1d

from numpy.polynomial.polynomial import polyval

from itertools import groupby
from operator import itemgetter
from dataclasses import dataclass, replace, fields


@dataclass
class ReductionParams:
    """Named, documented reduction parameters for :func:`build_trans_spectrum4`.

    This replaces the old convention of stacking 10 reduction parameters into
    a raw positional list (e.g. ``[0.2, 0.97, 51, 41, 5, 2, 5.0, 5.0, 5.0,
    5.0]``), where changing a value meant counting positions in the list with
    no names or documentation attached. Most of these parameters are "deep"
    (rarely changed in practice) but Antoine wants them kept visible and
    documented rather than removed, hence explicit named fields with
    sensible defaults instead of hardcoded numbers buried in the pipeline.

    In practice, only `mask_tellu`, `mask_wings` and `n_pc` are varied
    routinely; the rest are effectively fixed constants of the reduction
    (see individual field descriptions).

    For backward compatibility with existing code that still indexes into
    reduction parameters positionally (e.g. ``visit.params[5]``), this class
    also supports ``len()``, integer indexing/assignment (``params[i]``) and
    ``.copy()``, in the same order as the old 10-element list.

    Attributes
    ----------
    mask_tellu : float
        Telluric absorption fraction below which a pixel is masked as a deep
        telluric line (`lim_mask` in `build_trans_spectrum4`). Typically
        varied between 0.2 and 0.5 — deeper tellurics (lower value) mask
        more pixels.
    mask_wings : float
        Buffer/wing limit around masked telluric lines (`lim_buffer`).
        Typically varied between 0.9 and 0.98.
    reference_spec_box : int
        Width (in pixels) of the low-pass smoothing box kernel used when
        building the reference spectrum (`mo_box`, formerly "master-out").
        Rarely changed in practice.
    unused_legacy : float
        Dead parameter from the original 10-element positional list — never
        read by `build_trans_spectrum4` (confirmed: only indices 0, 1, 2, 4,
        5, 6, 7, 8, 9 are used). Kept only so that positional/list-style
        access (`params[i]`) stays compatible with older code that still
        expects a 10-element sequence.
    reference_spec_gauss_box : int
        Width of the Gaussian smoothing kernel used together with
        `reference_spec_box` when building the reference spectrum
        (`mo_gauss_box`). Rarely changed in practice.
    n_pc : int
        Number of principal components removed by the PCA step (`n_pca`).
        The one parameter that is essentially always varied/tuned per
        dataset.
    tresh : float
        Sigma-clipping threshold used to mask high-variance pixels *before*
        the PCA step (`tresh` in `build_trans_spectrum4`, feeds
        `extract.get_mask_noise`). Rarely changed in practice.
    tresh_lim : float
        Companion sigma-clipping limit to `tresh` (`tresh_lim`). Rarely
        changed in practice.
    last_tresh : float
        Sigma-clipping threshold used for a final round of masking *after*
        the PCA step, on the normalized post-PCA time series (`last_tresh`).
        Rarely changed in practice.
    last_tresh_lim : float
        Companion sigma-clipping limit to `last_tresh` (`last_tresh_lim`).
        Rarely changed in practice.
    """

    mask_tellu: float = 0.2
    mask_wings: float = 0.97
    reference_spec_box: int = 51
    unused_legacy: float = 41.0
    reference_spec_gauss_box: int = 5
    n_pc: int = 2
    tresh: float = 5.0
    tresh_lim: float = 5.0
    last_tresh: float = 5.0
    last_tresh_lim: float = 5.0

    def __len__(self):
        return len(fields(self))

    def __getitem__(self, index):
        return getattr(self, fields(self)[index].name)

    def __setitem__(self, index, value):
        setattr(self, fields(self)[index].name, value)

    def copy(self):
        """Return a shallow copy, mirroring `list.copy()` for old callers."""
        return replace(self)


def mask_deep_tellu(flux, tellu=None, path=None, tellu_list='list_tellu_tr',
                    limit_mask=0.5, limit_buffer=0.98, plot=False, new_mask_tellu=None):
    
    n_spec, nord, _ = flux.shape
    
    if new_mask_tellu is None:
        if tellu is None:
            _, wave_tell, tellu, _ = ext.read_all_sp_spirou_apero(path, tellu_list, wv_default="MASTER_WAVE.fits")
        tellu = np.ma.array(tellu, mask = ~np.isfinite(tellu))

        # Find the strong tellurics with a pad around in the order
        new_mask_tellu = np.empty_like(tellu)
        for iOrd in range(nord):
            new_mask = [ext.get_mask_tell(tell, limit_mask, limit_buffer, pad_masked=True) for tell in tellu[:,iOrd,:]]
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


def unberv(wave, flux_masked, berv, vr, counting = True):  #, norm=False
    
    n_spec, nord, _ = flux_masked.shape
    
    if isinstance(berv, u.Quantity):
        berv = berv.to(u.km/u.s).value
    # if isinstance(RV_sys, u.Quantity):
    #     RV_sys = RV_sys.to(u.km/u.s).value
    if isinstance(vr, u.Quantity):
        vr = vr.to(u.km/u.s).value

#     if clip is True:
#         # -- Sigma clipping of the spectra
#         flux_masked = sigma_clip(flux_masked, axis=0, sigma=4)

    flux_Sref = np.ones_like(flux_masked) * np.nan
    
    
    shifts = hm.calc_shift(-(berv+vr), kind='rel')
    print('')
    for n in range(n_spec):
        for iOrd in range(nord):
            if counting:
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


def build_reference_spec(wave, flux_Sref_norm, iOut, kind_lp='filter', 
                     box=201, gauss_box=5, reference_spec=None, kind_mo='median', 
                     clip_ratio=None, cont=False): #, light_curve
    
    nspec, nord, _ = flux_Sref_norm.shape
    
    if reference_spec is None:
        # --- Building the master out-of-transit spectrum
        if kind_mo == "median":
            reference_spec = np.ma.median(flux_Sref_norm[iOut],  axis=0)
        elif kind_mo == "mean":
            reference_spec = np.ma.mean(flux_Sref_norm[iOut],  axis=0)
    #     reference_spec = np.ma.average(flux_Sref_norm[iOut], weights=light_curve[iOut], axis=0)
        
    if cont is True:
        reference_spec_ratio = a.remove_pseudo_cont(wave[0], np.clip(reference_spec.copy(),0.95,None), 
                                                kWidth=30, wSize=55)
        reference_spec = reference_spec.copy()/reference_spec_ratio
#         reference_spec_ratio = a.remove_pseudo_cont(wave[0], reference_spec.copy(), 5, 31)
#         reference_spec = reference_spec/reference_spec_ratio
        
#     plt.figure()
#     plt.plot(wave[0],reference_spec.T)
    # plt.show()

    # -- Polynome fiting on the ratio of spec/master

    ratio  = np.ma.array(flux_Sref_norm/reference_spec)

#     fit = np.ma.array(np.ones_like(ratio)) * np.nan
    ratio_filt = np.ma.array(np.ones_like(ratio)) * np.nan
#     for n in range(n_spec):
    if kind_lp == 'poly' or kind_lp == 'filter':
        for iOrd in range(nord):
            hm.print_static(iOrd)

            # -- normalizing the spectra to get them at the same level as the reference_spec
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
#     plt.plot(wave[0,33], flux_Sref_norm[0,33].T/reference_spec[33])
#     plt.plot(wave[0,33], ratio_filt[0,33].T)
#     plt.plot(wave[0,33], sigma_clip(ratio_filt[0,33],5))
#     hm.stop()
    
    if reference_spec is None:
    #     new_reference_spec = np.ma.average(flux_norm_mo[iOut], weights=light_curve[iOut], axis=0)
        new_reference_spec = np.ma.median(flux_norm_mo[iOut], axis=0)
    else:
        new_reference_spec = reference_spec

    return flux_norm_mo, new_reference_spec, np.ma.masked_invalid(ratio_filt)


def build_reference_spec_pc(wave, flux_Sref_norm, iOut, plot=False, **kwargs):
    
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

    reference_spec = np.ma.median(flux_norm[iOut],  axis=0)
    
    if plot is True:
        plt.figure()
        plt.plot(wave[0,35], flux_norm[0,35])
        plt.plot(wave[0,35], reference_spec[35])

    return flux_norm, reference_spec


# TRANSMISSION SPECTRUM

from sklearn.decomposition import PCA

# returns the 'n_components' number of principal components of a dataset.  matrix is N x M, returns what is common to the M axis.
def PCA_decompose(matrix, n_components=None):
    if n_components == None:
        n_components = np.shape(matrix)[1]
    # Force the exact LAPACK SVD solver instead of sklearn's shape-based 'auto' heuristic:
    # for our matrices (n_components close to the full rank of the data), 'auto' already
    # picks 'full' in practice, but pinning it here makes the fit reproducible byte-for-byte
    # from one run to the next regardless of matrix shape, instead of relying on that
    # heuristic implicitly (see B3 cleanup notes: PCA is now fit once and reused across
    # reads with different n_pc, so this determinism guarantee matters more than before).
    pca = PCA(n_components=n_components, svd_solver='full')
    coefficients = pca.fit_transform(matrix)
    pcs = pca.components_
    return pca, pcs, coefficients

# removes 'n_pcs' number of principal components from input
def PCA_remove(matrix, pcs, coefficients, n_pcs, kind='rebuilt'):

    if n_pcs > pcs.shape[0]:
        raise ValueError(
            f'Requested n_pcs={n_pcs} principal components, but only {pcs.shape[0]} were '
            'fitted (n_comps at reduction time was too small for this read-time n_pc).'
        )

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



def clean_bad_pixels(wave, uncorr0, plot=False, t1=None, iOrd=34, tresh=4, tresh_lim=3):
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

# added condition to skip completely masked orders in the noise floor fits (happens sometimes with nirps)
def clean_bad_pixels_time(wave, uncorr0, tresh=3., plot=False, visit=None, iOrd=34, cmap=None, **kwargs):
#     if plot is True:
#         pf.plot_order(visit,iOrd,visit.uncorr, **kwargs)
    n_spec, nord,_ = uncorr0.shape
    
    median_wv = np.ma.median(uncorr0, axis=-1)[:,:,None]
    uncorr_mean_norm = uncorr0/median_wv
#     if plot is True:
#         pf.plot_order(visit,iOrd,uncorr_mean_norm, cmap=cmap, ylabel='uncorr_mean_norm', **kwargs)

    median_time = np.ma.median(uncorr_mean_norm, axis=0)[None,:,:]
    uncorr_norm = uncorr_mean_norm/median_time

#     if plot is True:
#         pf.plot_order(visit,iOrd, uncorr_norm, cbar=True, cmap=cmap, ylabel='uncorr_norm', **kwargs)

    noise_level = np.abs((uncorr_norm-np.ma.median(uncorr_norm, axis=-1)[:,:,None])/np.ma.std(uncorr_norm, axis=-1)[:,:,None])

#     if plot is True:
#         pf.plot_order(visit,iOrd, noise_level, cbar=True, cmap=cmap, ylabel='noise_level', **kwargs)

    good = np.where((noise_level <= 8) , uncorr_norm, np.nan)
    master = np.ma.masked_invalid(np.nanmean(good , axis=0))
    
    good_noise = np.where((noise_level <= 8), noise_level, np.nan)
    noise = np.ma.masked_invalid(np.nanmean(good_noise, axis=0))

#   when an entire order is masked, just return the noise fit as the noise itself, since noise is all nan anyway
    noise_floor = []
    for i in range(nord):
        poly_ord = 7
        if ~noise[i].mask.any():
            fit = ext.poly_out(noise[i], poly_ord, ind_fit=~noise[i].mask)
            while (fit <=0).any():
                poly_ord -= 1
                fit = ext.poly_out(noise[i], poly_ord, ind_fit=~noise[i].mask)
        else:
            # if the entire order is masked
            fit = noise[i]

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
#                 fct_sp=interp1d_masked(visit.wv[iord][~cond], uncorr_norm[n,iord][~cond], 
#                                    kind='cubic', fill_value='extrapolate')
                fct_sp=interp1d_masked(wave[iord], master[iord], 
                                   kind='cubic', fill_value='extrapolate')
                clipped_noise[n,iord][single] = np.ma.masked_array(fct_sp(wave[iord][single]))
            except ValueError:
                print('chunks are overlapping at {}, {}'.format(n, iord))
    #             print(visit.wv[iord][~cond], uncorr_norm[n,iord][~cond])
                pass
    #         print(fct_sp(visit.wv[iord][single][0]))
#             clipped_noise[n,iord][multiple] = np.nan
            try:
#                 fct_lin=interp1d_masked(visit.wv[iord][~cond], uncorr_norm[n,iord][~cond], 
#                                     kind='linear', fill_value='extrapolate')
                fct_lin=interp1d_masked(wave[iord], master[iord], 
                                    kind='linear', fill_value='extrapolate')
                clipped_noise[n,iord][multiple] = np.ma.masked_array(fct_lin(wave[iord][multiple]))
            except ValueError:
    #             print('chunks are overlapping at {}, {}'.format(n, iord))
    #             print(visit.wv[iord][~cond], uncorr_norm[n,iord][~cond])
                pass
            
            for k, g in groupby(enumerate(multiple), lambda ix : ix[0] - ix[1]):
                if len(list(map(itemgetter(1), g))) >= 4:
                    clipped_noise[n,iord][list(map(itemgetter(1), g))] = np.nan

    if plot is True:
        pf.plot_order(visit,iOrd, clipped_noise, cbar=True, cmap=cmap, ylabel='clipped_noise', **kwargs)
    if plot is True:
        pf.plot_order(visit,iOrd, clipped_noise*median_time, cbar=True, cmap=cmap, ylabel='clipped*median_time', **kwargs)
    if plot is True:
        pf.plot_order(visit,iOrd,visit.fl_norm, cbar=True, cmap=cmap, ylabel='visit.fl_norm', **kwargs)
        
    return np.ma.masked_invalid(clipped_noise*median_time)  #*median_wv




def resolve_reference_spec_exposures(iOut_temp, iOut, n_exposures):
    """Resolve which exposures are used to build the reference spectrum.

    Two modes:

    - `'all'` (default, matches `iOut_temp` passed as `'all'` or left `None` after this
      function's own default resolution upstream): every exposure is used. The planetary
      signal is negligible next to the star's and further diluted by the planet's own
      motion across exposures, so using all exposures actually *improves* the reference
      spectrum's signal-to-noise ratio (confirmed intentional behaviour, not a bug).
    - out-of-transit/out-of-eclipse only: pass `iOut_temp=None` to use the real `iOut`
      (computed from the orbit, see `planet_obs.py::gen_transit_model`).

    Before this was factored out, `iOut_temp` was silently overwritten to 'all' no matter
    what was passed in — restoring the `iOut_temp is None` branch below makes the choice
    real again, without changing the default ('all') behaviour.

    Parameters
    ----------
    iOut_temp : {'all', None} or array_like of int
        Requested mode, or an explicit array of exposure indices (used as-is).
    iOut : array_like of int
        The true out-of-transit/out-of-eclipse exposure indices for this sequence.
    n_exposures : int
        Total number of exposures in `flux` (used to build the 'all' index array, and to
        guard against an oversized `iOut_temp`).

    Returns
    -------
    numpy.ndarray of int
        Exposure indices to use when building the reference spectrum.
    """
    if iOut_temp is None:
        iOut_temp = iOut
    elif isinstance(iOut_temp, str) and iOut_temp == 'all':
        iOut_temp = np.arange(n_exposures)

    if iOut_temp.size > n_exposures:
        print('iOut size too big, flux size')
        iOut_temp = np.arange(n_exposures)

    return iOut_temp


def apply_pca_truncation(spec_trans, n_pca, n_comps=None, pca=None, clip_ts=None, norm=True, somme=False,
                          last_mask=True, tresh_lim=1., last_tresh=3, last_tresh_lim=1.):
    """Remove `n_pca` principal components from a transmission spectrum and finalize it.

    This is the n_pc-*dependent* tail of `build_trans_spectrum4` (PCA truncation, mean
    removal, final high-variance masking), split out so it can be re-run cheaply for a
    different `n_pca` without repeating the n_pc-*independent* steps upstream (normalization,
    stellar-frame shift, telluric masking, reference spectrum, `spec_trans` itself). Pass an
    already-fitted `pca` (e.g. saved at reduction time) to skip refitting entirely — only the
    truncation to `n_pca` components and the normalization/masking below are then recomputed.

    Parameters
    ----------
    spec_trans : numpy.ma.MaskedArray
        Transmission spectrum (n_exposures, n_orders, n_pixels), n_pc-independent.
    n_pca : int
        Number of principal components to remove.
    n_comps : int, optional
        Number of components to fit if `pca` is not provided (ignored otherwise).
    pca : sklearn.decomposition.PCA, optional
        Already-fitted PCA to reuse (transform only, no refit). If None, a new PCA is fit
        on `spec_trans` with `n_comps` components.
    clip_ts : float, optional
        Sigma-clipping threshold applied to `spec_trans` before PCA removal.
    norm : bool
        Whether to remove the mean and apply the final high-variance masking below.
    somme, last_mask, tresh_lim, last_tresh, last_tresh_lim :
        Same meaning as the corresponding parameters of `build_trans_spectrum4`.

    Returns
    -------
    clean_ts, ts_norm, final_ts, rebuilt, mask_last, pca
    """
    if clip_ts is not None:
        spec_trans = sigma_clip(spec_trans, clip_ts)

    clean_ts, rebuilt, pca = remove_dem_pca_all(spec_trans, n_pcs=n_pca, n_comps=n_comps, pca=pca)

    if norm is True:
        hm.print_static('Removing the mean \n')
        ts_norm = quick_norm(clean_ts, somme=False, take_all=False)

        if last_mask is True:
            print('Removing the remaining high variance pixels. \n')
            if last_tresh != tresh_lim:
                mask_last = [ext.get_mask_noise(f, last_tresh, last_tresh_lim, gwidth=0.01) for f in ts_norm.swapaxes(0, 1)]
                mask_last = mask_last | ts_norm.mask
                final_ts = np.ma.array(ts_norm, mask=mask_last)
            else:
                final_ts = sigma_clip(ts_norm, last_tresh)
                mask_last = final_ts.mask

            hm.print_static('Removing the mean. \n')
            final_ts = quick_norm(final_ts, somme=somme, take_all=False)
        else:
            mask_last = ts_norm.mask
            final_ts = ts_norm
    else:
        # `norm=False` is never used by any caller in the codebase (grep-confirmed) — kept
        # for signature symmetry with `build_trans_spectrum4`, `ts_norm` is simply unset.
        ts_norm = None
        mask_last = clean_ts.mask
        final_ts = clean_ts / np.ma.mean(clean_ts, axis=-1)[:, :, None]

    return clean_ts, ts_norm, final_ts, rebuilt, mask_last, pca


# def build_trans_spectrum4(wave, flux, light_curve, berv, RV_sys, vr, vrp, iIn, iOut,
#                          lim_mask=0.75, lim_buffer=0.97, tellu=None, path=None, mask_tellu=True, new_mask_tellu=None,
#                           mask_var=True, last_mask=True, iOut_temp=None, plot=False, #kind_mo_lp='filter',
#                           mo_box=51, mo_gauss_box=5, n_pca=1, n_comps=10, clip_ratio=None, clip_ts=None,
#                           poly_time=None, kind_mo="median", cont=False, cbp=False, ##blaze=None,
#                           tresh=3., tresh_lim=1., tresh2=3, tresh_lim2=1, noise=None, somme=False, norm=True,
#                           flux_masked=None, flux_Sref=None, flux_norm=None, flux_norm_mo=None, reference_spec=None,
#                           spec_trans=None, full_ts=None, unberv_it=True, wave_mo=None, template=None):
def build_trans_spectrum4(wave, flux, berv, RV_sys, vr, iOut,
                          lim_mask=0.5, lim_buffer=0.97, tellu=None, path=None, mask_tellu=True,
                          mask_var=True, last_mask=True, iOut_temp=None, plot=False,
                          mo_box=51, mo_gauss_box=5, n_pca=1, n_comps=10, clip_ratio=None, clip_ts=None,
                          poly_time=None, kind_mo="median", cont=False, cbp=False,
                          tresh=3., tresh_lim=1., last_tresh=3, last_tresh_lim=1, noise=None, somme=False, norm=True,
                          flux_masked=None, flux_Sref=None, flux_norm=None, flux_norm_mo=None, reference_spec=None,
                          spec_trans=None, clean_ts=None, unberv_it=True, counting = True, pca=None):
    """Build the transmission spectrum from raw flux, through PCA removal, in one pass.

    This runs the full reduction chain: median normalization, high-variance pixel masking,
    shift to the stellar reference frame (`unberv`), deep-telluric masking, reference
    spectrum (`reference_spec`/`build_reference_spec`), `spec_trans = flux_norm_mo / reference_spec`,
    and finally PCA removal of `n_pca` components (`apply_pca_truncation`).

    Every step up to and including `spec_trans` is independent of `n_pca` — pass any of the
    `flux_masked`/`flux_Sref`/`flux_norm`/`flux_norm_mo`/`reference_spec`/`spec_trans` arguments
    already computed (e.g. loaded from a saved reduction) to skip recomputing that step.
    Only the PCA removal (and, if fitting, its cost) actually depends on `n_pca` — pass an
    already-fitted `pca` to skip the fit too and only redo the cheap truncation to `n_pca`
    components (see `apply_pca_truncation`, which implements that tail on its own so it can
    be re-run for a different `n_pca` without going through this function again).

    Parameters
    ----------
    wave : numpy.ndarray
        Wavelength grid (n_exposures, n_orders, n_pixels).
    flux : numpy.ma.MaskedArray
        Raw flux (n_exposures, n_orders, n_pixels).
    berv, RV_sys, vr : array_like
        Barycentric and systemic radial velocities used to shift to the stellar frame.
    iOut : numpy.ndarray of int
        Default out-of-transit/eclipse exposure indices (used when `iOut_temp` is None).
    n_pca : int
        Number of principal components to remove from `spec_trans`.
    n_comps : int
        Number of components to fit if `pca` is not provided (ignored otherwise).
    pca : sklearn.decomposition.PCA, optional
        Already-fitted PCA to reuse instead of refitting (see `apply_pca_truncation`).
    flux_masked, flux_Sref, flux_norm, flux_norm_mo, reference_spec, spec_trans, clean_ts : optional
        Precomputed intermediate results to reuse instead of recomputing that step; each is
        independent of `n_pca` except `clean_ts` (which, if given, is unreachable — no caller
        in the codebase passes it, see note below).
    mask_tellu, mask_var, last_mask : bool
        Whether to apply telluric masking, high-variance pixel masking, and final masking.
    lim_mask, lim_buffer, mo_box, mo_gauss_box, kind_mo, clip_ratio, cont, cbp, tresh,
    tresh_lim, last_tresh, last_tresh_lim, clip_ts, somme, norm, unberv_it, counting :
        Tuning parameters for the corresponding sub-steps (masking thresholds, reference
        spectrum smoothing box sizes, PCA truncation/normalization options) — see the
        relevant helper (`mask_deep_tellu`, `build_reference_spec`, `apply_pca_truncation`).
    poly_time, noise : optional
        If `poly_time` is given, fits and removes a per-pixel 2nd-order polynomial in time
        from `spec_trans` before PCA removal (not used by the current pipeline default).
    path : str, optional
        Path passed through to `mask_deep_tellu` for telluric reference lookup.

    Returns
    -------
    flux_norm, flux_norm_mo, reference_spec, spec_trans, clean_ts, ts_norm, final_ts, rebuilt,
    pca, flux_Sref, flux_masked, ratio, mask_last, recon_time
    """
    rebuilt=np.ma.empty_like(flux)
    ratio=np.ma.empty_like(flux)
    
    n_spec, nord, npix = flux.shape
    
    if cbp is False:
        if flux_norm is None:
            hm.print_static('Normalizing by median. \n')
            flux_norm = flux/np.ma.median(np.clip(flux,0,None),axis=-1)[:,:,None]
    else:
        if flux_norm is None:
            flux_norm = clean_bad_pixels_time(np.mean(wave,axis=0), flux, tresh=tresh)#, plot=False, , tresh_lim=tresh_lim)
    if mask_var is True:
        hm.print_static('Masking high variance pixels (quick fix for OH lines). \n')
        new_mask = [ext.get_mask_noise(f, tresh, tresh_lim, gwidth=0.01, poly_ord=5) for f in flux_norm.swapaxes(0, 1)]
        new_mask = new_mask | flux_norm.mask
        flux_norm = np.ma.array(flux_norm, mask=new_mask)
        
    print('flux_norm all nan : {}'.format(flux_norm.mask.all()))    
    if flux_Sref is None:
        hm.print_static('Shifting everything in the stellar ref. frame and normalizing by the median \n')
        if unberv_it is True:
            print('Spectra ', end="")
            flux_Sref = unberv(wave, flux_norm, berv, vr, counting = counting)
            print('Telluriques ', end="")
            tellu_Sref = unberv(wave, tellu, berv, vr, counting = counting)
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
    iOut_temp = resolve_reference_spec_exposures(iOut_temp, iOut, flux.shape[0])

    if reference_spec is None:
        hm.print_static('Building the master out #1 \n')
        if flux_norm_mo is None:
            flux_norm_mo, reference_spec, ratio = build_reference_spec(wave, flux_masked, iOut_temp, 
                                            box=mo_box, gauss_box=mo_gauss_box, kind_mo=kind_mo, 
                                                               clip_ratio=clip_ratio, cont=cont)
        else:
            _, reference_spec, ratio = build_reference_spec(wave, flux_masked, iOut_temp,  
                                            box=mo_box, gauss_box=mo_gauss_box, kind_mo=kind_mo, 
                                                    clip_ratio=clip_ratio, cont=cont)
    else:
        if flux_norm_mo is None:
            flux_norm_mo, reference_spec, ratio = build_reference_spec(wave, flux_masked, iOut_temp, reference_spec=reference_spec,
                                            box=mo_box, gauss_box=mo_gauss_box, 
                                                               clip_ratio=clip_ratio, cont=cont)
    print('flux_norm_mo all nan : {}'.format(flux_norm_mo.mask.all()))
    print('reference_spec all nan : {}'.format(reference_spec.mask.all()))
    if spec_trans is None:
        hm.print_static('Building the transmission spectrum #1 \n')
        spec_trans = flux_norm_mo/reference_spec                             # comment out the division to keep the master out
        print('spec-trans all nan : {}'.format(spec_trans.mask.all()))
    if poly_time is not None:
        if noise is None:
            noise = np.tile(np.sqrt(np.ma.median(np.clip(flux,0,None),axis=-1)[:,:,None]),(1,1,npix))
            #np.std(spec_trans, axis=0)**2
#         for iOrd in range(nord):
#             spec_trans[:,iOrd] = ext.col_remove(spec_trans[:,iOrd])
        # --- Polynomial fit on time --- #
        hm.print_static('Removing 2nd ord polynome in time \n')
        recon_time = np.ones_like(spec_trans)*np.nan
        x = poly_time
        z_t = np.zeros((nord, npix, 3))

        for iord in range(nord):

            for col in range(npix):

                y = spec_trans[:,iord, col]

                if y.mask.all():
                    continue

                idx = np.isfinite(x) & np.isfinite(y)

                # Poly.fit(x[idx], y[idx], 2)
                z_t[iord,col] = np.polyfit(x[idx], y[idx], 2, w=1/noise[:,iord,col][idx])

                recon_time[idx, iord, col] = np.poly1d(z_t[iord,col])(x[idx])

        recon_time = np.ma.masked_invalid(recon_time)
        spec_trans = spec_trans/recon_time
    else:
        recon_time = np.ones_like(spec_trans)

    # NOTE: passing a precomputed `clean_ts` to skip this block entirely is not used by any
    # caller in the codebase (grep-confirmed) and was already unreachable before this refactor
    # (it relied on `pca`/`rebuilt`/`ts_norm`/`final_ts`/`mask_last` being set by magic from
    # outside the function). Not reproducing that dead path here.
    if clean_ts is None:
        hm.print_static('Removing the static noise with PCA and sigma cliping \n')
#         print(n_pca, n_comps)
        print(spec_trans.shape)
        clean_ts, ts_norm, final_ts, rebuilt, mask_last, pca = apply_pca_truncation(
            spec_trans, n_pca, n_comps=n_comps, pca=pca, clip_ts=clip_ts, norm=norm, somme=somme,
            last_mask=last_mask, tresh_lim=tresh_lim, last_tresh=last_tresh, last_tresh_lim=last_tresh_lim)
        print('clean_ts all nan : {}'.format(clean_ts.mask.all()))

    return flux_norm, flux_norm_mo, reference_spec, spec_trans, clean_ts, ts_norm, \
           final_ts, rebuilt, pca, flux_Sref, flux_masked, ratio, mask_last, recon_time
#, flux_BARYref, flux_SYSref, flux_Sref



def build_trans_spectrum_mod_new(visit, flux, z=None, z_t=None, plot=False,
                                 clip=3, npc=3, id_ord=34, xlim=[None,None]):
    
    if z is None:
        z = visit.z
    if z_t is None:
        z_t = visit.z_t    
        
    if plot is True:
        plot_order(visit, id_ord, flux)
        plt.colorbar()
        plt.xlim(*xlim)
#     uncorr_lp=flux
#     spec_trans=flux
    # --- Polynomial fit on master spectrum --- #
    hm.print_static('{}/10'.format(6))
#     recon_poly = np.ones_like(flux)*np.nan
    
#     x = np.nanmedian(flux, axis=0)#[iord]
#     idx_x = np.isfinite(x)

#     for iord in range(nord):
#         hm.print_static('{}/10  - {}  '.format(6, iord))

#         for n in range(visit.n_spec):
#             idx = idx_x[iord] & np.isfinite(flux[n,iord])
#             recon_poly[n,iord, idx] = polyval(x[iord][idx], z[n, iord][::-1])
#         recon_poly[:,iord, idx[iord]] = polyval(x[iord][idx[iord]], z[:, iord][::-1].T, tensor=True)

    recon_poly = visit.recon_poly #np.ma.masked_invalid(recon_poly)

    if plot is True:
        plot_order(visit, id_ord, recon_poly)
#         pf.plot_order(visit, id_ord, recon_poly)
        plt.colorbar()
        plt.xlim(*xlim)
        plot_order(visit, id_ord, sigma_clip(recon_poly/visit.recon_poly, 3))
#         pf.plot_order(visit, id_ord, recon_poly)
        plt.colorbar()
        plt.xlim(*xlim)
        

    uncorr_nostar = flux/recon_poly
    
    if plot is True:
        plot_order(visit, id_ord, uncorr_nostar)
        plt.colorbar()
        plt.xlim(*xlim)

    # --- Polynomial fit on time --- #
    hm.print_static('{}/10'.format(7))
    recon_time = np.ones_like(uncorr_nostar)*np.nan
    x = visit.t.value
    idx = np.isfinite(x)
    for iord in range(nord):
        hm.print_static('{}/10  - {}  '.format(7, iord))
#         for col in range(visit.npix):
#             idx &= np.isfinite(uncorr_nostar[:,iord,col])
#             recon_time[idx, iord, col] = polyval(x[idx], z_t[iord,col][::-1], tensor=True)
        recon_time[idx, iord, :] = polyval(x[idx], z_t[iord,:].T[::-1], tensor=True).T
#         print((recon_time[idx, iord, :] == 0).sum())
    recon_time[recon_time == 0] = np.nan
        
    
    recon_time = np.ma.masked_invalid(recon_time)


#     # --- Polynomial fit on master spectrum --- #
#     hm.print_static('{}/10'.format(6))
#     recon_poly = np.ones_like(uncorr_lp)*np.nan
#     z = np.zeros((visit.n_spec, visit.nord, 3))

#     x = np.nanmedian(uncorr_lp,axis=0)
#     idx_x = np.isfinite(x)
#     for iord in range(nord):

#         if idx_x[iord].sum()==0:
#             continue
        
#         for n in range(visit.n_spec):

#             y = uncorr_lp[n,iord]

#             idx = idx_x[iord] & np.isfinite(y)

#             # Poly.fit(x[idx], y[idx], 2)
#             z[n,iord] = np.polyfit(x[iord][idx], y[idx], 2)
#             recon_poly[n,iord, idx] = np.poly1d(z[n,iord])(x[iord][idx])

#     recon_poly = np.ma.masked_invalid(recon_poly)

#     if plot is True:
#         pf.plot_order(visit, id_ord, recon_poly)
#         plt.colorbar()
#         plt.xlim(*xlim)

#     uncorr_nostar = uncorr_lp/recon_poly
#     if plot is True:
#         pf.plot_order(visit, id_ord, uncorr_nostar)
#         plt.colorbar()
#         plt.xlim(*xlim)

#     # --- Polynomial fit on time --- #
#     hm.print_static('{}/10'.format(7))
#     recon_time = np.ones_like(uncorr_nostar)*np.nan
#     x = visit.t.value
#     z_t = np.zeros((visit.nord, visit.npix, 3))

#     for iord in range(nord):

#         for col in range(visit.npix):

#             y = uncorr_nostar[:,iord, col]

#             if y.mask.all():
#                 continue

#             idx = np.isfinite(x) & np.isfinite(y)

#             # Poly.fit(x[idx], y[idx], 2)
#             z_t[iord,col] = np.polyfit(x[idx], y[idx], 2)

#             recon_time[idx, iord, col] = np.poly1d(z_t[iord,col])(x[idx])

#     recon_time = np.ma.masked_invalid(recon_time)


    if plot is True:
        plot_order(visit, id_ord, sigma_clip(recon_time/visit.recon_time,3) )
        plt.colorbar()
        plt.xlim(*xlim)
        
        plot_order(visit, id_ord, uncorr_nostar/recon_time )
        plt.colorbar()
        plt.xlim(*xlim)

    spec_trans = uncorr_nostar/recon_time
    
    if plot is True:
        plot_order(visit, id_ord, sigma_clip(spec_trans, 3) )
        plt.colorbar()
        plt.xlim(*xlim)
    
#     visit.reference_spec = recon_time*recon_poly
#     visit.spec_trans = spec_trans

    # --- PCA clean up --- #
    hm.print_static('{}/10'.format(8))
    if npc >0 :
        pca_clean_ts, rebuilt, pca = remove_dem_pca_all(spec_trans, n_pcs=npc)#, pca= visit.pca)
#         visit.pca = pca
#         visit.rebuilt = rebuilt 
    else:
        pca_clean_ts = spec_trans
#         visit.rebuilt = np.ones_like(spec_trans)

    if plot is True:
        plot_order(visit, id_ord, pca_clean_ts )
        plt.colorbar()
        plt.xlim(*xlim)
    
    # --- Cleaning remaining deviant pixels --- #
#     hm.print_static('{}/10'.format(9))
# #     new_mask = [ext.get_mask_noise(f, 3, 1., gwidth=0.01) for f in pca_clean_ts.swapaxes(0,1)]
# #     new_mask = new_mask | pca_clean_ts.mask
#     final_ts = np.ma.array(pca_clean_ts, mask=visit.final.mask)
    final_ts = pca_clean_ts

    # --- Mean removal --- #
    hm.print_static('{}/10'.format(10))
    final_ts = ext.quick_norm(final_ts, take_all=False)

    if plot is True:
        plot_order(visit, id_ord, final_ts )
        plt.colorbar()
        plt.xlim(*xlim)
    
#     visit.final = final_ts
#     visit.reconstructed = visit.reference_spec * visit.rebuilt * visit.ratio * (visit.blaze/np.nanmax(visit.blaze, axis=-1)[:,:,None])
    
    
    return final_ts  #, final_ts_std



def calc_stacked_spectra(visit, flux=None, weight=None, pca_red=False, kind='average', 
                         iOrd0=None, iin=None, iout=None, RV=None, vr=None, vrp=None, alpha=None, RV_star=None):
    
    if flux is None:
        flux = visit.spec_trans
    if weight is None:
        weight = visit.light_curve
    if iin is None:
        iin = visit.iIn
    if iout is None:
        iout = visit.iOut
    if vr is None:
        vr = visit.vr
    if vrp is None:
        vrp = visit.vrp   
    if alpha is None:
        alpha = visit.alpha_frac
    
    spec_fin, _ = build_stacked_st(visit.wv, flux[iin], vr[iin], vrp[iin], 
                                   weight[iin], kind=kind, iOrd0=iOrd0, RV=RV, alpha=alpha[iin])

    spec_fin_out, _ = build_stacked_st(visit.wv, flux[iout], vr[iout], vrp[iout],
                                       weight[iout], kind=kind, iOrd0=iOrd0, RV=RV, alpha=alpha[iout])

    if RV_star is None:
        spec_fin_Sref = np.ma.average(flux[iin], axis=0, weights=weight[iin])
    else:
        spec_fin_Sref, _ = build_stacked_st(visit.wv, flux[iin], np.zeros_like(vr)[iin], np.zeros_like(vrp)[iin],
                                       weight[iin], kind=kind, iOrd0=iOrd0, RV=RV_star, alpha=alpha[iin])
    
    if pca_red is True:
        spec_fin_ts, _ = build_stacked_st(visit.wv, visit.final[iin], vr, vrp[iin], 
                                          weight[iin], kind=kind, iOrd0=iOrd0, RV=RV, alpha=alpha[iin])
        return spec_fin, spec_fin_out, spec_fin_Sref, spec_fin_ts
    else:
        return spec_fin, spec_fin_out, spec_fin_Sref

