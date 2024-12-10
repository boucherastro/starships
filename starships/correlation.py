import numpy as np
import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt

from starships import homemade as hm
# import .analysis as a
# from spirou_exo.extract import quick_norm
from .spectrum import quick_inject_clean
# from spirou_exo.transpec import remove_dem_pca_all
from .orbite import rv_theo_nu, rv_theo_t

from .mask_tools import interp1d_masked
from .transpec import build_trans_spectrum_mod2, build_trans_spectrum_mod_fast, \
                                remove_dem_pca_all, build_trans_spectrum4



def quick_correl(wave, flux, corrRV, mod_x, mod_y, wave_ref=None, 
                     get_logl=False, kind='BL', mod2d=False, expand_mask=0, noise=None, somme=False):
    
    n_spec, nord, npix = flux.shape
    correl = np.ma.zeros((n_spec, nord, corrRV.size))
    
    if expand_mask > 0:
        flux = hm.expand_mask(flux, n_points=expand_mask)

    if wave_ref is None:
        wave_ref = np.mean(wave, axis=0)

    # - Calculate the shift -
    shifts = hm.calc_shift(corrRV, kind='rel')
    
    if mod2d is False:
        # - Interpolate over the orginal data
        fct = interp1d_masked(mod_x, mod_y, kind='cubic', fill_value='extrapolate')
    
    if get_logl is True:
        N = (~np.isnan(flux)).sum(axis=-1)
#         print(N.shape)

    if (get_logl is True):
        sig, flux_norm, s2f, cst = calc_logl_OG_cst(flux, axis=2, sig=noise)
    
    for iOrd in range(nord):
        hm.print_static('{} / {}'.format(iOrd+1,nord))

        if flux[:,iOrd].mask.all():
            continue
            
        if mod2d is True:
            # - Interpolate over the orginal data
            fct = interp1d_masked(mod_x[iOrd], mod_y[iOrd], kind='cubic', fill_value='extrapolate')
        
        
        # - Evaluate it at the shifted grid
        model = fct(wave_ref[iOrd][:, None] / shifts[None,:])[None,:,:] #/ shifts[None,:]
#         model = quick_norm(model, somme=somme, take_all=False)
        model -= model.mean(axis=1)
        if somme is True:
            model /= np.sqrt(np.ma.sum(model**2, axis=1))#[:,None,:]
        
#         if iOrd == 8:
#             print(model.shape)
#             plt.figure()
#             plt.plot(wave_ref[iOrd,1900:2200], flux[0,iOrd,1900:2200, ])
#             plt.plot(wave_ref[iOrd,1900:2200], model[0,1900:2200, 51])
#             hm.stop()

        if get_logl is True:
            if kind == 'BL':
#                 print(iOrd, flux[:, iOrd, :, None].shape, model.shape, N[:,iOrd].shape, s2f[:,iOrd].shape)
                correl[:, iOrd, :] = calc_logl_BL_ord(flux[:, iOrd, :, None], model, N[:,iOrd, None], 
                                                      axis=1)
            elif kind == 'GG':
                correl[:, iOrd, :] = calc_logl_BL_ord(flux[:, iOrd, :, None]/noise[:,iOrd,:, None], 
                                                      model/noise[:,iOrd,:, None], 
                                                      N[:,iOrd, None], axis=1)

            elif kind == 'OG':
                correl[:, iOrd, :] = calc_logl_OG_ord(flux_norm[:,iOrd], model, sig[:,iOrd],
                                                      cst[:,iOrd], s2f=s2f[:,iOrd], axis=1)
        else:
            correl[:, iOrd, :] = np.ma.sum(flux[:, iOrd, :, None] * model, axis=1)

    return np.ma.masked_invalid(correl)



def quick_correl_3dmod(wave, flux, corrRV, mod_x, mod_y, wave_ref=None, 
                     get_logl=False, kind='BL', mod_1d=False):
    
    n_spec, nord, npix = flux.shape
    correl = np.ma.zeros((n_spec, nord, corrRV.size))

    if wave_ref is None:
        wave_ref = np.mean(wave, axis=0)

    # - Calculate the shift -
    shifts = hm.calc_shift(corrRV, kind='rel')
  
    if get_logl is True:
        N = (~np.isnan(flux)).sum(axis=-1)

    if (get_logl is True):
        sig, flux_norm, s2f, cst = calc_logl_OG_cst(flux[:, :, :, None], axis=2)
    
    for iOrd in range(nord):
        hm.print_static('{}  /  {}  '.format(iOrd+1,nord))

        if flux[:,iOrd].mask.all():
            continue
        
        for n in range(n_spec):
            # - Interpolate over the orginal data
            if mod_1d is False:
                fct = interp1d_masked(mod_x[n, iOrd], mod_y[n, iOrd], kind='cubic', fill_value='extrapolate')
            else:
                fct = interp1d_masked(mod_x, mod_y[n], kind='cubic', fill_value='extrapolate')

            # - Evaluate it at the shifted grid
            model = fct(wave_ref[iOrd][:, None] / shifts[None,:])
            model -= model.mean(axis=0)[None,:]

            if get_logl is True:
                if kind == 'BL':
#                     print(flux[n, iOrd, :, None].shape, model.shape, N[n,iOrd, None].shape)
                    correl[n, iOrd, :] = calc_logl_BL_ord(flux[n, iOrd, :, None], model, N[n,iOrd, None],
                                                          s2f=s2f[n,iOrd], axis=0)

                if kind == 'OG':
                    correl[n, iOrd, :] = calc_logl_OG_ord(flux_norm[n,iOrd], model, sig[n,iOrd],
                                                          cst[n,iOrd], s2f[n,iOrd], axis=0)
            else:
                correl[n, iOrd, :] = np.ma.sum(flux[n, iOrd, :, None] * model, axis=0)

    return np.ma.masked_invalid(correl)


def CCF_1D(wave, flux, corrRV, mod_x, mod_y):

    # - Calculate the shift -
    shifts = hm.calc_shift(corrRV, kind='rel')
    
    # - Interpolate over the orginal data
    fct = interp1d_masked(mod_x, mod_y, kind='cubic', fill_value='extrapolate')


    # - Evaluate it at the shifted grid
    model = fct(wave[:, None] / shifts[None,:])#[None,:,:]
    model -= model.mean(axis=0)

    # - Calculate the cross-correlation
    correl = np.ma.sum(flux[:, None] * model, axis=0)

    return np.ma.masked_invalid(correl)


def sum_logl(loglbl, icorr, orders, N, alpha=None, axis=0, del_idx=None,
             nolog=True, verbose=False, N_ord=None, scaling=None, calc_snr=False):
    """Sum the log likelihood over the orders and the spectra.
    This may be done differently depending on the log likelihood prescription
    (e.g. Brogi 2019, 2 possible versions of Gibson 2020).
    When `nolog`=True, the Bro.
    """

    if orders is None:
        orders = slice(None)
 
    if del_idx is not None:
        correlation = loglbl.copy()
        correlation[del_idx] = np.nan

    else:
        correlation = loglbl.copy()
    if alpha is not None:
        while alpha.ndim != loglbl.ndim:
            alpha = np.expand_dims(alpha, axis=1)
        correlation = correlation * alpha#[:,None,None,None,None,None]
        
    if scaling is not None:
#         scaling = np.ones(loglbl.shape[1])
        scaling = scaling[None,:]
        while scaling.ndim != loglbl.ndim:
            scaling = np.expand_dims(scaling, axis=-1)    
        correlation = correlation * scaling

    if verbose:  print('')
    if nolog is False:
        if N is not None: 
            logL_i = nolog2log(correlation, N, sum_N=False)

            logL_sum = np.ma.masked_invalid( np.nansum( np.nansum( logL_i[icorr][:, orders], \
                                                                   axis=axis), axis=axis))

            if verbose:  
                print('The log has not been taken yet')
                print('Taking the log, with N_i')
                print('Summing over everything')
        else:
            if verbose:  print('The log has already been taken, N is None')
            if verbose:  print('Summing over everything')
            if N_ord is None:
                N_ord = np.ones_like(correlation)
            
#             if calc_snr is True:
#                 swaped = np.swapaxes(correlation, 0,1)
# #                 print(swaped.shape)
#                 flat_correl_ord = np.reshape(correlation, (swaped.shape[0], np.prod(swaped.shape[1:])))
# #                 print(flat_correl_ord.shape)
#                 std_correl = np.ma.std(flat_correl_ord, axis=-1)#[None,:, None, None, None, None]
# #                 print(std_correl.shape)

#                 swaped_new_corr = swaped/std_correl[:, None, None, None, None, None]
# #                 print(swaped_new_corr.shape)
#                 correlation = np.swapaxes(swaped_new_corr , 0,1)
# #                 print(correlation.shape)
            
            logL_sum =  np.nansum( np.nansum( correlation[icorr][:, orders], 
                                             axis=axis), axis=axis) / np.ma.sum(N_ord[icorr][:, orders])
            
            #*alpha[icorr][:, orders]
#             print('Is nan everywhere?', np.isnan(logL_sum).all())
    else:

        nologL_sum_i =  np.ma.masked_invalid( np.nansum( np.nansum( correlation[icorr][:, orders] \
                                                                   , axis=axis), axis=axis))
#         print('somme des L_i', nologL_i.shape) 

        logL_sum =  nolog2log( nologL_sum_i, N[icorr][:,orders], sum_N=True)
        
#         print(np.isnan(logL_sum).all(),np.isnan(logL_sum).sum())
#         print('log des somme des L', logL_sum.shape)
        if verbose:  
            print('The log has not been taken yet')
            print('Summing over everything prior to taking the log')
            print('Taking the log, with sum of N')

    if verbose:  print('')
#     print(logL_sum.shape)
    return np.ma.masked_invalid(logL_sum)


def calc_chi2(flux, sig, model, axis=-1):

    return np.ma.sum((flux-model)**2/sig**2, axis=axis)


def calc_logl_chi2_scaled(flux, yerr, model, log_f, axis=-1):

    sigma_2 = yerr**2 * np.exp(2 * log_f)
    out = -0.5 * np.ma.sum((flux - model)**2 / sigma_2 + np.log(sigma_2), axis=axis)
    return out

    
def calc_logl_OG_cst(flux, axis=-1, sig=None):
    if sig is None:
        sig = np.ma.std(flux, axis=0)[None,:,:]
    flux_norm = flux/sig
    s2f = np.ma.var(flux_norm, axis=axis)
    cst = np.ma.sum(np.log(sig), axis=axis) ## =2
    
    return sig, flux_norm, s2f, cst


def calc_logl_OG_ord(flux_norm, model, sig_ord, cst, s2f, axis=-1):

    model /= sig_ord
    R = np.ma.sum(flux_norm * model, axis=axis)
    s2g = np.ma.sum(model**2, axis=axis) 
     
    return R - cst - (s2f + s2g)/2  


def calc_logl_BL_ord(flux, model, N, alpha=1., s2f=None, axis=-1, nolog=False):
    R = np.ma.sum(flux * model, axis=axis) 
    if s2f is None:
#         s2f = np.ma.var(flux, axis=axis)
#     s2g = np.ma.var(model, axis=axis)
        s2f = np.ma.sum(flux**2, axis=axis)
    s2g = np.ma.sum(model**2, axis=axis)
    
    if nolog is True:
        return (s2f - 2 * alpha * R + alpha**2 * s2g)
    else:
        return - N / 2 * np.log( 1/N * (s2f - 2 * alpha * R + alpha**2 * s2g) )

    
def calc_corr_ord(flux, model, axis=-1, N=1):
    return np.ma.sum(flux * model, axis=axis) / N

def calc_corr_ord_BL(flux, model, s2f=None, axis=-1):
    
    R = np.ma.sum(flux * model, axis=axis) 
    if s2f is None:
        s2f = np.ma.sum(flux**2, axis=axis)
    s2g = np.ma.sum(model**2, axis=axis)
   
    return R/np.sqrt(s2f*s2g)
    
    
def calc_logl_BL_ord_parts(flux, model, N, s2f=None, axis=-1, nolog=True):
    R = np.ma.sum(flux * model, axis=axis) 
    if s2f is None:
#         s2f = np.ma.var(flux, axis=axis)
        s2f = np.ma.sum(flux**2, axis=axis)
    s2g = np.ma.sum(model**2, axis=axis)
#     s2g = np.ma.var(model, axis=axis)
    if nolog is True:
        return (s2f - 2 * R + s2g), R, s2f, s2g
    else:
        return - N / 2 * np.log( 1/N * (s2f - 2 * R + s2g) ), R, s2f, s2g
#     return - N / 2 * np.log( 1/N * (s2f - 2 * R + s2g) ), R, s2f, s2g

def calc_logl_G_corr_ord(flux, model, N, s2f=None, axis=-1, nolog=True):
    R = np.ma.sum(flux * model, axis=axis) 
    if s2f is None:
#         s2f = np.ma.var(flux, axis=axis)
#     s2g = np.ma.var(model, axis=axis)
        s2f = np.ma.sum(flux**2, axis=axis)
    s2g = np.ma.sum(model**2, axis=axis)
    
    if nolog is True:
        return (s2f - 2 * R + s2g), R
    else:
        return - N / 2 * np.log( 1/N * (s2f - 2 * R + s2g) ), R
    
def calc_logl_G_plain_ord(flux, model, uncert, N=None, s2f=None, alpha=1., beta=1., axis=-1):
    """Compute log likelihood for Gibson2020 without the optimization of the noise.
    s2f (= sum(flux**2 / uncert**2)) can be pre-computed and passed to the function to save time.
    """
    
    # Compute N if not given
    if N is None:
        N = np.sum(~flux.mask, axis=axis)

    # Divide by the uncertainty
    flux = flux / uncert
    model = model / uncert

    # Compute eache term of the chi2
    R = np.ma.sum(flux * model, axis=axis) 
    if s2f is None:
        s2f = np.ma.sum(flux**2, axis=axis)
    s2g = np.ma.sum(model**2, axis=axis)
    
    chi2 = (s2f - 2 * alpha * R + alpha**2 * s2g) / beta**2
    uncert_sum = np.ma.sum(np.log(uncert), axis=axis)
    cst = -N / 2 * np.log(2. * np.pi) - N * np.log(beta) - uncert_sum
    logl = cst - chi2 / 2
    
    return logl
    



def nolog2log(nolog_L, N, sum_N=True):  #, sumit=False, axis=0
#     if sumit is True:
#         nolog_L = np.ma.masked_invalid(np.nansum(nolog_L, axis=axis))
    if sum_N is True:
        return - np.ma.sum(N) / 2 * np.log( 1/np.ma.sum(N) * nolog_L)
    else:
        while N.ndim != nolog_L.ndim:
            N = np.expand_dims(N, axis=-1)
        return - N / 2 * np.log( 1/N * nolog_L)


def calc_log_likelihood_grid_retrieval(RV, data_tr, planet, wave_mod, model, flux=None, s2f=None,
                             vrp_orb=None, vr_orb=None,
                             nolog=True,  alpha=None, kind_trans="emission"):

    if flux is None:
        flux = data_tr['flux']

    if s2f is None:
        s2f = data_tr['s2f']

    # Simulated velocities = (specified rv) + (planet's rv) - (star's rv) + (systemic rv)
    velocities = RV + vrp_orb - vr_orb + data_tr['RV_const']

    model_seq = gen_model_sequence_noinj(velocities, 
                                         # data_tr['wave'], data_tr['sep'],
                                         # data_tr['pca'], int(data_tr['params'][5]), #data_tr['noise'],
                                         data_tr=data_tr,
                                         planet=planet,
                                         model_wave=wave_mod[20:-20],
                                         model_spec=model[20:-20],
                                         kind_trans=kind_trans,
                                         alpha=alpha,# resol=resol
                                         )
    
#     model_seq = gen_model_sequence_retrieval([RV, vrp_orb-vr_orb, data_tr['RV_const']], data_tr, 
#                                              planet, wave_mod[20:-20], model[20:-20], 
#                                         kind_trans=kind_trans, alpha=alpha,# resol=resol
    #                                         )

    mod = model_seq/data_tr['noise']

    logl_BL = np.ma.zeros((mod.shape[0], mod.shape[1]))
    for iOrd in range(mod.shape[1]):

        if flux[:,iOrd].mask.all():
            continue
            
        logl_BL[:, iOrd] = calc_logl_BL_ord(flux[:,iOrd], mod[:,iOrd], data_tr['N'][:,iOrd],
                                                     s2f=s2f[:,iOrd], nolog=nolog)

    if not np.isfinite(logl_BL).all():
        return -np.inf
    
    return logl_BL
    
    
def gen_model_sequence_noinj(velocities, data_wave=None, data_sep=None, data_pca=None, data_npc=None, #data_noise,
                             planet=None, model_wave=None, model_spec=None, #resol=64000,norm=True,debug=False,
                            alpha=None, data_tr=None, data_recon=None,  **kwargs):
    """
    alpha: np.ndarray of shape (n_spec,)
        Fraction of planetary signal. Depends on `kind_trans`:
        - If 'transmission': fraction of the stellar disk hidden by the planet
        - If 'emission': fraction of the planetary disk not hidden by the star
    """

    if data_wave is None:
        data_wave = data_tr['wave']

    if data_sep is None:
        data_sep = data_tr['sep']

    if data_pca is None:
        data_pca = data_tr['pca']
    if data_npc is None:
        data_npc = int(data_tr['params'][5])
    if data_recon is None:
        # -- Uncomment the other 2 lines if you want the full reconstructed data to inject the model in
        data_recon = np.ones_like(data_wave)
        # data_recon = data_tr['reconstructed']
        # data_recon = data_recon/np.ma.median(data_recon,axis=-1)[:,:,None]/data_tr['ratio']/data_tr['mast_out']

    for arg in (planet, model_wave, model_spec):
        if arg is None:
            raise ValueError('`planet`, `model_wave` and `model_spec` need to be specified.')

    # --- inject model in an empty sequence of ones
    flux_inj, _ = quick_inject_clean(data_wave, data_recon,
                                                  model_wave, model_spec, 
                                                 velocities, data_sep, planet.R_star, planet.A_star, 
                                                                  RV=0.0, dv_star=0., 
                                                 R0 = planet.R_pl, alpha=alpha, **kwargs)
   
    # -- Remove the same number of pcas that were used to inject
    model_seq = build_trans_spectrum_mod_fast(flux_inj, data_pca, n_pca=data_npc)

    return model_seq


def gen_model_sequence_retrieval(velocities, data_tr, planet, model_wave, model_spec, #resol=64000, debug=False,
                                 alpha=None, norm=True, **kwargs):
    
    flux_inj, _ = quick_inject_clean(data_tr['wave'], data_tr['reconstructed'], 
                                                  model_wave, model_spec, 
                                                 np.sum(velocities), data_tr['sep'], planet.R_star, planet.A_star, 
                                                                  RV=0.0, dv_star=0., 
                                                 R0 = planet.R_pl, alpha=alpha, **kwargs)
   
    # -- Remove the same number of pcas that were used to inject

    model_seq = build_trans_spectrum_mod2(data_tr['wave'], flux_inj, data_tr['mast_out'], 
                                          data_tr['pca'], data_tr['noise'],
                                               plot=False, norm=norm, 
                                              ratio=data_tr['ratio'], #debug=debug, #blaze=blaze, 
#                           mo_box=data_tr['params'][2], mo_gauss_box=data_tr['params'][4], 
                                          n_pca=int(data_tr['params'][5]), n_comps=10)

    return model_seq



def unload_data(data_obj, kind_obj, 
                final=None, spec_trans=None, noise=None, ratio=None,
                pca=None, alpha=None, inj_alpha='ones',):
    
    if kind_obj == 'dict':
        if spec_trans is None:
            spec_trans = data_obj['spec_trans']

        if pca is None:
            pca= data_obj['pca']

        if final is None:
            final = data_obj['final']   

        N = data_obj['N']

        if noise is None:
            noise = data_obj['noise']
        if alpha is None:
            print('Injecting model w/ alpha = {}'.format(inj_alpha))
            if inj_alpha =='alpha':
                alpha = data_obj['alpha_frac']
            elif inj_alpha == 'ones':
                alpha = np.ones_like(data_obj['alpha_frac'])

        if ratio is None:
#             if tr.ratio_recon is True:
            ratio = data_obj['ratio']
#             else:
#                 ratio = None
        params0 = data_obj['params']
        scaling = data_obj['scaling']
        wave, vrp, icorr, clip_ts = data_obj['wave'], data_obj['vrp'], data_obj['icorr'], data_obj['clip_ts']
        t_start, RV_const, sep =  data_obj['t_start'], data_obj['RV_const'], data_obj['sep']
#         nu =data_obj['nu']
        
    if kind_obj == 'seq':
        if spec_trans is None:
            spec_trans = data_obj.spec_trans

        if pca is None:
            pca=data_obj.pca

        if final is None:
            final = data_obj.final   
        if noise is None:
            noise = data_obj.noise
        if alpha is None:
            print('Injecting model w/ alpha = {}'.format(inj_alpha))
            if inj_alpha =='alpha':
                alpha = data_obj.alpha_frac
            elif inj_alpha == 'ones':
                alpha = np.ones_like(data_obj.alpha_frac)

        if ratio is None:
            if data_obj.ratio_recon is True:
                ratio = data_obj.ratio
            else:
                ratio = None
        N = data_obj.N
        params0 = data_obj.params
        scaling = data_obj.scaling
        wave, vrp, icorr, clip_ts = data_obj.wave, data_obj.vrp, data_obj.icorr, data_obj.clip_ts
        t_start, RV_const, sep = data_obj.t_start, data_obj.RV_const, data_obj.sep
#         nu = data_obj.nu
    return spec_trans, pca, final, N, noise, alpha, ratio, \
            params0, wave, vrp, icorr, clip_ts, t_start, RV_const, sep, scaling


def calc_logl_injred(data_obj, kind_obj, planet, Kp_array, corrRV, n_pcas, wave_mod, models, 
                                 # resol,
                     kind_trans, final=None, spec_trans=None, noise=None, ratio=None,
                                 pca=None, alpha=None, inj_alpha='ones',
                                 get_GG=True, vrp_kind='t', nolog=True, 
                                 change_noise=False, force_npc=None, filename=None, **kwargs):
    
    if models.ndim < 2:
        models = models[None,:]
        
        
    spec_trans, pca, final, \
    N, noise, alpha, ratio, \
    params0, wave, vrp, icorr, \
    clip_ts, t_start, RV_const, sep, scaling = unload_data(data_obj, kind_obj, 
                                                      final=final, spec_trans=spec_trans, 
                                                      noise=noise, ratio=ratio,
                                                      pca=pca, alpha=alpha, inj_alpha=inj_alpha)

    n_spec, nord, _ = final.shape                   # change final for spec_trans to use step F of reduction in the correlation
#     if get_bl is True:
#         logl_BL = np.ma.zeros((tr.n_spec, tr.nord, Kp_array.size, corrRV.size, len(n_pcas), modelTD0.shape[0]))
    correl = np.ma.zeros((n_spec, nord, Kp_array.size, corrRV.size, len(n_pcas), models.shape[0]))
    logl_BL_sig = np.ma.zeros((n_spec, nord, Kp_array.size, corrRV.size, len(n_pcas), models.shape[0]))

    print('Starting with {} PCs'.format(params0[5]))
    params = params0.copy()
    for n, n_pc in enumerate(n_pcas):
#         print('n',n)
        print('Computing with'.format(n_pc))

        if (int(params[5]) != n_pc) or (change_noise is True):
            print(' Previous N_pc = {}, changing to {}  '.format(int(params[5]), n_pc))
            params[5] = n_pc
            print('Building final transmission spectrum with {} n_pc'.format(n_pc))

            # --- Will recompute the final transmission spectrum from spec_trans and pca 
            # --- (the rest is ignored)
            final, rebuilt, pca,  = build_trans_spectrum4(wave, spec_trans,   #_, _, _, mask_last, _
                                     vrp, planet.RV_sys, vrp, icorr, tellu=spec_trans, noise=noise, 
                                    lim_mask=params[0], lim_buffer=params[1],
                                    mo_box=params[2], mo_gauss_box=params[4],
                                    tresh=params[6], tresh_lim=params[7],
                                    last_tresh=params[8], last_tresh_lim=params[9], 
                                    n_pca=int(params[5]), clip_ts=clip_ts, clip_ratio=6, poly_time=None, 
                                    flux_masked=spec_trans, flux_Sref=spec_trans, flux_norm=spec_trans, 
                    flux_norm_mo=spec_trans, master_out=spec_trans, spec_trans=spec_trans, mask_var=False,
                                iOut_temp='all', cont=False)[6:9]

            N = (~np.isnan(final)).sum(axis=-1)
            
            if change_noise is True:
                print('Calculating noise with {} PCs'.format(params[5]))
                sig_col = np.ma.std(final, axis=0)[None,:,:]  #self.final  # self.spec_trans
                noise = sig_col*scaling
            #last_mask=False, n_comps=tr.n_comps, 

#             rebuilt = tr.rebuilt
#             final = np.ma.array(final, mask=mask_last) 
#             pca = tr.pca
            if kind_obj == 'seq':
                data_obj.params = params
                # params0 = params
                data_obj.N = N
            elif kind_obj == 'dict':
                data_obj['params'] = params
                # params0 = params
                data_obj['N'] = N
            
        if get_GG is True:
            flux = final/noise                     # change final for spec_trans to use step F of reduction in the correlation
    #         flux -= np.ma.mean(flux, axis=-1)[:,:,None]
        else:
            flux = final                     # change final for spec_trans to use step F of reduction in the correlation
        
        s2f_sig = np.ma.sum(flux**2, axis=-1)
        
#         if (get_bl is True) | (sfsg is True):
#             s2f = np.ma.sum(final**2, axis=-1)
        if force_npc is not None:
            n_pc_mod = force_npc
            print('Forcing model with : {}'.format(n_pc_mod))
        else:
            n_pc_mod = n_pc
            
        for f,specMod in enumerate(models):
#             print('f',f)

            for i,Kpi in enumerate(Kp_array):
#                 print('i',i)
                if vrp_kind == 'nu':
                    vrp_orb = rv_theo_nu(Kpi, nu*u.rad, planet.w, plnt=True).value
                elif vrp_kind == 't':
                    vrp_orb = rv_theo_t(Kpi, t_start*u.d, planet.mid_tr, planet.period, plnt=True).value

                vr_orb = -vrp_orb*(planet.M_pl/planet.M_star).decompose().value

                for v,vrad in enumerate(corrRV):
#                     print('v',v)
                    hm.print_static('         N_pca = {}, Kp = {}/{} = {:.2f}, File = {}/{}, RV = {}/{}  '.format(\
                                             n_pc, i+1,len(Kp_array),Kpi, f+1, models.shape[0], v+1,corrRV.size))

                    velocities = vrad + vrp_orb - vr_orb + RV_const
                    model_seq = gen_model_sequence_noinj(velocities,
                                         wave, sep, pca, int(params[5]),
                                         planet, wave_mod[20:-20], specMod[20:-20],
                                         kind_trans=kind_trans, alpha=alpha, #resol=resol,
                                                         **kwargs)
    
#                     model_seq = gen_model_sequence_retrieval([RV, vrp_orb-vr_orb, data_tr['RV_const']], data_tr, 
#                                                              planet, wave_mod[20:-20], specMod[20:-20], 
#                                                         kind_trans=kind_trans, alpha=alpha, #resol=resol,
#                                                         **kwargs)

                    if get_GG is True:
                        mod = model_seq/noise
                    else:
                        mod = model_seq

                    for iOrd in range(nord):
#                         print('iOrd',iOrd)

                        if final[:,iOrd].mask.all():     # change final for spec_trans to use step F of reduction in the correlation
                            continue

                        logl_BL_sig[:, iOrd, i, v, n, f],\
                        correl[:, iOrd, i, v, n, f] = calc_logl_G_corr_ord(flux[:,iOrd], mod[:,iOrd], 
                                                                            N[:,iOrd], s2f=s2f_sig[:,iOrd],
                                                                            nolog=nolog)
    out = []
    out.append(correl)
#     if get_GG is True:
    out.append(logl_BL_sig)
#     if get_bl is True:
#         out.append(logl_BL)

    if filename is not None:
        save_logl_seq(filename, correl, logl_BL_sig,
                      wave_mod[20:-20], specMod[20:-20], n_pcas, Kp_array, corrRV, kind_trans)
    
    return out
    

def gen_model_sequence(theta, tr, model_wave, model_spec, n_pcs=None,
                       # resol=64000, blaze=None, debug=False, iOut=None,
                       pca=None, norm=True, alpha=None,
                       reconstructed=None, ratio=None,
                       master_out=None,
                       **kwargs):
    
    vrad, vrp_orb, v_star = theta
#     vr = tr.vr
#     if isinstance(vr, u.Quantity):
#         vr = vr.to(u.km/u.s).value
    if master_out is None:
        master_out = tr.mast_out
    if pca is None:
        pca = tr.pca
    if reconstructed is None:
        reconstructed = tr.reconstructed
    if alpha is None:
        alpha = tr.alpha_frac
    if n_pcs is None:
        n_pcs=tr.params[5]
        
#     if vrad.ndim == 1 :
#         dv_pl = vrp_orb[:,None]+v_star+vrad[None,:]
#     else:
#     dv_pl = 

#     tr.inject_signal(model_wave, model_spec, RV=RV, dv_pl=vrp_orb+v_star, #-vr, #+tr.planet.RV_sys.value, 
#                      resol=resol, P_R=resol, flux=tr.reconstructed, alpha=np.ones_like(tr.alpha_frac), 
#                      wv_borders=[tr.wv_ext, tr.wv_bins], **kwargs) 
    tr.inject_signal(model_wave, model_spec, dv_pl=vrp_orb+v_star+vrad, #dv_star=0, RV=0, #-vr, #+tr.planet.RV_sys.value, 
                    flux=reconstructed, alpha=alpha,  **kwargs) 


    model_seq = build_trans_spectrum_mod2(tr.wave, tr.flux_inj, master_out, pca, tr.noise,
                                         plot=False, norm=norm, ratio=ratio,
#                           mo_box=tr.params[2], mo_gauss_box=tr.params[4], 
                                          n_pca=n_pcs, n_comps=10)
   
    return np.ma.masked_array(model_seq)


# import matplotlib.pyplot as plt

def quick_calc_logl_injred_class(tr, Kp_array, corrRV, n_pcas, modelWave0, modelTD0, 
                                 # resol,
                                 final=None, spec_trans=None, noise=None,
                                 nolog=True, pca=None, norm=True, alpha=None, inj_alpha='ones',
                                 get_GG=True, RVconst = 0.0, new_mask=None,
                                 vrp_kind='t',  master_out=None, iOut=None, ratio=None,
                                 reconstructed=None, change_noise=False, force_npc=None,
                                 filename=None, **kwargs):
    
    if modelTD0.ndim < 2:
        modelTD0 = modelTD0[None,:]
    if spec_trans is None:
#         rebuilt, pca = tr.rebuilt, tr.pca
        spec_trans = tr.spec_trans

    if pca is None:
        pca=tr.pca
        
    if final is None:
        final = tr.final   
    if noise is None:
        noise = tr.noise
    if alpha is None:
        print('Injecting model w/ alpha = {}'.format(inj_alpha))
        if inj_alpha =='alpha':
            alpha = tr.alpha_frac
        elif inj_alpha == 'ones':
            alpha = np.ones_like(tr.alpha_frac)
    
    if ratio is None:
        if tr.ratio_recon is True:
            ratio = tr.ratio
        else:
            ratio = None
    
#     if get_bl is True:
#         logl_BL = np.ma.zeros((tr.n_spec, tr.nord, Kp_array.size, corrRV.size, len(n_pcas), modelTD0.shape[0]))
    correl = np.ma.zeros((tr.n_spec, tr.nord, Kp_array.size, corrRV.size, len(n_pcas), modelTD0.shape[0]))
    logl_BL_sig = np.ma.zeros((tr.n_spec, tr.nord, Kp_array.size, corrRV.size, len(n_pcas), modelTD0.shape[0]))
    
    for n,n_pc in enumerate(n_pcas):
#         print('n',n)
        
        params = tr.params
        if (params[5] != n_pc) or (change_noise is True):
            print(' Previous N_pc = {}, changing to {}  '.format(params[5], n_pc))
            params[5] = n_pc
            print('Building final transmission spectrum with {} n_pc'.format(n_pc))
            tr.build_trans_spec(flux=spec_trans, params=params, 
                    flux_masked=tr.fl_masked, flux_Sref=tr.fl_Sref, flux_norm=tr.fl_norm, 
                    flux_norm_mo=tr.fl_norm_mo, master_out=tr.mast_out, spec_trans=spec_trans, mask_var=False,
                                change_noise=change_noise, iOut_temp='all', ratio_recon=tr.ratio_recon, 
                                clip_ts=tr.clip_ts, clip_ratio=tr.clip_ratio, cont=False)
            #last_mask=False, n_comps=tr.n_comps, 

            # rebuilt = tr.rebuilt
            final = np.ma.array(tr.final, mask=tr.last_mask) 
            pca = tr.pca
            # reconstructed=tr.reconstructed
            
        if get_GG is True:
            flux = final/noise
    #         flux -= np.ma.mean(flux, axis=-1)[:,:,None]
        else:
            flux = final

        if new_mask is not None:
            flux = np.ma.array(flux, mask=new_mask)

        s2f_sig = np.ma.sum(flux**2, axis=-1)
        
#         if (get_bl is True) | (sfsg is True):
#             s2f = np.ma.sum(final**2, axis=-1)
        if force_npc is not None:
            n_pc_mod = force_npc
            print('Forcing model with : {}'.format(n_pc_mod))
        else:
            n_pc_mod = n_pc
            
        for f,specMod in enumerate(modelTD0):
#             print('f',f)

            for i,Kpi in enumerate(Kp_array):
#                 print('i',i)
                if vrp_kind == 'nu':
                    vrp_orb = rv_theo_nu(Kpi, tr.nu*u.rad, tr.planet.w, plnt=True).value
                elif vrp_kind == 't':
                    vrp_orb = rv_theo_t(Kpi, tr.t, tr.planet.mid_tr, tr.planet.period, plnt=True).value

                vr_orb = -vrp_orb*(tr.planet.M_pl/tr.planet.M_star).decompose().value

                for v,vrad in enumerate(corrRV):
#                     print('v',v)
                    hm.print_static('         N_pca = {}, Kp = {}/{} = {:.2f}, File = {}/{}, RV = {}/{}  '.format(\
                                             n_pc, i+1,len(Kp_array),Kpi, f+1, modelTD0.shape[0], v+1,corrRV.size))
    
                    model_seq = gen_model_sequence([vrad, vrp_orb-vr_orb, RVconst], tr, modelWave0, specMod,  
                                                   pca=pca, n_pcs=n_pc_mod, norm=norm,
                                                   reconstructed=reconstructed, ratio=ratio, 
                                                   #blaze=blaze, debug=debug,  iOut=iOut,#resol=resol,
                                                   master_out=master_out, alpha=alpha, **kwargs)

                    if get_GG is True:
                        mod = model_seq/noise
                    else:
                        mod = model_seq
#                     mod -= np.ma.mean(mod, axis=-1)[:,:,None]
                    
#                     if debug is True:
#                         plt.figure()
#                         plt.imshow(model_seq[:,34][:,1900:2000])

                    for iOrd in range(tr.nord):
#                         print('iOrd',iOrd)

                        if final[:,iOrd].mask.all():
                            continue


                        logl_BL_sig[:, iOrd, i, v, n, f],\
                        correl[:, iOrd, i, v, n, f] = calc_logl_G_corr_ord(flux[:,iOrd], mod[:,iOrd], 
                                                                            tr.N[:,iOrd], s2f=s2f_sig[:,iOrd],
                                                                            nolog=nolog)
    out = []
    out.append(correl)
#     if get_GG is True:
    out.append(logl_BL_sig)
#     if get_bl is True:
#         out.append(logl_BL)
    if filename is not None:
        save_logl_seq(filename, correl, logl_BL_sig,
                      wave_mod[20:-20], specMod[20:-20], n_pcas, Kp_array, corrRV, kind_trans)
    
    return out



def quick_calc_logl_injred_class_parts(tr, Kp_array, corrRV, n_pcas, modelWave0, modelTD0, 
                                 resol=70000, final=None, spec_trans=None, noise=None, 
                                 debug=False, nolog=True, pca=None, norm=True, alpha=None, inj_alpha='ones',
                                 get_corr=True, get_GG=True, get_bl=False, sfsg=True, RVconst=0,
                                 mid_id_nu = None, vrp_kind='t',  master_out=None, iOut=None,
                                 reconstructed=None, blaze=None, **kwargs):
    
    if modelTD0.ndim < 2:
        modelTD0 = modelTD0[None,:]
    if spec_trans is None:
#         rebuilt, pca = tr.rebuilt, tr.pca
        spec_trans = tr.spec_trans
#     else:
#         _, rebuilt, pca = corr.remove_dem_pca_all(spec_trans, n_pcs=n_pcas[0], n_comps=tr.n_comps, plot=False)
    if pca is None:
        pca=tr.pca
        
    if final is None:
        final = tr.final   
    if noise is None:
        noise = tr.noise
    if alpha is None:
        print('Injecting model w/ alpha = {}'.format(inj_alpha))
        if inj_alpha =='alpha':
            alpha = tr.alpha_frac
        elif inj_alpha == 'ones':
            alpha = np.ones_like(tr.alpha_frac)
            
    if tr.ratio_recon is True:
        ratio = tr.ratio
    else:
        ratio = None
    
    logl_BL_sig = np.ma.zeros((tr.n_spec, tr.nord, Kp_array.size, corrRV.size, len(n_pcas), modelTD0.shape[0]))
    R_sig = np.ma.zeros((tr.n_spec, tr.nord, Kp_array.size, corrRV.size, len(n_pcas), modelTD0.shape[0]))
    s2f_val_sig = np.ma.zeros((tr.n_spec, tr.nord, Kp_array.size, corrRV.size, len(n_pcas), modelTD0.shape[0]))
    s2g_val_sig = np.ma.zeros((tr.n_spec, tr.nord, Kp_array.size, corrRV.size, len(n_pcas), modelTD0.shape[0]))
    
    for n,n_pc in enumerate(n_pcas):
#         print('n',n)

        params = tr.params
        if params[5] != n_pc:
            print(' Previous N_pc = {}, changing to {}  '.format(params[5], n_pc))
            params[5] = n_pc
            print('Building final transmission spectrum with {} n_pc'.format(n_pc))
            tr.build_trans_spec(flux=spec_trans, params=params, flux_masked=tr.fl_norm, 
                                  flux_Sref=tr.fl_norm, 
                                  flux_norm=tr.fl_norm, spec_trans=spec_trans, 
                                  flux_norm_mo=tr.fl_norm_mo, master_out=tr.mast_out, 
                                  last_mask=False, n_comps=tr.n_comps, mask_var=False)
            rebuilt = tr.rebuilt
            final = tr.final
            final.mask = tr.last_mask
            pca = tr.pca
            
        if get_GG is True:
            flux = final/noise
        else:
            flux = final
    #         flux -= np.ma.mean(flux, axis=-1)[:,:,None]
        s2f_sig = np.ma.sum(flux**2, axis=-1)
        
#         if (get_bl is True) | (sfsg is True):
#             s2f = np.ma.sum(final**2, axis=-1)

        for f,specMod in enumerate(modelTD0):
#             print('f',f)

            for i,Kpi in enumerate(Kp_array):
#                 print('i',i)
                if vrp_kind == 'nu':
                    vrp_orb = rv_theo_nu(Kpi, tr.nu*u.rad, tr.planet.w, plnt=True).value
                elif vrp_kind == 't':
                    vrp_orb = rv_theo_t(Kpi, tr.t, tr.planet.mid_tr, tr.planet.period, plnt=True).value
        
#                 if mid_id_nu is not None:
#                     vrp_orb = vrp_orb - rv_theo_nu(Kpi, mid_id_nu*u.rad, tr.planet.w, plnt=True).value
                vr_orb = -vrp_orb*(tr.planet.M_pl/tr.planet.M_star).decompose().value
                
                for v,vrad in enumerate(corrRV):
#                     print('v',v)
                    hm.print_static('         N_pca = {}, Kp = {}/{} = {:.2f}, File = {}/{}, RV = {}/{}  '.format(\
                                             n_pc, i+1,len(Kp_array),Kpi, f+1, modelTD0.shape[0], v+1,corrRV.size))

                    model_seq = gen_model_sequence([vrad, vrp_orb-vr_orb, RVconst], tr, modelWave0, specMod, pca=pca,
                                                   n_pcs=n_pc,  norm=norm,
                                                   # resol=resol,blaze=blaze, iOut=iOut,debug=debug,
                                                   reconstructed=reconstructed, ratio=ratio,
                                                   master_out=master_out,  alpha=alpha, **kwargs)
                    if get_GG is True:
                        mod = model_seq/noise
                    else:
                        mod = model_seq
#                     mod -= np.ma.mean(mod, axis=-1)[:,:,None]
                    
                    if debug is True:
                        plt.figure()
                        plt.imshow(model_seq[:,34][:,1900:2000])

                    for iOrd in range(tr.nord):
#                         print('iOrd',iOrd)

                        if final[:,iOrd].mask.all():
                            continue
#                         if nolog is False:

                        logl_BL_sig[:, iOrd, i, v, n, f], \
                        R_sig[:, iOrd, i, v, n, f], \
                        s2f_val_sig[:, iOrd, i, v, n, f], \
                        s2g_val_sig[:, iOrd, i, v, n, f] = calc_logl_BL_ord_parts(flux[:,iOrd], \
                                                                                  mod[:,iOrd], tr.N[:,iOrd], \
                                                                                  s2f=s2f_sig[:,iOrd], nolog=nolog)

#     out = []
#     if get_corr is True:
#         out.append(correl)
#     if get_sig is True:
#         out.append(logl_BL_sig)
#     if get_bl is True:
#         out.append(logl_BL)
    
    return [R_sig, logl_BL_sig, s2f_val_sig, s2g_val_sig]

def calc_split_correl(trb1, trb2, corrRV, Wave0, Model0, tr_merged=None, **kwargs):
    
    trb1.calc_correl(corrRV, Wave0, Model0, kind="BL", **kwargs)
    trb2.calc_correl(corrRV, Wave0, Model0, kind="BL", **kwargs)

    tr_new_correl = np.concatenate((trb1.correl, trb2.correl), axis=0)
    tr_new_logl = np.concatenate((trb1.logl, trb2.logl), axis=0)
    
    if tr_merged is not None:
        tr_merged.correl = tr_new_correl
        tr_merged.logl = tr_new_logl
    return tr_new_correl, tr_new_logl


def calc_logl_split_transits(trb1, trb2, Kp_array, corrRV0, n_pcas, Wave0, Model0, 
                             nolog=False, get_bl=False, **kwargs):
    
#     npc1 = trb1.params[5]
    og_npc2 = trb2.params[5]
    
    if get_bl is True:
        loblBL_p1, loblBL_p1_sig, loblBL_p1_BL = quick_calc_logl_injred_class(trb1, Kp_array, corrRV0, n_pcas, 
                                                                Wave0, Model0, nolog=nolog, **kwargs)
        trb2.params[5] = og_npc2
        print('')
        loblBL_p2, loblBL_p2_sig, loblBL_p2_BL = quick_calc_logl_injred_class(trb2, Kp_array, corrRV0, n_pcas, 
                                                                Wave0, Model0, nolog=nolog, **kwargs)
    else:
        loblBL_p1, loblBL_p1_sig = quick_calc_logl_injred_class(trb1, Kp_array, corrRV0, n_pcas, 
                                                            Wave0, Model0, nolog=nolog, **kwargs)
        trb2.params[5] = og_npc2
        print('')
        loblBL_p2, loblBL_p2_sig = quick_calc_logl_injred_class(trb2, Kp_array, corrRV0, n_pcas, 
                                                                Wave0, Model0, nolog=nolog, **kwargs)
    
    merge = np.concatenate((loblBL_p1,loblBL_p2), axis=0)
    merge_sig = np.concatenate((loblBL_p1_sig,loblBL_p2_sig), axis=0)
    
    if get_bl is True:
        merge_BL = np.concatenate((loblBL_p1_BL,loblBL_p2_BL), axis=0)

        return merge, merge_sig, merge_BL
    else:
        return merge, merge_sig


def save_logl_seq(filename, corr, logl, wave_mod, model, n_pcas, Kp_array, corrRV0, kind_trans):
    np.savez(filename, corr=corr, logl=logl, wave_mod=wave_mod, model=model,
             n_pcas=n_pcas, Kp_array=Kp_array, corrRV0=corrRV0, kind_trans=kind_trans)
    try:
        print('Saved correl at :', filename.with_suffix(".npz"))
    except AttributeError:
        print('Saved correl at :', filename + ".npz")


def load_logl_seq(filename):
    print('Loading correl from :', filename + '.npz')
    data = np.load(filename + '.npz')
    return data['corr'], data['logl'], data['Kp_array'], data['n_pcas'], data['corrRV0'], data
