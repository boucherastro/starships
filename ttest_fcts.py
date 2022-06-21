import numpy as np
# from astropy.table import Table
from starships import homemade as hm
# from spirou_exo.spectrum import find_R, quick_inject
from starships.analysis import bands, gauss
# from spirou_exo import transpec as ts
from starships import correlation as corr
from starships.orbite import rv_theo_nu, rv_theo_t
from starships.mask_tools import interp1d_masked
from mpl_toolkits.axes_grid1 import make_axes_locatable

import scipy as sp
import scipy.constants as cst
from scipy.optimize import curve_fit

from astropy import units as u
from astropy import constants as const
from astropy.stats import sigma_clip


import matplotlib.pyplot as plt
# from itertools import islice

# from astropy.convolution import convolve, Gaussian1DKernel



from matplotlib.ticker import PercentFormatter

def get_corr_in_out_trail(index, corrRV, ccf, tr, \
                          wind=0, speed_limit=4, limit_out=10, 
                          both_side=True, vrp=None, verbose=False):
    
    if vrp is None:
        vrp = tr.vrp.value
    
    in_ccf = []
    out_ccf = []

    for i in index:
#         
        idx_in = np.where((corrRV <= vrp[i]+wind+speed_limit) & \
                          (corrRV >= vrp[i]+wind-speed_limit))
#         if len(idx_in) < 1:
#             raise Exception("Sorry, no numbers below zero")

#         print(idx_in)
        if (i == 0) and (verbose is True):
            print(idx_in)
            print(corrRV[idx_in])
        try:
            in_ccf += list(ccf[i,idx_in].squeeze())
        except TypeError: 
            print(vrp[i]+wind+speed_limit, vrp[i]+wind-speed_limit, vrp[i],wind,speed_limit)
            print(idx_in, len(idx_in))
        
        if both_side is True:
            idx_out = np.where((corrRV > vrp[i]+wind+limit_out) | \
                           (corrRV < vrp[i]+wind-limit_out))
        else:
            idx_out = np.where((corrRV > vrp[i]+wind+limit_out))
#             idx_out = np.where((corrRV <= tr.vrp[index[i]].value+wind+limit_out+speed_limit) & \
#                           (corrRV >= tr.vrp[index[i]].value+wind+limit_out-speed_limit))
#         print(idx_out)
        try :
            out_ccf += list(ccf[i,idx_out].squeeze())
        except TypeError: 
            print(idx_in, len(idx_in))

    return in_ccf, out_ccf


def t_test_hist(sample1, sample2, label1, label2, title, ax=None, nb_x_gauss=101, 
                p0_estim=None, fig_name=''):
    
    if ax is None:
        fig,ax = plt.subplots(1,1, figsize=(8,5))
    
#     print(len(sample1),len(sample2))
    n, bins, patches = ax.hist(x=sample2, color='#0504aa', bins=20,
                                alpha=0.5, rwidth=0.85, weights=np.ones(len(sample2)) / len(sample2),
                                label=label2)
    
    if p0_estim is None:
        p0_estim = np.array([np.nanstd(sample2),n.max(),np.nanmean(sample2)])
   
    mids = 0.5*(bins[1:] + bins[:-1])
    x = np.linspace(mids.min(), mids.max(), nb_x_gauss)
    
    param, pcov = curve_fit(gauss, ydata=n, xdata=mids, p0=p0_estim)
#     print(param)
    ax.plot(x, gauss(x,*param), color='darkblue')#, label='Best-fit Gaussian')

#     print(sample2.size, sample1.size)
    x = np.linspace(np.nanmin(sample2), np.nanmax(sample2),101)
    n, bins, patches = ax.hist(x=sample1, color='limegreen', bins=bins,
                                alpha=0.5, rwidth=0.85, weights=np.ones(len(sample1)) / len(sample1),
                                label=label1)
    mids = 0.5*(bins[1:] + bins[:-1])
    try:
        param, pcov = curve_fit(gauss, ydata=n, xdata=mids, p0=p0_estim)
    except RuntimeError:
        print('No gauss found')

#     print(param)
    ax.plot(x, gauss(x,*param), color='darkgreen')#, label='Best-fit Gaussian')


    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title(title)
    ax.set_xlabel('Normalized correlation values', fontsize=16)
    ax.set_ylabel('% Occurence', fontsize=16)
    ax.legend(loc='upper left', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=13)
    if fig_name != '':
        ax.set_title(' ')
        fig.savefig('/home/boucher/spirou/Figures/fig_TTEST_intransit_{}.pdf'.format(fig_name))
    

def single_t_test(tr, corrRV, correlation, orders, ccf=None, speed_limit=4, wind=0, limit_out=10, \
           plot=True, both_side=True, logl=False, kind=1, equal_var=True, Kp=None, masked=True,
                 vrp=None, p0_estim=None, fig_name='', verbose=True, icorr=None):
    
    if ccf is None:
        ccf = np.ma.sum(correlation[:,orders],axis=1)
        
    if icorr is None:
        icorr = tr.iIn
        inotcorr = tr.iOut
    else:
        inotcorr = np.arange(tr.n_spec)[~icorr]
        
    if logl is True:
        ccf = ccf-np.nanmean(ccf,axis=-1)[:,None]
        
    if masked is True:
        ccf[(ccf == 0).all(axis=-1)] = np.nan
        
    if vrp is None:
        if Kp is None:
            vrp = tr.vrp.value
        else:
            vrp = rv_theo_nu(Kp, tr.nu*u.rad, tr.planet.w, plnt=True).value
    
    if plot is True:
        plt.figure(figsize=(10,5))
        plt.pcolormesh(corrRV, np.arange(tr.n_spec),ccf)
        # plt.plot(t1.vrp[t1.iIn], np.arange(t1.iIn.size),'k')
        plt.plot(tr.berv, np.arange(tr.n_spec),'b')
        plt.plot(vrp+wind-speed_limit, np.arange(tr.n_spec),'k')
        plt.plot(vrp+wind+speed_limit, np.arange(tr.n_spec),'k')
        
        if both_side is True:
            plt.plot(vrp+wind+limit_out, np.arange(tr.n_spec),'r')
            plt.plot(vrp+wind-limit_out, np.arange(tr.n_spec),'r')
        else:
            plt.plot(vrp+wind+limit_out, np.arange(tr.n_spec),'r')
#             plt.plot(tr.vrp.value+wind+limit_out+speed_limit, np.arange(tr.n_spec),'r')
#             plt.plot(tr.vrp.value+wind+limit_out-speed_limit, np.arange(tr.n_spec),'r')
            
        plt.axhline(tr.iIn[0],linestyle='--',color='white')
        plt.axhline(tr.iIn[-1],linestyle='--',color='white')
        plt.colorbar()
        

    in_ccf, out_ccf = get_corr_in_out_trail(tr.iIn, corrRV, ccf, tr, wind=wind, 
                                            speed_limit=speed_limit, limit_out=limit_out, 
                                            both_side=both_side, vrp=vrp, verbose=verbose)

    in_ccf_af, out_ccf_af = get_corr_in_out_trail(tr.iOut, corrRV, ccf, tr, wind=wind, 
                                                  speed_limit=speed_limit, limit_out=limit_out, 
                                            both_side=both_side, vrp=vrp, verbose=verbose)
    if kind == 1:
        A, B = in_ccf/np.nanstd(out_ccf), out_ccf/np.nanstd(out_ccf)
        C, D = in_ccf_af/np.std(out_ccf_af), out_ccf_af/np.std(out_ccf_af)
        title1 = 'In-transit'
        title2 = 'Out-of-transit'
        labelA='In-Trail'
        labelB='Out-of-Trail'
        labelC='In-Trail'
        labelD='Out-of-Trail'
    if kind == 2:
        A, B = in_ccf/np.nanstd(in_ccf_Af), in_ccf_af/np.nanstd(in_ccf_Af)
        C, D = out_ccf/np.nanstd(out_ccf_af), out_ccf_af/np.nanstd(out_ccf_af)
        title1 = 'In-Trail'
        title2 = 'Out-of-Trail'
        labelA='In-Transit'
        labelB='Out-of-Transit'
        labelC='In-Transit'
        labelD='Out-of-Transit'
        
    if plot is True:
        plt.figure()
        
        new_A = np.array(A)[np.isfinite(A)]
        new_B = np.array(B)[np.isfinite(B)]
        new_C = np.array(C)[np.isfinite(C)]
        new_D = np.array(D)[np.isfinite(D)]

        t_test_hist(new_A, new_B, labelA, labelB, title1, p0_estim=p0_estim, fig_name=fig_name)
        t_test_hist(new_C, new_D, labelC, labelD, title2, p0_estim=p0_estim)

    return sp.stats.ttest_ind(A, B, nan_policy='omit', equal_var=equal_var), \
           sp.stats.ttest_ind(C, D, nan_policy='omit', equal_var=equal_var)

def get_t_test_values(index, corrRV, ccf, vrp, \
                      RV=0, speed_limit=4, limit_out=10, both_side=True,
                      equal_var=True):
     
    in_ccf = []
    out_ccf = []
    
    if isinstance(vrp, u.Quantity):
        vrp = vrp.to(u.km/u.s).value

    for i in range(index.size):
        idx_in = np.where((corrRV <= vrp[index[i]]+RV+speed_limit) & \
                          (corrRV >= vrp[index[i]]+RV-speed_limit))
        if i ==0:
            print(idx_in)
            print(corrRV[idx_in])
        in_ccf += list(ccf[i,idx_in].squeeze())

        if both_side is True:
            idx_out = np.where((corrRV > vrp[index[i]]+RV+limit_out) | \
                           (corrRV < vrp[index[i]]+RV-limit_out))
        else:
#             idx_out = np.where((corrRV > tr.vrp[tr.iIn[i]].value+limit_out))
            idx_out = np.where((corrRV <= vrp[index[i]]+RV+limit_out+speed_limit) & \
                          (corrRV >= vrp[index[i]]+RV+limit_out-speed_limit))
            
        out_ccf += list(ccf[i,idx_out].squeeze())
        
    sigma, p_value =  sp.stats.ttest_ind(in_ccf, out_ccf, nan_policy='omit', equal_var=equal_var)
        
    return sigma, p_value

def ttest_map(tr, rv_grid, correlation, ccf=None, orders=np.arange(49), icorr=None, wind=0, RV_array=None,
              kp0=0, kp1=2, RV_limit=20, logl=False, plot=False, masked=False, RV=0, prf=False, Kp_array=None,
              speed_limit=4, limit_out=10, both_side=True, kp_step=1, rv_step=0.5, equal_var=True, verbose=False):
    if icorr is None:  
        icorr= tr.iIn
        
    if ccf is None:
        if logl is True:
            correlation = correlation-np.nanmean(correlation,axis=-1)[:,:,None]
        ccf = np.nansum(correlation[:,orders],axis=1)
    else:
        if logl is True:
            ccf = ccf-np.nanmean(ccf,axis=-1)[:,None]
    
    if masked is True:
        ccf[(ccf == 0).all(axis=-1)] = np.nan
        
    if prf is True:
#         vrp_orb0 = rv_theo_nu(tr.Kp.value, tr.nu * u.rad, tr.planet.w, plnt=True).value
        vrp_orb0 = rv_theo_t(tr.Kp.value, tr.t_start, tr.planet.mid_tr, tr.planet.period, plnt=True).value
        vr_orb0 = -vrp_orb0*(tr.planet.M_pl/tr.planet.M_star).decompose().value
    
    if Kp_array is None:
        Kp_array = np.arange(kp0, int(tr.Kp.value * kp1), kp_step)
    if RV_array is None:
        RV_array = np.arange(-RV_limit, RV_limit+0.5, rv_step)

    t_value = np.ones((Kp_array.size, RV_array.size))
    p_value = np.ones((Kp_array.size, RV_array.size))
    
    id_kp=hm.nearest(Kp_array, tr.Kp.value)
    for i, Kpi in enumerate(Kp_array):
        hm.print_static(i)
#         if i == id_kp:
#             print(Kp_array[i])
#         vrp_orb = rv_theo_nu(Kpi, tr.nu * u.rad, tr.planet.w, plnt=True).value
        vrp_orb = rv_theo_t(Kpi, tr.t_start, tr.planet.mid_tr, tr.planet.period, plnt=True).value
        vr_orb = -vrp_orb*(tr.planet.M_pl/tr.planet.M_star).decompose().value
        if prf is True:
            vrp_orb -= vrp_orb0
            vr_orb -= vr_orb0
        
        for j,rv in enumerate(RV_array):
#             t_value[i,j], p_value[i,j] = get_t_test_values(tr.iIn, rv_grid, ccf, vrp_orb, RV=rv, 
#                                     speed_limit=speed_limit, limit_out=limit_out, both_side=both_side,
#                                                           equal_var=equal_var)
#             print(tr.iIn, rv_grid, vrp_orb-vr_orb + RV)
            in_ccf, out_ccf = get_corr_in_out_trail(icorr, rv_grid, ccf, tr, wind=rv, 
                                            speed_limit=speed_limit, limit_out=limit_out, 
                                            both_side=both_side, vrp=vrp_orb + RV, verbose=verbose)
            A, B = in_ccf/np.nanstd(out_ccf), out_ccf/np.nanstd(out_ccf)
            new_A = np.array(A)[np.isfinite(A)]
            new_B = np.array(B)[np.isfinite(B)]
            t_value[i,j], p_value[i,j] = sp.stats.ttest_ind(A, B, nan_policy='omit', equal_var=equal_var)
        if i == id_kp and plot is True:
            plt.figure()
            plt.plot(RV_array, t_value[i,:])
            plt.xlabel(r'$v_{\rm rad}$ (km s$^{-1}$)', fontsize=16)
            plt.ylabel(r'$t$-test value', fontsize=16)
            print('Value at vrp + RV + wind')
            print(single_t_test(tr, rv_grid, ccf, orders, ccf=ccf, speed_limit=speed_limit, wind=wind, limit_out=limit_out, \
            plot=False, vrp=vrp_orb+RV ))
             
                
    if plot is True:
        print(Kp_array.size)
        if Kp_array.size > 1:
#             plt.figure()
#             plt.plot(RV_array, t_value.squeeze())
#             plt.xlabel(r'$v_{\rm rad}$ (km s$^{-1}$)', fontsize=16)
#             plt.ylabel(r'$t$-test value', fontsize=16)
#         else:
            plot_ttest_map_hist(tr, Kp_array, RV_array, t_value, p_value)
        
    return  Kp_array, RV_array, t_value, p_value, [speed_limit, limit_out, both_side, equal_var]


def ttest_map_2(tr, rv_grid, correlation, ccf=None, orders=np.arange(49), icorr=None,
              kp0=0, kp1=2, RV_limit=20, logl=False, plot=False, masked=False, RV=0, prf=False, Kp_array=None,
              speed_limit=4, limit_out=10, both_side=True, kp_step=1, rv_step=0.5, equal_var=True, verbose=False):

    if ccf is None:
        if logl is True:
            correlation = correlation-np.nanmean(correlation,axis=-1)[:,:,None]
        ccf = np.nansum(correlation[:,orders],axis=1)
    else:
        if logl is True:
            ccf = ccf-np.nanmean(ccf,axis=-1)[:,None]
    
    if masked is True:
        ccf[(ccf == 0).all(axis=-1)] = np.nan
        
    if prf is True:
#         vrp_orb0 = rv_theo_nu(tr.Kp.value, tr.nu * u.rad, tr.planet.w, plnt=True).value
        vrp_orb0 = rv_theo_t(tr.Kp.value, tr.t_start, tr.planet.mid_tr, tr.planet.period, plnt=True).value
        vr_orb0 = -vrp_orb0*(tr.planet.M_pl/tr.planet.M_star).decompose().value
    
    if Kp_array is None:
        Kp_array = np.arange(kp0, int(tr.Kp.value * kp1), kp_step)
    RV_array = np.arange(-RV_limit, RV_limit, rv_step)

    t_value = np.ones((Kp_array.size, RV_array.size))
    p_value = np.ones((Kp_array.size, RV_array.size))
    
    id_kp=hm.nearest(Kp_array, tr.Kp.value)
    for i, Kpi in enumerate(Kp_array):
        hm.print_static(i)
        if i == id_kp:
            print(Kp_array[i])
#         vrp_orb = rv_theo_nu(Kpi, tr.nu * u.rad, tr.planet.w, plnt=True).value
        vrp_orb = rv_theo_t(Kpi, tr.t_start, tr.planet.mid_tr, tr.planet.period, plnt=True).value
        vr_orb = -vrp_orb*(tr.planet.M_pl/tr.planet.M_star).decompose().value
        if prf is True:
            vrp_orb -= vrp_orb0
            vr_orb -= vr_orb0
        
        for j,rv in enumerate(RV_array):
#             t_value[i,j], p_value[i,j] = get_t_test_values(tr.iIn, rv_grid, ccf, vrp_orb, RV=rv, 
#                                     speed_limit=speed_limit, limit_out=limit_out, both_side=both_side,
#                                                           equal_var=equal_var)
#             print(tr.iIn, rv_grid, vrp_orb-vr_orb + RV)
#             in_ccf, out_ccf = get_corr_in_out_trail(tr.iIn, rv_grid, ccf, tr, wind=rv, 
#                                             speed_limit=speed_limit, limit_out=limit_out, 
#                                             both_side=both_side, vrp=vrp_orb + RV)
#             A, B = in_ccf/np.nanstd(out_ccf), out_ccf/np.nanstd(out_ccf)
#             new_A = np.array(A)[np.isfinite(A)]
#             new_B = np.array(B)[np.isfinite(B)]
#             t_value[i,j], p_value[i,j] = sp.stats.ttest_ind(A, B, nan_policy='omit', equal_var=equal_var)
            (t_value[i,j], p_value[i,j]),_ = single_t_test(tr, rv_grid, ccf, orders, ccf=ccf, wind=rv,
                                                           speed_limit=speed_limit, limit_out=limit_out, \
                                                            plot=False, vrp=vrp_orb, Kp=Kpi, verbose=verbose )
             
                
    if plot is True:
        print(Kp_array.size)
        if Kp_array.size == 1:
            plt.plot(RV_array, t_value.squeeze())
        else:
            plot_ttest_map_hist(tr, Kp_array, RV_array, t_value, p_value)
        
    return  Kp_array, RV_array, t_value, p_value, [speed_limit, limit_out, both_side, equal_var]


def ttest_fullmap_2(tr, rv_grid, correlation, ccf=None, orders=np.arange(49), icorr=None,
              kp0=0, kp1=2, RV_limit=20, logl=False, plot=False, masked=False, RV=0, prf=False, Kp_array=None,
              speed_limit=4, limit_out=10, both_side=True, kp_step=1, rv_step=0.5, equal_var=True, verbose=False):

    if ccf is None:
        if logl is True:
            correlation = correlation-np.nanmean(correlation,axis=-1)[:,:,None]
        ccf = np.nansum(correlation[:,orders],axis=1)
    else:
        if logl is True:
            ccf = ccf-np.nanmean(ccf,axis=-1)[:,None]
    
    if masked is True:
        ccf[(ccf == 0).all(axis=-1)] = np.nan
        
    if prf is True:
#         vrp_orb0 = rv_theo_nu(tr.Kp.value, tr.nu * u.rad, tr.planet.w, plnt=True).value
        vrp_orb0 = rv_theo_t(tr.Kp.value, tr.t_start, tr.planet.mid_tr, tr.planet.period, plnt=True).value
        vr_orb0 = -vrp_orb0*(tr.planet.M_pl/tr.planet.M_star).decompose().value

    if Kp_array is None:
        Kp_array = np.arange(kp0, int(tr.Kp.value * kp1), kp_step)
    RV_array = np.arange(-RV_limit, RV_limit, rv_step)

    t_value = np.ones((Kp_array.size, RV_array.size))
    p_value = np.ones((Kp_array.size, RV_array.size))
    
    id_kp=hm.nearest(Kp_array, tr.Kp.value)
    for i, Kpi in enumerate(Kp_array):
        hm.print_static(i)
        if i == id_kp:
            print(Kp_array[i])
#         vrp_orb = rv_theo_nu(Kpi, tr.nu * u.rad, tr.planet.w, plnt=True).value
        vrp_orb = rv_theo_t(Kpi, tr.t_start, tr.planet.mid_tr, tr.planet.period, plnt=True).value
        vr_orb = -vrp_orb*(tr.planet.M_pl/tr.planet.M_star).decompose().value
        if prf is True:
            vrp_orb -= vrp_orb0
            vr_orb -= vr_orb0
        
        for j,rv in enumerate(RV_array):
            (t_value[i,j], p_value[i,j]),_ = single_t_test(tr, rv_grid, ccf, orders, ccf=ccf, wind=rv,
                                                           speed_limit=speed_limit, limit_out=limit_out, \
                                                            plot=False, vrp=vrp_orb, Kp=Kpi, verbose=verbose )
             
                
    if plot is True:
        if Kp_array.size == 1:
            plt.plot(RV_array, t_value)
        else:
            plot_ttest_map_hist(tr, Kp_array, RV_array, t_value, p_value)
        
    return  Kp_array, RV_array, t_value, p_value, [speed_limit, limit_out, both_side, equal_var]


def calc_ttest_snr(ttest_vals, t0val):
    
    tval = ttest_vals.ravel()
    mean = np.average(tval)
    var = np.average((tval - mean)**2)
    std = np.sqrt(var)

    n, bins, patches = plt.hist(x=tval)
    mids = 0.5*(bins[1:] + bins[:-1])
    plt.axvline(t0val, color='k', linestyle=':')
    
    param, pcov = curve_fit(gauss, ydata=n, xdata=mids, p0=np.array([std,n.max(),0]))  # y[n]
    
    plt.plot(mids, gauss(mids, *param))
    return t0val/std


from scipy import special

def pval2sigma(pvalue, tail=2):
    return np.ma.masked_invalid(-special.erfinv(tail*pvalue-1)*np.sqrt(2))


def calc_n_sigma_lvl(values, sigmas, val=None, plot=False, inverse=False):
    
    
    tval = values.ravel()
    mean = np.average(tval)
    var = np.average((tval - mean)**2)
    std = np.sqrt(var)
    
    levels = np.array(sigmas)*std
    
    if plot is True:
        plt.figure()
        n, bins, patches = plt.hist(x=tval)
        mids = 0.5*(bins[1:] + bins[:-1])
        if val is not None:
            plt.axvline(val, color='k', linestyle=':', label='{:.2f} $\sigma$'.format(val/std))
        plt.axvline(mean, color='k', linestyle='-')
        for i, lvl in enumerate(levels):
            plt.axvline(mean+lvl, color=(0,(i)/levels.size,0.5), linestyle='-', 
                        label=r'{} $\sigma$'.format(sigmas[i]))
            plt.legend()
    if inverse is False:
        return np.sort(mean+levels)
    else:
        return np.sort(val-levels)

    

def calc_final_logl(tr, logl, logl_sig, tresh, index, del_ord=[], add_ord=[],
                    N_list=None, nolog_list=None, icorr=None, orders=None):  
    
    
    correlation = logl.copy()
    correlation_sig = logl_sig.copy()
    
    if orders is None:
        orders = list(remove_values_from_array(bands(tr.wv,'yjhk'), del_ord + ord_frac_tresh(tr, tresh)))+add_ord
    
    if N_list is None:
        N_list = [tr.N, tr.N]
    if nolog_list is None:
        nolog_list = [False, True]
        
    if icorr is None:
        icorr = tr.icorr
    print(index)
    logl_grid = corr.sum_logl(correlation, icorr, orders, N_list[0], 
                                   alpha=tr.alpha_frac, axis=0, del_idx=index, nolog=nolog_list[0])
    logl_grid_sig = corr.sum_logl(correlation_sig, icorr, orders, N_list[1], 
                                       alpha=tr.alpha_frac,axis=0, del_idx=index, nolog=nolog_list[1])
    return logl_grid, logl_grid_sig



from matplotlib.gridspec import GridSpec

def mo_fit(xy0, shift, scale):
    x, y0 = xy0
    y = hm.doppler_shift2(x,y0,shift)*scale
    
    return y
    

def plot_mo_fit_params(n_orders, tr, params, **kwargs):
    fig = plt.figure()
    n_spec=tr.n_spec
    # fig.suptitle("Controlling subplot sizes with width_ratios and height_ratios")

    gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey = ax1)
    ax3 = fig.add_subplot(gs[2], sharex = ax1)
    # ax4 = fig.add_subplot(gs[3])

    im = ax1.pcolormesh(np.arange(n_orders), tr.phase, params, **kwargs)
    cbar = fig.colorbar(im, ax=ax2, pad=0.1)
    cbar.set_label('Shift')
    
    ax2.plot(np.nanmean(params, axis=-1), tr.phase,'o-', alpha=0.8, label='Mean')
    ax3.plot(np.arange(n_orders), np.nanmean(params, axis=0),'o-', alpha=0.8)

    ax2.plot(np.nanmedian(params, axis=-1), tr.phase,'o-', alpha=0.8, label='Median')
    ax3.plot(np.arange(n_orders), np.nanmedian(params, axis=0),'o-', alpha=0.8)

    ax1.set_ylabel('Orbital phase')
    ax3.set_xlabel('Order number')
    ax3.set_ylabel('Shift')
    ax2.set_xlabel('Shift')
    ax2.legend()

def master_out_fit(tr, n_orders=49, plot_shift=True, plot_scale=False, 
                   master_out=None, fl_norm_mo=None, sigma=4):
    
    if master_out is None:
        master_out = tr.mast_out
    if fl_norm_mo is None:
        fl_norm_mo = tr.fl_norm_mo
    
    fit_params = []
    fit_err = []
    for n in range(tr.n_spec):
        fit_params_ord = []
        fit_err_ord = []
        for iord in range(n_orders):
            hm.print_static(n, iord, '  ')
            x = tr.wv[iord]
            if master_out.ndim == 2:
                y0 = master_out[iord]
            elif master_out.ndim == 3:
                y0 = master_out[n,iord]
            y = fl_norm_mo[n,iord]


            Y0 = y0[np.logical_not(np.isnan(y0.data))]
            X = x[np.logical_not(np.isnan(y0.data))]
            Y = y[np.logical_not(np.isnan(y0.data))]

            Y0 = Y0[np.logical_not(np.isnan(Y))]
            X = X[np.logical_not(np.isnan(Y))]
            Y = Y[np.logical_not(np.isnan(Y))]
            try:
                popt, pcov = curve_fit(mo_fit, [X, Y0], Y)
            except (RuntimeError, ValueError):
                popt, pcov = [np.nan, np.nan], np.array([[np.nan,np.nan],[np.nan,np.nan]])
            fit_params_ord.append(popt)
            fit_err_ord.append([pcov[0,0], pcov[1,1]])
        fit_params.append(fit_params_ord)
        fit_err.append(fit_err_ord)
    fit_params=np.array(fit_params)
    fit_err=np.array(fit_err)
    
    print(fit_params.shape)
    if plot_shift is True:
        plot_mo_fit_params(n_orders, tr, sigma_clip=(fit_params[:,:,0],sigma))
    if plot_scale is True:
        plot_mo_fit_params(n_orders, tr, sigma_clip=(fit_params[:,:,1],sigma))
    
    return fit_params, fit_err

def build_template_mo(tr, wave_temp, template, dv=None, norm=True):
    
    if dv is None:
        dv = tr.mid_berv+tr.mid_vr.value
        
    if template.ndim == 1:
        fct = interp1d_masked(wave_temp, template, kind='cubic', fill_value='extrapolate')
        if dv.ndim == 0:
            if dv != 0:
                shift = hm.calc_shift(dv)
            else:
                shift = 1.
            master_out = np.ones((tr.nord, tr.npix))*np.nan
            for iOrd in range(tr.nord):
                master_out[iOrd] = fct(tr.wv[iOrd]/shift)
                if norm is True:
                    master_out[iOrd] /= np.nanmedian(master_out[iOrd],axis=-1)
        else:
            shifts = hm.calc_shift(dv)
            master_out = np.ones((tr.n_spec, tr.nord, tr.npix))*np.nan
            for iOrd in range(tr.nord):
                master_out[:,iOrd] = fct(tr.wv[None,iOrd]/shifts[:,None])
            if norm is True:
                master_out /= np.nanmedian(master_out,axis=-1)[:,:,None]
                
    elif template.ndim == 2:
        master_out = np.ones((tr.n_spec, tr.nord, tr.npix))*np.nan
        for n in range(template.shape[0]):
            hm.print_static(n)
            fct = interp1d_masked(wave_temp, template[n], kind='cubic', fill_value='extrapolate')
            if dv.ndim == 0:
                if dv != 0:
                    shift = hm.calc_shift(dv)
                else:
                    shift = 1.
                for iOrd in range(tr.nord):
                    master_out[n,iOrd] = fct(tr.wv[iOrd]/shift)
                    if norm is True:
                        master_out[n,iOrd] /= np.nanmedian(master_out[n,iOrd],axis=-1)
            else:
                shifts = hm.calc_shift(dv)
                for iOrd in range(tr.nord):
                    master_out[n,iOrd] = fct(tr.wv[iOrd]/shifts[n])
#                     print(np.isnan(master_out[n,iOrd]).sum())
                if norm is True:
                    master_out[n,iOrd] /= np.nanmedian(master_out[n,iOrd])
                    
    master_out = np.ma.masked_invalid(master_out)
    
    return master_out


