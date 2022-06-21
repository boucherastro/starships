import numpy as np
from starships import homemade as hm
from starships.spectrum import find_R, quick_inject
from starships import analysis as a
# from spirou_exo import transpec as ts
# from spirou_exo import correlation as corr
import starships.ttest_fcts as nf
from starships.orbite import rv_theo_nu
from starships.mask_tools import interp1d_masked
from mpl_toolkits.axes_grid1 import make_axes_locatable

import scipy.constants as cst
import scipy as sp
from astropy import units as u
from astropy import constants as const

import matplotlib.pyplot as plt
from itertools import islice
from astropy.table import Table, Column



def plot_all_logl(corrRV0, loglbl, var_in, var_out, n_pcas, good_rv_idx=0, switch=False, n_lvl=None, 
                  vmin_in=None, vmax=None, title='', point=None, correl=False, cmap='inferno',
                  cbar_label=r'log $L$'):

    size_in, size_out = np.unique(var_in, return_counts=True)
    var_in_list = size_in[::-1]
    var_out_list = np.unique(var_out)
    
    var_out_list, var_out_nb = np.unique(var_out, return_counts=True)
    var_in_list, var_in_nb = np.unique(var_in, return_counts=True)
    var_in_list = var_in_list[::-1]    
    
    if switch is False:
        range_list = size_in.size
    else:
        range_list = var_out_list.size

    lstyles = ['-','--','-.',':']
    mark = ['o','.','+','x','*',',','d','v','s','^']
#     print(loglbl.shape)
    for idx_pca in range(len(n_pcas)):
        
        loglbl_npc =  loglbl[:,idx_pca]
#         if len(n_pcas) > 1 :
#             loglbl_npc =  loglbl[:,idx_pca]
#         else: 
#             loglbl_npc =  loglbl
#         print(loglbl_npc.shape)
        fig,ax = plt.subplots(1,2, figsize=(15,5)) #
        fig.suptitle(title+' // N PCs = {}'.format(n_pcas[idx_pca]), fontsize=16)

        if vmin_in is None:
            vmin = loglbl_npc.min()
        else:
            vmin = vmin_in
            
        # --- First figure ---
        for v,rv in enumerate(corrRV0):

            valeurs = (loglbl_npc)[v] #- loglbl_npc[0] #np.mean(loglbl_npc[np.array([0,2])],axis=0)
#             print(valeurs.shape)
            for i in range(range_list):
                if v == good_rv_idx:
                    if switch is False:
                        couleur = (0.5, 0.0, i/size_in.size)
                        labels='H2O = '+str(var_in[i*size_out[0]])
                    else:
                        couleur = (0.5, 0.0, i/size_out[0])
                        labels=str(var_out[i])+'K '
                else:
                    couleur = 'grey'
                    labels='_nolegend_'
                if switch is False:
#                     print(size_out[0])
                    ax[0].plot(var_out[i*size_out[0]:(1+i)*size_out[0]], 
                         valeurs[i*size_out[0]:(1+i)*size_out[0]], linestyle = lstyles[i%len(lstyles)], 
                         marker=mark[i%len(mark)], color=couleur,
                         label=labels)
                else:
                    ax[0].plot(np.log10(var_in_list), 
                         valeurs[np.where(var_out == var_out_list[i])], linestyle = lstyles[i%len(lstyles)], 
                         marker=mark[i%len(mark)], color=couleur,
                         label=labels)

        ax[0].legend(loc='upper right', fontsize=9)
        if switch is False:
            ax[0].set_xlabel(r'T$_{\rm eq}$ (K)', fontsize=16)
        else:
            ax[0].set_xlabel(r'log$_{10}$ VMR [H$_2$O]', fontsize=16)
        
#         ax[0].set_title()
        max_val = loglbl_npc.max()
        mean_val = np.nanmean(loglbl_npc)
        
#         ax[0].tight_layout()
        
        # --- Second figure ---
        loglbl_good = loglbl_npc[good_rv_idx]

        im_logl = loglbl_good.reshape(var_in_nb.size, var_out_nb.size).T
        max_val_idx = np.where(im_logl == im_logl.max())

#         plt.figure(figsize=(6,4))
    
        im = ax[1].pcolormesh(np.log10(var_in_list), var_out_list, im_logl, cmap=cmap, #shading='gouraud', 
                              vmax=vmax, vmin=vmin)
        ax[1].axvline(np.log10(var_in_list)[max_val_idx[1]],color='black',linestyle=':', alpha=0.3)
        ax[1].axhline(var_out_list[max_val_idx[0]],color='black',linestyle=':', alpha=0.3)
        ax[1].plot(np.log10(var_in_list)[max_val_idx[1]], var_out_list[max_val_idx[0]],'k+', 
                label='VMR={} // Teq={} // Max = {:.2f}'.format(var_in_list[max_val_idx[1]].data[0],
                                                            var_out_list[max_val_idx[0]].data[0],
                                                            im_logl.max()))
        ax[1].legend(loc='best', fontsize=9)
        ax[1].set_xlabel(r'log$_{10}$ VMR [H$_2$O]', fontsize=16)
        ax[1].set_ylabel(r'T$_{\rm P}$ (K)', fontsize=16)
        cbar = fig.colorbar(im, ax=ax[1], pad=0.01)
        
#         ylabel = r'log $L$'
        if correl is True:
            cbar_label = r'CCF SNR'
            max_val -= 9.7
        ax[0].set_ylim(mean_val-2, max_val+10)  #max_val-25
        ax[0].set_ylabel(cbar_label, fontsize=16)
        cbar.set_label(cbar_label, fontsize=16)
        
        clip_im = np.clip(im_logl, 
                  im_logl[np.where(var_out_list==var_out_list.min()),np.where(var_in_list == var_in_list.min())], None)
        levels = nf.calc_n_sigma_lvl(clip_im, [1,2,3], val=clip_im.max(), plot=False, inverse=True)
        
        print(levels)
        if correl is True:
            levels = list(clip_im.max()-np.array([1,2,3]))
            print(levels)

        ax[1].contour(np.log10(var_in_list), var_out_list, clip_im, levels, 
                        extent=(np.log10(var_in_list[0]),np.log10(var_in_list[-1]),\
                          var_out_list[0],var_out_list[-1]), cmap=cmap+'_r', alpha=0.7)#, 
#                           vmax=vmax, vmin=vmin)
        if point is not None:
            ax[1].plot(*point, 'o', color='dodgerblue')
            
            
def plot_logl_grid_i(corrRV0, loglbl, var_in, var_out, n_pcas, good_rv_idx=0, switch=False, n_lvl=None, 
                  vmin=None, vmax=None, title='', point=None, correl=False, cmap='inferno',
                  cbar_label=r'log $L$', title_it=True, tag_max=True, fig_name='', minmax='min', xlim_remove=0,
                    contours_ccf=[3,2,1]):

    size_in, size_out = np.unique(var_in, return_counts=True)
    var_in_list = size_in[::-1]
    var_out_list = np.unique(var_out)
    
    var_out_list, var_out_nb = np.unique(var_out, return_counts=True)
    var_in_list, var_in_nb = np.unique(var_in, return_counts=True)
    var_in_list = var_in_list[::-1]    
    
    if switch is False:
        range_list = size_in.size
    else:
        range_list = var_out_list.size

    lstyles = ['-','--','-.',':']
    mark = ['o','.','+','x','*',',','d','v','s','^']
#     print(loglbl.shape)
    for idx_pca in range(len(n_pcas)):
        
        loglbl_npc =  loglbl[:,idx_pca]

        fig,ax = plt.subplots(1,1, figsize=(8,6)) #
        if title_it is True:
            fig.suptitle(title+' // N PCs = {}'.format(n_pcas[idx_pca]), fontsize=16)

        if vmin is None:
            vmin = loglbl_npc.min()
#         else:
#             vmin = vmin_in
        
        
        max_val = loglbl_npc.max()
        mean_val = np.nanmean(loglbl_npc)
        
#         ax[0].tight_layout()
        
        # --- Second figure ---
        loglbl_good = loglbl_npc[good_rv_idx]

        im_logl = loglbl_good.reshape(var_in_nb.size, var_out_nb.size).T
        
        if minmax == 'min':
            max_val_idx = np.where(im_logl == im_logl.min())
        elif minmax == 'max':
            max_val_idx = np.where(im_logl == im_logl.max())

#         plt.figure(figsize=(6,4))
        x = np.log10(var_in_list)
        x_ext = np.insert(x,0,x[0]-np.diff(x)[0])
#         x_new = x_ext[1:]+0.5*np.diff(x_ext)
        x_new = np.insert(x_ext[1:]+0.5*np.diff(x_ext), 0, x_ext[0]+0.5*np.diff(x_ext)[0])
        
        y = var_out_list.data
        y_ext = np.append(y, y[-1]+np.diff(y)[-1])
#         y_new = y_ext[:-1]-0.5*np.diff(y_ext)
        y_new = np.append(y_ext[:-1]-0.5*np.diff(y_ext), y_ext[-1]-0.5*np.diff(y_ext)[-1])

        im = ax.pcolormesh(x_new, y_new, im_logl, cmap=cmap, #shading='gouraud', 
                              vmax=vmax, vmin=vmin, rasterized=True)
#         im = ax.imshow(im_logl, origin='lower', aspect='auto', 
#                                 extent=(np.log10(var_in_list).min(), np.log10(var_in_list).max(),
#                                 var_out_list.min(), var_out_list.max()), cmap=cmap)
        
        ax.axvline(np.log10(var_in_list)[max_val_idx[1]],color='black',linestyle=':', alpha=0.3)
        ax.axhline(var_out_list[max_val_idx[0]],color='black',linestyle=':', alpha=0.3)
        ax.plot(np.log10(var_in_list)[max_val_idx[1]], var_out_list[max_val_idx[0]],'k+')
        if tag_max is True:
            ax.axvline(np.log10(var_in_list)[max_val_idx[1]],color='black',linestyle=':', alpha=0.3)
            ax.axhline(var_out_list[max_val_idx[0]],color='black',linestyle=':', alpha=0.3)
            ax.plot(np.log10(var_in_list)[max_val_idx[1]], var_out_list[max_val_idx[0]],'k+', 
                label='VMR={} // Teq={} // Max = {:.2f}'.format(np.log10(var_in_list[max_val_idx[1]].data[0]),
                                                            var_out_list[max_val_idx[0]].data[0],
                                                            im_logl.max()))
            ax.legend(loc='best', fontsize=9)
        ax.set_xlabel(r'log$_{10}$ VMR [H$_2$O]', fontsize=16)
        ax.set_ylabel(r'T$_{\rm P}$ (K)', fontsize=16)
        cbar = fig.colorbar(im, ax=ax, pad=0.01)
        
        if correl is True:
            if cbar_label == r'log $L$':
                cbar_label = r'CCF SNR'
            max_val -= 9.7

        cbar.set_label(cbar_label, fontsize=16)

#         print(levels)
        if correl is True:
            clip_im = im_logl
            levels = list(clip_im.max()-np.array(contours_ccf))
#             print(levels)

        else:

            lnL = im_logl
#             dof=3
#             AIC = 2*dof-2*lnL
#             clip_im = np.exp((AIC.min() - AIC)/2)
#             levels = [1/100,1/10,1/2]
#             delta_BIC = -2*(lnL.max()-lnL)
            clip_im = im_logl
#             levels = [-20,-10, -6, -2]
            levels=[2,6,10]
        
#             clip_im = np.clip(im_logl, 
#                       im_logl[np.where(var_out_list==var_out_list.min()),np.where(var_in_list == var_in_list.min())], None)
#             levels = nf.calc_n_sigma_lvl(clip_im, [1,2,3], val=clip_im.max(), plot=False, inverse=True)

        ax.contour(np.log10(var_in_list), var_out_list, clip_im, levels, 
                        extent=(np.log10(var_in_list[0]),np.log10(var_in_list[-1]),\
                          var_out_list[0],var_out_list[-1]), cmap='tab20b_r', alpha=0.8) #cmap+'_r'  #'Spectral_r'
        if xlim_remove > 0:
            ax.set_xlim(None,x_new.max()-xlim_remove)
        else:
            ax.set_xlim(x_new.min()-xlim_remove,None)
        if point is not None:
            ax.plot(*point, 'o', color='dodgerblue')
        
        fig.savefig('/home/boucher/spirou/Figures/fig_grid_logl_'+fig_name+'.pdf')
            
            
def plot_all_orders_logl(tr, loglbl, var_in, var_out, 
                        cmap='inferno', tresh=0.5, color_range=10):

    size_in, size_out = np.unique(var_in, return_counts=True)
    var_in_list = size_in[::-1]
    var_out_list = np.unique(var_out)
    
    var_out_list, var_out_nb = np.unique(var_out, return_counts=True)
    var_in_list, var_in_nb = np.unique(var_in, return_counts=True)
    var_in_list = var_in_list[::-1]    

    fig, ax = plt.subplots(7,7, figsize=(16,12), sharex=True, sharey=True)
    mean_snr = np.ma.median(tr.SNR, axis=0)
    pix_frac = tr.N_frac
    
    for i in range(7):
        for j in range(7):
            if i*7+j in a.bands(tr.wv, 'y'):
                fg_color = 'goldenrod'
            if i*7+j in a.bands(tr.wv, 'j'):
                fg_color = 'olivedrab'
            if i*7+j in a.bands(tr.wv, 'h'):
                fg_color = 'steelblue'
            if i*7+j in a.bands(tr.wv, 'k'):
                fg_color = 'rebeccapurple'
            if pix_frac[i*7+j] < tresh:
                fg_color = 'firebrick'
            
            im_logl_ord = loglbl[i*7+j].reshape(var_in_nb.size, var_out_nb.size).T
#             if (im_logl_ord == 0).all():
#                 continue
            if (~np.isfinite(im_logl_ord)).all() or (im_logl_ord == 0).all() or (im_logl_ord.mask.all()):
#                 print('{} all masked'.format(i*7+j))
                continue
                
            vmax = im_logl_ord.max()
            vmin = vmax - color_range #im_logl_ord.min()
            
            max_val_idx = np.where(im_logl_ord == im_logl_ord.max())
            
                
            ax[i,j].pcolormesh(np.log10(var_in_list), var_out_list, im_logl_ord, 
                               cmap=cmap, shading='gouraud',vmax=vmax, vmin=vmin)
            ax[i,j].axvline(np.log10(var_in_list)[max_val_idx[1]],color='black',linestyle=':', alpha=0.3)
            ax[i,j].axhline(var_out_list[max_val_idx[0]],color='black',linestyle=':', alpha=0.3)
            ax[i,j].plot(np.log10(var_in_list)[max_val_idx[1]], var_out_list[max_val_idx[0]],'k+', 
                    label='VMR={} // Teq={} // Max = {:.2f}'.format(var_in_list[max_val_idx[1]].data[0],
                                                                var_out_list[max_val_idx[0]].data[0],
                                                                im_logl_ord.max()))    
            
            ax[i,j].set_title('{} - SNR:{:.0f} - {:.2f}'.format(i*7+j, mean_snr[i*7+j], pix_frac[i*7+j]), 
                              color=fg_color)
            clip_im = np.clip(im_logl_ord, im_logl_ord[np.where(var_out_list==var_out_list.min()),\
                                               np.where(var_in_list == var_in_list.min())], None)
            
            levels = nf.calc_n_sigma_lvl(clip_im, [1,2,3], val=clip_im.max(), plot=False, inverse=True)

            ax[i,j].contour(np.log10(var_in_list), var_out_list, clip_im, levels, 
                        extent=(np.log10(var_in_list[0]),np.log10(var_in_list[-1]),\
                          var_out_list[0],var_out_list[-1]), cmap=cmap+'_r', alpha=0.7, 
                          vmax=vmax, vmin=vmin)

            
from scipy.interpolate import interp2d
def plot_logl_and_corrmap(tr, var_out, var_in, corrRV0, logl_grid, corrRV, corr_map, 
                          good_rv_idx=0, icorr=None, cmap=None, orders=np.arange(49)):
    
    RV_sys = tr.planet.RV_sys.value
    Kp = tr.Kp.to(u.km/u.s).value
    
    var_out_list, var_out_nb = np.unique(var_out, return_counts=True)
    var_in_list, var_in_nb = np.unique(var_in, return_counts=True)
    var_in_list = var_in_list[::-1]
    
    snr_value = np.empty(var_out_list.shape)
    for i,var_out_i in enumerate(var_out_list):

        fig, (ax_log, ax_map) = plt.subplots(1,2, figsize=(12,3))
        idx_out = np.where(var_out == var_out_i)[0]
        logl_out_i = logl_grid[:,idx_out]

        for v in range(corrRV0.size):

            if v == good_rv_idx: 
                kwargs = {}
                kwargs['color'] = 'dodgerblue'
    #             kwargs['label'] = r'RV$_{\rmShift}$'+' = {}km/s'.format(rv)+ ', Injected RV position'
                kwargs['linestyle'] = '--'
                kwargs['marker'] = 'o'

            elif v == good_rv_idx+1:
                kwargs = {}
                kwargs['color'] = 'grey'
    #             kwargs['label'] = r'RV$_{\rmShift}$'+' = $\pm$[20,40,60,80,100]km/s'
                kwargs['linestyle'] = '-'
                kwargs['alpha'] = 0.5
                kwargs['marker'] = '.'
            else:
                kwargs = {}
                kwargs['color'] = 'grey'
                kwargs['label'] = '_nolegend_'
                kwargs['linestyle'] = '-'
                kwargs['alpha'] = 0.5
                kwargs['marker'] = '.'
            ax_log.plot(np.log10(var_in_list), logl_out_i[v], 'o--', **kwargs)

            if v == good_rv_idx:
                idx_in_max = np.argmax(logl_out_i[v])
                ax_log.plot(np.log10(var_in_list[idx_in_max]), logl_out_i[v][idx_in_max],'kx')
        ax_log.set_ylabel('log $L$', fontsize=14)
        ax_log.set_xlabel('VMRs [H2O]', fontsize=14)
        ax_log.set_title(r'T$_p$={}K'.format(int(var_out_i)))

        corr_map_max_i = corr_map[idx_out[idx_in_max]]
        ccf = np.nansum(corr_map_max_i[:,orders], axis=1)
#         if icorr is None:
#             l_curve = tr.light_curve
#             nunu = tr.nu
#             icorr_shape = ccf.shape[0]
#             berv_val = tr.berv
#         else:
#             if icorr.shape[0] < ccf.shape[0]:
#                 ccf = ccf[icorr]
#                 spec_num = icorr 
#             l_curve = tr.light_curve[icorr]
#             vrp = tr.vrp[icorr]
#             nunu = tr.nu[icorr]
#             icorr_shape = icorr.shape[0]
#             berv_val = tr.berv[icorr] 
        ccf, nunu, berv_val, vrp, alpha, icorr_shape = a.select_in_transit(icorr, ccf, tr.nu, tr.berv, 
                                                                         tr.vrp, tr.alpha_frac)

        interp_grid, Kp_array, sum_ccf, snr2, idx_bruit, idx_bruit2, courbe, snr = a.calc_snr_2d(ccf, 
                                                            corrRV, vrp, tr.Kp, nunu, tr.planet.w, 
                                                            limit_shift=100, interp_size=201,  
                                                                  RV_sys=tr.planet.RV_sys, kp0=0)

        snr_nonoise = np.ma.masked_invalid(snr2[~idx_bruit2][:,~idx_bruit])
        idx_max2 = np.where(snr_nonoise == snr_nonoise.max())
        idx_min2 = np.where(snr_nonoise == snr_nonoise.min()) 
        snr_fct = interp2d(interp_grid, Kp_array, snr2)

        fct_min = sp.optimize.fminbound(interp1d_masked(interp_grid, -courbe, kind="cubic"), 
                            interp_grid[~idx_bruit][int(np.clip(idx_max2[1]-6,0,None))],
                            interp_grid[~idx_bruit][int(np.clip(idx_max2[1]+6,None,
                                                                courbe[~idx_bruit].size-1))],
                                  full_output=1)
        rv_max = fct_min[0]
        snr_value[i] = snr_fct(rv_max, Kp)[0]


        maximum = True
        if maximum is True:
            idx_minmax = idx_max2
        else:
            idx_minmax = idx_min2
    #     hm.stop()
        im_corr = ax_map.imshow(snr2, origin='lower', aspect='auto', 
                                 extent=(interp_grid.min(), interp_grid.max(),
                                Kp_array.min(), Kp_array.max()), cmap=cmap)
        ax_map.set_ylabel('$K_p$ (km s$^{-1}$)', fontsize=14)
        ax_map.set_xlabel('$v_{\rm offset}$ (km s$^{-1}$)', fontsize=14)
        ax_map.axhline(Kp, linestyle=':', alpha=0.7, color='white') 
        ax_map.axvline(0, linestyle=':', alpha=0.7, color='white')
        ax_map.plot(0, Kp, 'k+', label=r'{:.2f} $\sigma$'.format(snr_fct(RV_sys, Kp)[0]))
        ax_map.plot(interp_grid[~idx_bruit][idx_max2[1]],
                    Kp_array[~idx_bruit2][idx_max2[0]],'bx', 
                    label=r'{:.2f} $\sigma$'.format(snr_nonoise.max()))
        ax_map.set_title('RV = {:.3f} // Kp = {:.2f}'.format(interp_grid[~idx_bruit][idx_max2[1]][0], 
                                                             Kp_array[~idx_bruit2][idx_max2[0]][0]), 
                         color='blue')

        ax_map.plot(rv_max, Kp, '+', color='dodgerblue', 
                        label=r'{:.2f} $\sigma$ = {:.2f} km s$^{-1}$ wind'.format(snr_value[i], (rv_max)))

        ax_map.legend(loc='best')
        divider = make_axes_locatable(ax_map)
        cax = divider.append_axes('right', size='3%', pad=0.05)
        cbar = fig.colorbar(im_corr,ax=ax_map, cax=cax)
        cbar.set_label('Correlation SNR', fontsize=14)
        
        fig.tight_layout()
    #     hm.stop()
    return snr_value


def plot_inverse(fractions, interp_grid, snrs_corr, snrs_logl, 
                 loglbl_tr, loglbl_tr_sig, loglbl_tr_frac, loglbl_tr_frac_sig):
    plt.figure(figsize=(10,5))
    for i in range(fractions.size):
        plt.plot(interp_grid, snrs_corr[i], color=(0,0.5,i/fractions.size),alpha=0.5, 
                 label = '{:.1f}'.format(fractions[i], loglbl_tr[i]))
    plt.legend(loc='lower left', fontsize=10)
    plt.ylabel('Correlation SNR')
    plt.xlabel('$v_{\rm offset}$')
    plt.title('Fraction of the inverse signal injected')
    plt.axhline(0,color='k',alpha=0.2, linestyle='--')

    plt.figure(figsize=(10,5))
    for i in range(fractions.size):
        plt.plot(interp_grid, snrs_logl[i], color=(0.5,0,i/fractions.size),alpha=0.5, 
                 label = '{:.1f}'.format(fractions[i], loglbl_tr[i]))
    plt.legend(loc='lower left', fontsize=10)
    plt.ylabel('log L SNR')
    plt.xlabel('$v_{\rm offset}$')
    plt.title('Fraction of the inverse signal injected')
    plt.axhline(0,color='k',alpha=0.2, linestyle='--')

    fig,ax = plt.subplots(1,2, figsize=(10,3))
    ax[0].plot(fractions, loglbl_tr,'o--')
    ax[0].set_title(r'without $\sigma$ division')
    ax[1].plot(fractions, loglbl_tr_sig,'o--')
    ax[1].set_title(r'with $\sigma$ division')

    fig,ax = plt.subplots(1,2, figsize=(10,3))
    ax[0].plot(fractions, loglbl_tr_frac,'o--')
    ax[0].set_title(r'without $\sigma$ division')
    ax[1].plot(fractions, loglbl_tr_frac_sig,'o--')
    ax[1].set_title(r'with $\sigma$ division')
    
    
def plot_all_orders_correl(corrRV, ccf, tr, icorr=None, logl=False, tresh=0.4, sharey=True,
                           vrp=None, RV_sys=None, vmin=None,vmax=None, vline=None, hline=None, kind='snr', 
                           return_snr=False):
    if icorr is None:
        icorr = tr.icorr
        
        
    if vrp is None:
        vrp = (tr.vrp-tr.vr).value
    if RV_sys is None:
        RV_sys = tr.planet.RV_sys
        
    fig, ax = plt.subplots(7,7, figsize=(16,12), sharex=True, sharey=sharey)
#     fig_shift, ax_shift = plt.subplots(7,7, figsize=(16,12), sharex=True, sharey=True)
    fig_single, ax_single = plt.subplots(7,7, figsize=(16,12), sharex=True, sharey=True)
    mean_snr = np.ma.median(tr.SNR, axis=0)
    pix_frac = tr.N_frac
    
    snr_list = []
    for i in range(7):
        for j in range(7):
            if i*7+j in a.bands(tr.wv, 'y'):
                fg_color = 'goldenrod'
            if i*7+j in a.bands(tr.wv, 'j'):
                fg_color = 'olivedrab'
            if i*7+j in a.bands(tr.wv, 'h'):
                fg_color = 'steelblue'
            if i*7+j in a.bands(tr.wv, 'k'):
                fg_color = 'rebeccapurple'
            if pix_frac[i*7+j] < tresh:
                fg_color = 'firebrick'
            
            if logl is True:
                ax[i,j].pcolormesh(corrRV, np.arange(ccf.shape[0]), \
                                   ccf[:,i*7+j]-np.nanmean(ccf[:,i*7+j], axis=-1)[:,None], vmin=vmin, vmax=vmax)
            else:
                ax[i,j].pcolormesh(corrRV, np.arange(ccf.shape[0]), ccf[:,i*7+j], vmin=vmin, vmax=vmax)
            ax[i,j].plot(vrp, np.arange(ccf.shape[0]), 'k:', alpha=0.5)

            
#             ax[i,j].plot(tr.berv, np.arange(ccf.shape[0]), 'r--', alpha=0.5)
            ax[i,j].set_title('{} - SNR:{:.0f} - {:.2f}'.format(i*7+j, mean_snr[i*7+j], pix_frac[i*7+j]), 
                              color=fg_color)
            
            shifted_corr, interp_grid, courbe, snr, _ = a.calc_snr_1d(ccf[icorr,i*7+j], corrRV, \
                                                            vrp[icorr], RV_sys=RV_sys)
            snr_list.append(snr)
            if kind == 'courbe':
                ax_single[i,j].plot(interp_grid, courbe)
            else:
                ax_single[i,j].plot(interp_grid, snr)
                ax_single[i,j].set_ylim(-4,4)
            ax_single[i,j].set_title('{} - SNR:{:.0f} - {:.2f}'.format(i*7+j, mean_snr[i*7+j], pix_frac[i*7+j]), 
                                     color=fg_color)
            ax_single[i,j].axvline(0, linestyle=':', alpha=0.5)
            if vline is not None:
                ax_single[i,j].axvline(vline, linestyle='-', alpha=0.5, color='navy')
            if hline is not None:
                ax_single[i,j].axhline(hline, linestyle='-', alpha=0.5, color='navy')
            ax_single[i,j].axvline(np.mean(tr.berv), linestyle='--', color='red', alpha=0.5)
            
    if return_snr is True:
        return interp_grid, snr_list
            
            
def plot_all_orders_spectra(tr, flux=None):
    
    fig, ax = plt.subplots(tr.nord,1, figsize=(15,72))
    mean_snr = np.nanmedian(tr.SNR, axis=0)
    
    pix_frac = tr.N_frac

    if flux is None:
        flux = tr.final

    for i in range(tr.nord):
        if i in a.bands(tr.wv, 'y'):
            fg_color = 'goldenrod'
        if i in a.bands(tr.wv, 'j'):
            fg_color = 'olivedrab'
        if i in a.bands(tr.wv, 'h'):
            fg_color = 'steelblue'
        if i in a.bands(tr.wv, 'k'):
            fg_color = 'rebeccapurple'

        ax[i].pcolormesh(tr.wv[i], np.arange(tr.n_spec), flux[:,i])

        ax[i].set_title('Ord:{} / SNR:{:.1f} / % good pixel:{:.1f} '.format(i, mean_snr[i],pix_frac[i]*100), 
                        color=fg_color)

import matplotlib as mpl        
        
def plot_steps(tr, iord, xlim=None, masking_limit=None, id_spec=0, fig_name='', 
               cmap=None, bad_color='red'):
    
    if cmap is None:
        cmap = mpl.cm.Greys_r
    cmap.set_bad(bad_color,1.)


    fig,(ax00,ax0,ax01,ax1,ax2,ax3,ax4) = plt.subplots(7,1, sharex=True, 
                                         gridspec_kw = {'height_ratios':[1.,1.,1.,1.,1.,1.,1.]},
                                        figsize=(13,16))
    uncorr_fl = tr.uncorr/(tr.blaze/np.nanmax(tr.blaze, axis=-1)[:,:,None])
    ax00.plot(tr.wv[iord], (uncorr_fl[id_spec,iord]/uncorr_fl[id_spec,iord].mean(axis=-1)[None]).T  + 0.4,
              'k', alpha=1, label='Uncorrected Flux')
    ax00.plot(tr.wv[iord], tr.tellu[id_spec,iord],'b', label='Telluric transmission', alpha=0.8)
    
    ax00.plot(tr.wv[iord],#*(1+(tr.berv[0]*u.km/u.s)/const.c), 
              (tr.flux[id_spec,iord]/tr.flux[id_spec,iord].mean(axis=-1)[None]).T + 0.4 + 0.4, 
              'g', alpha=1, label='Corrected Flux')
    
#     fl_norm= tr.flux/np.ma.median(tr.flux,axis=-1)[:,:,None]
#     ax00.plot(tr.wv[iord],#*(1+(tr.berv[0]*u.km/u.s)/const.c), 
#               (np.std(fl_norm[:,iord], axis=0)/np.std(fl_norm[:,iord], axis=0).mean())*0.4, 
#               'r', alpha=0.5, label='Scaled Mean Noise')

    if masking_limit is not None:
        ax00.axhline(masking_limit, color='blue', alpha=0.4, label='Tellu. Masking limit', linestyle=':')
#     ax00.plot(tr.wv[iord], tr.mast_out[iord],'r', label='Master Out')
    ax00.legend(loc='lower left')
#     ax00.set_title('A) Mean SNR = {:.2f}'.format(45, tr.SNR[:,iord].mean()))
    ax00.set_ylabel('Normalized\nFlux',fontsize=12)
    ax00.set_ylim(0.,2.1)
    
    divider00 = make_axes_locatable(ax00)
    cax00 = divider00.append_axes('right', size='3%', pad=0.05)
    cax00.axis('off')
    
# ax = plt.subplot(121)
# img = ax.imshow([np.arange(0,1,.1)],aspect="auto")
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("bottom", size="3%", pad=0.5)
# plt.colorbar(img, cax=cax, orientation='horizontal')

# ax2 = plt.subplot(122)
# ax2.plot(range(2))
# divider2 = make_axes_locatable(ax2)
# cax2 = divider2.append_axes("bottom", size="3%", pad=0.5)
# cax2.axis('off')
# plt.show()
    
#     if sub_spec is not None:
#         id_spec_part = np.where((tr.wv[iord] >= sub_spec[0]) & (tr.wv[iord] <= sub_spec[-1]))
#     else:
#         id_spec_part = np.where((tr.wv[iord] >= tr.wv[iord][0]) & (tr.wv[iord] <= tr.wv[iord][-1]))
    
#     ax00.plot(tr.wv[iord][id_spec_part], (tr.uncorr[0,iord]/tr.uncorr[0,iord].mean(axis=-1)[None])[id_spec_part].T,
#               'k', alpha=0.5, label='Uncorr. Flux Sample')
#     ax00.plot(tr.wv[iord][id_spec_part], (tr.uncorr[-1,iord]/tr.uncorr[-1,iord].mean(axis=-1)[None])[id_spec_part].T,
#               'k', alpha=0.5, label='')
    
#     ax00.plot(tr.wv[iord][id_spec_part],
#               (tr.flux[0,iord]/tr.flux[0,iord].mean(axis=-1)[None])[id_spec_part].T+0.3,
#               'g', alpha=0.5, label='Corr. Flux Sample')
#     ax00.plot(tr.wv[iord][id_spec_part],
#               (tr.flux[-1,iord]/tr.flux[-1,iord].mean(axis=-1)[None])[id_spec_part].T+0.3,
#               'g', alpha=0.5, label='')
    
#     ax00.plot(tr.wv[iord][id_spec_part], tr.tellu[0,iord][id_spec_part]-0.25,'b', label='Telluric transm.', alpha=0.5)
    

    if iord == 8:
        he_lines = [1.083206, 1.083322, 1.083331]
        for hel in he_lines:
            ax00.axvline(hel*(1+(tr.planet.RV_sys)/const.c), color='orange')
        ax00.axvline(1.083, color='green')

    im0 = ax0.pcolormesh(tr.wv[iord], tr.phase, \
                         tr.uncorr[:,iord]/(tr.blaze[:,iord]/tr.blaze[:,iord].max(axis=-1)[:,None]), 
                         cmap="Greys_r", rasterized=True) 
    ax0.axhline(tr.phase[tr.iIn[-1]], alpha=0.2, color='blue')
    ax0.axhline(tr.phase[tr.iIn[0]], alpha=0.2, color='blue')
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes('right', size='3%', pad=0.05)
    cbar00 = fig.colorbar(im0,ax=ax0, cax=cax)
#     ax0.set_title('B) Uncorrected Flux (Earth Rest Frame)')
    ax0.text(tr.wv[iord][40], tr.phase[-6], 'B) Uncorrected Flux (Earth Rest Frame)',
             fontsize = 12, bbox ={'facecolor':'white', 'alpha':0.8})
#     cbar00.set_label('Flux', fontsize=14)
    
    im01 = ax01.pcolormesh(tr.wv[iord], tr.phase, tr.flux[:,iord], cmap=cmap, rasterized=True) 
    ax01.axhline(tr.phase[tr.iIn[-1]], alpha=0.2, color='blue')
    ax01.axhline(tr.phase[tr.iIn[0]], alpha=0.2, color='blue')
    divider = make_axes_locatable(ax01)
    cax = divider.append_axes('right', size='3%', pad=0.05)
    cbar01 = fig.colorbar(im01,ax=ax01, cax=cax)
#     ax01.set_title('C) Telluric-Corrected Flux')
    ax01.text(tr.wv[iord][40], tr.phase[-6], 'C) Telluric-Corrected Flux',
             fontsize = 12, bbox ={'facecolor':'white', 'alpha':0.8})
    cbar01.set_label('Flux',y=1.2, fontsize=14)

    im1 = ax1.pcolormesh(tr.wv[iord], tr.phase, tr.fl_norm[:,iord], cmap=cmap, rasterized=True)#, vmin=0.85, vmax=1.08)
                         # vmax=1.60) 
    #, vmin=0.85, vmax=1.08
    ax1.axhline(tr.phase[tr.iIn[-1]], alpha=0.2, color='blue')
    ax1.axhline(tr.phase[tr.iIn[0]], alpha=0.2, color='blue')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='3%', pad=0.05)
    cbar0 = fig.colorbar(im1,ax=ax1, cax=cax)
#     ax1.set_title('D) Masked and Normalized Flux (Shifted to Star Rest Frame)')
    ax1.text(tr.wv[iord][40], tr.phase[-6], 'D) Normalized Flux (Shifted to Pseudo SRF)',
             fontsize = 12, bbox ={'facecolor':'white', 'alpha':0.8})

    im2 = ax2.pcolormesh(tr.wv[iord], tr.phase, tr.fl_norm_mo[:,iord], cmap=cmap, rasterized=True)#, vmin=0.85, vmax=1.08)
                         # vmax=1.60) 
    #, vmin=0.85, vmax=1.08
    ax2.axhline(tr.phase[tr.iIn[-1]], alpha=0.2, color='blue')
    ax2.axhline(tr.phase[tr.iIn[0]], alpha=0.2, color='blue')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='3%', pad=0.05)
    fig.colorbar(im2,ax=ax2, cax=cax)
#     ax2.set_title('E) Normalized to the Continuum Flux')
    ax2.text(tr.wv[iord][40], tr.phase[-6],'E) Normalized to the Continuum Flux',
             fontsize = 12, bbox ={'facecolor':'white', 'alpha':0.8})

    im3 = ax3.pcolormesh(tr.wv[iord], tr.phase, tr.spec_trans[:,iord], cmap=cmap, rasterized=True, vmin=0.955, vmax=1.035)
    ax3.axhline(tr.phase[tr.iIn[-1]], alpha=0.2, color='blue')
    ax3.axhline(tr.phase[tr.iIn[0]], alpha=0.2, color='blue')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='3%', pad=0.05)
    fig.colorbar(im3,ax=ax3, cax=cax)
#     ax3.set_title('F) Transmission Spectrum')
    ax3.text(tr.wv[iord][40], tr.phase[-6], 'F) Transmission Spectrum',
             fontsize = 12, bbox ={'facecolor':'white', 'alpha':0.8})

    im4 = ax4.pcolormesh(tr.wv[iord], tr.phase, tr.final[:,iord], cmap=cmap, rasterized=True, vmin=0.955-1, vmax=1.035-1)  
    ax4.axhline(tr.phase[tr.iIn[-1]], alpha=0.2, color='blue')
    ax4.axhline(tr.phase[tr.iIn[0]], alpha=0.2, color='blue')
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='3%', pad=0.05)
    cbar1 = fig.colorbar(im4,ax=ax4, cax=cax)
#     ax4.set_title('G) PCA-Corrected Transmission Spectrum ({} PC)'.format(tr.params[5]))
    ax4.text(tr.wv[iord][40], tr.phase[-6], 'G) PCA-Corrected Transmission Spectrum ({} PCs)'.format(tr.params[5]),
             fontsize = 12, bbox ={'facecolor':'white', 'alpha':0.8})

#     ax4.plot(tr.wv[iord][2044] * (1+tr.vrp/const.c), tr.phase)

    ax4.set_xlabel(r'Wavelength ($\mu$m)', fontsize=14)
    ax4.set_ylabel(r'Orbital phase ($\phi$)', fontsize=14)
    ax4.yaxis.set_label_coords(-0.07, 3)
    cbar1.set_label('Normalized flux', y=2.4, fontsize=14)
    if xlim is not None:
        ax4.set_xlim(*xlim)
        
    fig.subplots_adjust(hspace=0)

    fig.savefig('/home/boucher/spirou/Figures/fig_STEPS'+fig_name+'.pdf')

    
def plot_five_steps(tr, iord, xlim=None, masking_limit=0.8, fig_name='',
                     cmap=None, bad_color='red', length=16, id_spec=10):
    
    if cmap is None:
        cmap = mpl.cm.Greys_r
    cmap.set_bad(bad_color,1.)
    
    
    fig,(ax00, ax01,ax2,ax3,ax4) = plt.subplots(5,1, sharex=True, 
                                         gridspec_kw = {'height_ratios':[1.,1.,1.,1.,1.]},
                                        figsize=(length,8))
    
    uncorr_fl = tr.uncorr/(tr.blaze/np.nanmax(tr.blaze, axis=-1)[:,:,None])
    ax00.plot(tr.wv[iord], (uncorr_fl[id_spec,iord]/uncorr_fl[id_spec,iord].mean(axis=-1)[None]).T  + 0.45,
              'k', alpha=1, label='Uncorrected Flux')
    ax00.plot(tr.wv[iord], tr.tellu[id_spec,iord],'b', label='Telluric transmission', alpha=0.8)
    
    ax00.plot(tr.wv[iord],#*(1+(tr.berv[0]*u.km/u.s)/const.c), 
              (tr.flux[id_spec,iord]/tr.flux[id_spec,iord].mean(axis=-1)[None]).T + 0.45 + 0.45, 
              'g', alpha=1, label='Corrected Flux')
    if masking_limit is not None:
        ax00.axhline(masking_limit, color='blue', alpha=0.4, label='Tellu. Masking limit', linestyle=':')
#     ax00.plot(tr.wv[iord], tr.mast_out[iord],'r', label='Master Out')
    ax00.legend(loc='center', ncol=4, bbox_to_anchor =(0.5, 1.2))
#     ax00.set_title('A) Mean SNR = {:.2f}'.format(45, tr.SNR[:,iord].mean()))
    ax00.set_ylabel('Normalized\nFlux',fontsize=12)
    ax00.set_ylim(0.,2.5)
    
    divider00 = make_axes_locatable(ax00)
    cax00 = divider00.append_axes('right', size='3%', pad=0.05)
    cax00.axis('off')
    
    im01 = ax01.pcolormesh(tr.wv[iord], tr.phase, tr.flux[:,iord], cmap=cmap, rasterized=True) 
    if len(tr.iIn) > 0:
        ax01.axhline(tr.phase[tr.iIn[-1]], alpha=0.2, color='blue')
        ax01.axhline(tr.phase[tr.iIn[0]], alpha=0.2, color='blue')
    divider = make_axes_locatable(ax01)
    cax = divider.append_axes('right', size='3%', pad=0.05)
    cbar01 = fig.colorbar(im01,ax=ax01, cax=cax)
#     ax01.set_title('C) Tellu-Corrected Flux')
    cbar01.set_label('Flux', fontsize=14) #,y=1.2

    im2 = ax2.pcolormesh(tr.wv[iord], tr.phase, tr.fl_norm_mo[:,iord], cmap=cmap, rasterized=True)#, vmin=0.85, vmax=1.08)
                         # vmax=1.60) 
    #, vmin=0.85, vmax=1.08
    if len(tr.iIn) > 0:
        ax2.axhline(tr.phase[tr.iIn[-1]], alpha=0.2, color='blue')
        ax2.axhline(tr.phase[tr.iIn[0]], alpha=0.2, color='blue')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='3%', pad=0.05)
    fig.colorbar(im2,ax=ax2, cax=cax)
#     ax2.set_title('E) Normalized to the Continuum Flux')

#     im5 = ax5.pcolormesh(tr.wv[iord], tr.phase, tr.fl_norm_mo[:,iord], cmap=cmap, rasterized=True)#, vmin=0.85, vmax=1.08)
#                          # vmax=1.60) 
#     #, vmin=0.85, vmax=1.08
#     if len(tr.iIn) > 0:
#         ax5.axhline(tr.phase[tr.iIn[-1]], alpha=0.2, color='blue')
#         ax5.axhline(tr.phase[tr.iIn[0]], alpha=0.2, color='blue')
#     divider = make_axes_locatable(ax5)
#     cax = divider.append_axes('right', size='3%', pad=0.05)
#     fig.colorbar(im5,ax=ax5, cax=cax)
# #     ax2.set_title('E) Normalized to the Continuum Flux')


    im3 = ax3.pcolormesh(tr.wv[iord], tr.phase, tr.spec_trans[:,iord], cmap=cmap, rasterized=True)#, vmin=0.955, vmax=1.035)
    if len(tr.iIn) > 0:
        ax3.axhline(tr.phase[tr.iIn[-1]], alpha=0.2, color='blue')
        ax3.axhline(tr.phase[tr.iIn[0]], alpha=0.2, color='blue')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='3%', pad=0.05)
    fig.colorbar(im3,ax=ax3, cax=cax)
#     ax3.set_title('F) Transmission Spectrum')

    im4 = ax4.pcolormesh(tr.wv[iord], tr.phase, tr.final[:,iord], cmap=cmap, rasterized=True)#, vmin=0.955-1, vmax=1.035-1)  
    if len(tr.iIn) > 0:
        ax4.axhline(tr.phase[tr.iIn[-1]], alpha=0.2, color='blue')
        ax4.axhline(tr.phase[tr.iIn[0]], alpha=0.2, color='blue')
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='3%', pad=0.05)
    cbar1 = fig.colorbar(im4,ax=ax4, cax=cax)
#     ax4.set_title('G) PCA-Corrected Transmission Spectrum ({} PCs)'.format(tr.params[5]))

#     ax4.plot(tr.wv[iord][2044] * (1+tr.vrp/const.c), tr.phase)

    ax4.set_xlabel(r'Wavelength ($\mu$m)', fontsize=14)
    ax4.set_ylabel(r'Orbital phase ($\phi$)', fontsize=14)
    ax4.yaxis.set_label_coords(-0.05, 2)
    cbar1.set_label('Normalized flux', y=1.5, fontsize=14)
    if xlim is not None:
        ax4.set_xlim(*xlim)
        
    fig.subplots_adjust(hspace=0)

    ax01.text( tr.wv[iord][100],tr.phase[-15], 'A', fontsize = 12, bbox ={'facecolor':'white', 'alpha':0.8})
    ax2.text( tr.wv[iord][100],tr.phase[-15], 'B', fontsize = 12, bbox ={'facecolor':'white', 'alpha':0.8})
    ax3.text( tr.wv[iord][100],tr.phase[-15], 'C', fontsize = 12, bbox ={'facecolor':'white', 'alpha':0.8})
    ax4.text( tr.wv[iord][100],tr.phase[-15], 'D', fontsize = 12, bbox ={'facecolor':'white', 'alpha':0.8})

    fig.savefig('/home/boucher/spirou/Figures/fig_five_STEPS'+fig_name+'.pdf')

    return fig
    # cbar = plt.colorbar(im3, orientation="horizontal")
    
    
    
def plot_small_steps(tr, iord, xlim=None, masking_limit=0.8, fig_name='',
                     cmap=None, bad_color='red', length=14):
    
    if cmap is None:
        cmap = mpl.cm.Greys_r
    cmap.set_bad(bad_color,1.)
    
    
    fig,(ax01,ax2,ax3,ax4) = plt.subplots(4,1, sharex=True, 
                                         gridspec_kw = {'height_ratios':[1.,1.,1.,1.]},
                                        figsize=(length,7))
    
    im01 = ax01.pcolormesh(tr.wv[iord], tr.phase, tr.flux[:,iord], cmap=cmap, rasterized=True) 
    if len(tr.iIn) > 0:
        ax01.axhline(tr.phase[tr.iIn[-1]], alpha=0.2, color='blue')
        ax01.axhline(tr.phase[tr.iIn[0]], alpha=0.2, color='blue')
    divider = make_axes_locatable(ax01)
    cax = divider.append_axes('right', size='3%', pad=0.05)
    cbar01 = fig.colorbar(im01,ax=ax01, cax=cax)
#     ax01.set_title('C) Tellu-Corrected Flux')
    cbar01.set_label('Flux', fontsize=14) #,y=1.2

    im2 = ax2.pcolormesh(tr.wv[iord], tr.phase, tr.fl_norm_mo[:,iord], cmap=cmap, rasterized=True)#, vmin=0.85, vmax=1.08)
                         # vmax=1.60) 
    #, vmin=0.85, vmax=1.08
    if len(tr.iIn) > 0:
        ax2.axhline(tr.phase[tr.iIn[-1]], alpha=0.2, color='blue')
        ax2.axhline(tr.phase[tr.iIn[0]], alpha=0.2, color='blue')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='3%', pad=0.05)
    fig.colorbar(im2,ax=ax2, cax=cax)
#     ax2.set_title('E) Normalized to the Continuum Flux')

    im3 = ax3.pcolormesh(tr.wv[iord], tr.phase, tr.spec_trans[:,iord], cmap=cmap, rasterized=True)#, vmin=0.955, vmax=1.035)
    if len(tr.iIn) > 0:
        ax3.axhline(tr.phase[tr.iIn[-1]], alpha=0.2, color='blue')
        ax3.axhline(tr.phase[tr.iIn[0]], alpha=0.2, color='blue')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='3%', pad=0.05)
    fig.colorbar(im3,ax=ax3, cax=cax)
#     ax3.set_title('F) Transmission Spectrum')

    im4 = ax4.pcolormesh(tr.wv[iord], tr.phase, tr.final[:,iord], cmap=cmap, rasterized=True)#, vmin=0.955-1, vmax=1.035-1)  
    if len(tr.iIn) > 0:
        ax4.axhline(tr.phase[tr.iIn[-1]], alpha=0.2, color='blue')
        ax4.axhline(tr.phase[tr.iIn[0]], alpha=0.2, color='blue')
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='3%', pad=0.05)
    cbar1 = fig.colorbar(im4,ax=ax4, cax=cax)
#     ax4.set_title('G) PCA-Corrected Transmission Spectrum ({} PCs)'.format(tr.params[5]))

#     ax4.plot(tr.wv[iord][2044] * (1+tr.vrp/const.c), tr.phase)

    ax4.set_xlabel(r'Wavelength ($\mu$m)', fontsize=14)
    ax4.set_ylabel(r'Orbital phase ($\phi$)', fontsize=14)
    ax4.yaxis.set_label_coords(-0.05, 2)
    cbar1.set_label('Normalized flux', y=1.5, fontsize=14)
    if xlim is not None:
        ax4.set_xlim(*xlim)
        
    fig.subplots_adjust(hspace=0)

    fig.text(0.15, 0.83, 'A', fontsize = 12, bbox ={'facecolor':'white', 'alpha':0.8})
    fig.text(0.15, 0.63, 'B', fontsize = 12, bbox ={'facecolor':'white', 'alpha':0.8})
    fig.text(0.15, 0.44, 'C', fontsize = 12, bbox ={'facecolor':'white', 'alpha':0.8})
    fig.text(0.15, 0.26, 'D', fontsize = 12, bbox ={'facecolor':'white', 'alpha':0.8})

    fig.savefig('/home/boucher/spirou/Figures/fig_small_STEPS'+fig_name+'.pdf')

    return fig
    # cbar = plt.colorbar(im3, orientation="horizontal")
    
    
def plot_helium(tr, spec_fin_out, spec_fin, spec_fin_Sref, vrp=None,
                spec_fin_ts=None, add_RVsys=False, scale_y=1., iin=None, RV=0):

    he_lines = [1.083206, 1.083322, 1.083331]
    
    if iin is None:
        iin = tr.iIn
        
    if ~isinstance(RV, u.Quantity):
        RV = RV*u.km/u.s
    
#     if add_RVsys is True : 
#         vrp += tr.vrp + tr.planet.RV_sys + RV
#     else:
#         vrp = tr.vrp + RV
    if vrp is None:
        vrp = tr.vrp
        
    wave_shift_he = he_lines[0] * (1+((vrp+RV)/const.c).decompose())
    wave_shift_he2 = he_lines[1] * (1+((vrp+RV)/const.c).decompose())
    wave_shift_he3 = he_lines[2] * (1+((vrp+RV)/const.c).decompose())

    iord=8

    # ----- 1D -------
    
    plt.figure(figsize=(12,4))
#     fig,ax = plt.subplots(2,1,sharex=True, figsize=(15,8))
    plt.axhline(0,alpha=0.2, linestyle=":")

    plt.step(tr.wv[iord],spec_fin_out[iord]-1, where='mid', color='k', label='Out of transit', alpha=0.8)
    plt.fill_between(tr.wv[iord],spec_fin_out[iord]-1, step="mid", alpha=0.4, color='k')

    plt.step(tr.wv[iord],spec_fin[iord]-1, where='mid', color='c', label='Planet ref frame', alpha=0.8)
    plt.fill_between(tr.wv[iord],spec_fin[iord]-1, step="mid", alpha=0.4, color='c')

    plt.step(tr.wv[iord],spec_fin_Sref[iord]-1, where='mid', color='orange', label='Star ref frame', alpha=0.8)
    plt.fill_between(tr.wv[iord],spec_fin_Sref[iord]-1, step="mid", alpha=0.4, color='orange')

    if spec_fin_ts is not None:
        plt.step(tr.wv[iord],spec_fin_ts[iord]-1, where='mid', color='navy', 
                 label='Planet ref frame - pca corr', alpha=0.8)
        plt.fill_between(tr.wv[iord],spec_fin_ts[iord]-1, step="mid", alpha=0.4, color='navy')

    plt.legend(loc='lower right')

    for hel in he_lines:
        plt.axvline(hel * (1+(RV/const.c).decompose().value), color='red')
    plt.axvline(1.083 * (1+(RV/const.c).decompose().value), color='green')

    plt.xlim(1.0828-0.00005,1.0838+0.00005)
    plt.ylim(-0.02*scale_y,0.02*scale_y)
    plt.xlabel('Wavelength (um)')
    plt.ylabel('Excess absorption')
    
    _=plt.plot(tr.wv[iord].T, (tr.tellu[:,iord].T-1)*0.05+0.015, alpha=0.15, color='blue')

    # ----- 2D -------
    
    plt.figure(figsize=(15,4))
    im = plt.pcolormesh(tr.wv[iord], np.arange(tr.n_spec), tr.spec_trans[:,iord,:], cmap="viridis")

    plt.plot(wave_shift_he, np.arange(tr.n_spec), 'red',alpha=0.7)
    plt.plot(wave_shift_he2, np.arange(tr.n_spec), 'red',alpha=0.7)
    plt.plot(wave_shift_he3, np.arange(tr.n_spec), 'red',alpha=0.7)
    
    for hel in he_lines:
        plt.axvline(hel * (1+(RV/const.c).decompose().value), color='red',alpha=0.7)
    plt.axvline(1.083 * (1+(RV/const.c).decompose().value), color='black')
    
    plt.axhline(np.arange(tr.n_spec)[iin[0]], color='white', linestyle='--')
    plt.axhline(np.arange(tr.n_spec)[iin[-1]], color='white', linestyle='--')
    cbar = plt.colorbar(im, orientation='vertical')
    cbar.set_label(r'Excess absorption')
    im.set_clim(0.95,1.03)
#     plt.xlim(1.0828-0.0002,1.0838+0.0002)
    plt.xlim(1.0828-0.00005,1.0838+0.00005)
    
    plt.xlabel('Wavelength (um)')
    plt.ylabel(r'Spectrum number $\phi$')

    # plt.axvline(he_lines[0], color='orange',alpha=0.7)
    # plt.axvline(he_lines[1], color='orange',alpha=0.7)
    # plt.axvline(he_lines[2], color='orange',alpha=0.7)

#     plt.show()


def plot_detection_snrs(t, interp_grid_map, min_val_map, snrs_map, VMRs, Teq, id_cloud):

    size_vmr, size_teq = np.unique(VMRs[id_cloud], return_counts=True)

#     plt.figure()
    fig, ax = plt.subplots(1,2, figsize=(15,6))
    for i in range(size_vmr.size):
        ax[0].plot(Teq[id_cloud][i*size_teq[0]:(1+i)*size_teq[0]], 
                 min_val_map[i*size_teq[0]:(1+i)*size_teq[0]], '-o', 
                 label='H2O VMR = '+str(VMRs[id_cloud][i*size_teq[0]]))
    ax[0].legend(loc='best')
    ax[0].set_xlabel('Teq (K)')
    ax[0].set_ylabel('Correlation/logL value')

    idx_bruit = (interp_grid_map < t.planet.RV_sys.value-15) | (interp_grid_map > t.planet.RV_sys.value+15)
#     plt.figure()
    for i in range(size_vmr.size):
        ax[1].plot(Teq[id_cloud][i*size_teq[0]:(1+i)*size_teq[0]], 
                 (snrs_map[:,~idx_bruit]).max(axis=-1)[i*size_teq[0]:(1+i)*size_teq[0]], 
                 '-o', label='H2O VMR = '+str(VMRs[id_cloud][i*size_teq[0]]))
    ax[1].legend(loc='best')
    ax[1].set_xlabel('Teq (K)')
    ax[1].set_ylabel('SNR max')

    
#####################################################################################
import h5py
import corner
import shutil
import pygtc

def plot_mcmc_current_chains(filename, labels=None, truths=None,  
                             discard=0, param_no_zero=2, id_params=None, fig_name='',
                             show_titles=True, **corner_kwargs):

    copied_filename = filename  #shutil.copyfile(filename, hm.insert_str(filename, '_copy', -3)) 

    with h5py.File(copied_filename, "r") as f:

        samples = f['mcmc']['chain']
        if id_params is not None:
            samples = samples[:,:, id_params]
            
        ndim=np.array(samples).shape[-1]
        if labels is None:
            labels = ['' for i in range(ndim)]
            
        completed = np.where(samples[:,0,param_no_zero] == 0)[0]
        if completed.size == 0:
            cut_sample = samples[discard:,:,:]
            print('All Completed')
        else:
            cut_sample = samples[discard:completed[0],:,:]
            print('Completed {}/{}'.format(completed[0],samples.shape[0]))

        fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
        for i in range(ndim):
            ax = axes[i]
            ax.plot(cut_sample[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(cut_sample))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number");

        flat_samples = cut_sample.reshape(cut_sample.shape[0]*cut_sample.shape[1], ndim)

        fig = corner.corner(flat_samples, labels=labels, truths=truths,  # quantiles=[0.16, 0.5, 0.84],
                            show_titles=show_titles, **corner_kwargs);
        fig.savefig('/home/boucher/spirou/Figures/fig_mcmc'+fig_name+'.pdf')
        
#         GTC = pygtc.plotGTC(chains=[flat_samples], figureSize='APJ_page', paramNames=labels, nContourLevels=3,
#                            sigmaContourLevels=True)
        
    return fig
        
########################

def plot_ttest_map(tr, Kp_array, RV_array, sigma, p_value):
    
    fig, ax = plt.subplots(3,1, sharex=True, figsize=(12,6))

    im0 = ax[0].pcolormesh(RV_array, Kp_array, sigma)
    ax[0].set_ylabel('K_p')
    cbar = fig.colorbar(im0, ax=ax[0])
    cbar.set_label(r'$t$-test $\sigma$')
    ax[0].axhline(tr.Kp.value,color='white',alpha=0.5, linestyle=':')
    ax[0].axvline(0,color='white',alpha=0.5, linestyle=':')
    
    x = RV_array[(RV_array >= -15) & (RV_array <= 15)]
    y = sigma[hm.nearest(Kp_array, tr.Kp.value)][(RV_array >= -15) & (RV_array <= 15)]

    chose = a.find_max_spline(x, y.copy() , kind='max')
    print('T-val : Max value = {:.2f} // Max position = {:.2f}'.format(-chose[1], chose[0]))

    im1 = ax[1].pcolormesh(RV_array, Kp_array, np.log10(p_value), cmap='viridis_r')
#     ax[1].set_xlabel('RV shift')
    ax[1].set_ylabel('K_p')
    cbar = fig.colorbar(im1, ax=ax[1])
    cbar.set_label(r'log$_{10}$ p-value')
    ax[1].axhline(tr.Kp.value,color='white',alpha=0.5, linestyle=':')
    ax[1].axvline(0,color='white',alpha=0.5, linestyle=':')
    
    y = np.log10(p_value)[hm.nearest(Kp_array, tr.Kp.value)][(RV_array >= -15) & (RV_array <= 15)]
    chose = a.find_max_spline(x, np.ma.masked_invalid(y.copy()) , kind='min')
    print('P-val : Max value = {:.2f} // Max position = {:.2f}'.format(chose[1], chose[0]))
    
    im2 = ax[2].pcolormesh(RV_array, Kp_array, nf.pval2sigma(p_value), cmap='viridis')
    ax[2].set_xlabel('$v_{\rm offset}$')
    ax[2].set_ylabel('K_p')
    cbar = fig.colorbar(im2, ax=ax[2])
    cbar.set_label(r'Significance ($\sigma$)')
    ax[2].axhline(tr.Kp.value,color='white',alpha=0.5, linestyle=':')
    ax[2].axvline(0,color='white',alpha=0.5, linestyle=':')
    
    y = nf.pval2sigma(p_value)[hm.nearest(Kp_array, tr.Kp.value)][(RV_array >= -15) & (RV_array <= 15)]
    chose = a.find_max_spline(x, np.ma.masked_invalid(y.copy()) , kind='max')
    print('Sigma : Max value = {:.2f} // Max position = {:.2f}'.format(-chose[1], chose[0]))

    
    return -chose[1], chose[0]


def plot_ttest_map_hist(tr, corrRV, correlation, Kp_array, RV_array, sigma, ttest_params, ccf=None, 
                   orders=np.arange(49), masked=False, logl=False, plot_trail=False, 
                        Kp=None, RV=None, vrp=None, fig_name='', hist=True):
    
    speed_limit, limit_out, both_side, equal_var = ttest_params
    
    if hist is True:

        fig, ax = plt.subplots(2,1, figsize=(8,7))

        im0 = ax[0].pcolormesh(RV_array, Kp_array, sigma, rasterized=True)
        ax[0].set_ylabel(r'$K_{\rm P}$ (km s$^{-1}$)', fontsize=16)
        ax[0].set_xlabel(r'$v_{\rm rad}$ (km s$^{-1}$)', fontsize=16)

        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes('right', size='3%', pad=0.05)
        cbar = fig.colorbar(im0, ax=ax[0], cax=cax)
        cbar.set_label(r'$t$-test $\sigma$', fontsize=16)

        ax[0].axhline(tr.Kp.value,color='indigo',alpha=0.5, linestyle=':', label='Planet Rest Frame')
        ax[0].axvline(0,color='indigo',alpha=0.5, linestyle=':')
        fig.tight_layout(pad=1.0)


        if vrp is None:
            if Kp is None:
                Kp = tr.Kp.value
                vrp = tr.vrp.value
            else:
                vrp = rv_theo_nu(Kp, tr.nu*u.rad, tr.planet.w, plnt=True).value
        else:
            if Kp is None:
                Kp = tr.Kp.value

        x = RV_array[(RV_array >= -15) & (RV_array <= 15)]
        y = sigma[hm.nearest(Kp_array, Kp)][(RV_array >= -15) & (RV_array <= 15)]

        chose = a.find_max_spline(x, y.copy() , kind='max')
        print('T-val : Max value = {:.1f} // Max position = {:.1f}'.format(-chose[1], chose[0]))

        max_val = -chose[1]
        wind = chose[0]

        ax[0].scatter(wind, Kp, marker='+', color='k')#, 
    #                   label=r'{:.2f} // RV = {:.2f}'.format(max_val, wind))
    #     ax[0].legend(loc='lower right')


        if ccf is None:
            ccf = np.ma.sum(ccf[:,orders],axis=1)

        if logl is True:
            ccf = ccf-np.nanmean(ccf,axis=-1)[:,None]

        if masked is True:
            ccf[(ccf == 0).all(axis=-1)] = np.nan  

        if RV is not None:
            wind = RV
            print('Histogram for Kp = {:.2f} and RV = {:.2f}'.format(Kp,wind))

        if plot_trail is True:
            plt.figure()
            plt.pcolormesh(corrRV, np.arange(tr.n_spec),ccf)
            plt.plot(tr.berv, np.arange(tr.n_spec),'b')
            plt.plot(vrp+wind-speed_limit, np.arange(tr.n_spec),'k')
            plt.plot(vrp+wind+speed_limit, np.arange(tr.n_spec),'k')

            if both_side is True:
                plt.plot(vrp+wind+limit_out, np.arange(tr.n_spec),'r')
                plt.plot(vrp+wind-limit_out, np.arange(tr.n_spec),'r')
            else:
                plt.plot(vrp+wind+limit_out, np.arange(tr.n_spec),'r')

            plt.axhline(tr.iIn[0],linestyle='--',color='white')
            plt.axhline(tr.iIn[-1],linestyle='--',color='white')



        in_ccf, out_ccf = nf.get_corr_in_out_trail(tr.iIn, corrRV, ccf, tr, wind=wind, 
                                                speed_limit=speed_limit, limit_out=limit_out, 
                                                both_side=both_side, vrp=vrp)

    #     in_ccf_af, out_ccf_af = nf.get_corr_in_out_trail(tr.iOut, corrRV, ccf, tr, wind=wind, 
    #                                                   speed_limit=speed_limit, limit_out=limit_out, 
    #                                             both_side=both_side, vrp=vrp)

        A, B = in_ccf/np.nanstd(out_ccf), out_ccf/np.nanstd(out_ccf)
        title1 = ''
        labelA='In-Trail'
        labelB='Out-of-Trail'

        plt.figure()
        new_A = np.array(A)[np.isfinite(A)]
        new_B = np.array(B)[np.isfinite(B)]

        nf.t_test_hist(new_A, new_B, labelA, labelB, title1, ax[1])
        fig.tight_layout()

        fig.savefig('/home/boucher/spirou/Figures/fig_ttest{}.pdf'.format(fig_name))
        
    else:
        
        fig, ax = plt.subplots(1,1, figsize=(8,5))

        im0 = ax.pcolormesh(RV_array, Kp_array, sigma, rasterized=True)
        ax.set_ylabel(r'$K_{\rm P}$ (km s$^{-1}$)', fontsize=16)
        ax.set_xlabel(r'$v_{\rm rad}$ (km s$^{-1}$)', fontsize=16)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.05)
        cbar = fig.colorbar(im0, ax=ax, cax=cax)
        cbar.set_label(r'$t$-test $\sigma$', fontsize=16)

        ax.axhline(tr.Kp.value,color='indigo',alpha=0.5, linestyle=':', label='Planet Rest Frame')
        ax.axvline(0,color='indigo',alpha=0.5, linestyle=':')
        fig.tight_layout(pad=1.0)


        if vrp is None:
            if Kp is None:
                Kp = tr.Kp.value
                vrp = tr.vrp.value
            else:
                vrp = rv_theo_nu(Kp, tr.nu*u.rad, tr.planet.w, plnt=True).value
        else:
            if Kp is None:
                Kp = tr.Kp.value

        x = RV_array[(RV_array >= -15) & (RV_array <= 15)]
        y = sigma[hm.nearest(Kp_array, Kp)][(RV_array >= -15) & (RV_array <= 15)]

        chose = a.find_max_spline(x, y.copy() , kind='max')
        print('T-val : Max value = {:.1f} // Max position = {:.1f}'.format(-chose[1], chose[0]))

        max_val = -chose[1]
        wind = chose[0]

        ax.scatter(wind, Kp, marker='+', color='k')#, 
    #                   label=r'{:.2f} // RV = {:.2f}'.format(max_val, wind))
    #     ax[0].legend(loc='lower right')


        if ccf is None:
            ccf = np.ma.sum(ccf[:,orders],axis=1)

        if logl is True:
            ccf = ccf-np.nanmean(ccf,axis=-1)[:,None]

        if masked is True:
            ccf[(ccf == 0).all(axis=-1)] = np.nan  

        if RV is not None:
            wind = RV
            print('Histogram for Kp = {:.2f} and RV = {:.2f}'.format(Kp,wind))

#         if plot_trail is True:
#             plt.figure()
#             plt.pcolormesh(corrRV, np.arange(tr.n_spec),ccf)
#             plt.plot(tr.berv, np.arange(tr.n_spec),'b')
#             plt.plot(vrp+wind-speed_limit, np.arange(tr.n_spec),'k')
#             plt.plot(vrp+wind+speed_limit, np.arange(tr.n_spec),'k')

#             if both_side is True:
#                 plt.plot(vrp+wind+limit_out, np.arange(tr.n_spec),'r')
#                 plt.plot(vrp+wind-limit_out, np.arange(tr.n_spec),'r')
#             else:
#                 plt.plot(vrp+wind+limit_out, np.arange(tr.n_spec),'r')

#             plt.axhline(tr.iIn[0],linestyle='--',color='white')
#             plt.axhline(tr.iIn[-1],linestyle='--',color='white')



        in_ccf, out_ccf = nf.get_corr_in_out_trail(tr.iIn, corrRV, ccf, tr, wind=wind, 
                                                speed_limit=speed_limit, limit_out=limit_out, 
                                                both_side=both_side, vrp=vrp)

    #     in_ccf_af, out_ccf_af = nf.get_corr_in_out_trail(tr.iOut, corrRV, ccf, tr, wind=wind, 
    #                                                   speed_limit=speed_limit, limit_out=limit_out, 
    #                                             both_side=both_side, vrp=vrp)

        A, B = in_ccf/np.nanstd(out_ccf), out_ccf/np.nanstd(out_ccf)
        title1 = ''
        labelA='In-Trail'
        labelB='Out-of-Trail'

#         plt.figure()
        new_A = np.array(A)[np.isfinite(A)]
        new_B = np.array(B)[np.isfinite(B)]

#         nf.t_test_hist(new_A, new_B, labelA, labelB, title1, ax[1])
        fig.tight_layout()

        fig.savefig('/home/boucher/spirou/Figures/fig_ttest_map{}.pdf'.format(fig_name))

    return sp.stats.ttest_ind(A, B, nan_policy='omit', equal_var=equal_var)
    


# def plot_contrast(t, correlation, correlation_sig, index, icorr, n_pcas, vmrs, 
#                   limit_down=0, limit_up=0.98, n_pts=50, del_ord=[], add_ord=[], N_list=None, nolog_list=None):

#     contrast = []
#     contrast_sig = []

#     tresh_array = np.linspace(limit_down,limit_up, n_pts)
    
#     if N_list is None:
#         N_list = [t.N, t.N]
#     if nolog_list is None:
#         nolog_list = [False, True]

#     for tresh in tresh_array:
#         orders = nf.remove_values_from_array(a.bands(t.wv,'yjhk'), del_ord+nf.ord_frac_tresh(t, tresh))
#         orders = list(np.unique(list(orders)+list(add_ord)))

#         loglbl_contrast = corr.sum_logl(correlation, icorr, orders, N_list[0],
#                                         alpha=np.ones_like(t.alpha_frac), axis=0, 
#                                         del_idx=index, nolog=nolog_list[0]).squeeze() 
#         loglbl_contrast_sig = corr.sum_logl(correlation_sig, icorr, orders, N_list[1],
#                                             alpha=np.ones_like(t.alpha_frac), axis=0,
#                                             del_idx=index, nolog=nolog_list[1]).squeeze() 

#         mean_val = loglbl_contrast[:,np.where(np.log10(vmrs) == np.min(np.log10(vmrs)))].mean(axis=-1)
# #         print(np.min(np.log10(vmrs)), mean_val)
# #         print()
#         mean_val_sig = loglbl_contrast_sig[:,np.where(np.log10(vmrs) == np.min(np.log10(vmrs)))].mean(axis=-1)

#         cont = loglbl_contrast.max(axis=-1)-mean_val.squeeze()
#         cont_sig = loglbl_contrast_sig.max(axis=-1)-mean_val_sig.squeeze()
        
#         if (cont_sig.mask).all():
#             cont_sig = np.zeros_like(cont_sig)
        
#         contrast.append(cont)
#         contrast_sig.append(cont_sig)
        
#     contrast = np.ma.masked_invalid(contrast)
#     contrast_sig = np.ma.masked_invalid(contrast_sig)
    
#     id_max = np.where(contrast == np.max(contrast))
#     id_max_sig = np.where(contrast_sig == np.max(contrast_sig))

#     fig, ax = plt.subplots(1,2, figsize = (15,7))
#     im0 = ax[0].pcolormesh(n_pcas, tresh_array, contrast, shading='gouraud')
#     fig.colorbar(im0, ax=ax[0])
#     ax[0].plot(n_pcas[id_max[1][0]], tresh_array[id_max[0][0]], marker='+', color='white')
#     ax[0].text(n_pcas[id_max[1][0]], tresh_array[id_max[0][0]], '{:.3f}'.format(tresh_array[id_max[0][0]]), 
#                color='red')
#     for iord in range(t.nord):
#         ax[0].text(0.75*np.max(n_pcas)+iord/t.nord, t.N_frac[iord], '{}'.format(iord), color='white')
    
#     im1 = ax[1].pcolormesh(n_pcas, tresh_array, contrast_sig, shading='gouraud')
#     fig.colorbar(im1, ax=ax[1])
#     ax[1].plot(n_pcas[id_max_sig[1][0]], tresh_array[id_max_sig[0][0]], marker='+', color='white')
#     ax[1].text(n_pcas[id_max_sig[1][0]], tresh_array[id_max_sig[0][0]], 
#                '{:.3f}'.format(tresh_array[id_max_sig[0][0]]), color='red')
#     for iord in range(t.nord):
#         ax[1].text(0.75*np.max(n_pcas)+iord/t.nord, t.N_frac[iord], '{}'.format(iord), color='white')

#     return tresh_array, contrast, contrast_sig     


def plot_order(tr, iord, flux=None, xaxis=None, yaxis=None, show_slice=None, length=16,
               xlabel='', ylabel='', cbar=False, clim=[None,None], xlim=None, ylim=None, title='', **kwargs):
    
    if flux is None:
        flux = tr.final
        
    if xaxis is None:
        xaxis = tr.wv[iord]
    if yaxis is None:
        yaxis = np.arange(tr.n_spec)
    
    fig = plt.figure(figsize=(length,4))
    
    ax = fig.add_subplot(2,1,1)
    
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    
    im = ax.pcolormesh(xaxis, yaxis, flux[:,iord], vmin=clim[0], vmax=clim[1],**kwargs)
    ax.set_title(title)
    if cbar is True:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2%', pad=0.05)
        fig.colorbar(im,ax=ax, cax=cax)
    
    if show_slice is not None:
        ax2 = fig.add_subplot(2,1,2, sharex=ax)
        if show_slice == "mean":
            ax2.plot(xaxis, np.nanmean(flux[:,iord], axis=0), **kwargs)
        elif show_slice == "median":
            ax2.plot(xaxis, np.nanmedian(flux[:,iord], axis=0), **kwargs)
        else:
            ax2.plot(xaxis, flux[show_slice,iord], **kwargs)
#                vmin=np.nanpercentile(t1.final[:,45].data,1), vmax=0.050
#     return fig
        ax2.set_xlabel(xlabel)
    else:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    
        

# def calc_contrast(grid, x, y=None):

#     conds = (x == x.min())

#     if y is not None:
#         conds &= (y == y.min())
#         mean_val = grid[:,np.where(conds)]
#     else:
#         mean_val = grid[:,np.where(conds)].mean(axis=-1)
    
#     contrast = grid.max(axis=-1)-mean_val.squeeze()
#     return contrast

    

def plot_logl_grid(logl_grid, n_pcas, cases, cond, pCloud, corrRV0, sig='with', correl=False, 
                   var1='H2O', var2='Tmid', var_name='Pcloud', var_unit='Pa', fig_name='',minmax='max', **kwargs):
    uniq_p = np.unique(pCloud[cond])
    if uniq_p.ndim == 0:
        uniq_p = [uniq_p]
    for pcl in uniq_p:

        id_pcl= np.where(pCloud[cond] == pcl)[0]

        plot_logl_grid_i([corrRV0], logl_grid[:,:,id_pcl], cases[var1][cond][id_pcl], 
                      cases[var2][cond][id_pcl], n_pcas, 
                      good_rv_idx=0, switch=False, n_lvl=45, 
                      title = var_name+' = {} '.format(pcl)+var_unit, correl=correl,
                      fig_name='{}'.format(int(pcl))+fig_name, minmax=minmax,**kwargs)
        #plot_all_logl
    if minmax == 'max':
        idx = np.argmax(logl_grid, axis=-1).squeeze()
    elif minmax == 'min':
        idx = np.argmin(logl_grid, axis=-1).squeeze()
        
    mean_val = logl_grid[:,:,np.where((cases[var1][cond] == cases[var1][cond].min()) &\
                                      (cases[var2][cond] == cases[var2][cond].min()))].squeeze()
    
    contrast =  (logl_grid.max(axis=-1).T-mean_val).T.squeeze()

    if mean_val.ndim > 1:
        print(Table([n_pcas, cases[var1][cond][idx], cases[var2][cond][idx], pCloud[cond][idx],
             *[Column([round(col_i,2) for col_i in col.data]) for col in contrast]]))
    else:
        print('{} = {} // {} = {} // {} = {} '.format(var1, np.log10(cases[var1][cond][idx]), 
                                                      var2, cases[var2][cond][idx], 
                                                      var_name, pCloud[cond][idx]))
#         print('Contrast = ',*['{:.2f}'.format(cont) for cont in contrast])




def plot_airmass(list_tr, markers=['o','s','d'], 
                colors=['darkblue','dodgerblue','darkorange'], fig_name=''):

    plt.figure(figsize=(8,3.5))

    for i,tr in enumerate(list_tr):
        plt.plot(tr.phase, tr.AM,'-', marker=markers[i], color=colors[i], label='Transit {}'.format(i+1))

    phase_t1 = np.min([tr.phase[tr.iIn[0]] for tr in list_tr])
    phase_t2 = np.min([tr.phase[tr.total[0]] for tr in list_tr])
    phase_t3 = np.max([tr.phase[tr.total[-1]] for tr in list_tr])
    phase_t4 = np.max([tr.phase[tr.iIn[-1]] for tr in list_tr])

    plt.axvspan(phase_t1, phase_t4, alpha=0.2, label='Ingress/Egress')
    plt.axvspan(phase_t2, phase_t3, alpha=0.2)
    plt.axvspan(phase_t2, phase_t2, alpha=0.4, label='Total Transit')

    plt.ylabel('Airmass', fontsize=16)
    plt.xlabel(r'Orbital phase ($\phi$)', fontsize=16)
    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()

    plt.savefig('/home/boucher/spirou/Figures/fig_airmass_{}.pdf'.format(fig_name))

    fig, ax = plt.subplots(2,1, figsize=(9,8))
    
    hband = a.bands(tr.wv,'h')[2:-2]
    
    for i,tr in enumerate(list_tr):
        ax[0].plot(np.mean(tr.wv,axis=-1).T, np.nanmean(tr.SNR,axis=0).T,
                   '-', marker=markers[i], color=colors[i], label='Transit {}'.format(i+1))


    ax[0].set_ylabel('Mean S/N\nper order', fontsize=16)
    ax[0].set_xlabel(r'Wavelength ($\mu$m)', fontsize=16)
    ax[0].axvspan(np.mean(tr.wv,axis=-1)[28], np.mean(tr.wv,axis=-1)[36], alpha=0.2, color='darkorange',label='H-band')
    ax[0].legend(loc='upper left', fontsize=12) #, bbox_to_anchor=(0.9, 0.71)

    for i,tr in enumerate(list_tr):
        ax[1].plot(tr.phase, np.nanmean(tr.SNR[:, hband],axis=-1),'-', marker=markers[i], color=colors[i])


    ax[1].set_ylabel('Mean H-band S/N\nper exposure', fontsize=16)
    ax[1].set_xlabel(r'Orbital phase ($\phi$)', fontsize=16)

    ax[1].axvspan(phase_t1, phase_t4, alpha=0.2, label='Ingress/Egress')
    ax[1].axvspan(phase_t2, phase_t3, alpha=0.2)
    ax[1].axvspan(phase_t2, phase_t2, alpha=0.4, label='Total Transit')

    ax[1].legend(loc='best', fontsize=12) #, bbox_to_anchor=(0.9, 0.71)

    plt.savefig('/home/boucher/spirou/Figures/fig_SNR_{}.pdf'.format(fig_name))
        
            
# def plot_logl(corrRV0, loglbl, var_in, var_out, n_pcas, good_rv_idx=0, switch=False):

#     size_in, size_out = np.unique(var_in, return_counts=True)
#     var_in_list = size_in[::-1]
#     var_out_list = np.unique(var_out)
#     if switch is False:
#         range_list = size_in.size
#     else:
#         range_list = size_out[0]

#     lstyles = ['-','--','-.',':']
#     mark = ['o','.','+','x','*',',','d','v','s','^']

#     for idx_pca in range(len(n_pcas)):
#         if len(n_pcas) > 1 :
#             loglbl_npc =  loglbl[:,idx_pca]
#         else: 
#             loglbl_npc =  loglbl

#         plt.figure(figsize=(8,6))
#         for v,rv in enumerate(corrRV0):

#             valeurs = (loglbl_npc)[v] #- loglbl_npc[0] #np.mean(loglbl_npc[np.array([0,2])],axis=0)
            
#             for i in range(range_list):
#                 if v == good_rv_idx:
#                     if switch is False:
#                         couleur = (0.5, 0.0, i/size_in.size)
#                         labels='H2O VMR = '+str(var_in[i*size_out[0]])
#                     else:
#                         couleur = (0.5, 0.0, i/size_out[0])
#                         labels=str(var_out[i])+'K '
#                 else:
#                     couleur = 'grey'
#                     labels='_nolegend_'
#                 if switch is False:
#                     plt.plot(var_out[i*size_out[0]:(1+i)*size_out[0]], 
#                          valeurs[i*size_out[0]:(1+i)*size_out[0]], linestyle = lstyles[i%len(lstyles)], 
#                          marker=mark[i%len(mark)], color=couleur,
#                          label=labels)
#                 else:
#                     plt.plot(np.log10(var_in_list), 
#                          valeurs[np.where(var_out == var_out_list[i])], linestyle = lstyles[i%len(lstyles)], 
#                          marker=mark[i%len(mark)], color=couleur,
#                          label=labels)

#         plt.legend(loc='best', fontsize=9)
#         if switch is False:
#             plt.xlabel('Teq (K)', fontsize=16)
#         else:
#             plt.xlabel('log10 VMRs [H2O]', fontsize=16)
#         plt.ylabel(r'log $L$', fontsize=16)
#         plt.title('N PCs = {}'.format(n_pcas[idx_pca]), fontsize=16)
#         max_val = loglbl_npc.max()
#         if switch is False:
#             plt.ylim(max_val-20, max_val+5)
#         plt.tight_layout()
                        

# def plot_logl_map(tr, var_out, var_in, logl_grid, n_pcas, good_rv_idx=0, n_lvl=None, vmin=None):

#     var_out_list, var_out_nb = np.unique(var_out, return_counts=True)
#     var_in_list, var_in_nb = np.unique(var_in, return_counts=True)
#     var_in_list = var_in_list[::-1]
    
#     if vmin is None:
#         vmin=logl_grid.min()

#     for idx_pca in range(len(n_pcas)):
#         if len(n_pcas) > 1 :
#             loglbl_npc =  logl_grid[:,idx_pca]
#         else: 
#             loglbl_npc =  logl_grid

#         loglbl_npc = loglbl_npc[good_rv_idx]

#         im_logl = loglbl_npc.reshape(var_in_nb.size, var_out_nb.size).T

#         max_val_idx = np.where(im_logl == im_logl.max())

#         plt.figure(figsize=(6,4))
    
#         plt.pcolormesh(np.log10(var_in_list), var_out_list, im_logl, cmap='inferno', shading='gouraud', vmin=vmin)
#         plt.axvline(np.log10(var_in_list)[max_val_idx[1]],color='black',linestyle=':', alpha=0.3)
#         plt.axhline(var_out_list[max_val_idx[0]],color='black',linestyle=':', alpha=0.3)
#         plt.plot(np.log10(var_in_list)[max_val_idx[1]], var_out_list[max_val_idx[0]],'k+', 
#                 label='VMR={} // Teq={}'.format(var_in_list[max_val_idx[1]].data[0],
#                                             var_out_list[max_val_idx[0]].data[0] ))
#         plt.legend(loc='best', fontsize=9)
#         plt.xlabel('VMRs [H2O]', fontsize=16)
#         plt.ylabel('Teq (K)', fontsize=16)
#         cbar = plt.colorbar()
#         cbar.set_label(r'log $L$', fontsize=16)
#         plt.title('N PCs = {}'.format(n_pcas[idx_pca]), fontsize=16)
#         plt.tight_layout()
#         if n_lvl is not None:
#             plt.contour(np.log10(var_in_list), var_out_list, im_logl, n_lvl, 
#                         extent=(np.log10(var_in_list[0]),np.log10(var_in_list[-1]),\
#                           var_out_list[0],var_out_list[-1]), cmap='inferno_r', alpha=0.5, vmin=vmin)