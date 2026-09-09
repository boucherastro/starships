import numpy as np
from . import homemade as hm
from . import analysis as a
from . import retrieval_utils as ru
from . import ttest_fcts as nf
from .orbite import rv_theo_nu
from .mask_tools import interp1d_masked
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import scipy.constants as cst
import scipy as sp
from astropy import units as u
from astropy import constants as const

import matplotlib.pyplot as plt
# from itertools import islice
from astropy.table import Table, Column

from pathlib import Path

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.basicConfig()

# Initiate random number generator
rng = np.random.default_rng()


retrieval_plot_labels = { 'H2O': r"$\log_{10}$ H$_2$O",
                          'CO': r"$\log_{10}$ CO",
                          'CO2': r"$\log_{10}$ CO$_2$",
                          'FeH': r"$\log_{10}$ FeH",
                          'TiO': r"$\log_{10}$ TiO",
                          'VO': r"$\log_{10}$ VO",
                          'C2H2': r"$\log_{10}$ C$_2$H$_2$",
                          'HCN': r"$\log_{10}$ HCN",
                          'OH': r"$\log_{10}$ OH",
                          'H-': r"$\log_{10}$ H$^-$",
                          'temp': r"$T_{\rm P}$",
                          'cloud': r"$\log_{10}P_{clouds}$",
                          'rpl': r"$R_{\rm P}$",
                          'kp': r"$K_{\rm P}$",
                          'rv': r"$v_{\rm rad}$",
                          'tp_delta': r'$\log_{10} \delta$',
                          'tp_gamma': r'$\log_{10} \gamma$',
                          'tp_kappa': r'$\log_{10} \kappa$',
                          'tp_ptrans': r'$\log_{10} P_{trans}$',
                          'tp_alpha': r'$\alpha$'}

# Define colors gradations for plots
# Copied from pyGTCT.
colorsDict = {# Match pygtc up to v0.2.4
    'blues_old': ('#4c72b0', '#7fa5e3', '#b2d8ff'),
    'greens_old': ('#55a868', '#88db9b', '#bbffce'),
    'yellows_old': ('#f5964f', '#ffc982', '#fffcb5'),
    'reds_old': ('#c44e52', '#f78185', '#ffb4b8'),
    'purples_old': ('#8172b2', '#b4a5e5', '#37d8ff'),
    # New color scheme, dark colors match matplotlib v2
    'blues': ('#1f77b4', '#52aae7', '#85ddff'),
    'oranges': ('#ff7f0e', '#ffb241', '#ffe574'),
    'greens': ('#2ca02c', '#5fd35f', '#92ff92'),
    'reds': ('#d62728', '#ff5a5b', '#ff8d8e'),
    'purples': ('#9467bd', '#c79af0', '#facdff'),
    'browns': ('#8c564b', '#bf897e', '#f2bcb1'),
    'pinks': ('#e377c2', '#ffaaf5', '#ffddff'),
    'grays': ('#7f7f7f', '#b2b2b2', '#e5e5e5'),
    'yellows': ('#bcbd22', '#eff055', '#ffff88'),
    'cyans': ('#17becf', '#4af1ff', '#7dffff')
}
defaultColorsOrder = ['blues', 'oranges', 'greens', 'reds', 'purples',
                      'browns', 'pinks', 'grays', 'yellows', 'cyans']



def setup_default_plot_params():
    """
    Setup plot parameters for nice looking plots in latex.
    """
    # Set up latex fonts
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=16)

    # Set up tick parameters
    plt.rc('xtick', direction='in')
    plt.rc('xtick', top=True)
    plt.rc('xtick', bottom=True)
    plt.rc('xtick', labelsize=16)
    plt.rc('xtick.major', size=8)
    plt.rc('xtick.minor', size=4)
    plt.rc('xtick.major', width=1)
    plt.rc('xtick.minor', width=1)

    plt.rc('ytick', direction='in')
    plt.rc('ytick', left=True)
    plt.rc('ytick', right=True)
    plt.rc('ytick', labelsize=16)
    plt.rc('ytick.major', size=8)
    plt.rc('ytick.minor', size=4)
    plt.rc('ytick.major', width=1)
    plt.rc('ytick.minor', width=1)

    return

def get_plot_labels(params=None, retrieval_obj=None, species=None):
    """
    Get labels for plots from retrieval object.
    Args:
        retrieval_obj: imported retrieval code.
        Use `retrieval_obj = importlib.import_module(retrieval_code_filename)`

    Returns:
        labels for corner and chains plots
    """
    if params is None:
        if retrieval_obj is None:
            raise ValueError('Either params or retrieval_obj must be specified.')
        else:
            # Get all params names from retrieval object
            params = ru.get_all_param_names(retrieval_obj)
                
    if species is None and retrieval_obj is not None:
        species = retrieval_obj.species_in_prior
    elif species is None:
        species = list()
        
    # Get corresponding labels (if not found, use param name)
    labels = list()
    for key in params:
        try:
            lbl = retrieval_plot_labels[key]
        except KeyError:
            lbl = key
            if key in species:
                lbl = r"$\log_{10}$ " + key
                log.info(f"Species {key} not found in retrieval_plot_labels. Using {lbl}.")
            else:
                lbl = key
        labels.append(lbl)

    return labels

def get_plot_limits_from_data(data, pad=0.1):
    plt_limits = [np.min(data), np.max(data)]
    d_lim = plt_limits[1] - plt_limits[0]
    plt_limits[0] -= pad * d_lim
    plt_limits[-1] += pad * d_lim

    return plt_limits


def plot_all_logl(corrRV0, loglbl, var_in, var_out, n_pcas, good_rv_idx=0, switch=False,
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
            
            
def plot_logl_grid_i(corrRV0, loglbl, var_in, var_out, n_pcas, good_rv_idx=0, switch=False,
                  vmin=None, vmax=None, title='', point=None, correl=False, cmap='inferno',
                  cbar_label=r'log $L$', title_it=True, tag_max=True, fig_name='', minmax='min', xlim_remove=0,
                    contours_ccf=[3,2,1], path_fig=None):

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
                        extent=(np.log10(var_in_list[0]),np.log10(var_in_list[-1]),
                                var_out_list[0],var_out_list[-1]), cmap='tab20b_r', alpha=0.8) #cmap+'_r'  #'Spectral_r'
        if xlim_remove > 0:
            ax.set_xlim(None,x_new.max()-xlim_remove)
        else:
            ax.set_xlim(x_new.min()-xlim_remove,None)
        if point is not None:
            ax.plot(*point, 'o', color='dodgerblue')
        if path_fig is not None:
            fig.savefig(path_fig+'fig_grid_logl_'+fig_name+'.pdf')
            
            
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
        ax_map.set_ylabel('$K_p$ [km s$^{-1}$]', fontsize=14)
        ax_map.set_xlabel('$v_{\rm offset}$ [km s$^{-1}$]', fontsize=14)
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
    
    
# def plot_all_orders_correl(corrRV, ccf, tr, output_file=None, icorr=None, logl=False, tresh=0.4, sharey=True,
#                            vrp=None, RV_sys=None, vmin=None,vmax=None, vline=None, hline=None, kind='snr', 
#                            return_snr=False):
#     if icorr is None:
#         icorr = tr.icorr
        
        
#     if vrp is None:
#         vrp = (tr.vrp-tr.vr).value
#     if RV_sys is None:
#         RV_sys = tr.planet.RV_sys
        
#     fig, ax = plt.subplots(7,7, figsize=(16,12), sharex=True, sharey=sharey)
# #     fig_shift, ax_shift = plt.subplots(7,7, figsize=(16,12), sharex=True, sharey=True)
#     fig_single, ax_single = plt.subplots(7,7, figsize=(16,12), sharex=True, sharey=True)
#     mean_snr = np.ma.median(tr.SNR, axis=0)
#     pix_frac = tr.N_frac
    
#     snr_list = []
#     for i in range(7):
#         for j in range(7):
#             if i*7+j in a.bands(tr.wv, 'y'):
#                 fg_color = 'goldenrod'
#             if i*7+j in a.bands(tr.wv, 'j'):
#                 fg_color = 'olivedrab'
#             if i*7+j in a.bands(tr.wv, 'h'):
#                 fg_color = 'steelblue'
#             if i*7+j in a.bands(tr.wv, 'k'):
#                 fg_color = 'rebeccapurple'
#             if pix_frac[i*7+j] < tresh:
#                 fg_color = 'firebrick'
            
#             if logl is True:
#                 ax[i,j].pcolormesh(corrRV, np.arange(ccf.shape[0]), \
#                                    ccf[:,i*7+j]-np.nanmean(ccf[:,i*7+j], axis=-1)[:,None], vmin=vmin, vmax=vmax)
#             else:
#                 ax[i,j].pcolormesh(corrRV, np.arange(ccf.shape[0]), ccf[:,i*7+j], vmin=vmin, vmax=vmax)
#             ax[i,j].plot(vrp, np.arange(ccf.shape[0]), 'k:', alpha=0.5)

            
# #             ax[i,j].plot(tr.berv, np.arange(ccf.shape[0]), 'r--', alpha=0.5)
#             ax[i,j].set_title('{} - SNR:{:.0f} - {:.2f}'.format(i*7+j, mean_snr[i*7+j], pix_frac[i*7+j]), 
#                               color=fg_color)
            
#             shifted_corr, interp_grid, courbe, snr, _ = a.calc_snr_1d(ccf[icorr,i*7+j], corrRV, \
#                                                             vrp[icorr], RV_sys=RV_sys)
#             snr_list.append(snr)
#             if kind == 'courbe':
#                 ax_single[i,j].plot(interp_grid, courbe)
#             else:
#                 ax_single[i,j].plot(interp_grid, snr)
#                 ax_single[i,j].set_ylim(-4,4)
#             ax_single[i,j].set_title('{} - SNR:{:.0f} - {:.2f}'.format(i*7+j, mean_snr[i*7+j], pix_frac[i*7+j]), 
#                                      color=fg_color)
#             ax_single[i,j].axvline(0, linestyle=':', alpha=0.5)
#             if vline is not None:
#                 ax_single[i,j].axvline(vline, linestyle='-', alpha=0.5, color='navy')
#             if hline is not None:
#                 ax_single[i,j].axhline(hline, linestyle='-', alpha=0.5, color='navy')
#             ax_single[i,j].axvline(np.mean(tr.berv), linestyle='--', color='red', alpha=0.5)

#     if output_file is not None:
#         output_file = Path(output_file)
#         fig.savefig(output_file.with_stem(f'{output_file.stem}_2d'))
#         fig_single.savefig(output_file.with_stem(f'{output_file.stem}_single'))

#     if return_snr is True:
#         return interp_grid, snr_list

def plot_all_orders_correl(corrRV, ccf, tr, output_file=None, icorr=None, logl=False, tresh=0.4, sharey=True,
                            vrp=None, RV_sys=None, limit_shift=60., vmin=None,vmax=None, vline=None, hline=None, kind='snr',
                            return_snr=False, orders=None):
    
    """
    Plot Vrad vs exposure number for all orders, then Vrad vs SNR (in σ) for all orders.
    The text above each small plot indicates: {index} - SNR: {mean SNR of that order} - {fraction of pixels with signal in that order}. The color of this text is the wavelength band: yellow-orange = y band; green = j band; steel blue = h band; purple = k band; red = areas of strong telluric absorption.
    The dashed vertical line is at Vrad = 0, or the Vsys of the planet in the planetary rest frame.    
    Each bottom plot shows the sum of each column in its associated top plot, normalized. The horizontal blue line is set at 2σ. A peak above 2σ or 3σ indicates a detection signal in that order. This is useful to check from which order / wavelength range a detection signal really comes from.
       Args:
        corrRV (array-like): The RV values for correlation.
        ccf (array-like): The cross-correlation function.
        tr (object): The transit object.
        output_file (str, optional): The output file path. Defaults to None.
        icorr (int, optional): The index of the correlation. Defaults to None.
        logl (bool, optional): Whether to plot the logarithm of the correlation. Defaults to False.
        tresh (float, optional): The threshold for pixel fraction. Defaults to 0.4.
        sharey (bool, optional): Whether to share the y-axis among subplots. Defaults to True.
        vrp (array-like, optional): The radial velocity values. Defaults to None.
        RV_sys (float, optional): The system radial velocity. Defaults to None.
        limit_shift (float, optional): The limit for shifting. Defaults to 60.
        vmin (float, optional): The minimum value for the color scale. Defaults to None.
        vmax (float, optional): The maximum value for the color scale. Defaults to None.
        vline (float, optional): The vertical line position. Defaults to None.
        hline (float, optional): The horizontal line position. Defaults to None.
        kind (str, optional): The type of plot ('snr' or 'courbe'). Defaults to 'snr'.
        return_snr (bool, optional): Whether to return the SNR list. Defaults to False.
        orders (array-like, optional): Orders used in the analysis. If not None, adds a big red "X" on the orders that were not used in the analysis.

    Returns:
        list: The SNR list if return_snr is True.
    """
    
    if icorr is None:
        icorr = tr.icorr

    if vrp is None:
        vrp = (tr.vrp-tr.vr).value
    if RV_sys is None:
        RV_sys = tr.planet.RV_sys

    # Get the number of orders
    n_orders = ccf.shape[1]
    # Get the number of rows
    n_rows = int(np.ceil(n_orders/7))
    # Get the number of columns
    n_cols = int(np.ceil(n_orders/n_rows))
    
    fig, ax = plt.subplots(n_rows,n_cols, figsize=(16,12), sharex=True, sharey=sharey)
    fig_single, ax_single = plt.subplots(n_rows,n_cols, figsize=(16,12), sharex=True, sharey=True)

    mean_snr = np.ma.median(tr.SNR, axis=0)
    pix_frac = tr.N_frac

    snr_list = []
    for i in range(n_rows):
        for j in range(n_cols):
            index = i * n_cols + j
            if index >= n_orders:
                break  # Exit the loop if the index is out of bounds
            if index in a.bands(tr.wv, 'y'):
                fg_color = 'goldenrod'
            if index in a.bands(tr.wv, 'j'):
                fg_color = 'olivedrab'
            if index in a.bands(tr.wv, 'h'):
                fg_color = 'steelblue'
            if index in a.bands(tr.wv, 'k'):
                fg_color = 'rebeccapurple'
            if index >= len(pix_frac) or pix_frac[index] < tresh:
                fg_color = 'firebrick'

            if logl is True:
                ax[i,j].pcolormesh(corrRV, np.arange(ccf.shape[0]), \
                                   ccf[:,i*n_cols+j]-np.nanmean(ccf[:,i*n_cols+j], axis=-1)[:,None], vmin=vmin, vmax=vmax)
            else:
                ax[i,j].pcolormesh(corrRV, np.arange(ccf.shape[0]), ccf[:,i*n_cols+j], vmin=vmin, vmax=vmax)
            ax[i,j].plot(vrp, np.arange(ccf.shape[0]), 'k:', alpha=0.5)
            
            ax[i,j].set_title('{} - SNR:{:.0f} - {:.2f}'.format(i*n_cols+j, mean_snr[i*n_cols+j], pix_frac[i*n_cols+j]), color=fg_color)

            shifted_corr, interp_grid, courbe, snr, _ = a.calc_snr_1d(ccf[icorr,i*n_cols+j], corrRV, \
                                                            vrp[icorr], RV_sys=RV_sys, limit_shift=limit_shift)
            snr_list.append(snr)
            if kind == 'courbe':
                ax_single[i,j].plot(interp_grid, courbe)
            else:
                ax_single[i,j].plot(interp_grid, snr)
                ax_single[i,j].set_ylim(-4,4)
            ax_single[i,j].set_title('{} - SNR:{:.0f} - {:.2f}'.format(i*n_cols+j, mean_snr[i*n_cols+j], pix_frac[i*n_cols+j]),
                                        color=fg_color)
            ax_single[i,j].axvline(0, linestyle=':', alpha=0.5)
            if vline is not None:
                ax_single[i,j].axvline(vline, linestyle='-', alpha=0.5, color='navy')
            if hline is not None:
                ax_single[i,j].axhline(hline, linestyle='-', alpha=0.5, color='navy')
            ax_single[i,j].axvline(np.mean(tr.berv), linestyle='--', color='red', alpha=0.5)
            
            if orders is not None:
                if index not in orders:
                    left1, width1 = corrRV[0], corrRV[-1] - corrRV[0]
                    bottom1, height1 = 0, ccf.shape[0]
                    right1 = left1 + width1
                    top1 = bottom1 + height1
                    ax[i,j].text(0.5 * (left1 + right1), 0.4 * (bottom1 + top1), "X", horizontalalignment='center',
                                 verticalalignment='center', color="red",fontsize=65.)
                    
                    left2, width2 = interp_grid[0], interp_grid[-1] - interp_grid[0]
                    bottom2, height2 = -4, 8
                    right2 = left2 + width2
                    top2 = bottom2 + height2
                    ax_single[i,j].text(0.5 * (left2 + right2), 0.4 * (bottom2 + top2), "X", horizontalalignment='center',
                                        verticalalignment='center', color="red",fontsize=65.)

    if output_file is not None:
        output_file = Path(output_file)
        fig.savefig(output_file.with_stem(f'{output_file.stem}_2d'))
        fig_single.savefig(output_file.with_stem(f'{output_file.stem}_single'))

    if return_snr is True:
        return snr_list            
            
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
               cmap=None, bad_color='red', path_fig=None):
    
    '''
    Plot main steps of data reduction.
    
    id_spec:    index of the exposure
    '''
    
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
    if path_fig is not None:
        fig.savefig(path_fig+'fig_STEPS'+fig_name+'.pdf')

    
def plot_five_steps(tr, iord, xlim=None, masking_limit=0.8, fig_name='',
                     cmap=None, bad_color='red', length=16, id_spec=10, path_fig=None):
    
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

    ax01.text( tr.wv[iord][100],tr.phase[10], 'A', fontsize = 12, bbox ={'facecolor':'white', 'alpha':0.8})
    ax2.text( tr.wv[iord][100],tr.phase[10], 'B', fontsize = 12, bbox ={'facecolor':'white', 'alpha':0.8})
    ax3.text( tr.wv[iord][100],tr.phase[10], 'C', fontsize = 12, bbox ={'facecolor':'white', 'alpha':0.8})
    ax4.text( tr.wv[iord][100],tr.phase[10], 'D', fontsize = 12, bbox ={'facecolor':'white', 'alpha':0.8})

    if path_fig is not None:
        # fig.tight_layout()
        fig.savefig(path_fig +'fig_five_STEPS_'+fig_name+'.png')

    return fig
    # cbar = plt.colorbar(im3, orientation="horizontal")
    
    
    
def plot_small_steps(tr, iord, xlim=None, masking_limit=0.8, fig_name='',
                     cmap=None, bad_color='red', length=14, path_fig=None):
    
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

    if path_fig is not None:
        fig.savefig(path_fig+'fig_small_STEPS'+fig_name+'.pdf')

    return fig
    # cbar = plt.colorbar(im3, orientation="horizontal")
    
    
def plot_helium(tr, spec_fin_out, spec_fin, spec_fin_Sref, vrp=None,
                spec_fin_ts=None, scale_y=1., iin=None, RV=0):

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


def plot_mcmc_current_chains(filename, labels=None, truths=None,  
                             discard=0, param_no_zero=2, id_params=None, fig_name='',
                             show_titles=True, path_fig=None, **corner_kwargs):

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
        
        if path_fig is not None:
            fig.savefig(path_fig+'fig_mcmc'+fig_name+'.pdf')

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
                        show_rest_frame=True, Kp=None, RV=None, vrp=None, fig_name='',
                        path_fig=None, hist=True, cmap=None, tellu_loc=None):
    
    '''
    Plot Kp/Vrad map, T-test and Trail.
    
    In the Kp/Vrad map, the horizontal dotted line indicates the known Kp value of the planet. The vertical dotted line centers at the highest signal at the known Kp that is close to Vrad = 0. A signal at the intersection of these lines signifies a detection.
    
    The T-test plot tells us if the in-trail and out-of-trail distributions (both randomly selected values, respectively close to and far from the Vrad of the planet) are drawn from the same parent distribution.
    
    The trail plot is where the values used in the T-test are drawn from. The region outside of the black lines but within the red ones is not used because it is unclear if it is in-trail or out-of-trail.
    '''
    
    speed_limit, limit_out, both_side, equal_var = ttest_params
    
    if hist is True:

        fig, ax = plt.subplots(2,1, figsize=(8,7))

        im0 = ax[0].pcolormesh(RV_array, Kp_array, sigma, rasterized=True, cmap=cmap)
        ax[0].set_ylabel(r'$K_{\rm P}$ [km s$^{-1}$]', fontsize=16)
        ax[0].set_xlabel(r'$v_{\rm rad}$ [km s$^{-1}$]', fontsize=16)

        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes('right', size='3%', pad=0.05)
        cbar = fig.colorbar(im0, ax=ax[0], cax=cax)
        cbar.set_label(r'$t$-test $\sigma$', fontsize=16)

        ax[0].axhline(tr.Kp.value,color='r', linestyle=':', label='Planet Rest Frame')
        ax[0].axvline(0,color='r', linestyle=':')
        
        # Position of tellurics
        if tellu_loc is not None:
            if tellu_loc >= RV_array[0] and tellu_loc <= RV_array[-1]:  # if the telluric are within the limits of the x axis
                ax[0].axvline(tellu_loc, linestyle=":", alpha=0.85, color="saddlebrown")
            else:
                print("The telluric residuals are not located within the x axis limits of this plot")
        
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

        # ax[0].scatter(wind, Kp, marker='+', color='k')#,


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
            plt.figure(figsize=(8,5))
            plt.pcolormesh(corrRV, np.arange(tr.n_spec),ccf)
            plt.plot(tr.berv, np.arange(tr.n_spec),'b')
            plt.plot(vrp+wind-speed_limit, np.arange(tr.n_spec),'k', label="In-trail (inside the lines)")
            plt.plot(vrp+wind+speed_limit, np.arange(tr.n_spec),'k')

            if both_side is True:
                plt.plot(vrp+wind+limit_out, np.arange(tr.n_spec),'r', label="Out-of-trail (outside the lines)")
                plt.plot(vrp+wind-limit_out, np.arange(tr.n_spec),'r')
            else:
                plt.plot(vrp+wind+limit_out, np.arange(tr.n_spec),'r')

            plt.axhline(tr.iIn[0], linestyle='--', color='white', label="In transit observations")
            plt.axhline(tr.iIn[-1], linestyle='--', color='white')
            
            plt.xlabel(r'$v_{\rm rad}$ [km s$^{-1}$]', fontsize=16)
            plt.ylabel("Observation number", fontsize=16)
            plt.legend(loc="upper right")



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

        if path_fig is not None:
            fig.savefig(path_fig+'fig_ttest{}.pdf'.format(fig_name))
        
    else:
        
        fig, ax = plt.subplots(1,1, figsize=(8,5))

        im0 = ax.pcolormesh(RV_array, Kp_array, sigma, rasterized=True, cmap=cmap)
        ax.set_ylabel(r'$K_{\rm P}$ [km s$^{-1}$]', fontsize=16)
        ax.set_xlabel(r'$v_{\rm rad}$ [km s$^{-1}$]', fontsize=16)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.05)
        cbar = fig.colorbar(im0, ax=ax, cax=cax)
        cbar.set_label(r'$t$-test $\sigma$', fontsize=16)

        if show_rest_frame:
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

        # if show_max:
            # ax.scatter(wind, Kp, marker='+', color='k')#,
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

        if path_fig is not None:
            fig.savefig(path_fig+'fig_ttest_map{}.pdf'.format(fig_name))

    
    return sp.stats.ttest_ind(A, B, nan_policy='omit', equal_var=equal_var), fig
    


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


def plot_order(tr, iord, flux=None, xaxis=None, yaxis=None, show_slice=None, figsize=(16,4),
               xlabel='', ylabel='', cbar=False, clim=[None,None], xlim=None, ylim=None, 
               fontsize=12, title='', **kwargs):
    
    if flux is None:
        flux = tr.final
        
    if xaxis is None:
        xaxis = tr.wv[iord]
    if yaxis is None:
        yaxis = np.arange(tr.n_spec)
    
    fig = plt.figure(figsize=figsize)
    
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
        ax2.set_xlabel(xlabel, fontsize=fontsize)
    else:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    
    
        

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
                      good_rv_idx=0, switch=False,
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
                colors=['darkblue','dodgerblue','darkorange'], fig_name='', path_fig=None):

    ig, ax = plt.subplots(3,1, figsize=(9,8))

    # plt.figure(figsize=(8,3.5))

    for i,tr in enumerate(list_tr):
        ax[0].plot(tr.phase, tr.AM,'-', marker=markers[i], color=colors[i], label='Transit {}'.format(i+1))

    phase_t1 = np.min([tr.phase[tr.iIn[0]] for tr in list_tr])
    phase_t2 = np.min([tr.phase[tr.total[0]] for tr in list_tr])
    phase_t3 = np.max([tr.phase[tr.total[-1]] for tr in list_tr])
    phase_t4 = np.max([tr.phase[tr.iIn[-1]] for tr in list_tr])

    ax[0].axvspan(phase_t1, phase_t4, alpha=0.2, label='Ingress/Egress')
    ax[0].axvspan(phase_t2, phase_t3, alpha=0.2)
    ax[0].axvspan(phase_t2, phase_t2, alpha=0.4, label='Total Transit')

    ax[0].ylabel('Airmass', fontsize=16)
    ax[0].xlabel(r'Orbital phase ($\phi$)', fontsize=16)
    ax[0].legend(loc='upper left', fontsize=12)
    ax[0].tight_layout()

    # if path_fig is not None:
    #     plt.savefig(path_fig+'fig_airmass{}.pdf'.format(fig_name))

    # fig, ax = plt.subplots(2,1, figsize=(9,8))
    
    hband = a.bands(tr.wv,'h')[2:-2]
    
    for i,tr in enumerate(list_tr):
        ax[1].plot(np.mean(tr.wv,axis=-1).T, np.nanmean(tr.SNR,axis=0).T,
                   '-', marker=markers[i], color=colors[i], label='Transit {}'.format(i+1))


    ax[1].set_ylabel('Mean S/N\nper order', fontsize=16)
    ax[1].set_xlabel(r'Wavelength ($\mu$m)', fontsize=16)
    # ax[0].axvspan(np.mean(tr.wv,axis=-1)[28], np.mean(tr.wv,axis=-1)[36], alpha=0.2, color='darkorange',label='H-band')
    ax[1].legend(loc='upper left', fontsize=12) #, bbox_to_anchor=(0.9, 0.71)

    for i,tr in enumerate(list_tr):
        ax[2].plot(tr.phase, np.nanmean(tr.SNR[:, hband],axis=-1),'-', marker=markers[i], color=colors[i])


    ax[2].set_ylabel('Mean H-band S/N\nper exposure', fontsize=16)
    ax[2].set_xlabel(r'Orbital phase ($\phi$)', fontsize=16)

    ax[2].axvspan(phase_t1, phase_t4, alpha=0.2, label='Ingress/Egress')
    ax[2].axvspan(phase_t2, phase_t3, alpha=0.2)
    ax[2].axvspan(phase_t2, phase_t2, alpha=0.4, label='Total Transit')

    ax[2].legend(loc='best', fontsize=12) #, bbox_to_anchor=(0.9, 0.71)

    if path_fig is not None:
        plt.savefig(path_fig+'fig_SNR{}.pdf'.format(fig_name))


def plot_night_summary_NIRPS(list_tr, obs, markers=['o','s','d'], 
                colors=['darkblue','dodgerblue','darkorange'], fig_name='', path_fig=None):

    fig, ax = plt.subplots(6,1, figsize=(8,15))
    
    # plot mean s/n per order
    for i,tr in enumerate(list_tr):
        ax[0].plot(np.mean(tr.wv,axis=-1).T, np.nanmean(tr.SNR,axis=0).T,
                   '-', marker=markers[i], color=colors[i], label='Transit {}'.format(i+1))


    ax[0].set_ylabel('Mean S/N\nper order', fontsize=16)
    ax[0].set_xlabel(r'Wavelength ($\mu$m)', fontsize=16)
    ax[0].legend(loc='best', fontsize=12) #, bbox_to_anchor=(0.9, 0.71)

    # plot airmass
    for i,tr in enumerate(list_tr):
        ax[1].plot(tr.phase, tr.AM,'-', marker=markers[i], color=colors[i], label='Transit {}'.format(i+1))

    phase_t1 = np.min([tr.phase[tr.iIn[0]] for tr in list_tr])
    phase_t2 = np.min([tr.phase[tr.total[0]] for tr in list_tr])
    phase_t3 = np.max([tr.phase[tr.total[-1]] for tr in list_tr])
    phase_t4 = np.max([tr.phase[tr.iIn[-1]] for tr in list_tr])

    ax[1].axvspan(phase_t1, phase_t4, alpha=0.2, label='Ingress/Egress')
    ax[1].axvspan(phase_t2, phase_t3, alpha=0.2)
    ax[1].axvspan(phase_t2, phase_t2, alpha=0.4, label='Total Transit')

    ax[1].set_ylabel('Airmass', fontsize=16)
    # ax[1].set_xlabel(r'Orbital phase ($\phi$)', fontsize=16)
    ax[1].legend(loc='best', fontsize=12)
    

    # plot mean s/n per exposure in Y and H band
    hband = a.bands(tr.wv,'h')[2:-2]
    yband = a.bands(tr.wv,'y')[2:-2]
    
    for i,tr in enumerate(list_tr):
        ax[2].plot(tr.phase, np.nanmean(tr.SNR[:, hband],axis=-1),'-', marker=markers[i], color=colors[0], label = 'H-band')
        ax[2].plot(tr.phase, np.nanmean(tr.SNR[:, yband],axis=-1),'-', marker=markers[i], color='darkred', label = 'Y-band')

    ax[2].set_ylabel('Mean S/N\nper exposure', fontsize=16)
#     ax[2].set_xlabel(r'Orbital phase ($\phi$)', fontsize=16)

    ax[2].axvspan(phase_t1, phase_t4, alpha=0.2, label='Ingress/Egress')
    ax[2].axvspan(phase_t2, phase_t3, alpha=0.2)
    ax[2].axvspan(phase_t2, phase_t2, alpha=0.4, label='Total Transit')

    ax[2].legend(loc='best', fontsize=12, ncol = 2) #, bbox_to_anchor=(0.9, 0.71)
    
    # plot H2O telluric pre-clean exponent
    for i,tr in enumerate(list_tr):
        ax[3].plot(tr.phase, obs.headers_tellu.get_all('TLPEH2O')[0], '-', marker=markers[i], color=colors[0])
    
    ax[3].set_ylabel('Telluric exp. H2O', fontsize=16)
#     ax[3].set_xlabel(r'Orbital phase ($\phi$)', fontsize=16)

    ax[3].axvspan(phase_t1, phase_t4, alpha=0.2, label='Ingress/Egress')
    ax[3].axvspan(phase_t2, phase_t3, alpha=0.2)
    ax[3].axvspan(phase_t2, phase_t2, alpha=0.4, label='Total Transit')
    
    # plot other tellurice pre-clean exponents
    for i,tr in enumerate(list_tr):
        ax[4].plot(tr.phase, obs.headers_tellu.get_all('TLPEOTR')[0], '-', marker=markers[i], color=colors[i])
    
    ax[4].set_ylabel('Telluric exp. \nother species', fontsize=16)
    # ax[4].set_xlabel(r'Orbital phase ($\phi$)', fontsize=16)

    ax[4].axvspan(phase_t1, phase_t4, alpha=0.2, label='Ingress/Egress')
    ax[4].axvspan(phase_t2, phase_t3, alpha=0.2)
    ax[4].axvspan(phase_t2, phase_t2, alpha=0.4, label='Total Transit')

    start = np.array(obs.headers.get_all('HIERARCH ESO TEL AMBI FWHM START')[0])
    end = np.array(obs.headers.get_all('HIERARCH ESO TEL AMBI FWHM END')[0])
    mean_seeing = (start + end) / 2

    ax[5].plot(tr.phase, mean_seeing, '-', marker=markers[i], color = colors[i])
    ax[5].set_ylabel('Mean seeing', fontsize = 16)
    ax[5].set_xlabel(r'Orbital phase ($\phi$)', fontsize=16)
    ax[5].axvspan(phase_t1, phase_t4, alpha=0.2, label='Ingress/Egress')
    ax[5].axvspan(phase_t2, phase_t3, alpha=0.2)
    ax[5].axvspan(phase_t2, phase_t2, alpha=0.4, label='Total Transit')

    fig.tight_layout()

    if path_fig is not None:
        plt.savefig(path_fig+'night_summary{}.pdf'.format(fig_name), bbox_inches='tight')
        
            
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


## Plot profiles and spectra distributions from samples
def _get_fig_and_ax_inputs(fig, ax, nrows=1, ncols=1, **kwargs):
    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(nrows, ncols, **kwargs)
        else:
            ax = fig.gca()

    return fig, ax


def _get_idx_in_range(x_array, x_range):
    if x_range is None:
        idx = slice(None)
    else:
        cond = (x_range[0] <= x_array) & (x_array <= x_range[-1])
        idx, = np.nonzero(cond)

    return idx


def plot_p_profile_sample(pressures, sample_stats, line_color=None, region_color=None, p_range=(1e-6, 1e1), fig=None,
                          ax=None, alpha=0.2, tight_range=True, **kwargs):
    fig, ax = _get_fig_and_ax_inputs(fig, ax)

    idx = _get_idx_in_range(pressures, p_range)

    (line,) = ax.semilogy(sample_stats['median'][idx], pressures[idx], color=line_color, **kwargs)

    if region_color is None:
        region_color = line.get_color()

    for key in ['1-sig', '2-sig']:
        x1, x2 = sample_stats[key]
        ax.fill_betweenx(pressures[idx], x1[idx], x2[idx], color=region_color, alpha=alpha)

    if tight_range:
        ax.set_ylim(np.min(pressures[idx]), np.max(pressures[idx]))

    ylim = ax.get_ylim()
    if ylim[-1] > ylim[0]:
        ax.invert_yaxis()

    return fig, ax


def plot_tp_sample(pressures, temp_stats, line_color='forestgreen', region_color='limegreen', p_range=(1e-6, 1e1),
                   tight_range=True, fig=None, ax=None, alpha=0.5, **kwargs):
    fkwargs = dict(line_color=line_color, region_color=region_color, p_range=p_range, fig=fig, ax=ax, alpha=alpha,
                   tight_range=tight_range, **kwargs)
    fig, ax = plot_p_profile_sample(pressures, temp_stats, **fkwargs)

    ax.set_xlabel('Temperature [K]', fontsize=16)
    ax.set_ylabel('Pressure [bar]', fontsize=16)

    return fig, ax


def plot_spectra_sample(wave, spectra_stats, line_color='forestgreen', region_color='limegreen', wv_range=None,
                        scale_spec=1, fig=None, ax=None, show_2sig=True, **kwargs):

    if show_2sig:
        sigmas = ['1-sig', '2-sig']
    else:
        sigmas = ['1-sig']

    fig, ax = _get_fig_and_ax_inputs(fig, ax)

    idx = _get_idx_in_range(wave, wv_range)

    ax.plot(wave[idx], spectra_stats['median'][idx] * scale_spec, color=line_color, **kwargs)
    for key in ['1-sig', '2-sig']:
        y1, y2 = spectra_stats[key]
        plot_args = (wave[idx], y1[idx] * scale_spec, y2[idx] * scale_spec)
        ax.fill_between(*plot_args, color=region_color, alpha=0.5)

    return fig, ax


def plot_single_line_sample(wave, spectra_stats, centered=True, dv_units=False, sp_line_type='emission',
                            wv_range=None, **kwargs):
    """
    Plot a single line from a sample of spectra.
    Args:
        wave:
        spectra_stats:
        centered: bool
             If True, the line is centered at the maximum or minimum value of the median spectrum,
              depending on the `line_type` argument (absorption or emission line).
        dv_units:
        sp_line_type:
        wv_range:
        **kwargs:

    Returns:

    """
    # Initialize Center wavelength (will be an output of the function)
    wv_c = None

    if centered or dv_units:
        # In range
        idx = _get_idx_in_range(wave, wv_range)

        # Get index of max or min of the distributions (median, 1-sig, 2-sig)
        if sp_line_type == 'emission':
            _, idx_center = _get_stat_of_spectrum_distributions(np.max, spectra_stats, idx_range=idx)
        elif sp_line_type == 'absorption':
            _, idx_center = _get_stat_of_spectrum_distributions(np.min, spectra_stats, idx_range=idx)
        else:
            raise ValueError('`line_type` not valid.')

        # Center wavelengths
        wv_c = wave[idx_center]

        if centered:
            wave = wave - wv_c

            # Update wv_range
            if wv_range is not None:
                wv_range = (wv_range[0] - wv_c, wv_range[-1] - wv_c)

        if dv_units:
            wave = (const.c * wave / wv_c).to('km/s').value
            wv_range = [(const.c * wv_lim / wv_c).to('km/s').value for wv_lim in wv_range]
            xlabel = r"$\Delta$v [km/s]"
        else:
            xlabel = "Wavelength relative to line center [um]"

    fig, ax = plot_spectra_sample(wave, spectra_stats, wv_range=wv_range, **kwargs)
    ax.set_xlabel(xlabel)

    return fig, ax, wv_c

def _get_stat_of_spectrum_distributions(fct, spectra_stats, idx_range=None):
    if idx_range is None:
        idx_range = slice(None)

    results, index_list = [], []
    spectrum = spectra_stats['median']
    value = fct(spectrum[idx_range])
    idx = np.argmin(np.abs(spectrum - value))
    results.append(value)
    index_list.append(idx)

    for key in ['1-sig', '2-sig']:
        for spectrum in spectra_stats[key]:
            spectrum = spectrum
            value = fct(spectrum[idx_range])
            idx = np.argmin(np.abs(spectrum - value))
            results.append(value)
            index_list.append(idx)
    results = np.array(results)

    best_val = fct(results)
    idx_best = np.argmin(np.abs(results - best_val))
    idx_best = index_list[idx_best]

    return best_val, idx_best


def get_GTC_axes_idx(n_param):
    """
    Get the position of the axes of a corner plot for a given number of parameters.
    Args:
        n_param: integer
            Number of parameters in the corner plot.
    Returns:
        x-position: list with length == n_param
        y-position: list with length == n_param
        So if `ax` is the 2d array of axes, then
        `ax[x-position[0], y-position[0]]` gives the axe of the first parameter.
    """
    y_idx = []
    last_idx = 0
    for idx in np.arange(n_param - 1):
        y_idx.append(idx + last_idx)
        last_idx = y_idx[-1]

    x_idx = [i + last_idx for i in range(n_param - 1)]
    x_idx.append(x_idx[-1] + n_param)

    return x_idx, y_idx


def _assign_color_to_stat_key(keys, color_list):
    """Create a dictionary that assigns color to each keys and check that the number of color is sufficient.
    Returns a dictionnary to map each of these keys with a color."""

    color_dict = dict()

    # Get colors and assign one for each statistic in sample_stats (so each key)
    for idx_stat, key in enumerate(keys):
        try:
            color_dict[key] = color_list[idx_stat]
        except IndexError:
            raise IndexError(f"To many satistics to plot for available colors (length = {len(color_list)})")

    return color_dict


def plot_spectra_sample_GTC(wave, spectra_stats_list, colorsOrder=None, wv_range=None, scale_factor=1., fig=None,
                            ax=None, stats_keys=('1-sig', '2-sig'), **kwargs):
    """
    Plot intervals (1-sigma and 2-sigma) of a sample of spectra with pyGTC style.
    Many cases can be overlaid on the same plot.
    Args:
        wave: array or list of arrays
            Wavelengths of the spectra. If list, must have the same length as `spectra_stats_list`. Else, the same
            wavelength grid is used for all spectra.
        spectra_stats_list: list of dictionaries
            List of dictionaries containing the statistics of the spectra.
            Each dictionary must have the keys given by `stats_keys`.
            The length of the list is the number of cases to be overlaid on the same plot, the first one being on top.
        colorsOrder: list of strings
            List of colors to use for each case. If None, the default color order from pyGTC is used. Available colors
            are the same as pyGTC.
        wv_range: 2-tuple of floats
            Wavelength range to plot. If None, the full range is used.
        scale_factor: float
            Scale factor to apply to the spectra. Useful to change the units of the spectra.
        fig: figure object
            Figure object to use for the plot. If `ax` and `fig` are not specified, a new figure is created.
        ax: axes object
            Axes object to use for the plot. If `ax` and `fig` are not specified, a new figure is created.
        stats_keys: list of dict keys
            Keys of the dictionaries in `spectra_stats_list` that contain the statistics of the spectra to be plotted.
        **kwargs:
            Additional arguments to be passed to the plt.plot for the contour lines.

    Returns:
        fig, ax: figure and axes objects

    """

    fig, ax = _get_fig_and_ax_inputs(fig, ax)

    # Number of different cases for plot
    n_case = len(spectra_stats_list)

    # Use default color order from pyGTC
    if colorsOrder is None:
        colorsOrder = list(defaultColorsOrder[:n_case])

    # Make sure `wave` is a list with the same length as `spectra_stats_list`
    if len(wave) != n_case:
        wave = [wave for _ in range(n_case)]

    # Get idx for plot for each wave
    spec_idx_list = [_get_idx_in_range(wv, wv_range) for wv in wave]

    # Get color for each stats
    color_region = [_assign_color_to_stat_key(stats_keys, colorsDict[cs]) for cs in colorsOrder]

    # Plot region before the lines (so that the lines are on top)
    # The loop is reversed so that the first case is on top
    for i_case in reversed(range(n_case)):
        color_dict = color_region[i_case]
        wv = wave[i_case]
        idx_plt = spec_idx_list[i_case]
        # Plot regions corresponding statistics (1-sigma, 2-sigma, etc.)
        for key in reversed(stats_keys):
            y1, y2 = spectra_stats_list[i_case][key] * scale_factor
            ax.fill_between(wv[idx_plt], y1[idx_plt], y2[idx_plt], color=color_dict[key])

    # Plot contours. Skip the first case (which is the one on top).
    for i_case in reversed(range(1, n_case)):
        color_dict = color_region[i_case]
        wv = wave[i_case]
        idx_plt = spec_idx_list[i_case]
        # Plot contours corresponding statistics (1-sigma, 2-sigma, etc.)
        for key in reversed(stats_keys):
            y1, y2 = spectra_stats_list[i_case][key] * scale_factor
            ax.plot(wv[idx_plt], y1[idx_plt], color=color_dict[key], **kwargs)
            ax.plot(wv[idx_plt], y2[idx_plt], color=color_dict[key], **kwargs)

    return fig, ax


def plot_x_y_position(x, y, x_hole=0.2, y_hole=0.2, ax=None, fig=None, label=None,
                      vlines=True, hlines=True, linestyle='--', color='grey', **kwargs):
    """Plot horizontal and vertical line at a given position.
    Leave a hole at this position so the lines don't overplot at the wanted position."""
    
    fig, ax = _get_fig_and_ax_inputs(fig, ax)
    
    # x_hole is the size of the region around x where the vertical line is not shown.
    # y_hole is the size of the region around y where the horizontal line is not shown.
    # x_hole and y_hole are in fraction of the x and y range.
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_hole = x_hole * (x_max - x_min)
    y_hole = y_hole * (y_max - y_min)
    
    # Plot vertical and horizontal lines around the hole.
    kwargs['linestyle'] = linestyle
    kwargs['color'] = color
    if vlines:
        # Add label only to the first vertical line, so that it appears in the legend only once.
        ax.vlines(x, y_min, y - y_hole, label=label, **kwargs)
        ax.vlines(x, y + y_hole, y_max, **kwargs)
    if hlines:
        ax.hlines(y, x_min, x - x_hole, **kwargs)
        ax.hlines(y, x + x_hole, x_max, **kwargs)
    
    return fig, ax
    
    
def plot_tp_profiles_combined(n_draw, chains=None, yaml_file_list=None, get_tp_from_param=None,
                              prob_values=None, p_range=None, fig=None, ax=None,
                              tight_range=None, colorsOrder=None, log_level='WARNING', retrieval_obj=None):
    """
    Plot TP profile statistics from different walker chains.
    Args:
        n_draw: int
            Number of draws to take from each chain.
        chains: list of arrays
            List of chains. Each chain is a 2D array with shape (n_steps * n_wakers, n_params).
        yaml_file_list: list of str
            List of yaml files to setup the retrieval object. They must match the walker chains.
        get_tp_from_param: function
            Function to get the TP profile from the parameters. It must take the parameters as input and return the
            TP profile.
        prob_values: list of floats
             Percentiles to compute associated to `key_names`. Needs to have the same length as `key_names`.
             Default is (0.68, 0.954, 0.997).
        p_range: 2-tuple of floats
            Pressure range to plot. If None, the full range is used.
        fig: figure object
            Figure object to use for the plot. If `ax` and `fig` are not specified, a new figure is created.
        ax: axes object
            Axes object to use for the plot. If `ax` and `fig` are not specified, a new figure is created.
        tight_range: bool
            If True, the y-axis is set to the minimum and maximum pressure of all chains.
        colorsOrder: list of strings
            List of colors sets (from pyGTC) to use for each chain. If None, the default color order
            from pyGTC is used. Available colors are the same as pyGTC.

    Returns:
        fig, ax: figure and axes objects
    """
    
    # Import retrieval.py if no retrieval object given
    if retrieval_obj is None:
        import retrieval as retrieval_obj
    
    # Set log levels
    imported_libs = [retrieval_obj, ru]
    save_level = [im_lib.log.level for im_lib in imported_libs]
    for im_lib in imported_libs:
        im_lib.log.setLevel(log_level)

    # Number of chains
    n_chains = len(chains)

    # Find confidence intervals for each chains
    tp_stats_list = list()
    pressure_list = list()
    pressure_idx_list = list()
    for ch, yaml_file in zip(chains, yaml_file_list):
        retrieval_obj.setup_retrieval(yaml_file)
        n_draw_ch = np.min([n_draw, ch.shape[0]])
        profile_sample, pressures = ru.draw_tp_profiles_from_sample(n_draw_ch, ch, retrieval_obj=retrieval_obj)
        tp_stats = ru.get_stats_from_profile(profile_sample, prob_values=prob_values)
        # pressures idx for plot range
        idx_p = _get_idx_in_range(pressures, p_range)
        # Remove median from statistics
        del tp_stats['median']
        tp_stats_list.append(tp_stats)
        pressure_list.append(pressures)
        pressure_idx_list.append(idx_p)


    # Plot starts here

    # Init figure and axes if needed
    fig, ax = _get_fig_and_ax_inputs(fig, ax)

    # Get colors and assign for each statistic in sample_stats (so each key)
    color_region = [dict() for _ in range(n_chains)]
    # and assign for each statistic in sample_stats (so each key)
    for idx_ch, cs in enumerate(colorsOrder):
        stats = tp_stats_list[idx_ch]
        for idx_stat, key in enumerate(stats):
            try:
                color_region[idx_ch][key] = colorsDict[cs][idx_stat]
            except IndexError:
                raise IndexError(f"To many satistics to plot for available colors (length = {len(defaultColorsOrder)})")


    for idx_ch in reversed(range(n_chains)):
        stats = tp_stats_list[idx_ch]
        color_ch = color_region[idx_ch]
        pressures = pressure_list[idx_ch]
        idx_p = pressure_idx_list[idx_ch]
        for key, (x1, x2) in reversed(stats.items()):
            ax.fill_betweenx(pressures[idx_p], x1[idx_p], x2[idx_p], color=color_ch[key])

    for idx_ch in reversed(range(1, n_chains)):
        stats = tp_stats_list[idx_ch]
        color_ch = color_region[idx_ch]
        pressures = pressure_list[idx_ch]
        idx_p = pressure_idx_list[idx_ch]
        for key, (x1, x2) in reversed(stats.items()):
            ax.plot(x1[idx_p], pressures[idx_p], '-', color=color_ch[key])
            ax.plot(x2[idx_p], pressures[idx_p], '-', color=color_ch[key])

    if tight_range:
        all_p = [pressures[idx_p] for pressures, idx_p in zip(pressure_list, pressure_idx_list)]
        ax.set_ylim(np.min(all_p), np.max(all_p))

    ax.set_yscale('log')    

    ylim = ax.get_ylim()
    if ylim[-1] > ylim[0]:
        ax.invert_yaxis()


    ax.set_xlabel('Temperature [K]', fontsize=16)
    ax.set_ylabel('Pressure [bar]', fontsize=16)
    
    # set the log level to what it was
    for im_lib, level in zip(imported_libs, save_level):
        im_lib.log.setLevel(level)

    return fig, ax


def plot_mol_features_labels(mol_name, feature_list, y_feat, color=None,
                             dy_feat=0.02, txt_shift=None,
                             fig=None, ax=None,
                             plot_kwargs=None, text_kwargs=None):
    """Plot position of molecular bands.
    All values in y (`y_feat`, `dy_feat`, `txt_shift`) are in Axes coordinates,
    i.e. from 0 (bottom of axes) to 1 (top of axes)."""
    
    fig, ax = _get_fig_and_ax_inputs(fig, ax)
    
    if plot_kwargs is None:
        plot_kwargs = dict()
        
    if text_kwargs is None:
        text_kwargs = dict()  
        
    if txt_shift is None:
        txt_shift = 0.5 * dy_feat
        
    # Convert values in Axes scale to Data scale
    ylim = ax.get_ylim()
    dylim = ylim[-1] - ylim[0]
    txt_shift_y = dylim * txt_shift
    
    xlim = ax.get_xlim()
    dx = np.diff(xlim)
    txt_shift_x = dx * txt_shift

    # Consider y in  axes coordinates and x in Data coordinates 
    trans = ax.transAxes
    
    # Inverse transormation (puts display coordinates back to Data coordinates)
    inv = ax.transData.inverted()
    
    # y values for the plot (in Axes coordinates)
    y_axes_coords = np.full(3, y_feat)
    y_axes_coords[0] += dy_feat
    
    # Convert in Data coord
    y_plot = ylim[0] + y_axes_coords * dylim
    
    # Iterate over the feature list
    for x_feat in feature_list:

        x_plot = np.append(x_feat[0], x_feat)  # 3 points instead of 2
        
        # Plot
        ax.plot(x_plot, y_plot, color=color, **plot_kwargs)
        if color is None:
            color = ax.get_lines()[-1].get_color()
        
        x_in_plot = (xlim[0] <= x_plot[1]) & (x_plot[1] < xlim[-1])
        if x_in_plot:
            ax.text(x_plot[1] + txt_shift_x, y_plot[1] + txt_shift_y, mol_name, **text_kwargs)
            
    # Reset x and y limits
    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
        
    return fig, ax


def scatterplot_logl(flatten_sample, flatten_logl, n_max_pts=10000, tight_ylim=True, alpha=0.1,
                     color=None, fig=None, ax=None, ylabels=None, **kwargs):
    """
    Generate scatter plots of samples against their log-likelihood values.

    This function creates scatter plots for each parameter in the `flatten_sample` array against the corresponding
    log-likelihood values in `flatten_logl`. It supports plotting a maximum number of points to avoid overplotting.

    Parameters
    ----------
    flatten_sample : ndarray
        A 2D array of shape (n_samples, n_params) containing the sample values for each parameter.
    flatten_logl : ndarray
        A 1D array of length n_samples containing the log-likelihood values corresponding to each sample.
    n_max_pts : int, optional
        The maximum number of points to display in the scatter plot. If the number of samples exceeds this value,
        a random subset of `n_max_pts` samples will be selected for plotting. Default is 10000.
    tight_ylim : bool, optional
        If True, the y-axis limits are set to tightly encompass the range of log-likelihood values, with a small
        margin added. If False, the y-axis limits are determined automatically. Default is True.
    alpha : float, optional
        The alpha blending value, between 0 (transparent) and 1 (opaque), for the points in the scatter plot.
        Default is 0.1.
    color : str or None, optional
        The color of the points in the scatter plot. If None, the default color cycle is used. Default is None.
    fig : Figure or None, optional
        An existing matplotlib Figure object to plot on. If None, a new figure is created. Default is None.
    ax : Axes or None, optional
        An array of matplotlib Axes objects to plot on. If None, new axes are created. Default is None.
    ylabels : list of str or None, optional
        A list of labels for the y-axis, one for each parameter. If None, parameter indices are used as labels.
        Default is None.
    **kwargs
        Additional keyword arguments are passed to the `plot` function.

    Returns
    -------
    fig : Figure
        The matplotlib Figure object containing the plot.
    ax : Axes
        An array of matplotlib Axes objects containing the scatter plots, one for each parameter.

    Notes
    -----
    This function is designed to visualize the distribution of samples and their corresponding log-likelihood values
    in parameter space. It is particularly useful for examining the results of sampling algorithms in the context
    of Bayesian inference or optimization problems.
    """

    # Init figure and ax if not given
    n_param = flatten_sample.shape[-1]
    fig, ax =  _get_fig_and_ax_inputs(fig, ax, n_param, 1, figsize=(6, 3 * n_param))
    
    # ylabels
    if ylabels is None:
        ylabels = list(range(n_param))

    # Make sure the inputs are sorted with respect to logl
    sort_idx = np.argsort(flatten_logl)
    flatten_sample = flatten_sample[sort_idx]
    flatten_logl = flatten_logl[sort_idx]

    # Take random integers (no repeated value)
    rand_idx = rng.permutation(range(flatten_logl.shape[-1]))[:n_max_pts]

    # ylimits
    if tight_ylim:
        ymin, ymax = (np.quantile(flatten_logl, 0.1), np.max(flatten_logl))
        dy = ymax - ymin
        ylim = (ymin - 0.1 * dy, ymax + 0.1 * dy)
    else:
        ylim = (None, None)

    for i_param, ax_i in enumerate(ax):
        (lines,) = ax_i.plot(flatten_sample[rand_idx, i_param], flatten_logl[rand_idx], '.', 
                  color=color, alpha=alpha, **kwargs)
        color = lines.get_color()

        ax_i.set_ylim(*ylim)
        ax_i.set_xlabel(ylabels[i_param])

        # Add vertical lines for best logl and median
        ax_i.axvline(flatten_sample[-1, i_param], linestyle="-", color=color, label='best')
        ax_i.axvline(flatten_sample[len(flatten_sample)//2, i_param], linestyle="-.",
                     color=color, label='median')
        ax_i.axhline(flatten_logl[len(flatten_sample)//2], linestyle="-.", color=color)
        ax_i.legend()

    plt.tight_layout()

    return fig, ax


# =============================================================================
# logl_grid — trailing plot and Kp-Vsys map helpers
# =============================================================================

def oversample_image(image, scale_factor, x_coords=None, y_coords=None, method='cubic'):
    """Oversample a 2D image via cubic spline interpolation.

    Parameters
    ----------
    image : (ny, nx) ndarray
    scale_factor : float
    x_coords, y_coords : 1D arrays, optional
        Native pixel coordinates. Defaults to integer indices.
    method : str
        Passed to ``RegularGridInterpolator``.

    Returns
    -------
    oversampled_image, new_x, new_y
    """
    from scipy.interpolate import RegularGridInterpolator

    ny, nx = image.shape
    if x_coords is None:
        x_coords = np.arange(nx)
    if y_coords is None:
        y_coords = np.arange(ny)

    new_x = np.linspace(x_coords[0], x_coords[-1], int(nx * scale_factor))
    new_y = np.linspace(y_coords[0], y_coords[-1], int(ny * scale_factor))

    interpolator = RegularGridInterpolator((y_coords, x_coords), image, method=method)
    new_x_grid, new_y_grid = np.meshgrid(new_x, new_y)
    new_coords = np.array([new_y_grid.ravel(), new_x_grid.ravel()]).T
    oversampled = interpolator(new_coords).reshape(new_y_grid.shape)

    return oversampled, new_x, new_y


def _find_sequence_gaps(sequence_array, min_diff=1., prominence=0.5, find_peaks_kwargs=None):
    """Return indices and prominences of gaps in a monotone sequence."""
    from scipy.signal import find_peaks as _find_peaks

    diff = np.diff(sequence_array)
    if min_diff is not None:
        diff = diff * min_diff / diff.min()
    idx_gaps, props = _find_peaks(diff, prominence=prominence,
                                  **(find_peaks_kwargs or {}))
    return np.array(idx_gaps), props


def pcolormesh_ts(x_plot, y_plot, z_map, fig=None, ax=None, debug_plot=False, **kwargs):
    """pcolormesh for a time-series map with gaps automatically filled.

    Gaps in ``y_plot`` (e.g. between two visits) are detected and filled with
    NaN rows so that ``pcolormesh`` does not stretch across the gap.

    Parameters
    ----------
    x_plot : 1D array   (e.g. RV axis)
    y_plot : 1D array   (e.g. orbital phase, monotone)
    z_map  : (n_time, n_rv) array
    fig, ax : optional
    **kwargs : passed to ``ax.pcolormesh``

    Returns
    -------
    pcolormesh output
    """
    idx_gaps, props = _find_sequence_gaps(y_plot)
    gap_len = np.round(props['prominences']).astype(int) - 1

    dy = np.diff(y_plot)
    fill_values = [
        y_plot[idx] + (n + 1) * dy[idx] / (g + 1)
        for idx, g in zip(idx_gaps, gap_len)
        for n in range(g)
    ]
    fill_idx = np.repeat(idx_gaps + 1, gap_len)
    y_filled = np.insert(y_plot, fill_idx, fill_values)
    z_filled = np.insert(z_map, fill_idx, np.nan, axis=0)

    if debug_plot:
        y_nan = np.insert(y_plot, np.repeat(idx_gaps + 1, gap_len),
                          np.full(int(gap_len.sum()), np.nan))
        fig_d, ax_d = plt.subplots()
        ax_d.plot(y_nan, 'o', label='with gaps')
        ax_d.plot(y_filled, '.', label='filled')
        ax_d.legend()

    fig, ax = _get_fig_and_ax_inputs(fig, ax)
    out = ax.pcolormesh(x_plot, y_filled, z_filled, **kwargs)
    return out


def plot_peak_lightcurve(sequence_map, rv_array, noise_rv_limits, peak_rv_limits,
                         box_width=5, t_val=None, fig=None, ax=None,
                         orientation='horizontal'):
    """Plot the peak and noise lightcurves from a (n_rv, n_time) sequence map.

    Parameters
    ----------
    sequence_map : (n_rv, n_time) array
    rv_array : 1D array
    noise_rv_limits, peak_rv_limits : (low, high) tuples in km/s
    box_width : int  — boxcar smoothing kernel width
    t_val : 1D array, optional  — time/phase axis (default: integer index)
    orientation : 'horizontal' or 'vertical'
    """
    from astropy.convolution import convolve, Box1DKernel

    fig, ax = _get_fig_and_ax_inputs(fig, ax)
    if t_val is None:
        t_val = np.arange(sequence_map.shape[-1])

    box_ker = Box1DKernel(box_width)
    is_out = (rv_array < noise_rv_limits[0]) | (noise_rv_limits[-1] < rv_array)
    is_in = (peak_rv_limits[0] < rv_array) & (rv_array < peak_rv_limits[-1])

    _C_PEAK  = '#0072B2'  # Wong blue  — S/N signal
    _C_NOISE = '#AAAAAA'  # neutral gray — noise baseline

    peak_seq = np.ma.mean(sequence_map[is_in, :], axis=0).squeeze()
    peak_conv = convolve(peak_seq, box_ker, boundary='fill',
                         preserve_nan=True, fill_value=np.nan)
    noise_seq = np.ma.mean(sequence_map[is_out, :], axis=0).squeeze()

    if orientation == 'horizontal':
        ax.plot(t_val, peak_seq, 'o', markersize=5, color=_C_PEAK, zorder=3)
        ax.plot(t_val, peak_conv, color=_C_PEAK, alpha=0.55, lw=1.5)
        ax.plot(t_val, noise_seq, '.', color=_C_NOISE, ms=4)
        ax.axhline(0, linestyle='--', color=_C_NOISE, lw=0.8)
    elif orientation == 'vertical':
        ax.plot(peak_seq, t_val, 'o', markersize=5, color=_C_PEAK, zorder=3)
        ax.plot(peak_conv, t_val, color=_C_PEAK, alpha=0.55, lw=1.5)
        ax.plot(noise_seq, t_val, '.', color=_C_NOISE, ms=4)
        ax.axvline(0, linestyle='--', color=_C_NOISE, lw=0.8)
    else:
        raise ValueError(f"`orientation` must be 'horizontal' or 'vertical', got {orientation!r}")

    return fig, ax


def get_contours_posterior(post_grid_norm, dn, lvls=(0.39, 0.86, 0.99), renormalize=True):
    """Find posterior contour levels corresponding to given probability masses.

    Parameters
    ----------
    post_grid_norm : (n1, n2) array  — normalised posterior (linear, not log)
    dn : sequence of floats  — grid spacings (one per axis)
    lvls : sequence of floats  — probability masses (e.g. 0.6827 for 1-sigma)
    renormalize : bool  — renormalise cumulative sum to [0, 1]

    Returns
    -------
    lvl_post : posterior values at the requested contour levels
    lvls     : the requested probability masses
    """
    import warnings
    post_flat = post_grid_norm.flatten()
    post_sorted = np.sort(post_flat)[::-1]
    csum = np.cumsum(post_sorted)
    for dn_i in dn:
        csum *= dn_i
    if renormalize:
        csum /= csum[-1]
    lvls = np.array(lvls)
    lvl_post = post_sorted[np.sum(csum < lvls[:, None], axis=1) - 1]

    # Warn when a contour is so tight it encloses only a handful of pixels —
    # this usually means the posterior peak is sub-resolution (narrower than
    # one grid cell) and the sigma level is not reliable.
    for lvl, lv in zip(lvl_post, lvls):
        n_pix = int(np.sum(post_grid_norm >= lvl))
        if n_pix <= 4:
            warnings.warn(
                f'Contour at probability mass {lv:.4f} encloses only {n_pix} pixel(s). '
                'The posterior peak is likely sub-resolution; consider increasing '
                'the oversample factor in compute_kpvsys_posterior.',
                UserWarning, stacklevel=2,
            )
    return lvl_post, lvls


def plot_trailing_map(logl_map_norm, rv_array, phase, logl_1d_norm,
                      noise_rv_limits, peak_rv_limits,
                      logl_1d_norm_all=None, rv_expected=0.,
                      phase_contacts=None, contrib=None,
                      figsize=(8, 4), fig=None, save_path=None):
    """Three-panel trailing plot: phase×RV map + 1D logL profile + peak lightcurve.

    Parameters
    ----------
    logl_map_norm : (n_rv, n_exp) array
        Normalised logL map at a fixed Kp, already baseline-subtracted.
    rv_array : (n_rv,) array
    phase : (n_exp,) array
    logl_1d_norm : (n_rv,) array
        logL summed over out-of-eclipse exposures, normalised.
    noise_rv_limits, peak_rv_limits : (low, high) km/s
    logl_1d_norm_all : (n_rv,) array, optional
        Combined logL over all visits (shown in grey behind logl_1d_norm).
    rv_expected : float
        Expected v_sys for contact-point markers.
    phase_contacts : dict, optional
        Contact phases to mark: keys '1_4' and/or '2_3', values are lists of phases.
    contrib : (n_exp,) array, optional
        Per-exposure fractional contribution to the combined logL peak:
        ``ΔlogL_i(vsys_peak) / sum_j(ΔlogL_j)``.  When provided, plotted on a
        secondary x-axis (top) in the side panel in orange.  Compute in the
        notebook as::

            i_vsys = np.argmin(np.abs(lg.vsys_axis - vsys_peak))
            delta = logl_map_ts[i_vsys, :] - np.ma.median(logl_map_ts[is_out_rv, :], axis=0)
            contrib = delta / float(np.sum(delta[idx_signal]))

    figsize : tuple
    save_path : str or Path, optional
        If given, save the figure to this path.

    Returns
    -------
    fig, (ax_map, ax_bottom, ax_side)
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig = plt.figure(figsize=figsize)
    ax_map = fig.gca()
    divider = make_axes_locatable(ax_map)
    ax_bottom = divider.append_axes('bottom', size='20%', pad=0.07)
    ax_side = divider.append_axes('right', size='20%', pad=0.07)

    # --- Main map ---
    imgrid = pcolormesh_ts(rv_array, phase, logl_map_norm.T, ax=ax_map)
    ax_map.set_facecolor('slategray')

    # --- Side panel: peak lightcurve ---
    plot_peak_lightcurve(logl_map_norm, rv_array, noise_rv_limits, peak_rv_limits,
                         t_val=phase, box_width=5, ax=ax_side, orientation='vertical')

    # Overlay per-exposure fractional contribution on a secondary x-axis
    _C_CONTRIB = '#D55E00'  # Wong vermillion
    if contrib is not None:
        ax_contrib = ax_side.twiny()
        ax_contrib.plot(contrib, phase, 's', markersize=4, color=_C_CONTRIB, zorder=3)
        ax_contrib.axvline(0, linestyle='--', color=_C_CONTRIB, alpha=0.35, lw=0.8)
        ax_contrib.set_xlabel(r'$\Delta\mathcal{L}_i\,/\,\mathcal{L}_\mathrm{comb}$',
                              fontsize=8, color=_C_CONTRIB)
        ax_contrib.tick_params(axis='x', colors=_C_CONTRIB, labelsize=7)

    # --- Bottom panel: 1D logL profile ---
    if logl_1d_norm_all is not None:
        ax_bottom.plot(rv_array, logl_1d_norm_all, color=(0.4, 0.4, 0.4))
    ax_bottom.plot(rv_array, logl_1d_norm, 'k')

    # --- Reference lines ---
    for ax_i in [ax_map, ax_bottom]:
        ax_i.axvline(noise_rv_limits[0], linestyle=':', color='lightgray' if ax_i is ax_map else 'black')
        ax_i.axvline(noise_rv_limits[1], linestyle=':', color='lightgray' if ax_i is ax_map else 'black')

    # --- Contact-point markers ---
    if phase_contacts is not None:
        ylim = ax_map.get_ylim()
        styles = {'1_4': '-', '2_3': '-.'}
        for key, phases in phase_contacts.items():
            ls = styles.get(key, '--')
            for ph in phases:
                if ylim[0] <= ph <= ylim[1]:
                    plot_x_y_position(rv_expected, ph, x_hole=0.1,
                                      linestyle=ls, color='lightgray',
                                      ax=ax_map, vlines=False)
                    ax_side.axhline(ph, linestyle=ls, color='lightgray', linewidth=0.8)

    # --- Colorbar ---
    cax_list = []
    for _ in [ax_map, ax_side]:
        cax_list.append(divider.append_axes('top', size='5%', pad=0.05))
    cax_list[1].axis('off')
    fig.colorbar(imgrid, ax=ax_map, cax=cax_list[0], orientation='horizontal')
    cax_list[0].xaxis.set_ticks_position('top')
    cax_list[0].xaxis.set_label_position('top')
    cax_list[0].set_xlabel('S/N', fontsize=12)

    # --- Labels and layout ---
    ylim = ax_map.get_ylim()
    ax_side.set_yticklabels([])
    ax_side.set_ylim(ylim)
    ax_map.set_xticks([])
    ax_bottom.set_xlim(ax_map.get_xlim())
    ax_bottom.set_xlabel(r'$v_{\rm rad}$ (km s$^{-1}$)', fontsize=16)
    ax_bottom.set_ylabel('S/N')
    ax_map.set_ylabel('Orbital Phase', fontsize=14)
    ax_side.set_xlabel('S/N')

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')

    return fig, (ax_map, ax_bottom, ax_side)


def sigma2percent_2d(sigma):
    """Probability mass enclosed by an n-sigma ellipse in 2D.

    Uses the chi²(2) distribution: P(chi²(2) < sigma²) = 1 − exp(−sigma²/2).
    This differs from the 1D Gaussian values (erf(sigma/√2)).

    Parameters
    ----------
    sigma : float or array-like

    Returns
    -------
    float or array — probability mass in [0, 1)
    """
    return 1.0 - np.exp(-0.5 * np.asarray(sigma) ** 2)


def plot_2d_map(map_2d, x_axis, y_axis, margin_x=None, margin_y=None,
                x_label=r'$v_{\rm sys}$ (km s$^{-1}$)',
                y_label=r'$K_{\rm P}$ (km s$^{-1}$)',
                levels=None, level_style='contours', crosshair_hole=0.03,
                mark_peak=False, peak_hole=0.03,
                x_lim=None, y_lim=None, scale='linear',
                cbar_label=None, figsize=(6, 6), save_path=None):
    """Generic 2D (x, y) map: pcolormesh + optional 1D side panels + optional
    iso-value contours/crosshairs + optional peak marker.

    This knows nothing about confidence regions or cumulative probability --
    ``levels`` are plain VALUE thresholds in ``map_2d``'s own units (e.g.
    already-in-sigma values for an empirical or likelihood-ratio sigma map).
    See ``plot_posterior_2d`` for a thin wrapper that derives such
    thresholds from cumulative probability mass instead, for a genuine
    (normalizable) posterior -- it calls this function for everything else.

    Works equally for a strictly-positive posterior (``scale='log'``) and a
    signed map like a raw CCF or a sigma map (``scale='linear'`` -- log
    doesn't apply to negative values).

    Layout (when ``margin_x``/``margin_y`` are both given; otherwise just
    the 2D map panel)::

        ┌─────────────┐ ┌───┐
        │  2D map     │ │ y │  ← right panel: y-axis marginal (x = prob, y = y_axis)
        └─────────────┘ └───┘
        └─────────────┘        ← bottom panel: x-axis marginal (x = x_axis, y = prob)

    Parameters
    ----------
    map_2d : (n_x, n_y) array — any 2D map (posterior, CCF, sigma map, ...).
        May be a masked array; non-finite/masked entries are shown blank and
        excluded from the peak search.
    x_axis : (n_x,) array
    y_axis : (n_y,) array
    margin_x : (n_x,) array, optional — 1D curve for the bottom panel (a true
        marginal, a slice through ``map_2d`` at the peak, or anything else).
        No side panels are created unless BOTH ``margin_x`` and ``margin_y``
        are given.
    margin_y : (n_y,) array, optional — 1D curve for the right panel.
    x_label, y_label : str — axis labels
    levels : 1D array, optional — value thresholds in ``map_2d``'s own units
        (not probability masses — see ``plot_posterior_2d`` for that).
        Order doesn't matter, sorted internally.
    level_style : {'contours', 'crosshairs', 'none'}
        * ``'contours'`` — iso-value contours on the 2D map + matching
          position lines on the marginal panels (if present).
        * ``'crosshairs'`` — crosshair lines at the outermost (smallest)
          level's boundary, through the peak, mirrored on the marginal
          panels (if present).
        * ``'none'`` — no level indicators.
    crosshair_hole : float — fractional gap in the level-boundary crosshair
        lines (default 0.03).
    mark_peak : bool — if True, mark ``map_2d``'s maximum with its own small
        crosshair, independent of ``levels``/``level_style``. Default False.
    peak_hole : float — fractional gap in the peak-marker crosshair (default 0.03).
    x_lim, y_lim : (float, float), optional — display range limits.
    scale : {'linear', 'log'}
        Colour scale for the 2D map and the 1D panels' value axis. 'log'
        assumes a strictly positive map (e.g. a posterior) -- use 'linear'
        for anything signed (CCF, sigma map).
    cbar_label : str, optional — colourbar label. Default: no label.
    figsize : tuple
    save_path : str or Path, optional

    Returns
    -------
    fig, axes
        ``axes`` is ``(ax_map, ax_y, ax_x)`` when ``margin_x``/``margin_y``
        are both given (``ax_y`` = right panel, ``ax_x`` = bottom panel), or
        just ``ax_map`` otherwise.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    has_panels = margin_x is not None and margin_y is not None

    # --- Crop axes ---
    if x_lim is not None:
        mask = (x_axis >= x_lim[0]) & (x_axis <= x_lim[1])
        x_axis = x_axis[mask]
        map_2d = map_2d[mask, :]
        if has_panels:
            margin_x = margin_x[mask]
    if y_lim is not None:
        mask = (y_axis >= y_lim[0]) & (y_axis <= y_lim[1])
        y_axis = y_axis[mask]
        map_2d = map_2d[:, mask]
        if has_panels:
            margin_y = margin_y[mask]

    # --- Figure layout ---
    fig = plt.figure(figsize=figsize)
    ax_map = fig.gca()
    if has_panels:
        divider = make_axes_locatable(ax_map)
        ax_y = divider.append_axes('right', size='20%', pad=0.05)
        ax_x = divider.append_axes('bottom', size='20%', pad=0.07)

    # --- Main map ---
    norm = 'log' if scale == 'log' else None
    map_plot = np.ma.filled(np.ma.masked_invalid(map_2d), np.nan)
    imgrid = ax_map.pcolormesh(x_axis, y_axis, map_plot.T, norm=norm)

    # Peak location, needed by both level_style='crosshairs' and mark_peak --
    # masked/non-finite entries are excluded so they can never be "the peak".
    map_filled = np.ma.filled(np.ma.masked_invalid(map_2d), -np.inf)
    max_ind = np.unravel_index(np.argmax(map_filled), map_filled.shape)
    x_peak, y_peak = x_axis[max_ind[0]], y_axis[max_ind[1]]

    # --- Level indicators (contours / crosshairs at given VALUE thresholds) ---
    if levels is not None:
        levels_sorted = np.sort(np.asarray(levels, dtype=float))

        if level_style == 'contours':
            ax_map.contour(x_axis, y_axis, map_2d.T,
                           levels=levels_sorted, colors='w', linewidths=0.8)

        elif level_style == 'crosshairs':
            # Outermost = smallest threshold -> largest enclosed region, for
            # ANY monotonic map (superlevel sets shrink as the threshold grows).
            outermost_lvl = levels_sorted[0]
            i_lvl, j_lvl = np.nonzero(map_2d >= outermost_lvl)
            x_lo, x_hi = x_axis[i_lvl.min()], x_axis[i_lvl.max()]
            y_lo, y_hi = y_axis[j_lvl.min()], y_axis[j_lvl.max()]
            hole = crosshair_hole
            for xv in (x_lo, x_hi):
                plot_x_y_position(xv, y_peak, x_hole=hole, y_hole=hole,
                                  ax=ax_map, hlines=False)
            for yv in (y_lo, y_hi):
                plot_x_y_position(x_peak, yv, x_hole=hole, y_hole=hole,
                                  ax=ax_map, vlines=False)
            if has_panels:
                ax_x.axvline(x_lo, color='gray', linestyle='--')
                ax_x.axvline(x_hi, color='gray', linestyle='--')
                ax_y.axhline(y_lo, color='gray', linestyle='--')
                ax_y.axhline(y_hi, color='gray', linestyle='--')

        # --- Level position lines on marginals (contours mode) ---
        # For each level, project the enclosed region onto each axis: draw
        # lines at the min/max x (or y) that belong to the region. This
        # guarantees the marginal lines stay aligned with the 2D contours.
        if level_style == 'contours' and has_panels:
            for lvl in levels_sorted:
                i_above, j_above = np.nonzero(map_2d >= lvl)
                if not len(i_above):
                    continue
                x_lo, x_hi = x_axis[i_above.min()], x_axis[i_above.max()]
                y_lo, y_hi = y_axis[j_above.min()], y_axis[j_above.max()]
                ax_x.axvline(x_lo, linestyle=':', color='k', alpha=0.7)
                ax_x.axvline(x_hi, linestyle=':', color='k', alpha=0.7)
                ax_y.axhline(y_lo, linestyle=':', color='k', alpha=0.7)
                ax_y.axhline(y_hi, linestyle=':', color='k', alpha=0.7)

    # --- Simple peak marker (independent of levels/level_style) ---
    if mark_peak:
        plot_x_y_position(x_peak, y_peak, x_hole=peak_hole, y_hole=peak_hole,
                          ax=ax_map, color='cyan')

    # --- Marginal panels ---
    if has_panels:
        # Clip zeros before log-scale plotting to avoid blank axes.
        def _floor(arr):
            pos = arr[arr > 0]
            return float(pos.min()) * 1e-3 if len(pos) else 1e-300

        if scale == 'log':
            fx, fy = _floor(margin_x), _floor(margin_y)
            ax_x.semilogy(x_axis, np.maximum(margin_x, fx))
            ax_y.semilogx(np.maximum(margin_y, fy), y_axis)
            ax_x.set_ylim(bottom=fx * 0.5)
            ax_y.set_xlim(left=fy * 0.5)
        else:
            ax_x.plot(x_axis, margin_x)
            ax_y.plot(margin_y, y_axis)

    # --- Colourbar ---
    if has_panels:
        cax_list = []
        for _ in [ax_map, ax_y]:
            cax_list.append(divider.append_axes('top', size='3%', pad=0.05))
        cax_list[1].axis('off')
        fig.colorbar(imgrid, ax=ax_map, cax=cax_list[0], orientation='horizontal')
        cax_list[0].xaxis.set_ticks_position('top')
        cax_list[0].xaxis.set_label_position('top')
        if cbar_label:
            cax_list[0].set_xlabel(cbar_label, fontsize=12)
    else:
        cb = fig.colorbar(imgrid, ax=ax_map)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=12)

    # --- Labels and sync ---
    if has_panels:
        ax_y.set_yticklabels([])
        ax_map.set_xticks([])
        ax_y.set_ylim(ax_map.get_ylim())
        ax_x.set_xlim(ax_map.get_xlim())
        ax_map.set_ylabel(y_label, fontsize=16)
        ax_x.set_xlabel(x_label, fontsize=16)
    else:
        ax_map.set_xlabel(x_label, fontsize=16)
        ax_map.set_ylabel(y_label, fontsize=16)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')

    if has_panels:
        return fig, (ax_map, ax_y, ax_x)
    return fig, ax_map


def plot_posterior_2d(posterior, x_axis, y_axis, margin_x, margin_y,
                      x_label=r'$v_{\rm sys}$ (km s$^{-1}$)',
                      y_label=r'$K_{\rm P}$ (km s$^{-1}$)',
                      n_sigma=3, sigma_levels=None,
                      sigma_display='contours',
                      crosshair_hole=0.03,
                      x_lim=None, y_lim=None,
                      scale='log',
                      figsize=(6, 6), save_path=None):
    """Genuine-posterior 2D map: sigma contours/crosshairs + marginal panels.

    Thin wrapper around ``plot_2d_map`` -- this is the only place that knows
    about *cumulative probability mass*: it turns ``sigma_levels`` into
    actual posterior-value thresholds (``get_contours_posterior``) before
    handing everything else off to ``plot_2d_map``. Core implementation
    shared by ``plot_kpvsys_map`` and ``plot_alpha_rv_map``.

    Use ``plot_2d_map`` directly instead for anything that isn't a genuine,
    normalizable probability density -- a raw CCF map, or a sigma-scale map
    from ``logl_grid.compute_empirical_sigma_map``/
    ``compute_alpha_significance_map`` (those already report sigma values
    directly, with no cumulative-probability step needed at all).

    Parameters
    ----------
    posterior : (n_x, n_y) array — linear posterior
    x_axis : (n_x,) array
    y_axis : (n_y,) array
    margin_x : (n_x,) array — posterior marginalised over y
    margin_y : (n_y,) array — posterior marginalised over x
    x_label, y_label : str — axis labels
    n_sigma : int — number of sigma levels (1–5) when ``sigma_levels`` is None
    sigma_levels : list of float, optional — explicit sigma values
    sigma_display : {'contours', 'crosshairs', 'none'}
        Same meaning as ``plot_2d_map``'s ``level_style``, plus: in
        ``'crosshairs'`` mode, the outermost requested sigma value is
        printed as a text label on the map (e.g. "3σ").
    crosshair_hole : float — fractional gap in crosshair lines (default 0.03).
    x_lim, y_lim : (float, float), optional — display range limits.
    scale : {'log', 'linear'}
        Colour scale for the 2D map and the probability axes of the marginal panels.
    figsize : tuple
    save_path : str or Path, optional

    Returns
    -------
    fig, (ax_map, ax_y, ax_x)
        ``ax_y`` — right panel (y-axis marginal); ``ax_x`` — bottom panel (x-axis marginal)
    """
    # --- Crop axes FIRST (matches the pre-refactor behaviour: sigma levels
    # below are derived from the cropped region's own probability mass, not
    # the full grid's) ---
    if x_lim is not None:
        mask = (x_axis >= x_lim[0]) & (x_axis <= x_lim[1])
        x_axis, posterior, margin_x = x_axis[mask], posterior[mask, :], margin_x[mask]
    if y_lim is not None:
        mask = (y_axis >= y_lim[0]) & (y_axis <= y_lim[1])
        y_axis, posterior, margin_y = y_axis[mask], posterior[:, mask], margin_y[mask]

    d_x = x_axis[1] - x_axis[0]
    d_y = y_axis[1] - y_axis[0]

    # --- Sigma mass arrays -> actual posterior-value thresholds ---
    _s2d = [0.3935, 0.8647, 0.9889, 0.9997, 0.9999994]
    if isinstance(sigma_levels, str) and sigma_levels in ('auto', 'max'):
        # Find the maximum sigma that still encloses only the primary peak,
        # then draw a single contour at that level.
        sigma_iso = find_isolated_peak_sigma(posterior, x_axis, y_axis)
        log.info(f'Auto sigma level: {sigma_iso:.2f}σ (isolated primary peak)')
        sigma_levels = np.array([sigma_iso]) if sigma_iso > 0 else np.array([1.])
    if sigma_levels is not None:
        sigma_levels = np.asarray(sigma_levels, dtype=float)
        sigma_masses_2d = sigma2percent_2d(sigma_levels).tolist()
    else:
        sigma_levels = np.arange(1, n_sigma + 1, dtype=float)
        sigma_masses_2d = _s2d[:n_sigma]

    lvl_post, _ = get_contours_posterior(
        posterior, [d_x, d_y], lvls=sigma_masses_2d, renormalize=True,
    )

    level_style = 'none' if sigma_display == 'none' else sigma_display
    fig, axes = plot_2d_map(
        posterior, x_axis, y_axis, margin_x=margin_x, margin_y=margin_y,
        x_label=x_label, y_label=y_label,
        levels=lvl_post if sigma_display != 'none' else None,
        level_style=level_style, crosshair_hole=crosshair_hole,
        scale=scale,
        cbar_label='Probability density', figsize=figsize, save_path=None,
    )
    ax_map, ax_y, ax_x = axes

    # --- Text label for the outermost sigma level (crosshairs mode only) ---
    if sigma_display == 'crosshairs':
        outermost_lvl = float(np.sort(lvl_post)[0])
        j_lvl = np.nonzero(posterior >= outermost_lvl)[1]
        y_hi = y_axis[j_lvl.max()]
        outermost = sigma_levels[-1]
        label = (f'{outermost:.1f}' if outermost % 1 else f'{int(outermost)}') + r'$\sigma$'
        ax_map.text(x_axis.min(), y_hi, label, fontsize=16, weight='bold', color='white')

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')

    return fig, (ax_map, ax_y, ax_x)


def plot_kpvsys_map(posterior, vsys_axis, kp_axis, margin_vsys, margin_kp,
                    vsys_lim=None, kp_lim=None,
                    show_sigma=None, **kwargs):
    """Kp-Vsys posterior map.  Thin wrapper around ``plot_posterior_2d``."""
    import warnings as _w
    if show_sigma is not None:
        _w.warn("show_sigma is deprecated; use sigma_display.", DeprecationWarning, stacklevel=2)
        kwargs.setdefault('sigma_display', 'crosshairs' if show_sigma else 'none')
    fig, (ax_map, ax_kp, ax_vsys) = plot_posterior_2d(
        posterior, vsys_axis, kp_axis, margin_vsys, margin_kp,
        x_label=r'$v_{\rm rad}$ (km s$^{-1}$)',
        y_label=r'$K_{\rm P}$ (km s$^{-1}$)',
        x_lim=vsys_lim, y_lim=kp_lim,
        **kwargs,
    )
    return fig, (ax_map, ax_kp, ax_vsys)


def plot_contours_overlay(posteriors, vsys_axes, kp_axes,
                          labels=None, colors=None, linewidths=None,
                          n_sigma=1, sigma_levels=None,
                          vsys_lim=None, kp_lim=None,
                          rv_expected=None, kp_ref=None,
                          figsize=(6, 5), ax=None, save_path=None):
    """Overlay Kp-vsys sigma contours from multiple posteriors on a single panel.

    Useful for comparing per-visit detections or different molecular models.
    Each posterior contributes one set of contours drawn with its own colour.

    Parameters
    ----------
    posteriors : list of (n_vsys, n_kp) arrays — linear posteriors
    vsys_axes  : list of (n_vsys,) arrays
    kp_axes    : list of (n_kp,)  arrays
    labels : list of str, optional
    colors : list of str, optional  (default: Paul Tol bright palette)
    linewidths : list of float, optional (default: 1.5 for all)
    n_sigma : int — number of sigma levels when ``sigma_levels`` is None (default 1)
    sigma_levels : list of float, optional — explicit sigma values
    vsys_lim, kp_lim : (float, float), optional — axis display limits
    rv_expected : float, optional — dashed reference line at this vsys
    kp_ref : float, optional — dashed reference line at this Kp
    figsize : tuple
    ax : matplotlib Axes, optional — draw into existing axes
    save_path : str or Path, optional

    Returns
    -------
    fig, ax
    """
    from matplotlib.lines import Line2D

    # Paul Tol "bright" colorblind-safe palette
    _TOL = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']

    n = len(posteriors)
    if colors is None:
        colors = [_TOL[i % len(_TOL)] for i in range(n)]
    if labels is None:
        labels = [f'Dataset {i + 1}' for i in range(n)]
    if linewidths is None:
        linewidths = [1.5] * n

    # Sigma mass thresholds
    _s2d = [0.3935, 0.8647, 0.9889, 0.9997, 0.9999994]
    if sigma_levels is not None:
        sigma_masses = sigma2percent_2d(np.asarray(sigma_levels, dtype=float)).tolist()
    else:
        sigma_masses = _s2d[:n_sigma]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    handles = []
    for posterior, vsys, kp, label, color, lw in zip(
        posteriors, vsys_axes, kp_axes, labels, colors, linewidths
    ):
        # Crop to display limits
        vsys_p, kp_p, post_p = vsys, kp, posterior
        if vsys_lim is not None:
            m = (vsys >= vsys_lim[0]) & (vsys <= vsys_lim[1])
            vsys_p, post_p = vsys[m], post_p[m, :]
        if kp_lim is not None:
            m = (kp >= kp_lim[0]) & (kp <= kp_lim[1])
            kp_p, post_p = kp[m], post_p[:, m]

        d_vsys = vsys_p[1] - vsys_p[0]
        d_kp   = kp_p[1]   - kp_p[0]
        lvl_post, _ = get_contours_posterior(
            post_p, [d_vsys, d_kp], lvls=sigma_masses, renormalize=True,
        )
        ax.contour(vsys_p, kp_p, post_p.T,
                   levels=lvl_post[::-1], colors=[color], linewidths=lw)
        handles.append(Line2D([0], [0], color=color, lw=lw, label=label))

    ax.legend(handles=handles, fontsize=10)
    ax.set_xlabel(r'$v_{\rm sys}$ (km s$^{-1}$)', fontsize=14)
    ax.set_ylabel(r'$K_{\rm P}$ (km s$^{-1}$)', fontsize=14)

    if vsys_lim is not None:
        ax.set_xlim(vsys_lim)
    if kp_lim is not None:
        ax.set_ylim(kp_lim)
    if rv_expected is not None:
        ax.axvline(rv_expected, color='gray', linestyle='--', alpha=0.5, lw=0.8)
    if kp_ref is not None:
        ax.axhline(kp_ref, color='gray', linestyle='--', alpha=0.5, lw=0.8)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')

    return fig, ax


def find_isolated_peak_sigma(posterior, vsys_axis, kp_axis,
                              sigma_max=10., n_steps=200,
                              min_secondary_pixels=3):
    """Find the maximum sigma contour that encloses only the primary peak.

    Brute-force scan: tests ``n_steps`` sigma values from 0 to ``sigma_max``.
    At each step, the region ``posterior >= threshold(sigma)`` is labelled into
    connected components.  The scan stops the first time a secondary component
    with ≥ ``min_secondary_pixels`` pixels appears alongside the primary peak,
    and returns the last sigma where the primary was still alone.

    Parameters
    ----------
    posterior : (n_x, n_y) array — linear posterior (from ``compute_kpvsys_posterior``)
    vsys_axis : (n_vsys,) array
    kp_axis   : (n_kp,)  array
    sigma_max : float — upper limit of the scan (default 10)
    n_steps   : int   — number of sigma values tested (default 200, ~0.05σ step)
    min_secondary_pixels : int
        A secondary component must have at least this many pixels to count as a
        genuine secondary peak (filters single-pixel noise spikes).

    Returns
    -------
    float — maximum sigma for which the primary peak is isolated (0.0 if never)
    """
    import warnings
    from scipy import ndimage

    d_vsys = vsys_axis[1] - vsys_axis[0]
    d_kp   = kp_axis[1]   - kp_axis[0]
    max_ind = np.unravel_index(np.argmax(posterior), posterior.shape)

    last_valid = 0.
    # Scan from small sigma upward; break on first secondary-peak detection.
    sigmas = np.linspace(0., sigma_max, n_steps + 1)[1:]  # skip sigma=0
    for sigma in sigmas:
        mass = sigma2percent_2d(sigma)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            lvl, _ = get_contours_posterior(
                posterior, [d_vsys, d_kp], lvls=[mass], renormalize=True,
            )
        labeled, n_labels = ndimage.label(posterior >= lvl[0])

        peak_label = labeled[max_ind]
        if peak_label == 0:
            break  # primary peak dropped below threshold

        has_secondary = any(
            lbl != peak_label and np.sum(labeled == lbl) >= min_secondary_pixels
            for lbl in range(1, n_labels + 1)
        )
        if has_secondary:
            break

        last_valid = sigma

    return last_valid


def plot_logl_per_order(map_orders, vsys_axis, kp_axis,
                         idx_orders=None, n_valid=None, n_col=6,
                         shared_clim=True, figsize=None, cmap='viridis', save_path=None):
    """Grid of Kp-vsys maps, one panel per spectral order.

    Useful for diagnosing which orders drive the detection: a genuine planet
    signal should appear consistently at the same (vsys, Kp) across orders,
    while telluric or instrumental artefacts appear at fixed wavelengths.

    Parameters
    ----------
    map_orders : (n_vsys, n_kp, n_orders) array
        Map values per order.  Typically the result of
        ``get_logl(sum_axis=-2)`` or ``get_ccf(sum_axis=-2)``,
        which sum over exposures but keep the order axis.
    vsys_axis : (n_vsys,) array — v_sys grid in km/s
    kp_axis : (n_kp,) array — Kp grid in km/s
    idx_orders : array-like, optional
        True spectral order indices (for panel titles).  If None, uses
        0, 1, 2, … up to n_orders.
    n_valid : (n_orders,) array, optional
        Mean number of valid spectral pixels per order.  Displayed in the
        title of each panel alongside the order index.
    n_col : int
        Number of columns in the subplot grid.  Default: 6.
    shared_clim : bool
        If True (default), all panels share the same colour scale (global
        min/max), making it easy to compare signal amplitude across orders.
        If False, each panel auto-scales independently.
    figsize : tuple, optional
        Figure size.  Default: (2.2 × n_col, 2.0 × n_rows).
    cmap : str
        Matplotlib colormap name.  Default: 'viridis'.
    save_path : str or Path, optional
        If given, save the figure to this path.

    Returns
    -------
    fig, axes : (n_rows, n_col) array of Axes
    """
    n_orders = map_orders.shape[-1]
    n_rows = int(np.ceil(n_orders / n_col))
    if figsize is None:
        figsize = (2.2 * n_col, 2.0 * n_rows)
    if idx_orders is None:
        idx_orders = np.arange(n_orders)

    if shared_clim:
        vmin = float(np.ma.min(map_orders))
        vmax = float(np.ma.max(map_orders))
    else:
        vmin = vmax = None

    fig, axes = plt.subplots(n_rows, n_col, figsize=figsize)
    for idx, ax_i in enumerate(np.ravel(axes)):
        if idx < n_orders:
            ax_i.pcolormesh(vsys_axis, kp_axis, map_orders[:, :, idx].T,
                            vmin=vmin, vmax=vmax, cmap=cmap)
            title = f'Ord {idx_orders[idx]}'
            if n_valid is not None:
                title += f', N={int(n_valid[idx])}'
            ax_i.set_title(title, fontsize='small')
        ax_i.set_xticks([])
        ax_i.set_yticks([])

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    return fig, axes


def plot_loo_contributions(posterior_full, log_delta, contributions,
                            vsys_axis, kp_axis, idx_orders=None,
                            order_labels=None, n_col=6,
                            vsys_lim=None, kp_lim=None,
                            contributions_frac=None,
                            kp_ref=None, vsys_ref=None,
                            figsize=None, save_path=None):
    """Two-panel LOO (leave-one-out) order contribution figure.

    Top panel — bar chart
        One bar per order showing the fractional contribution to the total
        detected signal (if contributions_frac is provided), or the raw
        Δlog P[peak] otherwise.
        Positive (blue) = order helps; negative (vermillion) = order hurts.

    Bottom panel — grid of difference maps
        For each order k, a Kp-vsys heatmap of
        ``log_post_full_norm − log_post_loo_norm_k`` using a diverging colormap.
        Shows *where* in the Kp-vsys plane each order contributes.

    Parameters
    ----------
    posterior_full : (n_vsys, n_kp) — full posterior from ``compute_loo_order_contributions``
    log_delta      : (n_orders, n_vsys, n_kp) — per-order log-posterior difference
    contributions  : (n_orders,) — ``log_delta`` at the reference peak
    vsys_axis, kp_axis : 1-D arrays
    idx_orders     : 1-D int array, optional — order indices (for x-tick labels)
    order_labels   : list of str, optional — override x-tick labels
    n_col          : int — columns in the difference-map grid
    vsys_lim, kp_lim : (float, float), optional — zoom limits for the maps
    contributions_frac : (n_orders,) array, optional
        If provided, the bar chart shows fractional contribution (0–1 scale)
        instead of raw Δlog P.
    kp_ref, vsys_ref : float, optional
        Expected planet location to mark on the maps (km/s).
    figsize        : tuple, optional
    save_path      : str or Path, optional

    Returns
    -------
    fig, (ax_bar, axes_maps)
    """
    n_orders = len(contributions)
    n_row_maps = int(np.ceil(n_orders / n_col))

    if figsize is None:
        figsize = (max(8, n_col * 1.8), 4 + n_row_maps * 2.2)

    fig = plt.figure(figsize=figsize)
    # Top 25% = bar chart; bottom 75% = grid of maps
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.35)
    ax_bar = fig.add_subplot(gs[0])
    gs_maps = gs[1].subgridspec(n_row_maps, n_col, hspace=0.05, wspace=0.05)

    # ── Bar chart ────────────────────────────────────────────────────────────
    bar_values = contributions_frac if contributions_frac is not None else contributions
    x = np.arange(n_orders)
    colors = np.where(bar_values >= 0, '#0072B2', '#D55E00')  # blue / vermillion
    ax_bar.bar(x, bar_values, color=colors, width=0.7)
    ax_bar.axhline(0, color='k', lw=0.8)
    if contributions_frac is not None:
        ax_bar.set_ylabel('Fraction of signal\n(off-peak LOO)', fontsize=10)
    else:
        ax_bar.set_ylabel(r'$\Delta f_\mathrm{off\text{-}peak}$ (LOO)', fontsize=10)
    ax_bar.set_title('Order contribution (LOO)', fontsize=11)

    if order_labels is not None:
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(order_labels, rotation=45, ha='right', fontsize=7)
    elif idx_orders is not None:
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(idx_orders, rotation=45, ha='right', fontsize=7)
    else:
        ax_bar.set_xlabel('Order index', fontsize=11)

    # ── Difference maps ──────────────────────────────────────────────────────
    # Crop axes if limits are provided
    vsys_plot = vsys_axis
    kp_plot   = kp_axis
    log_delta_plot = log_delta
    if vsys_lim is not None:
        m = (vsys_axis >= vsys_lim[0]) & (vsys_axis <= vsys_lim[1])
        vsys_plot      = vsys_axis[m]
        log_delta_plot = log_delta_plot[:, m, :]
    if kp_lim is not None:
        m = (kp_axis >= kp_lim[0]) & (kp_axis <= kp_lim[1])
        kp_plot        = kp_axis[m]
        log_delta_plot = log_delta_plot[:, :, m]

    # Symmetric colour limits across all orders
    vmax = float(np.nanpercentile(np.abs(log_delta_plot), 99))
    vmax = max(vmax, 1e-6)

    peak_ind = np.unravel_index(np.argmax(posterior_full), posterior_full.shape)
    # Show the reference location: supplied kp_ref/vsys_ref if given, else map peak.
    ref_vsys = vsys_ref if vsys_ref is not None else vsys_axis[peak_ind[0]]
    ref_kp   = kp_ref   if kp_ref   is not None else kp_axis[peak_ind[1]]

    axes_maps = []
    for k in range(n_orders):
        row_i, col_i = divmod(k, n_col)
        ax_i = fig.add_subplot(gs_maps[row_i, col_i])
        axes_maps.append(ax_i)

        ax_i.pcolormesh(vsys_plot, kp_plot, log_delta_plot[k].T,
                        cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax_i.axvline(ref_vsys, color='k', lw=0.5, ls='--', alpha=0.5)
        ax_i.axhline(ref_kp,   color='k', lw=0.5, ls='--', alpha=0.5)

        lbl = (order_labels[k] if order_labels is not None
               else str(idx_orders[k]) if idx_orders is not None
               else str(k))
        ax_i.text(0.04, 0.96, lbl, transform=ax_i.transAxes,
                  fontsize=6, va='top', color='k')
        ax_i.set_xticks([])
        ax_i.set_yticks([])

    # Turn off unused panels
    for k in range(n_orders, n_row_maps * n_col):
        row_i, col_i = divmod(k, n_col)
        fig.add_subplot(gs_maps[row_i, col_i]).axis('off')

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')

    return fig, (ax_bar, axes_maps)


def plot_alpha_marginal(alpha_array, logl_at_peak, log_p_alpha,
                         vsys_peak=None, kp_peak=None,
                         fig=None, ax=None, save_path=None):
    """Plot the logL profile and marginal posterior as a function of alpha.

    Two curves are shown:

    * **Slice at peak** (black) — ``logL(α, vsys_peak, Kp_peak)``: how the
      logL varies with the model amplitude at the best-fit position.
    * **Full marginal** (blue) — ``log P(α | data) ∝ ∫∫ L(α, v, K) dv dK``:
      the alpha posterior marginalised over all (vsys, Kp).

    **Interpretation:**

    * Detection: both curves are peaked near alpha ≈ 1 (the model amplitude
      is recovered near its true value).
    * Non-detection: the marginal is flat/declining — alpha is unconstrained,
      favouring low values because the data are consistent with no planet.

    The two curves are separately normalised to zero at their maximum so they
    can be overlaid on the same axis regardless of their absolute offsets.

    Parameters
    ----------
    alpha_array : (n_alpha,) array
    logl_at_peak : (n_alpha,) array
        logL values at the posterior peak, one per alpha.
        From ``logl_grid.compute_alpha_marginal``.
    log_p_alpha : (n_alpha,) array
        Log marginal posterior P(α|data).
        From ``logl_grid.compute_alpha_marginal``.
    vsys_peak, kp_peak : float, optional
        Peak position in km/s — used only for the axis title.
    fig, ax : optional  — supply existing Figure/Axes to embed in a multi-panel
        figure.

    Returns
    -------
    fig, ax
    """
    fig, ax = _get_fig_and_ax_inputs(fig, ax)

    # Shift both curves to zero at their maximum for overlay
    logl_norm     = logl_at_peak  - logl_at_peak.max()
    log_p_norm    = log_p_alpha   - log_p_alpha.max()

    ax.plot(alpha_array, logl_norm,  'k-',  label=r'$\log L(\alpha)$ at peak')
    ax.plot(alpha_array, log_p_norm, 'b--', label=r'$\log P(\alpha \,|\, \mathrm{data})$')
    ax.axvline(1., linestyle=':', color='gray', label=r'$\alpha = 1$')
    ax.axhline(0., linestyle=':', color='lightgray')

    ax.set_xlabel(r'$\alpha$ (model amplitude)', fontsize=14)
    ax.set_ylabel(r'$\log L$ (relative)', fontsize=14)
    ax.legend()

    if vsys_peak is not None and kp_peak is not None:
        ax.set_title(
            fr'Alpha marginal  ($v_\mathrm{{sys}}={vsys_peak:.1f}$, '
            fr'$K_P={kp_peak:.1f}$ km/s)',
        )

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    return fig, ax


def plot_alpha_rv_map(posterior, alpha_axis, rv_axis,
                      margin_alpha, margin_rv,
                      rv_label=r'$K_{\rm P}$ (km s$^{-1}$)', **kwargs):
    """Alpha × RV posterior map.  Thin wrapper around ``plot_posterior_2d``.

    Parameters
    ----------
    posterior    : (n_alpha, n_rv) array
    alpha_axis   : (n_alpha,) array
    rv_axis      : (n_rv,)    array — Kp or vsys in km/s
    margin_alpha : (n_alpha,) array — posterior marginalised over rv
    margin_rv    : (n_rv,)    array — posterior marginalised over alpha
    rv_label     : str — y-axis label
    **kwargs     : forwarded to ``plot_posterior_2d``

    Returns
    -------
    fig, (ax_map, ax_alpha, ax_rv)
        ``ax_alpha`` — bottom panel; ``ax_rv`` — right panel.
    """
    kwargs.setdefault('figsize', (6, 5))
    fig, (ax_map, ax_rv, ax_alpha) = plot_posterior_2d(
        posterior, alpha_axis, rv_axis, margin_alpha, margin_rv,
        x_label=r'$\alpha$ (model amplitude)',
        y_label=rv_label,
        **kwargs,
    )
    # Alpha = 1 reference lines
    ax_map.axvline(1., color='w', linestyle='--', linewidth=0.8, alpha=0.6)
    ax_alpha.axvline(1., color='gray', linestyle='--', linewidth=0.8)
    return fig, (ax_map, ax_alpha, ax_rv)
