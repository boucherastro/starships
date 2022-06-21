import starships.orbite as o
import starships.analysis as a
import starships.correlation as corr
import starships.homemade as hm
import starships.ttest_fcts as nf
import starships.plotting_fcts as pf
from starships import spectrum as spectrum

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec 
import astropy.units as u
import astropy.constants as const

from scipy.interpolate import interp1d, interp2d
from scipy.optimize import curve_fit, fsolve
from mpl_toolkits.axes_grid1 import make_axes_locatable

    
class Correlations():

    def __init__(self, data, kind=None, rv_grid=None, n_pcas=None, kp_array=None): 
        
        self.data = data
        self.kind = kind
        self.rv_grid = rv_grid  
        self.n_pcas = n_pcas
        self.kp_array = kp_array
            
    def calc_ccf(self, orders=None, N=1, alpha=None, index=None, ccf0=None, rm_vert_mean=False,):  #v
        
        if ccf0 is None:
            if orders is None:
                orders = list(np.arange(49))
#             self.orders = orders
#             self.ord_frac = 100*len(orders)/(49)#-len(np.where(t.N0.mask.all(axis=0))[0]))
            self.ccf0 = np.ma.sum(self.data[:, orders] * N[:,orders], axis=1)
        else:
            self.ccf0 = ccf0
            
            
        if rm_vert_mean is True:
            n_spec, n_RV = self.ccf0.shape
            self.ccf00 = self.ccf0.copy()
            yyy=self.ccf0-np.mean(self.ccf0, axis=-1)[:,None]
            
            recon_time = np.ones_like(yyy)*np.nan
            x = np.arange(n_spec)
            z_t = np.zeros((n_spec, 3))

            for col in range(n_RV):

                y = yyy[:,col]

                if y.mask.all():
                    continue

                idx = np.isfinite(x) & np.isfinite(y)

                # Poly.fit(x[idx], y[idx], 2)
                z_t = np.polyfit(x[idx], y[idx], 2)

                recon_time[idx,  col] = np.poly1d(z_t)(x[idx])

            recon_time = np.ma.masked_invalid(recon_time)
            self.ccf0 = yyy-recon_time

        if alpha is None:
            alpha = np.ones_like(self.ccf0)
        if alpha.shape != self.ccf0.shape:
            alpha = alpha[:, None]
        
        self.ccf = self.ccf0.copy()*alpha
        if index is not None:
            self.ccf[np.unique(index)]=0
            
    def calc_logl(self, tr, icorr=None, orders=None, index=None, 
                  N=None, nolog=None, alpha=None, inj_alpha='ones', **kwargs):  
        
        if orders is None:
            orders = list(np.arange(49))
        if icorr is None:
            icorr = tr.icorr
        if alpha is None:
            if inj_alpha =='alpha':
                alpha = np.ones_like(tr.alpha_frac)
            elif inj_alpha == 'ones':
                alpha = tr.alpha_frac
#             alpha = np.ones_like(tr.alpha_frac)  #tr.alpha_frac  # np.ones_like(tr.alpha_frac)
        self.logl0 = np.nansum( self.data[:, orders], axis=1)
        self.logl = corr.sum_logl(self.data, icorr, orders, N, alpha=alpha, 
                                  axis=0, del_idx=index, nolog=nolog, **kwargs)
        
    def calc_logl_snr(self, n_pca=None, **kwargs):

        if ~hasattr(self,'interp_grid'):
            self.interp_grid = self.rv_grid

        if (n_pca is not None) and (len(self.n_pcas) > 1):
            id_pc = np.where(np.array(self.n_pcas) == n_pca)[0]
            try:
                self.courbe = (self.logl.squeeze()[:,id_pc]).squeeze()
            except :
                print('The requested PC is not available, but taking ', self.n_pcas[0])
                self.courbe = (self.logl.squeeze()[:,0]).squeeze()
        else:
            self.courbe = self.logl.squeeze()


        self.get_snr_1d(**kwargs)
        self.find_max()

        
    def plot_multi_npca(self, logl=None, vlines=[0], kind='snr', kind_courbe='bic', ylim=None, 
                        no_signal=None, ground_type='mean',
                        hlines=None, legend=True, title='', max_rv=None, force_max_rv=None, **kwargs):
    
        lstyles = ['-','--','-.',':']

        self.interp_grid = self.rv_grid
        if logl is None:
            logl = self.logl
        
        val = []
        pos = []
        snr = []
        courbe = []
        
        fig,ax = plt.subplots(1,2, figsize=(10,4))
        for i in range(len(self.n_pcas)):
            couleur = (0.5,0.1,i/len(self.n_pcas))
            
            if len(self.n_pcas) > 1:
#                 print(logl.shape)
                self.courbe = (logl[:,:,i]).squeeze()
            else:
                self.courbe = logl.squeeze()
            
#             if kind == 'courbe':
            self.get_snr_1d(max_rv=max_rv, **kwargs)
#             self.snr=self.courbe
#             print(self.snr[[0,-1]])
            if force_max_rv is None:
                self.find_max()
            else:
                if kind == 'snr':
                    arr = self.snr
                elif kind == 'courbe':
                    arr = self.courbe
                fct = interp1d(self.interp_grid, arr)
                self.pos = force_max_rv
                self.max = fct(force_max_rv)
            
            val.append(self.max)
            pos.append(self.pos)
            snr.append(self.snr)
            courbe.append(self.courbe)
            
            
            for line in vlines:
                ax[1].axvline(line)
            
            if kind == 'snr':
                ax[1].plot(self.rv_grid, self.snr,  color=couleur, linestyle=lstyles[i%len(lstyles)],
                           label='{:.1f} / {:.2f} / {:.2f}'.format(self.n_pcas[i], self.max, self.pos))
            elif kind == 'courbe':
                if kind_courbe == 'bic':
                    ax[1].plot(self.rv_grid, self.courbe-self.courbe[self.idx_bruit_rv].mean(),  
                           color=couleur, linestyle=lstyles[i%len(lstyles)],
                       label='{:.1f} / {:.2f} / {:.2f}'.format(self.n_pcas[i], self.max, self.pos))
                if kind_courbe == 'abs':
                    ax[1].plot(self.rv_grid, self.courbe,  
                           color=couleur, linestyle=lstyles[i%len(lstyles)],
                       label='{:.1f} / {:.2f} / {:.2f}'.format(self.n_pcas[i], self.max, self.pos))
                
            if legend is True:
                ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
            if ylim is not None:
                ax[1].set_ylim(*ylim)
                
        val = np.array(val)
        self.npc_val = val
        self.npc_pos = np.array(pos)
        self.npc_snr = np.array(snr)
        self.npc_courbe = np.array(courbe)

        print('Max value at {} npc = {} at {} km/s'.format(self.n_pcas[np.argmax(val)], \
                                                           val.max(), self.npc_pos[np.argmax(val)]))
        if hlines is not None:
            for line in hlines:
                ax[0].axhline(line)
        if kind == 'snr':
            ax[0].plot(self.n_pcas, val,'o--')
            ax[0].plot(self.n_pcas[np.argmax(val)], val[np.argmax(val)],'ro')
        elif kind == 'courbe':
            idx_pos_max = hm.nearest(self.rv_grid, self.npc_pos)
            maxima = [self.npc_courbe[i, idx_pos_max[i]] for i in range(len(self.n_pcas))]
            
            max_val = np.array(maxima)
            self.npc_max_abs = max_val
            
            if ground_type == 'mean':
                ground_lvl = self.npc_courbe[:, self.idx_bruit_rv].mean(axis=-1)
            elif ground_type == 'max_noise':
                ground_lvl = np.abs(self.npc_courbe[:, self.idx_bruit_rv]).max(axis=-1)
            elif ground_type == 'no_signal_idx':
                ground_lvl = self.npc_courbe[no_signal, idx_pos_max]
            elif ground_type == 'no_signal_curve':
                ground_lvl = [no_signal.npc_courbe[i, idx_pos_max[i]] for i in range(len(self.n_pcas))]
            elif ground_type == 'no_signal_value':
                ground_lvl = no_signal
            
            delta_bic = 2*(np.array(maxima) - ground_lvl)
            self.npc_bic = delta_bic #np.log10(delta_bic)
            
            if kind_courbe == 'bic':
                ax[0].plot(self.n_pcas, delta_bic,'o--')
#                 ax[0].axhline(1)
#                 ax[0].set_ylim(-10,100)
            elif kind_courbe == 'abs':
                ax[0].plot(self.n_pcas, max_val,'o--')
            
        ax[0].set_ylabel(title)

   
    
    def get_snr_2d(self, ccf2d=None, Kp_array=None, interp_grid=None, \
                   kp_limit=70, rv_limit=15, RV_sys=0, Kp0=151):
        
        if interp_grid is None:
            interp_grid = self.interp_grid
        else:
            self.interp_grid = interp_grid
        if Kp_array is None:
            Kp_array = self.Kp_array 
        else:
            self.Kp_array = Kp_array 
        if ccf2d is None:
            ccf2d = self.sum_ccf  
        else:
            self.sum_ccf = ccf2d  
            self.curve_fct2d = interp2d(interp_grid, Kp_array, ccf2d)
            
        idx_bruit_rv = (interp_grid < RV_sys - rv_limit) | (interp_grid > RV_sys + rv_limit)
        idx_bruit_kp = (Kp_array < Kp0 - kp_limit) | (Kp_array > Kp0 + kp_limit)
        self.idx_bruit_rv = idx_bruit_rv
        self.idx_bruit_kp = idx_bruit_kp
        self.idx_bruit_rv0 = idx_bruit_rv

        bruit2d = np.ma.std(ccf2d[idx_bruit_kp][:, idx_bruit_rv])
        self.bruit2d = bruit2d
        snr2d = (ccf2d - np.ma.mean(ccf2d[idx_bruit_kp][:, idx_bruit_rv])) / \
                 np.ma.masked_invalid(bruit2d)
        self.snr2d = snr2d
        self.snr_fct2d = interp2d(interp_grid, Kp_array, snr2d)
            
    def calc_correl_snr_2d(self, tr, icorr=None, limit_shift=60, interp_size=201, RV_sys=0,
                           kp0=0, kp1=2, rv_limit=15, kp_limit=70, RV_shift=0, vr_orb=None, vrp_kind='t'):
    
        if isinstance(RV_sys, u.Quantity):
            RV_sys = (RV_sys.to(u.km / u.s)).value
        if icorr is None:
            icorr = tr.icorr
        self.icorr = icorr
        self.Kp0 = (tr.Kp.to(u.km / u.s)).value
        self.RV_shift = RV_shift
        
        interp_grid = np.linspace(-limit_shift - 2 * (tr.vrp)[icorr].value[0] + RV_sys,
                                  limit_shift - 2 * (tr.vrp)[icorr].value[-1] + RV_sys, 
                                  interp_size).squeeze()
        self.interp_grid = interp_grid
        
        Kp_array = np.arange(kp0, int(tr.Kp.value * kp1))
        self.Kp_array = Kp_array

        sum_ccf = np.zeros((Kp_array.size, interp_grid.size))
        for i, Kpi in enumerate(Kp_array):
            hm.print_static(i)
            
            if vrp_kind == 'nu':
#                 print('nu')
                vrp_orb = o.rv_theo_nu(Kpi, tr.nu[icorr]*u.rad, tr.planet.w, plnt=True).to(u.km/u.s)
            elif vrp_kind == 't':
#                 print('t')
                vrp_orb = o.rv_theo_t(Kpi, tr.t_start[icorr], tr.planet.mid_tr, tr.planet.period, plnt=True).to(u.km/u.s)

#             vrp_orb = o.rv_theo_nu(Kpi, tr.nu[icorr] * u.rad, tr.planet.w, plnt=True).to(u.km/u.s)
            if vr_orb is None:
                vr_orb = (-vrp_orb*(tr.planet.M_pl/tr.planet.M_star).decompose()).to(u.km/u.s)
#                 vr_orb = tr.vr[icorr] #(Kpi*u.km/u.s/tr.planet.M_star*tr.planet.M_pl).to(u.km/u.s)
#             print(vrp_orb.shape,vr_orb.shape,RV_shift[icorr].shape)
            shifted_ccf = np.ma.masked_invalid(a.shift_correl(interp_grid, self.ccf[icorr], 
                                               self.rv_grid, (vrp_orb-vr_orb).value+RV_shift[icorr]))
            sum_ccf[i] = np.ma.sum(shifted_ccf, axis=0)
           
        self.sum_ccf = sum_ccf
        self.curve_fct2d = interp2d(interp_grid, Kp_array, sum_ccf)

        self.get_snr_2d(kp_limit=kp_limit, rv_limit=rv_limit, RV_sys=RV_sys, 
                   Kp0 = (tr.Kp.to(u.km / u.s)).value)
        
        sum_ccf_nonoise = sum_ccf[~self.idx_bruit_kp][:, ~self.idx_bruit_rv]
        self.sum_ccf_nonoise = np.ma.masked_invalid(sum_ccf_nonoise)

        self.idx_max = np.where(sum_ccf_nonoise == sum_ccf_nonoise.max())
        
    def find_max(self, interp_grid=None, snr=None, kind='rv', 
                 kind_max='spline', minmax='max', p0_estim=None):
        
        if interp_grid is None:
            interp_grid = self.interp_grid
        if snr is None:
            snr = self.snr
            
        if kind == "rv":
            idx_bruit = self.idx_bruit_rv0
        elif kind == "kp":
            idx_bruit = self.idx_bruit_kp
        if kind_max == 'spline':
            pos_max = a.find_max_spline(interp_grid[~idx_bruit], snr[~idx_bruit].copy(), 
                                        kind=minmax)
        elif kind_max == 'gauss': 
            if p0_estim is None:
                p0_estim = [np.std(snr[~idx_bruit]), np.nanmax(snr[~idx_bruit]),0]
            pos_max, pos_err = a.find_max_gaussian(interp_grid[~idx_bruit], snr[~idx_bruit].copy(), 
                                          kind=minmax, p0_estim=p0_estim, error_gauss=True)
            pos_max = (pos_max[2], -pos_max[1])
            self.pos_err = pos_err[2] 
            self.max_err = pos_err[1]
        
#         plt.figure()
#         plt.plot(self.interp_grid, self.snr.copy())
#         plt.plot(self.interp_grid[~self.idx_bruit_rv], 
#                                   self.snr[~self.idx_bruit_rv].copy())
#         plt.axvline(pos_max[0])
#         plt.axhline(-pos_max[1])
#         print("max = ", -pos_max[1])
        if kind == "rv":
            self.pos = pos_max[0]
            self.max = -pos_max[1]
        elif kind == "kp":
            self.pos_kp = pos_max[0]
            self.max_kp = -pos_max[1]
    
    def get_curve_at_slice(self, Kp_slice, **kwargs):
        
        if isinstance(Kp_slice, u.Quantity):
            Kp_slice = (Kp_slice.to(u.km / u.s)).value
        
        self.snr = self.snr_fct2d(self.interp_grid, Kp_slice)
        self.courbe = self.curve_fct2d(self.interp_grid, Kp_slice)
        self.Kp_slice = Kp_slice
        
        self.find_max( **kwargs)
        
        
    def get_snr_1d(self, interp_grid=None, courbe=None, rv_limit=8, 
                   RV_sys=0, plot=False, debug=False, max_rv=None, add_mask=None): #
        
        if interp_grid is None:
            interp_grid = self.interp_grid
        else:
            self.interp_grid = interp_grid
        if courbe is None:
            courbe = self.courbe
        else:
            self.courbe = courbe
            
        if debug is True:
            plt.figure()
            plt.plot(interp_grid, self.courbe)
            
        if max_rv is not None:
            id_sub = (interp_grid <= max_rv) | (interp_grid >= -max_rv)
            interp_grid = interp_grid[id_sub]
            courbe = courbe[id_sub]

        idx_bruit = (interp_grid < RV_sys - rv_limit) | (interp_grid > RV_sys + rv_limit)
        idx_bruit0 = idx_bruit.copy()
        if add_mask is not None:
            idx_bruit &= ((interp_grid < add_mask[0] - add_mask[1]) | (interp_grid > add_mask[0] + add_mask[1]))
        
        if plot is True:
            plt.figure()
            plt.plot(interp_grid, courbe)
            plt.plot(interp_grid[idx_bruit], courbe[idx_bruit])
        
        self.idx_bruit_rv = idx_bruit
        self.idx_bruit_rv0 = idx_bruit0
        bruit = np.nanstd(courbe[idx_bruit])
        self.bruit = bruit
        self.snr = (courbe - np.ma.mean(courbe[idx_bruit])) / np.ma.masked_invalid(bruit)
        
        if plot is True:
            plt.figure()
            plt.plot(interp_grid, self.snr)
            plt.plot(interp_grid[idx_bruit], self.snr[idx_bruit])
        
        
    def calc_correl_snr_1d(self, tr, Kp=None, icorr=None, limit_shift=60, interp_size=201, 
                           RV_sys=0, rv_limit=8, plot=True, RV = 0., RV_shift=0,
                           vrp_kind = 't', vrp_orb=None, vr_orb=None):

        if isinstance(RV_sys, u.Quantity):
            RV_sys = (RV_sys.to(u.km / u.s)).value
        if icorr is None:
            icorr = tr.icorr
        if Kp is None:
            Kp = tr.Kp.value
        if vrp_orb is None:  
            if vrp_kind == 'nu':
                vrp_orb = o.rv_theo_nu(Kp, tr.nu*u.rad, tr.planet.w, plnt=True).to(u.km/u.s)
            elif vrp_kind == 't':
                vrp_orb = o.rv_theo_t(Kp, tr.t_start, tr.planet.mid_tr, tr.planet.period, plnt=True).to(u.km/u.s)
        if vr_orb is None:
            vr_orb = (-vrp_orb*(tr.planet.M_pl/tr.planet.M_star).decompose()).to(u.km/u.s)
#             dvr = o.rv_theo_nu(Kp, tr.nu * u.rad, tr.planet.w, plnt=True)

        interp_grid = np.linspace(-limit_shift + RV_sys, limit_shift + RV_sys, interp_size).squeeze()
        self.interp_grid = interp_grid

        ccf_shifted = np.ma.masked_invalid(a.shift_correl(interp_grid, self.ccf, 
                                           self.rv_grid, (vrp_orb-vr_orb).value+RV_shift))
        self.shifted_ccf = ccf_shifted
        
        if plot is True:
            self.plot_PRF(tr, RV=RV)
#             idx_mid = hm.nearest(self.interp_grid, RV)
#             colum3 = np.ma.mean(self.shifted_ccf[:,idx_mid-1:idx_mid+2],axis=-1)
            
#             fig = plt.figure() 
#             fig.set_figheight(6) 
#             fig.set_figwidth(10) 

#             spec = gridspec.GridSpec(ncols=2, nrows=1,  width_ratios=[3, 1]) 

#             ax0 = fig.add_subplot(spec[0]) 
#             ax0.pcolormesh(self.interp_grid, np.arange(tr.n_spec)[icorr], self.shifted_ccf[icorr])
#             ax0.axvline(RV, alpha=0.5, color='k')

#             ax1 = fig.add_subplot(spec[1]) 
#             ax1.plot(colum3[icorr], np.arange(tr.n_spec)[icorr],'o-') 
#             ax1.set_xlim(colum3[icorr].min()*1.5, colum3[icorr].max()*1.5)

        self.courbe = np.ma.sum(ccf_shifted[icorr], axis=0)
        
        self.get_snr_1d(rv_limit=rv_limit, RV_sys=RV_sys)
        
        
    def find_1sig(self, limit_rv=5., step=1.):
    
        def f(x, lvl):
            return fct_snr(x) - lvl

        fct_snr=interp1d(self.interp_grid, self.snr, kind='cubic')

        down, up = fsolve(f, self.pos-limit_rv, args=(self.max-step)), fsolve(f, self.pos+limit_rv, args=(self.max-step))

        print(r'{:.1f} + {:.1f} - {:.1f}'.format(self.pos, (up-self.pos)[0], (self.pos-down)[0]))

        return up-self.pos, self.pos-down
    
    
    def find_1sig_kp(self, limit_rv=15):
    
        self.snr_kp = self.snr_fct2d(self.pos, self.Kp_array).squeeze()
        self.find_max(interp_grid=self.Kp_array, snr=self.snr_kp, kind='kp')
    #     plt.figure()
    #     plt.plot(self.Kp_array, self.snr_kp)
        def f(x, lvl):
            return fct_snr(x) - lvl

        fct_snr=interp1d(self.Kp_array, self.snr_kp, kind='cubic')

        down, up = fsolve(f, self.pos_kp-limit_rv, args=(self.max_kp-1)), \
                    fsolve(f, self.pos_kp+limit_rv, args=(self.max_kp-1))

        print(r'{:.0f} + {:.0f} - {:.0f}'.format(self.pos_kp, (up-self.pos_kp)[0], (self.pos_kp-down)[0]))

        return up-self.pos_kp, self.pos_kp-down
    

    def full_plot(self, tr, icorr, wind=None, fit_gauss=False, 
                  fig_larg=8, fig_haut=3, cmap='plasma', fig_name='', tag_max=False,
                  Kp_slice=None, clim=None, get_logl=False, hline=None, save_fig='', show_legend=True):

        figs, (ax1,ax0,ax2) = plt.subplots(3, 1, figsize=(fig_larg, 3 * fig_haut), sharex=False)

        axi = [ax0,ax2]
        
        
        ##### LIGNE VERTICALE À VSYS=0 EN POINTILLÉ
                ###### LIGNE VERTICALE EN TRait plein avec un trou (comme la horizontale) 
                
                ######## + LIGNE HORIZONTALE EN POINTILLÉE POUR SÉPARER LES 2 TRANSITS

        ### --- CCF --- 
        
        if get_logl is True:
            yyy = a.remove_means(self.ccf0[:,None,:], 1).squeeze()
        else:
            yyy = self.ccf0
        im1 = ax1.pcolormesh(self.rv_grid, tr.phase.value, yyy, cmap=cmap, rasterized=True)

        ax1.plot((tr.berv-tr.planet.RV_sys.value), tr.phase.value, '--',color='darkred', alpha=0.8, label='BERV')
#         ax1.set_xlim(self.interp_grid[0], self.interp_grid[-1])
        ax1.set_ylabel(r'$\phi$', fontsize=14)
        if fig_name != '':
            ax1.set_title(fig_name, fontsize=16)

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='2%', pad=0.05)
        cbar = figs.colorbar(im1, ax=ax1, cax=cax)
        cbar.set_label('CCF', fontsize=14)

        iout = np.delete(np.arange(tr.vrp.size), icorr)
        ax1.plot(tr.vrp[iout].value+self.RV_shift[iout]+tr.mid_vrp.value, tr.phase.value[iout], 
                 'k.', alpha=0.5, label=r'Expected $v_{\rm P}$')
        ax1.plot(tr.vrp.value+self.RV_shift+tr.mid_vrp.value, tr.phase.value, 
                 'k', alpha=0.35, label=r'Expected $v_{\rm P}$')
        if hline is not None:
            ax1.axhline(hline, color='white', linestyle='-')

        # --- SUM CCF 2D --- 

#         imaxi2 = axi[0].imshow(self.snr2d, origin='lower', aspect='auto',
#                                extent=(self.interp_grid.min(), self.interp_grid.max(),
#                                self.Kp_array.min(), self.Kp_array.max()), cmap=cmap)
        imaxi2 = axi[0].pcolormesh(self.interp_grid, self.Kp_array, self.snr2d, cmap=cmap, rasterized=True)
        axi[0].set_ylabel(r'$K_{\rm P}$ (km s$^{-1}$)', fontsize=14)

        self.get_curve_at_slice(self.Kp_array[~self.idx_bruit_kp][self.idx_max[0]])
        print('Highest SNR = {} // Kp = {} // RV = {} '.format(self.max, 
                                                  self.Kp_array[~self.idx_bruit_kp][self.idx_max[0]],
                                                               self.pos))
        self.get_curve_at_slice(tr.Kp)
        print(r'Max SNR = {:.2f}$\sigma$, Max position = {:.2f}'.format(self.max, self.pos))
        print('')
        hm.printmd(r'Max SNR = **{:.2f}**$\sigma$, Max position = {:.2f}'.format(self.max, self.pos))
        print('')
        
        if tag_max is True:
            axi[0].plot(self.pos, tr.Kp.to(u.km / u.s).value,'k+', 
                label='Pos = {:.2f} // Max = {:.2f}'.format(self.pos, self.max))
            axi[0].legend(loc='best')

        axi[0].axhline(tr.Kp.to(u.km / u.s).value, linestyle='-', alpha=0.7, color='indigo', 
                      xmin=0,xmax=hm.nearest(self.interp_grid, self.pos-10)/self.interp_grid.size )
        axi[0].axhline(tr.Kp.to(u.km / u.s).value, linestyle='-', alpha=0.7, color='indigo',
                      xmin=hm.nearest(self.interp_grid, self.pos+10)/self.interp_grid.size ,xmax=1) 
        
#         axi[0].axvline(self.pos, linestyle='-', alpha=0.7, color='indigo', 
#                           ymin=0, ymax=(self.Kp_array[~self.idx_bruit_kp][self.idx_max[0]]-50)/self.Kp_array[-1] )
#         axi[0].axvline(self.pos, linestyle='-', alpha=0.7, color='indigo',
#                           ymin=(self.Kp_array[~self.idx_bruit_kp][self.idx_max[0]]+50)/self.Kp_array[-1] , ymax=1) 
        axi[0].axvline(self.pos, linestyle='-', alpha=0.7, color='indigo', 
                          ymin=0, ymax=(tr.Kp.value - 60)/self.Kp_array[-1] )
        axi[0].axvline(self.pos, linestyle='-', alpha=0.7, color='indigo',
                          ymin=(tr.Kp.value + 60)/self.Kp_array[-1] , ymax=1) 
    

        axi[0].axvline(0, color='indigo', linestyle=':')
        axi[0].set_ylim(0, self.Kp_array[-1])

        if wind is not None:
            if wind != 0 :
                axi[0].axvline(wind, linestyle=':', color='dodgerblue', label=r'Wind')

        divider = make_axes_locatable(axi[0])
        cax = divider.append_axes('right', size='2%', pad=0.05)
        cbar = figs.colorbar(imaxi2, ax=axi[0], cax=cax)
        if clim is not None:
            imaxi2.set_clim(clim[0],clim[1])

        cbar.set_label('CCF SNR', fontsize=14)

        ### --- Sum CCF 1D --- 
        axi[1].plot(self.interp_grid, self.snr.squeeze(), color='indigo', label='Expected Kp')

        axi[1].axvline(0, color='indigo', alpha=0.5, linestyle=':', label=r"Planet Rest frame")
#         axi[1].axvline(self.pos, color='indigo', alpha=0.5, linestyle='-')
        axi[1].axhline(0, color='indigo', linestyle=':', alpha=0.5)


        if wind is not None:
            if wind != 0:
                axi[1].axvline(wind, linestyle=':', color='dodgerblue',
                           label=str(wind) + ' km/s Wind')

        axi[1].set_ylabel('CCF SNR', fontsize=14)
        axi[1].set_xlabel(r'$v_{\rm offset}$ (km s$^{-1}$)', fontsize=14)
        


        if clim is not None:
            axi[1].set_ylim(clim[0],clim[1])

        if Kp_slice is not None:
            
            if isinstance(Kp_slice,list):
                Kp_slice = np.array(Kp_slice)
            else:
                Kp_slice = [Kp_slice]

            for k,kpslice in enumerate(Kp_slice):
                if isinstance(kpslice, u.Quantity):
                    kpslice = (kpslice.to(u.km / u.s)).value
                
                if (kpslice < 0):
                    axi[0].set_ylim(self.Kp_array[0], self.Kp_array[-1])   
                
                vrp_orb_slice = o.rv_theo_nu(kpslice, tr.nu * u.rad, tr.planet.w, plnt=True)
                ax1.plot(vrp_orb_slice[iout].value, tr.phase.value[iout], 
                 '.', color=(0.3,0, (k+1)/len(Kp_slice)), alpha=0.5, label=r'Kp = {:.2f}'.format(kpslice))
        
                axi[0].axhline(kpslice, linestyle='--', color=(0.3,0, (k+1)/len(Kp_slice)), 
                               label='Kp = {:.2f}'.format(kpslice))
                self.get_curve_at_slice(kpslice)
                print('Kp Slice = {:.2f} // SNR = {} // RV = {} '.format(kpslice, self.max, self.pos))

                axi[1].plot(self.interp_grid, self.snr.squeeze(), color=(0.3,0, (k+1)/len(Kp_slice)), 
                            label='Kp = {:.2f}'.format(kpslice), linestyle='--')
                
            
                
#         axi[0].legend(loc='lower left')
#         axi[1].legend(loc='lower left')
        if show_legend is True:
            ax1.legend(loc='best')
        figs.tight_layout(pad=1.0)
       
        pos = ax0.get_position()
        pos2 = ax2.get_position()
        ax2.set_position([pos.x0,pos2.y0,pos.width-0.02, pos2.height])
        
        if save_fig != '':
            figs.savefig('/home/boucher/spirou/Figures/CCF_'+save_fig+'.pdf')
#             fig.savefig('/home/boucher/spirou/Figures/STEPS'+fig_name+'.pdf')

    def fit_gauss_1d(self, Kp=None):
        
        if ~hasattr(self, 'snr'):
            if Kp is None:
                Kp = self.Kp0
            print('Taking slice at {} km/s'.format(Kp))
            self.get_curve_at_slice(Kp)
        
        plt.figure()
        plt.plot(self.interp_grid, self.snr)
        plt.plot(self.interp_grid[~self.idx_bruit_rv], self.snr[~self.idx_bruit_rv])
        max_val = self.snr[~self.idx_bruit_rv].max()
        
        try:
            param, pcov = curve_fit(a.gauss, self.interp_grid[~self.idx_bruit_rv], 
                                    self.snr[~self.idx_bruit_rv], p0=[3, 1, 0])
            err_param = np.sqrt(np.diag(pcov))
            print('Gauss fit : RV = {:.3f} +/- {:.3f} // Ampli = {:.3f}'.format(param[2], 
                                                                                err_param[2], param[1]))
            plt.plot(self.interp_grid[20:-20], a.gauss(self.interp_grid[20:-20], *param), 
                        color='c', label='Gaussian fit', alpha=0.5)
        except RuntimeError:
            print('Could not fit a gaussian')
            

        
#     def fit_gauss_1d_kp(self, rv=None):
        
#         if ~hasattr(self, 'pos'):
#             if rv is None:
#                 rv = self.pos
#             else:
#                 rv = 0
#             print('Taking slice at {} km/s'.format(rv))
#             self.snr_kp = self.snr_fct2d(rv, self.Kp_array)
        
        
#         pos_max = a.find_max_spline(self.Kp_array[~self.idx_bruit_kp], 
#                                   self.snr_kp[~self.idx_bruit_kp].copy(), kind='max')
        
#         self.pos_kp = pos_max[0]
#         self.max_kp = -pos_max[1]
        
#         plt.figure()
#         plt.plot(self.Kp_array, self.snr_kp)
#         plt.plot(self.Kp_array[~self.idx_bruit_kp], self.snr[~self.idx_bruit_kp])
#         max_val = self.snr_kp[~self.idx_bruit_kp].max()
        
#         try:
#             param, pcov = curve_fit(a.gauss, self.kp_array[~self.idx_bruit_kp], 
#                                     self.snr_kp[~self.idx_bruit_kp], p0=[3, 1, 0])
#             err_param = np.sqrt(np.diag(pcov))
#             print('Gauss fit : Kp = {:.3f} +/- {:.3f} // Ampli = {:.3f}'.format(param[2], 
#                                                                                 err_param[2], param[1]))
#             plt.plot(self.Kp_array[20:-20], a.gauss(self.Kp_array[20:-20], *param), 
#                         color='c', label='Gaussian fit', alpha=0.5)
#         except RuntimeError:
#             print('Could not fit a gaussian')

            
            
    def fit_gauss_2d(self, get_theta=False, limit_rv=10, limit_kp=70):
        
        abs_max_pos_kp = self.Kp_array[~self.idx_bruit_kp][self.idx_max[0]]
        abs_max_pos_rv = self.interp_grid[~self.idx_bruit_rv][self.idx_max[1]]

        idx_bruit_rv = (self.interp_grid < abs_max_pos_rv - limit_rv) | (self.interp_grid > abs_max_pos_rv + limit_rv)
        idx_bruit_kp = (self.Kp_array < abs_max_pos_kp - limit_kp) | \
                       (self.Kp_array > abs_max_pos_kp + limit_kp)

        x = self.interp_grid[~idx_bruit_rv]
        y = self.Kp_array[~idx_bruit_kp]

        fct_x = interp1d(np.arange(x.size),x, fill_value='extrapolate')
        fct_y = interp1d(np.arange(y.size),y, fill_value='extrapolate')

        Xin, Yin = np.mgrid[0:x.size, 0:y.size]

        data = self.snr2d[~idx_bruit_kp][:, ~idx_bruit_rv]
        max_val = data.max()

        params, err_params = a.fitgaussian(data, get_theta=get_theta)
        
        if get_theta is True:
            fit = a.gaussian2(*params)
        else:
            fit = a.gaussian(*params)

        xc, yc = fct_x(params[2]), fct_y(params[1])
#         sig_x, sig_y = np.abs(params[4] * np.diff(x)[0]), np.abs(params[3] * np.diff(y)[0])
        err_x = err_params[2]* np.diff(x)[0]
        err_y = err_params[1]* np.diff(y)[0]
    #         try:
        print('Gauss 2d fit : RV = {:.3f} +/- {:.3f} // Kp = {:.3f} +/- {:.3f} '.format(xc, err_x, \
                                                                                        yc, err_y))
        print('Amp = {:.2f} +/- {:.2f}'.format(params[0], err_params[0]))
        print('x0 = {:.2f} +/- {:.2f}'.format(params[2], err_params[2]))
        print('y0 = {:.2f} +/- {:.2f}'.format(params[1], err_params[1]))
        print('sig_x = {:.2f} +/- {:.2f}'.format(params[4], err_params[4]))
        print('sig_y = {:.2f} +/- {:.2f}'.format(params[3], err_params[3]))
        print('offset = {:.2f} +/- {:.2f}'.format(params[5], err_params[5]))
        
        fig, ax = plt.subplots(1,2, figsize=(10,6), sharey=True)  #plt.figure()
        ax[0].pcolormesh(x,y,data)
        ax[0].axhline(y[hm.nearest(y, self.Kp0)], linestyle='--', alpha=0.5)
        ax[0].axhline(yc, linestyle='-', alpha=0.7)
        ax[0].axvline(xc, linestyle='-', alpha=0.7)
        ax[1].pcolormesh(x,y,fit(*np.indices(data.shape)))
        ax[1].axhline(y[hm.nearest(y, self.Kp0)], linestyle='--', alpha=0.5)
        ax[1].axhline(yc, linestyle='-', alpha=0.7)
        ax[1].axvline(xc, linestyle='-', alpha=0.7)

        ax[0].set_xlabel(r'$v_{\rm offset}$ (km s$^{-1}$)')
        ax[0].set_ylabel(r'$K_{\rm P}$ (km s$^{-1}$)')
        ax[1].set_xlabel(r'$v_{\rm rad}$ (km s$^{-1}$)')
        
#         x1, y1 = centroid_com(data)
        x2, y2 = centroid_quadratic(data)
#         x3, y3 = centroid_2dg(data)

        marker = '+'
        ms, mew = 15, 2.
        ax[0].plot(fct_x(x2), fct_y(y2), color='white', marker=marker, ms=ms, mew=mew)
        ax[1].plot(fct_x(x2), fct_y(y2), color='white', marker=marker, ms=ms, mew=mew)

        plt.figure()
        plt.matshow(data, cmap=plt.cm.gist_earth_r, aspect=0.2)
        plt.axhline(hm.nearest(y, self.Kp0), linestyle='-', alpha=0.5)
    #   plt.axvline(hm.nearest(x, RV_sys.value), linestyle='-', alpha=0.5)

        plt.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper)
        plt.axhline(params[1], linestyle=':', alpha=0.5)
        plt.axvline(params[2], linestyle=':', alpha=0.5)
    #     plt.invert_yaxis()

        plt.figure()
        plt.matshow(data, cmap=plt.cm.gist_earth_r, aspect=0.2)
        plt.axhline(hm.nearest(y, self.Kp0), linestyle='-', alpha=0.5)
    #   plt.axvline(hm.nearest(x, RV_sys.value), linestyle='-', alpha=0.5)

        plt.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper)
        plt.axhline(params[1], linestyle=':', alpha=0.5)
        plt.axvline(params[2], linestyle=':', alpha=0.5)
    #     plt.invert_yaxis()

    
    def calc_ccf2d(self, tr, ccf=None, kind='logl_corr', id_pc=None, 
                   remove_mean=False, debug=False, index=None):
        if ccf is None:
            if kind == "shift":
                ccf = self.shifted_ccf
            elif kind == 'logl_corr':
                ccf = np.ma.masked_invalid(np.ma.sum(self.data, axis=1)).squeeze()
                print(ccf.shape)

            elif kind == "logl_sig":
                nolog_L_sig = np.ma.masked_invalid(np.ma.sum(self.data, axis=1)).squeeze()
                ccf = corr.nolog2log(nolog_L_sig, tr.N, sum_N=True).squeeze()
                remove_mean = True

        if remove_mean is True:     
            if ccf.ndim == 2:
                ccf = ccf-np.nanmedian(ccf,axis=1)[:,None]
            elif ccf.ndim == 3:
                ccf = ccf-np.nanmedian(ccf,axis=1)[:,None, :]

        if ccf.ndim == 3:
            if id_pc is None:
                ccf = ccf[:,:,0]
            else:
                ccf = ccf[:,:,id_pc]
    #                 ccf = ccf-np.nanmedian(ccf,axis=-1)[:,None]
    #         print(ccf.shape)
            
        if debug is True:
            plt.figure()
            plt.pcolormesh(ccf)
            
        if index is not None:
            ccf[index] = 0

        self.map_prf = ccf
#         print(ccf.mean())
        
        
        
        return ccf
    
    def plot_PRF(self, tr, interp_grid=None, ccf=None, RV=0., icorr=None, split_fig=[0], peak_center=None,
                     hlines=None, texts=None, kind='logl_corr', index=None, snr_1d=None, labels=None, clim=None, 
                     fig_name='', extension='.pdf', id_pc=None, map_kind='snr', debug=False, remove_mean=False,
                 minus_kp=False, figwidth=10):

#         if ccf is None:
#             if kind == "shift":
#                 ccf = self.shifted_ccf
#             elif kind == 'logl_corr':
#                 ccf = np.ma.masked_invalid(np.ma.sum(self.data, axis=1)).squeeze()
#     #                 if nolog_L.ndim == 2:
#     #                     ccf = 1/np.ma.sum(tr.N, axis=1)[:,None]*nolog_L
#     #                 elif nolog_L.ndim == 3:
#     #                     ccf = 1/np.ma.sum(tr.N, axis=1)[:,None,None]*nolog_L
#                 print(ccf.shape)

#             elif kind == "logl_sig":
#                 nolog_L_sig = np.ma.masked_invalid(np.ma.sum(self.data, axis=1)).squeeze()
#                 ccf = corr.nolog2log(nolog_L_sig, tr.N, sum_N=True).squeeze()
#                 remove_mean = True

#         if remove_mean is True:     
#             if ccf.ndim == 2:
#                 ccf = ccf-np.nanmedian(ccf,axis=1)[:,None]
#             elif ccf.ndim == 3:
#                 ccf = ccf-np.nanmedian(ccf,axis=1)[:,None, :]

#         if ccf.ndim == 3:
#             if id_pc is None:
#                 ccf = ccf[:,:,0]
#             else:
#                 ccf = ccf[:,:,id_pc]
#     #                 ccf = ccf-np.nanmedian(ccf,axis=-1)[:,None]
#     #         print(ccf.shape)
#         if debug is True:
#             plt.figure()
#             plt.pcolormesh(ccf)


#         self.map_prf = ccf
#         print(ccf.mean())

        ccf = self.calc_ccf2d(tr, ccf=ccf, kind=kind, id_pc=id_pc, 
                              remove_mean=remove_mean, index=index)
    
        
    
        if interp_grid is None:
            interp_grid = self.interp_grid
            
        if peak_center is not None:
            nb_pix = np.round(peak_center/2.3)
            peak_rv = np.arange(-nb_pix*2.3 + self.pos, (nb_pix+1)*2.3 + self.pos, 2.3)
            peak_ccf = np.ones((tr.n_spec, peak_rv.size))*np.nan

            for n in range(tr.n_spec):
                fct = interp1d(self.rv_grid, ccf[n])    
                peak_ccf[n] = fct(peak_rv)

            interp_grid = peak_rv
#             rv_grid = peak_rv
            ccf = peak_ccf
            idx_bruit_rv = (interp_grid < RV - 15) | (interp_grid > RV + 15)
        else:
            idx_bruit_rv = self.idx_bruit_rv 
#         if icorr is None:
#             icorr = np.arange(tr.n_spec)

        idx_mid = hm.nearest(interp_grid, RV)

        colum3 = np.ma.mean(ccf[:,idx_mid-1:idx_mid+2]/np.nanstd(ccf[:,idx_bruit_rv]),axis=-1)

        if index is not None:
            colum3[index] = np.nan
            colum3 = np.ma.masked_invalid(colum3)
        
        x = interp_grid

        if len(split_fig) > 1:
            y = []
            z = []
            id_range = []
            for i in range(len(split_fig))[1:][::-1]:

                id_range.append([split_fig[i-1],split_fig[i]])
    #             print(id_range)
                y.append(tr.phase[split_fig[i-1]:split_fig[i]]) #np.arange(tr.n_spec)[:split_fig]
                if map_kind == 'snr':
                    z.append(ccf[split_fig[i-1]:split_fig[i]]/ \
                             np.nanstd(ccf[:,idx_bruit_rv][split_fig[i-1]:split_fig[i]]))
                if map_kind == 'curve':
                    z.append(ccf[split_fig[i-1]:split_fig[i]])
        else:
            y=tr.phase  #np.arange(split_fig, tr.n_spec)-split_fig
            if map_kind == 'snr':
                z=ccf/np.nanstd(ccf[:,idx_bruit_rv])
            if map_kind == 'curve':
                z=ccf

        idx_mid = hm.nearest(interp_grid, RV)        
        colum3 = np.ma.mean(ccf[:,idx_mid-1:idx_mid+2]/np.nanstd(ccf[:,idx_bruit_rv]),axis=-1)
        bin_ccf = spectrum.box_binning(colum3, box_size=3)        

        if len(split_fig) > 1:

            height = []
            for i in range(len(split_fig)-1):
                height.append(np.sum(tr.dt[id_range[i][0]:id_range[i][1]]).to(u.h).value)
            height.append(height[0]/2)

            fig = plt.figure(constrained_layout=True, figsize=(figwidth,8))
            gs = fig.add_gridspec(len(split_fig), 3, width_ratios=[3, 3, 1.3], height_ratios=height)

            ax_map = []
            ax_ccf = []
            for i in range(len(split_fig)-1):
            #     print(i)
                curr_ax_map = fig.add_subplot(gs[i, :-1])
                ax_map.append(curr_ax_map)
                ax_ccf.append( fig.add_subplot(gs[i, -1], sharey=curr_ax_map))

            ax_snr = fig.add_subplot(gs[-1, :-1], sharex=ax_map[0])  
        else:
            fig = plt.figure(constrained_layout=True, figsize=(10,6))
            gs = fig.add_gridspec(2, 2, width_ratios=[3, 1.1], 
                                  height_ratios=[3, 1])
            ax_map = fig.add_subplot(gs[0, :-1])
            ax_ccf = fig.add_subplot(gs[0, -1], sharey=ax_map)
            ax_snr = fig.add_subplot(gs[-1, :-1], sharex=ax_map)

        berv_rv = -tr.vrp.value-tr.mid_vrp.value-tr.RV_sys-(tr.mid_vr.value+tr.vr.value)-(tr.mid_berv+tr.berv)

        if len(split_fig) > 1:
            for i in range(len(split_fig))[:-1]:
        #         print(i)
                im = ax_map[i].pcolormesh(x, y[i], z[i] , cmap='plasma', rasterized=True,
                                         vmin=np.nanpercentile(z[i],0.1), 
                                         vmax=np.nanpercentile(z[i],99.9))
                if index is not None:
                    idx_zero = np.where((z[i] == 0).all(axis=-1) == True)[0]
                    print(idx_zero)
                    if i == 0:
                        idx_end =idx_zero[-1]+2
                    else:
                        if idx_zero[-1] == len(y)-1:
                            idx_zero[-1]
                        else:
                            idx_end = idx_zero[-1]+1
                    print(idx_zero[0],idx_end)
                    ax_map[i].pcolormesh(x, y[i][idx_zero[0]:idx_end], z[i][idx_zero[0]:idx_end] , 
                                     cmap='nipy_spectral_r', rasterized=True,)
                try: 
                    id_out = tr.iOut[(tr.iOut>=id_range[i][0]) & (tr.iOut<id_range[i][1])]-id_range[i][0]
                    ax_map[i].plot(np.ones_like(y[i])[id_out]*RV, y[i][id_out], 'k.', alpha=0.5)
                except IndexError:
                    ax_map[i].axvline(RV, color='k', linestyle=':', alpha=0.8)

                ax_map[i].axvline(0, linestyle='-', color='black', alpha=0.6)
                if minus_kp is False:
                    ax_map[i].plot(berv_rv[id_range[i][0]:id_range[i][1]], y[i], '--', color='navy')
                else:
                    ax_map[i].plot(berv_rv[id_range[i][0]:id_range[i][1]][::-1], y[i], '--', color='navy')
                if clim is not None:
                    im.set_clim(clim[0],clim[1])
#                 ax_map[i].set_ylim(tr.phase.min(), tr.phase.max())
            ax_map[0].set_ylabel(r'Orbital Phase', fontsize=16, y=-0.2*(len(split_fig)-1))
            
        else:
            ax_map.set_ylabel(r'Orbital Phase', fontsize=16)
            im = ax_map.pcolormesh(x, y, z , cmap='plasma', rasterized=True)
            ax_map.plot(np.ones_like(y)[tr.iOut]*RV, y[tr.iOut],'k.', alpha=0.2)
            ax_map.axvline(0, linestyle=':', color='black', alpha=0.8)
            if minus_kp is False:
                ax_map.plot(berv_rv, y, '--', color='darkred')
            else:
                ax_map.plot(berv_rv[::-1], y, '--', color='darkred')
            if clim is not None:
                im.set_clim(clim[0],clim[1])

        if hlines is not None:
            for hline in hlines:
        #         print(hline)
                ax_map[hline[0]].axhline(hline[1], linestyle=hline[2], color=hline[3], alpha=0.99)
        if texts is not None:
            for text in texts:
                ax_map[text[0]].text(text[1], text[2], text[3], fontsize=16, 
                                     bbox ={'facecolor':'white', 'alpha':0.99})

        # ---- MEAN CCF ----
        if len(split_fig) > 1:
            for i in range(len(split_fig))[:-1]:
        #         print(i)    
                ax_ccf[i].plot(colum3[id_range[i][0]:id_range[i][1]], y[i],'.') 
                plt.setp(ax_ccf[i].get_yticklabels(), visible=False)
                ax_ccf[i].set_xlim(np.nanmin(colum3)*1.5, np.nanmax(colum3)*1.5)
                ax_ccf[i].axvline(0, linestyle=':', color='k', alpha=0.5)
                crossing_cond = (berv_rv[id_range[i][0]:id_range[i][1]] >= RV-2.3) &\
                                (berv_rv[id_range[i][0]:id_range[i][1]] <= RV+2.3)
                

                ax_ccf[i].set_xlabel(r'$\overline{\rm CCF}$', fontsize=13)

                ax_ccf[i].plot(bin_ccf[id_range[i][0]:id_range[i][1]], y[i],'-', color='darkorange', alpha=1) 
                ax_ccf[i].plot(colum3[id_range[i][0]:id_range[i][1]][crossing_cond], y[i][crossing_cond],
                               '.', color='lightgrey')
#                 ax_ccf[i].set_ylim(y[i][0]-np.diff(y[i])[0], y[i][-1]+np.diff(y[i])[-1])
#                 ax_ccf[i].set_ylim(tr.phase.min(), tr.phase.max())
        else:
            ax_ccf.plot(colum3, y,'.') 
            plt.setp(ax_ccf.get_yticklabels(), visible=False)
            ax_ccf.set_xlim(np.nanmin(colum3)*1.5, np.nanmax(colum3)*1.5)
            ax_ccf.axvline(0, linestyle=':', color='k', alpha=0.5)
            crossing_cond = (berv_rv >= RV-2.3) & (berv_rv <= RV+2.3)
            ax_ccf.plot(colum3[crossing_cond], y[crossing_cond],'.', color='grey')

            ax_ccf.set_xlabel(r'$\overline{\rm CCF}$', fontsize=13)

            ax_ccf.plot(bin_ccf, y,'-', color='darkorange', alpha=1) 
#             ax_ccf.set_ylim(y[0]-np.diff(y)[0], y[-1]+np.diff(y)[-1])

        if hlines is not None:
            for hline in hlines:
        #         print(hline)
                ax_ccf[hline[0]].axhline(hline[1], linestyle=hline[2], color='black', alpha=0.8)


        # ---- SNR ---- 

        if snr_1d is None:
            labels = [None]
            if id_pc is not None:
                snr_1d = [self.npc_snr[id_pc]]
            else:
                snr_1d = [self.snr]

        for i,snr_i in enumerate(snr_1d):
            ax_snr.plot(interp_grid, snr_i, label=labels[i])

        ax_snr.set_xlabel(r'$v_{\rm rad}$ (km s$^{-1}$)', fontsize=16)
        ax_snr.axhline(0, linestyle=':', color='k', alpha=0.5)
        ax_snr.set_ylabel(r'S/N', fontsize=13)
        if len(labels) > 1:
            ax_snr.legend(loc='best')

        if len(split_fig) > 1:
            for i in range(len(split_fig))[:-1]:
                plt.setp(ax_map[i].get_xticklabels(), visible=False)
                ax_map[i].tick_params(axis='both', which='major', labelsize=12)
                ax_ccf[i].tick_params(axis='both', which='major', labelsize=12)
            plt.setp(ax_map[0].get_xticklabels(), visible=False)
        else:
            plt.setp(ax_map.get_xticklabels(), visible=False)
        ax_snr.tick_params(axis='both', which='major', labelsize=12)


        fig.subplots_adjust(hspace=0.05, wspace=0.03)

        if fig_name != '':
            fig.savefig('/home/boucher/spirou/Figures/fig_CCF_2D_'+fig_name+'.pdf', rasterize=True) 

        
        # --- overplot all CCF ---
        if len(split_fig) > 1:
            ymin = np.min(tr.phase)
            ymax = np.max(tr.phase)
            diff_y = [np.diff(y[i])[0] for i in range(len(split_fig))[:-1]]
            print(ymin,ymax)
            print(diff_y)
            print(np.mean(diff_y))
            
            
            common_y = np.arange(ymin,ymax,np.mean(diff_y))
            ccf_interp = []
            for i in range(len(split_fig))[:-1]:
                fct_ccf2d = interp2d(x, y[i], z[i], fill_value=0)
                ccf_interp.append(fct_ccf2d(x, common_y))
            fig2 = plt.figure()
            plt.pcolormesh(x, common_y, np.array(ccf_interp).sum(axis=0) , cmap='plasma', rasterized=True)
            plt.xlabel(r'$v_{\rm rad}$ (km s$^{-1}$)', fontsize=16)
            plt.ylabel(r'Orbital Phase', fontsize=16)
            plt.axhline(tr.phase[tr.iIn[0]], color='white',linestyle=':')
            plt.axhline(tr.phase[tr.iIn[-1]], color='white',linestyle=':')
            plt.axvline(0, color='black',linestyle=':', alpha=0.7)
            
        
        
#     def plot_PRF(self, tr, interp_grid=None, ccf=None, RV=0., icorr=None, split_fig=0,
#                  hlines=None, texts=None, kind='shift', index=None, snr_1d=None, labels=None, clim=None, 
#                  fig_name='', extension='.pdf', id_pc=None, map_kind='snr', debug=False, remove_mean=False):
            
#         if ccf is None:
#             if kind == "shift":
#                 ccf = self.shifted_ccf
#             elif kind == 'logl_corr':
#                 ccf = np.ma.masked_invalid(np.ma.sum(self.data, axis=1)).squeeze()
# #                 if nolog_L.ndim == 2:
# #                     ccf = 1/np.ma.sum(tr.N, axis=1)[:,None]*nolog_L
# #                 elif nolog_L.ndim == 3:
# #                     ccf = 1/np.ma.sum(tr.N, axis=1)[:,None,None]*nolog_L
#                 print(ccf.shape)

#             elif kind == "logl_sig":
#                 nolog_L_sig = np.ma.masked_invalid(np.ma.sum(self.data, axis=1)).squeeze()
#                 ccf = corr.nolog2log(nolog_L_sig, tr.N, sum_N=True).squeeze()
#                 remove_mean = True
                
#         if remove_mean is True:     
#             if ccf.ndim == 2:
#                 ccf = ccf-np.nanmedian(ccf,axis=1)[:,None]
#             elif ccf.ndim == 3:
#                 ccf = ccf-np.nanmedian(ccf,axis=1)[:,None, :]
        
#         if ccf.ndim == 3:
#             if id_pc is None:
#                 ccf = ccf[:,:,0]
#             else:
#                 ccf = ccf[:,:,id_pc]
# #                 ccf = ccf-np.nanmedian(ccf,axis=-1)[:,None]
# #         print(ccf.shape)
#         if debug is True:
#             plt.figure()
#             plt.pcolormesh(ccf)
        
#         if interp_grid is None:
#             interp_grid = self.interp_grid

#         if icorr is None:
#             icorr = np.arange(tr.n_spec)

#         idx_mid = hm.nearest(interp_grid, RV)
        
#         colum3 = np.ma.mean(ccf[:,idx_mid-1:idx_mid+2]/np.nanstd(ccf[:,self.idx_bruit_rv]),axis=-1)
        
#         if index is not None:
#             ccf[index] = 0
#             colum3[index] = np.nan
#             colum3 = np.ma.masked_invalid(colum3)
#     #     idx_0 = hm.nearest(interp_grid, 0.)
#     #     colum0 = np.ma.mean(ccf[:,idx_0-1:idx_0+2],axis=-1)
        
#         self.map_prf = ccf
#         print(ccf.mean())
# #         fig = plt.figure() 
# #         fig.set_figheight(6) 
# #         fig.set_figwidth(10) 

#         if split_fig > 0:
 
#             fig = plt.figure(constrained_layout=True, figsize=(10,8))
#             gs = fig.add_gridspec(3, 3, width_ratios=[3,3, 1.1], 
#                                   height_ratios=[3*(tr.n_spec-split_fig)*np.diff(tr.phase[split_fig:])[0]/tr.n_spec,
#                                                  3*split_fig*np.diff(tr.phase[:split_fig])[0]/tr.n_spec, 
#                                                  1*np.diff(tr.phase[:split_fig])[0]])
#             ax02 = fig.add_subplot(gs[0, :-1])
#             ax01 = fig.add_subplot(gs[1, :-1], sharex=ax02)
#             ax12 = fig.add_subplot(gs[0, -1], sharey=ax02)
#             ax11 = fig.add_subplot(gs[1, -1], sharey=ax01)
#             ax2 = fig.add_subplot(gs[-1, :-1], sharex=ax02)
        
#         else:
#             fig = plt.figure(constrained_layout=True, figsize=(10,8))
#             gs = fig.add_gridspec(2, 2, width_ratios=[3, 1.1], 
#                                   height_ratios=[3, 1])
#             ax02 = fig.add_subplot(gs[0, :-1])
#             ax12 = fig.add_subplot(gs[0, -1], sharey=ax02)
#             ax2 = fig.add_subplot(gs[-1, :-1], sharex=ax02)


#         x = interp_grid
#         y2 = tr.phase[split_fig:]  #np.arange(split_fig, tr.n_spec)-split_fig
#         if map_kind == 'snr':
#             z2 = ccf[split_fig:]/np.nanstd(ccf[:,self.idx_bruit_rv][split_fig:])
#         if map_kind == 'curve':
#             z2 = ccf[split_fig:]
        
#         if split_fig > 0:
#             y1 = tr.phase[:split_fig] #np.arange(tr.n_spec)[:split_fig]
#             if map_kind == 'snr':
#                 z1 = ccf[:split_fig]/np.nanstd(ccf[:,self.idx_bruit_rv][:split_fig])
#             if map_kind == 'curve':
#                 z1 = ccf[:split_fig]
        
        
#         berv_rv = -tr.vrp.value-tr.mid_vrp.value-tr.RV_sys-(tr.mid_vr.value+tr.vr.value)-(tr.mid_berv+tr.berv)
#         im2 = ax02.pcolormesh(x, y2, z2 , cmap='plasma', rasterized=True)
#         ax02.plot(np.ones_like(y2)[tr.iOut[tr.iOut>=split_fig]-split_fig]*RV,
#                  y2[tr.iOut[tr.iOut>=split_fig]-split_fig],'k.', alpha=0.2)
#         ax02.set_ylabel(r'Exposure Number', fontsize=16)
#         ax02.axvline(0, linestyle=':', color='black', alpha=0.8)
#         ax02.plot(berv_rv[split_fig:], y2, '--', color='darkred')
        
#         if split_fig > 0:
#             ax02.set_ylabel(r'Orbital Phase', fontsize=16, y=-0.00)
#             im1 = ax01.pcolormesh(x, y1, z1 , cmap='plasma', rasterized=True)
#             ax01.plot(np.ones_like(y1)[tr.iOut[tr.iOut<split_fig]]*RV,
#                      y1[tr.iOut[tr.iOut<split_fig]],'k.', alpha=0.2)
#             ax01.axvline(0, linestyle=':', color='black', alpha=0.8)
#             ax01.plot(berv_rv[:split_fig], y1, '--', color='darkred')
            

#         if clim is not None:
#             if split_fig > 0:
#                 im1.set_clim(clim[0],clim[1])
#             im2.set_clim(clim[0],clim[1])

# #         ax0.axhline(tr.iIn[0], linestyle=':', color='white', alpha=0.8)
# #         ax0.axhline(tr.iIn[-1]+1, linestyle=':', color='white', alpha=0.8)

#         if hlines is not None:
#             for hline in hlines:
#                 if split_fig > 0:
#                     if hline[-1] == 1:
#                         ax01.axhline(hline[0], linestyle=hline[1], color=hline[2], alpha=0.8)
#                 if hline[-1] == 2:
#                     ax02.axhline(hline[0], linestyle=hline[1], color=hline[2], alpha=0.8)
#         if texts is not None:
#             for text in texts:
#                 if split_fig > 0:
#                     if text[-1] == 1:
#                         ax01.text(text[0], text[1], text[2], bbox ={'facecolor':'white', 'alpha':0.8})
#                 if text[-1] == 2:
#                     ax02.text(text[0], text[1], text[2], bbox ={'facecolor':'white', 'alpha':0.8})
            

#         # ---- MEAN CCF ----
# #         ax1 = fig.add_subplot(spec[1], sharey=ax0) 
#     #     ax1.plot(colum0[icorr], np.arange(tr.n_spec)[icorr],'ko-', alpha=0.3) 

    
#         ax12.plot(colum3[split_fig:], y2,'.') 
#         plt.setp(ax12.get_yticklabels(), visible=False)
#         ax12.set_xlim(np.nanmin(colum3[icorr])*1.5, np.nanmax(colum3[icorr])*1.5)
#         ax12.axvline(0, linestyle=':', color='k', alpha=0.5)
#         crossing_cond = (berv_rv[split_fig:] >= RV-2.3) & (berv_rv[split_fig:] <= RV+2.3)
#         ax12.plot(colum3[split_fig:][crossing_cond], y2[crossing_cond],'r.')

#         if split_fig > 0:
#             ax11.plot(colum3[:split_fig], y1,'.') 
#             plt.setp(ax11.get_yticklabels(), visible=False)
#             ax11.set_xlim(np.nanmin(colum3[icorr])*1.5, np.nanmax(colum3[icorr])*1.5)
#             ax11.axvline(0, linestyle=':', color='k', alpha=0.5)
#             crossing_cond = (berv_rv[:split_fig] >= RV-2.3) & (berv_rv[:split_fig] <= RV+2.3)
#             ax11.plot(colum3[:split_fig][crossing_cond], y1[crossing_cond],'r.')

#             ax11.set_xlabel(r'$\overline{\rm CCF}$', fontsize=13)
#         else:
#             ax12.set_xlabel(r'$\overline{\rm CCF}$', fontsize=13)

# #         ax1.axhline(tr.iIn[0], linestyle=':', color='black', alpha=0.8)
# #         ax1.axhline(tr.iIn[-1]+1, linestyle=':', color='black', alpha=0.8)
        
#         if hlines is not None:
#             for hline in hlines:
#                 if split_fig > 0:
#                     if hline[-1] == 1:
#                         ax11.axhline(hline[0], linestyle=hline[1], color='black', alpha=0.8)
#                 if hline[-1] == 2:
#                     ax12.axhline(hline[0], linestyle=hline[1], color='black', alpha=0.8)

#         bin_ccf = spectrum.box_binning(colum3[icorr], box_size=3)
#         ax12.plot(bin_ccf[split_fig:], y2,'-', color='darkorange', alpha=1) 
#         ax12.set_ylim(y2[0]-np.diff(y2)[0], y2[-1]+np.diff(y2)[-1])
        
#         if split_fig > 0:
#             ax11.plot(bin_ccf[:split_fig], y1,'-', color='darkorange', alpha=1) 
#             ax11.set_ylim(y1[0]-np.diff(y1)[0], y1[-1]+np.diff(y1)[-1])
        
        
#         # ---- SNR ---- 
        
#         if snr_1d is None:
#             labels = [None]
#             if id_pc is not None:
#                 snr_1d = [self.npc_snr[id_pc]]
#             else:
#                 snr_1d = [self.snr]
            
# #         ax2 = fig.add_subplot(spec[2], sharex=ax0)
#     #     ax2.plot(interp_grid, np.nansum(ccf[tr.icorr], axis=0))
#         for i,snr_i in enumerate(snr_1d):
#             ax2.plot(interp_grid, snr_i, label=labels[i]) 
#         ax2.set_xlabel(r'$v_{\rm rad}$ (km s$^{-1}$)', fontsize=16)
#         ax2.axhline(0, linestyle=':', color='k', alpha=0.5)
#         ax2.set_ylabel(r'SNR', fontsize=13)
#         if len(labels) > 1:
#             ax2.legend(loc='best')

#         plt.setp(ax02.get_xticklabels(), visible=False)
#         ax2.tick_params(axis='both', which='major', labelsize=12)
        
#         if split_fig > 0:
#             plt.setp(ax01.get_xticklabels(), visible=False)

#             ax01.tick_params(axis='both', which='major', labelsize=12)
#             ax11.tick_params(axis='both', which='major', labelsize=12)
        
        
#         fig.subplots_adjust(hspace=0.05, wspace=0.03)
        
#         if fig_name != '':
#             fig.savefig('/home/boucher/spirou/Figures/CCF_2D_'+fig_name+'.pdf', rasterize=True) 
            
        
            
    def ccf_map_plot(self, tr, fit_gauss=False, fig_larg=8, fig_haut=3, cmap='plasma', 
                     Kp_slice=None, clim=None, save_fig='', map2d=None, minmax='max', label_curve='All Tr',
                    snr_1d=None, labels=None, force_max_pos=None, fig_name='', tag_max=False):
        
        if map2d is None:
            map2d = self.snr2d

        figs, axi = plt.subplots(2, 1, figsize=(fig_larg, 2 * fig_haut), sharex=True)

        # --- Kp vs vrad map 2D --- 

        imaxi2 = axi[0].pcolormesh(self.interp_grid, self.Kp_array, map2d, cmap=cmap, rasterized=True)
        axi[0].set_ylabel(r'$K_{\rm P}$ (km s$^{-1}$)', fontsize=14)
        
        if fig_name != '':
            axi[0].set_title(fig_name, fontsize=16)

        self.get_curve_at_slice(self.Kp_array[~self.idx_bruit_kp][self.idx_max[0]], minmax=minmax)
        print('Highest SNR = {} // Kp = {} // RV = {} '.format(self.max, 
                                                  self.Kp_array[~self.idx_bruit_kp][self.idx_max[0]],
                                                               self.pos))
        self.get_curve_at_slice(tr.Kp, minmax=minmax)
        print(r'Max SNR = {:.2f}$\sigma$, Max position = {:.2f}'.format(self.max, self.pos))
        print('')
        hm.printmd(r'Max SNR = **{:.2f}**$\sigma$, Max position = {:.2f}'.format(self.max, self.pos))
        print('')
        
        if force_max_pos is None:
            pos_max = self.pos
        else:
            pos_max = force_max_pos
        
            # - Lines enclosing the maximum -
        axi[0].axhline(tr.Kp.to(u.km / u.s).value, linestyle='-', alpha=0.7, color='indigo', 
                      xmin=0,xmax=hm.nearest(self.interp_grid, pos_max-10)/self.interp_grid.size )
        axi[0].axhline(tr.Kp.to(u.km / u.s).value, linestyle='-', alpha=0.7, color='indigo',
                      xmin=hm.nearest(self.interp_grid, pos_max+10)/self.interp_grid.size ,xmax=1) 

#         axi[0].axvline(self.pos, linestyle='-', alpha=0.7, color='indigo', 
#                           ymin=0, ymax=(self.Kp_array[~self.idx_bruit_kp][self.idx_max[0]]-50)/self.Kp_array[-1] )
#         axi[0].axvline(self.pos, linestyle='-', alpha=0.7, color='indigo',
#                           ymin=(self.Kp_array[~self.idx_bruit_kp][self.idx_max[0]]+50)/self.Kp_array[-1] , ymax=1) 
        axi[0].axvline(pos_max, linestyle='-', alpha=0.7, color='indigo', 
                          ymin=0, ymax=(tr.Kp.value - 60)/self.Kp_array[-1] )
        axi[0].axvline(pos_max, linestyle='-', alpha=0.7, color='indigo',
                          ymin=(tr.Kp.value + 60)/self.Kp_array[-1] , ymax=1) 

        axi[0].axvline(0, color='indigo', linestyle=':')
        axi[0].set_ylim(0, self.Kp_array[-1])

        divider = make_axes_locatable(axi[0])
        cax = divider.append_axes('right', size='2%', pad=0.05)
        cbar = figs.colorbar(imaxi2, ax=axi[0], cax=cax)
        if clim is not None:
            imaxi2.set_clim(clim[0],clim[1])

        cbar.set_label('SNR', fontsize=14)
        
        if tag_max is True:
            axi[0].plot(self.pos, tr.Kp.to(u.km / u.s).value,'k+', 
                label='Pos = {:.2f} // Max = {:.2f}'.format(self.pos, self.max))
            axi[0].legend(loc='best')

        ### --- Sum CCF 1D slice --- 
        
        axi[1].plot(self.interp_grid, self.snr.squeeze(), color='indigo', label=label_curve)
        
        axi[1].axvline(0, color='indigo', alpha=0.5, linestyle=':',)# label=r"Planet Rest frame")
        axi[1].axhline(0, color='indigo', linestyle=':', alpha=0.5)

        axi[1].set_ylabel('SNR', fontsize=14)
        axi[1].set_xlabel(r'$v_{\rm rad}$ (km s$^{-1}$)', fontsize=14)
        
        axi[1].axvline(pos_max, linestyle='-', alpha=0.7, color='indigo')

        if clim is not None:
            axi[1].set_ylim(clim[0],clim[1])

        if Kp_slice is not None:
            
            if isinstance(Kp_slice,list):
                Kp_slice = np.array(Kp_slice)
            else:
                Kp_slice = [Kp_slice]

            for k,kpslice in enumerate(Kp_slice):
                if isinstance(kpslice, u.Quantity):
                    kpslice = (kpslice.to(u.km / u.s)).value
                
                if kpslice < self.Kp_array.min():
                    continue
                
                if (kpslice < 0):
                    axi[0].set_ylim(self.Kp_array[0], self.Kp_array[-1])   
                
#                 vrp_orb_slice = o.rv_theo_nu(kpslice, tr.nu * u.rad, tr.planet.w, plnt=True)
#                 ax1.plot(vrp_orb_slice[iout].value, np.arange(tr.vrp.size)[iout], 
#                  '.', color=(0.3,0, (k+1)/len(Kp_slice)), alpha=0.5, label=r'Kp = {:.2f}'.format(kpslice))
        
                axi[0].axhline(kpslice, linestyle='--', color=(0.3,0, (k+1)/len(Kp_slice)), 
                               label='Kp = {:.2f}'.format(kpslice))
                self.get_curve_at_slice(kpslice, minmax=minmax)
                print('Kp Slice = {:.2f} // SNR = {} // RV = {} '.format(kpslice, self.max, self.pos))

                axi[1].plot(self.interp_grid, self.snr.squeeze(), color=(0.3,0, (k+1)/len(Kp_slice)), 
                            label='Kp = {:.2f}'.format(kpslice), linestyle='--')
                
#         if wind is not None:
#             if wind != 0 :
#                 axi[0].axvline(wind, linestyle=':', color='dodgerblue', label=r'Wind')
#                 axi[1].axvline(wind, linestyle=':', color='dodgerblue', label=str(wind) + ' km/s Wind')            
                
#         if show_legend is True:
#             ax1.legend(loc='best')
        figs.tight_layout(pad=1.0)
       
        pos = axi[0].get_position()
        pos2 = axi[1].get_position()
        axi[1].set_position([pos.x0,pos2.y0,pos.width-0.02, pos2.height])
        
        if snr_1d is not None:
            for i,snr_i in enumerate(snr_1d):
                axi[1].plot(self.interp_grid, snr_i, label=labels[i], alpha=0.6) 

            axi[1].legend(loc='best')
        
        if save_fig != '':
            figs.savefig('/home/boucher/spirou/Figures/fig_CCF_map_'+save_fig+'.pdf')#, rasterize=True)            

    def ttest_value(self, tr, orders=np.arange(49), wind=None, vrp=None, plot=True, 
                    kind='corr', speed_limit=2.5, ccf0=None, peak_center=None, verbose=True,**kwargs):
        if wind is None:
            wind = self.pos
        if kind == 'corr':
            if vrp is None:
                vrp = tr.vrp.value+tr.RV_const+tr.mid_vrp.value
            ccf = self.ccf0
        elif kind == 'logl':
            if vrp is None:
                vrp = np.zeros_like(tr.vrp.value)#+tr.RV_const+tr.mid_vrp.value
            ccf = self.map_prf
        
        if ccf0 is not None:
            ccf = ccf0.copy()
            
        if peak_center is not None:
            nb_pix = np.round(peak_center/2.3)
            peak_rv = np.arange(-nb_pix*2.3 + wind, (nb_pix+1)*2.3 + wind, 2.3)
            peak_ccf = np.ones((tr.n_spec, peak_rv.size))*np.nan

            for n in range(tr.n_spec):
                fct = interp1d(self.rv_grid, ccf[n])    
                peak_ccf[n] = fct(peak_rv)

            (t_in, p_in), (t_out, p_out) = nf.single_t_test(tr, peak_rv, peak_ccf, orders, wind=wind, 
                                                            ccf=peak_ccf.copy(), vrp=vrp,
                                                            plot=plot, speed_limit=speed_limit,
                                                            verbose=verbose, **kwargs) 
        else:
            (t_in, p_in), (t_out, p_out) = nf.single_t_test(tr, self.rv_grid, ccf.copy(), orders, wind=wind, 
                                                        ccf=ccf.copy(), vrp=vrp, plot=plot, speed_limit=speed_limit, 
                                                            verbose=verbose, **kwargs) 
        if verbose is True:
            print('In-transit t-val = {:.2f} / p-val {:.2e} / sig = {:.2f}'.format(t_in, p_in, nf.pval2sigma(p_in)))
            print('Out-of-transit t-val = {:.2f} / p-val {:.2e} / sig = {:.2f}'.format(t_out, p_out, nf.pval2sigma(p_out)))
        self.ttest_val = t_in
        
    def ttest_map(self, tr, kind='corr', vrp=None, orders=np.arange(49), 
                  kp0=0, RV_limit=100, kp_step=5, rv_step=1, RV=None, 
                  fig_name='', speed_limit=2.5, ccf=None, peak_center=None, hist=True,**kwargs):
        
        if kind == 'corr':
            if ccf is None:
                ccf = self.ccf0
            prf = False
            
            if RV is None:
                RV = tr.RV_const
#             if vrp is None:
#                 vrp = tr.vrp.value+tr.RV_const+tr.mid_vrp.value
        elif kind == 'logl':
            if ccf is None:
                ccf = self.map_prf
            prf = True
            
            if RV is None:
                RV = 0
#             if vrp is None:
#                 vrp = np.zeros_like(tr.vrp.value)+tr.RV_const #+tr.mid_vrp.value

        if peak_center is not None:
            nb_pix = np.round(peak_center/2.3)
            peak_rv = np.arange(-nb_pix*2.3 + self.pos, (nb_pix+1)*2.3 + self.pos, 2.3)
            peak_ccf = np.ones((tr.n_spec, peak_rv.size))*np.nan

            for n in range(tr.n_spec):
                fct = interp1d(self.rv_grid, ccf[n])    
                peak_ccf[n] = fct(peak_rv)
            rv_grid = peak_rv
            ccf = peak_ccf
            print(peak_rv[[0,-1]])
            RV_array = rv_grid
        else:
            rv_grid = self.rv_grid
            RV_array = None
        
        Kp_array, RV_array, \
        t_value, p_value, ttest_params = nf.ttest_map(tr, rv_grid, ccf.copy(),  ccf=ccf.copy(),
                                                      orders=orders, kp0=kp0, RV_limit=RV_limit,
                                                      kp_step=kp_step, rv_step=rv_step, RV=RV, RV_array=RV_array,
                                                      prf=prf, speed_limit=speed_limit, wind=self.pos, **kwargs)
        
        self.ttest_map_data = [Kp_array, RV_array, t_value, ttest_params]
        self.ttest_map_kp = Kp_array
        self.ttest_map_rv = RV_array
        self.ttest_map_tval = t_value
        self.ttest_map_params = ttest_params
        
        if Kp_array.size > 1 :
            self.plot_ttest_map(tr, vrp=vrp, kind=kind, orders=orders, fig_name=fig_name, hist=hist)
        
        
        
    def plot_ttest_map(self, tr, 
                       RV=None, kind='corr', vrp=None, orders=np.arange(49), fig_name='', hist=True):
        
        Kp_array, RV_array, t_value, ttest_params = self.ttest_map_kp, self.ttest_map_rv, \
                                                    self.ttest_map_tval, self.ttest_map_params
        
        if RV is None:
            RV = self.pos
        
        if kind == 'corr':
            ccf = self.ccf0
            prf = False
            if vrp is None:
                vrp = tr.vrp.value+tr.RV_const+tr.mid_vrp.value
                
        elif kind == 'logl':
            ccf = self.map_prf
            prf = True
            if vrp is None:
                vrp = np.zeros_like(tr.vrp.value)+tr.RV_const #+tr.mid_vrp.value
                
        t_in, p_in = pf.plot_ttest_map_hist(tr, self.rv_grid, ccf.copy(), Kp_array, RV_array, t_value, ttest_params, 
                               orders=orders, plot_trail=True, masked=True, ccf=ccf.copy(),
                              vrp=vrp, RV=RV, fig_name=fig_name, hist=hist)
        print(t_in, p_in, nf.pval2sigma(p_in))
        
def plot_ccf_timeseries(t, rv_star, correlation, plot_gauss=True, plot_spline=True, x0_estim=0,
                    orders=np.arange(49), rv_limit=30, berv=False, RV=0, limit_rv=2, iOrd=None, plot=True):

    val_ga =[]
    pos_ga =[]
    val_sp =[]
    pos_sp =[]
    pos_err_ga = []

    tc = Correlations(correlation.copy(), rv_grid=rv_star, kind='correl')
    if iOrd is None:
        tc.calc_ccf(orders=orders, N=t.N_frac[None,:,None]**2, index=None, alpha=np.ones_like(t.alpha_frac))
    else:
        tc.ccf0 = tc.data[:,iOrd]
    if plot is True:
        fig, ax = plt.subplots(1,3, figsize=(14,5))  # plt.figure()
    for n in range(t.n_spec):
    #     for iord in range(49):
        if plot is True:
            ax[0].pcolormesh(rv_star[(rv_star >=-limit_rv)&(rv_star<= limit_rv)],
                         np.arange(t.n_spec), 
                         tc.ccf0[:,(rv_star >=-limit_rv)&(rv_star<= limit_rv)])

        try:
            tc.get_snr_1d(interp_grid=rv_star, courbe=tc.ccf0[n], 
                          rv_limit=rv_limit, RV_sys=x0_estim, plot=False)
            tc.find_max(interp_grid=rv_star, snr=tc.ccf0[n], kind_max='gauss')
            val_ga.append(tc.max)
            pos_ga.append(tc.pos)
            pos_err_ga.append(tc.pos_err)
            if plot is True:
                ax[2].axvline(tc.pos)
        except ValueError:
            val_ga.append(np.nan)
            pos_ga.append(np.nan)
            pos_err_ga.append(np.nan)
        if plot is True:
            ax[2].plot(rv_star,tc.ccf0[n],'k')
            ax[2].plot(rv_star[~tc.idx_bruit_rv],tc.ccf0[n][~tc.idx_bruit_rv],'r')
        try:    
            tc.find_max(interp_grid=rv_star, snr=tc.ccf0[n], kind_max='spline')
            val_sp.append(tc.max)
            pos_sp.append(tc.pos)
            if plot is True:
                ax[2].axvline(tc.pos, color='g')
        except ValueError:
            val_sp.append(np.nan)
            pos_sp.append(np.nan)
    val_ga = np.ma.masked_invalid(val_ga)
    pos_ga = np.ma.masked_invalid(pos_ga) 
    pos_err_ga = np.ma.masked_invalid(pos_err_ga) 
    val_sp = np.ma.masked_invalid(val_sp)
    pos_sp = np.ma.masked_invalid(pos_sp) 
    if plot is True:
        ax[0].plot(np.array(pos_ga), np.arange(t.n_spec),'x', color='white')
        ax[0].plot(np.array(pos_sp), np.arange(t.n_spec),'rx')
    #     ax[0].set_xlim(-2,2)
        ax[0].axvline(0, linestyle=':', alpha=0.5, color='k')

    #     plt.figure()
        if plot_gauss is True:
            ax[1].errorbar(t.phase, pos_ga, yerr=pos_err_ga, marker='o')
        if plot_spline is True:
            ax[1].plot(t.phase, pos_sp,'go-')
        if berv is True:
            ax[1].plot(t.phase, t.berv+t.vr.value+RV,color='orange')
        ax[1].axvline(t.phase[t.iIn[0]], color='k', alpha=0.3, linestyle='--')
        ax[1].axvline(t.phase[t.iIn[-1]], color='k', alpha=0.3, linestyle='--')
        ax[1].axvline(0, color='k', alpha=0.3, linestyle=':')
    

    return tc, pos_ga, pos_err_ga, pos_sp