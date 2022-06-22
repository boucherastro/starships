
import emcee
import h5py

import corner
import numpy as np

import pymc3
# import pygtc

from starships import homemade as hm
from starships import analysis as a
from starships import spectrum as spectrum
import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d
import astropy.units as u
import astropy.constants as const

from petitRADTRANS import Radtrans

from collections import OrderedDict
import starships.petitradtrans_utils as prt
from petitRADTRANS import nat_cst as nc
# import numpy as np

# bin_down = np.array([11225,11409,11594,11779,11963,12148,12333,12517,12702,12887,13071,13256,13441,13625,13810,13995,14179,14364,14549,14733,14918,15102,15287,15472,15656,15841,16026,16210])/1e4

# bin_up = np.array([11409, 11594, 11779, 11963, 12148, 12333, 12517, 12702, 12887, 13071, 13256, 13441, 13625, 13810, 13995, 14179, 14364, 14549, 14733, 14918, 15102, 15287, 15472, 15656, 15841, 16026, 16210, 16395])/1e4

# HST_wave = (bin_up - bin_down) / 2 + bin_down

# TD = np.array([0.993,0.995,1.001,0.991,1.001,0.991,0.992,1.000,0.993,0.997,1.010,1.016,1.054,1.051,1.052,1.046,1.053,1.046,1.042,1.031,1.028,1.012,0.999,1.001,0.983,0.986,0.975,0.967])

# HST_data = TD/100

# TD_err_up = np.array([0.009,0.008,0.009,0.010,0.008,0.008,0.008,0.012,0.008,0.009,0.007,0.008,0.009,0.011,0.008,0.007,0.007,0.010,0.009,0.009,0.009,0.010,0.010,0.010,0.009,0.009,0.012,0.013])/100

# TD_err_down = np.array([0.009,0.007,0.010,0.009,0.009,0.008,0.007,0.011,0.008,0.009,0.008,0.008,0.010,0.011,0.008,0.007,0.006,0.011,0.009,0.010,0.012,0.010,0.009,0.011,0.009,0.009,0.011,0.012])/100

# HST_data_err = np.sqrt(TD_err_up**2+TD_err_down**2)


# bin_down_vis = np.array([2898, 3700, 4041, 4261, 4426, 4536, 4646, 4756, 4921, 5030, 5140, 5250, 5360, 5469, 5579, 5500, 5600, 5700, 5800, 5878, 5913, 6070, 6200, 6300, 6450, 6600, 6800, 7000, 7200, 7450, 7645, 7720, 8100, 8485, 8985])/1e4

# bin_up_vis = np.array([3700, 4041, 4261, 4426, 4536, 4646, 4756, 4921, 5030, 5140, 5250, 5360, 5469, 5579, 5688, 5600, 5700, 5800, 5878, 5913, 6070, 6200, 6300, 6450, 6600, 6800, 7000, 7200, 7450, 7645, 7720, 8100, 8485, 8985, 10240])/1e4

# td_vis = np.array([1.050, 1.048, 1.027, 1.028, 1.025, 1.035, 1.013, 1.023, 1.028, 1.034, 1.005, 1.024, 1.024, 1.007, 1.036, 1.023, 1.047, 1.014, 1.051, 1.066, 1.026, 1.028, 1.022, 1.036, 0.995, 1.004, 0.997, 1.009, 1.018, 1.003, 1.020, 1.010, 0.986, 1.005, 1.025])

# td_up_vis = np.array([0.037, 0.018, 0.014, 0.014, 0.009, 0.016, 0.015, 0.011, 0.012, 0.016, 0.014, 0.019, 0.019, 0.016, 0.014, 0.027, 0.022, 0.014, 0.018, 0.024, 0.014, 0.020, 0.018, 0.015, 0.020, 0.013, 0.016, 0.013, 0.011, 0.017, 0.022, 0.020, 0.026, 0.023, 0.018])/100

# td_down_vis = np.array([0.044, 0.021, 0.015, 0.013, 0.012, 0.018, 0.013, 0.011, 0.012, 0.015, 0.012, 0.014, 0.016, 0.017, 0.015, 0.021, 0.023, 0.014, 0.020, 0.023, 0.017, 0.016, 0.020, 0.016, 0.017, 0.012, 0.012, 0.014, 0.011, 0.016, 0.026, 0.019, 0.018, 0.018, 0.017])/100

# HST_wave_VIS = (bin_up_vis - bin_down_vis) / 2 + bin_down_vis

# HST_data_VIS = td_vis/100

# HST_data_err_VIS = np.sqrt(td_up_vis**2+td_down_vis**2)

# sp_wave = np.array([3.5,4.5])
# sp_data = np.array([0.993/100, 1.073/100])
# sp_data_err = np.array([0.005/100, 0.006/100])

# mols = ['H2O','CO','CO2','FeH','CH4','HCN','NH3','C2H2','TiO', 'OH','Na','K']



def guillot_global(P,kappa_IR,gamma,grav,T_int,T_equ):
    ''' Returns a temperature array, in units of K,
    of the same dimensions as the pressure P
    (in bar). For this the temperature model of Guillot (2010)
    is used (his Equation 29).

    Args:
        P:
            numpy array of floats, containing the input pressure in bars.
        kappa_IR (float):
            The infrared opacity in units of :math:`\\rm cm^2/s`.
        gamma (float):
            The ratio between the visual and infrated opacity.
        grav (float):
            The planetary surface gravity in units of :math:`\\rm cm/s^2`.
        T_int (float):
            The planetary internal temperature (in units of K).
        T_equ (float):
            The planetary equilibrium temperature (in units of K).
    '''
    tau = P*1e6*kappa_IR/grav
    T_irr = T_equ*np.sqrt(2.)
    T = (0.75 * T_int**4. * (2. / 3. + tau) + \
      0.75 * T_irr**4. / 4. * (2. / 3. + 1. / gamma / 3.**0.5 + \
      (gamma / 3.**0.5 - 1. / 3.**0.5 / gamma)* \
      np.exp(-gamma * tau *3.**0.5)))**0.25
    return T




def add_contrib_mol(atom, nb_mols, list_mols, abunds, abund0 = 0., samples=None):
    
    
    abund = 0. + abund0
    
    if samples is not None:
        samp = np.zeros_like(samples[:,0]) + abund0
    
    for i in range(nb_mols):
        mol = list_mols[i]
        if atom in mol:
            idx = mol.index(atom)+1
            if idx < len(mol):
                num = mol[idx]
                try:
                    num = float(num)
                except ValueError:
                    if num.islower():
                        num = 0.
                    else:
                        num = 1.
            else:
                num = 1.
            abund += num*abunds[i]
            
            if samples is not None:
                samp += num*samples[:,i]
                
    if samples is not None:
        return abund, samp
    else:
        return abund, 0.


def print_abund_ratios_any(params, nb_mols=None, samples=None, fe_sur_h = 0, n_sur_h=0,
                               sol_values=None, stellar_values=None, errors=None, H2=0.75, N2=10**(-4.5),
                              list_mols=['H2O','CO','CO2','FeH','CH4','HCN','NH3','C2H2','TiO',\
                                        'OH','Na','K'], fig_name='', prob=0.68):
    if nb_mols is None:
        nb_mols = len(list_mols)
        
    values = []
    samp_values = []
    samp_values_err = []
    median_values = []
    
    bins = np.arange(-7,-1,0.2)
    
    if sol_values is None:
        sol_values = [0.54, 8.46 - 12, 8.69 - 12, 7.83 - 12]
#     sol_c_sur_h = 8.46 - 12  # -3.5
#     sol_o_sur_h = 8.69 - 12  # -3.27
#     sol_n_sur_h = -7.8708 ## <- sans N2 // avec N2 --> #7.83 - 12  # -3.27
    if stellar_values is None:
#     sol_n_sur_o = sol_n_sur_h - sol_o_sur_h
        stellar_values = [0.54] + list(np.array(sol_values[1:]) + fe_sur_h)

#     stellar_c_sur_h = sol_c_sur_h+fe_sur_h
#     stellar_o_sur_h = sol_o_sur_h+fe_sur_h
#     stellar_n_sur_h = sol_n_sur_h+fe_sur_h
    
    
    
#     stellar_n_sur_o = stellar_n_sur_h - stellar_o_sur_h

    abunds = 10**(np.array(params[:nb_mols]))

    print('')

    if samples is not None:
        samps = 10**(np.array(samples[:,:nb_mols]))
    else:
        samps = None
            
    abund_c, samp_c = add_contrib_mol('C', nb_mols, list_mols, abunds, samples=samps)
    abund_o, samp_o = add_contrib_mol('O', nb_mols, list_mols, abunds, samples=samps)
    abund_n, samp_n = add_contrib_mol('N', nb_mols, list_mols, abunds, samples=samps,)
#                                       abund0 = 2*(1-np.sum(abunds))*N2)
    abund_h, samp_h = add_contrib_mol('H', nb_mols, list_mols, abunds, samples=samps, 
                                      abund0 = 2*(1-np.sum(abunds))*H2)
#     print(abund_h)
#     stellar_n_sur_h = np.log10(10**(stellar_n_sur_h) - 2*N2/abund_h)
    c_sur_o = abund_c / abund_o 
    
    values.append(c_sur_o)
    
    if samples is not None:
        samp_c_sur_o = samp_c / samp_o 

        fig = plt.figure()

        maximum, errbars2 = calc_hist_max_and_err(samp_c_sur_o, bins=50, bin_size=2, plot=True, prob=prob)
#         plt.title('C/O = {:.4f} + {:.4f} - {:.4f}\n {:.2f} - {:.2f}'.format(maximum,
#                                                                             errbars2[1]-maximum,
#                                                                             maximum-errbars2[0],
#                                                                            errbars2[0], errbars2[1]))
        plt.annotate(r'C/O = {:.2f}$^{{+{}}}_{{-{}}}$'.format(maximum, 
                                                              '{:.2f}'.format(errbars2[1]-maximum),
                                                              '{:.2f}'.format(maximum-errbars2[0])
                                                             ), \
                     xy=(0.1,0.75), xycoords='axes fraction', fontsize=16 )
        plt.ylabel('% Occurence', fontsize=16)
        plt.xlabel('C/O', fontsize=16)
        plt.axvline(0.54, label='Solar', linestyle='--', color='k')
        if stellar_values is not None:
            plt.axvline(stellar_values[0], label='Stellar', linestyle=':', color='red')
        plt.xlim(0,1.0)
        plt.legend()
        samp_values.append(maximum)
        samp_values_err.append(errbars2)
        
        median_values.append(np.median(samp_c_sur_o))
        
        fig.savefig('/home/boucher/spirou/Figures/fig_C_sur_O_distrib{}.pdf'.format(fig_name))
    
    print('C/O = {:.4f}'.format(c_sur_o))
    print('')

    c_sur_h = abund_c / abund_h 
    

    print('C/H = {:.5e} = 10^{:.4f}'.format(c_sur_h, np.log10(c_sur_h)))
#     print('Fake C/H = {:.5e} = 10^{:.4f}'.format(10**sol_c_sur_h*(c_sur_h/10**stellar_c_sur_h), 
#                                                  np.log10(10**sol_c_sur_h*(c_sur_h/10**stellar_c_sur_h))))
    print('C/H / C/H_sol = {:.4f} = 10^{:.4f}'.format(c_sur_h/10**sol_values[1], 
                                                   np.log10(c_sur_h/10**sol_values[1])))
    print('C/H / C/H_* = {:.4f} = 10^{:.4f}'.format(c_sur_h/10**stellar_values[1], 
                                                 np.log10(c_sur_h/10**stellar_values[1])))
    print('')

    o_sur_h = abund_o / abund_h 
    
    

    print('O/H = {:.5e} = 10^{:.4f}'.format(o_sur_h, np.log10(o_sur_h)))
#     print('Fake O/H = {:.5e} = 10^{:.4f}'.format(10**sol_o_sur_h*(o_sur_h/10**stellar_o_sur_h), 
#                                                  np.log10(10**sol_o_sur_h*(o_sur_h/10**stellar_o_sur_h))))
    print('O/H / O/H_sol = {:.4f} = 10^{:.4f}'.format(o_sur_h/10**sol_values[2], 
                                                   np.log10(o_sur_h/10**sol_values[2])))
    print('O/H / O/H_* = {:.4f} = 10^{:.4f}'.format(o_sur_h/10**stellar_values[2], 
                                                 np.log10(o_sur_h/10**stellar_values[2])))
    print('')
    
    values.append(np.log10(c_sur_h))
    values.append(np.log10(o_sur_h))
    
    n_sur_h = abund_n / abund_h *n_sur_h

    if n_sur_h != 0:
        values.append(np.log10(n_sur_h))
        print('N/H = {:.5e} = 10^{:.4f}'.format(n_sur_h, np.log10(n_sur_h)))
    #     print('Fake N/H = {:.5e} = 10^{:.4f}'.format(10**sol_n_sur_h*(n_sur_h/10**stellar_n_sur_h), 
    #                                                  np.log10(10**sol_n_sur_h*(n_sur_h/10**stellar_n_sur_h))))
        print('N/H / N/H_sol = {:.4f} = 10^{:.4f}'.format(n_sur_h/10**sol_values[3], 
                                                       np.log10(n_sur_h/10**sol_values[3])))
        print('N/H / N/H_* = {:.4f} = 10^{:.4f}'.format(n_sur_h/10**stellar_values[3], 
                                                     np.log10(n_sur_h/10**stellar_values[3])))
        print('')
    
#     n_sur_o = abund_n / abund_o 
    
#     if n_sur_o != 0:
#         print('N/O = {:.4f} = 10^{:.4f}'.format(n_sur_o, np.log10(n_sur_o)))
#         print('N/O / N/O_sol = {:.4f} = 10^{:.4f}'.format(n_sur_o/10**sol_n_sur_o, 
#                                                        np.log10(n_sur_o/10**sol_n_sur_o)))
#         print('N/O / N/O_* = {:.4f} = 10^{:.4f}'.format(n_sur_o/10**stellar_n_sur_o, 
#                                                      np.log10(n_sur_o/10**stellar_n_sur_o)))
#         print('')
    
    

    if samples is not None:

        samp_c_sur_h = samp_c / samp_h 
        samp_o_sur_h = samp_o / samp_h 
        samp_n_sur_h = samp_n / samp_h 
#         samp_n_sur_o = samp_n / samp_o 

        fig = plt.figure()

        maximum2, errbars22 = calc_hist_max_and_err(np.log10(samp_c_sur_h), bins=bins, bin_size=6, plot=True,
                                                    color='dodgerblue', label='C', prob=prob)
        print('C/H = {:.4f} + {:.4f} - {:.4f}'.format(maximum2,errbars22[1]-maximum2,maximum2-errbars22[0]))
        print('[C/H] = {:.4f} + {:.4f} - {:.4f}'.format(maximum2-sol_values[1],
                                                      errbars22[1]-maximum2,maximum2-errbars22[0]))
        print('[C/H] = {:.4f}x solar = {:.4f} -- {:.4f}'.format(10**(maximum2-sol_values[1]),
                                                      10**(errbars22[0]-sol_values[1]), 
                                                         10**(errbars22[1]-sol_values[1])))
        maximum3, errbars23 = calc_hist_max_and_err(np.log10(samp_o_sur_h), bins=bins, bin_size=6, plot=True,
                                                    color='darkorange', label='O', prob=prob)
        print('O/H = {:.4f} + {:.4f} - {:.4f}'.format(maximum3,errbars23[1]-maximum3,maximum3-errbars23[0]))
        print('[O/H] = {:.4f} + {:.4f} - {:.4f}'.format(maximum3-sol_values[2],
                                                      errbars23[1]-maximum3,maximum3-errbars23[0]))
        print('[O/H] = {:.4f}x solar = {:.4f} -- {:.4f}'.format(10**(maximum3-sol_values[2]),
                                                      10**(errbars23[0]-sol_values[2]), 
                                                         10**(errbars23[1]-sol_values[2])))
        
#         plt.axvline(np.median(np.log10(samp_c_sur_h)), color='purple', linestyle = ':')
#         plt.axvline(np.median(np.log10(samp_o_sur_h)), color='gold', linestyle = ':')
        
        print(np.median(np.log10(samp_c_sur_h))-stellar_values[1])
        print(np.median(np.log10(samp_o_sur_h))-stellar_values[2])

        samp_values.append(maximum2)
        samp_values_err.append(errbars22)
        samp_values.append(maximum3)
        samp_values_err.append(errbars23)
        median_values.append(np.median(samp_c_sur_h))
        median_values.append(np.median(samp_o_sur_h))
        
        if n_sur_h != 0:
            maximum3, errbars23 = calc_hist_max_and_err(np.log10(samp_n_sur_h), bins=bins, bin_size=6, plot=True,
                                                        color='forestgreen', label='N', prob=prob)
            print('N/H = {:.4f} + {:.4f} - {:.4f}'.format(maximum3,errbars23[1]-maximum3,maximum3-errbars23[0]))
            plt.axvline(stellar_values[3], linestyle='-.', color='darkgreen')
            samp_values.append(maximum3)
            samp_values_err.append(errbars23)
            median_values.append(np.median(samp_n_sur_h))

        plt.axvline(stellar_values[1], linestyle='-.', color='royalblue', label='Stellar')#, zorder=32)
        plt.axvline(stellar_values[2], linestyle='-.', color='orangered', )#label='')
        
#         plt.axvline(sol_values[1], linestyle='--', color='royalblue', label='Solar')
#         plt.axvline(sol_values[2], linestyle='--', color='orangered', )#label='')
        
        plt.ylabel('% Occurence', fontsize=16)
        plt.xlabel('$X$/H', fontsize=16)

        plt.legend(fontsize=12)
        
        fig.savefig('/home/boucher/spirou/Figures/fig_X_sur_H_distrib{}.pdf'.format(fig_name))
    
    if samples is not None:
        
        return values, samp_values, samp_values_err, [samp_c, samp_o, samp_h], median_values
    else:
        return values
   

    
def plot_abund_ratios(params, nb_mols=None, samples=None, fe_sur_h = 0, n_sur_h=0,
                               sol_values=None, stellar_values=None, errors=None, H2=0.75, N2=10**(-4.5),
                              list_mols=['H2O','CO','CO2','FeH','CH4','HCN','NH3','C2H2','TiO',\
                                        'OH','Na','K'], fig_name='', prob=0.68):
    if nb_mols is None:
        nb_mols = len(list_mols)
        
        
    bins = np.arange(-3,3,0.2)
    
    if sol_values is None:
        sol_values = [0.54, 8.46 - 12, 8.69 - 12, 7.83 - 12]
    if stellar_values is None:
#     sol_n_sur_o = sol_n_sur_h - sol_o_sur_h
        stellar_values = [0.54] + list(np.array(sol_values[1:]) + fe_sur_h)

    abunds = 10**(np.array(params[:nb_mols]))

    print('')

    if samples is not None:
        samps = 10**(np.array(samples[:,:nb_mols]))
    else:
        samps = None
            
    abund_c, samp_c = add_contrib_mol('C', nb_mols, list_mols, abunds, samples=samps)
    abund_o, samp_o = add_contrib_mol('O', nb_mols, list_mols, abunds, samples=samps)
    abund_n, samp_n = add_contrib_mol('N', nb_mols, list_mols, abunds, samples=samps,)
#                                       abund0 = 2*(1-np.sum(abunds))*N2)
    abund_h, samp_h = add_contrib_mol('H', nb_mols, list_mols, abunds, samples=samps, 
                                     abund0 = 2*(1-np.sum(abunds))*H2)
    
    c_sur_h = abund_c / abund_h 
    
    print('C/H = {:.5e} = 10^{:.4f}'.format(c_sur_h, np.log10(c_sur_h)))
#     print('Fake C/H = {:.5e} = 10^{:.4f}'.format(10**sol_c_sur_h*(c_sur_h/10**stellar_c_sur_h), 
#                                                  np.log10(10**sol_c_sur_h*(c_sur_h/10**stellar_c_sur_h))))
    print('C/H / C/H_sol = {:.4f} = 10^{:.4f}'.format(c_sur_h/10**sol_values[1], 
                                                   np.log10(c_sur_h/10**sol_values[1])))
    print('C/H / C/H_* = {:.4f} = 10^{:.4f}'.format(c_sur_h/10**stellar_values[1], 
                                                 np.log10(c_sur_h/10**stellar_values[1])))
    print('')

    o_sur_h = abund_o / abund_h 

    print('O/H = {:.5e} = 10^{:.4f}'.format(o_sur_h, np.log10(o_sur_h)))
#     print('Fake O/H = {:.5e} = 10^{:.4f}'.format(10**sol_o_sur_h*(o_sur_h/10**stellar_o_sur_h), 
#                                                  np.log10(10**sol_o_sur_h*(o_sur_h/10**stellar_o_sur_h))))
    print('O/H / O/H_sol = {:.4f} = 10^{:.4f}'.format(o_sur_h/10**sol_values[2], 
                                                   np.log10(o_sur_h/10**sol_values[2])))
    print('O/H / O/H_* = {:.4f} = 10^{:.4f}'.format(o_sur_h/10**stellar_values[2], 
                                                 np.log10(o_sur_h/10**stellar_values[2])))
    print('')

    
    n_sur_h = abund_n / abund_h *n_sur_h

    if n_sur_h != 0:

        print('N/H = {:.5e} = 10^{:.4f}'.format(n_sur_h, np.log10(n_sur_h)))
    #     print('Fake N/H = {:.5e} = 10^{:.4f}'.format(10**sol_n_sur_h*(n_sur_h/10**stellar_n_sur_h), 
    #                                                  np.log10(10**sol_n_sur_h*(n_sur_h/10**stellar_n_sur_h))))
        print('N/H / N/H_sol = {:.4f} = 10^{:.4f}'.format(n_sur_h/10**sol_values[3], 
                                                       np.log10(n_sur_h/10**sol_values[3])))
        print('N/H / N/H_* = {:.4f} = 10^{:.4f}'.format(n_sur_h/10**stellar_values[3], 
                                                     np.log10(n_sur_h/10**stellar_values[3])))
        print('')
     

    if samples is not None:

        samp_c_sur_h = np.log10(samp_c / samp_h) -sol_values[1]
        samp_o_sur_h = np.log10(samp_o / samp_h) -sol_values[2]
        if n_sur_h != 0:
            samp_n_sur_h = np.log10(samp_n / samp_h) -sol_values[3]
#         samp_n_sur_o = samp_n / samp_o 

        fig = plt.figure()

        maximum2, errbars22 = calc_hist_max_and_err(samp_c_sur_h, bins=bins, bin_size=6, plot=True,
                                                    color='dodgerblue', label='C/H', prob=prob)
        print('C/H = {:.4f} + {:.4f} - {:.4f}'.format(maximum2,errbars22[1]-maximum2,maximum2-errbars22[0]))
        maximum3, errbars23 = calc_hist_max_and_err(samp_o_sur_h, bins=bins, bin_size=6, plot=True,
                                                    color='darkorange', label='O/H', prob=prob)
        print('O/H = {:.4f} + {:.4f} - {:.4f}'.format(maximum3,errbars23[1]-maximum3,maximum3-errbars23[0]))
                
        if n_sur_h != 0:
            maximum3, errbars23 = calc_hist_max_and_err(samp_n_sur_h, bins=bins, bin_size=6, plot=True,
                                                        color='forestgreen', label='N/H', prob=prob)
            print('N/H = {:.4f} + {:.4f} - {:.4f}'.format(maximum3,errbars23[1]-maximum3,maximum3-errbars23[0]))
            plt.axvline(stellar_values[3], linestyle='-.', color='darkgreen')

        plt.axvline(0, linestyle='--', color='black', label='Solar')
#         plt.axvline(fe_sur_h, linestyle=':', color='red', label='Stellar')#, zorder=32)
        plt.axvline(stellar_values[1]- sol_values[1], linestyle=':', color='royalblue', label='C/H Stellar')
        plt.axvline(stellar_values[2]- sol_values[2], linestyle=':', color='orangered', label='O/H Stellar')

        plt.ylabel('% Occurence', fontsize=16)
        plt.xlabel('[$X$/H]', fontsize=16)

        plt.legend(fontsize=13)
        
        fig.savefig('/home/boucher/spirou/Figures/fig_X_sur_H_distrib_sol{}.pdf'.format(fig_name))
    
    return fig
   

    
def mol_frac(params, solar_ab, stellar_ab, 
             mols = ['H2O','CO','CO2','FeH','CH4','HCN','NH3','C2H2','TiO','OH','Na','K'],
            errors=None, nb_mol=None):
    if nb_mol is None:
        nb_mol = len(mols)
    
    abunds = 10**(params[:nb_mol])
    
    print('')
#     if sol_ab is not None:
    for i in range(nb_mol):
        print('{} = {:.4e} = 10^({:.4f}) = 10^ {:.4f} sol = 10^{:.4f} *'.format(mols[i],abunds[i], params[i], 
                                                           np.log10(abunds/solar_ab[:nb_mol])[i],
                                                          np.log10(abunds/stellar_ab[:nb_mol])[i]))
    print('')
    if errors is not None:
        for i in range(nb_mol):
            print('{} = {:.4f} + {:.4f} - {:.4f} == {:.3f} to {:.3f}'.format(mols[i], params[i],
                                                                    errors[i][1]-params[i], params[i]-errors[i][0],
                                                             errors[i][0], errors[i][1]))

            
            
def calc_hist_max_and_err(sample, bins=50, bin_size=2, plot=True, color=None,
                          label=None, prob=0.68, fill=True, weight=1.):
    
    if fill:
        his, edges,_ = plt.hist(sample, bins=bins, alpha=0.3, color=color, label=label,
                               weights=np.ones(len(sample)) / len(sample)*100*weight) 
        mids = 0.5*(edges[1:] + edges[:-1])
        
    else:
        his, edges = np.histogram(sample, bins=bins,weights=np.ones(len(sample)) / len(sample)*100*weight)
        mids = 0.5*(edges[1:] + edges[:-1])
        plt.step(edges[1:], his, color=color, label=label)    
        
    binned = spectrum.box_binning(his,bin_size)
    max_samp = a.find_max_spline(mids, binned, binning=True)[0]

    err_samp = pymc3.stats.hdi(sample, prob)                   
    if plot:
#         plt.plot(mids, binned) 
        plt.axvline(max_samp, color=color)
        plt.axvspan(err_samp[1], err_samp[0], color=color, alpha=0.2)
#         plt.axvline(err_samp[0], color='k', linestyle='--')
#         plt.axvline(err_samp[1], color='k', linestyle='--')
    
    
    return max_samp,err_samp



def plot_best_mod(params, atmos_obj, alpha_vis=0.5, alpha=0.7, color=None, label='', iso=False, 
                  wl_lim =None, back_color='royalblue', add_hst=True, ylim=[0.0092, 0.0122], **kwargs):
    
    fig = plt.figure(figsize=(15,5))

    if add_hst:
        plt.errorbar(HST_wave, HST_data, HST_data_err, color='k',marker='.',linestyle='none', zorder=32, 
                     alpha=0.9, label='Spake+2021')
        plt.errorbar(HST_wave_VIS, HST_data_VIS, HST_data_err_VIS, color='k',marker='.',linestyle='none',
                     alpha=alpha_vis, zorder=31)
        plt.errorbar(sp_wave, sp_data, sp_data_err, color='k',marker='.',linestyle='none')
    
    if wl_lim is not None:
        x,y = calc_best_mod_any(params, atmos_obj, **kwargs)
        plt.plot(x[x>wl_lim],y[x>wl_lim], alpha=alpha, label=label, color=color)
    else:
        plt.plot(*calc_best_mod_any(params, atmos_obj, **kwargs), alpha=alpha, label=label, color=color)


    plt.axvspan(0.95,2.55, color=back_color, alpha=0.1, label='SPIRou range')

    plt.legend(loc='upper left', fontsize=13)
    plt.ylim(*ylim)
    plt.ylabel(r'$(R_{P}/R_{*})^2$', fontsize=16)
    plt.xlabel(r'Wavelength ($\mu$m)', fontsize=16)

    plt.xscale('log')

    tick_labels = [0.3,.4,.5,.6,.7,.8,.9,1,1.5,2, 2.5,3,3.5,4,5]
    plt.xticks(tick_labels)
    plt.gca().set_xticklabels(tick_labels)
    
    return fig



def calc_best_mod_any(params, planet, atmos_obj, temp_params, P0=10e-3, 
                      scatt=False, gamma_scat=-1.7, kappa_factor=0.36,
                      TP=False,  radius_param=2, cloud_param=1,  #kappa_IR=-3, gamma=-1.5,
                      scale=1., haze=None, nb_mols=None, kind_res='low', \
                      list_mols=None, iso=False, kind_trans='transmission', **kwargs):
    
    species_all0 = OrderedDict({})
    
    if kind_res == "low":
        
        if list_mols is None:
            list_mols=['H2O_HITEMP', 'CO_all_iso_HITEMP', 'CO2', 'FeH', \
                                   'CH4', 'HCN', 'NH3', 'C2H2','TiO_all_Plez', \
                                   'OH', 'Na_allard', 'K_allard', 'VO']
        
        
        for mol in list_mols:
            species_all0[mol] = [1e-99]
#                             'H2O_HITEMP': [1e-99], 
#                            'CO_all_iso_HITEMP':[1e-99],
#                           'CO2':[1e-99],
#                               'FeH':[1e-99],
#                               'C2H2': [1e-99],
#                             'CH4': [1e-99],
#                             'HCN': [1e-99],
#                             'NH3': [1e-99],
#                             'TiO_all_Plez': [1e-99],
#                                 'OH': [1e-99],
#                                 'VO': [1e-99],
#                                 'Na_allard' : [1e-99],
#                                 'K_allard' : [1e-99]
#                               })
    elif kind_res == "high":
        if list_mols is None:
            list_mols=['H2O_main_iso', 'CO_all_iso', 'CO2_main_iso', 'FeH_main_iso', \
                       'CH4_main_iso', 'HCN_main_iso', 'NH3_main_iso', 'C2H2_main_iso',\
                       'TiO_all_iso', 'OH_SCARLET', 'Na', 'K', 'VO']
#         species_all0 = OrderedDict({})
        
        for mol in list_mols:
            species_all0[mol] = [1e-99]
#                         'H2O_main_iso': [1e-99], 
#                        'CO_all_iso':[1e-99],
#                       'CO2_main_iso':[1e-99],
#                            'FeH_main_iso':[1e-99],
#                           'C2H2_main_iso': [1e-99], 
#                             'CH4_main_iso': [1e-99], 
#                             'HCN_main_iso': [1e-99], 
#                             'NH3_main_iso': [1e-99], 
#                             'TiO_all_iso': [1e-99],
#                             'OH_SCARLET': [1e-99],
#                             'VO': [1e-99],
#                             'Na' : [1e-99],
#                             'K' : [1e-99]
#                             })
    
    if nb_mols is None:
        nb_mols = len(list_mols)
        
    species_all = species_all0.copy()
    
    for i in range(nb_mols):
#         print(list_mols[i],10**params[i])
        species_all[list_mols[i]] = [10**params[i]]
                        

#     print(nb_mols, params)
    temp_params['T_eq'] = params[nb_mols+0]
    
    if cloud_param is not None:
        cloud = 10**(params[nb_mols+cloud_param])
    else:
        cloud = None
    radius = params[nb_mols+radius_param] * const.R_jup

    temp_params['gravity'] = (const.G * planet.M_pl / (radius)**2).cgs.value
    
    if TP is True:
        temp_params['kappa_IR'] = 10**params[-2]
        temp_params['gamma'] = 10**params[-1]
        
#     print(temp_params['kappa_IR'], temp_params['gamma'])

#     temperature = nc.guillot_global(pressures, 10**kappa_IR, 10**gamma, gravity, T_int, T_equ)
    if iso :
        temperature = temp_params['T_eq']*np.ones_like(temp_params['pressures'])
    else:
        temperature = nc.guillot_global(temp_params['pressures'], 
                                     temp_params['kappa_IR'], 
                                     temp_params['gamma'], 
                                     temp_params['gravity'], 
                                     temp_params['T_int'], 
                                     temp_params['T_eq'])
        
#     plt.plot(temperature, np.log10(temp_params['pressures']))
#     else:
#         gravity = (const.G * planet.M_pl / (radius)**2).cgs.value
#         temperature = nc.guillot_global(pressures, 10**kappa_IR, 10**gamma, gravity, T_int, T_equ)
#     plt.figure()
#     plt.plot(temperature, pressures)
#     plt.figure()

    if scatt is True:
        gamma_scat = params[-2]
        kappa_factor = params[-1]
    
    _, wave, model_rp = prt.calc_multi_full_spectrum(planet, species_all, atmos_full=atmos_obj, 
                             pressures=temp_params['pressures'], T=temp_params['T_eq'], 
                                                     temperature=temperature, plot=False,
                             P0=P0, haze=haze, cloud=cloud, path=None, rp=radius, 
                             gamma_scat = gamma_scat, kappa_factor=kappa_factor, kind_trans=kind_trans, **kwargs )
    
    if kind_trans =='transmission':
        out = np.array(model_rp)[0]/1e6*scale
    elif kind_trans == 'emission':
        out = np.array(model_rp)[0]

    return wave, out



# def calc_best_mod_any(params, atmos_obj, P0=10e-3, 
#                       scatt=True, gamma_scat=-1.7, kappa_factor=0.36,
#                       kappa_IR=-3, gamma=-1.5, radius_param=2,
#                       scale=1., haze=None, nb_mols=None, kind_res='low', \
#                       list_mols=None, iso=False, **kwargs):
    
#     if kind_res == "low":
#         if list_mols is None:
#             list_mols=['H2O_HITEMP', 'CO_all_iso_HITEMP', 'CO2', 'FeH', \
#                                    'CH4', 'HCN', 'NH3', 'C2H2', 'TiO_all_Plez', \
#                                    'OH', 'Na_allard', 'K_allard']
#         species_all0 = OrderedDict({
#                             'H2O_HITEMP': [1e-99], 
#                            'CO_all_iso_HITEMP':[1e-99],
#                           'CO2':[1e-99],
#                               'FeH':[1e-99],
#                               'C2H2': [1e-99],
#                             'CH4': [1e-99],
#                             'HCN': [1e-99],
#                             'NH3': [1e-99],
#                             'TiO_all_Plez': [1e-99],
#                                 'OH': [1e-99],
#                                 'Na_allard' : [1e-99],
#                                 'K_allard' : [1e-99]
#                               })
#     elif kind_res == "high":
#         if list_mols is None:
#             list_mols=['H2O_main_iso', 'CO_all_iso', 'CO2_main_iso', 'FeH_main_iso', \
#                        'C2H2_main_iso', 'CH4_main_iso', 'HCN_main_iso', 'NH3_main_iso', \
#                        'TiO_all_iso', 'OH_SCARLET', 'Na', 'K']
#         species_all0 = OrderedDict({
#                         'H2O_main_iso': [1e-99], 
#                        'CO_all_iso':[1e-99],
#                       'CO2_main_iso':[1e-99],
#                            'FeH_main_iso':[1e-99],
#                           'C2H2_main_iso': [1e-99], 
#                             'CH4_main_iso': [1e-99], 
#                             'HCN_main_iso': [1e-99], 
#                             'NH3_main_iso': [1e-99], 
#                             'TiO_all_iso': [1e-99],
#                             'OH_SCARLET': [1e-99],
#                             'Na' : [1e-99],
#                             'K' : [1e-99]
#                             })
    
#     if nb_mols is None:
#         nb_mols = len(list_mols)
        
#     species_all = species_all0.copy()
    
#     for i in range(nb_mols):
# #         print(list_mols[i],10**params[i])
#         species_all[list_mols[i]] = [10**params[i]]
                        

# #     print(nb_mols, params)
#     T_equ =  params[nb_mols+0]
#     cloud = 10**(params[nb_mols+1])
#     radius = params[nb_mols+radius_param] * const.R_jup

#     gravity = (const.G * planet.M_pl / (radius)**2).cgs.value
#     temperature = nc.guillot_global(pressures, 10**kappa_IR, 10**gamma, gravity, T_int, T_equ)
#     if iso :
#         temperature = T_equ*np.ones_like(pressures)
#     else:
#         gravity = (const.G * planet.M_pl / (radius)**2).cgs.value
#         temperature = nc.guillot_global(pressures, 10**kappa_IR, 10**gamma, gravity, T_int, T_equ)
        
#     if scatt is True:
#         gamma_scat = params[-2]
#         kappa_factor = params[-1]

#     _, wave_low, model_rp_low = prt.calc_multi_full_spectrum(planet, species_all, atmos_full=atmos_obj, 
#                              pressures=pressures, T=T_equ, temperature=temperature, plot=False,
#                              P0=P0, haze=haze, cloud=cloud, path=None, rp=radius, 
#                              gamma_scat = gamma_scat, kappa_factor=kappa_factor, **kwargs )

#     return wave_low, np.array(model_rp_low)[0]/1e6*scale


def downgrade_mod(wlen, flux_lambda, down_wave, Raf=75):
    
    _, resamp_prt = spectrum.resampling(wlen, flux_lambda, Raf=Raf, Rbf=1000, sample=wlen)
    binned_prt_hst = spectrum.box_binning(resamp_prt, 1000/Raf)
    fct_prt= interp1d(wlen, binned_prt_hst)

    return fct_prt(down_wave)


def read_walkers_file(filename, discard=0, id_params=None, param_no_zero=0):

    with h5py.File(filename, "r") as f:

        samples = f['mcmc']['chain']
        if id_params is not None:
            samples = samples[:,:, id_params]

        ndim=np.array(samples).shape[-1]
#         if labels is None:
#             labels = ['' for i in range(ndim)]

        completed = np.where(samples[:,0,param_no_zero] == 0)[0]
        if completed.size == 0:
            cut_sample = samples[discard:,:,:]
            print('All Completed')
        else:
            cut_sample = samples[discard:completed[0],:,:]
            print('Completed {}/{}'.format(completed[0],samples.shape[0]))

    return cut_sample


def single_max_dist(samp, bins='auto', start=0, end=-1, bin_size=6, plot=False):

    if plot is False:
        his, edges = np.histogram(samp, bins=bins, density=True)
    else:
        his, edges,_ = plt.hist(samp, bins=bins)
    mids = 0.5*(edges[1:] + edges[:-1])
    
    binned = spectrum.box_binning(his[start:end], bin_size)
    maxs = a.find_max_spline(mids[start:end], binned, binning=True)[0]

    errors = pymc3.stats.hdi(samp, 0.68)

    return maxs, errors

def find_dist_maxs(sample_all, labels, bin_size=6, flag_id=None, plot=True, prob=0.68):
    
    n_params = sample_all.shape[-1]
    maxs = []
    errors = []
    for i in range(n_params):

    #     if i == 9 :
    #         sample_all[:, :, i] = ((sample_all[:, :, i]/u.day).to('1/s') * maxs[6] * const.R_jup.to(u.km)).value
        samp = sample_all[:, :, i].ravel()  #cut_sample[:, :, i].ravel()

#         his, edges = np.histogram(samp, bins='auto',density=True)
#         mids = 0.5*(edges[1:] + edges[:-1])
#         if flag_id is not None and i == flag_id[0]:
#             binned = spectrum.box_binning(his[30:], flag_id[1])
#             maximum = a.find_max_spline(mids[30:], binned, binning=True)[0]
#         else:
#             binned = spectrum.box_binning(his,bin_size)
#             maximum = a.find_max_spline(mids, binned, binning=True)[0]
#         maxs.append(maximum)

        plt.figure(figsize=(7,3))
        his, edges,_ = plt.hist(samp, bins='auto') 
        mids = 0.5*(edges[1:] + edges[:-1])
#         print(i)
        if flag_id is not None :
            flag = np.array(flag_id)
#             print(flag)
            if i in flag[:,0]:
#                 print('i in flag')
                binned = spectrum.box_binning(his[30:], flag[flag[:,0] == i][0][1])
                maximum = a.find_max_spline(mids[30:], binned, binning=True)[0]
                plt.plot(mids[30:], binned) 
            else:
                binned = spectrum.box_binning(his,bin_size)
                maximum = a.find_max_spline(mids, binned, binning=True)[0]
                plt.plot(mids, binned)
        else:
            binned = spectrum.box_binning(his,bin_size)
            maximum = a.find_max_spline(mids, binned, binning=True)[0]
            plt.plot(mids, binned)
        maxs.append(maximum)

        errbars2 = pymc3.stats.hdi(samp, prob)
        errors.append(errbars2)

        plt.axvline(maximum, color='k')
        plt.axvline(errbars2[0], color='k', linestyle='--')
        plt.axvline(errbars2[1], color='k', linestyle='--')
        print(maximum, errbars2-maximum, errbars2)

    if plot is True:
        fig, ax = plt.subplots(len(labels), 2,constrained_layout=True, 
                       figsize=(10,len(labels)), sharex='col', sharey='row',
                      gridspec_kw={'width_ratios': [5,1]})

        for i in range(n_params):

            ax[i,0].plot(sample_all[:, :, i], "k", alpha=0.3)
            ax[i,0].set_xlim(0, len(sample_all))
            ax[i,0].set_ylabel(labels[i])
            sample = sample_all[:, :, i].reshape(sample_all.shape[0]*sample_all.shape[1], 1)
            ax[i,1].hist(sample,
                         bins=30, orientation='horizontal', weights = np.ones(len(sample)) / len(sample)*100,
                        color='k')
        ax[-1,0].set_xlabel("Steps")
        
#         fig, axes = plt.subplots(n_params, figsize=(10, n_params), sharex=True)
#         for i in range(n_params):
#             ax = axes[i]
#             ax.plot(sample_all[:, :, i], "k", alpha=0.3)
#             ax.set_xlim(0, len(sample_all))
#             ax.set_ylabel(labels[i])
#             ax.yaxis.set_label_coords(-0.1, 0.5)

#         axes[-1].set_xlabel("step number");
    
    return maxs, errors


# def plot_corner(sample_all, labels, param_no_zero=4):
# #     print(sample_all.shape)
#     ndim = sample_all.shape[-1]
    
#     flat_samples = sample_all.reshape(sample_all.shape[0]*sample_all.shape[1], ndim)

#     fig = corner.corner(flat_samples, labels=labels, 
# #                         truths=[None,None,None,None, None, 130,None, None, None, None],  
#                         # quantiles=[0.16, 0.5, 0.84],
#                         show_titles=True,)# range=[(-6,-1), (-8,-1), (-8,-1), (-8,-1), (700,1200), 
#                                            #     (-5,0), (1.0,1.5), (125,135), (-10,-5), (0,20)])#, **corner_kwargs);

    
#     # --- Extract the axes
#     axes = np.array(fig.axes).reshape((ndim, ndim))
#     for i in range(ndim):
#         axes[ndim-1,i].set_xlabel(labels[i],fontsize=16)

#     for i in range(1, ndim):
#         axes[i,0].set_ylabel(labels[i],fontsize=16)

#     for i in range(ndim):
#     #     if i == 3:
#     #         maxs[i] = 0
#     #         errors[i][1] = 0

#         axes[i,i].axvline(maxs[i], color='k')
#         axes[i,i].axvline(errors[i][0], color='k', linestyle='--')
#         axes[i,i].axvline(errors[i][1], color='k', linestyle='--')

#         if i == param_no_zero:
#             float_str_moins = "{:.0f}".format(errors[i][0]-maxs[i])
#             float_str_plus = "{:.0f}".format(errors[i][1]-maxs[i])

#             axes[i,i].set_title(r' {} = {:.0f}$_{{{}}}^{{+{}}}$'.format(labels[i], maxs[i], 
#                                                                         float_str_moins, float_str_plus))
#         else:
#             float_str_moins = "{:.2f}".format(errors[i][0]-maxs[i])
#             float_str_plus = "{:.2f}".format(errors[i][1]-maxs[i])

#             axes[i,i].set_title(r' {} = {:.2f}$_{{{}}}^{{+{}}}$'.format(labels[i], maxs[i], 
#                                                                             float_str_moins, float_str_plus))

#     return flat_samples

# def print_abund_ratios_4mols_fast(logmaxs, H_scale=1., sol_ab=None, stellar_ab=None,
#                             m_sur_h=-0.193, H2=0.75):
#     sol_c_sur_h = 8.46-12
#     sol_o_sur_h = 8.69-12
    
#     stellar_c_sur_h = sol_c_sur_h+m_sur_h
#     stellar_o_sur_h = sol_o_sur_h+m_sur_h
#     mols=['H2O','CO','CO2','FeH']
    

#     abunds = 10**(np.array(logmaxs[:4]))

#     print('')
#     abund_c = (abunds[1]+abunds[2])
#     abund_o = (abunds[0]+abunds[1]+2*abunds[2])
#     abund_h = (2*abunds[0]+abunds[3] + 2*(1-np.sum(abunds[:4]))*H2*H_scale )
#     c_sur_o = abund_c / abund_o 

#     print('C/O = {:.4f}'.format(c_sur_o))
#     print('')

#     c_sur_h = abund_c / abund_h 

#     print('C/H = {:.4f}'.format(np.log10(c_sur_h)))
#     print('C/H / C/H_sol = {:.4f}'.format(10**(np.log10(c_sur_h)-sol_c_sur_h)))
#     print('C/H / C/H_* = {:.4f}'.format(10**(np.log10(c_sur_h)-stellar_c_sur_h)))
#     print('')

#     o_sur_h = abund_o / abund_h 
    
#     print('O/H = {:.4f}'.format(np.log10(o_sur_h)))
#     print('O/H / O/H_sol = {:.4f}'.format(10**(np.log10(o_sur_h)-sol_o_sur_h)))
#     print('O/H / O/H_* = {:.4f}'.format(10**(np.log10(o_sur_h)-stellar_o_sur_h)))

#     print('')
#     if sol_ab is not None:
#         for i in range(4):
#             print('{} = {:.4f} sol = {:.4f} *'.format(mols[i], np.log10(abunds/sol_ab[:4])[i],
#                                                               np.log10(abunds/stellar_ab[:4])[i]))
  

import emcee

def read_walker_prob(filename, tol=20, discard=0):
    
    # --- read backend
    reader = emcee.backends.HDFBackend(filename, read_only=True)
    fin_pos = reader.get_chain(discard=discard)
    logl = reader.get_log_prob(discard=discard) 

    # --- merge all walkers
    flat_logl = np.reshape(logl,(fin_pos.shape[0]*fin_pos.shape[1]))#.shape
    flat_fin_pos = np.reshape(fin_pos,(fin_pos.shape[0]*fin_pos.shape[1], fin_pos.shape[-1]))#.shape

    # --- place walker in order from worst [0] to best[-1] logL
    ord_pos = flat_fin_pos[np.argsort(flat_logl)]
    ord_logl = flat_logl[np.argsort(flat_logl)]
    
    autocorr = emcee.autocorr.integrated_time(fin_pos, tol=tol)
    
    return ord_pos, ord_logl, autocorr

def plot_corner_logl(ord_pos, labels, param_x = 0, param_y = 1, n_pts=1000, cmap='PuRd_r', n_reverse=20):
    couleurs = hm.get_colors(cmap, n_pts)

    for i in range(n_pts):
        pos_i = ord_pos[-i]
        plt.scatter(pos_i[param_x],pos_i[param_y],marker='o', color=couleurs[i],alpha=0.4)
    plt.plot(ord_pos[-n_reverse:][:,param_x],
             ord_pos[-n_reverse:][:,param_y],'o-', color=couleurs[0], alpha=0.1, zorder=32)
    plt.scatter(ord_pos[-1][param_x],ord_pos[-1][param_y],marker='x', color='k', zorder=33)
    plt.xlabel(labels[param_x])
    plt.ylabel(labels[param_y])

    
    
def plot_ratios_corner(samps, values_compare, color='blue', add_solar=True, **kwargs):
    samp_c, samp_o, samp_h = samps
    c_sur_o = samp_c/samp_o
    samp_c_o_sur_h = (samp_c + samp_o)/samp_h
    sol_c_o_sur_h = np.sum(10**(np.array(values_compare)[1:]))
#     stel_c_o_sur_h = np.sum(10**(np.array(values_stellar)[1:]))

    fig = corner.corner(np.concatenate((np.log10(samp_c_o_sur_h[:,None]/sol_c_o_sur_h), c_sur_o[:,None]), axis=1), 
                        labels=['[(C+O)/H]', 'C/O'], 
                            show_titles=True, color=color, **kwargs)

#     _ = corner.corner(np.concatenate((np.log10(samp_c_o_sur_h[:,None]/stel_c_o_sur_h), c_sur_o[:,None]), axis=1), 
#                       labels=['(C+O)/H', 'C/O'], 
#                             show_titles=True, color='red', **kwargs)

    if add_solar :
        axes = np.array(fig.axes).reshape((2,2))
        axes[1,0].scatter([0.0], [0.54], marker='o', color='k')
        
    return fig


def plot_tp_profile(params, planet, errors, nb_mols, pressures, 
                    kappa = -3, gamma = -1.5, T_int=500, 
                    plot_limits=False, label='', color=None, 
                    radius_param = 2, zorder=None):

    T_eq = params[nb_mols]
    kappa_IR = 10**(kappa)
    gamma = 10**(gamma)
#     T_int = 500.
    if planet.M_pl.ndim == 1:
        planet.M_pl = planet.M_pl[0]
    gravity = (const.G * planet.M_pl / (params[nb_mols+radius_param]*const.R_jup)**2).cgs.value
    print(type(gravity))


    temperature = guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_eq)
    temperature_up = guillot_global(pressures, kappa_IR, gamma, gravity, T_int, errors[nb_mols][1])
    temperature_down = guillot_global(pressures, kappa_IR, gamma, gravity, T_int, errors[nb_mols][0])
    print(temperature[0], temperature_up[0], temperature_down[0])

    if plot_limits:
        plt.plot(guillot_global(pressures, kappa_IR, gamma, gravity, T_int, 300),  
                     np.log10(pressures), ':', alpha=0.5, color='grey', label='T-P limits')
        plt.plot(guillot_global(pressures, kappa_IR, gamma, gravity, T_int, 1400),  
                     np.log10(pressures), ':', alpha=0.5, color='grey')


    plt.plot(temperature,  np.log10(pressures), label=label, color=color)
    plt.fill_betweenx(np.log10(pressures), temperature_down, temperature_up, alpha=0.15, color=color, zorder=zorder)

    t1bar = guillot_global(1, kappa_IR, gamma, gravity, T_int, T_eq)

#     print(t1bar, kappa_IR, gamma, gravity, T_int, errors[nb_mols][1])
    print('T @ 1 bar = {:.0f} + {:.0f} - {:.0f} '.format(t1bar,
                               guillot_global(1, kappa_IR, gamma, gravity, T_int, errors[nb_mols][1])-t1bar,
                           t1bar -guillot_global(1, kappa_IR, gamma, gravity, T_int, errors[nb_mols][0]) ))




# def calc_best_mod(params, atmos_obj, gamma_scat=-1.71, kappa_factor=0.36):
    
#     species_low = OrderedDict({'H2O_HITEMP': [10**params[0]],
#                         'CO_all_iso_HITEMP': [10**params[1]],
#                         'CO2': [10**params[2]],
#                           'FeH':[10**params[3]]})

#     T_equ =  params[4]
#     cloud = 10**(params[5])
#     radius = params[6] * const.R_jup

#     gravity = (const.G * planet.M_pl / (radius)**2).cgs.value
#     temperature = nc.guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)

#     _, wave_low, model_rp_low = prt.calc_multi_full_spectrum(planet, species_low, atmos_full=atmos_obj, 
#                              pressures=pressures, T=T_equ, temperature=temperature, plot=False,
#                              P0=P0, haze=haze, cloud=cloud, path=None, rp=radius, 
#                              gamma_scat = gamma_scat, kappa_factor=kappa_factor )

#     return wave_low, np.array(model_rp_low)[0]/1e6

# def plot_best_mod(params, atmos_obj, label='', **kwargs):
    
#     fig = plt.figure(figsize=(15,5))

#     plt.errorbar(HST_wave, HST_data, HST_data_err, color='k',marker='.',linestyle='none', zorder=32, 
#                  alpha=0.9, label='Spake+2021')
#     plt.errorbar(HST_wave_VIS, HST_data_VIS, HST_data_err_VIS, color='k',marker='.',linestyle='none',
#                  alpha=0.3, zorder=31)
#     plt.errorbar(sp_wave, sp_data, sp_data_err, color='k',marker='.',linestyle='none')


#     wave_low, model_rp_low = calc_best_mod(params, atmos_obj, **kwargs)

#     plt.plot(wave_low, model_rp_low, alpha=0.7, label=label)


#     plt.axvspan(0.95,2.55, color='royalblue', alpha=0.1, label='SPIRou range')

#     plt.legend(loc='upper left')
#     plt.ylim(0.0092, 0.0122)
#     plt.ylabel(r'$(R_{P}/R_{*})^2$', fontsize=16)
#     plt.xlabel(r'Wavelength ($\mu$m)', fontsize=16)

#     plt.xscale('log')

#     tick_labels = [0.3,.4,.5,.6,.7,.8,.9,1,1.5,2, 2.5,3,3.5,4,5]
#     plt.xticks(tick_labels)
#     plt.gca().set_xticklabels(tick_labels)
    

# def read_walkers_file(filename, discard=0, id_params=None, param_no_zero=4, labels=None):

#     with h5py.File(filename, "r") as f:

#         samples = f['mcmc']['chain']
#         if id_params is not None:
#             samples = samples[:,:, id_params]

#         ndim=np.array(samples).shape[-1]
# #         if labels is None:
# #             labels = ['' for i in range(ndim)]

#         completed = np.where(samples[:,0,param_no_zero] == 0)[0]
#         if completed.size == 0:
#             cut_sample = samples[discard:,:,:]
#             print('All Completed')
#         else:
#             cut_sample = samples[discard:completed[0],:,:]
#             print('Completed {}/{}'.format(completed[0],samples.shape[0]))
            
# #             if sample_all.ndim >2:

#         if labels is not None:
#             n_params = ndim
#             fig, axes = plt.subplots(n_params, figsize=(10, n_params), sharex=True)
#             for i in range(n_params):
#                 ax = axes[i]
#                 ax.plot(samples[:completed[0], :, i], "k", alpha=0.3)
# #                 ax.set_xlim(0, len(samples))
#                 ax.set_ylabel(labels[i])
#                 ax.yaxis.set_label_coords(-0.1, 0.5)
#                 ax.axvline(discard, color='red')
# #                 ax.set_xlim(None,completed[0])

#             axes[-1].set_xlabel("step number")
            

#     return cut_sample


# # def find_dist_maxs(sample_all, labels, bin_size=6, plot=True):
    
# #     n_params = sample_all.shape[-1]
# #     maxs = []
# #     errors = []
# #     for i in range(n_params):

# #     #     if i == 9 :
# #     #         sample_all[:, :, i] = ((sample_all[:, :, i]/u.day).to('1/s') * maxs[6] * const.R_jup.to(u.km)).value
# #         if sample_all.ndim == 2:
# #             samp = sample_all[:, i]
# #         else:
# #             samp = sample_all[:, :, i].ravel()  #cut_sample[:, :, i].ravel()

# #         his, edges = np.histogram(samp, bins='auto',density=True)
# #         mids = 0.5*(edges[1:] + edges[:-1])
# #         binned = spectrum.box_binning(his,bin_size)
# #         maximum = a.find_max_spline(mids, binned, binning=True)[0]
# #         maxs.append(maximum)

# #         plt.figure(figsize=(7,3))
# #         his, edges,_ = plt.hist(samp, bins='auto') 
# #         mids = 0.5*(edges[1:] + edges[:-1])
# #         binned = spectrum.box_binning(his,bin_size)
# #         maximum = a.find_max_spline(mids, binned, binning=True)[0]
# #         plt.plot(mids, binned) 

# #         errbars2 = pymc3.stats.hdi(samp, 0.68)
# #         errors.append(errbars2)

# #         plt.axvline(maximum, color='k')
# #         plt.axvline(errbars2[0], color='k', linestyle='--')
# #         plt.axvline(errbars2[1], color='k', linestyle='--')
# #         print(maximum, errbars2-maximum, errbars2)
       
# #     if sample_all.ndim >2 and plot:

# #         fig, axes = plt.subplots(n_params, figsize=(10, n_params), sharex=True)
# #         for i in range(n_params):
# #             ax = axes[i]
# #             ax.plot(sample_all[:, :, i], "k", alpha=0.3)
# #             ax.set_xlim(0, len(sample_all))
# #             ax.set_ylabel(labels[i])
# #             ax.yaxis.set_label_coords(-0.1, 0.5)

# #         axes[-1].set_xlabel("step number");
    
# #     return maxs, errors

# def find_dist_maxs(sample_all, labels, bin_size=6, flag_id=None, plot=True, prob=0.68):
    
#     n_params = sample_all.shape[-1]
#     maxs = []
#     errors = []
#     for i in range(n_params):

#     #     if i == 9 :
#     #         sample_all[:, :, i] = ((sample_all[:, :, i]/u.day).to('1/s') * maxs[6] * const.R_jup.to(u.km)).value
#         samp = sample_all[:, :, i].ravel()  #cut_sample[:, :, i].ravel()

# #         his, edges = np.histogram(samp, bins='auto',density=True)
# #         mids = 0.5*(edges[1:] + edges[:-1])
# #         if flag_id is not None and i == flag_id[0]:
# #             binned = spectrum.box_binning(his[30:], flag_id[1])
# #             maximum = a.find_max_spline(mids[30:], binned, binning=True)[0]
# #         else:
# #             binned = spectrum.box_binning(his,bin_size)
# #             maximum = a.find_max_spline(mids, binned, binning=True)[0]
# #         maxs.append(maximum)

#         plt.figure(figsize=(7,3))
#         his, edges,_ = plt.hist(samp, bins='auto') 
#         mids = 0.5*(edges[1:] + edges[:-1])
# #         print(i)
#         if flag_id is not None :
#             flag = np.array(flag_id)
# #             print(flag)
#             if i in flag[:,0]:
# #                 print('i in flag')
#                 binned = spectrum.box_binning(his[30:], flag[flag[:,0] == i][0][1])
#                 maximum = a.find_max_spline(mids[30:], binned, binning=True)[0]
#                 plt.plot(mids[30:], binned) 
#             else:
#                 binned = spectrum.box_binning(his,bin_size)
#                 maximum = a.find_max_spline(mids, binned, binning=True)[0]
#                 plt.plot(mids, binned)
#         else:
#             binned = spectrum.box_binning(his,bin_size)
#             maximum = a.find_max_spline(mids, binned, binning=True)[0]
#             plt.plot(mids, binned)
#         maxs.append(maximum)

#         errbars2 = pymc3.stats.hdi(samp, prob)
#         errors.append(errbars2)

#         plt.axvline(maximum, color='k')
#         plt.axvline(errbars2[0], color='k', linestyle='--')
#         plt.axvline(errbars2[1], color='k', linestyle='--')
#         print(maximum, errbars2-maximum, errbars2)

#     if plot is True:
#         fig, axes = plt.subplots(n_params, figsize=(10, n_params), sharex=True)
#         for i in range(n_params):
#             ax = axes[i]
#             ax.plot(sample_all[:, :, i], "k", alpha=0.3)
#             ax.set_xlim(0, len(sample_all))
#             ax.set_ylabel(labels[i])
#             ax.yaxis.set_label_coords(-0.1, 0.5)

#         axes[-1].set_xlabel("step number");
    
#     return maxs, errors


def plot_corner(sample_all, labels, param_no_zero=4, maxs=None, errors=None, plot=True,**kwargs):
#     print(sample_all.shape)
    ndim = sample_all.shape[-1]
    
    if sample_all.ndim == 2:
        flat_samples = sample_all
    else:
        flat_samples = sample_all.reshape(sample_all.shape[0]*sample_all.shape[1], ndim)

    if plot is True:
        fig = corner.corner(flat_samples, labels=labels, 
    #                         truths=[None,None,None,None, None, 130,None, None, None, None],  
                            # quantiles=[0.16, 0.5, 0.84],
                            show_titles=True,**kwargs)# range=[(-6,-1), (-8,-1), (-8,-1), (-8,-1), (700,1200), 
                                               #     (-5,0), (1.0,1.5), (125,135), (-10,-5), (0,20)])#, **corner_kwargs);

        if maxs is None:
            maxs, errors = find_dist_maxs(sample_all, labels, bin_size=6)

        # --- Extract the axes
        axes = np.array(fig.axes).reshape((ndim, ndim))
        for i in range(ndim):
            axes[ndim-1,i].set_xlabel(labels[i],fontsize=16)

        for i in range(1, ndim):
            axes[i,0].set_ylabel(labels[i],fontsize=16)

        for i in range(ndim):
        #     if i == 3:
        #         maxs[i] = 0
        #         errors[i][1] = 0

            axes[i,i].axvline(maxs[i], color='k')
            axes[i,i].axvline(errors[i][0], color='k', linestyle='--')
            axes[i,i].axvline(errors[i][1], color='k', linestyle='--')

            if i == param_no_zero:
                float_str_moins = "{:.0f}".format(errors[i][0]-maxs[i])
                float_str_plus = "{:.0f}".format(errors[i][1]-maxs[i])

                axes[i,i].set_title(r' {} = {:.0f}$_{{{}}}^{{+{}}}$'.format(labels[i], maxs[i], 
                                                                            float_str_moins, float_str_plus))
            else:
                float_str_moins = "{:.2f}".format(errors[i][0]-maxs[i])
                float_str_plus = "{:.2f}".format(errors[i][1]-maxs[i])

                axes[i,i].set_title(r' {} = {:.2f}$_{{{}}}^{{+{}}}$'.format(labels[i], maxs[i], 
                                                                                float_str_moins, float_str_plus))
    else:
        fig = plt.figure()
    
    return fig, flat_samples



# def print_abund_ratios_4mols_fast(logmaxs, H_scale=1., sol_ab=None, stellar_ab=None,
#                             m_sur_h=-0.193):
#     sol_c_sur_h = 8.46-12
#     sol_o_sur_h = 8.69-12
    
#     stellar_c_sur_h = sol_c_sur_h+m_sur_h
#     stellar_o_sur_h = sol_o_sur_h+m_sur_h
#     mols=['H2O','CO','CO2','FeH']
    

#     abunds = 10**(np.array(logmaxs[:4]))

#     print('')
#     abund_c = (abunds[1]+abunds[2])
#     abund_o = (abunds[0]+abunds[1]+2*abunds[2])
#     abund_h = (2*abunds[0]+abunds[3] + 2*(1-np.sum(abunds[:4]))*0.75*H_scale )
#     c_sur_o = abund_c / abund_o 

#     print('C/O = {:.4f}'.format(c_sur_o))
#     print('')

#     c_sur_h = abund_c / abund_h 

#     print('C/H = {:.4f}'.format(np.log10(c_sur_h)))
#     print('C/H / C/H_sol = {:.4f}'.format(10**(np.log10(c_sur_h)-sol_c_sur_h)))
#     print('C/H / C/H_* = {:.4f}'.format(10**(np.log10(c_sur_h)-stellar_c_sur_h)))
#     print('')

#     o_sur_h = abund_o / abund_h 
    
#     print('O/H = {:.4f}'.format(np.log10(o_sur_h)))
#     print('O/H / O/H_sol = {:.4f}'.format(10**(np.log10(o_sur_h)-sol_o_sur_h)))
#     print('O/H / O/H_* = {:.4f}'.format(10**(np.log10(o_sur_h)-stellar_o_sur_h)))

#     print('')
#     if sol_ab is not None:
#         for i in range(4):
#             print('{} = {:.4f} sol = {:.4f} *'.format(mols[i], np.log10(abunds/sol_ab[:4])[i],
#                                                               np.log10(abunds/stellar_ab[:4])[i]))


# def read_walker_prob(filename):
    
#     # --- read backend
#     reader = emcee.backends.HDFBackend(filename, read_only=True)
#     fin_pos = reader.get_chain()
#     logl = reader.get_log_prob() 

#     # --- merge all walkers
#     flat_logl = np.reshape(logl,(fin_pos.shape[0]*fin_pos.shape[1]))#.shape
#     flat_fin_pos = np.reshape(fin_pos,(fin_pos.shape[0]*fin_pos.shape[1], fin_pos.shape[-1]))#.shape

#     # --- place walker in order from worst [0] to best[-1] logL
#     ord_pos = flat_fin_pos[np.argsort(flat_logl)]
#     ord_logl = flat_logl[np.argsort(flat_logl)]
    
#     return ord_pos, ord_logl


# def plot_corner_logl(ord_pos, labels, param_x = 0, param_y = 1, n_pts=1000, cmap='PuRd_r', n_reverse=20):
#     couleurs = hm.get_colors(cmap, n_pts)

#     for i in range(n_pts):
#         pos_i = ord_pos[-i]
#         plt.scatter(pos_i[param_x],pos_i[param_y],marker='o', color=couleurs[i],alpha=0.4)
#     plt.plot(ord_pos[-n_reverse:][:,param_x],
#              ord_pos[-n_reverse:][:,param_y],'o-', color=couleurs[0], alpha=0.1, zorder=32)
#     plt.scatter(ord_pos[-1][param_x],ord_pos[-1][param_y],marker='x', color='k', zorder=33)
#     plt.xlabel(labels[param_x])
#     plt.ylabel(labels[param_y])
    
    
    





# import emcee
# import h5py

# import corner
# import numpy as np

# import pymc3
# # import pygtc

# from spirou_exo.utils import homemade as hm
# from spirou_exo import analysis as a
# from spirou_exo import spectrum as spectrum
# import matplotlib.pyplot as plt
# # from scipy.interpolate import interp1d
# import astropy.units as u
# import astropy.constants as const

# from petitRADTRANS import Radtrans

# from collections import OrderedDict
# import petitradtrans as prt
# from petitRADTRANS import nat_cst as nc
# # import numpy as np

# # bin_down = np.array([11225,11409,11594,11779,11963,12148,12333,12517,12702,12887,13071,13256,13441,13625,13810,13995,14179,14364,14549,14733,14918,15102,15287,15472,15656,15841,16026,16210])/1e4

# # bin_up = np.array([11409, 11594, 11779, 11963, 12148, 12333, 12517, 12702, 12887, 13071, 13256, 13441, 13625, 13810, 13995, 14179, 14364, 14549, 14733, 14918, 15102, 15287, 15472, 15656, 15841, 16026, 16210, 16395])/1e4

# # HST_wave = (bin_up - bin_down) / 2 + bin_down

# # TD = np.array([0.993,0.995,1.001,0.991,1.001,0.991,0.992,1.000,0.993,0.997,1.010,1.016,1.054,1.051,1.052,1.046,1.053,1.046,1.042,1.031,1.028,1.012,0.999,1.001,0.983,0.986,0.975,0.967])

# # HST_data = TD/100

# # TD_err_up = np.array([0.009,0.008,0.009,0.010,0.008,0.008,0.008,0.012,0.008,0.009,0.007,0.008,0.009,0.011,0.008,0.007,0.007,0.010,0.009,0.009,0.009,0.010,0.010,0.010,0.009,0.009,0.012,0.013])/100

# # TD_err_down = np.array([0.009,0.007,0.010,0.009,0.009,0.008,0.007,0.011,0.008,0.009,0.008,0.008,0.010,0.011,0.008,0.007,0.006,0.011,0.009,0.010,0.012,0.010,0.009,0.011,0.009,0.009,0.011,0.012])/100

# # HST_data_err = np.sqrt(TD_err_up**2+TD_err_down**2)


# # bin_down_vis = np.array([2898, 3700, 4041, 4261, 4426, 4536, 4646, 4756, 4921, 5030, 5140, 5250, 5360, 5469, 5579, 5500, 5600, 5700, 5800, 5878, 5913, 6070, 6200, 6300, 6450, 6600, 6800, 7000, 7200, 7450, 7645, 7720, 8100, 8485, 8985])/1e4

# # bin_up_vis = np.array([3700, 4041, 4261, 4426, 4536, 4646, 4756, 4921, 5030, 5140, 5250, 5360, 5469, 5579, 5688, 5600, 5700, 5800, 5878, 5913, 6070, 6200, 6300, 6450, 6600, 6800, 7000, 7200, 7450, 7645, 7720, 8100, 8485, 8985, 10240])/1e4

# # td_vis = np.array([1.050, 1.048, 1.027, 1.028, 1.025, 1.035, 1.013, 1.023, 1.028, 1.034, 1.005, 1.024, 1.024, 1.007, 1.036, 1.023, 1.047, 1.014, 1.051, 1.066, 1.026, 1.028, 1.022, 1.036, 0.995, 1.004, 0.997, 1.009, 1.018, 1.003, 1.020, 1.010, 0.986, 1.005, 1.025])

# # td_up_vis = np.array([0.037, 0.018, 0.014, 0.014, 0.009, 0.016, 0.015, 0.011, 0.012, 0.016, 0.014, 0.019, 0.019, 0.016, 0.014, 0.027, 0.022, 0.014, 0.018, 0.024, 0.014, 0.020, 0.018, 0.015, 0.020, 0.013, 0.016, 0.013, 0.011, 0.017, 0.022, 0.020, 0.026, 0.023, 0.018])/100

# # td_down_vis = np.array([0.044, 0.021, 0.015, 0.013, 0.012, 0.018, 0.013, 0.011, 0.012, 0.015, 0.012, 0.014, 0.016, 0.017, 0.015, 0.021, 0.023, 0.014, 0.020, 0.023, 0.017, 0.016, 0.020, 0.016, 0.017, 0.012, 0.012, 0.014, 0.011, 0.016, 0.026, 0.019, 0.018, 0.018, 0.017])/100

# # HST_wave_VIS = (bin_up_vis - bin_down_vis) / 2 + bin_down_vis

# # HST_data_VIS = td_vis/100

# # HST_data_err_VIS = np.sqrt(td_up_vis**2+td_down_vis**2)

# # sp_wave = np.array([3.5,4.5])
# # sp_data = np.array([0.993/100, 1.073/100])
# # sp_data_err = np.array([0.005/100, 0.006/100])


# def calc_best_mod(params, atmos_obj, gamma_scat=-1.71, kappa_factor=0.36):
    
#     species_low = OrderedDict({'H2O_HITEMP': [10**params[0]],
#                         'CO_all_iso_HITEMP': [10**params[1]],
#                         'CO2': [10**params[2]],
#                           'FeH':[10**params[3]]})

#     T_equ =  params[4]
#     cloud = 10**(params[5])
#     radius = params[6] * const.R_jup

#     gravity = (const.G * planet.M_pl / (radius)**2).cgs.value
#     temperature = nc.guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)

#     _, wave_low, model_rp_low = prt.calc_multi_full_spectrum(planet, species_low, atmos_full=atmos_obj, 
#                              pressures=pressures, T=T_equ, temperature=temperature, plot=False,
#                              P0=P0, haze=haze, cloud=cloud, path=None, rp=radius, 
#                              gamma_scat = gamma_scat, kappa_factor=kappa_factor )

#     return wave_low, np.array(model_rp_low)[0]/1e6

# def plot_best_mod(params, atmos_obj, label='', **kwargs):
    
#     fig = plt.figure(figsize=(15,5))

#     plt.errorbar(HST_wave, HST_data, HST_data_err, color='k',marker='.',linestyle='none', zorder=32, 
#                  alpha=0.9, label='Spake+2021')
#     plt.errorbar(HST_wave_VIS, HST_data_VIS, HST_data_err_VIS, color='k',marker='.',linestyle='none',
#                  alpha=0.3, zorder=31)
#     plt.errorbar(sp_wave, sp_data, sp_data_err, color='k',marker='.',linestyle='none')


#     wave_low, model_rp_low = calc_best_mod(params, atmos_obj, **kwargs)

#     plt.plot(wave_low, model_rp_low, alpha=0.7, label=label)


#     plt.axvspan(0.95,2.55, color='royalblue', alpha=0.1, label='SPIRou range')

#     plt.legend(loc='upper left')
#     plt.ylim(0.0092, 0.0122)
#     plt.ylabel(r'$(R_{P}/R_{*})^2$', fontsize=16)
#     plt.xlabel(r'Wavelength ($\mu$m)', fontsize=16)

#     plt.xscale('log')

#     tick_labels = [0.3,.4,.5,.6,.7,.8,.9,1,1.5,2, 2.5,3,3.5,4,5]
#     plt.xticks(tick_labels)
#     plt.gca().set_xticklabels(tick_labels)
    

# def read_walkers_file(filename, discard=0, id_params=None, param_no_zero=4, labels=None):

#     with h5py.File(filename, "r") as f:

#         samples = f['mcmc']['chain']
#         if id_params is not None:
#             samples = samples[:,:, id_params]

#         ndim=np.array(samples).shape[-1]
# #         if labels is None:
# #             labels = ['' for i in range(ndim)]

#         completed = np.where(samples[:,0,param_no_zero] == 0)[0]
#         if completed.size == 0:
#             cut_sample = samples[discard:,:,:]
#             print('All Completed')
#         else:
#             cut_sample = samples[discard:completed[0],:,:]
#             print('Completed {}/{}'.format(completed[0],samples.shape[0]))
            
# #             if sample_all.ndim >2:

#         if labels is not None:
#             n_params = ndim
#             fig, axes = plt.subplots(n_params, figsize=(10, n_params), sharex=True)
#             for i in range(n_params):
#                 ax = axes[i]
#                 ax.plot(samples[:completed[0], :, i], "k", alpha=0.3)
# #                 ax.set_xlim(0, len(samples))
#                 ax.set_ylabel(labels[i])
#                 ax.yaxis.set_label_coords(-0.1, 0.5)
#                 ax.axvline(discard, color='red')
# #                 ax.set_xlim(None,completed[0])

#             axes[-1].set_xlabel("step number")
            

#     return cut_sample


# # def find_dist_maxs(sample_all, labels, bin_size=6, plot=True):
    
# #     n_params = sample_all.shape[-1]
# #     maxs = []
# #     errors = []
# #     for i in range(n_params):

# #     #     if i == 9 :
# #     #         sample_all[:, :, i] = ((sample_all[:, :, i]/u.day).to('1/s') * maxs[6] * const.R_jup.to(u.km)).value
# #         if sample_all.ndim == 2:
# #             samp = sample_all[:, i]
# #         else:
# #             samp = sample_all[:, :, i].ravel()  #cut_sample[:, :, i].ravel()

# #         his, edges = np.histogram(samp, bins='auto',density=True)
# #         mids = 0.5*(edges[1:] + edges[:-1])
# #         binned = spectrum.box_binning(his,bin_size)
# #         maximum = a.find_max_spline(mids, binned, binning=True)[0]
# #         maxs.append(maximum)

# #         plt.figure(figsize=(7,3))
# #         his, edges,_ = plt.hist(samp, bins='auto') 
# #         mids = 0.5*(edges[1:] + edges[:-1])
# #         binned = spectrum.box_binning(his,bin_size)
# #         maximum = a.find_max_spline(mids, binned, binning=True)[0]
# #         plt.plot(mids, binned) 

# #         errbars2 = pymc3.stats.hdi(samp, 0.68)
# #         errors.append(errbars2)

# #         plt.axvline(maximum, color='k')
# #         plt.axvline(errbars2[0], color='k', linestyle='--')
# #         plt.axvline(errbars2[1], color='k', linestyle='--')
# #         print(maximum, errbars2-maximum, errbars2)
       
# #     if sample_all.ndim >2 and plot:

# #         fig, axes = plt.subplots(n_params, figsize=(10, n_params), sharex=True)
# #         for i in range(n_params):
# #             ax = axes[i]
# #             ax.plot(sample_all[:, :, i], "k", alpha=0.3)
# #             ax.set_xlim(0, len(sample_all))
# #             ax.set_ylabel(labels[i])
# #             ax.yaxis.set_label_coords(-0.1, 0.5)

# #         axes[-1].set_xlabel("step number");
    
# #     return maxs, errors

# def find_dist_maxs(sample_all, labels, bin_size=6, flag_id=None, plot=True, prob=0.68):
    
#     n_params = sample_all.shape[-1]
#     maxs = []
#     errors = []
#     for i in range(n_params):

#     #     if i == 9 :
#     #         sample_all[:, :, i] = ((sample_all[:, :, i]/u.day).to('1/s') * maxs[6] * const.R_jup.to(u.km)).value
#         samp = sample_all[:, :, i].ravel()  #cut_sample[:, :, i].ravel()

# #         his, edges = np.histogram(samp, bins='auto',density=True)
# #         mids = 0.5*(edges[1:] + edges[:-1])
# #         if flag_id is not None and i == flag_id[0]:
# #             binned = spectrum.box_binning(his[30:], flag_id[1])
# #             maximum = a.find_max_spline(mids[30:], binned, binning=True)[0]
# #         else:
# #             binned = spectrum.box_binning(his,bin_size)
# #             maximum = a.find_max_spline(mids, binned, binning=True)[0]
# #         maxs.append(maximum)

#         plt.figure(figsize=(7,3))
#         his, edges,_ = plt.hist(samp, bins='auto') 
#         mids = 0.5*(edges[1:] + edges[:-1])
# #         print(i)
#         if flag_id is not None :
#             flag = np.array(flag_id)
# #             print(flag)
#             if i in flag[:,0]:
# #                 print('i in flag')
#                 binned = spectrum.box_binning(his[30:], flag[flag[:,0] == i][0][1])
#                 maximum = a.find_max_spline(mids[30:], binned, binning=True)[0]
#                 plt.plot(mids[30:], binned) 
#             else:
#                 binned = spectrum.box_binning(his,bin_size)
#                 maximum = a.find_max_spline(mids, binned, binning=True)[0]
#                 plt.plot(mids, binned)
#         else:
#             binned = spectrum.box_binning(his,bin_size)
#             maximum = a.find_max_spline(mids, binned, binning=True)[0]
#             plt.plot(mids, binned)
#         maxs.append(maximum)

#         errbars2 = pymc3.stats.hdi(samp, prob)
#         errors.append(errbars2)

#         plt.axvline(maximum, color='k')
#         plt.axvline(errbars2[0], color='k', linestyle='--')
#         plt.axvline(errbars2[1], color='k', linestyle='--')
#         print(maximum, errbars2-maximum, errbars2)

#     if plot is True:
#         fig, axes = plt.subplots(n_params, figsize=(10, n_params), sharex=True)
#         for i in range(n_params):
#             ax = axes[i]
#             ax.plot(sample_all[:, :, i], "k", alpha=0.3)
#             ax.set_xlim(0, len(sample_all))
#             ax.set_ylabel(labels[i])
#             ax.yaxis.set_label_coords(-0.1, 0.5)

#         axes[-1].set_xlabel("step number");
    
#     return maxs, errors


# def plot_corner(sample_all, labels, param_no_zero=4, maxs=None, errors=None, plot=True,**kwargs):
# #     print(sample_all.shape)
#     ndim = sample_all.shape[-1]
    
#     if sample_all.ndim == 2:
#         flat_samples = sample_all
#     else:
#         flat_samples = sample_all.reshape(sample_all.shape[0]*sample_all.shape[1], ndim)

#     if plot is True:
#         fig = corner.corner(flat_samples, labels=labels, 
#     #                         truths=[None,None,None,None, None, 130,None, None, None, None],  
#                             # quantiles=[0.16, 0.5, 0.84],
#                             show_titles=True,**kwargs)# range=[(-6,-1), (-8,-1), (-8,-1), (-8,-1), (700,1200), 
#                                                #     (-5,0), (1.0,1.5), (125,135), (-10,-5), (0,20)])#, **corner_kwargs);

#         if maxs is None:
#             maxs, errors = find_dist_maxs(sample_all, labels, bin_size=6)

#         # --- Extract the axes
#         axes = np.array(fig.axes).reshape((ndim, ndim))
#         for i in range(ndim):
#             axes[ndim-1,i].set_xlabel(labels[i],fontsize=16)

#         for i in range(1, ndim):
#             axes[i,0].set_ylabel(labels[i],fontsize=16)

#         for i in range(ndim):
#         #     if i == 3:
#         #         maxs[i] = 0
#         #         errors[i][1] = 0

#             axes[i,i].axvline(maxs[i], color='k')
#             axes[i,i].axvline(errors[i][0], color='k', linestyle='--')
#             axes[i,i].axvline(errors[i][1], color='k', linestyle='--')

#             if i == param_no_zero:
#                 float_str_moins = "{:.0f}".format(errors[i][0]-maxs[i])
#                 float_str_plus = "{:.0f}".format(errors[i][1]-maxs[i])

#                 axes[i,i].set_title(r' {} = {:.0f}$_{{{}}}^{{+{}}}$'.format(labels[i], maxs[i], 
#                                                                             float_str_moins, float_str_plus))
#             else:
#                 float_str_moins = "{:.2f}".format(errors[i][0]-maxs[i])
#                 float_str_plus = "{:.2f}".format(errors[i][1]-maxs[i])

#                 axes[i,i].set_title(r' {} = {:.2f}$_{{{}}}^{{+{}}}$'.format(labels[i], maxs[i], 
#                                                                                 float_str_moins, float_str_plus))
#     else:
#         fig = plt.figure()
    
#     return fig, flat_samples

# def print_abund_ratios_4mols_fast(logmaxs, H_scale=1., sol_ab=None, stellar_ab=None,
#                             m_sur_h=-0.193):
#     sol_c_sur_h = 8.46-12
#     sol_o_sur_h = 8.69-12
    
#     stellar_c_sur_h = sol_c_sur_h+m_sur_h
#     stellar_o_sur_h = sol_o_sur_h+m_sur_h
#     mols=['H2O','CO','CO2','FeH']
    

#     abunds = 10**(np.array(logmaxs[:4]))

#     print('')
#     abund_c = (abunds[1]+abunds[2])
#     abund_o = (abunds[0]+abunds[1]+2*abunds[2])
#     abund_h = (2*abunds[0]+abunds[3] + 2*(1-np.sum(abunds[:4]))*0.75*H_scale )
#     c_sur_o = abund_c / abund_o 

#     print('C/O = {:.4f}'.format(c_sur_o))
#     print('')

#     c_sur_h = abund_c / abund_h 

#     print('C/H = {:.4f}'.format(np.log10(c_sur_h)))
#     print('C/H / C/H_sol = {:.4f}'.format(10**(np.log10(c_sur_h)-sol_c_sur_h)))
#     print('C/H / C/H_* = {:.4f}'.format(10**(np.log10(c_sur_h)-stellar_c_sur_h)))
#     print('')

#     o_sur_h = abund_o / abund_h 
    
#     print('O/H = {:.4f}'.format(np.log10(o_sur_h)))
#     print('O/H / O/H_sol = {:.4f}'.format(10**(np.log10(o_sur_h)-sol_o_sur_h)))
#     print('O/H / O/H_* = {:.4f}'.format(10**(np.log10(o_sur_h)-stellar_o_sur_h)))

#     print('')
#     if sol_ab is not None:
#         for i in range(4):
#             print('{} = {:.4f} sol = {:.4f} *'.format(mols[i], np.log10(abunds/sol_ab[:4])[i],
#                                                               np.log10(abunds/stellar_ab[:4])[i]))


# def read_walker_prob(filename):
    
#     # --- read backend
#     reader = emcee.backends.HDFBackend(filename, read_only=True)
#     fin_pos = reader.get_chain()
#     logl = reader.get_log_prob() 

#     # --- merge all walkers
#     flat_logl = np.reshape(logl,(fin_pos.shape[0]*fin_pos.shape[1]))#.shape
#     flat_fin_pos = np.reshape(fin_pos,(fin_pos.shape[0]*fin_pos.shape[1], fin_pos.shape[-1]))#.shape

#     # --- place walker in order from worst [0] to best[-1] logL
#     ord_pos = flat_fin_pos[np.argsort(flat_logl)]
#     ord_logl = flat_logl[np.argsort(flat_logl)]
    
#     return ord_pos, ord_logl


# def plot_corner_logl(ord_pos, labels, param_x = 0, param_y = 1, n_pts=1000, cmap='PuRd_r', n_reverse=20):
#     couleurs = hm.get_colors(cmap, n_pts)

#     for i in range(n_pts):
#         pos_i = ord_pos[-i]
#         plt.scatter(pos_i[param_x],pos_i[param_y],marker='o', color=couleurs[i],alpha=0.4)
#     plt.plot(ord_pos[-n_reverse:][:,param_x],
#              ord_pos[-n_reverse:][:,param_y],'o-', color=couleurs[0], alpha=0.1, zorder=32)
#     plt.scatter(ord_pos[-1][param_x],ord_pos[-1][param_y],marker='x', color='k', zorder=33)
#     plt.xlabel(labels[param_x])
#     plt.ylabel(labels[param_y])
    
    
    