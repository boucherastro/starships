from pathlib import Path

import h5py

import corner
import numpy as np

import arviz as az
from functools import partial

from starships import homemade as hm
from starships import analysis as a
from starships import spectrum as spectrum
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import astropy.units as u
import astropy.constants as const

# from .plotting_fcts import _get_idx_in_range
import random

from . import petitradtrans_utils as prt

try:
    from petitRADTRANS.physics import guillot_global, guillot_modif
except ModuleNotFoundError:
    try:
        from petitRADTRANS.nat_cst import guillot_global, guillot_modif
    except ModuleNotFoundError:
        print('petitRADTRANS is not installed on this system')

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Init random generator (used in functions later to draw from samples
rng = np.random.default_rng()


# def guillot_global(P,kappa_IR,gamma,grav,T_int,T_equ):
#     ''' Returns a temperature array, in units of K,
#     of the same dimensions as the pressure P
#     (in bar). For this the temperature model of Guillot (2010)
#     is used (his Equation 29).

#     Args:
#         P:
#             numpy array of floats, containing the input pressure in bars.
#         kappa_IR (float):
#             The infrared opacity in units of :math:`\\rm cm^2/s`.
#         gamma (float):
#             The ratio between the visual and infrated opacity.
#         grav (float):
#             The planetary surface gravity in units of :math:`\\rm cm/s^2`.
#         T_int (float):
#             The planetary internal temperature (in units of K).
#         T_equ (float):
#             The planetary equilibrium temperature (in units of K).
#     '''
#     tau = P*1e6*kappa_IR/grav
#     T_irr = T_equ*np.sqrt(2.)
#     T = (0.75 * T_int**4. * (2. / 3. + tau) + \
#       0.75 * T_irr**4. / 4. * (2. / 3. + 1. / gamma / 3.**0.5 + \
#       (gamma / 3.**0.5 - 1. / 3.**0.5 / gamma)* \
#       np.exp(-gamma * tau *3.**0.5)))**0.25
#     return T


def add_contrib_mol(atom, nb_mols, list_mols, abunds, abund0=0., samples=None):
    abund = 0. + abund0

    if samples is not None:
        samp = np.zeros_like(samples[:, 0]) + abund0

    for i in range(nb_mols):
        mol = list_mols[i]
        if atom in mol:
            idx = mol.index(atom) + 1
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
            abund += num * abunds[i]

            if samples is not None:
                samp += num * samples[:, i]

    if samples is not None:
        return abund, samp
    else:
        return abund, 0.


def print_abund_ratios_any(params, nb_mols=None, samples=None, fe_sur_h=0, n_sur_h=0, sol_values=None,
                           stellar_values=None, errors=None, H2=0.85, N2=10 ** (-4.5),
                           list_mols=['H2O', 'CO', 'CO2', 'FeH', 'CH4', 'HCN', 'NH3', 'C2H2', 'TiO', 'OH', 'Na', 'K'],
                           fig_name='', prob=0.68, bins=None):
    if nb_mols is None:
        nb_mols = len(list_mols)
    if bins is None:
        bins = np.arange(0, 3.0, 0.05)
        xlims = [0, 1.5]
    else:
        xlims = [bins[0], bins[-1]]

    values = []
    samp_values = []
    samp_values_err = []
    median_values = []

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

    abunds = 10 ** (np.array(params[:nb_mols]))

    print('')

    if samples is not None:
        samps = 10 ** (np.array(samples[:, :nb_mols]))
    else:
        samps = None

    abund_c, samp_c = add_contrib_mol('C', nb_mols, list_mols, abunds, samples=samps)
    abund_o, samp_o = add_contrib_mol('O', nb_mols, list_mols, abunds, samples=samps)
    abund_n, samp_n = add_contrib_mol('N', nb_mols, list_mols, abunds, samples=samps, )
    #                                       abund0 = 2*(1-np.sum(abunds))*N2)
    abund_h, samp_h = add_contrib_mol('H', nb_mols, list_mols, abunds, samples=samps,
                                      abund0=2 * (1 - np.sum(abunds)) * H2)
    #     print(abund_h)
    #     stellar_n_sur_h = np.log10(10**(stellar_n_sur_h) - 2*N2/abund_h)
    c_sur_o = abund_c / abund_o

    values.append(c_sur_o)

    if samples is not None:
        samp_c_sur_o = samp_c / samp_o

        #         fig = plt.figure()
        #         plt.hist(samp_c_sur_o,)
        fig = plt.figure()
        maximum, errbars2 = calc_hist_max_and_err(samp_c_sur_o, bins=bins, bin_size=2, plot=True, prob=prob)
        #         plt.title('C/O = {:.4f} + {:.4f} - {:.4f}\n {:.2f} - {:.2f}'.format(maximum,
        #                                                                             errbars2[1]-maximum,
        #                                                                             maximum-errbars2[0],
        #                                                                            errbars2[0], errbars2[1]))
        plt.annotate(r'C/O = {:.2f}$^{{+{}}}_{{-{}}}$'.format(maximum, '{:.2f}'.format(errbars2[1] - maximum),
                                                              '{:.2f}'.format(maximum - errbars2[0])), xy=(0.1, 0.75),
                     xycoords='axes fraction', fontsize=16)
        plt.ylabel('% Occurence', fontsize=16)
        plt.xlabel('C/O', fontsize=16)
        plt.axvline(0.54, label='Solar', linestyle='--', color='k')
        if stellar_values is not None:
            plt.axvline(stellar_values[0], label='Stellar', linestyle=':', color='red')

        plt.xlim(*xlims)
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
    print('C/H / C/H_sol = {:.4f} = 10^{:.4f}'.format(c_sur_h / 10 ** sol_values[1],
                                                      np.log10(c_sur_h / 10 ** sol_values[1])))
    print('C/H / C/H_* = {:.4f} = 10^{:.4f}'.format(c_sur_h / 10 ** stellar_values[1],
                                                    np.log10(c_sur_h / 10 ** stellar_values[1])))
    print('')

    o_sur_h = abund_o / abund_h

    print('O/H = {:.5e} = 10^{:.4f}'.format(o_sur_h, np.log10(o_sur_h)))
    #     print('Fake O/H = {:.5e} = 10^{:.4f}'.format(10**sol_o_sur_h*(o_sur_h/10**stellar_o_sur_h),
    #                                                  np.log10(10**sol_o_sur_h*(o_sur_h/10**stellar_o_sur_h))))
    print('O/H / O/H_sol = {:.4f} = 10^{:.4f}'.format(o_sur_h / 10 ** sol_values[2],
                                                      np.log10(o_sur_h / 10 ** sol_values[2])))
    print('O/H / O/H_* = {:.4f} = 10^{:.4f}'.format(o_sur_h / 10 ** stellar_values[2],
                                                    np.log10(o_sur_h / 10 ** stellar_values[2])))
    print('')

    values.append(np.log10(c_sur_h))
    values.append(np.log10(o_sur_h))

    n_sur_h = abund_n / abund_h * n_sur_h

    if n_sur_h != 0:
        values.append(np.log10(n_sur_h))
        print('N/H = {:.5e} = 10^{:.4f}'.format(n_sur_h, np.log10(n_sur_h)))
        #     print('Fake N/H = {:.5e} = 10^{:.4f}'.format(10**sol_n_sur_h*(n_sur_h/10**stellar_n_sur_h),
        #                                                  np.log10(10**sol_n_sur_h*(n_sur_h/10**stellar_n_sur_h))))
        print('N/H / N/H_sol = {:.4f} = 10^{:.4f}'.format(n_sur_h / 10 ** sol_values[3],
                                                          np.log10(n_sur_h / 10 ** sol_values[3])))
        print('N/H / N/H_* = {:.4f} = 10^{:.4f}'.format(n_sur_h / 10 ** stellar_values[3],
                                                        np.log10(n_sur_h / 10 ** stellar_values[3])))
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

        bins = np.arange(-7, -1, 0.2)

        samp_c_sur_h = samp_c / samp_h
        samp_o_sur_h = samp_o / samp_h
        samp_n_sur_h = samp_n / samp_h
        #         samp_n_sur_o = samp_n / samp_o

        fig = plt.figure()

        maximum2, errbars22 = calc_hist_max_and_err(np.log10(samp_c_sur_h), bins=bins, bin_size=6, plot=True,
                                                    color='dodgerblue', label='C', prob=prob)
        print('C/H = {:.4f} + {:.4f} - {:.4f}'.format(maximum2, errbars22[1] - maximum2, maximum2 - errbars22[0]))
        print('C/H = {:.4e} = {:.4e} -- {:.4e}'.format(10 ** maximum2, 10 ** errbars22[0], 10 ** errbars22[1]))
        print('[C/H] = {:.4f} + {:.4f} - {:.4f}'.format(maximum2 - sol_values[1], errbars22[1] - maximum2,
                                                        maximum2 - errbars22[0]))
        print('[C/H] = {:.4f}x solar = {:.4f} -- {:.4f}'.format(10 ** (maximum2 - sol_values[1]),
                                                                10 ** (errbars22[0] - sol_values[1]),
                                                                10 ** (errbars22[1] - sol_values[1])))
        print('[C/H]* = {:.4f} + {:.4f} - {:.4f}'.format(maximum2 - stellar_values[1], errbars22[1] - maximum2,
                                                         maximum2 - errbars22[0]))
        print('[C/H]* = {:.4f}x solar = {:.4f} -- {:.4f}'.format(10 ** (maximum2 - stellar_values[1]),
                                                                 10 ** (errbars22[0] - stellar_values[1]),
                                                                 10 ** (errbars22[1] - stellar_values[1])))
        maximum3, errbars23 = calc_hist_max_and_err(np.log10(samp_o_sur_h), bins=bins, bin_size=6, plot=True,
                                                    color='darkorange', label='O', prob=prob)
        print('O/H = {:.4f} + {:.4f} - {:.4f}'.format(maximum3, errbars23[1] - maximum3, maximum3 - errbars23[0]))
        print('O/H = {:.4e} = {:.4e} -- {:.4e}'.format(10 ** maximum3, 10 ** errbars23[0], 10 ** errbars23[1]))
        print('[O/H] = {:.4f} + {:.4f} - {:.4f}'.format(maximum3 - sol_values[2], errbars23[1] - maximum3,
                                                        maximum3 - errbars23[0]))
        print('[O/H] = {:.4f}x solar = {:.4f} -- {:.4f}'.format(10 ** (maximum3 - sol_values[2]),
                                                                10 ** (errbars23[0] - sol_values[2]),
                                                                10 ** (errbars23[1] - sol_values[2])))
        print('[O/H]* = {:.4f} + {:.4f} - {:.4f}'.format(maximum3 - stellar_values[2], errbars23[1] - maximum3,
                                                         maximum3 - errbars23[0]))
        print('[O/H]* = {:.4f}x solar = {:.4f} -- {:.4f}'.format(10 ** (maximum3 - stellar_values[2]),
                                                                 10 ** (errbars23[0] - stellar_values[2]),
                                                                 10 ** (errbars23[1] - stellar_values[2])))

        #         plt.axvline(np.median(np.log10(samp_c_sur_h)), color='purple', linestyle = ':')
        #         plt.axvline(np.median(np.log10(samp_o_sur_h)), color='gold', linestyle = ':')

        print(np.median(np.log10(samp_c_sur_h)) - stellar_values[1])
        print(np.median(np.log10(samp_o_sur_h)) - stellar_values[2])

        samp_values.append(maximum2)
        samp_values_err.append(errbars22)
        samp_values.append(maximum3)
        samp_values_err.append(errbars23)
        median_values.append(np.median(samp_c_sur_h))
        median_values.append(np.median(samp_o_sur_h))

        if n_sur_h != 0:
            maximum3, errbars23 = calc_hist_max_and_err(np.log10(samp_n_sur_h), bins=bins, bin_size=6, plot=True,
                                                        color='forestgreen', label='N', prob=prob)
            print('N/H = {:.4f} + {:.4f} - {:.4f}'.format(maximum3, errbars23[1] - maximum3, maximum3 - errbars23[0]))
            plt.axvline(stellar_values[3], linestyle='-.', color='darkgreen')
            samp_values.append(maximum3)
            samp_values_err.append(errbars23)
            median_values.append(np.median(samp_n_sur_h))

        plt.axvline(stellar_values[1], linestyle='-.', color='royalblue', label='Stellar')  # , zorder=32)
        plt.axvline(stellar_values[2], linestyle='-.', color='orangered', )  # label='')

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


def plot_c_sur_o(params, nb_mols=None, samples=None, fe_sur_h=0, n_sur_h=0, sol_values=None, stellar_values=None,
                 errors=None, H2=0.85, N2=10 ** (-4.5),
                 list_mols=['H2O', 'CO', 'CO2', 'FeH', 'CH4', 'HCN', 'NH3', 'C2H2', 'TiO', 'OH', 'Na', 'K'],
                 fig_name='', prob=0.68, color=None, label='', pos=(0.1, 0.75), add_infos=True, plot=True, bins=None,
                 **kwargs):
    if nb_mols is None:
        nb_mols = len(list_mols)

    if bins is None:
        bins = np.arange(0, 3.0, 0.02)

    values = []
    samp_values = []
    samp_values_err = []

    if sol_values is None:
        sol_values = [0.54, 8.46 - 12, 8.69 - 12, 7.83 - 12]

    if stellar_values is None:
        stellar_values = [0.54] + list(np.array(sol_values[1:]) + fe_sur_h)

    abunds = 10 ** (np.array(params[:nb_mols]))

    print('')

    if samples is not None:
        samps = 10 ** (np.array(samples[:, :nb_mols]))
    else:
        samps = None

    abund_c, samp_c = add_contrib_mol('C', nb_mols, list_mols, abunds, samples=samps)
    abund_o, samp_o = add_contrib_mol('O', nb_mols, list_mols, abunds, samples=samps)
    #     abund_n, samp_n = add_contrib_mol('N', nb_mols, list_mols, abunds, samples=samps,)
    #                                       abund0 = 2*(1-np.sum(abunds))*N2)
    abund_h, samp_h = add_contrib_mol('H', nb_mols, list_mols, abunds, samples=samps,
                                      abund0=2 * (1 - np.sum(abunds)) * H2)
    #     print(abund_h)
    #     stellar_n_sur_h = np.log10(10**(stellar_n_sur_h) - 2*N2/abund_h)
    c_sur_o = abund_c / abund_o

    values.append(c_sur_o)

    if samples is not None:
        samp_c_sur_o = samp_c / samp_o

        #         fig = plt.figure()

        maximum, errbars2 = calc_hist_max_and_err(samp_c_sur_o, bins=bins, bin_size=2, prob=prob, color=color,
                                                  label=label, plot=plot, **kwargs)
        if plot is True:
            plt.annotate(r'C/O = {:.2f}$^{{+{}}}_{{-{}}}$'.format(maximum, '{:.2f}'.format(errbars2[1] - maximum),
                                                                  '{:.2f}'.format(maximum - errbars2[0])), xy=pos,
                         xycoords='axes fraction', fontsize=16, color=color)
        plt.ylabel('% Occurence', fontsize=16)
        plt.xlabel('C/O', fontsize=16)

        if add_infos:
            plt.axvline(0.54, label='Solar', linestyle='--', color='k')
            if stellar_values is not None:
                plt.axvline(stellar_values[0], label='Stellar', linestyle=':', color='red')
            plt.xlim(0, 1.0)
        plt.legend(fontsize=14)

        samp_values.append(maximum)
        samp_values_err.append(errbars2)

    return values, samp_values, samp_values_err


def plot_abund_ratios(params, nb_mols=None, samples=None, fe_sur_h=0, n_sur_h=0, sol_values=None, stellar_values=None,
                      H2=0.85,  # errors=None,  N2=10**(-4.5),
                      list_mols=['H2O', 'CO', 'CO2', 'FeH', 'CH4', 'HCN', 'NH3', 'C2H2', 'TiO', 'OH', 'Na', 'K'],
                      plot_all=True, fig=None, fig_name=None, path_fig='', prob=0.68, **kwargs):
    if nb_mols is None:
        nb_mols = len(list_mols)

    bins = np.arange(-3, 3, 0.2)

    if sol_values is None:
        sol_values = [0.54, 8.46 - 12, 8.69 - 12, 7.83 - 12]
    if stellar_values is None:
        # sol_n_sur_o = sol_n_sur_h - sol_o_sur_h
        stellar_values = [0.54] + list(np.array(sol_values[1:]) + fe_sur_h)

    abunds = 10 ** (np.array(params[:nb_mols]))

    print('')

    if samples is not None:
        samps = 10 ** (np.array(samples[:, :nb_mols]))
    else:
        samps = None

    abund_c, samp_c = add_contrib_mol('C', nb_mols, list_mols, abunds, samples=samps)
    abund_o, samp_o = add_contrib_mol('O', nb_mols, list_mols, abunds, samples=samps)
    abund_n, samp_n = add_contrib_mol('N', nb_mols, list_mols, abunds, samples=samps, )
    #                                       abund0 = 2*(1-np.sum(abunds))*N2)
    abund_h, samp_h = add_contrib_mol('H', nb_mols, list_mols, abunds, samples=samps,
                                      abund0=2 * (1 - np.sum(abunds)) * H2)

    c_sur_h = abund_c / abund_h

    print('C/H = {:.5e} = 10^{:.4f}'.format(c_sur_h, np.log10(c_sur_h)))
    #     print('Fake C/H = {:.5e} = 10^{:.4f}'.format(10**sol_c_sur_h*(c_sur_h/10**stellar_c_sur_h),
    #                                                  np.log10(10**sol_c_sur_h*(c_sur_h/10**stellar_c_sur_h))))
    print('C/H / C/H_sol = {:.4f} = 10^{:.4f}'.format(c_sur_h / 10 ** sol_values[1],
                                                      np.log10(c_sur_h / 10 ** sol_values[1])))
    print('C/H / C/H_* = {:.4f} = 10^{:.4f}'.format(c_sur_h / 10 ** stellar_values[1],
                                                    np.log10(c_sur_h / 10 ** stellar_values[1])))
    print('')

    o_sur_h = abund_o / abund_h

    print('O/H = {:.5e} = 10^{:.4f}'.format(o_sur_h, np.log10(o_sur_h)))
    #     print('Fake O/H = {:.5e} = 10^{:.4f}'.format(10**sol_o_sur_h*(o_sur_h/10**stellar_o_sur_h),
    #                                                  np.log10(10**sol_o_sur_h*(o_sur_h/10**stellar_o_sur_h))))
    print('O/H / O/H_sol = {:.4f} = 10^{:.4f}'.format(o_sur_h / 10 ** sol_values[2],
                                                      np.log10(o_sur_h / 10 ** sol_values[2])))
    print('O/H / O/H_* = {:.4f} = 10^{:.4f}'.format(o_sur_h / 10 ** stellar_values[2],
                                                    np.log10(o_sur_h / 10 ** stellar_values[2])))
    print('')

    n_sur_h = abund_n / abund_h * n_sur_h

    if n_sur_h != 0:
        print('N/H = {:.5e} = 10^{:.4f}'.format(n_sur_h, np.log10(n_sur_h)))
        #     print('Fake N/H = {:.5e} = 10^{:.4f}'.format(10**sol_n_sur_h*(n_sur_h/10**stellar_n_sur_h),
        #                                                  np.log10(10**sol_n_sur_h*(n_sur_h/10**stellar_n_sur_h))))
        print('N/H / N/H_sol = {:.4f} = 10^{:.4f}'.format(n_sur_h / 10 ** sol_values[3],
                                                          np.log10(n_sur_h / 10 ** sol_values[3])))
        print('N/H / N/H_* = {:.4f} = 10^{:.4f}'.format(n_sur_h / 10 ** stellar_values[3],
                                                        np.log10(n_sur_h / 10 ** stellar_values[3])))
        print('')

    if samples is not None:

        samp_c_sur_h = np.log10(samp_c / samp_h) - sol_values[1]
        samp_o_sur_h = np.log10(samp_o / samp_h) - sol_values[2]
        if n_sur_h != 0:
            samp_n_sur_h = np.log10(samp_n / samp_h) - sol_values[3]
        #         samp_n_sur_o = samp_n / samp_o

        if fig is None:
            fig = plt.figure()

        maximum2, errbars22 = calc_hist_max_and_err(samp_c_sur_h, bins=bins, bin_size=6, plot=True, color='dodgerblue',
                                                    label='C/H', prob=prob, **kwargs)
        print('C/H = {:.4f} + {:.4f} - {:.4f}'.format(maximum2, errbars22[1] - maximum2, maximum2 - errbars22[0]))
        maximum3, errbars23 = calc_hist_max_and_err(samp_o_sur_h, bins=bins, bin_size=6, plot=True, color='darkorange',
                                                    label='O/H', prob=prob, **kwargs)
        print('O/H = {:.4f} + {:.4f} - {:.4f}'.format(maximum3, errbars23[1] - maximum3, maximum3 - errbars23[0]))

        if n_sur_h != 0:
            maximum3, errbars23 = calc_hist_max_and_err(samp_n_sur_h, bins=bins, bin_size=6, plot=True,
                                                        color='forestgreen', label='N/H', prob=prob)
            print('N/H = {:.4f} + {:.4f} - {:.4f}'.format(maximum3, errbars23[1] - maximum3, maximum3 - errbars23[0]))
            plt.axvline(stellar_values[3] - sol_values[3], linestyle='-.', color='darkgreen')
        if plot_all:
            plt.axvline(0, linestyle='--', color='black', label='Solar')
            #         plt.axvline(fe_sur_h, linestyle=':', color='red', label='Stellar')#, zorder=32)
            plt.axvline(stellar_values[1] - sol_values[1], linestyle=':', color='royalblue', label='C/H Stellar')
            plt.axvline(stellar_values[2] - sol_values[2], linestyle=':', color='orangered', label='O/H Stellar')

            plt.ylabel('% Occurence', fontsize=16)
            plt.xlabel('[$X$/H]', fontsize=16)

            plt.legend(fontsize=13)

        if fig_name is not None:
            fig.savefig(path_fig + 'fig_X_sur_H_distrib_sol{}.pdf'.format(fig_name))

    return fig


def mol_frac(params, solar_ab, stellar_ab,
             mols=['H2O', 'CO', 'CO2', 'FeH', 'CH4', 'HCN', 'NH3', 'C2H2', 'TiO', 'OH', 'Na', 'K'], errors=None,
             nb_mol=None):
    if nb_mol is None:
        nb_mol = len(mols)

    abunds = 10 ** (params[:nb_mol])

    print('')
    #     if sol_ab is not None:
    for i in range(nb_mol):
        print('{} = {:.4e} = 10^({:.4f}) // 10^ {:.4f} = {:.4f} sol // 10^{:.4f} = {:.4f} *'.format(mols[i], abunds[i],
                                                                                                    params[i], np.log10(
                abunds / solar_ab[:nb_mol])[i], (abunds / solar_ab[:nb_mol])[i], np.log10(abunds / stellar_ab[:nb_mol])[
                                                                                                        i], (
                                                                                                                abunds / stellar_ab[
                                                                                                                         :nb_mol])[
                                                                                                        i]))
    print('')
    if errors is not None:
        for i in range(nb_mol):
            print(
                '{} = {:.4f} + {:.4f} - {:.4f} == {:.3f} to {:.3f}'.format(mols[i], params[i], errors[i][1] - params[i],
                                                                           params[i] - errors[i][0], errors[i][0],
                                                                           errors[i][1]))
            print('{} = {:.3f} to {:.3f} x sol = {:.3f} to {:.3f} x *'.format(mols[i],
                                                                              10 ** (errors[i][0]) / solar_ab[:nb_mol][
                                                                                  i],
                                                                              10 ** (errors[i][1]) / solar_ab[:nb_mol][
                                                                                  i], 10 ** (errors[i][0]) /
                                                                              stellar_ab[:nb_mol][i],
                                                                              10 ** (errors[i][1]) /
                                                                              stellar_ab[:nb_mol][i]))


def calc_hist_max_and_err(sample, bins=50, bin_size=2, plot=True, color=None, label=None, prob=0.68, fill=True,
                          weight=1.):
    if fill:
        his, edges, _ = plt.hist(sample, bins=bins, alpha=0.3, color=color, label=label,
                                 weights=np.ones(len(sample)) / len(sample) * 100 * weight)
        mids = 0.5 * (edges[1:] + edges[:-1])

    else:
        his, edges = np.histogram(sample, bins=bins, weights=np.ones(len(sample)) / len(sample) * 100 * weight)
        mids = 0.5 * (edges[1:] + edges[:-1])
        plt.step(edges[1:], his, color=color, label=label)

    binned = spectrum.box_binning(his, bin_size)
    max_samp = a.find_max_spline(mids, binned, binning=True)[0]

    err_samp = az.hdi(sample, prob)
    if plot:
        #         plt.plot(mids, binned)
        plt.axvline(max_samp, color=color)
        plt.axvspan(err_samp[1], err_samp[0], color=color, alpha=0.15)
    #         plt.axvline(err_samp[0], color='k', linestyle='--')
    #         plt.axvline(err_samp[1], color='k', linestyle='--')

    return max_samp, err_samp


def plot_best_mod(params, planet, atmos_obj, temp_params, alpha_vis=0.5, alpha=0.7, color=None, label='', wl_lim=None,
                  back_color='royalblue', add_hst=True, ylim=[0.0092, 0.0122], **kwargs):
    fig = plt.figure(figsize=(15, 5))

    if add_hst:
        plt.errorbar(HST_wave, HST_data, HST_data_err, color='k', marker='.', linestyle='none', zorder=32, alpha=0.9,
                     label='Spake+2021')
        plt.errorbar(HST_wave_VIS, HST_data_VIS, HST_data_err_VIS, color='k', marker='.', linestyle='none',
                     alpha=alpha_vis, zorder=31)
        plt.errorbar(sp_wave, sp_data, sp_data_err, color='k', marker='.', linestyle='none')

    if wl_lim is not None:
        x, y = calc_best_mod_any(params, planet, atmos_obj, temp_params, **kwargs)
        plt.plot(x[x > wl_lim], y[x > wl_lim], alpha=alpha, label=label, color=color)
    else:
        plt.plot(*calc_best_mod_any(params, atmos_obj, **kwargs), alpha=alpha, label=label, color=color)

    plt.axvspan(0.95, 2.55, color=back_color, alpha=0.1, label='SPIRou range')

    plt.legend(loc='upper left', fontsize=13)
    plt.ylim(*ylim)
    plt.ylabel(r'$(R_{P}/R_{*})^2$', fontsize=16)
    plt.xlabel(r'Wavelength ($\mu$m)', fontsize=16)

    plt.xscale('log')

    tick_labels = [0.3, .4, .5, .6, .7, .8, .9, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5]
    plt.xticks(tick_labels)
    plt.gca().set_xticklabels(tick_labels)

    return fig


import matplotlib.gridspec as gridspec


def add_plot_mod(wv_mod, mod, ax1, ax2, ax3, params, temp_params, kind_temp, TP, label='', lim_ax1=[1.0, 1.7],
                 alpha=0.6, **kwargs):
    mask = (wv_mod >= lim_ax1[0]) & (wv_mod <= lim_ax1[1])

    ax1.plot(wv_mod[mask], mod[mask], label=label, alpha=alpha, )
    ax2.plot(calc_tp_profile(params, temp_params, kind_temp=kind_temp, TP=TP, **kwargs),
             np.log10(temp_params['pressures']), label=label)
    ax3.plot(wv_mod, mod, label=label, alpha=alpha, )


def plot_best_mods(ret_params, ret, kind_res='low', params=None, hst_data=None, spit_data=None):
    models = []

    wv_mod, mod1 = calc_best_mod_any(ret_params[ret]['meds'],  # chose1, #
                                     ret_params[ret]['planet'], ret_params[ret]['atmos'],
                                     ret_params[ret]['temp_params'], kind_res=kind_res, **ret_params[ret]['kwargs'],
                                     params_id=ret_params[ret]['params_id'])
    models.append(mod1)

    wv_mod, mod2 = calc_best_mod_any(ret_params[ret]['maxs'],  # chose2, #
                                     ret_params[ret]['planet'], ret_params[ret]['atmos'],
                                     ret_params[ret]['temp_params'], kind_res=kind_res, **ret_params[ret]['kwargs'],
                                     params_id=ret_params[ret]['params_id'])
    models.append(mod2)

    wv_mod, mod3 = calc_best_mod_any(ret_params[ret]['best'],  # chose3, #
                                     ret_params[ret]['planet'], ret_params[ret]['atmos'],
                                     ret_params[ret]['temp_params'], kind_res=kind_res, **ret_params[ret]['kwargs'],
                                     params_id=ret_params[ret]['params_id'])
    models.append(mod3)
    if params is not None:
        wv_mod, mod4 = calc_best_mod_any(params, ret_params[ret]['planet'], ret_params[ret]['atmos'],
                                         ret_params[ret]['temp_params'], kind_res=kind_res, **ret_params[ret]['kwargs'],
                                         params_id=ret_params[ret]['params_id'])
        models.append(mod4)

    models = np.array(models)

    fig = plt.figure(figsize=(17, 9), tight_layout=True)
    gs = gridspec.GridSpec(2, 2)

    ax_zoom = fig.add_subplot(gs[0, 0])
    ax_tp = fig.add_subplot(gs[0, 1])
    ax_all = fig.add_subplot(gs[1, :])

    ##############################################
    ##############################################

    # ret_params[ret]['temp_params']['T_eq'] = chose1[12]
    add_plot_mod(wv_mod, mod1, ax_zoom, ax_tp, ax_all, ret_params[ret]['meds'],  # chose1,
                 ret_params[ret]['temp_params'], ret_params[ret]['kwargs']['kind_temp'],
                 ret_params[ret]['kwargs']['TP'], label='Meds', nb_mols=ret_params[ret]['nb_mols'],
                 params_id=ret_params[ret]['params_id'])

    # ret_params[ret]['temp_params']['T_eq'] = chose2[12]
    add_plot_mod(wv_mod, mod2, ax_zoom, ax_tp, ax_all, ret_params[ret]['maxs'],  # chose2,
                 ret_params[ret]['temp_params'], ret_params[ret]['kwargs']['kind_temp'],
                 ret_params[ret]['kwargs']['TP'], label='Maxs', nb_mols=ret_params[ret]['nb_mols'],
                 params_id=ret_params[ret]['params_id'])

    # ret_params[ret]['temp_params']['T_eq'] = chose3[12]
    add_plot_mod(wv_mod, mod3, ax_zoom, ax_tp, ax_all, ret_params[ret]['best'], ret_params[ret]['temp_params'],
                 ret_params[ret]['kwargs']['kind_temp'], ret_params[ret]['kwargs']['TP'], label='Best',
                 nb_mols=ret_params[ret]['nb_mols'], params_id=ret_params[ret]['params_id'])

    if params is not None:
        add_plot_mod(wv_mod, mod4, ax_zoom, ax_tp, ax_all, params, ret_params[ret]['temp_params'],
                     ret_params[ret]['kwargs']['kind_temp'], ret_params[ret]['kwargs']['TP'], label='Best',
                     nb_mols=ret_params[ret]['nb_mols'], params_id=ret_params[ret]['params_id'])

    ##############################################
    ##############################################
    # ax_zoom.plot(HST_wave, down1,'o',color='limegreen')

    if hst_data is not None:
        HST_wave, HST_data, HST_data_err = hst_data

        ax_zoom.errorbar(HST_wave, HST_data, HST_data_err, color='k', marker='.', linestyle='none')
    # ax_zoom.errorbar(HST_wave_VIS, HST_data_VIS, HST_data_err_VIS,color='k',marker='.', linestyle='none')
    ax_zoom.set_xlim(1.0, 1.7)
    ax_zoom.set_ylim(np.min((mod1[(wv_mod <= 1.7) & (wv_mod >= 1.1)], mod2[(wv_mod <= 1.7) & (wv_mod >= 1.1)],
                             mod3[(wv_mod <= 1.7) & (wv_mod >= 1.1)])) - 0.0005, np.max((mod1[(wv_mod <= 1.7) & (
                wv_mod >= 1.1)], mod2[(wv_mod <= 1.7) & (wv_mod >= 1.1)], mod3[(wv_mod <= 1.7) & (
                wv_mod >= 1.1)])) + 0.0005)

    ax_tp.set_ylim(1, -8)
    ax_tp.legend(fontsize=16)
    ax_tp.set_xlabel('Temperature (K)', fontsize=16)
    ax_tp.set_ylabel(r'Pressure (bar)', fontsize=16)

    # ax_tp.plot(temp_eq,  np.log10(press_eq),'--',color='indigo',  label='Chem. Equil. ATMO Spake+2021')
    # ax_tp.plot(temp_free,  np.log10(press_free), '--',color='orchid', label='Free Chem. ATMO Spake+2021')
    # ax_tp.plot(temp_nem,  np.log10(press_nem),'--',color='orangered',  label='Free Chem. Nemesis Spake+2021')
    if hst_data is not None:
        ax_all.errorbar(HST_wave, HST_data, HST_data_err, color='k', marker='.', linestyle='none', zorder=50)
    # ax_all.errorbar(HST_wave_VIS, HST_data_VIS, HST_data_err_VIS,
    #                 color='k',marker='.', linestyle='none', zorder=51)
    if spit_data is not None:
        spit_wave, spit_data, spit_data_err, spit_bins = spit_data
        ax_all.errorbar(spit_wave, spit_data, spit_data_err, xerr=spit_bins, color='k', marker='.', linestyle='none',
                        zorder=52)
    ax_all.set_xlabel('Wavelength (um)', fontsize=16)
    ax_all.set_ylabel(r'$(R_p/R_*)^2$', fontsize=16)

    tick_labels = [.6, .7, .8, .9, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5]
    plt.xscale('log')
    plt.xticks(tick_labels)
    ax_all.set_xticklabels(tick_labels)

    return fig, wv_mod, models


def calc_best_mod_any(params, planet, atmos_obj, temp_params, P0=10e-3, scatt=False, gamma_scat=-1.7,
                      kappa_factor=np.log10(0.36), TP=False, radius_param=2, cloud_param=1, scale=1., haze=None,
                      nb_mols=None, kind_res='low', list_mols=None, kind_temp='', kind_trans='transmission',
                      plot_abundance=False, params_id=None, plot_TP=False, change_line_list=None, add_line_list=None,
                      **kwargs):
    #     species_all0 = OrderedDict({})

    if list_mols is None:
        list_mols = ['H2O', 'CO', 'CO2', 'FeH', 'CH4', 'HCN', 'NH3', 'C2H2', 'TiO', 'VO', 'OH', 'Na', 'K']

    if kind_res == "low":
        species_all0 = prt.select_mol_list(list_mols, list_values=None, kind_res='low',
                                           change_line_list=change_line_list, add_line_list=add_line_list)
    elif kind_res == "high":
        species_all0 = prt.select_mol_list(list_mols, list_values=None, kind_res='high')

    if nb_mols is None:
        nb_mols = len(list_mols)

    species_all = species_all0.copy()

    for i, mol_i in enumerate(species_all.keys()):
        #         print(list_mols[i],10**params[i])
        species_all[mol_i] = [10 ** params[i]]

    #     print(nb_mols, params)
    temp_params['T_eq'] = params[nb_mols + 0]
    # print(temp_params['T_eq'])

    if cloud_param is not None:
        cloud = 10 ** (params[nb_mols + cloud_param])
    else:
        cloud = None
    if radius_param is not None:
        radius = params[nb_mols + radius_param] * const.R_jup
    else:
        radius = planet.R_pl

    temp_params['gravity'] = (const.G * planet.M_pl / (radius) ** 2).cgs.value

    temperature = calc_tp_profile(params, temp_params, nb_mols=nb_mols, kind_temp=kind_temp,
                                  params_id=params_id)  # , TP=TP

    if plot_TP:
        plt.figure()
        plt.plot(temperature, np.log10(temp_params['pressures']))
        plt.ylim(2, -6)

    if scatt is True:
        gamma_scat = params[-2]
        kappa_factor = 10 ** params[-1]
    # print(species_all)
    _, wave, model_rp = prt.calc_multi_full_spectrum(planet, species_all, atmos_full=atmos_obj,
                                                     pressures=temp_params['pressures'], T=temp_params['T_eq'],
                                                     temperature=temperature, plot=False, P0=P0, haze=haze, cloud=cloud,
                                                     path=None, rp=radius, gamma_scat=gamma_scat,
                                                     kappa_factor=kappa_factor, kind_trans=kind_trans,
                                                     plot_abundance=plot_abundance, **kwargs)

    if kind_trans == 'transmission':
        out = np.array(model_rp)[0] / 1e6 * scale
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


def downgrade_mod(wlen, flux_lambda, down_wave, Rbf=1000, Raf=75):
    _, resamp_prt = spectrum.resampling(wlen, flux_lambda, Raf=Raf, Rbf=Rbf, sample=wlen)
    binned_prt_hst = spectrum.box_binning(resamp_prt, Rbf / Raf)
    fct_prt = interp1d(wlen, binned_prt_hst)

    return fct_prt(down_wave)


def read_walkers_file(filename, discard=0, discard_after=None, id_params=None, param_no_zero=0):
    with h5py.File(filename, "r") as f:

        if discard_after is None:
            samples = f['mcmc']['chain']
        else:
            samples = f['mcmc']['chain'][:discard_after]
        if id_params is not None:
            samples = samples[:, :, id_params]

        # ndim=np.array(samples).shape[-1]
        #         if labels is None:
        #             labels = ['' for i in range(ndim)]

        completed = np.where(samples[:, 0, param_no_zero] == 0)[0]
        if completed.size == 0:
            cut_sample = samples[discard:, :, :]
            print('All Completed')
        else:
            cut_sample = samples[discard:completed[0], :, :]
            print('Completed {}/{}'.format(completed[0], samples.shape[0]))

    return cut_sample


def read_walkers_prob(filename, discard=0, discard_after=None):
    with h5py.File(filename, "r") as f:

        if discard_after is None:
            fin_pos = f['mcmc']['chain'][discard:]
            logl = f['mcmc']['log_prob'][discard:]
        else:
            fin_pos = f['mcmc']['chain'][discard:discard_after]
            logl = f['mcmc']['log_prob'][discard:discard_after]

    # --- merge all walkers
    flat_logl = np.reshape(logl, (fin_pos.shape[0] * fin_pos.shape[1]))  # .shape
    flat_fin_pos = np.reshape(fin_pos, (fin_pos.shape[0] * fin_pos.shape[1], fin_pos.shape[-1]))  # .shape

    # --- place walker in order from worst [0] to best[-1] logL
    ord_pos = flat_fin_pos[np.argsort(flat_logl)]
    ord_logl = flat_logl[np.argsort(flat_logl)]

    # autocorr = emcee.autocorr.integrated_time(fin_pos, tol=tol)

    return ord_pos, ord_logl


def single_max_dist(samp, bins='auto', start=0, end=-1, bin_size=6, plot=False):
    if plot is False:
        his, edges = np.histogram(samp, bins=bins, density=True)
    else:
        his, edges, _ = plt.hist(samp, bins=bins)
    mids = 0.5 * (edges[1:] + edges[:-1])

    binned = spectrum.box_binning(his[start:end], bin_size)
    maxs = a.find_max_spline(mids[start:end], binned, binning=True)[0]

    errors = az.hdi(samp, 0.68)

    return maxs, errors


def find_dist_maxs(sample_all, labels=None, bin_size=6, flag_id=None, plot=True, prob=0.68, print_gen_walkers=False,
                   cut_sample=None):
    n_params = sample_all.shape[-1]

    if labels is None:
        labels = [f'{idx}' for idx in range(n_params)]

    maxs = []
    errors = []
    #     print(n_params)
    if plot is True:
        fig, ax = plt.subplots(len(labels), 2, constrained_layout=True, figsize=(10, len(labels)), sharex='col',
                               sharey='row', gridspec_kw={'width_ratios': [5, 1]})

    for i in range(n_params):

        #     if i == 9 :
        #         sample_all[:, :, i] = ((sample_all[:, :, i]/u.day).to('1/s') * maxs[6] * const.R_jup.to(u.km)).value
        samp = sample_all[:, :, i].ravel()  # cut_sample[:, :, i].ravel()
        sample_i = sample_all[:, :, i].reshape(sample_all.shape[0] * sample_all.shape[1], 1)

        if cut_sample is not None:
            samp = cut_sample[:, i]
            sample_i = cut_sample[:, i]
        #         his, edges = np.histogram(samp, bins='auto',density=True)
        #         mids = 0.5*(edges[1:] + edges[:-1])
        #         if flag_id is not None and i == flag_id[0]:
        #             binned = spectrum.box_binning(his[30:], flag_id[1])
        #             maximum = a.find_max_spline(mids[30:], binned, binning=True)[0]
        #         else:
        #             binned = spectrum.box_binning(his,bin_size)
        #             maximum = a.find_max_spline(mids, binned, binning=True)[0]
        #         maxs.append(maximum)
        if plot:
            #         print(i)
            ax[i, 0].plot(sample_all[:, :, i], "k", alpha=0.3)
            ax[i, 0].set_xlim(0, len(sample_all))
            ax[i, 0].set_ylabel(labels[i])

            his, edges, _ = ax[i, 1].hist(sample_i, bins=30, orientation='horizontal',
                                          weights=np.ones(len(sample_i)) / len(sample_i) * 100, color='k')
        else:
            his, edges = np.histogram(samp, bins=30, weights=np.ones(len(samp)) / len(samp) * 100, )

        mids = 0.5 * (edges[1:] + edges[:-1])
        #         print(i)
        if flag_id is not None:
            flag = np.array(flag_id)
            #             print(flag)
            if i in flag[:, 0]:
                #                 print('i in flag')
                binned = spectrum.box_binning(his[30:], flag[flag[:, 0] == i][0][1])
                maximum = a.find_max_spline(mids[30:], binned, binning=True)[0]
                if plot:
                    ax[i, 1].plot(binned, mids[30:], color='darkorange')
            else:
                binned = spectrum.box_binning(his, bin_size)
                maximum = a.find_max_spline(mids, binned, binning=True)[0]
                if plot:
                    ax[i, 1].plot(binned, mids, color='darkorange')
        else:
            binned = spectrum.box_binning(his, bin_size)
            maximum = a.find_max_spline(mids, binned, binning=True)[0]
            if plot:
                ax[i, 1].plot(binned, mids, color='darkorange')
        maxs.append(maximum)

        errbars2 = az.hdi(samp, prob)
        errors.append(errbars2)

        if plot:
            ax[i, 1].axhline(maximum, color='dodgerblue')
            ax[i, 1].axhspan(errbars2[0], errbars2[1], alpha=0.2, color='dodgerblue')
        #         ax[i,1].axhline(errbars2[0], linestyle='--')
        #         ax[i,1].axhline(errbars2[1], linestyle='--')
        print(maximum, errbars2 - maximum, errbars2)

    if print_gen_walkers:
        gen_walkers_init(errors)
    #     if plot:
    #         ax[-1,0].set_xlabel("Steps")

    #         fig, axes = plt.subplots(n_params, figsize=(10, n_params), sharex=True)
    #         for i in range(n_params):
    #             ax = axes[i]
    #             ax.plot(sample_all[:, :, i], "k", alpha=0.3)
    #             ax.set_xlim(0, len(sample_all))
    #             ax.set_ylabel(labels[i])
    #             ax.yaxis.set_label_coords(-0.1, 0.5)

    #         axes[-1].set_xlabel("step number");

    return maxs, errors


def gen_walkers_init(errors):
    params_str = ['{:6.1f}'.format(err_i) for err_i in np.array(errors)[:, 0]]
    err_str = ['{:6.1f}'.format(err_i) for err_i in np.array(errors)[:, 1] - np.array(errors)[:, 0]]

    print("pos = \ "[:-1] + "\n [" + ', '.join(params_str) + "] + \ "[:-1])
    print("([" + ', '.join(err_str) + '] * \ '[:-1])
    print("np.random.uniform(size=(nwalkers, {})))".format(np.array(errors)[:, 0].size))


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
    flat_logl = np.reshape(logl, (fin_pos.shape[0] * fin_pos.shape[1]))  # .shape
    flat_fin_pos = np.reshape(fin_pos, (fin_pos.shape[0] * fin_pos.shape[1], fin_pos.shape[-1]))  # .shape

    # --- place walker in order from worst [0] to best[-1] logL
    ord_pos = flat_fin_pos[np.argsort(flat_logl)]
    ord_logl = flat_logl[np.argsort(flat_logl)]

    if tol > 1:
        autocorr = emcee.autocorr.integrated_time(fin_pos, tol=tol)
    else:
        autocorr = np.zeros_like(ord_logl)

    return ord_pos, ord_logl, autocorr


def plot_corner_logl(ord_pos, labels, param_x=0, param_y=1, n_pts=1000, cmap='PuRd_r', n_reverse=20):
    couleurs = hm.get_colors(cmap, n_pts)

    for i in range(n_pts):
        pos_i = ord_pos[-i]
        plt.scatter(pos_i[param_x], pos_i[param_y], marker='o', color=couleurs[i], alpha=0.4)
    plt.plot(ord_pos[-n_reverse:][:, param_x], ord_pos[-n_reverse:][:, param_y], 'o-', color=couleurs[0], alpha=0.1,
             zorder=32)
    plt.scatter(ord_pos[-1][param_x], ord_pos[-1][param_y], marker='x', color='k', zorder=33)
    plt.xlabel(labels[param_x])
    plt.ylabel(labels[param_y])


def show_logl_per_param(ret_dict, ord_pos=None, ord_logl=None):
    if ord_pos is None:
        ord_pos = ret_dict['ord_pos']
    if ord_logl is None:
        ord_logl = ret_dict['ord_logl']
    fig, ax = plt.subplots(ord_pos.shape[-1], 1, figsize=(6, 3 * ord_pos.shape[-1]))

    # ylimits
    ymin, ymax = (np.quantile(ord_logl, 0.5), np.max(ord_logl))
    dy = ymax - ymin
    ylim = (ymin - 0.1 * dy, ymax + 0.1 * dy)

    for i_param, ax_i in enumerate(ax):
        ax_i.plot(ord_pos[:, i_param], ord_logl[:], ".k", alpha=0.1)
        #         ax_i.plot(ord_pos_best[:,i_param], ord_logl_best, ".", color='darkgreen')
        ax_i.set_ylim(*ylim)
        ax_i.set_xlabel(ret_dict['labels'][i_param])
    #     ax_i.axhline(23374, linestyle=":")
    plt.tight_layout()


def plot_ratios_corner(samps, values_compare, color='blue', add_solar=True, **kwargs):
    samp_c, samp_o, samp_h = samps
    c_sur_o = samp_c / samp_o
    samp_c_o_sur_h = (samp_c + samp_o) / samp_h
    sol_c_o_sur_h = np.sum(10 ** (np.array(values_compare)[1:]))
    #     stel_c_o_sur_h = np.sum(10**(np.array(values_stellar)[1:]))

    fig = corner.corner(np.concatenate((np.log10(samp_c_o_sur_h[:, None] / sol_c_o_sur_h), c_sur_o[:, None]), axis=1),
                        labels=['[(C+O)/H]', 'C/O'], show_titles=True, color=color, **kwargs)

    #     _ = corner.corner(np.concatenate((np.log10(samp_c_o_sur_h[:,None]/stel_c_o_sur_h), c_sur_o[:,None]), axis=1),
    #                       labels=['(C+O)/H', 'C/O'],
    #                             show_titles=True, color='red', **kwargs)

    if add_solar:
        axes = np.array(fig.axes).reshape((2, 2))
        axes[1, 0].scatter([0.0], [0.54], marker='o', color='k')

    return fig


def calc_tp_profile(params, temp_params, kind_temp='', TP=True, T_eq=None, pressures=None, nb_mols=None, params_id=None,
                    verbose=False):
    if nb_mols is None:
        print('Assuming that there are {} nb mols'.format(nb_mols))

    if pressures is None:
        pressures = temp_params['pressures']

    if params_id is not None:
        if verbose:
            print('Taking into account that there are {} molecules'.format(nb_mols))
        if params_id['tp_kappa'] is not None:
            temp_params['kappa_IR'] = 10 ** params[nb_mols + params_id['tp_kappa']]

        if params_id['tp_gamma'] is not None:
            if verbose:
                print(params, nb_mols, params_id['tp_gamma'])
            temp_params['gamma'] = 10 ** params[nb_mols + params_id['tp_gamma']]

        if params_id['tp_delta'] is not None:
            temp_params['delta'] = 10 ** params[nb_mols + params_id['tp_delta']]

        if params_id['tp_ptrans'] is not None:
            temp_params['ptrans'] = 10 ** params[nb_mols + params_id['tp_ptrans']]

        if params_id['tp_alpha'] is not None:
            temp_params['alpha'] = params[nb_mols + params_id['tp_alpha']]

        if params_id['tp_tint'] is not None:
            temp_params['T_int'] = params[nb_mols + params_id['tp_tint']]

        if params_id['temp'] is not None:
            temp_params['T_eq'] = params[nb_mols + params_id['temp']]

    if T_eq is not None:
        temp_params['T_eq'] = T_eq

    # print(temp_params, T_eq)
    if kind_temp == 'iso':
        temperatures = T_eq * np.ones_like(pressures)
    elif kind_temp == 'modif':
        if params_id is None:
            if TP is True:
                temp_params['delta'] = 10 ** params[-4]
                temp_params['gamma'] = 10 ** params[-3]
                temp_params['ptrans'] = 10 ** params[-2]
                temp_params['alpha'] = params[-1]

        temperatures = guillot_modif(pressures, temp_params['delta'], temp_params['gamma'], temp_params['T_int'],
                                     temp_params['T_eq'], temp_params['ptrans'], temp_params[
                                         'alpha'])  # print(  #       temp_params['delta'], \  #       temp_params['gamma'], \  #       temp_params['T_int'], \  #       T_eq,\  #       temp_params['ptrans'], \  #       temp_params['alpha'])
    else:
        if params_id is None:
            if TP is True:
                temp_params['kappa_IR'] = 10 ** params[-2]
                temp_params['gamma'] = 10 ** params[-1]
        temperatures = guillot_global(pressures, temp_params['kappa_IR'], temp_params['gamma'], temp_params['gravity'],
                                      temp_params['T_int'], temp_params['T_eq'])

    return temperatures


def plot_tp_profile(params, planet, errors, nb_mols, temp_params, params_id=None, kappa=-3, gamma=-1.5, T_int=500,
                    plot_limits=False, label='', color=None, radius_param=2, TP=True, zorder=None, kind_temp=''):
    #     T_eq = params[nb_mols]
    #     kappa_IR = 10**(kappa)
    #     gamma = 10**(gamma)

    #     T_int = 500.
    if planet.M_pl.ndim == 1:
        planet.M_pl = planet.M_pl[0]
    if radius_param is not None:
        temp_params['gravity'] = (const.G * planet.M_pl / (params[nb_mols + radius_param] * const.R_jup) ** 2).cgs.value

    temperature = calc_tp_profile(params, temp_params, kind_temp=kind_temp, TP=TP, nb_mols=nb_mols, params_id=params_id)
    temperature_up = calc_tp_profile(params, temp_params, kind_temp=kind_temp, TP=TP, T_eq=errors[nb_mols][1],
                                     nb_mols=nb_mols, params_id=params_id)
    temperature_down = calc_tp_profile(params, temp_params, kind_temp=kind_temp, TP=TP, T_eq=errors[nb_mols][0],
                                       nb_mols=nb_mols, params_id=params_id)
    print(temperature[0], temperature_up[0], temperature_down[0])

    if plot_limits:
        plt.plot(calc_tp_profile(params, temp_params, kind_temp=kind_temp, TP=TP, T_eq=500, nb_mols=nb_mols,
                                 params_id=params_id), np.log10(temp_params['pressures']), ':', alpha=0.5, color='grey',
                 label='T-P limits')
        plt.plot(calc_tp_profile(params, temp_params, kind_temp=kind_temp, TP=TP, T_eq=4000, nb_mols=nb_mols,
                                 params_id=params_id), np.log10(temp_params['pressures']), ':', alpha=0.5, color='grey')

    plt.plot(temperature, np.log10(temp_params['pressures']), label=label, color=color)
    plt.fill_betweenx(np.log10(temp_params['pressures']), temperature_down, temperature_up, alpha=0.15, color=color,
                      zorder=zorder)

    t1bar = \
    calc_tp_profile(params, temp_params, kind_temp=kind_temp, TP=TP, pressures=1, nb_mols=nb_mols, params_id=params_id)[
        0]

    #     print(t1bar, kappa_IR, gamma, gravity, T_int, errors[nb_mols][1])
    print('T @ 1 bar = {:.0f} + {:.0f} - {:.0f} '.format(t1bar,
                                                         calc_tp_profile(params, temp_params, kind_temp=kind_temp,
                                                                         TP=TP, T_eq=errors[nb_mols][1], pressures=1,
                                                                         nb_mols=nb_mols, params_id=params_id)[
                                                             0] - t1bar, t1bar -
                                                         calc_tp_profile(params, temp_params, kind_temp=kind_temp,
                                                                         TP=TP, T_eq=errors[nb_mols][0], pressures=1,
                                                                         nb_mols=nb_mols, params_id=params_id)[0]))


def plot_rot_ker(theta, planet, nb_mols, params_id, resol, left_val=1., right_val=1., alpha=0.7, no_label=False,
                 fig=None, color_comb='dodgerblue', color_ker='dodgerblue', label='Instrum. * Rot. (S)', **kwargs):
    if params_id['cloud_r'] is not None:
        right_val = theta[nb_mols + params_id['cloud_r']]
    if params_id['rpl'] is not None:
        radius = theta[nb_mols + params_id['rpl']]
    else:
        radius = planet.R_pl.to(u.R_jup).value

    if (params_id['wind_l'] is not None) or (params_id['wind_gauss'] is not None):
        if params_id['wind_r'] is None:
            if params_id['wind_l'] is not None:
                omega = [theta[nb_mols + params_id['wind_l']]]
            else:
                omega = [theta[nb_mols + params_id['wind_gauss']]]
        else:
            omega = [theta[nb_mols + params_id['wind_l']], theta[nb_mols + params_id['wind_r']]]

        rot_kwargs = {}
        if (params_id['wind_gauss'] is not None):
            rot_kwargs['gauss'] = True
            rot_kwargs['x0'] = 0
            rot_kwargs['fwhm'] = theta[nb_mols + params_id['wind_gauss']] * 1e3

    rotker = spectrum.RotKerTransitCloudy(radius * const.R_jup, planet.M_pl, theta[nb_mols + params_id['temp']] * u.K,
                                          np.array(omega) / u.day, resol, left_val=left_val, right_val=right_val,
                                          step_smooth=250., v_mid=0., **rot_kwargs, **kwargs)

    res_elem = rotker.res_elem
    v_grid, kernel, cloud = rotker.get_ker(n_os=1000)

    gauss_ker0 = hm.gauss(v_grid, 0.0, FWHM=res_elem)
    gauss_ker = gauss_ker0 / gauss_ker0.sum()
    _, ker_degraded = rotker.degrade_ker(n_os=1000)

    if (params_id['rv'] is not None):
        rv_shift = theta[nb_mols + params_id['rv']]
    else:
        rv_shift = 0.0

    if fig is None:
        fig = plt.figure(figsize=(7, 5))
        label1 = 'Instrum. res. elem.'
        label2 = 'Rotation kernel'
        plt.plot(v_grid / 1e3, gauss_ker, color="k", label=label1)
        plt.axvline(res_elem / 2e3, linestyle='--', color='gray')
        plt.axvline(-res_elem / 2e3, linestyle='--', color='gray')
    else:
        label2 = None
    if no_label:
        label = None
        label2 = None

    plt.plot(v_grid / 1e3 + rv_shift, kernel, color=color_ker, label=label2, zorder=10, linestyle=':', alpha=alpha)
    plt.plot(v_grid / 1e3 + rv_shift, ker_degraded, color=color_comb, label=label, zorder=12, alpha=alpha * 1.3)
    plt.legend(fontsize=13)
    return fig


def gen_params_id0(**kwargs):
    params_id = {'temp': None, 'cloud': None, 'rpl': None, 'kp': None, 'rv': None, 'wind_l': None, 'wind_r': None,
        'cloud_r': None, 'wind_gauss': None, 'tp_kappa': None, 'tp_delta': None, 'tp_gamma': None, 'tp_ptrans': None,
        'tp_alpha': None, 'scat_gamma': None, 'scat_factor': None}
    for key in list(kwargs.keys()):
        params_id[key] = kwargs[key]

    return params_id


def gen_params_id(list_params):
    params_id = {'temp': None, 'cloud': None, 'rpl': None, 'kp': None, 'rv': None, 'wind_l': None, 'wind_r': None,
        'cloud_r': None, 'wind_gauss': None, 'tp_kappa': None, 'tp_delta': None, 'tp_gamma': None, 'tp_ptrans': None,
        'tp_alpha': None, 'tp_tint': None, 'scat_gamma': None, 'scat_factor': None}
    count = 0
    for param in list_params:
        params_id[param] = count
        count += 1

    return params_id


def gen_params_id_p(params_priors):
    params_id = {'temp': None, 'cloud': None, 'rpl': None, 'kp': None, 'rv': None, 'wind_l': None, 'wind_r': None,
        'cloud_r': None, 'wind_gauss': None, 'tp_kappa': None, 'tp_delta': None, 'tp_gamma': None, 'tp_ptrans': None,
        'tp_alpha': None, 'tp_tint': None, 'scat_gamma': None, 'scat_factor': None}
    count = 0
    for param in list(params_priors.keys()):
        if (param != 'abund'):
            params_id[param] = count
            count += 1

    return params_id


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

# #         errbars2 = az.hdi(samp, 0.68)
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


def plot_corner(sample_all, labels=None, param_no_zero=4, maxs=None, errors=None, plot=True, **kwargs):
    #     print(sample_all.shape)
    ndim = sample_all.shape[-1]

    if labels is None:
        labels = [f'{idx}' for idx in range(ndim)]

    if sample_all.ndim == 2:
        flat_samples = sample_all
    else:
        flat_samples = sample_all.reshape(sample_all.shape[0] * sample_all.shape[1], ndim)

    if plot is True:
        fig = corner.corner(flat_samples, labels=labels,
                            #                         truths=[None,None,None,None, None, 130,None, None, None, None],
                            # quantiles=[0.16, 0.5, 0.84],
                            show_titles=True, **kwargs)  # range=[(-6,-1), (-8,-1), (-8,-1), (-8,-1), (700,1200),
        #     (-5,0), (1.0,1.5), (125,135), (-10,-5), (0,20)])#, **corner_kwargs);

        if maxs is None:
            maxs, errors = find_dist_maxs(sample_all, labels, bin_size=6)

        # --- Extract the axes
        axes = np.array(fig.axes).reshape((ndim, ndim))
        for i in range(ndim):
            axes[ndim - 1, i].set_xlabel(labels[i], fontsize=16)

        for i in range(1, ndim):
            axes[i, 0].set_ylabel(labels[i], fontsize=16)

        for i in range(ndim):
            #     if i == 3:
            #         maxs[i] = 0
            #         errors[i][1] = 0

            axes[i, i].axvline(maxs[i], color='k')
            axes[i, i].axvline(errors[i][0], color='k', linestyle='--')
            axes[i, i].axvline(errors[i][1], color='k', linestyle='--')

            if i == param_no_zero:
                float_str_moins = "{:.0f}".format(errors[i][0] - maxs[i])
                float_str_plus = "{:.0f}".format(errors[i][1] - maxs[i])

                axes[i, i].set_title(
                    r' {} = {:.0f}$_{{{}}}^{{+{}}}$'.format(labels[i], maxs[i], float_str_moins, float_str_plus))
            else:
                float_str_moins = "{:.2f}".format(errors[i][0] - maxs[i])
                float_str_plus = "{:.2f}".format(errors[i][1] - maxs[i])

                axes[i, i].set_title(
                    r' {} = {:.2f}$_{{{}}}^{{+{}}}$'.format(labels[i], maxs[i], float_str_moins, float_str_plus))
    else:
        fig = plt.figure()

    return fig, flat_samples


def gen_ret_dict(ret_params, ret_name, files, labels, discard=0, tol=8, plot=True, filename2=None, sig2=True,
                 corner=False, discard_after=None, min_logl=None):
    ret_params[ret_name]['filename'] = files

    sample_all_files = []
    ord_pos_files = []
    ord_logl_files = []
    for file in files:
        try:
            ord_pos, ord_logl, _ = read_walker_prob(file, tol=tol)
            ord_pos_files.append(ord_pos)
            ord_logl_files.append(ord_logl)
        except OSError:
            print('Corrupted file, use discard_after=N to try and see where it causes problem.')

        sample_all = read_walkers_file(file, discard=discard, discard_after=discard_after)
        sample_all_files.append(sample_all)

    if len(files) > 1:
        # cut_sample2 = read_walkers_file(filename2, discard=discard)
        sample_all = np.concatenate(sample_all_files, axis=0)
        ord_pos = np.concatenate(ord_pos_files, axis=0)
        ord_logl = np.concatenate(ord_logl_files, axis=0)

    print(sample_all.shape)

    if min_logl is not None:
        cut_sample = ord_pos[ord_logl > min_logl]
    else:
        cut_sample = None

    maxs, errors = find_dist_maxs(sample_all, labels, bin_size=6, plot=plot,
                                  cut_sample=cut_sample)  # , prob=0.954)#, plot=False)#
    if sig2 is True:
        print('')
        print('2sigma values')
        maxs2s, errors2s = find_dist_maxs(sample_all, labels, bin_size=6, prob=0.954, plot=False, cut_sample=cut_sample)
        # maxs3s, errors3s = find_dist_maxs(sample_all, labels, bin_size=6, prob=0.9973, plot=False)
        ret_params[ret_name]['errors2s'] = errors2s.copy()

    if corner is True:
        fig, flat_samples = plot_corner(sample_all, labels, param_no_zero=2, maxs=maxs, errors=errors)
    else:
        flat_samples = sample_all.reshape(sample_all.shape[0] * sample_all.shape[1], sample_all.shape[-1])

    ret_params[ret_name]['sample'] = sample_all.copy()
    ret_params[ret_name]['ord_pos'] = ord_pos.copy()
    ret_params[ret_name]['ord_logl'] = ord_logl.copy()
    ret_params[ret_name]['flat_sample'] = flat_samples
    ret_params[ret_name]['best'] = ret_params[ret_name]['ord_pos'][-1].copy()
    if min_logl is not None:
        ret_params[ret_name]['meds'] = np.median(cut_sample, axis=0).copy()
    else:
        ret_params[ret_name]['meds'] = np.median(flat_samples, axis=0).copy()
    ret_params[ret_name]['maxs'] = maxs.copy()
    ret_params[ret_name]['errors'] = errors.copy()
    if min_logl is not None:
        ret_params[ret_name]['cut_sample'] = cut_sample


#     print(maxs)
#     return

def plot_TP_profile_samples(ret_params, ret_names,  # nb_mols,
                            title='', labels=None, colors=None, dark_colors=None, kind=['best'], xlim=None, slim=False,
                            n_samp=100, samples=None, confid_interv=False, show_2sig=True, linestyle=['-', '--', ':'],
                            params=None, **kwargs):
    if slim:
        fig = plt.figure(figsize=(3, 5))
    else:
        fig, ax = plt.subplots(1, 1)

    if colors is None:
        colors = [(200 / 255, 30 / 255, 0 / 255), (100 / 255, 190 / 255, 240 / 255), 'gold']
    if dark_colors is None:
        dark_colors = ['darkred', 'royalblue', 'darkgoldenrod']
    if labels is None:
        labels = ['1', '2', '3']
    if xlim is None:
        xlim = [500, 10000]

    for ret_i, ret_name_i in enumerate(ret_names):
        temp_sample = []

        if slim is False:
            if samples is None:
                sample = ret_params[ret_name_i]['flat_sample']
            else:
                if samples[ret_i] is None:
                    sample = ret_params[ret_name_i]['flat_sample']
                else:
                    sample = samples[ret_i]

            for i in range(n_samp):
                hm.print_static(i)
                params_i = sample[random.randint(0, sample.shape[0] - 1)]  # [random.randint(0, sample.shape[1] - 1)]
                #                 print(params_i)
                #                 print(ret_params[ret_name_i]['nb_mols'])
                #                 print(ret_params[ret_name_i]['temp_params'])
                profile_i = calc_tp_profile(params_i, ret_params[ret_name_i]['temp_params'],
                                            nb_mols=ret_params[ret_name_i]['nb_mols'],
                                            kind_temp=ret_params[ret_name_i]['kwargs']['kind_temp'],
                                            params_id=ret_params[ret_name_i]['params_id'])
                if confid_interv is False:
                    plt.plot(profile_i, ret_params[ret_name_i]['temp_params']['pressures'], color=colors[ret_i],
                             alpha=0.1)
                temp_sample.append(profile_i)

        temp_sample = np.array(temp_sample)
        if confid_interv:
            temp_stats = {'1-sig': [], '2-sig': []}
            for temp_sample_p in temp_sample.T:
                temp_stats['1-sig'].append(az.hdi(temp_sample_p, hdi_prob=.68))
                temp_stats['2-sig'].append(az.hdi(temp_sample_p, hdi_prob=.954))
            for key, val in temp_stats.items():
                temp_stats[key] = np.array(val).T

            temp_stats['median'] = np.median(temp_sample, axis=0)
            fig, ax = plot_tp_sample(ret_params[ret_name_i]['temp_params']['pressures'], temp_stats,
                                     line_color=dark_colors[ret_i], region_color=colors[ret_i], fig=fig, ax=ax,
                                     show_2sig=show_2sig, label=labels[ret_i], **kwargs)

        if kind is not None:
            for i, kind_i in enumerate(kind):
                if len(kind) > 1:
                    label_kind = kind_i
                else:
                    label_kind = ''
                plt.plot(calc_tp_profile(ret_params[ret_name_i][kind_i], ret_params[ret_name_i]['temp_params'],
                                         nb_mols=ret_params[ret_name_i]['nb_mols'],
                                         kind_temp=ret_params[ret_name_i]['kwargs']['kind_temp'],
                                         params_id=ret_params[ret_name_i]['params_id']),
                         ret_params[ret_name_i]['temp_params']['pressures'], label=labels[ret_i] + ' ' + label_kind,
                         color=dark_colors[ret_i], zorder=50, linestyle=linestyle[i])
        if params is not None:
            for i, params_i in enumerate(params):
                plt.plot(calc_tp_profile(params_i, ret_params[ret_name_i]['temp_params'],
                                         nb_mols=ret_params[ret_name_i]['nb_mols'],
                                         kind_temp=ret_params[ret_name_i]['kwargs']['kind_temp'],
                                         params_id=ret_params[ret_name_i]['params_id']),
                         ret_params[ret_name_i]['temp_params']['pressures'], label=labels[ret_i] + ' ' + label_kind,
                         color=(1.0, 0, i * 0.2), zorder=50, linestyle=linestyle[i])

    plt.ylim(1e1, 1e-6)
    plt.yscale('log')
    plt.legend(fontsize=13, loc="upper left", bbox_to_anchor=(1.1, 1.0))
    plt.xlabel(r'Temperature [K]', fontsize=16)
    plt.ylabel(r'Pressure [bar]', fontsize=16)
    plt.title(title, fontsize=16)
    plt.xlim(*xlim)
    return fig


def plot_tp_sample(pressures, temp_stats, line_color='forestgreen', region_color='limegreen', p_range=(1e-6, 1e1),
                   tight_range=True, fig=None, ax=None, alpha=0.5, **kwargs):
    fkwargs = dict(line_color=line_color, region_color=region_color, p_range=p_range, fig=fig, ax=ax, alpha=alpha,
                   tight_range=tight_range, **kwargs)
    fig, ax = plot_p_profile_sample(pressures, temp_stats, **fkwargs)

    ax.set_xlabel('Temperature [K]', fontsize=16)
    ax.set_ylabel('Pressure [bar]', fontsize=16)

    return fig, ax


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
        idx = (x_range[0] <= x_array) & (x_array <= x_range[-1])

    return idx


def plot_p_profile_sample(pressures, sample_stats, line_color=None, region_color=None, p_range=(1e-6, 1e1), fig=None,
                          ax=None, alpha=0.2, tight_range=True, show_2sig=True, **kwargs):
    if show_2sig:
        sigmas = ['1-sig', '2-sig']
    else:
        sigmas = ['1-sig']

    fig, ax = _get_fig_and_ax_inputs(fig, ax)

    idx = _get_idx_in_range(pressures, p_range)

    (line,) = ax.semilogy(sample_stats['median'][idx], pressures[idx], color=line_color, **kwargs)

    if region_color is None:
        region_color = line.get_color()

    for key in sigmas:
        x1, x2 = sample_stats[key]
        ax.fill_betweenx(pressures[idx], x1[idx], x2[idx], color=region_color, alpha=alpha)

    if tight_range:
        ax.set_ylim(np.min(pressures[idx]), np.max(pressures[idx]))

    ylim = ax.get_ylim()
    if ylim[-1] > ylim[0]:
        ax.invert_yaxis()

    return fig, ax


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


# def lnprob(theta, ):
#     total = 0.
#
#     total += log_prior_params(theta, nb_mols, params_id, params_prior)
#
#     #     print('Prior ',total)
#
#     if not np.isfinite(total):
#         #         print('prior -inf')
#         return -np.inf
#
#     if params_id['cloud'] is not None:
#         pcloud = 10 ** (theta[nb_mols + params_id['cloud']])  # in bars
#     else:
#         pcloud = None
#
#     if params_id['rpl'] is not None:
#         rpl = theta[nb_mols + params_id['rpl']]
#         temp_params['gravity'] = (const.G * planet.M_pl / (rpl * const.R_jup) ** 2).cgs.value
#     else:
#         rpl = planet.R_pl.to(u.R_jup).value
#
#     if params_id['tp_kappa'] is not None:
#         kappa = 10 ** theta[nb_mols + params_id['tp_kappa']]
#     else:
#         kappa = temp_params['kappa_IR']
#
#     if params_id['tp_gamma'] is not None:
#         gamma = 10 ** theta[nb_mols + params_id['tp_gamma']]
#     else:
#         gamma = temp_params['gamma']
#
#     if params_id['tp_delta'] is not None:
#         delta = 10 ** theta[nb_mols + params_id['tp_delta']]
#     else:
#         delta = temp_params['delta']
#
#     if params_id['tp_ptrans'] is not None:
#         ptrans = 10 ** theta[nb_mols + params_id['tp_ptrans']]
#     else:
#         ptrans = temp_params['ptrans']
#
#     if params_id['tp_alpha'] is not None:
#         alpha = theta[nb_mols + params_id['tp_alpha']]
#     else:
#         alpha = temp_params['alpha']
#
#     if params_id['scat_gamma'] is not None:
#         gamma_scat = theta[nb_mols + params_id['scat_gamma']]
#     else:
#         gamma_scat = None
#
#     if params_id['scat_factor'] is not None:
#         factor = theta[nb_mols + params_id['scat_factor']]  # * (5.31e-31*u.m**2/u.u).cgs.value
#     else:
#         factor = None
#
#     if (params_id['wind_l'] is not None) or (params_id['wind_gauss'] is not None):
#         if params_id['wind_r'] is None:
#             omega = [theta[nb_mols + params_id['wind_l']]]
#         else:
#             omega = [theta[nb_mols + params_id['wind_l']], theta[nb_mols + params_id['wind_r']]]
#
#         rot_kwargs = {
#             'rot_params': [rpl * const.R_jup, planet.M_pl,
#                            theta[nb_mols + params_id['temp']] * u.K,
#                            omega]
#         }
#         if (params_id['wind_gauss'] is not None):
#             rot_kwargs['gauss'] = True
#             rot_kwargs['x0'] = 0
#             rot_kwargs['fwhm'] = theta[nb_mols + params_id['wind_gauss']] * 1e3
#
#     else:
#         rot_kwargs = {'rot_params': None}
#
#     #     if plot:
#     #         plt.figure(figsize=(16,8))
#
#     # --- Generating the temperature profile
#     if kind_temp == "modif":
#         temperatures = guillot_modif(temp_params['pressures'], delta, gamma,
#                                      temp_params['T_int'],
#                                      theta[nb_mols + params_id['temp']],
#                                      ptrans, alpha)
#     elif kind_temp == 'iso':
#         temperatures = theta[nb_mols + params_id['temp']] * np.ones_like(temp_params['pressures'])
#     else:
#         temperatures = guillot_global(temp_params['pressures'],
#                                       kappa, gamma, temp_params['gravity'],
#                                       temp_params['T_int'],
#                                       theta[nb_mols + params_id['temp']])
#
#     ####################
#     # --- HIGH RES --- #
#     ####################
#
#     if (retrieval_type == 'JR') or (retrieval_type == 'HRR'):
#         #         print('High res')
#
#         atmos_high, species_high = prt_high
#
#         # --- Updating the abundances
#         for i_mol, mol in enumerate(list(species_high.keys())[:len(list_mols)]):
#             species_high[mol] = 10 ** (theta[i_mol])
#
#         if continuum_opacities is not None:
#             if 'H-' in continuum_opacities:
#                 species_high['H-'] = 10 ** theta[nb_mols - 1]
#                 species_high['H'] = 10 ** (-99.)
#                 species_high['e-'] = 10 ** (-6.0)
#
#         #         print(species_high)
#
#         # --- Generating the model
#         wv_high, model = prt.retrieval_model_plain(atmos_high, species_high, planet, temp_params['pressures'],
#                                                    temperatures, temp_params['gravity'], P0, pcloud,
#                                                    rpl * const.R_jup.cgs.value, planet.R_star.cgs.value,
#                                                    kind_trans=kind_trans, dissociation=dissociation,
#                                                    fct_star=fct_star,
#                                                    gamma_scat=gamma_scat, kappa_factor=factor)
#
#         if not np.isfinite(model[100:-100]).all():
#             print("NaN in high res model spectrum encountered")
#             return -np.inf
#
#         # --- Downgrading and broadening the model (if broadening is included)
#         wv_high, model_high = prt.prepare_model(wv_high, model, int(1e6 / opacity_sampling),
#                                                 Raf=instrum['resol'], **rot_kwargs)
#         #         print(wv_high, model_high)
#
#         if plot:
#             plt.plot(wv_high, model_high, alpha=0.3)
#
#         logl_i = []
#         # --- Computing the logL for all sequences
#         for tr_i in data_trs.keys():
#             #             print(tr_i)
#             vrp_orb = rv_theo_t(theta[nb_mols + params_id['kp']], \
#                                 data_trs[tr_i]['t_start'] * u.d, planet.mid_tr, planet.period, plnt=True).value
#             logl_tr = corr.calc_log_likelihood_grid_retrieval(theta[nb_mols + params_id['rv']],
#                                                               data_trs[tr_i], planet, wv_high, model_high,
#                                                               data_trs[tr_i]['flux'], data_trs[tr_i]['s2f'],
#                                                               vrp_orb=vrp_orb, vr_orb=-vrp_orb * Kp_scale,
#                                                               # resol=instrum['resol'],
#                                                               nolog=nolog,
#                                                               # inj_alpha=inj_alpha,
#                                                               alpha=np.ones_like(data_trs[tr_i]['t_start']),
#                                                               kind_trans=kind_trans)
#
#             #             print(data_trs[tr_i]['flux'][0])
#             if not np.isfinite(logl_tr).all():
#                 return -np.inf
#
#             logl_i.append(logl_tr)
#         #             print(logl_tr)
#         # --- Computing the total logL
#         #         spirou_logl = corr.sum_logl(np.concatenate(np.array(logl_i), axis=0),
#         #                                data_info['trall_icorr'], orders, data_info['trall_N'],
#         #                                axis=0, del_idx=data_info['bad_indexs'],
#         #                                      nolog=True, alpha=data_info['trall_alpha_frac'])
#         #         total += spirou_logl.copy()
#         total += corr.sum_logl(np.concatenate(np.array(logl_i), axis=0),
#                                data_info['trall_icorr'], orders, data_info['trall_N'],
#                                axis=0, del_idx=data_info['bad_indexs'],
#                                nolog=True, alpha=data_info['trall_alpha_frac'])
#
#     #         print('SPIRou', spirou_logl)
#     #         print('SPIRou', total)
#     ###################
#     # --- LOW RES --- #
#     ###################
#     if ((retrieval_type == 'JR') and (spitzer is not None)) or (retrieval_type == 'LRR') or (prt_low is not None):
#         #         print('Low res')
#
#         atmos_low, species_low = prt_low
#
#         # --- Updating the abundances
#         for i_mol, mol in enumerate(list(species_low.keys())[:len(list_mols)]):
#             species_low[mol] = 10 ** (theta[i_mol])
#
#         if continuum_opacities is not None:
#             if 'H-' in continuum_opacities:
#                 species_low['H-'] = 10 ** theta[nb_mols - 1]
#                 species_low['H'] = 10 ** (-99.)
#                 species_low['e-'] = 10 ** (-6.0)
#         #         print(species_low)
#
#         # --- Generating the model
#         wv_low, model_low = \
#             prt.retrieval_model_plain(atmos_low, species_low, planet, temp_params['pressures'],
#                                       temperatures, temp_params['gravity'], P0, pcloud,
#                                       rpl * const.R_jup.cgs.value, planet.R_star.cgs.value,
#                                       kind_trans=kind_trans, dissociation=dissociation,
#                                       fct_star=fct_star_low,
#                                       gamma_scat=gamma_scat, kappa_factor=factor)
#
#         if np.sum(np.isnan(model_low)) > 0:
#             print("NaN in low res model spectrum encountered")
#             return -np.inf
#
#         if plot:
#             plt.plot(wv_low, model_low)
#
#         if spitzer is not None:
#             #             print('Spitzer')
#             spit_wave, spit_data, spit_data_err, wave_sp, fct_sp = spitzer
#             if plot:
#                 plt.errorbar(spit_wave, spit_data, spit_data_err,
#                              color='k', marker='.', linestyle='none', zorder=32)
#
#             spit_mod = []
#             for wave_i, fct_i in zip(wave_sp, fct_sp):
#                 # --- Computing the model broadband point
#                 cond = (wv_low >= wave_i[0]) & (wv_low <= wave_i[-1])
#                 spit_mod.append(np.average(model_low[cond], weights=fct_i(wv_low[cond])))
#
#             if plot:
#                 plt.plot(spit_wave, np.array(spit_mod), '.')
#             # --- Computing the logL
#             #             spitzer_logl = -1/2*corr.calc_chi2(spit_data, spit_data_err, np.array(spit_mod))
#             #             total += spitzer_logl.copy()
#             total += -1 / 2 * corr.calc_chi2(spit_data, spit_data_err, np.array(spit_mod))
#
#     #             print('Spitzer', spitzer_logl)
#     if (retrieval_type == 'JR') or (retrieval_type == 'LRR'):
#         #         print('HST')
#
#         if (retrieval_type == 'JR') and (spitzer is None):
#             # --- If no Spitzer or STIS data is included, only the high res model is generated
#             # and this is the model that will be downgraded for WFC3
#             wv_low, model_low = wv_high, model_high
#             Rbf = instrum['resol']
#             R_sampling = int(1e6 / opacity_sampling)
#         else:
#             Rbf = 1000
#             R_sampling = 1000
#         #         print(Rbf)
#         for instrument in hst.keys():
#             hst_wave, hst_data, hst_data_err, hst_res = hst[instrument]
#             #             print(hst_wave, hst_data, hst_data_err, hst_res)
#
#             cond = (wv_low >= hst_wave[0] - 0.05) & (wv_low <= hst_wave[-1] + 0.05)
#
#             _, resamp_prt = spectrum.resampling(wv_low[cond], model_low[cond],
#                                                 Raf=hst_res, Rbf=Rbf, sample=wv_low[cond])
#             binned_prt_hst = spectrum.box_binning(resamp_prt, R_sampling / hst_res)
#             fct_prt = interp1d(wv_low[cond], binned_prt_hst)
#             mod = fct_prt(hst_wave)
#             #             print(mod)
#             if plot:
#                 plt.plot(wv_low[cond], model_low[cond])
#                 plt.errorbar(hst_wave, hst_data, hst_data_err,
#                              color='k', marker='.', linestyle='none', zorder=32)
#                 plt.plot(hst_wave, mod, '.')
#                 plt.xlim(None, 2.0)
#             # --- Computing the logL
#             #             hst_logl = -1/2*corr.calc_chi2(hst_data, hst_data_err, mod)
#             #             total += hst_logl.copy()
#             total += -1 / 2 * corr.calc_chi2(hst_data, hst_data_err, mod)
#
#         del wv_low, model_low
#
#     if retrieval_type != 'LRR':
#         del wv_high, model_high
#
#     gc.collect()
#
#     return total


def init_uniform_prior(prior_inputs, n_wlkr):
    """Initialize from uniform prior"""
    low, high = prior_inputs
    sample = np.random.uniform(low, high, size=(n_wlkr, 1))

    return sample


def init_from_prior(n_wlkrs, prior_init_func, prior_dict, n_mol=1, special_treatment=None):
    """Code to initialize walkers based on the prior function.
    The `prior_init_func` is a dict of functions associated to each prior type.
    Each of these functions is given the arguments after the prior name
    in the global variable `params_prior` plus the number of walkers `n_wlkrs`.
    The function must return an array with size=(n_wlkr, 1).
    """

    log.info(f"Generating initial walkers from prior.")

    if special_treatment is None:
        special_treatment = dict()

    walkers_init = []

    # Use the prior for all parameters
    for key, prior in prior_dict.items():

        if key in special_treatment:
            prior = special_treatment[key]
            log.info(f"Specific treatment for '{key}' initialization -> {prior}")

        prior_name, prior_args = prior[0], prior[1:]
        try:
            init_func = prior_init_func[prior_name]
        except KeyError:
            raise KeyError(f"No function associated to prior '{prior_name}' for param '{key}'. "
                           "Specify it in input `prior_init_func`.")

        #### Remove the first part of the if once the molecule names
        #### are directly given as parameters, not 'abund'.
        if key == 'abund':
            for _ in range(n_mol):
                # Run init function
                sample = init_func(prior_args, n_wlkrs)
                walkers_init.append(sample)
        else:
            # Run init function
            sample = init_func(prior_args, n_wlkrs)
            walkers_init.append(sample)

    # Stack all parameters
    walkers_init = np.hstack(walkers_init)

    return walkers_init


def init_from_burnin(n_walkers, n_best_min=1000, quantile=None, wlkr_file=None, wlkr_path=''):
    """Draw `n_walkers` walkers among the ones with the best logL found in `wlkr_file`.
    It returns the parameters values to initiate a new sampler (n_walker, n_parameters).
    The values are drawn within the N best walkers (`n_best_min`) or the best `quantile`.
    Can be set to None to force to use the other alternative."""

    if wlkr_file is None:
        raise NotImplementedError("The walker file (`wlkr_file`) is required for now.")
    else:
        wlkr_file = Path(wlkr_file)
        wlkr_path = Path(wlkr_path)

    # Ordered logl and posteriors (flatten)
    output = read_walker_prob(wlkr_path / wlkr_file, tol=1)
    ord_pos = output[0]
    ord_logl = output[1]

    # Best walkers among the x quantile or at least the n best
    n_total = ord_pos.shape[0]

    # Take most restrictive values
    if n_best_min is None:
        n_best_min = n_total
    else:
        n_best_min = np.min([n_best_min, n_total])

    if quantile is None:
        n_sample = n_best_min
    else:
        n_quantile = int(quantile * n_total)  # Number of values in quantile
        n_sample = np.min([n_quantile, n_best_min])

    n_sample = np.max([n_walkers, n_sample])  # At least the number of walkers

    # Take random integers (no repeated value)
    rng = np.random.default_rng()
    random_int = rng.permutation(range(n_sample))[:n_walkers]

    # Associated index in ord_pos
    idx = n_total - random_int
    walker_init = ord_pos[idx, :]
    logl = ord_logl[idx]

    return walker_init, logl


## Utilities to compute model profiles and spectra from sample
def get_tp_from_retrieval(param, retrieval_obj):
    prt_args, prt_kwargs, rot_kwargs, _ = retrieval_obj.prepare_prt_inputs(param)

    pressures = retrieval_obj.temp_params['pressures']
    temperatures = prt_args[0]

    return pressures, temperatures


def gen_abundances_default(param, retrieval_obj=None, tp_fct=None, vmrh2he=None):
    if retrieval_obj is None:
        raise ValueError('For now, `retrieval_obj` must be specified.')

    if tp_fct is None:
        tp_fct = partial(get_tp_from_retrieval, retrieval_obj=retrieval_obj)

    if vmrh2he is None:
        vmrh2he = [0.85, 0.15]

    pressures, temp_profile = tp_fct(param)
    include_dissociation = retrieval_obj.dissociation

    species_dict = retrieval_obj.update_abundances(param)
    mol_names = list(species_dict.keys())
    mol_abundances = list(species_dict.values())

    _, _, out = prt.gen_abundances(mol_names, mol_abundances, pressures, temp_profile, verbose=False, vmrh2he=vmrh2he,
                                   dissociation=include_dissociation, scale=1.0, plot=False)

    return out


def draw_profiles_form_sample(n_draw, flat_samples, list_mols=None, retrieval_obj=None, get_tp_from_param=None,
                              get_mol_profile=None):
    """
    Compute profiles (temperature and abundance profiles) as a function of pressure for `n_draw`
    in a flattened sample of parameters.
    The (imported) retrieval that was used to generate the sample can be given as an input.
    Args:
        n_draw: integer
        flat_samples: 2d array (n_sample, n_params)
        list_mols: list of molecules (optional)
        retrieval_obj: imported retrieval object. (optional)
                       Use `retrieval_obj = importlib.import_module(retrieval_code_filename)`
        get_tp_from_param: Function that returns the tp-profile (output -> pressure, temperature).
                           Default is the one used in the `retrieval_obj`
        get_mol_profile: Function that returns the abundances (output -> dict('molecule': molecule_profile)).
                         Default is `.petitradtrans_utils.gen_abundances`.

    Returns:
        dictionnary with 'temperature' and all molecules profiles.
    """
    if list_mols is None:
        list_mols = retrieval_obj.list_mols + retrieval_obj.continuum_opacities
    else:
        list_mols = list_mols.copy()

    if get_tp_from_param is None:
        # Function of the retrieval
        get_tp_from_param = partial(get_tp_from_retrieval, retrieval_obj=retrieval_obj)

    if get_mol_profile is None:
        get_mol_profile = partial(gen_abundances_default, retrieval_obj=retrieval_obj)

    # Take random integers (no repeated value)
    random_idx = rng.permutation(range(flat_samples.shape[0]))[:n_draw]

    # Other molecule profiles will be added to this dictionary on the fly
    profile_samples = {'temperature': []}

    for param_i in flat_samples[random_idx]:

        param_i = param_i.copy()

        pressures, temp_profile = get_tp_from_param(param_i)

        profile_samples['temperature'].append(temp_profile)

        # Molecule profiles
        mols_profiles = get_mol_profile(param_i)

        for key, val in mols_profiles.items():
            try:
                profile_samples[key]
            except KeyError:
                profile_samples[key] = list()
            profile_samples[key].append(val)

    # Convert to array
    for key, val in profile_samples.items():
        profile_samples[key] = np.array(val)

    return profile_samples, pressures


def draw_tp_profiles_from_sample(n_draw, flat_samples, retrieval_obj=None, get_tp_from_param=None):
    """
    Compute profiles (temperature and abundance profiles) as a function of pressure for `n_draw`
    in a flattened sample of parameters.
    The (imported) retrieval that was used to generate the sample can be given as an input.
    Args:
        n_draw: integer
        flat_samples: 2d array (n_sample, n_params)
        retrieval_obj: imported retrieval object. (optional)
                       Use `retrieval_obj = importlib.import_module(retrieval_code_filename)`
        get_tp_from_param: Function that returns the tp-profile (output -> pressure, temperature).
                           Default is the one used in the `retrieval_obj`
    Returns:
        Array af temperature profiles.
    """
    if get_tp_from_param is None:
        # Function of the retrieval
        get_tp_from_param = partial(get_tp_from_retrieval, retrieval_obj=retrieval_obj)

    # Take random integers (no repeated value)
    random_idx = rng.permutation(range(flat_samples.shape[0]))[:n_draw]

    # Other molecule profiles will be added to this dictionary on the fly
    tp_samples = []

    for param_i in flat_samples[random_idx]:

        param_i = param_i.copy()

        pressures, temp_profile = get_tp_from_param(param_i)

        tp_samples.append(temp_profile)

    # Convert to array
    tp_samples = np.array(tp_samples)

    return tp_samples, pressures


def get_stats_from_profile(profile_sample, key_names=None,
                           prob_values=None, log_scale=False):
    """
    Compute stats from sample of profiles or spectra (sample of 1d arrays)
    Args:
        profile_sample: 2d array
            Sample of profiles or spectra with shape = (number of sample, length of profile or spectra).
        key_names: list of strings
             Names of the statistics to compute. Each names is associated with a percentile given by `prob_values`.
             Default is ('1-sig', '2-sig', '3-sig').
        prob_values: list of floats
             Percentiles to compute associated to `key_names`. Needs to have the same length as `key_names`.
             Default is (0.68, 0.954, 0.997).
        log_scale: boolean
            If True, compute the stats in log space.

    Returns:
        dictionary with the keys specified in `key_names`. The 'median' is added to the dictionary.
    """
    # Default values
    if key_names is None:
        if prob_values is None:
            key_names = ['1-sig', '2-sig', '3-sig']
            prob_values = [0.68, 0.954, 0.997]
        else:
            key_names = [str(p_val) for p_val in prob_values]

    # Compute stats in log space?
    if log_scale:
        profile_sample = np.log(profile_sample)

    # Initialize
    stats = {key: [] for key  in key_names}
    # Compute stats for each sample
    for sample_p in profile_sample.T:
        # Iterate over the different percentiles
        for key, p_val in zip(key_names, prob_values):
            stats[key].append(az.hdi(sample_p, hdi_prob=p_val))
    # Convert to array
    for key, val in stats.items():
        stats[key] = np.array(val).T

    # Add median to stats
    stats['median'] = np.median(profile_sample, axis=0)

    # Convert back to linear space
    if log_scale:
        for key in stats:
            stats[key] = np.exp(stats[key])

    return stats


# Functions to work with spectra samples drawn from a retrieval chain
def get_shift_spec_sample(spec_sample, wave=None, idx=None, wv_norm=None):
    if idx is None:
        idx = unpack_wv_norm_and_get_idx(wave, wv_norm)

    out = np.mean(spec_sample[:, idx], axis=-1)[:, None]

    return out


def unpack_wv_norm_and_get_idx(wave, wv_norm):
    try:
        idx = _get_idx_in_range(wave, wv_norm)
    except TypeError:
        # Assume non-iterable
        idx = [np.searchsorted(wave, wv_norm)]
    except ValueError:
        # Assume wrong shape, so many ranges
        idx = [_get_idx_in_range(wave, wv_range) for wv_range in wv_norm]
        idx = np.concatenate(idx)  # idx = np.logical_or.reduce(idx)??

    return idx


def normalize_spec_sample(spec_sample, wave=None, shift_fct=None, wv_norm=None, scale_fct=None):
    if shift_fct is not None:
        try:
            y_shift = shift_fct(spec_sample, axis=-1)[:, None]
        except TypeError:
            y_shift = np.apply_along_axis(shift_fct, -1, spec_sample)[:, None]

    elif wv_norm is None:
        y_shift = 0

    elif wave is None:
        raise ValueError('`wave` must be given to be used with `wv_norm`.')

    else:
        y_shift = get_shift_spec_sample(spec_sample, wave=wave, wv_norm=wv_norm)

    # Apply shift
    out = spec_sample - y_shift

    # Compute scaling factor
    if scale_fct is not None:
        try:
            y_scale = scale_fct(out, axis=-1)[:, None]
        except TypeError:
            y_scale = np.apply_along_axis(scale_fct, -1, out)[:, None]
    else:
        y_scale = 1

    # Apply scaling
    out = out / y_scale

    return out


def get_all_param_names(retrieval_obj):
    """
    Get all parameters names (species + others) from the retrieval object.
    Args:
        retrieval_obj: retrieval object
            Created by importing the retrieval code.
    Returns:
        list of all parameters names.
    """

    # Get params names (other then species)
    params_id = retrieval_obj.params_id
    valid_params = [key for key, idx in params_id.items() if idx is not None]

    # Sort with respect to param_id
    valid_params = sorted(valid_params, key=lambda x: params_id[x])

    # Combine all params
    params = retrieval_obj.list_mols + retrieval_obj.continuum_opacities + valid_params

    return params