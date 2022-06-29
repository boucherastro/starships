# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:47:17 2016

@author: AnneBoucher

"""
from __future__ import division
import numpy as np

from starships import homemade as hm
from starships.extract import get_var_res, get_res
from astroquery.vizier import Vizier
from astropy import units as u
from astropy import constants as const
from astropy.table import Table, Column, QTable
import astropy.io.ascii as ascii_ap
from astropy.io import fits
import matplotlib.pyplot as plt
import scipy.constants as cst
from scipy.interpolate import interp1d
from scipy.io.idl import readsav

import os.path
# from .config import *
from importlib import reload
# import spirou_exo.analysis as a

# from scipy.interpolate import interp1d

# from hapi import transmittanceSpectrum

#------------------------------
# --- Get the model spectrum of a star depending on it's temperature and logg ---
# --- teff=[1200,7000]K or spt [M6,T8.5] and logg=[2.5,5.5] ---


def star(teff=2800, logg=5.0, spt='', ref='', want=''):

    # --- Selecting de model parameters according to the spectral type
    if spt != '':
        if len(spt) == 2:
            spt += '.0'
        x, temp, sptype = hm.relations(want=want, ref=ref)

        try:
            ind = sptype.index(spt)
        except ValueError:
            print('The spectral type you entered is not available')
            spt = input('Enter a spectral type between ' + str(sptype[0]) +
                        ' and ' + str(sptype[len(sptype) - 1]) + ' : ')
            ind = sptype.index(spt)
        teff = temp[ind]

    # --- Reading the .fits
    pat = masterDir + 'Modeles/'
    modele = 'lte0' + str((round(teff / 50.) * 50.) / 100.) + '-' + str(logg) + \
             '-0.0a+0.0.BT-Settl.spec.7'
    print('Star model : {}'.format(modele))

    # --- If the model at 50K precision isn't available, Teff is changed at a 100K precision
    try:
        hdulist = fits.open(pat + modele + '.fits')
    except:
        print("The model : " + modele + " doesn't exist. The Teff " +
              str(round(teff / 50.) * 50.) + " is changed to " + str(round(teff, -2)))
        modele = 'lte0' + str(round(teff, -2) / 100.) + '-' + str(logg) + \
                 '-0.0a+0.0.BT-Settl.spec.7'

    try:
        hdulist = fits.open(pat + modele + '.fits')
    except:
        modele = 'lte0' + str(round(teff, -2) / 100.) + '-' + str(logg) + \
                 '-0.0a+0.0.BT-Settl.spec.7'
        hdulist = fits.open(pat + modele + '.fits')

    tbdata = hdulist[1].data

    params = {'Teff': teff, 'logg': logg}

    # --- To access each individual array:
#    wl = tbdata.field('wavelength')
#    flux = tbdata.field('flux')
#    return wl,flux

    # wl = um / flux =  J/s/micron/m²
    return tbdata.field('wavelength'), tbdata.field('flux'), params


#------------------------------
# --- Get the model spectrum of a star similar to HD208458 with limb-darkening
# --- Teff=6000K and logg=4.5, with Z=Z_0 ---


def star_ld():

    # --- Without the .sav file
    # file = '/Users/AnneBoucher/Documents/Doctorat/Python/t6000l45z0_ldnl.dat'
    # nbvar = 6
    # wl, c1, c2, c3, c4, flux = hm.readcol(file, nbvar)

    # wl = wl.astype(np.float)*10  # - To angstrom
    # flux = flux.astype(np.float)
    # c1, c2, c3, c4 = c1.astype(np.float), c2.astype(np.float), \
    #     c3.astype(np.float), c4.astype(np.float)

    # --- With the .sav file
    file = masterDir + 'starModels/t6000l45z0_ldnl.dat.sav'
    chose = readsav(file)
    wl = chose['wl'] * 10  # to Angstrom
    flux = chose['flux']
    c1, c2, c3, c4 = chose['c1'], chose['c2'], chose['c3'], chose['c4']

    flux = -flux * cst.pi * (42.0 * c1 + 70.0 * c2 + 90.0 * c3 + 105.0 * c4 - 210.0) / 210.0
    flux = 4.0 * flux * 2.99792458e18 / (wl * wl)  # - To ergs cm^{-2} s^{-1} A^{-1}
    flux = flux * 10  # - To convert from u.erg / u.s / u.cm**2 / u.Angstrom).to(u.J / u.s / u.m**2 / u.um)
    flux[flux < 0] = 0

    wl = wl * 1.e-4  # - To micron

    # plt.plot(wl, flux)
    # plt.xlim(1, 5)
    params = {'Teff': 6000, 'logg': 4.5}

    return np.squeeze(wl), np.squeeze(flux), params


#------------------------------
# --- Spectrum of a hot-Jupiter covering J,H, and K.
# --- Solar composition spectrum for HD 209458 b


# def planet(systeme, Z=1, T=1600, cloud = None, species=None, PRT=False):
    
#     pat = masterDir + 'planetModels/'
    
# #     if PRT is True:
        
# #         filename = hm.replace_with_check(systeme, [' '], '_') + '_PRT_T{}K'.format(T)
            
# # #         species = OrderedDict({'H2O_main_iso':[-3, -4, -5], 
# # #                        'CO_all_iso':[-3, -4, -5]})
# #         for mol in species.keys():
# #             filename += '_{}{:.2f}'.format(mol,species[mol]) 
# #         if cloud is not None:
# #             filename += '_cloud{:.1f}'.format(cloud)
            
# #         dppm = np.load(pat + hm.replace_with_check(systeme, [' '], '_') + '/' + filename + '.npy')
# #         wave = np.load(pat + hm.replace_with_check(systeme, [' '], '_') + '/' + \
# #                       hm.replace_with_check(systeme, [' '], '_') + '_Spectrum_PRT_wave.npy')
# # #         spec = Table.read(pat + filename + '.ecsv', format='ascii.ecsv', delimiter=',')
# # #         return spec['wave'], spec['dppm'], [],[],[]
# #         print(filename)
# #         return wave, dppm * 1e-6, [],[],[]
    
#     if np.logical_or(systeme == 'HD 209458', systeme == 'HD 209458 b'):
#         filename = '20160828_20h08m04s_HD_209458_b_Metallicity1_Spectrum.csv'
#         spec = Table.read(pat + filename, format='ascii.ecsv', delimiter=',')
#         return spec['Wavelength'], spec['Transit Depth']*1e-6, spec.meta['params']
#     elif systeme == 'TRAPPIST-1':
#         filename = 'TRAPPIST_1_d_WellMixed_H2O_1_Spectrum_FullRes_dppm.csv'
#         spec = Table.read(pat + filename, format='ascii.ecsv', delimiter=',')
#         return spec['wave'], spec['dppm']*1e-6, spec.meta
# #     elif systeme == 'GJ3470':
# #         filename = 'GJ_3470_b_Metallicity1.0_CtoO0.54_pCloud100000.0mbar_cHaze1e-10_pQuench1e-99_TpTeq_Spectrum_FullRes_dppm.csv'
# #         spec = Table.read(pat + filename, format='ascii.ecsv', delimiter=',')
# #         return spec['wave'], spec['dppm']*1e-6, spec.meta
#     else:
#         try:
#             filename = hm.replace_with_check(systeme, [' ', '-'], '_') + \
#             ('_Metallicity{:.1f}_pCloud{:.1f}mbar_Spectrum_FullRes.csv').format(Z, cloud)
# #             ("_Metallicity{:.1f}_CtoO0.54_pCloud{:.1f}mbar_cHaze1e-10_pQuench1e-99_TpTeq_Spectrum_FullRes.csv").format(Z, cloud)
# #             ('_Metallicity{:.1f}_pCloud{:.1f}mbar_Spectrum_FullRes.csv').format(Z, cloud)
#             spec = Table.read(pat + filename, format='ascii.ecsv', delimiter=',')
        
#         except:
#             try:
#                 filename = hm.replace_with_check(systeme, [' ', '-'], '_') + \
#              ("_Metallicity{:.1f}_CtoO0.54_pCloud{:.1f}mbar_cHaze1e-10_pQuench1e-99_TpTeq_Spectrum_FullRes.csv").format(Z, cloud)
#                 spec = Table.read(pat + filename, format='ascii.ecsv', delimiter=',')
            
#             except FileNotFoundError:
#                 print("Didn't find the file, adding a 'b' to the name ... ", end=' ')
#                 filename = hm.replace_with_check(systeme+' b', [' ', '-'], '_') +('_Metallicity{:.1f}_pCloud{:.1f}mbar_Spectrum_FullRes.csv').format(Z, cloud)
#                 spec = Table.read(pat + filename, format='ascii.ecsv', delimiter=',')
#                 print('but it worked!')
        
#         print(spec.meta)
#         print(filename)
        
#         try:
#             return spec['wave'][:-1], spec['dppm'][:-1]*1e-6, spec['dppm[H2O]'][:-1]*1e-6, spec['dppm[CO]'][:-1]*1e-6, spec['dppm[CH4]'][:-1]*1e-6
        
#         except KeyError:
#             return spec['wave'][:-1], spec['dppm'][:-1]*1e-6, [],[],[]

# #    fig,ax=plt.subplots()
# #    ax.plot(spec['Wavelength'],spec['Transit Depth'])
# #    ax.set_xlabel(r'Wavelength $[\mu m]$')
# #    ax.set_ylabel(r'Transit Depth [ppm]')

# #    wl=spec['Wavelength']
# #    depth=spec['Transit Depth']
# #    params=spec.meta['params']
# #    return wl,depth,params



#------------------------------
#------------- from hapi -----------------

def volumeConcentration(p,T):
    cBolts = const.k_B.cgs
    return (p/9.869233e-7)/(cBolts*T) # CGS

def transmittanceSpectrum(Omegas,AbsorptionCoefficient,Environment={'l':100.},
                          Wavenumber=None):
    """
    INPUT PARAMETERS: 
        Wavenumber/Omegas:   wavenumber grid                    (required)
        AbsorptionCoefficient:  absorption coefficient on grid  (required)
        Environment:  dictionary containing path length in cm.
                      Default={'l':100.}
    OUTPUT PARAMETERS: 
        Wavenum: wavenumber grid
        Xsect:  transmittance spectrum calculated on the grid
    ---
    DESCRIPTION:
        Calculate a transmittance spectrum (dimensionless) based
        on previously calculated absorption coefficient.
        Transmittance spectrum is calculated at an arbitrary
        optical path length 'l' (1 m by default)
    ---
    EXAMPLE OF USAGE:
        nu,trans = transmittanceSpectrum(nu,coef)
    ---
    """
    # compatibility with older versions
    if Wavenumber: Omegas=Wavenumber
    l = Environment['l']
    Xsect = np.exp(-(AbsorptionCoefficient*l).value)
    return Omegas,Xsect

#------------------------------


# def model_out(model_corr, star_name='HD 189733', 
#               Z=1, cloud=1e5, mol='', T=1600, p=1, 
#               trans_spec=True, PRT=False, species=None,
#               plot_mod=False):
    
#     if model_corr == 'model':
#         if star_name == 'HD 209458':
#             modelWave0, modelTD0, _ = planet(star_name, PRT=PRT)
#         else:
#             modelWave0, modelTD0, specH2O, specCO, specCH4 = planet(star_name, 
#                                                                     Z=Z, cloud=cloud, T=T,
#                                                                     PRT=PRT, species=species)
#             if mol == 'H2O':
#                 modelTD0 = specH2O
#             elif mol == 'CO':
#                 modelTD0 = specCO
#             elif mol == 'CH4':
#                 modelTD0 = specCH4
                
#     elif model_corr == 'exomol':
#         data= ascii_ap.read(masterDir +
#            'HITRAN/Xsec/{}/{}_3920-10526_{:d}K_0.010000.sigma'.format(model_corr, mol, int(T)))
#         modelWave0 = 1e4/data['col1'][::-1] # cm-1 to um
#         if trans_spec is True:
#             modelTD0 = transmittanceSpectrum(data['col1'], 
#                    data['col2'] * volumeConcentration(p,T))[1][::-1] # cm2/molec
#         else:
#             modelTD0 = data['col2'][::-1]

#     elif model_corr == 'hitemp': 
#         data = np.load(masterDir + 
#                        'HITRAN/Xsec/{}/{}_{:d}K_{:.1f}atm.npz'.format(model_corr, mol, int(T),p))
#         modelWave0 = data['wave'] # um
#         if trans_spec is True:
#             modelTD0 = transmittanceSpectrum(1e4/modelWave0[::-1], 
#                                              data['coef_cm'][::-1])[1][::-1] # cm-1
#         else:
#             modelTD0 = data['coef']

#     elif model_corr == 'hitran': 
#         data = np.load(masterDir + 
#                        'HITRAN/Xsec/{}/{}_{:d}.npz'.format(model_corr, mol, int(T)))
#         modelWave0 = data['wave'][::-1] # um
#         if trans_spec is True:
#             modelTD0 = data['trans'][::-1]
#         else:
#             modelTD0 = data['coef'][::-1]

#     elif model_corr == 'll_hitran': 
#         data = np.load(masterDir + 
#                        'HITRAN/line_list/{}_{:d}.npz'.format(mol, int(T)))
#         modelWave0 = 1e4/data['nu1'][::-1] # cm-1 to um
#         modelTD0 = data['sw1'][::-1]
    
#     elif model_corr == 'll_hitemp': 
#         data = np.load(masterDir + 
#                        'HITRAN/line_list/{}_HITEMP_{:d}.npz'.format(mol, int(T)))
#         if trans_spec is True:
#             modelWave0 = data['wave'][::-1] # um
#             modelTD0 = data['trans'][::-1]
#         else:
#             modelWave0 = 1e4/data['nu1'][::-1] # cm-1 to um
#             modelTD0 = data['sw1'][::-1]

#     elif model_corr == 'll_exomol':
#         data = np.load(masterDir +
#                        'HITRAN/line_list/{}_POKAZATEL_0900-2600nm_{:d}K.npz'.format(mol, int(T)))
#         modelWave0 = data['wave'][::-1] # um
#         modelTD0 = data['sw'][::-1]

#     if plot_mod is True:
#         plt.plot(modelWave0, modelTD0)
#         max_value = modelTD0.max() * 1.001
#         plt.plot([1.002, 1.115], [max_value, max_value])
#         plt.plot([1.16, 1.325], [max_value, max_value])
#         plt.plot([1.5, 1.7], [max_value, max_value])
#         plt.plot([2.1, 2.34], [max_value, max_value])

#     return np.array(modelWave0), np.array(modelTD0)

#------------------------------


def telluric(airmass=1.0, wv=1.6, tapas=False):
    """
    # --- Transmission spectrum above the Mauna Kea in the IR
    # --- airmass [1.0,1.5,2.0] and water vapor [1.0,1.6,3.0,5.0]mm
    """
    if tapas is False:
        pat = masterDir + 'Telluric/GeminiIR/'

        am = str(int(airmass * 10))
        wv = str(int(wv * 10))

        file = 'mktrans_zm_' + wv + '_' + am + '.dat'

    #    wl,trans=np.loadtxt(pat+file,unpack=True)
        wl, trans = hm.readcol(pat + file, 2)
        wl, trans = wl.astype(np.float), trans.astype(np.float)

        params = {'Airmass': airmass, 'WaterVapor': wv}
    else:
        file = masterDir + 'Telluric/model_tapas.sav'

        airmass = 2.116893
        wv = 2.903600

        chose = readsav(file)
        wl = chose['wl_nm'] / 1e3  # from nm to um
        trans = np.clip(chose['trans_tell'], 0, None)

        params = {'Airmass': airmass, 'WaterVapor': wv}

    return np.squeeze(wl), np.squeeze(trans), params


def loadTellTrans(date_id):
    date = ['16Nov2003', '1Jan2018']
    files = [masterDir + 'telluricModels/TAPAS/za0/' + date[date_id] + '/tapas_00000' + str(i) + '.ipac.sav' for i in range(1, 7)]

    # mol = ['H2O', 'O3', 'O2', 'CO2', 'CH4', 'N2O']
    transMol = np.empty((6, 4360408))

    for i in range(6):
        # data = ascii.read(files[i], format='ipac')
        # wl, trans = data['wavelength'], data['transmittance']
        sav = readsav(files[i])
        if i == 0:  # H2O
            wave = sav['wl'][::-1] / 1e3
        transMol[i, :] = np.clip(sav['trans'][::-1], 0, 1)

    transTot = np.prod(transMol, axis=0)
    return wave, transMol, transTot

#------------------------------
# --- Resampling function ---
# - Change the sampling of a spectrum from an initial resolution of Rbf
# - to a final resolution of Raf

def resampling(wl, flux, Raf, Rbf=None, lb_range=None, ptsPerElem=1, sample=None, rot_ker=None):#, plot=False):
    """
    Spectrum resampling, assuming a R that is constant
    Rbf is the sampling resolution, not the actual resolution.
    If Rbf is not given, we take the highest value in the array
    """
    # - On définit le range de longueur d'onde
    lb0 = np.min(wl)
    lb1 = np.max(wl)
    if lb_range is None:
        lb_range = [lb0, lb1]
    # - Si la résolution d'avant n'est pas donnée, on la calcule
    if Rbf is None:
        Rmax, Rmin, R_all = find_R(wl)
        if isRcst(wl):  # value 10 is arbitrary
            Rbf = R_all
            # -- We check if R is constant over the range, and if not we resample
    #         wl, flux = sample_new_R(wl, flux, Rbf, lb_range, sample=wl)
            # npt = wl.size
            # Rbf = np.round((1 / ptsPerElem) / ((lb1 / lb0)**(1 / npt) - 1))
            # print('R was {}'.format(Rbf))
        else:
            Rbf = Rmax
            # d_wl = np.diff(wl)
            # Rbf = np.round(np.max(wl[1:] / d_wl))
#             print('R was {}'.format(Rbf))
    # - Si on veut réduire la résolution :
    # - On définit la largeur de la gaussienne qu'on veut
    # We want to express the gaussian fwhm in units of the 
    # sampling of the input.
    R_sampling = Rbf
    # If the sampling is at R = cst, then it is at
    # delta_v = cst since R = light_speed/delta_v, so
    # delta_v_after / delta_v_sampling = R_sampling / R_after
    fwhm = R_sampling / Raf
    if isinstance(Raf, int) or isinstance(Raf, np.float64):
        if Raf < Rbf :
            if rot_ker is None:
#                 print('Resampling without rotation')
                # - On on va vouloir un vecteur de taille ~7xfwhm
                taille = hm.myround(fwhm * 7, base=2) + 1
                # - On génère notre gaussienne avec le nb de point (pxl) qu'on vient de calculer
                profil = hm.gauss(np.arange(taille), mean=(taille - 1) / 2, FWHM=fwhm)

            else:
#                 print('Resampling with rotation')
                # R_af is implicitly included in the rot_ker object
                try:
                    profil = rot_ker.resample(R_sampling, n_os=1000, pad=7)
                except KernelIndexError:
                    # - On on va vouloir un vecteur de taille ~7xfwhm
                    taille = hm.myround(fwhm * 7, base=2) + 1
                    # - On génère notre gaussienne avec le nb de point (pxl) qu'on vient de calculer
                    profil = hm.gauss(np.arange(taille), mean=(taille - 1) / 2, FWHM=fwhm)
#             print(flux)
#             print(profil)
            # - On convolue notre ancien spectre avec le nouveau profil
            new_flux = np.convolve(flux, profil, mode='same')
            # - Notre nouveau spectre a la bonne résolution, mais pas le bon échantillonnage
            # - On va donc ajuster l'échantillonage :
            new_lb, new_flux = sample_new_R(wl, new_flux, Raf, lb_range,
                                            ptsPerElem=ptsPerElem, sample=sample)
        elif Raf > Rbf:
#             print('Raf > Rbf')
            new_lb, new_flux = sample_new_R(wl, flux, Raf, lb_range,
                                            ptsPerElem=ptsPerElem, sample=sample)
    elif (Raf.any() < Rbf).any():
#         print('Raf = continuous array < Rbf')
        new_flux = gaussConv(flux, fwhm)
        # - Notre nouveau spectre a la bonne résolution, mais pas le bon échantillonnage
        # - On va donc ajuster l'échantillonage :
        new_lb, new_flux = sample_new_R(wl, new_flux, Raf, lb_range,
                                        ptsPerElem=ptsPerElem, sample=sample)
    return new_lb, new_flux


def _get_rot_ker_tr_v(v_grid, omega, r_p, z_h):
        x_v = v_grid / omega
        out = np.zeros_like(x_v)
        idx = np.abs(x_v) < r_p
        arg1 = (r_p + z_h)**2 - (x_v)**2
        arg2 = r_p**2 - (x_v)**2
        out[idx] = (np.sqrt(arg1[idx]) - np.sqrt(arg2[idx])) / z_h
        idx = (np.abs(x_v) >= r_p) & (np.abs(x_v) <= (r_p + z_h))
        arg1 = (r_p + z_h)**2 - (x_v)**2
        out[idx] = np.sqrt(arg1[idx]) / z_h
        return out
def rot_ker_transit_v(pl_rad, pl_mass, t_eq, omega, resolution, mu=None, n_os=1, pad=5, iplot=False):
    '''Rotational kernel for transit at dv constant (constant resolution)
    pl_rad: scalar astropy quantity, planet radius
    pl_mass: scalar astropy quantity, planet mass
    t_eq: scalar astropy quantity, planet equilibrium temperature
    omega: array-like astropy quantity, 1 or 2 elements
        Rotation frequency. If 2 elements, different for
        each hemisphere. The first element will be for
        the negative speed (blue-shifted), and the 
        second for positive speed (red-shifted)
    resolution: scalar (float or int)
        spectral resolution
    mu: scalar astropy quantity, mean molecular mass
    n_os: scalar, oversampling (to sample the kernel)
    pad: scalar
        pad around the kernel in units of resolution
        elements. Values of the pad are set to zero.
    iplot: bool, plot or not
    '''
    if mu is None:
        mu = 2 * u.u
    g_surf = const.G * pl_mass / pl_rad**2
    scale_height = const.k_B * t_eq / (mu * g_surf)
    z_h = scale_height.to('m').value
    r_p = pl_rad.to('m').value
    res_elem = const.c / resolution
    res_elem = res_elem.to('m/s').value
    omega = omega.to('1/s').value
    ker_h_len = (r_p + z_h) * omega.max()
    v_max = ker_h_len + pad * res_elem
    delta_v = res_elem / n_os
    v_grid = np.arange(-v_max, v_max, delta_v)
    v_grid -= np.mean(v_grid)
    if omega.size == 2:
        kernel = np.zeros_like(v_grid)
        idx_minus = (v_grid < 0)
        # Negative v
        args = (v_grid[idx_minus], omega[0], r_p, z_h)
        kernel[idx_minus] = _get_rot_ker_tr_v(*args)
        # Positive (remaining index)
        args = (v_grid[~idx_minus], omega[1], r_p, z_h)
        kernel[~idx_minus] = _get_rot_ker_tr_v(*args)
    else:
        kernel = _get_rot_ker_tr_v(v_grid, omega, r_p, z_h)
    if iplot:
        plt.figure()
        plt.plot(v_grid/1e3, kernel)
        plt.axvline(res_elem/2e3, linestyle=':', label='Resolution element')
        plt.axvline(-res_elem/2e3, linestyle=':')
        plt.legend()
        plt.xlabel('dv [km/s]')
        plt.ylabel('Kernel')
    return kernel


class RotKerTransit:
    def __init__(self, pl_rad, pl_mass, t_eq, omega, resolution, mu=None):
        '''Rotational kernel for transit at dv constant (constant resolution)
        pl_rad: scalar astropy quantity, planet radius
        pl_mass: scalar astropy quantity, planet mass
        t_eq: scalar astropy quantity, planet equilibrium temperature
        omega: array-like astropy quantity, 1 or 2 elements
            Rotation frequency. If 2 elements, different for
            each hemisphere. The first element will be for
            the negative speed (blue-shifted), and the 
            second for positive speed (red-shifted)
        resolution: scalar (float or int)
            spectral resolution
        mu: scalar astropy quantity, mean molecular mass
        '''
        if mu is None:
            mu = 2.3 * u.u
        g_surf = const.G * pl_mass / pl_rad**2
        scale_height = const.k_B * t_eq / (mu * g_surf)
        z_h = 5 * scale_height.to('m').value
        r_p = pl_rad.to('m').value
        res_elem = const.c / resolution
        res_elem = res_elem.to('m/s').value
        omega = omega.to('1/s').value
        self.res_elem = res_elem
        self.omega = omega
        self.z_h = z_h
        self.r_p = r_p
    def get_ker(self, n_os=None, pad=7, norm=True):
        '''
        n_os: scalar, oversampling (to sample the kernel)
        pad: scalar
            pad around the kernel in units of resolution
            elements. Values of the pad are set to zero.
        '''
        res_elem = self.res_elem
        omega = self.omega
        z_h = self.z_h
        r_p = self.r_p 
        ker_h_len = (r_p + z_h) * omega.max()
        v_max = ker_h_len + pad * res_elem
        if n_os is None:
            # Find adequate sampling
            delta_v = np.abs(z_h * omega.min() / 100)
        else:
            delta_v = res_elem / n_os
        v_grid = np.arange(-v_max, v_max, delta_v)
        v_grid -= np.mean(v_grid)
        if omega.size == 2:
            kernel = np.zeros_like(v_grid)
            idx_minus = (v_grid < 0)
            # Negative v
            args = (v_grid[idx_minus], omega[0], r_p, z_h)
            kernel[idx_minus] = _get_rot_ker_tr_v(*args)
            # Positive (remaining index)
            args = (v_grid[~idx_minus], omega[1], r_p, z_h)
            kernel[~idx_minus] = _get_rot_ker_tr_v(*args)
        else:
            kernel = _get_rot_ker_tr_v(v_grid, omega, r_p, z_h)
        if norm:
            kernel /= kernel.sum()
        return v_grid, kernel
    
    def degrade_ker(self, rot_ker=None, v_grid=None, norm=True, **kwargs):
        '''kwargs are passed to get_ker method'''
        fwhm = self.res_elem
        if rot_ker is None: 
            v_grid, rot_ker = self.get_ker(norm=norm, **kwargs)
        gauss_ker = hm.gauss(v_grid, 0.0, FWHM=fwhm)
        out_ker = np.convolve(rot_ker, gauss_ker, mode='same')
        if norm:
            out_ker /= out_ker.sum()
        return v_grid, out_ker
    
    def resample(self, res_sampling, **kwargs):
        '''
        res_sampling: resolution of the sampling needed
        kwargs are passed to degrade_ker method
        '''
        dv_new = const.c / res_sampling
        dv_new = dv_new.to('m/s').value
        v_grid, kernel = self.degrade_ker(**kwargs)
        ker_spl = interp1d(v_grid, kernel, kind='linear')
        v_grid = np.arange(v_grid.min(), v_grid.max(), dv_new)
        out_ker = ker_spl(v_grid)
        return out_ker
    
    def show(self, norm=True, **kwargs):
        res_elem = self.res_elem
        v_grid, kernel = self.get_ker(norm=norm, **kwargs)
        gauss_ker = hm.gauss(v_grid, 0.0, FWHM=res_elem)
        if norm:
            gauss_ker /= gauss_ker.sum()
        _, ker_degraded = self.degrade_ker(norm=norm, **kwargs)
        fig = plt.figure()
        plt.plot(v_grid/1e3, gauss_ker, "--", color="gray",
                 label='Instrumental resolution element')
        plt.axvline(res_elem/2e3, linestyle='--', color='gray')
        plt.axvline(-res_elem/2e3, linestyle='--', color='gray')
        plt.plot(v_grid/1e3, kernel,
                 label='Rotation kernel')
        plt.plot(v_grid/1e3, ker_degraded,
                 label='Instrumental * rotation')
        plt.legend()
        plt.xlabel('dv [km/s]')
        plt.ylabel('Kernel')
        
        
from astropy.convolution import Gaussian1DKernel
def box_smoothed_step(v_grid, left_val, right_val, box_width, v_mid=0):
    # box width in units of dv 
    dv_grid = v_grid[1] - v_grid[0]
    box_width = box_width / dv_grid
    # Gaussian smoothing kernel
    g_ker = Gaussian1DKernel(box_width).array
    # Apply step funnction
    y_val = np.zeros(v_grid.shape)
    y_val[v_grid < v_mid] = left_val
    y_val[v_grid >= v_mid] = right_val
    # Pad with connstants at the boundaries
    pad_size = np.ceil(len(g_ker) / 2).astype(int) - 1
    pad_left = np.repeat(y_val[0], pad_size)
    pad_right = np.repeat(y_val[-1], pad_size)
    y_val = np.concatenate([pad_left, y_val, pad_right])
    # Smooth the step function
    y_smooth = np.convolve(y_val, g_ker, mode='valid')
    # Normalize so that if the function was equal
    # to 1 everywhere
    y_ones = np.convolve(np.ones_like(y_val), g_ker, mode='valid')
    norm = y_ones.sum()
    y_smooth = y_smooth / norm * len(y_smooth)
    return y_smooth


def lorentz(x, x0=0, gamma=1., amp=1.):
    return amp * gamma**2 / ( (x-x0)**2 + gamma**2 )

def v_de_phi(phi, gamma1, gamma2, amp1, amp2):
    
    output = np.ones_like(phi) * np.nan
    
    # --- Entre -pi/2 et pi/2 ---
    
    region = np.where((phi > -np.pi/2) & (phi <= np.pi/2))
    
    lorentz1 = lorentz(phi[region], x0=np.pi/2, gamma=gamma1, amp=amp1)
    lorentz2 = lorentz(phi[region], x0=-np.pi/2, gamma=gamma2, amp=amp2)
    
    output[region] = lorentz1+lorentz2
    
    # --- Entre -pi et -pi/2 ---
    
    region = np.where((phi >= -np.pi) & (phi <= -np.pi/2))
    
    lorentz1 = lorentz(phi[region], x0=-3*np.pi/2, gamma=gamma1, amp=amp1)
    lorentz2 = lorentz(phi[region], x0=-np.pi/2, gamma=gamma2, amp=amp2)
    
    output[region] = lorentz1+lorentz2
    
    # --- Entre pi/2 et pi ---
    
    region = np.where((phi > np.pi/2) & (phi <= np.pi))
    
    lorentz1 = lorentz(phi[region], x0=np.pi/2, gamma=gamma1, amp=amp1)
    lorentz2 = lorentz(phi[region], x0=3*np.pi/2, gamma=gamma2, amp=amp2)
    
    output[region] = lorentz1+lorentz2

    return output




class KernelIndexError(IndexError):
    pass


class RotKerTransitCloudy:
    def __init__(self, pl_rad, pl_mass, t_eq, omega, resolution,
                 left_val=1, right_val=1, step_smooth=1, v_mid=0, mu=None, 
                 gauss=False, x0=0., fwhm=4000.,  
                 gamma1=2/np.pi, gamma2=2/np.pi, amp1=5, amp2=5, vphi=False):
        '''Rotational kernel for transit at dv constant (constant resolution)
        pl_rad: scalar astropy quantity, planet radius
        pl_mass: scalar astropy quantity, planet mass
        t_eq: scalar astropy quantity, planet equilibrium temperature
        omega: array-like astropy quantity, 1 or 2 elements
            Rotation frequency. If 2 elements, different for
            each hemisphere. The first element will be for
            the negative speed (blue-shifted), and the 
            second for positive speed (red-shifted)
        resolution: scalar (float or int)
            spectral resolution
        left_val: float
            Transmission value at the bluer part of the kernel.
            Between 0 and 1. Default is 1 (no clouds)
        right_val: float
            Transmission value at the redest part of the kernel.
            Between 0 and 1. Default is 1 (no clouds)
        step_smooth: float
            fwhm of the gaussian kernel to smooth the clouds
            transmission step function. Default is 1.
        v_mid: float
            velocity where the step occurs in the clouds
            transmission step function. Default is 0.
        mu: scalar astropy quantity, mean molecular mass
        '''
        if mu is None:
            mu = 2.3 * u.u
        g_surf = const.G * pl_mass / pl_rad**2
        scale_height = const.k_B * t_eq / (mu * g_surf)
        z_h = 5 * scale_height.to('m').value
        r_p = pl_rad.to('m').value
        res_elem = const.c / resolution
        res_elem = res_elem.to('m/s').value
        omega = omega.to('1/s').value
        self.res_elem = res_elem
        self.omega = omega
        self.z_h = z_h
        self.r_p = r_p
        self.left_val = left_val
        self.right_val = right_val
        self.step_smooth = step_smooth
        self.v_mid = v_mid
        
        self.gauss = gauss
        self.x0 = x0
        self.fwhm = fwhm
        
        self.vphi = vphi
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.amp1 = amp1
        self.amp2 = amp2
        
    def get_ker(self, n_os=None, pad=7, v_grid=None):
        '''
        n_os: scalar, oversampling (to sample the kernel)
        pad: scalar
            pad around the kernel in units of resolution
            elements. Values of the pad are set to zero.
        '''
        res_elem = self.res_elem
        omega = self.omega
        z_h = self.z_h
        r_p = self.r_p 
        clouds_args = (self.left_val,
                       self.right_val,
                       self.step_smooth,
                       self.v_mid)
        ker_h_len = (r_p + z_h) * omega.max()
        v_max = ker_h_len + pad * res_elem
        if n_os is None:
            # Find adequate sampling
            delta_v = np.abs(z_h * omega.min() / 100)
        else:
            delta_v = res_elem / n_os
        if v_grid is None:
            v_grid = np.arange(-v_max, v_max, delta_v)
            v_grid -= np.mean(v_grid)
        if omega.size == 2:
            kernel = np.zeros_like(v_grid)
            idx_minus = (v_grid < 0)
            # Negative v
            args = (v_grid[idx_minus], omega[0], r_p, z_h)
            kernel[idx_minus] = _get_rot_ker_tr_v(*args)
            # Positive (remaining index)
            args = (v_grid[~idx_minus], omega[1], r_p, z_h)
            kernel[~idx_minus] = _get_rot_ker_tr_v(*args)
        else:
            kernel = _get_rot_ker_tr_v(v_grid, omega, r_p, z_h)
        # normalize
        kernel /= kernel.sum()
        # Get cloud transmission function
        idx_valid = (kernel > 0)
        if idx_valid.sum() <= 1:
            raise KernelIndexError("Kernel size too small for grid.")
        clouds = np.ones_like(kernel) * np.nan
        clouds[idx_valid] = box_smoothed_step(v_grid[idx_valid], *clouds_args)
        kernel[idx_valid] = kernel[idx_valid] * clouds[idx_valid]
        return v_grid, kernel, clouds
    
    def get_ker_vphi(self, n_os=1000, pad=7):
        '''
        n_os: scalar, oversampling (to sample the kernel)
        pad: scalar
            pad around the kernel in units of resolution
            elements. Values of the pad are set to zero.
        gamma1 and gamma2 : spread of the two peaks
        amp1 and amp2 : maximal values of the two peaks (in km/s)
        (usually 1 is negative and 2 is positive)
        '''
        res_elem = self.res_elem#*(u.m/u.s).to(u.km/u.s)
#         print(res_elem)
        phi_args = (self.gamma1,
                    self.gamma2,
                    self.amp1*1e3,
                    self.amp2*1e3)

        phi = np.linspace(-np.pi, np.pi, 10000)
        v_tot = v_de_phi(phi, *phi_args)
        self.v_de_phi = v_tot
        self.phi = phi
        

        v_max = np.max(np.abs([self.amp1*1e3, self.amp2*1e3])) + pad*res_elem  #.value
        delta_v = res_elem / (n_os/20)

        v_grid = np.arange(-v_max, v_max, delta_v)
        v_grid -= np.mean(v_grid)

        diff = np.diff(v_grid)

        edges = list(v_grid[:-1]-0.5*diff)
        edges.append(v_grid[-1]-0.5*diff[-1])
        edges.append(v_grid[-1]+0.5*diff[-1])
#         edges = np.array(edges)

        kernel, _ = np.histogram(v_tot, bins = edges)

#         kernel, bins = np.histogram(v_tot, bins=int(n_os/2), range=(-v_max, v_max)) #np.array(edges))
#         v_grid = 0.5*(bins[1:] + bins[:-1])        
        kernel = np.array(kernel)
#         ker_sum = kernel.sum()
        kernel = kernel/(kernel.sum())


#         v_max = np.max(np.abs([self.amp1, self.amp2])) + pad*res_elem*(u.m/u.s).to(u.km/u.s)#.value
#         kernel, bins = np.histogram(v_tot, bins=int(n_os/2), range=[-v_max, v_max])
#         v_grid = 0.5*(bins[1:] + bins[:-1])

#         kernel = np.array(kernel)
#         ker_sum = kernel.sum()

#         kernel = kernel/ker_sum

        return v_grid, kernel

    def get_ker_gauss(self, n_os=None, pad=7):
        
        res_elem = self.res_elem
        omega = self.omega
        z_h = self.z_h
        r_p = self.r_p 
        print(res_elem,omega,z_h,r_p)

        ker_h_len = (r_p + z_h) * omega.max()
        v_max = ker_h_len + pad * res_elem
        if n_os is None:
            # Find adequate sampling
            delta_v = np.abs(z_h * omega.min() / 100)
        else:
            delta_v = res_elem / n_os
#         print(delta_v, v_max)
        v_grid = np.arange(-v_max, v_max, delta_v)
        v_grid -= np.mean(v_grid)
        
        x0 = self.x0
        fwhm = self.fwhm
        
        kernel = hm.gauss(v_grid, x0, FWHM=fwhm)
        # normalize
        kernel /= kernel.sum()
        # Get cloud transmission function
        idx_valid = (kernel > 0)
        if idx_valid.sum() <= 1:
            raise KernelIndexError("Kernel size too small for grid.")

        return v_grid, kernel

    
#     def v_de_phi_kernel(gamma1, gamma2, amp1, amp2, plot=False, n_os=1000, pad=7, R=64000):

#         res_elem = const.c / R
#         phi = np.linspace(-np.pi, np.pi, 10000)

#         v_tot = v_de_phi(phi, gamma1, gamma2, amp1, amp2)

#         delta_v = res_elem / (n_os/10)

#         v_max = np.max(np.abs([amp1, amp2])) + pad*res_elem.to(u.km/u.s).value
#     #     print(v_max)

#         v_grid = np.arange(-v_max, v_max, delta_v.to(u.km/u.s).value)
#         v_grid -= np.mean(v_grid)
#     #     print(v_grid[[0,-1]])
#         diff = np.diff(v_grid)
#     #     print(delta_v, diff[0])
#         edges = list(v_grid[:-1]-0.5*diff)
#         edges.append(v_grid[-1]-0.5*diff[-1])
#         edges.append(v_grid[-1]+0.5*diff[-1])
#         edges = np.array(edges)

#         kernel, bins = np.histogram(v_tot, bins=edges,)# range=[-v_max, v_max])  #bins=int(n_os/2)
#     #     v_grid = 0.5*(bins[1:] + bins[:-1])

#         kernel=np.array(kernel)
#         ker_sum = kernel.sum()
#         kernel = kernel/ker_sum

#         if plot is True:
#             plt.figure()
#             plt.plot(phi, v_tot)
#         if plot is True:
#                 plt.figure()
#                 plt.plot(v_grid, kernel)

#         return v_grid, kernel
        
    
    def degrade_ker(self, **kwargs): #rot_ker=None, v_grid=None,
        '''kwargs are passed to get_ker method'''
        fwhm = self.res_elem
#         if rot_ker is None:
        if self.vphi is True:
            v_grid, rot_ker = self.get_ker_vphi(**kwargs)
        elif self.gauss is True:
            v_grid, rot_ker = self.get_ker_gauss(**kwargs)
        else:
            v_grid, rot_ker, _ = self.get_ker(**kwargs)
                
        norm = rot_ker.sum()
        gauss_ker = hm.gauss(v_grid, 0.0, FWHM=fwhm)
        out_ker = np.convolve(rot_ker, gauss_ker, mode='same')
        out_ker /= out_ker.sum()
        out_ker *= norm
        return v_grid, out_ker
    
    def resample(self, res_sampling, **kwargs):
        '''
        res_sampling: resolution of the sampling needed
        kwargs are passed to degrade_ker method
        '''
        dv_new = const.c / res_sampling
        dv_new = dv_new.to('m/s').value
        v_grid, kernel = self.degrade_ker(**kwargs)
        norm = kernel.sum()
        ker_spl = interp1d(v_grid, kernel, kind='linear')
        v_grid = np.arange(v_grid.min(), v_grid.max(), dv_new)
        out_ker = ker_spl(v_grid)
        out_ker /= out_ker.sum()
        out_ker *= norm
        return out_ker
    
    def show(self, **kwargs):
        res_elem = self.res_elem
        if self.vphi is False:
            v_grid, kernel, clouds = self.get_ker(**kwargs)
        else:
            v_grid, kernel = self.get_ker_vphi(**kwargs)
        gauss_ker = hm.gauss(v_grid, 0.0, FWHM=res_elem)
        gauss_ker /= gauss_ker.sum()
        _, ker_degraded = self.degrade_ker(**kwargs)
        fig = plt.figure()
        lines = plt.plot(v_grid/1e3, gauss_ker, color="gray",
                            label='Instrum. res. elem.')
        plt.axvline(res_elem/2e3, linestyle='--', color='gray')
        plt.axvline(-res_elem/2e3, linestyle='--', color='gray')
        new_line = plt.plot(v_grid/1e3, kernel, label='Rotation kernel')
        lines += new_line
        new_line = plt.plot(v_grid/1e3, ker_degraded, 
                            label='Instrum. * rotation')
        lines += new_line
        twin_ax = plt.gca().twinx()
        if self.vphi is False:
            new_line = twin_ax.plot(v_grid/1e3, clouds, label='Transmission clouds',
                                color='g', alpha=0.5, linestyle=':')
        lines += new_line
        labs = [line.get_label() for line in lines]
        plt.legend(lines, labs)
        plt.xlabel('dv [km/s]')
        plt.ylabel('Kernel')
        twin_ax.set_ylim(-0.05, 1.05)


# def resampling(wl, flux, Raf, Rbf=None, lb_range=None, ptsPerElem=1, sample=None):
#     """
#     Spectrum resampling, assuming a R that is constant
#     If Rbf is not given, we take the highest value in the array
#     """

#     # - On définit le range de longueur d'onde
#     lb0 = np.min(wl)
#     lb1 = np.max(wl)

#     if lb_range is None:
#         lb_range = [lb0, lb1]

#     # - Si la résolution d'avant n'est pas donnée, on la calcule
#     if Rbf is None:
#         Rmax, Rmin, R_all = find_R(wl)
#         if isRcst(wl):  # value 10 is arbitrary
#             Rbf = R_all
#             # -- We check if R is constant over the range, and if not we resample
#     #         wl, flux = sample_new_R(wl, flux, Rbf, lb_range, sample=wl)
#             # npt = wl.size
#             # Rbf = np.round((1 / ptsPerElem) / ((lb1 / lb0)**(1 / npt) - 1))
#             # print('R was {}'.format(Rbf))
#         else:
#             Rbf = Rmax
#             # d_wl = np.diff(wl)
#             # Rbf = np.round(np.max(wl[1:] / d_wl))
# #             print('R was {}'.format(Rbf))

#     # - Si on veut réduire la résolution :
#     # - On définit la largeur de la gaussienne qu'on veut
#     larg_gauss = Rbf / Raf  # ptsPerElem
#     if isinstance(Raf, int) or isinstance(Raf, np.float64):
#         if Raf < Rbf :
#             # - On on va vouloir un vecteur de taille ~7xlarg_gauss
#             taille = hm.myround(larg_gauss * 7, base=2) + 1
#             # - On génère notre gaussienne avec le nb de point (pxl) qu'on vient de calculer
#             profil = hm.gauss(np.arange(taille), mean=(taille - 1) / 2, FWHM=larg_gauss)
            
            
# #             ### convolve avec 
# #             np.convolve(profil,kernel, mode='same')

            
#             # - On convolue notre ancien spectre avec le nouveau profil
#             new_flux = np.convolve(flux, profil, mode='same')
#             # - Notre nouveau spectre a la bonne résolution, mais pas le bon échantillonnage
#             # - On va donc ajuster l'échantillonage :
#             new_lb, new_flux = sample_new_R(wl, new_flux, Raf, lb_range,
#                                             ptsPerElem=ptsPerElem, sample=sample)
#         elif Raf > Rbf:
#             new_lb, new_flux = sample_new_R(wl, flux, Raf, lb_range,
#                                             ptsPerElem=ptsPerElem, sample=sample)
#     elif (Raf.any() < Rbf).any():
#         new_flux = gaussConv(flux, larg_gauss)
#         # - Notre nouveau spectre a la bonne résolution, mais pas le bon échantillonnage
#         # - On va donc ajuster l'échantillonage :
#         new_lb, new_flux = sample_new_R(wl, new_flux, Raf, lb_range,
#                                         ptsPerElem=ptsPerElem, sample=sample)

#     return new_lb, new_flux

# --- older ---
# def resampling(wl, flux, Raf, Rbf=None, lb_range=None, ptsPerElem=1, sample=None):
#     """
#     Spectrum resampling, assuming a R that is constant
#     If Rbf is not given, we take the highest value in the array
#     """

#     # - On définit le range de longueur d'onde
#     lb0 = np.min(wl)
#     lb1 = np.max(wl)

#     if lb_range is None:
#         lb_range = [lb0, lb1]

#     # - Si la résolution d'avant n'est pas donnée, on la calcule
#     if Rbf is None:
#         Rmax, Rmin, R_all = find_R(wl)
#         if isRcst(wl):  # value 10 is arbitrary
#             Rbf = R_all
#             # -- We check if R is constant over the range, and if not we resample
#     #         wl, flux = sample_new_R(wl, flux, Rbf, lb_range, sample=wl)
#             # npt = wl.size
#             # Rbf = np.round((1 / ptsPerElem) / ((lb1 / lb0)**(1 / npt) - 1))
#             # print('R was {}'.format(Rbf))
#         else:
#             Rbf = Rmax
#             # d_wl = np.diff(wl)
#             # Rbf = np.round(np.max(wl[1:] / d_wl))
# #             print('R was {}'.format(Rbf))

#         # - Si on veut réduire la résolution :
#     if (Raf.any() < Rbf).any():
#         # - On définit la largeur de la gaussienne qu'on veut
#         larg_gauss = Rbf / Raf  # ptsPerElem
#         if isinstance(Raf, int) or isinstance(Raf, np.float64):
#             # - On on va vouloir un vecteur de taille ~7xlarg_gauss
#             taille = hm.myround(larg_gauss * 7, base=2) + 1
#             # - On génère notre gaussienne avec le nb de point (pxl) qu'on vient de calculer
#             profil = hm.gauss(np.arange(taille), mean=(taille - 1) / 2, FWHM=larg_gauss)
#             # - On convolue notre ancien spectre avec le nouveau profil
#             new_flux = np.convolve(flux, profil, mode='same')
#         else:
#             new_flux = gaussConv(flux, larg_gauss)
            

#         # - Notre nouveau spectre a la bonne résolution, mais pas le bon échantillonnage
#         # - On va donc ajuster l'échantillonage :
#         new_lb, new_flux = sample_new_R(wl, new_flux, Raf, lb_range,
#                                         ptsPerElem=ptsPerElem, sample=sample)

#     # - si on veut oversampler :
#     else:
#         new_lb, new_flux = sample_new_R(wl, flux, Raf, lb_range,
#                                         ptsPerElem=ptsPerElem, sample=sample)

#         # fct = interp1d(wl, flux, fill_value='extrapolate')
#         # new_npt = np.floor(np.log(lb_range[1] / lb_range[0]) / np.log(1 / (ptsPerElem * Raf) + 1))
#         # new_lb = np.exp(np.arange(new_npt) / (new_npt - 1) * np.log(lb_range[1] / lb_range[0])) * lb_range[0]  # lb0*(1/(2*Raf)+1)**(np.arange(new_npt))

#         # new_flux = fct(new_lb)

#     return new_lb, new_flux


def find_R(wl, ptsPerElem=1):
    """
    Compute the "sampling resolution" power of the given spectrum (for a given sampling)
    It will take the highest value it can find, if R is not constant (if constant delta_wl)
    """
    # if ptsPerElem is not None:
    #     R = np.round((1 / ptsPerElem) / ((np.max(wl) / np.min(wl))**(1 / wl.size) - 1))
    # else:
    d_wl = np.diff(wl)
    R_all = np.round(wl[1:] / d_wl / ptsPerElem)

    # print('R is {}'.format(R))

    return R_all.max(), R_all.min(), np.append(R_all, R_all[-1]-1)


from spirou_exo.utils.mask_tools import interp1d_masked


def sample_new_R(wl, flux, R, lb_range, ptsPerElem=1, sample=None):
    # - On génère une "fonction" à partir de notre spectre pour interpoler
    
    if ~isinstance(flux, np.ma.core.MaskedArray):
        flux = np.ma.array(flux, mask=~np.isfinite(flux))
    
    fct = interp1d_masked(wl, flux, kind='cubic', fill_value='extrapolate')
    
    if sample is None:
        # - On calcule le nombre de point nécessaire pour avoir le bon échantillonnage
        # - Chaque élément = 1 pixel (point) spectral = 20 pixels physiques
        new_npt = np.floor(np.log(lb_range[1] / lb_range[0]) / np.log(1 / (ptsPerElem * R) + 1))

        # - On génère un nouveau vecteur de longueur d'onde avec le bon nb de point
        new_lb = np.exp(np.arange(new_npt) / (new_npt - 1) * np.log(lb_range[1] / lb_range[0])) * lb_range[0]  # lb0*(1/(2*Raf)+1)**(np.arange(new_npt))
        # - On calcul de nouveau spectre avec notre interpolation
        # new_flux = fct(new_lb)
    else:
        # If thr user wants to use its own sampling grid, but for a spectrum at the desired R
        new_lb = sample
    
#     print(new_lb[0], new_lb[-1])
#     plt.figure()
#     plt.plot(wl, flux)
#     plt.plot(new_lb, fct(new_lb), 'r')
# #     plt.plot(new_lb, new_flux2, 'g')

#     hm.stop()
        
    return new_lb, fct(new_lb)


def isRcst(wl):
    """
    Determines if R is constant troughout the whole array
    """
    return np.abs(np.diff([find_R(wl)[0], find_R(wl)[1]])) > 10


def gaussConv(x,fwhm):
    #convolution with a Gaussian kernel with fwhm specified for each pixel
    #fwhm is an array with as many elements as x
    sig=fwhm/2.35
    sig2=sig**2
    deltaMAX=np.ceil(3*fwhm.max()).astype(int)

    xx=np.copy(x)
    delta=1
    while delta<=deltaMAX:
        xx[deltaMAX:x.size-deltaMAX]+=np.exp((-0.5*delta**2)/sig2[deltaMAX:x.size-deltaMAX])*\
            (x[deltaMAX+delta:x.size-deltaMAX+delta]+x[deltaMAX-delta:x.size-deltaMAX-delta])
        delta+=1

    xx*=1./(2*np.pi)**(0.5)/sig
    xx[:deltaMAX]=np.nan
    xx[-deltaMAX:]=np.nan

    return xx


def gen_mod(wave, flux, samp_wave, RV=None, Rbf=None):
    """
    Transorm a 1d-spectrum to a 2d-spectrum on the given wavelength grid
    wave : wavelength array of the 1d-spectrum
    flux : flux of the 1d-spectrum (can be single or multiple epochs)
    samp_wave : wavelength 2d-array grid to be sampled on
    RV : (optional) if the flux needs to be shifted for n-epochs
    Rbf : Initial resolution of the spectrum
    """
    
    try:
        n_spec, n_ord, npix = samp_wave.shape
    except ValueError:
        samp_wave = np.expand_dims(samp_wave, axis=0)
        n_spec, n_ord, npix = samp_wave.shape
        

    if flux.ndim > 1:
        if RV is not None:
            flux_shift = hm.doppler_shift2(wave, flux, RV)
            n_spec = RV.size
        else:
            flux_shift = flux.copy()
    else:
        flux_shift = np.expand_dims(flux.copy(), axis=0)  
        flux_shift = np.tile(flux_shift, (n_spec,1))

    # wave_ord = np.empty((n_spec, n_ord, 4088))
    flux_ord = np.empty((n_spec, n_ord, npix))

    for n in range(n_spec):
        for iOrd in range(n_ord):
            hm.print_static(iOrd, '--', n)
#             print(samp_wave[n, iOrd][0]-0.01,samp_wave[n, iOrd][-1]+0.01)
            i0, i1 = np.searchsorted(wave, [samp_wave[n, iOrd][0]-0.01, \
                                            samp_wave[n, iOrd][-1]+0.01])
            x = wave[i0:i1]
            y = flux_shift[n, i0:i1]

            R_more = get_var_res(iOrd, x)

            _, flux_ord[n,iOrd] = resampling(x, y, R_more,                            
                                              # get_res(iOrd),
                                              Rbf=Rbf,
                                              sample=samp_wave[n, iOrd])
#             hm.stop()
#             flux_ord[n,iOrd] = norm4correl_1d(flux_ord[n,iOrd])
#     print('done')
    return flux_ord.squeeze()  #wave_ord.squeeze(), 


#------------------------------
# --- Compute the synthetic flux of a given spectrum for a given filter (2MASS or WISE) ---


def flux_synt(wl, flux, cat, filtre):
    if cat == '2mass':
        file = masterDir + '/data/transmission_2mass.sav'
        sav = readsav(file)
        if filtre == 'j':
            wl_trans = sav.ltransj
            trans_fct = sav.ttransj
        elif filtre == 'h':
            wl_trans = sav.ltransh
            trans_fct = sav.ttransh
        elif filtre == 'k':
            wl_trans = sav.ltransk
            trans_fct = sav.ttransk
    elif cat == 'wise':
        file = masterDir + '/data/transmission_wise.sav'
        sav = readsav(file)
        if filtre == 'w1':
            wl_trans = sav.ltransw1
            trans_fct = sav.ttransw1
            print(wl_trans)
        elif filtre == 'w2':
            wl_trans = sav.ltransw2
            trans_fct = sav.ttransw2
        elif filtre == 'w3':
            wl_trans = sav.ltransw3
            trans_fct = sav.ttransw3
        elif filtre == 'w4':
            wl_trans = sav.ltransw4
            trans_fct = sav.ttransw4

    x = wl[np.logical_and(wl >= np.min(wl_trans), wl <= np.max(wl_trans))]
    y = flux[np.logical_and(wl >= np.min(wl_trans), wl <= np.max(wl_trans))]

    fct_trans = interp1d(wl_trans, trans_fct)
    full_trans = fct_trans(x)

    fsynt = np.trapz(y * full_trans, x=x) / np.trapz(full_trans, x=x)

    return fsynt

#------------------------------


def mag2flux(mag, cat, emag=None, filtre=None):
    # --- Compute the flux of a given magnitude in a given filter (2MASS or WISE) ---

    mag = np.array(mag)
    if cat == '2mass':
        pointx = [1.235, 1.662, 2.159] * u.um
        p0 = np.array([3.129E-13, 1.133E-13, 4.283E-14]) * 1.e4 * u.W / u.m**2 / u.um
        ep0 = np.array([5.464E-15, 2.212E-15, 8.053E-16]) * 1.e4 * u.W / u.m**2 / u.um
        ind = {'j': 0, 'h': 1, 'k': 2}
    elif cat == 'wise':
        pointx = [3.4, 4.6, 11.56, 22] * u.um
        p0 = np.array([8.1787E-15, 2.4150E-15, 6.5151E-17, 5.0901E-18]) * 1.e4 * u.W / u.m**2 / u.um
        ep0 = np.array([1.2118E-16, 3.5454E-17, 9.8851E-19, 1.0783E-19]) * 1.e4 * u.W / u.m**2 / u.um
        ind = {'w1': 0, 'w2': 1, 'w3': 2, 'w4': 3}

    if filtre:
        if len(filtre) > 1:
            li = []
            for f in filtre:
                li.append(ind[f])
            idx = li
        else:
            idx = ind[filtre]
        p0 = p0[idx]
        ep0 = ep0[idx]
        pointx = pointx[idx]

    flambda = 10.**(-mag / 2.5).squeeze() * p0

    if emag is not None:
        emag = np.array(emag)
        part1 = ep0 * (10.**(-mag / 2.5))
        part2 = emag * p0 * (10.**(-mag / 2.5)) * np.log(10.) * (-1. / 2.5)
        eflambda = np.sqrt(part1**2. + part2**2.)
        # return flambda, eflambda, pointx
    else:
        eflambda = np.zeros_like(flambda)
        # return flambda, pointx

    return flambda, eflambda, pointx

#------------------------------





def scale_spectrum(wl, flux, jmag=None, hmag=None, kmag=None,
                   name=None, return_factor=False, iprint=False):
    '''
    Scale a spectrum to the observed magnitude
    Give at least one magnitude OR the name of the target
    'wl': numpy array in um or quantity array.
    'flux': numpy array in units of J/s/um/m**2 or quantity array
    '''

    uflux = 1.
    if isinstance(flux, u.Quantity):
        flux = flux.to("J/(s*micron*m^2)").value
        uflux = u.Unit("J/(s*micron*m^2)")
    if isinstance(wl, u.Quantity):
        wl = wl.to(u.um).value

    if name:
        if iprint:
            print("Searching in Vizier for " + name + "...")
            print("(Faster if you provide magnitudes instead of a name)")
        try:
            result = Vizier.query_object(name, radius=10 * u.arcsec, catalog=['2MASS'])[0]
        except:
            print("ASTROQUERY IS NOT WORKING....")
        if iprint:
            print(result)
        jmag, hmag, kmag = result['Jmag'][0], result['Hmag'][0], result['Kmag'][0]
        emag = np.array([result['e_Jmag'], result['e_Hmag'], result['e_Kmag']])
    else:
        emag = [None, None, None]

    scaling = []

    if jmag is not None:
        fl_synt = flux_synt(wl, flux, '2mass', 'j') * u.W / u.um / u.m**2
        fl_obs, efl_obs, pointx = mag2flux(jmag, '2mass', filtre='j', emag=emag[0])
        scaling.append((fl_obs / fl_synt))

    if hmag is not None:
        fl_synt = flux_synt(wl, flux, '2mass', 'h') * u.W / u.um / u.m**2
        fl_obs, efl_obs, pointx = mag2flux(hmag, '2mass', filtre='h', emag=emag[1])
        scaling.append((fl_obs / fl_synt))

    if kmag is not None:
        fl_synt = flux_synt(wl, flux, '2mass', 'k') * u.W / u.um / u.m**2
        fl_obs, efl_obs, pointx = mag2flux(kmag, '2mass', filtre='k', emag=emag[2])
        scaling.append((fl_obs / fl_synt))

    scale_factor = np.mean(np.array(scaling))

    print('J = {} \n H = {} \n K = {}'.format(jmag, hmag, kmag))

    flux *= scale_factor

    output = flux * uflux, [jmag, hmag, kmag]
    if return_factor:
        output += scale_factor,

    return output


def through_telluric(flux, T, airmass=1):
    return np.nan_to_num(flux * np.power(T, airmass))


def photon_count(wl, flux, R=73500, extra=False, ptsPerElem=1):
    # --- Compute the photon count for a given flux
    if isinstance(wl, u.Quantity):
        wl = wl.to(u.um).value
    if isinstance(flux, u.Quantity):
        flux = flux.to("J/(s*micron)").value

    if wl.size > 1:
        E_photon = const.h * (const.c / (wl * u.um)).decompose()
        R = find_R(wl)[0]
        d_wl = wl * u.um / (ptsPerElem * R)
    else:
        E_photon = const.h * (const.c / (wl * u.um)).decompose()
        d_wl = wl * u.um / (ptsPerElem * R)

    N_photon = np.clip((flux * d_wl / E_photon).value, 0, None)

    # -------------------

    if extra is True:
        return N_photon, E_photon, d_wl
    else:
        return N_photon


def integ_time(wl, flux, N_lim=50):  # Real SPIRou N_lim = 150

    npix = 20
    saturation = npix * 40000  # photons/pixel *20 pixels physique pour chaque éléments de mon vecteur en longueur d'onde
    READ = 5.57  # sec  # Read time for SPIRou
    photons = photon_count(wl, flux, ptsPerElem=2)  # photons/s/20pxl spectral or 1/2 elem. of resolution

    # Number of READ to reach the saturation limit
    N = np.floor(saturation / (READ * photons.max()))

    # - Setting a limit on the number of READ
    if N > N_lim:
        N = N_lim

    EIT = (N - 1) * READ  # Effective Integration time
    TT = (N + 1) * READ   # Total time
    DC = EIT / TT       # Duty Cycle
    PCT = N * READ      # Photon Collection Time

    dic = {'N': N,
           'EIT': EIT,
           'TT': TT,
           'DC': DC,
           'PCT': PCT,
           'phot': photons}

    return dic


def add_noise(N_photon, dt, all_source=False, n_transit=1, noisy=True, boost=1.):

    if boost != 1:
        N_photon *= 1. - (boost / 50.)
    # --- Shot Noise
    signal = np.random.poisson(lam=np.floor(N_photon)).astype(float)  # on applique le bruit de photon # N_photon
    photNoise = np.sqrt(N_photon)  # RMS du bruit de photon (sigma du bruit phot) pour une pose
    # shot = signal*dt

    if all_source is True:
        # - All the noise sources :
        npix = 20   # Number of pixels per spectral pixel (=1/2 Res. Elem.)
        background = 0
        RON = 5     # [e-] Read-out Noise
        dark = 0.007  # [e-/s] Dark Current (includes the small background)

        noise = np.sqrt(n_transit * npix * (background * dt + dark * dt + RON**2)) * boost  # autres sources de bruit
        signal += np.random.randn(signal.size) * noise  # ajoute les autres sources de bruit
        noise_tot = np.sqrt(photNoise**2 + noise**2)  # bruit tot
    else:
        noise_tot = photNoise  # - Noise = sqrt(S) where S[é] = s[é/s]*t[s]

    # signal *= E_photon / delta_wl
    # noise_tot *= E_photon / delta_wl

    if noisy is False:
        signal = N_photon

    return np.clip(signal, 0, None), noise_tot


def out_SPIROU(wl, flux, t_exp=None, noisy=True, nb_transit=1, print_info=False,
               Rbf=None, NIRPS=False, OH_emis=False, throughput=0.1,
               lb_range=[1.001, 2.35], Res=73500, boost_noise=1.):

    if isinstance(lb_range, u.Quantity):
        lb_range = list(lb_range.to('um').value)

    # - SPIRou parameters --

    if NIRPS is True:
        Res = 100000
        lb_range = [1.001, 1.8]

    area_CFHT = (81729 * u.cm**2).to(u.m**2)  # 81700 / 88300 / (cst.pi*(D/2)**2-cst.pi*(obscur/2)**2)
    # throughput = 0.10  # 10%
    # D = 3.58 * u.m
    # obscur = 0.95 * u.m
    # lb_range = [1.001, 2.35]
    # elem_res = 4.2 * u.km/u.s

    # - Resampling of the original spectrum to the resolution and sampling of SPIRou
    new_wl, spectre = resampling(wl, flux, Res, Rbf=Rbf, ptsPerElem=2, lb_range=lb_range)
    spectre = spectre * u.Unit("J/(s*micron*m^2)")

# - Spectrum in J/s/um/m**2 times the mirror area, t_exp and throughput
# - to get Energy per unit of wavelength per second

    spectre = (spectre * area_CFHT * throughput).to(u.J / u.um / u.s)

    # -- Setting the integration time
    if t_exp is not None:
        if isinstance(t_exp, u.Quantity):
            t_exp = t_exp.to(u.s).value
        oh = 40  # 11  # Overheads = 1rst READ + 1 RESET = 2*5.5 sec
        dt = t_exp - oh
        if print_info is True:
            print('T exp = {}'.format(dt))

    else:
        # - Computing the integration time at the saturation limit
        # Limited at N_lim=50, but can be changed
        out = integ_time(new_wl, spectre)  # , N_lim=X
        dt = out['EIT']

        if print_info is True:
            print('Max el/pxl/sec = {}'.format(out['phot'].max() / 20))
            # print('Max el/resElem/sec = {}'.format(np.floor(out['phot']*2)))
            print('N = {}'.format(out['N']))
            print('T exp = {}'.format(dt))
            print('T tot = {}'.format(out['TT']))
            print('DC = {}'.format(out['DC']))

            print('Median el/pxl/exposure = {}'.format(np.median(out['phot']) / 20 * dt))
            print('Median el/resElem/exposure = {}'.format(np.median(out['phot']) * 2 * dt))

    # if noisy is True:
    N_photon = photon_count(new_wl, spectre * dt * nb_transit, ptsPerElem=1)

    if OH_emis is True:
        data = np.load(masterDir + 'tellStandards/OHspec.npz')
        # oh_phot = data['oh_phot'] / u.s  # Nphot/s
        # oh_wv = data['wv']
        # oh_fct = interp1d(oh_wv, oh_flux * dt)  # Nphot
        N_photon += data['oh_phot'] / u.s * dt

    signal, noise = add_noise(N_photon, dt, all_source=True, n_transit=nb_transit,
                              noisy=noisy, boost=boost_noise)
    # else:
    #     # signal = spectre
    #     signal, E_photon, delta_wl = photon_count(new_wl, spectre * dt * nb_transit, extra=True)
    #     noise = np.zeros_like(signal)

    # print(np.where(signal < 0)[0].shape)
    # print('')

    return new_wl, signal, noise


# def inject_a_signal(wave, flux, wave_model, flux_model, dv_pl, sep, R_star, A_star, 
#                     resol=70000, boost=1., level=0., tr=None):

#     n_spec, nord, _ = flux.shape
    
#     # --- Spectre en transmission de l'atmosphère de la planète
#     P_wl, uniq_idx = np.unique(wave_model, return_index=True)
#     P_depth = np.array(flux_model[uniq_idx])
#     P_R = (find_R(P_wl)[2]).mean()  # resolution power # 250000 #
#     print(P_R)

#     # --- PLANET ROTATION 
# #     Vsini_planet = 3.4*u.km/u.s
# #     P_depth = rotation_spectrum(P_depth, Vsini_planet, P_R)
    
#     # --- Boosting the signal but keeping the same transit depth
    
#     P_depth *= boost
#     TD =  P_depth * 1e-6 # /cds.ppm     # - Transit depth not in ppm   
#     TD -= level
#     print('3 Sigma Planet signal = {:4.0f} ppm'.format(3*np.std(P_depth)))
#     print('Mean transit depth = {:1.2f}%'.format(TD.mean()*100))
    
#     # -- Doppler shifting it 
#     TD_shift = hm.doppler_shift2(P_wl, TD, dv_pl)
    
#     flux_inj=np.empty_like(flux)
#     Rpl=np.empty_like(flux)
#     for tt in range(n_spec):
#         hm.print_static(tt)
#         for iOrd in range(nord):

#             if nord > 1:
#                 resol = get_var_res(iOrd, P_wl)
#             elif nord == 1:
#                 resol = np.ones(P_wl.size) * resol

#             # --- Resampling it
#             P_wl_resamp, P_depth_resamp = resampling(P_wl, TD_shift[tt], 
#                                                               resol, Rbf=P_R, 
#                                                               sample=wave[tt,iOrd])

#             # - Stellar disk fraction that is hidden by the planet - (Overlapping circles - Kipping2011)
#             Rpl[tt, iOrd, :] = np.sqrt(P_depth_resamp) * R_star.to(u.m)  # - Planet radius
    
# #             alpha = hm.circle_overlap(R_star.to(u.m), Rpl[tt, iOrd, :]*u.m, sep[tt]) / A_star.to(u.m**2*u.rad)*u.rad
#             alpha = hm.calc_tr_lightcurve(p, coeffs, self.t.value, ld_model=ld_model)

#             flux_inj[tt, iOrd, :] = (1. - alpha) * flux[tt, iOrd, :]  # flux[random_spec_idx[tt], iOrd, :]
            
#     return np.ma.array(flux_inj, mask=~np.isfinite(flux_inj)), Rpl


from astropy.convolution import convolve, Box1DKernel
def box_binning(model, box_size=2):
#     print("Binning with box kernel of ", box_size)
#     return np.convolve(model, Box1DKernel(box_size).array, mode='same')
    return convolve(model, Box1DKernel(box_size))


def calc_bin_edges(tr, pad=0):
    
    wv_borders_ord = []
    wv_ord = []
    for iOrd in range(tr.nord):
        wv = tr.wv[iOrd]
        dwv = np.diff(wv)
        wv_border_down = np.insert(wv[1:]-dwv/2, 0, wv[0]-dwv[0]/2)
        wv_borders = np.append(wv_border_down, wv[-1]+dwv[-1]/2)
        
        if pad > 0:
            wv_border_temp = wv_borders
            wv_temp = wv
            for i in range(pad):
                wv_border_temp = np.insert(wv_border_temp, 0, wv_border_temp[0]-dwv[:10].mean())
                wv_borders2 = np.append(wv_border_temp, wv_border_temp[-1]+dwv[-10:].mean())
                wv_border_temp = wv_borders2

                wv_temp = np.insert(wv_temp, 0, wv_temp[0]-dwv[:10].mean())
                wv_2 = np.append(wv_temp, wv_temp[-1]+dwv[-10:].mean())
                wv_temp = wv_2

            wv_borders_ord.append(wv_borders2)
            wv_ord.append(wv_2)
        else:
            wv_borders_ord.append(wv_borders)
            wv_ord.append(wv)
            
        
    wv_borders_ord = np.array(wv_borders_ord)
    tr.wv_bins = wv_borders_ord
    tr.wv_ext = wv_ord


from scipy import stats
def binning_model(wave0, model0, wv_borders):
    binned_mod, _, _ = stats.binned_statistic(wave0, model0, 'median', bins=wv_borders)
    return binned_mod


def quick_inject_clean(wave, flux, P_x, P_y, dv_pl, sep, R_star, A_star,
                  R0=None, alpha=None, RV=0, dv_star=0, kind_trans='transmission'):

    _, nord, _ = flux.shape
#     wv = np.mean(wave, axis=0)

    # - Calculate the shift -
#     if dv_star != 0:
#         dv_pl += dv_star
#     if RV != 0:
#         dv_pl += RV 
    shifts = hm.calc_shift(dv_pl, kind='rel')

    if (P_y.ndim == 1):
        # - Interpolate over the orginal data
        fct = interp1d_masked(P_x, P_y, kind='cubic', fill_value='extrapolate')

#     if R0 is None:
#         R0 = np.sqrt(P_y.mean()) * R_star.to(u.m)
#     if alpha is None:
#         alpha = np.array([hm.circle_overlap(R_star.to(u.m), R0.to(u.m), sep_i) / A_star for sep_i in sep])

    flux_inj=np.empty_like(flux)
    P_depth_shift=np.empty_like(flux)
    
    for iOrd in range(nord):
        
        if P_y.ndim == 1:
            # - Interpolating on spirou grid while shifting
            P_depth_shift[:, iOrd, :] = fct(wave[:, iOrd] / shifts[:,None])   # shifts1 / shifts2
        else:
            # - Interpolating the model to shift
            fct = interp1d_masked(P_x[iOrd], P_y[iOrd], kind='cubic', fill_value='extrapolate')
            # - Evaluating on spirou grid while shifting
            P_depth_shift[:, iOrd, :] = fct(wave[:, iOrd] / shifts[:,None])   # shifts1 / shifts2
            
        # - Adding the planet contribution 
        if kind_trans == "transmission":
            flux_inj[:, iOrd, :] = (1. - alpha[:,None] * np.ma.masked_invalid(P_depth_shift[:, iOrd, :])) * flux[:, iOrd, :] 
        elif kind_trans == "emission":
            flux_inj[:, iOrd, :] = (1. + alpha[:,None] * np.ma.masked_invalid(P_depth_shift[:, iOrd, :])) * flux[:, iOrd, :] 

    return np.ma.masked_invalid(flux_inj), P_depth_shift




def quick_inject_clean_allrv(wave, flux, P_x, P_y, dv_pl, sep, R_star, A_star,
                 boost=1., level=0., resol=70000, P_R=70000, R0=None, 
                 alpha=None, verbose=False, RV=0, dv_star=0):

    n_spec, nord, npix = flux.shape
#     wv = np.mean(wave, axis=0)

    # - Calculate the shift -
#     if dv_star != 0:
#         dv_pl += dv_star
#     if RV != 0:
#         dv_pl += RV 
    shifts = hm.calc_shift(dv_pl, kind='rel')

    if (P_y.ndim == 1):
        # - Interpolate over the orginal data
        fct = interp1d_masked(P_x, P_y, kind='cubic', fill_value='extrapolate')

    flux_inj=np.ones((n_spec, nord, npix, dv_pl.shape[-1]))*np.nan
    P_depth_shift=np.ones((n_spec, nord, npix, dv_pl.shape[-1]))*np.nan
    
    for iOrd in range(nord):
        
        if P_y.ndim == 1:
            # - Interpolating on spirou grid while shifting
            P_depth_shift[:, iOrd, :, :] = fct(wave[:, iOrd, :, None] / shifts[:,None,:])   # shifts1 / shifts2
        else:
            # - Interpolating the model to shift
            fct = interp1d_masked(P_x[iOrd], P_y[iOrd], kind='cubic', fill_value='extrapolate')
            # - Evaluating on spirou grid while shifting
            P_depth_shift[:, iOrd, :, :] = fct(wave[:, iOrd, None] / shifts[:,None,:])   # shifts1 / shifts2
            
        # - Adding the planet contribution 
        flux_inj[:, iOrd, :,:] = (1. - alpha[:,None,None] * np.ma.masked_invalid(P_depth_shift[:, iOrd, :,:])) * \
                                  flux[:, iOrd, :,None] 
        
    return np.ma.masked_invalid(flux_inj)



def quick_inject(wave, flux, P_x, P_y, dv_pl, sep, R_star, A_star,
                 boost=1., level=0., resol=70000, P_R=70000, R0=None, 
                 alpha=None, verbose=False, binning=False, wv_borders=None, shift_it='after'):

    n_spec, nord, _ = flux.shape
    if R0 is None:
        R0 = np.sqrt(P_y.mean()) * R_star.to(u.m)

    wv = np.mean(wave, axis=0)
    # - Building wave borders
    if wv_borders is None:
        wv_ext, wv_bins = [], []
        for iOrd in range(nord):
            dwv = np.diff(wv[iOrd])
            wv_border_down = np.insert(wv[1:]-dwv/2, 0, wv[0]-dwv[0]/2)
            wv_borders = np.append(wv_border_down, wv[-1]+dwv[-1]/2)
            wv_bins.append(wv_borders)
            wv_ext.append(wave[0, iOrd, :])
    else:
        wv_ext, wv_bins = wv_borders
                
#     # --- Boosting the signal but keeping the same transit depth
#     TD =  P_depth * boost  # - P_depth (planet Transit depth) must not be in ppm 
#     TD -= level
#     if verbose is True:
#         print('3 Sigma Planet signal = {:4.0f} ppm'.format(3*np.std((P_depth*boost)*1e6)))
#         print('Mean transit depth = {:1.2f}%'.format(TD.mean()*100))
    
#     # --- Resampling it
#     if P_R != resol:
#         P_x, P_y = resampling(P_wl, TD, np.ones_like(TD)*resol, Rbf=P_R, sample=P_wl)#, sample=wave[tt,iOrd])
#     else:
#         P_x, P_y = P_wl, TD
    
    # - Calculate the shift -
    shifts = hm.calc_shift(dv_pl, kind='rel')

    if (P_y.ndim == 1):
        # - Interpolate over the orginal data
        fct = interp1d_masked(P_x, P_y, kind='cubic', fill_value='extrapolate')

    if alpha is None:
        alpha = np.array([hm.circle_overlap(R_star.to(u.m), R0.to(u.m), sep_i) / A_star for sep_i in sep])

    flux_inj=np.empty_like(flux)
    P_depth_shift=np.empty_like(flux)
    
#     plt.figure()
#     plt.plot(P_x, P_y, 'k')
    for iOrd in range(nord):
        if binning is False:
            if shift_it == 'after':
                if P_y.ndim == 1:
                    # - Interpolating on spirou grid while shifting
                    P_depth_shift[:, iOrd, :] = fct(wave[:, iOrd] / shifts[:,None])
                else:
                    # - Interpolating the model to shift
                    fct = interp1d_masked(P_x[iOrd], P_y[iOrd], kind='cubic', fill_value='extrapolate')
                    # - Evaluating on spirou grid while shifting
                    P_depth_shift[:, iOrd, :] = fct(wave[:, iOrd] / shifts[:,None])
                
#             elif shift_it == 'before':
#                 wv_sh_lim = np.concatenate((wv[iOrd][0]/shifts[[0,-1]], wv[iOrd][-1]/shifts[[0,-1]]))
#                 cond = (P_x >= wv_sh_lim.min()) & (P_x <= wv_sh_lim.max())
#                 shifted = fct(P_x[cond][None,:]/shifts[:,None])
                
#                 for n in range(n_spec):
#                     hm.print_static(iOrd, n, '  ')
#                     fct_sh = interp1d_masked(P_x[cond], shifted[n], kind='cubic', fill_value='extrapolate')
# #                     if P_depth.ndim == 1:
#                     # - Interpolating on spirou grid while shifting
#                     P_depth_shift[n, iOrd, :] = fct_sh(wave[n, iOrd])
# #                     else:
# #                         # - Interpolating the model to shift
# #                         fct = interp1d_masked(P_x[iOrd], P_y[iOrd], kind='cubic', fill_value='extrapolate')
# #                         # - Evaluating on spirou grid while shifting
# #                         P_depth_shift[n, iOrd, :] = fct_sh(wave[n, iOrd])
#         else:  
#             if shift_it == 'after':
#                 hm.print_static(iOrd, '  ')
#                 wv_sh_lim = np.concatenate((wv_ext[iOrd][0]/shifts[[0,-1]], wv_ext[iOrd][-1]/shifts[[0,-1]]))
#                 cond = (P_x >= wv_sh_lim.min()) & (P_x <= wv_sh_lim.max())

#                 binned, _, _ = stats.binned_statistic(P_x[cond], P_y[cond], 'mean', bins=wv_bins[iOrd])
                
#                 # - Interpolating the spirou grid to shift
#                 fct = interp1d_masked(wv_ext[iOrd], np.ma.masked_invalid(binned), kind='cubic', fill_value='extrapolate')
#     #           # - Shifting it
#                 P_depth_shift[:, iOrd, :] = fct(wave[:, iOrd] / shifts[:,None])

#             elif shift_it == 'before':
#                 wv_sh_lim = np.concatenate((wv[iOrd][0]/shifts[[0,-1]], wv[iOrd][-1]/shifts[[0,-1]]))
#                 cond = (P_x >= wv_sh_lim.min()) & (P_x <= wv_sh_lim.max())
#                 shifted = fct(P_x[cond][None,:]/shifts[:,None])

#                 for n in range(n_spec):
#                     hm.print_static(iOrd, n, '  ')
# #                     P_depth_shift[n, iOrd, :] = binning_model(P_x[cond], shifted[n], wv_bins[iOrd])
#                     P_depth_shift[n, iOrd, :], _, _ = stats.binned_statistic(P_x[cond], shifted[n], 
#                                                                             'median', bins=wv_bins[iOrd])
            
        # - Adding the planet contribution 
        flux_inj[:, iOrd, :] = (1. - alpha[:,None] * np.ma.masked_invalid(P_depth_shift[:, iOrd, :])) * flux[:, iOrd, :] 
#     hm.stop()
    return np.ma.masked_invalid(flux_inj), P_depth_shift



# def quick_inject(wave, flux, P_wl, P_depth, dv_pl, sep, R_star, A_star,
#                  boost=1., level=0., resol=70000, P_R=70000, R0=None, 
#                  alpha=None, verbose=False, binning=False, wv_borders=None, shift_it='after'):

#     n_spec, nord, _ = flux.shape
#     if R0 is None:
#         R0 = np.sqrt(P_depth.mean()) * R_star.to(u.m)

#     wv = np.mean(wave, axis=0)
#     # - Building wave borders
#     if wv_borders is None:
#         dwv = np.diff(wv[iOrd])
#         wv_border_down = np.insert(wv[1:]-dwv/2, 0, wv[0]-dwv[0]/2)
#         wv_borders = np.append(wv_border_down, wv[-1]+dwv[-1]/2)
#         wv_ext = wave[0, iOrd, :]
#     else:
#         wv_ext, wv_bins = wv_borders
                
#     # --- Boosting the signal but keeping the same transit depth
#     TD =  P_depth * boost  # - P_depth (planet Transit depth) must not be in ppm 
#     TD -= level
#     if verbose is True:
#         print('3 Sigma Planet signal = {:4.0f} ppm'.format(3*np.std((P_depth*boost)*1e6)))
#         print('Mean transit depth = {:1.2f}%'.format(TD.mean()*100))
    
#     # --- Resampling it
#     if P_R != resol:
#         P_x, P_y = resampling(P_wl, TD, np.ones_like(TD)*resol, Rbf=P_R, sample=P_wl)#, sample=wave[tt,iOrd])
#     else:
#         P_x, P_y = P_wl, TD
    
#     # - Calculate the shift -
#     shifts = hm.calc_shift(dv_pl, kind='rel')

#     if (P_depth.ndim == 1):
#         # - Interpolate over the orginal data
#         fct = interp1d_masked(P_x, P_y, kind='cubic', fill_value='extrapolate')

#     if alpha is None:
#         alpha = np.array([hm.circle_overlap(R_star.to(u.m), R0.to(u.m), sep_i) / A_star for sep_i in sep])

#     flux_inj=np.empty_like(flux)
#     P_depth_shift=np.empty_like(flux)
    
# #     plt.figure()
# #     plt.plot(P_x, P_y, 'k')
#     for iOrd in range(nord):
#         if binning is False:
#             if shift_it == 'after':
#                 if P_depth.ndim == 1:
#                     # - Interpolating on spirou grid while shifting
#                     P_depth_shift[:, iOrd, :] = fct(wave[:, iOrd] / shifts[:,None])
#                 else:
#                     # - Interpolating the model to shift
#                     fct = interp1d_masked(P_x[iOrd], P_y[iOrd], kind='cubic', fill_value='extrapolate')
#                     # - Evaluating on spirou grid while shifting
#                     P_depth_shift[:, iOrd, :] = fct(wave[:, iOrd] / shifts[:,None])
                
# #             elif shift_it == 'before':
# #                 wv_sh_lim = np.concatenate((wv[iOrd][0]/shifts[[0,-1]], wv[iOrd][-1]/shifts[[0,-1]]))
# #                 cond = (P_x >= wv_sh_lim.min()) & (P_x <= wv_sh_lim.max())
# #                 shifted = fct(P_x[cond][None,:]/shifts[:,None])
                
# #                 for n in range(n_spec):
# #                     hm.print_static(iOrd, n, '  ')
# #                     fct_sh = interp1d_masked(P_x[cond], shifted[n], kind='cubic', fill_value='extrapolate')
# # #                     if P_depth.ndim == 1:
# #                     # - Interpolating on spirou grid while shifting
# #                     P_depth_shift[n, iOrd, :] = fct_sh(wave[n, iOrd])
# # #                     else:
# # #                         # - Interpolating the model to shift
# # #                         fct = interp1d_masked(P_x[iOrd], P_y[iOrd], kind='cubic', fill_value='extrapolate')
# # #                         # - Evaluating on spirou grid while shifting
# # #                         P_depth_shift[n, iOrd, :] = fct_sh(wave[n, iOrd])
# #         else:  
# #             if shift_it == 'after':
# #                 hm.print_static(iOrd, '  ')
# #                 wv_sh_lim = np.concatenate((wv_ext[iOrd][0]/shifts[[0,-1]], wv_ext[iOrd][-1]/shifts[[0,-1]]))
# #                 cond = (P_x >= wv_sh_lim.min()) & (P_x <= wv_sh_lim.max())

# #                 binned, _, _ = stats.binned_statistic(P_x[cond], P_y[cond], 'mean', bins=wv_bins[iOrd])
                
# #                 # - Interpolating the spirou grid to shift
# #                 fct = interp1d_masked(wv_ext[iOrd], np.ma.masked_invalid(binned), kind='cubic', fill_value='extrapolate')
# #     #           # - Shifting it
# #                 P_depth_shift[:, iOrd, :] = fct(wave[:, iOrd] / shifts[:,None])

# #             elif shift_it == 'before':
# #                 wv_sh_lim = np.concatenate((wv[iOrd][0]/shifts[[0,-1]], wv[iOrd][-1]/shifts[[0,-1]]))
# #                 cond = (P_x >= wv_sh_lim.min()) & (P_x <= wv_sh_lim.max())
# #                 shifted = fct(P_x[cond][None,:]/shifts[:,None])

# #                 for n in range(n_spec):
# #                     hm.print_static(iOrd, n, '  ')
# # #                     P_depth_shift[n, iOrd, :] = binning_model(P_x[cond], shifted[n], wv_bins[iOrd])
# #                     P_depth_shift[n, iOrd, :], _, _ = stats.binned_statistic(P_x[cond], shifted[n], 
# #                                                                             'median', bins=wv_bins[iOrd])
            
#         # - Adding the planet contribution 
#         flux_inj[:, iOrd, :] = (1. - alpha[:,None] * np.ma.masked_invalid(P_depth_shift[:, iOrd, :])) * flux[:, iOrd, :] 
# #     hm.stop()
#     return np.ma.masked_invalid(flux_inj), P_depth_shift




def rotating_spectrum(flux, Vsini, R=70000, ptsPerElem=1):
    """
    #convolve spectrum by rotation broadening profile (assume Gaussian here)
    #wavelength array is approximately at constant R
    #use a fixed pixel width, good enough
    """

    if not isinstance(Vsini,u.Quantity):
        Vsini = Vsini*u.km/u.s
    
    fwhm=R*(Vsini/const.c).decompose()*ptsPerElem  # 2 pix/res el. assumed in S_R
    g=np.exp(-np.mgrid[-5*fwhm:5*fwhm+1]**2/2/(fwhm/2.35)**2)
    g/=g.sum()

    return np.convolve(flux,g,'same')


class Spectrum(QTable):
    '''
    Generic class for spectra.
    To be use for specific spectrum (ex: PHOENIX spectrum) class

    A Spectrum instance is a astropy.table.Table with at least
    a "wave" column and a "flux" column
    '''
    y = "flux"
    scaled = False
    scale_factor = None

    def __init__(self, data=None, wv=np.array([]), flux=np.array([]), **kwargs):

        if data is not None:
            super().__init__(data=data, **kwargs)
            if isinstance(data, Spectrum):
                self.__dict__ = data.__dict__.copy()
        else:
            if not np.array(flux).any():
                flux = np.array(wv * 0)
            super().__init__(data={"wave": wv, self.y: flux}, **kwargs)

    def scale_spectrum(self, jmag=None, hmag=None, kmag=None, name=None,
                       iprint=False, scale_factor=None):

        if self.scaled:
            print("Spectrum already scaled")
            return

        if scale_factor:
            self[self.y] *= scale_factor
        else:
            wl, flux = self["wave"], self[self.y]

            flux, [J, H, K], scale_factor = \
                scale_spectrum(wl, flux, jmag=jmag, hmag=hmag,
                               kmag=kmag, name=name, return_factor=True, iprint=print)
            self[self.y] = flux
            self.mag = {"J": J, "H": H, "K": K}

        self.scale_factor = scale_factor
        self.scaled = True

    def resample(self, Rba, Rbf=None, **kwargs):

        wv, flux = self["wave"], self[self.y]

        try:
            Rbf = self.meta["resPower"]
        except KeyError:
            Rbf = Rbf

        return resampling(wv, flux, Rba, Rbf=Rbf, **kwargs)

    def prod(self, *keys, but=["wave"]):

        keys = list(keys)
        if not keys:
            keys = self.keys()
            for key in but:
                keys.remove(key)

        out = 1.
        for key in keys:
            out *= self[key]

        return out

    def plot(self, *args, fig=None, ax=None, **kwargs):

        if ax is None and fig is None:
            fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(self["wave"], self[self.y], *args, **kwargs)
        return fig, ax

    def conv_vsini(self, vsini, Rbf=None):
        '''
        Convolve star spectrum by rotation broadening profile (assume Gaussian here)
        wavelength array is approximately at constant R
        use a fixed pixel width, good enough
        '''

        flux = self[self.y]

        try:
            Rbf = self.meta["resPower"]
        except KeyError:
            Rbf = Rbf

        try:
            fwhm = Rbf * (vsini / const.c).decompose()
        except:
            fwhm = Rbf * vsini / const.c.value

        g = np.exp(-np.mgrid[-5 * fwhm:5 * fwhm + 1]**2 / 2 / (fwhm / 2.35)**2)
        g /= g.sum()

        self[self.y] = np.convolve(flux, g, 'same') * flux.unit


class PHOENIXspectrum(Spectrum):

    file_frame = 'lte%05g-%1.2f-%1.1f.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
    model_dir = "PHOENIX_HiRes/Z-%1.1f/"
    

    def __init__(self, data=None, Teff=None, logg=None,
                 metallicity=0.0, **kwargs):

        if data is None:
            # Read flux
            masterDir = '/home/boucher/spirou/'
            
            file = self.file_frame % (Teff, logg, metallicity)
            path = masterDir + self.model_dir % (metallicity)
            try:
                hdul = fits.open(path + file)
            except FileNotFoundError:
                print("No such file or directory: "
                      + path + file + "\n Trying to download PHOENIX HiRes" +
                      "spectrum using PHOENIXspectrum.get_hires_spectrum()...")
                self.get_hires_spectrum(teff=Teff, logg=logg, Z=metallicity)
                hdul = fits.open(path + file)

            # Read header
            # self._get_param(hdul[0].header)
            kwargs["meta"] = dict(hdul[0].header)
            funit = kwargs["meta"].pop("BUNIT")
            flux = hdul[0].data * u.Unit(funit)

            # Read wv grid (in a distinct file)
            file = "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
            path = masterDir + 'PHOENIX_HiRes/'
            hdul = fits.open(path + file)
            wvunit = hdul[0].header['UNIT']
            wv = hdul[0].data * u.Unit(wvunit)

            super().__init__(wv=wv, flux=flux, **kwargs)

        else:
            super().__init__(data=data, **kwargs)

    @classmethod
    def get_hires_spectrum(cls, teff=5000, logg=5.0, Z=0.0):

        Z = hm.myround(Z, base=0.5)
        teff = hm.myround(teff, 100)
        logg = hm.myround(logg, base=0.5)
        ModelFileName = cls.file_frame % (teff, logg, Z)
        path = masterDir + cls.model_dir % (Z) + ModelFileName
        url_link = "ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/"\
            + "PHOENIX-ACES-AGSS-COND-2011/Z-{:1.1f}/".format(Z) + ModelFileName

        if os.path.isfile(path) is False:
            print('Downloading the Model ' + ModelFileName + ' ...', end=' ')
            bashCommand = "wget  -O {} '{}'".format(path, url_link)
            os.system(bashCommand)
            print('Done!')
        else:
            print("You already have the model")
            print("(Teff = %05g, logg = %1.2f, Z = %1.1f" % (teff, logg, Z))


class Benneke(Spectrum):

    file_frame = "%s_Metallicity%1.1f_CtoO%1.2f_pCloud%.1f" \
        + "mbar_cHaze%1.0e_pQuench%1.0e_TpPrevTp_Spectrum_FullRes.csv"
    model_dir = "Benneke_emission/"
    y = "thermal"

    def __init__(self, data=None, pl_name=None, metallicity=1.0, CtoO=0.54,
                 pCloud=100000, cHaze=1e-10, pQuench=1e-99, file=None, **kwargs):

        if data is None:
            # Name of the file and path
            if file is None:
                pl_name = "_".join(pl_name.split(' '))  # Convert spaces to underscore
                file = self.file_frame % (pl_name, metallicity, CtoO, pCloud, cHaze, pQuench)
            path = masterDir + self.model_dir

            # Read model
            temp = Spectrum.read(path + file, format='ascii.ecsv')
            super().__init__(temp)
        else:
            super().__init__(data=data, **kwargs)


class Tellurics(Spectrum):

    y = "total"
    file_frame = "tapas_00000{}.ipac.sav"
    model_dir = "telluricModels/TAPAS/za0/{}/"
    molecules = ['H2O', 'O3', 'O2', 'CO2', 'CH4', 'N2O']

    def __init__(self, date_id=0, date=['16Nov2003', '1Jan2018'],
                 **kwargs):

        if 'data' in kwargs:
            super().__init__(**kwargs)
        elif 'wv' in kwargs:
            flux = kwargs.pop('flux', np.array(kwargs['wv'] * 0))
            super().__init__(flux=flux, **kwargs)
        else:
            path = masterDir + self.model_dir.format(date[date_id])
            spec = {}
            for i, mol in enumerate(self.molecules, start=1):
                file = self.file_frame.format(i)
                sav = readsav(path + file)
                if i == 1:
                    spec["wave"] = sav['wl'][::-1] / 1e3
                    spec["total"] = np.ones_like(spec["wave"])
                spec[mol] = np.clip(sav['trans'][::-1], 0, 1)
                spec["total"] *= spec[mol]
            resPower = hm.myround(find_R(spec["wave"])[1], base=100)
            kwargs["meta"] = {"resPower": resPower}
            super().__init__(data=spec, **kwargs)


class OH_lines(Spectrum):

    file = 'OH_list.txt'
    model_dir = "telluricModels/"
    y = "Intensity"

    def __init__(self, data=None, file=None, **kwargs):

        if data is None:
            # Name of the file and path
            if file is None:
                file = self.file
            path = masterDir + self.model_dir

            # Read model
            temp = Spectrum.read(path + file, format='ascii')
            temp.rename_column('Wave', 'wave')
            temp['wave'].unit = 'Angstrom'
            super().__init__(temp)
        else:
            super().__init__(data=data, **kwargs)
