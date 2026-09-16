# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:47:17 2016

@author: AnneBoucher

"""
from __future__ import division
import numpy as np

from . import homemade as hm
from .extract import get_var_res
from astroquery.vizier import Vizier
from astropy import units as u
from astropy import constants as const
from astropy.table import Table, Column, QTable
from astropy.convolution import Gaussian1DKernel
# import astropy.io.ascii as ascii_ap
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scipy.constants as cst
from scipy.interpolate import interp1d
from scipy.io.idl import readsav
import logging

import os.path
# from .config import *
# from importlib import reload

from .mask_tools import interp1d_masked
# import spirou_exo.analysis as a

# from scipy.interpolate import interp1d

# Set logger to see the debug messages
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# from hapi import transmittanceSpectrum
masterDir = '/home/boucher/spirou/'
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
      


#############################################
# --- Code related to convolution kernels ---
#############################################

# --- Utility functions for convolution and kernels ---

def resampling(wl, flux, Raf, Rbf=None, lb_range=None, ptsPerElem=1, sample=None, rot_ker=None):#, plot=False):
    """
    Spectrum resampling, assuming a R that is constant.
    Change the sampling of a spectrum from an initial sampling resolution of Rbf
    to a final sampling resolution of Raf.
    *** Rbf is the sampling resolution, not the actual resolution. ***
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
    if np.isscalar(Raf):
        if Raf < Rbf :
            if rot_ker is None:
#                 print('Resampling without rotation')
                # - On on va vouloir un vecteur de taille ~7xfwhm
                taille = hm.myround(fwhm * 7, base=2) + 1
                # - On génère notre gaussienne avec le nb de point (pxl) qu'on vient de calculer
                profil = hm.gauss(np.arange(taille), mean=(taille - 1) / 2, FWHM=fwhm)

            else:
                if isinstance(rot_ker, np.ndarray):
                    profil = rot_ker
                else:
    #                 print('Resampling with rotation')
                    # R_af is implicitly included in the rot_ker object
                    try:
                        profil = rot_ker.resample(R_sampling, n_os=500, pad=7)
                    except KernelIndexError:
                        # - On on va vouloir un vecteur de taille ~7xfwhm
                        taille = hm.myround(fwhm * 7, base=2) + 1
                        # - On génère notre gaussienne avec le nb de point (pxl) qu'on vient de calculer
                        profil = hm.gauss(np.arange(taille), mean=(taille - 1) / 2, FWHM=fwhm)
            
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


def convert_default_units(list_of_args, default_units):
    """Converts the arguments to default units. If the argument is not an astropy quantity,
    it is left unchanged, so assume it is already in the default units.
    list_of_args: list of astropy quantities or arrays
    default_units: list of strings
    """
    values = tuple()
    for quant, unit in zip(list_of_args, default_units):
        try:
            quant = quant.to(unit).value
        except AttributeError:
            pass
        values += (quant,)
    return values


# --- Utility classes for convolution kernels ---

class KernelIndexError(IndexError):
    pass

# --- Base classes for kernels ---

class BaseKer:

    def __init__(self, resolution):
        '''Base class for rotation kernels with multiple regions.
        resolution: scalar (float or int)
            spectral resolution of the instrument
        '''
        # Get resolution element in m/s
        res_elem = const.c / resolution
        res_elem = res_elem.to('m/s').value

        # Save attributes
        self.res_elem = res_elem


    def degrade_ker(self, rot_ker=None, v_grid=None, norm=True, fwhm=None, **kwargs):
        ''' 
        General method to degrade a kernel
        Parameters
        ----------
        rot_ker: array-like
            kernel to degrade. If None, the method get_ker is called to get the kernel
            specific to the class (won't work with the BaseKer class, but it will with subclasses).
        v_grid: array-like
            velocity grid of the kernel. If None, the method get_ker is called to get the velocity grid,
            like rot_ker. It won't work with the BaseKer class, but it will with subclasses.
        norm: bool
            Wheter to normalize the kernel or not. Default is True.
        fwhm: float
            FWHM of the gaussian kernel to convolve with the kernel. If None, the FWHM is set to the
            resolution element of the kernel object.
        kwargs:
          Passed to `get_ker` method.
        '''
        if fwhm is None:
            fwhm = self.res_elem

        if rot_ker is None: 
            results = self.get_ker(norm=norm, **kwargs)
            v_grid, rot_ker = results[:2]
            
        gauss_ker = hm.gauss(v_grid, 0.0, FWHM=fwhm)
        gauss_ker /= gauss_ker.sum()
        
        out_ker = np.convolve(rot_ker, gauss_ker, mode='same')
        
        if norm:
            out_ker /= out_ker.sum()
        return v_grid, out_ker
    
    def resample(self, res_sampling, rot_ker=None, v_grid=None, norm=True, fwhm=None, **kwargs):
        ''' Resample a kernel to a given sampling resolution. It also downgrades the kernel
        to a specified fwhm or to the default resolution element of the kernel object.
        Parameters
        ----------
        res_sampling: float
            resolution of the sampling needed
        rot_ker: array-like
            kernel to degrade. If None, the method get_ker is called to get the kernel
            specific to the class (won't work with the BaseKer class, but it will with subclasses).
        v_grid: array-like
            velocity grid of the kernel. If None, the method get_ker is called to get the velocity grid,
            like rot_ker. It won't work with the BaseKer class, but it will with subclasses.
        norm: bool
            Wheter to normalize the kernel or not. Default is True.
        fwhm: float
            FWHM of the gaussian kernel to convolve with the kernel. If None, the FWHM is set to the
            resolution element of the kernel object.
        kwargs are passed to degrade_ker method
        '''
        dv_new = const.c / res_sampling
        dv_new = dv_new.to('m/s').value
        v_grid, kernel = self.degrade_ker(rot_ker=rot_ker, v_grid=v_grid, norm=norm, fwhm=fwhm, **kwargs)
        dv_old = np.diff(v_grid).mean()
        ker_spl = interp1d(v_grid, kernel, kind='linear')
        v_grid = np.arange(v_grid.min(), v_grid.max(), dv_new)
        # Re-center (found 2026-08-28, Antoine): np.arange(min, max, step) anchors at
        # `min`, not necessarily landing symmetrically -- left uncorrected, the
        # *returned* kernel array (the only thing most callers, e.g. a plain
        # np.convolve(flux, kernel), actually use) is not centered on its own middle
        # index, silently introducing a small systematic velocity offset. Matches
        # get_ker()'s own convention (`v_grid -= np.mean(v_grid)`) instead of losing it
        # here. The resulting shift was tiny in the case that surfaced this
        # (~0.5 m/s out of a ~1200 m/s pixel, well-sampled kernel) but scales with how
        # coarsely the kernel is sampled (`n_os`), so worth fixing at the source
        # rather than relying on it staying negligible.
        v_grid -= np.mean(v_grid)
        out_ker = ker_spl(v_grid)
        out_ker *= dv_new / dv_old
        return out_ker
    
    def show(self, norm=True, **kwargs):
        """ Plot the kernel and the degraded kernel
        Parameters
        ----------
        norm: bool
            Wheter to normalize the kernel or not. Default is True.
        kwargs are passed to get_ker method.
        """
        res_elem = self.res_elem
        v_grid, kernel = self.get_ker(norm=norm, **kwargs)
        gauss_ker = hm.gauss(v_grid, 0.0, FWHM=res_elem)
        # if norm:
        #     gauss_ker /= gauss_ker.sum()
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
        
        axtwin = plt.gca().twinx()
        axtwin.plot(v_grid/1e3, gauss_ker, "--", color="gray",
                    label='Instrumental resolution element')

        return fig


class BaseKerMulti:
    """
    Base class for rotation kernels with multiple regions.

    Parameters
    ----------
    resolution: scalar (float or int)
        spectral resolution of the instrument
    """

    def __init__(self, resolution):
        """Base class for rotation kernels with multiple regions.
        resolution: scalar (float or int)
            spectral resolution of the instrument
        """

        # Get resolution element in m/s
        res_elem = const.c / resolution
        res_elem = res_elem.to('m/s').value

        # Save attributes
        self.res_elem = res_elem

    def degrade_ker(self, rot_ker_list=None, v_grid=None, norm=True, fwhm=None, **kwargs):
        """
        General method to degrade a kernel
        Parameters
        ----------
        rot_ker_list: list of array-like
            List of kernel to degrade. If None, the method get_ker is called to get the kernel
            specific to the class.
        v_grid: array-like
            velocity grid of the kernel. If None, the method get_ker is called to get the velocity grid,
            like rot_ker.
        norm: bool
            Wheter to normalize the kernel or not. Default is True.
        fwhm: float
            FWHM of the gaussian kernel to convolve with the kernel. If None, the FWHM is set to the
            resolution element of the kernel object.
        kwargs:
          Passed to `get_ker` method if `rot_ker_list` is None.
        """
        # Get default values if not specified
        if fwhm is None:
            fwhm = self.res_elem

        if rot_ker_list is None: 
            v_grid, rot_ker_list = self.get_ker(norm=norm, **kwargs)
        gauss_ker = hm.gauss(v_grid, 0.0, FWHM=fwhm)

        out_ker_list = [np.convolve(rot_ker, gauss_ker, mode='same') for rot_ker in rot_ker_list]

        # Make sure the sum over all kernels is 1 (conservation of flux)
        if norm:
            all_sum = np.sum(out_ker_list)
            out_ker_list = [ker / all_sum for ker in out_ker_list]

        return v_grid, out_ker_list
    
    def resample(self, res_sampling, rot_ker_list=None, v_grid=None, norm=True, fwhm=None, **kwargs):
        """ Resample a kernel to a given sampling resolution. It also downgrades the kernel
        to a specified fwhm or to the default resolution element of the kernel object.
        Parameters
        ----------
        res_sampling: float
            resolution of the sampling needed
        rot_ker_list: list of array-like
            List of kernels to degrade. If None, the method get_ker is called to get the kernel
            specific to the class.
        v_grid: array-like
            velocity grid of the kernel. If None, the method get_ker is called to get the velocity grid,
            like rot_ker.
        norm: bool
            Wheter to normalize the kernel or not. Default is True.
        fwhm: float
            FWHM of the gaussian kernel to convolve with the kernel. If None, the FWHM is set to the
            resolution element of the kernel object.
        kwargs are passed to degrade_ker method
        """
        # Find the dv equivalent to the sampling resolution
        dv_new = const.c / res_sampling
        dv_new = dv_new.to('m/s').value

        # Get default values if not specified
        if fwhm is None:
            fwhm = self.res_elem

        if rot_ker_list is None: 
            v_grid, rot_ker_list = self.degrade_ker(rot_ker_list=rot_ker_list, v_grid=v_grid, norm=norm, fwhm=fwhm, **kwargs)

        # Interpolate the kernel to the new sampling
        out_ker_list = []
        for rot_ker in rot_ker_list:
            ker_spl = interp1d(v_grid, rot_ker, kind='linear')
            v_grid_new = np.arange(v_grid.min(), v_grid.max(), dv_new)
            # Re-center (found 2026-08-28, see BaseKer.resample's identical fix for
            # the full explanation): np.arange(min, max, step) anchors at `min`, not
            # necessarily landing symmetrically, so the returned kernel isn't centered
            # on its own middle index unless corrected here.
            v_grid_new -= np.mean(v_grid_new)
            out_ker_list.append(ker_spl(v_grid_new))
            
        # Make sure the sum over all kernels is 1 (conservation of flux)
        if norm == True:
            all_sum = np.sum(out_ker_list)
            out_ker_list = [ker / all_sum for ker in out_ker_list]
            
        return out_ker_list


# --- Specific classes and functions for kernels ---

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


class RotKerTransit(BaseKer):
    """Geometric wind-broadening kernel for transit (transmission) spectroscopy.

    Derived purely from the planet's radius/mass/equilibrium temperature/rotation
    (or wind) frequency -- as opposed to `RotKerTransitCloudy`, this one has no
    cloud/vphi options. Inherits `degrade_ker`/`resample`/`show` from `BaseKer`
    (Chantier A Phase 3): this class used to hand-duplicate its own copies of those
    three methods, which had drifted from `BaseKer`'s -- notably `resample` was
    missing the `dv_new / dv_old` rescale that keeps the kernel's discrete sum
    normalized after interpolating onto a different-density velocity grid, so a
    resampled kernel's amplitude was off by a `dv_old / dv_new` factor. `BaseKer`'s
    versions fix that (and add the optional `fwhm` override) for free.
    """

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
        super().__init__(resolution)
        if mu is None:
            mu = 2.3 * u.u
        g_surf = const.G * pl_mass / pl_rad**2
        scale_height = const.k_B * t_eq / (mu * g_surf)
        z_h = 5 * scale_height.to('m').value
        r_p = pl_rad.to('m').value
        omega = omega.to('1/s').value
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


class SolidRotationKernel(BaseKer):
    """Rotational-broadening kernel for a uniformly rotating, uniformly bright disk.

    This is the *phase-independent* default 'emission' rotation kernel (Chantier A
    Phase 3, `model_sequence.py::_build_default_rotation_kernel`): the planet's whole
    observer-facing disk broadens its own emitted flux by solid-body rotation, with
    no notion of citrus regions or orbital phase. It used to be built by reusing
    `CitrusRotationKernel` with a single citrus boundary (`citrus_phases=[0.0]`), on
    the assumption that one boundary degenerates into "the whole disk, phase
    independent". **That assumption was wrong**: `citrus_to_ker`'s region-boundary
    geometry is built for >= 2 boundaries, and a single self-referencing boundary
    (`reg_2 == reg_1`) falls into an unintended branch -- verified numerically
    (Chantier A Phase 3 follow-up, 2026-08-28) to give a kernel that is *not*
    symmetric around v=0 (centroid off by ~3 km/s for a WASP-33b-like case) and,
    for orbital phases other than exactly 0.0, entirely zero. This class replaces
    that reuse with the closed-form profile directly, verified symmetric.

    Physics: for a disk of projected radius `pl_rad` rotating at equatorial
    velocity `v_eq = pl_rad * omega`, the flux Doppler-shifted to line-of-sight
    velocity v is proportional to the chord length of the disk at that velocity --
    the classic "uniform disk" rotational-broadening profile (Gray's formula with
    zero limb darkening): `G(v) = sqrt(1 - (v / v_eq)**2)` for `|v| <= v_eq`, 0
    otherwise.

    Parameters
    ----------
    pl_rad : scalar astropy quantity
        Planet radius.
    omega : scalar astropy quantity
        Rotation rate (rad/s) -- `rot_factor` times the orbital angular frequency
        for the tidally-locked assumption used by `_build_default_rotation_kernel`.
    resolution : scalar (float or int)
        Spectral resolution of the instrument.
    """

    def __init__(self, pl_rad, omega, resolution):
        super().__init__(resolution)
        pl_rad, omega = convert_default_units([pl_rad, omega], ['m', '1/s'])
        self.v_eq = pl_rad * omega

    def get_ker(self, n_os=None, pad=7, norm=True):
        """
        Parameters
        ----------
        n_os : scalar, optional
            Oversampling (to sample the kernel). If None, a default sampling is
            derived from `v_eq`, same convention as `CitrusRotationKernel.get_ker`.
        pad : scalar
            Pad around the kernel, in units of resolution elements. Values in the
            pad are set to zero.
        norm : bool
            Whether to normalize the kernel to unit sum. Default True.

        Returns
        -------
        v_grid : np.ndarray
        kernel : np.ndarray
        """
        res_elem = self.res_elem
        v_eq = self.v_eq

        v_max = v_eq + pad * res_elem
        if n_os is None:
            delta_v = np.abs(v_eq / 10)
            log.info(f'No oversampling given. Using delta_v = {delta_v:.2f} m/s')
        else:
            delta_v = res_elem / n_os

        v_grid = np.arange(-v_max, v_max, delta_v)
        v_grid -= np.mean(v_grid)

        kernel = np.zeros_like(v_grid)
        visible = np.abs(v_grid) <= v_eq
        kernel[visible] = np.sqrt(1 - (v_grid[visible] / v_eq) ** 2)

        if norm:
            kernel /= kernel.sum()

        return v_grid, kernel


class RotKerTransitCloudy(BaseKer):
    """
    Rotational kernel for transit at dv constant (constant resolution)

    Parameters
    ----------
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
    """

    def __init__(self, pl_rad, pl_mass, t_eq, omega, resolution,
                 left_val=1, right_val=1, step_smooth=1, v_mid=0, mu=None, 
                 gauss=False, x0=0., fwhm=4000.,  
                 gamma1=2/np.pi, gamma2=2/np.pi, amp1=5, amp2=5, vphi=False):
        """
        Rotational kernel for transit at dv constant (constant resolution)
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
        
        """
        super().__init__(resolution)

        if mu is None:
            mu = 2.3 * u.u
        g_surf = const.G * pl_mass / pl_rad**2
        scale_height = const.k_B * t_eq / (mu * g_surf)
        z_h = 5 * scale_height.to('m').value
        r_p = pl_rad.to('m').value
        omega = omega.to('1/s').value

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
        
    def get_ker(self, n_os=None, pad=7, v_grid=None, norm=False):
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

        # Get cloud transmission function
        idx_valid = (kernel > 0)
        norm_factor = kernel[idx_valid].sum()  # normalization factor
        if idx_valid.sum() <= 1:
            idx_valid = np.searchsorted(v_grid, 0)
            kernel[idx_valid] = 1
            # raise KernelIndexError("Kernel size too small for grid.")
        clouds = np.ones_like(kernel) * np.nan
        clouds[idx_valid] = box_smoothed_step(v_grid[idx_valid], *clouds_args)        
        kernel[idx_valid] = kernel[idx_valid] * clouds[idx_valid]
        kernel /= norm_factor
        
        # normalize to unity
        if norm:
            kernel /= kernel.sum()

        return v_grid, kernel, clouds
    
    # TODO: Should be the get_ker method in a separate class
    def get_ker_vphi(self, n_os=1000, pad=7, norm=False):
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
        if norm:
            kernel = kernel/(kernel.sum())


#         v_max = np.max(np.abs([self.amp1, self.amp2])) + pad*res_elem*(u.m/u.s).to(u.km/u.s)#.value
#         kernel, bins = np.histogram(v_tot, bins=int(n_os/2), range=[-v_max, v_max])
#         v_grid = 0.5*(bins[1:] + bins[:-1])

#         kernel = np.array(kernel)
#         ker_sum = kernel.sum()

#         kernel = kernel/ker_sum

        return v_grid, kernel

    # TODO: Should be the get_ker method in a separate class
    def get_ker_gauss(self, n_os=None, pad=7):
        
        res_elem = self.res_elem
        omega = self.omega
        z_h = self.z_h
        r_p = self.r_p 
#         print(res_elem,omega,z_h,r_p)

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


def citrus_to_ker(y_coord, citrus_phases, phase, radius=1.):
    
    # Put the phase back between valid phases
    citrus_phases = np.mod(citrus_phases, 1)
    
    # Not valid range if oout of the projected sphere
    valid_range = np.abs(y_coord) <= radius

    disk_bounds = np.zeros_like(y_coord)
    disk_bounds[valid_range] = np.sqrt(radius**2 - y_coord[valid_range]**2)
    zeros = np.zeros_like(disk_bounds)
    
    total_phases = (citrus_phases + phase) % 1
    
    western = (y_coord < 0.) & valid_range
    eastern = (y_coord >= 0.) & valid_range
    
    # First, get the values of the projected longitudes for each citrus boundaries
    proj_longitutes = []
    for phase_i in citrus_phases:
        
        # Initiate with zeros
        p_long_i = np.full_like(y_coord, np.nan)
#         p_long_i = np.where(valid_range, np.nan, 0.)
        
        # Find what is the phase and longitude seen by the observer
        total_phase = (phase_i + phase) % 1
        longitude_pos = total_phase * 2. * np.pi
        # If the projected longitude is visible, then compute the boundary
        if (abs(total_phase) <= 0.25) or 0.75 <= abs(total_phase) < 1.:
            
            # Special case at 0
            if total_phase != 0.:
                proj_y = y_coord / np.sin(longitude_pos)
                sqrt_arg = radius**2 - proj_y**2
                # Valid srqt arg and same y with same sign as sin(longitude_pos)                
                cond = (sqrt_arg >= 0) & (proj_y > 0)
                p_long_i[cond] = np.sqrt(sqrt_arg[cond])
                # Put zeros where sqrt is not valid
                cond = (sqrt_arg < 0) & (proj_y > 0)
                p_long_i[cond] = 0.
            else:
                p_long_i[eastern] = disk_bounds[eastern]
                p_long_i[western] = 0.
                
        proj_longitutes.append(p_long_i)
    

    # Then compute the vertical lengths
    v_lengths = []
    
    # idx = 0
    log.debug(f'total phases: {total_phases}')
    n_bounds = len(total_phases)
    for idx in range(n_bounds):
        reg_1 = f'{idx + 1}'
        reg_2 = f'{(idx - 1) % n_bounds  + 1}'
        if (0 < total_phases[idx] <= 0.25):
            log.debug(f'Bound {reg_1}-{reg_2} is eastern')
            if np.isfinite(proj_longitutes[idx - 1]).any():
                if np.isfinite(proj_longitutes[idx - 1][western]).any():
                    log.debug(f'Bound {reg_2}-{reg_1} is western')
                    bnd12 = np.nanmax([proj_longitutes[idx], zeros], axis=0)
                    bnd12[western] = np.nanmax([proj_longitutes[idx - 1], zeros], axis=0)[western]
                    bnd21 = zeros

                else:
                    log.debug(f'Bound {reg_2}-{reg_1} is eastern')
                    bnd12 = np.nanmax([proj_longitutes[idx], zeros], axis=0)
                    bnd21 = np.nanmax([proj_longitutes[idx - 1], zeros], axis=0)
                
                v_len_0 = bnd12 - bnd21

                # if v_len_0.any() < 0:
                #     v_len_0 = disk_bounds - v_len_0

            else:
                log.debug(f'Bound {reg_2}-{reg_1} is not visible.')
                bnd12 = np.nanmin([proj_longitutes[idx], disk_bounds], axis=0)
                bnd21 = 0.
                v_len_0 = bnd12 - bnd21
            
        elif (0.75 <= total_phases[idx] < 1.):
            log.debug(f'Bound {reg_1}-{reg_2} is western')
            if np.isfinite(proj_longitutes[idx - 1]).any():
                if (0. < total_phases[idx - 1] <= 0.25):
                    log.debug(f'Bound {reg_2}-{reg_1} is eastern')
                    bnd12 = np.nanmax([proj_longitutes[idx], zeros], axis=0)
                    bnd12[eastern] = np.nanmax([proj_longitutes[idx - 1], zeros], axis=0)[eastern]
                    bnd21 = disk_bounds
                elif total_phases[idx - 1] == 0.:
                    log.debug(f'Bound {reg_2}-{reg_1} is centered')
                    bnd12 = np.nanmax([proj_longitutes[idx], zeros], axis=0)
                    bnd21 = zeros
                    log.debug(f'{((bnd12 - bnd21) < 0).any()}')
                else:
                    log.debug(f'Bound {reg_2}-{reg_1} is western')
                    bnd12 = np.nanmax([proj_longitutes[idx], zeros], axis=0)
                    bnd21 = np.nanmax([proj_longitutes[idx - 1], zeros], axis=0)

            else:
                log.debug(f'Bound {reg_2}-{reg_1} is not visible.')
                bnd12 = np.nanmin([proj_longitutes[idx], disk_bounds], axis=0)
                bnd21 = disk_bounds
            v_len_0 = bnd21 - bnd12

        elif total_phases[idx] == 0.:
            log.debug(f'Bound {reg_1}-{reg_2} is centered')
            # Special case at 0
            if np.isfinite(proj_longitutes[idx - 1]).any():
                if np.isfinite(proj_longitutes[idx - 1][eastern]).any():
                    log.debug(f'Bound {reg_2}-{reg_1} is eastern')
                    bnd12 = np.nanmax([proj_longitutes[idx], zeros], axis=0)
                    bnd12[eastern] = np.nanmax([proj_longitutes[idx - 1], zeros], axis=0)[eastern]
                    bnd21 = disk_bounds
                else:
                    log.debug(f'Bound {reg_2}-{reg_1} is western')
                    bnd12 = zeros
                    bnd21 = np.nanmax([proj_longitutes[idx - 1], zeros], axis=0)

            else:
                log.debug(f'Bound {reg_2}-{reg_1} is not visible.')
                bnd12 = np.nanmin([proj_longitutes[idx], disk_bounds], axis=0)
                bnd21 = 0.
            v_len_0 = bnd21 - bnd12

        elif (0.75 < total_phases[idx - 1] < 1.):
            log.debug(f'Bound {reg_1}-{reg_2} is hidden.')
            log.debug(f'Bound {reg_2}-{reg_1} is western')
            bnd12 = disk_bounds
            bnd21 = np.nanmin([proj_longitutes[idx - 1], disk_bounds], axis=0)
            v_len_0 = bnd21 - bnd12

        elif 0. < total_phases[idx - 1] < 0.25:
            log.debug(f'Bound {reg_1}-{reg_2} is hidden.')
            log.debug(f'Bound {reg_2}-{reg_1} is eastern')
            bnd21 = 0.
            bnd12 = np.nanmin([proj_longitutes[idx - 1], disk_bounds], axis=0)
            v_len_0 = bnd21 - bnd12

        elif total_phases[idx - 1] == 0.:  # Special case at 0
            log.debug(f'Bound {reg_1}-{reg_2} is hidden.')
            log.debug(f'Bound {reg_2}-{reg_1} is centered')
            bnd12 = np.nanmax([proj_longitutes[idx - 1], zeros], axis=0)
            bnd21 = zeros
            v_len_0 = bnd12 - bnd21

        elif (0.25 <= total_phases[idx - 1] <= 0.75) and total_phases[idx] < total_phases[idx - 1]:
            log.debug(f'Both bounds are hidden.')
            log.debug(f'But bounds {reg_1}-{reg_2} < {reg_2}-{reg_1} '
                      f'so we are seeing region {reg_1}')
            v_len_0 = disk_bounds
        
        else:
            log.debug(f'Both bounds are hidden.')
            log.debug(f'But bounds {reg_1}-{reg_2} > {reg_2}-{reg_1} '
                      f'so we are not seeing region {reg_1}')
            v_len_0 = zeros.copy()

        if (v_len_0 < 0).any():
            log.debug(f'Negative length detected. Correcting.')
            v_len_0 = disk_bounds - np.abs(v_len_0)

        v_lengths.append(v_len_0)

    return v_lengths


def plot_sphere(phase_limits, colors=None, viewing_phase=0.):
    
    phase_limits = (np.array(phase_limits) % 1)
    
    # Define the sphere's surface
    theta = np.linspace(0, 2.*np.pi, 100)  # longitude
    phi = np.linspace(0, np.pi, 100)  # latitude
    theta, phi = np.meshgrid(theta, phi)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # Define a list of colors for the different regions
    color_map = np.zeros(theta.shape, dtype=float)
    for idx, end in enumerate(phase_limits):
        start = phase_limits[idx - 1]
        start = start * 2 * np.pi
        end = end * 2 * np.pi
        if (start < end):
            cond = (theta >= start) & (theta < end)
        else:
            cond = (theta >= start) | (theta < end)
        color_map[cond] = idx / (len(phase_limits) - 1)
    
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cmap = ListedColormap(colors[:len(phase_limits)])

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cmap(color_map), shade=False)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_aspect('equal')

    # Remove axes
    ax.axis('off')

    # Set the viewing angle to longitude 180
    ax.view_init(azim=viewing_phase * 360, elev=0)
    
    # Add legend
    for idx, end in enumerate(phase_limits):
        start = phase_limits[idx - 1]
        color = colors[idx]
        ax.plot([0], [0], 's', color=color, label=f'{start:0.2}° to {end}°')

    ax.legend(loc='upper right')

    return fig, ax


class CitrusRotationKernel(BaseKerMulti):
    """
    Class to compute the rotation kernel for a planet with a solid rotation
    and a given spectral resolution.

    Parameters
    ----------
    citrus_phases: list of floats
        phases of the citrus boundaries
    pl_rad: scalar astropy quantity, planet radius in m
    omega: scalar astropy quantity, angular frequency (2 pi / period) in rad/s
    resolution: scalar (float or int)
        spectral resolution of the instrument

    Attributes 
    ----------
    res_elem: scalar float
        resolution element in m/s   
    omega: scalar float
        rotation rate in rad/s

    """


    def __init__(self, citrus_phases, pl_rad, omega, resolution):
        '''Rotational kernel for regions separated by constant longitudes (citrus slices)
        at dv constant (constant resolution). Solid rotation is assumed.

        citrus_phases: list of floats
            phases of the citrus boundaries
        pl_rad: scalar astropy quantity, planet radius
        omega: scalar astropy quantity, rotation rate
        resolution: scalar (float or int)
            spectral resolution of the instrument
       
        '''
        # Initiate the base class (resolution is set there)
        super().__init__(resolution)

        # Convert to default units
        pl_rad, omega = convert_default_units([pl_rad, omega], ['m', '1/s'])

        # Save attributes
        self.omega = omega
        self.r_p = pl_rad
        self.citrus_phases = citrus_phases
        self.n_kernel = len(citrus_phases)

    def get_ker(self, phase, n_os=None, pad=7, norm=True):
        """
        Get the kernel for a given phase

        Parameters
        ----------
        phase: scalar float
            phase of the observed object. Between 0 and 1.
        n_os: scalar, oversampling (to sample the kernel)
        pad: scalar
            pad around the kernel in units of resolution
            elements. Values of the pad are set to zero.
        norm: bool
            Wheter to normalize the kernel or not. Default is True.
            The kernel will be normalized so that the sum over all
            citrus regions is 1 (so a kernel for a single region
            will be not necessarily be normalized to 1).

        Returns
        -------
        v_grid: array-like
            velocity grid of the kernel
        citrus_ker: list of array-like
            list of kernels for each citrus region
        """

        # Get some values from the attributes
        res_elem = self.res_elem
        omega = self.omega
        r_p = self.r_p
        citrus_phases = self.citrus_phases

        # Get the kernel half length
        ker_h_len = r_p * omega

        ###############################################
        # Define the velocity grid to sample the kernel
        ###############################################

        # Get the maximum velocity range
        v_max = ker_h_len + pad * res_elem

        # Find adequate sampling if oversampling is not given
        if n_os is None:
            delta_v = np.abs(r_p * omega / 10)
            log.info(f'No oversampling given. Using delta_v = {delta_v:.2f} m/s')
        else:
            delta_v = res_elem / n_os

        # Define the velocity grid
        v_grid = np.arange(-v_max, v_max, delta_v)
        # Centered on zero
        v_grid -= np.mean(v_grid)

        ###############################################
        # Define the kernel
        ###############################################
        # Convert velocity to horizontal coordinate
        x_v = v_grid / omega
        citrus_ker = citrus_to_ker(x_v, citrus_phases, phase, radius=r_p)

        # Make sure the sum over all kernels is 1 (conservation of flux)
        # Also, make sure it is not zeros everywhere (that would mean that
        # the grid sampling was larger than the rotation kernel)
        # In this case, set the value closest to zero to 1 (delta function)
        if norm:
            all_sum = np.sum(citrus_ker)
            if all_sum == 0:
                log.warning('The kernel is zero everywhere. '
                            'The grid sampling is probably too large. Using delta function.')
                idx = np.argmin(np.abs(x_v))
                for ker in citrus_ker:
                    ker[idx] = 1.
                all_sum = np.sum(citrus_ker)
            citrus_ker = [ker / all_sum for ker in citrus_ker]

        return v_grid, citrus_ker
    
    def show(self, norm=True, **kwargs):
        """ Plot the kernel and the degraded kernel
        Parameters
        ----------
        norm: bool
            Wheter to normalize the kernel or not. Default is True.
        kwargs are passed to get_ker method.
        """
        res_elem = self.res_elem
        v_grid, ker_list = self.get_ker(norm=norm, **kwargs)
        gauss_ker = hm.gauss(v_grid, 0.0, FWHM=res_elem)
        if norm:
            gauss_ker /= gauss_ker.sum()
        v_grid_degraded, ker_list_degraded = self.degrade_ker(norm=norm, **kwargs)
        
        fig = plt.figure()
        ax_native = plt.gca()
        ax_degraded = ax_native.twinx()

        ax_degraded.plot(v_grid/1e3, gauss_ker, ":", color="gray",
                 label='Instrumental resolution element')
        ax_native.axvline(res_elem/2e3, linestyle=':', color='gray')
        ax_native.axvline(-res_elem/2e3, linestyle=':', color='gray')
        for idx, ker in enumerate(ker_list):
            ax_native.plot(v_grid/1e3, ker,
                    label=f'Rotation kernel {idx + 1}')
            color = ax_native.lines[-1].get_color()
            ax_degraded.plot(v_grid_degraded/1e3, ker_list_degraded[idx],
                              '--', color=color,
                     label=f'Instrumental * rotation kernel {idx + 1}')

        # Make one combined legend for both axes
        lines_native, labels_native = ax_native.get_legend_handles_labels()
        lines_degraded, labels_degraded = ax_degraded.get_legend_handles_labels()
        ax_native.legend(lines_native + lines_degraded, labels_native + labels_degraded,
                         loc='upper right')

        # Add labels
        plt.xlabel('dv [km/s]')
        ax_native.set_ylabel('Kernel')
        ax_degraded.set_ylabel('Kernel * Instrumental')
        
        # Plot the sphere
        phase_limits = self.citrus_phases
        fig_sphere, _ = plot_sphere(phase_limits, viewing_phase=-kwargs['phase'])

        return fig, fig_sphere


#####################################################
# --- Hotspot + wind rotation kernel (Chantier A) ---
#####################################################
#
# Unlike CitrusRotationKernel above, whose analytic chord-length trick
# (citrus_to_ker) only works because solid-body rotation makes v_los depend
# on longitude alone, a genuine 2D brightness feature (a displaced hotspot)
# or a wind field that depends on latitude (a jet, a day-to-night flow)
# couples both surface coordinates together and leaves no closed-form
# shortcut. The functions below build the kernel numerically instead:
# discretize the visible surface on a grid, evaluate brightness and
# line-of-sight velocity at each point, and histogram the result into a
# velocity kernel -- the same integral citrus_to_ker solves analytically,
# just carried out on a grid. They are deliberately geometry-agnostic
# (bin_weighted_velocities especially makes no assumption about where its
# samples came from), so they are reusable beyond HotspotWindRotationKernel
# below -- e.g. a numerical cross-check of citrus_to_ker itself, or a future
# transit/transmission terminator-ring kernel built the same way (see
# tutorials/rotation_kernel_examples/ for the full derivation, including the
# transit-ring analytic result that inspired this approach).
#
# Sky-plane geometry (unit sphere, edge-on orbit, zero obliquity -- same
# implicit assumption already used by every kernel above):
#
#   phi        : planet-centered latitude (rad), spin axis = pole
#   lam        : planet-centered longitude (rad), increases eastward
#                (lam = 0 is an arbitrary reference meridian -- for the
#                hotspot kernel below, the substellar meridian)
#   lam_obs    : sub-observer longitude, see phase_to_lam_obs
#
#   x = cos(phi) * sin(lam - lam_obs)    (east-west on the sky)
#   y = sin(phi)                          (north-south on the sky)
#   z = cos(phi) * cos(lam - lam_obs)    (toward the observer)
#
# A point is visible when z > 0, and for emission the projected-area
# (Lambertian) weight is simply mu = z (since R = 1).


def phase_to_lam_obs(phase):
    """Convert an orbital phase to the sub-observer longitude `lam_obs`.

    Centralizes the phase convention so every caller (kernel construction,
    plotting) agrees with each other by construction instead of duplicating
    the formula. Phase=0 is mid-transit (nightside facing the observer,
    `lam_obs = +-pi`), phase=0.5 is secondary eclipse (dayside facing the
    observer, `lam_obs = 0`) -- matching `CitrusRotationKernel`,
    `planet_obs.Planet.phase`, and every retrieval yaml in this repo.

    The *sign* of `lam_obs`'s drift with phase (this function's one genuine
    physics content, beyond that phase=0.5 <-> lam_obs=0 alignment) was
    originally guessed wrong. Caught by a direct physical check: for a
    tidally-locked planet on a prograde orbit, a fixed surface feature (e.g.
    a hotspot) must drift toward, and disappear over, the *eastern* limb as
    phase increases past 0.5 -- verified by re-deriving the sub-observer
    longitude from orbital mechanics (a prograde circular orbit, matching
    `orbite.rv_theo_t`'s sign convention: `RV_planet(phase) =
    +K*sin(2*pi*phase)`, redshift positive), which gives
    `lam_obs = -2*pi*(phase - 0.5)`, not `+2*pi*(phase - 0.5)`.

    Parameters
    ----------
    phase : scalar float
        Orbital phase, any real number (wrapped to [0, 1) internally).

    Returns
    -------
    float
        Sub-observer longitude (rad).
    """
    return -2 * np.pi * ((phase - 0.5) % 1.0)


def wrap_to_pi(angle):
    """Wrap an angle (or array of angles) to the range (-pi, pi].

    Longitude differences need this because longitude is periodic: e.g. the
    angular distance between lam=350 deg and lam=10 deg is 20 deg, not 340 deg.

    Parameters
    ----------
    angle : np.ndarray
        Angle(s) in radians, any range.

    Returns
    -------
    np.ndarray
        Equivalent angle(s) wrapped to (-pi, pi].
    """
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


def smooth_threshold(value, width):
    """Smoothed step function: ~1 where `value > 0`, ~0 where `value < 0`.

    A `tanh` transition of scale `width` centered on `value = 0`, i.e. the
    same softening already used for the hotspot ellipse boundary
    (`HotspotWindRotationKernel.hot_region_weight`), factored out here so it
    can be reused for *any* hard latitude cutoff in the wind field (the jet's
    `|phi| < jet_half_lat` and the day-to-night flow's `|phi| > onset`) --
    a plain boolean mask there would give the wind speed itself a
    discontinuity at the cutoff latitude, which (like the hotspot's hard
    edge) introduces ringing into the kernel after convolution with the
    instrumental profile.

    Parameters
    ----------
    value : np.ndarray
        Signed distance from the threshold (positive = inside/active,
        negative = outside/inactive), in the same units as `width`.
    width : float
        Width of the transition. `width -> 0` recovers a hard step.

    Returns
    -------
    np.ndarray
        Smoothed mask, same shape as `value`, values in (0, 1).
    """
    return 0.5 * (1.0 + np.tanh(value / width))


def equal_area_latlon_grid(n_lat, n_lon):
    """Build a latitude/longitude grid over the full sphere with equal-area cells.

    A naive uniform grid in (phi, lam) over-samples the poles (cells shrink
    in physical area as cos(phi) -> 0 near the poles), which would silently
    over-weight polar surface elements in the kernel integral. To avoid that,
    we sample uniformly in sin(phi) instead of phi: since the sphere's area
    element is dOmega = cos(phi) dphi dlam = d(sin phi) dlam, a grid that is
    uniform in (sin(phi), lam) has *exactly* equal cell area, and the solid
    angle per cell is then just a constant -- no cos(phi) weighting needed
    anywhere else in the pipeline.

    Parameters
    ----------
    n_lat : int
        Number of latitude bins (uniform in sin(phi), covering the full
        -pi/2 to pi/2 range).
    n_lon : int
        Number of longitude bins (uniform in lam, covering the full
        -pi to pi range).

    Returns
    -------
    phi : np.ndarray, shape (n_lat, n_lon)
        Latitude of each grid cell center (rad), via meshgrid.
    lam : np.ndarray, shape (n_lat, n_lon)
        Longitude of each grid cell center (rad), via meshgrid.
    d_omega : float
        Solid angle of a single grid cell (sr), same for every cell.
    """
    # Cell edges uniform in sin(phi) in [-1, 1], and cell centers at the
    # midpoint of each edge pair (in sin(phi) space, not in phi space).
    sin_phi_edges = np.linspace(-1.0, 1.0, n_lat + 1)
    sin_phi_centers = 0.5 * (sin_phi_edges[:-1] + sin_phi_edges[1:])
    phi_centers = np.arcsin(sin_phi_centers)

    lon_edges = np.linspace(-np.pi, np.pi, n_lon + 1)
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    phi, lam = np.meshgrid(phi_centers, lon_centers, indexing='ij')

    d_sin_phi = sin_phi_edges[1] - sin_phi_edges[0]
    d_lon = lon_edges[1] - lon_edges[0]
    d_omega = d_sin_phi * d_lon

    return phi, lam, d_omega


def project_to_sky(phi, lam, lam_obs):
    """Project planet-centered spherical coordinates onto the sky plane.

    Parameters
    ----------
    phi : np.ndarray
        Latitude (rad).
    lam : np.ndarray
        Longitude (rad), same shape as `phi`.
    lam_obs : float
        Sub-observer longitude (rad), see `phase_to_lam_obs`.

    Returns
    -------
    x, y, z : np.ndarray
        Sky-plane coordinates on the unit sphere (east-west, north-south,
        toward observer).
    mu : np.ndarray
        Projected-area (Lambertian) weight, mu = z (>= 0 on the visible
        hemisphere, meaningless -- and masked out by `visible` -- on the far
        side).
    visible : np.ndarray of bool
        True where the surface element faces the observer (z > 0).
    """
    dlam = lam - lam_obs
    x = np.cos(phi) * np.sin(dlam)
    y = np.sin(phi)
    z = np.cos(phi) * np.cos(dlam)
    mu = z
    visible = z > 0
    return x, y, z, mu, visible


def los_velocity_from_zonal(v_zonal, phi, lam, lam_obs):
    """Project a purely zonal (east-west tangential) velocity field to the line of sight.

    For a velocity vector tangent to circles of latitude (the "zonal"
    direction -- no north-south component, so this cannot break north-south
    symmetry by itself), the line-of-sight projection collapses to a clean
    1D formula independent of latitude except through `v_zonal` itself:

        v_los(phi, lam) = v_zonal(phi, lam) * sin(lam - lam_obs)

    Sign convention: **redshift-positive** (matching `orbite.rv_theo_t`'s
    `RV_planet(phase) = +K*sin(2*pi*phase)`, the convention used for Kp/Vsys
    throughout this repo) -- verified against a direct physical check
    (`phase_to_lam_obs`'s docstring): a prograde-rotating hotspot must drift
    toward, and disappear over, the eastern limb as phase increases past
    0.5, and a day-to-night flow (moving from the visible dayside into the
    hidden nightside) must be redshifted (receding) when viewed at secondary
    eclipse (`los_velocity_from_day_night`'s docstring) -- neither held with
    an earlier, un-derived guess at this sign.

    Parameters
    ----------
    v_zonal : np.ndarray
        Local eastward tangential speed (m/s) at each grid point. Can
        already include solid rotation, a jet, a day-to-night term, etc. --
        this function only cares about the total zonal speed.
    phi : np.ndarray
        Latitude (rad), same shape as `v_zonal` (unused directly here, kept
        for a consistent call signature -- the latitude dependence already
        lives inside `v_zonal`).
    lam : np.ndarray
        Longitude (rad), same shape as `v_zonal`.
    lam_obs : float
        Sub-observer longitude (rad).

    Returns
    -------
    np.ndarray
        Line-of-sight velocity (m/s), same shape as `v_zonal`.
    """
    return v_zonal * np.sin(lam - lam_obs)


def los_velocity_from_day_night(speed, phi, lam, lam_obs):
    """Project a "radially outward from the substellar point" velocity field to the line of sight.

    A day-to-night flow's most direct path is *not* tangent to circles of
    latitude (that is only true exactly on the equator) -- it follows the
    great circle through the substellar point (lam=0, phi=0) and the local
    grid point, pointing away from the substellar point and toward the
    antistellar point. Seen face-on at secondary eclipse, this is exactly
    the "center-to-limb, in every direction" picture: radiating outward from
    the disk center (the substellar point) toward the edge.

    Deriving the local unit vector for that direction (call it `e_psi`, the
    direction of increasing angular distance `psi` from the substellar
    point) and projecting it onto the line of sight the same way
    `los_velocity_from_zonal` does for `e_lambda` gives, after simplifying:

        cos(psi) = cos(phi) * cos(lam)                    (angular distance from substellar point)
        v_los(phi, lam) = -speed(phi, lam) *
            (cos(psi) * cos(phi) * cos(lam - lam_obs) - cos(lam_obs)) / sin(psi)

    Sign convention: **redshift-positive**, same as `los_velocity_from_zonal`
    (see that function's docstring) -- checked directly here too: at
    secondary eclipse (`lam_obs=0`), this reduces to `+speed * sin(psi)`,
    always >= 0 (redshifted), matching the physical expectation that gas
    flowing from the visible dayside into the hidden nightside is receding
    from the observer, and growing from 0 at the disk center (`psi=0`, the
    substellar point) to its largest values at the limb (`psi` approaching
    90 deg) -- not a spatially uniform shift.

    Consistency check: exactly on the equator (phi=0), this reduces to
    `sign(sin(lam)) * los_velocity_from_zonal(speed, 0, lam, lam_obs)` --
    i.e. proportional to `los_velocity_from_zonal`'s own `e_lambda`
    projection, with the sign flip at lam=0 an earlier version of this
    function applied everywhere (not just on the equator, which was the bug
    this version fixes: away from the equator, a point at high latitude is
    *not* reached most directly by moving along its own line of latitude).

    Parameters
    ----------
    speed : np.ndarray
        Local day-to-night flow speed (m/s, always >= 0 -- the outward
        direction is already encoded in the projection itself, so this does
        not need an explicit sign flip from the caller). Typically
        `day_night_speed * mask(|phi| > onset)`.
    phi, lam : np.ndarray
        Latitude/longitude (rad), same shape as `speed`.
    lam_obs : float
        Sub-observer longitude (rad).

    Returns
    -------
    np.ndarray
        Line-of-sight velocity (m/s), same shape as `speed`.
    """
    cos_psi = np.cos(phi) * np.cos(lam)
    # Clip away from exactly 0 to avoid a 0/0 division right at the
    # substellar/antistellar points -- those two points are single grid
    # cells at most, and both sit at phi=0, so any onset latitude > 0
    # already gives them zero weight through `speed`; the clip just keeps
    # the arithmetic finite there instead of producing a stray NaN that
    # `speed * finite_but_wrong_value` would not otherwise clean up.
    sin_psi = np.sqrt(np.clip(1 - cos_psi**2, 1e-12, None))
    dlam = lam - lam_obs
    e_psi_dot_zobs = (cos_psi * np.cos(phi) * np.cos(dlam) - np.cos(lam_obs)) / sin_psi
    return -speed * e_psi_dot_zobs


def make_velocity_grid(v_max, res_elem, n_os=None, pad=7.0):
    """Build the centered velocity grid a kernel is sampled on.

    Mirrors the convention already used throughout this module
    (`SolidRotationKernel.get_ker`, `CitrusRotationKernel.get_ker`, ...): pad
    the grid by a few resolution elements beyond the kernel's physical
    extent, and center it exactly on v=0 (`np.arange` does not guarantee
    that on its own).

    Parameters
    ----------
    v_max : float
        Largest line-of-sight speed (m/s) the kernel can physically reach
        (a safe upper bound is enough, it only sets the grid extent).
    res_elem : float
        Instrumental resolution element (m/s).
    n_os : float, optional
        Oversampling factor relative to `res_elem`. If None, a default
        sampling of `res_elem / 10` is used.
    pad : float
        Extra padding around the kernel, in units of `res_elem`.

    Returns
    -------
    np.ndarray
        Centered velocity grid (m/s).
    """
    v_edge = v_max + pad * res_elem
    delta_v = res_elem / 10 if n_os is None else res_elem / n_os
    v_grid = np.arange(-v_edge, v_edge, delta_v)
    v_grid -= np.mean(v_grid)
    return v_grid


def bin_weighted_velocities(v_los, weight, v_grid, norm=True):
    """Turn a cloud of (line-of-sight velocity, weight) samples into a kernel.

    This is the fully generic core of this whole section: it makes no
    assumption about where `v_los` and `weight` came from -- a lat/lon grid
    over a disk, a ring of points around a transit terminator, an
    irregularly-sampled mesh, anything. It is the discrete equivalent of the
    brightness-weighted-velocity-distribution integral every rotation kernel
    in this module ultimately computes: each grid cell of surface weight
    `weight[i]` contributes a Dirac spike at `v_los[i]`, and summing (binning)
    those spikes over many cells approximates the smooth kernel.

    Cells outside the visible hemisphere (or otherwise excluded) should
    simply carry `weight = 0` -- they still contribute a (harmless) empty
    bin count.

    Parameters
    ----------
    v_los : np.ndarray
        Line-of-sight velocity samples (m/s), any shape.
    weight : np.ndarray
        Weight (brightness x projected area x solid angle, or any other
        relevant measure) of each sample, same shape as `v_los`.
    v_grid : np.ndarray
        Uniformly-spaced, centered velocity grid to bin onto (as returned by
        `make_velocity_grid`).
    norm : bool
        Whether to normalize the resulting kernel to unit sum. Default True.

    Returns
    -------
    np.ndarray
        Kernel evaluated on `v_grid` (same length).
    """
    # Build bin edges from the (assumed uniform) grid spacing, so that
    # v_grid[i] sits at the center of its own bin.
    dv = np.diff(v_grid).mean()
    edges = np.concatenate([v_grid - dv / 2, [v_grid[-1] + dv / 2]])

    kernel, _ = np.histogram(v_los.ravel(), bins=edges, weights=weight.ravel())

    if norm:
        total = kernel.sum()
        if total == 0:
            # Grid resolution too coarse compared to v_grid, or an
            # entirely-invisible/zero-weight configuration: fall back to a
            # delta function at v=0 rather than dividing by zero, matching
            # the behaviour CitrusRotationKernel.get_ker already has for
            # the analogous degenerate case.
            idx = np.argmin(np.abs(v_grid))
            kernel[idx] = 1.0
        else:
            kernel = kernel / total

    return kernel


def plot_sky_view(ax, phi, lam, color, phase, cmap='inferno', vmin=None, vmax=None):
    """Plot a color map as seen by the observer at a given orbital phase.

    Only the visible hemisphere is drawn (the far side is left blank), using
    the exact same sky-plane projection (`project_to_sky`) the kernel itself
    integrates over -- so this is literally "what the kernel calculation
    sees" at that phase, not a separate/approximate rendering of it. The
    generic primitive `HotspotWindRotationKernel.show()`'s panels are built
    from (mirrors `plot_sphere`'s role for `CitrusRotationKernel.show()`).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.
    phi, lam : np.ndarray
        Surface grid (rad), e.g. `kernel.phi`, `kernel.lam`.
    color : np.ndarray
        Value to color-map at each grid point (e.g. a hot/cold weight, or a
        line-of-sight velocity), same shape as `phi`.
    phase : scalar float
        Orbital phase (0 = transit, 0.5 = secondary eclipse -- see
        `phase_to_lam_obs`).
    cmap, vmin, vmax :
        Forwarded to `pcolormesh`.

    Returns
    -------
    matplotlib.collections.QuadMesh
        The plotted mesh (handy for attaching a shared colorbar).
    """
    lam_obs = phase_to_lam_obs(phase)
    x, y, z, mu, visible = project_to_sky(phi, lam, lam_obs)

    # NaN out the far side so pcolormesh leaves it blank instead of drawing
    # it (it would otherwise draw at whatever `color` value sits there, even
    # though those grid points are physically not facing the observer).
    plotted = np.where(visible, color, np.nan)
    mesh = ax.pcolormesh(x, y, plotted, shading='nearest', cmap=cmap, vmin=vmin, vmax=vmax)

    # Disk outline, for a clean "planet" look even where color is NaN right
    # at the limb (grid resolution can leave a ragged edge otherwise).
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color='0.3', lw=0.8))
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'phase = {phase:.2f}', fontsize=9)

    return mesh


class HotspotWindRotationKernel(BaseKerMulti):
    """Rotation kernel for a planet with an elliptical hotspot and a 3-component wind field.

    Physical picture
    -----------------
    The planet's disk is split into two brightness regions:

    - A "hot" region: an ellipse in (longitude, latitude) space, centered on
      the substellar meridian by default but allowed an east-west (and, if
      really needed, north-south) offset -- this is the classic
      eastward-shifted hotspot seen on many hot Jupiters.
    - A "cold" region: everything else.

    Both regions share the same surface wind field, the sum of three
    components:

    1. Solid-body rotation, rate `omega_rot` (typically the tidally-locked
       value, 2*pi / orbital period) -- *zonal* (tangent to circles of
       latitude around the spin axis).
    2. A superrotating equatorial jet: an extra `jet_delta_omega` confined
       to `|latitude| < jet_half_lat` -- also zonal.
    3. A day-to-night flow: a constant speed `day_night_speed` confined to
       `|latitude| > day_night_lat_onset`, directed *away from the
       substellar point along the great circle through it* (a toy model of
       the day-to-night overturning circulation -- see
       tutorials/rotation_kernel_examples/ for the full derivation). This is
       **not** zonal -- seen face-on at secondary eclipse it radiates
       outward from the disk center toward the limb in every direction,
       only coinciding with the zonal (east-west) direction exactly on the
       equator. It needs its own line-of-sight projection
       (`los_velocity_from_day_night`), separate from the other two (see
       `zonal_wind_field` and `day_night_speed_field`).

    None of the three components has a north-south component of its own
    (each individually preserves north-south symmetry), but only the first
    two are tangent to latitude circles -- worth keeping straight since it
    is easy to assume "no north-south component" implies "zonal", which is
    not the same thing away from the equator.

    This class returns *two* kernels (hot region, cold region), each
    carrying only the *geometric* (visibility x projected-area x
    solid-angle) weight of its region -- exactly like `CitrusRotationKernel`
    returns one kernel per longitude slice. The actual brightness/
    temperature contrast between the hot and cold regions is deliberately
    *not* baked in here: it belongs downstream, as the per-region spectrum
    weight (`theta_dict['spec_scale']` in
    `model_sequence.combine_regions_with_kernel`), the same way the citrus
    kernel is already used. Keeping that split makes this drop-in compatible
    with the existing multi-region retrieval pipeline.

    Because the hotspot is a genuine 2D brightness feature and the
    jet/day-night masks depend on latitude, there is no closed-form
    shortcut left (unlike `CitrusRotationKernel`) -- the kernel is built
    numerically, via the generic grid + histogram machinery earlier in this
    section (`equal_area_latlon_grid`, `bin_weighted_velocities`, ...).

    See `tutorials/rotation_kernel_examples/` for the full theoretical
    derivation and worked examples (this class started life there as
    `starships_analysis/hotspot_wind_kernel/`, folded in once validated).

    Parameters
    ----------
    pl_rad : scalar astropy quantity
        Planet radius.
    omega_rot : scalar astropy quantity
        Solid-body rotation rate (rad/s) -- the tidally-locked value is
        `2 * pi * u.rad / P_orb` for most hot Jupiters.
    resolution : scalar (float or int)
        Spectral resolution of the instrument.
    hotspot_lon : scalar astropy quantity (angle)
        East-west offset of the hotspot center from the substellar meridian
        (positive = eastward, the usual advected-hotspot direction).
    hotspot_half_lon : scalar astropy quantity (angle)
        Half-width of the hotspot ellipse in longitude.
    hotspot_half_lat : scalar astropy quantity (angle)
        Half-width of the hotspot ellipse in latitude.
    hotspot_lat : scalar astropy quantity (angle), optional
        North-south offset of the hotspot center. Default 0 deg (no N-S
        symmetry breaking) -- included for completeness, but a nonzero value
        should be a deliberate choice, not the default assumption.
    hotspot_edge_width : float, optional
        Softness of the hotspot boundary, in units of the ellipse's own
        normalized radius (0 = infinitely sharp edge, ~0.1-0.3 = a gradual
        transition). A soft edge avoids the ringing a hard-edged brightness
        step would otherwise introduce after convolution with the
        instrumental profile. Default 0.2.
    jet_delta_omega : scalar astropy quantity, optional
        Extra angular rotation rate of the equatorial superrotating jet, on
        top of `omega_rot`. Default 0 (no jet).
    jet_half_lat : scalar astropy quantity (angle), optional
        Latitude half-extent of the jet (active for `|latitude| < jet_half_lat`).
        Default 0 deg, which disables the jet regardless of `jet_delta_omega`
        (an empty latitude band).
    day_night_speed : scalar astropy quantity, optional
        Constant speed of the day-to-night flow, directed away from the
        substellar *point* along the great circle through it (not along a
        line of latitude -- see the class docstring). Default 0 (no flow).
    day_night_lat_onset : scalar astropy quantity (angle), optional
        Latitude beyond which the day-to-night flow is active
        (`|latitude| > day_night_lat_onset`). Default 90 deg, which disables
        the flow regardless of `day_night_speed` (no latitude satisfies it).
    wind_mask_edge_width : scalar astropy quantity (angle), optional
        Softness of the jet's and day-to-night flow's latitude cutoffs
        (`jet_half_lat`, `day_night_lat_onset`) -- the same idea as
        `hotspot_edge_width`, but for the wind field's latitude masks rather
        than the brightness map: a hard cutoff would give the wind *speed*
        itself a discontinuity at that latitude, which introduces ringing
        into the kernel the same way a hard-edged hotspot would. Applied via
        a single shared width (not a separate one per mask) since both are
        the same kind of feature -- a latitude threshold -- and there is no
        reason to expect them to need different smoothing scales. Does not
        affect whether a component is disabled: `jet_delta_omega=0` or
        `day_night_speed=0` (the defaults) already zero out that component's
        contribution regardless of how soft its mask is. Default 3 deg.
    n_lat, n_lon : int, optional
        Resolution of the equal-area surface grid the kernel is numerically
        integrated on. Defaults (181, 361) are generous for a smooth kernel;
        lower them for quick exploratory plots.

    Attributes
    ----------
    res_elem : float
        Instrumental resolution element (m/s), set by `BaseKerMulti.__init__`.
    """

    def __init__(self, pl_rad, omega_rot, resolution, *,
                 hotspot_lon, hotspot_half_lon, hotspot_half_lat,
                 hotspot_lat=0 * u.deg, hotspot_edge_width=0.2,
                 jet_delta_omega=0 / u.s, jet_half_lat=0 * u.deg,
                 day_night_speed=0 * u.m / u.s, day_night_lat_onset=90 * u.deg,
                 wind_mask_edge_width=3 * u.deg,
                 n_lat=181, n_lon=361):
        super().__init__(resolution)

        (pl_rad, omega_rot,
         hotspot_lon, hotspot_lat, hotspot_half_lon, hotspot_half_lat,
         jet_delta_omega, jet_half_lat,
         day_night_speed, day_night_lat_onset, wind_mask_edge_width) = convert_default_units(
            [pl_rad, omega_rot,
             hotspot_lon, hotspot_lat, hotspot_half_lon, hotspot_half_lat,
             jet_delta_omega, jet_half_lat,
             day_night_speed, day_night_lat_onset, wind_mask_edge_width],
            ['m', '1/s', 'rad', 'rad', 'rad', 'rad', '1/s', 'rad', 'm/s', 'rad', 'rad'])

        self.r_p = pl_rad
        self.omega_rot = omega_rot
        self.hotspot_lon = hotspot_lon
        self.hotspot_lat = hotspot_lat
        self.hotspot_half_lon = hotspot_half_lon
        self.hotspot_half_lat = hotspot_half_lat
        self.hotspot_edge_width = hotspot_edge_width
        self.jet_delta_omega = jet_delta_omega
        self.jet_half_lat = jet_half_lat
        self.day_night_speed = day_night_speed
        self.day_night_lat_onset = day_night_lat_onset
        self.wind_mask_edge_width = wind_mask_edge_width

        # The surface grid does not depend on orbital phase, so it is built
        # once here and reused by every `get_ker` call.
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.phi, self.lam, self.d_omega = equal_area_latlon_grid(n_lat, n_lon)

    def zonal_wind_field(self, phi, lam):
        """Total eastward (zonal) wind speed at each grid point, in the planet frame.

        Solid-body rotation plus the superrotating jet -- both genuinely
        *zonal* (tangent to circles of latitude around the spin axis). The
        day-to-night flow is *not* zonal (see `day_night_speed_field` and
        `los_velocity_from_day_night`): its most direct path from the
        substellar to the antistellar point only coincides with a line of
        latitude on the equator, so it needs its own projection rather than
        being folded into this sum.

        Both masks below are smoothed (not boolean) latitude thresholds --
        see `wind_mask_edge_width` in the class docstring -- but a disabled
        component (default parameters, `jet_delta_omega=0`) still
        contributes exactly zero regardless of the mask's value, since the
        mask only ever multiplies an already-zero speed.

        Parameters
        ----------
        phi : np.ndarray
            Latitude (rad).
        lam : np.ndarray
            Longitude (rad), lam=0 at the substellar meridian, same shape as `phi`.

        Returns
        -------
        np.ndarray
            Eastward wind speed (m/s), same shape as `phi`.
        """
        return self.solid_rotation_field(phi) + self.jet_field(phi)

    def solid_rotation_field(self, phi):
        """Solid-body rotation speed (m/s, eastward) at each grid point.

        Tangential speed shrinks toward the poles as the lever arm
        `R*cos(phi)` shrinks. Split out from `zonal_wind_field` as its own
        method so a diagnostic plot can show this component on its own
        (`plot_velocity_components`).

        Parameters
        ----------
        phi : np.ndarray
            Latitude (rad).

        Returns
        -------
        np.ndarray
            Eastward speed (m/s), same shape as `phi`.
        """
        return self.omega_rot * self.r_p * np.cos(phi)

    def jet_field(self, phi):
        """Superrotating-jet extra eastward speed (m/s) at each grid point.

        Same `cos(phi)` lever arm as solid rotation, confined to a latitude
        band around the equator (smoothed cutoff at `+-jet_half_lat`, see
        `wind_mask_edge_width`). Split out from `zonal_wind_field` as its own
        method so a diagnostic plot can show this component on its own
        (`plot_velocity_components`).

        Parameters
        ----------
        phi : np.ndarray
            Latitude (rad).

        Returns
        -------
        np.ndarray
            Eastward speed (m/s), same shape as `phi`.
        """
        jet_mask = smooth_threshold(self.jet_half_lat - np.abs(phi), self.wind_mask_edge_width)
        return self.jet_delta_omega * self.r_p * np.cos(phi) * jet_mask

    def day_night_speed_field(self, phi):
        """Day-to-night flow speed at each grid point (always >= 0), confined to high latitude.

        Unlike `zonal_wind_field`, this is a *speed*, not a vector component
        -- the outward-from-substellar *direction* is applied separately by
        `los_velocity_from_day_night`, which is why there is no `sign(lam)`
        here anymore (that used to fake the "away from substellar on both
        flanks" behavior using the zonal direction, which only happens to be
        correct exactly on the equator).

        Parameters
        ----------
        phi : np.ndarray
            Latitude (rad).

        Returns
        -------
        np.ndarray
            Day-to-night flow speed (m/s), same shape as `phi`.
        """
        # Smoothed cutoff at +-day_night_lat_onset, see wind_mask_edge_width.
        day_night_mask = smooth_threshold(np.abs(phi) - self.day_night_lat_onset,
                                           self.wind_mask_edge_width)
        return self.day_night_speed * day_night_mask

    def los_velocity_components(self, phase):
        """Line-of-sight velocity, split into its physical components, at a given phase.

        Same total as `get_ker` computes internally, just not summed --
        meant for diagnostic plots (`plot_velocity_components`) that show
        each wind mechanism's own contribution to what the observer sees,
        rather than only the combined result.

        Parameters
        ----------
        phase : scalar float
            Orbital phase (see `get_ker`'s docstring for the convention).

        Returns
        -------
        dict of np.ndarray
            Keys `'solid_rotation'`, `'jet'`, `'day_night'`, `'total'` (the
            exact sum of the first three) -- each an array of LOS velocities
            (m/s) on the kernel's own `(phi, lam)` grid.
        """
        lam_obs = phase_to_lam_obs(phase)
        phi, lam = self.phi, self.lam

        los_solid = los_velocity_from_zonal(self.solid_rotation_field(phi), phi, lam, lam_obs)
        los_jet = los_velocity_from_zonal(self.jet_field(phi), phi, lam, lam_obs)
        los_day_night = los_velocity_from_day_night(self.day_night_speed_field(phi),
                                                      phi, lam, lam_obs)

        return {
            'solid_rotation': los_solid,
            'jet': los_jet,
            'day_night': los_day_night,
            'total': los_solid + los_jet + los_day_night,
        }

    def hot_region_weight(self, phi, lam):
        """Fraction of the local brightness attributed to the hot region, in [0, 1].

        A smoothed indicator of the hotspot ellipse: `equal to 1` well inside
        the ellipse, `equal to 0` well outside, with a `tanh` transition of
        relative width `hotspot_edge_width` around the ellipse boundary
        (normalized elliptical radius = 1).

        Parameters
        ----------
        phi : np.ndarray
            Latitude (rad).
        lam : np.ndarray
            Longitude (rad), same shape as `phi`.

        Returns
        -------
        np.ndarray
            Hot-region weight, same shape as `phi`. `1 - this` is the
            cold-region weight (the two are a strict partition of unity).
        """
        # wrap_to_pi handles the case where the hotspot straddles the +-pi
        # longitude branch cut.
        dlam = wrap_to_pi(lam - self.hotspot_lon)
        dphi = phi - self.hotspot_lat
        ellipse_radius = np.sqrt((dlam / self.hotspot_half_lon) ** 2
                                  + (dphi / self.hotspot_half_lat) ** 2)
        # Same smoothed-threshold helper the wind masks use (see
        # wind_mask_edge_width): active (~1) where ellipse_radius < 1.
        return smooth_threshold(1.0 - ellipse_radius, self.hotspot_edge_width)

    def get_ker(self, phase, n_os=None, pad=7, norm=True):
        """Get the [hot region, cold region] kernels for a given orbital phase.

        Parameters
        ----------
        phase : scalar float
            Orbital phase, between 0 and 1. Standard convention (matching
            `CitrusRotationKernel`, `Planet.phase`, and every retrieval yaml
            in this repo): phase=0 is mid-transit (nightside facing the
            observer), phase=0.5 is secondary eclipse (dayside facing the
            observer).
        n_os : scalar, optional
            Oversampling of the velocity grid (see `make_velocity_grid`).
        pad : scalar
            Padding around the kernel, in units of resolution elements.
        norm : bool
            Whether to normalize so the two kernels sum to 1 (flux
            conservation across regions), matching `CitrusRotationKernel`'s
            convention. Default True.

        Returns
        -------
        v_grid : np.ndarray
        ker_list : list of two np.ndarray
            `[kernel_hot, kernel_cold]`.
        """
        lam_obs = phase_to_lam_obs(phase)
        phi, lam, d_omega = self.phi, self.lam, self.d_omega

        _, _, _, mu, visible = project_to_sky(phi, lam, lam_obs)

        v_los = self.los_velocity_components(phase)['total']

        w_hot = self.hot_region_weight(phi, lam)

        # Purely geometric weight (visibility x projected area x solid
        # angle), split between the two regions -- no brightness contrast
        # baked in here, see the class docstring.
        base_weight = np.where(visible, mu, 0.0) * d_omega
        weight_hot = base_weight * w_hot
        weight_cold = base_weight * (1.0 - w_hot)

        # Safe upper bound on the largest line-of-sight speed reachable by
        # any combination of the three wind components -- used only to size
        # the velocity grid, does not need to be tight.
        v_max = ((abs(self.omega_rot) + abs(self.jet_delta_omega)) * self.r_p
                  + abs(self.day_night_speed))
        v_grid = make_velocity_grid(v_max, self.res_elem, n_os=n_os, pad=pad)

        kernel_hot = bin_weighted_velocities(v_los, weight_hot, v_grid, norm=False)
        kernel_cold = bin_weighted_velocities(v_los, weight_cold, v_grid, norm=False)

        if norm:
            total = kernel_hot.sum() + kernel_cold.sum()
            if total == 0:
                # Grid too coarse relative to v_grid (or a degenerate,
                # entirely-invisible configuration): fall back to a delta
                # function at v=0, same convention as CitrusRotationKernel.
                idx = np.argmin(np.abs(v_grid))
                kernel_hot = np.zeros_like(v_grid)
                kernel_hot[idx] = 1.0
                kernel_cold = np.zeros_like(v_grid)
            else:
                kernel_hot = kernel_hot / total
                kernel_cold = kernel_cold / total

        return v_grid, [kernel_hot, kernel_cold]

    def brightness_map(self):
        """Convenience helper: the full-sphere hot/cold weight map, for plotting.

        Not used by `get_ker` itself (which works cell-by-cell on the same
        grid); exposed separately for diagnostic plots (`show`,
        `plot_regions`). The hotspot is fixed in the planet frame, so this
        does not depend on orbital phase.

        Returns
        -------
        phi, lam : np.ndarray
            The surface grid (rad).
        w_hot : np.ndarray
            Hot-region weight at each grid point, in [0, 1].
        """
        return self.phi, self.lam, self.hot_region_weight(self.phi, self.lam)

    def plot_regions(self, phase, ax=None):
        """Sky view of the hot/cold region split at a given phase.

        Parameters
        ----------
        phase : scalar float
        ax : matplotlib.axes.Axes, optional
            Drawn into a new figure/axes if not given.

        Returns
        -------
        matplotlib.collections.QuadMesh
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(4, 4))
        phi, lam, w_hot = self.brightness_map()
        mesh = plot_sky_view(ax, phi, lam, w_hot, phase, cmap='inferno', vmin=0, vmax=1)
        ax.set_title(f'Regions (1=hot, 0=cold)\nphase={phase:.3f}', fontsize=9)
        return mesh

    def plot_los_velocity(self, phase, ax=None, vmax=None):
        """Sky view of the total line-of-sight velocity at a given phase.

        Parameters
        ----------
        phase : scalar float
        ax : matplotlib.axes.Axes, optional
            Drawn into a new figure/axes if not given.
        vmax : float, optional
            Color scale half-range (km/s), symmetric around 0. Defaults to
            this map's own peak magnitude -- pass one explicitly to compare
            several panels (e.g. `plot_velocity_components`'s) on the same
            scale.

        Returns
        -------
        matplotlib.collections.QuadMesh
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(4, 4))
        v_los = self.los_velocity_components(phase)['total'] / 1e3
        if vmax is None:
            vmax = np.nanmax(np.abs(v_los))
        mesh = plot_sky_view(ax, self.phi, self.lam, v_los, phase,
                              cmap='coolwarm', vmin=-vmax, vmax=vmax)
        ax.set_title(f'LOS velocity [km/s]\nphase={phase:.3f}', fontsize=9)
        return mesh

    def plot_velocity_components(self, phase, axes=None):
        """Sky view of each wind mechanism's own LOS velocity, side by side.

        All three panels share one color scale (the largest magnitude among
        the three), so their relative importance is directly comparable -- a
        component disabled by its own default parameters just plots as a
        uniformly blank disk rather than a mismatched, misleadingly
        "significant looking" color range of its own.

        Parameters
        ----------
        phase : scalar float
        axes : sequence of 3 matplotlib.axes.Axes, optional
            Drawn into a new figure/axes if not given.

        Returns
        -------
        matplotlib.collections.QuadMesh
            The last panel's mesh (handy for attaching one shared colorbar).
        """
        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        components = self.los_velocity_components(phase)
        labels = ['solid_rotation', 'jet', 'day_night']
        titles = ['Solid rotation', 'Jet', 'Day-to-night']
        vmax = max(np.nanmax(np.abs(components[label])) for label in labels) / 1e3
        vmax = max(vmax, 1e-9)  # avoid a degenerate vmin=vmax=0 if all three are disabled
        mesh = None
        for ax, label, title in zip(axes, labels, titles):
            mesh = plot_sky_view(ax, self.phi, self.lam, components[label] / 1e3, phase,
                                  cmap='coolwarm', vmin=-vmax, vmax=vmax)
            ax.set_title(title, fontsize=9)
        return mesh

    def plot_kernel(self, phase, ax=None, n_os=5, fwhm=None):
        """Native and instrument-degraded kernel, hot and cold regions, at a given phase.

        Parameters
        ----------
        phase : scalar float
        ax : matplotlib.axes.Axes, optional
            Drawn into a new figure/axes if not given.
        n_os : scalar, optional
            Forwarded to `get_ker`/`degrade_ker` (velocity-grid oversampling).
        fwhm : float, optional
            Forwarded to `degrade_ker` (instrumental FWHM). Defaults to the
            kernel's own resolution element if not given.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        v_grid, (ker_hot, ker_cold) = self.get_ker(phase=phase, n_os=n_os)
        v_grid_d, (ker_hot_d, ker_cold_d) = self.degrade_ker(phase=phase, n_os=n_os, fwhm=fwhm)
        ax.plot(v_grid / 1e3, ker_hot, color='C1', label='hot (native)')
        ax.plot(v_grid / 1e3, ker_cold, color='C0', label='cold (native)')
        ax.plot(v_grid_d / 1e3, ker_hot_d, '--', color='C1', label='hot (degraded)')
        ax.plot(v_grid_d / 1e3, ker_cold_d, '--', color='C0', label='cold (degraded)')
        ax.set_xlabel('v [km/s]')
        ax.set_ylabel('Kernel')
        ax.set_title(f'Native vs. instrument-degraded kernel\nphase={phase:.3f}', fontsize=9)
        ax.legend(fontsize=7)
        return ax

    def show(self, phase, n_os=5, fwhm=None):
        """All diagnostic panels for this kernel at one phase, in a single figure.

        The `HotspotWindRotationKernel` equivalent of
        `CitrusRotationKernel.show()`: regions, LOS velocity, the three wind
        components separately, and the native vs. instrument-degraded
        kernel, all in one call instead of writing each panel by hand.

        Parameters
        ----------
        phase : scalar float
        n_os, fwhm :
            Forwarded to `plot_kernel`.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=(13, 10))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.8], hspace=0.55, wspace=0.3)

        ax_regions = fig.add_subplot(gs[0, 0])
        self.plot_regions(phase, ax=ax_regions)

        ax_vlos = fig.add_subplot(gs[0, 1])
        mesh_vlos = self.plot_los_velocity(phase, ax=ax_vlos)
        fig.colorbar(mesh_vlos, ax=ax_vlos, shrink=0.8, label='km/s')

        axes_components = [fig.add_subplot(gs[1, i]) for i in range(3)]
        mesh_components = self.plot_velocity_components(phase, axes=axes_components)
        fig.colorbar(mesh_components, ax=axes_components, shrink=0.8, label='km/s')

        ax_kernel = fig.add_subplot(gs[2, :])
        self.plot_kernel(phase, ax=ax_kernel, n_os=n_os, fwhm=fwhm)

        fig.suptitle(f'HotspotWindRotationKernel diagnostics — phase={phase:.3f}')
        return fig


#####################################################
# --- Other section (not sure yet how to call it) ---
#####################################################


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
#                     resol=70000, boost=1., level=0., visit=None):

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


def calc_bin_edges(visit, pad=0):
    
    wv_borders_ord = []
    wv_ord = []
    for iOrd in range(visit.nord):
        wv = visit.wv[iOrd]
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
    visit.wv_bins = wv_borders_ord
    visit.wv_ext = wv_ord


from scipy import stats
def binning_model(wave0, model0, wv_borders):
    binned_mod, _, _ = stats.binned_statistic(wave0, model0, 'median', bins=wv_borders)
    return binned_mod


def quick_inject_clean(wave, flux, P_x, P_y, dv_pl, sep, R_star, A_star,
                  R0=None, RV=0, dv_star=0,
                       alpha=None,  kind_trans='transmission'):
    """
    alpha: 
        Fraction of planetary signal. Depends on `kind_trans`:
        - If 'transmission': fraction of the stellar disk hidden by the planet
        - If 'emission': fraction of the planetary disk not hidden by the star
    """

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
                 boost=1., level=0., resol=70000, R0=None,
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
                 boost=1., level=0., resol=70000, R0=None,
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
