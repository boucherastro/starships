# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 14:25:57 2016

@author: AnneBoucher

"""
#import time
import numpy as np
import time
import itertools
from astropy.io import fits
from astropy.table import Table
from astropy import constants as const
from astropy import units as u
import scipy.constants as cst
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d
from scipy.sparse import find, diags, csr_matrix
import matplotlib.pyplot as plt
import logging
import matplotlib as mpl

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from .mask_tools import interp1d_masked
interp1d_masked.iprint = False

# def idx2(array,value):
#    time1 = time.time()
#    indices = []
#    idx = -1
#    while True:
#        try:
#            idx = array.index(value, idx+1)
#            indices.append(idx)
#        except ValueError:
#            break
#    time2 = time.time()
#    print(time2-time1)
#    return indices

# ----------------------------------------------------------------


def gauss(x, mean=0, sigma=1, FWHM=None):
       
    if FWHM is not None:
        sigma = fwhm2sigma(FWHM)  # FWHM / (2 * np.sqrt(2 * np.log(2)))
        
    if x.ndim>1:
        mean = np.expand_dims(mean, axis=-1)
        sigma = np.expand_dims(sigma, axis=-1)
        
    G = np.exp(-0.5 * ((x - mean) / sigma)**2) / np.sqrt(2 * cst.pi * sigma**2)

    if x.ndim>1:
        G /= G.sum(axis=-1)[:,None]  # Normalization
    else:
        G /= G.sum()
        
    return G


def sigma2fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))


def fwhm2sigma(fwhm):
    """Convert a full width half max to a standard deviation, assuming a gaussian
    Parameters
    ----------
    fwhm : float
        Full-width half-max of a gaussian.
    Returns
    -------
    sigma : float
        Standard deviation of a gaussian.
    """

    sigma = fwhm / np.sqrt(8. * np.log(2.))

    return sigma

 # ----------------------------------------------------------------



def myround(x, base=1., typeRound=None):
    '''Rounding number to the base you want
       You can choose the typeRound = 'ceil' or 'floor' '''
    if typeRound is None:
        return base * np.round(x / base)
    elif typeRound == 'ceil':
        return base * np.ceil(x / base)
    elif typeRound == 'floor':
        return base * np.floor(x / base)
# ----------------------------------------------------------------


def alpha2nb(char):
    """alpha character to numeric"""
    return ord(char) - 96
# ----------------------------------------------------------------


import batman 
from astropy.modeling.physical_models import BlackBody as bb

def calc_tr_lightcurve(planet, coeffs, time, T0,
                       # fl_ratio=0.0001, t0 = 0.5,
                       ld_model='linear', kind_trans='transmission'):
    params = batman.TransitParams()
    params.t0 = planet.mid_tr.value                       #time of inferior conjunction
    params.per = planet.period.to(u.d).value                      #orbital period
    params.rp = (planet.R_pl/planet.R_star).decompose().value                      #planet radius (in units of stellar radii)
    params.a = (planet.ap/planet.R_star).decompose().value                       #semi-major axis (in units of stellar radii)
    params.inc = planet.incl.to(u.deg).value                     #orbital inclination (in degrees)
    params.ecc = planet.excent                      #eccentricity
    try:
        params.w = planet.w.to(u.deg).value                     #longitude of periastron (in degrees)
    except:
        params.w = 90.
    params.u = coeffs                #limb darkening coefficients [u1, u2]
    params.limb_dark = ld_model       #limb darkening model
    # Note that for circular orbits, batman uses the convention params.w = 90. 
    # The units for params.t0 and params.per can be anything as long as they are consistent.

    # We also need to specify the times at which we wish to calculate the model:

#     t = t1.t_start.value #nplanet.linspace(-0.05, 0.05, 100)
    # Using these parameters, we initialize the model and calculate a model light curve:

    if kind_trans == 'emission':
        transittype = "secondary"
        bb_star = bb(planet.Teff)
        bb_pl = bb(planet.Tp)
        fl_ratio = bb_pl(1.4*u.um)/bb_star(1.4*u.um)
        params.fp = fl_ratio
        params.t_secondary = T0.value
        # if tag == 'secondary':
        #     params.t_secondary += 0.5*planet.period.to(u.d).value
    else:
        transittype = "primary"
    m = batman.TransitModel(params, time, transittype=transittype)    #initializes model
    alpha = m.light_curve(params)          #calculates light curve
    
    if kind_trans == 'emission':
        return alpha-1
    elif kind_trans == 'transmission':
         return 1-alpha



def circle_overlap(R, r, S):
    '''
    α(R, r; S) is the area of overlap between two circles, of radii R and r
    with separation S
    '''
    try:
        r = mat_vect(r, col=1, size=S.size)
    except AttributeError:
        r = u.Quantity([r])
        r = mat_vect(r, col=1, size=S.size)
    S = mat_vect(S, col=0, size=r.size)

    alpha = np.empty_like(S) * u.m

    alpha[S >= (r + R)] = 0

    alpha[S <= (R - r)] = cst.pi * r[S <= (R - r)]**2

    i = np.nonzero((S < (R + r)) & (S > (R - r)))
    k0 = np.arccos((S[i]**2 + r[i]**2 - R**2) / (2 * S[i] * r[i])).value
    k1 = np.arccos((S[i]**2 + R**2 - r[i]**2) / (2 * S[i] * R)).value
    k2 = np.sqrt((4 * (S[i] * R)**2 - (R**2 + S[i]**2 - r[i]**2)**2) / 4).decompose()
    alpha[i] = r[i]**2 * k0 + R**2 * k1 - k2

    return alpha.squeeze()

    # alpha = np.empty_like(S)*u.m

    # i, = np.nonzero(S >= (R+r))
    # alpha[i] = 0

    # i, = np.nonzero(S <= (R-r))
    # alpha[i] = cst.pi*r**2

    # i, = np.nonzero((S < (R+r)) & (S > (R-r)))
    # k0 = np.arccos((S[i]**2+r**2-R**2)/(2*S[i]*r)).value
    # k1 = np.arccos((S[i]**2+R**2-r**2)/(2*S[i]*R)).value
    # k2 = np.sqrt((4*(S[i]*R)**2-(R**2+S[i]**2-r**2)**2)/4).decompose()
    # alpha[i] = r**2*k0+R**2*k1-k2
#


def relations(ref='', want='', value=None, plot=False):
    '''
    # --- To find the polynomial relations from Faherty2016 and Filippazzo2015 ---
    # ref = faherty2016, filippazzo2015 [M6-T9], pecaut2013 [O9 (0.5),M9], pecaut2013_no0.5 [O9,M9], pecaut2013_PMS [F0,M9]
    # want = the number associated to the wanted relation, if known
    # value = input parameter value, if known (or a specific spectral type)
    # plot = True to plot the relation
    '''
    beginning = True

    while beginning is True:

        pat = '/home/boucher/spirou/Sptype-Teff/'
        if ref == '':
            file = pat + 'faherty2016'
        else:
            file = pat + ref

        # --- Reading the .fits
        hdulist = fits.open(file + '.fits')
        tbdata = hdulist[1].data

        # --- To access each individual array:
        out_param = tbdata.field('out_param')
        out_type = tbdata.field('out_type')
        start = tbdata.field('start')
        in_param = tbdata.field('in_param')
        end = tbdata.field('end')
    #    rms=tbdata.field('rms')
        factors = np.array([tbdata.field('c0'), tbdata.field('c1'),
                            tbdata.field('c2'), tbdata.field('c3'),
                            tbdata.field('c4'), tbdata.field('c5'),
                            tbdata.field('c6')])

        if want == '':
            print('Here is a list of the possible output, with their needed input :')
            for i in range(len(out_param)):
                print('%d --- Out : %s, Type : %s and In : %s' % (i, out_param[i], out_type[i], in_param[i]))

            while True:
                try:
                    want = int(input('Which Output do you want? Type in the associated # : '))
                    if want in range(len(out_param)):
                        break
                    else:
                        print('This output is not in the list. Try again.')
                except:
                    print('This is not a valid answer. Enter a number, please.')

        x = np.around(np.arange(start[want], end[want], 0.1), 1)

        fac = factors[:, want]
        spt = []

        # --- Attributing the good letter for the SpT
        if in_param[want].upper().strip() == 'SPT':
            for num in x:
                if num < -50:
                    spt_letter = "O"
                elif num >= -50 and num < -40:
                    spt_letter = "B"
                elif num >= -40 and num < -30:
                    spt_letter = "A"
                elif num >= -30 and num < -20:
                    spt_letter = "F"
                elif num >= -20 and num < -10:
                    spt_letter = "G"
                elif num >= -10 and num < 0:
                    spt_letter = "K"
                elif num >= 0 and num < 10:
                    spt_letter = "M"
                elif num >= 10 and num < 20:
                    spt_letter = "L"
                elif num >= 20 and num < 30:
                    spt_letter = "T"
                elif num > 30:
                    spt_letter = "Y"
                spt.append(spt_letter + '{:.1f}'.format(num % 10))

        # --- I make sure x and factors have the right dimension for multiplication
    #    x=x.reshape(x.shape[0],1)
    #    fac=factors[:,want].reshape(1,factors[:,want].shape[0])

        # --- Calculate the polynomial function for the chosen out_param, with respect to the limits of the in_param
        pol = np.zeros(x.size)
        for i in range(7):
            pol += (x**i) * fac[i]

        # --- Plotting the relation
        if plot is True:
            plt.plot(x, pol)
            plt.xlabel(in_param[want])
            plt.ylabel(out_param[want] + ' ' + out_type[want])

        # --- Depending on what is demended, it return differents values

        if type(value) is str:
            if spt != []:
                if len(value) == 2:
                    value += '.0'
                try:
                    ind = spt.index(value)
                except ValueError:

                    print('The spectral type you entered is not available : ', value, ' for ', out_param[want])
                    value_temp = value

                    value = input('Enter a spectral type between ' + str(spt[0]) + ' and ' + str(spt[len(spt) - 1]) + ', or type "ref" to change reference : ')

                    if value == 'ref':
                        ref = input('Enter the new ref code : ')
                        value = value_temp
                        continue
                    else:
                        beginning = False

                    ind = spt.index(value)

                if plot is True:
                    plt.plot(x[ind], pol[ind], 'or')

                return x[ind], pol[ind], [spt[ind], out_param[want]]

            else:
                return x, pol, [in_param[want], out_param[want]]

        elif type(value) is float:
            try:
                ind = x.index(value)
            except ValueError:
                print('The value you entered is not available')
                value = input('Enter a value between ' + str(x[0]) + ' and ' + str(x[len(x) - 1]) + ' : ')
                ind = spt.index(value)

            if plot is True:
                plt.plot(x[ind], pol[ind], 'or')

            return x[ind], pol[ind], [out_param[want], in_param[want]]

        else:
            if spt != []:
                return x, pol, spt
            else:
                return x, pol, [in_param[want], out_param[want]]

# ----------------------------------------------------------------




def find_radius(spt, want='', ref=''):
    '''
    # --- To find the star radius from Lbol from Faherty2016 and Filippazzo2015 ---
    # ref = faherty2016 [M7-T8/L7], filippazzo2015 [M6-T9]
    # want = the number associated to the wanted relation, if known
    # value = input parameter value, if known (or a specific spectral type)
    # plot = True to plot the relation

    Ex. : find_radius('M6',ref = 'filippazzo2015',want = 32)
    '''

    want1, ref1 = want, ref
    x, log_Lbol, params = relations(ref=ref1, want=want1, value=spt)
    Lbol = 10**log_Lbol * 3.8275e26 * u.W
    x, Teff, params = relations(want=18, value=spt)
    Teff = Teff * u.K

    R_etoile = np.sqrt(Lbol / (4 * cst.pi * const.sigma_sb)) * Teff**(-2)
    #print('R_etoile : ','{:.4e}'.format(R_etoile),' = ','{:.4f}'.format(R_etoile/const.R_sun),' R_Sun')
    return R_etoile

# ----------------------------------------------------------------


def dd2dms(dd):
    negative = dd < 0
    dd = abs(dd)
    minutes, seconds = divmod(dd * 3600, 60)
    degrees, minutes = divmod(minutes, 60)
    if negative:
        if degrees > 0:
            degrees = -degrees
        elif minutes > 0:
            minutes = -minutes
        else:
            seconds = -seconds
    return (degrees, minutes, seconds)

# ----------------------------------------------------------------


def doppler_shift(wl_in, wl_out, flux, dv, obs=False, inverse=False, mask_it=True,
                  kind='linear', fill_value="extrapolate", scale=True):
    """
    dv = v_receiver - v_source should be positive when the source and the receiver
    are moving towards each other
    """
    if inverse is True:
        dv *= -1.

    shift = calc_shift(dv)
    # print(shift)

# def doppler_shift(wl_av, wl_ap, flux, dvo, dvs):
#     shift = (1.+dvs/const.c)/(1.+dvo/const.c)

    try:
        unit = wl_in.unit
    except AttributeError:
        unit = 1.

#     if wl_out is None:
#         wl_out = wl_in

#     Lambda0 = np.log(wl_in/unit)
#     DLambda = np.log(1./shift)
#     Lambda_obs = Lambda0+DLambda
#     wl_obs = np.exp(Lambda_obs)

    wl_obs = (wl_in / unit) * shift  # +1

    # print(flux)
    if mask_it is True:
        fct_flux = interp1d_masked(wl_obs, flux, kind=kind, fill_value=fill_value)
    else:
        fct_flux = interp1d(wl_obs, flux, kind=kind, fill_value=fill_value)

    # extrapolating may not be the best idea when doppler shifting
    new_flux = fct_flux(wl_out)

    # print(new_flux)
    if scale is True:
        if inverse is False:
            new_flux /= shift  # shift
        else:
            new_flux *= shift  # shift

    if obs is False:
        return new_flux, shift
    else:
        return new_flux, shift, wl_obs

# ----------------------------------------------------------------


def test_doppler_shift(dv=None):
    x = np.arange(0.5, 2.5, 2e-5) * u.um
    y = np.sin(x.value)

    if dv is None:
        dvo = 20000 * u.km / u.s
        dvs = -10000 * u.km / u.s
        dv = dvo - dvs

    print('dv = ', dv)
    xnew = np.arange(1.001, 2.35, 1.3e-5)

    y_scale, shift = doppler_shift(x, xnew, y, dv)
    y_noscale, shift = doppler_shift(x, xnew, y, dv, scale=False)

    plt.plot(x, y, 'kx', label='og')
    plt.plot(xnew, y_scale, 'x', label='shifted')  # /(1+(-dv/const.c).decompose())
    plt.plot(xnew, y_noscale, 'x', label='shifted', color='green')  # /(1+(-dv/const.c).decompose())

    
def calc_shift(dv, kind='rel'):
    # - take in entry wl_source and give the shift to wl_observer
    if isinstance(dv, u.Quantity):
        dv = dv.to(u.km / u.s)
    else:
        dv = dv * u.km / u.s
    beta = (dv / const.c).decompose()
    
    if kind == 'rel':
        shift = np.sqrt((1. + beta) / (1 - beta)).value
    elif kind == 'classic':
        shift = (1+beta)
    return shift


def doppler_shift2(x, y, dv, scale=False, xout=None, kind='rel'):
    # - Calculate the shift -
    shifts = calc_shift(dv, kind=kind)
    
    # - Interpolate over the orginal data
    fct = interp1d_masked(x, y, kind='cubic', fill_value='extrapolate')
    
    # - Evaluate it at the shifted grid
    if dv.ndim > 0:
        yp = np.array([fct(x/shift) for shift in shifts])
    else:
        yp = fct(x/shifts)
    
    # - Evaluate at a given grid
    if xout is not None:
        fct_shifted = interp1d_masked(x, yp, kind='cubic', fill_value='extrapolate')
        yp = fct_shifted(xout)
    
    # - To be exaustive, one would want to scale the spectrum after shifting it
    # - (This effect is very small at the shifts we usually use)
    if scale is True:
#         print(yp.shape, shifts.shape)
        try:
            yp /= shifts  # shift
        except ValueError:
            yp /= shifts[:,None]
    
    return yp

def doppler_shift3(x, y, dv, scale=False, xout=None):
    # - Calculate the shift -
    shifts = (1 + (dv/const.c).decompose().value )  #  calc_shift(dv)
    
    new_x = x * shifts

    # - Interpolate over the shifted grid
    fct = interp1d_masked(new_x, y, kind='cubic', fill_value='extrapolate')

    yp = fct(x)

    # - Evaluate at a given grid
    if xout is not None:
        yp = fct(xout)
    
    # - To be exaustive, one would want to scale the spectrum after shifting it
    # - (This effect is very small at the shifts we usually use)
    if scale is True:
        yp /= shifts

    return yp


def test_doppler_shift2(dv=-3000):
    x = np.linspace(0, 10, 300)
    y = np.sin(2*x)
    dv = dv * u.km / u.s
    
    gamma = 1/np.sqrt(1-(dv/const.c)**2)

    fct = interp1d_masked(x, y, kind='cubic', fill_value='extrapolate')
    shift = calc_shift(dv, kind='rel')
    print(shift, 1/shift)
    yp = fct(x / shift)/shift
    plt.figure()
    plt.plot(x, y, 'b', label='Original')
    plt.plot(x, yp, ':', label='Shifted 2')

    y2 = doppler_shift(x, x, y, dv)[0]
    y3 = doppler_shift2(x, y, dv, scale=True)
    
    y4 = doppler_shift3(x, y, dv, scale=True)
    
    plt.plot(x, y2, 'g', label='Shifted 1')
    plt.plot(x, y3, 'r', label='Shifted 2, scaled')
    plt.plot(x, y4, color='pink', linestyle='--', label='Shifted 3, scaled', alpha=0.8)
    leg = plt.legend(loc='best')


def mat_vect(vec, col=0, size=None):
    '''
    np.tile does the same thing.... or use var[:, None] to add a dimension
    Voir la commande rang, col = np.indices((shape)) pour avoir une matrice
    avec le numéro des indices des colonnes et des rangées en même temps
    dist = np.sqrt((np.indices((dim,dim))[0]-dim/2)**2+(np.indices((dim,dim))[1]-dim/2)**2)
    '''
    if size is None:
        size = vec.size

    if col == 1:
        mat = vec.reshape(vec.size, 1) * np.ones((1, size))
    else:
        mat = np.ones((size, 1)) * vec.reshape(1, vec.size)

    return mat

# ----------------------------------------------------------------


def dist(dim, point=[0, 0], center=False, xy=False):
    '''
    Créer un tableau carré ont la valeur de chaque élément est proportionnel à la distance eulidiene.
    Utile pour créer un tableau de fréquence pour aplications FFT. Adapté de la fonction dist de IDL.
    '''
    # dist = np.sqrt((np.indices((dim,dim))[0]-dim/2)**2+(np.indices((dim,dim))[1]-dim/2)**2)

    if center is True:
        point = [int(dim / 2), int(dim / 2)]
    x = mat_vect(np.arange(dim), col=0)
    y = mat_vect(np.arange(dim), col=1)

    distance = np.sqrt((x - point[0])**2 + (y - point[1])**2)
    # insère une ligne
    if xy is False:
        return distance
    else:
        return distance, x - point[0], y - point[0]

 # ----------------------------------------------------------------


def cart2pol(x, y):

    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


# def closest(array,mynum,value=False):
#    #min(array, key=lambda x:abs(x-mynum))
#
#    return np.where(abs(x-mynum) == )
 # ----------------------------------------------------------------


def nearest(array, mynumber, axis=-1):
    if isinstance(mynumber,float):
        return (np.abs(array - mynumber)).argmin()
    else:
        return (np.abs(array - mynumber[:,None])).argmin(axis=axis)

 # ----------------------------------------------------------------
 # To calculation code execution time


def tic():
    return time.time()


def toc(start):
    print('Execution time = {}'.format(time.time() - start))

 # ----------------------------------------------------------------


# def bisec(function, a, b, tol, nmax, *args):  # ,**kwargs
#     # INPUT: Function f, endpoint values a, b, tolerance TOL, maximum iterations NMAX
#     # CONDITIONS: a < b, either f(a) < 0 and f(b) > 0 or f(a) > 0 and f(b) < 0
#     # OUTPUT: value which differs from a root of f(x)=0 by less than TOL
#     N = 1
#     while N <= nmax:  # limit iterations to prevent infinite loop
#         c = (a + b) / 2.  # new midpoint
#         diff = (b - a) / 2.

#         Fc = function([c], *args)
#         Fa = function([a], *args)
#         if Fc == 0. or diff < tol:
#             return c  # solution found
#             break
#         N += 1  # increment step counter

#         if Fc * Fa > 0:  # new interval
#             a = c
#         else:
#             b = c
#     print("Method failed. Nmax has been attained.")

#  # ----------------------------------------------------------------


def two_scales(ax1, common_x, data1, data2, c1='r', c2='b', labelx='Common X',
               label1='data1', label2='data2'):
    """
    Parameters
    ----------
    ax : axis
        Axis to put two scales on
    common_x : array-like
        x-axis values for both datasets
    data1: array-like
        Data for left hand scale
    data2 : array-like
        Data for right hand scale
    c1 : color
        Color for line 1
    c2 : color
        Color for line 2

    Returns
    -------
    ax : axis
        Original axis
    ax2 : axis
        New twin axis
    """
    ax2 = ax1.twinx()

    ax1.plot(common_x, data1, color=c1)
    ax1.set_xlabel(labelx)
    ax1.set_ylabel(label1)

    ax2.plot(common_x, data2, color=c2)
    ax2.set_ylabel(label2)
    return ax1, ax2

# --- Example
# - Create some mock data
# t = np.arange(0.01, 10.0, 0.01)
# s1 = np.exp(t)
# s2 = np.sin(2 * np.pi * t)

# - Create axes
# fig, ax = plt.subplots()
# ax1, ax2 = two_scales(ax, t, s1, s2, 'r', 'b')

# --- Change color of each axis


def color_y_axis(ax, color):
    """Color your axes."""
    for t in ax.get_yticklabels():
        t.set_color(color)
    return None

# --- Example
# color_y_axis(ax1, 'r')
# color_y_axis(ax2, 'b')

# ----------------------------------------------------------------


def planck(wav, T):
    a = 2.0 * const.h * const.c**2
    b = const.h * const.c / (wav * const.k_B * T)
    intensity = a / ((wav**5) * (np.exp(b) - 1.0))
    return intensity.to(u.W / u.m**2 / u.um).value

# ----------------------------------------------------------------


# def polyfit2d(x, y, z, order=2):
#     ncols = (order + 1)**2
#     G = np.zeros((x.size, ncols))
#     ij = itertools.product(range(order + 1), range(order + 1))
#     for k, (i, j) in enumerate(ij):
#         G[:, k] = x**i * y**j
#     m, _, _, _ = np.linalg.lstsq(G, z)
#     return G, m


def polyfit2d(x, y, z, order=2):
    G = poly_matrix(x, y, order=order)
    m, _, _, _ = np.linalg.lstsq(G, z, rcond=1e-6)
    return G, m


def poly_matrix(x, y, order=2):
    """ generate Matrix use with lstsq """
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order + 1), range(order + 1))
    for k, (i, j) in enumerate(ij):
        G[:, k] = x**i * y**j
    return G

# ----------------------------------------------------------------


def stop():
    raise Exception('exit')

# ----------------------------------------------------------------


# def find_good(x, factor=0.3, interval=500):
#     '''
#     Find in a spectrum the indices where the flux is >0.3 times the median flux
#     '''
#     index = []
#     for i in range(x.size):
#         if i < interval:
#             med_i = np.median(x[:i + interval])
#         elif i >= interval & i < x.size - interval:
#             med_i = np.median(x[i - interval:i + interval])
#         elif i >= x.size - interval:
#             med_i = np.median(x[i - interval:])

#         if x[i] > factor * med_i:
#             index.append(i)

#     return np.array(index)

# # ----------------------------------------------------------------


def get_colors(cmap_str, nb, inv=False):
    '''generate a color array from a deesired colormap'''
    cmap = getattr(mpl.cm, cmap_str)
    colors = [cmap(x) for x in np.linspace(0, 1.0, nb)]
    # Inverse the colors if needed
    if inv:
        colors = colors[::-1]

    return colors

# ----------------------------------------------------------------


def print_static(*var):

#     print("\r", end="")
#     print(*var, end="")
    print(*var, end='\r', flush=True)

# ----------------------------------------------------------------


# from IPython.display import Markdown, display
# def printmd(string):
#     display(Markdown(string))
    
# ----------------------------------------------------------------    

    
def expand_mask(flux, n_points=3):
    if ~isinstance(flux, np.ma.MaskedArray):
        flux = np.ma.array(flux, mask=np.isnan(flux))
    
    block = np.ones((n_points+1)*2+1)
    block[0], block[-1] = 0, 0
    
    flux_mask = convolve1d(flux.copy().mask, block, axis=-1,
                               mode='nearest').astype(bool)
    flux.mask = flux_mask
    
    return flux


# ----------------------------------------------------------------    

def find_n_maxs(array, n=3, kind='max'):
    """ --- find all the local maxima --- """
    if kind == 'max':
        return np.argpartition(array,-n)[-n:]
    elif kind == 'min':
        return np.argpartition(array,n)[:n]

# ----------------------------------------------------------------       
    
def replace_with_check(text, characters_list, change_to):
    """ --- Replace every instance of a character in a string by something else --- """
    for ch in characters_list:
        if ch in text:
            text = text.replace(ch, change_to)
    return text  
# ----------------------------------------------------------------    


import inspect
def retrieve_name(var):
    """ --- To get the name of a variable as a string --- """
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def insert_str(string, str_to_insert, index):
    return string[:index] + str_to_insert + string[index:]



# ==============================================================================
# Code for building convolution matrix (c matrix).
# ==============================================================================

def gaussians(x, x0, sig, amp=None):
    """Gaussian function
    Parameters
    ----------
    x : array[float]
        Array of points over which gaussian to be defined.
    x0 : float
        Center of the gaussian.
    sig : float
        Standard deviation of the gaussian.
    amp : float
        Value of the gaussian at the center.
    Returns
    -------
    values : array[float]
        Array of gaussian values for input x.
    """

    # Amplitude term
    if amp is None:
        amp = 1. / np.sqrt(2. * np.pi * sig**2.)

    values = amp * np.exp(-0.5 * ((x - x0) / sig) ** 2.)

    return values

def to_2d(kernel, grid_range):
    """ Build a 2d kernel array with a constant 1D kernel (input)
    Parameters
    ----------
    kernel : array[float]
        Input 1D kernel.
    grid_range : list[int]
        Indices over which convolution is defined on grid.
    Returns
    -------
    kernel_2d : array[float]
        2D array of input 1D kernel tiled over axis with
        length equal to difference of grid_range values.
    """

    # Assign range where the convolution is defined on the grid
    a, b = grid_range

    # Get length of the convolved axis
    n_k_c = b - a

    # Return a 2D array with this length
    kernel_2d = np.tile(kernel, (n_k_c, 1)).T

    return kernel_2d


def _get_wings(fct, grid, h_len, i_a, i_b):
    """Compute values of the kernel at grid[+-h_len]
    Parameters
    ----------
    fct : callable
        Function that returns the value of the kernel, given
        a grid value and the center of the kernel.
        fct(grid, center) = kernel
        grid and center have the same length.
    grid : array[float]
        grid where the kernel is projected
    h_len : int
        Half-length where we compute kernel value.
    i_a : int
        Index of grid axis 0 where to apply convolution.
        Once the convolution applied, the convolved grid will be
        equal to grid[i_a:i_b].
    i_b : int
        index of grid axis 1 where to apply convolution.
    Returns
    -------
    left : array[float]
        Kernel values at left wing.
    right : array[float]
        Kernel values at right wing.
    """

    # Save length of the non-convolved grid
    n_k = len(grid)

    # Get length of the convolved axis
    n_k_c = i_b - i_a

    # Init values
    left, right = np.zeros(n_k_c), np.zeros(n_k_c)

    # Add the left value on the grid
    # Possibility that it falls out of the grid;
    # take first value of the grid if so.
    i_grid = np.max([0, i_a - h_len])

    # Save the new grid
    grid_new = grid[i_grid:i_b - h_len]

    # Re-use dummy variable `i_grid`
    i_grid = len(grid_new)

    # Compute kernel at the left end.
    # `i_grid` accounts for smaller length.
    ker = fct(grid_new, grid[i_b - i_grid:i_b])
    left[-i_grid:] = ker

    # Add the right value on the grid
    # Possibility that it falls out of the grid;
    # take last value of the grid if so.
    # Same steps as the left end (see above)
    i_grid = np.min([n_k, i_b + h_len])
    grid_new = grid[i_a + h_len:i_grid]
    i_grid = len(grid_new)
    ker = fct(grid_new, grid[i_a:i_a + i_grid])
    right[:i_grid] = ker

    return left, right


def trpz_weight(grid, length, shape, i_a, i_b):
    """Compute weights due to trapezoidal integration
    Parameters
    ----------
    grid : array[float]
        grid where the integration is projected
    length : int
        length of the kernel
    shape : tuple[int]
        shape of the compact convolution 2d array
    i_a : int
        Index of grid axis 0 where to apply convolution.
        Once the convolution applied, the convolved grid will be
        equal to grid[i_a:i_b].
    i_b : int
        index of grid axis 1 where to apply convolution.
    Returns
    -------
    out : array[float]
        2D array with shape according to input shape
    """

    # Index of each element on the convolution matrix
    # with respect to the non-convolved grid
    # `i_grid` has the shape (N_k_convolved, kernel_length - 1)
    i_grid = np.indices(shape)[0] - (length // 2)
    i_grid = np.arange(i_a, i_b)[None, :] + i_grid[:-1, :]

    # Set values out of grid to -1
    i_bad = (i_grid < 0) | (i_grid >= len(grid) - 1)
    i_grid[i_bad] = -1

    # Delta lambda
    d_grid = np.diff(grid)

    # Compute weights from trapezoidal integration
    weight = 0.5 * d_grid[i_grid]
    weight[i_bad] = 0

    # Fill output
    out = np.zeros(shape)
    out[:-1] += weight
    out[1:] += weight

    return out


def fct_to_array(fct, grid, grid_range, thresh=1e-5, length=None):
    """Build a compact kernel 2d array based on a kernel function
    and a grid to project the kernel
    Parameters
    ----------
    fct : callable
        Function that returns the value of the kernel, given
        a grid value and the center of the kernel.
        fct(grid, center) = kernel
        grid and center have the same length.
    grid : array[float]
        Grid where the kernel is projected
    grid_range : list[int] or tuple[int]
        Indices of the grid where to apply the convolution.
        Once the convolution applied, the convolved grid will be
        equal to grid[grid_range[0]:grid_range[1]].
    thresh : float, optional
        Threshold to cut the kernel wings. If `length` is specified,
        `thresh` will be ignored.
    length : int, optional
        Length of the kernel. Must be odd.
    Returns
    -------
    kern_array : array[float]
        2D array of kernel projected onto grid.
    """

    # Assign range where the convolution is defined on the grid
    i_a, i_b = grid_range

    # Init with the value at kernel's center
    out = fct(grid, grid)[i_a:i_b]

    # Add wings
    if length is None:
        # Generate a 2D array of the grid iteratively until
        # thresh is reached everywhere.

        # Init parameters
        length = 1
        h_len = 0  # Half length

        # Add value on each sides until thresh is reached
        while True:
            # Already update half-length
            h_len += 1

            # Compute next left and right ends of the kernel
            left, right = _get_wings(fct, grid, h_len, i_a, i_b)

            # Check if they are all below threshold.
            if (left < thresh).all() and (right < thresh).all():
                break  # Stop iteration
            else:
                # Update kernel length
                length += 2

                # Set value to zero if smaller than threshold
                left[left < thresh] = 0.
                right[right < thresh] = 0.

                # add new values to output
                out = np.vstack([left, out, right])

        # Weights due to integration (from the convolution)
        weights = trpz_weight(grid, length, out.shape, i_a, i_b)

    elif (length % 2) == 1:  # length needs to be odd
        # Generate a 2D array of the grid iteratively until
        # specified length is reached.

        # Compute number of half-length
        n_h_len = (length - 1) // 2

        # Simply iterate to compute needed wings
        for h_len in range(1, n_h_len + 1):
            # Compute next left and right ends of the kernel
            left, right = _get_wings(fct, grid, h_len, i_a, i_b)

            # Add new kernel values
            out = np.vstack([left, out, right])

        # Weights due to integration (from the convolution)
        weights = trpz_weight(grid, length, out.shape, i_a, i_b)

    else:
        msg = "`length` provided to `fct_to_array` must be odd."
        log.critical(msg)
        raise ValueError(msg)

    kern_array = (out * weights)
    return kern_array


def cut_ker(ker, n_out=None, thresh=None):
    """Apply a cut on the convolution matrix boundaries.
    Parameters
    ----------
    ker : array[float]
        convolution kernel in compact form, so
        shape = (N_ker, N_k_convolved)
    n_out : int, list[int] or tuple[int]
        Number of kernel's grid point to keep on the boundaries.
        If an int is given, the same number of points will be
        kept on each boundaries of the kernel (left and right).
        If 2 elements are given, it corresponds to the left and right
        boundaries.
    thresh : float
        threshold used to determine the boundaries cut.
        If n_out is specified, this is ignored.
    Returns
    ------
    ker : array[float]
        The same kernel matrix as the input ker, but with the cut applied.
    """

    # Assign kernel length and number of kernels
    n_ker, n_k_c = ker.shape

    # Assign half-length of the kernel
    h_len = (n_ker - 1) // 2

    # Determine n_out with thresh if not given
    if n_out is None:

        if thresh is None:
            # No cut to apply
            return ker
        else:
            # Find where to cut the kernel according to thresh
            i_left = np.where(ker[:, 0] >= thresh)[0][0]
            i_right = np.where(ker[:, -1] >= thresh)[0][-1]

            # Make sure it is on the good wing. Take center if not.
            i_left = np.minimum(i_left, h_len)
            i_right = np.maximum(i_right, h_len)

    # Else, unpack n_out
    else:
        # Could be a scalar or a 2-elements object)
        try:
            i_left, i_right = n_out
        except TypeError:
            i_left, i_right = n_out, n_out

        # Find the position where to cut the kernel
        # Make sure it is not out of the kernel grid,
        # so i_left >= 0 and i_right <= len(kernel)
        i_left = np.maximum(h_len - i_left, 0)
        i_right = np.minimum(h_len + i_right, n_ker - 1)

    # Apply the cut
    for i_k in range(0, i_left):
        # Add condition in case the kernel is larger
        # than the grid where it's projected.
        if i_k < n_k_c:
            ker[:i_left - i_k, i_k] = 0

    for i_k in range(i_right + 1 - n_ker, 0):
        # Add condition in case the kernel is larger
        # than the grid where it's projected.
        if -i_k <= n_k_c:
            ker[i_right - n_ker - i_k:, i_k] = 0

    return ker


def sparse_c(ker, n_k, i_zero=0):
    """Convert a convolution kernel in compact form (N_ker, N_k_convolved)
    to sparse form (N_k_convolved, N_k)
    Parameters
    ----------
    ker : array[float]
        Convolution kernel in compact form, with shape (N_kernel, N_kc)
    n_k : int
        Length of the original grid
    i_zero : int
        Position of the first element of the convolved grid
        in the original grid.
    Returns
    -------
    matrix : array[float]
        Sparse form of the input convolution kernel
    """

    # Assign kernel length and convolved axis length
    n_ker, n_k_c = ker.shape

    # Algorithm works for odd kernel grid
    if n_ker % 2 != 1:
        err_msg = "Length of the convolution kernel given to sparse_c should be odd."
        log.critical(err_msg)
        raise ValueError(err_msg)

    # Assign half-length
    h_len = (n_ker - 1) // 2

    # Define each diagonal of the sparse convolution matrix
    diag_val, offset = [], []
    for i_ker, i_k_c in enumerate(range(-h_len, h_len + 1)):

        i_k = i_zero + i_k_c

        if i_k < 0:
            diag_val.append(ker[i_ker, -i_k:])
        else:
            diag_val.append(ker[i_ker, :])

        offset.append(i_k)

    # Build convolution matrix
    matrix = diags(diag_val, offset, shape=(n_k_c, n_k), format="csr")

    return matrix


def get_c_matrix(kernel, grid, bounds=None, i_bounds=None, norm=True,
                 sparse=True, n_out=None, thresh_out=None, **kwargs):
    """Return a convolution matrix
    Can return a sparse matrix (N_k_convolved, N_k)
    or a matrix in the compact form (N_ker, N_k_convolved).
    N_k is the length of the grid on which the convolution
    will be applied, N_k_convolved is the length of the
    grid after convolution and N_ker is the maximum length of
    the kernel. If the default sparse matrix option is chosen,
    the convolution can be applied on an array f | f = fct(grid)
    by a simple matrix multiplication:
    f_convolved = c_matrix.dot(f)
    Parameters
    ----------
    kernel: ndarray (1D or 2D), callable
        Convolution kernel. Can be already 2D (N_ker, N_k_convolved),
        giving the kernel for each items of the convolved grid.
        Can be 1D (N_ker), so the kernel is the same. Can be a callable
        with the form f(x, x0) where x0 is the position of the center of
        the kernel. Must return a 1D array (len(x)), so a kernel value
        for each pairs of (x, x0). If kernel is callable, the additional
        kwargs `thresh` and `length` will be used to project the kernel.
    grid: one-d-array:
        The grid on which the convolution will be applied.
        For example, if C is the convolution matrix,
        f_convolved = C.f(grid)
    bounds: 2-elements object
        The bounds of the grid on which the convolution is defined.
        For example, if bounds = (a,b),
        then grid_convolved = grid[a <= grid <= b].
        It dictates also the dimension of f_convolved
    sparse: bool, optional
        return a sparse matrix (N_k_convolved, N_k) if True.
        return a matrix (N_ker, N_k_convolved) if False.
    n_out: integer or 2-integer object, optional
        Specify how to deal with the ends of the convolved grid.
        `n_out` points will be used outside from the convolved
        grid. Can be different for each ends if 2-elements are given.
    thresh_out: float, optional
        Specify how to deal with the ends of the convolved grid.
        Points with a kernel value less then `thresh_out` will
        not be used outside from the convolved grid.
    thresh: float, optional
        Only used when `kernel` is callable to define the maximum
        length of the kernel. Truncate when `kernel` < `thresh`
    length: int, optional
        Only used when `kernel` is callable to define the maximum
        length of the kernel.
    """

    # Define range where the convolution is defined on the grid.
    # If `i_bounds` is not specified, try with `bounds`.
    if i_bounds is None:

        if bounds is None:
            a, b = 0, len(grid)
        else:
            a = np.min(np.where(grid >= bounds[0])[0])
            b = np.max(np.where(grid <= bounds[1])[0]) + 1

    else:
        # Make sure it is absolute index, not relative
        # So no negative index.
        if i_bounds[1] < 0:
            i_bounds[1] = len(grid) + i_bounds[1]

        a, b = i_bounds

    # Generate a 2D kernel depending on the input
    if callable(kernel):
        kernel = fct_to_array(kernel, grid, [a, b], **kwargs)
    elif kernel.ndim == 1:
        kernel = to_2d(kernel, [a, b])

    if kernel.ndim != 2:
        msg = ("Input kernel to get_c_matrix must be callable or"
               " array with one or two dimensions.")
        log.critical(msg)
        raise ValueError(msg)
    # Kernel should now be a 2-D array (N_kernel x N_kc)

    # Normalize if specified
    if norm:
        kernel = kernel / np.nansum(kernel, axis=0)

    # Apply cut for kernel at boundaries
    kernel = cut_ker(kernel, n_out, thresh_out)

    if sparse:
        # Convert to a sparse matrix.
        kernel = sparse_c(kernel, len(grid), a)

    return kernel


# ==============================================================================
# One to one mappping
# ==============================================================================

class OneToOneMap:
    
    def __init__(self, keys, values):
        
        self.keys = tuple(keys)
        self.values = tuple(values)
        
    def __getitem__(self, args):
        
        if isinstance(args, list):
            idx_list = [self.keys.index(key) for key in args]
            out = tuple(self.values[idx] for idx in idx_list)
        else:
            idx = self.keys.index(args)
            out = self.values[idx]
            
        return out
    
    def inverse(self):
        return OneToOneMap(self.values, self.keys)
    
    
# ==============================================================================
# Function to deal with command line arguments
# ==============================================================================
# Get kwargs from command line (if called from command line)
def unpack_kwargs_from_command_line(sys_argv):
    kwargs_in = dict([arg.split('=') for arg in sys_argv[1:]])
    if kwargs_in:
        for key, value in kwargs_in.items():
            print(f"Keyword argument from command line: {key} = {value}")
    return kwargs_in


def get_kwargs_with_message(key, kwargs_from_cmd_line, default_val=None):
    """Function to access keywords from command line and print an message when using a default value."""
    try:
        val = kwargs_from_cmd_line[key]
    except KeyError:
        print(f"{key} not provided, using default value of {default_val}")
        val = default_val
    
    return val