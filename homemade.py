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
import matplotlib.pyplot as plt
import matplotlib as mpl

from starships.mask_tools import interp1d_masked
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
    return fwhm / np.sqrt(8 * np.log(2))

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

def calc_tr_lightcurve(planet, coeffs, time, 
                       fl_ratio=0.0001, t0 = 0.5, ld_model='linear', kind_trans='transmission'):
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
#         bb_star = bb(planet.Teff)
#         bb_pl = bb(planet.Teq)
        params.fp = fl_ratio
        params.t_secondary = planet.mid_tr.value + 0.5*planet.period.to(u.d).value
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


def get_colors(cmap, nb, inv=False):
    '''generate a color array from a deesired colormap'''
    exec('color = [mpl.cm.{}(x) for x in np.linspace(0, 1.0, {})]'.format(cmap, nb), globals())
    #colors = [mpl.cm.magma(x) for x in np.linspace(0, 1.0, nb_plot)][::-1]
    colors = color[::-1] if inv is True else color
    return colors

# ----------------------------------------------------------------


def print_static(*var):

#     print("\r", end="")
#     print(*var, end="")
    print(*var, end='\r', flush=True)

# ----------------------------------------------------------------


from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))
    
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