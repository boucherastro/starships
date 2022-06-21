#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 13:32:23 2017

@author: AnneBoucher
"""

import numpy as np
import scipy.constants as cst
from astropy import units as u
from astropy import constants as const
import matplotlib.pyplot as plt
from . import observatory as observ
from PyAstronomy import pyasl as al
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import Angle
# from .utils import homemade as hm


def t2trueanom(P, t, t0=0, e=0):
    # --- Going from time (t, input) to true anomaly (nu, output) ---

    # --- Mean motion [rad/s]
    n = 2 * cst.pi / P
    # n=np.sqrt(cst.G*(m1+m2)/a^3)

    # --- Mean anomaly [rad]
    M = np.array((n * (t - t0)).decompose())

    # --- Eccentric anomaly [rad]
    # - Defining E0
    # if e >= 0.3:
    #     E = np.zeros_like(M) + cst.pi
    # else:
    #     E = M
    E = M

    # - Newton-Raphson iterator
    correction = np.zeros_like(M) + 100

    while (correction >= 0.001).all():
        correction = (E - (M + e * np.sin(E))) / (1 - e * np.cos(E))
        E = E - correction

    # --- True anomaly [rad]
    argument = np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2)
    nu = 2 * np.arctan(argument)
    # nu2 = np.arccos((np.cos(E) - e) / (1 - e * np.cos(E)))
    # nu = nu2
    return np.array(nu)


def trueanom2t(P, nu, t0=0, e=0):
    # --- Going from true anomaly (nu, input) to time (t, output) ---

    # --- Eccentric anomaly [rad]
    argument = np.tan(nu / 2) / np.sqrt((1 + e) / (1 - e))
    E = 2 * np.arctan(argument)

    # --- Mean anomaly [rad]
    M = E - e * np.sin(E)

    # --- Mean motion [rad/s]
    n = 2 * cst.pi / P
    # n=np.sqrt(cst.G*(m1+m2)/a^3)

    # --- Time
    t_t0 = M / n
    t = t_t0 + t0

    return np.array(t)


def rv(nu, P, e=0, i=cst.pi / 2, w=0, Mp=const.M_earth, Mstar=const.M_sun):

    if isinstance(nu, u.Quantity):
        nu = nu.to(u.rad)
    else:
        nu = np.array(nu) * u.rad

    # --- Mass function
    Mprime = Mp**3 / (Mstar + Mp)**2

    # --- Semi-major axis (Kepler's law)
    astar = (P.to(u.s)**2 * const.G * Mprime / (4 * cst.pi**2))**(1 / 3)
    # astar=Mp*ap/Mstar
    ap = Mstar * astar / Mp

    # --- Radial velocity semi-amplitude
    K = (2 * cst.pi * astar / P * np.sin(i) / np.sqrt(1 - e * e)).decompose()
    # K2=(2*cst.pi*const.G/P)**(1/3)*Mp*np.sin(i)/(Mstar+Mp)**(2/3)/np.sqrt(1-e*e)
    Kp = (2 * cst.pi * ap / P * np.sin(i) / np.sqrt(1 - e * e)).decompose()

    # --- Radial velocity
    # - Star
    rv = -K * (np.cos(nu + w) + e * np.cos(w))

    # - Planet
    rvp = Kp * (np.cos(nu + w) + e * np.cos(w))
    # rvp=Kp*(np.cos(nu+w+cst.pi)+e*np.cos(w+cst.pi))

    return K, rv, Kp, rvp


def rv_theo(K_star, t, t_ref, P, nu, w, e=0, plnt=False):
    if plnt is True:
        rv1 = K_star * np.sin(2 * cst.pi* u.rad * ((t - t_ref) / P).decompose())
        rv2 = K_star * (np.cos(nu + w) + e * np.cos(w))
    else:
        rv1 = -K_star * np.sin(2 * cst.pi* u.rad * ((t - t_ref) / P).decompose())
        rv2 = -K_star * (np.cos(nu + w) + e * np.cos(w))
        
    return rv1, rv2

def rv_theo_t(K_star, t, t_ref, P, plnt=False):
    if isinstance(K_star, u.Quantity):
        K_star = K_star.to(u.km/u.s)
    else:
        K_star = K_star*u.km/u.s
        
    rv = -K_star * np.sin(2 * cst.pi* u.rad * ((t - t_ref) / P).decompose())    
    if plnt is True:
        rv *= -1
    return rv

def rv_theo_nu(K_star, nu, w, e=0, plnt=False):
    if isinstance(K_star, u.Quantity):
        K_star = K_star.to(u.km/u.s)
    else:
        K_star = K_star*u.km/u.s
        
    rv = -K_star * (np.cos(nu + w) + e * np.cos(w))  
    if plnt is True:
        rv *= -1 
    return rv

def Kp_theo(K_star, M_star, M_pl):
    if isinstance(K_star, u.Quantity):
        K_star = K_star.to(u.km/u.s)
    else:
        K_star = K_star*u.km/u.s
    
    return (M_star*K_star/M_pl).decompose()


def position(nu, e=0, i=cst.pi / 2, w=0, omega=0, ap=None, Rstar=None,
             P=None, Mp=const.M_earth, Mstar=const.M_sun, plot=False):

    if isinstance(nu, u.Quantity):
        nu = nu.to(u.rad)
    else:
        nu = np.array(nu) * u.rad

    if isinstance(w, u.Quantity):
        w = w.to(u.rad)
    else:
        w = np.array(w) * u.rad

    # Compute ap with P if no value given or use 1 AU
    if ap is None:
        if P is not None:
            Mprime = Mp**3 / (Mstar + Mp)**2
            astar = (P.to(u.s)**2 * const.G * Mprime / (4 * cst.pi**2))**(1 / 3)
            ap = Mstar * astar / Mp  # - Barycentre
        else:
            ap = const.au

    r = ap * (1 - e * e) / (1 + e * np.cos(nu))
    if isinstance(r, u.Quantity):  # SI units
        r = r.to(u.m)

    x = r * (np.cos(omega) * np.cos(w + nu) - np.sin(omega) * np.sin(w + nu) * np.cos(i))
    y = r * (np.sin(omega) * np.cos(w + nu) + np.cos(omega) * np.sin(w + nu) * np.cos(i))
    z = r * np.sin(w + nu) * np.sin(i)

    separation = np.sqrt(x * x + y * y)

    bRstar = ap * np.cos(i) * (1 - e * e) / (1 + e * np.sin(w))

    if plot:
        plt.figure()
        plt.plot(x, y)
        circle = plt.Circle((0, 0), Rstar.value, color='#FACC2E')
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_artist(circle)
        plt.axis('equal')
        plt.axvline(0, color='k', ls=':')
        plt.axhline(0, color='k', ls=':')
        plt.xlabel(r'$X_p$')
        plt.ylabel(r'$Y_p$')
        # plt.axis([,,,])

    return r, x, y, z, separation, bRstar


def transit(Rs, Rp, sep, z=None, nu=None, r=None, vr=None, i_tperi=None, w=None):

    if z is None:  # No distinction between transit and eclipse
        z = np.zeros_like(np.array(sep)) - 1

    out = np.where(sep >= (Rs + Rp))[0]
    limb = np.where(
            (sep < (Rs + Rp)) &
            (sep >= (Rs - Rp)) &
            (z < 0)
        )[0]
    transit = np.where(
            (sep < (Rs - Rp)) &
            (sep >= 0) &
            (z < 0)
        )[0]

    if nu is not None:
        if nu.size == sep.size:
            nu = np.squeeze(nu)

            if r is not None:
                plt.figure()
                ax1 = plt.subplot(111, projection='polar')
                ax1.plot(nu[out], r[out], 'r')
                ax1.plot(nu[limb], r[limb], 'gx')
                ax1.plot(nu[transit], r[transit], 'b+')
                if (i_tperi < nu.size) :
                    if (i_tperi is not None):

                        ax1.plot(nu[i_tperi], r[i_tperi], '*')
                    if w is not None:
                        to_observer = nu[i_tperi] - w.to(u.rad).value - cst.pi / 2.
                        ax1.plot([0, to_observer], [0, 1.8e10], '--k')
                        ax1.text(to_observer, 1.8e10, 'To observer')
                ax1.set_rmax(2.0e10)
                plt.show()

#            plt.figure()
#            plt.polar(nu[out],r[out],'r')
#            plt.polar(nu[limb],r[limb],'gx')
#            plt.polar(nu[transit],r[transit],'b+')

            if vr is not None:
                phi = nu / (2 * np.pi) + 0.25
                plt.figure()
                plt.plot(phi[out], vr[out], 'r')
                plt.plot(phi[limb], vr[limb], 'gx')
                plt.plot(phi[transit], vr[transit], 'b+')
                plt.xlabel(r'Orbital phase $\phi$')
                plt.ylabel(r'RV [$km/s$]')
                # plt.xlim(-2, -1)
                # plt.ylim(-25, 25)
                plt.show()

#            plt.figure()
#            x=np.linspace(0,cst.pi,nu.size)
#            plt.plot(x[out],nu[out],'r')
#            plt.plot(x[limb],nu[limb],'gx')
#            plt.plot(x[transit],nu[transit],'b+')

#            if limb.size > 0:
#                plt.axis([(x[limb])[0]-0.01,(x[limb])[-1]+0.01,(nu[limb])[0]-0.05,(nu[limb])[-1]+0.05])
#            elif transit.size > 0:
#                plt.axis([(x[transit])[0]-0.01,(x[transit])[0]+0.01,(nu[transit])[0]-0.05,(nu[transit])[0]+0.05])

    return out, limb, transit


def where_is_the_transit(t, mid_tr, periode, trandur):

    nb_per = (t-mid_tr)/periode.to(u.d)
    nieme_tr_since_tr0 = np.round(nb_per)

    limit_d = nieme_tr_since_tr0*periode.to(u.d)+mid_tr - 0.5*trandur.to(u.d) 
    limit_u = nieme_tr_since_tr0*periode.to(u.d)+mid_tr + 0.5*trandur.to(u.d) 

    iIn = []
    iOut = []
    nb_time = np.arange(t.size)
    for down,up,i in zip(limit_d,limit_u, nb_time):
        if (t[i] <= up) & (t[i] >= down):
            iIn.append(i)
        elif (t[i] > up) | (t[i] < down):
            iOut.append(i)
    
    return np.array(iIn), np.array(iOut)


def where_eclipse(Rs, Rp, sep, z=None, nu=None, r=None, vr=None, i_tperi=None, w=None):

    if z is None:  # No distinction between transit and eclipse
        z = np.zeros_like(np.array(sep)) + 1

    if isinstance(w, u.Quantity):
        w = w.to(u.rad)
    elif w is not None:
        w = np.array(w) * u.rad

    out = np.squeeze(np.where(sep >= (Rs + Rp)))

    ecl_tot = np.squeeze(
        np.where(
            (sep < (Rs - Rp)) &
            (sep >= 0) &
            (z > 0)
        ))

    ecl_part = np.squeeze(
        np.where(
            (sep < (Rs + Rp)) &
            (sep >= (Rs - Rp)) &
            (z > 0)
        ))

    if nu is not None:
        if nu.size == sep.size:
            nu = np.squeeze(nu)

            if r is not None:
                plt.figure()
                ax1 = plt.subplot(111, projection='polar')
                ax1.plot(nu[out], r[out], 'r')
                ax1.plot(nu[ecl_part], r[ecl_part], 'gx')
                ax1.plot(nu[ecl_tot], r[ecl_tot], 'b+')
                if i_tperi is not None:
                    ax1.plot(nu[i_tperi], r[i_tperi], '*')
                if w is not None:
                    to_observer = nu[i_tperi] - w.to(u.rad).value - cst.pi / 2.
                    ax1.plot([0, to_observer], [0, 1.8e10], '--k')
                    ax1.text(to_observer, 1.8e10, 'To observer')
                ax1.set_rmax(2.0e10)
                plt.show()

            if vr is not None:
                plt.figure()
                plt.plot(nu[out], vr[out], 'r')
                plt.plot(nu[ecl_part], vr[ecl_part], 'gx')
                plt.plot(nu[ecl_tot], vr[ecl_tot], 'b+')
                plt.xlabel(r'$\nu$ [rad]')
                plt.ylabel(r'RV [$km/s$]')
                plt.show()

    return out, ecl_part, ecl_tot


def barryc_correc(observatoire, radec, date):

    if isinstance(date, u.Quantity):
        date = date.to(u.d).value

    obs = observ.observ_param(observatoire)
    lon = Angle(obs['Longitude'], unit=u.deg).deg
    lat = Angle(obs['Latitude'], unit=u.deg).deg
    alt = obs['Altitude']  # meters

    c = SkyCoord(radec[0], radec[1], frame='icrs', unit=(u.deg, u.deg))
    # crd2 = [SkyCoord(ra[i], dec[i]) for i in range(len(ra))]  # Faster way if many objects
    ra2000 = c.ra.deg
    dec2000 = c.dec.deg
    # correc, hjd = al.helcorr(lon, lat, alt, ra2000, dec2000, date)
    print(lon, lat, alt, ra2000, dec2000, date[0])
    corr_hjd = np.array([al.helcorr(lon, lat, alt, ra2000, dec2000, date[i]) for i in range(len(date))])
    correc = corr_hjd[:, 0] * u.km / u.s
    # hjd = corr_hjd[:, 1]

    return correc

# ---------------------------------
# --- Orbit time calculation ---


def orbit_time(R_pl, R_star, periode, ap, b, *opt, e=0, i=cst.pi / 2, w=0, omega=0, debug=False):
    '''
    Return the transit duration "tt" (t3-t2)
    and the full transit duration "fulltt" (t4-t1)
    opt = (nu,transit,limb) -> all 3 inputs must be numpy arrays of the same size
    '''

    # Calculations

    p = R_pl / R_star

    TT = (periode / cst.pi * np.arcsin(np.sqrt(((1 - p)**2 - b**2) / ((ap / R_star)**2 - b**2))).value).to(u.min)

    fullTT = (periode / cst.pi * np.arcsin(np.sqrt(((1 + p)**2 - b**2) / ((ap / R_star)**2 - b**2))).value).to(u.min)
    if debug:
        print(TT, fullTT)

    # Options

    if opt:
        # Unpack
        nu, transit, limb = opt

        if transit.size != 0:
            temps_transit = trueanom2t(periode, nu[transit], e=e)
            tt = ((temps_transit[-1] - temps_transit[0]) * periode.unit).to(u.min)
            # tt = ((temps_transit[np.int(temps_transit.size / 2) - 1]
            #        - temps_transit[0]) * periode.unit).to(u.min)

        if limb.size != 0:
            temps_transit = trueanom2t(periode, nu[limb], e=e)
            fulltt = ((temps_transit[-1] - temps_transit[0]) * periode.unit).to(u.min)
            # fulltt = ((temps_transit[np.int(temps_transit.size / 2) - 1]
            #            - temps_transit[0]) * periode.unit).to(u.min)
        if debug:
            print(tt, fulltt)

    return TT, fullTT

# ---------------------------------
# ------ Airmass calculation --------


def airmass(name, radec, observatoire, time, plot=False):
    """
    Compute the airmass
    time : can be a single value or an array of Time('time') value
    ex: date0 = Time(time)
        delta_time = np.linspace(-5, 29, 100)*u.hour
        obstime = date0+delta_time
    """
    try:
        obj_name = SkyCoord.from_name(name)
    except:
        obj_name = SkyCoord(radec[0], radec[1], frame='icrs', unit=(u.deg, u.deg))

    obs = observ.observ_param(observatoire)

    loc = EarthLocation(lat=Angle(obs['Latitude'], unit=u.deg).deg,
                        lon=Angle(obs['Longitude'], unit=u.deg).deg,
                        height=obs['Altitude'])

    frame = AltAz(obstime=time, location=loc)
    altazs = obj_name.transform_to(frame)

    if plot is True:
        plt.plot(time.value, altazs.secz, 'o')
        plt.ylim(3, 1)
        plt.xlabel('Time')
        plt.ylabel('Airmass [Sec(z)]')
        plt.show()

    return altazs.secz

# ---------------------------------

# e = 0.5
# i = np.radians(90)
# w = np.radians(0)
# omega = np.radians(0)
#
# annee=u.year.to(u.s)
# t=np.linspace(0,annee,100000)
#
# nu=t2trueanom(annee,t,e=e)

# K,vr,Kp,vrp=rv(nu,annee*u.s,e=e,i=i,w=w)
#
# temps=trueanom2t(annee,nu,e=e)
#
# rp,x,y,z,sep=position(nu,e=e,i=i,w=w,omega=omega,plot=True)
#
# transit(const.R_sun,const.R_earth,sep,nu=nu,r=rp,rv=vrp)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x,y,z)
# ax.scatter(x,y,z,s=1)

# plt.figure()
# plt.plot(t,vrp)
# plt.figure()
# plt.plot(t,vr)
# plt.figure()
# plt.polar(nu,rp)

# -- Pour M_p<<M_*
# P=sqrt(ap^3/Mstar)*365.25*24
#
# --------------------
# Mp=4.*pi*Rp^3.*rho/3.
# astar=Mp*ap/Mstar
##
# -- Pour M_p arbitraire, orbites relatives
# arel=astar+ap
# chose=4*pi^2*arel^3/(Grav*(Mstar+Mp))
# P_rel=sqrt(chose)
##
# -- Pour M_p arbitraire, orbites absolues
# Mprime=Mp^3/(Mstar+Mp)^2
# chose=4*pi^2*astar^3/(Grav*Mprime)
# P_abs=sqrt(chose)

# ---- Polar plots (différentes manières)

#            r = r/1e11#np.arange(0, 2, 0.01)
#            theta = nu#2 * np.pi * r

#            ax1=plt.subplot(111,projection='polar')
#            ax1.plot(nu[out],r[out],color='r')
#            ax1.plot(nu[limb],r[limb],color='g')
#            ax1.plot(nu[transit],r[transit],color='b') # ou simplement "bx" pour avoir un symbol = x
#            ax1.set_rmax(2.0)
#            plt.show()

#            ax = plt.subplot(111, projection='polar')
#            ax.plot(theta[out], r[out])
#            ax.set_rmax(2)
#            #ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
#            ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
#            ax.grid(True)
#            ax.set_title("A line plot on a polar axis", va='bottom')
#            plt.show()

#            plt.figure()
#            plt.polar(nu[out],rp[out],'r')
#            plt.polar(nu[limb],rp[limb],'gx')
#            plt.polar(nu[transit],rp[transit],'b+')
