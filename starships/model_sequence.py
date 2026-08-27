"""Shared per-exposure model generation for the high-resolution retrieval pipeline.

Chantier A Phase 2 (see ``Notes/plan_revision_starships.md`` for the full design
discussion). This module is meant to be used identically by ``retrieval.py`` (MCMC) and
``logl_grid.py`` (Kp-Vsys grids) -- see Chantier C, which asks that both compute their
model in as close to the same way as possible.

It fixes Chantier A bug #2: in emission, the planet flux (Fp) and the stellar flux
(Fstar) used to be combined into a single ``Fp/Fstar`` ratio right after petitRADTRANS
generation (``petitradtrans_utils.py::retrieval_model_plain``), and that combined ratio
was then Doppler-shifted **as one rigid object** per exposure
(``correlation.py::gen_model_sequence_noinj`` -> ``spectrum.py::quick_inject_clean``), at
a velocity dominated by the planet's orbital motion (~100-200 km/s). The star's own
reflex motion (~0.1-0.2 km/s) is negligible in comparison, so the net effect was to drag
the (stationary, in reality) stellar absorption lines along at the planet's velocity --
contaminating the fit, since the PCA step cannot separate a stellar contribution that
moves like the planetary signal.

The fix: keep Fp and Fstar separate for as long as possible, and Doppler-shift them
**independently** per exposure, only recombining into a ratio at the very end. Two
building blocks are provided for that:

- ``precompute_theta_model()`` -- the "1D" step: generate Fp/Fstar for one set of
  atmospheric parameters (one `theta`) and degrade them once, before any Doppler shift.
  This replaces the relevant part of ``retrieval.py::prepare_model_high_or_low``.
- ``build_model_sequence()`` -- the "2D" step: Doppler-shift Fp and Fstar independently
  per exposure and recombine into the model sequence compared to the data. This replaces
  ``correlation.py::gen_model_sequence_noinj``/``spectrum.py::quick_inject_clean`` for
  callers that need Fp/Fstar kept separate.

``combine_regions()`` is a third, smaller building block: a generic weighted sum used
both for the citrus-region combination (``retrieval.py::prepare_model_multi_reg``) and
for the cloudy/clear blend (previously a special-cased branch in
``prepare_model_high_or_low``) -- the cloudy/clear blend is really just a 2-region case
of the same weighted-sum mechanism, with no phase-dependent kernel.

Notes on cost (raised by Antoine while reviewing this design): keeping Fstar separate
costs one extra cubic-spline evaluation per order compared to the old single-ratio
shift, in both cases fully vectorized across exposures -- a modest, unavoidable price
for actually fixing bug #2 in emission. There is no per-exposure Python loop here, so
whether the star is treated as fixed (default, see ``vr_orb`` below) or shifted by a
real per-exposure velocity does not change the cost of this step. The genuinely
expensive addition that must stay optional is a phase-dependent rotation kernel
(Chantier A Phase 3): applying a different convolution per exposure. That hook is left
as a `None`/identity default here (``region_kernels``) precisely so it costs nothing
until Phase 3 wires it in.
"""
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from . import homemade as hm
from .convolution import degrade_and_resample
from .mask_tools import interp1d_masked
from . import petitradtrans_utils as prt

ArrayOrScalar = Union[float, np.ndarray]


def combine_regions(sub_models: Sequence[np.ndarray], weights: Sequence[ArrayOrScalar]) -> np.ndarray:
    """Weighted sum of sub-model spectra sharing the same wavelength grid.

    Generic replacement for two combinations that used to be special-cased in
    `retrieval.py`:

    - Summing the citrus-region contributions in `prepare_model_multi_reg` (one
      sub-model per longitude region, weight = `theta_dict['spec_scale']`).
    - Blending the cloudy/clear model in `prepare_model_high_or_low` (two sub-models,
      weights = `cloud_fraction` and `1 - cloud_fraction`).

    Both are a weighted sum of sub-models on the same grid, with no phase-dependent
    kernel involved -- a phase-dependent rotation kernel (Chantier A Phase 3) would be
    applied to each sub-model *before* it reaches this function, not here.

    Parameters
    ----------
    sub_models : sequence of np.ndarray
        Model spectra to combine, all on the same wavelength grid.
    weights : sequence of float or np.ndarray
        One weight per sub-model (scalar, or per-exposure array broadcastable against
        the corresponding sub-model).

    Returns
    -------
    np.ndarray
        `sum(weight_i * sub_model_i)`.
    """
    # Multiply each sub-model by its own weight (broadcasting handles both a single
    # scalar weight and a per-exposure weight array), then sum over the sub-models
    # (axis=0 of the list -- NOT the wavelength axis, which is preserved).
    weighted = [np.asarray(w) * np.asarray(m) for w, m in zip(weights, sub_models)]
    return np.sum(weighted, axis=0)


def _as_value(x):
    """Strip an astropy Unit if present, otherwise return `x` unchanged."""
    return getattr(x, 'value', x)


def precompute_theta_model(
        atmo_obj,
        species: dict,
        planet,
        theta_dict: dict,
        kind_trans: str,
        resolution: float,
        native_resolution: float,
        fct_star=None,
        **retrieval_model_kwargs,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Generate Fp/Fstar for one theta and pre-convolve them once, before Doppler shifts.

    Thin wrapper around `petitradtrans_utils.retrieval_model_plain(..., return_fp_fstar=True)`
    followed by a single degradation step (`convolution.degrade_and_resample`, Chantier A
    Phase 1), applied to Fp (and to Fstar, in emission, so that both end up at the same
    resolution and can be validly divided later) -- mirrors the degradation convention
    already used by `petitradtrans_utils.py::prepare_model` (trim the last native-grid
    point, evaluate the degraded spectrum back onto its own trimmed grid, drop 15 points
    at each edge to discard the convolution boundary).

    This is the "pre-convolution, once per theta" step of Chantier A Phase 2: it runs
    once for a given set of atmospheric parameters, before the per-exposure Doppler loop
    (`build_model_sequence`) -- not once per exposure.

    Parameters
    ----------
    atmo_obj : petitRADTRANS.Radtrans
        Atmosphere object, as expected by `retrieval_model_plain`.
    species : dict
        Species dict, as expected by `retrieval_model_plain`.
    planet : starships.planet_obs.Planet
        Used for `planet.Teff` when falling back to a blackbody stellar spectrum.
    theta_dict : dict
        One region's parameter dict, as produced by `retrieval.py::unpack_theta`. Must
        contain the keys `retrieval_model_plain` expects (`pressures`, `temperatures`,
        `gravity`, `P0`, `p_cloud`, `R_pl`, `R_star`, ...).
    kind_trans : {'transmission', 'emission'}
    resolution : float
        Target resolving power for the pre-convolution step.
    native_resolution : float
        Native/physical resolving power of the petitRADTRANS output (`Rbf`).
    fct_star : callable or 'blackbody' or None, optional
        Forwarded to `retrieval_model_plain` (see `retrieval.py::init_stellar_spectrum`).
    **retrieval_model_kwargs
        Forwarded to `retrieval_model_plain` (`C_to_O`, `Fe_to_H`, `gamma_scat`,
        `kappa_factor`, `specie_2_lnlst`, `dissociation`, ...).

    Returns
    -------
    wave : np.ndarray
    Fp : np.ndarray
        Pre-convolved planet flux (transmission depth in transmission mode).
    Fstar : np.ndarray or None
        Pre-convolved stellar flux, at the same resolution as `Fp`. `None` in
        transmission (no separate stellar flux there).
    """
    # --- Step 1: run petitRADTRANS for this theta, asking for Fp/Fstar separately ---
    # (retrieval_model_plain still returns the old combined ratio by default everywhere
    # else in the package -- return_fp_fstar=True is what this module needs.)
    args = [theta_dict[key] for key in ('pressures', 'temperatures', 'gravity', 'P0',
                                        'p_cloud', 'R_pl', 'R_star')]
    wave, Fp, Fstar = prt.retrieval_model_plain(
        atmo_obj, species, planet, *args,
        kind_trans=kind_trans, fct_star=fct_star, return_fp_fstar=True,
        **retrieval_model_kwargs,
    )

    # --- Step 2: degrade Fp (and Fstar) once, before any Doppler shift ---
    # Trim the last native-grid point (same convention as petitradtrans_utils.prepare_model,
    # petitRADTRANS's own frequency grid has one point of dubious value at the edge).
    wave_trim = wave[:-1]
    # `_as_value` strips the astropy Unit if retrieval_model_plain returned a Quantity
    # (emission does; transmission returns a plain array) -- degrade_and_resample works
    # on plain numpy arrays.
    Fp_trim = _as_value(Fp)[:-1]

    # Degrade the raw (petitRADTRANS-native-resolution) Fp down to `resolution`, evaluated
    # back on its own (trimmed) wavelength grid -- this is the "pre-convolution, once per
    # theta" step, done here rather than in the per-exposure loop of build_model_sequence().
    Fp_pre = degrade_and_resample(wave_trim, Fp_trim, resolution=resolution,
                                  input_resolution=native_resolution, sample=wave_trim)
    Fp_pre = np.ma.masked_invalid(Fp_pre)

    if Fstar is not None:
        # Degrade Fstar with the exact same target resolution/grid as Fp. This matters:
        # Fp and Fstar must end up at the *same* resolution before build_model_sequence()
        # divides one by the other, otherwise the ratio would mix a smooth (still-native)
        # stellar spectrum with a properly-degraded planet spectrum.
        Fstar_trim = _as_value(Fstar)[:-1]
        Fstar_pre = degrade_and_resample(wave_trim, Fstar_trim, resolution=resolution,
                                         input_resolution=native_resolution, sample=wave_trim)
        Fstar_pre = np.ma.masked_invalid(Fstar_pre)
    else:
        # Transmission: no separate stellar flux to degrade.
        Fstar_pre = None

    # Drop the convolution boundary (same 15-point edge trim as prepare_model()) --
    # gauss_convolve()'s 'valid' mode already trims some of it, this removes what's left
    # of the edge region where the kernel didn't have a full window to work with.
    wave_out = wave_trim[15:-15]
    Fp_out = Fp_pre[15:-15]
    Fstar_out = Fstar_pre[15:-15] if Fstar_pre is not None else None

    return wave_out, Fp_out, Fstar_out


def build_model_sequence(
        wave: np.ndarray,
        Fp: np.ndarray,
        data_wave: np.ndarray,
        vrp_orb: ArrayOrScalar,
        Fstar: Optional[np.ndarray] = None,
        vr_orb: ArrayOrScalar = 0.0,
        alpha: ArrayOrScalar = 1.0,
        kind_trans: str = 'emission',
        RV: float = 0.0,
        region_kernel=None,
) -> np.ma.MaskedArray:
    """Build the per-exposure model sequence, Doppler-shifting Fp and Fstar independently.

    Replaces `correlation.py::gen_model_sequence_noinj` / `spectrum.py::quick_inject_clean`
    for callers that need Fp/Fstar kept separate (Chantier A bug #2 fix). Mirrors
    `quick_inject_clean`'s single-spectrum (`P_y.ndim == 1`) branch and injection formula,
    applied independently to Fp (at the planet's velocity) and Fstar (at the star's
    velocity, `vr_orb`, default fixed / not shifted at all -- see below) instead of to an
    already-combined ratio shifted once at a single rigid velocity (the bug).

    Parameters
    ----------
    wave, Fp : np.ndarray
        Pre-convolved planet spectrum (`precompute_theta_model`'s output), one theta.
    data_wave : np.ndarray, shape (n_exp, n_ord, n_pix)
        Instrument wavelength grid to evaluate the model on, one row per exposure.
    vrp_orb : float or np.ndarray, shape (n_exp,)
        Planet orbital velocity, km/s. `RV` (the fitted residual velocity, if any) is
        added to it here, same convention as the old `velocities = RV + vrp_orb - ...`.
    Fstar : np.ndarray, optional
        Pre-convolved stellar spectrum, same wavelength grid as `Fp`. Required in
        emission (the whole point of this function); `None` in transmission, where
        there is no separate stellar flux (`kind_trans='transmission'` then shifts `Fp`
        alone, exactly like the old `quick_inject_clean`).
    vr_orb : float or np.ndarray, shape (n_exp,), default 0.0
        Stellar reflex velocity, km/s. Default: the star is **not** Doppler-shifted at
        all (`vr_orb=0`) -- a deliberate approximation confirmed by Antoine: the reflex
        motion (~0.1-0.2 km/s) is negligible next to the planet's orbital velocity and
        the BERV for essentially every target. Pass the real per-exposure velocity
        (`Observations.vr`, saved by `planet_obs.py::save_sequences` since Chantier A
        Phase 2) for the rare case where full physical accuracy is wanted -- the cost is
        the same either way (see module docstring), this is purely an accuracy choice.
    alpha : float or np.ndarray, shape (n_exp,), default 1.0
        Occultation fraction (`Observations.alpha_frac`). Default 1.0 assumes full
        visibility for every exposure (no eclipse/transit modulation) -- pass the real
        `alpha_frac` to have the injection follow the true light curve.
    kind_trans : {'transmission', 'emission'}
    RV : float, default 0.0
        Residual/fitted radial velocity added to `vrp_orb` (and to `vr_orb`, in
        emission -- a global RV offset shifts everything, star included).
    region_kernel : callable, optional
        Hook for a phase/region-dependent rotation kernel (Chantier A Phase 3, not
        implemented yet): if given, called as `region_kernel(wave, Fp)` and expected to
        return a (possibly per-exposure) convolved `Fp` before the Doppler shift below.
        `None` (default) applies no extra kernel -- this is what keeps this function
        cheap until Phase 3 wires a real kernel in.

    Returns
    -------
    np.ma.MaskedArray, shape (n_exp, n_ord, n_pix)
        Model flux, matching `data_wave`'s shape -- same convention as
        `quick_inject_clean`'s output (an "injected" sequence built from a flat unity
        baseline; PCA removal, needed to compare with real reduced data, is the caller's
        job, exactly as with the function this replaces).
    """
    # Emission needs a real Fstar to divide by -- without it there is nothing to fix
    # (the whole point of this function, vs. the old gen_model_sequence_noinj, is to keep
    # Fp and Fstar apart), so fail loudly instead of silently falling back to Fp alone.
    if kind_trans == 'emission' and Fstar is None:
        raise ValueError("`Fstar` is required when kind_trans='emission' -- pass the "
                          "`Fstar` returned by `precompute_theta_model` (or omit it and "
                          "use kind_trans='transmission' if there truly is no stellar "
                          "contribution to separate).")

    n_exp, n_ord, n_pix = data_wave.shape

    # --- Planet velocity: one value per exposure (vrp_orb + the fitted residual RV) ---
    # A scalar vrp_orb (e.g. a single representative velocity) is broadcast to every
    # exposure; a per-exposure array (the usual case -- vrp_orb varies across a transit)
    # is used as-is. `calc_shift` (same convention as the old quick_inject_clean) turns a
    # km/s velocity into the multiplicative wavelength shift factor sqrt((1+beta)/(1-beta)).
    vrp_orb = np.atleast_1d(np.asarray(vrp_orb, dtype=float)) + RV
    if vrp_orb.size == 1:
        vrp_orb = np.full(n_exp, vrp_orb[0])
    shifts_p = hm.calc_shift(vrp_orb, kind='rel')

    # Optional Chantier A Phase 3 hook: a phase/region-dependent rotation kernel would be
    # convolved into Fp here, before the Doppler shift below. Left as a no-op (Fp
    # unchanged) until Phase 3 actually implements `get_ker`/`degrade_ker`.
    Fp_for_kernel = region_kernel(wave, Fp) if region_kernel is not None else Fp
    # Single spline built once (not per exposure) -- interp1d_masked lets it be evaluated
    # at every exposure's (Doppler-shifted) wavelength grid below in one vectorized call.
    fct_p = interp1d_masked(wave, Fp_for_kernel, kind='cubic', fill_value='extrapolate')

    if Fstar is not None and kind_trans == 'emission':
        # --- Stellar velocity: independent from the planet's (this is the actual bug fix) ---
        # Default vr_orb=0.0 -> shifts_star is (numerically) 1 for every exposure, i.e. the
        # star is not shifted at all (Antoine-approved approximation, see docstring). Passing
        # a real per-exposure `vr` array here instead makes the star follow its true reflex
        # motion -- same cost either way, this is purely an accuracy choice.
        vr_orb_arr = np.atleast_1d(np.asarray(vr_orb, dtype=float)) + RV
        if vr_orb_arr.size == 1:
            vr_orb_arr = np.full(n_exp, vr_orb_arr[0])
        shifts_star = hm.calc_shift(vr_orb_arr, kind='rel')
        fct_star = interp1d_masked(wave, Fstar, kind='cubic', fill_value='extrapolate')

    # --- Occultation fraction (alpha_frac): per-exposure light-curve weight ---
    # Default 1.0 (full visibility, no eclipse/transit modulation) if the caller doesn't
    # pass the real `alpha_frac` -- same broadcasting as the two velocities above.
    alpha_arr = np.atleast_1d(np.asarray(alpha, dtype=float))
    if alpha_arr.size == 1:
        alpha_arr = np.full(n_exp, alpha_arr[0])

    # --- Per-order loop (mirrors quick_inject_clean's structure) ---
    # Only `i_ord` is looped over in plain Python; every exposure within an order is
    # handled in one vectorized call (fct_p/fct_star evaluated on the whole (n_exp, n_pix)
    # sub-array at once), exactly like the function this replaces.
    model_seq = np.empty((n_exp, n_ord, n_pix))
    for i_ord in range(n_ord):
        # Evaluate the (fixed) planet spectrum at each exposure's own, Doppler-shifted
        # wavelength grid: dividing the instrument grid by the shift factor is equivalent
        # to shifting the model spectrum by `-vrp_orb` (same convention as calc_shift).
        fp_shifted = fct_p(data_wave[:, i_ord] / shifts_p[:, None])
        if Fstar is not None and kind_trans == 'emission':
            # Star evaluated at *its own* shift (possibly all-1, i.e. unshifted) --
            # independent from the planet's shift above. This is the actual fix: the old
            # code shifted a single already-combined Fp/Fstar ratio by one rigid velocity.
            fstar_shifted = fct_star(data_wave[:, i_ord] / shifts_star[:, None])
            depth = np.ma.masked_invalid(fp_shifted) / np.ma.masked_invalid(fstar_shifted)
        else:
            # Transmission (or no Fstar at all): nothing to divide, `depth` is just the
            # shifted planet spectrum -- identical to the old quick_inject_clean here.
            depth = np.ma.masked_invalid(fp_shifted)

        # Same injection formula as quick_inject_clean: transmission removes light
        # (1 - alpha*depth, alpha = fraction of the stellar disk occulted by the planet),
        # emission adds light (1 + alpha*depth, alpha = fraction of the dayside visible).
        if kind_trans == 'transmission':
            model_seq[:, i_ord, :] = 1. - alpha_arr[:, None] * depth
        elif kind_trans == 'emission':
            model_seq[:, i_ord, :] = 1. + alpha_arr[:, None] * depth
        else:
            raise ValueError(f"kind_trans must be 'transmission' or 'emission', got {kind_trans!r}")

    return np.ma.masked_invalid(model_seq)
