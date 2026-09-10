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
both for the multi-region combination (``retrieval.py::prepare_model_multi_reg`` --
`theta_regions`/`region_id`, an arbitrary user-defined split of the planet into
regions with independent parameters, not necessarily "citrus" longitude slices) and
for the cloudy/clear blend (previously a special-cased branch in
``prepare_model_high_or_low``) -- the cloudy/clear blend is really just a 2-region case
of the same weighted-sum mechanism, with no phase-dependent kernel.

Notes on cost (raised by Antoine while reviewing this design): keeping Fstar separate
costs one extra cubic-spline evaluation per order compared to the old single-ratio
shift, in both cases fully vectorized across exposures -- a modest, unavoidable price
for actually fixing bug #2 in emission. There is no per-exposure Python loop here, so
whether the star is treated as fixed (default, see ``vr_orb`` below) or shifted by a
real per-exposure velocity does not change the cost of this step.

Chantier A Phase 3 adds two more rotation-kernel mechanisms, deliberately kept at two
different cost tiers depending on whether the kernel varies with orbital phase:

- A *phase-independent* kernel (plain vsini-style broadening in emission, or wind
  broadening in transmission) does not change across a visit, so it is applied once
  per theta in ``precompute_theta_model()`` (its ``rotation_kernel`` argument) --
  same cost tier as the resolution pre-convolution it sits next to.
- A *phase-dependent* kernel (a multi-region kernel -- ``retrieval.py``'s ``get_ker``,
  user-pluggable via ``get_ker_file``, e.g. a citrus/longitude-slice geometry like
  ``spectrum.CitrusRotationKernel``, but nothing here assumes that specific geometry)
  genuinely changes shape per exposure (regions rotating into/out of view), so it
  cannot share one spline across exposures the way the fast path above does --
  ``build_model_sequence()``'s ``region_kernel`` hook, when given, builds one spline
  *per exposure* instead of one shared spline, roughly ``n_exp`` times more expensive
  for that part of the computation. Confirmed acceptable with Antoine (2026-08-27):
  this only affects the opt-in multi-region case, never the default
  (``region_kernel=None``) path used by the rest of the pipeline.
"""
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import astropy.units as u
import astropy.constants as const

from . import homemade as hm
from . import spectrum
from . import transpec
from .extract import quick_norm
from .convolution import degrade_and_resample
from .mask_tools import interp1d_masked
from . import petitradtrans_utils as prt

ArrayOrScalar = Union[float, np.ndarray]


def combine_regions(sub_models: Sequence[np.ndarray], weights: Sequence[ArrayOrScalar]) -> np.ndarray:
    """Weighted sum of sub-model spectra sharing the same wavelength grid.

    Generic replacement for two combinations that used to be special-cased in
    `retrieval.py`:

    - Summing the region contributions in `prepare_model_multi_reg` (one sub-model
      per region -- e.g. a citrus/longitude slice, or any other user-defined split --
      weight = `theta_dict['spec_scale']`).
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


def generate_native_fp_fstar(
        atmo_obj,
        species: dict,
        planet,
        theta_dict: dict,
        kind_trans: str,
        fct_star=None,
        **retrieval_model_kwargs,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Run petitRADTRANS for one theta and trim to the native grid, no degradation.

    This is `precompute_theta_model`'s "Step 1" (petitRADTRANS call + trim the last
    native-grid point), factored out so it can also be reused, un-degraded, by the
    multi-region path (`combine_regions_with_kernel`, Chantier A Phase 3): the
    per-region kernel returned by `retrieval.py`'s `get_ker` already bakes in the
    resolution degradation (`spectrum.BaseKerMulti.degrade_ker`, e.g. via
    `spectrum.CitrusRotationKernel.resample` for the documented citrus example), so
    degrading `Fp` here too, before that kernel is even applied, would degrade it
    twice.

    Parameters
    ----------
    atmo_obj, species, planet, theta_dict, kind_trans, fct_star, retrieval_model_kwargs
        Same as `precompute_theta_model` (forwarded as-is to
        `petitradtrans_utils.retrieval_model_plain`).

    Returns
    -------
    wave : np.ndarray
    Fp : np.ndarray
        Native-resolution planet flux (transmission depth in transmission mode).
    Fstar : np.ndarray or None
        Native-resolution stellar flux. `None` in transmission.
    """
    args = [theta_dict[key] for key in ('pressures', 'temperatures', 'gravity', 'P0',
                                        'p_cloud', 'R_pl', 'R_star')]
    wave, Fp, Fstar = prt.retrieval_model_plain(
        atmo_obj, species, planet, *args,
        kind_trans=kind_trans, fct_star=fct_star, return_fp_fstar=True,
        **retrieval_model_kwargs,
    )

    # Trim the last native-grid point (petitRADTRANS's own frequency grid has one
    # point of dubious value at the edge) -- same convention as
    # petitradtrans_utils.prepare_model/precompute_theta_model.
    wave_trim = wave[:-1]
    Fp_trim = _as_value(Fp)[:-1]
    Fstar_trim = _as_value(Fstar)[:-1] if Fstar is not None else None

    return wave_trim, Fp_trim, Fstar_trim


def combine_regions_with_kernel(wave: np.ndarray, Fp_list: Sequence[np.ndarray],
                                rot_ker_list: Sequence[np.ndarray],
                                weights: Sequence[ArrayOrScalar]) -> np.ndarray:
    """Convolve each region's native Fp with its own kernel and combine (Phase 3).

    Generic multi-region combination step: `theta_regions`/`region_id` split the
    planet into an arbitrary number of regions with independent atmospheric
    parameters (a citrus/longitude-slice geometry is the documented example, via
    `spectrum.CitrusRotationKernel`, but nothing here assumes that specific
    geometry -- `rot_ker_list` can come from any user-supplied `get_ker`).

    Meant to be called once *per exposure*, as the body of a `region_kernel`
    closure passed to `build_model_sequence` (see that function's `region_kernel`
    parameter): a genuinely phase-dependent kernel changes shape with orbital phase
    (regions rotating into/out of view), so this cannot be precomputed once per
    theta the way the phase-independent default kernel (`precompute_theta_model`'s
    `rotation_kernel` argument) is.

    Mirrors the old combined-ratio multi-region path (`retrieval.py::prepare_model_multi_reg`
    + `petitradtrans_utils.prepare_model`'s `rot_ker` argument, applied via
    `spectrum.py::resampling`): each region's kernel
    (e.g. `spectrum.CitrusRotationKernel.resample`) already bakes in both the
    geometric, phase-dependent visibility weighting (`get_ker`'s own
    flux-conservation normalization: the kernels sum to 1 across regions at a given
    phase) *and* the resolution degradation (`spectrum.BaseKerMulti.degrade_ker`) --
    so `Fp_list` must be the *native* (undegraded) flux from
    `generate_native_fp_fstar`, not `precompute_theta_model`'s already-degraded
    output. `weights` (`theta_dict['spec_scale']` per region) is applied on top,
    exactly like the old path's `model_i *= theta_dict['spec_scale']` -- a
    deliberately separate, independent weight from the kernel's own normalization,
    not a double-count of it.

    Parameters
    ----------
    wave : np.ndarray
        Wavelength grid the combined spectrum is returned on. Its length must be
        `len(Fp_list[i]) - 30` (see `Returns` below) -- i.e. the native grid
        produced by `generate_native_fp_fstar`, edge-trimmed by 15 points on each
        side the same way `precompute_theta_model` trims its own output.
    Fp_list : sequence of np.ndarray
        One native-resolution planet-flux array per region (un-trimmed --
        `generate_native_fp_fstar`'s direct output), all on the same native grid.
    rot_ker_list : sequence of np.ndarray
        One rotation-kernel array per region, at the current phase and resampled to
        the same native sampling as `Fp_list` (`retrieval.py`'s
        `get_ker(theta_regions, phase=phase_i, ..., model_resolution=<native res>)`),
        same length as each entry of `Fp_list`.
    weights : sequence of float or np.ndarray
        One weight per region (`theta_dict['spec_scale']`), forwarded to
        `combine_regions`.

    Returns
    -------
    np.ndarray
        Combined planet-flux spectrum for this phase, trimmed by 15 points on each
        side (same convolution-boundary convention as `precompute_theta_model`) so
        its length matches `wave`.
    """
    convolved = [np.convolve(Fp_i, ker_i, mode='same')[15:-15]
                for Fp_i, ker_i in zip(Fp_list, rot_ker_list)]
    return combine_regions(convolved, weights)


def _build_default_rotation_kernel(rotation_kernel: Optional[str], theta_dict: dict, planet,
                                   sampling_resolution: float) -> Optional[np.ndarray]:
    """Build the phase-independent, "once per theta" rotation kernel (Chantier A Phase 3).

    Covers the two cases that do *not* need a per-exposure kernel (see
    `build_model_sequence`'s `region_kernel` for the phase-dependent multi-region
    case, which does): a fixed geometric wind-broadening kernel in transmission, or a
    simple (single-region) solid-rotation kernel in emission -- both are properties of
    the planet's own spectrum, independent of orbital phase, so they only need to be
    computed once per theta, here, rather than once per exposure.

    Parameters
    ----------
    rotation_kernel : {'transmission', 'emission'} or None
        Which default kernel to build (`retrieval.py`'s `rotation_kernel` YAML key).
        `None` (default, and the only option for existing configs that don't set this
        key) means no default kernel -- returns `None`.
    theta_dict : dict
        One region's parameter dict (`retrieval.py::unpack_theta`). Must contain
        `R_pl`, `M_pl`, `T_eq`, `wind` for `'transmission'`, or `R_pl`, `rot_factor`
        for `'emission'`.
    planet : starships.planet_obs.Planet
        Used for `planet.period` (`'emission'` only, to turn `rot_factor` into an
        angular rotation frequency assuming solid/tidally-locked rotation).
    sampling_resolution : float
        Resolving power to resample the kernel onto -- must match the sampling
        density of the array it will be convolved with (`np.convolve`, `mode='same'`
        requires the same grid). Here, that is `Fp` *before* the resolution
        pre-convolution/degrade step (native/dense sampling), not the final
        instrument resolution -- same order as the old `spectrum.resampling`'s
        rot_ker path (convolve first, degrade to instrument resolution after).

    Returns
    -------
    np.ndarray or None
        1D kernel array, ready for `np.convolve(Fp, kernel, mode='same')`. `None` if
        `rotation_kernel` is `None`.
    """
    if rotation_kernel is None:
        return None

    # NOTE on `theta_dict['R_pl']`'s units (found 2026-08-28, alongside the
    # SolidRotationKernel fix below): `retrieval.py::unpack_theta` already converts
    # `R_pl` to cgs centimeters in place (`combined_dict['R_pl'] *= const.R_jup.cgs.value`)
    # before `theta_dict` ever reaches this function -- unlike `M_pl`, which
    # `setup_retrieval` stores as a bare float in Mjup (see the M_pl note below).
    # Both branches used to multiply `theta_dict['R_pl']` by `const.R_jup` again (as
    # if it were still a bare Jupiter-radii count), silently inflating the radius by
    # ~7e9x (verified numerically: 8.6e17 m instead of ~1.2e8 m for a WASP-33b-like
    # case) -- fixed here by attaching the correct existing unit (`u.cm`) instead of
    # re-multiplying by `const.R_jup`.
    if rotation_kernel == 'transmission':
        # Same physical kernel as the (dead) RotKerTransitCloudy(gauss=True) path it
        # replaces -- geometric wind broadening from the planet's own scale height,
        # not an ad hoc gaussian. `omega` convention (wind value in units of 1/day)
        # kept identical to the old `rot_kwargs` in `prepare_model_high_or_low`.
        # NOTE: unlike that old code (which passed `theta_dict['M_pl']` to
        # RotKerTransitCloudy with no unit attached -- theta_dict stores it as a
        # plain float in Mjup, `retrieval.py::setup_retrieval`'s
        # `fixed_params['M_pl'] = planet.M_pl.to('Mjup').value` -- silently giving
        # `g_surf = const.G * pl_mass / pl_rad**2` the wrong units), `const.M_jup` is
        # attached explicitly here.
        ker_obj = spectrum.RotKerTransit(
            theta_dict['R_pl'] * u.cm, theta_dict['M_pl'] * const.M_jup,
            theta_dict['T_eq'] * u.K, np.array([theta_dict['wind']]) / u.day, sampling_resolution,
        )
        kernel = ker_obj.resample(sampling_resolution, n_os=500, pad=7)
    elif rotation_kernel == 'emission':
        # Simple, phase-independent solid-rotation kernel (spectrum.SolidRotationKernel,
        # Chantier A Phase 3 follow-up, 2026-08-28). Used to be built by reusing
        # CitrusRotationKernel with a single citrus boundary ([0.0]), on the assumption
        # that one boundary degenerates into "the whole disk, phase independent" -- that
        # assumption was wrong (Antoine + verified numerically): citrus_to_ker's
        # boundary geometry is built for >= 2 boundaries, and the single
        # self-referencing boundary gave a kernel that was *not* symmetric around v=0,
        # and was entirely zero for any phase other than exactly 0.0 (masked by the
        # existing unit test's unrealistically small v_eq, which fell back to
        # CitrusRotationKernel's own "kernel is zero everywhere -> delta function"
        # safety net regardless of phase). SolidRotationKernel computes the closed-form
        # profile directly instead.
        angular_freq = theta_dict['rot_factor'] * 2 * np.pi / planet.period[0].to('s').value
        ker_obj = spectrum.SolidRotationKernel(
            theta_dict['R_pl'] * u.cm, angular_freq, sampling_resolution,
        )
        kernel = ker_obj.resample(sampling_resolution, n_os=500, pad=7)
    else:
        raise ValueError(f"rotation_kernel must be None, 'transmission' or 'emission', "
                         f"got {rotation_kernel!r}")

    return kernel


def precompute_theta_model(
        atmo_obj,
        species: dict,
        planet,
        theta_dict: dict,
        kind_trans: str,
        resolution: float,
        native_resolution: float,
        fct_star=None,
        rotation_kernel: Optional[str] = None,
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
    rotation_kernel : {'transmission', 'emission'} or None, optional
        Chantier A Phase 3: the phase-*independent* default rotation kernel to apply
        to `Fp` (only) once, here -- see `_build_default_rotation_kernel`. `None`
        (default) applies no kernel, unchanged behaviour for existing configs. For
        the phase-*dependent* multi-region kernel, see `build_model_sequence`'s
        `region_kernel` instead (applied per exposure).
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
    # else in the package -- return_fp_fstar=True is what this module needs.) Factored
    # out into generate_native_fp_fstar() so the multi-region path
    # (combine_regions_with_kernel) can reuse it without the degradation step below.
    wave_trim, Fp_trim, Fstar_trim = generate_native_fp_fstar(
        atmo_obj, species, planet, theta_dict, kind_trans, fct_star=fct_star,
        **retrieval_model_kwargs,
    )

    # --- Optional default rotation kernel (Chantier A Phase 3, phase-independent) ---
    # Convolved at native sampling, *before* degrading to `resolution` -- same order
    # as the old spectrum.resampling(rot_ker=...) path (convolve first, degrade after)
    # and as the composition order documented in the plan (kernel applied to the raw
    # planet spectrum, resolution-degrade applied last). Fstar is never convolved:
    # rotation broadening is a property of the planet's own spectrum only.
    default_kernel = _build_default_rotation_kernel(rotation_kernel, theta_dict, planet,
                                                     sampling_resolution=native_resolution)
    if default_kernel is not None:
        Fp_trim = np.convolve(Fp_trim, default_kernel, mode='same')

    # Degrade the raw (petitRADTRANS-native-resolution) Fp down to `resolution`, evaluated
    # back on its own (trimmed) wavelength grid -- this is the "pre-convolution, once per
    # theta" step, done here rather than in the per-exposure loop of build_model_sequence().
    Fp_pre = degrade_and_resample(wave_trim, Fp_trim, resolution=resolution,
                                  input_resolution=native_resolution, sample=wave_trim)
    Fp_pre = np.ma.masked_invalid(Fp_pre)

    if Fstar_trim is not None:
        # Degrade Fstar with the exact same target resolution/grid as Fp. This matters:
        # Fp and Fstar must end up at the *same* resolution before build_model_sequence()
        # divides one by the other, otherwise the ratio would mix a smooth (still-native)
        # stellar spectrum with a properly-degraded planet spectrum.
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
        phase: Optional[ArrayOrScalar] = None,
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
        For the multi-region case (`region_kernel` given, see below), `Fp` is
        instead a list of native-resolution, per-region planet spectra
        (`model_sequence.generate_native_fp_fstar`'s output, one call per region) --
        this function never reads `Fp` directly in that case, only forwards it
        untouched to `region_kernel`, so the type is opaque to it either way.
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
        Hook for a phase-*dependent* rotation kernel (Chantier A Phase 3), e.g. a
        multi-region kernel (`retrieval.py`'s `get_ker`, user-pluggable): if given,
        called once *per exposure* as `region_kernel(wave, Fp, phase_i)` (`phase_i`
        from `phase` below) and expected to return that exposure's convolved `Fp`,
        evaluated before the Doppler shift. The multi-region case wires this
        up as a closure over `combine_regions_with_kernel` (`retrieval.py`, one call
        per exposure with that exposure's own phase, `Fp` being the per-region list
        described above) -- `region_kernel` always returns the fully convolved and
        combined *spectrum* for that phase (not the raw kernel(s)); this function
        stays unaware of how many regions there are or how they are combined.
        Because the kernel's shape genuinely changes with orbital phase (e.g.
        regions rotating into/out of view), this requires building one
        spline *per exposure* instead of the single shared spline used when
        `region_kernel` is None -- roughly `n_exp` times more expensive for this
        part of the computation (confirmed acceptable with Antoine: this only
        affects the opt-in multi-region case, not the default path). For a
        *phase-independent* kernel (plain vsini-style broadening in emission, or
        wind broadening in transmission), use `precompute_theta_model`'s
        `rotation_kernel` argument instead -- applied once per theta, not once per
        exposure, since it does not vary across a visit. `None` (default) applies
        no extra kernel -- the fast, single-spline path below.
    phase : float or np.ndarray, shape (n_exp,), optional
        Orbital phase of each exposure, forwarded to `region_kernel`. Required if
        `region_kernel` is given (ignored otherwise).

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

    # --- Optional Chantier A Phase 3 hook: phase-dependent rotation kernel ---
    if region_kernel is not None:
        if phase is None:
            raise ValueError("`phase` is required when `region_kernel` is given.")
        phase_arr = np.atleast_1d(np.asarray(phase, dtype=float))
        if phase_arr.size == 1:
            phase_arr = np.full(n_exp, phase_arr[0])
        # One spline *per exposure*: the kernel's shape genuinely changes with phase
        # (e.g. regions rotating into/out of view), so a single shared spline
        # would be wrong here -- unlike the fast path below, this cannot be
        # vectorized across exposures (see docstring for the resulting ~n_exp cost).
        fct_p_per_exp = [
            interp1d_masked(wave, region_kernel(wave, Fp, phase_arr[i_exp]),
                            kind='cubic', fill_value='extrapolate')
            for i_exp in range(n_exp)
        ]
    else:
        # Single spline built once (not per exposure) -- interp1d_masked lets it be
        # evaluated at every exposure's (Doppler-shifted) wavelength grid below in
        # one vectorized call. Fast path used whenever no phase-dependent kernel is
        # requested (the vast majority of calls).
        fct_p = interp1d_masked(wave, Fp, kind='cubic', fill_value='extrapolate')

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
        if region_kernel is not None:
            # No vectorized shortcut here: each exposure has its own spline (built
            # above from its own phase-dependent kernel), so it must be evaluated
            # separately at its own shifted grid.
            fp_shifted = np.ma.array([
                fct_p_per_exp[i_exp](data_wave[i_exp, i_ord] / shifts_p[i_exp])
                for i_exp in range(n_exp)
            ])
        else:
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


def apply_pca_to_model(
    flux: np.ndarray,
    pca,
    reference_spec: Optional[np.ndarray] = None,
    ratio: Optional[np.ndarray] = None,
    n_pca: int = 2,
    n_comps: int = 10,
    norm: bool = True,
    somme: bool = False,
    plot: bool = False,
) -> np.ma.MaskedArray:
    """Apply the same PCA-truncation processing to an already-built model sequence
    that was applied to the real reduced data, so the two are directly comparable.

    Moved here from ``transpec.py`` (Chantier C1) -- despite living next to the real-data
    reduction code, it was only ever called on model sequences (already Doppler-shifted/
    injected by `quick_inject_clean` or `build_model_sequence` -- generating that sequence
    is a distinct, earlier step, not part of this function). Merges what used to be two
    near-duplicate functions, `build_trans_spectrum_mod2` and `build_trans_spectrum_mod_fast`
    -- the latter was exactly the former with `reference_spec`/`ratio` left out (dividing by
    a reference spectrum only matters when `flux` was injected into real reconstructed data;
    the old `gen_model_sequence_noinj`'s default -- injecting into a flat sequence of ones --
    made that division a no-op, hence the separate "fast" copy). `wave`/`noise`, accepted by
    the old `build_trans_spectrum_mod2` but never actually used in its body, are dropped.

    Parameters
    ----------
    flux : np.ndarray
        Model sequence (n_exposures, n_orders, n_pixels), already built (Doppler-shifted,
        combined) -- this function does not generate it.
    pca : sklearn.decomposition.PCA
        Already-fitted PCA object, the same one the real data was truncated with
        (`transpec.apply_pca_truncation`).
    reference_spec : np.ndarray, optional
        Reference spectrum to divide by before PCA removal, matching
        `Observations.reference_spec`. Left out (default) when `flux` was injected into a
        flat baseline of ones, where dividing by it would be a no-op.
    ratio : np.ndarray, optional
        Extra normalization factor, applied (divided) before `reference_spec`.
    n_pca : int, default 2
        Number of PCA components to remove (see `transpec.remove_dem_pca_all`). No PCA
        removal at all when 0.
    n_comps : int, default 10
        Passed through to `transpec.remove_dem_pca_all` (sigma-clipping component count).
    norm : bool, default True
        If True (default), normalize with `quick_norm` (removes the mean, same convention
        as the real data). If False, just divide by the per-exposure/order mean.
    somme : bool, default False
        Passed through to `quick_norm`.
    plot : bool, default False
        Passed through to `transpec.remove_dem_pca_all` (diagnostic plot).

    Returns
    -------
    np.ma.MaskedArray
        PCA-truncated, normalized model sequence, same shape as `flux`.
    """
    flux_norm = flux / np.ma.median(flux, axis=-1)[:, :, None]
    if ratio is not None:
        flux_norm = flux_norm / ratio
    if reference_spec is not None:
        flux_norm = flux_norm / reference_spec

    if n_pca > 0:
        full_ts, _, _ = transpec.remove_dem_pca_all(
            flux_norm, pca=pca, n_pcs=n_pca, n_comps=n_comps, plot=plot)
    else:
        full_ts = flux_norm

    if norm:
        final_ts = quick_norm(full_ts, somme=somme, take_all=False)
    else:
        final_ts = full_ts / np.ma.mean(full_ts, axis=-1)[:, :, None]

    return final_ts


def gen_model_sequence_noinj(
    velocities: np.ndarray,
    data_wave: Optional[np.ndarray] = None,
    data_sep: Optional[np.ndarray] = None,
    data_pca=None,
    data_npc: Optional[int] = None,
    planet=None,
    model_wave: Optional[np.ndarray] = None,
    model_spec: Optional[np.ndarray] = None,
    alpha: Optional[np.ndarray] = None,
    data_visit: Optional[dict] = None,
    data_recon: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ma.MaskedArray:
    """Old combined-ratio model sequence: inject one model spectrum at a single rigid velocity.

    Moved here from ``correlation.py`` (Chantier C1) -- it belongs with model generation,
    not correlation. Kept only as a fallback path in ``logl_grid.py::_get_chi2_detailed``
    for older saved model files that don't have ``Fp_high``/``Fstar_high`` split apart yet
    (see ``load_model``). New code should use ``build_model_sequence`` instead, which
    fixes Chantier A bug #2 (Fp/Fstar Doppler-shifted independently per exposure) -- this
    function still has that bug for any model file that actually falls back to it.

    Parameters
    ----------
    velocities : np.ndarray
        Per-exposure velocities (km/s) at which to inject the model.
    data_wave, data_sep, data_pca, data_npc : optional
        Data-side quantities needed by the injection/PCA-removal step. Any left as
        `None` are read from `data_visit` instead (see below).
    planet : Planet
        Used for `R_star`/`A_star`/`R_pl`.
    model_wave, model_spec : np.ndarray
        Native-resolution model wavelength/spectrum to inject.
    alpha : np.ndarray of shape (n_spec,), optional
        Fraction of planetary signal actually visible. Depends on `kind_trans`:
        - transmission: fraction of the stellar disk hidden by the planet.
        - emission: fraction of the planetary disk not hidden by the star.
    data_visit : dict, optional
        Fallback source for `data_wave`/`data_sep`/`data_pca`/`data_npc` when those
        aren't passed directly (see `planet_obs.py::_visit_to_data_dict`).
    data_recon : np.ndarray, optional
        Reconstructed data sequence to inject the model into. Defaults to an array of
        ones (inject into an otherwise-flat sequence), which is what every real caller
        uses today.
    **kwargs
        Passed through to `spectrum.quick_inject_clean`.

    Returns
    -------
    np.ma.MaskedArray
        Model sequence, PCA-truncated the same way the data was.
    """
    if data_wave is None:
        data_wave = data_visit['wave']

    if data_sep is None:
        data_sep = data_visit['sep']

    if data_pca is None:
        data_pca = data_visit['pca']
    if data_npc is None:
        data_npc = int(data_visit['params'][5])
    if data_recon is None:
        # -- Uncomment the other 2 lines if you want the full reconstructed data to inject the model in
        data_recon = np.ones_like(data_wave)
        # data_recon = data_visit['reconstructed']
        # data_recon = data_recon/np.ma.median(data_recon,axis=-1)[:,:,None]/data_visit['ratio']/data_visit['reference_spec']

    for arg in (planet, model_wave, model_spec):
        if arg is None:
            raise ValueError('`planet`, `model_wave` and `model_spec` need to be specified.')

    # --- inject model in an empty sequence of ones
    flux_inj, _ = spectrum.quick_inject_clean(data_wave, data_recon,
                                               model_wave, model_spec,
                                               velocities, data_sep, planet.R_star, planet.A_star,
                                               RV=0.0, dv_star=0.,
                                               R0=planet.R_pl, alpha=alpha, **kwargs)

    # -- Remove the same number of pcas that were used to inject
    model_seq = apply_pca_to_model(flux_inj, data_pca, n_pca=data_npc)

    return model_seq
