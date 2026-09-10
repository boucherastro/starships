"""
logl_grid — Kp × v_sys logL grid computation for high-resolution spectroscopy.

Workflow
--------
1. Compute::

    setup_logl_grid(yaml) → load_logl_grid_data() → load_model(npz)
    → results = compute_logl_grid() → save_logl_grid(results, out_path, stem)

2. Analyse::

    load_logl_results(file_list, path)
    logl = get_logl(alpha=1., sum_axis=(-2, -1))

The module uses module-level globals (same pattern as starships.retrieval) so
that multiprocessing workers inherit large arrays without memory duplication via
fork.

Extra keys to add to the retrieval YAML
----------------------------------------
::

    rv_grid:  {min: -200, max: 200,  step: 5}   # km/s
    kp_grid:  {min: 0,   max: 500,  step: 5}    # km/s
    logl_kind: BL                                # BL or G
    logl_grid_output_path: ~/scratch/DataAnalysis/SPIRou/logl_grids
    n_processes_per_cpu: 3
    apply_alpha: true                            # true = modulate template by alpha_frac (realistic);
                                                 # false = uniform injection (all frames equal weight)
    use_real_stellar_rv: false                   # Chantier A Phase 2, Fp/Fstar-separated model files
                                                 # only: true = Doppler-shift Fstar at the star's real
                                                 # per-exposure reflex velocity (needs `vr` in the high-res
                                                 # .npz); false (default) = keep the star fixed beyond its
                                                 # RV_const baseline (negligible reflex motion in practice)

Output filenames encode the grid parameters so that different grids for the
same model do not overwrite each other::

    {model_stem}_kp{min}_{max}_{step}_rv{min}_{max}_{step}[_noalpha]_visit{i}.npz

Negative values are written with a leading ``m`` (e.g., -100 → m100).

Command-line usage (Narval)
---------------------------
::

    python -m starships.logl_grid \
        --config ~/scratch/.../retrieval_inputs.yaml \
        --specfile take3_HRR_all.npz
"""

import logging
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from astropy import units as u

import starships.planet_obs as pl_obs
from starships import model_sequence as model_seq
from starships.orbite import rv_theo_t
from starships.planet_obs import Observations

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level globals
# Declared here for discoverability; set by setup_logl_grid / load_* functions.
# ---------------------------------------------------------------------------

# --- From setup_logl_grid (YAML keys) ---
global pl_name, pl_params
global high_res_file_stem_list, high_res_path
global instrum, instrum_param_list, wv_range_high
global kind_trans, orders
# logl_grid-specific
global rv_grid, kp_grid, logl_kind
global logl_grid_output_path, n_processes_per_cpu
global apply_alpha, use_real_stellar_rv
# Derived in setup_logl_grid
global obs, planet, Kp_scale

# --- From load_logl_grid_data ---
global data_visits, data_info_list, idx_orders, axis_sum

# --- From load_model ---
global wv_high, model_high
# Fp/Fstar kept separate (Chantier A Phase 2) -- None if the loaded model .npz uses
# the older, combined-ratio-only format (see load_model()).
global Fp_high, Fstar_high

# --- Workers (set per-visit before Pool.map) ---
global _current_data_visit, _current_alpha_frac


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

# NOTE: This function must live in *this* module so that globals() modifies
# logl_grid's own namespace (same trick as retrieval.setup_retrieval).
def setup_logl_grid(input_parameters, **kwargs):
    """Set up module globals for logl grid computation.

    Reads the same YAML as ``setup_retrieval``, plus the logl_grid-specific
    keys listed in the module docstring.
    """
    from starships.retrieval import unpack_input_parameters, get_wv_range
    from starships.instruments import load_instrum

    input_params = unpack_input_parameters(input_parameters, **kwargs)

    _KEYS = [
        'pl_name', 'pl_params',
        'high_res_file_stem_list', 'high_res_path',
        'instrum', 'kind_trans', 'orders',
        'rv_grid', 'kp_grid', 'logl_kind',
        'logl_grid_output_path', 'n_processes_per_cpu',
        'apply_alpha',
        # B3 (Chantier B): PCA truncation happens at read time now (see
        # save_reduced_sequence/load_reduced_sequence) -- n_pc must come from the config
        # (broadcast to match high_res_file_stem_list by unpack_input_parameters above),
        # not from the loaded file itself.
        'n_pc',
    ]
    for key in _KEYS:
        if key in input_params:
            globals()[key] = input_params[key]

    # Defaults for optional logl_grid keys
    global rv_grid, kp_grid, logl_kind, n_processes_per_cpu, apply_alpha, use_real_stellar_rv
    if 'rv_grid' not in globals():
        rv_grid = None
    if 'kp_grid' not in globals():
        kp_grid = None
    logl_kind = input_params.get('logl_kind', 'BL')
    n_processes_per_cpu = input_params.get('n_processes_per_cpu', 3)
    apply_alpha = bool(input_params.get('apply_alpha', True))
    # Chantier A Phase 2: same option as retrieval.py::setup_retrieval -- Doppler-shift
    # Fstar at the star's real per-exposure reflex velocity (needs `vr` in the loaded
    # .npz, see planet_obs.py::save_sequences) instead of keeping it fixed. Default
    # False: negligible next to the planet's orbital velocity and the BERV for
    # essentially every target (Antoine-confirmed approximation).
    use_real_stellar_rv = bool(input_params.get('use_real_stellar_rv', False))

    # Build planet / obs
    global obs, planet, Kp_scale
    obs = Observations(name=input_params['pl_name'], pl_kwargs=input_params['pl_params'])
    planet = obs.planet
    Kp_scale = (planet.M_pl / planet.M_star).decompose().value
    planet.all_params = None  # free unused memory

    # Instrument info
    global instrum_param_list, wv_range_high
    instrum_param_list = [load_instrum(name) for name in input_params['instrum']]
    wv_range_high = get_wv_range([p['high_res_wv_lim'] for p in instrum_param_list])
    log.info(f'Wavelength range (high res): {wv_range_high}')


def load_logl_grid_data():
    """Load high-res data into globals. Run after setup_logl_grid."""
    global data_visits, data_info_list, idx_orders, axis_sum

    data_visits = []
    data_info_list = []

    for high_res_file_stem, n_pc_i in zip(high_res_file_stem_list, n_pc):
        log.info(f'Loading: {high_res_path / high_res_file_stem}')
        # B3: n_pc applied at read time (one value per file); reuse `planet` (config
        # pl_params overrides applied) instead of a fresh ExoFile lookup per visit.
        data_info_i, data_visit_i = pl_obs.load_sequences(
            high_res_file_stem, n_pc_i, path=high_res_path, planet=planet
        )
        data_visits.append(data_visit_i)
        data_info_list.append(data_info_i)

    n_ord = data_visits[0]['flux'].shape[1]
    idx_orders = orders if orders is not None else np.arange(n_ord)
    axis_sum = -1  # sum over spectral pixel axis


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(model_path):
    """Load a model spectrum NPZ.

    Sets ``wv_high``/``model_high`` globals (combined ratio, as before). If the file
    also has ``fp_mod``/``fstar_mod`` keys -- the format written when the model was
    generated with Fp/Fstar kept separate (Chantier A Phase 2, see
    `model_sequence.py::precompute_theta_model`) -- also sets ``Fp_high``/
    ``Fstar_high``, which `_get_chi2_detailed` then uses to fix bug #2 (star's reflex
    RV getting dragged at the planet's orbital velocity). Older model files without
    those keys still work exactly as before, just without the fix (``Fp_high``/
    ``Fstar_high`` stay `None`).
    """
    global wv_high, model_high, Fp_high, Fstar_high
    model_file = np.load(model_path)
    try:
        wv_high = model_file['wave']
    except KeyError:
        wv_high = model_file['wave_mod']

    if 'fp_mod' in model_file and 'fstar_mod' in model_file:
        Fp_high = model_file['fp_mod']
        Fstar_high = model_file['fstar_mod']
        # Kept for anything downstream that still expects the combined ratio
        # (diagnostics, plotting) -- computed here rather than saved twice.
        model_high = Fp_high / Fstar_high
    else:
        Fp_high = None
        Fstar_high = None
        try:
            model_high = model_file['spec']
        except KeyError:
            model_high = model_file['mod_spec']

    model_file.close()
    if not np.isfinite(model_high[100:-100]).all():
        raise ValueError(f'NaN found in model spectrum: {model_path}')
    log.info(f'Model loaded: {Path(model_path).name}'
            + (' (Fp/Fstar separated)' if Fp_high is not None else ' (combined ratio only)'))


# ---------------------------------------------------------------------------
# Chi2 terms (per-exposure, per-order)
# ---------------------------------------------------------------------------

def _calc_chi2_terms(model, data_visit):
    """Compute model-dependent chi2 terms (f×g, g²) for one visit."""
    flux = data_visit['flux'][:, idx_orders]
    noise = data_visit['noise'][:, idx_orders]
    model_norm = model[:, idx_orders] / noise
    f_x_g = np.ma.sum(model_norm * flux, axis=axis_sum)
    s2g = np.ma.sum(model_norm ** 2, axis=axis_sum)
    return f_x_g, s2g


# Set per-visit in compute_logl_grid before Pool.map; inherited by workers via fork.
_current_data_visit = None
_current_alpha_frac = None
_current_apply_alpha = True


def _get_chi2_detailed(theta):
    """Worker: compute chi2 terms for one (v_sys, kp) grid point.

    Uses the Fp/Fstar-separated engine (`model_sequence.py`, Chantier A Phase 2,
    fixes bug #2) when the loaded model has `Fp_high`/`Fstar_high` (see `load_model`);
    falls back to the old combined-ratio path (`model_sequence.py::gen_model_sequence_noinj`,
    moved out of `correlation.py` in Chantier C1) for older model files that only have the
    combined ratio.
    """
    v_sys, kp = theta
    data_visit = _current_data_visit

    # Planet's velocity relative to the star (not to the observer -- see
    # model_sequence.py / retrieval.py::lnprob for the same composition).
    vrp_orb = rv_theo_t(
        kp, data_visit['t_start'] * u.d, planet.mid_tr, planet.period, plnt=True
    ).value

    n_pc = int(data_visit['params'][5])
    alpha_arg = _current_alpha_frac if _current_apply_alpha else np.ones_like(_current_alpha_frac)

    if Fp_high is not None:
        # Chantier A Phase 2: Fp/Fstar kept separate, Doppler-shifted independently
        # per exposure -- fixes bug #2 (star's tiny reflex RV dragged at the
        # planet's orbital velocity by the old rigid combined-ratio shift below).
        # Same defaults/composition as retrieval.py::lnprob: star assumed fixed
        # beyond its RV_const baseline unless use_real_stellar_rv is set and the
        # loaded data actually has a per-exposure `vr` (older .npz files don't).
        if use_real_stellar_rv and data_visit.get('vr') is not None:
            vr_orb = data_visit['vr'].to(u.km / u.s).value
        else:
            vr_orb = 0.0

        model_seq_arr = model_seq.build_model_sequence(
            wv_high[20:-20], Fp_high[20:-20], data_visit['wave'], vrp_orb,
            Fstar=Fstar_high[20:-20], vr_orb=vr_orb, alpha=alpha_arg,
            kind_trans=kind_trans, RV=v_sys + data_visit['RV_const'])
    else:
        # Older model file (combined ratio only, see load_model) -- old path,
        # unchanged, same bug as before for this particular model file.
        velocities = v_sys + vrp_orb - vrp_orb * Kp_scale + data_visit['RV_const']
        model_seq_arr = model_seq.gen_model_sequence_noinj(
            velocities,
            data_wave=data_visit['wave'],
            data_sep=data_visit['sep'],
            data_pca=data_visit['pca'],
            data_npc=n_pc,
            planet=planet,
            model_wave=wv_high[20:-20],
            model_spec=model_high[20:-20],
            kind_trans=kind_trans,
            alpha=alpha_arg,
        )
    return _calc_chi2_terms(model_seq_arr, data_visit)


# ---------------------------------------------------------------------------
# Grid computation
# ---------------------------------------------------------------------------

def _make_grid(grid_cfg):
    # step/2 as the upper-bound buffer: includes max without ever adding an extra point.
    # Using step itself as the buffer causes np.arange to overshoot when max is not an
    # exact multiple of step (e.g. min=227, max=228, step=5 would give [227, 232]).
    return np.arange(grid_cfg['min'], grid_cfg['max'] + grid_cfg['step'] / 2, grid_cfg['step'])


def estimate_logl_grid_memory(n_vsys, n_kp, n_exp_per_visit, n_ord,
                               n_visits=1, n_alpha=31, dtype_bytes=8):
    """Print a memory budget for logl grid computation and analysis.

    Helps decide before running whether available RAM is sufficient.

    Parameters
    ----------
    n_vsys, n_kp : int — grid dimensions (number of vsys / Kp points)
    n_exp_per_visit : int — number of exposures per visit
    n_ord : int — number of spectral orders kept
    n_visits : int — number of visits (default 1)
    n_alpha : int — alpha values for marginalization (default 31)
    dtype_bytes : int — bytes per float (8 for float64)
    """
    def _fmt(n_bytes):
        if n_bytes < 1e6:
            return f'{n_bytes / 1e3:.1f} KB'
        if n_bytes < 1e9:
            return f'{n_bytes / 1e6:.0f} MB'
        return f'{n_bytes / 1e9:.2f} GB'

    n_exp_total  = n_visits * n_exp_per_visit
    grid_cell    = n_vsys * n_kp * n_ord * dtype_bytes  # base unit: one (vsys,kp,ord) array

    # ── Grid computation (compute_and_save_logl_grid, one visit at a time) ──
    # pool.map returns a list (n_vsys*n_kp,) of 2×n_exp×n_ord arrays.
    # np.array(outputs) allocates all of them contiguously.
    outputs_mem      = 2 * grid_cell * n_exp_per_visit   # shape (n_vsys*n_kp, 2, n_exp, n_ord)
    visit_terms_mem  = 2 * grid_cell * n_exp_per_visit   # cross_terms + squared_terms after reshape
    compute_peak     = outputs_mem + visit_terms_mem      # both exist simultaneously before del outputs

    # ── Analysis (load_logl_results + get_logl) ──
    # Loaded globals: cross_terms + squared_terms, all visits concatenated along exp axis.
    globals_mem = 2 * grid_cell * n_exp_total   # dominant term
    extras_mem  = globals_mem * 0.15            # s2f, N, uncert_sum, phase … ≈ 15 % overhead
    logl_cube   = n_alpha * n_vsys * n_kp * dtype_bytes
    analysis_peak = globals_mem + extras_mem + logl_cube

    width = 58
    sep   = '─' * width
    print(sep)
    print('logL grid — memory estimate')
    print(sep)
    print(f'  Grid    : {n_vsys} vsys × {n_kp} Kp  ({n_vsys*n_kp:,} points)')
    print(f'  Data    : {n_exp_per_visit} exp/visit × {n_visits} visit(s), {n_ord} orders')
    print(f'  α array : {n_alpha} values')
    print(sep)
    print('COMPUTATION  (compute_and_save_logl_grid — per visit)')
    print(f'  pool.map outputs         : {_fmt(outputs_mem)}')
    print(f'  cross + squared terms    : {_fmt(visit_terms_mem)}')
    print(f'  Peak (before del outputs): {_fmt(compute_peak)}')
    print()
    print('ANALYSIS  (load_logl_results + _build_logl_cube)')
    print(f'  Loaded globals (all visits): {_fmt(globals_mem + extras_mem)}')
    print(f'    cross_terms + squared    : {_fmt(globals_mem)}')
    print(f'    other arrays (≈15 %)     : {_fmt(extras_mem)}')
    print(f'  logL cube ({n_alpha}×{n_vsys}×{n_kp})    : {_fmt(logl_cube)}')
    print(f'  Analysis peak            : {_fmt(analysis_peak)}')
    print(sep)
    overall_peak = max(compute_peak, analysis_peak)
    print(f'  Overall peak             : {_fmt(overall_peak)}')
    print(f'  Recommended RAM          : ≥ {_fmt(overall_peak * 1.5)}  (×1.5 safety margin)')
    print(sep)


def compute_logl_grid(rv_array=None, kp_array=None, n_process=None):
    """Run Pool.map over (rv, kp) grid for all loaded visits.

    Returns
    -------
    list of dict
        One dict per visit with keys: cross_terms, squared_terms, kp, vsys.
    """
    global _current_data_visit, _current_alpha_frac, _current_apply_alpha

    if rv_array is None:
        rv_array = _make_grid(rv_grid)
    if kp_array is None:
        kp_array = _make_grid(kp_grid)
    if n_process is None:
        try:
            n_cpu = int(os.environ['SLURM_CPUS_PER_TASK'])
        except KeyError:
            n_cpu = 1
        n_process = n_cpu * n_processes_per_cpu

    kp_step  = kp_array[1]  - kp_array[0]  if len(kp_array) > 1 else float('nan')
    rv_step  = rv_array[1]  - rv_array[0]  if len(rv_array) > 1 else float('nan')
    log.info(
        f'Kp  grid: [{kp_array[0]:.2f}, {kp_array[-1]:.2f}] km/s  '
        f'step={kp_step:.2f}  n={len(kp_array)}'
    )
    log.info(
        f'vsys grid: [{rv_array[0]:.2f}, {rv_array[-1]:.2f}] km/s  '
        f'step={rv_step:.2f}  n={len(rv_array)}'
    )
    log.info(
        f'Grid size: {len(rv_array) * len(kp_array)} points  '
        f'({len(rv_array)} × {len(kp_array)})'
    )
    log.info(f'apply_alpha = {apply_alpha}  (template modulated by alpha_frac: {apply_alpha})')
    log.info(f'use_real_stellar_rv = {use_real_stellar_rv} '
            f'(Fp/Fstar engine only, ignored for older combined-ratio-only model files)')

    _current_apply_alpha = apply_alpha

    kp_mesh, vsys_mesh = np.meshgrid(kp_array, rv_array)
    theta_grid = np.array([np.ravel(vsys_mesh), np.ravel(kp_mesh)]).T

    results = []
    for i, (data_visit, di) in enumerate(zip(data_visits, data_info_list), start=1):
        _current_data_visit = data_visit
        _current_alpha_frac = di['all_alpha_frac']

        log.info(f'Computing grid for visit {i}/{len(data_visits)} with {n_process} processes ...')
        with Pool(n_process) as pool:
            outputs = pool.map(_get_chi2_detailed, theta_grid)

        outputs = np.array(outputs)
        data_shape = outputs.shape[2:]
        cross_terms, squared_terms = [
            np.reshape(outputs[:, idx], (*kp_mesh.shape, *data_shape))
            for idx in range(2)
        ]
        results.append(dict(
            cross_terms=cross_terms,
            squared_terms=squared_terms,
            kp=kp_mesh,
            vsys=vsys_mesh,
        ))
    return results


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def _fmt_v(v):
    """Format a grid value for use in a filename (no '.', 'm' for minus)."""
    v = float(v)
    iv = int(v)
    s = str(iv) if iv == v else f'{v:.3g}'.replace('.', 'p')
    return s.replace('-', 'm')


def _build_grid_stem_suffix(rv_array, kp_array, apply_alpha_flag):
    """Build the grid-parameter suffix for the output filename.

    Example: ``_kp200_300_5_rvm100_100_1`` or ``_kp200_300_5_rvm100_100_1_noalpha``
    """
    kp_step = kp_array[1] - kp_array[0] if len(kp_array) > 1 else 0
    rv_step = rv_array[1] - rv_array[0] if len(rv_array) > 1 else 0
    suffix = (
        f'_kp{_fmt_v(kp_array[0])}_{_fmt_v(kp_array[-1])}_{_fmt_v(kp_step)}'
        f'_rv{_fmt_v(rv_array[0])}_{_fmt_v(rv_array[-1])}_{_fmt_v(rv_step)}'
    )
    if not apply_alpha_flag:
        suffix += '_noalpha'
    return suffix


def _compute_contact_phases(planet_obj, kind_trans_str):
    """Return contact phases [T1, T2, T3, T4] in phase units.

    T1/T4 bracket the full transit/eclipse (external contacts).
    T2/T3 bracket the fully-in phase (internal contacts).

    Phases are relative to phase 0 for transmission and phase 0.5 for
    emission (secondary eclipse centred at 0.5).

    If the orbital geometry does not allow computing T2/T3 (grazing or
    missing parameters), T2 = T1 and T3 = T4.
    """
    # Force scalar floats: planet Quantity values can be 1-element arrays.
    T14_phase = float((planet_obj.trandur / planet_obj.period).decompose().value)

    phase_T1 = -T14_phase / 2
    phase_T4 = +T14_phase / 2
    phase_T2, phase_T3 = phase_T1, phase_T4  # fallback

    try:
        k       = float((planet_obj.R_pl / planet_obj.R_star).decompose().value)
        # Impact parameter for a circular orbit: b = a·cos(i)/R_star
        b       = float((planet_obj.ap * np.cos(planet_obj.incl) / planet_obj.R_star).decompose().value)
        # Scaled semi-major axis along the sky plane: (a·sin i)/R_star
        a_sin_i = float((planet_obj.ap * np.sin(planet_obj.incl) / planet_obj.R_star).decompose().value)

        # Winn (2010) Eq. 14-15: T_ij = (P/π)·arcsin(sqrt((1±k)²−b²) / (a sin i / R_star))
        # The arcsin argument must be divided by a_sin_i; omitting this factor makes
        # the argument > 1 for typical hot Jupiters (a/R_star ≈ 10), giving arcsin = NaN.
        arg_14 = np.sqrt(max(0., (1 + k) ** 2 - b ** 2)) / a_sin_i
        arg_23 = np.sqrt(max(0., (1 - k) ** 2 - b ** 2)) / a_sin_i

        if 0 < arg_14 < 1 and 0 < arg_23 < 1:
            T23_phase = T14_phase * np.arcsin(arg_23) / np.arcsin(arg_14)
            phase_T2 = -T23_phase / 2
            phase_T3 = +T23_phase / 2
    except Exception as exc:
        log.debug(f'Could not compute T23: {exc}')

    if kind_trans_str == 'emission':
        return np.array([0.5 + phase_T1, 0.5 + phase_T2, 0.5 + phase_T3, 0.5 + phase_T4])
    return np.array([phase_T1, phase_T2, phase_T3, phase_T4])


def _save_visit_result(visit_result, data_visit, di, output_path,
                        file_stem, grid_suffix, contact_phases, visit_idx):
    """Write one visit's grid result to an NPZ file.

    Extracted so that both ``save_logl_grid`` and
    ``compute_and_save_logl_grid`` can share the same serialisation logic.
    """
    noise = data_visit['noise'][:, idx_orders]
    flux  = data_visit['flux'][:, idx_orders]
    uncert_sum_v = np.sum(np.ma.log(noise), axis=axis_sum)
    s2f_v        = np.sum(flux ** 2,        axis=axis_sum)

    # Phase is not stored in data_visit by load_sequences — compute it from
    # t_start and the planet orbital parameters (set by setup_logl_grid).
    t_start = data_visit['t_start']
    phase = ((t_start * u.d - planet.mid_tr) / planet.period).decompose().value
    phase -= np.round(phase.mean())
    if kind_trans == 'emission':
        # Emission: secondary eclipse at phase 0.5 → shift to [0, 1] range.
        if (phase < 0).all():
            phase += 1.0

    saved = dict(
        **visit_result,
        alpha_frac=di['all_alpha_frac'],
        icorr=di['all_icorr'],
        N=di['all_N'],
        bad_indexs=np.empty(0),
        s2f=s2f_v,
        uncert_sum=uncert_sum_v,
        phase=phase,
        t_start=t_start,
        contact_phases=contact_phases,
        kind_trans=np.array([kind_trans]),
        apply_alpha=np.array([apply_alpha]),
    )

    for key, val in list(saved.items()):
        if not isinstance(val, np.ndarray):
            log.warning(f'{key}: not an ndarray, saving as empty.')
            saved[key] = np.empty(0)
        elif val.dtype == object:
            log.warning(f'{key}: dtype=object, saving as empty.')
            saved[key] = np.empty(0)

    filename = f'{file_stem}{grid_suffix}_visit{visit_idx}.npz'
    np.savez(output_path / filename, **saved)
    log.info(f'Saved: {output_path / filename}')


def save_logl_grid(results, output_path, file_stem):
    """Save grid results to NPZ, one file per visit.

    The filename embeds grid parameters so that runs with different grids
    for the same model do not overwrite each other.

    Note: ``results`` must already be fully computed and held in memory.
    For large grids or many visits, prefer ``compute_and_save_logl_grid``
    which writes each visit to disk before computing the next.
    """
    output_path = Path(output_path).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    vsys_ax = results[0]['vsys'][:, 0]
    kp_ax   = results[0]['kp'][0, :]
    grid_suffix = _build_grid_stem_suffix(vsys_ax, kp_ax, apply_alpha)

    contact_phases = _compute_contact_phases(planet, kind_trans)
    log.info(
        'Contact phases  [T1, T2, T3, T4]: '
        + ', '.join(f'{p:.4f}' for p in contact_phases)
    )

    for i, (data_visit, di, visit_res) in enumerate(
        zip(data_visits, data_info_list, results), start=1
    ):
        _save_visit_result(visit_res, data_visit, di, output_path,
                           file_stem, grid_suffix, contact_phases, i)


def compute_and_save_logl_grid(output_path, file_stem,
                                rv_array=None, kp_array=None, n_process=None):
    """Compute the logl grid and save each visit immediately after computation.

    Memory-efficient alternative to ``compute_logl_grid`` + ``save_logl_grid``:
    each visit's arrays are written to disk and freed before the next visit is
    computed.  Peak RAM usage is proportional to *one* visit's grid rather than
    all visits combined.

    Parameters
    ----------
    output_path : str or Path
    file_stem : str — model filename stem (used as NPZ filename prefix)
    rv_array, kp_array : 1-D arrays, optional — override YAML grid definitions
    n_process : int, optional — number of worker processes (default: SLURM_CPUS_PER_TASK × n_processes_per_cpu)
    """
    global _current_data_visit, _current_alpha_frac, _current_apply_alpha

    if rv_array is None:
        rv_array = _make_grid(rv_grid)
    if kp_array is None:
        kp_array = _make_grid(kp_grid)
    if n_process is None:
        try:
            n_cpu = int(os.environ['SLURM_CPUS_PER_TASK'])
        except KeyError:
            n_cpu = 1
        n_process = n_cpu * n_processes_per_cpu

    kp_step = kp_array[1] - kp_array[0] if len(kp_array) > 1 else float('nan')
    rv_step = rv_array[1] - rv_array[0] if len(rv_array) > 1 else float('nan')
    log.info(
        f'Kp  grid: [{kp_array[0]:.2f}, {kp_array[-1]:.2f}] km/s  '
        f'step={kp_step:.2f}  n={len(kp_array)}'
    )
    log.info(
        f'vsys grid: [{rv_array[0]:.2f}, {rv_array[-1]:.2f}] km/s  '
        f'step={rv_step:.2f}  n={len(rv_array)}'
    )
    log.info(
        f'Grid size: {len(rv_array) * len(kp_array)} points  '
        f'({len(rv_array)} × {len(kp_array)})'
    )
    log.info(f'apply_alpha = {apply_alpha}')
    log.info(f'use_real_stellar_rv = {use_real_stellar_rv}')

    _current_apply_alpha = apply_alpha

    kp_mesh, vsys_mesh = np.meshgrid(kp_array, rv_array)
    theta_grid = np.array([np.ravel(vsys_mesh), np.ravel(kp_mesh)]).T

    output_path = Path(output_path).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    grid_suffix = _build_grid_stem_suffix(rv_array, kp_array, apply_alpha)

    contact_phases = _compute_contact_phases(planet, kind_trans)
    log.info(
        'Contact phases  [T1, T2, T3, T4]: '
        + ', '.join(f'{p:.4f}' for p in contact_phases)
    )

    for i, (data_visit, di) in enumerate(zip(data_visits, data_info_list), start=1):
        _current_data_visit = data_visit
        _current_alpha_frac = di['all_alpha_frac']

        log.info(
            f'Computing grid for visit {i}/{len(data_visits)} '
            f'with {n_process} processes ...'
        )
        with Pool(n_process) as pool:
            outputs = pool.map(_get_chi2_detailed, theta_grid)

        outputs = np.array(outputs)
        data_shape = outputs.shape[2:]
        cross_terms_v, squared_terms_v = [
            np.reshape(outputs[:, idx], (*kp_mesh.shape, *data_shape))
            for idx in range(2)
        ]
        del outputs  # free raw worker output before saving

        visit_result = dict(
            cross_terms=cross_terms_v,
            squared_terms=squared_terms_v,
            kp=kp_mesh,
            vsys=vsys_mesh,
        )
        _save_visit_result(visit_result, data_visit, di, output_path,
                           file_stem, grid_suffix, contact_phases, i)
        del cross_terms_v, squared_terms_v, visit_result
        log.info(f'Visit {i} written and freed from memory.')


# ---------------------------------------------------------------------------
# Post-processing — load results + compute logL / CCF
# ---------------------------------------------------------------------------

# Module globals populated by load_logl_results
cross_terms = None
squared_terms = None
s2f = None
uncert_sum = None
N = None
vsys_axis = None
kp_axis = None
_loaded_extra = {}

_CONCAT_AXIS = {
    'cross_terms': -2,
    'squared_terms': -2,
    'alpha_frac': 0,
    'N': 0,
    's2f': 0,
    'uncert_sum': 0,
    't_start': 0,
    'phase': 0,
    'kp': None,
    'vsys': None,
    'icorr': 'add',
    'bad_indexs': 'add',
    # Per-planet / per-run constants — take value from first file only
    'contact_phases': None,
    'kind_trans': None,
    'apply_alpha': None,
}


def load_logl_results(file_list, path=None):
    """Load and combine logl grid NPZ files into module globals.

    After this call, ``get_logl`` and ``get_ccf`` operate without arguments.

    Parameters
    ----------
    file_list : list of str or Path
        NPZ files to load and concatenate (one per visit).
    path : str or Path, optional
        Directory containing the files.
    """
    import re
    import warnings

    global cross_terms, squared_terms, s2f, uncert_sum, N
    global vsys_axis, kp_axis, _loaded_extra

    path = Path(path) if path else Path('.')

    # Check for duplicate visit numbers before loading.
    # Each file should correspond to a distinct visit; loading two files for
    # the same visit silently doubles its exposures and corrupts the analysis.
    visit_numbers = {}
    for f in file_list:
        m = re.search(r'_visit(\d+)', Path(f).name)
        visit_num = m.group(1) if m else Path(f).stem
        if visit_num in visit_numbers:
            warnings.warn(
                f"Duplicate visit '{visit_num}' detected:\n"
                f"  already loaded: {visit_numbers[visit_num]}\n"
                f"  duplicate:      {Path(f).name}\n"
                "Each visit should appear only once. Check your glob pattern.",
                UserWarning, stacklevel=2,
            )
        else:
            visit_numbers[visit_num] = Path(f).name
    combined = {}

    for filename in file_list:
        zf = np.load(path / filename)
        for key in zf.keys():
            exp_axis = _CONCAT_AXIS.get(key, 0)
            if key in combined and exp_axis is not None:
                if exp_axis == 'add':
                    offset = combined['N'].shape[0]
                    combined[key] = np.concatenate(
                        [combined[key], zf[key] + offset], axis=0
                    )
                else:
                    combined[key] = np.concatenate(
                        [combined[key], zf[key]], axis=exp_axis
                    )
            else:
                combined[key] = zf[key]

    cross_terms = combined['cross_terms']
    squared_terms = combined['squared_terms']
    s2f = combined['s2f']
    uncert_sum = combined['uncert_sum']
    N = combined['N']
    vsys_axis = combined['vsys'][:, 0]
    kp_axis = combined['kp'][0, :]
    _loaded_extra = combined


def _chi2_from_terms(ct, st, sf, alpha=1.):
    """chi2(alpha) = sf - 2*alpha*ct + alpha**2*st (Brogi & Line 2019, eq. 6-ish).

    The one-line formula both `_logl_from_chi2_terms` (below) and
    `retrieval.py::lnprob` (Chantier A Phase 2) need -- `lnprob` needs the *raw* chi2
    (not yet converted to logL), because it sums chi2 across several visits first
    (`correlation.py::sum_logl`) and only takes the log once, at the very end, not
    once per visit. See `_logl_from_chi2_terms` for the parameter meanings.
    """
    return sf - 2 * alpha * ct + alpha ** 2 * st


def _logl_from_chi2_terms(ct, st, sf, N, alpha=1., beta=1., kind='BL', uncert_sum=None):
    """Compute logL from pre-summed chi2 terms -- globals-free core, shared.

    Chantier A Phase 2: factored out of `get_logl` below (which reads its terms from
    this module's globals, populated by the Kp-Vsys grid scan) so that
    `retrieval.py::lnprob` can compute a logL with the *exact same formula* from terms
    it computes fresh at every MCMC step (see `model_sequence.py`), instead of
    depending on the older, less clearly documented `correlation.py::calc_logl_BL_ord`.
    `get_logl` itself keeps all its grid-slicing/summing logic -- only the final
    "chi2 terms -> logL" conversion lives here now.

    Parameters
    ----------
    ct : np.ma.array
        Cross term, sum(model/noise * flux/noise) -- "R" in `correlation.py`'s
        notation, "f_x_g" in `_calc_chi2_terms` below.
    st : np.ma.array
        Squared model term, sum((model/noise)**2) -- "s2g".
    sf : np.ma.array
        Squared data term, sum((flux/noise)**2) -- "s2f", model-independent
        (precomputed once per visit, not recomputed at every grid point/MCMC step).
    N : np.ma.array or int
        Number of valid (unmasked) pixels contributing to the sum.
    alpha : float or np.ndarray, broadcastable against `ct`/`st`/`sf`/`N`
        Model scaling factor (fit or fixed at 1 for "as-is" injection).
    beta : float
        Noise scaling factor, Gibson ('G') prescription only.
    kind : {'BL', 'G'}
        LogL prescription: 'BL' = Brogi & Line (no explicit noise scaling fitted),
        'G' = Gibson et al. (explicit `beta`).
    uncert_sum : np.ma.array, optional
        sum(log(noise)) -- only needed for `kind='G'`.

    Returns
    -------
    np.ma.array
        Same broadcast shape as `sf - alpha*ct + alpha**2*st`.
    """
    chi2 = _chi2_from_terms(ct, st, sf, alpha=alpha)
    if kind == 'BL':
        return -N / 2 * np.ma.log(chi2 / N)
    elif kind == 'G':
        cst = -N / 2 * np.ma.log(2. * np.pi) - N * np.log(float(beta)) - uncert_sum
        return cst - 0.5 * chi2 / float(beta) ** 2
    else:
        raise ValueError(f"logl kind must be 'BL' or 'G', got {kind!r}")


def get_logl(alpha=1., beta=1., kind='BL', idx_orders=None, idx_exposure=None, sum_axis=None):
    """Compute logL from the loaded grid terms.

    Parameters
    ----------
    alpha : float or 1-D array-like
        Model scaling factor.  When an array is passed together with
        ``sum_axis``, a vectorised fast path pre-sums the chi² terms and
        then broadcasts alpha, avoiding a Python loop.  The output gains a
        leading alpha axis: shape ``(n_alpha, *spatial_dims)``.
        Scalar alpha preserves the original output shape (backward compatible).
    beta : float
        Noise scaling factor (Gibson prescription only).
    kind : {'BL', 'G'}
        logL prescription: 'BL' = Brogi & Line, 'G' = Gibson.
    idx_orders : array-like, optional
        Orders to include (default: all).
    idx_exposure : array-like, optional
        Exposure indices to include (default: all).
    sum_axis : int or tuple, optional
        Axis/axes over which to sum before returning.

    Returns
    -------
    np.ma.array
        Shape ``(n_alpha, *spatial_dims)`` when alpha is a 1-D array,
        ``(*spatial_dims,)`` when alpha is a scalar.
    """
    N_ma = np.ma.array(N, mask=(N == 0))

    exp_slice = slice(None) if idx_exposure is None else np.array(idx_exposure)[:, None]
    ord_slice = np.arange(N.shape[-1]) if idx_orders is None else np.array(idx_orders)
    idx = (..., exp_slice, ord_slice)

    ct = cross_terms[idx]
    st = squared_terms[idx]
    sf = s2f[idx]
    N_i = N_ma[idx]
    us = uncert_sum[idx]

    alpha_arr = np.asarray(alpha, dtype=float)
    scalar_alpha = alpha_arr.ndim == 0

    if not scalar_alpha and sum_axis is not None:
        # --- Vectorised fast path ---
        # chi²(α) = sf_sum − 2α·ct_sum + α²·st_sum is linear in the pre-summed
        # terms, so summing first is algebraically exact and avoids an intermediate
        # (n_alpha, …, n_exp, n_ord) tensor.
        ct_sum = np.ma.sum(ct,  axis=sum_axis)
        st_sum = np.ma.sum(st,  axis=sum_axis)
        sf_sum = np.ma.sum(sf,  axis=sum_axis)
        n_sum  = np.ma.sum(N_i, axis=sum_axis)
        # Add a leading "alpha axis" so alpha_arr broadcasts against the (already
        # summed) spatial dimensions -- the actual chi2->logL math is shared with the
        # scalar path below via `_logl_from_chi2_terms`.
        alpha_bcast = alpha_arr.reshape((-1,) + (1,) * ct_sum.ndim)
        us_sum = np.ma.sum(us, axis=sum_axis) if kind == 'G' else None
        return _logl_from_chi2_terms(ct_sum, st_sum, sf_sum, n_sum, alpha=alpha_bcast,
                                     beta=beta, kind=kind, uncert_sum=us_sum)

    # --- Scalar / original path (backward compatible) ---
    if sum_axis is not None:
        ct = np.ma.sum(ct, axis=sum_axis)
        st = np.ma.sum(st, axis=sum_axis)
        sf = np.ma.sum(sf, axis=sum_axis)
        N_i = np.ma.sum(N_i, axis=sum_axis)
        if kind == 'G':
            us = np.sum(us, axis=sum_axis)

    return _logl_from_chi2_terms(ct, st, sf, N_i, alpha=alpha_arr, beta=beta, kind=kind,
                                 uncert_sum=us if kind == 'G' else None)


def get_ccf(kind='BL', idx_orders=None, idx_exposure=None, sum_axis=None):
    """Compute CCF from the loaded grid terms.

    Parameters
    ----------
    kind : {'BL', 'G'}
        Normalisation convention.
    idx_orders, idx_exposure, sum_axis : same as get_logl.

    Returns
    -------
    np.ma.array
    """
    N_ma = np.ma.array(N, mask=(N == 0))

    exp_slice = slice(None) if idx_exposure is None else np.array(idx_exposure)[:, None]
    ord_slice = np.arange(N.shape[-1]) if idx_orders is None else np.array(idx_orders)
    idx = (..., exp_slice, ord_slice)

    if kind == 'BL':
        return np.ma.sum(cross_terms[idx] / N_ma[idx], axis=sum_axis)
    elif kind == 'G':
        return np.ma.sum(cross_terms[idx], axis=sum_axis)
    else:
        raise ValueError(f"CCF kind must be 'BL' or 'G', got {kind!r}")


# ---------------------------------------------------------------------------
# Analysis helpers — posterior computation for Kp-Vsys maps
# ---------------------------------------------------------------------------

def _log_simps(log_arr, x_or_dx, use_x=False, axis=0):
    """Numerically stable integration of exp(log_arr) via Simpson's rule.

    Computing ∫ exp(f) dx directly overflows or underflows for large |f|.
    The standard log-sum-exp trick shifts by the *maximum* before exponentiating::

        log(∫ exp(f) dx) = log(∫ exp(f − c) dx) + c    where c = max(f)

    After the shift, the dominant contribution is exp(0) = 1 (no overflow) and
    all other terms are ≤ 1; any that underflow to 0 are negligible anyway.

    Parameters
    ----------
    log_arr : ndarray
        Log of the integrand.
    x_or_dx : float or 1D array
        Uniform step size (float) or coordinate array (1D array).
    use_x : bool
        If True, x_or_dx is a coordinate array; if False it is a step size.
    axis : int
        Axis along which to integrate.

    Returns
    -------
    ndarray or float — log of the integral
    """
    from scipy import integrate
    # Shift by the MAXIMUM along the integration axis (standard log-sum-exp trick).
    # Shifting by the max ensures the dominant contribution is exp(0) = 1 and all
    # other terms are ≤ 1, so there is no overflow regardless of the dynamic range.
    # Shifting by the min would require exp(max − min) for the dominant term, which
    # overflows if the logL range exceeds ~710.  Underflow of the small-contribution
    # terms (exp(min − max) → 0) is acceptable since they are negligible.
    # Normalise the input to a plain float array before doing any arithmetic.
    # np.nanmax on a masked array returns the fill_value (typically 1e20) rather
    # than the max of the valid data — which would make exp_shift = 1e20 and
    # cause all exp(logL − 1e20) terms to underflow to zero.
    # Converting masked positions to -inf avoids the fill-value contamination:
    # -inf contributes zero to the integral (exp(-inf) = 0), which is correct.
    if isinstance(log_arr, np.ma.MaskedArray):
        log_arr = log_arr.filled(-np.inf)

    exp_shift = np.nanmax(log_arr)   # ignores NaN; -inf is just the smallest value
    if not np.isfinite(exp_shift):
        # Every position is -inf or NaN (no valid data) — integral is zero.
        result_shape = list(log_arr.shape)
        del result_shape[axis]
        return np.full(result_shape, -np.inf)

    # -inf → exp(-inf) = 0: positions with no valid data contribute nothing.
    # NaN (from cubic-spline edge artefacts) also map to 0.
    arr = np.where(np.isfinite(log_arr), np.exp(log_arr - exp_shift), 0.)

    if use_x:
        result = integrate.simps(arr, x=x_or_dx, axis=axis)
    else:
        result = integrate.simps(arr, dx=x_or_dx, axis=axis)
    with np.errstate(divide='ignore'):
        return np.log(result) + exp_shift


def compute_kpvsys_posterior(alpha_array=None, idx_signal=None,
                              idx_orders=None, oversample=2, kind='BL'):
    """Compute the alpha-marginalized Kp-Vsys posterior.

    Since the model amplitude alpha is not known a priori, we marginalize over
    it rather than fixing it to 1::

        P(Kp, v_sys | data) = ∫ L(Kp, v_sys, α) dα

    Integration is done via Simpson's rule in log-space (_log_simps) for
    numerical stability.  The posterior is then oversampled by cubic spline
    interpolation before computing sigma contours: the contour integrals
    converge faster on a finer grid, so oversampling improves the accuracy
    of the sigma levels without recomputing the expensive logl grid.

    Parameters
    ----------
    alpha_array : 1D array, optional
        Alpha values to marginalize over.
        Default: 31 points linearly spaced in [0.01, 2.0].
        The range covers sub- to super-solar model amplitudes.
    idx_signal : 1D int array, optional
        Indices of exposures where the planet signal is detectable
        (alpha_frac > 0.5).  Default: derived from loaded alpha_frac.
    idx_orders : array-like, optional
        Spectral orders to include.  Default: all.
    oversample : int
        Oversampling factor applied to log(posterior) via cubic interpolation
        before computing marginals and contours.  Default: 2.
    kind : {'BL', 'G'}
        LogL prescription: 'BL' = Brogi & Line (default), 'G' = Gibson.

    Returns
    -------
    posterior : (n_vsys_os, n_kp_os) array
        Alpha-marginalized, oversampled posterior (linear, not log).
    vsys_os : (n_vsys_os,) array  — oversampled v_sys axis
    kp_os   : (n_kp_os,)  array  — oversampled Kp axis
    margin_vsys : (n_vsys_os,) array  — posterior marginalized over Kp
    margin_kp   : (n_kp_os,)   array  — posterior marginalized over v_sys
    """

    if alpha_array is None:
        # 31 points in [0.01, 2.0] gives adequate integration accuracy.
        alpha_array = np.linspace(0.01, 2., 31)

    if idx_signal is None:
        # alpha_frac is the fraction of the total planet signal received during
        # each exposure.  It equals 1 during full eclipse/transit and 0 when
        # the planet is hidden by the star.  Exposures with alpha_frac > 0.5
        # are those where we expect a detectable signal.
        alpha_frac = _loaded_extra['alpha_frac']
        (idx_signal,) = np.nonzero(alpha_frac > 0.5)

    # --- Marginalise over alpha in log-space ---
    log.info(f'Marginalising over {len(alpha_array)} alpha values ...')
    logl_map_alpha = _build_logl_cube(alpha_array, idx_signal, idx_orders, kind=kind)
    d_alpha = alpha_array[1] - alpha_array[0]
    log_posterior = _log_simps(logl_map_alpha, d_alpha, axis=0)   # (n_vsys, n_kp)

    return _posterior_from_log_map(log_posterior, vsys_axis, kp_axis, oversample)


def compute_kpvsys_posterior_fixed_alpha(alpha, idx_signal=None, idx_orders=None,
                                          oversample=2, kind='BL'):
    """Compute the Kp-Vsys posterior/likelihood map at a FIXED alpha.

    Same output convention as ``compute_kpvsys_posterior`` (drop-in for
    ``plot_kpvsys_map``), but alpha is held fixed instead of marginalized
    over. Typical uses: ``alpha=0.`` (null-hypothesis map -- should come out
    ~flat after normalization, a useful sanity check) or
    ``alpha=alpha_best`` from ``compute_alpha_significance`` (visualize the
    map at the maximum-likelihood amplitude).

    See ``compute_kpvsys_bayes_factor`` and ``compute_alpha_significance``
    for two different ways of turning a fixed-alpha comparison into a
    detection statistic.

    Parameters
    ----------
    alpha : float
        Fixed model scaling factor.
    idx_signal : 1D int array, optional
        Same as ``compute_kpvsys_posterior``.
    idx_orders : array-like, optional
        Spectral orders to include. Default: all.
    oversample : int
        Oversampling factor, see ``compute_kpvsys_posterior``.
    kind : {'BL', 'G'}
        LogL prescription: 'BL' = Brogi & Line (default), 'G' = Gibson.

    Returns
    -------
    posterior : (n_vsys_os, n_kp_os) array
        Fixed-alpha, oversampled posterior/likelihood (linear, max = 1).
    vsys_os : (n_vsys_os,) array  — oversampled v_sys axis
    kp_os   : (n_kp_os,)  array  — oversampled Kp axis
    margin_vsys : (n_vsys_os,) array  — marginalized over Kp
    margin_kp   : (n_kp_os,)   array  — marginalized over v_sys
    """
    if idx_signal is None:
        alpha_frac = _loaded_extra['alpha_frac']
        (idx_signal,) = np.nonzero(alpha_frac > 0.5)

    log_map = get_logl(alpha=float(alpha), idx_orders=idx_orders,
                        idx_exposure=idx_signal, kind=kind, sum_axis=(-2, -1))

    return _posterior_from_log_map(log_map, vsys_axis, kp_axis, oversample)


def compute_kpvsys_bayes_factor(alpha_array=None, idx_signal=None, idx_orders=None,
                                 kind='BL', vsys_bounds=None, kp_bounds=None):
    """Compute the Bayes factor between the signal and null models.

    Signal model: alpha, vsys, AND Kp are all free parameters, each with its
    own *uniform* prior (alpha over ``alpha_array``'s range, vsys/Kp over
    ``vsys_bounds``/``kp_bounds``, default: the full loaded grid). Null
    model: alpha=0 (no planet signal, no vsys/Kp dependence at all)::

        log(BF) = log[ integral( L(vsys,Kp,a) da dvsys dKp )
                        / (a_range * vsys_range * kp_range) ]
                  - logL(alpha=0)

    Each ``1 / range`` term is an Occam factor coming from that parameter's
    prior normalization -- it is what makes the Bayes factor penalize a
    vague/wide prior (or a large, mostly-empty search volume). vsys and Kp
    get their own Occam factor for the same reason alpha does: they are free
    parameters of the signal model that the null model does not have at all,
    so there is nothing for their priors to cancel against. **The result
    depends on all three ranges chosen** (the alpha range explicitly, the
    vsys/Kp ranges via ``vsys_bounds``/``kp_bounds`` or implicitly through
    whatever grid was computed), more so for marginal detections. Report
    log(BF) for a couple of different ranges as a sensitivity check rather
    than trusting a single number (see the ``detection_significance``
    tutorial notebook for a worked example) -- but see the two distinct
    failure modes below before doing so.

    Two different things can happen when you narrow a range, and they are
    easy to conflate:

    - **Truncation**: the narrower range excludes where the likelihood is
      actually high (e.g. an ``alpha_array`` that stops before the true
      best-fit alpha). This makes log(BF) mechanically lower because real
      evidence got left out of the integral -- it says nothing about
      robustness, it is simply the wrong range for this data. Always check
      that a candidate range/bounds fully covers the peak (e.g. against
      ``compute_alpha_significance``'s ``alpha_best``/``vsys_best``/
      ``kp_best``) before reading anything into a lower log(BF).
    - **Genuine Occam dilution**: the range already fully covers the peak,
      and is simply made wider still by adding "empty" territory the data
      disfavor. *This* is the sensitivity effect worth reporting.

    ``vsys_bounds``/``kp_bounds`` in particular carry a **look-elsewhere
    risk**: choosing them *after* seeing where the peak landed on the full
    grid is double-dipping (you'd be reporting the evidence for a model you
    only get to test because the data already told you where to look), not
    a legitimate a priori restriction. Only narrow vsys/Kp using bounds
    justified independently of this grid (e.g. known orbital constraints),
    never by eye-balling the map first.

    Parameters
    ----------
    alpha_array : 1D array, optional
        Alpha values to marginalize over, defining the prior's support.
        Default: 31 points linearly spaced in [0.01, 2.0].
    idx_signal : 1D int array, optional
        Indices of exposures where the planet signal is detectable.
        Default: derived from loaded alpha_frac.
    idx_orders : array-like, optional
        Spectral orders to include. Default: all.
    kind : {'BL', 'G'}
        LogL prescription: 'BL' = Brogi & Line (default), 'G' = Gibson.
    vsys_bounds : (float, float), optional
        Restrict the vsys integral/prior to this sub-range of the loaded
        grid. Default: the full ``vsys_axis`` range.
    kp_bounds : (float, float), optional
        Restrict the Kp integral/prior to this sub-range of the loaded grid.
        Default: the full ``kp_axis`` range.

    Returns
    -------
    dict
        ``log_bf`` : natural-log Bayes factor (signal vs null).
        ``alpha_range``, ``vsys_range``, ``kp_range`` : widths of the three
        uniform priors actually used.
        ``logl_evidence_signal`` : log-evidence of the signal model.
        ``logl_null`` : logL of the null model (alpha=0).
    """
    if alpha_array is None:
        alpha_array = np.linspace(0.01, 2., 31)
    if idx_signal is None:
        alpha_frac = _loaded_extra['alpha_frac']
        (idx_signal,) = np.nonzero(alpha_frac > 0.5)

    # --- Select the vsys/Kp sub-grid to integrate over (default: everything) ---
    if vsys_bounds is None:
        vsys_mask = np.ones_like(vsys_axis, dtype=bool)
    else:
        vsys_mask = (vsys_axis >= vsys_bounds[0]) & (vsys_axis <= vsys_bounds[1])
    if kp_bounds is None:
        kp_mask = np.ones_like(kp_axis, dtype=bool)
    else:
        kp_mask = (kp_axis >= kp_bounds[0]) & (kp_axis <= kp_bounds[1])
    vsys_sel = vsys_axis[vsys_mask]
    kp_sel = kp_axis[kp_mask]

    # --- Evidence of the signal model: integrate over alpha, then vsys, then Kp ---
    logl_cube = get_logl(alpha=alpha_array, idx_orders=idx_orders,
                          idx_exposure=idx_signal, kind=kind, sum_axis=(-2, -1))
    logl_cube = logl_cube[:, vsys_mask][:, :, kp_mask]               # (n_alpha, n_vsys_sel, n_kp_sel)
    d_alpha = alpha_array[1] - alpha_array[0]
    log_ev_vsys_kp = _log_simps(logl_cube, d_alpha, axis=0)          # (n_vsys_sel, n_kp_sel)

    d_vsys = vsys_sel[1] - vsys_sel[0]
    d_kp   = kp_sel[1]   - kp_sel[0]
    log_ev_signal_raw = _log_simps(_log_simps(log_ev_vsys_kp, d_kp, axis=1), d_vsys, axis=0)

    # Occam factors: normalize each integral by its (uniform) prior density.
    # alpha, vsys, and Kp are all free parameters of the signal model that the
    # null model simply doesn't have -- each needs its own normalization, there
    # is no shared prior to cancel out of the ratio.
    alpha_range = float(alpha_array[-1] - alpha_array[0])
    vsys_range  = float(vsys_sel[-1] - vsys_sel[0])
    kp_range    = float(kp_sel[-1] - kp_sel[0])
    log_ev_signal = float(log_ev_signal_raw) - np.log(alpha_range) - np.log(vsys_range) - np.log(kp_range)

    # --- Evidence of the null model: alpha=0 => chi2 = sf only, independent of vsys/Kp ---
    logl_null_map = get_logl(alpha=0., idx_orders=idx_orders, idx_exposure=idx_signal,
                              kind=kind, sum_axis=(-2, -1))
    logl_null = float(np.ma.median(logl_null_map))

    return dict(log_bf=log_ev_signal - logl_null, alpha_range=alpha_range,
                vsys_range=vsys_range, kp_range=kp_range,
                logl_evidence_signal=log_ev_signal, logl_null=logl_null)


def _profile_alpha_logl(idx_signal, idx_orders, kind='BL'):
    """Per-grid-point analytic profile likelihood: MLE alpha (>= 0) and logL there.

    ``chi2(alpha) = sf - 2*alpha*ct + alpha**2*st`` is an upward parabola in
    alpha (``st > 0``) for *both* ``kind='BL'`` and ``kind='G'`` -- both logL
    prescriptions are strictly decreasing functions of chi2 (see
    ``_logl_from_chi2_terms``), so maximizing either one over alpha is
    exactly equivalent to minimizing this same chi2, whose unconstrained
    minimum is at ``alpha = ct/st`` -- a closed form, no numerical optimizer
    needed. The alpha >= 0 constraint (the model amplitude can't be
    negative) clips negative solutions to the boundary alpha=0, which is
    also exactly where the null model sits.

    Shared by ``compute_alpha_significance`` (evaluated at the grid's best
    point) and ``compute_alpha_significance_map`` (evaluated everywhere).

    Returns
    -------
    alpha_map : (n_vsys, n_kp) array -- per-point alpha MLE (>= 0).
    logl_best_map : (n_vsys, n_kp) array -- logL at ``alpha_map``.
    logl_null_map : (n_vsys, n_kp) array -- logL at alpha=0 (constant across
        vsys/Kp; returned as a full map only for shape convenience).
    """
    ct, st, sf, N, us = get_chi2_components(idx_orders=idx_orders, idx_exposure=idx_signal,
                                             sum_axis=(-2, -1))
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha_unclipped = np.ma.where(st > 0, ct / st, 0.)
    alpha_map = np.ma.maximum(alpha_unclipped, 0.)
    logl_best_map = _logl_from_chi2_terms(ct, st, sf, N, alpha=alpha_map, kind=kind, uncert_sum=us)
    logl_null_map = _logl_from_chi2_terms(ct, st, sf, N, alpha=0., kind=kind, uncert_sum=us)
    return alpha_map, logl_best_map, logl_null_map


def _alpha_significance_full(idx_signal, idx_orders, kind):
    """Core computation shared by ``compute_alpha_significance`` and
    ``compute_alpha_significance_map`` -- full per-point maps plus the
    location of the map's own peak. Both public functions are thin
    extractions from this, so the D/p-value/sigma formulas exist in exactly
    one place.
    """
    from scipy import stats

    alpha_map, logl_best_map, logl_null_map = _profile_alpha_logl(idx_signal, idx_orders, kind)
    logl_null = float(np.ma.median(logl_null_map))

    # Numerical safety: D should be >= 0 at the true (per-point) MLE (logl_best_map
    # >= logl_null by construction, since alpha=0 is always at least as good as the
    # unconstrained optimum when that optimum is clipped to the boundary); guard
    # against tiny negative values from floating-point noise.
    D_map = np.ma.maximum(2. * (logl_best_map - logl_null), 0.)
    p_map = 0.5 * stats.chi2.sf(np.ma.filled(D_map, 0.), df=1)
    p_map = np.ma.array(p_map, mask=np.ma.getmaskarray(D_map))
    sigma_map = np.ma.array(stats.norm.isf(p_map), mask=np.ma.getmaskarray(D_map))

    sigma_filled = np.ma.filled(sigma_map, -np.inf)
    i_v, i_k = np.unravel_index(np.argmax(sigma_filled), sigma_filled.shape)

    return dict(alpha_map=alpha_map, logl_best_map=logl_best_map, logl_null=logl_null,
                D_map=D_map, p_map=p_map, sigma_map=sigma_map, i_v=i_v, i_k=i_k)


def compute_alpha_significance(idx_signal=None, idx_orders=None, kind='BL'):
    """Compute a detection significance via the profile likelihood-ratio test.

    Compares H0: alpha=0 (no signal) against H1: alpha=alpha_best, the
    maximum-likelihood estimate jointly optimized over vsys, Kp, and alpha
    (closed form -- see ``_profile_alpha_logl``). This sidesteps the need to
    choose a prior over alpha (unlike ``compute_kpvsys_bayes_factor``), at
    the cost of moving from a Bayesian to a frequentist framework.

    Since alpha >= 0 is a *boundary* of the parameter space (the model
    amplitude cannot be negative), the null distribution of the test
    statistic ``D = 2 * (logL_best - logL_null)`` is **not** a plain
    chi-square with 1 degree of freedom. By Chernoff's theorem (Chernoff
    1954; Self & Liang 1987), it is a 50/50 mixture of a point mass at 0
    and chi2(df=1), so the p-value is ``0.5 * chi2(df=1).sf(D)`` -- using
    the unadjusted chi2(df=1) survival function would overstate the
    significance by roughly a factor of 2 in probability.

    A thin extraction (at the grid's best point only) of
    ``compute_alpha_significance_map``, which computes the exact same
    statistic at *every* (vsys, Kp) grid point -- see there for a full map
    instead of just this single point's numbers.

    Parameters
    ----------
    idx_signal : 1D int array, optional
        Indices of exposures where the planet signal is detectable.
        Default: derived from loaded alpha_frac.
    idx_orders : array-like, optional
        Spectral orders to include. Default: all.
    kind : {'BL', 'G'}
        LogL prescription: 'BL' = Brogi & Line (default), 'G' = Gibson.

    Returns
    -------
    dict
        ``alpha_best``, ``vsys_best``, ``kp_best`` : MLE location.
        ``logl_best``, ``logl_null`` : logL at the MLE and at alpha=0.
        ``D`` : the test statistic, 2*(logl_best - logl_null).
        ``p_value`` : Chernoff-mixture-corrected p-value.
        ``sigma`` : ``p_value`` converted to an equivalent Gaussian sigma.
    """
    if idx_signal is None:
        alpha_frac = _loaded_extra['alpha_frac']
        (idx_signal,) = np.nonzero(alpha_frac > 0.5)

    core = _alpha_significance_full(idx_signal, idx_orders, kind)
    i_v, i_k = core['i_v'], core['i_k']

    return dict(
        alpha_best=float(core['alpha_map'][i_v, i_k]),
        vsys_best=float(vsys_axis[i_v]), kp_best=float(kp_axis[i_k]),
        logl_best=float(core['logl_best_map'][i_v, i_k]), logl_null=core['logl_null'],
        D=float(core['D_map'][i_v, i_k]), p_value=float(core['p_map'][i_v, i_k]),
        sigma=float(core['sigma_map'][i_v, i_k]),
    )


def compute_alpha_significance_map(idx_signal=None, idx_orders=None, kind='BL'):
    """Compute the profile likelihood-ratio detection-significance MAP.

    Like ``compute_alpha_significance``, but evaluates the statistic at
    *every* (vsys, Kp) grid point instead of reporting it only at the single
    best one: for each grid point, find the alpha that maximizes logL there
    (closed form, see ``_profile_alpha_logl``), then the Chernoff-corrected
    sigma comparing that point's best logL to the null. Same statistic as
    ``compute_alpha_significance``, evaluated everywhere instead of once --
    the two agree exactly at the map's own peak (both are thin extractions
    of ``_alpha_significance_full``).

    **The look-elsewhere caveat applies just as much here as to the
    single-point version, not less.** Each grid point's sigma is a formally
    valid test *taken in isolation*; the map does not correct for having
    evaluated a whole grid of them. Don't read "look, an entire region is
    above 3 sigma" as stronger evidence than the single best point already
    gives -- it isn't a multiple-detections argument, it's the same one
    detection's neighbourhood.

    Parameters
    ----------
    idx_signal : 1D int array, optional
        Indices of exposures where the planet signal is detectable.
        Default: derived from loaded alpha_frac.
    idx_orders : array-like, optional
        Spectral orders to include. Default: all.
    kind : {'BL', 'G'}
        LogL prescription: 'BL' = Brogi & Line (default), 'G' = Gibson.

    Returns
    -------
    dict
        ``sigma_map`` : (n_vsys, n_kp) array -- Chernoff-corrected sigma at
        every grid point (can be ``+inf`` where the p-value underflows to 0
        in float64, for a very significant point).
        ``alpha_map`` : (n_vsys, n_kp) array -- per-point alpha MLE.
        ``vsys_best``, ``kp_best``, ``alpha_best`` : location of the map's
        maximum.
        ``peak_sigma`` : sigma at that location (matches
        ``compute_alpha_significance``'s ``sigma`` on the same data).
        ``logl_null`` : logL of the null model (alpha=0).
        ``vsys_coords``, ``kp_coords`` : the native grid axes (echoed back
        for convenience, e.g. for ``plotting_fcts.plot_alpha_significance_map``).
    """
    if idx_signal is None:
        alpha_frac = _loaded_extra['alpha_frac']
        (idx_signal,) = np.nonzero(alpha_frac > 0.5)

    core = _alpha_significance_full(idx_signal, idx_orders, kind)
    i_v, i_k = core['i_v'], core['i_k']

    return dict(sigma_map=core['sigma_map'], alpha_map=core['alpha_map'],
                vsys_best=float(vsys_axis[i_v]), kp_best=float(kp_axis[i_k]),
                alpha_best=float(core['alpha_map'][i_v, i_k]),
                peak_sigma=float(core['sigma_map'][i_v, i_k]),
                vsys_coords=vsys_axis.copy(), kp_coords=kp_axis.copy(),
                logl_null=core['logl_null'])


def compute_empirical_sigma_map(map_2d, method='clip', box_vsys=None, box_kp=None,
                                 clip_sigma=2., clip_iter=4,
                                 vsys_coords=None, kp_coords=None):
    """Estimate a Kp-Vsys detection sigma map via an empirical noise floor.

    A widely used, purely empirical alternative to ``compute_kpvsys_posterior``
    / ``compute_kpvsys_bayes_factor`` / ``compute_alpha_significance``: instead
    of a formal likelihood/probability comparison against a null hypothesis,
    this just measures how many standard deviations the map's peak sits above
    a noise floor estimated directly from the map itself -- no prior, no
    reference to alpha=0, no asymptotic distribution assumption. Two ways to
    estimate that noise floor:

    - ``method='box'``: standard deviation within a user-chosen (vsys, Kp)
      sub-region assumed to contain no signal (``box_vsys``/``box_kp``).
    - ``method='clip'``: iterative sigma-clipping over the *entire* map
      (removes outliers -- including the peak itself -- each iteration),
      using the standard deviation of what survives as the noise floor.

    Both give ``sigma_map = (map_2d - median) / noise_std``; the value at the
    map's maximum is the headline "N sigma" figure this method reports.

    This is a common convention in the HRCCS literature but, unlike the other
    three methods in this module, has no formal null-hypothesis test behind
    it -- it assumes the estimated noise floor is representative and
    Gaussian, which is not guaranteed (e.g. a ``box`` region that isn't
    really signal-free, or ``clip`` iterations that under/over-clip a
    non-Gaussian map). Treat it as a convenient cross-check, not a
    replacement for ``compute_alpha_significance``/``compute_kpvsys_bayes_factor``
    (see the ``detection_significance`` tutorial notebook for a comparison).

    **Never feed this the *linear* (exponentiated) output of**
    ``compute_kpvsys_posterior``/``compute_kpvsys_posterior_fixed_alpha``
    **directly.** Those posteriors are ``exp(logL - max)`` by construction, so
    almost the entire map away from the peak is squashed to ~0 (numerical
    underflow, not a real noise floor) -- box/clip would then measure that
    underflow, giving a ``noise_std`` many orders of magnitude too small and a
    wildly inflated, meaningless "sigma" (this is not a matter of
    oversampling -- plain ``oversample=1`` posteriors have the exact same
    problem -- it's the exponentiation itself that breaks the assumption).

    A CCF/logL-*scale* map is what this method needs -- ``get_ccf(...)``
    qualifies directly, and so does ``np.log(posterior)`` (the log-posterior,
    i.e. exactly what the sigma-scale colour maps in this module's plots
    already show by default, ``plot_posterior_2d``'s ``scale='log'``): taking
    the log undoes the exponentiation and recovers a logL-like quantity, as
    long as ``posterior`` hasn't underflowed to exact 0.0 anywhere (check
    ``np.isfinite(np.log(posterior)).all()`` -- non-finite values are masked
    out of the noise-floor estimate automatically, but a very large fraction
    of them would defeat the purpose).

    Non-finite values (``-inf``/``NaN``, e.g. from ``np.log(posterior)``
    underflow, or masked/invalid grid points) are automatically excluded from
    the median/noise-floor estimate and can't become the reported peak.

    Originally written by Joost Wardenier, improved by Mathis Bouffard (June
    2025) as ``plotting_fcts.calculate_KpVsys_map``, tied to the older
    ``correlation_class.Correlations`` CCF object; generalized here to
    operate on any (vsys, Kp) map -- e.g. from ``get_ccf(idx_exposure=idx_signal,
    sum_axis=(-2, -1))`` -- so it can be compared directly against the other
    detection statistics in this module.

    Parameters
    ----------
    map_2d : (n_vsys, n_kp) array
        A Kp-Vsys map -- e.g. from ``get_ccf`` (native grid), or an
        oversampled posterior from ``compute_kpvsys_posterior`` (its
        ``vsys_os``/``kp_os`` output, *not* the module's native
        ``vsys_axis``/``kp_axis``, matches this case -- pass them via
        ``vsys_coords``/``kp_coords`` below).
    method : {'box', 'clip'}
        Noise-floor estimation method. Default: 'clip'.
    box_vsys, box_kp : (float, float), optional
        (min, max) bounds in km/s defining the signal-free region used by
        ``method='box'``. Required if ``method='box'``.
    clip_sigma : float
        Sigma threshold for ``method='clip'``'s iterative sigma-clipping.
        Default: 2.
    clip_iter : int
        Maximum number of sigma-clipping iterations. Default: 4.
    vsys_coords, kp_coords : 1D array, optional
        Coordinates matching ``map_2d``'s two axes. Default: the module's
        native ``vsys_axis``/``kp_axis`` globals -- correct for a map fresh
        off ``get_ccf``, but **wrong (and will raise an ``IndexError``, or
        silently mismatch, for anything on a different grid**, most commonly
        an oversampled posterior: pass its own ``vsys_os``/``kp_os`` here
        instead.

    Returns
    -------
    dict
        ``sigma_map`` : (n_vsys, n_kp) array, ``(map_2d - median) / noise_std``.
        ``peak_sigma`` : sigma value at map_2d's maximum.
        ``vsys_best``, ``kp_best`` : location of that maximum.
        ``noise_std`` : the estimated noise floor used for the normalization.
        ``method`` : the method used ('box' or 'clip').
        ``vsys_coords``, ``kp_coords`` : the coordinates actually used (echoed
        back for convenience, e.g. for ``plotting_fcts.plot_empirical_sigma_map``).
        ``box_vsys``, ``box_kp`` : the box bounds used (``None`` for ``method='clip'``).
    """
    if vsys_coords is None:
        vsys_coords = vsys_axis
    if kp_coords is None:
        kp_coords = kp_axis
    if map_2d.shape != (len(vsys_coords), len(kp_coords)):
        raise ValueError(
            f"map_2d.shape={map_2d.shape} doesn't match "
            f"(len(vsys_coords), len(kp_coords))=({len(vsys_coords)}, {len(kp_coords)}). "
            "If map_2d came from compute_kpvsys_posterior (oversample > 1), pass its "
            "own vsys_os/kp_os as vsys_coords/kp_coords -- they don't match the "
            "module's native vsys_axis/kp_axis used by default."
        )

    # Mask non-finite values (-inf/NaN, e.g. from np.log(posterior) underflow, or
    # already-masked/invalid grid points) so they can't poison the noise-floor
    # estimate or get reported as the peak.
    map_ma = np.ma.masked_invalid(map_2d)

    if method == 'box':
        if box_vsys is None or box_kp is None:
            raise ValueError("method='box' requires box_vsys and box_kp.")
        vsys_mask = (vsys_coords >= box_vsys[0]) & (vsys_coords <= box_vsys[1])
        kp_mask = (kp_coords >= box_kp[0]) & (kp_coords <= box_kp[1])
        submap = map_ma[np.ix_(vsys_mask, kp_mask)]
        noise_std = float(np.ma.std(submap))
    elif method == 'clip':
        from astropy.stats import sigma_clip
        clipped = sigma_clip(map_ma, sigma=clip_sigma, maxiters=clip_iter)
        noise_std = float(np.ma.std(clipped))
    else:
        raise ValueError(f"method must be 'box' or 'clip', got {method!r}")

    median = float(np.ma.median(map_ma))
    sigma_map = (map_ma - median) / noise_std

    map_filled = np.ma.filled(map_ma, -np.inf)
    i_v, i_k = np.unravel_index(np.argmax(map_filled), map_filled.shape)
    peak_sigma = float(sigma_map[i_v, i_k])

    return dict(sigma_map=sigma_map, peak_sigma=peak_sigma,
                vsys_best=float(vsys_coords[i_v]), kp_best=float(kp_coords[i_k]),
                noise_std=noise_std, method=method,
                vsys_coords=vsys_coords, kp_coords=kp_coords,
                box_vsys=box_vsys, box_kp=box_kp)


def _marginals_from_log(log_post, x_axis, y_axis):
    """Compute 1D marginals from a 2D log-posterior using per-slice max shifts.

    Per-slice max (instead of the global max used by _log_simps) ensures that
    rows/columns far from the peak are not all shifted to underflow — they each
    keep their own scale and contribute a smoothly varying integral value.

    Parameters
    ----------
    log_post : (n_x, n_y) array  — log-posterior, may contain -inf
    x_axis   : (n_x,) array
    y_axis   : (n_y,) array

    Returns
    -------
    margin_x : (n_x,) array  — posterior marginalised over y  (not normalised)
    margin_y : (n_y,) array  — posterior marginalised over x  (not normalised)
    """
    from scipy import integrate as _int

    lp = np.where(np.isfinite(log_post), log_post, -np.inf)

    # margin_x(i) = ∫ exp(log_post[i, :]) dy  — integrate over y (axis=1)
    mx = np.nanmax(lp, axis=1, keepdims=True)          # (n_x, 1) — per-row max
    mx_safe = np.where(np.isfinite(mx), mx, 0.)
    shifted_x = np.where(np.isfinite(lp), np.exp(lp - mx_safe), 0.)
    intg_x = _int.simps(shifted_x, x=y_axis, axis=1)  # (n_x,)
    with np.errstate(divide='ignore'):
        margin_x = np.exp(np.log(intg_x) + mx_safe.squeeze(1))

    # margin_y(j) = ∫ exp(log_post[:, j]) dx  — integrate over x (axis=0)
    my = np.nanmax(lp, axis=0, keepdims=True)          # (1, n_y) — per-col max
    my_safe = np.where(np.isfinite(my), my, 0.)
    shifted_y = np.where(np.isfinite(lp), np.exp(lp - my_safe), 0.)
    intg_y = _int.simps(shifted_y, x=x_axis, axis=0)  # (n_y,)
    with np.errstate(divide='ignore'):
        margin_y = np.exp(np.log(intg_y) + my_safe.squeeze(0))

    return margin_x, margin_y


def _posterior_from_log_map(log_map, axis0_coords, axis1_coords, oversample=2):
    """Normalise, optionally oversample, exponentiate, and compute marginals.

    Shared post-processing for any 2D log-likelihood/log-posterior map
    (vsys x Kp, alpha x Kp, alpha x vsys) -- factored out of
    ``compute_kpvsys_posterior``, ``compute_alpha_kp_posterior``, and
    ``compute_alpha_vsys_posterior``, which only differ in which axis they
    have already integrated out before calling this.

    Parameters
    ----------
    log_map : (n0, n1) array
        Log-likelihood or log-posterior map, may contain -inf/NaN.
    axis0_coords : (n0,) array
        Coordinates for the first axis of ``log_map``.
    axis1_coords : (n1,) array
        Coordinates for the second axis of ``log_map``.
    oversample : int
        Oversampling factor applied via cubic-spline interpolation before
        computing marginals (display only). ``oversample <= 1`` disables it.

    Returns
    -------
    posterior : (n0_out, n1_out) array
        Normalised posterior (linear, max = 1).
    axis0_out, axis1_out : arrays
        Output coordinates (oversampled if requested).
    margin_axis0, margin_axis1 : arrays
        1D marginals (not renormalised) over axis1 and axis0 respectively.
    """
    finite_lp = log_map[np.isfinite(log_map)]
    lp_max = float(finite_lp.max()) if len(finite_lp) else 0.
    log_post_norm = log_map - lp_max

    if oversample > 1:
        from starships.plotting_fcts import oversample_image
        # Replace any residual -inf before spline interpolation.
        floor_val = log_post_norm[np.isfinite(log_post_norm)].min()
        log_post_safe = np.where(np.isfinite(log_post_norm), log_post_norm, floor_val)
        log_post_os, axis1_os, axis0_os = oversample_image(
            log_post_safe, oversample, x_coords=axis1_coords, y_coords=axis0_coords,
        )
        posterior = np.exp(log_post_os)
        axis0_out, axis1_out = axis0_os, axis1_os
        lp_for_margins = log_post_os   # per-slice max on oversampled grid
    else:
        posterior = np.exp(log_post_norm)
        axis0_out, axis1_out = axis0_coords.copy(), axis1_coords.copy()
        lp_for_margins = log_post_norm

    margin_axis0, margin_axis1 = _marginals_from_log(lp_for_margins, axis0_out, axis1_out)
    return posterior, axis0_out, axis1_out, margin_axis0, margin_axis1


def _build_logl_cube(alpha_array, idx_signal, idx_orders, kind='BL'):
    """Build and pre-normalise the (n_alpha, n_vsys, n_kp) logL cube.

    Shared by compute_kpvsys_posterior, compute_alpha_kp_posterior, and
    compute_alpha_vsys_posterior so that the normalisation logic lives in one place.
    Uses the vectorised fast path in get_logl (array alpha + sum_axis).
    """
    logl_cube = get_logl(
        alpha=np.asarray(alpha_array),
        idx_orders=idx_orders,
        idx_exposure=idx_signal,
        kind=kind,
        sum_axis=(-2, -1),
    )  # shape (n_alpha, n_vsys, n_kp)
    # Pre-normalise to max=0 so that exp() calls stay within float64 range.
    if isinstance(logl_cube, np.ma.MaskedArray):
        _shift = float(logl_cube.max())
    else:
        _finite = logl_cube[np.isfinite(logl_cube)]
        _shift = float(_finite.max()) if len(_finite) > 0 else 0.
    return logl_cube - _shift


# ---------------------------------------------------------------------------
# Leave-one-out order contributions
# ---------------------------------------------------------------------------

def get_chi2_components(idx_orders=None, idx_exposure=None, sum_axis=None):
    """Return pre-summed chi² components for a subset of orders / exposures.

    Parameters
    ----------
    idx_orders : array-like, optional — orders to include (default: all)
    idx_exposure : array-like, optional — exposures to include (default: all)
    sum_axis : int or tuple, optional — axes to sum over

    Returns
    -------
    ct, st, sf, n_pix, us : masked arrays — cross-term, model², data², N, log-noise sum
    """
    N_ma = np.ma.array(N, mask=(N == 0))
    exp_slice = slice(None) if idx_exposure is None else np.array(idx_exposure)[:, None]
    ord_slice  = np.arange(N.shape[-1]) if idx_orders is None else np.array(idx_orders)
    idx = (..., exp_slice, ord_slice)

    ct    = cross_terms[idx]
    st    = squared_terms[idx]
    sf    = s2f[idx]
    n_pix = N_ma[idx]
    us    = uncert_sum[idx]

    if sum_axis is not None:
        ct    = np.ma.sum(ct,    axis=sum_axis)
        st    = np.ma.sum(st,    axis=sum_axis)
        sf    = np.ma.sum(sf,    axis=sum_axis)
        n_pix = np.ma.sum(n_pix, axis=sum_axis)
        us    = np.ma.sum(us,    axis=sum_axis)

    return ct, st, sf, n_pix, us


def _logl_from_components(ct_s, st_s, sf_s, n_s, us_s, alpha_array,
                           kind='BL', beta=1.):
    """Compute the (n_alpha, n_vsys, n_kp) logL cube from pre-summed chi² components.

    Returns the **un-normalised** cube so the caller can apply a consistent
    shift before comparing full vs. leave-one-out cubes.
    """
    alpha_arr   = np.asarray(alpha_array, dtype=float)
    n_spatial   = ct_s.ndim
    alpha_bcast = alpha_arr.reshape((-1,) + (1,) * n_spatial)
    chi2 = sf_s - 2 * alpha_bcast * ct_s + alpha_bcast ** 2 * st_s
    # Guard against chi2 <= 0 (can occur for LOO subsets with bad/noisy orders).
    # Masking these points prevents log(0) / log(negative) infinities.
    chi2_safe = np.ma.masked_where(chi2 <= 0, chi2)
    if kind == 'BL':
        return -n_s / 2 * np.ma.log(chi2_safe / n_s)
    elif kind == 'G':
        cst = -n_s / 2 * np.ma.log(2. * np.pi) - n_s * np.log(float(beta)) - us_s
        return cst - 0.5 * chi2_safe / float(beta) ** 2
    else:
        raise ValueError(f"kind must be 'BL' or 'G', got {kind!r}")


def compute_loo_order_contributions(alpha_array=None, idx_signal=None,
                                     idx_orders=None, kind='BL',
                                     kp_ref=None, vsys_ref=None,
                                     vsys_excl=30., kp_excl=30.):
    """Compute leave-one-out (LOO) order contributions via off-peak probability.

    The scalar contribution for order k measures how the Kp-vsys posterior
    concentrates toward the signal when order k is included:

        contribution_k = f_off_loo_k - f_off_full

    where f_off = sum(posterior[off-peak region]) / sum(posterior[all]) is the
    fraction of the marginalised posterior probability that lies *outside* the
    signal region.  A good order concentrates probability in the peak, so
    removing it increases f_off (positive contribution).  A bad order spreads
    probability away from the peak, so removing it decreases f_off (negative).

    The off-peak region is defined as the grid points that are simultaneously
    far from the reference location in both vsys and Kp (i.e. the "corners" of
    the grid), controlled by ``vsys_excl`` and ``kp_excl``.  Using a region
    that is clearly away from any plausible signal makes the metric insensitive
    to the exact shape of the peak and robust to N-scaling effects.

    Only two calls to ``get_chi2_components`` are needed (total + per-order),
    and the LOO for each order k is obtained by subtracting its chi² components
    from the total — exact because chi² components are additive.

    Parameters
    ----------
    alpha_array : 1D array, optional — default 31 pts in [0.01, 2]
    idx_signal  : 1D int array, optional — default derived from alpha_frac > 0.5
    idx_orders  : array-like, optional — orders to include (default: all)
    kind        : {'BL', 'G'}
    kp_ref      : float, optional
        Expected planet Kp (km/s).  Centre of the exclusion zone for the
        off-peak mask.  Falls back to the map maximum if not provided.
    vsys_ref    : float, optional
        Expected planet vsys (km/s).  Used together with kp_ref.
    vsys_excl   : float — half-width of the vsys exclusion zone (km/s, default 30)
    kp_excl     : float — half-width of the Kp exclusion zone (km/s, default 30)

    Returns
    -------
    posterior_full : (n_vsys, n_kp) — alpha-marginalised posterior (max = 1)
    log_delta      : (n_orders, n_vsys, n_kp)
        ``log_post_full_norm − log_post_loo_norm_k`` at each grid point.
        Positive = order k boosts the posterior there; negative = it hurts.
        Each posterior is independently normalised (max = 0).
    contributions  : (n_orders,) — f_off_loo_k - f_off_full
        Positive = order sharpens the peak (good); negative = order spreads it.
        Orders with no valid pixels are assigned 0.
    contributions_frac : (n_orders,) — contributions as a fraction of the sum
        of all positive contributions.  Good orders: 0 to 1; bad orders: negative.
    idx_orders_out : (n_orders,) — order indices used
    """
    if alpha_array is None:
        alpha_array = np.linspace(0.01, 2., 31)
    if idx_signal is None:
        alpha_frac = _loaded_extra['alpha_frac']
        (idx_signal,) = np.nonzero(alpha_frac > 0.5)
    if idx_orders is None:
        idx_orders = np.arange(N.shape[-1])
    idx_orders = np.asarray(idx_orders)
    n_orders_used = len(idx_orders)
    d_alpha = alpha_array[1] - alpha_array[0]

    # ── Total chi² components (all orders, signal exposures) ──────────────
    ct_tot, st_tot, sf_tot, n_tot, us_tot = get_chi2_components(
        idx_orders=idx_orders, idx_exposure=idx_signal, sum_axis=(-2, -1),
    )  # shape (n_vsys, n_kp) for ct/st/sf/us; scalar for n_tot

    # ── Per-order chi² components (signal exposures summed, order axis kept) ─
    ct_ord, st_ord, sf_ord, n_ord, us_ord = get_chi2_components(
        idx_orders=idx_orders, idx_exposure=idx_signal, sum_axis=-2,
    )  # shape (n_vsys, n_kp, n_orders) for ct/st/sf/us; (n_orders,) for n_ord

    # ── Full posterior ─────────────────────────────────────────────────────
    logl_full_unnorm = _logl_from_components(
        ct_tot, st_tot, sf_tot, n_tot, us_tot, alpha_array, kind=kind,
    )  # (n_alpha, n_vsys, n_kp)
    log_post_full = _log_simps(logl_full_unnorm, d_alpha, axis=0)  # (n_vsys, n_kp)

    # Normalised posterior (max = 1) used for maps and fraction computation.
    lp_max_full = float(log_post_full[np.isfinite(log_post_full)].max())
    log_post_full_norm = log_post_full - lp_max_full
    posterior_full = np.exp(log_post_full_norm)

    # Reference location for the off-peak exclusion zone.
    if kp_ref is not None and vsys_ref is not None:
        i_vsys_ref = int(np.argmin(np.abs(vsys_axis - vsys_ref)))
        i_kp_ref   = int(np.argmin(np.abs(kp_axis   - kp_ref)))
    else:
        i_vsys_ref, i_kp_ref = np.unravel_index(
            np.argmax(posterior_full), posterior_full.shape
        )
    vsys_ctr = float(vsys_axis[i_vsys_ref])
    kp_ctr   = float(kp_axis[i_kp_ref])

    # Off-peak mask: grid points far from the reference in BOTH vsys and Kp.
    # Using AND (corners) ensures the region is cleanly off-signal.
    off_vsys = np.abs(vsys_axis - vsys_ctr) > vsys_excl   # shape (n_vsys,)
    off_kp   = np.abs(kp_axis   - kp_ctr)   > kp_excl     # shape (n_kp,)
    off_mask = np.outer(off_vsys, off_kp)                  # shape (n_vsys, n_kp)
    if off_mask.sum() < 10:
        import warnings
        warnings.warn(
            "Less than 10 off-peak grid points with vsys_excl="
            f"{vsys_excl} km/s and kp_excl={kp_excl} km/s. "
            "Consider reducing the exclusion zone.",
            UserWarning, stacklevel=2,
        )

    # Off-peak fraction for the full posterior.
    total_full = float(posterior_full.sum())
    f_off_full = float(posterior_full[off_mask].sum()) / total_full

    # ── Leave-one-out loop ─────────────────────────────────────────────────
    n_vsys_sz, n_kp_sz = log_post_full.shape
    log_delta     = np.zeros((n_orders_used, n_vsys_sz, n_kp_sz))
    contributions = np.zeros(n_orders_used)

    for k in range(n_orders_used):
        # Skip orders with no valid pixels — their contribution is exactly 0.
        if float(np.ma.filled(n_ord[k], 0)) == 0:
            continue

        ct_loo = ct_tot - ct_ord[..., k]
        st_loo = st_tot - st_ord[..., k]
        sf_loo = sf_tot - sf_ord[..., k]
        n_loo  = n_tot  - float(np.ma.filled(n_ord[k], 0))
        us_loo = us_tot - us_ord[..., k]

        logl_loo_unnorm = _logl_from_components(
            ct_loo, st_loo, sf_loo, n_loo, us_loo, alpha_array, kind=kind,
        )
        log_post_loo = _log_simps(logl_loo_unnorm, d_alpha, axis=0)

        # Guard: skip if LOO posterior is entirely degenerate.
        finite_loo = log_post_loo[np.isfinite(log_post_loo)]
        if len(finite_loo) == 0:
            continue

        # Off-peak fraction for this LOO posterior.
        lp_max_loo = float(finite_loo.max())
        log_post_loo_norm = log_post_loo - lp_max_loo
        post_loo = np.exp(log_post_loo_norm)
        total_loo = float(post_loo.sum())
        f_off_loo = float(post_loo[off_mask].sum()) / total_loo

        # Positive = removing order k raises f_off → order was concentrating
        # probability in the peak → good contribution.
        contributions[k] = f_off_loo - f_off_full

        # Normalised difference map for spatial visualisation.
        log_delta[k] = log_post_full_norm - log_post_loo_norm

    # Fractional contribution: each order as a share of total positive contribution.
    pos_total = float(contributions[contributions > 0].sum())
    contributions_frac = (contributions / pos_total
                          if pos_total > 0 else np.zeros_like(contributions))

    return posterior_full, log_delta, contributions, contributions_frac, idx_orders


def compute_alpha_kp_posterior(alpha_array=None, idx_signal=None,
                                idx_orders=None, oversample=2, kind='BL'):
    """Compute the vsys-marginalised alpha × Kp posterior.

    P(α, Kp | data) = ∫ L(α, vsys, Kp) d(vsys)

    Useful for checking whether the model amplitude α is constrained as a
    function of Kp, independently of the system velocity.

    Parameters
    ----------
    alpha_array : 1D array, optional — alpha values (default 31 pts in [0.01, 2]).
    idx_signal : 1D int array, optional
    idx_orders : array-like, optional
    oversample : int — oversampling factor for the 2D posterior.
    kind : {'BL', 'G'} — logL prescription (default 'BL').

    Returns
    -------
    posterior : (n_alpha_os, n_kp_os) array
    alpha_os  : (n_alpha_os,) array
    kp_os     : (n_kp_os,)   array
    margin_alpha : (n_alpha_os,) array — posterior marginalised over Kp
    margin_kp    : (n_kp_os,)   array — posterior marginalised over alpha
    """

    if alpha_array is None:
        alpha_array = np.linspace(0.01, 2., 31)
    if idx_signal is None:
        alpha_frac = _loaded_extra['alpha_frac']
        (idx_signal,) = np.nonzero(alpha_frac > 0.5)

    logl_cube = _build_logl_cube(alpha_array, idx_signal, idx_orders, kind=kind)
    d_vsys = vsys_axis[1] - vsys_axis[0]
    log_posterior = _log_simps(logl_cube, d_vsys, axis=1)   # (n_alpha, n_kp)

    return _posterior_from_log_map(log_posterior, alpha_array, kp_axis, oversample)


def compute_alpha_vsys_posterior(alpha_array=None, idx_signal=None,
                                  idx_orders=None, oversample=2, kind='BL'):
    """Compute the Kp-marginalised alpha × vsys posterior.

    P(α, vsys | data) = ∫ L(α, vsys, Kp) d(Kp)

    Parameters
    ----------
    alpha_array : 1D array, optional — alpha values (default 31 pts in [0.01, 2]).
    idx_signal : 1D int array, optional
    idx_orders : array-like, optional
    oversample : int — oversampling factor.

    Returns
    -------
    posterior : (n_alpha_os, n_vsys_os) array
    alpha_os  : (n_alpha_os,) array
    vsys_os   : (n_vsys_os,) array
    margin_alpha : (n_alpha_os,) array — posterior marginalised over vsys
    margin_vsys  : (n_vsys_os,) array — posterior marginalised over alpha
    """

    if alpha_array is None:
        alpha_array = np.linspace(0.01, 2., 31)
    if idx_signal is None:
        alpha_frac = _loaded_extra['alpha_frac']
        (idx_signal,) = np.nonzero(alpha_frac > 0.5)

    logl_cube = _build_logl_cube(alpha_array, idx_signal, idx_orders, kind=kind)
    d_kp = kp_axis[1] - kp_axis[0]
    log_posterior = _log_simps(logl_cube, d_kp, axis=2)   # (n_alpha, n_vsys)

    return _posterior_from_log_map(log_posterior, alpha_array, vsys_axis, oversample)


def get_log_norm_posterior(param_1, param_2, post_grid):
    """Return the log-normalised posterior on a (param_1, param_2) grid.

    Subtracts the log-evidence (2D Simpson integral over the grid) from the
    log-posterior, giving a proper normalised log-probability::

        log P_norm = log P - log ∫∫ exp(log P) dp1 dp2

    Parameters
    ----------
    param_1, param_2 : 1D arrays  (e.g. vsys_axis, kp_axis)
    post_grid : (n1, n2) array  — log-posterior values (any offset)

    Returns
    -------
    (n1, n2) array — log-posterior normalised so that exp() integrates to 1
    """
    from scipy import integrate

    d1 = param_1[-1] - param_1[0]
    d2 = param_2[-1] - param_2[0]

    exp_shift = np.floor(post_grid.min())
    pg = np.exp(post_grid - exp_shift)
    log_evidence = np.log(integrate.simps(integrate.simps(pg, dx=d1), dx=d2)) + exp_shift

    return post_grid - log_evidence


def compute_alpha_marginal(alpha_array=None, idx_signal=None, idx_orders=None):
    """Compute the marginal logL as a function of alpha.

    Two quantities are returned:

    * ``logl_at_peak`` — the 1D slice of the logl cube at the peak (vsys, Kp).
      Shows how the logL varies with model amplitude at the best-fit position.
    * ``log_p_alpha`` — the full marginal posterior::

          log P(α | data) = log ∫∫ L(α, vsys, Kp) d(vsys) d(Kp)

      A detection produces a peaked marginal (alpha constrained near 1).
      A non-detection gives a flat, uninformative marginal.

    Parameters
    ----------
    alpha_array : 1D array, optional
        Alpha values to scan.  Default: 31 points in [0.01, 2.0].
    idx_signal : 1D int array, optional
        Exposures to include.  Default: alpha_frac > 0.5 from loaded data.
    idx_orders : array-like, optional
        Orders to include.  Default: all.

    Returns
    -------
    alpha_array : 1D array
    logl_at_peak : 1D array — logL at (vsys_peak, Kp_peak) for each alpha
    log_p_alpha : 1D array — log marginal posterior over alpha
    vsys_peak, kp_peak : floats — position of the alpha-marginalised peak
    """
    if alpha_array is None:
        alpha_array = np.linspace(0.01, 2., 31)

    if idx_signal is None:
        alpha_frac = _loaded_extra['alpha_frac']
        (idx_signal,) = np.nonzero(alpha_frac > 0.5)

    # Build the 3D logl cube: shape (n_alpha, n_vsys, n_kp)
    logl_map_alpha = _build_logl_cube(alpha_array, idx_signal, idx_orders)
    d_alpha = alpha_array[1] - alpha_array[0]

    # Peak position: use the alpha-marginalised map
    log_post = _log_simps(logl_map_alpha, d_alpha, axis=0)
    max_ind = np.unravel_index(np.argmax(log_post), log_post.shape)
    vsys_peak = vsys_axis[max_ind[0]]
    kp_peak   = kp_axis[max_ind[1]]

    # 1D slice at the peak
    logl_at_peak = logl_map_alpha[:, max_ind[0], max_ind[1]]

    # Full marginal P(α|data): integrate L over (vsys, Kp)
    # Step 1: integrate over Kp axis (axis=-1)
    d_kp   = kp_axis[1]   - kp_axis[0]
    d_vsys = vsys_axis[1] - vsys_axis[0]
    log_marg_kp  = _log_simps(logl_map_alpha, d_kp,   axis=2)  # (n_alpha, n_vsys)
    log_p_alpha  = _log_simps(log_marg_kp,    d_vsys, axis=1)  # (n_alpha,)

    return alpha_array, logl_at_peak, log_p_alpha, vsys_peak, kp_peak


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

def _find_npz_files(out_path, stem):
    """Return sorted list of NPZ files matching ``{stem}_visit*.npz``."""
    files = sorted(out_path.glob(f'{stem}_visit*.npz'))
    if not files:
        raise FileNotFoundError(
            f'No NPZ files found matching {out_path / stem}_visit*.npz\n'
            'Run without --no-compute to generate them first.'
        )
    return files


def _make_trailing_plot(out_path, stem, kp_val, rv_expected=0.,
                         noise_rv_width=15., peak_rv_width=2.):
    """Generate and save a trailing plot from existing NPZ files.

    Parameters
    ----------
    kp_val : float
        Reference Kp (km/s) at which to slice the 3D map.
    rv_expected : float
        Expected v_sys (km/s) — used to centre the noise and peak windows.
    noise_rv_width : float
        Half-width of the RV baseline window (km/s). Points *outside*
        (rv_expected ± noise_rv_width) define the noise for normalisation.
    peak_rv_width : float
        Half-width of the peak window (km/s) for the lightcurve side panel.
    """
    from starships.plotting_fcts import plot_trailing_map

    files = _find_npz_files(out_path, stem)

    # The noise window is centred on rv_expected so the normalisation baseline
    # is symmetric around the expected signal position.
    noise_rv_limits = (rv_expected - noise_rv_width, rv_expected + noise_rv_width)
    peak_rv_limits  = (rv_expected - peak_rv_width,  rv_expected + peak_rv_width)

    # is_out_rv: boolean mask selecting RV values outside the noise window.
    # These points are free of planet signal and define the per-exposure baseline.
    is_out_rv = (vsys_axis < noise_rv_limits[0]) | (noise_rv_limits[1] < vsys_axis)

    # --- Combined 1D profile (all visits, out-of-eclipse exposures only) ---
    # Load all visits first to compute the reference combined profile shown in
    # grey behind the per-visit profile.  Only out-of-eclipse exposures are
    # summed: in-eclipse exposures do not contain planet signal and would
    # dilute the detection.
    load_logl_results(files)
    alpha_frac_all = _loaded_extra['alpha_frac']
    (idx_signal_all,) = np.nonzero(alpha_frac_all > 0.5)

    # Use the nearest Kp in the grid (argmin of absolute difference), not searchsorted.
    # searchsorted returns the insertion point, which can point to the wrong neighbour
    # when kp_val falls between two grid points.
    i_kp = int(np.argmin(np.abs(kp_axis - kp_val)))
    log.info(f'Trailing plot at Kp = {kp_axis[i_kp]:.2f} km/s (requested {kp_val:.2f})')

    logl_1d_all = get_logl(idx_exposure=idx_signal_all,
                            alpha=1., sum_axis=(-2, -1))[:, i_kp]
    logl_1d_all_norm = logl_1d_all - np.ma.median(logl_1d_all[is_out_rv])
    logl_1d_all_norm /= np.ma.std(logl_1d_all_norm[is_out_rv])

    # --- Per-visit trailing maps ---
    for i_file, filename in enumerate(files, start=1):
        load_logl_results([filename])

        # Phase may be absent from the NPZ (saved as an empty array when the data
        # pipeline did not produce it).  Fall back to integer exposure index.
        phase = _loaded_extra.get('phase', None)
        if phase is None or len(phase) == 0:
            phase = np.arange(N.shape[-2])

        # Sum over spectral pixels only (axis=-1) to keep the time axis.
        # Result shape: (n_vsys, n_kp, n_exp) → sliced to (n_vsys, n_exp).
        logl_map_ts = get_logl(alpha=1., sum_axis=-1)[:, i_kp, :]

        # Out-of-eclipse mask for this visit only
        visit_alpha = _loaded_extra['alpha_frac']
        (visit_signal,) = np.nonzero(visit_alpha > 0.5)

        # 1D profile: sum over out-of-eclipse exposures and orders → (n_vsys,)
        logl_1d = get_logl(idx_exposure=visit_signal, alpha=1.,
                            sum_axis=(-2, -1))[:, i_kp]

        # Normalise trailing map to S/N units.
        # For each exposure, subtract the median logL over the out-of-signal
        # RV range, then divide by the standard deviation over that range.
        # This removes the per-exposure baseline drift and puts all exposures
        # on a common scale.
        logl_map_norm = (logl_map_ts
                         - np.ma.median(logl_map_ts[is_out_rv, :], axis=0)[None, :])
        logl_map_norm /= np.ma.std(logl_map_ts[is_out_rv, :], axis=0)[None, :]

        logl_1d_norm = logl_1d - np.ma.median(logl_1d[is_out_rv])
        logl_1d_norm /= np.ma.std(logl_1d_norm[is_out_rv])

        # Build contact-phase dict for the plot (T1/T4 = solid, T2/T3 = dash-dot)
        contact_phases = _loaded_extra.get('contact_phases', None)
        if contact_phases is not None and len(contact_phases) == 4:
            phase_contacts = {
                '1_4': [float(contact_phases[0]), float(contact_phases[3])],
                '2_3': [float(contact_phases[1]), float(contact_phases[2])],
            }
        else:
            phase_contacts = None

        save_path = out_path / f'trailing_{stem}_visit{i_file}.pdf'  # i_file starts at 1
        plot_trailing_map(
            logl_map_norm, vsys_axis, phase, logl_1d_norm,
            noise_rv_limits=noise_rv_limits,
            peak_rv_limits=peak_rv_limits,
            logl_1d_norm_all=logl_1d_all_norm,
            rv_expected=rv_expected,
            phase_contacts=phase_contacts,
            save_path=save_path,
        )
        log.info(f'Saved trailing plot: {save_path}')

    # Reload all visits so the module globals are in a consistent state
    load_logl_results(files)


def _make_kpvsys_plot(out_path, stem, alpha_array=None, oversample=2):
    """Generate and save a Kp-Vsys map from existing NPZ files.

    Parameters
    ----------
    alpha_array : 1D array, optional
        Alpha values to marginalize over (see compute_kpvsys_posterior).
    oversample : int
        Oversampling factor for the posterior (see compute_kpvsys_posterior).
    """
    from starships.plotting_fcts import plot_kpvsys_map

    files = _find_npz_files(out_path, stem)
    load_logl_results(files)

    # Full posterior computation: alpha marginalisation + oversampling + marginals
    posterior, vsys_os, kp_os, margin_vsys, margin_kp = compute_kpvsys_posterior(
        alpha_array=alpha_array, oversample=oversample,
    )

    save_path = out_path / f'kpvsys_{stem}.pdf'
    plot_kpvsys_map(posterior, vsys_os, kp_os, margin_vsys, margin_kp,
                    save_path=save_path)
    log.info(f'Saved Kp-Vsys map: {save_path}')


def main():
    """Entry point for ``run_starships_logl_grid`` (registered in setup.py).

    Usage::

        # Compute grid
        run_starships_logl_grid --config ~/scratch/.../cfg.yaml --specfile ~/scratch/.../model.npz

        # Compute + plot
        run_starships_logl_grid --config cfg.yaml --specfile model.npz --plot trailing --kp 227
        run_starships_logl_grid --config cfg.yaml --specfile model.npz --plot kpvsys

        # Plot only (NPZ already computed)
        run_starships_logl_grid --config cfg.yaml --specfile model.npz --plot trailing --kp 227 --no-compute
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog='run_starships_logl_grid',
        description='Compute a Kp × v_sys logL grid for high-resolution spectroscopy.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--config', required=True, type=Path,
        help='Path to the retrieval input YAML (~ is expanded).',
    )
    parser.add_argument(
        '--specfile', required=True, type=Path,
        help='Path to the model spectrum NPZ (keys: wave, spec). ~ is expanded.',
    )
    parser.add_argument(
        '--plot', choices=['trailing', 'kpvsys'], default=None,
        help='Which figure to produce after computing (or with --no-compute).',
    )
    parser.add_argument(
        '--kp', type=float, default=0.,
        metavar='KP_KM_S',
        help='Reference Kp in km/s for the trailing plot Kp slice.',
    )
    parser.add_argument(
        '--rv', type=float, default=0.,
        metavar='RV_KM_S',
        help='Expected vsys in km/s — centres the noise window for the trailing plot.',
    )
    parser.add_argument(
        '--no-compute', action='store_true',
        help='Skip grid computation; load existing NPZ files and go straight to plotting.',
    )

    args = parser.parse_args()

    # Expand ~ in paths supplied on the command line
    config   = args.config.expanduser()
    specfile = args.specfile.expanduser()

    setup_logl_grid(config)

    out_path = (
        Path(logl_grid_output_path).expanduser()
        if logl_grid_output_path
        else Path('.')
    )
    stem = specfile.stem

    if not args.no_compute:
        load_logl_grid_data()
        load_model(specfile)

        rv_arr = _make_grid(rv_grid)
        kp_arr = _make_grid(kp_grid)
        n_exp_per_visit = max(dt['flux'].shape[0] for dt in data_visits)
        estimate_logl_grid_memory(
            n_vsys=len(rv_arr),
            n_kp=len(kp_arr),
            n_exp_per_visit=n_exp_per_visit,
            n_ord=len(idx_orders),
            n_visits=len(data_visits),
        )

        compute_and_save_logl_grid(out_path, stem)

    if args.plot == 'trailing':
        _make_trailing_plot(out_path, stem, args.kp, rv_expected=args.rv)
    elif args.plot == 'kpvsys':
        _make_kpvsys_plot(out_path, stem)


if __name__ == '__main__':
    main()
