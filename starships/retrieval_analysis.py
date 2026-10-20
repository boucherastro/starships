"""Tools to analyse the output of a retrieval (walker files) -- shared by the analysis
notebooks (`tutorials/retrieval_analysis/`) and by the automatic figure script.

The notebooks are interactive: the user looks at the chains, decides which walkers/steps
to keep, then generates spectra, profiles and figures from what was kept. The functions
of this module are the building blocks of those steps. They take arrays (or the imported
`starships.retrieval` module, after `retrieval.setup_retrieval(yaml_file)`) as inputs and
return arrays, so they can be used one by one in a notebook or chained in a script.

Conventions
-----------
- A *posterior sample* is a flattened array `pos` of shape (n_samples, n_params) sorted by
  log-likelihood (worst first, **best last**), with the matching array `logl`. This is the
  format returned by `load_posterior_sample` and used everywhere else.
- Parameter order in the chains is the order of `retrieval.params_prior` *after*
  `retrieval.setup_retrieval` (which adds the fitted species and the per-region parameters).

Sections
--------
1. Reading and selecting the walkers (`load_posterior_sample`, `select_samples`,
   `combine_walker_files`).
2. What can be done with a run (`describe_run`).
3. Profiles, spectra and rotation kernels drawn from a sample (`draw_region_profiles`,
   `equilibrium_profiles`, `draw_spectra`, `synthetic_data_samples`, `draw_rotation_kernels`). These need the
   `retrieval` module set up with the run's yaml file; the spectra need petitRADTRANS.
4. Abundance ratios computed from a region's abundance profiles (`abundance_ratio_profiles`,
   `metallicity_profiles`, `profile_stats_at_pressures`, `fit_chemical_equilibrium_at_pressure`).
   Single-region only for now (see that section's docstring).

This module is being built step by step; figures will be added in a later section.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import astropy.constants as const
import emcee
import h5py
import numpy as np

from . import retrieval_utils as ru
from .petitradtrans_utils import gen_abundances, mass_frac_2_vmr

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

PathLike = Union[str, Path]


# =====================================================================================
# Reading and selecting the walkers
# =====================================================================================

def load_posterior_sample(walker_file: PathLike, discard: int = 0,
                          tol: int = 8) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Read a walker file and return its flattened posterior sample, sorted by log-likelihood.

    Parameters
    ----------
    walker_file : str or Path
        Walker file (`.h5`) written by the retrieval (emcee HDF5 backend).
    discard : int, default 0
        Number of steps to discard at the beginning of each walker's chain (burn-in).
    tol : int, default 8
        Tolerance of the autocorrelation-time estimate (`emcee.autocorr.integrated_time`).
        The estimate is only reliable when the chain is longer than `tol` times the
        autocorrelation time. If it is not (typical for burn-in files or very short runs),
        the autocorrelation time is skipped with a warning instead of raising an error.
        Use `tol=0` to skip it on purpose (faster for long chains).

    Returns
    -------
    ord_pos : np.ndarray
        Parameters of every (step, walker) pair, shape (n_samples, n_params), from the
        lowest to the highest log-likelihood (the **last** row is the best sample).
    ord_logl : np.ndarray
        Log-likelihood of each row of `ord_pos`, shape (n_samples,).
    autocorr : np.ndarray or None
        Estimated integrated autocorrelation time of each parameter (in steps), or None if
        it could not be estimated (or `tol=0`).

    Raises
    ------
    ValueError
        If `discard` is not smaller than the number of steps in the file.
    """
    walker_file = Path(walker_file)

    # Check the number of steps first, to give a clear message instead of an empty array later.
    n_steps = emcee.backends.HDFBackend(str(walker_file), read_only=True).iteration
    if discard >= n_steps:
        raise ValueError(f"`discard`={discard} must be smaller than the number of steps in "
                         f"{walker_file.name} ({n_steps}).")

    try:
        ord_pos, ord_logl, autocorr = ru.read_walker_prob(walker_file, tol=tol, discard=discard)
    except emcee.autocorr.AutocorrError:
        # The chain is too short for a reliable autocorrelation estimate. Not a problem for
        # exploring the sample, so read again without it.
        log.warning(f"Chain too short to estimate the autocorrelation time with tol={tol} "
                    f"({n_steps - discard} steps kept): autocorrelation not computed.")
        ord_pos, ord_logl, _ = ru.read_walker_prob(walker_file, tol=0, discard=discard)
        autocorr = None

    if tol <= 1:
        autocorr = None  # `read_walker_prob` returns zeros in that case: not a real estimate

    return ord_pos, ord_logl, autocorr


def select_samples(ord_pos: np.ndarray, ord_logl: np.ndarray,
                   param_names: Optional[Sequence[str]] = None,
                   logl_min: Optional[float] = None,
                   param_cuts: Optional[Dict[Union[str, int], Tuple[Optional[float], Optional[float]]]] = None,
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only the samples above a log-likelihood threshold and/or inside parameter ranges.

    This is the declarative version of the manual selection done in the interactive
    notebook (`cond = ord_logl > x; cond &= ord_pos[:, idx] < y`), used by the automatic
    figure script. All the conditions are combined with a logical AND.

    Parameters
    ----------
    ord_pos, ord_logl : np.ndarray
        Posterior sample and its log-likelihoods (see `load_posterior_sample`).
    param_names : sequence of str, optional
        Name of each column of `ord_pos` (e.g. `list(retrieval.params_prior)`). Only needed
        if `param_cuts` uses names instead of column indices.
    logl_min : float, optional
        Keep only samples with `logl > logl_min`. None keeps everything.
    param_cuts : dict, optional
        `{parameter: (min, max)}`, the parameter being a name (needs `param_names`) or a
        column index. Use None for a missing bound, e.g. `{'kp': (190, 200), 'rv': (10, None)}`.
        The bounds are compared with the values stored in the chains (so log10 for
        parameters with a log-uniform prior).

    Returns
    -------
    pos_sub, logl_sub : np.ndarray
        The selected samples, still sorted by log-likelihood.

    Raises
    ------
    KeyError
        If a parameter name of `param_cuts` is not in `param_names`.
    ValueError
        If nothing is left after the selection (better than failing later with an empty array).
    """
    keep = np.ones(len(ord_logl), dtype=bool)

    if logl_min is not None:
        keep &= ord_logl > logl_min

    for param, (low, high) in (param_cuts or {}).items():
        if isinstance(param, str):
            if param_names is None:
                raise ValueError("`param_names` is needed to select a parameter by name.")
            if param not in param_names:
                raise KeyError(f"Parameter '{param}' not found. Available: {list(param_names)}")
            idx = list(param_names).index(param)
        else:
            idx = param
        if low is not None:
            keep &= ord_pos[:, idx] > low
        if high is not None:
            keep &= ord_pos[:, idx] < high

    if not keep.any():
        raise ValueError("No sample left after the selection: relax `logl_min`/`param_cuts`.")

    log.info(f"Selection keeps {keep.sum()} / {len(keep)} samples.")

    return ord_pos[keep], ord_logl[keep]


def combine_walker_files(walker_files: Sequence[PathLike], file_out: PathLike,
                         walker_path: Optional[PathLike] = None, n_steps: Optional[int] = None,
                         overwrite: bool = False) -> Path:
    """Merge several walker files into one, stacking their walkers side by side.

    Typical use: the burn-in of a retrieval is run as a SLURM array, giving one walker file
    per array task (each with its own walkers). To continue from the best of them, or to
    look at all of them together, they are merged into a single file (readable with
    `load_posterior_sample` and usable as `walker_file_in` in the retrieval yaml file).

    Only the last `n_steps` steps of each file are kept, so that all the files contribute
    the same number of steps (the merged chains must have a rectangular shape).

    Parameters
    ----------
    walker_files : sequence of str or Path
        Files to merge (all must have the same number of parameters).
    file_out : str or Path
        Merged file to write.
    walker_path : str or Path, optional
        Directory of `walker_files` and `file_out` (if they are not full paths already).
    n_steps : int, optional
        Number of steps to keep from the end of each file. By default the smallest number of
        steps completed among the files (so nothing is lost from the shortest file). It can
        be set to a smaller number to keep only the end of the chains, e.g. `n_steps=20`.
    overwrite : bool, default False
        Replace `file_out` if it already exists. Otherwise a `FileExistsError` is raised.

    Returns
    -------
    Path
        Path of the merged file.

    Raises
    ------
    FileExistsError
        If `file_out` exists and `overwrite` is False.
    ValueError
        If the files do not have the same number of parameters, or `n_steps` is larger
        than the shortest file.
    """
    walker_path = Path(walker_path) if walker_path is not None else Path('.')
    files_in = [walker_path / Path(f) for f in walker_files]
    file_out = walker_path / Path(file_out)

    if file_out.exists() and not overwrite:
        raise FileExistsError(f"{file_out} already exists. Use `overwrite=True` to replace it.")

    # Number of completed steps in each file (the HDF5 datasets can be longer than that
    # because emcee allocates them in blocks)
    completed = []
    for f_in in files_in:
        with h5py.File(f_in, 'r') as f:
            completed.append(int(f['mcmc'].attrs['iteration']))
            log.info(f"{f_in.name}: {completed[-1]} steps")

    n_available = min(completed)
    if n_steps is None:
        n_steps = n_available
    elif n_steps > n_available:
        raise ValueError(f"`n_steps`={n_steps} is larger than the shortest file ({n_available} steps).")

    # Collect the last `n_steps` steps of every file
    chains, log_probs, accepted = [], [], []
    for f_in, last_step in zip(files_in, completed):
        with h5py.File(f_in, 'r') as f:
            dset = f['mcmc']
            chains.append(dset['chain'][last_step - n_steps:last_step])
            log_probs.append(dset['log_prob'][last_step - n_steps:last_step])
            accepted.append(dset['accepted'][...])

    ndims = {chain.shape[-1] for chain in chains}
    if len(ndims) != 1:
        raise ValueError(f"The files do not have the same number of parameters: {sorted(ndims)}")

    # Stack the walkers (axis 1 of chain and log_prob, axis 0 of accepted)
    combined = {'chain': np.concatenate(chains, axis=1),
                'log_prob': np.concatenate(log_probs, axis=1),
                'accepted': np.concatenate(accepted)}

    # Write in the format of the emcee HDF5 backend, so that emcee/`load_posterior_sample` can read it
    with h5py.File(file_out, 'w') as f:
        grp = f.create_group('mcmc')
        for key, val in combined.items():
            grp[key] = val
        grp.attrs['has_blob'] = False
        for key, val in zip(['iteration', 'nwalkers', 'ndim'], combined['chain'].shape):
            grp.attrs[key] = val

    log.info(f"Merged {len(files_in)} files into {file_out} "
             f"({combined['chain'].shape[1]} walkers, {n_steps} steps).")

    return file_out


# =====================================================================================
# What can be done with a given retrieval run?
# =====================================================================================

@dataclass
class RunInfo:
    """What a retrieval run allows to do, detected from the loaded `retrieval` module.

    The notebooks and the figure script use this to switch sections on/off (instead of
    crashing when e.g. there is no photometry, or no high-resolution model).

    Attributes
    ----------
    retrieval_type : str
        'HRR' (high-res only), 'LRR' (low-res only) or 'JR' (joint), as in the yaml file.
    kind_trans : str
        'emission' or 'transmission'.
    region_ids : list of int
        Region labels (`region_id` in the yaml file). More than one means a multi-region model.
    can_model_high : bool
        A high-resolution (line-by-line) model can be computed. This does not depend on
        the retrieval type: it is true whenever `retrieval.wv_range_high` is not empty, i.e.
        when the yaml file lists a high-res instrument (`instrum`) or a low-res instrument
        modelled line-by-line (`opacity_mode: 'lbl'`). So an LRR run may well have it.
    can_model_low : bool
        A low-resolution (correlated-k) model can be computed (`retrieval.wv_range_low`).
    has_high_res_data : bool
        High-resolution observations were fitted (retrieval type 'HRR' or 'JR').
    spectrophotometric, photometric : dict
        `{instrument name: 'low' or 'high'}`: which model (c-k or line-by-line) generates the
        synthetic data of each low-resolution instrument.
    """
    retrieval_type: str
    kind_trans: str
    region_ids: List[int]
    can_model_high: bool
    can_model_low: bool
    has_high_res_data: bool
    spectrophotometric: Dict[str, str] = field(default_factory=dict)
    photometric: Dict[str, str] = field(default_factory=dict)

    @property
    def n_regions(self) -> int:
        return len(self.region_ids)

    @property
    def is_multi_region(self) -> bool:
        return self.n_regions > 1

    @property
    def low_res_instruments(self) -> Dict[str, str]:
        """All the low-resolution instruments (spectrophotometric + photometric)."""
        return {**self.spectrophotometric, **self.photometric}

    @property
    def has_low_res_data(self) -> bool:
        return len(self.low_res_instruments) > 0

    def summary(self) -> str:
        """Multi-line, human readable description (printed at the top of the notebooks)."""
        yes_no = {True: 'yes', False: 'no'}
        lines = [f"Retrieval type          : {self.retrieval_type} ({self.kind_trans})",
                 f"Regions                 : {self.region_ids}"
                 + (' (multi-region)' if self.is_multi_region else ''),
                 f"High-res observations   : {yes_no[self.has_high_res_data]}",
                 f"High-res model possible : {yes_no[self.can_model_high]}",
                 f"Low-res model possible  : {yes_no[self.can_model_low]}"]
        for name, model_type in self.low_res_instruments.items():
            lines.append(f"Low-res instrument      : {name} (data modelled at "
                         f"{'high (lbl)' if model_type == 'high' else 'low (c-k)'} resolution)")
        return '\n'.join(lines)


def describe_run(retrieval_obj) -> RunInfo:
    """Detect what a retrieval run allows to do (see `RunInfo`).

    Parameters
    ----------
    retrieval_obj : module
        The `starships.retrieval` module, after `retrieval_obj.setup_retrieval(yaml_file)`.

    Returns
    -------
    RunInfo
    """
    def model_types(data_dict: Optional[dict]) -> Dict[str, str]:
        # `model_type` is only assigned by `setup_retrieval` for joint retrievals or when a
        # low-res instrument is modelled line-by-line; otherwise the c-k ('low') model is used.
        return {name: infos.get('model_type', 'low') for name, infos in (data_dict or {}).items()}

    retrieval_type = retrieval_obj.retrieval_type

    return RunInfo(
        retrieval_type=retrieval_type,
        kind_trans=retrieval_obj.kind_trans,
        region_ids=list(retrieval_obj.region_id),
        can_model_high=len(retrieval_obj.wv_range_high) > 0,
        can_model_low=len(retrieval_obj.wv_range_low) > 0,
        has_high_res_data=retrieval_type in ('HRR', 'JR'),
        spectrophotometric=model_types(getattr(retrieval_obj, 'spectrophotometric_data', None)),
        photometric=model_types(getattr(retrieval_obj, 'photometric_data', None)),
    )


# =====================================================================================
# Profiles, spectra and rotation kernels drawn from a posterior sample
# =====================================================================================
# All the functions below draw `n_draw` random samples among the rows of `flat_samples`
# (typically the selection made in the notebook). The draws are not repeated. Pass `seed`
# to get the same draws again (e.g. to compare the profiles and the spectra of the same
# samples, or to reproduce a figure).

def draw_sample_indices(n_draw: int, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
    """Random row indices for `n_draw` distinct samples (fewer if `n_samples` is smaller)."""
    rng = np.random.default_rng(seed)
    return rng.choice(n_samples, size=min(n_draw, n_samples), replace=False)


def compute_region_profiles(theta_regions: List[dict], retrieval_obj
                            ) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]]]:
    """Temperature and abundance (volume mixing ratio) profiles of every region for one theta.

    Same recipe as the notebook: the abundances are computed with
    `petitradtrans_utils.gen_abundances` from the parameters of each region (free chemistry,
    H2 and He fill the rest, with dissociation if `retrieval_obj.dissociation`).

    Parameters
    ----------
    theta_regions : list of dict
        One parameter dictionary per region, from `retrieval_obj.unpack_theta`.
    retrieval_obj : module
        The `starships.retrieval` module, set up with the run's yaml file.

    Returns
    -------
    pressures : np.ndarray
        Pressure grid (bar), shared by all the regions.
    profiles : list of dict
        One dictionary per region, `{'temperature': array, 'H2O': array, ...}` with one
        array per species (all the species of the model, including H2 and He).

    Raises
    ------
    NotImplementedError
        For chemical-equilibrium retrievals (the abundances then do not come from the
        fitted parameters directly).
    """
    if getattr(retrieval_obj, 'chemical_equilibrium', False):
        raise NotImplementedError("Abundance profiles are only implemented for free-chemistry "
                                  "retrievals (chemical_equilibrium: false).")

    species = retrieval_obj.line_opacities + retrieval_obj.continuum_opacities + retrieval_obj.other_species

    profiles = []
    for theta_dict in theta_regions:
        # `gen_abundances` appends H2 and He to the lists it receives: give it copies
        _, _, vmr = gen_abundances(list(species), [theta_dict[key] for key in species],
                                   theta_dict['pressures'], theta_dict['temperatures'],
                                   verbose=False, vmrh2he=[0.85, 0.15],
                                   dissociation=retrieval_obj.dissociation, plot=False)
        profile = {'temperature': np.asarray(theta_dict['temperatures'])}
        profile.update({name: np.asarray(profile_vmr) for name, profile_vmr in vmr.items()})
        profiles.append(profile)

    return theta_regions[0]['pressures'], profiles


def draw_region_profiles(n_draw: int, flat_samples: np.ndarray, retrieval_obj,
                         seed: Optional[int] = None, vmr_range: Tuple[float, float] = (1e-20, 1.),
                         temperature_range: Tuple[float, float] = (0., 10000.)
                         ) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]]]:
    """Draw profiles (temperature and abundances) of every region from a posterior sample.

    Parameters
    ----------
    n_draw : int
        Number of samples to draw.
    flat_samples : np.ndarray
        Posterior sample, shape (n_samples, n_params) (e.g. `ord_pos_sub` in the notebook).
    retrieval_obj : module
        The `starships.retrieval` module, set up with the run's yaml file.
    seed : int, optional
        Seed of the random draw (see the note above this section).
    vmr_range, temperature_range : 2-tuple of float
        Profiles are clipped to these ranges, to keep the statistics (computed in log
        space for the abundances) finite.

    Returns
    -------
    pressures : np.ndarray
        Pressure grid (bar), shape (n_pressures,).
    profile_samples : list of dict
        One dictionary per region (same order as `retrieval_obj.region_id`), each with the
        keys of `compute_region_profiles` and arrays of shape (n_drawn, n_pressures).
    """
    idx = draw_sample_indices(n_draw, len(flat_samples), seed)

    n_regions = len(retrieval_obj.region_id)
    collected = [dict() for _ in range(n_regions)]
    for params in flat_samples[idx]:
        theta_regions = retrieval_obj.unpack_theta(params.copy())
        pressures, profiles = compute_region_profiles(theta_regions, retrieval_obj)
        for region_samples, profile in zip(collected, profiles):
            for name, values in profile.items():
                region_samples.setdefault(name, []).append(values)

    # Convert to arrays and clip to reasonable values
    profile_samples = []
    for region_samples in collected:
        region_arrays = {}
        for name, values in region_samples.items():
            low, high = temperature_range if name == 'temperature' else vmr_range
            region_arrays[name] = np.clip(np.array(values), low, high)
        profile_samples.append(region_arrays)

    return pressures, profile_samples


def compute_profile_stats(profile_samples: Dict[str, np.ndarray]) -> Dict[str, dict]:
    """Median and confidence intervals (1, 2 and 3 sigma) of every profile of one region.

    Computed in log space for the abundances and in linear space for the temperature, as
    in the notebook (see `retrieval_utils.get_stats_from_profile` for the content of each
    entry).

    Parameters
    ----------
    profile_samples : dict
        One element of the list returned by `draw_region_profiles`.

    Returns
    -------
    dict
        Same keys as `profile_samples`, each with the statistics dictionary.
    """
    return {name: ru.get_stats_from_profile(values, log_scale=(name != 'temperature'))
            for name, values in profile_samples.items()}


def equilibrium_profiles(profile_samples: Dict[str, np.ndarray], pressures: np.ndarray,
                         c_to_o: float = 0.95, metallicity: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Abundance profiles expected from chemical equilibrium, for a sample of temperature profiles.

    For each drawn temperature profile, the abundances come from the petitRADTRANS
    equilibrium-chemistry tables (`poor_mans_nonequ_chem.interpol_abundances`) at a fixed C/O and
    metallicity. To compare with the free-chemistry profiles of the same draws
    (`draw_region_profiles`). Needs petitRADTRANS.

    NOTE on naming: `metallicity` here is the input knob of petitRADTRANS's equilibrium-chemistry
    tables (a bulk enrichment of every heavy element) -- it is NOT literally the elemental Fe/H
    ratio, even though Fe/H is the standard proxy used to estimate it (the default below) and the
    two are numerically equal by construction in that default case. Use `metallicity_profiles(...,
    'Fe/H')` for the actual elemental ratio (e.g. to plot it, or to pass a measured value in here).

    Parameters
    ----------
    profile_samples : dict
        Profiles of one region from `draw_region_profiles`: 'temperature' and the abundances of all
        the species, arrays of shape (n_draw, n_pressures).
    pressures : np.ndarray
        Pressure grid (bar) of the profiles.
    c_to_o : float, default 0.95
        Carbon-to-oxygen ratio, the same at all pressures (chosen by hand).
    metallicity : float, optional
        Equilibrium-chemistry metallicity, in dex. By default it is estimated from the
        free-chemistry Fe/H elemental ratio of each draw if `Fe` is in `profile_samples` (the
        standard metallicity-from-Fe/H proxy), and solar (0) otherwise.

    Returns
    -------
    dict
        `{species: array (n_draw, n_pressures)}` of volume mixing ratios, for all the species of the
        equilibrium tables.
    """
    # petitRADTRANS is only needed here, so it is imported here
    from petitRADTRANS.poor_mans_nonequ_chem import interpol_abundances

    species = [key for key in profile_samples if key != 'temperature']
    temperatures = profile_samples['temperature']

    if metallicity is None and 'Fe' in profile_samples:
        abundances = [profile_samples[key] for key in species]
        Fe_profiles = ru.get_atom_abundance('Fe', abundances, list_species=species)
        H_profiles = ru.get_atom_abundance('H', abundances, list_species=species)
        # log10 VMR_Fe / VMR_H = -4.46 (=12. - 7.54)  (solar protosolar metallicity form Asplund 2009 table 5)
        # This is the elemental Fe/H ratio, used here only as the standard proxy for the
        # equilibrium-chemistry metallicity knob -- the two concepts are not the same thing.
        metallicity_estimate = np.log10(Fe_profiles / H_profiles) - (-4.46)
    else:
        metallicity_estimate = np.full(temperatures.shape, 0. if metallicity is None else metallicity)

    c_to_o_profile = c_to_o * np.ones_like(pressures)

    collected: Dict[str, list] = {}
    for i_sample, temperature in enumerate(temperatures):
        mass_fractions = interpol_abundances(c_to_o_profile, metallicity_estimate[i_sample], temperature, pressures)
        vmrs = mass_frac_2_vmr(mass_frac_dict=mass_fractions)
        for specie, vmr in vmrs.items():
            collected.setdefault(specie, []).append(vmr)

    return {specie: np.array(values) for specie, values in collected.items()}


def draw_spectra(n_draw: int, flat_samples: np.ndarray, retrieval_obj, kind_res: str = 'low',
                 species_list: Optional[list] = None, by_region: bool = False,
                 seed: Optional[int] = None
                 ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
    """Draw model spectra from a posterior sample, optionally region by region and species by species.

    The spectra are made with `retrieval_obj.prepare_static_model`, the same entry point the
    retrieval uses for the low-resolution data, so they match what was fitted. This needs
    petitRADTRANS (the first call is slow, the atmosphere objects are initialised).

    Parameters
    ----------
    n_draw : int
        Number of samples to draw.
    flat_samples : np.ndarray
        Posterior sample, shape (n_samples, n_params).
    retrieval_obj : module
        The `starships.retrieval` module, set up with the run's yaml file.
    kind_res : {'low', 'high'}, default 'low'
        'low' is the correlated-k model, fast. 'high' is the line-by-line model, much slower.
        Check `describe_run(retrieval_obj).can_model_high` first.
    species_list : list, optional
        Species (or lists of species, to be taken together) whose contribution is also
        computed, as in `retrieval_utils.get_contribution`. Each one adds 2 spectra per
        draw. Default: only the total spectrum.
    by_region : bool, default False
        For a multi-region model, also compute the spectrum each region would give if it
        covered the whole planet (`'region_<id>'`), in addition to the combined one
        (`'global'`). Ignored for a single region.
    seed : int, optional
        Seed of the random draw (same draws as `draw_region_profiles` with the same seed).

    Returns
    -------
    waves : dict
        `{'global': wavelengths, 'region_1': ...}`. Kept per entry because the combined
        multi-region spectrum can have a slightly shorter grid than the single-region ones.
    spectra : dict
        `{'global': {'All': array (n_drawn, n_wave), 'H2O only': ..., 'Without H2O': ...,
        'No Species': ...}, 'region_1': {...}}`. The species entries only exist if
        `species_list` is given.
    """
    idx = draw_sample_indices(n_draw, len(flat_samples), seed)

    waves: Dict[str, np.ndarray] = {}
    collected: Dict[str, Dict[str, list]] = {}
    for params in flat_samples[idx]:
        theta_regions = retrieval_obj.unpack_theta(params.copy())

        # What to compute for this draw: the whole planet, and maybe each region on its own
        to_compute = {'global': theta_regions}
        if by_region and len(theta_regions) > 1:
            for region, theta_dict in zip(retrieval_obj.region_id, theta_regions):
                to_compute[f'region_{region}'] = [theta_dict]

        for name, theta_list in to_compute.items():
            if species_list:
                wave, spectra = ru.get_contribution(species_list, retrieval_obj, theta_list, mode=kind_res)
            else:
                wave, spec, _ = retrieval_obj.prepare_static_model(theta_list, kind_res)
                spectra = {'All': spec}
            waves[name] = wave
            for key, spec in spectra.items():
                collected.setdefault(name, {}).setdefault(key, []).append(spec)

    spectra_out = {name: {key: np.array(values) for key, values in entry.items()}
                   for name, entry in collected.items()}

    return waves, spectra_out


def synthetic_data_samples(spec_samples_low: Optional[np.ndarray], retrieval_obj,
                           wave_low: Optional[np.ndarray] = None,
                           spec_samples_high: Optional[np.ndarray] = None,
                           wave_high: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Synthetic data of every low-resolution instrument for a set of drawn spectra.

    Each instrument uses the model that was used to fit it: the correlated-k spectra
    (`spec_samples_low`) or the line-by-line ones (`spec_samples_high`), see
    `describe_run(retrieval_obj).low_res_instruments`. Spectrophotometry and photometry are
    both included.

    Parameters
    ----------
    spec_samples_low : np.ndarray or None
        Low-resolution spectra of the draws, shape (n_draw, n_wave), e.g. `spectra['global']['All']`
        from `draw_spectra(..., kind_res='low')`. Can be None if no instrument uses it.
    retrieval_obj : module
        The `starships.retrieval` module, set up with the run's yaml file.
    wave_low : np.ndarray, optional
        Wavelengths of `spec_samples_low` (`waves['global']`).
    spec_samples_high, wave_high : np.ndarray, optional
        Same for the line-by-line spectra, needed only if an instrument is modelled at high
        resolution.

    Returns
    -------
    dict
        `{instrument name: array (n_draw, n_data_points)}`.

    Raises
    ------
    ValueError
        If an instrument needs the low- or high-resolution spectra and they were not given.
    """
    model_types = describe_run(retrieval_obj).low_res_instruments
    for model_type, spec_samples in [('low', spec_samples_low), ('high', spec_samples_high)]:
        needing = [name for name, kind in model_types.items() if kind == model_type]
        if needing and spec_samples is None:
            raise ValueError(f"{needing} use the {model_type}-resolution model: give "
                             f"`spec_samples_{model_type}` (draw them with kind_res='{model_type}').")

    n_draw = len(spec_samples_low if spec_samples_low is not None else spec_samples_high)

    collected: Dict[str, list] = {}
    for i_draw in range(n_draw):
        synt_data = ru.get_syntetic_data_low_res(
            wv_low=wave_low, spec_low=None if spec_samples_low is None else spec_samples_low[i_draw],
            wv_high=wave_high, spec_high=None if spec_samples_high is None else spec_samples_high[i_draw],
            retrieval_obj=retrieval_obj)
        for name, data in synt_data.items():
            collected.setdefault(name, []).append(data)

    return {name: np.array(values) for name, values in collected.items()}


def draw_rotation_kernels(n_draw: int, flat_samples: np.ndarray, retrieval_obj,
                          phase: Optional[float] = None, instrum: Optional[dict] = None,
                          seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Draw the rotation kernel of every region from a posterior sample (multi-region models).

    Calls the `get_ker` function of the run (`get_ker_file` in the yaml file) for each draw,
    exactly as the retrieval does, so the kernels include the instrumental resolution and
    the weight of each region (visibility) at `phase`. The kernels of the regions add up to 1.

    Parameters
    ----------
    n_draw : int
        Number of samples to draw.
    flat_samples : np.ndarray
        Posterior sample, shape (n_samples, n_params).
    retrieval_obj : module
        The `starships.retrieval` module, set up with the run's yaml file.
    phase : float, optional
        Orbital phase (0 = mid-transit, 0.5 = secondary eclipse) at which the kernels are
        computed. Default: mid-eclipse (0.5) in emission, mid-transit (0) in transmission.
    instrum : dict, optional
        Instrument parameters passed to `get_ker` (needs at least its `'resol'`). Default:
        the first instrument of the run.
    seed : int, optional
        Seed of the random draw.

    Returns
    -------
    velocity : np.ndarray
        Velocity axis (km/s), centred on 0, shape (n_v,).
    kernels : np.ndarray
        Kernels, shape (n_drawn, n_regions, n_v). The kernel size depends on the parameters
        (e.g. the wind speed), so each one is placed on a common velocity axis (padded with
        zeros).

    Raises
    ------
    ValueError
        If the run has no `get_ker_file`, or if no instrument is available.
    """
    if phase is None:
        phase = 0.5 if retrieval_obj.kind_trans == 'emission' else 0.0

    if instrum is None:
        try:
            instrum = retrieval_obj.instrum_param_list[0]
        except (AttributeError, IndexError):
            raise ValueError("No instrument in this run: pass `instrum` (needs at least 'resol').")

    idx = draw_sample_indices(n_draw, len(flat_samples), seed)

    kernels_list = []
    for params in flat_samples[idx]:
        theta_regions = retrieval_obj.unpack_theta(params.copy())
        region_kernels = retrieval_obj.get_ker(theta_regions, phase=phase, planet=retrieval_obj.planet,
                                               instrum=instrum,
                                               model_resolution=retrieval_obj.prt_res['high'])
        if region_kernels[0] is None:
            raise ValueError("This run has no rotation kernel to draw: `get_ker_file` is not set "
                             "in its yaml file.")
        kernels_list.append([np.asarray(ker) for ker in region_kernels])

    # The kernels are sampled at constant velocity steps (the model resolution) and centred on
    # their middle pixel, see `spectrum.BaseKerMulti.resample`.
    dv_kms = (const.c / retrieval_obj.prt_res['high']).to('km/s').value
    n_max = max(len(ker) for region_kernels in kernels_list for ker in region_kernels)
    velocity = (np.arange(n_max) - (n_max - 1) / 2) * dv_kms

    kernels = np.zeros((len(kernels_list), len(kernels_list[0]), n_max))
    for i_draw, region_kernels in enumerate(kernels_list):
        for i_reg, ker in enumerate(region_kernels):
            v_ker = (np.arange(len(ker)) - (len(ker) - 1) / 2) * dv_kms
            kernels[i_draw, i_reg] = np.interp(velocity, v_ker, ker, left=0., right=0.)

    return velocity, kernels


# =====================================================================================
# Abundance ratios
# =====================================================================================
# Elemental abundance ratios (C/O, [X/H]...) computed from a sample of abundance profiles
# (`draw_region_profiles`'s output for one region). Only meaningful for a single region: for a
# multi-region model, `draw_region_profiles` already gives one profile sample per region, so the
# functions below can be called once per region if needed, but there is no dedicated multi-region
# helper here yet.

def abundance_ratio_profiles(profile_samples: Dict[str, np.ndarray], ratio_string: str,
                             abund0: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Elemental abundance ratio (e.g. 'C/O') as a function of pressure, for a sample of profiles.

    Each species contributes to the numerator/denominator atom counts through
    `retrieval_utils.get_atom_abundance` (e.g. H2O contributes 2 atoms of H and 1 of O).

    Parameters
    ----------
    profile_samples : dict
        One region's profiles from `draw_region_profiles`: 'temperature' (ignored here) and the
        abundance (volume mixing ratio) of every species, arrays of shape (n_draw, n_pressures).
    ratio_string : str
        `'<numerator>/<denominator>'`, e.g. `'C/O'`, `'C/H'`.
    abund0 : dict, optional
        `{element: abundance}` added to an element's count for atoms not carried by any species
        in `profile_samples` (e.g. `{'H': 0.84}` if H2 itself was not part of the free-chemistry
        fit). Default: 0 for every element (same as `get_atom_abundance`'s default).

    Returns
    -------
    np.ndarray
        Ratio, shape (n_draw, n_pressures) (linear, not dex).
    """
    if abund0 is None:
        abund0 = {}
    atom_num, atom_denom = ratio_string.split('/')
    species = [key for key in profile_samples if key != 'temperature']
    abundances = [profile_samples[key] for key in species]

    num = ru.get_atom_abundance(atom_num, abundances, species, abund0=abund0.get(atom_num, 0.))
    denom = ru.get_atom_abundance(atom_denom, abundances, species, abund0=abund0.get(atom_denom, 0.))

    return num / denom


def metallicity_profiles(profile_samples: Dict[str, np.ndarray], ratio_string: str,
                         reference_values: Optional[Dict[str, float]] = None,
                         abund0: Optional[Dict[str, float]] = None) -> np.ndarray:
    """`[X/Y]` relative to a reference composition (solar by default), as a function of pressure.

    `[X/Y] = log10(VMR_X / VMR_Y) - log10(N_X / N_Y)_reference`, in dex.

    Parameters
    ----------
    profile_samples : dict
        One region's profiles from `draw_region_profiles`.
    ratio_string : str
        `'<numerator>/<denominator>'`, e.g. `'Fe/H'`, `'C/H'`, `'O/H'`.
    reference_values : dict, optional
        `{element: 12 + log10(N_element / N_H)}`, the standard solar-abundance convention.
        Default: `retrieval_utils.solar_asplund2009`. Use `retrieval_utils.protosolar_asplund2009`
        for the protosolar values instead.
    abund0 : dict, optional
        Forwarded to `abundance_ratio_profiles`.

    Returns
    -------
    np.ndarray
        `[X/Y]`, shape (n_draw, n_pressures), in dex.
    """
    if reference_values is None:
        reference_values = ru.solar_asplund2009

    atom_num, atom_denom = ratio_string.split('/')
    ratio = abundance_ratio_profiles(profile_samples, ratio_string, abund0=abund0)
    reference_ratio_dex = reference_values[atom_num] - reference_values[atom_denom]

    return np.log10(ratio) - reference_ratio_dex


def profile_stats_at_pressures(profile: np.ndarray, pressures: np.ndarray,
                               pressure_list: Sequence[float], log_scale: bool = False
                               ) -> Dict[float, Dict[str, Any]]:
    """Statistics of a profile sample at a few chosen pressures.

    A ratio can vary a lot with pressure (a free-chemistry retrieval does not enforce a constant
    composition with altitude), so its value depends on which pressure is chosen for the
    comparison -- this computes the statistics at each of several pressures side by side, to make
    that dependence visible instead of picking one pressure silently.

    Parameters
    ----------
    profile : np.ndarray
        Sample of profiles, shape (n_draw, n_pressures), e.g. from `abundance_ratio_profiles`,
        `metallicity_profiles` or `draw_region_profiles`'s output.
    pressures : np.ndarray
        Pressure grid (bar) of `profile`, shape (n_pressures,).
    pressure_list : sequence of float
        Pressures (bar) to compute the statistics at. Each one is looked up with
        `np.searchsorted` (same convention as elsewhere in the code, e.g. the retrieval's own
        `ipress = np.searchsorted(pressures, ...)` -- an insertion index, clipped to the array
        bounds, not a true nearest-neighbor search). Pass a value already on `pressures` (or very
        close to it) to avoid the ambiguity this can cause on a coarse or log-spaced grid.
    log_scale : bool, default False
        Compute the statistics in log space (e.g. for a ratio spanning orders of magnitude, but
        not for one already in dex like `metallicity_profiles`'s output).

    Returns
    -------
    dict
        `{pressure: stats}` (the exact values of `pressure_list`, not the grid ones actually used),
        one entry per element of `pressure_list`. Each `stats` dict has scalar values: 'median',
        '1-sig', '2-sig', '3-sig' (each a `(low, high)` pair, see `retrieval_utils.get_stats_from_profile`).
    """
    out = {}
    for pressure in pressure_list:
        idx = int(np.clip(np.searchsorted(pressures, pressure), 0, len(pressures) - 1))
        stats = ru.get_stats_from_profile(profile[:, idx:idx + 1], log_scale=log_scale)
        out[pressure] = {key: (val[0] if key == 'median' else tuple(np.asarray(val)[:, 0]))
                         for key, val in stats.items()}

    return out


def fit_chemical_equilibrium_at_pressure(profile_samples: Dict[str, np.ndarray], pressures: np.ndarray,
                                         pressure: float, mol_list: Sequence[str] = ('H2O', 'CO'),
                                         metallicity: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Chemical-equilibrium C/O (and metallicity) that reproduces the free-chemistry abundances
    of `mol_list` at one pressure, fitted independently for each draw. Needs petitRADTRANS.

    Two least-squares fits are done per draw (log-abundance residuals, `mol_list` values only):
    one with both C/O and metallicity free, one with metallicity fixed (`metallicity`, or --
    default -- the value implied by the free-chemistry Fe/H elemental ratio of that same draw,
    same proxy/convention as `equilibrium_profiles`; solar if neither is available). See that
    function's docstring for why "metallicity" and "Fe/H" are not the same thing, even though one
    is used here to estimate the other by default.

    Parameters
    ----------
    profile_samples : dict
        One region's profiles from `draw_region_profiles` ('temperature' and the abundances).
        `mol_list` (and 'Fe', unless `metallicity` is given) must be in it.
    pressures : np.ndarray
        Pressure grid (bar) of the profiles.
    pressure : float
        Pressure (bar) of the fit. The closest grid point is used.
    mol_list : sequence of str, default ('H2O', 'CO')
        Species used to constrain the fit.
    metallicity : float, optional
        Fixes the metallicity of the "fixed metallicity" fit (dex).

    Returns
    -------
    dict
        'C_to_O', 'metallicity' (both from the free fit) and 'C_to_O_fixed_metallicity' (from the
        fixed-metallicity fit). Each is a 1d array with one value per draw whose fit converged --
        shorter than `n_draw` if some draws failed (a fit failing at an extreme, ill-constrained
        pressure is expected sometimes, not itself a bug).
    """
    from petitRADTRANS.poor_mans_nonequ_chem import interpol_abundances
    from scipy.optimize import least_squares

    idx = int(np.clip(np.searchsorted(pressures, pressure), 0, len(pressures) - 1))
    pressure_value = pressures[idx]
    temperatures = profile_samples['temperature']

    if metallicity is None and 'Fe' in profile_samples:
        species = [key for key in profile_samples if key != 'temperature']
        abundances = [profile_samples[key] for key in species]
        Fe_profiles = ru.get_atom_abundance('Fe', abundances, species)
        H_profiles = ru.get_atom_abundance('H', abundances, species)
        metallicity_estimate = (np.log10(Fe_profiles / H_profiles) - (-4.46))[:, idx]
    else:
        metallicity_estimate = np.full(temperatures.shape[0], 0. if metallicity is None else metallicity)

    def model_vmrs(c_to_o, fe_to_h, temperature):
        mass_fractions = interpol_abundances(c_to_o, fe_to_h, temperature, pressure_value)
        vmrs = mass_frac_2_vmr(mass_frac_dict=mass_fractions)
        # Real petitRADTRANS always returns each species as a length-1 array, even for scalar
        # inputs (confirmed empirically on Narval, 2026-09-22) -- squeeze to a genuine scalar so
        # the residuals below broadcast against `vmr_observed` (shape (n_mol,)) as intended,
        # instead of silently upgrading to (n_mol, n_mol) (the bug the fake-petitRADTRANS unit
        # tests could not catch, since the fake returned plain scalars).
        return np.array([np.asarray(vmrs[mol]).reshape(-1)[0] for mol in mol_list])

    def residuals_both(x, temperature, vmr_observed):
        return np.log(vmr_observed) - np.log(model_vmrs(x[0], x[1], temperature))

    def residuals_fixed(x, temperature, vmr_observed, fe_to_h):
        return np.log(vmr_observed) - np.log(model_vmrs(x[0], fe_to_h, temperature))

    c_to_o_fit, metallicity_fit, c_to_o_fixed = [], [], []
    for i_draw, temperature in enumerate(temperatures):
        vmr_observed = np.array([profile_samples[mol][i_draw, idx] for mol in mol_list])

        result = least_squares(residuals_both, [0.54, 0.], bounds=((0.1, -2), (1.6, 3)),
                               args=(temperature[idx], vmr_observed))
        if result.success:
            c_to_o_fit.append(result.x[0])
            metallicity_fit.append(result.x[1])

        result = least_squares(residuals_fixed, [0.54], bounds=(0.1, 1.6),
                               args=(temperature[idx], vmr_observed, metallicity_estimate[i_draw]))
        if result.success:
            c_to_o_fixed.append(result.x[0])

    return {'C_to_O': np.array(c_to_o_fit), 'metallicity': np.array(metallicity_fit),
           'C_to_O_fixed_metallicity': np.array(c_to_o_fixed)}
