"""Find the best-fit (maximum a posteriori) of a retrieval, starting from a burn-in.

Console script: ``run_starships_optimize``.

The MCMC (``retrieval.py``) explores the whole posterior, but the samples it
returns are typically several log-likelihood units *below* the true maximum
(in ``d`` dimensions, a typical posterior sample sits about ``d/2`` below it).
This tool polishes a burn-in into a proper best-fit: the maximum of ``lnprob``
(log-prior + log-likelihood, exactly the function the MCMC samples), which is
what you want for a best-fit spectrum, a chi2_min, a BIC, or a likelihood ratio.
It does **not** improve the sampling itself, and it does not replace the
sampling phase of the retrieval.

How it works
------------
1. Read the burn-in file (emcee HDF5 backend) and rank all its samples by
   ``lnprob``.
2. Start from one of the best samples. With several starts (a SLURM array,
   see ``submit_starships_optimize``) each task takes a different one, far
   enough from the others (see ``min_separation``).
3. Run CMA-ES (``pip install cma``), a derivative-free optimizer that works
   well in ~30 dimensions and tolerates a slightly noisy likelihood. It works
   in *whitened* coordinates: the covariance of the best burn-in samples
   is used as the initial search shape, so it does not waste evaluations
   learning the degeneracies between parameters (e.g. the TP profile) that the
   burn-in already knows about.
4. Save the best point, its ``lnprob``, and the convergence history to an
   ``.npz`` file. Unless resuming (``--resume``), this never overwrites an existing
   ``optimum_<burnin>_start<N>.npz``/``_state.pkl`` pair -- a numeric suffix is
   added instead (``optimum_<burnin>_start<N>_0.npz``, ...), the same convention
   ``retrieval.py`` already uses for ``walker_file_out``. A quick unrelated run
   (e.g. a resource test) that happens to reuse the same YAML/burn-in/start-index
   can therefore not destroy a real run's finished result.

Several starts converging to the same ``lnprob`` (within ~0.5) is the
reliability check: if they do not, the likelihood is rugged or multimodal and
a single optimum means little. Use ``--compare`` to print that comparison.

Parameters that are poorly constrained (clouds, ...) can make the maximum
ill-defined, and the MAP of a 27-dimensional posterior is *not* a good summary
of the posterior (volume effects). Report it as "best-fit", not as the
central value of a parameter.

Each ``lnprob`` call costs the same as one MCMC step (tens of seconds for
large retrievals), so the whole optimization typically costs a few thousand
evaluations, about 1% of a retrieval. It cannot be run locally: it calls
petitRADTRANS through ``retrieval.lnprob``.

Usage::

    # One optimization, from the best burn-in sample
    run_starships_optimize params.yaml --walker-file walker_burnin.h5

    # The 3rd start of a multi-start (this is what each SLURM array task does)
    run_starships_optimize params.yaml --walker-file walker_burnin.h5 --start-index 3

    # Compare the results of several starts
    run_starships_optimize --compare optimum_*_start*.npz

    # Peek at a run that has not finished yet (or was killed): no petitRADTRANS
    # needed, reads the state file the run above keeps updating every generation.
    run_starships_optimize --inspect optimum_walker_burnin_start1_state.pkl
"""
import argparse
import logging
import os
import pickle
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import emcee
import numpy as np

log = logging.getLogger(__name__)


##########################################
# Burn-in reading and start selection
##########################################

def load_burnin_samples(walker_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read all the samples of a burn-in file, ranked from best to worst.

    Parameters
    ----------
    walker_file : Path
        emcee HDF5 backend (the file written by ``run_starships_retrieval``, or a
        combination of several such files).

    Returns
    -------
    samples : np.ndarray, shape (n_samples, n_dim)
        All the finite samples (every step of every walker), best ``lnprob`` first.
    lnprob : np.ndarray, shape (n_samples,)
        The ``lnprob`` (log-prior + log-likelihood) of each sample.
    """
    # Only the steps that were actually completed are returned by emcee, so a
    # burn-in that was cut short by the walltime is fine.
    reader = emcee.backends.HDFBackend(str(walker_file), read_only=True)
    samples = reader.get_chain(flat=True)
    lnprob = reader.get_log_prob(flat=True)

    # A -inf/NaN sample has no meaning as a starting point
    finite = np.isfinite(lnprob)
    samples, lnprob = samples[finite], lnprob[finite]

    order = np.argsort(lnprob)[::-1]  # best first
    return samples[order], lnprob[order]


def whitening_from_samples(samples: np.ndarray, rel_floor: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """Build the linear map between the parameters and *whitened* coordinates.

    Whitened coordinates ``z`` are such that the samples have unit variance and
    no correlations in every direction: ``x = x0 + L @ z`` and ``z = L_inv @ (x - x0)``,
    with ``L @ L.T`` equal to the covariance of ``samples``. Searching in this
    space makes the problem look "round" to the optimizer, whatever the units or
    the degeneracies of the parameters.

    Parameters
    ----------
    samples : np.ndarray, shape (n_samples, n_dim)
        Samples describing the region of interest (e.g. the best burn-in samples).
    rel_floor : float
        Eigenvalues of the covariance smaller than ``rel_floor`` times the largest
        one are raised to that level. Keeps the map invertible when a direction
        has (almost) no variance in the samples.

    Returns
    -------
    L : np.ndarray, shape (n_dim, n_dim)
        Whitened -> parameter coordinates (without the offset ``x0``).
    L_inv : np.ndarray, shape (n_dim, n_dim)
        Parameter -> whitened coordinates (without the offset ``x0``).
    """
    cov = np.atleast_2d(np.cov(samples, rowvar=False))

    # Eigen-decomposition (rather than a Cholesky) because it keeps working when the
    # covariance is singular, which happens when a parameter is (nearly) constant.
    eigval, eigvec = np.linalg.eigh(cov)
    eigval = np.clip(eigval, rel_floor * eigval.max(), None)

    L = eigvec * np.sqrt(eigval)
    L_inv = (eigvec / np.sqrt(eigval)).T
    return L, L_inv


def select_starts(samples_best_first: np.ndarray, L_inv: np.ndarray, n_starts: int,
                  min_separation: float = 3.0) -> np.ndarray:
    """Pick the indices of the starting points of a multi-start optimization.

    Goes down the list of samples (best first) and keeps a sample only if it is
    at least ``min_separation`` from all the ones already kept. Without this,
    the top of the list would be filled with the successive steps of the same
    walker, i.e. the same point ``n_starts`` times.

    Parameters
    ----------
    samples_best_first : np.ndarray, shape (n_samples, n_dim)
        Burn-in samples, best ``lnprob`` first (see ``load_burnin_samples``).
    L_inv : np.ndarray, shape (n_dim, n_dim)
        Parameter -> whitened map (see ``whitening_from_samples``); distances are
        measured in whitened units (1 = one posterior standard deviation).
        Two independent samples of the best region are typically ``sqrt(2 * n_dim)``
        apart, i.e. about 7 for 27 parameters.
    n_starts : int
        Number of starting points wanted.
    min_separation : float
        Minimum whitened distance between two starting points.

    Returns
    -------
    np.ndarray of int
        Indices in ``samples_best_first``, in order (the first one is always the best
        sample). It can be shorter than ``n_starts`` if the samples do not contain
        enough well-separated points.
    """
    whitened = samples_best_first @ L_inv.T
    kept: List[int] = []
    for i in range(len(whitened)):
        if kept:
            distances = np.linalg.norm(whitened[kept] - whitened[i], axis=1)
            if distances.min() < min_separation:
                continue
        kept.append(i)
        if len(kept) == n_starts:
            break
    return np.array(kept, dtype=int)


##########################################
# CMA-ES
##########################################

def _ask_feasible(es, to_x: Callable[[np.ndarray], np.ndarray], is_feasible: Callable[[np.ndarray], bool],
                  max_rounds: int = 1000) -> List[np.ndarray]:
    """Ask CMA-ES for a full generation of candidates that are all inside the prior.

    Testing whether a candidate is inside the prior is free (no forward model), so
    the candidates outside are simply drawn again instead of wasting an expensive
    ``lnprob`` call on them.
    """
    accepted: List[np.ndarray] = []
    for _ in range(max_rounds):
        for z in es.ask(es.popsize - len(accepted)):
            if is_feasible(to_x(z)):
                accepted.append(z)
        if len(accepted) == es.popsize:
            return accepted
    raise RuntimeError(f"Could not draw {es.popsize} candidates inside the prior after {max_rounds} rounds. "
                       "The search distribution is almost entirely outside the prior: is the burn-in "
                       "consistent with the YAML (same priors)?")


def default_popsize(n_dim: int, n_cpu: int = 1) -> int:
    """CMA-ES default population size (``4 + 3 ln(n_dim)``), rounded up to a multiple of ``n_cpu``.

    All the candidates of a generation are evaluated in parallel, so a
    generation of 13 on 24 CPUs would leave 11 of them idle.
    """
    popsize = 4 + int(3 * np.log(n_dim))
    return int(np.ceil(popsize / n_cpu) * n_cpu)


def optimize_cmaes(neg_lnprob_batch: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, L: np.ndarray,
                   is_feasible: Callable[[np.ndarray], bool], sigma0: float = 0.3,
                   popsize: Optional[int] = None, max_evals: int = 5000,
                   max_seconds: Optional[float] = None, tolfun: float = 0.1, seed: Optional[int] = None,
                   state_file: Optional[Path] = None, resume: bool = False) -> Dict:
    """Minimize ``-lnprob`` with CMA-ES, in whitened coordinates around ``x0``.

    Parameters
    ----------
    neg_lnprob_batch : callable
        Takes an array (n_candidates, n_dim) of parameters and returns the
        ``-lnprob`` of each one, shape (n_candidates,). This is where the
        candidates of a generation are evaluated (in parallel, if wanted).
        Non-finite values are allowed (e.g. a failed model): they are ranked last.
    x0 : np.ndarray, shape (n_dim,)
        Starting point, in parameter space.
    L : np.ndarray, shape (n_dim, n_dim)
        Whitened -> parameter map (see ``whitening_from_samples``).
    is_feasible : callable
        Takes one parameter vector and returns False if it is outside the prior.
    sigma0 : float
        Initial step size, in whitened units (1 = one std of the best burn-in
        samples). CMA-ES adapts it, so it only needs to be the right order of
        magnitude; smaller is safer when ``x0`` is already a good point.
    popsize : int, optional
        Candidates per generation (default: see ``default_popsize``).
    max_evals : int
        Stop after this number of ``lnprob`` evaluations (in total, including
        those of a run that is resumed). Checked between generations, so the
        last generation can go over it by up to ``popsize``.
    max_seconds : float, optional
        Stop after this much time (of this session), at the end of the generation
        in progress. Use it to finish cleanly before the SLURM walltime.
    tolfun : float
        Stop when the best ``lnprob`` has changed by less than this over the last
        ~``10 + 30 * n_dim / popsize`` generations. 0.1 is far below any
        scientifically meaningful difference in log-likelihood.
    seed : int, optional
        Random seed (for reproducibility).
    state_file : Path, optional
        If given, the full state is saved there after every generation (atomically).
    resume : bool
        Continue from ``state_file`` instead of starting again from ``x0``.

    Returns
    -------
    dict
        ``x_best`` (n_dim,), ``lnprob_best``, ``n_evals``, ``stop_reason`` (str),
        ``history`` (n_generations, 4): columns are the number of evaluations, the
        best ``lnprob`` so far, the median ``lnprob`` of the generation and the
        CMA-ES step size. ``theta_trace`` (n_evals, n_dim) and ``lnprob_trace``
        (n_evals,): every candidate ever evaluated (in evaluation order), not just
        the best one. Each is a real ``lnprob`` call, so nothing evaluated is
        thrown away: near the end, the trace approximates a local (Laplace) picture
        of the posterior around the optimum -- CMA-ES's own search covariance
        tracks the local curvature -- which is informative even if the run is
        killed before ``tolfun`` converges. It is not the true posterior shape
        (CMA-ES concentrates on the neighbourhood of the best point it has found,
        and a Laplace approximation misses skew and other modes), but it comes at
        zero extra cost.
    """
    try:
        import cma
    except ImportError as err:
        raise ImportError("run_starships_optimize needs the `cma` package: `pip install cma`.") from err

    n_dim = len(x0)

    def to_x(z: np.ndarray) -> np.ndarray:
        # Whitened coordinates -> parameters
        return x0 + L @ np.asarray(z)

    # --- Initialise (or restore) the optimizer state ---
    if resume:
        if state_file is None or not Path(state_file).is_file():
            raise FileNotFoundError(f"Cannot resume: state file {state_file} does not exist.")
        with open(state_file, 'rb') as f:
            state = pickle.load(f)
        es = pickle.loads(state['es'])
        best_x, best_f = state['best_x'], state['best_f']
        history = state['history']
        # 'theta_trace'/'lnprob_trace' were added after this run may have started (a state
        # file from before then has neither key) -- fall back to an empty trace rather than
        # KeyError, since a state file's *format* long outlives the code version that wrote it.
        theta_trace = list(state.get('theta_trace', []))
        lnprob_trace = list(state.get('lnprob_trace', []))
        elapsed_before = state['elapsed']
        log.info(f"Resuming from {state_file}: {es.countevals} evaluations already done, "
                 f"best lnprob = {-best_f:.3f}")
        if not theta_trace and es.countevals > 0:
            log.warning("This state file predates the evaluation trace: the history of "
                       f"{es.countevals} evaluations already done is not recoverable, only new "
                       "evaluations from here on will be traced.")
    else:
        if popsize is None:
            popsize = default_popsize(n_dim)
        opts = {
            'popsize': popsize,
            'tolfun': tolfun,
            # Stop when the steps are 0.1% of the posterior width: far below what matters.
            'tolx': 1e-3,
            # Silence CMA-ES's own output and files, we log what we need ourselves.
            'verbose': -9,
            'verb_log': 0,
        }
        if seed is not None:
            opts['seed'] = seed
        es = cma.CMAEvolutionStrategy(np.zeros(n_dim), sigma0, opts)
        best_x, best_f = np.array(x0, dtype=float), np.inf
        history = []
        theta_trace, lnprob_trace = [], []
        elapsed_before = 0.

    # maxfevals is set here (not in opts) so that it can be changed when resuming
    es.opts.set({'maxfevals': max_evals})

    t_start = time.time()
    stop_reason = ''
    while True:
        stop = es.stop()
        if stop:
            stop_reason = ', '.join(f'{key}={val}' for key, val in stop.items())
            break
        if max_seconds is not None and time.time() - t_start > max_seconds:
            stop_reason = 'time limit'
            break

        # --- One generation ---
        z_batch = _ask_feasible(es, to_x, is_feasible)
        x_batch = np.array([to_x(z) for z in z_batch])
        f_batch = np.asarray(neg_lnprob_batch(x_batch), dtype=float)

        # A failed candidate (non-finite lnprob, e.g. negative temperatures in the TP
        # profile) just needs to rank last. CMA-ES only uses the ranking, so the exact
        # value does not matter.
        finite = np.isfinite(f_batch)
        penalty = f_batch[finite].max() + 1. if finite.any() else 1e30
        f_batch = np.where(finite, f_batch, penalty)

        es.tell(z_batch, list(f_batch))

        # Every candidate this generation actually cost a real lnprob call, not
        # just the best one -- keep all of them, they are not free to redo.
        theta_trace.extend(x_batch)
        lnprob_trace.extend(-f_batch)

        i_best = int(np.argmin(f_batch))
        if f_batch[i_best] < best_f:
            best_f, best_x = float(f_batch[i_best]), x_batch[i_best].copy()

        history.append([es.countevals, -best_f, -float(np.median(f_batch)), es.sigma])
        log.info(f"generation {len(history):4d} | evaluations = {es.countevals:5d} | "
                 f"best lnprob = {-best_f:.3f} | median of generation = {-np.median(f_batch):.3f} | "
                 f"sigma = {es.sigma:.3f} | elapsed = {(elapsed_before + time.time() - t_start) / 3600:.2f} h")

        if state_file is not None:
            _save_state(state_file, es, best_x, best_f, history, theta_trace, lnprob_trace,
                       elapsed_before + time.time() - t_start)

    return {
        'x_best': best_x,
        'lnprob_best': -best_f,
        'n_evals': int(es.countevals),
        'stop_reason': stop_reason,
        'history': np.array(history).reshape(-1, 4),
        'theta_trace': np.array(theta_trace).reshape(-1, n_dim),
        'lnprob_trace': np.array(lnprob_trace),
    }


def _save_state(state_file: Path, es, best_x: np.ndarray, best_f: float, history: list, theta_trace: list,
                lnprob_trace: list, elapsed: float) -> None:
    """Write the optimizer state so a run cut by the walltime can be resumed.

    Includes every candidate evaluated so far (``theta_trace``/``lnprob_trace``),
    so that a run killed mid-way still leaves the full evaluation trace on disk,
    not just the best point -- see ``optimize_cmaes``'s docstring for why that
    trace is worth keeping. At a few thousand evaluations x tens of parameters
    this is a few MB at most, cheap to rewrite every generation.
    """
    state = {'es': es.pickle_dumps(), 'best_x': best_x, 'best_f': best_f, 'history': history,
             'theta_trace': theta_trace, 'lnprob_trace': lnprob_trace, 'elapsed': elapsed}
    # Write to a temporary file first, so that a job killed in the middle of the
    # write cannot destroy the previous (good) state.
    tmp_file = Path(str(state_file) + '.tmp')
    with open(tmp_file, 'wb') as f:
        pickle.dump(state, f)
    os.replace(tmp_file, state_file)


def load_state(state_file: Path) -> Dict:
    """Read a CMA-ES state file, including while the run that owns it is still going.

    Safe to call on a running or killed job: ``_save_state`` writes a new file and
    renames it over the old one, so a reader never sees a half-written file. Needs
    only ``cma``/``numpy``/``pickle`` -- no petitRADTRANS, no retrieval YAML -- so
    it works from a login node or a laptop, on a copy of the state file if wanted.

    Parameters
    ----------
    state_file : Path
        The ``*_state.pkl`` file next to the result written by ``optimize_from_burnin``.
        A file from before the evaluation trace existed is read fine too, just with
        an empty ``theta_trace``/``lnprob_trace`` (nothing to recover retroactively).

    Returns
    -------
    dict
        Same shape as ``optimize_cmaes``'s return value (``x_best``, ``lnprob_best``,
        ``n_evals``, ``history``, ``theta_trace``, ``lnprob_trace``), plus ``elapsed_s``.
    """
    with open(state_file, 'rb') as f:
        state = pickle.load(f)
    es = pickle.loads(state['es'])
    n_dim = len(state['best_x'])
    return {
        'x_best': state['best_x'],
        'lnprob_best': -state['best_f'],
        'n_evals': es.countevals,
        'history': np.array(state['history']).reshape(-1, 4),
        'theta_trace': np.array(state.get('theta_trace', [])).reshape(-1, n_dim),
        'lnprob_trace': np.array(state.get('lnprob_trace', [])),
        'elapsed_s': state['elapsed'],
    }


def _free_output_paths(output_dir: Path, base_stem: str, resume: bool) -> Tuple[Path, Path]:
    """Pick the ``(result_file, state_file)`` pair for a run, without ever overwriting
    another run's files.

    When not resuming, a fresh optimization must never silently overwrite an existing
    ``.npz``/``_state.pkl`` pair -- that pair might belong to a completed run holding
    thousands of expensive ``lnprob`` evaluations (this happened for real: a quick,
    unrelated test run that reused the default ``start_index=1`` destroyed a start's
    finished result). So instead, a numeric suffix is appended, exactly like
    ``retrieval.prepare_run`` already does for ``walker_file_out`` -- same convention,
    same reason. Resuming is the one case allowed to reuse the exact existing pair, since
    that is the whole point of ``--resume``.

    Parameters
    ----------
    output_dir : Path
        Directory the files go in.
    base_stem : str
        The name the files would have with no collision, e.g. ``optimum_<burnin>_start1``
        (the ``.npz``/``_state.pkl`` extensions are added here).
    resume : bool
        If True, always return the un-suffixed pair (``optimize_cmaes`` itself raises
        ``FileNotFoundError`` if the state file it names does not exist).

    Returns
    -------
    result_file, state_file : Path, Path
        Guaranteed to not already exist as a pair, unless ``resume`` is True.
    """
    result_file = output_dir / f'{base_stem}.npz'
    state_file = output_dir / f'{base_stem}_state.pkl'
    if resume:
        return result_file, state_file

    for suffix_idx in range(100):
        if not result_file.is_file() and not state_file.is_file():
            return result_file, state_file
        log.info(f"{result_file.name} (or its state file) already exists and this is not a "
                 f"--resume: trying {base_stem}_{suffix_idx} instead.")
        result_file = output_dir / f'{base_stem}_{suffix_idx}.npz'
        state_file = output_dir / f'{base_stem}_{suffix_idx}_state.pkl'
    raise FileExistsError(f"Could not find a free output file for '{base_stem}' in {output_dir} "
                          "after 100 tries.")


##########################################
# Driver: connects everything to the retrieval
##########################################

def _neg_lnprob(theta: np.ndarray) -> float:
    """``-lnprob`` of one parameter vector (module-level so that a Pool can call it)."""
    import starships.retrieval as ret
    return -float(ret.lnprob(theta))


def optimize_from_burnin(config: Path, walker_file: Path, start_index: int = 1, n_cpu: Optional[int] = None,
                         output_dir: Optional[Path] = None, top_fraction: float = 0.1,
                         min_separation: float = 3.0, sigma0: float = 0.3, popsize: Optional[int] = None,
                         max_evals: int = 5000, max_hours: Optional[float] = None, tolfun: float = 0.1,
                         seed: Optional[int] = None, resume: bool = False) -> Path:
    """Find the best-fit of a retrieval starting from its burn-in (see the module docstring).

    Parameters
    ----------
    config : Path
        The retrieval YAML (the one that goes with the burn-in file).
    walker_file : Path
        The burn-in (emcee HDF5 backend). A bare file name is looked for in the
        ``walker_path`` of the YAML.
    start_index : int
        Which starting point to use (1 = best burn-in sample, 2 = the best one that
        is well separated from the first, ...). Different values give independent
        optimizations, to be run in parallel as a SLURM array.
    n_cpu : int, optional
        Number of processes evaluating ``lnprob`` in parallel (default: the
        ``SLURM_CPUS_PER_TASK`` of the job, else 1 = no multiprocessing).
    output_dir : Path, optional
        Where to write the results (default: the ``walker_path`` of the YAML).
    top_fraction : float
        Fraction of the burn-in samples (the best ones) used to estimate the
        covariance that shapes the search. At least ``10 * n_dim`` samples are used.
    min_separation : float
        Minimum whitened distance between two starting points (see ``select_starts``).
    sigma0, popsize, max_evals, tolfun, seed, resume
        See ``optimize_cmaes``.
    max_hours : float, optional
        Stop cleanly after this many hours (set it a bit below the SLURM walltime).

    Returns
    -------
    Path
        The ``.npz`` file with the results (``theta_best``, ``lnprob_best``, ``logl_best``,
        ``param_names``, ``theta_start``, ``history``, ...).
    """
    # Imported here so that the rest of this module (and its tests) does not need
    # petitRADTRANS, which retrieval.py pulls in.
    import starships.retrieval as ret
    from starships import retrieval_utils as ru

    config = Path(config).expanduser()

    ###################
    # Setup the retrieval: defines the priors, the data, and lnprob (all as globals of `ret`)
    ###################
    ret.setup_retrieval(config)
    if ret.retrieval_type in ('JR', 'HRR'):
        ret.load_high_res_data()

    param_names = list(ret.params_prior.keys())
    n_dim = len(param_names)

    if n_cpu is None:
        n_cpu = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))

    walker_file = Path(walker_file).expanduser()
    if not walker_file.is_file():
        walker_file = Path(ret.walker_path) / walker_file
    if not walker_file.is_file():
        raise FileNotFoundError(f"Burn-in file not found: {walker_file}")
    output_dir = Path(ret.walker_path) if output_dir is None else Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    ###################
    # Burn-in: covariance and starting point
    ###################
    samples, lnprob_samples = load_burnin_samples(walker_file)
    if samples.shape[1] != n_dim:
        raise ValueError(f"The burn-in has {samples.shape[1]} parameters but the YAML defines {n_dim} "
                         f"({param_names}). Is it the YAML that goes with this burn-in?")
    log.info(f"Burn-in: {len(samples)} samples, best lnprob = {lnprob_samples[0]:.3f}, "
             f"median = {np.median(lnprob_samples):.3f}")

    # Shape of the search: covariance of the best samples only. The whole burn-in
    # can still be spread over the prior if it has not converged, and would give a
    # search region far too wide.
    n_top = min(len(samples), max(int(top_fraction * len(samples)), 10 * n_dim))
    L, L_inv = whitening_from_samples(samples[:n_top])

    start_indices = select_starts(samples, L_inv, start_index, min_separation)
    if len(start_indices) < start_index:
        raise ValueError(f"Only {len(start_indices)} starting points separated by at least {min_separation} "
                         f"(whitened units) exist in the burn-in, but start_index = {start_index}. "
                         "Use a smaller start_index or min_separation.")
    x0 = samples[start_indices[start_index - 1]]
    lnprob_start = float(lnprob_samples[start_indices[start_index - 1]])
    log.info(f"Start #{start_index}: lnprob = {lnprob_start:.3f}")

    ###################
    # lnprob evaluation (serial or in parallel)
    ###################
    # First call in the main process: checks that lnprob works, gives the cost of one
    # evaluation, and (most importantly) initializes the atmosphere objects *before*
    # the Pool forks, so the workers inherit them instead of each rebuilding them
    # (same trick as in retrieval.main).
    t0 = time.time()
    lnprob_start_check = -_neg_lnprob(x0)
    log.info(f"lnprob at the start point = {lnprob_start_check:.3f} (in the burn-in file: {lnprob_start:.3f}); "
             f"first call took {time.time() - t0:.0f} s (includes the one-time initialization)")
    if not np.isfinite(lnprob_start_check):
        raise ValueError("lnprob is not finite at the starting point. Is the YAML the one used for the burn-in?")

    def is_feasible(theta: np.ndarray) -> bool:
        return bool(np.isfinite(ru.log_prior(theta, ret.params_prior, prior_func_dict=ret.prior_func_dict)))

    if popsize is None:
        popsize = default_popsize(n_dim, n_cpu)
    log.info(f"CMA-ES: {n_dim} parameters, population = {popsize}, {n_cpu} process(es)")

    base_stem = f'optimum_{walker_file.stem}_start{start_index}'
    result_file, state_file = _free_output_paths(output_dir, base_stem, resume)
    log.info(f"Results will be saved to {result_file}")

    max_seconds = None if max_hours is None else max_hours * 3600.
    t_run = time.time()

    def run(neg_lnprob_batch: Callable) -> Dict:
        return optimize_cmaes(neg_lnprob_batch, x0, L, is_feasible, sigma0=sigma0, popsize=popsize,
                              max_evals=max_evals, max_seconds=max_seconds, tolfun=tolfun, seed=seed,
                              state_file=state_file, resume=resume)

    if n_cpu > 1:
        with Pool(n_cpu) as pool:
            # chunksize=1: each call is long, so hand them out one at a time to balance the load
            result = run(lambda x_batch: np.array(pool.map(_neg_lnprob, list(x_batch), chunksize=1)))
    else:
        result = run(lambda x_batch: np.array([_neg_lnprob(x) for x in x_batch]))

    ###################
    # Save and report
    ###################
    theta_best = result['x_best']
    log_prior_best = float(ru.log_prior(theta_best, ret.params_prior, prior_func_dict=ret.prior_func_dict))
    np.savez(result_file,
             theta_best=theta_best, param_names=np.array(param_names),
             lnprob_best=result['lnprob_best'],
             # lnprob = log-prior + log-likelihood: the log-likelihood alone is what is left
             # after removing the prior (they are equal for uniform priors).
             log_prior_best=log_prior_best, logl_best=result['lnprob_best'] - log_prior_best,
             theta_start=x0, lnprob_start=lnprob_start, start_index=start_index,
             n_evals=result['n_evals'], stop_reason=result['stop_reason'], history=result['history'],
             # Every candidate CMA-ES evaluated, not just the best one -- each cost a real
             # lnprob call, so it is saved even if the run is stopped early (see
             # optimize_cmaes's docstring: this traces the local shape of the posterior
             # around the optimum, at no extra cost).
             theta_trace=result['theta_trace'], lnprob_trace=result['lnprob_trace'],
             elapsed_s=time.time() - t_run, walker_file=str(walker_file), config=str(config))

    log.info(f"Stopped: {result['stop_reason']} after {result['n_evals']} evaluations.")
    log.info(f"lnprob: start = {lnprob_start:.3f} -> best = {result['lnprob_best']:.3f} "
             f"(improvement = {result['lnprob_best'] - lnprob_start:.3f})")
    log.info(f"{'parameter':<24s}{'start':>14s}{'best':>14s}")
    for name, x_start, x_best in zip(param_names, x0, theta_best):
        log.info(f"{name:<24s}{x_start:>14.5g}{x_best:>14.5g}")
    log.info(f"Results saved to {result_file}")

    return result_file


def compare_results(result_files: Sequence[Path], tol: float = 0.5) -> Dict:
    """Print how well several optimizations (starts) agree with each other.

    If the starts reach the same ``lnprob`` (within ``tol``) the maximum is reliable.
    If not, the likelihood is rugged or has several modes and no single "best-fit"
    is trustworthy: look at the parameters that differ between the starts.

    Parameters
    ----------
    result_files : sequence of Path
        ``.npz`` files written by ``optimize_from_burnin``.
    tol : float
        Maximum difference of ``lnprob`` between starts to call them consistent.

    Returns
    -------
    dict
        ``lnprob_best`` (one value per file), ``spread`` (max - min) and ``consistent`` (bool).
    """
    results = [np.load(f) for f in result_files]
    lnprob_best = np.array([float(r['lnprob_best']) for r in results])
    names = [str(n) for n in results[0]['param_names']]

    print(f"{'file':<60s}{'start':>6s}{'lnprob start':>15s}{'lnprob best':>15s}{'evals':>7s}  stop")
    for f, r in zip(result_files, results):
        print(f"{Path(f).name:<60s}{int(r['start_index']):>6d}{float(r['lnprob_start']):>15.3f}"
              f"{float(r['lnprob_best']):>15.3f}{int(r['n_evals']):>7d}  {r['stop_reason']}")

    spread = float(lnprob_best.max() - lnprob_best.min())
    consistent = spread <= tol
    print(f"\nlnprob spread between starts: {spread:.3f} -> "
          + ("consistent (the maximum looks reliable)." if consistent else
             f"NOT consistent (> {tol}): the optimizer did not converge to the same maximum."))

    # Parameter values of every start, in the order of decreasing lnprob
    order = np.argsort(lnprob_best)[::-1]
    print(f"\n{'parameter':<24s}" + ''.join(f"{'start ' + str(int(results[i]['start_index'])):>14s}" for i in order))
    for j, name in enumerate(names):
        print(f"{name:<24s}" + ''.join(f"{float(results[i]['theta_best'][j]):>14.5g}" for i in order))

    return {'lnprob_best': lnprob_best, 'spread': spread, 'consistent': consistent}


def inspect_state(state_file: Path, output_file: Optional[Path] = None) -> Path:
    """Snapshot a running (or killed) optimization's evaluation trace to an ``.npz`` file.

    Reads only the state file (see ``load_state``): no petitRADTRANS, no retrieval
    YAML, so this can run on a login node, or on a copy of the state file pulled
    somewhere else, while the SLURM job that owns it is still running. Useful to
    see where an optimization has gotten to without waiting for it to finish, or to
    recover the evaluation trace of a run that was killed before writing its final
    result (``optimize_from_burnin``'s ``.npz`` is only written at the very end).

    Parameters
    ----------
    state_file : Path
        The ``*_state.pkl`` file next to the (possibly not-yet-written) result of an
        ``optimize_from_burnin`` run.
    output_file : Path, optional
        Where to save the snapshot (default: ``state_file`` with ``_state.pkl``
        replaced by ``_snapshot.npz``).

    Returns
    -------
    Path
        The saved snapshot file, with the same ``theta_trace``/``lnprob_trace``/
        ``history``/``x_best``/``lnprob_best`` fields as the final result.
    """
    state_file = Path(state_file)
    state = load_state(state_file)
    if output_file is None:
        stem = state_file.name.removesuffix('_state.pkl')
        output_file = state_file.with_name(f'{stem}_snapshot.npz')

    np.savez(output_file, x_best=state['x_best'], lnprob_best=state['lnprob_best'], n_evals=state['n_evals'],
             history=state['history'], theta_trace=state['theta_trace'], lnprob_trace=state['lnprob_trace'],
             elapsed_s=state['elapsed_s'], state_file=str(state_file))

    print(f"n_evals so far: {state['n_evals']}  (elapsed: {state['elapsed_s'] / 3600:.2f} h)")
    print(f"best lnprob so far: {state['lnprob_best']:.3f}")
    print(f"trace: {state['theta_trace'].shape[0]} evaluated points -> saved to {output_file}")
    return output_file


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point of ``run_starships_optimize`` (see the module docstring)."""
    parser = argparse.ArgumentParser(
        prog='run_starships_optimize', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', type=Path, nargs='?', help='Retrieval YAML that goes with the burn-in.')
    parser.add_argument('--walker-file', type=Path,
                        help='Burn-in file (emcee HDF5). A bare name is looked for in the YAML walker_path.')
    parser.add_argument('--start-index', type=int, default=None,
                        help='Which starting point: 1 = best burn-in sample, 2 = best one well separated '
                             'from the first, ... (default: $SLURM_ARRAY_TASK_ID, else 1).')
    parser.add_argument('--n-cpu', type=int, default=None,
                        help='Processes evaluating lnprob in parallel (default: $SLURM_CPUS_PER_TASK, else 1).')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Where to write the results (default: YAML walker_path).')
    parser.add_argument('--max-evals', type=int, default=5000, help='Maximum number of lnprob evaluations.')
    parser.add_argument('--max-hours', type=float, default=None,
                        help='Stop cleanly after this many hours (set it below the SLURM walltime).')
    parser.add_argument('--popsize', type=int, default=None,
                        help='CMA-ES candidates per generation (default: 4 + 3 ln(n_dim), rounded up to n_cpu).')
    parser.add_argument('--sigma0', type=float, default=0.3,
                        help='Initial step size in whitened units (1 = std of the best burn-in samples).')
    parser.add_argument('--tolfun', type=float, default=0.1,
                        help='Stop when the best lnprob changed less than this over many generations.')
    parser.add_argument('--top-fraction', type=float, default=0.1,
                        help='Fraction of the best burn-in samples used to shape the search.')
    parser.add_argument('--min-separation', type=float, default=3.0,
                        help='Minimum distance (in std of the best samples) between different starts.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--resume', action='store_true',
                        help='Continue a run cut by the walltime, from its saved state.')
    parser.add_argument('--compare', type=Path, nargs='+', metavar='RESULT.npz',
                        help='Do not optimize: compare the results of several starts.')
    parser.add_argument('--inspect', type=Path, metavar='STATE.pkl',
                        help='Do not optimize: snapshot a running/killed run\'s evaluation trace '
                             '(*_state.pkl) to an .npz file, without needing petitRADTRANS.')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    if args.compare:
        compare_results(args.compare)
        return

    if args.inspect:
        inspect_state(args.inspect)
        return

    if args.config is None or args.walker_file is None:
        parser.error('config and --walker-file are required (unless using --compare/--inspect).')

    start_index = args.start_index
    if start_index is None:
        # Same convention as the retrieval: the tasks of a SLURM array are numbered from 1
        start_index = int(os.environ.get('SLURM_ARRAY_TASK_ID', 1))

    optimize_from_burnin(args.config, args.walker_file, start_index=start_index, n_cpu=args.n_cpu,
                         output_dir=args.output_dir, top_fraction=args.top_fraction,
                         min_separation=args.min_separation, sigma0=args.sigma0, popsize=args.popsize,
                         max_evals=args.max_evals, max_hours=args.max_hours, tolfun=args.tolfun,
                         seed=args.seed, resume=args.resume)


if __name__ == '__main__':
    main()
