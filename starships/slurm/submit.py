"""Submit a SLURM resource-estimation sweep. Two workloads (``--workload``):

- ``lnprob`` (default): one job per ``n_cpu:n_iter_per_cpu`` pair given (e.g.
  ``1:2 2:3 4:6 8:12``), each with ``--cpus-per-task`` set to exactly that
  ``n_cpu``. Wraps :mod:`starships.slurm.worker`'s ``run_starships_resource_
  test``, which times independent ``lnprob()`` calls -- raw compute scaling
  vs ``n_cpu``, see that module's docstring.
- ``retrieval``: one job per point, each a REAL short ``run_starships_
  retrieval`` run (``n_steps_burnin`` and either ``n_walkers_per_cpu`` or
  ``n_walkers`` overridden on its command line -- all already supported
  directly by ``retrieval.main()``/``prepare_run()``, no separate test tool
  needed here, nothing of ``retrieval.py`` reimplemented). Tests real MCMC
  ensemble parallelization/synchronization efficiency: within one step,
  ``n_walkers`` spread across ``n_cpu`` workers parallelizes fine, but every
  walker must finish before the ensemble can propose the next step -- more
  walkers per cpu means that fixed per-step overhead is amortized over more
  real compute per worker per step (Antoine, 2026-09-17). ``lnprob`` mode's
  independent calls cannot reveal this at all. Two ways to sweep it,
  mutually exclusive:

  - ``--n-cpu N`` fixed, one job per ``n_walkers_per_cpu`` value given (e.g.
    ``2 4 8 14``) -- the original design, isolates the walkers-per-cpu
    efficiency effect at one fixed ``n_cpu``.
  - ``--n-walkers N`` fixed, one job per ``n_cpu`` value given (e.g.
    ``10 12 14 24``) -- Antoine, 2026-09-17: the practical constraint is
    normally a *target walker count* (MCMC needs "enough" walkers,
    independent of how many cpus happen to be available), so this sweeps the
    actually-relevant question directly ("for MY fixed walker count, what
    n_cpu should I use?") with REAL measurements at each ``n_cpu``, instead
    of extrapolating from a single fixed-``n_cpu`` sweep's efficiency curve
    (which :func:`starships.slurm.estimation.plot_ensemble_efficiency`'s
    ``n_walkers_target`` option can only approximate, assuming efficiency at
    a given ``n_walkers_per_cpu`` transfers across different ``n_cpu`` --
    untested beyond that one sweep's own fixed ``n_cpu``). Passes ``n_walkers``
    directly (not ``n_walkers_per_cpu``) to ``run_starships_retrieval`` --
    ``retrieval.py`` uses ``n_walkers`` as-is when given, no divisibility by
    ``n_cpu`` required (``multiprocessing.Pool`` chunks unevenly on its own if
    needed).

Console script: ``run_starships_resource_sweep``. Run from whatever directory
you want the sweep's metadata (``last_sweep.json``) and job logs saved in
(typically a per-project scratch directory) -- nothing needs to be copied or
installed there first, this is a plain installed command.

Jobs within a sweep are grouped by a shared ``sweep_id`` (used in job names
and log filenames) and their job IDs are recorded in ``last_sweep.json`` so
:mod:`starships.slurm.parse` can find them all afterwards with no arguments.
Deliberately NOT a single ``sbatch --array`` submission for either workload:
a job array forces the same ``--cpus-per-task`` on every task, which does not
fit ``lnprob`` mode's varying ``n_cpu`` (Antoine's call, 2026-09-17).

Why ``lnprob`` mode's ``n_iter_per_cpu`` is given per point, not once for the
whole sweep (Antoine, 2026-09-17): each worker's average call time is
noisier with fewer calls (dominated by the residual post-fork "first touch"
cost -- see :mod:`starships.slurm.worker`'s docstring), so a good estimate
needs more calls per worker at higher ``n_cpu``, not the same number
everywhere. And since all of a job's workers run in parallel, extra calls
per worker cost comparatively little *extra wall time* as ``n_cpu`` grows --
so scaling ``n_iter_per_cpu`` up with ``n_cpu`` also keeps every job in the
sweep roughly the same wall-clock length, instead of the low-``n_cpu`` jobs
(fewer, slower workers) taking much longer than the high-``n_cpu`` ones for
the same total work. There's no formula for "the right" iteration count here
(that would need already knowing the per-call cost this sweep exists to
measure) -- just pick more for higher ``n_cpu``, and check the resulting
per-job wall times (`pool wall time` in each job's log) once the sweep is
done; adjust and re-run a point if one came out far shorter or longer than
the rest. ``retrieval`` mode's ``n_steps_burnin`` is naturally already "per
walker" (every walker goes through the same steps), so it is a single value
shared across the whole sweep instead (``--n-steps``).

Each job's ``--wrap`` command calls the target console script
(``run_starships_resource_test`` or ``run_starships_retrieval``) by its
*absolute path*, resolved from ``sys.prefix`` (the active Python
environment's root) -- not by ``source``-ing any particular venv's
``activate`` script. This is what makes the whole tool venv-agnostic:
whichever Python environment ``run_starships_resource_sweep`` itself was
launched from (yours, a labmate's, a student's) is exactly the one used
inside the submitted jobs too, with nothing hard-coded.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


def _console_script_executable(name: str) -> Path:
    """Absolute path to a console script installed in the currently-active
    Python environment (e.g. ``run_starships_resource_test``,
    ``run_starships_retrieval``) -- see module docstring for why this
    replaces activating a venv by name.

    Uses ``sys.prefix`` (the active environment's root), not
    ``sys.executable``'s own directory -- some Python installs (e.g. certain
    macOS venvs wrapping a Framework build) have ``sys.executable`` be a
    symlink whose *resolved* target lives outside the venv's ``bin/``
    entirely, which would silently look in the wrong place. ``sys.prefix`` is
    exactly what pip itself used to decide where to install console scripts,
    so it is the robust way to find them again.
    """
    candidate = Path(sys.prefix) / 'bin' / name
    if not candidate.exists():
        raise FileNotFoundError(
            f'Could not find {name} in {Path(sys.prefix) / "bin"} -- is starships installed '
            f'in this environment (pip install -e . or pip install starships)?')
    return candidate


def _parse_sweep_point(token: str) -> Tuple[int, int]:
    """Parse one ``"n_cpu:n_iter_per_cpu"`` CLI token (lnprob workload) into
    ``(n_cpu, n_iter_per_cpu)``, both positive integers."""
    if ':' not in token:
        raise ValueError(f'{token!r} is not "n_cpu:n_iter_per_cpu" (e.g. "4:6")')
    n_cpu_str, n_iter_str = token.split(':', 1)
    n_cpu, n_iter_per_cpu = int(n_cpu_str), int(n_iter_str)
    if n_cpu <= 0 or n_iter_per_cpu <= 0:
        raise ValueError(f'{token!r}: both n_cpu and n_iter_per_cpu must be positive')
    return n_cpu, n_iter_per_cpu


def submit_lnprob_sweep(config: str, sweep_points: List[Tuple[int, int]], account: str,
                        mem: str, time_limit: str, out_dir: Path) -> dict:
    """Submit one SLURM job per ``(n_cpu, n_iter_per_cpu)`` in ``sweep_points``
    (the ``lnprob`` workload) and write the sweep's metadata file.

    Parameters
    ----------
    config : str
        Path to the retrieval input YAML file (passed through to each job).
    sweep_points : list of (int, int)
        ``(n_cpu, n_iter_per_cpu)`` pairs, one job each -- see module
        docstring for why ``n_iter_per_cpu`` is chosen per point rather than
        shared across the whole sweep.
    account : str
        SLURM ``--account`` for every job (no default -- this is
        institution/user-specific, unlike ``mem``/``time_limit`` below).
    mem, time_limit : str
        SLURM ``--mem``/``--time`` for every job -- generous placeholders are
        fine (the whole point of the sweep is not knowing the real numbers
        yet); narrow down once real results come back if these are
        consistently too high (or bump them up if a job runs out). Real
        KELT-20b (SPIRou+JWST/NIRSpec G395H) runs peaked at 28-56 GB, well
        under the 64G default -- lower it for a similar config.
    out_dir : Path
        Where to write ``last_sweep.json`` and the ``logs/`` directory.

    Returns
    -------
    dict
        The sweep metadata that was written to ``last_sweep.json``.
    """
    resource_test_exe = _console_script_executable('run_starships_resource_test')
    log_dir = out_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    sweep_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    jobs = []
    for n_cpu, n_iter_per_cpu in sweep_points:
        log_file = log_dir / f'{sweep_id}_ncpu{n_cpu}_%j.txt'
        cmd = [
            'sbatch', '--parsable',
            f'--account={account}',
            f'--cpus-per-task={n_cpu}',
            f'--mem={mem}',
            f'--time={time_limit}',
            f'--job-name=resprofile_{sweep_id}_ncpu{n_cpu}',
            f'--output={log_file}',
            '--mail-type=FAIL',
            f'--wrap={resource_test_exe} {config} {n_cpu} {n_iter_per_cpu}',
        ]
        job_id = subprocess.run(cmd, capture_output=True, text=True, check=True
                                ).stdout.strip()
        print(f'n_cpu={n_cpu} (n_iter_per_cpu={n_iter_per_cpu}) -> job {job_id}')
        # %j is resolved by sbatch on the cluster side using this same job ID,
        # so we can compute the final filename ourselves rather than
        # re-discovering it later.
        resolved_log = str(log_file).replace('%j', job_id)
        jobs.append(dict(n_cpu=n_cpu, n_iter_per_cpu=n_iter_per_cpu, job_id=job_id,
                        log_file=resolved_log))

    sweep = dict(sweep_id=sweep_id, workload='lnprob', config=config, jobs=jobs)
    state_file = out_dir / 'last_sweep.json'
    with open(state_file, 'w') as f:
        json.dump(sweep, f, indent=2)

    print(f'\nSubmitted {len(jobs)} jobs for sweep {sweep_id}.')
    print(f'Sweep info saved to {state_file}.')
    print('Once all jobs complete (squeue -u $USER to check), run: run_starships_resource_parse')
    return sweep


def submit_retrieval_sweep(config: str, n_cpu: int, n_walkers_per_cpu_list: List[int],
                           n_steps: int, account: str, mem: str, time_limit: str,
                           out_dir: Path) -> dict:
    """Submit one SLURM job per ``n_walkers_per_cpu`` in
    ``n_walkers_per_cpu_list`` (the ``retrieval`` workload, fixed ``n_cpu``)
    and write the sweep's metadata file.

    Each job is a real, short ``run_starships_retrieval`` run -- ``n_cpu``
    (via ``--cpus-per-task``, read from ``$SLURM_CPUS_PER_TASK`` by
    ``retrieval.py`` itself), ``n_walkers_per_cpu`` and ``n_steps_burnin``
    are all just command-line overrides already natively supported by
    ``retrieval.main()``/``prepare_run()`` -- nothing here reimplements or
    modifies anything in ``retrieval.py``.

    Parameters
    ----------
    config : str
        Path to the retrieval input YAML file.
    n_cpu : int
        Fixed ``--cpus-per-task`` for every job in this sweep (this workload
        varies ``n_walkers_per_cpu``, not ``n_cpu`` -- see module docstring).
    n_walkers_per_cpu_list : list of int
        ``n_walkers_per_cpu`` values to test, one job each
        (``n_walkers = n_cpu * n_walkers_per_cpu``).
    n_steps : int
        ``n_steps_burnin`` override for every job -- shared across the sweep
        (naturally already "per walker", see module docstring).
    account, mem, time_limit, out_dir
        See ``submit_lnprob_sweep``.

    Returns
    -------
    dict
        The sweep metadata that was written to ``last_sweep.json``.
    """
    retrieval_exe = _console_script_executable('run_starships_retrieval')
    log_dir = out_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    sweep_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    jobs = []
    for n_walkers_per_cpu in n_walkers_per_cpu_list:
        log_file = log_dir / f'{sweep_id}_ensemble_wpc{n_walkers_per_cpu}_%j.txt'
        cmd = [
            'sbatch', '--parsable',
            f'--account={account}',
            f'--cpus-per-task={n_cpu}',
            f'--mem={mem}',
            f'--time={time_limit}',
            f'--job-name=resprofile_{sweep_id}_wpc{n_walkers_per_cpu}',
            f'--output={log_file}',
            '--mail-type=FAIL',
            f'--wrap={retrieval_exe} yaml_file={config} '
            f'n_walkers_per_cpu={n_walkers_per_cpu} n_steps_burnin={n_steps}',
        ]
        job_id = subprocess.run(cmd, capture_output=True, text=True, check=True
                                ).stdout.strip()
        n_walkers = n_cpu * n_walkers_per_cpu
        print(f'n_walkers_per_cpu={n_walkers_per_cpu} (n_walkers={n_walkers}) -> job {job_id}')
        resolved_log = str(log_file).replace('%j', job_id)
        jobs.append(dict(n_walkers_per_cpu=n_walkers_per_cpu, job_id=job_id,
                        log_file=resolved_log))

    sweep = dict(sweep_id=sweep_id, workload='retrieval', config=config, n_cpu=n_cpu,
                n_steps=n_steps, jobs=jobs)
    state_file = out_dir / 'last_sweep.json'
    with open(state_file, 'w') as f:
        json.dump(sweep, f, indent=2)

    print(f'\nSubmitted {len(jobs)} jobs for sweep {sweep_id} (n_cpu={n_cpu} fixed).')
    print(f'Sweep info saved to {state_file}.')
    print('Once all jobs complete (squeue -u $USER to check), run: run_starships_resource_parse')
    return sweep


def submit_retrieval_sweep_fixed_walkers(config: str, n_walkers: int, n_cpu_list: List[int],
                                         n_steps: int, account: str, mem: str, time_limit: str,
                                         out_dir: Path, real_cpu: Optional[int] = None) -> dict:
    """Submit one SLURM job per ``n_cpu`` in ``n_cpu_list`` (the ``retrieval``
    workload, fixed ``n_walkers``) and write the sweep's metadata file.

    Complements ``submit_retrieval_sweep`` (fixed ``n_cpu``, varying
    ``n_walkers_per_cpu``) -- see module docstring for when to use which.
    Passes ``n_walkers`` directly to ``run_starships_retrieval`` (not
    ``n_walkers_per_cpu``), so it stays exactly fixed regardless of ``n_cpu``
    -- ``retrieval.py`` uses it as-is when given, no divisibility by ``n_cpu``
    required.

    Parameters
    ----------
    config : str
        Path to the retrieval input YAML file.
    n_walkers : int
        Fixed walker count for every job in this sweep.
    n_cpu_list : list of int
        ``n_cpu`` values to test, one job each -- this is the *logical*
        ``Pool``/``EnsembleSampler`` worker count (``n_cpu=`` passed
        explicitly to ``run_starships_retrieval``, bypassing its own
        ``$SLURM_CPUS_PER_TASK`` auto-detection). Normally also the *real*
        core count requested per job (``--cpus-per-task``) -- unless
        ``real_cpu`` is given (see below).
    n_steps : int
        ``n_steps_burnin`` override for every job.
    account, mem, time_limit, out_dir
        See ``submit_lnprob_sweep``.
    real_cpu : int, optional
        If given, every job requests this FIXED real ``--cpus-per-task``
        (physical cores) regardless of the logical ``n_cpu`` being tested --
        i.e. an oversubscription/hyperthreading test (Antoine, 2026-09-18):
        does telling ``Pool``/``EnsembleSampler`` to use MORE worker
        processes than real allocated cores help throughput (smaller
        per-worker share of the fixed walker count) despite the workers now
        contending for fewer real cores? Untested territory -- an earlier,
        unrelated over-subscription attempt in ``logl_grid.py`` (see
        ``Notes/plan_revision_starships.md``, chantier C) made things worse
        there (CPU/memory-bound workers, not I/O-bound), so there is no
        default assumption this will help here either; that is exactly what
        this option is for finding out. Omit it to keep the normal
        one-real-core-per-logical-worker behaviour.

    Returns
    -------
    dict
        The sweep metadata that was written to ``last_sweep.json``.
    """
    retrieval_exe = _console_script_executable('run_starships_retrieval')
    log_dir = out_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    sweep_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    jobs = []
    for n_cpu in n_cpu_list:
        cpus_per_task = real_cpu if real_cpu is not None else n_cpu
        log_file = log_dir / f'{sweep_id}_ensemble_ncpu{n_cpu}_%j.txt'
        cmd = [
            'sbatch', '--parsable',
            f'--account={account}',
            f'--cpus-per-task={cpus_per_task}',
            f'--mem={mem}',
            f'--time={time_limit}',
            f'--job-name=resprofile_{sweep_id}_ncpu{n_cpu}',
            f'--output={log_file}',
            '--mail-type=FAIL',
            # n_cpu is always passed explicitly (not left to
            # $SLURM_CPUS_PER_TASK auto-detection) so oversubscription
            # (n_cpu > cpus_per_task) works: retrieval.py would otherwise
            # just read back cpus_per_task, silently undoing the test.
            f'--wrap={retrieval_exe} yaml_file={config} n_walkers={n_walkers} n_cpu={n_cpu} '
            f'n_steps_burnin={n_steps}',
        ]
        job_id = subprocess.run(cmd, capture_output=True, text=True, check=True
                                ).stdout.strip()
        wpc = n_walkers / n_cpu
        oversub_note = f', real_cpu={cpus_per_task}' if real_cpu is not None else ''
        print(f'n_cpu={n_cpu} (n_walkers_per_cpu≈{wpc:.1f}{oversub_note}) -> job {job_id}')
        resolved_log = str(log_file).replace('%j', job_id)
        jobs.append(dict(n_cpu=n_cpu, real_cpu=cpus_per_task, job_id=job_id,
                        log_file=resolved_log))

    sweep = dict(sweep_id=sweep_id, workload='retrieval', config=config, n_walkers=n_walkers,
                n_steps=n_steps, jobs=jobs)
    state_file = out_dir / 'last_sweep.json'
    with open(state_file, 'w') as f:
        json.dump(sweep, f, indent=2)

    print(f'\nSubmitted {len(jobs)} jobs for sweep {sweep_id} (n_walkers={n_walkers} fixed'
         f'{oversub_note if real_cpu is not None else ""}).')
    print(f'Sweep info saved to {state_file}.')
    print('Once all jobs complete (squeue -u $USER to check), run: run_starships_resource_parse')
    return sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', help='Retrieval input YAML file.')
    parser.add_argument('points', nargs='+', metavar='POINT',
                        help='lnprob workload (default): "n_cpu:n_iter_per_cpu" pairs, one job '
                             'each, e.g. 1:2 2:3 4:6 8:12. retrieval workload with --n-cpu: bare '
                             'n_walkers_per_cpu values, e.g. 2 4 8 14. retrieval workload with '
                             '--n-walkers: bare n_cpu values, e.g. 10 12 14 24. See this '
                             'command\'s full --help text.')
    parser.add_argument('--workload', choices=['lnprob', 'retrieval'], default='lnprob',
                        help='See module description above. Default: lnprob.')
    parser.add_argument('--n-cpu', type=int, default=None,
                        help='(retrieval workload) fixed n_cpu for every job -- points are '
                             'n_walkers_per_cpu values. Exactly one of --n-cpu/--n-walkers is '
                             'required for --workload retrieval.')
    parser.add_argument('--n-walkers', type=int, default=None,
                        help='(retrieval workload) fixed n_walkers for every job -- points are '
                             'n_cpu values. Exactly one of --n-cpu/--n-walkers is required for '
                             '--workload retrieval.')
    parser.add_argument('--n-steps', type=int, default=None,
                        help='(retrieval workload, required) n_steps_burnin override for '
                             'every job.')
    parser.add_argument('--real-cpu', type=int, default=None,
                        help='(retrieval workload with --n-walkers only) fixed REAL '
                             '--cpus-per-task for every job, decoupled from the logical n_cpu '
                             'points being tested -- an oversubscription/hyperthreading test '
                             '(n_cpu > --real-cpu means more worker processes than physical '
                             'cores). Omit for the normal 1 logical worker = 1 real core '
                             'behaviour.')
    parser.add_argument('--account', required=True, help='SLURM --account for every job.')
    parser.add_argument('--mem', default='64G', help='SLURM --mem for every job (default: 64G '
                                                      '-- likely generous, see --help).')
    parser.add_argument('--time', dest='time_limit', default='00:30:00',
                        help='SLURM --time for every job (default: 00:30:00).')
    parser.add_argument('--out-dir', type=Path, default=Path.cwd(),
                        help='Where to write last_sweep.json and logs/ (default: current '
                             'directory).')
    args = parser.parse_args()

    if args.workload == 'lnprob':
        try:
            sweep_points = [_parse_sweep_point(tok) for tok in args.points]
        except ValueError as e:
            parser.error(str(e))
        n_cpu_values = [n_cpu for n_cpu, _ in sweep_points]
        if len(set(n_cpu_values)) != len(n_cpu_values):
            parser.error('n_cpu values must be distinct (one job per n_cpu).')
        submit_lnprob_sweep(args.config, sweep_points, args.account, args.mem, args.time_limit,
                            args.out_dir)
    else:
        if args.n_steps is None:
            parser.error('--n-steps is required for --workload retrieval')
        if bool(args.n_cpu) == bool(args.n_walkers):
            parser.error('exactly one of --n-cpu/--n-walkers is required for --workload '
                         'retrieval (not both, not neither).')
        try:
            point_values = [int(tok) for tok in args.points]
        except ValueError:
            parser.error('retrieval workload points must be plain integers, not "n_cpu:n_iter" '
                         'pairs.')
        if any(n <= 0 for n in point_values):
            parser.error('points must be positive integers.')
        if len(set(point_values)) != len(point_values):
            parser.error('points must be distinct (one job per value).')

        if args.real_cpu is not None and args.n_cpu is not None:
            parser.error('--real-cpu only applies to --workload retrieval --n-walkers (it '
                         'decouples the real allocation from the logical n_cpu points, which '
                         'only vary in --n-walkers mode).')

        if args.n_cpu is not None:
            submit_retrieval_sweep(args.config, args.n_cpu, point_values, args.n_steps,
                                   args.account, args.mem, args.time_limit, args.out_dir)
        else:
            submit_retrieval_sweep_fixed_walkers(args.config, args.n_walkers, point_values,
                                                 args.n_steps, args.account, args.mem,
                                                 args.time_limit, args.out_dir,
                                                 real_cpu=args.real_cpu)


if __name__ == '__main__':
    main()
