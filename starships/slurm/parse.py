"""Parse a completed resource-estimation sweep into plots and an optional
resource table for a real retrieval run. Handles both workloads
:mod:`starships.slurm.submit` can produce (branches on the sweep's
``"workload"`` field -- ``"lnprob"`` or ``"retrieval"``).

Console script: ``run_starships_resource_parse``. Run with no arguments from
the same directory ``run_starships_resource_sweep`` was run from (reads
``last_sweep.json`` there by default). ``lnprob`` workload: pass
``--n-walkers``/``--n-steps`` to also get an estimated ``--mem``/``--time``
table for a real run of that size. ``retrieval`` workload: pass
``--baseline-time-per-iter-s`` (read off a prior ``lnprob``-workload sweep's
own printed table) to also get an efficiency-vs-pure-compute comparison.

For BOTH workloads, memory always comes from ``sacct`` (SLURM's own
cgroup-based accounting): real, ground-truth peak memory (``MaxRSS``) for
that job's whole process tree (parent + every forked ``Pool`` worker), and
its overall state (to make sure the job actually finished before trusting
its numbers). Time differs by workload:

- ``lnprob``: from that job's own stdout log, written by
  :mod:`starships.slurm.worker`: the "pool wall time" line, i.e. how long the
  ``multiprocessing.Pool`` block itself took (excludes one-time
  setup/data-loading time, which does not depend on ``n_cpu`` and would bias
  the per-``n_cpu`` time comparison if included).
- ``retrieval``: from ``sacct``'s own ``Elapsed`` directly (the whole job,
  setup included this time -- there is no separate "pool wall time" line to
  read, since these jobs run the real, unmodified ``run_starships_retrieval``
  rather than a purpose-built timer). Use enough ``--n-steps`` in the sweep
  to keep the one-time setup cost a small fraction of the total.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from starships.slurm.estimation import (estimate_run_resources, plot_ensemble_efficiency,
                                        plot_resource_scaling)

# sacct can lag a few seconds behind a job actually finishing before its
# accounting is fully populated -- retry a handful of times rather than
# reporting a false "MaxRSS missing" error the moment every job leaves the
# queue. (It can also be down entirely for longer stretches, e.g. a
# cluster-wide slurmdbd outage -- these retries do not help with that; if
# every attempt fails, re-run this command later, no need to resubmit jobs.)
SACCT_RETRY_ATTEMPTS = 5
SACCT_RETRY_DELAY_S = 10

_POOL_TIME_RE = re.compile(r'pool wall time:\s*([\d.]+)s')


def load_sweep_state(path: Path) -> dict:
    if not path.exists():
        sys.exit(f'No sweep file found at {path} -- run run_starships_resource_sweep first, '
                 f'or pass --sweep-file to point at an older sweep.')
    with open(path) as f:
        return json.load(f)


def _parse_max_rss_to_mb(raw: str) -> Optional[float]:
    """Convert sacct's MaxRSS string (e.g. '1234567K', '512.3M', '2G') to MB.

    sacct reports MaxRSS with a unit suffix (K/M/G, base-1024) attached
    directly to the number, or empty if that step's accounting has nothing to
    report (e.g. the top-level job step, as opposed to its '.batch' step --
    see run_sacct's own filtering for why only '.batch' rows are read).
    """
    raw = raw.strip()
    if not raw:
        return None
    unit = raw[-1].upper()
    scale = {'K': 1 / 1024, 'M': 1.0, 'G': 1024.0}
    if unit in scale:
        return float(raw[:-1]) * scale[unit]
    # No recognized suffix -- sacct is documented to always attach one for
    # MaxRSS, so this is unexpected; treat the raw number as already being in
    # KB (sacct's base unit when no suffix is shown) rather than guessing.
    return float(raw) / 1024


def _parse_elapsed_to_s(raw: str) -> Optional[float]:
    """Convert sacct's Elapsed string ('[D-]HH:MM:SS') to seconds."""
    raw = raw.strip()
    if not raw:
        return None
    days = 0
    if '-' in raw:
        day_str, raw = raw.split('-', 1)
        days = int(day_str)
    h, m, s = (int(x) for x in raw.split(':'))
    return days * 86400 + h * 3600 + m * 60 + s


def run_sacct(job_id: str) -> dict:
    """Query sacct for one job's State/Elapsed/MaxRSS, from its '.batch' step
    (the step that actually ran run_starships_resource_test and has the real
    MaxRSS -- the parent job-ID step's MaxRSS is typically empty)."""
    cmd = ['sacct', '-j', job_id, '--format=JobID,State,Elapsed,MaxRSS',
           '--parsable2', '--noheader']
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        # sacct --parsable2 is usually delimiter-only (no trailing '|'), but
        # some versions/configs do add a trailing empty field -- accept
        # either rather than silently skipping every real line if that
        # differs from what was tested against here.
        fields = line.split('|')
        if len(fields) < 4:
            continue
        step_id, state, elapsed, max_rss = fields[:4]
        if step_id.endswith('.batch'):
            return dict(state=state, elapsed_s=_parse_elapsed_to_s(elapsed),
                       max_rss_mb=_parse_max_rss_to_mb(max_rss))
    return dict(state=None, elapsed_s=None, max_rss_mb=None)


def parse_pool_wall_time(log_file: Path) -> Optional[float]:
    if not log_file.exists():
        return None
    text = log_file.read_text()
    match = _POOL_TIME_RE.search(text)
    return float(match.group(1)) if match else None


def _get_sacct_or_die(job_id: str, label: str, log_file: Path) -> dict:
    """Query sacct for one job, retrying a few times (see
    SACCT_RETRY_ATTEMPTS), and exit with a clear message (rather than
    silently building results from partial/wrong data) if it never comes back
    COMPLETED with a MaxRSS. Shared by both workloads' result-building.

    Parameters
    ----------
    job_id : str
    label : str
        How to refer to this job in an error message (e.g. ``'n_cpu=4'`` or
        ``'n_walkers_per_cpu=8'``).
    log_file : Path
        Only used to point the user at it in an error message.
    """
    sacct_info = None
    for attempt in range(SACCT_RETRY_ATTEMPTS):
        sacct_info = run_sacct(job_id)
        if sacct_info['state'] == 'COMPLETED' and sacct_info['max_rss_mb'] is not None:
            break
        if attempt < SACCT_RETRY_ATTEMPTS - 1:
            time.sleep(SACCT_RETRY_DELAY_S)

    if sacct_info['state'] != 'COMPLETED':
        sys.exit(f'Job {job_id} ({label}) has sacct state {sacct_info["state"]!r}, not '
                 f'COMPLETED -- check `squeue -u $USER` / `sacct -j {job_id}` and the log at '
                 f'{log_file} before re-running this sweep.')
    if sacct_info['max_rss_mb'] is None:
        sys.exit(f'Job {job_id} ({label}) is COMPLETED but sacct has no MaxRSS yet after '
                 f'{SACCT_RETRY_ATTEMPTS} retries -- try running run_starships_resource_parse '
                 f'again in a minute (or later, if sacct itself is down cluster-wide -- check '
                 f'`sacct -j {job_id}` directly).')
    return sacct_info


def build_results(sweep: dict) -> list:
    """Query sacct + logs for every job in an ``lnprob``-workload sweep and
    build the ``results: List[dict]`` structure
    :mod:`starships.slurm.estimation`'s ``plot_resource_scaling``/
    ``estimate_run_resources`` expect."""
    results = []
    for job in sweep['jobs']:
        n_cpu, job_id, log_file = job['n_cpu'], job['job_id'], Path(job['log_file'])
        n_iter_per_cpu = job['n_iter_per_cpu']

        sacct_info = _get_sacct_or_die(job_id, f'n_cpu={n_cpu}', log_file)

        pool_wall_time_s = parse_pool_wall_time(log_file)
        if pool_wall_time_s is None:
            sys.exit(f'Could not find a "pool wall time" line in {log_file} for job {job_id} '
                     f'(n_cpu={n_cpu}) -- did run_starships_resource_test crash before '
                     f'finishing? Check the log directly.')

        results.append(dict(
            n_cpu=n_cpu,
            max_rss_mb=sacct_info['max_rss_mb'],
            elapsed_s=sacct_info['elapsed_s'],
            pool_wall_time_s=pool_wall_time_s,
            n_iter_per_cpu=n_iter_per_cpu,
            # Every worker runs the same n_iter_per_cpu calls in parallel with
            # the others (see worker.py) -- pool_wall_time_s already reflects
            # one worker's full run, so no n_cpu factor is needed here (unlike
            # the old "n_iter_total split across workers" scheme).
            avg_time_per_iter_s=pool_wall_time_s / n_iter_per_cpu,
        ))
    return sorted(results, key=lambda e: e['n_cpu'])


def build_ensemble_results(sweep: dict) -> list:
    """Query sacct for every job in a ``retrieval``-workload sweep and build
    the ``results: List[dict]`` structure
    :mod:`starships.slurm.estimation`'s ``plot_ensemble_efficiency`` expects.
    No log-parsing needed (unlike ``build_results``) -- these jobs run the
    real, unmodified ``run_starships_retrieval``, which prints nothing this
    tool greps for; time comes from ``sacct``'s ``Elapsed`` instead (see
    module docstring).

    Handles both ``retrieval``-workload sweep shapes (see
    :mod:`starships.slurm.submit`'s two ``submit_retrieval_sweep*``
    functions): fixed ``n_cpu`` (``sweep['n_cpu']``, jobs carry
    ``n_walkers_per_cpu``) or fixed ``n_walkers`` (``sweep['n_walkers']``,
    jobs carry ``n_cpu``) -- detected from which key the sweep has.
    """
    n_steps = sweep['n_steps']
    fixed_n_cpu = sweep.get('n_cpu')  # None if this sweep fixed n_walkers instead
    fixed_n_walkers = sweep.get('n_walkers')  # None if this sweep fixed n_cpu instead

    results = []
    for job in sweep['jobs']:
        job_id, log_file = job['job_id'], Path(job['log_file'])

        if fixed_n_cpu is not None:
            n_cpu, n_walkers_per_cpu = fixed_n_cpu, job['n_walkers_per_cpu']
            n_walkers = n_cpu * n_walkers_per_cpu
            real_cpu = job.get('real_cpu', n_cpu)
            label = f'n_walkers_per_cpu={n_walkers_per_cpu}'
        else:
            n_cpu, n_walkers = job['n_cpu'], fixed_n_walkers
            n_walkers_per_cpu = n_walkers / n_cpu  # not necessarily an integer
            real_cpu = job.get('real_cpu', n_cpu)  # older sweeps: no oversubscription, defaults to n_cpu
            label = f'n_cpu={n_cpu}' + (f' (real_cpu={real_cpu})' if real_cpu != n_cpu else '')

        sacct_info = _get_sacct_or_die(job_id, label, log_file)

        results.append(dict(
            n_cpu=n_cpu,
            real_cpu=real_cpu,
            n_walkers_per_cpu=n_walkers_per_cpu,
            n_walkers=n_walkers,
            max_rss_mb=sacct_info['max_rss_mb'],
            elapsed_s=sacct_info['elapsed_s'],
            # Whole job (setup + n_steps real MCMC steps) -- see module
            # docstring for why, unlike build_results, this can't exclude
            # setup time here.
            time_per_step_s=sacct_info['elapsed_s'] / n_steps,
        ))
    return sorted(results, key=lambda e: e['n_walkers_per_cpu'])


def print_table(results: list) -> None:
    print(f"{'n_cpu':>6} | {'MaxRSS (GB)':>12} | {'pool time (s)':>14} | {'avg s/iter':>10}")
    for e in results:
        print(f"{e['n_cpu']:>6} | {e['max_rss_mb'] / 1024:>12.2f} | "
              f"{e['pool_wall_time_s']:>14.1f} | {e['avg_time_per_iter_s']:>10.2f}")


def print_ensemble_table(results: list, baseline_time_per_iter_s: Optional[float]) -> None:
    # n_walkers_per_cpu is a float when this sweep fixed n_walkers and varied
    # n_cpu instead (see build_ensemble_results) -- always shown to 1 decimal
    # so it stays readable either way. real_cpu only differs from n_cpu for
    # an oversubscription sweep (--real-cpu) -- shown so it's not mistaken
    # for a data-entry mistake when it does.
    show_real_cpu = any(e['real_cpu'] != e['n_cpu'] for e in results)
    header = f"{'n_cpu':>6} | " + (f"{'real_cpu':>8} | " if show_real_cpu else '') + \
            f"{'n_walkers_per_cpu':>17} | {'n_walkers':>9} | {'MaxRSS (GB)':>12} | {'s/step':>8}"
    if baseline_time_per_iter_s is not None:
        header += f" | {'efficiency':>10}"
    print(header)
    for e in results:
        row = f"{e['n_cpu']:>6} | "
        if show_real_cpu:
            row += f"{e['real_cpu']:>8} | "
        row += (f"{e['n_walkers_per_cpu']:>17.1f} | {e['n_walkers']:>9} | "
               f"{e['max_rss_mb'] / 1024:>12.2f} | {e['time_per_step_s']:>8.2f}")
        if baseline_time_per_iter_s is not None:
            ideal_s = e['n_walkers_per_cpu'] * baseline_time_per_iter_s
            efficiency_pct = 100 * ideal_s / e['time_per_step_s']
            row += f" | {efficiency_pct:>9.0f}%"
        print(row)


def print_estimates(estimates: list) -> None:
    print(f"\n{'n_cpu':>6} | {'measured':>8} | {'walkers/cpu':>12} | "
          f"{'est. time':>12} | {'est. mem (GB)':>14}")
    for e in estimates:
        hh = e['est_wall_time_s'] / 3600
        print(f"{e['n_cpu']:>6} | {str(e['measured']):>8} | {e['n_walkers_per_cpu']:>12} | "
              f"{hh:>10.1f}h | {e['est_memory_mb'] / 1024:>14.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sweep-file', type=Path, default=Path.cwd() / 'last_sweep.json',
                        help='Sweep metadata file written by run_starships_resource_sweep '
                             '(default: last_sweep.json in the current directory).')
    parser.add_argument('--plot-path', type=Path, default=None,
                        help='Where to save the scaling plot (default: '
                             'resource_scaling_<sweep_id>.png in the current directory).')
    parser.add_argument('--n-walkers', type=int, default=None,
                        help='If given (with --n-steps), also print an estimated '
                             '--mem/--time table for a real run of this many walkers.')
    parser.add_argument('--n-steps', type=int, default=None,
                        help='Total MCMC steps for the real run (burnin + sampling).')
    parser.add_argument('--extra-n-cpu', type=int, nargs='*', default=(),
                        help='n_cpu values to also estimate for, beyond what was swept '
                             '(e.g. --extra-n-cpu 16 32) -- e.g. the n_cpu you actually plan '
                             'to run the real retrieval at. Also produces a second, zoomed-out '
                             '"_extrapolated" plot marking these projected points.')
    parser.add_argument('--available-cpus', type=int, default=None,
                        help='Physical CPUs actually available per job, if known -- draws '
                             'the oversubscription boundary on the plot (lnprob workload only).')
    parser.add_argument('--baseline-time-per-iter-s', type=float, default=None,
                        help='(retrieval workload) pure per-lnprob()-call cost in seconds, '
                             'read off a prior lnprob-workload sweep\'s own printed table -- if '
                             'given, also prints/plots efficiency vs that ideal, no-overhead '
                             'baseline.')
    parser.add_argument('--n-steps-target', type=int, default=None,
                        help='(retrieval workload, needs --n-walkers-target and '
                             '--baseline-time-per-iter-s too) also plot the estimated total wall '
                             'time for a real run of this many MCMC steps, AT A FIXED walker '
                             'count (--n-walkers-target) -- the implied n_cpu varies per point '
                             'tested (n_cpu = n_walkers_target / n_walkers_per_cpu), since the '
                             'walker count is normally the real constraint, not n_cpu.')
    parser.add_argument('--n-walkers-target', type=int, default=None,
                        help='(retrieval workload, with --n-steps-target) the walker count to '
                             'hold fixed for that total-time estimate -- typically the number of '
                             'walkers you actually intend to use for a real run.')
    parser.add_argument('--node-mem-gb', type=float, default=None,
                        help='(retrieval workload) total RAM of one compute node (GB, e.g. 250 '
                             'for a standard Narval node) -- shades memory above it on the plot '
                             'as unavailable within a single node (multiprocessing.Pool cannot '
                             'span nodes).')
    args = parser.parse_args()

    if bool(args.n_walkers) != bool(args.n_steps):
        parser.error('--n-walkers and --n-steps must be given together.')
    if bool(args.n_steps_target) != bool(args.n_walkers_target):
        parser.error('--n-steps-target and --n-walkers-target must be given together.')

    sweep = load_sweep_state(args.sweep_file)
    print(f"Parsing sweep {sweep['sweep_id']} ({len(sweep['jobs'])} jobs)...")

    if sweep.get('workload', 'lnprob') == 'retrieval':
        results = build_ensemble_results(sweep)
        print_ensemble_table(results, args.baseline_time_per_iter_s)

        plot_path = args.plot_path or Path.cwd() / f"ensemble_efficiency_{sweep['sweep_id']}.png"
        plot_ensemble_efficiency(results, str(plot_path),
                                 baseline_time_per_iter_s=args.baseline_time_per_iter_s,
                                 n_steps_target=args.n_steps_target,
                                 n_walkers_target=args.n_walkers_target,
                                 node_mem_gb=args.node_mem_gb)
        print(f'\nPlot saved to {plot_path}')
        if args.baseline_time_per_iter_s is None:
            print('\n(Pass --baseline-time-per-iter-s S -- from a prior lnprob-workload sweep\'s '
                  'table -- to also get an efficiency-vs-pure-compute comparison.)')
        return

    results = build_results(sweep)
    print_table(results)

    plot_path = args.plot_path or Path.cwd() / f"resource_scaling_{sweep['sweep_id']}.png"
    saved_paths = plot_resource_scaling(results, str(plot_path), available_cpus=args.available_cpus,
                                        extra_n_cpu=args.extra_n_cpu)
    print('\nPlot(s) saved to:')
    for p in saved_paths:
        print(f'  {p}')

    if args.n_walkers:
        estimates = estimate_run_resources(results, args.n_walkers, args.n_steps,
                                           extra_n_cpu=args.extra_n_cpu)
        print_estimates(estimates)
    else:
        print('\n(Pass --n-walkers N --n-steps S to also get an estimated --mem/--time '
              'table for a real run.)')


if __name__ == '__main__':
    main()
