"""Run one point of a SLURM resource-estimation sweep for a STARSHIPS retrieval.

This is the per-job workload submitted by ``run_starships_resource_sweep``
(see :mod:`starships.slurm.submit`): it runs a real, small
``lnprob()`` workload across a real ``multiprocessing.Pool(n_cpu)``, inside a
single SLURM job allocated exactly ``--cpus-per-task=<n_cpu>``. It
deliberately does NOT try to measure its own memory usage -- that is
``sacct``'s job (SLURM's own cgroup-based accounting, read *after* the job
finishes, by :mod:`starships.slurm.parse`). Trying to measure memory from
inside the process itself (e.g. sampling ``/proc``) was a real source of bugs
in an earlier version of this tool (see the "Design history" section in the
package README) -- ``sacct`` avoids that whole class of mistakes for free,
since it is exactly what SLURM itself will enforce a ``--mem`` request
against.

This script only measures *time*, by timing the ``Pool`` block directly with
``time.time()`` and printing the result -- see "pool wall time" below, which
``parse.py`` greps out of this job's stdout log. Wall time is not available
from ``sacct`` for just the ``Pool`` portion, only for the whole job (setup +
Pool combined), and setup time (loading the retrieval config and high-res
data) does not depend on ``n_cpu`` -- mixing it into the per-``n_cpu`` time
comparison would bias the fit.

Console script: ``run_starships_resource_test <retrieval_config.yaml> <n_cpu>
<n_iter_per_cpu>``. ``n_iter_per_cpu`` is how many ``lnprob()`` calls EACH
worker runs (not a total split across them) -- see
:mod:`starships.slurm.submit`'s docstring for why the right value grows with
``n_cpu``: a worker with only 1-2 calls has its average dominated by the
residual post-fork "first touch" cost (see below), not the true steady-state
per-iteration cost, and more workers running in parallel means extra calls
per worker cost little *extra wall time* for the job as a whole.

This only measures RAW compute scaling (independent lnprob() calls) -- it
does NOT capture the ensemble-level synchronization cost of a real MCMC run
(every walker must finish before the ensemble can propose the next step). To
test that instead -- e.g. whether more walkers per cpu improves efficiency --
just run a real, short ``run_starships_retrieval`` with ``n_walkers_per_cpu``
and ``n_steps_burnin`` overridden on the command line (both already
supported directly by ``retrieval.main()``/``prepare_run()``, no separate
tool needed) -- see :mod:`starships.slurm.submit`'s ``--ensemble-*``
options.
"""
import argparse
import time
from multiprocessing import Pool
from typing import Tuple

import starships.retrieval as retrieval


def _worker_call(args: Tuple[int, int]) -> None:
    """Run this worker's ``n_iter_per_cpu`` calls and print a one-line
    summary. Runs inside a forked ``Pool`` worker -- printing (not returning
    large data) keeps this simple and avoids pickling anything heavy back to
    the parent process."""
    worker_i, n_iter_per_cpu = args
    thetas = [retrieval.walker_init[k % len(retrieval.walker_init)] for k in range(n_iter_per_cpu)]
    t0 = time.time()
    for theta in thetas:
        logl = retrieval.lnprob(theta)
    dt = time.time() - t0
    print(f'[worker {worker_i}] n_iter={len(thetas)}, total_time={dt:.2f}s, '
          f'avg_time_per_iter={dt / len(thetas):.2f}s', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='Retrieval input YAML file.')
    parser.add_argument('n_cpu', type=int, help='multiprocessing.Pool size to test.')
    parser.add_argument('n_iter_per_cpu', type=int,
                        help='lnprob() calls EACH worker runs (not a total).')
    args = parser.parse_args()

    print(f'=== n_cpu={args.n_cpu}, n_iter_per_cpu={args.n_iter_per_cpu}, '
          f'config={args.config} ===', flush=True)

    t0 = time.time()
    retrieval.setup_retrieval(args.config)
    if retrieval.retrieval_type in ('JR', 'HRR'):
        retrieval.load_high_res_data()
    print(f'setup done in {time.time() - t0:.1f}s (not counted in "pool wall time" below)',
          flush=True)

    if len(retrieval.walker_init) == 0:
        raise ValueError('walker_init is empty -- did setup_retrieval run correctly?')

    # Warm-up in the PARENT, before the Pool is created, so forked workers
    # inherit the already-initialized petitRADTRANS opacity objects via
    # copy-on-write instead of each re-paying that (large) initialization cost
    # on their own -- same reasoning as prepare_run()'s own pre-run sanity
    # check in retrieval.py, which this mirrors deliberately.
    t0 = time.time()
    warm_logl = retrieval.lnprob(retrieval.walker_init[0])
    print(f'warm-up lnprob={warm_logl}, {time.time() - t0:.1f}s', flush=True)

    worker_args = [(worker_i, args.n_iter_per_cpu) for worker_i in range(args.n_cpu)]

    # This is the number parse.py actually reads back out of the log -- see
    # this file's module docstring for why it's timed here rather than read
    # from sacct's whole-job Elapsed.
    t0 = time.time()
    with Pool(args.n_cpu) as pool:
        pool.map(_worker_call, worker_args)
    pool_wall_time_s = time.time() - t0
    print(f'pool wall time: {pool_wall_time_s:.2f}s', flush=True)
    print('=== done -- real memory usage is read from sacct, not this script ===', flush=True)


if __name__ == '__main__':
    main()
