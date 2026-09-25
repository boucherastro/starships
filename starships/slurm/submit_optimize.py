"""Submit a best-fit optimization from a burn-in (``run_starships_optimize``) to SLURM.

Console script: ``submit_starships_optimize``. See :mod:`starships.retrieval_optimize`
for what the optimization does and how to read its results.

Usage::

    submit_starships_optimize CONFIG.yaml --walker-file BURNIN.h5 [--n-starts N] \\
        --account ACCT --cpus-per-task N --mem MEM --time TIME [options] \\
        [-- <extra run_starships_optimize options>]

With ``--n-starts N`` > 1 the job is submitted as a SLURM array of N tasks. Each
task is an independent optimization from a different starting point of the burn-in
(task 1 starts from the best burn-in sample, task 2 from the best one that is well
separated from it, ...). If they all reach the same ``lnprob`` the maximum is reliable;
compare them with ``run_starships_optimize --compare optimum_*.npz``.

Anything after a lone ``--`` (which must come last) is forwarded verbatim to every
``run_starships_optimize`` call, e.g. ``-- --max-hours 40 --max-evals 3000``. Set
``--max-hours`` a bit below the SLURM ``--time``: the optimization then stops cleanly
and saves its result instead of being killed. (If it is killed anyway, its state is
saved after every generation: rerun the same command with ``-- --resume``.)

The YAML is copied to scratch *once* and shared by all the tasks of one submission, so
the original can be edited right away (see :mod:`starships.slurm.common`).
``run_starships_optimize`` evaluates the model in parallel over ``$SLURM_CPUS_PER_TASK``
processes; the model is initialized once and shared between them (not duplicated).

Example::

    submit_starships_optimize params_kelt20_burnin.yaml \\
        --walker-file walker_steps_kelt20_burnin.h5 --n-starts 3 \\
        --account def-rdoyon --cpus-per-task 2 --mem 23750M --time 2-00:00:00 \\
        --job-name K20_optimize -- --max-hours 44
"""
import argparse
import shlex
import sys
from pathlib import Path
from typing import Optional, Sequence

from starships.slurm.common import (
    add_sbatch_arguments, build_sbatch_command, console_script_executable, log_file_pattern,
    make_temp_yaml_copy, record_submission, resolve_scratch_dir, submit_job,
)


def submit_optimize(config: Path, walker_file: Path, n_starts: int, extra_args: Sequence[str],
                    args: argparse.Namespace) -> Optional[str]:
    """Copy the YAML, build the ``sbatch`` command and submit it.

    Parameters
    ----------
    config : Path
        The retrieval YAML that goes with the burn-in (original; the copy is made here).
    walker_file : Path
        The burn-in file. Made absolute if it exists from the current directory (the job
        does not run from here); a bare name is left as is and looked for in the
        YAML ``walker_path`` by the job.
    n_starts : int
        Number of independent optimizations (SLURM array size); 1 means a plain single job.
    extra_args : sequence of str
        Forwarded verbatim to ``run_starships_optimize``.
    args : argparse.Namespace
        Parsed shared SLURM options.

    Returns
    -------
    str or None
        Job ID, or None with ``--dry-run``.
    """
    scratch_dir = resolve_scratch_dir(args.scratch_dir)
    optimize_exe = console_script_executable('run_starships_optimize')

    yaml_copy = make_temp_yaml_copy(config, scratch_dir, tool_name='submit_starships_optimize')
    print(f'Input YAML: {config}')
    print(f'  copied to: {yaml_copy}  (this copy is what the job reads)')

    walker_file = walker_file.expanduser()
    if walker_file.is_file():
        walker_file = walker_file.resolve()

    use_array = n_starts > 1
    array_spec = f'1-{n_starts}' if use_array else None
    log_file = log_file_pattern(scratch_dir, array=use_array)

    # In an array, each task finds its own start index from $SLURM_ARRAY_TASK_ID
    wrap = f'{optimize_exe} {yaml_copy} --walker-file {shlex.quote(str(walker_file))}'
    extra = ' '.join(shlex.quote(a) for a in extra_args)
    if extra:
        wrap += f' {extra}'

    cmd = build_sbatch_command(args, wrap, log_file, array=array_spec)
    if use_array:
        print(f'{n_starts} independent optimizations (SLURM array {array_spec}).')
    print(f'Log file(s): {log_file}')

    job_id = submit_job(cmd, dry_run=args.dry_run)
    if job_id is not None:
        record_submission(scratch_dir, job_id, cmd, yaml_copy=yaml_copy)
    return job_id


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog='submit_starships_optimize', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', type=Path, help='Retrieval YAML that goes with the burn-in.')
    parser.add_argument('--walker-file', type=Path, required=True, metavar='BURNIN.h5',
                        help='Burn-in file (emcee HDF5). A bare name is looked for in the YAML walker_path.')
    parser.add_argument('--n-starts', type=int, default=1,
                        help='Number of independent optimizations from different burn-in samples, '
                             'submitted as a SLURM array 1-N (default: 1, plain single job).')
    add_sbatch_arguments(parser, default_job_name='starships_optimize')

    # Everything after a lone "--" is forwarded verbatim to run_starships_optimize.
    # Split by hand rather than with argparse.REMAINDER (see submit_logl_grid.main).
    argv = sys.argv[1:] if argv is None else list(argv)
    if '--' in argv:
        split_at = argv.index('--')
        argv, extra_args = argv[:split_at], argv[split_at + 1:]
    else:
        extra_args = []
    args = parser.parse_args(argv)

    if args.n_starts < 1:
        parser.error('--n-starts must be >= 1')
    if '--start-index' in extra_args:
        parser.error('--start-index cannot be forwarded: each SLURM task chooses its own start '
                     '(use --n-starts N).')

    config = args.config.expanduser().resolve()
    if not config.is_file():
        parser.error(f'config file not found: {config}')

    submit_optimize(config, args.walker_file, args.n_starts, extra_args, args)


if __name__ == '__main__':
    main()
