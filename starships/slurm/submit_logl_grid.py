"""Submit Kp x v_sys logL grid computations (``run_starships_logl_grid``) to SLURM.

Console script: ``submit_starships_logl_grid``. Python replacement for the
bash launcher ``run_sbatch_logl_grid.sh``, which hard-coded a list of model
files and used a SLURM array to run one grid per file. Here the model files
are simply listed on the command line, and **one plain job is submitted per
model file** (no array): each job then has its own explicit ``--specfile``,
its own name and its own log (by job ID), and a single file can be re-run without
touching the others. (Same reasoning as ``run_starships_resource_sweep``,
see the README in this directory.)

Usage::

    submit_starships_logl_grid CONFIG.yaml --specfile MODEL.npz [MODEL2.npz ...] \\
        --account ACCT --cpus-per-task N --mem MEM --time TIME [options] \\
        [-- <extra run_starships_logl_grid options>]

Anything after a lone ``--`` (which must come last) is forwarded verbatim to every
``run_starships_logl_grid`` call (e.g. ``-- --plot trailing --kp 227``), so
new options of that command need no change here.

The YAML is copied to scratch *once* and shared by all the jobs of one
submission (they all use the same grid settings by construction), so the
original can be edited right away for the next batch (see
:mod:`starships.slurm.common`). ``run_starships_logl_grid`` sizes its worker
pool from ``$SLURM_CPUS_PER_TASK`` x the YAML's ``n_processes_per_cpu``.

Example::

    submit_starships_logl_grid retrieval_inputs_kelt20_dayside.yaml \\
        --specfile model_spectra/kelt20b_D2_H2O_only.npz model_spectra/kelt20b_D2_all.npz \\
        --account def-dlafre --cpus-per-task 4 --mem 20G --time 07:00:00 --job-name logl_K20
"""
import argparse
import shlex
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from starships.slurm.common import (
    add_sbatch_arguments, build_sbatch_command, console_script_executable, log_file_pattern,
    make_temp_yaml_copy, record_submission, resolve_scratch_dir, submit_job,
)


def submit_logl_grids(config: Path, specfiles: Sequence[Path], extra_args: Sequence[str],
                      args: argparse.Namespace) -> List[str]:
    """Copy the YAML once, then submit one ``run_starships_logl_grid`` job
    per model file.

    Parameters
    ----------
    config : Path
        The user's retrieval/logl-grid YAML (original; the copy is made here).
    specfiles : sequence of Path
        Model ``.npz`` files (``retrieval.save_planet_model`` output), one job
        each.
    extra_args : sequence of str
        Forwarded verbatim to every ``run_starships_logl_grid`` call.
    args : argparse.Namespace
        Parsed shared SLURM options.

    Returns
    -------
    list of str
        Job IDs in ``specfiles`` order (empty with ``--dry-run``).
    """
    scratch_dir = resolve_scratch_dir(args.scratch_dir)
    logl_exe = console_script_executable('run_starships_logl_grid')

    yaml_copy = make_temp_yaml_copy(config, scratch_dir, tool_name='submit_starships_logl_grid')
    print(f'Input YAML: {config}')
    print(f'  copied to: {yaml_copy}  (this copy is what the jobs read)')

    extra = ' '.join(shlex.quote(a) for a in extra_args)
    job_ids = []
    for specfile in specfiles:
        # One job per model file, named after it so squeue stays readable
        # (logs are named by job ID, see common.log_file_pattern).
        job_name = f'{args.job_name}_{specfile.stem}'
        log_file = log_file_pattern(scratch_dir)
        wrap = f'{logl_exe} --config {yaml_copy} --specfile {specfile}'
        if extra:
            wrap += f' {extra}'
        cmd = build_sbatch_command(args, wrap, log_file, job_name=job_name)

        print(f'\n[{specfile.name}]  log: {log_file}')
        job_id = submit_job(cmd, dry_run=args.dry_run)
        if job_id is not None:
            record_submission(scratch_dir, job_id, cmd, yaml_copy=yaml_copy)
            job_ids.append(job_id)

    if job_ids:
        print(f'\nSubmitted {len(job_ids)} job(s): {", ".join(job_ids)}')
    return job_ids


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog='submit_starships_logl_grid', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', type=Path, help='Retrieval / logl-grid input YAML file.')
    parser.add_argument('--specfile', type=Path, nargs='+', required=True, metavar='MODEL.npz',
                        help='Model file(s) to compute a grid for, one SLURM job each.')
    add_sbatch_arguments(parser, default_job_name='starships_logl_grid')

    # Everything after a lone "--" is forwarded verbatim to run_starships_logl_grid.
    # Split by hand rather than with argparse.REMAINDER: a REMAINDER positional
    # placed after `config` would also swallow this tool's own --options.
    argv = sys.argv[1:] if argv is None else list(argv)
    if '--' in argv:
        split_at = argv.index('--')
        argv, extra_args = argv[:split_at], argv[split_at + 1:]
    else:
        extra_args = []
    args = parser.parse_args(argv)

    config = args.config.expanduser().resolve()
    if not config.is_file():
        parser.error(f'config file not found: {config}')

    specfiles = [p.expanduser().resolve() for p in args.specfile]
    missing = [p for p in specfiles if not p.is_file()]
    if missing:
        parser.error('model file(s) not found: ' + ', '.join(str(p) for p in missing))

    submit_logl_grids(config, specfiles, extra_args, args)


if __name__ == '__main__':
    main()
