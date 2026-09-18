"""Submit a STARSHIPS stellar spectrum fit (``starships.stellar_fit``) to SLURM.

Console script: ``submit_starships_stellar_fit``. Python replacement for the
bash launcher ``run_sbatch_fit_stellar_spectrum.sh`` -- same two-step
workflow, with the PHOENIX download step made explicit instead of relying on
"run the script by hand = download, run it through sbatch = fit":

1. **Download the PHOENIX model files** the fit needs, *on the login node*
   (compute nodes have no internet access; the fit would fail there on the
   first missing file). Done by default, right before submitting, by calling
   ``starships.stellar_fit.download_phoenix_files`` on the YAML. It is
   idempotent -- files already present are not downloaded again -- so leaving
   it on costs nothing after the first time. ``--no-download`` skips it,
   ``--download-only`` does only this step and does not submit.
2. **Submit the fit** (``run_starships_stellar_fit <yaml copy>``) with the
   requested SLURM resources.

Usage::

    submit_starships_stellar_fit CONFIG.yaml \\
        --account ACCT --cpus-per-task N --mem MEM --time TIME [options]

The YAML is copied to scratch before submission and the copy is what the job
reads, so the original can be edited right away for the next run (see
:mod:`starships.slurm.common`).

Note on parallelism: ``stellar_fit`` sizes its dynesty worker pool from the
YAML's ``n_workers`` key, NOT from ``$SLURM_CPUS_PER_TASK`` (unlike
``retrieval.py``/``logl_grid.py``). This tool warns when the two disagree,
since asking for 8 cores and running 1 worker (or the reverse) is a common
and silent waste.

Example::

    submit_starships_stellar_fit stellar_fit_inputs_kelt20.yaml \\
        --account def-dlafre --cpus-per-task 8 --mem 20G --time 08:00:00 --job-name fit_star
"""
import argparse
from pathlib import Path
from typing import Optional

import yaml

from starships.slurm.common import (
    add_sbatch_arguments, build_sbatch_command, console_script_executable, log_file_pattern,
    make_temp_yaml_copy, record_submission, resolve_scratch_dir, submit_job,
)


def download_phoenix(yaml_file: Path) -> None:
    """Run the PHOENIX download step for ``yaml_file`` (login node only).

    Imported lazily: ``starships.stellar_fit`` pulls in the whole modelling
    stack, which ``--help`` or ``--no-download`` should not have to pay for.
    """
    from starships.stellar_fit import download_phoenix_files
    print('Downloading PHOENIX model files needed by this config (skips files already '
          'present)...')
    download_phoenix_files(str(yaml_file))


def warn_if_workers_mismatch(yaml_params: dict, cpus_per_task: int) -> Optional[str]:
    """Return a warning string if the YAML's ``n_workers`` does not match the
    requested ``--cpus-per-task`` (None if consistent or not set)."""
    n_workers = yaml_params.get('n_workers')
    if n_workers is None:
        return None
    try:
        n_workers = int(n_workers)
    except (TypeError, ValueError):
        return None
    if n_workers != cpus_per_task:
        return (f'WARNING: YAML n_workers={n_workers} but --cpus-per-task={cpus_per_task}. '
                f'stellar_fit uses n_workers from the YAML for its worker pool, so the job '
                f'will {"leave cores idle" if n_workers < cpus_per_task else "oversubscribe"} '
                f'-- set both to the same value.')
    return None


def submit_stellar_fit(config: Path, args: argparse.Namespace) -> Optional[str]:
    """Copy the YAML, (optionally) download PHOENIX files, build and submit
    the ``sbatch`` command.

    Parameters
    ----------
    config : Path
        The user's stellar-fit YAML (original; the copy is made here).
    args : argparse.Namespace
        Parsed options (shared SLURM options + this tool's download flags).

    Returns
    -------
    str or None
        Job ID, or None with ``--dry-run`` / ``--download-only``.
    """
    scratch_dir = resolve_scratch_dir(args.scratch_dir)
    fit_exe = console_script_executable('run_starships_stellar_fit')

    yaml_copy = make_temp_yaml_copy(config, scratch_dir, tool_name='submit_starships_stellar_fit')
    print(f'Input YAML: {config}')
    print(f'  copied to: {yaml_copy}  (this copy is what the job reads)')

    # The download reads the *copy* too: it is the exact config the job will
    # run, so the set of files it needs is exactly what gets downloaded.
    if not args.no_download:
        download_phoenix(yaml_copy)
    if args.download_only:
        print('--download-only: not submitting.')
        return None

    log_file = log_file_pattern(scratch_dir)
    wrap = f'{fit_exe} {yaml_copy}'
    cmd = build_sbatch_command(args, wrap, log_file)
    print(f'Log file: {log_file}')

    job_id = submit_job(cmd, dry_run=args.dry_run)
    if job_id is not None:
        record_submission(scratch_dir, job_id, cmd, yaml_copy=yaml_copy)
    return job_id


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='submit_starships_stellar_fit', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', type=Path, help='Stellar fit input YAML file.')
    parser.add_argument('--no-download', action='store_true',
                        help='Skip the PHOENIX download step (files already present, or you '
                             'ran --download-only earlier).')
    parser.add_argument('--download-only', action='store_true',
                        help='Only download the PHOENIX files this config needs, do not submit.')
    add_sbatch_arguments(parser, default_job_name='starships_stellar_fit')
    args = parser.parse_args()

    if args.no_download and args.download_only:
        parser.error('--no-download and --download-only contradict each other.')

    config = args.config.expanduser().resolve()
    if not config.is_file():
        parser.error(f'config file not found: {config}')
    with open(config) as f:
        yaml_params = yaml.load(f, Loader=yaml.FullLoader) or {}

    warning = warn_if_workers_mismatch(yaml_params, args.cpus_per_task)
    if warning:
        print(warning)

    submit_stellar_fit(config, args)


if __name__ == '__main__':
    main()
