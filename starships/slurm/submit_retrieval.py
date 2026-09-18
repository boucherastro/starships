"""Submit a STARSHIPS retrieval (``run_starships_retrieval``) to SLURM.

Console script: ``submit_starships_retrieval``. Python replacement for the
bash pair ``send_job_retrieval.sh`` + ``run_retrieval.sh`` -- same workflow,
but with nothing to copy or edit: every SLURM resource is a command-line
option, the YAML is copied to scratch before submission (so it can be edited
right away without touching the queued job), and the job runs the installed
``run_starships_retrieval`` from the current Python environment.

Usage::

    submit_starships_retrieval MODE CONFIG.yaml [key=value ...] \\
        --account ACCT --cpus-per-task N --mem MEM --time TIME [options]

``MODE`` is ``burnin`` or ``sampling``. It does not change what
``run_starships_retrieval`` receives (``retrieval.py`` decides the phase by
itself from ``walker_file_in``: None means burnin, otherwise sampling) -- it
selects the SLURM shape of the submission and lets this tool check that the
YAML + overrides are consistent with what you asked for, *before* the job
waits in the queue only to crash on startup:

- ``burnin``: ``walker_file_in`` must be null. With ``--n-burnin-runs N`` > 1
  the job is submitted as a SLURM array of N *independent burnin retrievals*
  -- literally N separate ``run_starships_retrieval`` runs of the same YAML,
  each with its own walkers, over the (few) ``n_steps_burnin`` steps. Together
  they act as one very large burnin, exploring N times more of the parameter
  space than a single run would; the sampling phase then starts from the
  best of all of them (``init_mode: from_burnin``). This is the
  ``slurm_array_behaviour: 'burnin'`` mechanism of ``retrieval.py``, which
  appends the array task index to each run's ``walker_file_out``; the YAML
  must have it set. (Not to be confused with the walkers/chains *inside* one
  retrieval -- every one of these N runs has its full set of walkers.)
- ``sampling``: ``walker_file_in`` must be set (the burnin chain to start
  from), typically together with ``init_mode``. Single job, no array.

Any ``key=value`` positional token is forwarded verbatim to
``run_starships_retrieval``, which uses it to override that key of the YAML
(``retrieval.main()`` reads them from the command line and converts the
types it knows about, see ``retrieval_inputs.convert_cmd_line_to_types``).
The same key=value tokens are also used *here* for the consistency checks
above, so overriding on the command line is a perfectly good way to switch
phase without editing the YAML.

Examples::

    # 8 independent burnin retrievals (one SLURM array), 1.5 days each
    submit_starships_retrieval burnin retrieval_inputs_kelt20.yaml --n-burnin-runs 8 \\
        --account def-dlafre --cpus-per-task 24 --mem 100G --time 1-12:00:00 \\
        --job-name K20_burnin

    # Sampling, started from one of those burnin runs
    submit_starships_retrieval sampling retrieval_inputs_kelt20.yaml \\
        walker_file_in=K20_burnin_3.h5 init_mode=from_burnin \\
        --account def-dlafre --cpus-per-task 24 --mem 100G --time 3-00:00:00

    # See what would be submitted, without submitting
    submit_starships_retrieval burnin cfg.yaml --n-burnin-runs 8 --account x \\
        --cpus-per-task 24 --mem 100G --time 1-00:00:00 --dry-run

Sizing ``--cpus-per-task``/``--mem``/``--time`` is what
``run_starships_resource_sweep`` (:mod:`starships.slurm.submit`) is for --
see the README in this directory.
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from starships.slurm.common import (
    add_sbatch_arguments, build_sbatch_command, console_script_executable, log_file_pattern,
    make_temp_yaml_copy, record_submission, resolve_scratch_dir, submit_job,
)

MODES = ('burnin', 'sampling')
# Strings that the command line / YAML may use to mean "no value" -- mirrors
# what retrieval_inputs.convert_cmd_line_to_types accepts.
_NULL_STRINGS = ('null', 'none', '')


def parse_overrides(tokens: List[str]) -> Dict[str, str]:
    """Turn ``['a=1', 'b=x']`` into ``{'a': '1', 'b': 'x'}``.

    Values are kept as strings: ``run_starships_retrieval`` does its own type
    conversion. Only the split is done here, plus a clear error for a token
    without ``=`` (a very easy typo to make on a long command line).
    """
    overrides = {}
    for tok in tokens:
        if '=' not in tok:
            raise ValueError(f'{tok!r} is not a key=value override')
        key, value = tok.split('=', 1)
        if not key:
            raise ValueError(f'{tok!r}: empty key')
        overrides[key] = value
    return overrides


def _is_null(value) -> bool:
    """True for YAML ``null`` (Python None) and for the strings the command
    line uses to mean the same thing."""
    return value is None or (isinstance(value, str) and value.strip().lower() in _NULL_STRINGS)


def check_mode_consistency(mode: str, yaml_params: dict, overrides: Dict[str, str],
                           n_burnin_runs: int) -> List[str]:
    """Check that the YAML (+ command-line overrides) matches the requested
    ``mode`` and return a list of problems (empty if all is well).

    Done at submission time, on purpose: the same mistakes made inside the
    job would only show up after the job started, i.e. after the wait in the
    queue, and ``retrieval.py``'s own checks are not all as explicit as these.

    Parameters
    ----------
    mode : {'burnin', 'sampling'}
    yaml_params : dict
        The loaded YAML.
    overrides : dict
        Command-line ``key=value`` overrides (take precedence over the YAML).
    n_burnin_runs : int
        Number of independent burnin retrievals requested (``--n-burnin-runs``).

    Returns
    -------
    list of str
        Human-readable problems. Empty means consistent.
    """
    def effective(key):
        return overrides[key] if key in overrides else yaml_params.get(key)

    problems = []
    walker_file_in = effective('walker_file_in')

    if mode == 'burnin':
        if not _is_null(walker_file_in):
            problems.append(
                f"mode is 'burnin' but walker_file_in={walker_file_in!r}: retrieval.py treats "
                f"any non-null walker_file_in as the sampling phase (n_steps_sampling, walkers "
                f"initialised from that file). Use mode 'sampling', or override "
                f"walker_file_in=null.")
        if n_burnin_runs > 1 and effective('slurm_array_behaviour') != 'burnin':
            problems.append(
                f"--n-burnin-runs {n_burnin_runs} submits a SLURM array, which retrieval.py only "
                f"accepts with slurm_array_behaviour: 'burnin' (it raises otherwise, and would not "
                f"give each run its own walker_file_out). Set it in the YAML or override "
                f"slurm_array_behaviour=burnin.")
    else:  # sampling
        if _is_null(walker_file_in):
            problems.append(
                "mode is 'sampling' but walker_file_in is null: the run would start from "
                "scratch (burnin phase, n_steps_burnin). Give the burnin run to start from, "
                "e.g. walker_file_in=<file>.h5 init_mode=from_burnin, or use mode 'burnin'.")
    return problems


def submit_retrieval(mode: str, config: Path, overrides: Dict[str, str], n_burnin_runs: int,
                     args: argparse.Namespace) -> Optional[str]:
    """Copy the YAML, build the ``sbatch`` command and submit it.

    Parameters
    ----------
    mode : {'burnin', 'sampling'}
    config : Path
        The user's retrieval YAML (the *original*; the copy is made here).
    overrides : dict
        ``key=value`` overrides forwarded to ``run_starships_retrieval``.
    n_burnin_runs : int
        Independent burnin retrievals (SLURM array size); 1 means a plain
        single job.
    args : argparse.Namespace
        Parsed shared SLURM options (see ``common.add_sbatch_arguments``).

    Returns
    -------
    str or None
        Job ID, or None with ``--dry-run``.
    """
    scratch_dir = resolve_scratch_dir(args.scratch_dir)
    retrieval_exe = console_script_executable('run_starships_retrieval')

    yaml_copy = make_temp_yaml_copy(config, scratch_dir, tool_name='submit_starships_retrieval')
    print(f'Input YAML: {config}')
    print(f'  copied to: {yaml_copy}  (this copy is what the job reads)')

    use_array = mode == 'burnin' and n_burnin_runs > 1
    array_spec = f'1-{n_burnin_runs}' if use_array else None
    log_file = log_file_pattern(scratch_dir, array=use_array)

    # retrieval.main() reads yaml_file=... and any other key=value from argv.
    override_str = ' '.join(f'{k}={v}' for k, v in overrides.items())
    wrap = f'{retrieval_exe} yaml_file={yaml_copy}'
    if override_str:
        wrap += f' {override_str}'

    cmd = build_sbatch_command(args, wrap, log_file, array=array_spec)
    if use_array:
        print(f'Mode: burnin, {n_burnin_runs} independent burnin retrievals '
              f'(SLURM array {array_spec}).')
    else:
        print(f'Mode: {mode}, single job.')
    print(f'Log file(s): {log_file}')

    job_id = submit_job(cmd, dry_run=args.dry_run)
    if job_id is not None:
        record_submission(scratch_dir, job_id, cmd, yaml_copy=yaml_copy)
    return job_id


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='submit_starships_retrieval', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=MODES,
                        help="'burnin' (walker_file_in null, optional --n-burnin-runs array) or "
                             "'sampling' (walker_file_in set).")
    parser.add_argument('config', type=Path, help='Retrieval input YAML file.')
    parser.add_argument('overrides', nargs='*', metavar='key=value',
                        help='Overrides forwarded verbatim to run_starships_retrieval '
                             '(e.g. walker_file_in=burnin_3.h5 init_mode=from_burnin).')
    parser.add_argument('--n-burnin-runs', type=int, default=1,
                        help='burnin only: number of independent burnin retrievals (each a full '
                             'run_starships_retrieval with its own walkers), submitted as a '
                             'SLURM array 1-N to act as one very large burnin (default: 1, '
                             'plain single job).')
    add_sbatch_arguments(parser, default_job_name='starships_retrieval')
    # Intermixed: lets key=value tokens and --options come in any order.
    args = parser.parse_intermixed_args()

    if args.n_burnin_runs < 1:
        parser.error('--n-burnin-runs must be >= 1')
    if args.mode == 'sampling' and args.n_burnin_runs > 1:
        parser.error('--n-burnin-runs only applies to mode burnin (sampling is a single job).')

    try:
        overrides = parse_overrides(args.overrides)
    except ValueError as e:
        parser.error(str(e))

    config = args.config.expanduser().resolve()
    if not config.is_file():
        parser.error(f'config file not found: {config}')
    with open(config) as f:
        yaml_params = yaml.load(f, Loader=yaml.FullLoader) or {}

    problems = check_mode_consistency(args.mode, yaml_params, overrides, args.n_burnin_runs)
    if problems:
        parser.error('inconsistent submission:\n  - ' + '\n  - '.join(problems))

    submit_retrieval(args.mode, config, overrides, args.n_burnin_runs, args)


if __name__ == '__main__':
    main()
