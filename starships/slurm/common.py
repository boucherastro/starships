"""Shared building blocks for the ``submit_starships_*`` console scripts.

These are the pieces that every "submit a STARSHIPS job to SLURM" tool in
:mod:`starships.slurm` needs, factored out once so that the per-tool modules
(:mod:`starships.slurm.submit_retrieval`,
:mod:`starships.slurm.submit_stellar_fit`,
:mod:`starships.slurm.submit_logl_grid`) only contain what is specific to
their workload. Nothing here is science code -- it is purely the plumbing
between the command line and ``sbatch``.

Design choices (all inherited from the earlier ``run_starships_resource_sweep``
tool, see :mod:`starships.slurm.submit`, and from Antoine's bash launchers
``send_job_retrieval.sh``/``run_sbatch_*.sh`` that these tools replace):

- **No ``#SBATCH`` header files, no bash script to copy.** Every SLURM
  resource (``--account``, ``--cpus-per-task``, ``--mem``, ``--time``, ...)
  is a plain command-line option of the ``submit_starships_*`` command, and
  the job's command is passed to ``sbatch --wrap``. So a student with
  ``starships`` installed can submit a job with nothing but the installed
  command -- there is no template script to find, copy and edit first.
- **The input YAML is copied before submission** (:func:`make_temp_yaml_copy`)
  and the *copy* is what the job reads. A job can sit in the queue for
  hours; without the copy, editing the YAML to prepare the *next* run would
  silently change what the *queued* run does once it starts. The copy is
  timestamped and kept in a scratch directory, so it also serves as a record
  of exactly what each job was given.
- **No venv activation anywhere.** The job calls the target console script
  by its absolute path inside the *currently active* Python environment
  (:func:`console_script_executable`), so whichever environment you launch
  the submit command from is the one used inside the job, with nothing
  hard-coded. See ``README.md`` in this directory for the full rationale.
- **``--dry-run``** everywhere: builds everything (including the YAML copy)
  and prints the exact ``sbatch`` command instead of running it. Use it to
  check a submission before spending queue time, or to see what the tool
  does without a cluster at all.
"""
import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Union

from starships.config import get_scratch_dir

# Sub-directories created inside the scratch directory (see resolve_scratch_dir).
TEMP_YAML_SUBDIR = 'temp_yaml_files'   # timestamped YAML copies read by the jobs
LOG_SUBDIR = 'sbatch_outputs'          # SLURM stdout/stderr logs
SUBMISSIONS_LOG_NAME = 'submissions.log'  # one line per submission, for traceability


# ---------------------------------------------------------------------------
# Locating things
# ---------------------------------------------------------------------------

def console_script_executable(name: str) -> Path:
    """Absolute path to a console script installed in the currently-active
    Python environment (e.g. ``run_starships_retrieval``).

    This is what makes the whole toolset venv-agnostic: instead of
    ``source``-ing a particular venv's ``activate`` inside the job, the job
    calls the script by its absolute path, resolved *here*, at submission
    time, from the environment the submit command itself runs in.

    Uses ``sys.prefix`` (the active environment's root), not
    ``sys.executable``'s own directory -- some Python installs (e.g. certain
    macOS venvs wrapping a Framework build) have ``sys.executable`` be a
    symlink whose *resolved* target lives outside the venv's ``bin/``
    entirely, which would silently look in the wrong place. ``sys.prefix`` is
    exactly what pip itself used to decide where to install console scripts,
    so it is the robust way to find them again.

    Parameters
    ----------
    name : str
        Console script name as registered in ``setup.py``.

    Returns
    -------
    Path
        Absolute path to the executable.

    Raises
    ------
    FileNotFoundError
        If the script is not installed in the active environment.
    """
    candidate = Path(sys.prefix) / 'bin' / name
    if not candidate.exists():
        raise FileNotFoundError(
            f'Could not find {name} in {Path(sys.prefix) / "bin"} -- is starships installed '
            f'in this environment (pip install -e . or pip install starships)?')
    return candidate


def resolve_scratch_dir(explicit: Optional[Union[str, Path]] = None) -> Path:
    """Directory where temporary YAML copies and SLURM logs go.

    Resolution order (first one found wins):

    1. ``explicit`` -- a directory given on the command line (``--scratch-dir``).
    2. ``scratch_dir`` from the STARSHIPS config (``~/.starships/config.yaml``,
       see :mod:`starships.config`).
    3. The ``$SCRATCH`` environment variable (set on every Digital Research
       Alliance of Canada cluster, e.g. Narval).

    Note that a plain system temp directory (``/tmp``) is deliberately NOT a
    fallback: the temporary YAML copy is read by the job on a *compute* node,
    so it has to live on a filesystem shared between login and compute nodes
    -- ``/tmp`` on the login node is not.

    Parameters
    ----------
    explicit : str or Path, optional
        Directory given explicitly by the user; takes precedence over
        everything else.

    Returns
    -------
    Path
        The scratch directory, ``~`` expanded and made absolute. Not created
        here (callers create the sub-directories they need).

    Raises
    ------
    ValueError
        If nothing is set anywhere.
    """
    import os

    if explicit is not None:
        return Path(explicit).expanduser().resolve()

    from_config = get_scratch_dir()
    if from_config is not None:
        return from_config.expanduser().resolve()

    from_env = os.environ.get('SCRATCH')
    if from_env:
        return Path(from_env).expanduser().resolve()

    raise ValueError(
        'No scratch directory found. Either pass --scratch-dir, set scratch_dir in '
        '~/.starships/config.yaml (starships.config.edit_config(scratch_dir=...)), or '
        'make sure the $SCRATCH environment variable is set (it is on Narval).')


# ---------------------------------------------------------------------------
# Temporary YAML copy
# ---------------------------------------------------------------------------

def make_temp_yaml_copy(yaml_file: Union[str, Path], scratch_dir: Union[str, Path],
                        tool_name: str = 'starships.slurm') -> Path:
    """Copy ``yaml_file`` to a timestamped file under
    ``<scratch_dir>/temp_yaml_files/`` and return the copy's path.

    The copy is what the submitted job reads (see module docstring for why).
    A short comment header is prepended recording where the copy came from
    and when -- YAML comments are ignored by every loader ``starships`` uses,
    so the job sees exactly the original content. (``retrieval.py`` re-saves
    its own versioned YAML alongside the run's outputs; that re-dump drops
    comments, which is fine -- this header is only for someone browsing the
    temp directory later.)

    Naming: ``<original stem>_<YYYYmmdd_HHMMSS>.yaml``. If two submissions
    happen within the same second (e.g. a loop), a ``_1``, ``_2``, ... suffix
    keeps them distinct instead of overwriting.

    Parameters
    ----------
    yaml_file : str or Path
        The user's input YAML (retrieval, stellar fit or logl grid config).
    scratch_dir : str or Path
        Root scratch directory (see :func:`resolve_scratch_dir`); the
        ``temp_yaml_files/`` sub-directory is created inside it if needed.
    tool_name : str, optional
        Name written in the header comment, so the copy says which command
        made it.

    Returns
    -------
    Path
        Absolute path to the new copy.
    """
    yaml_file = Path(yaml_file).expanduser().resolve()
    if not yaml_file.is_file():
        raise FileNotFoundError(f'Input YAML file not found: {yaml_file}')

    temp_dir = Path(scratch_dir) / TEMP_YAML_SUBDIR
    temp_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    copy_path = temp_dir / f'{yaml_file.stem}_{timestamp}{yaml_file.suffix}'
    # Same-second collision guard (see docstring).
    counter = 1
    while copy_path.exists():
        copy_path = temp_dir / f'{yaml_file.stem}_{timestamp}_{counter}{yaml_file.suffix}'
        counter += 1

    header = (f'# Temporary copy made by {tool_name} on {now.isoformat(timespec="seconds")}\n'
              f'# Original file: {yaml_file}\n'
              f'# This is the file actually read by the submitted SLURM job -- edit the\n'
              f'# original freely, it will not affect the queued job.\n')
    copy_path.write_text(header + yaml_file.read_text())
    return copy_path


# ---------------------------------------------------------------------------
# Command-line options shared by every submit tool
# ---------------------------------------------------------------------------

def add_sbatch_arguments(parser: argparse.ArgumentParser, default_job_name: str) -> None:
    """Add the SLURM resource options shared by every ``submit_starships_*``
    command to ``parser``.

    These replace, one for one, the ``#SBATCH`` header lines (or the
    ``sbatch_args=(...)`` "EDIT THIS SECTION" block) of the bash launchers
    this toolset replaces. ``--account``, ``--cpus-per-task``, ``--mem`` and
    ``--time`` are *required*: there is no sensible default for a real run,
    and silently falling back on the cluster's defaults (e.g. a 1 h time
    limit for a 3-day retrieval) is exactly the kind of mistake this tool
    should prevent.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The tool's parser, modified in place.
    default_job_name : str
        Default for ``--job-name`` (each tool has its own).
    """
    group = parser.add_argument_group(
        'SLURM resources',
        'Replace the #SBATCH header of a bash launcher. --nodes=1 and --mail-type are '
        'always set; anything else can be passed through with --sbatch-arg.')
    group.add_argument('--account', required=True,
                       help='SLURM --account (required, e.g. def-dlafre).')
    group.add_argument('--cpus-per-task', type=int, required=True,
                       help='SLURM --cpus-per-task (required). The STARSHIPS entry points '
                            'read $SLURM_CPUS_PER_TASK themselves to size their worker pool.')
    group.add_argument('--mem', required=True,
                       help='SLURM --mem for the job (required, e.g. 60G).')
    group.add_argument('--time', dest='time_limit', required=True,
                       help='SLURM --time (required, e.g. 1-12:00:00 or 08:00:00).')
    group.add_argument('--job-name', default=default_job_name,
                       help=f'SLURM --job-name, what squeue shows (default: {default_job_name}).')
    group.add_argument('--mail-type', default='FAIL',
                       help='SLURM --mail-type (default: FAIL).')
    group.add_argument('--sbatch-arg', action='append', default=[], metavar='ARG',
                       help='Extra raw sbatch option, repeatable. Because the value itself '
                            'starts with "--", write it with "=": --sbatch-arg=--mail-user=me@x.ca')
    group.add_argument('--pre-cmd', default=None, metavar='CMD',
                       help='Shell command run inside the job before the STARSHIPS command '
                            '(e.g. "module load StdEnv/2020"). Not needed when the submitting '
                            'shell already has the right environment: sbatch exports it to the '
                            'job by default.')

    paths = parser.add_argument_group(
        'Where files go',
        'Temporary YAML copies go to <scratch>/temp_yaml_files/, SLURM logs to '
        '<scratch>/sbatch_outputs/out_sbatch_<jobid>.txt. <scratch> is --scratch-dir, else scratch_dir from '
        '~/.starships/config.yaml, else $SCRATCH.')
    paths.add_argument('--scratch-dir', default=None,
                       help='Override the scratch directory (see above).')
    paths.add_argument('--dry-run', action='store_true',
                       help='Do everything except calling sbatch: make the YAML copy and print '
                            'the exact sbatch command that would be run.')


# ---------------------------------------------------------------------------
# Building and running the sbatch command
# ---------------------------------------------------------------------------

def build_sbatch_command(args: argparse.Namespace, wrap_command: str, log_file: Path,
                         job_name: Optional[str] = None,
                         array: Optional[str] = None) -> List[str]:
    """Assemble the full ``sbatch`` command line from the parsed shared
    options plus the job's own command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed options from a parser that went through
        :func:`add_sbatch_arguments`.
    wrap_command : str
        The command the job runs (goes to ``sbatch --wrap``). If
        ``args.pre_cmd`` is set it is prepended with ``&&``.
    log_file : Path
        SLURM ``--output`` pattern (may contain ``%j`` / ``%A_%a``).
    job_name : str, optional
        Overrides ``args.job_name`` (used when one submission produces several
        jobs, e.g. one per model file, each with a distinct name).
    array : str, optional
        SLURM ``--array`` spec (e.g. ``"1-8"``); omitted when None.

    Returns
    -------
    list of str
        Ready for ``subprocess.run``. ``--parsable`` is always included so
        that the job ID comes back cleanly on stdout.
    """
    if args.pre_cmd:
        wrap_command = f'{args.pre_cmd} && {wrap_command}'

    cmd = [
        'sbatch', '--parsable',
        f'--account={args.account}',
        '--nodes=1',
        f'--cpus-per-task={args.cpus_per_task}',
        f'--mem={args.mem}',
        f'--time={args.time_limit}',
        f'--job-name={job_name or args.job_name}',
        f'--output={log_file}',
        f'--mail-type={args.mail_type}',
    ]
    if array is not None:
        cmd.append(f'--array={array}')
    cmd.extend(args.sbatch_arg)
    cmd.append(f'--wrap={wrap_command}')
    return cmd


def log_file_pattern(scratch_dir: Path, array: bool = False) -> Path:
    """SLURM ``--output`` pattern for a job: ``<scratch>/sbatch_outputs/out_sbatch_%j.txt``
    (or ``out_sbatch_%A_%a.txt`` for an array job, so each task gets its own
    log). Creates the log directory if needed.

    Named after the job ID only, never the job name (Antoine, 2026-09-18): a
    job ID is unique and is what ``squeue``/``sacct``/``submissions.log`` show,
    so a log is easy to find from it; a job name is not guaranteed distinct
    between submissions. Same convention as the bash launchers this replaces.
    """
    log_dir = scratch_dir / LOG_SUBDIR
    log_dir.mkdir(parents=True, exist_ok=True)
    suffix = '%A_%a' if array else '%j'
    return log_dir / f'out_sbatch_{suffix}.txt'


def submit_job(cmd: Sequence[str], dry_run: bool = False) -> Optional[str]:
    """Print and run an ``sbatch`` command, returning the job ID.

    Parameters
    ----------
    cmd : sequence of str
        From :func:`build_sbatch_command`.
    dry_run : bool, optional
        If True, only print the command and return None.

    Returns
    -------
    str or None
        The job ID printed by ``sbatch --parsable`` (for an array job, the
        array's master ID), or None in dry-run mode.
    """
    # shlex.join quotes the --wrap=... part properly, so the printed line can
    # be copy-pasted into a shell as-is.
    print('sbatch command:')
    print('  ' + shlex.join(cmd))
    if dry_run:
        print('  (dry run: not submitted)')
        return None
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    job_id = result.stdout.strip()
    print(f'Submitted job {job_id}')
    return job_id


def record_submission(scratch_dir: Path, job_id: str, cmd: Sequence[str],
                      yaml_copy: Optional[Path] = None) -> None:
    """Append one line describing a submission to
    ``<scratch>/sbatch_outputs/submissions.log``.

    A plain text, append-only record of *which* YAML copy and command each
    job ID corresponds to. Handy weeks later when only a job ID or a log file
    is left and the question is "what exactly did that run use?".
    """
    log_dir = scratch_dir / LOG_SUBDIR
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().isoformat(timespec='seconds')
    yaml_note = f' yaml={yaml_copy}' if yaml_copy is not None else ''
    with open(log_dir / SUBMISSIONS_LOG_NAME, 'a') as f:
        f.write(f'{stamp} job={job_id}{yaml_note} cmd={shlex.join(cmd)}\n')
