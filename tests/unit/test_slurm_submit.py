"""``starships.slurm`` job launchers (``submit_starships_*``) -- pure plumbing
tests, no SLURM, no petitRADTRANS.

``sbatch`` is never called: ``subprocess.run`` in ``starships.slurm.common``
is replaced by a fake that records the command and returns a job ID, and the
console-script lookup is pointed at a fake path so the tests do not depend
on ``starships`` being pip-installed in the running environment.
"""
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import starships.slurm.common as common
import starships.slurm.submit_logl_grid as submit_logl_grid
import starships.slurm.submit_retrieval as submit_retrieval
import starships.slurm.submit_stellar_fit as submit_stellar_fit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_sbatch(monkeypatch):
    """Replace subprocess.run with a recorder returning incrementing job IDs."""
    calls = []

    def fake_run(cmd, capture_output, text, check):
        calls.append(list(cmd))
        return SimpleNamespace(stdout=f'{1000 + len(calls)}\n', returncode=0)

    monkeypatch.setattr(common.subprocess, 'run', fake_run)
    return calls


@pytest.fixture
def fake_exe(monkeypatch):
    """Point the console-script lookup at a fake absolute path."""
    monkeypatch.setattr(common, 'console_script_executable',
                        lambda name: Path('/fake/venv/bin') / name)
    # The submit_* modules imported the name directly; patch there too.
    for mod in (submit_retrieval, submit_stellar_fit, submit_logl_grid):
        monkeypatch.setattr(mod, 'console_script_executable',
                            lambda name: Path('/fake/venv/bin') / name)


@pytest.fixture
def scratch(tmp_path):
    d = tmp_path / 'scratch'
    d.mkdir()
    return d


@pytest.fixture
def retrieval_yaml(tmp_path):
    cfg = {'walker_file_in': None, 'init_mode': 'from_burnin',
           'slurm_array_behaviour': 'burnin', 'n_steps_burnin': 300, 'n_cpu': None}
    p = tmp_path / 'retrieval_inputs_test.yaml'
    p.write_text(yaml.dump(cfg))
    return p


def _wrap_of(cmd):
    """The --wrap=... payload of a recorded sbatch command."""
    return next(a for a in cmd if a.startswith('--wrap='))[len('--wrap='):]


def _opt(cmd, name):
    """Value of a --name=value option in a recorded sbatch command."""
    return next(a for a in cmd if a.startswith(f'--{name}='))[len(f'--{name}='):]


# ---------------------------------------------------------------------------
# common.py
# ---------------------------------------------------------------------------

class TestResolveScratchDir:
    def test_explicit_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv('SCRATCH', '/env/scratch')
        monkeypatch.setattr(common, 'get_scratch_dir', lambda: Path('/config/scratch'))
        assert common.resolve_scratch_dir(tmp_path) == tmp_path.resolve()

    def test_config_before_env(self, monkeypatch):
        monkeypatch.setenv('SCRATCH', '/env/scratch')
        monkeypatch.setattr(common, 'get_scratch_dir', lambda: Path('/config/scratch'))
        assert common.resolve_scratch_dir() == Path('/config/scratch')

    def test_env_fallback(self, monkeypatch):
        monkeypatch.setenv('SCRATCH', '/env/scratch')
        monkeypatch.setattr(common, 'get_scratch_dir', lambda: None)
        assert common.resolve_scratch_dir() == Path('/env/scratch')

    def test_nothing_set_is_an_error(self, monkeypatch):
        monkeypatch.delenv('SCRATCH', raising=False)
        monkeypatch.setattr(common, 'get_scratch_dir', lambda: None)
        with pytest.raises(ValueError, match='scratch'):
            common.resolve_scratch_dir()


class TestMakeTempYamlCopy:
    def test_copy_is_timestamped_and_loads_identically(self, retrieval_yaml, scratch):
        copy = common.make_temp_yaml_copy(retrieval_yaml, scratch)

        assert copy.parent == scratch / common.TEMP_YAML_SUBDIR
        assert copy.name.startswith('retrieval_inputs_test_')
        assert copy.suffix == '.yaml'
        # Header comment present, content unchanged for a YAML loader.
        text = copy.read_text()
        assert text.startswith('# Temporary copy made by')
        assert str(retrieval_yaml.resolve()) in text
        assert yaml.safe_load(text) == yaml.safe_load(retrieval_yaml.read_text())

    def test_same_second_copies_do_not_collide(self, retrieval_yaml, scratch):
        first = common.make_temp_yaml_copy(retrieval_yaml, scratch)
        second = common.make_temp_yaml_copy(retrieval_yaml, scratch)
        assert first != second
        assert first.exists() and second.exists()

    def test_missing_input_raises(self, scratch, tmp_path):
        with pytest.raises(FileNotFoundError):
            common.make_temp_yaml_copy(tmp_path / 'nope.yaml', scratch)


class TestBuildSbatchCommand:
    def _args(self, **over):
        base = dict(account='def-x', cpus_per_task=4, mem='20G', time_limit='01:00:00',
                    job_name='job', mail_type='FAIL', sbatch_arg=[], pre_cmd=None)
        base.update(over)
        return SimpleNamespace(**base)

    def test_basic_shape(self):
        cmd = common.build_sbatch_command(self._args(), 'echo hi', Path('/logs/job_%j.txt'))
        assert cmd[:2] == ['sbatch', '--parsable']
        assert '--nodes=1' in cmd
        assert _opt(cmd, 'cpus-per-task') == '4'
        assert _opt(cmd, 'output') == '/logs/job_%j.txt'
        assert not any(a.startswith('--array') for a in cmd)
        assert cmd[-1] == '--wrap=echo hi'

    def test_array_extra_args_and_pre_cmd(self):
        args = self._args(sbatch_arg=['--mail-user=me@x.ca'], pre_cmd='module load StdEnv/2020')
        cmd = common.build_sbatch_command(args, 'echo hi', Path('/l/j_%A_%a.txt'),
                                          job_name='other', array='1-8')
        assert _opt(cmd, 'array') == '1-8'
        assert _opt(cmd, 'job-name') == 'other'
        assert '--mail-user=me@x.ca' in cmd
        assert _wrap_of(cmd) == 'module load StdEnv/2020 && echo hi'


class TestSubmitJob:
    def test_returns_job_id(self, fake_sbatch):
        assert common.submit_job(['sbatch', '--wrap=x']) == '1001'
        assert fake_sbatch == [['sbatch', '--wrap=x']]

    def test_dry_run_does_not_call_sbatch(self, fake_sbatch):
        assert common.submit_job(['sbatch', '--wrap=x'], dry_run=True) is None
        assert fake_sbatch == []


# ---------------------------------------------------------------------------
# submit_retrieval.py
# ---------------------------------------------------------------------------

class TestParseOverrides:
    def test_splits_on_first_equal(self):
        assert submit_retrieval.parse_overrides(['a=1', 'b=x=y']) == {'a': '1', 'b': 'x=y'}

    def test_rejects_token_without_equal(self):
        with pytest.raises(ValueError, match='key=value'):
            submit_retrieval.parse_overrides(['oops'])


class TestCheckModeConsistency:
    base = {'walker_file_in': None, 'slurm_array_behaviour': 'burnin'}

    def test_burnin_ok(self):
        assert submit_retrieval.check_mode_consistency('burnin', self.base, {}, 8) == []

    def test_burnin_with_walker_file_in_is_flagged(self):
        problems = submit_retrieval.check_mode_consistency(
            'burnin', {**self.base, 'walker_file_in': 'chain.h5'}, {}, 1)
        assert len(problems) == 1 and 'walker_file_in' in problems[0]

    def test_override_can_fix_it(self):
        problems = submit_retrieval.check_mode_consistency(
            'burnin', {**self.base, 'walker_file_in': 'chain.h5'}, {'walker_file_in': 'null'}, 1)
        assert problems == []

    def test_array_needs_slurm_array_behaviour(self):
        problems = submit_retrieval.check_mode_consistency(
            'burnin', {**self.base, 'slurm_array_behaviour': None}, {}, 8)
        assert len(problems) == 1 and 'slurm_array_behaviour' in problems[0]
        # ...but a single run is fine without it
        assert submit_retrieval.check_mode_consistency(
            'burnin', {**self.base, 'slurm_array_behaviour': None}, {}, 1) == []

    def test_sampling_needs_walker_file_in(self):
        problems = submit_retrieval.check_mode_consistency('sampling', self.base, {}, 1)
        assert len(problems) == 1 and 'walker_file_in' in problems[0]
        assert submit_retrieval.check_mode_consistency(
            'sampling', self.base, {'walker_file_in': 'chain.h5'}, 1) == []


class TestSubmitRetrievalCli:
    def _run(self, monkeypatch, argv):
        monkeypatch.setattr('sys.argv', ['submit_starships_retrieval'] + argv)
        submit_retrieval.main()

    def test_burnin_array_submission(self, monkeypatch, fake_sbatch, fake_exe, retrieval_yaml,
                                     scratch):
        self._run(monkeypatch, ['burnin', str(retrieval_yaml), '--n-burnin-runs', '8',
                                '--account', 'def-x', '--cpus-per-task', '24', '--mem', '100G',
                                '--time', '1-12:00:00', '--job-name', 'K20',
                                '--scratch-dir', str(scratch)])
        assert len(fake_sbatch) == 1
        cmd = fake_sbatch[0]
        assert _opt(cmd, 'array') == '1-8'
        assert _opt(cmd, 'output').endswith('out_sbatch_%A_%a.txt')
        wrap = _wrap_of(cmd)
        assert wrap.startswith('/fake/venv/bin/run_starships_retrieval yaml_file=')
        # The job reads the COPY, not the original.
        yaml_path = Path(wrap.split('yaml_file=')[1].split()[0])
        assert yaml_path.parent == scratch / common.TEMP_YAML_SUBDIR
        assert yaml_path.exists()
        assert yaml_path != retrieval_yaml
        # Submission recorded for traceability.
        log = (scratch / common.LOG_SUBDIR / common.SUBMISSIONS_LOG_NAME).read_text()
        assert 'job=1001' in log and str(yaml_path) in log

    def test_sampling_with_overrides_forwarded(self, monkeypatch, fake_sbatch, fake_exe,
                                               retrieval_yaml, scratch):
        self._run(monkeypatch, ['sampling', str(retrieval_yaml),
                                'walker_file_in=K20_3.h5', 'init_mode=from_burnin',
                                '--account', 'def-x', '--cpus-per-task', '24', '--mem', '100G',
                                '--time', '3-00:00:00', '--scratch-dir', str(scratch)])
        cmd = fake_sbatch[0]
        assert not any(a.startswith('--array') for a in cmd)
        assert _opt(cmd, 'output').endswith('out_sbatch_%j.txt')
        assert _wrap_of(cmd).endswith(' walker_file_in=K20_3.h5 init_mode=from_burnin')

    def test_inconsistent_mode_exits_before_submitting(self, monkeypatch, fake_sbatch, fake_exe,
                                                       retrieval_yaml, scratch):
        with pytest.raises(SystemExit):
            self._run(monkeypatch, ['sampling', str(retrieval_yaml),  # walker_file_in null
                                    '--account', 'def-x', '--cpus-per-task', '1', '--mem', '1G',
                                    '--time', '00:10:00', '--scratch-dir', str(scratch)])
        assert fake_sbatch == []

    def test_dry_run_copies_yaml_but_does_not_submit(self, monkeypatch, fake_sbatch, fake_exe,
                                                     retrieval_yaml, scratch):
        self._run(monkeypatch, ['burnin', str(retrieval_yaml), '--dry-run',
                                '--account', 'def-x', '--cpus-per-task', '1', '--mem', '1G',
                                '--time', '00:10:00', '--scratch-dir', str(scratch)])
        assert fake_sbatch == []
        assert list((scratch / common.TEMP_YAML_SUBDIR).glob('*.yaml'))

    def test_missing_required_resource_is_an_error(self, monkeypatch, fake_sbatch, fake_exe,
                                                   retrieval_yaml, scratch):
        with pytest.raises(SystemExit):  # no --time
            self._run(monkeypatch, ['burnin', str(retrieval_yaml), '--account', 'def-x',
                                    '--cpus-per-task', '1', '--mem', '1G',
                                    '--scratch-dir', str(scratch)])
        assert fake_sbatch == []


# ---------------------------------------------------------------------------
# submit_stellar_fit.py
# ---------------------------------------------------------------------------

class TestSubmitStellarFitCli:
    @pytest.fixture
    def fit_yaml(self, tmp_path):
        p = tmp_path / 'stellar_fit_inputs_test.yaml'
        p.write_text(yaml.dump({'n_workers': 8, 'n_live_points': 100}))
        return p

    @pytest.fixture
    def fake_download(self, monkeypatch):
        calls = []
        monkeypatch.setattr(submit_stellar_fit, 'download_phoenix', lambda y: calls.append(y))
        return calls

    def _run(self, monkeypatch, argv):
        monkeypatch.setattr('sys.argv', ['submit_starships_stellar_fit'] + argv)
        submit_stellar_fit.main()

    def test_download_then_submit(self, monkeypatch, fake_sbatch, fake_exe, fake_download,
                                  fit_yaml, scratch):
        self._run(monkeypatch, [str(fit_yaml), '--account', 'def-x', '--cpus-per-task', '8',
                                '--mem', '20G', '--time', '08:00:00', '--scratch-dir', str(scratch)])
        assert len(fake_download) == 1
        # Download uses the copy (the exact config the job will run).
        assert fake_download[0].parent == scratch / common.TEMP_YAML_SUBDIR
        assert len(fake_sbatch) == 1
        assert _wrap_of(fake_sbatch[0]) == f'/fake/venv/bin/run_starships_stellar_fit {fake_download[0]}'

    def test_no_download(self, monkeypatch, fake_sbatch, fake_exe, fake_download, fit_yaml,
                         scratch):
        self._run(monkeypatch, [str(fit_yaml), '--no-download', '--account', 'def-x',
                                '--cpus-per-task', '8', '--mem', '20G', '--time', '08:00:00',
                                '--scratch-dir', str(scratch)])
        assert fake_download == []
        assert len(fake_sbatch) == 1

    def test_download_only(self, monkeypatch, fake_sbatch, fake_exe, fake_download, fit_yaml,
                           scratch):
        self._run(monkeypatch, [str(fit_yaml), '--download-only', '--account', 'def-x',
                                '--cpus-per-task', '8', '--mem', '20G', '--time', '08:00:00',
                                '--scratch-dir', str(scratch)])
        assert len(fake_download) == 1
        assert fake_sbatch == []

    def test_workers_mismatch_warning(self):
        assert submit_stellar_fit.warn_if_workers_mismatch({'n_workers': 8}, 8) is None
        assert submit_stellar_fit.warn_if_workers_mismatch({}, 8) is None
        warning = submit_stellar_fit.warn_if_workers_mismatch({'n_workers': 1}, 8)
        assert warning and 'n_workers=1' in warning


# ---------------------------------------------------------------------------
# submit_logl_grid.py
# ---------------------------------------------------------------------------

class TestSubmitLoglGridCli:
    def _run(self, monkeypatch, argv):
        monkeypatch.setattr('sys.argv', ['submit_starships_logl_grid'] + argv)
        submit_logl_grid.main()

    def test_one_job_per_specfile_sharing_one_yaml_copy(self, monkeypatch, fake_sbatch, fake_exe,
                                                        retrieval_yaml, scratch, tmp_path):
        specs = [tmp_path / f'model_{s}.npz' for s in ('H2O', 'CO', 'all')]
        for s in specs:
            s.write_bytes(b'')
        self._run(monkeypatch, [str(retrieval_yaml), '--specfile', *map(str, specs),
                                '--account', 'def-x', '--cpus-per-task', '4', '--mem', '20G',
                                '--time', '07:00:00', '--job-name', 'logl',
                                '--scratch-dir', str(scratch),
                                '--', '--plot', 'trailing', '--kp', '227'])
        assert len(fake_sbatch) == 3
        yaml_copies = set()
        for cmd, spec in zip(fake_sbatch, specs):
            assert _opt(cmd, 'job-name') == f'logl_{spec.stem}'
            assert _opt(cmd, 'output').endswith('out_sbatch_%j.txt')
            wrap = _wrap_of(cmd)
            assert wrap.startswith('/fake/venv/bin/run_starships_logl_grid --config ')
            assert f'--specfile {spec.resolve()}' in wrap
            assert wrap.endswith(' --plot trailing --kp 227')
            yaml_copies.add(wrap.split('--config ')[1].split()[0])
        # All three jobs read the same single copy.
        assert len(yaml_copies) == 1
        assert Path(yaml_copies.pop()).parent == scratch / common.TEMP_YAML_SUBDIR

    def test_missing_specfile_is_an_error(self, monkeypatch, fake_sbatch, fake_exe,
                                          retrieval_yaml, scratch, tmp_path):
        with pytest.raises(SystemExit):
            self._run(monkeypatch, [str(retrieval_yaml), '--specfile', str(tmp_path / 'no.npz'),
                                    '--account', 'def-x', '--cpus-per-task', '4', '--mem', '20G',
                                    '--time', '07:00:00', '--scratch-dir', str(scratch)])
        assert fake_sbatch == []
