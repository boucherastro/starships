"""
Regression tests for logL computations.

These tests verify that numerical results are unchanged after non-scientific
code modifications (cleanup, refactoring, dependency updates, etc.).

The tests use calc_logl_injred (via quick_calc_logl_injred_class), which is
the actual science pipeline: it injects a model into the data, accounts for
the planet's orbital motion, removes PCA systematics, and computes the logL.
This produces a peak at V_sys — unlike quick_correl which sums incoherently.

Prerequisites (on Narval):
    1. Set paths in ~/.starships/config.yaml (via starships.config.edit_config)
    2. Copy config_template.yaml → ~/.starships/regression_config.yaml and fill in paths
    3. Run generate_golden.py once to create golden references
    4. Run `pytest tests/regression/ -v` after every code change

Each dataset in regression_config.yaml must have:
    npz_path   : path to the reduced NPZ (passed to load_reduced_sequence)
    n_pc       : number of PCA components to remove (read-time argument since B3)
    model_path : path to the model NPZ (keys: 'wave', 'spec')
    pl_name    : planet name for load_reduced_sequence (e.g. "WASP-33 b")
    kind_trans : 'emission' or 'transmission'

Tests are automatically SKIPPED if the config or golden outputs are missing,
so they do not break local development or CI environments without real data.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

# ---------------------------------------------------------------------------
# Config paths
# ---------------------------------------------------------------------------

_REGRESSION_CONFIG = Path.home() / '.starships' / 'regression_config.yaml'

_config_available = _REGRESSION_CONFIG.exists()


def _golden_dir():
    from starships.config import get_regression_golden_dir
    return get_regression_golden_dir()


def _dataset_names():
    if not _config_available:
        return []
    with open(_REGRESSION_CONFIG) as f:
        cfg = yaml.safe_load(f)
    return list(cfg.get('datasets', {}).keys())


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

requires_regression_data = pytest.mark.skipif(
    not _config_available,
    reason=(
        "Regression config not found. "
        "On Narval: copy config_template.yaml → ~/.starships/regression_config.yaml"
    ),
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def regression_config():
    with open(_REGRESSION_CONFIG) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope='session')
def corrRV(regression_config):
    g = regression_config['rv_grid']
    return np.arange(g['min'], g['max'] + g['step'], g['step'])


def _load_transit(ds_config):
    """Load full Observations object from NPZ via load_reduced_sequence.

    Planet parameters from the retrieval_config YAML override ExoFile defaults,
    matching how parameters are set in the reduction/retrieval pipeline.

    B3: `n_pc` is now a read-time argument (no longer baked into the saved file), so it
    must be given explicitly in `regression_config.yaml` (`n_pc:` per dataset).
    """
    import starships.planet_obs as pl_obs
    from pipeline.reduction import pl_param_units

    path    = Path(ds_config['npz_path']).expanduser()
    pl_name = ds_config['pl_name']
    n_pc    = ds_config['n_pc']
    if not path.exists():
        pytest.skip(f"Reduced data not found: {path}")

    pl_kwargs = {}
    ret_cfg_key = ds_config.get('retrieval_config')
    if ret_cfg_key:
        ret_cfg_path = Path(ret_cfg_key).expanduser()
        if ret_cfg_path.exists():
            with open(ret_cfg_path) as f:
                ret_cfg = yaml.safe_load(f)
            if ret_cfg.get('pl_params'):
                pl_kwargs = pl_param_units(ret_cfg)
        else:
            print(f"  Warning: retrieval_config not found: {ret_cfg_path}")

    return pl_obs.load_reduced_sequence(path, n_pc, name=pl_name, plot=False,
                                        pl_kwargs=pl_kwargs or None)


def _load_golden(ds_name):
    path = _golden_dir() / f'{ds_name}_logl.npz'
    if not path.exists():
        pytest.skip(f"Golden output missing for {ds_name}: {path}\n"
                    "Run generate_golden.py to create it.")
    golden = np.load(path, allow_pickle=True)
    if 'logl_1d' not in golden.files:
        pytest.skip(
            f"Golden for {ds_name} uses old quick_correl format (no 'logl_1d' key).\n"
            "Re-generate with: python generate_golden.py"
        )
    return golden


def _run_logl(ds_config, corrRV):
    """Run calc_logl_injred and return the 1D logL profile."""
    import starships.correlation as corr
    from starships.correlation_class import Correlations

    visit         = _load_transit(ds_config)
    model      = np.load(Path(ds_config['model_path']).expanduser())
    kind_trans = ds_config.get('kind_trans', 'emission')
    n_pc       = int(visit.params[5])
    Kp_array   = np.array([visit.Kp.value])

    _, logl_map = corr.calc_logl_injred(
        visit, 'seq', visit.planet, Kp_array, corrRV, [n_pc],
        model['wave'], model['spec'], kind_trans,
        counting=False,
    )

    logl_obj = Correlations(logl_map, kind='logl', rv_grid=corrRV,
                            n_pcas=[n_pc], kp_array=Kp_array)
    logl_obj.calc_logl(visit, orders=np.arange(visit.nord),
                       N=visit.N, nolog=True, icorr=visit.icorr, std_robust=True)

    return np.array(logl_obj.logl).squeeze()   # (n_rv,)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLogLProfileClassic:
    """
    Verify that the logL pipeline (calc_logl_injred) produces identical
    numerical results before and after code modifications.

    Tolerance: rtol=1e-10 for full profile.
    These are deterministic floating-point operations on the same machine.
    """

    _cache: dict = {}

    def _logl_1d(self, regression_config, corrRV, ds_name):
        """Run calc_logl_injred once per dataset and cache the result."""
        if ds_name not in self._cache:
            ds_cfg = regression_config['datasets'][ds_name]
            self._cache[ds_name] = _run_logl(ds_cfg, corrRV)
        return self._cache[ds_name]

    @requires_regression_data
    @pytest.mark.parametrize("ds_name", _dataset_names())
    def test_logl_profile_unchanged(self, regression_config, corrRV, ds_name):
        """Full logL(RV) profile must match the golden reference (rtol=1e-10)."""
        logl_1d = self._logl_1d(regression_config, corrRV, ds_name)
        golden  = _load_golden(ds_name)

        np.testing.assert_allclose(
            logl_1d, golden['logl_1d'], rtol=1e-10,
            err_msg=f"[{ds_name}] logL(RV) profile differs from golden reference",
        )

    @requires_regression_data
    @pytest.mark.parametrize("ds_name", _dataset_names())
    def test_peak_rv_unchanged(self, regression_config, corrRV, ds_name):
        """The logL peak position (V_sys) must not shift after code changes."""
        logl_1d  = self._logl_1d(regression_config, corrRV, ds_name)
        golden   = _load_golden(ds_name)
        rv_step  = regression_config['rv_grid']['step']

        peak_rv     = corrRV[np.nanargmax(logl_1d)]
        peak_rv_ref = golden['corrRV'][np.nanargmax(golden['logl_1d'])]

        assert abs(peak_rv - peak_rv_ref) <= rv_step / 2, (
            f"[{ds_name}] logL peak shifted: {peak_rv:.1f} km/s "
            f"(was {peak_rv_ref:.1f} km/s)"
        )

        ds_cfg       = regression_config['datasets'][ds_name]
        expected     = ds_cfg.get('expected_peak_rv')
        expected_tol = ds_cfg.get('expected_peak_rv_tol', rv_step * 2)
        if expected is not None:
            assert abs(peak_rv - expected) <= expected_tol, (
                f"[{ds_name}] Peak RV unexpected: {peak_rv:.1f} km/s "
                f"(expected {expected:.1f} ± {expected_tol:.1f} km/s)"
            )
