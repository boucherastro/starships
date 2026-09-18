"""
Regression tests for the logL(RV) profile computed by the `logl_grid` engine.

These tests verify that numerical results are unchanged after non-scientific
code modifications (cleanup, refactoring, dependency updates, etc.).

They run the harmonized model engine (`starships.logl_grid`, Chantier C -- the same
per-visit model preparation as `retrieval.py::lnprob`: `_prepare_visit_model` +
`_get_chi2_detailed`, then `save_logl_grid`/`load_logl_results`/`get_logl`) on one
reduced visit, at the reference Kp, over the `rv_grid` of the regression config, and
compare the resulting 1D logL(RV) profile to a golden reference. Before Chantier C
this file exercised the old classic-CCF engine (`correlation.calc_logl_injred`), whose
golden encoded Chantier A bug #2 -- deliberately dropped (Antoine, 2026-09-18).

Prerequisites (on Narval):
    1. Set paths in ~/.starships/config.yaml (via starships.config.edit_config)
    2. Copy config_template.yaml → ~/.starships/regression_config.yaml and fill in paths
    3. Run generate_golden.py once to create golden references
    4. Run `pytest tests/regression/ -v` after every code change

Each dataset in regression_config.yaml must have:
    npz_path   : path to the reduced NPZ (passed to load_reduced_sequence)
    n_pc       : number of PCA components to remove (read-time argument since B3)
    model_path : path to the model NPZ -- a `model_sequence.PlanetModel` file written by
                 `retrieval.save_planet_model`, or an old-engine combined-ratio file
                 (`wave`/`spec`), loaded as the legacy `combined_ratio` form
    pl_name    : planet name for load_reduced_sequence (e.g. "WASP-33 b")
    kind_trans : 'emission' or 'transmission'
    kp_ref     : optional, Kp (km/s) at which the RV profile is computed (default: the
                 planet's own Kp)
    instrum    : instrument name for `instruments.load_instrum` (default 'spirou')
    get_ker_file : optional, only for a multi-region model (same key as the retrieval YAML)

Tests are automatically SKIPPED if the config or golden outputs are missing,
so they do not break local development or CI environments without real data.
"""

import tempfile
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


def _planet_kwargs(ds_config):
    """Planet parameter overrides from the dataset's optional `retrieval_config` YAML,
    matching how parameters are set in the reduction/retrieval pipeline."""
    from pipeline.reduction import pl_param_units

    ret_cfg_key = ds_config.get('retrieval_config')
    if not ret_cfg_key:
        return None
    ret_cfg_path = Path(ret_cfg_key).expanduser()
    if not ret_cfg_path.exists():
        print(f"  Warning: retrieval_config not found: {ret_cfg_path}")
        return None
    with open(ret_cfg_path) as f:
        ret_cfg = yaml.safe_load(f)
    return pl_param_units(ret_cfg) if ret_cfg.get('pl_params') else None


def _load_transit(ds_config):
    """Load one visit through `planet_obs.load_sequences` -- the exact entry point
    `logl_grid.load_logl_grid_data`/`retrieval.load_high_res_data` use (Chantier C1).

    Returns the `(planet, data_info, data_visit)` triple `logl_grid` works with.
    B3: `n_pc` is a read-time argument (no longer baked into the saved file), so it
    must be given explicitly in `regression_config.yaml` (`n_pc:` per dataset).
    """
    import starships.planet_obs as pl_obs
    from starships.planet_obs import Observations

    path = Path(ds_config['npz_path']).expanduser()
    if not path.exists():
        pytest.skip(f"Reduced data not found: {path}")

    obs = Observations(name=ds_config['pl_name'], pl_kwargs=_planet_kwargs(ds_config))
    planet = obs.planet
    # `load_sequences` appends the `_data_trs_[0].npz` suffix itself.
    stem = path.name.replace('_data_trs_0.npz', '').replace('_data_trs_.npz', '')
    data_info, data_visit = pl_obs.load_sequences(stem, ds_config['n_pc'], path=path.parent,
                                                  planet=planet)
    return planet, data_info, data_visit


def planet_kp(planet):
    """Planet's orbital RV semi-amplitude Kp (km/s) from its parameters -- the same
    orbital solution `Observations.gen_rv_sequence` uses (`orbite.rv`), which a bare
    `Planet` does not carry as an attribute."""
    from starships import orbite

    _, _, kp, _ = orbite.rv(np.array([0.0]), planet.period, e=planet.excent, i=planet.incl,
                            w=planet.w, Mp=planet.M_pl, Mstar=planet.M_star)
    return float(np.squeeze(kp.to('km/s').value))


def _load_golden(ds_name):
    path = _golden_dir() / f'{ds_name}_logl.npz'
    if not path.exists():
        pytest.skip(f"Golden output missing for {ds_name}: {path}\n"
                    "Run generate_golden.py to create it.")
    golden = np.load(path, allow_pickle=True)
    if 'engine' not in golden.files or str(golden['engine']) != 'logl_grid':
        pytest.skip(
            f"Golden for {ds_name} was generated by the old classic-CCF engine "
            "(pre-Chantier C).\nRe-generate with: python generate_golden.py --only logl"
        )
    return golden


def compute_logl_1d(ds_config, corrRV):
    """Compute the 1D logL(RV) profile at `kp_ref` with the `logl_grid` engine.

    Exercises the full compute/save/load chain of `logl_grid` on one visit:
    `load_model` → `compute_logl_grid` (in-process, `n_process=1`) →
    `save_logl_grid` → `load_logl_results` → `get_logl(alpha=1, summed over
    exposures and orders)`. Shared with `generate_golden.py`.

    Returns
    -------
    planet : starships.planet_obs.Planet
    logl_1d : np.ndarray, shape (n_rv,)
    """
    import starships.logl_grid as lg
    from starships.instruments import load_instrum

    planet, data_info, data_visit = _load_transit(ds_config)
    kind_trans = ds_config.get('kind_trans', 'emission')

    # Module globals `setup_logl_grid`/`load_logl_grid_data` would set from a full
    # retrieval YAML -- set directly from the (much smaller) regression config.
    lg.planet = planet
    lg.kind_trans = kind_trans
    lg.instrum_param_list = [load_instrum(ds_config.get('instrum', 'spirou'))]
    lg.apply_alpha = True
    lg.use_real_stellar_rv = False
    lg.logl_grouping = 'per_instrument'
    lg.data_visits = [data_visit]
    lg.data_info_list = [data_info]
    lg.idx_orders = np.arange(data_visit['flux'].shape[1])
    lg.axis_sum = -1
    get_ker_file = ds_config.get('get_ker_file')
    if get_ker_file is None:
        lg.get_ker = lambda theta_regions, phase=None, planet=None, instrum=None, \
                            model_resolution=None: [None for _ in theta_regions]
    else:
        from starships.retrieval_utils import load_custom_get_ker
        lg.get_ker = load_custom_get_ker(Path(get_ker_file).expanduser())

    lg.load_model(Path(ds_config['model_path']).expanduser())

    # Kp of the RV profile: the dataset's `kp_ref` if given, else the planet's own Kp
    # (what the old classic-CCF test used, via `visit.Kp`).
    kp_ref = ds_config.get('kp_ref')
    if kp_ref is None:
        kp_ref = planet_kp(planet)
    kp_array = np.array([float(kp_ref)])
    results = lg.compute_logl_grid(rv_array=corrRV, kp_array=kp_array, n_process=1)

    with tempfile.TemporaryDirectory() as tmp:
        lg.save_logl_grid(results, tmp, 'regression')
        files = sorted(Path(tmp).glob('regression*_visit*.npz'))
        lg.load_logl_results(files)
        logl = lg.get_logl(alpha=1., sum_axis=(-2, -1))   # (n_vsys, n_kp)

    return planet, np.asarray(logl)[:, 0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLogLProfile:
    """
    Verify that the `logl_grid` engine produces identical numerical results
    before and after code modifications.

    Tolerance: rtol=1e-10 for the full profile.
    These are deterministic floating-point operations on the same machine.
    """

    _cache: dict = {}

    def _logl_1d(self, regression_config, corrRV, ds_name):
        """Run the engine once per dataset and cache the result."""
        if ds_name not in self._cache:
            ds_cfg = regression_config['datasets'][ds_name]
            _, self._cache[ds_name] = compute_logl_1d(ds_cfg, corrRV)
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
