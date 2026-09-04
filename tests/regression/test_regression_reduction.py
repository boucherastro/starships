"""
Regression tests for the reduction pipeline.

These tests re-run the reduction from raw FITS files and compare the output
with the existing reduced NPZ (which is the golden reference — no separate
golden generation step is needed, the NPZ files already on Narval ARE the
reference).

Requirements:
    - Raw FITS files accessible at obs_dir (Narval or local)
    - Pipeline config YAML for each dataset
    - ~/.starships/regression_config.yaml with a 'reduction_datasets' section

These tests are SLOW (5-15 min per night) and require raw data, so they are
opt-in only. Run with:

    pytest tests/regression/ -v -m reduction

They are automatically skipped in all other contexts.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

# ---------------------------------------------------------------------------
# Config and markers
# ---------------------------------------------------------------------------

_REGRESSION_CONFIG = Path.home() / '.starships' / 'regression_config.yaml'
_config_available  = _REGRESSION_CONFIG.exists()


def _reduction_dataset_names():
    if not _config_available:
        return []
    with open(_REGRESSION_CONFIG) as f:
        cfg = yaml.safe_load(f)
    return list(cfg.get('reduction_datasets', {}).keys())


pytestmark = pytest.mark.reduction  # entire module requires -m reduction


requires_reduction_data = pytest.mark.skipif(
    not _config_available,
    reason=(
        "Reduction config not found. "
        "On Narval: add 'reduction_datasets' to ~/.starships/regression_config.yaml"
    ),
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def reduction_config():
    with open(_REGRESSION_CONFIG) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_FINAL_NOISE_THRESHOLD = 0.05  # max |Δfinal| / noise tolerated (5 % of per-pixel noise)


class TestReductionPipeline:
    """
    Re-run the reduction pipeline from raw FITS files and verify the output
    matches the existing reduced NPZ files.

    The comparison covers:
      - final  (KEY: the transmission/emission spectrum fed to retrievals and
                cross-correlation).  Because final is centred near 0, rtol is
                meaningless.  We use |Δfinal| / noise < _FINAL_NOISE_THRESHOLD
                instead, which is the physically meaningful acceptance criterion.
      - wave   (wavelength solution — bit-identical expected)
      - flux   (raw blaze-normalised counts — sanity check only)

    noise is intentionally NOT tested separately: noise = std(final) × scaling,
    so if final and scaling are both within tolerance, noise is too.

    All comparisons exclude masked pixels (garbage values in .data at masked
    positions would give spurious failures).
    """

    @requires_reduction_data
    @pytest.mark.parametrize("ds_name", _reduction_dataset_names())
    def test_final_spectrum_unchanged(self, reduction_config, ds_name):
        """max |Δfinal| / noise must be below _FINAL_NOISE_THRESHOLD.

        final is a transmission spectrum centred near 0 (std ~ 0.02), so rtol
        is not meaningful.  The physically correct criterion is that differences
        stay well below the per-pixel noise level.
        """
        transit, golden = self._reduce_and_load_golden(reduction_config, ds_name)
        for key in ('final', 'mask_final', 'noise', 'mask_noise'):
            if key not in golden.files:
                pytest.skip(f"[{ds_name}] '{key}' absent from golden NPZ — "
                            "re-generate with generate_golden.py --only reduction")

        mask_final = transit.final.mask | golden['mask_final'].astype(bool)
        mask_noise = np.asarray(transit.noise.mask) | golden['mask_noise'].astype(bool)
        valid = ~(mask_final | mask_noise)

        diff = np.abs(transit.final.data[valid] - golden['final'][valid])
        noise_ref = golden['noise'][valid]
        diff_over_noise = diff / np.where(noise_ref > 0, noise_ref, np.nan)

        max_frac = float(np.nanmax(diff_over_noise))
        p95_frac = float(np.nanpercentile(diff_over_noise, 95))
        assert max_frac < _FINAL_NOISE_THRESHOLD, (
            f"[{ds_name}] max |Δfinal|/noise = {max_frac:.4e} "
            f"(p95 = {p95_frac:.4e}) exceeds threshold {_FINAL_NOISE_THRESHOLD:.0%}"
        )

    @requires_reduction_data
    @pytest.mark.parametrize("ds_name", _reduction_dataset_names())
    def test_reduced_wave_unchanged(self, reduction_config, ds_name):
        """Re-reduced wavelength solution must match the golden NPZ."""
        transit, golden = self._reduce_and_load_golden(reduction_config, ds_name)
        np.testing.assert_allclose(
            transit.wave, golden['wave'],
            rtol=1e-10,
            err_msg=f"[{ds_name}] Wavelength solution differs from golden NPZ",
        )

    @requires_reduction_data
    @pytest.mark.parametrize("ds_name", _reduction_dataset_names())
    def test_pca_subspace_unchanged(self, reduction_config, ds_name):
        """PCA object must have at least n_pc components (sanity check)."""
        transit, golden = self._reduce_and_load_golden(reduction_config, ds_name)
        ds_cfg = reduction_config['reduction_datasets'][ds_name]
        n_pc   = ds_cfg['n_pc']
        assert transit.pca.components_.shape[0] >= n_pc, (
            f"[{ds_name}] PCA object has fewer components than n_pc={n_pc}: "
            f"{transit.pca.components_.shape}"
        )

    # ------------------------------------------------------------------
    # Shared reduction helper (cached per test session via lru_cache-like
    # pattern — reduction is expensive, run once per dataset name)
    # ------------------------------------------------------------------

    _cache: dict = {}

    def _reduce_and_load_golden(self, reduction_config, ds_name):
        """Run reduction once per dataset and cache the result."""
        if ds_name not in self._cache:
            self._cache[ds_name] = _run_reduction(reduction_config, ds_name)
        return self._cache[ds_name]


class TestPipelineReductionEntryPoint:
    """Same non-regression check as `TestReductionPipeline`, but exercising the actual
    `pipeline.reduction` entry-point functions (`load_planet`, `build_trans_spec`,
    `build_reduction_params`) instead of re-implementing the notebook code path inline.

    This is the path Chantier B's B1 sub-phase changed (named `ReductionParams` instead of
    a positional list, `list_recon` loading, real per-exposure exclusion, etc.) — the other
    tests in this module never actually call `pipeline/reduction.py`, so they wouldn't catch
    a regression introduced there.
    """

    _cache: dict = {}

    @requires_reduction_data
    @pytest.mark.parametrize("ds_name", _reduction_dataset_names())
    def test_final_spectrum_unchanged(self, reduction_config, ds_name):
        transit, golden = self._reduce_and_load_golden(reduction_config, ds_name)
        for key in ('final', 'mask_final', 'noise', 'mask_noise'):
            if key not in golden.files:
                pytest.skip(f"[{ds_name}] '{key}' absent from golden NPZ")

        mask_final = transit.final.mask | golden['mask_final'].astype(bool)
        mask_noise = np.asarray(transit.noise.mask) | golden['mask_noise'].astype(bool)
        valid = ~(mask_final | mask_noise)

        diff = np.abs(transit.final.data[valid] - golden['final'][valid])
        noise_ref = golden['noise'][valid]
        diff_over_noise = diff / np.where(noise_ref > 0, noise_ref, np.nan)

        max_frac = float(np.nanmax(diff_over_noise))
        assert max_frac < _FINAL_NOISE_THRESHOLD, (
            f"[{ds_name}] max |Δfinal|/noise = {max_frac:.4e} exceeds threshold "
            f"{_FINAL_NOISE_THRESHOLD:.0%} (pipeline.reduction entry point)"
        )

    def _reduce_and_load_golden(self, reduction_config, ds_name):
        if ds_name not in self._cache:
            self._cache[ds_name] = _run_reduction_via_pipeline(reduction_config, ds_name)
        return self._cache[ds_name]


class TestReadTimeNPCConsistency:
    """B3 (Chantier B): `n_pc` is now applied at read time (PCA fit once, truncated on
    demand) instead of being baked into the reduction/save step. This checks the actual
    end-to-end guarantee through the real pipeline entry points (`reduce_data`'s cache-hit
    path, `load_reduced_sequence`): reading a cached reduction at a *different* `n_pc` than
    the one that triggered the original reduction must give the exact same `final`/`noise` as
    an independent, from-scratch reduction run directly at that `n_pc`.

    This is a real regression test: it caught a real bug during development (`clip_ts` wasn't
    threaded through `load_reduced_sequence`'s reuse call, silently skipping the sigma-clip
    step applied before PCA truncation, producing a ~0.0078 discrepancy in `final` -- about a
    third of its typical scale, nowhere near float-noise-sized).
    """

    @requires_reduction_data
    @pytest.mark.parametrize("ds_name", _reduction_dataset_names())
    def test_cached_reload_at_different_npc_matches_fresh_reduction(self, reduction_config, ds_name, tmp_path):
        import yaml as _yaml
        import pipeline.reduction as red

        ds_cfg = reduction_config['reduction_datasets'][ds_name]
        pipeline_cfg_path = Path(ds_cfg['pipeline_config']).expanduser()
        if not pipeline_cfg_path.exists():
            pytest.skip(f"Pipeline config not found: {pipeline_cfg_path}")
        with open(pipeline_cfg_path) as f:
            config_dict = _yaml.safe_load(f)

        obs_dir = Path(config_dict.get('obs_dir', '')).expanduser()
        if not obs_dir.exists():
            pytest.skip(f"Raw data directory not found: {obs_dir}")
        config_dict['obs_dir'] = obs_dir

        visit_name = ds_cfg['visit_name']
        mask_tellu = ds_cfg['mask_tellu']
        mask_wings = ds_cfg['mask_wings']
        n_pc_reduction = ds_cfg['n_pc']
        n_pc_reload = n_pc_reduction + 1  # deliberately different from the reduction's own n_pc

        planet, obs = red.load_planet(config_dict, visit_name)

        cached_dir = tmp_path / 'cached'
        cached_dir.mkdir()
        print(f"\n  [{ds_name}] first reduce_data call, n_pc={n_pc_reduction} (writes the file) ...")
        red.reduce_data(config_dict, planet, obs, cached_dir, cached_dir,
                         n_pc_reduction, mask_tellu, mask_wings, visit_name, plot=False)

        print(f"  [{ds_name}] second reduce_data call, n_pc={n_pc_reload} "
              "(same mask_tellu/mask_wings -> cache-hit, read-time PCA truncation only) ...")
        transit_cached = red.reduce_data(config_dict, planet, obs, cached_dir, cached_dir,
                                          n_pc_reload, mask_tellu, mask_wings, visit_name, plot=False)

        fresh_dir = tmp_path / 'fresh'
        fresh_dir.mkdir()
        print(f"  [{ds_name}] independent reduce_data call directly at n_pc={n_pc_reload} "
              "(separate scratch dir, no cache reuse) ...")
        transit_fresh = red.reduce_data(config_dict, planet, obs, fresh_dir, fresh_dir,
                                         n_pc_reload, mask_tellu, mask_wings, visit_name, plot=False)

        np.testing.assert_array_equal(
            transit_cached.final.mask, transit_fresh.final.mask,
            err_msg=f"[{ds_name}] final mask differs between cache-hit and fresh reduction at n_pc={n_pc_reload}",
        )
        valid = ~transit_cached.final.mask
        np.testing.assert_allclose(
            transit_cached.final.data[valid], transit_fresh.final.data[valid],
            rtol=1e-10, atol=1e-12,
            err_msg=f"[{ds_name}] final at n_pc={n_pc_reload}: cache-hit read-time truncation "
                    "diverges from an independent from-scratch reduction at the same n_pc",
        )
        np.testing.assert_allclose(
            transit_cached.noise.data[~transit_cached.noise.mask],
            transit_fresh.noise.data[~transit_fresh.noise.mask],
            rtol=1e-10, atol=1e-12,
            err_msg=f"[{ds_name}] noise (fixed at noise_npc) differs between cache-hit and fresh reduction",
        )


class TestPerVisitPlanetOverride:
    """B3 (Chantier B): a per-visit planet parameter override used at reduction time (e.g.
    `mid_tr`, for a TTV/resonant system like Mathis's TRAPPIST-1 retrieval, where the transit
    epoch genuinely differs from one visit to the next) must survive a save/load round trip,
    and must not leak into a *different* visit loaded afterwards with the same shared planet
    object (`retrieval.py` reuses one `planet` across every visit it loads).
    """

    @requires_reduction_data
    @pytest.mark.parametrize("ds_name", _reduction_dataset_names())
    def test_override_round_trips_without_leaking_into_a_shared_planet(self, reduction_config, ds_name, tmp_path):
        import yaml as _yaml
        import astropy.units as u
        import pipeline.reduction as red
        import starships.planet_obs as pl_obs

        ds_cfg = reduction_config['reduction_datasets'][ds_name]
        pipeline_cfg_path = Path(ds_cfg['pipeline_config']).expanduser()
        if not pipeline_cfg_path.exists():
            pytest.skip(f"Pipeline config not found: {pipeline_cfg_path}")
        with open(pipeline_cfg_path) as f:
            config_dict = _yaml.safe_load(f)

        obs_dir = Path(config_dict.get('obs_dir', '')).expanduser()
        if not obs_dir.exists():
            pytest.skip(f"Raw data directory not found: {obs_dir}")
        config_dict['obs_dir'] = obs_dir

        # Inject a synthetic per-visit mid_tr override, like a TTV config would.
        config_dict.setdefault('pl_params', {})
        overridden_mid_tr = 2454163.5  # arbitrary, just needs to differ from the ExoFile default
        config_dict['pl_params']['mid_tr'] = {'value': overridden_mid_tr, 'unit': 'd'}

        visit_name = ds_cfg['visit_name']
        planet, obs = red.load_planet(config_dict, visit_name)
        assert planet.reduction_overrides, "load_planet should have recorded the mid_tr override"
        np.testing.assert_allclose(planet.mid_tr.to(u.d).value, overridden_mid_tr)

        scratch_dir = tmp_path / 'scratch'
        scratch_dir.mkdir()
        transit = red.reduce_data(config_dict, planet, obs, scratch_dir, scratch_dir,
                                   ds_cfg['n_pc'], ds_cfg['mask_tellu'], ds_cfg['mask_wings'],
                                   visit_name, plot=False)
        np.testing.assert_allclose(transit.planet.mid_tr.to(u.d).value, overridden_mid_tr)

        # Reload with a *different*, non-overridden shared planet passed in explicitly (as
        # retrieval.py would for a second visit reusing the same planet) -- the saved
        # per-visit override must still win, and the shared planet object must not be mutated.
        shared_planet, _ = red.load_planet({**config_dict, 'pl_params': {}}, visit_name)
        shared_mid_tr_before = float(shared_planet.mid_tr.to(u.d).value)

        reloaded = pl_obs.load_reduced_sequence(
            f"retrieval_input_{visit_name}_maskwings{ds_cfg['mask_wings']*100:n}"
            f"_masktellu{ds_cfg['mask_tellu']*100:n}",
            ds_cfg['n_pc'], path=scratch_dir, planet=shared_planet, name=planet.name,
        )

        assert reloaded.planet is not shared_planet, (
            "load_reduced_sequence must apply the saved override to a private copy, "
            "not mutate the shared planet passed in"
        )
        np.testing.assert_allclose(reloaded.planet.mid_tr.to(u.d).value, overridden_mid_tr)
        np.testing.assert_allclose(shared_planet.mid_tr.to(u.d).value, shared_mid_tr_before)


def _run_reduction_via_pipeline(reduction_config, ds_name):
    """Re-run the reduction through the real `pipeline.reduction` entry point and return
    (transit, golden_data)."""
    import yaml as _yaml
    import pipeline.reduction as red

    ds_cfg = reduction_config['reduction_datasets'][ds_name]

    pipeline_cfg_path = Path(ds_cfg['pipeline_config']).expanduser()
    if not pipeline_cfg_path.exists():
        pytest.skip(f"Pipeline config not found: {pipeline_cfg_path}")
    with open(pipeline_cfg_path) as f:
        config_dict = _yaml.safe_load(f)

    obs_dir = Path(config_dict.get('obs_dir', '')).expanduser()
    if not obs_dir.exists():
        pytest.skip(f"Raw data directory not found: {obs_dir}")

    golden_path = Path(ds_cfg['golden_npz']).expanduser()
    if not golden_path.exists():
        pytest.skip(f"Golden NPZ not found: {golden_path}")

    visit_name = ds_cfg['visit_name']
    config_dict['obs_dir'] = obs_dir  # Path, matching known-working usage in _run_reduction()

    print(f"\n  [{ds_name}] pipeline.reduction.load_planet ...")
    planet, obs = red.load_planet(config_dict, visit_name)

    n_pc       = ds_cfg['n_pc']
    mask_tellu = ds_cfg['mask_tellu']
    mask_wings = ds_cfg['mask_wings']
    bad_indexs = config_dict['bad_indexs'].get(visit_name, []) if config_dict['bad_indexs'] else []

    print(f"  [{ds_name}] pipeline.reduction.build_trans_spec "
          f"(n_pc={n_pc}, mask_tellu={mask_tellu}, mask_wings={mask_wings}) ...")
    list_tr = red.build_trans_spec(config_dict, n_pc, mask_tellu, mask_wings, obs, planet,
                                    bad_indexs=bad_indexs)
    transit = list_tr['1']

    golden = np.load(golden_path, allow_pickle=True)
    return transit, golden


# ---------------------------------------------------------------------------
# Reduction runner (module-level so it can be used standalone)
# ---------------------------------------------------------------------------

def _run_reduction(reduction_config, ds_name):
    """Re-run the reduction following the notebook code path and return (transit, golden_data)."""
    import yaml as _yaml
    import astropy.units as u
    import astropy.constants as const
    import starships.planet_obs as pl_obs
    from starships.planet_obs import Observations
    from pipeline.reduction import pl_param_units

    ds_cfg = reduction_config['reduction_datasets'][ds_name]

    # Load the pipeline config YAML for this dataset
    pipeline_cfg_path = Path(ds_cfg['pipeline_config']).expanduser()
    if not pipeline_cfg_path.exists():
        pytest.skip(f"Pipeline config not found: {pipeline_cfg_path}")
    with open(pipeline_cfg_path) as f:
        config_dict = _yaml.safe_load(f)

    # Check raw FITS are accessible
    obs_dir = Path(config_dict.get('obs_dir', '')).expanduser()
    if not obs_dir.exists():
        pytest.skip(f"Raw data directory not found: {obs_dir}")

    # Check golden NPZ exists
    golden_path = Path(ds_cfg['golden_npz']).expanduser()
    if not golden_path.exists():
        pytest.skip(f"Golden NPZ not found: {golden_path}")

    # Build planet kwargs from config (skip null values, same as retrieval.py)
    pl_kwargs = pl_param_units(config_dict) if config_dict.get('pl_params') else {}

    # Create Observations and load raw data (notebook code path). Which raw-file format to
    # expect (external blaze/wave calibration files vs. bundled/embedded extensions) is
    # entirely determined by `instrument` now (Chantier B, B2 follow-up) -- use e.g.
    # instrument: 'SPIRou-APERO-CADC'/'NIRPS-APERO-CADC' in the dataset's pipeline config for
    # the bundled format, instead of a separate cadc flag/fetch_data(CADC=...) here.
    visit_name = ds_cfg['visit_name']
    list_filenames = {
        'list_e2ds':  f'list_e2ds_{visit_name}',
        'list_tcorr': f'list_tcorr_{visit_name}',
        'list_recon': f'list_recon_{visit_name}',
    }

    instrument = config_dict.get('instrument', 'SPIRou-APERO')
    print(f"\n  [{ds_name}] Loading raw data from {obs_dir} (visit: {visit_name}, instrument: {instrument}) ...")
    obs = Observations(name=config_dict['pl_name'], instrument=instrument, pl_kwargs=pl_kwargs)
    obs.fetch_data(obs_dir, **list_filenames)
    obs.n_spec = len(obs.filenames)  # not set automatically by fetch_data

    # All exposures; remove bad ones if specified in config
    all_exp = np.arange(obs.n_spec)
    bad = config_dict.get('bad_indexs', {}).get(visit_name, [])
    transit_tags = [np.delete(all_exp, bad) if bad else all_exp]

    # Reduction parameters (notebook format)
    n_pc       = ds_cfg['n_pc']
    mask_tellu = ds_cfg['mask_tellu']
    mask_wings = ds_cfg['mask_wings']
    params_all = [[mask_tellu, mask_wings, 51, 41, 5, n_pc, 5.0, 5.0, 5.0, 5.0]]

    kwargs_gen_tr = {
        'coeffs':     config_dict['coeffs'],
        'ld_model':   config_dict['ld_model'],
        'do_tr':      [1],
        'kind_trans': config_dict['kind_trans'],
        'polynome':   [False],
        'cbp':        True,
    }
    kwargs_build_ts = {
        'clip_ratio':  config_dict['clip_ratio'],
        'clip_ts':     config_dict['clip_ts'],
        'unberv_it':   config_dict['unberv_it'],
    }

    print(f"  [{ds_name}] Running reduction "
          f"(n_pc={n_pc}, mask_tellu={mask_tellu}, mask_wings={mask_wings}) ...")
    list_tr = pl_obs.generate_all_transits(
        obs, transit_tags, [0.0], params_all, config_dict['iout_all'],
        counting=False, **kwargs_gen_tr, **kwargs_build_ts,
    )
    transit = list_tr['1']

    golden = np.load(golden_path, allow_pickle=True)
    return transit, golden
