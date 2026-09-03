"""
Generate golden (reference) outputs for STARSHIPS regression tests.

Run this ONCE on Narval with the current code before any cleanup or refactoring.
The outputs are saved in the directory returned by get_regression_golden_dir()
(defaults to ~/.starships/regression_golden/) and are used by the pytest
regression tests to verify that code changes do not alter scientific results.

Usage::

    # Default: all goldens (logl + model)
    python tests/regression/generate_golden.py

    # Custom config and output directory
    python tests/regression/generate_golden.py \\
        --config /path/to/regression_config.yaml \\
        --output-dir /path/to/golden/

    # Generate only specific type
    python tests/regression/generate_golden.py --only logl
    python tests/regression/generate_golden.py --only model
    python tests/regression/generate_golden.py --only reduction

    # Also save diagnostic plots (PNG) alongside the golden files
    python tests/regression/generate_golden.py --plots

Setup::

    from starships.config import edit_config
    edit_config(regression_golden_dir='/scratch/user/starships_regression/golden')
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import yaml


def load_regression_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_logl_1d(ds_config: dict, corrRV: np.ndarray):
    """Compute the logL 1D profile using calc_logl_injred (full pipeline).

    Uses load_single_sequences to get the transit object (orbital parameters,
    PCA, noise, etc.), then runs calc_logl_injred with the correct Kp and
    applies Correlations.calc_logl normalization (MAD/0.6745, in-transit sum).

    This matches exactly what is done in Correlations.ipynb / plot_ccflogl.

    Parameters
    ----------
    ds_config : dataset section from regression_config.yaml
        Required keys: npz_path, model_path, pl_name
        Optional keys: kind_trans (default 'emission')
    corrRV    : RV grid (km/s)

    Returns
    -------
    tr        : loaded Observations object
    logl_1d   : 1D logL profile, shape (n_rv,)
    """
    import starships.planet_obs as pl_obs
    import starships.correlation as corr
    from starships.correlation_class import Correlations

    from pipeline.reduction import pl_param_units

    npz_path   = Path(ds_config['npz_path']).expanduser()
    pl_name    = ds_config['pl_name']
    kind_trans = ds_config.get('kind_trans', 'emission')

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

    tr       = pl_obs.load_single_sequences(npz_path, pl_name, plot=False,
                                            pl_kwargs=pl_kwargs or None)
    model    = np.load(Path(ds_config['model_path']).expanduser())
    n_pc     = int(tr.params[5])
    Kp_array = np.array([tr.Kp.value])

    _, logl_map = corr.calc_logl_injred(
        tr, 'seq', tr.planet, Kp_array, corrRV, [n_pc],
        model['wave'], model['spec'], kind_trans,
        counting=False,
    )

    logl_obj = Correlations(logl_map, kind='logl', rv_grid=corrRV,
                            n_pcas=[n_pc], kp_array=Kp_array)
    logl_obj.calc_logl(tr, orders=np.arange(tr.nord),
                       N=tr.N, nolog=True, icorr=tr.icorr, std_robust=True)

    return tr, np.array(logl_obj.logl).squeeze()  # (n_rv,)


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def _plot_logl(ds_name, corrRV, logl_1d, tr, plots_dir):
    """Save logL(RV) profile (1D) as a PNG file."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plots] matplotlib not available — skipping plots.")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    peak_rv = corrRV[np.nanargmax(logl_1d)]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(corrRV, logl_1d, lw=1.5)
    ax.axvline(peak_rv, color='r', ls='--', label=f'Pic = {peak_rv:+.1f} km/s')
    ax.axvline(float(tr.RV_const), color='gray', ls=':', lw=1,
               label=f'V_sys config = {float(tr.RV_const):.1f} km/s')
    ax.set_xlabel('RV (km/s)')
    ax.set_ylabel('logL  [in-transit, Σ ordres]')
    ax.set_title(f'{ds_name} — logL 1D  (Kp={tr.Kp:.1f}, n_pc={int(tr.params[5])})')
    ax.legend()
    fig.tight_layout()
    out = plots_dir / f'{ds_name}_logl_profile.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Plot : {out}")


def _plot_model(ds_name, wv, model, plots_dir):
    """Save model spectrum as a PNG file."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plots] matplotlib not available — skipping plot.")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(wv, model, lw=0.8)
    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Model spectrum')
    ax.set_title(f'{ds_name} — model spectrum')
    fig.tight_layout()
    out = plots_dir / f'{ds_name}_model_spectrum.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Plot : {out}")


def _plot_reduction(ds_name, transit, plots_dir):
    """Save PCA components and a few example spectra as PNG files."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plots] matplotlib not available — skipping plot.")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    # PCA components
    comps = transit.pca.components_
    n_comp = min(5, comps.shape[0])
    fig, axes = plt.subplots(n_comp, 1, figsize=(12, 2 * n_comp), sharex=True)
    if n_comp == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(comps[i], lw=0.8)
        ax.set_ylabel(f'PC {i+1}')
    axes[-1].set_xlabel('Pixel index (flattened)')
    fig.suptitle(f'{ds_name} — PCA components')
    fig.tight_layout()
    out = plots_dir / f'{ds_name}_pca_components.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Plot : {out}")

    # Example spectra: first order, first 5 exposures
    if transit.flux is not None:
        flux_data = np.ma.filled(transit.flux, np.nan)
        n_exp = min(5, flux_data.shape[0])
        wave = transit.wave  # shape (n_ord, n_pix) or similar
        fig, ax = plt.subplots(figsize=(12, 4))
        for i in range(n_exp):
            ax.plot(flux_data[i, 0, :], lw=0.6, alpha=0.7, label=f'exp {i}')
        ax.set_xlabel('Pixel index')
        ax.set_ylabel('Flux (order 0)')
        ax.set_title(f'{ds_name} — first {n_exp} exposures, order 0')
        ax.legend(fontsize=7)
        fig.tight_layout()
        out = plots_dir / f'{ds_name}_flux_example.png'
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  Plot : {out}")


# ---------------------------------------------------------------------------
# Golden generators
# ---------------------------------------------------------------------------

def generate_logl_goldens(cfg, output_dir, plots_dir=None):
    """Generate golden NPZ files for logL regression tests.

    Uses calc_logl_injred (full pipeline with orbital correction) instead of
    quick_correl.  Requires pl_name and kind_trans in each dataset config.
    """
    corrRV = np.arange(
        cfg['rv_grid']['min'],
        cfg['rv_grid']['max'] + cfg['rv_grid']['step'],
        cfg['rv_grid']['step'],
    )

    summary = {}
    for ds_name, ds_cfg in cfg.get('datasets', {}).items():
        print(f"\n{'='*60}")
        print(f"  {ds_name}")
        print(f"{'='*60}")

        if 'pl_name' not in ds_cfg:
            print(f"  SKIP: 'pl_name' missing from dataset config — "
                  "add it to regression_config.yaml")
            continue

        print(f"  Loading transit ({ds_cfg['pl_name']}) and running "
              f"calc_logl_injred ...")
        tr, logl_1d = compute_logl_1d(ds_cfg, corrRV)

        peak_rv = corrRV[np.nanargmax(logl_1d)]

        out_path = output_dir / f'{ds_name}_logl.npz'
        np.savez(
            out_path,
            corrRV  = corrRV,
            logl_1d = logl_1d,
        )

        print(f"  Pic   : RV = {peak_rv:+.1f} km/s  (V_sys config = {float(tr.RV_const):.1f} km/s)")
        print(f"  Max logL : {float(np.nanmax(logl_1d)):.6f}")
        print(f"  Saved : {out_path}")
        summary[ds_name] = {'peak_rv': float(peak_rv),
                            'max_logl': float(np.nanmax(logl_1d))}

        if plots_dir is not None:
            _plot_logl(ds_name, corrRV, logl_1d, tr, plots_dir)

    return summary


def generate_lnprob_goldens(cfg, output_dir, plots_dir=None):
    """Generate golden NPZ files for lnprob regression tests.

    Runs the full retrieval chain (setup_retrieval + load_high_res_data +
    lnprob) at a fixed theta vector and saves the resulting scalar.
    Requires petitRADTRANS and the reduced high-res NPZ data.
    """
    try:
        import petitRADTRANS  # noqa: F401
    except ImportError:
        print("\n  [lnprob] petitRADTRANS not available — skipping lnprob goldens.")
        return {}

    from starships import retrieval as ret

    summary = {}
    for ds_name, ds_cfg in cfg.get('lnprob_datasets', {}).items():
        print(f"\n{'='*60}")
        print(f"  {ds_name}  (lnprob)")
        print(f"{'='*60}")

        ret_cfg_path = Path(ds_cfg['retrieval_config']).expanduser()
        if not ret_cfg_path.exists():
            print(f"  SKIP: retrieval config not found: {ret_cfg_path}")
            continue

        print(f"  Setting up retrieval from {ret_cfg_path.name} ...")
        ret.setup_retrieval(input_parameters=ret_cfg_path)

        print("  Loading high-res data ...")
        ret.load_high_res_data()

        theta_params = ds_cfg['theta_params']
        missing = [k for k in ret.params_prior.keys() if k not in theta_params]
        if missing:
            print(f"  SKIP: theta_params missing keys: {missing}")
            continue

        theta = np.array([theta_params[k] for k in ret.params_prior.keys()])

        print(f"  Evaluating lnprob (n_params={len(theta)}) ...")
        lnprob_val = float(ret.lnprob(theta))

        if not np.isfinite(lnprob_val):
            print(f"  WARNING: lnprob = {lnprob_val} — theta may be outside the prior.")

        out_path = output_dir / f'{ds_name}_lnprob.npz'
        np.savez(out_path, lnprob=lnprob_val, theta=theta)

        print(f"  lnprob : {lnprob_val:.6f}")
        print(f"  Saved  : {out_path}")
        summary[ds_name] = {'lnprob': lnprob_val}

        if plots_dir is not None:
            _plot_lnprob(ds_name, lnprob_val, theta, list(ret.params_prior.keys()), plots_dir)

    return summary


def _plot_lnprob(ds_name, lnprob_val, theta, param_names, plots_dir):
    """Save a bar chart of the theta vector as a PNG file."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plots] matplotlib not available — skipping plot.")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(8, len(theta) * 0.5), 4))
    ax.bar(range(len(theta)), theta)
    ax.set_xticks(range(len(theta)))
    ax.set_xticklabels(param_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('theta value (sampled space)')
    ax.set_title(f'{ds_name} — theta vector  (lnprob = {lnprob_val:.4f})')
    fig.tight_layout()
    out = plots_dir / f'{ds_name}_lnprob_theta.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Plot : {out}")


def generate_model_goldens(cfg, output_dir, plots_dir=None):
    """Generate golden NPZ files for model regression tests (requires petitRADTRANS)."""
    try:
        import petitRADTRANS  # noqa: F401
    except ImportError:
        print("\n  [model] petitRADTRANS not available — skipping model goldens.")
        return {}

    from starships import retrieval as ret

    summary = {}
    for ds_name, ds_cfg in cfg.get('model_datasets', {}).items():
        print(f"\n{'='*60}")
        print(f"  {ds_name}  (model)")
        print(f"{'='*60}")

        ret_cfg_path = Path(ds_cfg['retrieval_config']).expanduser()
        if not ret_cfg_path.exists():
            print(f"  SKIP: retrieval config not found: {ret_cfg_path}")
            continue

        print(f"  Setting up retrieval from {ret_cfg_path.name} ...")
        ret.setup_retrieval(input_parameters=ret_cfg_path)

        theta_params = ds_cfg['theta_params']
        missing = [k for k in ret.params_prior.keys() if k not in theta_params]
        if missing:
            print(f"  SKIP: theta_params missing keys: {missing}")
            continue

        theta = np.array([theta_params[k] for k in ret.params_prior.keys()])

        print(f"  Generating model spectrum (mode={ds_cfg['mode']}) ...")
        theta_regions = ret.unpack_theta(theta)
        wv, model = ret.prepare_model_multi_reg(theta_regions, mode=ds_cfg['mode'])

        out_path = Path(ds_cfg['golden_npz']).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, wv=wv, model=model)

        print(f"  wv    : {wv.shape}  [{wv.min():.4f}, {wv.max():.4f}] µm")
        print(f"  model : {model.shape}  [{model.min():.3e}, {model.max():.3e}]")
        print(f"  Saved : {out_path}")
        summary[ds_name] = {'wv_range': [float(wv.min()), float(wv.max())],
                            'model_max': float(model.max())}

        if plots_dir is not None:
            _plot_model(ds_name, wv, model, plots_dir)

    return summary


def generate_reduction_goldens(cfg, plots_dir=None):
    """Re-run the reduction pipeline and overwrite the golden NPZ files.

    The golden_npz path in reduction_datasets IS the reference file — running
    this function regenerates it on the current machine so that regression tests
    pass on the same platform (MKL on Narval, Accelerate on macOS, etc.).
    """
    import yaml as _yaml
    import numpy as np
    import starships.planet_obs as pl_obs
    from starships.planet_obs import Observations
    from pipeline.reduction import pl_param_units

    summary = {}
    for ds_name, ds_cfg in cfg.get('reduction_datasets', {}).items():
        print(f"\n{'='*60}")
        print(f"  {ds_name}  (reduction)")
        print(f"{'='*60}")

        pipeline_cfg_path = Path(ds_cfg['pipeline_config']).expanduser()
        if not pipeline_cfg_path.exists():
            print(f"  SKIP: pipeline config not found: {pipeline_cfg_path}")
            continue

        with open(pipeline_cfg_path) as f:
            config_dict = _yaml.safe_load(f)

        obs_dir = Path(config_dict.get('obs_dir', '')).expanduser()
        if not obs_dir.exists():
            print(f"  SKIP: raw data directory not found: {obs_dir}")
            continue

        golden_path = Path(ds_cfg['golden_npz']).expanduser()

        pl_kwargs = pl_param_units(config_dict) if config_dict.get('pl_params') else {}

        visit_name = ds_cfg['visit_name']
        list_filenames = {
            'list_e2ds':  f'list_e2ds_{visit_name}',
            'list_tcorr': f'list_tcorr_{visit_name}',
            'list_recon': f'list_recon_{visit_name}',
        }

        # Which raw-file format to expect (external blaze/wave calibration files vs.
        # bundled/embedded extensions) is entirely determined by `instrument` (Chantier B, B2
        # follow-up) -- use e.g. instrument: 'NIRPS-APERO-CADC' in the dataset's pipeline
        # config for the bundled format, instead of a separate cadc flag here.
        instrument = config_dict.get('instrument', 'SPIRou-APERO')
        print(f"  Loading raw data from {obs_dir} (visit: {visit_name}, instrument: {instrument}) ...")
        obs = Observations(name=config_dict['pl_name'], instrument=instrument, pl_kwargs=pl_kwargs)
        obs.fetch_data(obs_dir, **list_filenames)
        obs.n_spec = len(obs.filenames)

        all_exp = np.arange(obs.n_spec)
        bad = config_dict.get('bad_indexs', {}).get(visit_name, [])
        transit_tags = [np.delete(all_exp, bad) if bad else all_exp]

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

        print(f"  Running reduction "
              f"(n_pc={n_pc}, mask_tellu={mask_tellu}, mask_wings={mask_wings}) ...")
        list_tr = pl_obs.generate_all_transits(
            obs, transit_tags, [0.0], params_all, config_dict['iout_all'],
            counting=False, **kwargs_gen_tr, **kwargs_build_ts,
        )
        transit = list_tr['1']

        golden_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            golden_path,
            final        = transit.final.filled(0),
            mask_final   = transit.final.mask,
            flux         = transit.flux.filled(0),
            mask_flux    = transit.flux.mask,
            wave         = transit.wave,
            noise        = transit.noise.filled(0),
            mask_noise   = transit.noise.mask,
            components_  = transit.pca.components_,
        )
        print(f"  flux  : {transit.flux.shape}")
        print(f"  wave  : {transit.wave.shape}")
        print(f"  Saved : {golden_path}")
        summary[ds_name] = {'n_spec': transit.flux.shape[0],
                            'n_ord':  transit.flux.shape[1]}

        if plots_dir is not None:
            _plot_reduction(ds_name, transit, plots_dir)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate regression golden outputs for STARSHIPS'
    )
    parser.add_argument(
        '--config',
        default=str(Path.home() / '.starships' / 'regression_config.yaml'),
        help='Path to regression config YAML (default: ~/.starships/regression_config.yaml)',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Override output directory for logL/model golden files '
             '(default: from starships.config.get_regression_golden_dir())',
    )
    parser.add_argument(
        '--only',
        choices=['logl', 'model', 'lnprob', 'reduction'],
        default=None,
        help='Generate only one type of golden (default: logl + model + lnprob)',
    )
    parser.add_argument(
        '--plots',
        action='store_true',
        help='Save diagnostic PNG plots alongside the golden files',
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        print("Copy tests/regression/config_template.yaml to "
              "~/.starships/regression_config.yaml and fill in the paths.")
        sys.exit(1)

    from starships.config import get_regression_golden_dir

    cfg = load_regression_config(config_path)

    output_dir = Path(args.output_dir) if args.output_dir else get_regression_golden_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = (output_dir / 'plots') if args.plots else None

    summary = {}

    if args.only == 'logl' or args.only is None:
        logl_summary = generate_logl_goldens(cfg, output_dir, plots_dir=plots_dir)
        summary.update(logl_summary)

    if args.only == 'model' or args.only is None:
        model_summary = generate_model_goldens(cfg, output_dir, plots_dir=plots_dir)
        summary.update({f'[model] {k}': v for k, v in model_summary.items()})

    if args.only == 'lnprob' or args.only is None:
        lnprob_summary = generate_lnprob_goldens(cfg, output_dir, plots_dir=plots_dir)
        summary.update({f'[lnprob] {k}': v for k, v in lnprob_summary.items()})

    if args.only == 'reduction':
        red_summary = generate_reduction_goldens(cfg, plots_dir=plots_dir)
        summary.update({f'[reduction] {k}': v for k, v in red_summary.items()})

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for name, res in summary.items():
        if 'peak_rv' in res:
            print(f"  {name:<40}  peak RV = {res['peak_rv']:+7.1f} km/s  "
                  f"logL_max = {res['max_logl']:.4f}")
        elif 'lnprob' in res:
            print(f"  {name:<40}  lnprob = {res['lnprob']:.6f}")
        elif 'wv_range' in res:
            print(f"  {name:<40}  wv [{res['wv_range'][0]:.3f}, "
                  f"{res['wv_range'][1]:.3f}] µm  model_max = {res['model_max']:.3e}")
        else:
            print(f"  {name:<40}  {res['n_spec']} exposures, {res['n_ord']} orders")

    print(f"\n  Golden outputs saved to: {output_dir.resolve()}")
    if plots_dir is not None:
        print(f"  Plots saved to:          {plots_dir.resolve()}")
    print("\n  Next: after any code change, run")
    print("    pytest tests/regression/ -v")


if __name__ == '__main__':
    main()
