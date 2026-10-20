"""Command-line tool: make every retrieval-analysis figure automatically, no walker manipulation.

The analysis notebooks (`tutorials/retrieval_analysis/`) are interactive: you look at the chains
and decide by hand which walkers/steps to keep. This script is for the other case -- you already
know the retrieval converged and just want the figures, unattended, from a burn-in cut (and
optionally a log-likelihood floor). If you need to inspect the chains first, use a notebook
instead; this script has no equivalent of that.

Usage::

    run_starships_retrieval_figures --yaml retrieval_inputs.yaml \\
        --walker-file walker_steps_x.h5 --discard 200 --output-dir figures/

    # From Python, e.g. to fold into another script:
    from starships import retrieval, retrieval_figures
    retrieval.setup_retrieval('retrieval_inputs.yaml')
    retrieval_figures.make_all_figures(retrieval, 'walker_steps_x.h5', discard=200,
                                       output_dir='figures/')

What it makes (each section is skipped, with a log message, if the run does not have what it
needs -- e.g. no abundance ratios for a multi-region run, no rotation kernels for a single-region
one): corner plot, log-likelihood scatter plot, the model spectrum (saved for `logl_grid` whenever
a high-resolution model is possible), the comparison with low-resolution data, the contribution of
each species, temperature/abundance profile confidence intervals, spectrum confidence intervals
with the data, rotation kernels and abundance ratios (both multi-/single-region specific, see
above).

A failure in one section is logged and does not stop the others -- this is meant to run
unattended, so one broken figure should not lose the rest.
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import matplotlib
matplotlib.use('Agg')  # no display needed/available when run as a batch script
import matplotlib.pyplot as plt
import numpy as np

from . import retrieval_analysis as ra
from . import retrieval_utils as ru
from .plotting_fcts import (colorsDict, get_plot_labels, get_plot_limits_from_data,
                            plot_contributions, plot_p_profile_sample, plot_profile_combined,
                            plot_spectra_sample_GTC, plot_tp_sample, scatterplot_logl,
                            setup_default_plot_params)

log = logging.getLogger(__name__)

PathLike = Union[str, Path]


def _save(fig, output_dir: Path, name: str, formats: Sequence[str]) -> List[Path]:
    """Save `fig` under `output_dir/name.<fmt>` for each format, then close it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = output_dir / f'{name}.{fmt}'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        paths.append(path)
    plt.close(fig)
    return paths


def make_all_figures(retrieval_obj, walker_file: PathLike, walker_path: Optional[PathLike] = None,
                     discard: int = 0, logl_min: Optional[float] = None, best: str = 'median',
                     n_draw: int = 50, output_dir: PathLike = '.', formats: Sequence[str] = ('png',),
                     seed: int = 0, ratio_pressures: Sequence[float] = (1e-4, 1e-2, 1e0),
                     equilibrium_pressure: float = 1e-2, equilibrium_mol_list: Sequence[str] = ('H2O', 'CO'),
                     ) -> Dict[str, List[Path]]:
    """Make and save every figure this run allows (see the module docstring for the list).

    Parameters
    ----------
    retrieval_obj : module
        The `starships.retrieval` module, already set up (`retrieval.setup_retrieval(yaml_file)`).
    walker_file : str or Path
        Walker file to read.
    walker_path : str or Path, optional
        Directory of `walker_file`, if it is not a full path already.
    discard : int, default 0
        Burn-in steps discarded at the start of each walker's chain.
    logl_min : float, optional
        Additional log-likelihood floor applied after `discard` (see `retrieval_analysis.select_samples`).
    best : {'median', 'best'}, default 'median'
        Which sample of the (post-selection) posterior is used for the single-parameter-set figures
        (model spectrum, contributions): the median-logL sample, or the single best one.
    n_draw : int, default 50
        Number of samples drawn for every confidence-interval figure (profiles, spectra, kernels).
    output_dir : str or Path, default '.'
        Directory the figures (and the saved model, if any) are written to.
    formats : sequence of str, default ('png',)
        File extensions to save each figure as (e.g. `('png', 'pdf')`).
    seed : int, default 0
        Seed shared by every random draw, so a re-run with the same inputs reproduces the same figures.
    ratio_pressures : sequence of float, default (1e-4, 1e-2, 1e0)
        Pressures (bar) compared in the abundance-ratio figure (single-region only).
    equilibrium_pressure : float, default 1e-2
        Pressure (bar) of the chemical-equilibrium fit (single-region only).
    equilibrium_mol_list : sequence of str, default ('H2O', 'CO')
        Species used to constrain the chemical-equilibrium fit (single-region only).

    Returns
    -------
    dict
        `{figure name: [saved paths]}` for every figure that was actually made (a section skipped
        because the run does not support it, or one that failed, has no entry).
    """
    output_dir = Path(output_dir)
    run = ra.describe_run(retrieval_obj)
    log.info(run.summary())

    walker_file = Path(walker_path) / Path(walker_file) if walker_path is not None else Path(walker_file)
    pos, logl, _ = ra.load_posterior_sample(walker_file, discard=discard)
    param_names = list(retrieval_obj.params_prior)
    if logl_min is not None:
        pos, logl = ra.select_samples(pos, logl, param_names=param_names, logl_min=logl_min)
    log.info(f'{len(logl)} samples kept for the figures (discard={discard}, logl_min={logl_min}).')

    if best not in ('median', 'best'):
        raise ValueError(f"`best`={best!r} not valid, choose 'median' or 'best'.")
    best_params = pos[len(pos) // 2] if best == 'median' else pos[-1]
    theta_regions = retrieval_obj.unpack_theta(best_params)
    labels = get_plot_labels(params=retrieval_obj.params_prior)
    setup_default_plot_params()

    saved: Dict[str, List[Path]] = {}
    attempted: List[str] = []

    def section(name):
        """Decorator-like helper: run a figure-making section, log and skip it on failure."""
        def register(fct):
            attempted.append(name)
            try:
                result = fct()
            except Exception:
                log.exception(f"Section '{name}' failed, skipping it.")
                return
            if result:
                saved[name] = result
        return register

    # --- Corner plot
    @section('corner')
    def _():
        fig, _flat = ru.plot_corner(pos[:, None, :], labels, param_no_zero=4, plot_datapoints=False)
        return _save(fig, output_dir, 'corner', formats)

    # --- Log-likelihood scatter plot
    @section('logl_scatter')
    def _():
        fig, _ax = scatterplot_logl(pos, logl, n_max_pts=10000, tight_ylim=True, alpha=0.1,
                                    color='k', ylabels=labels)
        return _save(fig, output_dir, 'logl_scatter', formats)

    # --- Model spectrum (+ save for logl_grid whenever a high-res model is possible)
    wv_low, spec_low, wv_high, spec_high = None, None, None, None

    @section('model_spectrum')
    def _():
        nonlocal wv_low, spec_low, wv_high, spec_high
        wv_low, spec_low, _r = retrieval_obj.prepare_static_model(theta_regions, 'low')
        paths = []
        fig, ax = plt.subplots()
        ax.plot(wv_low, spec_low * 1e6, label='c-k')
        if run.can_model_high:
            wv_high, spec_high, _r = retrieval_obj.prepare_static_model(theta_regions, 'high')
            ax.plot(wv_high, spec_high * 1e6, label='lbl', alpha=0.7)
            model_path = output_dir / 'planet_model.npz'
            retrieval_obj.save_planet_model(theta_regions, model_path)
            paths.append(model_path)
            log.info(f'Model saved for logl_grid: {model_path}')
        ax.set_xlabel('Wavelength [um]')
        ax.set_ylabel(f'{run.kind_trans} spectrum [ppm]')
        ax.legend()
        return _save(fig, output_dir, 'model_spectrum', formats) + paths

    # --- Comparison with low-resolution data
    @section('low_res_data')
    def _():
        if not run.has_low_res_data:
            log.info('No spectrophotometric/photometric data: skipping low_res_data.')
            return None
        synt_data = ru.get_syntetic_data_low_res(wv_low, spec_low, wv_high, spec_high,
                                                 retrieval_obj=retrieval_obj)
        n_subplots = len(retrieval_obj.spectrophotometric_data) + 1
        fig, ax = plt.subplots(n_subplots, 1, figsize=(8, 4 * n_subplots), squeeze=False)
        ax = ax[:, 0]
        for ax_i in ax:
            if spec_high is not None:
                ax_i.plot(wv_high, spec_high * 1e6, 'lightblue', zorder=-1)
            ax_i.plot(wv_low, spec_low * 1e6)
            ax_i.set_ylabel(f'{run.kind_trans} spectrum [ppm]')
        for i_instru, (name, infos) in enumerate(retrieval_obj.spectrophotometric_data.items()):
            x, y, yerr = infos['wave'], infos['data'] * 1e6, infos['err'] * 1e6
            for ax_i in [ax[i_instru], ax[-1]]:
                ax_i.errorbar(x, y, yerr=yerr, fmt='.', label=name)
            ax[i_instru].plot(x, synt_data[name] * 1e6, '.', label='Model binned')
            ax[i_instru].set_xlim(get_plot_limits_from_data(x))
            ax[i_instru].set_ylim(get_plot_limits_from_data(y, pad=0.2))
            ax[i_instru].legend(fontsize='small')
        for name, infos in retrieval_obj.photometric_data.items():
            x, y, yerr, xlim = (infos[key] for key in ['wave', 'data', 'err', 'wv_coverages'])
            xerr = np.abs(np.transpose(xlim) - x)
            ax[-1].errorbar(x, y * 1e6, yerr=yerr * 1e6, xerr=xerr, fmt='.', label=name)
        ax[-1].set_xlabel('Wavelength [um]')
        ax[-1].legend(fontsize='small')
        plt.tight_layout()
        return _save(fig, output_dir, 'low_res_data', formats)

    # --- Contribution of each species (one figure per region + the whole planet if multi-region)
    @section('contributions')
    def _():
        theta_by_name = {'global': theta_regions}
        if run.is_multi_region:
            theta_by_name.update({f'region_{r}': [t] for r, t in zip(run.region_ids, theta_regions)})
        paths = []
        for name, theta_list in theta_by_name.items():
            wv_c, spec_dict = ru.get_contribution(retrieval_obj.line_opacities, retrieval_obj, theta_list, mode='low')
            fig, _ax = plot_contributions(wv_c, spec_dict, run.kind_trans, figsize=(10, 16), title=name)
            paths += _save(fig, output_dir, f'contributions_{name}', formats)
        return paths

    # --- Profiles (temperature + abundances), one figure per region
    @section('profiles')
    def _():
        pressures, profile_samples_by_region = ra.draw_region_profiles(n_draw, pos, retrieval_obj, seed=seed)
        paths = []
        for region, samples in zip(run.region_ids, profile_samples_by_region):
            stats = ra.compute_profile_stats(samples)
            species = [key for key in samples if key != 'temperature']
            fig, ax = plt.subplots(1, 1 + len(species), sharey=True, figsize=(3 * (1 + len(species)), 4))
            plot_tp_sample(pressures, stats['temperature'], ax=ax[0])
            for ax_i, specie in zip(ax[1:], species):
                plot_p_profile_sample(pressures, stats[specie], ax=ax_i, label=specie)
                ax_i.set_xscale('log')
                ax_i.set_xlim(1e-12, 1e0)
                ax_i.legend(fontsize='x-small')
            fig.suptitle(f'Region {region}' if run.is_multi_region else None)
            plt.tight_layout()
            paths += _save(fig, output_dir, f'profiles_region{region}', formats)
        return paths

    # --- Spectrum confidence intervals, with the data
    @section('spectra_ci')
    def _():
        waves, spec_samples = ra.draw_spectra(n_draw, pos, retrieval_obj, kind_res='low', seed=seed)
        wv_s, spec_s = waves['global'], spec_samples['global']['All']
        stats = ru.get_stats_from_profile(spec_s)

        spec_high_samples, wv_high_s = None, None
        if 'high' in run.low_res_instruments.values():
            waves_h, spec_samples_h = ra.draw_spectra(n_draw, pos, retrieval_obj, kind_res='high', seed=seed)
            wv_high_s, spec_high_samples = waves_h['global'], spec_samples_h['global']['All']

        fig, ax = plt.subplots(figsize=(9, 4.5))
        plot_spectra_sample_GTC(wv_s, [stats], colorsOrder=['greens'], ax=ax, scale_factor=100)
        if run.has_low_res_data:
            synt_data = ra.synthetic_data_samples(spec_s, retrieval_obj, wave_low=wv_s,
                                                  spec_samples_high=spec_high_samples, wave_high=wv_high_s)
            for name, infos in {**retrieval_obj.spectrophotometric_data, **retrieval_obj.photometric_data}.items():
                x = infos['wave']
                y, yerr = infos['data'] * 100, infos['err'] * 100
                ax.errorbar(x, y, yerr=yerr, fmt='.', label=name)
        ax.set_xlabel('Wavelength [um]')
        ax.set_ylabel(r'F$_p$/F$_\star$ [%]')
        ax.legend(fontsize='small')
        return _save(fig, output_dir, 'spectra_ci', formats)

    # --- Rotation kernels (multi-region only)
    @section('rotation_kernels')
    def _():
        if not run.is_multi_region:
            log.info('Single-region run: skipping rotation_kernels.')
            return None
        phase = 0.5 if run.kind_trans == 'emission' else 0.0
        velocity, kernels = ra.draw_rotation_kernels(n_draw, pos, retrieval_obj, phase=phase, seed=seed)
        fig, ax = plt.subplots(figsize=(7, 4))
        for i_reg, region in enumerate(run.region_ids):
            stats = ru.get_stats_from_profile(kernels[:, i_reg, :])
            ax.plot(velocity, stats['median'], label=f'Region {region}')
            ax.fill_between(velocity, *stats['1-sig'], alpha=0.3)
        ax.set_xlabel('Velocity [km/s]')
        ax.set_ylabel('Kernel')
        ax.set_title(f'Phase {phase}')
        ax.legend(fontsize='small')
        return _save(fig, output_dir, 'rotation_kernels', formats)

    # --- Abundance ratios (single-region only, see retrieval_analysis's module docstring)
    @section('abundance_ratios')
    def _():
        if run.is_multi_region:
            log.info('Multi-region run: skipping abundance_ratios (single-region only for now).')
            return None
        pressures, profile_samples_by_region = ra.draw_region_profiles(n_draw, pos, retrieval_obj, seed=seed)
        profile_samples = profile_samples_by_region[0]

        ratios = {'C/O': ra.abundance_ratio_profiles(profile_samples, 'C/O')}
        is_dex = {'C/O': False}
        for r in ['C/H', 'O/H', 'Fe/H']:
            atom_num = r.split('/')[0]
            if atom_num in profile_samples or atom_num == 'H':
                ratios[r] = ra.metallicity_profiles(profile_samples, r)
                is_dex[r] = True

        fig, ax = plt.subplots(1, len(ratios), figsize=(4 * len(ratios), 4), squeeze=False)
        ax = ax[0]
        for ax_i, (name, profile) in zip(ax, ratios.items()):
            stats_by_p = ra.profile_stats_at_pressures(profile, pressures, ratio_pressures,
                                                       log_scale=not is_dex[name])
            xs = np.arange(len(ratio_pressures))
            medians = np.array([stats_by_p[p]['median'] for p in ratio_pressures])
            err1 = np.array([stats_by_p[p]['1-sig'] for p in ratio_pressures])
            ax_i.errorbar(xs, medians, yerr=[medians - err1[:, 0], err1[:, 1] - medians], fmt='o', capsize=4)
            ax_i.set_xticks(xs)
            ax_i.set_xticklabels([f'{p:.0e}' for p in ratio_pressures])
            ax_i.set_title(name)
            if is_dex[name]:
                ax_i.axhline(0., linestyle='--', color='grey')
        plt.tight_layout()
        paths = _save(fig, output_dir, 'abundance_ratios', formats)

        # The fit falls back to a solar metallicity estimate when 'Fe' isn't a fitted species, so
        # it is always attempted here (see fit_chemical_equilibrium_at_pressure's docstring).
        eq_fit = ra.fit_chemical_equilibrium_at_pressure(profile_samples, pressures, equilibrium_pressure,
                                                         mol_list=list(equilibrium_mol_list))
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        ax[0].hist(eq_fit['C_to_O'], bins=20, color='C0')
        ax[0].set_xlabel('Equilibrium C/O')
        ax[1].hist(eq_fit['metallicity'], bins=20, color='C2')
        ax[1].set_xlabel('Equilibrium metallicity [dex]')
        plt.tight_layout()
        paths += _save(fig, output_dir, 'chemical_equilibrium', formats)
        return paths

    skipped = [name for name in attempted if name not in saved]
    log.info(f"Done: {len(saved)}/{len(attempted)} sections produced ({', '.join(saved)})."
             + (f" Skipped: {', '.join(skipped)}." if skipped else ''))
    return saved


def main() -> None:
    """Entry point for ``run_starships_retrieval_figures`` (registered in setup.py)."""
    parser = argparse.ArgumentParser(
        prog='run_starships_retrieval_figures',
        description='Make every retrieval-analysis figure automatically from a walker file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--yaml', required=True, type=Path,
                       help='Retrieval input YAML (~ is expanded).')
    parser.add_argument('--walker-file', required=True, type=Path,
                       help='Walker file to read (~ is expanded). If not a full path, '
                            'combined with --walker-path or the yaml file\'s own walker_path.')
    parser.add_argument('--walker-path', type=Path, default=None,
                       help='Directory of --walker-file, if it is not a full path already. '
                            'Default: the yaml file\'s own walker_path.')
    parser.add_argument('--discard', type=int, default=0,
                       help='Burn-in steps discarded at the start of each walker.')
    parser.add_argument('--logl-min', type=float, default=None,
                       help='Additional log-likelihood floor applied after --discard.')
    parser.add_argument('--best', choices=['median', 'best'], default='median',
                       help="Which sample is used for the single-parameter-set figures.")
    parser.add_argument('--n-draw', type=int, default=50,
                       help='Number of samples drawn for every confidence-interval figure.')
    parser.add_argument('--output-dir', type=Path, default=Path('.'),
                       help='Directory the figures (and the saved model, if any) are written to.')
    parser.add_argument('--format', dest='formats', nargs='+', default=['png'],
                       help='File extensions to save each figure as.')
    parser.add_argument('--seed', type=int, default=0,
                       help='Seed shared by every random draw.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    from . import retrieval  # imported here: this triggers heavy imports (petitRADTRANS, ...)
    retrieval.setup_retrieval(args.yaml.expanduser())

    walker_path = args.walker_path.expanduser() if args.walker_path else retrieval.walker_path
    make_all_figures(retrieval, args.walker_file.expanduser(), walker_path=walker_path,
                     discard=args.discard, logl_min=args.logl_min, best=args.best,
                     n_draw=args.n_draw, output_dir=args.output_dir.expanduser(),
                     formats=args.formats, seed=args.seed)


if __name__ == '__main__':
    main()
