"""Fit/plot/extrapolate helpers for the SLURM resource-estimation sweep.

These are plain functions of a ``results: List[dict]`` structure -- they do
not care *how* that data was measured, only that each entry has the keys
listed in ``estimate_run_resources``'s docstring. :mod:`starships.slurm.parse`
is the piece that builds ``results`` (by combining
``sacct`` and each job's log), and calls into these functions. No dependency
on the rest of ``starships`` here (no ``petitRADTRANS``, no retrieval
machinery) -- this module is plain ``numpy``/``matplotlib``, usable
standalone if you just want the fitting/plotting logic.
"""
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Applied to every memory estimate before it is reported -- real runs tend to
# run a little hotter than a short sanity sweep (longer runs, more walkers
# touching more of the parameter space, etc.), so a flat safety margin is
# cheaper than being surprised by an OOM'd job. Adjust here if it turns out to
# be too conservative (or not conservative enough) in practice.
DEFAULT_MEMORY_SAFETY_MARGIN = 1.5


def fit_linear(x: Sequence[float], y: Sequence[float]) -> Optional[Tuple[float, float]]:
    """Least-squares line ``y = slope * x + intercept`` through ``(x, y)``.

    Returns ``None`` if fewer than 2 points are given (a line cannot be fit
    through a single point). Used by ``plot_resource_scaling`` and
    ``estimate_run_resources`` to extrapolate both memory and per-iteration
    time from the points actually measured in a sweep to an ``n_cpu`` that
    was not tested.

    Parameters
    ----------
    x, y : sequence of float
        Coordinates of the measured points (e.g. ``n_cpu`` and
        ``max_rss_mb``).

    Returns
    -------
    (slope, intercept) or None
    """
    if len(x) < 2:
        return None
    slope, intercept = np.polyfit(np.asarray(x, dtype=float), np.asarray(y, dtype=float), 1)
    return float(slope), float(intercept)


def estimate_run_resources(results: List[dict], n_walkers_target: int, n_steps: int,
                           extra_n_cpu: Sequence[int] = (),
                           memory_safety_margin: float = DEFAULT_MEMORY_SAFETY_MARGIN,
                           ) -> List[dict]:
    """Estimate wall time and memory for a real MCMC run, for every ``n_cpu``
    tested in the sweep plus any ``extra_n_cpu`` values -- the practical
    question this whole sweep exists to answer ("if I request 16 or 32 CPUs
    for the real run, what ``--mem``/``--time`` do I need?"), not just a plot.

    Wall time: walkers are split round-robin across ``n_cpu`` parallel
    workers, ``ceil(n_walkers_target / n_cpu)`` walkers each, run sequentially
    within a worker over ``n_steps`` MCMC steps. Since all workers run in
    parallel, wall time is set by the busiest one:
    ``n_walkers_per_cpu * n_steps * avg_time_per_iter_s``.

    .. note::
       This uses one flat ``avg_time_per_iter_s`` per ``n_cpu`` (elapsed Pool
       time divided by the number of calls made during the sweep) rather than
       separating out a one-time per-worker warm-up cost -- a deliberate
       simplification versus tracking every individual call's timing. It is a
       slight overestimate for a real run with many steps (where the one-time
       warm-up cost is amortized over far more calls than the short sweep
       used), and folded into by the safety margin below in practice.

    Memory: for an ``n_cpu`` that was actually tested, use the measured
    ``max_rss_mb`` directly. For an untested ``n_cpu`` (from ``extra_n_cpu``),
    extrapolate via ``fit_linear`` through the measured ``(n_cpu, max_rss_mb)``
    points (needs >=2 sweep points; falls back to the closest tested point's
    value with only 1). Memory does *not* scale linearly with ``n_cpu`` in
    general -- large shared arrays (e.g. petitRADTRANS opacity tables) may be
    shared copy-on-write across forked workers, so total memory can grow much
    slower than ``n_cpu`` -- but a straight line through what was *actually
    measured* is a more honest estimate than assuming either "flat" or
    "fully duplicated per worker" ahead of time. Widen the sweep (test more
    ``n_cpu`` values) if the extrapolation needs to reach far past what was
    measured.

    Parameters
    ----------
    results : list of dict
        One entry per ``n_cpu`` tested by the sweep (built by
        :mod:`starships.slurm.parse`), each with keys ``n_cpu``,
        ``max_rss_mb``, ``avg_time_per_iter_s``.
    n_walkers_target : int
        Total number of walkers the real run will use.
    n_steps : int
        Total number of MCMC steps the real run will take (e.g.
        ``n_steps_burnin + n_steps_sampling``).
    extra_n_cpu : sequence of int, optional
        ``n_cpu`` values to also estimate for, beyond what the sweep tested
        (e.g. 16, 32 for a bigger real run than was actually swept).
    memory_safety_margin : float, optional
        Multiplicative safety margin applied to every memory estimate (see
        module-level ``DEFAULT_MEMORY_SAFETY_MARGIN``).

    Returns
    -------
    list of dict
        One entry per candidate ``n_cpu`` (measured ones first, in the order
        given in ``results``, then ``extra_n_cpu`` in the order given): keys
        ``n_cpu``, ``measured`` (bool), ``n_walkers_per_cpu``,
        ``est_wall_time_s``, ``est_memory_mb`` (already includes the safety
        margin).
    """
    results_sorted = sorted(results, key=lambda e: e['n_cpu'])
    tested_n_cpu = [entry['n_cpu'] for entry in results_sorted]
    mem_fit = fit_linear(tested_n_cpu, [e['max_rss_mb'] for e in results_sorted])
    time_fit = fit_linear(tested_n_cpu, [e['avg_time_per_iter_s'] for e in results_sorted])

    estimates = []
    for n_cpu in list(tested_n_cpu) + [n for n in extra_n_cpu if n not in tested_n_cpu]:
        measured = n_cpu in tested_n_cpu
        closest = min(results_sorted, key=lambda e: abs(e['n_cpu'] - n_cpu))

        if measured:
            entry = next(e for e in results_sorted if e['n_cpu'] == n_cpu)
            memory_mb = entry['max_rss_mb']
            avg_time_per_iter_s = entry['avg_time_per_iter_s']
        else:
            if mem_fit is not None:
                slope, intercept = mem_fit
                memory_mb = max(intercept + slope * n_cpu, 0.0)
            else:
                memory_mb = closest['max_rss_mb']
            if time_fit is not None:
                slope, intercept = time_fit
                avg_time_per_iter_s = max(intercept + slope * n_cpu, 0.0)
            else:
                avg_time_per_iter_s = closest['avg_time_per_iter_s']

        n_walkers_per_cpu = int(np.ceil(n_walkers_target / n_cpu))
        est_wall_time_s = n_walkers_per_cpu * n_steps * avg_time_per_iter_s

        estimates.append(dict(n_cpu=n_cpu, measured=measured,
                              n_walkers_per_cpu=n_walkers_per_cpu,
                              est_wall_time_s=est_wall_time_s,
                              est_memory_mb=memory_mb * memory_safety_margin))
    return estimates


def plot_resource_scaling(results: List[dict], plot_path: str,
                          available_cpus: Optional[int] = None,
                          extra_n_cpu: Sequence[int] = ()) -> List[str]:
    """Save a 2-panel figure summarizing a resource-estimation sweep.

    Left panel: peak memory (``max_rss_mb``, from ``sacct``'s ``MaxRSS`` --
    real, ground-truth, cgroup-based physical memory for the whole job,
    parent + every forked worker, with no risk of double-counting shared
    pages) in GB, vs ``n_cpu`` -- the number relevant to sizing a real run's
    ``--mem`` request. A "naive scaling" reference line (``base * n_cpu /
    base_n_cpu``, what you'd expect with *no* memory shared between workers)
    and a linear fit through the actual measured points (see ``fit_linear``,
    only drawn with >=2 points -- the empirically observed rate, which can
    fall anywhere between "flat" and "naive scaling" depending on how much
    really is shared copy-on-write) are both extended back to ``n_cpu=0``: the
    fit's intercept there estimates the *shared* memory floor (independent of
    ``n_cpu``), marked with a star.

    Right panel: average wall time per ``lnprob()`` call per process
    (``avg_time_per_iter_s``) vs ``n_cpu``, with the same kind of linear fit.
    A rising trend here is the signature of CPU contention (more processes
    than physical cores actually help with, or memory-bandwidth/cache
    pressure between workers) -- useful context if testing oversubscription
    (``n_cpu`` beyond ``available_cpus``) later.

    If ``available_cpus`` is given, both panels get a shaded region marking
    ``n_cpu > available_cpus`` (oversubscription: real contention on those
    cores, not genuine additional parallelism) -- not used by the sweep
    itself yet, but left in since testing oversubscription on purpose is a
    natural next use of this same sweep mechanism (see the package
    docstring).

    If ``extra_n_cpu`` is given (``n_cpu`` values not tested by the sweep,
    e.g. the ``n_cpu`` you actually intend to run the real retrieval at), a
    **second** figure is also saved, next to the first with ``_extrapolated``
    inserted before the extension -- zoomed out far enough to show each of
    those points, each marked distinctly (diamond, annotated with its
    projected value) from the measured points. The first (default) figure
    always stays zoomed to just the measured range, so it stays readable even
    when a requested ``extra_n_cpu`` is far beyond what was tested -- see
    Antoine, 2026-09-17: "une version zoomed-out incluant la valeur estimée et
    une avec les valeurs calculées." Values already present in ``results`` are
    silently skipped (already shown as a measured point, nothing to project).

    Parameters
    ----------
    results : list of dict
        As built by :mod:`starships.slurm.parse` -- see
        ``estimate_run_resources``'s docstring for the expected keys.
    plot_path : str
        Where to save the (zoomed, measured-range-only) figure (PNG).
    available_cpus : int, optional
        Number of physical CPUs actually available per job, if known -- draws
        the oversubscription boundary described above.
    extra_n_cpu : sequence of int, optional
        ``n_cpu`` values to also project and mark on a second, zoomed-out
        figure (see above). Projected the same way as
        ``estimate_run_resources`` (linear fit through the measured points,
        or the closest measured point's value with fewer than 2 points to
        fit) -- but memory/time only, independent of any particular real-run
        size (no ``n_walkers``/``n_steps`` needed here).

    Returns
    -------
    list of str
        Path(s) actually written: always ``[plot_path]``, plus the zoomed-out
        path if ``extra_n_cpu`` produced anything to project.
    """
    # Imported here, not at module level, so importing this module for its
    # fitting/estimation functions alone (e.g. from a quick interactive
    # session) never requires matplotlib or forces a backend choice on the
    # caller. 'Agg' is the non-interactive, file-only backend -- appropriate
    # here since this only ever runs on a headless Narval node to save a PNG,
    # never to pop up a window.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    results_sorted = sorted(results, key=lambda e: e['n_cpu'])
    n_cpus = np.array([e['n_cpu'] for e in results_sorted], dtype=float)

    # Zoomed figure: just the measured range, as before.
    fig = _render_scaling_figure(results_sorted, n_cpus, x_extrap_max=n_cpus.max() * 1.3,
                                 available_cpus=available_cpus, project_n_cpu=())
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    log.info('Saved resource-scaling plot to %s', plot_path)
    saved_paths = [plot_path]

    # Zoomed-out figure: only produced if there is something to project.
    project_n_cpu = sorted(n for n in extra_n_cpu if n not in n_cpus)
    if project_n_cpu:
        x_extrap_max = max(n_cpus.max() * 1.3, max(project_n_cpu) * 1.15)
        fig = _render_scaling_figure(results_sorted, n_cpus, x_extrap_max=x_extrap_max,
                                     available_cpus=available_cpus,
                                     project_n_cpu=project_n_cpu)
        extrapolated_path = _insert_suffix(plot_path, '_extrapolated')
        fig.savefig(extrapolated_path, dpi=150)
        plt.close(fig)
        log.info('Saved zoomed-out (extrapolated) resource-scaling plot to %s',
                 extrapolated_path)
        saved_paths.append(extrapolated_path)

    return saved_paths


def _insert_suffix(path: str, suffix: str) -> str:
    """'foo/bar.png' -> 'foo/bar<suffix>.png'."""
    p = Path(path)
    return str(p.with_name(p.stem + suffix + p.suffix))


def _render_scaling_figure(results_sorted: List[dict], n_cpus: np.ndarray, x_extrap_max: float,
                           available_cpus: Optional[int], project_n_cpu: Sequence[int]):
    """Build one 2-panel scaling figure (not saved) -- shared by both the
    zoomed and zoomed-out figures ``plot_resource_scaling`` produces, so the
    two never drift apart in what they draw beyond their x-range and
    projected-point markers."""
    import matplotlib.pyplot as plt

    fig, (ax_mem, ax_time) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left panel: peak memory (GB) vs n_cpu ---
    max_rss_gb = np.array([e['max_rss_mb'] for e in results_sorted]) / 1024.0
    base_n_cpu, base_rss_gb = n_cpus[0], max_rss_gb[0]

    x_naive = np.linspace(0, x_extrap_max, 50)
    y_naive = base_rss_gb * (x_naive / base_n_cpu)
    ax_mem.plot(x_naive, y_naive, '--', color='gray',
                label='naive scaling (no sharing, base × n_cpu)', zorder=1)

    mem_fit = fit_linear(n_cpus, max_rss_gb * 1024.0)  # fit in MB, plot in GB
    if mem_fit is not None:
        slope_mb, intercept_mb = mem_fit
        slope_gb, intercept_gb = slope_mb / 1024.0, intercept_mb / 1024.0
        x_lin = np.array([0, x_extrap_max])
        ax_mem.plot(x_lin, intercept_gb + slope_gb * x_lin, '-.', color='C4',
                    label=f'linear fit (shared floor ≈{intercept_gb:.1f} GB)', zorder=2)
        ax_mem.scatter([0], [intercept_gb], marker='*', s=140, color='C4', zorder=4)

    ax_mem.plot(n_cpus, max_rss_gb, 'o-', color='C1', label='measured (sacct MaxRSS)', zorder=3)

    _mark_projected_points(ax_mem, n_cpus, max_rss_gb * 1024.0, project_n_cpu, mem_fit,
                           value_scale=1 / 1024.0, fmt='{:.1f} GB')
    _mark_oversubscription(ax_mem, available_cpus, x_extrap_max)
    ax_mem.set_xlabel('n_cpu (parallel processes requested)')
    ax_mem.set_ylabel('peak memory (GB) -- sacct MaxRSS, whole job')
    ax_mem.set_title('Memory vs parallelism\n(size --mem from the linear-fit line)')
    ax_mem.set_xticks(np.unique(np.concatenate([[0], n_cpus, project_n_cpu])))
    ax_mem.legend(fontsize=7, loc='upper left')

    # --- Right panel: average time per iteration (per process) vs n_cpu ---
    avg_time_s = np.array([e['avg_time_per_iter_s'] for e in results_sorted])
    ax_time.plot(n_cpus, avg_time_s, 'o-', color='C2', label='measured', zorder=3)

    time_fit = fit_linear(n_cpus, avg_time_s)
    if time_fit is not None:
        slope, intercept = time_fit
        x_lin = np.array([0, x_extrap_max])
        ax_time.plot(x_lin, np.maximum(intercept + slope * x_lin, 0.0), '-.', color='C4',
                     label='linear fit', zorder=2)

    _mark_projected_points(ax_time, n_cpus, avg_time_s, project_n_cpu, time_fit,
                           value_scale=1.0, fmt='{:.1f}s')
    _mark_oversubscription(ax_time, available_cpus, x_extrap_max)
    ax_time.set_xlabel('n_cpu (parallel processes requested)')
    ax_time.set_ylabel('avg. lnprob() time per process (s)')
    ax_time.set_title('Per-iteration cost vs parallelism\n(rising = CPU contention)')
    ax_time.set_xticks(np.unique(np.concatenate([n_cpus, project_n_cpu])))
    ax_time.legend(fontsize=8)

    fig.tight_layout()
    return fig


def _mark_projected_points(ax, n_cpus: np.ndarray, values, project_n_cpu: Sequence[int],
                           linear_fit: Optional[Tuple[float, float]], value_scale: float,
                           fmt: str) -> None:
    """Mark each n_cpu in project_n_cpu with a diamond at its fit-projected
    value (falling back to the closest measured value if there are fewer than
    2 points to fit -- same rule as estimate_run_resources), annotated with
    that value. `values` are in the same units `linear_fit` was computed in
    (e.g. MB); `value_scale` converts to the axes' displayed units (e.g. GB)
    for both the marker and its annotation.
    """
    if not project_n_cpu:
        return
    labelled = False
    for n in project_n_cpu:
        if linear_fit is not None:
            slope, intercept = linear_fit
            projected = max(intercept + slope * n, 0.0)
        else:
            closest_idx = int(np.argmin(np.abs(n_cpus - n)))
            projected = values[closest_idx]
        y = projected * value_scale
        ax.scatter([n], [y], marker='D', s=70, color='C3', zorder=5,
                   label=None if labelled else 'projected (n_cpu requested, not tested)')
        ax.annotate(fmt.format(y), (n, y), textcoords='offset points', xytext=(6, 6), fontsize=8,
                   color='C3')
        labelled = True


def _mark_oversubscription(ax, available_cpus: Optional[int], x_extrap_max: float) -> None:
    """Shade n_cpu > available_cpus on the given axes, if available_cpus is known."""
    if available_cpus is None:
        return
    ax.axvline(available_cpus, linestyle=':', color='red', alpha=0.6, zorder=2)
    if x_extrap_max > available_cpus:
        ax.axvspan(available_cpus, x_extrap_max, color='red', alpha=0.05,
                   label=f'> available CPUs ({available_cpus}): real contention', zorder=0)


def plot_ensemble_efficiency(results: List[dict], plot_path: str,
                             baseline_time_per_iter_s: Optional[float] = None,
                             n_steps_target: Optional[int] = None,
                             n_walkers_target: Optional[int] = None,
                             node_mem_gb: Optional[float] = None) -> None:
    """Save a 2-panel figure for a ``retrieval``-workload sweep (real
    ``emcee.EnsembleSampler`` runs at a fixed ``n_cpu``, varying
    ``n_walkers_per_cpu`` -- see :mod:`starships.slurm.submit`'s docstring).

    Left panel: peak memory (GB, ``sacct`` ``MaxRSS``) vs ``n_walkers_per_cpu``.
    This whole sweep is fundamentally a single-node resource-optimization
    question (Antoine, 2026-09-17): ``multiprocessing.Pool`` only shares
    memory within one node, so both ``n_cpu`` and total memory are capped by
    whatever one node has (a standard Narval node: 64 cores, 250 GB) --
    crossing that line means a different, unsupported parallelization scheme
    (e.g. MPI across nodes), not just a bigger ``--mem`` request. If
    ``node_mem_gb`` is given, a shaded region above it marks memory that is
    NOT available on a single node.

    Right panel: efficiency (%) vs ``n_walkers_per_cpu`` -- NOT raw time per
    step, which is not the useful quantity here (Antoine, 2026-09-17): a step
    with more walkers per cpu does more real work, so of course it takes
    longer -- that alone says nothing about whether the extra walkers were
    "worth it". Efficiency is ``ideal / measured``, where ``ideal =
    n_walkers_per_cpu * baseline_time_per_iter_s`` is what a step would cost
    with zero inter-worker synchronization overhead (every walker must finish
    before the ensemble can propose the next step) -- so it already factors
    out "more walkers = more work" and isolates the actual synchronization
    overhead. A rising, saturating efficiency curve is the "sweet spot"
    signature: below the knee, adding walkers per cpu is nearly free
    (overhead shrinks fast); past it, efficiency gains flatten out, so
    further increases mostly just add real compute time for comparatively
    little overhead improvement.

    If ``n_steps_target`` AND ``n_walkers_target`` are also given, a second
    line (secondary y-axis) shows the *estimated total wall time* for a real
    run of ``n_steps_target`` steps at a FIXED ``n_walkers_target`` -- NOT at
    whatever ``n_walkers`` this sweep's ``n_cpu`` happened to test (Antoine,
    2026-09-17): in practice the number of walkers is the real constraint
    (MCMC needs "enough" walkers, chosen independently of how many cpus you
    have), so the practical question is "for MY fixed walker count, what
    ``n_cpu`` should I actually request?" -- not "what happens if I grow the
    ensemble alongside n_cpu" (which is what this sweep's own tested
    ``n_walkers`` do, and would give a misleadingly steep, ever-climbing
    time curve that has nothing to do with the walkers-per-cpu trade-off
    itself). For each ``n_walkers_per_cpu`` tested, the implied
    ``n_cpu = n_walkers_target / n_walkers_per_cpu`` is computed, and its
    time per step is estimated as ``n_walkers_per_cpu * baseline_time_per_iter_s
    / efficiency`` (assuming the efficiency measured at this
    ``n_walkers_per_cpu`` transfers to other ``n_cpu`` -- untested beyond this
    sweep's own fixed ``n_cpu``, but the closest estimate available from a
    single sweep). Reading both curves together answers "how do I optimize
    resources and compute time for a given (large, MCMC-favoring) number of
    walkers": pick a point past the efficiency knee, before the total-time
    line climbs further than you're willing to pay for. This total-time
    curve is generally DEcreasing in ``n_walkers_per_cpu`` here (fewer cpus
    for the same fixed walker count = more serial work per step, but that can
    still lose to raw added parallelism from more cpus) -- the opposite
    direction from CPU-hours: since CPU-hours spent = ``n_cpu *
    wall_time_per_step * n_steps = n_walkers_target * baseline_time_per_iter_s
    * n_steps / efficiency``, CPU-hours are *inversely* proportional to
    efficiency alone (independent of how the fixed walker count is split) --
    so maximizing efficiency minimizes CPU-hours/cost, while minimizing wall
    time favours more cpus (lower ``n_walkers_per_cpu``) even at worse
    efficiency. These two curves do not necessarily agree on the same
    "best" point -- that tension (faster vs. cheaper) is the actual trade-off
    to weigh, not something this plot resolves for you.

    Parameters
    ----------
    results : list of dict
        As built by :mod:`starships.slurm.parse`'s ``build_ensemble_results``
        -- keys ``n_walkers_per_cpu``, ``max_rss_mb``, ``time_per_step_s``.
    plot_path : str
        Where to save the figure (PNG).
    baseline_time_per_iter_s : float, optional
        Pure per-``lnprob()``-call cost, for the efficiency comparison
        described above. Without it, the right panel falls back to the raw
        ``time_per_step_s`` (still useful as a sanity check, just not the
        "is this worth it" answer).
    n_steps_target : int, optional
        Real MCMC steps to estimate total wall time for -- needs
        ``n_walkers_target`` and ``baseline_time_per_iter_s`` too (see above
        for why a *fixed* walker count, not this sweep's own tested
        ``n_walkers``).
    n_walkers_target : int, optional
        The walker count to hold fixed for that total-time estimate (see
        above) -- typically the number of walkers you actually intend to use
        for a real run, independent of this sweep's own ``n_cpu``.
    node_mem_gb : float, optional
        Total RAM of one compute node (GB) -- shades memory above it on the
        left panel as unavailable within a single node (see above).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    results_sorted = sorted(results, key=lambda e: e['n_walkers_per_cpu'])
    wpc = np.array([e['n_walkers_per_cpu'] for e in results_sorted], dtype=float)
    time_per_step = np.array([e['time_per_step_s'] for e in results_sorted])

    # This sweep may have held n_cpu fixed (varying n_walkers_per_cpu) or
    # n_walkers fixed (varying n_cpu, so n_walkers_per_cpu itself is derived,
    # possibly non-integer) -- see submit.py's two submit_retrieval_sweep*
    # functions. Title text reflects whichever was actually fixed.
    n_cpu_values = {e['n_cpu'] for e in results_sorted}
    n_walkers_values = {e['n_walkers'] for e in results_sorted}
    if len(n_cpu_values) == 1:
        fixed_label = f'n_cpu={n_cpu_values.pop()} fixed'
    elif len(n_walkers_values) == 1:
        fixed_label = f'n_walkers={n_walkers_values.pop()} fixed'
    else:
        fixed_label = 'neither n_cpu nor n_walkers fixed'

    show_cost_panel = n_steps_target is not None and n_walkers_target is not None
    fig, axes = plt.subplots(1, 3 if show_cost_panel else 2, figsize=(18 if show_cost_panel else 12, 5))
    ax_mem, ax_eff = axes[0], axes[1]

    # --- Panel 1: peak memory (GB) vs n_walkers_per_cpu ---
    max_rss_gb = np.array([e['max_rss_mb'] for e in results_sorted]) / 1024.0
    ax_mem.plot(wpc, max_rss_gb, 'o-', color='C1', label='measured (sacct MaxRSS)')
    ax_mem.set_xlabel('n_walkers_per_cpu')
    ax_mem.set_ylabel('peak memory (GB) -- sacct MaxRSS, whole job')
    ax_mem.set_title(f'Memory vs walkers/cpu ({fixed_label})')
    ax_mem.set_xticks(wpc)
    if node_mem_gb is not None:
        y_top = max(node_mem_gb * 1.1, max_rss_gb.max() * 1.1)
        ax_mem.axhline(node_mem_gb, linestyle=':', color='red', alpha=0.6,
                       label=f'single-node limit ({node_mem_gb:.0f} GB)')
        ax_mem.axhspan(node_mem_gb, y_top, color='red', alpha=0.05)
        ax_mem.set_ylim(0, y_top)
    ax_mem.legend(fontsize=8)

    # --- Panel 2: efficiency (%) vs n_walkers_per_cpu, the "sweet spot" curve ---
    if baseline_time_per_iter_s is None:
        # Fallback: no baseline to compute efficiency against -- show the raw
        # measurement rather than nothing, but this does NOT answer "is it
        # worth it" (see docstring).
        ax_eff.plot(wpc, time_per_step, 'o-', color='C2', label='measured (whole job / n_steps)')
        ax_eff.set_ylabel('time per real MCMC step (s) -- raw, no baseline given')
        ax_eff.set_title('Time per step vs walkers/cpu\n(pass baseline_time_per_iter_s for '
                         'efficiency instead)')
        efficiency_pct = None
    else:
        ideal = wpc * baseline_time_per_iter_s
        efficiency_pct = 100 * ideal / time_per_step
        ax_eff.plot(wpc, efficiency_pct, 'o-', color='C3', label='efficiency (ideal / measured)')
        ax_eff.axhline(100, linestyle=':', color='gray', alpha=0.6, label='100% (no overhead)')
        ax_eff.set_ylabel('efficiency (%) -- ideal / measured time per step')
        ax_eff.set_ylim(0, max(105, efficiency_pct.max() * 1.1))
        ax_eff.set_title('Efficiency vs walkers/cpu\n(saturating = the "sweet spot" past the knee)')
        ax_eff.legend(fontsize=8)

    ax_eff.set_xlabel('n_walkers_per_cpu')
    ax_eff.set_xticks(wpc)

    # --- Panel 3 (only with both targets given): wall time AND CPU-hours for
    # a FIXED (n_walkers_target, n_steps_target) -- see docstring for why
    # these are plotted for a fixed walker count, not this sweep's own
    # n_walkers, and for why they move in OPPOSITE directions. ---
    if show_cost_panel:
        ax_cost = axes[2]
        # n_cpu implied by each n_walkers_per_cpu tested, for the FIXED
        # n_walkers_target (not this sweep's own n_cpu/n_walkers).
        implied_n_cpu = n_walkers_target / wpc
        time_per_step_est = wpc * baseline_time_per_iter_s / (efficiency_pct / 100.0)
        est_total_h = time_per_step_est * n_steps_target / 3600.0
        # CPU-hours = n_cpu * wall_time -- algebraically this reduces to
        # (n_walkers_target * baseline_time_per_iter_s * n_steps_target) /
        # efficiency, i.e. exactly the efficiency curve again, just inverted
        # and rescaled by these three constants (Antoine, 2026-09-17) -- shown
        # here as a concrete number (what you'd actually be billed) rather
        # than an abstract %.
        cpu_hours = implied_n_cpu * est_total_h

        line_wall = ax_cost.plot(wpc, est_total_h, 's--', color='C4', label='est. wall time (h)')
        ax_cost.set_ylabel('est. wall time (h)', color='C4')
        ax_cost.tick_params(axis='y', labelcolor='C4')
        for x, y, n_cpu_implied in zip(wpc, est_total_h, implied_n_cpu):
            ax_cost.annotate(f'n_cpu≈{n_cpu_implied:.0f}', (x, y), textcoords='offset points',
                            xytext=(6, -10), fontsize=7, color='C4')

        ax_cpuh = ax_cost.twinx()
        line_cpuh = ax_cpuh.plot(wpc, cpu_hours, '^-', color='C5', label='est. CPU-hours')
        ax_cpuh.set_ylabel('est. CPU-hours (billed)', color='C5')
        ax_cpuh.tick_params(axis='y', labelcolor='C5')

        lines = line_wall + line_cpuh
        ax_cost.legend(lines, [l.get_label() for l in lines], fontsize=8, loc='upper center')
        ax_cost.set_xlabel('n_walkers_per_cpu')
        ax_cost.set_xticks(wpc)
        ax_cost.set_title(f'Cost vs walkers/cpu: {n_steps_target} steps, '
                         f'{n_walkers_target} walkers fixed\n(wall time ↑ but CPU-hours ↓ with '
                         f'walkers/cpu -- speed vs. cost)')

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    log.info('Saved ensemble-efficiency plot to %s', plot_path)
