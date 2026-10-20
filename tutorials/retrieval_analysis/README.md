# Retrieval analysis tutorials

Interactive notebooks to look at the output of a retrieval (`starships.retrieval`): inspect the
walker chains, choose which steps/walkers to keep, generate models and figures from the result.
Built on `starships.retrieval_analysis` (the reusable library) and `starships.plotting_fcts`.
Needs petitRADTRANS and a real walker file for most sections -- the walker-reading/selection
functions themselves don't, see `starships.retrieval_analysis`'s own docstring for what does.

- `analyse_retrieval.ipynb`: single-region retrievals. Adapts to what the run actually has
  (high-res only, low-res only, joint; with or without photometry) -- the loading cell prints
  what was detected, and sections that need something the run doesn't have are skipped. Covers:
  walker selection, the model spectrum (and saving it for `logl_grid_examples/`, including one
  model per line-opacity species on request), comparison with low-resolution data, species
  contributions, temperature/abundance profile confidence intervals, spectrum confidence
  intervals, and abundance ratios (C/O, [X/H], chemical equilibrium comparison).
- `analyse_retrieval_multi_region.ipynb`: same, for a multi-region model (e.g. day/night,
  hotspot) -- a superset of the single-region notebook's sections, plus the ones specific to
  comparing regions: rotation kernels, profiles/spectra/contributions region by region versus
  the combined (global) one. Abundance ratios are not implemented for multi-region models yet.

Both optionally load a `starships.retrieval_optimize` best-fit (see
`retrieval_optimize_examples/`) to compare its search trace with the MCMC chain's own
log-likelihoods and to generate a model from a genuine maximum-a-posteriori fit instead of a
walker sample -- off by default (`use_optimizer_result = False`).

For the automatic, non-interactive equivalent (every figure from a walker file in one command,
no manual chain inspection), see `starships.retrieval_figures` / `run_starships_retrieval_figures`.

Still a first pass (built/validated against real KELT-20b data on Narval, not yet polished with
further feedback) -- expect more comments from Antoine.
