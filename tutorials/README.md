# STARSHIPS tutorials

This is the canonical home for curated, user-facing tutorial notebooks — as opposed to the
exploratory / working notebooks scattered across `notebook_examples/`, `wasp33_examples/`,
`wasp127_example/`, and `wasp33_additionnal_examples/` (those remain scratch material used
as reference during cleanup, not meant to be polished).

## Convention

One subfolder per stabilized STARSHIPS workflow. Inside each subfolder, notebooks generally
come in two flavours:

- **Quickstart** — run the workflow end-to-end with minimal explanation (e.g. `kpvsys_map.ipynb`,
  `stellar_fit_tutorial.ipynb`).
- **Explained** — a deeper walkthrough of the parameters and what they control, meant to double
  as reference documentation (e.g. `kpvsys_explained.ipynb`, `Reduction_parameters_explained.ipynb`).

A workflow only gets a folder here once its API has stabilized — otherwise the tutorial documents
an interface that is still moving. See `Notes/plan_revision_starships.md`, Chantier E, for the
cleanup roadmap driving this.

## Reading order

Folders are listed here in the order a new user actually needs them — i.e. the dependency chain
of the STARSHIPS workflow, not alphabetical order. This ordering is also what will drive the
table of contents once these tutorials get built into a Sphinx/ReadTheDocs site (not set up yet).

1. `reduction/` — turn raw instrument data into a reduced sequence
   (`pipeline/reduction.py`, `starships.transpec.ReductionParams`). Start here.
2. `rotation_kernel_examples/` — planetary rotation/wind kernels (`starships.spectrum`), a
   forward-modeling building block used when generating model spectra. Self-contained theory
   and math, no data or fitted products needed -- independent of the other folders here.
3. `stellar_fit_examples/` — fit the stellar spectrum (`starships.stellar_fit`). Needed as an
   input for emission retrievals; independent of retrieval/logl-map machinery otherwise.
4. *(planned)* `retrieval/` — run and inspect a retrieval (`starships.retrieval`). Not created yet
   — Chantier A/C API still moving.
5. `logl_grid_examples/` — log-likelihood / Kp-Vsys detection maps (`starships.logl_grid`),
   typically run as a lighter-weight alternative or complement to a full retrieval.

More folders (correlation/ttest, ...) will be added as their respective chantiers stabilize the
underlying API — insert them in this list at the point where a user would actually reach for them,
not appended at the end by default.
