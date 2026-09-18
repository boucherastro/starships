# `starships.slurm` — SLURM resource-estimation sweep

Answers two practical questions before launching a real retrieval on Narval,
via two workloads (`--workload`, default `lnprob`):

- **`lnprob`**: "if I ask for N cpus, how much `--mem`/`--time` do I actually
  need?" Runs a small, real workload (a handful of `lnprob()` calls across a
  real `multiprocessing.Pool`) at several `n_cpu` values, each as its own
  SLURM job, then reads SLURM's own accounting (`sacct`) for the real numbers
  and fits a line through them to extrapolate to the `n_cpu` a real run would
  use.
- **`retrieval`**: "does putting more walkers per cpu improve efficiency?"
  (Antoine, 2026-09-17: within one MCMC step, `n_walkers` spread across
  `n_cpu` workers parallelizes fine, but every walker must finish before the
  ensemble can propose the next step -- more walkers per cpu amortizes that
  fixed per-step synchronization cost over more real compute per worker.) At
  a **fixed** `n_cpu`, runs a real, short `run_starships_retrieval` (the
  actual, unmodified retrieval entry point -- nothing reimplemented) at
  several `n_walkers_per_cpu` values, and compares the measured time per real
  MCMC step against the ideal, no-overhead cost from a prior `lnprob` sweep.

This is a normal part of the `starships` package (`starships.slurm`) — after
a plain `pip install starships`, the 3 console scripts below are on the PATH.
No repository checkout, no file copying, no editing a script's sbatch header
by hand.

## Recommended workflow for a new retrieval

A practical, step-by-step procedure for sizing `--cpus-per-task`/`--mem`/
`--time` and choosing `n_walkers`/`n_cpu` before committing to a real (long,
expensive) retrieval run. Every step submits real, short SLURM jobs — confirm
with whoever manages the allocation first, same rule as any job submission.

1. **Baseline: raw per-call cost.** A quick `lnprob` sweep across a few
   `n_cpu` values gives the pure compute cost per call — the "ideal, no
   overhead" reference every later step compares against:
   ```bash
   run_starships_resource_sweep <config.yaml> 1:2 2:3 4:5 8:7 --account <acct>
   # ... wait for the jobs (squeue -u $USER), then:
   run_starships_resource_parse
   ```
   Read off the `avg s/iter` column — call it `<baseline>` below. It should
   be roughly stable across `n_cpu`; if it climbs steeply instead, that
   itself is worth knowing (real CPU contention on this node/config).

2. **Know your `ndim` before picking any `n_walkers`.** `emcee` refuses to
   run with too few walkers relative to the number of free parameters
   (`ndim`) — a `RuntimeError`/`ValueError` you'll hit immediately otherwise.
   A cheap, quick check (no SLURM job needed — `setup_retrieval` alone takes
   seconds, it does not touch petitRADTRANS opacities):
   ```bash
   python -c "
   import starships.retrieval as retrieval
   retrieval.setup_retrieval('<config.yaml>')
   print('ndim =', retrieval.walker_init.shape[1])
   "
   ```
   `n_walkers` must be `>= 2 * ndim` (emcee's hard minimum) — plan for well
   above that (4-8x `ndim` is a common rule of thumb, more if your
   parameters are strongly correlated, e.g. a rotation kernel combined with
   multi-region parameters — a case where you likely want to lean toward the
   higher end).

3. **Find the walkers-per-cpu efficiency "sweet spot".** At a candidate
   `n_cpu` (e.g. what you'd normally request), sweep a handful of
   `n_walkers_per_cpu` values with a small `--n-steps` (2-5 is enough to get
   a stable reading):
   ```bash
   run_starships_resource_sweep <config.yaml> --workload retrieval \
       --n-cpu 16 --n-steps 2 4 8 16 32 --account <acct>
   # ... wait, then:
   run_starships_resource_parse --baseline-time-per-iter-s <baseline> \
       --node-mem-gb 250
   ```
   The efficiency panel climbs then flattens ("saturates") — that knee is
   roughly where more walkers per cpu stop being worth it. Check the memory
   panel too: it should stay well under the single-node line (250 GB on a
   standard Narval node); if it climbs steeply instead, that's a real limit
   on how far you can push this.

4. **(If you have a target walker count) find the best `n_cpu` for it.**
   Step 3 only tested one `n_cpu`. If `ndim`/science needs pin down a
   specific `n_walkers` target, test a few `n_cpu` values directly, and weigh
   wall-clock speed against CPU-hours — Compute Canada bills by CPU-hour, and
   these two move in *opposite* directions (see "What's measured, and why"
   below):
   ```bash
   run_starships_resource_sweep <config.yaml> --workload retrieval \
       --n-walkers 256 --n-steps 2 10 16 24 32 --account <acct>
   # ... wait, then:
   run_starships_resource_parse --baseline-time-per-iter-s <baseline> \
       --n-steps-target <real_n_steps> --n-walkers-target 256 --node-mem-gb 250
   ```
   `--n-steps-target` is your real run's total step count (burnin +
   sampling) — the cost panel then shows actual estimated wall-time/
   CPU-hours per `n_cpu` tested, a concrete number rather than an abstract
   percentage. `--real-cpu` (with `--n-walkers`) is also available here for
   an oversubscription test (more logical workers than real cores) — see
   `--help`, still experimental/exploratory as of this writing.

5. **Size `--mem`/`--time` for the real run** from whichever point you picked
   in step 3/4: take the measured `MaxRSS` with a safety margin (this tool's
   `lnprob` workload already applies a ×1.5 margin automatically in its own
   `est. mem` column — do the same by hand here), and the estimated wall time
   from step 4's cost panel (or `time_per_step * your_real_n_steps` from step
   3's table if you skipped step 4).

Steps 3 and 4 answer different questions and can disagree on the "best"
point — step 3's efficiency alone tells you where returns diminish; step 4
tells you whether that translates to faster *or* cheaper for your specific
walker count. There's no single number this tool hands you — it's meant to
make an informed choice easy, not replace one.

## Usage

```bash
# --- lnprob workload (default) ---

# 1. Submit the sweep (from wherever you want last_sweep.json/logs/ saved,
#    e.g. a per-project scratch directory). Each point is "n_cpu:n_iter_per_cpu"
#    -- use MORE iterations at higher n_cpu (see "Why n_iter_per_cpu grows
#    with n_cpu" below):
run_starships_resource_sweep <retrieval_config.yaml> 1:2 2:3 4:6 8:12 \
    --account <your_slurm_account> [--mem 64G] [--time 00:30:00]

# 2. Wait for the jobs to finish (squeue -u $USER).

# 3. Parse the results (same directory, no arguments needed for the basic case):
run_starships_resource_parse
# Or, to also get an estimated --mem/--time table for a real run, and a
# zoomed-out plot projecting the n_cpu you actually intend to run at:
run_starships_resource_parse --n-walkers 84 --n-steps 5000 --extra-n-cpu 16

# --- retrieval workload (walkers-per-cpu efficiency) ---
# See "Recommended workflow for a new retrieval" above for the full
# step-by-step procedure -- this is just the two sweep shapes it uses.

# 1a. Fixed n_cpu, one job per n_walkers_per_cpu value, --n-steps shared
#     (n_steps_burnin override -- see "Why n_iter_per_cpu grows with n_cpu"
#     below for why a single shared value is fine here, unlike lnprob's
#     per-point n_iter_per_cpu):
run_starships_resource_sweep <retrieval_config.yaml> --workload retrieval \
    --n-cpu 8 --n-steps 5 2 4 8 14 --account <your_slurm_account>

# 1b. Or, fixed n_walkers instead (one job per n_cpu value) -- for "what
#     n_cpu should I use for MY fixed walker count":
run_starships_resource_sweep <retrieval_config.yaml> --workload retrieval \
    --n-walkers 256 --n-steps 5 10 16 24 32 --account <your_slurm_account>

# 1c. --n-walkers also supports an oversubscription test: --real-cpu fixes
#     the REAL --cpus-per-task while the points are the LOGICAL n_cpu passed
#     to Pool/EnsembleSampler (n_cpu > --real-cpu = more worker processes
#     than physical cores):
run_starships_resource_sweep <retrieval_config.yaml> --workload retrieval \
    --n-walkers 256 --n-steps 5 --real-cpu 16 32 --account <your_slurm_account>

# 2. Wait for the jobs to finish, then parse -- pass the pure per-call cost
#    read off a prior lnprob sweep's table (the "avg s/iter" column) for the
#    efficiency comparison:
run_starships_resource_parse --baseline-time-per-iter-s 44
```

`run_starships_resource_parse --help` / `run_starships_resource_sweep --help`
list every option (`--extra-n-cpu`, `--available-cpus` for the
oversubscription boundary on the plot, `--baseline-time-per-iter-s`,
`--n-steps-target`/`--n-walkers-target`/`--node-mem-gb`, etc).

## Why `n_iter_per_cpu` grows with `n_cpu`

Each sweep point's `n_iter_per_cpu` is how many `lnprob()` calls EACH worker
runs (not a total split across them) — chosen per point, not shared across
the whole sweep (Antoine, 2026-09-17). Two reasons to use more at higher
`n_cpu`:

- A worker's average call time is noisier with fewer calls — dominated by the
  residual post-fork "first touch" cost rather than the true steady-state
  per-iteration cost.
- Since all of a job's workers run in parallel, extra calls per worker cost
  comparatively little *extra wall time* as `n_cpu` grows — so scaling
  `n_iter_per_cpu` up with `n_cpu` also keeps every job in the sweep roughly
  the same wall-clock length, instead of the low-`n_cpu` jobs (fewer, slower
  workers) taking much longer than the high-`n_cpu` ones for the same total
  work.

There's no formula for "the right" iteration count (that would need already
knowing the per-call cost this sweep exists to measure) — just pick more for
higher `n_cpu`, and check the resulting per-job wall times (`pool wall time`
in each job's log) once the sweep is done; adjust and re-run a point if one
came out far shorter or longer than the rest.

**Always confirm with whoever manages the allocation before running this for
real** — it submits real SLURM jobs on a shared compute resource, same rule
as any other job submission.

## What's measured, and why

- **Memory**: `sacct`'s `MaxRSS` (SLURM's own cgroup-based accounting) for
  each job's `.batch` step. This is ground truth for "will this OOM" --
  whatever SLURM enforces `--mem` against -- and correctly accounts for a
  forked `multiprocessing.Pool` (parent + all workers, shared pages counted
  once, not once per worker) -- simpler and more reliable than sampling
  memory from inside the running process.
- **Time**: NOT taken from `sacct`'s `Elapsed` directly, because that includes
  one-time job setup (`setup_retrieval`, loading high-res data) which does not
  depend on `n_cpu` and would bias the per-`n_cpu` comparison. Instead,
  `worker.py` times just the `multiprocessing.Pool` block itself and prints it
  ("pool wall time"), which `parse.py` reads back out of the job's log.
- Only the **peak memory** and **total pool time** per `n_cpu` are used (no
  per-`lnprob()`-call detail, no distinction between a worker's first call and
  later calls) — sufficient for fitting/extrapolating a line through `n_cpu`.

## Why not a SLURM array

A single `sbatch --array=...` submission uses the *same* `--cpus-per-task`
for every task in the array — SLURM does not support a different core count
per array index within one submission. Testing several `n_cpu` values would
therefore mean either over-allocating cores for the smaller ones (request the
max everywhere) or under-allocating for the larger ones. Antoine's call
(2026-09-17): skip the array, submit one job per `n_cpu` in a loop instead,
each with its exact `--cpus-per-task` — `submit.py` groups them with a shared
`sweep_id` (job names, log filenames) and a metadata file (`last_sweep.json`)
instead of a real SLURM array ID, so `parse.py` still only needs one command
with no arguments.

## Why no venv activation anywhere

`submit.py` resolves `run_starships_resource_test`'s absolute path from
`sys.prefix` (the active Python environment's root, i.e. `<sys.prefix>/bin/
run_starships_resource_test`) rather than `source`-ing any particular venv by
name. `sys.prefix` is used instead of `sys.executable`'s own directory
because some Python installs (e.g. certain macOS venvs wrapping a Framework
build) have `sys.executable` be a symlink whose *resolved* target lives
outside the venv's `bin/` entirely -- caught during local testing (2026-09-17)
before it ever reached Narval. Whichever Python environment you launched the
sweep from is exactly the one used inside the submitted jobs, with nothing
hard-coded — this is what makes the tool work out of the box for anyone with
`starships` installed, not just on Antoine's own venv path.

## Oversubscription / hyperthreading

`--real-cpu` (`--workload retrieval --n-walkers`, see "Usage" above) tests
requesting more *logical* `Pool`/`EnsembleSampler` workers than *real*
allocated cores. **First real test (2026-09-18, KELT-20b `day2_2regions`,
256 walkers, real 16 cores): it hurts, not helps** — logical `n_cpu=32` (2x
oversubscription) vs logical `n_cpu=16` (no oversubscription) on the same 16
real cores was both slower (1080 vs 983 s/step) and used more memory (92 vs
61 GB), not less — each extra logical worker still carries its own working
memory even though it's sharing real cores, and the added contention costs
more than the smaller per-worker walker share saves. Consistent with the
unrelated `logl_grid.py` oversubscription finding
(`Notes/plan_revision_starships.md`, chantier C): CPU/memory-bound workloads
like this one don't benefit from oversubscription the way I/O-bound ones can.
Not recommended based on this one data point; the `lnprob` workload's
`plot_resource_scaling` separately accepts `available_cpus` to shade an
`n_cpu > available_cpus` region on its plots, for the same kind of check on
raw compute scaling.

## Job launchers: `submit_starships_*`

The second family of tools in `starships.slurm` (added 2026-09-18). They
replace the bash `sbatch` launchers that used to live in `retrievals/` and in
per-project `scripts/` folders on Narval (`send_job_retrieval.sh` +
`run_retrieval.sh`, `run_sbatch_fit_stellar_spectrum.sh`,
`run_sbatch_logl_grid.sh`) — scripts that had to be copied, then edited by
hand (the `#SBATCH` header / `sbatch_args=(...)` "EDIT THIS SECTION" block,
a hard-coded venv path, a hard-coded list of model files) for every project.
With `starships` installed, there is nothing to copy any more:

| Command | Replaces | Runs in the job |
|---|---|---|
| `submit_starships_retrieval` | `send_job_retrieval.sh` + `run_retrieval.sh` | `run_starships_retrieval` |
| `submit_starships_stellar_fit` | `run_sbatch_fit_stellar_spectrum.sh` | `run_starships_stellar_fit` (new console script for `python -m starships.stellar_fit`) |
| `submit_starships_logl_grid` | `run_sbatch_logl_grid.sh` | `run_starships_logl_grid` |

**Which Python environment?** The one you activated before running the
command -- nothing else. The job calls the console script of that environment
by absolute path, so no `module load` and no `activate` are needed inside the
job (verified on Narval, 2026-09-18: a venv's Python runs self-contained,
under the default `StdEnv/2023` as well as with an empty environment; the
only external requirement is petitRADTRANS's `pRT_input_data_path` variable,
normally set once in `~/.bashrc`). For a student, the minimum-effort setup is
therefore: activate a shared environment installed under `projects/`, then
run the command -- see "Running on a cluster (SLURM / Narval)" in the
[main README](../../README.md#running-on-a-cluster-slurm--narval), the
user-facing entry point for all of this.

All three share `starships/slurm/common.py` and the same conventions:

- **SLURM resources are command-line options** — `--account`,
  `--cpus-per-task`, `--mem`, `--time` are *required* (no silent fallback on
  the cluster's defaults, e.g. a 1 h limit for a 3-day retrieval);
  `--job-name`, `--mail-type` (default `FAIL`), `--sbatch-arg=--anything`
  (repeatable pass-through), `--pre-cmd "module load ..."` are optional.
  `--nodes=1` is always set. Same venv-agnostic mechanism as the sweep (see
  "Why no venv activation anywhere" above): the job calls the installed
  console script of the environment you submit from.
- **The YAML is copied before submission, and the job reads the copy.** A
  job can wait in the queue for hours; editing the YAML to prepare the next
  run must not change what the queued run does. Copies go to
  `<scratch>/temp_yaml_files/<stem>_<YYYYmmdd_HHMMSS>.yaml` with a comment
  header saying where they came from. This was `send_job_retrieval.sh`'s
  behaviour; it now applies to the stellar fit and logl grid launchers too.
- **`<scratch>`** is `--scratch-dir`, else `scratch_dir` from
  `~/.starships/config.yaml`, else `$SCRATCH` (always set on Narval). SLURM
  logs go to `<scratch>/sbatch_outputs/out_sbatch_%j.txt` (`_%A_%a` for an
  array) — named by job ID, not job name, since only the ID is guaranteed
  unique and it is what `squeue`/`sacct` show — and every submission appends
  one line (date, job ID, YAML copy,
  full `sbatch` command) to `<scratch>/sbatch_outputs/submissions.log` —
  so a job ID can always be traced back to exactly what it ran.
- **`--dry-run`** does everything (including the YAML copy) except calling
  `sbatch`, and prints the exact command. Works anywhere, cluster or not.

```bash
# --- Retrieval: burnin as 8 independent burnin retrievals (SLURM array), then sampling ---
submit_starships_retrieval burnin retrieval_inputs_kelt20.yaml --n-burnin-runs 8 \
    --account def-dlafre --cpus-per-task 24 --mem 100G --time 1-12:00:00 --job-name K20_burnin
submit_starships_retrieval sampling retrieval_inputs_kelt20.yaml \
    walker_file_in=K20_burnin_3.h5 init_mode=from_burnin \
    --account def-dlafre --cpus-per-task 24 --mem 100G --time 3-00:00:00 --job-name K20_sampling

# --- Stellar fit: PHOENIX download on the login node (default, idempotent), then the fit ---
submit_starships_stellar_fit stellar_fit_inputs_kelt20.yaml \
    --account def-dlafre --cpus-per-task 8 --mem 20G --time 08:00:00 --job-name fit_star

# --- logL grids: one job per model file, options after "--" go to run_starships_logl_grid ---
submit_starships_logl_grid retrieval_inputs_kelt20_dayside.yaml \
    --specfile model_spectra/kelt20b_D2_H2O_only.npz model_spectra/kelt20b_D2_all.npz \
    --account def-dlafre --cpus-per-task 4 --mem 20G --time 07:00:00 --job-name logl_K20 \
    -- --plot trailing --kp 227
```

Per-tool specifics (full detail in each module's docstring / `--help`):

- `submit_starships_retrieval MODE CONFIG [key=value ...]`: `MODE` is
  `burnin` or `sampling`. `key=value` tokens are forwarded verbatim to
  `run_starships_retrieval` (which overrides those YAML keys itself).
  `retrieval.py` decides the phase from `walker_file_in` (null = burnin), so
  the tool checks **at submission time** that YAML + overrides match the
  requested mode — and that `--n-burnin-runs N > 1` (a SLURM array of N
  independent burnin *retrievals*, each with its own full set of walkers,
  acting together as one very large burnin) comes with
  `slurm_array_behaviour: 'burnin'`,
  without which `retrieval.py` raises on startup. Catching these before the
  queue wait is the whole point of having a `MODE` argument at all.
- `submit_starships_stellar_fit CONFIG`: runs
  `stellar_fit.download_phoenix_files` on the login node first (compute nodes
  have no internet; the download skips files already present) —
  `--no-download` / `--download-only` to split the two steps. Warns when the
  YAML's `n_workers` (what `stellar_fit` actually uses for its pool) differs
  from `--cpus-per-task`.
- `submit_starships_logl_grid CONFIG --specfile A.npz B.npz ...`: one job per
  model file (not an array — each job has an explicit `--specfile`, its own
  name/log, and can be re-run alone), all sharing one YAML copy.

Not migrated: `run_sbatch_profile_memory.sh` — it wrapped
`retrievals/profile_memory_scaling.py`, the ad hoc predecessor of the
resource sweep above, which no longer exists in the repo.

## Possible future extension: more of `starships.slurm`

The resource sweep and the three job launchers above are what lives here as
of 2026-09-18. Other SLURM/cluster-specific operational tooling belongs here
too, kept separate from the science code.
