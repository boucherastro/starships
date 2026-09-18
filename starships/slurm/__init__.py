"""SLURM resource-estimation sweep for STARSHIPS retrievals.

Answers a practical question before launching a real retrieval on Narval (or
any SLURM cluster): "if I ask for N cpus, how much ``--mem``/``--time`` do I
actually need?" It runs a small, real workload (a handful of ``lnprob()``
calls across a real ``multiprocessing.Pool``) at several ``n_cpu`` values,
each as its own SLURM job, then reads SLURM's own accounting (``sacct``) for
the real numbers and fits a line through them to extrapolate to the ``n_cpu``
a real run would use.

This is exposed entirely through 3 console scripts (registered in
``setup.py``, available on the PATH after a plain ``pip install starships`` --
no repository checkout or file copying needed):

- ``run_starships_resource_test`` (:mod:`starships.slurm.worker`) -- runs ONE
  ``n_cpu`` value's workload. This is what gets submitted to SLURM; not
  normally called directly.
- ``run_starships_resource_sweep`` (:mod:`starships.slurm.submit`) -- submits
  one real SLURM job per ``n_cpu`` to test, each calling
  ``run_starships_resource_test`` with an exact ``--cpus-per-task``. Run this
  first, from whatever directory you want the sweep's metadata/logs saved in.
- ``run_starships_resource_parse`` (:mod:`starships.slurm.parse`) -- once all
  jobs finish, reads their ``sacct`` accounting + logs, produces the scaling
  plot, and (optionally) an estimated ``--mem``/``--time`` table for a real
  run. Run this second, from the same directory.

See each module's docstring for the full design rationale (in particular why
memory comes from ``sacct`` rather than any self-measurement inside the
running job, and why this deliberately does NOT use a SLURM job array).
:mod:`starships.slurm.estimation` holds the underlying fit/plot/extrapolate
functions, usable standalone if needed.

This subpackage (``starships.slurm``) is the home for SLURM/cluster-specific
operational tooling in general, kept deliberately separate from the science
code (``retrieval.py``, ``logl_grid.py``, ...). Besides the resource sweep
above, it holds the **job launchers** that replace the old copy-and-edit bash
``sbatch`` scripts (``send_job_retrieval.sh``, ``run_sbatch_*.sh``) -- three
more console scripts, one per workload, sharing :mod:`starships.slurm.common`:

- ``submit_starships_retrieval`` (:mod:`starships.slurm.submit_retrieval`) --
  burnin (optionally N chains as a SLURM array) or sampling retrieval.
- ``submit_starships_stellar_fit`` (:mod:`starships.slurm.submit_stellar_fit`)
  -- PHOENIX download on the login node, then the fit.
- ``submit_starships_logl_grid`` (:mod:`starships.slurm.submit_logl_grid`) --
  one Kp x v_sys grid job per model file.

All take the SLURM resources as command-line options (no ``#SBATCH`` header
anywhere), copy the input YAML to scratch before submitting (so the original
can be edited without touching the queued job), and run the installed console
scripts of the current Python environment. See ``README.md`` here.
"""
