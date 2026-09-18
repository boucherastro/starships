# starships
Spectral Transmission And Radiation Search for High resolutIon Planet Signal
*WARNING*: STARSHIPS produces a lot of data, so be careful with your disk space. We encourage you to run it on a cluster.

*NOTE*: The Notebook examples will be availble soon.

## Description

STARSHIPS is a python package designed to extract the transmission and emission spectra of exoplanets from high-resolution spectroscopy. It can search for spectral signatures by performing cross-correlation with a model spectrum. STARSHIPS can also perform atmosperic retrievals based on petitRADTRANS models. The retrieval framework also allows to combine Low and High resolution data.

### Instruments

STARSHIPS can be adapted to work with any high resolution spectrograph. The current version is tested with:
- SPIRou
- NIRPS
- IGRINS (The input spectra need a little tweaking)
- HARPS (comming soon)
- CRIRES (comming soon)

At low resolution, it works with any instrument that provides a spectrum in the form of a wavelength and spectrum. The instrument resolution needs to be provided and is assumed to be constant (lambda/delta_lambda). For photometry, the transmission function for each band is also required.

## Installation

Eventually, there will be  an installation via a simple pip install.
For now, you will have to clone this repository
````shell
git clone https://github.com/boucherastro/starships.git
````
Then navigate to the directory
```shell
cd starships
```
You should be in the main branch by default, but if not, switch to it:
```shell
git checkout main
```
The current versions is tested for python 3.9.
Note that petitRADTRANS (version 2.7) needs to be installed separately.

If you are on compute canada servers, I've put the exact instruction in the installation guidelines.

**On a cluster (Narval) with a shared installation already made by your
group, skip this whole section**: see
[Running on a cluster (SLURM / Narval)](#running-on-a-cluster-slurm--narval).

### General guidelines for installation:
(after you cloned the `starships` repository)

#### 1. Create a python environment
First, you should make yourself a new Python environment with python 3.9, and then activate it.

Examples:
```shell
conda create --name myenv python=3.9
conda activate myenv 
```
or (depending on how your environnements work)
```shell
virtualenv --python="/path/to/python3.9" "/path/to/new/virtualenv/"
source /path/to/new/virtualenv/bin/activate
```
NOTE: If you don't know your path to python3.9, run `which python` or `which python3`

For **compute canada**:
If you are on compute canada, you can use the following script to install starships and petitRADTRANS. 
```shell
module load StdEnv/2020
module load gcc python/3.9 mpi4py hdf5 netcdf
virtualenv /path/to/new/virtualenv/
pip install --no-index --upgrade pip
source /path/to/new/virtualenv/bin/activate
```
A good name for the virtual environment would be `starships_39` for example, so you would do `virtualenv ~/path/to/starships_39`. I personnaly put my environments in ~/venvs/ so I would do `virtualenv ~/venvs/starships_39`.

#### 2. Install STARSHIPS
Navigate into starhips directory (`cd /path/to/starships`). Then install the package in editor mode (temporary until we upload starships on pyPI).

`pip install -e .`

starships relies on atmospheric models from petitRADTRANS, so also install that :

`pip install petitRADTRANS==2.4.9`

Note that it could work with versions up tp 2.7, but we have not tested it yet. We plan to include the latest version (>= 3) in the future.

For petitRADTRANS, you'll need the input_data folder, where the corr-k and line-by-line (lbl) opacities grids are. This folder is very massive, so if you are on a cluster it would be best share this folder to other users or to point toward an existing one already on the server.

```shell
echo export pRT_input_data_path="/path/to/where/the/existing/or/new/folder/is/input_data\" >> ~/.bashrc
source ~/.bashrc
```
For narval cluster of **compute canada**:
```shell
echo export pRT_input_data_path=\"~/projects/def-dlafre/bouchea3/input_data" >> ~/.bashrc
source ~/.bashrc
```

Check if the installation worked (in python):

```python
from petitRADTRANS import Radtrans
atmosphere = Radtrans(line_species = ['CH4'], continuum_opacities=['H2-H2'])
```
You may need to install other packages (we are still building this), so note what is missing when you run the code at tell us please.

#### 3. Keep a local version of exofile (optional)
(optional, but useful without internet connexion)

(Required to work on **compute canada** clusters)

Finally, with exofile, you can keep a local version of the nasa exoplanet archive to access planetary parameters.
To do so, in python:
```python
from exofile.archive import ExoFile
from exofile.config import edit_param

# Load the exofile (need internet connexion)
database = ExoFile.load(query=True,
                        use_alt_file=True,
                        warn_units=False,
                        warn_local_file=False)

# Save it wherever you want
database.write('/path/to/my_local_exofile_alt.ecsv')

# Specify the path in the code parameters (this will be saved)
edit_param(exofile_alt='/path/to/my_local_exofile_alt.ecsv')

```
If you want to use the database without internet, use `ExoFile.load(query=False, ...)`

### Adding the environment to jupyter notebook (or any iPython kernel)

You can create a kernel for the virtual environment to use it in Jupyter notebooks. I generally do this will all my environment. This will also become handy in VSCode to execute some specific part of a python code or play in an interactive ipython console. Here is how to do it:

 ```bash
 # Make sure you are in the virtual environment (source /path/to/new/virtualenv/bin/activate)
 mkdir -p ~/.local/share/jupyter/kernels # Create the directory where the kernels are saved
 pip install --no-index ipykernel  # Install the ipykernel package
 python -m ipykernel install --user --name starships_39 --display-name "STARSHIPS Kernel (3.9)"  # Create the kernel
 ```

 See https://docs.alliancecan.ca/wiki/Advanced_Jupyter_configuration#Python_kernel for more details.

## Usage

Curated example notebooks live in [`tutorials/`](tutorials/) (see its README
for the recommended reading order: data reduction, rotation kernels, stellar
fit, retrieval, logL / Kp-vsys maps). The heavy steps (retrievals, stellar
fits, logL grids) are meant to run as SLURM jobs: see the next section.

## Running on a cluster (SLURM / Narval)

Everything here assumes Narval (Digital Research Alliance of Canada), but
applies to any SLURM cluster. The commands come with the package: nothing to
clone, no `sbatch` script to copy or edit.

### 1. Environment: use the shared installation

If someone in your group already installed STARSHIPS in a shared location
(typically a virtual environment under `~/projects/<account>/shared/`), you
do not need to install anything or load any module -- ask your supervisor for
the actual path.

**One-time setup** -- tell petitRADTRANS where its (shared) opacity data
lives, with one line in your `~/.bashrc`:

```shell
echo 'export pRT_input_data_path="/home/$USER/projects/def-dlafre/bouchea3/input_data"' >> ~/.bashrc
source ~/.bashrc
```

**Every session** -- activate the shared environment. This is the only
command to learn: it makes the `starships` package importable in Python and
its commands (`submit_starships_retrieval`, ...) available in your shell:

```shell
source /project/def-rdoyon/shared/starships_env/bin/activate   # ask for the real path
```

For notebooks, start JupyterLab from that same activated environment, or
register it as a Jupyter kernel (see
[Adding the environment to jupyter notebook](#adding-the-environment-to-jupyter-notebook-or-any-ipython-kernel)).
Using one environment for both the notebooks and the submitted jobs
guarantees they run the same code.

Notes:
- No `module load` is needed to *use* the shared environment (its Python is
  self-contained); the `module load` lines in the installation guidelines are
  only needed to *create* an environment.
- Never `pip install` into the shared environment. If you need to modify
  STARSHIPS itself, create your own environment following the
  [installation guidelines](#general-guidelines-for-installation) instead;
  the commands below then use that one -- whichever environment is active
  when you submit is the one the job runs with.

### 2. Submit jobs

Each heavy step is one command, run from any directory. It takes the SLURM
resources as options, copies your YAML to scratch before submitting (so you
can keep editing the original while the job waits in the queue -- the job
reads the copy) and submits the job:

```shell
# Retrieval: 8 independent burnin retrievals (one SLURM array), then sampling
submit_starships_retrieval burnin my_retrieval.yaml --n-burnin-runs 8 \
    --account def-dlafre --cpus-per-task 24 --mem 100G --time 1-12:00:00
submit_starships_retrieval sampling my_retrieval.yaml \
    walker_file_in=my_burnin_3.h5 init_mode=from_burnin \
    --account def-dlafre --cpus-per-task 24 --mem 100G --time 3-00:00:00

# Stellar spectrum fit (downloads the PHOENIX models first, on the login node)
submit_starships_stellar_fit my_stellar_fit.yaml \
    --account def-dlafre --cpus-per-task 8 --mem 20G --time 08:00:00

# logL / Kp-vsys grids, one job per model file
submit_starships_logl_grid my_retrieval.yaml --specfile model_A.npz model_B.npz \
    --account def-dlafre --cpus-per-task 4 --mem 20G --time 07:00:00
```

- `--account`, `--cpus-per-task`, `--mem` and `--time` are required (no
  silent fallback on the cluster defaults). `--help` lists everything else.
- Add `--dry-run` to see the exact `sbatch` command without submitting.
- `key=value` tokens given to `submit_starships_retrieval` override the
  corresponding keys of the YAML for that job.
- Logs go to `$SCRATCH/sbatch_outputs/out_sbatch_<jobid>.txt`; the YAML
  copies to `$SCRATCH/temp_yaml_files/`; and `$SCRATCH/sbatch_outputs/submissions.log`
  records which copy and command each job ID ran.

### 3. Size the resources first

To find out how many CPUs, how much memory and how much time a retrieval
needs before launching it for real, use `run_starships_resource_sweep` (a few
short test jobs, then `run_starships_resource_parse` reads SLURM's accounting
and extrapolates).

Full reference for all of these commands, and the design choices behind them:
[`starships/slurm/README.md`](starships/slurm/README.md).
