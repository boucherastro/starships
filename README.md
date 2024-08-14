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
module load gcc python/3.9 mpi4py
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

To use starships, we will include example Notebooks in the close future. 

