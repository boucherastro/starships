# starships
Spectral Transmission And Radiation Search for High resolutIon Planet Signal

## Installation

Eventually, there will be  an installation via pip install.
For now, you will have to clone this repository
````shell
git clone https://github.com/boucherastro/.git
````
and use the file ``requirements.txt``
to install the packages needed.
The current versions is tested for python 3.9.
Note that petitRADTRANS needs to be installed separately.

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
For **compute canada**:
```shell
module load gcc python/3.9 mpi4py
virtualenv /path/to/new/virtualenv/
source /path/to/new/virtualenv/bin/activate
```

#### 2. Install required packages
Then you have to install the relevant Python packages :

`pip install -r /path/to/starships/requirements.txt`

starships relies on atmospheric models from petitRADTRANS, so also install that :

`pip install petitRADTRANS==2.4.9`

For this, you'll need the input_data folder, where the corr-k and line-by-line (lbl) opacities grids are. This folder is very massive, so it would be best to point toward an existing one already on the server.

```shell
echo export pRT_input_data_path=\"/path/to/where/the/existing/or/new/folder/is/input_data\" >> ~/.bashrc
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
You may need to install other packages (we are still building this), so note what is missing when you run the code at tell me please

#### 3. Keep a local version of exofile
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


## Usage 

To use starships, we will include example Notebooks in the future. 

