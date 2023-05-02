# starships
Spectral Transmission And Radiation Search for High resolutIon Planet Signal

## Installation

You will need to install a bunch of stuff before you can use starships.

First, you should make yourself a new Python environnement, and then activate it.

Examples:
```
conda create --name myenv
conda activate myenv 
```
or
```
virtualenv --no-download ~/.virtualenvs/myenv
source ~/.virtualenvs/myenv/bin/activate
```

Then you have to install the relevent Python packages :

`pip install astropy exo-k h5py matplotlib mpi4py numpy pymultinest scipy molmass seaborn corner`

`starships` relies on atmospheric models from `petitRADTRANS`, so also install that :

`pip install petitRADTRANS`

For this, you'll need the input_data folder, where the corr-k and line-by-line (lbl) opacities grids are. This folder is very massive, so it would be best to point toward an existing one already on the server.

```
echo export pRT_input_data_path=\"/path/to/where/the/existing/or/new/folder/is/input_data\" >> ~/.bashrc
source ~/.bashrc
```

Check if the installation worked:

```
from petitRADTRANS import Radtrans
atmosphere = Radtrans(line_species = ['CH4'], continuum_opacities=['H2-H2'])
```

You'll have to install the batman package, but it sometimes have problems. So you can try it as is :

`pip install batman-package`

And if it doesn't work, you can try with an older version:

`pip install batman-package==2.4.8`


Then, you can install exofile, directly with:

`pip install exofile`

Or with another route:

```
git clone https://github.com/AntoineDarveau/exofile.git
cd exofile/
python -m pip install -U .
```

Install another bunch of packages (which are best downloaded after exofile):

`pip install astroquery sklearn arviz emcee`

And finally, you can clone starships! 

`git clone https://github.com/boucherastro/starships.git`


## Usage 

To use starships, we will include example Notebooks in the future. 

