from setuptools import setup, find_packages
from starships import __version__

setup(
    name='starships',
    version=__version__,
    packages=find_packages(),
    install_requires=[
        'astropy==5.1',
        'batman-package==2.4.8',
        'corner<=2.2.1',
        'emcee<=3.1.3',
        'exo-k<=1.2.0',
        'exofile>=0.2.2',
        # 'h5py==3.6',
        'matplotlib<3.7',
        'numpy>=1.23.0,<1.26.0',
        'scikit-learn>=1.0.2,<=1.3.1',
        'astroquery==0.4.6',
        'molmass',
        'arviz<=0.13',
        # 'netcdf4==1.5.*',
        'PyAstronomy==0.18.0'
    ],
    entry_points={
        'console_scripts': [
            'run_starships_retrieval=starships.retrieval:main',
            'run_starships_logl_grid=starships.logl_grid:main',
            'starships=starships.cli:main',
            # SLURM resource-estimation sweep (starships.slurm) -- see that
            # subpackage's README for usage.
            'run_starships_resource_test=starships.slurm.worker:main',
            'run_starships_resource_sweep=starships.slurm.submit:main',
            'run_starships_resource_parse=starships.slurm.parse:main',
            # Job launchers (starships.slurm) -- replace the old bash sbatch
            # scripts; see that subpackage's README for usage.
            'run_starships_stellar_fit=starships.stellar_fit:cli',
            'submit_starships_retrieval=starships.slurm.submit_retrieval:main',
            'submit_starships_stellar_fit=starships.slurm.submit_stellar_fit:main',
            'submit_starships_logl_grid=starships.slurm.submit_logl_grid:main',
        ]
    },
    author='Antoine Darveau-Bernier',
    author_email='antoine.darveau@gmail.com',
    description='A package for analysing high-resoltion spectroscopic data of exoplanets',
    license='MIT',
    keywords='starships analysis',
    url='https://github.com/boucherastro/.git',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
)
