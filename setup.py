from setuptools import setup, find_packages

setup(
    name='starships',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        'astropy==5.1',
        'batman-package==2.4.8',
        'corner<=2.2.1',
        'emcee<=3.1.3',
        'exo-k<=1.2.0',
        'exofile>=0.2.2',
        'h5py>=3.7,<3.11.0',
        'matplotlib<3.7',
        'numpy==1.23.0',
        'scikit-learn==1.0.2',
        'astroquery==0.4.6',
        'molmass',
        'arviz==0.13.0',
        'netcdf4==1.5.*',
        'PyAstronomy==0.18.0'
    ],
    entry_points={
        'console_scripts': [
            'starships=starships.cli:main'
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
