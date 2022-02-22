The CSV files contain the parameters used to generate the potentials. The jupyter notebook 'load_potentials.ipynb' shows how to load and process the parameters.

The HDF5 files contain solutions of the Schroedinger equation corresponding to the potential found using Numerov's method. The jupyter notebook 'load_hdf5.iypnb' shows how to calculate the remaining quantities such as the kinetic energy from these solutions.

Note that there is no HDF5 file for the M=100000 training set as this file exceeds
the 100 MB GitHub file size limit.
