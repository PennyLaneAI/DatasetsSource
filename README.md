# Datasets Generator
This repository containing the source code for generating quantum datasets.

The underlying data generation pipelines can be used after installing the above as a `quantum_datasets` module as

``` bash
git clone https://github.com/PennyLaneAI/DatasetsSource
cd DatasetsSource
pip install .
```

Once, the module is installed, one may use the generation methods - `qchem_data_generate` and `qspin_data_generate` for generating datasets for Quantum Chemistry and Quantum Many-body Physics as shown below:

### Code Snippets

For quantum chemistry, you will have to provide it the path for the `XYZ` files. We use a modified version of the standard `XYZ` files, which can be found in the `xyzfiles` folder for reference.

``` python
import quantum_datasets as qd
qd.qchem_data_generate("xyzfiles/q.1-1/schm-1/h2.xyz")
```

For the quantum spin systems, we have to provide the name of the spin systems (for example: "Ising", "Heisenberg", "FermiHubbard" or "BoseHubbard"), and the periodicity and layout of the lattice.

``` python
import quantum_datasets as qd
qd.qspin_data_generate("Ising", periodicity=True, layout=(1, 4), num_systems=1000)
```

### Datasets
All the datasets will be generated and written under the `datasets` folder with distinct subfolders for `qchem` and `qspin`.