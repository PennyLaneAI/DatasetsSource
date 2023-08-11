# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import networkx as nx
from tqdm.auto import tqdm
from pennylane import numpy as pnp

from .qchem import ChemDataPipeline
from .qspin import SpinDataPipeline
from .spin import IsingModel, HeisenbergModel, FermiHubbardModel, BoseHubbardModel
from .chem import read_xyz, bond_struct, geom_struct

systypes = {
    "ising": IsingModel,
    "heisenberg": HeisenbergModel,
    "fermihubbard": FermiHubbardModel,
    "bosehubbard": BoseHubbardModel,
}


# pylint: disable=dangerous-default-value
def qchem_data_generate(
    xyz_path, basis="STO-3G", bondlengths=[], use_bond_struct=False, folder_path=None
):
    r"""Generates data for Molecular Systems"""
    data_pipeline = ChemDataPipeline()
    _, mol_name, symbols, charge, geometry, gs_bond = read_xyz(xyz_path)
    geometries, gs_bonds = [geometry], [gs_bond]

    bond_lengths = bondlengths
    if use_bond_struct:
        bond_lengths += list(bond_struct[mol_name])

    for bond in bond_lengths:
        bond = float(bond)
        gs_bonds.append(bond)
        geometries.append(pnp.array(geom_struct[mol_name](bond), requires_grad=False))

    for bond, geom in (bond_bar := tqdm(zip(gs_bonds, geometries))):
        data_pipeline.pipeline(
            molname=mol_name,
            symbols=symbols,
            geometry=pnp.array(geom, requires_grad=False) * 1.88973,  # Ang to Bohr conversion.
            charge=charge,
            basis_name=basis,
            descriptor=f"{round(bond, 3)}",
            filename=folder_path if folder_path else "",
            prog_bar=bond_bar,
        )
        gc.collect()

def qspin_data_generate(sysname, periodicity, layout, num_systems=100, folder_path=None):
    r"""Generates data for Spin Systems"""
    data_pipeline = SpinDataPipeline()
    lat = nx.grid_2d_graph(layout[0], layout[1], periodic=periodicity)
    try:
        spinsys = systypes[sysname.lower()]
    except KeyError as exc:
        raise NotImplementedError(
            "Only Ising, Heisenberg, FermiHubbard and BoseHubbard models are supported."
        ) from exc

    periodic = "closed" if periodicity else "open"
    lattice = "rectangular" if layout[0] > 1 else "chain"
    sites = f"{layout[0]}x{layout[1]}"

    file_name = f"datasets/qspin/{sysname.lower()}/{periodic}/{lattice}/{sites}/{sysname.lower()}_{periodic}_{lattice}_{sites}"
    system = spinsys(num_systems, layout, lat, periodicity)
    for _ in (pbar := tqdm(range(1))):
        data_pipeline.pipeline(sysname.lower(), system, filepath=file_name, prog_bar=pbar)
        gc.collect()