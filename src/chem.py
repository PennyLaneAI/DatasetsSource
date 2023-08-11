# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility objects and methods for quantum chemistry data generation."""

import re
import numpy as np
import pennylane as qml

# fmt: off
elem_symbols = [
        '_',
        'H',  'He',
        'Li', 'Be', 'B',  'C',  'N',  'O',  'F',  'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P',  'S',  'Cl', 'Ar',
        'K',  'Ca', 'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
        'Ge', 'As', 'Se', 'Br', 'Kr',
        'Rb', 'Sr', 'Y',  'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
        'Sn', 'Sb', 'Te', 'I', 'Xe',
        'Cs', 'Ba',
        'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
        'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
        'Fr', 'Ra',
        'Ac', 'Th', 'Pa', 'U',  'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
        'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]
# fmt: on

geom_struct = {
    "H2": lambda x: np.array([[-0.5 * x, 0.0, 0.0], [0.5 * x, 0.0, 0.0]]),  # linear
    "HeH+": lambda x: np.array([[x / 3, 0.0, 0.0], [-2 * x / 3, 0.0, 0.0]]),  # linear
    "H3+": lambda x: np.array(
        [
            [0.0, x / np.sqrt(3), 0.0],  # planar - equilateral
            [-0.5 * x, -0.5 * x / np.sqrt(3), 0.0],
            [0.5 * x, -0.5 * x / np.sqrt(3), 0.0],
        ]
    ),
    "H4": lambda x: np.array(
        [
            [-1.5 * x, 0.0, 0.0],
            [-0.5 * x, 0.0, 0.0],  # linear - chain
            [0.5 * x, 0.0, 0.0],
            [1.5 * x, 0.0, 0.0],
        ]
    ),
    "LiH": lambda x: np.array([[x / 4, 0.0, 0.0], [-3 * x / 4, 0.0, 0.0]]),  # linear
    "HF": lambda x: np.array([[-9 * x / 10, 0.0, 0.0], [x / 10, 0.0, 0.0]]),  # linear
    "OH-": lambda x: np.array([[x / 9, 0.0, 0.0], [-8 * x / 9, 0.0, 0.0]]),  # linear
    "H6": lambda x: np.array(
        [
            [-2.5 * x, 0.0, 0.0],
            [-1.5 * x, 0.0, 0.0],  # linear - chain
            [-0.5 * x, 0.0, 0.0],
            [0.5 * x, 0.0, 0.0],
            [1.5 * x, 0.0, 0.0],
            [2.5 * x, 0.0, 0.0],
        ]
    ),
    "BeH2": lambda x: np.array([[0.0, 0.0, 0.0], [-x, 0.0, 0.0], [x, 0.0, 0.0]]),  # linear
    "H2O": lambda x: np.array(
        [
            [x * np.cos(0.911725) / 5, 0.0, 0.0],  # H-O-H 104.476
            [-4 * x * np.cos(0.911725) / 5, -x * np.sin(0.911725), 0.0],
            [-4 * x * np.cos(0.911725) / 5, x * np.sin(0.911725), 0.0],
        ]
    ),
    "H8": lambda x: np.array(
        [
            [-3.5 * x, 0.0, 0.0],
            [-2.5 * x, 0.0, 0.0],  # linear - chain
            [-1.5 * x, 0.0, 0.0],
            [-0.5 * x, 0.0, 0.0],
            [0.5 * x, 0.0, 0.0],
            [1.5 * x, 0.0, 0.0],
            [2.5 * x, 0.0, 0.0],
            [3.5 * x, 0.0, 0.0],
        ]
    ),
    "BH3": lambda x: np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, x, 0.0],  # planaer H-B-H 120
            [-x * np.sqrt(3) / 2, -0.5 * x, 0.0],
            [x * np.sqrt(3) / 2, -0.5 * x, 0.0],
        ]
    ),
    "NH3": lambda x: np.array(
        [
            [0.0, 0.0, x * 0.375324205 * 3 / 10],
            # np.sqrt(4*np.cos(0.976)**2 - 1)/np.sqrt(3) # HNH = 106.78
            # 2 * np.sin(0.923175)/np.sqrt(3)
            # np.sin(0.923175)/np.sqrt(3)
            [0.0, 2 * x * 0.4634468, -x * 0.375324205 * 7 / 10],
            [-x * np.sin(0.9318313), -x * 0.4634468, -x * 0.375324205 * 7 / 10],
            [x * np.sin(0.9318313), -x * 0.4634468, -x * 0.375324205 * 7 / 10],
        ]
    ),
    "H3O+": lambda x: np.array(
        [
            [0.0, 0.0, x * 0.3019369 * 3 / 11],
            [0.0, 2 * x * 0.4766635, -x * 0.3019369 * 8 / 11],
            [-x * np.sin(0.97127573), -x * 0.4766635, -x * 0.3019369 * 8 / 11],
            [x * np.sin(0.97127573), -x * 0.4766635, -x * 0.3019369 * 8 / 11],
        ]
    ),
    "CH4": lambda x: np.array(
        [
            [0.0, 0.0, 0.0],
            [x / np.sqrt(3), x / np.sqrt(3), x / np.sqrt(3)],
            [-x / np.sqrt(3), -x / np.sqrt(3), x / np.sqrt(3)],
            [-x / np.sqrt(3), x / np.sqrt(3), -x / np.sqrt(3)],
            [x / np.sqrt(3), -x / np.sqrt(3), -x / np.sqrt(3)],
        ]
    ),
    "NH4+": lambda x: np.array(
        [
            [0.0, 0.0, 0.0],
            [
                x / np.sqrt(3),
                x / np.sqrt(3),
                x / np.sqrt(3),
            ],
            [-x / np.sqrt(3), -x / np.sqrt(3), x / np.sqrt(3)],
            [-x / np.sqrt(3), x / np.sqrt(3), -x / np.sqrt(3)],
            [x / np.sqrt(3), -x / np.sqrt(3), -x / np.sqrt(3)],
        ]
    ),
    "Li2": lambda x: np.array([[-0.5 * x, 0.0, 0.0], [0.5 * x, 0.0, 0.0]]),  # linear
    "C2": lambda x: np.array([[-0.5 * x, 0.0, 0.0], [0.5 * x, 0.0, 0.0]]),  # linear
    "N2": lambda x: np.array([[-0.5 * x, 0.0, 0.0], [0.5 * x, 0.0, 0.0]]),  # linear
    "O2": lambda x: np.array([[-0.5 * x, 0.0, 0.0], [0.5 * x, 0.0, 0.0]]),  # linear
    "He2": lambda x: np.array([[-0.5 * x, 0.0, 0.0], [0.5 * x, 0.0, 0.0]]),  # linear
    "H10": lambda x: np.array(  # linear - chain
        [
            [-4.5 * x, 0.0, 0.0],
            [-3.5 * x, 0.0, 0.0],
            [-2.5 * x, 0.0, 0.0],
            [-1.5 * x, 0.0, 0.0],
            [-0.5 * x, 0.0, 0.0],
            [0.5 * x, 0.0, 0.0],
            [1.5 * x, 0.0, 0.0],
            [2.5 * x, 0.0, 0.0],
            [3.5 * x, 0.0, 0.0],
            [4.5 * x, 0.0, 0.0],
        ]
    ),
    "NeH+": lambda x: np.array([[x / 11, 0.0, 0.0], [-10 * x / 11, 0.0, 0.0]]),  # linear
    "HCN": lambda x: np.array(  # depends on the angle
        [
            [-1.642 * np.cos(x), 1.642 * np.sin(x), 0.0],  # H-O-H 104.476
            [0.0, 0.0, 0.0],
            [1.156, 0.0, 0.0],
        ]
    ),
    "H5": lambda x: np.array(
        [
            [-2.0 * x, 0.0, 0.0],
            [-1.0 * x, 0.0, 0.0],  # linear - chain
            [0.0, 0.0, 0.0],
            [1.0 * x, 0.0, 0.0],
            [2.0 * x, 0.0, 0.0],
        ]
    ),
    "H7": lambda x: np.array(
        [
            [-3.0 * x, 0.0, 0.0],
            [-2.0 * x, 0.0, 0.0],
            [-1.0 * x, 0.0, 0.0],  # linear - chain
            [0.0, 0.0, 0.0],
            [1.0 * x, 0.0, 0.0],
            [2.0 * x, 0.0, 0.0],
            [3.0 * x, 0.0, 0.0],
        ]
    ),
    "CO": lambda x: np.array([[0.0, 0.0, 0.0], [0.0, 0.0, x]]),  # linear
}


bond_struct = {
    "H2": np.linspace(0.5, 2.5, 11),  # linear
    "HeH+": np.linspace(0.5, 2.5, 11),  # np.linspace(0.5, 2.1, 41),  # linear
    "H3+": np.linspace(0.5, 2.1, 41),
    "H4": np.linspace(0.5, 1.3, 41),  # linear - chain
    "LiH": np.linspace(0.9, 2.1, 41),  # linear
    "HF": np.linspace(0.5, 2.1, 41),  # linear
    "OH-": np.linspace(0.5, 2.1, 41),  # linear
    "H6": np.linspace(0.5, 1.3, 41),  # linear - chain
    "BeH2": np.linspace(0.5, 2.1, 41)[37:38],  # linear
    "H2O": np.linspace(0.5, 2.1, 41),  # H-O-H 104.476
    "H8": np.linspace(0.5, 0.9, 41)[33:],  # linear - chain
    "BH3": np.linspace(0.5, 1.7, 41)[33:],  # planaer H-B-H 120
    "NH3": np.linspace(0.5, 1.7, 41)[3:4],
    "H3O+": np.linspace(0.7, 1.3, 41),
    "CH4": np.linspace(0.5, 2.5, 11),
    "NH4+": np.linspace(0.5, 2.1, 41),
    "Li2": np.linspace(1.5, 3.5, 11),  # linear
    "C2": np.linspace(0.5, 2.5, 11),  # linear
    "N2": np.linspace(0.5, 2.5, 11),  # linear
    "O2": np.linspace(0.5, 2.5, 11),  # linear
    "He2": np.linspace(4.5, 6.5, 11),  # linear
    "H10": np.linspace(0.5, 1.5, 11),  # linear
    "NeH+": np.linspace(0.5, 2.5, 11),  # linear
    "HCN": np.linspace(0.0, np.pi, 11),  # angular
    "H5": np.linspace(0.5, 1.5, 11),  # linear
    "H7": np.linspace(0.5, 1.5, 11),  # linear
    "CO": np.linspace(0.5, 2.5, 11),  # linear
}


def read_xyz(path):
    """Reads a molecule from its (custom) xyz file"""
    if path.split(".")[-1] != "xyz":
        raise NotImplementedError("Currently supports only xyz files")
    with open(path, "r", encoding="utf-8") as xyz_file:
        lines = xyz_file.readlines()
        num_atoms = int(lines[0])
        mol_name, charge = (lines[1].strip("\n")).split(" ")
        charge = int(charge)
        opt_bond = float(lines[3])
        symbols = []
        geometry = []
        for line in lines[4:]:
            data = re.split(" |\t", line)  # multiple delimiters
            symbols.append(data[0])
            geometry.append(np.array(list(map(float, data[1:]))))
    return (
        num_atoms,
        mol_name,
        symbols,
        charge,
        np.array(geometry),
        opt_bond,
    )


def is_commuting_obs(ham1, ham2):
    """Check for commutivity between two observables."""
    commute = True
    for op1 in ham1.ops:
        for op2 in ham2.ops:
            if not qml.grouping.is_commuting(op1, op2):
                commute = False
                break
    return commute


def triple_excitation_matrix(gamma):
    """Build Triple Excitation gates for buildding AllSingleDoubleTriples."""
    cos = qml.math.cos(gamma / 2)
    sin = qml.math.sin(gamma / 2)
    mat = qml.math.diag([1.0] * 7 + [cos] + [1.0] * 48 + [cos] + [1.0] * 7)
    mat = qml.math.scatter_element_add(mat, (7, 56), -sin)
    mat = qml.math.scatter_element_add(mat, (56, 7), sin)
    return mat


def core_orbitals(mol):
    """Calculating core orbitals for a given Moelcule object."""
    count = 0
    for elem in mol.symbols:
        atomic_z = elem_symbols.index(elem)
        if atomic_z > 2:
            count += 1
        if atomic_z > 10:
            count += 4
        if atomic_z > 18:
            count += 4
        if atomic_z > 36:
            count += 9
        if atomic_z > 54:
            count += 9
        if atomic_z > 86:
            count += 16
    return list(range(count))


def excitations(electrons, orbitals, delta_sz=0):
    """Calculating single, double and triple excitations."""
    if not electrons > 0:
        raise ValueError(
            f"The number of active electrons has to be greater than 0 \n"
            f"Got n_electrons = {electrons}"
        )

    if orbitals <= electrons:
        raise ValueError(
            f"The number of active spin-orbitals ({orbitals}) "
            f"has to be greater than the number of active electrons ({electrons})."
        )

    if delta_sz not in (0, 1, -1, 2, -2):
        raise ValueError(
            f"Expected values for 'delta_sz' are 0, +/- 1 and +/- 2 but got ({delta_sz})."
        )

    # define the spin projection 'sz' of the single-particle states
    s_z = np.array([0.5 if (i % 2 == 0) else -0.5 for i in range(orbitals)])

    singles = [
        [r, p]
        for r in range(electrons)
        for p in range(electrons, orbitals)
        if s_z[p] - s_z[r] == delta_sz
    ]

    doubles = [
        [s, r, q, p]
        for s in range(electrons - 1)
        for r in range(s + 1, electrons)
        for q in range(electrons, orbitals - 1)
        for p in range(q + 1, orbitals)
        if (s_z[p] + s_z[q] - s_z[r] - s_z[s]) == delta_sz
    ]

    triples = [
        [u, t, s, r, q, p]
        for u in range(electrons)
        for t in range(u + 1, electrons)
        for s in range(t + 1, electrons)
        for r in range(electrons, orbitals)
        for q in range(r + 1, orbitals)
        for p in range(q + 1, orbitals)
        if (s_z[p] + s_z[q] + s_z[r] - s_z[s] - s_z[t] - s_z[u]) == delta_sz
    ]

    return singles, doubles, triples
