# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility objects and methods for quantum many-body physics data generation."""

import functools
import numpy as np
import networkx as nx
from tqdm import tqdm
import pennylane as qml


# pylint: disable=fixme, invalid-name, too-many-instance-attributes
class SpinSystem:
    """Spin System Class"""

    def __init__(
        self,
        num_systems: int,
        stype: str,
        periodicity: bool,
        lattice: nx.generators.lattice,
        layout: tuple,
    ) -> None:
        self.num_systems = num_systems
        self.type = stype
        if self.type not in ["Ising", "Heisenberg", "FermiHubbard", "BoseHubbard"]:
            raise NotImplementedError(
                "Only Ising, Heisenberg, FermiHubbard and BoseHubbard models are supported."
            )
        self.periodicity = periodicity
        self.lattice = lattice
        self.layout = layout
        self.build_lattice_structure()
        self.num_sites = len(self.nodes.values())
        self.hamiltonian = None
        self.ham_string = None
        self.num_qubits = None

    def build_lattice_structure(self):
        """Builds the lattice structure for the spin system"""
        node_list = {node: idx for idx, node in enumerate(self.lattice.nodes.keys())}
        edge_list = []
        for nodes in self.lattice.edges.keys():
            edge_list.append((node_list[nodes[0]], node_list[nodes[1]]))
        self.nodes, self.edges = node_list, edge_list

    def build_phase_data(self, psi, order_op):
        """Builds the phase data for the spin system"""
        return np.conjugate(psi.T) @ order_op @ psi


# pylint: disable=line-too-long, anomalous-backslash-in-string, too-many-arguments, too-many-instance-attributes, too-many-arguments, too-many-instance-attributes
class IsingModel(SpinSystem):
    """IsingModel System Class"""

    def __init__(self, num_systems, layout, lattice, periodicity, J=-1) -> None:
        super().__init__(num_systems, "Ising", periodicity, lattice, layout)
        self.J = J
        # self.h = np.linspace(0, -4*self.J, self.num_systems)
        self.m = layout[0]
        self.n = layout[1]
        self.h = np.linspace(
            0,
            4 * np.abs(self.J) * len(self.lattice.edges) / self.num_sites,
            self.num_systems,
        )
        self.parameters = {"J": self.J, "h": self.h}
        self.num_phases = 2
        self.order_mat = None
        self.ham_string = r"J\sum_{\langle i,j\rangle}^{} \sigma_i^z\sigma_j^z + h\sum_i \sigma_i^x"
        self.order_params = {
            "M": r"The order parameter is the magnetization in the Z direction, defined by \langle M_z \rangle =\langle |\sum_i \sigma_i^z|\rangle"
        }
        self.data = {
            "name": "Transverse Field Ising",
            "parameter": "External magnetic field in X direction, h",
            "ham_eq": self.ham_string,
            "order_params": self.order_params,
        }
        self.num_qubits = self.num_sites
        # self.hamiltonian = self.build_hamiltonian()

    def build_hamiltonian(self):
        """Create hamiltonians for all h magnetic field values."""
        n_particles = self.m * self.n
        edges = self.edges
        hamils = []
        Jz = self.J
        for i in tqdm(range(self.num_systems)):
            coeffs = [Jz] * len(edges)
            observables = [qml.PauliZ(i) @ qml.PauliZ(j) for (i, j) in edges]
            coeffs = coeffs + [self.h[i]] * n_particles
            observables = observables + [
                qml.PauliX(i) for i in range(n_particles)
            ]  # observables + external mag. field
            H = qml.Hamiltonian(coeffs, observables)
            hamils.append(H)

        return hamils

    def build_order_params(self, psi0):
        """Builds order parameter and phase data for Spin systems"""
        m, n = self.m, self.n
        Mz = qml.Hamiltonian([1] * m * n, [qml.PauliZ(i) for i in range(m * n)])
        Mz = qml.utils.sparse_hamiltonian(Mz, wires=Mz.wires)
        Mz = np.abs(np.array((Mz)))
        self.order_mat = Mz
        return self.build_phase_data(psi0, Mz)


# pylint: disable=line-too-long, anomalous-backslash-in-string, too-many-arguments, too-many-instance-attributes, too-many-instance-attributes
class HeisenbergModel(SpinSystem):
    """HeisenbergModel System Class"""

    def __init__(self, num_systems, layout, lattice, periodicity, J=-1) -> None:
        super().__init__(num_systems, "Heisenberg", periodicity, lattice, layout)
        self.Jxy = J
        self.Jz = np.linspace(0, -2, self.num_systems)
        self.parameters = {"Jxy": self.Jxy, "Jz": self.Jz}
        self.num_phases = 2
        # self.hamiltonian = self.build_hamiltonian()
        self.m = layout[0]
        self.n = layout[1]
        self.order_mat = None
        self.ham_string = r"J_{xy}\sum_{\langle i,j\rangle}(\sigma_i^x\sigma_j^x+\sigma_i^y\sigma_j^y) + J_z\sum_{\langle i,j\rangle} \sigma_i^z \sigma_j^z"
        self.order_params = {
            "M": r"The order parameter is the magnetization in the Z direction, defined by \langle M_z \rangle =\langle |\sum_i \sigma_i^z|\rangle"
        }
        self.data = {
            "name": "Heisenberg XXZ",
            "parameter": "Z coupling constant, J_z",
            "ham_eq": self.ham_string,
            "order_params": self.order_params,
        }
        self.num_qubits = self.num_sites

    def build_hamiltonian(self):
        """Create hamiltonians for all Jz coupling constants."""
        edges = self.edges
        hamils = []
        Jxy = self.Jxy
        for Jz in tqdm(self.Jz):
            coeffs = [Jz] * len(edges) + [Jxy] * len(edges) + [Jxy] * len(edges)
            observables = (
                [qml.PauliZ(i) @ qml.PauliZ(j) for (i, j) in edges]
                + [qml.PauliX(i) @ qml.PauliX(j) for (i, j) in edges]
                + [qml.PauliY(i) @ qml.PauliY(j) for (i, j) in edges]
            )
            H = qml.Hamiltonian(coeffs, observables)
            hamils.append(H)

        return hamils

    def build_order_params(self, psi0):
        """Builds order parameter and phase data for Spin systems"""
        m, n = self.m, self.n
        Mz = qml.Hamiltonian([1] * m * n, [qml.PauliZ(i) for i in range(m * n)])
        Mz = qml.utils.sparse_hamiltonian(Mz, wires=Mz.wires)
        Mz = np.abs(np.array((Mz)))
        self.order_mat = Mz
        return self.build_phase_data(psi0, Mz)


# pylint: disable=line-too-long, anomalous-backslash-in-string, too-many-arguments, too-many-instance-attributes
class FermiHubbardModel(SpinSystem):
    """FermiHubbard System Class"""

    def __init__(self, num_systems, layout, lattice, periodicity, t=1) -> None:
        super().__init__(num_systems, "FermiHubbard", periodicity, lattice, layout)
        self.t = t
        self.U = np.linspace(0, 300 * self.t, self.num_systems)
        self.build_qubit_structure()
        self.parameters = {"t": self.t, "U": self.U}
        self.ham_string = r"-t ( \sum_{\langle i, j\rangle, \sigma} \hat{c}^\dagger_i\hat{c}_j + h.c.) + U (\sum_i )\hat{n}_{i\uparrow}\hat{n}_{i\downarrow}"
        self.order_params = {
            "M": r"The sum of absolute values of the long-range order components of local magnetic moments, defined by M=\sum_i |\frac{n_{i\uparrow} - n_{i\downarrow}}{2}|"
        }
        self.data = {
            "name": "Fermi-Hubbard",
            "parameter": "Hopping parameter t, Interaction parameter u",
            "ham_eq": self.ham_string,
            "order_params": self.order_params,
        }
        self.num_phases = None
        self.num_qubits = 2 * self.num_sites

    def build_hamiltonian(self):
        """Hamiltonian obtained from arXiv:2112.14077"""
        hamils = []
        t_term = qml.Hamiltonian([], [])
        for i, j in self.sites_pair:
            ham = qml.Hamiltonian(*qml.qchem.jordan_wigner([i, j])) + qml.Hamiltonian(
                *qml.qchem.jordan_wigner([j, i])
            )
            qml.simplify(ham)
            t_term += ham
        qml.simplify(-self.t * t_term)
        for idx in tqdm(range(self.num_systems), leave=False):
            u_term = qml.Hamiltonian([], [])
            for i in self.sites:
                u_term += qml.Hamiltonian(
                    [self.U[idx] / 4], [qml.PauliZ(i) @ qml.PauliZ(i + len(self.sites))]
                )

            hamils.append(t_term + u_term)
        return hamils

    def build_qubit_structure(self):
        """Build qubits structure for the spin system lattice"""
        self.sites = list(
            self.nodes.values()
        )  # {node:idx for idx, node in enumerate(self.lattice.nodes.keys())}
        iter_sites = self.sites if self.periodicity else self.sites[:-1]
        self.sites_pair = [[(i) % self.num_sites, (i + 1) % self.num_sites] for i in iter_sites] + [
            [(i) % (self.num_sites) + self.num_sites, (i + 1) % (self.num_sites) + self.num_sites]
            for i in iter_sites
        ]

    def build_order_params(self, psi0):
        """Build order parameters data for the spin system"""
        m_lor_op = 0
        for i in tqdm(self.sites, leave=False):
            opn1 = qml.Hamiltonian(*qml.qchem.jordan_wigner([i, i]))
            n1 = (
                psi0.T.conjugate()
                @ qml.utils.sparse_hamiltonian(opn1, wires=range(2 * self.num_sites))
                @ psi0
            )
            opn2 = qml.Hamiltonian(
                *qml.qchem.jordan_wigner([i + len(self.sites), i + len(self.sites)])
            )
            n2 = (
                psi0.T.conjugate()
                @ qml.utils.sparse_hamiltonian(opn2, wires=range(2 * self.num_sites))
                @ psi0
            )
            m_lor_op += 0.5 * np.abs(n1 - n2)

        m2_op = 0
        for i in tqdm(self.sites, leave=False):
            ops2 = qml.Hamiltonian(
                [0.25, -0.25, 0.25, -0.25],
                [
                    qml.Identity(i),
                    qml.PauliZ(i),
                    qml.Identity(i + len(self.sites)),
                    qml.PauliZ(i + len(self.sites)),
                ],
            )
            mat_ops2 = qml.utils.sparse_hamiltonian(ops2, wires=range(2 * self.num_sites))
            m2_op += psi0.T.conjugate() @ mat_ops2 @ mat_ops2 @ psi0
        m2_op /= len(self.sites)
        return m_lor_op, m2_op


# pylint: disable=line-too-long, anomalous-backslash-in-string, too-many-arguments, too-many-instance-attributes
class BoseHubbardModel(SpinSystem):
    """BoseHubbard System Class"""

    def __init__(self, num_systems, layout, lattice, periodicity, t=1) -> None:
        super().__init__(num_systems, "BoseHubbard", periodicity, lattice, layout)
        self.t = t
        self.U = np.linspace(0, 7 * self.t, self.num_systems)
        self.build_qubit_structure()
        self.parameters = {"t": self.t, "U": self.U}
        self.ham_string = r"-t ( \sum_{\langle i, j\rangle} \hat{c}^\dagger_i\hat{c}_j + h.c.) + U (\sum_i )\hat{n}_{i}\hat{n}_{i}"
        self.order_params = {
            "M": "The sum of absolute values of the long-range order components of local magnetic moments, defined by M=\sum_i |\frac{n_{i} - n_{i+L}}{2}|"
        }
        self.data = {
            "name": "Bose-Hubbard",
            "parameter": "Hopping parameter t, Interaction parameter u",
            "ham_eq": self.ham_string,
            "order_params": self.order_params,
        }
        self.num_phases = None
        self.num_qubits = 2 * self.num_sites

    def build_qubit_structure(self):
        """Build qubits structure for the spin system lattice"""
        self.sites = list(
            self.nodes.values()
        )  # {node:idx for idx, node in enumerate(self.lattice.nodes.keys())}
        iter_sites = self.sites if self.periodicity else self.sites[:-1]
        self.sites_pair = [[(i) % self.num_sites, (i + 1) % self.num_sites] for i in iter_sites]

    # pylint: disable=unnecessary-lambda-assignment
    def build_hamiltonian(self):
        """Build Hamiltonians for Bose-Hubbar spin systems"""
        sigmap = lambda qb: 0.5 * (qml.PauliX(qb) + 1j * qml.PauliY(qb))
        sigmam = lambda qb: 0.5 * (qml.PauliX(qb) - 1j * qml.PauliY(qb))
        identp = lambda qb: 0.5 * qml.op_sum(qml.Identity(qb), qml.PauliZ(qb))
        identm = lambda qb: 0.5 * qml.op_sum(qml.Identity(qb), -qml.PauliZ(qb))

        trdict = {"11": identm, "00": identp, "01": sigmap, "10": sigmam}

        def bop_map(t, base=0, dag=False):
            r"""Helper function for then binary mapping from Bosons fock space to qubit via arXiv:2105.12563"""

            a_op = (
                np.sqrt(2 ** (t - 1)) * sigmam(base)
                if not dag
                else np.sqrt(2 ** (t - 1)) * sigmap(base)
            )

            if t > 1:
                a_op @= qml.simplify(
                    functools.reduce(
                        lambda i, j: i @ j,
                        [sigmap(base + i) if not dag else sigmam(base + i) for i in range(1, t)],
                    )
                )  # sqrt(2**t) term

                b_op = (
                    [
                        functools.reduce(
                            lambda i, j: i @ j,
                            [
                                trdict[k + l](base + idx + 1)
                                for idx, (k, l) in enumerate(
                                    zip(bin(m)[2:].zfill(t - 1), bin(m - 1)[2:].zfill(t - 1))
                                )
                            ],
                        )
                        for m in range(1, 2 ** (t - 1))
                    ]
                    if not dag
                    else [
                        functools.reduce(
                            lambda i, j: i @ j,
                            [
                                trdict[k + l](base + idx + 1)
                                for idx, (k, l) in enumerate(
                                    zip(bin(m - 1)[2:].zfill(t - 1), bin(m)[2:].zfill(t - 1))
                                )
                            ],
                        )
                        for m in range(1, 2 ** (t - 1))
                    ]
                )  # creation and anhillation operators for the left/right summation terms

                for idx, i in enumerate(range(0, 2 ** (t - 1) - 1)):
                    a_op = qml.op_sum(
                        a_op, qml.s_prod(np.sqrt(i + 1), qml.simplify(identp(base) @ b_op[idx]))
                    )

                for idx, i in enumerate(range(2 ** (t - 1) + 1, 2 ** (t))):
                    a_op = qml.op_sum(
                        a_op, qml.s_prod(np.sqrt(i), qml.simplify(identm(base) @ b_op[idx]))
                    )

            return qml.simplify(a_op)

        def binary_bosonic_map(boson_op, fock_sites=2):
            r"""Binary mapping from Bosons fock space to qubit via arXiv:2105.12563"""

            if fock_sites <= 0:
                raise ValueError(f"Length of fock space must be positive, got {fock_sites}.")

            b_ops = []
            for ind, op in enumerate(boson_op):
                if op >= 0:
                    b_op = bop_map(fock_sites, base=ind * fock_sites, dag=not bool(op))
                    if op > 1:
                        b_op = qml.pow(b_op, z=op, lazy=False)
                else:
                    b_op = qml.Identity(ind * fock_sites)
                    for idx in range(1, fock_sites):
                        b_op = qml.prod(b_op, qml.Identity(ind * fock_sites + idx))
                b_ops.append(b_op)
            # pylint: disable=unnecessary-lambda
            bos_op = functools.reduce(lambda i, j: qml.prod(i, j), b_ops)
            # pylint: disable=protected-access
            bos_op._wires = qml.wires.Wires(range(len(boson_op) * fock_sites))
            return qml.simplify(bos_op)

        def op_sum_to_ham(op_sum_obj):
            coeffs, ops = [], []
            for term in list(op_sum_obj):
                try:
                    coeffs.append(term.data[0][0])
                except IndexError:
                    coeffs.append(term.data[0])
                if any(
                    [
                        isinstance(term.hyperparameters["base"], x)
                        for x in [qml.Identity, qml.PauliX, qml.PauliY, qml.PauliZ]
                    ]
                ):
                    ops.append(term.hyperparameters["base"])
                else:
                    if isinstance(term.hyperparameters["base"], qml.ops.op_math.Pow):
                        t_op = term.hyperparameters["base"]
                        term.hyperparameters["base"] = qml.prod(
                            *[
                                qml.pow(x, t_op.hyperparameters["z"], lazy=False)
                                for x in t_op.hyperparameters["base"].operands
                            ]
                        )
                    ops.append(
                        qml.simplify(
                            functools.reduce(
                                lambda i, j: i @ j, term.hyperparameters["base"].operands
                            )
                        )
                    )
            return qml.Hamiltonian(coeffs, ops)

        hamils = []
        fock_sites = 2

        t_term = 0 * qml.op_sum(qml.Identity(0), qml.Identity(1))
        for i, j in self.sites_pair:
            b_op1, b_op2 = [], []
            for k in range(max(i, j) + 1):
                if k == i:
                    b_op1.append(1)
                    b_op2.append(0)
                elif k == j:
                    b_op1.append(0)
                    b_op2.append(1)
                else:
                    b_op1.append(-1)
                    b_op2.append(-1)
            ham = qml.simplify(
                binary_bosonic_map(b_op1, fock_sites)
                + qml.simplify(binary_bosonic_map(b_op2, fock_sites))
            )
            t_term += ham
        qml.simplify(-self.t * t_term)

        for idx in tqdm(range(self.num_systems), leave=False):
            u_term = 0 * qml.op_sum(qml.Identity(0), qml.Identity(1))
            for i in self.sites:
                b_op1, b_op2 = [], []
                for k in range(i + 1):
                    if k == i:
                        b_op1.append(1)
                        b_op2.append(0)
                    else:
                        b_op1.append(-1)
                        b_op2.append(-1)

                u_term += qml.simplify(
                    float(self.U[idx])
                    * qml.prod(
                        binary_bosonic_map(b_op1, fock_sites), binary_bosonic_map(b_op2, fock_sites)
                    )
                )

            simp_ham = op_sum_to_ham(qml.simplify(t_term + u_term))
            hamils.append(qml.simplify(simp_ham))

        return hamils

    def build_order_params(self, psi0):
        """Build order parameters data for the spin system"""
        m_lor_op = 0
        for i in tqdm(self.sites, leave=False):
            opn1 = qml.Hamiltonian(*qml.qchem.jordan_wigner([i, i]))
            n1 = (
                psi0.T.conjugate()
                @ qml.utils.sparse_hamiltonian(opn1, wires=range(2 * self.num_sites))
                @ psi0
            )
            opn2 = qml.Hamiltonian(
                *qml.qchem.jordan_wigner([i + len(self.sites), i + len(self.sites)])
            )
            n2 = (
                psi0.T.conjugate()
                @ qml.utils.sparse_hamiltonian(opn2, wires=range(2 * self.num_sites))
                @ psi0
            )
            m_lor_op += 0.5 * np.abs(n1 - n2)

        m2_op = 0
        for i in tqdm(self.sites, leave=False):
            ops2 = qml.Hamiltonian(
                [0.25, -0.25, 0.25, -0.25],
                [
                    qml.Identity(i),
                    qml.PauliZ(i),
                    qml.Identity(i + len(self.sites)),
                    qml.PauliZ(i + len(self.sites)),
                ],
            )
            mat_ops2 = qml.utils.sparse_hamiltonian(ops2, wires=range(2 * self.num_sites))
            m2_op += psi0.T.conjugate() @ mat_ops2 @ mat_ops2 @ psi0
        m2_op /= len(self.sites)
        return m_lor_op, m2_op
