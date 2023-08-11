# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of quantum many-body physics data generation pipeline."""

import os
import sys
from pathlib import Path
import numpy as np
import scipy as sp
from tqdm.auto import tqdm
import pennylane as qml

from .pipeline import DataPipeline

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OMP_PROC_BIND"] = "spread"
os.environ["OMP_PLACES"] = "threads"
sys.setrecursionlimit(10000)


# pylint: disable=fixme
# pylint: disable=invalid-name
class SpinDataPipeline(DataPipeline):
    """Quantum Spins Data Pipeline Class"""

    @staticmethod
    def get_groundstate(ham, use_sparse=False):
        """Builds the ground state data for the spin system using"""
        if use_sparse:
            sp_ham = qml.utils.sparse_hamiltonian(ham)
            eigvals, eigvecs = sp.linalg.eigh(sp_ham)
        else:
            mat = qml.matrix(ham)
            eigvals, eigvecs = np.linalg.eigh(mat)
        states = np.transpose(eigvecs)  # doing it this way is slow
        return np.min(eigvals), states[np.argmin(eigvals)]

    @staticmethod
    def get_groundstate_lanczos(ham):
        """Builds the ground state for the spin system using Lanczos"""
        try:
            # pylint: disable=import-outside-toplevel
            import pylanczos
        except ImportError:
            print("Please install pylancoz for using Lancoz method: pip install pylancoz")
        sp_ham = qml.utils.sparse_hamiltonian(ham)
        engine = pylanczos.PyLanczos(sp_ham, False, 1)
        eigval, eigvec = engine.run()
        return eigval, eigvec

    @staticmethod
    def corr_function(i, j):
        """Builds correlation function X(i)*X(j) + Y(i)*Y(j) + Z(i)*Z(j)"""
        # TODO: Add more correlation functions for each
        ops = []
        for op in [qml.PauliX, qml.PauliY, qml.PauliZ]:
            if i != j:
                ops.append(op(i) @ op(j))
            else:
                ops.append(qml.Identity(i))
        return ops

    @staticmethod
    def build_exact_corrmat(coups, corrs, circuit, psi, num_qubits):
        """Builds a correlation matrix for the correlation matrix"""
        corr_mat_exact = np.zeros((num_qubits, num_qubits))
        for idx, (i, j) in enumerate(coups):
            corr = corrs[idx]
            if i == j:
                corr_mat_exact[i][j] = 1.0
            else:
                corr_mat_exact[i][j] = (
                    np.sum(np.array([circuit(psi, observables=[o]) for o in corr]).T) / 3
                )
                corr_mat_exact[j][i] = corr_mat_exact[i][j]
        return corr_mat_exact

    @staticmethod
    def circuit(psi, observables):
        """Builds a ground state preparation circuit"""
        psi = psi / np.linalg.norm(psi)  # normalize the state
        qml.QubitStateVector(psi, wires=range(int(np.log2(len(psi)))))
        return [qml.expval(o) for o in observables]

    @staticmethod
    def gen_class_shadow(num_sites, groundstate):
        """Build classical shadows for the spin system"""
        dev = qml.device("default.qubit", wires=num_sites, shots=1000)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(groundstate, wires=range(num_sites))
            return qml.classical_shadow(wires=range(num_sites))

        outcomes, unitary_ensmb = circuit()

        return outcomes, unitary_ensmb

    @staticmethod
    def gen_order_params(groundstate, order_mat):
        """Build order parameter values"""
        order = np.conj(groundstate) @ order_mat @ groundstate
        return order

    def pipeline(self, sysname, spinsys, filepath="", prog_bar=None):
        """Implements the data generation pipeline"""

        if prog_bar is None:
            raise ValueError("Please initialize progress bar for verbose output")

        prog_bar.set_description("Spin System Generation")

        if not filepath:
            filepath = f"data/qspin/{sysname}/{spinsys.layout}-{spinsys.num_systems}"

        path = "/".join(filepath.split("/")[:-1])
        Path(path).mkdir(parents=True, exist_ok=True)

        f = self.append_data(filepath + "_full.dat")
        present_keys = []

        f["spin_system"] = spinsys.data

        prog_bar.set_description("Build Hamiltonians")
        if "hamiltonians" not in present_keys:
            f["hamiltonians"] = spinsys.build_hamiltonian()
        f["parameters"] = spinsys.parameters

        prog_bar.set_description("GroundState Calculation")
        if "ground_energies" not in present_keys:
            if spinsys.num_sites < 8:  # use exact solution
                groundstate_data = np.array(
                    [
                        list(self.get_groundstate(f["hamiltonians"][i]))
                        for i in tqdm(range(spinsys.num_systems))
                    ],
                    dtype=object,
                )
            else:  # use lanczos
                prog_bar.set_description(
                    "GroundState Calculation (Lanczos with Sparse Hamiltonians)"
                )
                groundstate_data = np.array(
                    [self.get_groundstate_lanczos(ham) for ham in tqdm(f["hamiltonians"])]
                )

            f["ground_energies"] = np.array(groundstate_data[:, 0].astype("complex128"))
            f["ground_states"] = np.array(groundstate_data[:, 1])

        # TODO: Add more correlation functions and uncomment the below
        # prog_bar.set_description("CorrelationMat Calculation")
        # coups = list(it.product(range(spinsys.num_sites), repeat=2))
        # corrs = [self.corr_function(i, j) for i, j in coups]
        # expval_exact = []
        # for x in tqdm(groundstate_data[:, 1]):
        #     expval_exact.append(self.build_exact_corrmat(coups, corrs,
        # circuit_exact, x, spinsys.num_sites)) # for x in data
        # f['correlation_matrix'] = expval_exact

        prog_bar.set_description("ClassShadow Calculation")
        if "shadow_basis" not in present_keys:
            shadows = []
            for x in tqdm(groundstate_data[:, 1]):
                shadows.append(self.gen_class_shadow(spinsys.num_qubits, x))
            shadows = np.array(shadows)
            f["shadow_basis"] = shadows[:, 1]
            f["shadow_meas"] = shadows[:, 0]

        prog_bar.set_description("Order Parameter Calculation")
        if "order_params" not in present_keys:
            f["order_params"] = np.array(
                [
                    spinsys.build_order_params(groundstate)
                    for groundstate in tqdm(groundstate_data[:, 1])
                ]
            )
            f["num_phases"] = spinsys.num_phases

        self.write_data(f, filepath + "_full.dat")
        self.write_data_seperated(f, filepath)
