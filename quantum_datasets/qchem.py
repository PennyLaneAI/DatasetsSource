# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of quantum chemistry data generation pipeline."""

import os
import sys
import functools as ft
from pathlib import Path
import itertools as it
from operator import itemgetter

import numpy as np
import scipy as sp
from tqdm.auto import tqdm
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.pauli import pauli_word_to_string, string_to_pauli_word
from pyscf import gto

from .pipeline import DataPipeline
from .chem import triple_excitation_matrix, excitations, core_orbitals

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OMP_PROC_BIND"] = "spread"
os.environ["OMP_PLACES"] = "threads"
sys.setrecursionlimit(10000)

qml.disable_return()


# pylint: disable=fixme, invalid-name, line-too-long, no-member
# pylint: disable=too-many-locals, too-many-arguments, too-many-instance-attributes
class ChemDataPipeline(DataPipeline):
    """Quantum Chemistry Data Pipeline Class"""

    def __init__(self):
        self.tapered_hf_state = None
        self.tapered_wire_map = None
        self.opt_asd_params = None
        self.opt_ag_params = None
        self.opt_tap_params = None
        self.aux_ops = None
        self.tapered_aux_ops = None
        self.red_mats = None
        super().__init__()

    # pylint: disable=redefined-outer-name, too-many-branches, too-many-statements
    def run_adaptive_vqe(self, mol, hf_state, ham, classical_energy, use_triples=False):
        """Runs VQE routine implementing AdaptiveGivens template"""

        singles, doubles = qml.qchem.excitations(mol.n_electrons, len(ham.wires))

        triples = [] if not use_triples else excitations(mol.n_electrons, len(ham.wires))[2]

        def circuit_1(params, wires, excitations):
            qml.BasisState(hf_state, wires=wires)
            for i, excitation in enumerate(excitations):
                if len(excitation) == 4:
                    qml.DoubleExcitation(params[i], wires=excitation)
                elif len(excitation) == 2:
                    qml.SingleExcitation(params[i], wires=excitation)
                else:
                    qml.QubitUnitary(triple_excitation_matrix(params[i]), wires=excitation)

        def circuit_2(params, wires, excitations, gates_select, params_select):
            qml.BasisState(hf_state, wires=wires)
            for i, gate in enumerate(gates_select):
                if len(gate) == 4:
                    qml.DoubleExcitation(params_select[i], wires=gate)
                elif len(gate) == 2:
                    qml.SingleExcitation(params_select[i], wires=gate)
                else:
                    qml.QubitUnitary(triple_excitation_matrix(params[i]), wires=gate)
            for i, gate in enumerate(excitations):
                if len(gate) == 4:
                    qml.DoubleExcitation(params[i], wires=gate)
                elif len(gate) == 2:
                    qml.SingleExcitation(params[i], wires=gate)
                else:
                    qml.QubitUnitary(triple_excitation_matrix(params[i]), wires=gate)

        dev = qml.device("lightning.qubit", wires=ham.wires)

        @qml.qnode(dev, diff_method="adjoint")
        def cost_fn_1(param, excitations):
            circuit_1(param, wires=ham.wires, excitations=excitations)
            return qml.expval(ham)

        circuit_gradient = qml.grad(cost_fn_1, argnum=0)
        params = pnp.array([0.0] * len(doubles))
        grads = circuit_gradient(params, excitations=doubles)

        doubles_select = [doubles[i] for i in range(len(doubles)) if abs(grads[i]) > 1.0e-5]
        doubles_indice = [len(triples) + i for i in range(len(doubles)) if abs(grads[i]) > 1.0e-5]

        triples_select = (
            []
            if not use_triples
            else [triples[i] for i in range(len(triples)) if abs(grads[i]) > 1.0e-5]
        )
        triples_indice = (
            [] if not use_triples else [i for i in range(len(triples)) if abs(grads[i]) > 1.0e-5]
        )
        # TODO: Add another training loop for TripleExcitations before DoubleExcitations

        opt = qml.AdamOptimizer(stepsize=0.01)
        params_doubles = pnp.zeros(len(doubles_select) + len(triples_select), requires_grad=True)
        if self.opt_asd_params is not None:
            for idx, exc_itx in enumerate(doubles_indice):
                params_doubles[idx] = self.opt_asd_params[exc_itx]

        params_doubles, prev_energy = opt.step_and_cost(
            cost_fn_1, params_doubles, excitations=triples_select + doubles_select
        )
        for _ in (grad_bar := tqdm(range(100), leave=False)):
            params_doubles, energy = opt.step_and_cost(
                cost_fn_1, params_doubles, excitations=triples_select + doubles_select
            )
            grad_bar.set_description(f"GradExc: {np.round(pnp.abs(energy - prev_energy), 5)}")
            if pnp.abs(energy - prev_energy) <= 1e-3:
                break

            prev_energy = energy

        @qml.qnode(dev, diff_method="adjoint")
        def cost_fn_2(param, excitations, gates_select, params_select):
            circuit_2(
                param,
                wires=ham.wires,
                excitations=excitations,
                gates_select=gates_select,
                params_select=params_select,
            )
            return qml.expval(ham)

        circuit_gradient = qml.grad(cost_fn_2, argnum=0)
        params = pnp.array([0.0] * len(singles))
        grads = circuit_gradient(
            params,
            excitations=singles,
            gates_select=triples_select + doubles_select,
            params_select=params_doubles,
        )

        singles_select = [singles[i] for i in range(len(singles)) if abs(grads[i]) > 1.0e-5]
        singles_indice = [
            len(triples) + len(doubles) + i for i in range(len(singles)) if abs(grads[i]) > 1.0e-5
        ]
        gates_select = triples_select + doubles_select + singles_select
        params = pnp.zeros(len(gates_select), requires_grad=True)
        if self.opt_asd_params is not None:
            for idx, exc_itx in enumerate(doubles_indice + singles_indice):
                params[idx] = self.opt_asd_params[exc_itx]

        opt = qml.AdamOptimizer(stepsize=0.01)
        params, prev_energy = opt.step_and_cost(cost_fn_1, params, excitations=gates_select)
        for _ in (pbar := tqdm(range(1000), position=0, leave=True)):
            params, energy = opt.step_and_cost(cost_fn_1, params, excitations=gates_select)
            chem_acc = pnp.round(pnp.abs(classical_energy - energy), 5)
            pbar.set_description(f"AdaptGiv: Chem. Acc.  = {chem_acc} Hartree\t")
            if pnp.abs(chem_acc) <= 1e-3 or pnp.abs(energy - prev_energy) <= 1e-6:
                break

            prev_energy = energy

        self.opt_ag_params = params.copy()

        all_params = pnp.zeros(len(doubles + singles), requires_grad=True)
        if self.opt_asd_params is not None:
            all_params = self.opt_asd_params.copy()

        for idx, exc_itx in enumerate(triples_indice + doubles_indice + singles_indice):
            all_params[exc_itx] = params[idx]
        self.opt_asd_params = all_params.copy()

        gates_set = []
        for i, excitation in enumerate(gates_select):
            if len(excitation) == 4:
                gates_set.append(qml.DoubleExcitation(params[i], wires=excitation))
            else:
                gates_set.append(qml.SingleExcitation(params[i], wires=excitation))

        return params, gates_set, energy

    @staticmethod
    def convert_ham_obs(hamiltonian, wire_map):
        """Converts Hamiltonian object to dictionary object"""
        hamil_dict = {}
        coeffs, ops = hamiltonian.terms()
        for coeff, op in zip(coeffs, ops):
            hamil_dict.update({qml.grouping.pauli_word_to_string(op, wire_map): coeff})
        return hamil_dict

    @staticmethod
    def convert_ham_dict(hamil_dict, wire_map):
        """Converts dictionary object to Hamiltonian object"""
        coeffs, ops = [], []
        for key, val in hamil_dict.items():
            coeffs.append(val)
            ops.append(qml.grouping.string_to_pauli_word(key, wire_map))
        return qml.Hamiltonian(coeffs, ops)

    @staticmethod
    def get_qwc_groupings(hamiltonian):
        """Get qubit-wise commuting groupings"""
        obs, parts, coeffs = qml.grouping.optimize_measurements(hamiltonian.ops, hamiltonian.coeffs)
        idx = np.nonzero(["Identity" in qb.name for qb in parts[0]])[0]
        if len(idx):
            obs.insert(0, [])
            parts.insert(0, [qml.Identity(wires=[0])])
            coeffs.insert(0, coeffs[0][idx[0] : idx[0] + 1])
            parts[1].pop(idx[0])
            coeffs[1] = np.concatenate((coeffs[1][: idx[0]], coeffs[1][idx[0] + 1 :]))
        return (coeffs, parts, obs)

    @staticmethod
    def get_brt_groupings(molecule):
        """Get basis-rotation groupings"""
        _, one, two = qml.qchem.electron_integrals(molecule)()
        coeffs, ops, unitaries = qml.qchem.basis_rotation(one, two, tol_factor=1.0e-5)
        return (coeffs, ops, unitaries)

    @staticmethod
    def get_qwc_samples(groupings, vqe_gates, hf_state, hamiltonian, shots=10000, check=False):
        """Get samples for qubit-wise commuting groupings"""

        def qc_kit(obs):
            qml.BasisState(hf_state, wires=hamiltonian.wires)
            for op in vqe_gates:
                qml.apply(op)
            for ob in obs:
                qml.apply(ob)
            return qml.counts()

        coeffs, parts, obs = groupings
        dev = qml.device("lightning.qubit", wires=len(hamiltonian.wires), shots=shots)
        circuit = qml.QNode(qc_kit, dev)

        def calc_expec_qwc(coeffs, parts, obs):
            """Check expectation values for energy reconstruction and validation"""
            energy = 0.0
            for idx, _ in enumerate(obs):
                res_dict = circuit(obs[idx])
                exps = np.zeros(len(parts[idx]))
                for ind, (part, coeff) in enumerate(zip(parts[idx], coeffs[idx])):
                    meas_wires = set(part.wires) if not part.name == "Identity" else ()
                    nkey = list(
                        map(
                            "".join,
                            list(
                                it.product(
                                    "01",
                                    repeat=len(set(dev.wires).intersection(meas_wires)),
                                )
                            ),
                        )
                    )
                    if not meas_wires:
                        exp = dev.shots
                    else:
                        new_dict = {nk: 0 for nk in nkey}
                        for key, counts in res_dict.items():
                            ky = "".join(itemgetter(*set(part.wires))(key))
                            new_dict[ky] += res_dict[key]
                        exp = 0.0
                        for key, counts in new_dict.items():
                            exp += (-1) ** (sum(int(i) for i in key)) * counts
                    exps[ind] = exp * coeff / dev.shots

                energy += np.sum(exps)
            return energy

        if check:
            # TODO: Add check to test precision for groupings
            _ = calc_expec_qwc(coeffs, parts, obs)

        res_dicts = []
        for idx in tqdm(range(len(obs)), leave=False):
            res_dict = circuit(obs[idx])
            res_dicts.append(res_dict)

        return res_dicts

    # pylint: disable=import-outside-toplevel, dangerous-default-value
    @staticmethod
    def get_initial_states(
        symbols,
        geometry,
        charge,
        basis_name,
        init_state_tol=1e-4,
        state_qualities=(1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.01, 0.001),
    ):
        """
        Get initial states for quantum chemistry simulations.
        This function performs Hartree-Fock (HF) and Complete Active Space Configuration Interaction (CASCI) calculations
        to determine the initial states for quantum chemistry simulations. It iterates over different spin states to find
        the true ground state, and builds an initial state from that of varying quality, either by
        truncating the number of Slater determinants in the perfect ground state description, or
        by mixing in a relevant fraction of the first excited state, when truncation is not possible.

        Args:
            symbols (list of str): List of atomic symbols.
            geometry (list of list of float): List of atomic coordinates in Bohr.
            charge (int): The charge of the molecule.
            basis_name (str): The basis set name.
            init_state_tol (float, optional): Tolerance for which Slater determinants to keep in initial state preparation.
                All determinants with coefficients below this value are set to zero. Default is 1e-4.
            state_qualities (Sequence[float], optional): List of state qualities for truncation and re-allocation. 
                Default is ``(1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.01, 0.001)``.

        Returns:
            tuple: A tuple containing:
            - dets (np.ndarray): Array of determinants.
            - coeffs (np.ndarray): Array of coefficients representing the determinants.

        Notes:
            - The function uses PySCF for quantum chemistry calculations and PennyLane for quantum computing operations.
        """

        try:
            from overlapper.utils import get_mol_attrs
            from overlapper.state import do_hf, do_casci, casci_state
            from pennylane.qchem.convert import _sign_chem_to_phys
        except ImportError as exc:
            raise ImportError(
                "To use the get_initial_states function, please install the overlapper package via "
                "pip install git+https://github.com/XanaduAI/Overlapper.git"
            ) from exc

        def _casci_to_superposition(keys, norbs):
            new_dets = []
            for key in keys:
                new_det = []
                int_a, int_b = key
                bin_a = bin(int_a)[2:][::-1]
                bin_b = bin(int_b)[2:][::-1]
                bin_a += "0" * (norbs - len(bin_a))
                bin_b += "0" * (norbs - len(bin_b))
                for bi, ba in zip(bin_a, bin_b):
                    new_det.append(int(bi))
                    new_det.append(int(ba))
                new_dets.append(new_det)

            return np.array(new_dets)

        bohr_to_ang = 0.529177210903 # conversion factor from Bohr to Angstrom
        atom = [[name, np.array(geom) * bohr_to_ang] for (name, geom) in zip(symbols, geometry)]
        mol = gto.M(atom=atom, basis=basis_name, charge=charge)

        # loop over spins to find true ground state
        spins = list(range(9))
        hf_solver_array = []
        solver_array = []
        e_array = []
        ss_array = []
        sz_array = []
        for sz in spins:
            print(f"\n\tBegin calculation with Sz = {sz}.")
            # create molecule and set its spin
            try:
                assert (
                    sz % 2 == int(mol.nelectron) % 2
                ), f"Proposed spin {sz} impossible for given number of electrons {mol.nelectron}."
                mol_copy = gto.M(atom=atom, basis=basis_name, charge=charge, spin=sz)
                ncas, nelecas_a, nelecas_b = get_mol_attrs(mol_copy)
                assert (
                    nelecas_a <= ncas
                ), f"More electrons in a sector {nelecas_a} than orbitals {ncas}."
                assert (
                    nelecas_b <= ncas
                ), f"More electrons in b sector {nelecas_b} than orbitals {ncas}."
                # run an HF calculation
                if np.abs(sz) < 1e-2:
                    hftype = "rhf"
                else:
                    hftype = "rohf"
                hf, hf_e, hf_ss, hf_sz = do_hf(mol_copy, hftype=hftype)
                print(f"\nRHF energy: {hf_e[0]:.3f}")
                print(f"\nRHF S^2 and Sz: {hf_ss[0]:.3f}, {hf_sz[0]:.3f}")
                # run a full CI calculation
                casci, casci_e, casci_ss, casci_sz = do_casci(
                    hf, ncas, (nelecas_a, nelecas_b), maxiter=1000, mem=8500, nroots=2, verbose=0
                )
                print(f"\nCASCI energy: {casci_e}")
                print(f"\nCASCI spins: {casci_ss}, {casci_sz}")

            except AssertionError:
                print(
                    f"\nCalculation on molecule with Sz = {sz} could not be done."
                    f" Check your electron counts.\n"
                )
                continue

            hf_solver_array.append(hf)
            solver_array.append(casci)
            e_array.append(casci_e)
            ss_array.append(casci_ss)
            sz_array.append(casci_sz)

        assert (
            len(solver_array) > 0
        ), "None of the spin calculations you supplied worked. Pick different spins."

        # select true ground state
        minvals = [elem[0] for elem in e_array]
        minidx = np.where(minvals == min(minvals))[0][0]
        hf = hf_solver_array[minidx]
        casci, casci_e, casci_ss, casci_sz = (
            solver_array[minidx],
            e_array[minidx],
            ss_array[minidx],
            sz_array[minidx],
        )
        casci_sz = np.array(np.array(casci_sz).round(decimals=0), dtype=int)
        print(f"All energies {e_array}")
        print(f"Lowest energy sector index {minidx}")
        print(
            f"Sector picked: energy of solver {casci.e_tot}, energy array "
            f"{casci_e}, spins {casci_ss}, spin projections {casci_sz}"
        )

        # grab the coefficients and Slaters
        wf_0 = casci_state(casci, state=0, tol=init_state_tol)
        wf_0 = _sign_chem_to_phys(wf_0, mol.nao)
        wf_1 = casci_state(casci, state=1, tol=init_state_tol)
        wf_1 = _sign_chem_to_phys(wf_1, mol.nao)
        dets_0, coeffs_0 = _casci_to_superposition(wf_0.keys(), mol.nao), np.array(
            list(wf_0.values())
        )
        dets_1, coeffs_1 = _casci_to_superposition(wf_1.keys(), mol.nao), np.array(
            list(wf_1.values())
        )

        # loop over state quality -- truncate and re-allocate
        ovlps = np.cumsum(np.abs(coeffs_0) ** 2)
        for state_quality in state_qualities:

            idx = np.where((state_quality > ovlps) & (state_quality < ovlps))

            dets = []
            coeffs = []

            if idx[0].size == 0:

                for jj, elem in enumerate(dets_0):
                    dets += [elem]
                    jj_idx = np.where(np.all(np.array(dets_1) == np.array(elem), axis=1))[0]
                    if len(jj_idx) > 0:
                        coeffs += [
                            float(
                                np.sqrt(state_quality) * coeffs_0[jj]
                                + np.sqrt(1 - state_quality) * coeffs_1[jj_idx][0]
                            )
                        ]
                    else:
                        coeffs += [float(np.sqrt(state_quality) * coeffs_0[jj])]

                for jj, elem in enumerate(dets_1):
                    if elem not in dets_0:
                        dets += [elem]
                        coeffs += [np.sqrt(1 - state_quality) * coeffs_1[jj]]

                dets = np.array(dets)
                coeffs = np.array(coeffs)

            else:
                dets = np.array(dets_0)[:idx]  # [D x ncas], list of idx lists, each of length ncas
                coeffs = np.array(coeffs_0)[:idx]  # [D]

        return dets, coeffs

    @staticmethod
    def get_brt_samples(groupings, vqe_gates, hf_state, hamiltonian, shots=10000, check=False):
        """Get samples for basis-rotation groupings"""

        def brt_kit(u):
            qml.BasisState(hf_state, wires=hamiltonian.wires)
            for op in vqe_gates:
                qml.apply(op)
            qml.BasisRotation(wires=range(len(u)), unitary_matrix=u)
            return qml.counts()

        coeffs, ops, unitaries = groupings
        dev = qml.device("lightning.qubit", wires=len(hamiltonian.wires), shots=shots)
        circuit = qml.QNode(brt_kit, dev)

        def calc_expec_brt(coeffs, ops, unitaries):
            """Check expectation values for energy reconstruction and validation"""
            energy = 0.0
            for idx, _ in enumerate(unitaries):
                res_dict = circuit(unitaries[idx])
                exps = np.zeros(len(ops[idx]))
                for ind, (part, coeff) in enumerate(zip(ops[idx], coeffs[idx])):
                    meas_wires = set(part.wires) if not part.name == "Identity" else ()
                    nkey = list(
                        map(
                            "".join,
                            list(
                                it.product(
                                    "01",
                                    repeat=len(set(dev.wires).intersection(meas_wires)),
                                )
                            ),
                        )
                    )
                    if not meas_wires:
                        exp = dev.shots
                    else:
                        new_dict = {nk: 0 for nk in nkey}
                        for key, counts in res_dict.items():
                            ky = "".join(itemgetter(*set(part.wires))(key))
                            new_dict[ky] += res_dict[key]
                        exp = 0.0
                        for key, counts in new_dict.items():
                            exp += (-1) ** (sum(int(i) for i in key)) * counts
                    exps[ind] = exp * coeff / dev.shots
                energy += np.sum(exps)
            return energy

        if check:
            # TODO: Add check to test precision for groupings
            _ = calc_expec_brt(coeffs, ops, unitaries)

        res_dicts = []
        for idx in tqdm(range(len(unitaries)), leave=False):
            res_dict = circuit(unitaries[idx])
            res_dicts.append(res_dict)
        return res_dicts

    def cached_sparse_matrix(self, hamiltonian):
        """Caches sparse matrix generation for Hamiltonians. Useful when you have sufficient memory bandwidth."""

        wires = hamiltonian.wires
        n = len(wires)
        matrix = sp.sparse.csr_matrix((2**n, 2**n), dtype="complex128")

        coeffs = qml.math.toarray(hamiltonian.data)
        if self.red_mats is None:
            red_mats_flag = True
            self.red_mats = []
        else:
            red_mats_flag = False

        temp_mats = []
        for ind, (coeff, op) in enumerate(zip(coeffs, hamiltonian.ops)):
            if red_mats_flag:
                obs = []
                for o in qml.operation.Tensor(op).obs:
                    if len(o.wires) > 1:
                        # todo: deal with operations created from multi-qubit operations such as Hermitian
                        raise ValueError(
                            f"Can only sparsify Hamiltonians whose constituent observables consist of "
                            f"(tensor products of) single-qubit operators; got {op}."
                        )
                    obs.append(o.matrix())

                # Array to store the single-wire observables which will be Kronecker producted together
                mat = []
                # i_count tracks the number of consecutive single-wire identity matrices encountered
                # in order to avoid unnecessary Kronecker products, since I_n x I_m = I_{n+m}
                i_count = 0
                for wire_lab in wires:
                    if wire_lab in op.wires:
                        if i_count > 0:
                            mat.append(sp.sparse.eye(2**i_count, format="coo"))
                        i_count = 0
                        idx = op.wires.index(wire_lab)
                        # obs is an array storing the single-wire observables which
                        # make up the full Hamiltonian term
                        sp_obs = sp.sparse.coo_matrix(obs[idx])
                        mat.append(sp_obs)
                    else:
                        i_count += 1

                if i_count > 0:
                    mat.append(sp.sparse.eye(2**i_count, format="coo"))

                self.red_mats.append(
                    ft.reduce(lambda i, j: sp.sparse.kron(i, j, format="coo"), mat)
                )

            red_mat = self.red_mats[ind] * coeff

            temp_mats.append(red_mat.tocsr())
            # Value of 100 arrived at empirically to balance time savings vs memory use. At this point
            # the `temp_mats` are summed into the final result and the temporary storage array is
            # cleared.
            if (len(temp_mats) % 150) == 0:
                matrix += sum(temp_mats)
                temp_mats = []

        matrix += sum(temp_mats)
        return matrix

    # pylint: disable=too-many-branches, too-many-statements
    # pylint: disable=dangerous-default-value, too-many-positional-arguments
    def pipeline(
        self,
        molname,
        symbols,
        geometry,
        charge,
        basis_name,
        descriptor,
        update_keys=[],
        skip_keys=[],
        filename="",
        prog_bar=None,
    ):
        """Implements the data generation pipeline"""

        if prog_bar is None:
            raise ValueError("Please initialize progress bar for verbose output")

        prog_bar.set_description("Molecule Generation")
        mol = qml.qchem.Molecule(symbols, geometry, charge=charge, basis_name=basis_name)

        path = f"datasets/qchem/{molname}/{mol.basis_name.upper()}/{descriptor}/"
        Path(path).mkdir(parents=True, exist_ok=True)
        if not filename:
            filename = f"{path}{molname}_{mol.basis_name.upper()}_{descriptor}_full.dat"

        f = {}  # self.append_data(filename)
        present_keys = list(f.keys())
        f["molecule"] = mol
        wire_map = None

        if "hamiltonian" not in skip_keys and (
            "hamiltonian" not in present_keys or "hamiltonian" in update_keys
        ):
            if mol.n_orbitals <= 40:
                prog_bar.set_description("Hamiltonian Generation")
                hamiltonian, qubits = qml.qchem.molecular_hamiltonian(
                    symbols,
                    geometry,
                    charge=charge,
                    basis=basis_name,
                    method="pyscf",
                )
                active_electrons = mol.n_electrons
                active_orbitals = mol.n_orbitals
            else:
                # TODO: Add support for active orbitals in the Molecule class
                core_orbs = core_orbitals(mol)
                active_electrons = mol.n_electrons - 2 * len(core_orbs)
                active_orbitals = mol.n_orbitals - len(core_orbs)
                hamiltonian, qubits = qml.qchem.molecular_hamiltonian(
                    symbols,
                    geometry,
                    charge=charge,
                    method="pyscf",
                    active_electrons=active_electrons,
                    active_orbitals=active_orbitals,
                )

            prog_bar.set_description("SparseHam Generation")
            sparse_ham = hamiltonian.sparse_matrix()
            # use self.cached_sparse_matrix(hamiltonian) if huge memory access and
            # data generation is being done for multiple geometries.
            sparse_hamiltonian = qml.SparseHamiltonian(sparse_ham, hamiltonian.wires)

            wire_map = dict(enumerate(hamiltonian.wires))
            f["hamiltonian"] = {
                "terms": self.convert_ham_obs(hamiltonian, wire_map),
                "wire_map": wire_map,
            }
            f["sparse_hamiltonian"] = sparse_ham  # sparse hamiltonian

            prog_bar.set_description("Groupings Generation")
            f["qwc_groupings"] = self.get_qwc_groupings(hamiltonian)
            f["basis_rot_groupings"] = self.get_brt_groupings(mol)

            self.write_data(f, filename)
        else:
            wire_map = f["wire_map"] if f["wire_map"] is not None else None
            hamiltonian = (
                self.convert_ham_dict(f["hamiltonian"], wire_map)
                if f["hamiltonian"] is not None
                else None
            )
            sparse_ham = f["sparse_hamiltonian"] if f["sparse_hamiltonian"] is not None else None
            sparse_hamiltonian = (
                qml.SparseHamiltonian(sparse_ham, hamiltonian.wires)
                if sparse_ham is not None
                else None
            )

        if "symmetries" not in skip_keys and (
            "symmetries" not in present_keys or "symmetries" in update_keys
        ):
            prog_bar.set_description("Symmetry Generation")
            generators = qml.symmetry_generators(hamiltonian)
            paulixops = qml.paulix_ops(generators, len(hamiltonian.wires))
            paulix_sector = qml.qchem.optimal_sector(hamiltonian, generators, mol.n_electrons)

            f["symmetries"] = generators
            f["paulix_ops"] = paulixops
            f["optimal_sector"] = paulix_sector
            self.write_data(f, filename)
        else:
            generators = f["symmetries"]
            paulixops = f["paulix_ops"]
            paulix_sector = f["optimal_sector"]

        if "hamiltonian" not in skip_keys and (
            "hamiltonian" not in present_keys or "hamiltonian" in update_keys
        ):
            prog_bar.set_description("AuxillaryObs Generation")

            if self.aux_ops is None:
                self.aux_ops = [
                    qml.qchem.particle_number(len(hamiltonian.wires)),
                    qml.qchem.spin2(mol.n_electrons, 2 * mol.n_orbitals),
                    qml.qchem.spinz(len(hamiltonian.wires)),
                ]

            observables = [
                hamiltonian,
                *qml.qchem.dipole_moment(mol)(),
            ] + self.aux_ops

            hf_state = pnp.where(pnp.arange(len(hamiltonian.wires)) < mol.n_electrons, 1, 0)
            f["dipole_op"] = observables[1:4]
            f["num_op"] = observables[4]
            f["spin2_op"] = observables[5]
            f["spinz_op"] = observables[6]
            f["hf_state"] = hf_state

            prog_bar.set_description("FCI Energy Generation")
            if charge:
                eigvals, eigvecs = sp.sparse.linalg.eigsh(
                    sparse_ham, which="SA", k=20, return_eigenvectors=True
                )
                num_op_mat = qml.matrix(f["num_op"])
                num_op_mat = qml.matrix(f["num_op"])
                num_op_vec = np.diag(num_op_mat).real
                energies = []
                for index in tqdm(range(len(eigvals)), leave=False):
                    eigvec = eigvecs[:, index]
                    num_part = np.dot(num_op_vec, np.abs(eigvec) ** 2)
                    spn_part = eigvec.conjugate() @ f["spin2_op"] @ eigvec
                    print(index, spn_part)
                    if int(np.round(num_part.real)) == mol.n_electrons:
                        energies.append(eigvals[index])
                energies = np.array(energies)
                classical_energy = energies[0]
            else:
                energies = qml.eigvals(sparse_hamiltonian, k=2 * qubits - 1)
                classical_energy = qml.eigvals(sparse_hamiltonian, k=2 * qubits - 1)[
                    0
                ]  # energies[0]

            # TODO: Use sparse matrices when number of orbitals are large
            f["fci_spectrum"] = np.array([classical_energy])
            f["fci_energy"] = classical_energy
            self.write_data(f, filename)
        else:
            observables = [
                hamiltonian,
                f["dipole_op"],
                f["num_op"],
                f["spin2_op"],
                f["spinz_op"],
            ]
            hf_state = f["hf_state"]
            classical_energy = f["fci_energy"]
            energies = f["fci_spectrum"]

        if "symmetries" not in skip_keys and (
            "symmetries" not in present_keys or "symmetries" in update_keys
        ):
            prog_bar.set_description("TaperingHF Generation")
            if self.tapered_hf_state is None:
                self.tapered_hf_state = qml.qchem.taper_hf(
                    generators,
                    paulixops,
                    paulix_sector,
                    mol.n_electrons,
                    len(hamiltonian.wires),
                )

            def taper_obs_with_bar(idx, observable):
                prog_bar.set_description(f"TaperingObs {idx} Generation")
                return qml.taper(observable, generators, paulixops, paulix_sector)

            if self.tapered_aux_ops is None:
                self.tapered_aux_ops = [
                    taper_obs_with_bar(idx, observable)
                    if len(observable.ops)
                    else qml.Hamiltonian([], [])
                    for idx, observable in enumerate(self.aux_ops)
                ]

            tapered_obs = [
                taper_obs_with_bar(idx, observable)
                if len(observable.ops)
                else qml.Hamiltonian([], [])
                for idx, observable in enumerate(observables[:4])
            ] + self.tapered_aux_ops

            coeffs, ops = tapered_obs[0].terms()
            wire_map = {wire: itx for itx, wire in enumerate(tapered_obs[0].wires.tolist())}
            ops = [
                string_to_pauli_word(pauli_word_to_string(op, wire_map=wire_map), wire_map=wire_map)
                for op in ops
            ]
            tapered_obs[0] = qml.Hamiltonian(coeffs=coeffs, observables=ops)

            f["tapered_hamiltonian"] = {
                "terms": self.convert_ham_obs(tapered_obs[0], wire_map),
                "wire_map": wire_map,
            }
            f["tapered_dipole_op"] = tapered_obs[1:4]
            f["tapered_num_op"] = tapered_obs[4]
            f["tapered_spin2_op"] = tapered_obs[5]
            f["tapered_spinz_op"] = tapered_obs[6]
            f["tapered_hf_state"] = self.tapered_hf_state
            self.write_data(f, filename)

            # TODO: Use the tapered Sparse Hamiltonian when number of orbitals are large
            # prog_bar.set_description(f"FCI Energy Generation 2")
            # sparse_taper_ham = qml.SparseHamiltonian(tapered_obs[0].sparse_matrix(), tapered_obs[0].wires)
            # f["fci_energy"] = qml.eigvals(sparse_taper_ham)[0] #energies[0]

        else:
            tapered_obs = [
                f["tapered_hamiltonian"],
                f["tapered_dipole_op"],
                f["tapered_num_op"],
                f["tapered_spin2_op"],
                f["tapered_spinz_op"],
            ]
            self.tapered_hf_state = f["tapered_hf_state"]
            self.tapered_wire_map = f["tapered_wire_map"]

        prog_bar.set_description("VQE Execution")
        if "vqe_params" not in skip_keys and (
            "vqe_params" not in present_keys or "vqe_params" in update_keys
        ):
            params, circuit, energy = self.run_adaptive_vqe(
                mol, hf_state, sparse_hamiltonian, classical_energy
            )
            f["vqe_energy"] = energy
            f["vqe_params"] = params
            f["vqe_gates"] = circuit
        else:
            f["vqe_energy"] = None
            f["vqe_params"] = None
            f["vqe_gates"] = None

        prog_bar.set_description("Samples Generation")
        if "samples" not in skip_keys and (
            "samples" not in present_keys or "samples" in update_keys
        ):
            f["qwc_samples"] = self.get_qwc_samples(
                f["qwc_groupings"],
                f["vqe_gates"],
                f["hf_state"],
                hamiltonian,
                shots=10000,
            )
            f["basis_rot_samples"] = self.get_brt_samples(
                f["basis_rot_groupings"],
                f["vqe_gates"],
                f["hf_state"],
                hamiltonian,
                shots=10000,
            )

        prog_bar.set_description("Initial State Preparation")
        if "initial_states" not in skip_keys and (
            "initial_states" not in present_keys or "initial_states" in update_keys
        ):
            f["initial_state_dets"], f["initial_state_coeffs"] = self.get_initial_states(
                symbols, geometry, charge, basis_name, init_state_tol=1e-6
            )

        prog_bar.set_description("We did it again!")
        self.write_data(f, filename)
        self.write_data_seperated(f, f"{path}{molname}_{mol.basis_name.upper()}_{descriptor}")
