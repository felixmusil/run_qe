import numpy as np
from utils import rotation_matrix,isCellSkewed,unskewCell,get_symprec,get_relative_angle
import ase
import spglib as spg



def frame2qe_format(frame,sg):
    from qe_input import NOPROBLEM, SG2ibrav
    if sg in NOPROBLEM:
        return frame.copy()
    else:
        ibrav = SG2ibrav(sg)

        change_func = ibrav2func[ibrav]
        custom_frame, _, _ = change_func(frame,sg)
        return custom_frame

def get_ibrav0_frame(frame, sg):
    symprec = get_symprec(frame, sg)
    if symprec is None:
        print 'Not possible'
        return None
    (lattice, positions, numbers) = spg.standardize_cell(
                            frame, to_primitive=True,no_idealize=False,
                            symprec=symprec, angle_tolerance=-1.0)
    primitive_atoms = ase.Atoms(cell=lattice, scaled_positions=positions, numbers=numbers)

    cell = primitive_atoms.get_cell()
    pos = primitive_atoms.get_positions()

    inequivalent_pos = pos[0]

    return primitive_atoms, cell, inequivalent_pos

def get_ibrav2_frame(frame, sg):
    symprec = get_symprec(frame, sg)
    if symprec is None:
        print 'Not possible'
        return None
    (lattice, positions, numbers) = spg.standardize_cell(
                            frame, to_primitive=True,no_idealize=False,
                            symprec=symprec, angle_tolerance=-1.0)
    primitive_atoms = ase.Atoms(cell=lattice, scaled_positions=positions, numbers=numbers)

    cell = primitive_atoms.get_cell()
    pos = primitive_atoms.get_positions()

    axis = [0, 0, 1]
    theta = np.pi / 2.
    rot = rotation_matrix(axis, theta)

    cellp = np.dot(cell, rot.T)
    cellp[np.abs(cellp) < 1e-10] = 0.
    posp = np.dot(pos, rot.T)
    inequivalent_pos = posp[0]

    primitive_atoms.set_cell(cellp)
    primitive_atoms.set_positions(posp)

    a = primitive_atoms.get_cell_lengths_and_angles()[0] * 2
    # only a is used by QE
    b, c, alpha, beta, gamma = [0.] * 5
    cell_par = [a, b, c, alpha, beta, gamma]
    return primitive_atoms, cell_par, inequivalent_pos

def get_ibrav5_frame(frame, sg):
    symprec = get_symprec(frame, sg)
    if symprec is None:
        print 'Not possible'
        return None
    (lattice, positions, numbers) = spg.standardize_cell(
                            frame, to_primitive=True,no_idealize=False,
                            symprec=symprec, angle_tolerance=-1.0)
    primitive_atoms = ase.Atoms(cell=lattice, scaled_positions=positions, numbers=numbers)

    cell = primitive_atoms.get_cell()
    pos = primitive_atoms.get_positions()

    axis = [0, 0, 1]
    theta = - 60. * np.pi / 180.
    rot = rotation_matrix(axis, theta)

    cellp = np.dot(cell, rot.T)
    cellp[np.abs(cellp) < 1e-10] = 0.
    posp = np.dot(pos, rot.T)
    inequivalent_pos = posp[0]

    primitive_atoms.set_cell(cellp)
    primitive_atoms.set_positions(posp)

    a, _, _, alpha, _, _ = primitive_atoms.get_cell_lengths_and_angles()
    # only a is used by QE
    b, c, beta, gamma = [0.] * 4
    cell_par = [a, b, c, alpha, beta, gamma]
    return primitive_atoms, cell_par, inequivalent_pos

def get_ibrav9_frame(frame, sg):
    symprec = get_symprec(frame, sg)
    if symprec is None:
        print 'Not possible'
        return None
    (lattice, positions, numbers) = spg.standardize_cell(
                            frame, to_primitive=True,no_idealize=False,
                            symprec=symprec, angle_tolerance=-1.0)
    primitive_atoms = ase.Atoms(cell=lattice, scaled_positions=positions, numbers=numbers)

    pos = primitive_atoms.get_positions()

    inequivalent_pos = pos[0]

    a, b, c, _, _, _ = primitive_atoms.get_cell_lengths_and_angles()
    # only a is used by QE
    alpha, beta, gamma = [0.] * 3
    cell_par = [a, b, c, alpha, beta, gamma]
    return primitive_atoms, cell_par, inequivalent_pos

def get_ibrav12_frame(frame, sg):
    symprec = get_symprec(frame, sg)
    if symprec is None:
        print 'Not possible'
        return None
    (lattice, positions, numbers) = spg.standardize_cell(
                            frame, to_primitive=False,no_idealize=False,
                            symprec=symprec, angle_tolerance=-1.0)
    primitive_atoms = ase.Atoms(cell=lattice, scaled_positions=positions, numbers=numbers)


    pos = primitive_atoms.get_positions()

    inequivalent_pos = pos[0]

    cell_par = primitive_atoms.get_cell_lengths_and_angles()
    return primitive_atoms, cell_par, inequivalent_pos

ibrav2func = {3: get_ibrav0_frame,
                7:get_ibrav0_frame,
                10:get_ibrav0_frame,
                11:get_ibrav0_frame,
                13:get_ibrav0_frame,
                14:get_ibrav0_frame,
                2:get_ibrav2_frame,
                5:get_ibrav5_frame,
                -9:get_ibrav9_frame,
                -12:get_ibrav12_frame,
}