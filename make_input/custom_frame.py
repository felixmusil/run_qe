import numpy as np
from utils import rotation_matrix,isCellSkewed,unskewCell,get_symprec,get_relative_angle
import ase
import spglib as spg



def frame2qe_format(frame,sg):
    '''
    cell_par, inequivalent_pos are useful here because QE sometimes needs mixed input that are not found in the
    object that we need to look at ultimatly.
    :param frame:
    :param sg:
    :return:
    '''
    from qe_input import NOPROBLEM, SG2ibrav,tricky_sg

    if sg in NOPROBLEM:
        cell_par = frame.get_cell_lengths_and_angles()
        inequivalent_pos = frame.get_scaled_positions()[0].reshape((1,-1))
        custom_frame = frame.copy()

    elif sg in tricky_sg:
        ibrav = 0

        change_func = ibrav2func[ibrav]
        custom_frame,  cell_par, inequivalent_pos  = change_func(frame, sg)

    else:
        ibrav = SG2ibrav(sg)

        change_func = ibrav2func[ibrav]
        custom_frame,  cell_par, inequivalent_pos = change_func(frame,sg)

    return custom_frame ,  cell_par, inequivalent_pos

def get_ibrav0_frame(frame, sg):
    symprec = get_symprec(frame, sg)
    if symprec is None:
        print 'Not possible'
        return None
    (lattice, positions, numbers) = spg.standardize_cell(
                            frame, to_primitive=True,no_idealize=False,
                            symprec=symprec, angle_tolerance=-1.0)
    primitive_atoms = ase.Atoms(cell=lattice, scaled_positions=positions, numbers=numbers)
    # primitive_atoms = frame
    cell = primitive_atoms.get_cell()
    pos = primitive_atoms.get_positions()

    return primitive_atoms, cell, pos

def get_ibrav3_frame(frame, sg):
    '''
    hard to reproduce exactly QE primitive cell but it is similar.
    
    :param frame: 
    :param sg: 
    :return: 
    '''
    symprec = get_symprec(frame, sg)
    if symprec is None:
        print 'Not possible'
        return None
    (lattice, positions, numbers) = spg.standardize_cell(
                            frame, to_primitive=True,no_idealize=False,
                            symprec=symprec, angle_tolerance=-1.0)
    primitive_atoms = ase.Atoms(cell=lattice, scaled_positions=positions, numbers=numbers)


    # QE ibrav=2 and space group mix primitive cell and position in the standard cell
    inequivalent_pos = frame.get_scaled_positions()[0].reshape((1,-1))


    # QE takes the a from the standard cell and not the primitive one
    cell_par = frame.get_cell_lengths_and_angles()

    return primitive_atoms, cell_par, inequivalent_pos

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
    cellp[np.abs(cellp) < 1e-7] = 0.
    posp = np.dot(pos, rot.T)

    # QE ibrav=2 and space group mix primitive cell and position in the standard cell
    inequivalent_pos = frame.get_scaled_positions()[0].reshape((1,-1))

    # primitive atom here match QE cell definition but the standard cell information is used in the input
    primitive_atoms.set_cell(cellp)
    primitive_atoms.set_positions(posp)

    # QE takes the a from the standard cell and not the primitive one
    cell_par = frame.get_cell_lengths_and_angles()

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

    primitive_atoms.set_cell(cellp)
    primitive_atoms.set_positions(posp)

    inequivalent_pos = primitive_atoms.get_scaled_positions()[0].reshape((1, -1))

    cell_par = primitive_atoms.get_cell_lengths_and_angles()

    return primitive_atoms, cell_par, inequivalent_pos

def get_ibrav8_frame(frame, sg):
    '''
    
    :param frame: 
    :param sg: 
    :return: 
    '''
    symprec = get_symprec(frame, sg)
    if symprec is None:
        print 'Not possible'
        return None
    (lattice, positions, numbers) = spg.standardize_cell(
        frame, to_primitive=True, no_idealize=False,
        symprec=symprec, angle_tolerance=-1.0)
    primitive_atoms = ase.Atoms(cell=lattice, scaled_positions=positions, numbers=numbers)


    inequivalent_pos = primitive_atoms.get_scaled_positions()[0].reshape((1,-1))

    cell_par = primitive_atoms.get_cell_lengths_and_angles()

    return primitive_atoms, cell_par, inequivalent_pos

def get_ibrav9_frame(frame, sg):
    '''
    hard to reproduce exactly QE primitive cell but it is similar.

    :param frame: 
    :param sg: 
    :return: 
    '''
    symprec = get_symprec(frame, sg)
    if symprec is None:
        print 'Not possible'
        return None
    (lattice, positions, numbers) = spg.standardize_cell(
        frame, to_primitive=True, no_idealize=False,
        symprec=symprec, angle_tolerance=-1.0)
    primitive_atoms = ase.Atoms(cell=lattice, scaled_positions=positions, numbers=numbers)

    # QE ibrav=2 and space group mix primitive cell and position in the standard cell
    inequivalent_pos = frame.get_scaled_positions()[0].reshape((1, -1))

    # QE takes the a from the standard cell and not the primitive one
    cell_par = frame.get_cell_lengths_and_angles()

    return frame.copy(), cell_par, inequivalent_pos

def get_ibrav12_frame(frame, sg):
    symprec = get_symprec(frame, sg)
    if symprec is None:
        print 'Not possible'
        return None
    (lattice, positions, numbers) = spg.standardize_cell(
                            frame, to_primitive=False,no_idealize=False,
                            symprec=symprec, angle_tolerance=-1.0)
    primitive_atoms = ase.Atoms(cell=lattice, scaled_positions=positions, numbers=numbers)

    inequivalent_pos = primitive_atoms.get_scaled_positions()[0].reshape((1,-1))

    cell_par = primitive_atoms.get_cell_lengths_and_angles()

    return primitive_atoms, cell_par, inequivalent_pos

def get_standard_frame(frame,sg,primitive=True):
    symprec = get_symprec(frame, sg)
    if symprec is None:
        print 'Not possible'
        return None

    to_primitive = True if primitive else False

    (lattice, positions, numbers) = spg.standardize_cell(
                            frame, to_primitive=to_primitive,no_idealize=False,
                            symprec=symprec, angle_tolerance=-1.0)
    primitive_atoms = ase.Atoms(cell=lattice, scaled_positions=positions, numbers=numbers)
    return primitive_atoms

ibrav2func = {3: get_ibrav3_frame,
                7:get_ibrav0_frame,
                10:get_ibrav0_frame,
                11:get_ibrav0_frame,
                13:get_ibrav0_frame,
                14:get_ibrav0_frame,
                2:get_ibrav2_frame,
                5:get_ibrav5_frame,
                8:get_ibrav8_frame,
                9:get_ibrav9_frame,
                91:get_ibrav9_frame,
                -12:get_ibrav12_frame,
}