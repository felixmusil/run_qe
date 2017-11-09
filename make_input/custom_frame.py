import numpy as np
from utils import rotation_matrix,isCellSkewed,unskewCell,get_symprec,get_relative_angle
import ase
import spglib as spg
from qe_input import SG2ibrav, ibrav0,NOPROBLEM
from raw_info import sgwyck2qewyck,z2symb

def frame2qe_format_new(crystal,sites_z,sym_data):

    sg = sym_data['number']

    if sg in ibrav0:
        custom_frame, cell_par, positions_data = get_ibrav0_frame_new(crystal)
    elif sg in NOPROBLEM:
        custom_frame, cell_par, positions_data = get_frame_no_mod_new(crystal, sites_z, sym_data)

    else:
        ibrav = SG2ibrav(sg)

        change_func = ibrav2func_new[ibrav]
        custom_frame,  cell_par, positions_data = change_func(crystal, sites_z, sym_data)

    return custom_frame ,  cell_par, positions_data

def get_frame_no_mod_new(crystal, sites_z, sym_data):

    spaceGroupIdx = sym_data['number']
    positions_data = {'wyckoffs': [],'species': [],'positions': []}

    for it, equi in enumerate(sym_data['equivalent_atoms']):
        if equi > len(sites_z):
            raise NotImplementedError('There are too many inequivalent positions {} '
                                      'compared to the sites {}'.format(
                np.unique(sym_data['equivalent_atoms']), sites_z))


        if equi not in positions_data.keys():
            positions_data['wyckoffs'].append(sgwyck2qewyck[(spaceGroupIdx,sym_data['wyckoffs'][it])])
            positions_data['species'].append(z2symb[sym_data['std_types'][it]])
            positions_data['positions'].append(list(crystal.get_scaled_positions()[it].reshape((1, -1))))


    # QE takes the a from the standard cell and not the primitive one
    cell_par = crystal.get_cell_lengths_and_angles()

    return crystal.copy(), cell_par, positions_data

def get_ibrav5_frame_new(crystal, sites_z, sym_data):

    spaceGroupIdx = sym_data['number']
    positions_data = {'wyckoffs': [], 'species': [], 'positions': []}

    for it, equi in enumerate(sym_data['equivalent_atoms']):
        if equi > len(sites_z):
            raise NotImplementedError('There are too many inequivalent positions {} '
                                      'compared to the sites {}'.format(
                np.unique(sym_data['equivalent_atoms']), sites_z))

        if equi not in positions_data.keys():
            positions_data['wyckoffs'].append(sgwyck2qewyck[(spaceGroupIdx, sym_data['wyckoffs'][it])])
            positions_data['species'].append(z2symb[sym_data['std_types'][it]])
            positions_data['positions'].append(list(crystal.get_scaled_positions()[it].reshape((1, -1))))

    (lattice, positions, numbers) = spg.standardize_cell(
        crystal, to_primitive=True, no_idealize=False,
        symprec=1e-5, angle_tolerance=-1.0)

    primitive_atoms = ase.Atoms(cell=lattice, scaled_positions=positions, numbers=numbers)
    cell_par = primitive_atoms.get_cell_lengths_and_angles()

    return primitive_atoms, cell_par, positions_data

def get_ibrav0_frame_new(crystal):

    (lattice, positions, numbers) = spg.standardize_cell(
        crystal, to_primitive=True,no_idealize=False,
                            symprec=1e-5, angle_tolerance=-1.0)
    primitive_atoms = ase.Atoms(cell=lattice, scaled_positions=positions, numbers=numbers)
    # primitive_atoms = frame
    cell = primitive_atoms.get_cell()
    pos = primitive_atoms.get_positions()
    species = primitive_atoms.get_chemical_symbols()
    positions_data = { 'species': species, 'positions': pos}
    return primitive_atoms, cell, positions_data

ibrav2func_new = {
    5:get_ibrav5_frame_new,
}


#############################################################################################

def frame2qe_format(frame,sg):
    '''
    cell_par, inequivalent_pos are useful here because QE sometimes needs mixed input that are not found in the
    object that we need to look at ultimatly.
    :param frame:
    :param sg:
    :return:
    '''
    from qe_input import NOPROBLEM, SG2ibrav,ibrav0


    if sg in ibrav0:
        custom_frame, cell_par, inequivalent_pos = get_ibrav0_frame(frame, sg)
    elif sg in NOPROBLEM:
        custom_frame, cell_par, inequivalent_pos = get_frame_no_mod(frame, sg)

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


    inequivalent_pos = frame.get_scaled_positions()[0].reshape((1, -1))

    cell_par = primitive_atoms.get_cell_lengths_and_angles()

    return primitive_atoms, cell_par, inequivalent_pos


def get_ibrav13_frame(frame, sg):

    if isCellSkewed(frame):
        primitive_atoms = unskewCell(frame)
    else:
        primitive_atoms = frame

    inequivalent_pos = primitive_atoms.get_scaled_positions()[0].reshape((1, -1))

    cell_par = primitive_atoms.get_cell_lengths_and_angles()
    #cell_par = cell_par[[1,0,2]]
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

def get_frame_no_mod(frame, sg):
    '''
    hard to reproduce exactly QE primitive cell but it is similar.

    :param frame: 
    :param sg: 
    :return: 
    '''

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

ibrav2func = {3: get_frame_no_mod,
                7:get_frame_no_mod,
                10:get_frame_no_mod,
                11:get_frame_no_mod,
                -13:get_frame_no_mod,
                14:get_frame_no_mod,
                2:get_frame_no_mod,
                5:get_ibrav5_frame,
                8:get_frame_no_mod,
                9:get_frame_no_mod,
                91:get_frame_no_mod,
                -12:get_frame_no_mod,
}