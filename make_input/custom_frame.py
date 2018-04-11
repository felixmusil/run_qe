import numpy as np
from utils import rotation_matrix,isCellSkewed,unskewCell,get_symprec,get_relative_angle
import ase
import spglib as spg
from collections import OrderedDict
from raw_info import sgwyck2qewyck,z2symb,sgwyck2qeibrav5,sgwyck2site_generator,sgwyck2qemask,sgwyck2qewyck

def frame2qe_format_new(crystal,sites_z,sym_data,force_ibrav0):
    from qe_input import SG2ibrav, ibrav0, NOPROBLEM
    sg = sym_data['number']

    if np.unique(sym_data['equivalent_atoms']) > len(sites_z):
        raise NotImplementedError('There are too many inequivalent positions {} '
                                  'compared to the sites {}'.format(
            np.unique(sym_data['equivalent_atoms']), sites_z))

    if sg in ibrav0 or force_ibrav0:
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
    aa = []
    # positions = crystal.get_scaled_positions()
    # species = crystal.get_atomic_numbers()
    # positions = np.mod(np.round(primitive_atoms.get_scaled_positions(),decimals=7),[1,1,1])

    positions = np.mod(np.round(sym_data['std_positions'],decimals=7),[1,1,1])
    species = sym_data['std_types']

    for it, equi in enumerate(sym_data['equivalent_atoms']):

        sgwyck = (spaceGroupIdx,sym_data['wyckoffs'][it])
        site_generator = sgwyck2site_generator[sgwyck]
        match_generator = is_matching_generator(positions[it], site_generator)

        # print '######'
        # print positions[it], site_generator
        # print match_generator
        if equi not in aa and match_generator is True:
            positions_data['wyckoffs'].append(sgwyck2qewyck[sgwyck])
            positions_data['species'].append(z2symb[species[it]])
            positions_data['positions'].append(positions[it])
            aa.append(equi)

        if len(positions_data['positions']) == len(sites_z):
            break
    if len(positions_data['positions']) == 0:
        print str(sgwyck) + ','

    # QE takes the a from the standard cell and not the primitive one
    # cell_par = crystal.get_cell_lengths_and_angles()
    cell_par = ase.geometry.cell_to_cellpar(sym_data['std_lattice'])
    return crystal.copy(), cell_par, positions_data

def get_ibrav5_frame_new(crystal, sites_z, sym_data):
    from raw_info import sgwyck2qeibrav5
    primitive_atoms = get_std_frame(crystal)
    sym_data = spg.get_symmetry_dataset(primitive_atoms)
    spaceGroupIdx = sym_data['number']
    positions_data = {'wyckoffs': [], 'species': [], 'positions': []}

    aa = []
    # positions = sym_data['std_positions'] np.mod(sym_data['std_positions'],[1,1,1])

    positions = np.mod(np.round(primitive_atoms.get_scaled_positions(),decimals=7),[1,1,1])
    species = primitive_atoms.get_atomic_numbers()

    for it, equi in enumerate(sym_data['equivalent_atoms']):

        sgwyck = (spaceGroupIdx, sym_data['wyckoffs'][it])
        site_generator = sgwyck2site_generator[sgwyck]
        match_generator = is_matching_generator(positions[it],site_generator)

        # print '######'
        # print positions[it], site_generator
        # print match_generator
        if equi not in aa and match_generator is True:
            positions_data['wyckoffs'].append(sgwyck2qeibrav5[sgwyck])
            positions_data['species'].append(z2symb[species[it]])
            positions_data['positions'].append(positions[it])
            aa.append(equi)
        if len(positions_data['positions']) == len(sites_z):
            break

    if len(positions_data['positions']) == 0:
        print str(sgwyck)+','

    cell_par = primitive_atoms.get_cell_lengths_and_angles()

    return primitive_atoms, cell_par, positions_data

def is_matching_generator(position,site_generator):
    match_generator = True
    # check if the selected postitions are matching properly the site generator
    # are the non free parameters in the proper place in positions[it]
    is_str, is_float = OrderedDict(), OrderedDict()
    for jj, (pos, s_gen) in enumerate(zip(position, site_generator)):
        if isinstance(s_gen, float):
            if np.abs(s_gen - pos) > 1e-5:
                match_generator = False
            is_float[jj] = s_gen
        elif isinstance(s_gen, str):
            is_str[jj] = s_gen

    # are the free parameters in the proper order
    if len(is_str) > 1 and site_generator != ['x', 'y', 'z']:

        #x, y, z = position
        for ii,s_gen in is_str.iteritems():
            if 'x' == s_gen:
                x = position[ii]
            elif 'y' == s_gen:
                y = position[ii]
            elif 'z' == s_gen:
                z = position[ii]

            #print 'There is : {}'.format(site_generator)


        if len(is_str) == 3:
            # TODO should replace eval with something safer
            ppos = map(eval, is_str.values())
        elif len(is_str) == 2:
            ppos = np.zeros(3)
            ppos[is_str.keys()] = map(eval, is_str.values())
            ppos[is_float.keys()] = is_float.values()
        # print positions[it],pp
        ppos = np.mod(np.round(ppos, decimals=7), [1, 1, 1])
        # print '#########'
        # print x
        # print is_str
        # print position, ppos
        if not np.allclose(position, ppos, atol=1e-3):
            match_generator = False

    return match_generator



def get_ibrav0_frame_new(crystal):

    # (lattice, positions, numbers) = spg.standardize_cell(
    #     crystal, to_primitive=False,no_idealize=False,
    #                         symprec=1e-5, angle_tolerance=-1.0)
    # primitive_atoms = ase.Atoms(cell=lattice, scaled_positions=positions, numbers=numbers)
    primitive_atoms = crystal
    cell = primitive_atoms.get_cell()
    pos = [list(pp) for pp in primitive_atoms.get_positions()]
    species = primitive_atoms.get_chemical_symbols()
    positions_data = { 'species': species, 'positions': pos}
    return primitive_atoms, cell, positions_data

def get_std_frame(crystal,to_primitive=True,symprec=1e-5):
    (lattice, positions, numbers) = spg.standardize_cell(
        crystal, to_primitive=to_primitive, no_idealize=False,
        symprec=symprec, angle_tolerance=-1.0)
    std_atoms = ase.Atoms(cell=lattice, scaled_positions=positions, numbers=numbers)
    return std_atoms

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