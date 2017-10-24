import numpy as np
from raw_info import missClassificationCorrection
import spglib as spg
import shutil,os

def qp2ase(qpatoms):
    from ase import Atoms as aseAtoms
    positions = qpatoms.get_positions()
    cell = qpatoms.get_cell()
    numbers = qpatoms.get_atomic_numbers()
    pbc = qpatoms.get_pbc()
    atoms = aseAtoms(numbers=numbers, cell=cell, positions=positions, pbc=pbc)

    for key, item in qpatoms.arrays.iteritems():
        if key in ['positions', 'numbers', 'species', 'map_shift', 'n_neighb']:
            continue
        atoms.set_array(key, item)

    return atoms

def ase2qp(aseatoms):
    from quippy import Atoms as qpAtoms
    positions = aseatoms.get_positions()
    cell = aseatoms.get_cell()
    numbers = aseatoms.get_atomic_numbers()
    pbc = aseatoms.get_pbc()
    return qpAtoms(numbers=numbers,cell=cell,positions=positions,pbc=pbc)


def crystal2SpaceGroupIdx(crystal,symtol):
    '''Identify the space group index (from 1,230) of a given crystal (ase Atoms class).
        Return an int from 1 to 230.'''
    import spglib as spg
    try:
        
        return int(spg.get_spacegroup(crystal,symprec=symtol).split(' ')[1][1:-1])
    except:
        #print crystal
        #print symtol
        #print spg.get_spacegroup(crystal,symprec=symtol)
        return -1


def get_symprec(crystal,spaceGroupIdx):
    '''
    Find which symprec is required to detect properly the sg
    '''
    detectedSpaceGroup = []
    symprec = None
    tols = [1e-5,1e-4,1e-3,1e-2]
    for tol in tols:
        detectedSpaceGroup.append(crystal2SpaceGroupIdx(crystal,tol))
        if spaceGroupIdx == detectedSpaceGroup[-1]:
            symprec = tol
            return symprec
        elif spaceGroupIdx in missClassificationCorrection.keys():
            if missClassificationCorrection[spaceGroupIdx] == detectedSpaceGroup[-1]:
                symprec = tol
                return symprec
    return symprec


def isInSP(crystal, spaceGroupIdx, tols=[1e-5, 1e-4, 1e-3, 1e-2]):
    SPgood = False
    detectedSpaceGroup = []

    for tol in tols:
        detectedSpaceGroup.append(crystal2SpaceGroupIdx(crystal, tol))
    if spaceGroupIdx in detectedSpaceGroup:
        SPgood = True
    elif spaceGroupIdx in missClassificationCorrection.keys():
        if missClassificationCorrection[spaceGroupIdx] in detectedSpaceGroup:
            SPgood = True
    return SPgood

def crystal2SpaceGroupIdx(crystal,symtol):
    '''Identify the space group index (from 1,230) of a given crystal (ase Atoms class).
        Return an int from 1 to 230.'''
    try:
        return int(spg.get_spacegroup(crystal,symprec=symtol).split(' ')[1][1:-1])
    except:

        return -1


def isCellSkewed(frame):
    params = np.abs(np.cos(frame.get_cell_lengths_and_angles()[3:]*np.pi/180.))
    return np.any(params>=0.5)
def unskewCell(frame):
    if isCellSkewed(frame):
        dd = ase2qp(frame)
        dd.set_cutoff(20.)
        dd.unskew_cell()
        dd.wrap()
        ee = qp2ase(dd)
        return ee
    else:
        return frame.copy()


def get_kpts(frame, Nkpt=1000):
    ir_cell = frame.get_reciprocal_cell()
    ir_l = np.linalg.norm(ir_cell, axis=1)

    n_kpt = float(Nkpt) / frame.get_number_of_atoms()
    # makes sure there are at least 3 kpts per directions
    if n_kpt < 27:
        n_kpt = 27

    n_reg = np.power(n_kpt, 1. / 3.)
    #print n_reg
    # find the midle val to norm with it
    mid = list(set(range(3)).difference([ir_l.argmin(), ir_l.argmax()]))[0]
    ir_l = ir_l / ir_l[mid]
    # get number of k point per directions
    nb = np.array(np.round(n_reg * ir_l), dtype=np.int64)

    return nb
    
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    see: https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula
    """
    import math
    import numpy as np
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def get_relative_angle(v1,v2,radian=True):
    ''' 
    Compute angle between 2 vectors
    from https://math.stackexchange.com/questions/1143354/numerically-stable-method-for-angle-between-3d-vectors
    '''
    import numpy as np
    v = np.asarray(v1)
    u = np.asarray(v2)
    nv = np.linalg.norm(v)
    nu = np.linalg.norm(u)
    angle = 2*np.arctan2(np.linalg.norm( nv * u - nu * v ) , \
                         np.linalg.norm( nv * u + nu * v )  )
    if radian:
        return angle
    else:
        return angle*180 /np.pi

def check_suffix(file_path):
    suffix = 0
    while os.path.isfile(file_path+'-{}'.format(suffix)):
        suffix += 1
    new_file_path = file_path +'-{}'.format(suffix)
    return new_file_path


def change_input(fn, replace_dict=None, add_dict=None):
    new_fn = check_suffix(fn + ".bak")

    shutil.move(fn, new_fn)

    destination = open(fn, "w")
    source = open(new_fn, "r")
    print fn
    for line in source:

        mod = False
        if replace_dict is not None:
            for name, val in replace_dict.iteritems():
                if name in line:
                    destination.write('  ' + name + ' = ' + val + " \n")
                    mod = True
        if add_dict is not None:
            for name, val in add_dict.iteritems():
                if name in line:
                    destination.write(line)
                    destination.write('  ' + val[0] + ' = ' + val[1] + " \n")
                    mod = True
        if not mod:
            destination.write(line)

    source.close()
    destination.close()