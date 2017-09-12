import numpy as np
from raw_info import missClassificationCorrection


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
    
    
def get_kpts(frame,Nkpt=1000):
    ir_cell = frame.get_reciprocal_cell()
    ir_l = np.linalg.norm(ir_cell,axis=1)
    
    n_kpt = float(Nkpt) / frame.get_number_of_atoms()
    n_reg = np.power(n_kpt,1./3.).round()
    # find the midle val to norm with it
    mid = list(set(range(3)).difference([ir_l.argmin(),ir_l.argmax()]))
    ir_l = ir_l / ir_l[mid]
    # get number of k point per directions
    nb = np.array(np.ceil(n_reg*ir_l),dtype=np.int64)
    # makes sure there are at least 3 kpts per directions
    nb[nb <3] = 3
    
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