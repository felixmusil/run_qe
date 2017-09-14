import ase
import numpy as np
import spglib as spg
from custom_frame import frame2qe_format
from utils import get_kpts

from SSSP_acc_PBE_info import PP_names,rhocutoffs,wfccutoffs

# List of the Space groups that do not need special care
NOPROBLEM = [16, 17, 18, 19, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 47, 48, 49, 50,
             51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,75, 76, 77, 78, 81, 83,
             84, 85, 86, 89, 90, 91, 92, 93, 94, 95, 96, 99,
             100, 101, 102, 103, 104, 105, 106, 111, 112, 113, 114, 115, 116, 117,
             118, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
             136, 137, 138,143, 144, 145, 147, 149, 150, 151, 152, 153, 154, 156,
             157, 158, 159,
             162, 163, 164, 165, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177,
             178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
             192, 193, 194,195, 198, 200, 201, 205, 207, 208, 212, 213, 215, 218,
             221, 222, 223, 224]
# List of spacegroups that need ibrav0 input type
ibrav0 = [5, 8, 9, 12, 15,23, 24,20, 21, 35, 36, 37, 38, 39, 40, 41, 63,
          64, 65, 66, 67, 68,
          44, 45, 46, 71, 72, 73, 74,22, 42, 43,
          69, 70,79, 80, 82,87, 88, 97, 98, 107, 108, 109,110, 119, 120, 121, 122,
          139, 140, 141, 142,197, 199, 204, 206, 211, 214, 217, 220, 229, 230]
# List of spacegroups that need to be modified
TOMODIFY = [1,2,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 35, 36, 37, 38, 39, 40, 41,
            42, 43, 44, 45, 46, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 79, 80, 82, 87, 88, 97,
            98, 107, 108, 109, 110, 119, 120, 121, 122, 139, 140, 141, 142, 146, 148, 155, 160, 161, 166,
            167, 196, 197, 199, 202, 203, 204, 206, 209, 210, 211, 214, 216, 217, 219, 220, 225, 226, 227,
            228, 229, 230]

frame2change = {3: [197, 199, 204, 206, 211, 214, 217, 220, 229, 230],
                7:[79, 80, 82, 87, 88, 97, 98, 107, 108, 109, 110, 119, 120, 121, 122,
                   139, 140, 141, 142],
                10:[22, 42, 43, 69, 70],
                11:[23, 24, 44, 45, 46, 71, 72, 73, 74],
                13:[5, 8, 9, 12, 15],
                14:[1,2],
                2:[196, 202, 203, 209, 210, 216, 219, 225, 226, 227, 228],
                5:[146, 148, 155, 160, 161, 166, 167],
                -9:[20, 21, 35, 36, 37, 38, 39, 40, 41, 63, 64, 65, 66, 67, 68],
                -12:[3, 4, 6, 7, 10, 11, 13, 14],
}

def makeQEInput(crystal,spaceGroupIdx,WyckTable,SGTable,ElemTable,
                zatom = 14,rhocutoff = None,wfccutoff = None,
                calculation_type='"scf"',smearing=1e-2,
                pressure=0,press_conv_thr=0.5,cell_factor=2,
                etot_conv_thr=1e-4,forc_conv_thr=1e-3,nstep=150,
                scf_conv_thr=1e-6,
                kpt = [2,2,2],Nkpt=None,kpt_offset = [0,0,0],
                ppPath='"./pseudo/SSSP_acc_PBE/"'):

    new_crystal = frame2qe_format(crystal,spaceGroupIdx)

    PP = [PP_names[zatom]]

    if wfccutoff is None:
        wfccutoff = wfccutoffs[zatom]
    if rhocutoff is None:
        rhocutoff = rhocutoffs[zatom]

    if Nkpt is not None:
        kpt = list(get_kpts(new_crystal,Nkpt=Nkpt))
    # pressure in kBar and en_tot/forces in a.u.
    kwargs = dict(crystal=new_crystal,spaceGroupIdx=spaceGroupIdx,WyckTable=WyckTable,
                  SGTable=SGTable,ElemTable=ElemTable,
                  etot_conv_thr=etot_conv_thr, forc_conv_thr=forc_conv_thr,nstep=nstep,
                 zatom = zatom,rhocutoff = rhocutoff,wfccutoff = wfccutoff,
                  scf_conv_thr=scf_conv_thr,
                  pressure=pressure, press_conv_thr=press_conv_thr, cell_factor=cell_factor,
                 calculation_type=calculation_type,smearing=smearing,
                 kpt = kpt,kpt_offset = kpt_offset,ppPath=ppPath,
                 PP=PP)
    if spaceGroupIdx in ibrav0:
        qeInput = makeQEInput_ibrav0(**kwargs)
    else:
        qeInput = makeQEInput_sg(**kwargs)

    return qeInput


def makeQEInput_sg(crystal,spaceGroupIdx,WyckTable,SGTable,ElemTable,
                 zatom = 14,rhocutoff = 20 * 4,wfccutoff = 20,
                 calculation_type='"scf"',smearing=1e-2,
                etot_conv_thr=1e-4, forc_conv_thr=1e-3,nstep=150,
                scf_conv_thr=1e-6,
                 pressure=0,press_conv_thr=0.5,cell_factor=2,
                 kpt = [2,2,2],kpt_offset = [0,0,0],ppPath='"./pseudo/"',
                 PP=['Si.pbe-n-rrkjus_psl.1.0.0.UPF']):

    
    elemInfo = ElemTable[ElemTable['z'] == zatom]
    species = [elemInfo['sym'].values[0],]*1
    mass = [elemInfo['mass'].values[0]] # in atomic unit
    #PP = ['Si.pbe-n-rrkjus_psl.1.0.0.UPF']
    nvalence = elemInfo['val'].values[0]
    
    
    wyck = [SG2wyckoff(spaceGroupIdx,WyckTable)]
    positions = crystal.get_scaled_positions()[0].reshape((1,-1))
    cellParam = crystal.get_cell_lengths_and_angles()
    ibrav = SG2ibrav(spaceGroupIdx)
    Natom = crystal.get_number_of_atoms()
    if ibrav == -12:
        uniqueb = '.TRUE.'
    else:
        uniqueb = '.FALSE.'
    # if ibrav == -9:
    #     ibrav = 9
    nbnd = get_number_of_bands(Natom=Natom,Nvalence=nvalence)

    d2r = np.pi / 180.

    # define name list of QE
    control = {'calculation' : calculation_type,
              'outdir' : '"./out/"',
              'prefix' : '"qe"',
              'pseudo_dir' : ppPath,
              'restart_mode' : '"from_scratch"',
              'verbosity' : '"high"',
              'wf_collect' : '.false.',
              'nstep': '{:.0f}'.format(nstep),
              'etot_conv_thr' : '{:.5f}'.format(etot_conv_thr*Natom),
              'forc_conv_thr' : '{:.5f}'.format(forc_conv_thr),
        }
    controlKeys = ['calculation', 'outdir', 'prefix', 'pseudo_dir', 'restart_mode',
                   'verbosity', 'wf_collect','nstep', 'etot_conv_thr', 'forc_conv_thr']
    system = {
              'ecutrho' :   '{:.5f}'.format(rhocutoff),
              'ecutwfc' :   '{:.5f}'.format(wfccutoff),
              'ibrav' : str(ibrav),
              'nat' : str(1),
              'nbnd' : str(nbnd),
              'ntyp' : str(1),
              'occupations' : '"smearing"',
              'smearing' : '"cold"',
              'degauss' :   '{:.6f}'.format(smearing),
              'space_group' : str(spaceGroupIdx),
              'uniqueb':uniqueb,
              'A' : str(cellParam[0]),
              'B' : str(cellParam[1]),
              'C' : str(cellParam[2]),
              'cosAB' : '{:.5f}'.format(np.cos(cellParam[5]*d2r)),
              'cosAC' : '{:.5f}'.format(np.cos(cellParam[4]*d2r)),
              'cosBC' : '{:.5f}'.format(np.cos(cellParam[3]*d2r)),
        }
    syskeys = [ 'ecutrho','ecutwfc','ibrav', 'nat','nbnd','ntyp' ,
               'occupations', 'smearing','degauss','space_group','uniqueb','A','B' ,'C', 'cosAB', 'cosAC', 'cosBC' ]
    electrons = {'conv_thr':'{:.8f}'.format(scf_conv_thr)}
    cellkeys = ['cell_factor', 'press', 'press_conv_thr']
    # pressure in [KBar]
    cell = {
        'cell_factor': '{:.5f}'.format(cell_factor),
        'press': '{:.5f}'.format(pressure),
        'press_conv_thr': '{:.5f}'.format(press_conv_thr),
    }

    # define cards of QE
    optionkey = 'unit'
    atomic_sp = {'ATOMIC_SPECIES':{'species':np.unique(species), 
                                   'mass':mass,'PP':PP},'unit':''}
    atspkeys = ['species','mass','PP']
    atomic_pos = {'ATOMIC_POSITIONS':{'species':species, 'wickoffs':wyck,
                                      'positions':list(positions)},
                  'unit':'crystal_sg'}
    atposkeys = ['species', 'wickoffs','positions']
    kpoints = {'K_POINTS':{'kpoints':kpt+kpt_offset},'unit':'automatic'}
    kptkeys = ['kpoints']

    qeInput = ''

    qeInput += makeNamelist('&CONTROL',control,controlKeys)
    qeInput += makeNamelist( '&SYSTEM' ,system,syskeys)
    qeInput += makeNamelist( '&ELECTRONS',electrons)
    qeInput += makeNamelist('&CELL', cell)
    qeInput += makeCard(atomic_sp,cardKeys = atspkeys,optionKey = optionkey,T = True) 
    qeInput += makeCard(atomic_pos,cardKeys = atposkeys,optionKey = optionkey,T = True) 
    qeInput += makeCard(kpoints,cardKeys = kptkeys,optionKey = optionkey,T = False)
   
    return qeInput


def makeQEInput_ibrav0(crystal,WyckTable,SGTable,ElemTable,spaceGroupIdx=10,
                 zatom = 14,rhocutoff = 20 * 4,wfccutoff = 20,smearing=1e-2,
                pressure=0,press_conv_thr=0.5,cell_factor=2,
                etot_conv_thr=1e-4,forc_conv_thr=1e-3,nstep=150,
                 scf_conv_thr=1e-6,
                kpt = [2,2,2],kpt_offset = [0,0,0],calculation_type='"scf"',
                ppPath='"./pseudo/"',PP=['Si.pbe-n-rrkjus_psl.1.0.0.UPF']):

    elemInfo = ElemTable[ElemTable['z'] == zatom]
    
    mass = [elemInfo['mass'].values[0]] # in atomic unit
    #PP = ['Si.pbe-n-rrkjus_psl.1.0.0.UPF']
    nvalence = elemInfo['val'].values[0]
    

    positions = crystal.get_positions()
    lattice = crystal.get_cell()
    ibrav = 0
    Natom = crystal.get_number_of_atoms()
    species = [elemInfo['sym'].values[0],]*Natom


    nbnd = get_number_of_bands(Natom=Natom,Nvalence=nvalence)

    # define name list of QE
    control = {'calculation' : calculation_type,
              'outdir' : '"./out/"',
              'prefix' : '"qe"',
              'pseudo_dir' : ppPath,
              'restart_mode' : '"from_scratch"',
              'verbosity' : '"high"',
              'wf_collect' : '.false.',
              'nstep':'{:.0f}'.format(nstep),
              'etot_conv_thr': '{:.5f}'.format(etot_conv_thr*Natom),
              'forc_conv_thr': '{:.5f}'.format(forc_conv_thr),
        }
    controlKeys = ['calculation','outdir','prefix','pseudo_dir', 'restart_mode',
                   'verbosity' ,'wf_collect','nstep','etot_conv_thr','forc_conv_thr']
    system = {
              'ecutrho' :   '{:.5f}'.format(rhocutoff),
              'ecutwfc' :   '{:.5f}'.format(wfccutoff),
              'ibrav' : str(ibrav),
              'nat' : str(Natom),
              'nbnd' : str(nbnd),
              'ntyp' : str(1),
              'occupations' : '"smearing"',
              'smearing' : '"cold"',
              'degauss' :   '{:.6f}'.format(smearing),
        }
    syskeys = [ 'ecutrho','ecutwfc','ibrav', 'nat','nbnd','ntyp' ,
               'occupations', 'smearing','degauss' ]
    electrons = {'conv_thr':'{:.8f}'.format(scf_conv_thr)}

    cellkeys = ['cell_factor', 'press', 'press_conv_thr']
    # pressure in [KBar]
    cell = {
        'cell_factor': '{:.5f}'.format(cell_factor),
        'press': '{:.5f}'.format(pressure) ,
        'press_conv_thr': '{:.5f}'.format(press_conv_thr),
    }

    # define cards of QE
    optionkey = 'unit'
    atomic_sp = {'ATOMIC_SPECIES':{'species':np.unique(species), 'mass':mass,'PP':PP},'unit':''}
    atspkeys = ['species','mass','PP']
    atomic_pos = {'ATOMIC_POSITIONS':{'species':species,'positions':positions},'unit':'angstrom'}
    atposkeys = ['species','positions']
    kpoints = {'K_POINTS':{'kpoints':kpt+kpt_offset},'unit':'automatic'}
    kptkeys = ['kpoints']
    cell_param = {'CELL_PARAMETERS':{'cell':lattice},'unit':'angstrom'}
    cellkeys = ['cell']


    qeInput = ''

    qeInput += makeNamelist('&CONTROL',control,controlKeys)
    qeInput += makeNamelist( '&SYSTEM' ,system,syskeys)
    qeInput += makeNamelist( '&ELECTRONS',electrons)
    qeInput += makeNamelist('&CELL', cell)
    qeInput += makeCard(atomic_sp,cardKeys = atspkeys,optionKey = optionkey,T = True) 
    qeInput += makeCard(atomic_pos,cardKeys = atposkeys,optionKey = optionkey,T = True) 
    qeInput += makeCard(kpoints,cardKeys = kptkeys,optionKey = optionkey,T = False)
    qeInput += makeCard(cell_param,cardKeys = cellkeys,optionKey = optionkey,T = True)
    
    inputDic = qeInput
        
    return inputDic




def makeCard(cardDict,cardKeys=None,optionKey='unit',
             T=False, endl = ' \n',eq = ' = ',stl = '  '):
    """Transform a Card block in dict form into a string."""
    from itertools import izip_longest
    from  tabulate import tabulate
    cardStr = ''
    if cardKeys is None:
        keys =  cardDict.keys()
        keys.remove(optionKey)
        cardKeys = cardDict[keys[0]]
        
    if optionKey in cardDict.keys():
        keys =  cardDict.keys()
        keys.remove(optionKey)
        for key in keys:
            cardStr += key + stl
        cardStr += cardDict[optionKey] + endl

        for key,val in cardDict.items():
            if isinstance(val, dict):
                l = [val[cardKey] for cardKey in cardKeys]
                ll = map(list, izip_longest(*l)) if T else l
                cardStr += stl + tabulate(ll, tablefmt="plain")\
                    .replace('[','').replace(']','').replace(',','').replace('\n',endl+stl) + endl

        return cardStr
    else:
        print 'cardDict has no key called: {} (optionKey)'.format(optionKey)
    
def makeNamelist(namelistName,namelistDict,namelistKeys=None, 
                 endl = ' \n',eq = ' = ',stl = '  '):
    """Transform a Name List block in dict form into a string."""
    if namelistKeys is None:
        namelistKeys = namelistDict.keys()
    namelistStr = namelistName + endl
    for key in namelistKeys:
        namelistStr += stl + key + eq + namelistDict[key] + endl
    namelistStr += '/' + endl
    return namelistStr


def get_number_of_bands(Natom,Nvalence):
    if int(0.2*Natom*Nvalence/2.) > 4:
        nbnd = int(Natom*Nvalence/2.+int(0.2*Natom*Nvalence/2.))
    else:
        nbnd = int(Natom*Nvalence/2.+4)
    return nbnd

def SG2ibrav(spaceGroupIdx,SGTable=None):
    from raw_info import bravaisLattice2ibrav

    if SGTable is None:
        from raw_info import SG2BravaisLattice
        bravaisLattice = SG2BravaisLattice[spaceGroupIdx]
    else:
        sgInfo = SGTable[SGTable['Table No.'] == spaceGroupIdx]
        bravaisLattice = sgInfo['Crystal System'].values[0].encode('utf-8') + ' ' + sgInfo['Full notation'].values[0][0].encode('utf-8')

    ibrav = bravaisLattice2ibrav[bravaisLattice]
   
    return ibrav

def SG2wyckoff(spaceGroupIdx,wyckTable):
    highestMultiplicitySite = wyckTable[spaceGroupIdx].loc[0,:]
    return str(highestMultiplicitySite['Multiplicity']) + \
                highestMultiplicitySite['Wyckoff letter']