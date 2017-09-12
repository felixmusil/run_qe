import ase
import numpy as np
import spglib as spg

# List of the Space groups that do not need special care
NOPROBLEM = [16, 17, 18, 19, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 47, 48, 49, 50,
             51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 6275, 76, 77, 78, 81, 83,
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
ibrav0 = [5, 8, 9, 12, 15,23, 24, 44, 45, 46, 71, 72, 73, 74,22, 42, 43,
          69, 70,79, 80, 82,87, 88, 97, 98, 107, 108, 109,110, 119, 120, 121, 122,
          139, 140, 141, 142,197, 199, 204, 206, 211, 214, 217, 220, 229, 230]
# List of spacegroups that need to be modified
TOMODIFY = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 35, 36, 37, 38, 39, 40, 41,
            42, 43, 44, 45, 46, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 79, 80, 82, 87, 88, 97,
            98, 107, 108, 109, 110, 119, 120, 121, 122, 139, 140, 141, 142, 146, 148, 155, 160, 161, 166,
            167, 196, 197, 199, 202, 203, 204, 206, 209, 210, 211, 214, 216, 217, 219, 220, 225, 226, 227,
            228, 229, 230]

frame2change = {0: ibrav0,
                2:[196, 202, 203, 209, 210, 216, 219, 225, 226, 227, 228],
                5:[146, 148, 155, 160, 161, 166, 167],
                -9:[20, 21, 35, 36, 37, 38, 39, 40, 41, 63, 64, 65, 66, 67, 68],
                -12:[3, 4, 6, 7, 10, 11, 13, 14],
}



def makeQEInput_sg(crystal,spaceGroupIdx,WyckTable,SGTable,ElemTable,
                 zatom = 14,rhocutoff = 20 * 4,wfccutoff = 20,
                 calculation_type='"scf"',
                 kpt = [2,2,2],kpt_offset = [0,0,0],ppPath='"./pseudo/"',
                 PP=['Si.pbe-n-rrkjus_psl.1.0.0.UPF']):
    
    inputDic = {}
    
    elemInfo = ElemTable[ElemTable['z'] == zatom]
    species = [elemInfo['sym'].values[0],]*1
    mass = [elemInfo['mass'].values[0]] # in atomic unit
    #PP = ['Si.pbe-n-rrkjus_psl.1.0.0.UPF']
    nvalence = elemInfo['val'].values[0]
    
    
    wyck = [SG2wyckoff(spaceGroupIdx,WyckTable)]
    positions = crystal.get_scaled_positions()[0]
    cellParam = crystal.get_cell_lengths_and_angles()
    ibrav = SG2ibrav(spaceGroupIdx,SGTable)
    #multipicity = WyckTable[spaceGroupIdx]['Multiplicity'][0]
    multipicity = crystal.get_number_of_atoms()

    if int(0.2*multipicity*nvalence/2.) > 4:
        nbnd = int(multipicity*nvalence/2.+int(0.2*multipicity*nvalence/2.))
    else:
        nbnd = int(multipicity*nvalence/2.+4)

    # define name list of QE
    control = {'calculation' : calculation_type,
              'outdir' : '"./out/"',
              'prefix' : '"qe"',
              'pseudo_dir' : '"../pseudo/"',
              'restart_mode' : '"from_scratch"',
              'verbosity' : '"high"',
              'wf_collect' : '.false.',
        }
    controlKeys = ['calculation','outdir','prefix','pseudo_dir', 'restart_mode','verbosity' ,'wf_collect']
    system = {
              'ecutrho' :   '{:.5f}'.format(rhocutoff),
              'ecutwfc' :   '{:.5f}'.format(wfccutoff),
              'ibrav' : str(ibrav),
              'nat' : str(1),
              'nbnd' : str(nbnd),
              'ntyp' : str(1),
              'occupations' : '"smearing"',
              'smearing' : '"cold"',
              'degauss' :   '1.0000000000d-02',
              'space_group' : str(spaceGroupIdx),
              'A' : str(cellParam[0]),
              'B' : str(cellParam[1]),
              'C' : str(cellParam[2]),
              'cosAB' : '{:.5f}'.format(np.cos(cellParam[5]*np.pi/180.)),
              'cosAC' : '{:.5f}'.format(np.cos(cellParam[4]*np.pi/180.)),
              'cosBC' : '{:.5f}'.format(np.cos(cellParam[3]*np.pi/180.)),
        }
    syskeys = [ 'ecutrho','ecutwfc','ibrav', 'nat','nbnd','ntyp' ,
               'occupations', 'smearing','degauss','space_group','A','B' ,'C', 'cosAB', 'cosAC', 'cosBC' ]
    electrons = {'conv_thr':'1.0000000000d-06'}

    # define cards of QE
    optionkey = 'unit'
    atomic_sp = {'ATOMIC_SPECIES':{'species':np.unique(species), 
                                   'mass':mass,'PP':PP},'unit':''}
    atspkeys = ['species','mass','PP']
    atomic_pos = {'ATOMIC_POSITIONS':{'species':species, 'wickoffs':wyck,
                                      'positions':positions},'unit':'crystal_sg'}
    atposkeys = ['species', 'wickoffs','positions']
    kpoints = {'K_POINTS':{'kpoints':kpt+kpt_offset},'unit':'automatic'}
    kptkeys = ['kpoints']

    qeInput = ''

    qeInput += makeNamelist('&CONTROL',control)
    qeInput += makeNamelist( '&SYSTEM' ,system,syskeys)
    qeInput += makeNamelist( '&ELECTRONS',electrons)
    qeInput += makeCard(atomic_sp,cardKeys = atspkeys,optionKey = optionkey,T = True) 
    qeInput += makeCard(atomic_pos,cardKeys = atposkeys,optionKey = optionkey,T = True) 
    qeInput += makeCard(kpoints,cardKeys = kptkeys,optionKey = optionkey,T = False)
   
    return qeInput


def makeQEInput_ibrav0(crystal,WyckTable,SGTable,ElemTable,spaceGroupIdx=10,
                 zatom = 14,rhocutoff = 20 * 4,wfccutoff = 20,
                        kpt = [2,2,2],kpt_offset = [0,0,0],calculation_type='"scf"',
                        ppPath='"./pseudo/"',PP=['Si.pbe-n-rrkjus_psl.1.0.0.UPF']):
    inputDic = {}
    
    elemInfo = ElemTable[ElemTable['z'] == zatom]
    
    mass = [elemInfo['mass'].values[0]] # in atomic unit
    #PP = ['Si.pbe-n-rrkjus_psl.1.0.0.UPF']
    nvalence = elemInfo['val'].values[0]
    
    (lattice, positions, numbers) = spg.find_primitive(crystal, symprec=1e-05, angle_tolerance=-1.0)
    primitive_atoms = ase.Atoms(cell=lattice,scaled_positions=positions,numbers=numbers)
    
    positions = primitive_atoms.get_positions()
    cell = primitive_atoms.get_cell()
    ibrav = 0
    Natom = primitive_atoms.get_number_of_atoms()
    species = [elemInfo['sym'].values[0],]*Natom
    
    multiplicity = Natom
    if int(0.2*multiplicity*nvalence/2.) > 4:
        nbnd = int(multiplicity*nvalence/2.+int(0.2*multiplicity*nvalence/2.))
    else:
        nbnd = int(multiplicity*nvalence/2.+4)

    # define name list of QE
    control = {'calculation' : calculation_type,
              'outdir' : '"./out/"',
              'prefix' : '"qe"',
              'pseudo_dir' : ppPath,
              'restart_mode' : '"from_scratch"',
              'verbosity' : '"high"',
              'wf_collect' : '.false.',
        }
    controlKeys = ['calculation','outdir','prefix','pseudo_dir', 'restart_mode','verbosity' ,'wf_collect']
    system = {
              'ecutrho' :   '{:.5f}'.format(rhocutoff),
              'ecutwfc' :   '{:.5f}'.format(wfccutoff),
              'ibrav' : str(ibrav),
              'nat' : str(Natom),
              'nbnd' : str(nbnd),
              'ntyp' : str(1),
              'occupations' : '"smearing"',
              'smearing' : '"cold"',
              'degauss' :   '1.0000000000d-02',
        }
    syskeys = [ 'ecutrho','ecutwfc','ibrav', 'nat','nbnd','ntyp' ,
               'occupations', 'smearing','degauss' ]
    electrons = {'conv_thr':'1.0000000000d-06'}

    # define cards of QE
    optionkey = 'unit'
    atomic_sp = {'ATOMIC_SPECIES':{'species':np.unique(species), 'mass':mass,'PP':PP},'unit':''}
    atspkeys = ['species','mass','PP']
    atomic_pos = {'ATOMIC_POSITIONS':{'species':species,'positions':positions},'unit':'angstrom'}
    atposkeys = ['species','positions']
    kpoints = {'K_POINTS':{'kpoints':kpt+kpt_offset},'unit':'automatic'}
    kptkeys = ['kpoints']
    cell_param = {'CELL_PARAMETERS':{'cell':cell},'unit':'angstrom'}
    cellkeys = ['cell']


    qeInput = ''

    qeInput += makeNamelist('&CONTROL',control)
    qeInput += makeNamelist( '&SYSTEM' ,system,syskeys)
    qeInput += makeNamelist( '&ELECTRONS',electrons)
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
                cardStr += stl + tabulate(ll, tablefmt="plain").replace('[','').replace(']','').replace(',','').replace('\n',endl+stl) + endl
        
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