import cPickle as pck
import pandas as pd

from make_input.qe_input import makeQEInput
from make_input.qe_run import run_qe_local
from make_input.raw_info import SG2BravaisLattice

calculation_type = '"scf"'

zatom = 14

sg_list = [160,202,200]


kpt = [2,2,2]
Nkpt = None
# rhocutoff ,wfccutoff = None,None
rhocutoff ,wfccutoff = 20*4,20
smearing = 1e-2
etot_conv_thr = 1e-4
forc_conv_thr = 1e-3
nstep = 150
scf_conv_thr = 1e-1

dataPath = '/home/felix/git/run_qe/test_run/test_ibrav2/'
ppPath='"/home/felix/git/run_qe/pseudo/SSSP_acc_PBE/"'

fileNames = {}
infoPath = './info/'
structurePath = './structures/'

fileNames['crystals'] = structurePath + 'partial_input_crystals_sg3-230.pck'

fileNames['wyck'] = infoPath+'SpaceGroup-multiplicity-wickoff-info.pck'
fileNames['general info'] = infoPath+'SpaceGroup-general-info.pck'
fileNames['elements info'] = infoPath+'General-Info-Elements-fast.pck'



with open(fileNames['crystals'],'rb') as f:
    crystals = pck.load(f)
with open(fileNames['wyck'],'rb') as f:
    WyckTable = pck.load(f)
SGTable = pd.read_pickle(fileNames['general info'])
ElemTable = pd.read_pickle(fileNames['elements info'])


dirNames = {(sg,it):dataPath+'sg_{}-f_{}'.format(sg,it)
            for sg in sg_list for it in range(len(crystals[sg]))}

# crystal = crystals[sg][it]
# dirName = dataPath + 'test_run/sg_{}-f_{}'.format(sg,it)

for (sg,it),dirName in dirNames.iteritems():
    print dirName
    crystal = crystals[sg][it]
    input_str = makeQEInput(crystal,sg,WyckTable,SGTable,ElemTable,
                    zatom = zatom,rhocutoff = rhocutoff,wfccutoff = wfccutoff,
                    calculation_type=calculation_type,smearing=smearing,
                    pressure=0,press_conv_thr=0.5,cell_factor=2,
                    etot_conv_thr=1e-4,forc_conv_thr=1e-3,nstep=150,
                    scf_conv_thr=scf_conv_thr,print_forces=False,
                    kpt = kpt,Nkpt=Nkpt ,kpt_offset = [0,0,0],
                    ppPath=ppPath)

    print 'sending the calc'
    exitstatus = run_qe_local(input_str,dirName,verbose=False,
                 path2mpi='/usr/bin/',np=4,path2pw='/home/felix/source/qe-6.1/bin/')

    print exitstatus