import cPickle as pck
import pandas as pd

from make_input.qe_input import makeQEInput
from make_input.qe_run import run_qe_hpc
from make_input.qe_run import run_qe_hpc
from make_input.raw_info import SG2BravaisLattice

calculation_type = '"scf"'

zatom = 14
sg,it = 30,0
kpt = [2,2,2]
Nkpt = None
# rhocutoff ,wfccutoff = None,None
rhocutoff ,wfccutoff = 20*4,20
smearing = 1e-2
etot_conv_thr = 1e-4
forc_conv_thr = 1e-3
nstep = 150
scf_conv_thr = 1e-2

dataPath = '/scratch/musil/qmat/data/'
ppPath='"/scratch/musil/qmat/run_qe/pseudo/SSSP_acc_PBE/"'

hpc = 'deneb'
node = 1
tasks = 8
cpus_per_tasks = 2
mem = 63000
time = '00:30:00'
debug = True


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

# dirNames = {(sg,it):'../test_run/check_ibrav0_sym/sg_{}-f_{}'.format(sg,it)
#             for sg in ibrav0 for it in range(len(crystals[sg]))}

crystal = crystals[sg][it]
dirName = dataPath + 'test_run/sg_{}-f_{}'.format(sg,it)

print 'Calc in folder:'
print dirName

input_str = makeQEInput(crystal,sg,WyckTable,SGTable,ElemTable,
                zatom = zatom,rhocutoff = rhocutoff,wfccutoff = wfccutoff,
                calculation_type=calculation_type,smearing=smearing,
                pressure=0,press_conv_thr=0.5,cell_factor=2,
                etot_conv_thr=1e-4,forc_conv_thr=1e-3,nstep=150,
                scf_conv_thr=1e-6,
                kpt = kpt,Nkpt=Nkpt ,kpt_offset = [0,0,0],
                ppPath=ppPath)

print 'sending the calc'
exitstatus = run_qe_hpc(input_str,dirName,verbose=False,hpc=hpc, node=node, tasks=tasks,
                cpus_per_tasks=cpus_per_tasks, mem=mem, time=time, debug=debug)

print exitstatus