import cPickle as pck
import pandas as pd

from make_input.qe_input import makeQEInput,frame2change
from make_input.qe_run import run_qe_hpc
from make_input.qe_run import run_qe_hpc
from make_input.raw_info import SG2BravaisLattice
from tqdm import tqdm
from make_input.raw_info import bravaisLattice2ibrav,SG2BravaisLattice

ssg = range(1,230+1)
sg2ibrav = {}
ibrav2sg = {ibrav:[] for ibrav in bravaisLattice2ibrav.values()}
for sg in ssg:
    bl = SG2BravaisLattice[sg]
    ibrav = bravaisLattice2ibrav[bl]
    ibrav2sg[ibrav].append(sg)
    sg2ibrav[sg] = ibrav


calculation_type = '"vc-relax"'

zatom = 14

kpt = [2,2,2]
Nkpt = None
# rhocutoff ,wfccutoff = None,None
rhocutoff ,wfccutoff = 20*4,20
smearing = 1e-2
etot_conv_thr = 1e-4
forc_conv_thr = 1e-3
nstep = 2
scf_conv_thr = 1e-1

hpc = 'deneb'
node = 1
tasks = 8
cpus_per_tasks = 2
mem = 63000
time = '00:10:00'
debug = False

dataPath = '/scratch/musil/qmat/data/'
ppPath='"/scratch/musil/qmat/run_qe/pseudo/SSSP_acc_PBE/"'


fileNames = {}
infoPath = './info/'
structurePath = './structures/'

fileNames['crystals'] = structurePath + 'input_crystals_sg1-230-18-10-17.pck'

fileNames['wyck'] = infoPath+'SpaceGroup-multiplicity-wickoff-info.pck'
fileNames['general info'] = infoPath+'SpaceGroup-general-info.pck'
fileNames['elements info'] = infoPath+'General-Info-Elements-fast.pck'



with open(fileNames['crystals'],'rb') as f:
    crystals = pck.load(f)
with open(fileNames['wyck'],'rb') as f:
    WyckTable = pck.load(f)
SGTable = pd.read_pickle(fileNames['general info'])
ElemTable = pd.read_pickle(fileNames['elements info'])

dirNames = {(sg,it):dataPath + 'check_input_errors_relax/sg_{}-f_{}'.format(sg,it)
            for sg in  range(1,230+1) for it in range(len(crystals[sg]))}

# crystal = crystals[sg][it]
# dirName = dataPath + 'test_run/sg_{}-f_{}'.format(sg,it)

# print 'Calc in folder:'
# print dirName
print 'sending the calcs'
pbar = tqdm(total=len(dirNames),ascii=True)
for (sg,it),dirName in dirNames.iteritems():
    crystal = crystals[sg][it]
    input_str = makeQEInput(crystal,sg,WyckTable,SGTable,ElemTable,
                    zatom = zatom,rhocutoff = rhocutoff,wfccutoff = wfccutoff,
                    calculation_type=calculation_type,smearing=smearing,
                    pressure=0,press_conv_thr=0.5,cell_factor=2,
                    etot_conv_thr=1e-4,forc_conv_thr=1e-3,nstep=nstep,
                    scf_conv_thr=scf_conv_thr,print_forces=False,
                    kpt = kpt,Nkpt=Nkpt ,kpt_offset = [0,0,0],
                    ppPath=ppPath)


    exitstatus = run_qe_hpc(input_str,dirName,verbose=False,hpc=hpc, node=node, tasks=tasks,
                    cpus_per_tasks=cpus_per_tasks, mem=mem, time=time, debug=debug)

    pbar.update()

pbar.close()