import cPickle as pck
import numpy as np
from make_input.qe_input import makeQEInput_new
from make_input.qe_run import run_qe_hpc
from tqdm import tqdm
from make_input.SSSP_acc_PBE_info import wfccutoffs,rhocutoffs

dry_run = True

calculation_type = '"vc-relax"'

sites_z = [14]

kpt = [2,2,2]
Nkpt = 1000
# rhocutoff ,wfccutoff = None,None
rhocutoff ,wfccutoff = [], []
for zatom in sites_z:
    rhocutoff.append(rhocutoffs[zatom])
    wfccutoff.append( wfccutoffs[zatom])

rhocutoff = np.max(rhocutoff)
wfccutoff = np.max(wfccutoff)


smearing = 1e-3
etot_conv_thr = 1e-4
forc_conv_thr = 1e-4
nstep = 100
scf_conv_thr = 1e-6
symprec = 1e-5

hpc = 'fidis'
node = 1
tasks = 14
cpus_per_tasks = 2
mem = 63000
time = '24:00:00'
debug = False

dataPath = '/scratch/musil/qmat/data/'
ppPath='"/home/musil/git/run_qe/pseudo/SSSP_acc_PBE/"'


fileNames = {}

structurePath = './structures/'

fileNames['crystals'] = structurePath + 'structures_141117.pck'


with open(fileNames['crystals'],'rb') as f:
    crystals = pck.load(f)

dirNames = {it:dataPath + 'run_relax_Si_new/idx_{}'.format(it)
            for it, _ in enumerate(crystals)}

# crystal = crystals[sg][it]
# dirName = dataPath + 'test_run/sg_{}-f_{}'.format(sg,it)

# print 'Calc in folder:'
# print dirName
print 'sending the calcs'
pbar = tqdm(total=len(dirNames),ascii=True)
for it,dirName in dirNames.iteritems():
    crystal = crystals[it]

    input_str = makeQEInput_new(crystal, sites_z, symprec=symprec,
                                rhocutoff=rhocutoff, wfccutoff=wfccutoff,
                                calculation_type=calculation_type, smearing=smearing,
                                pressure=0, press_conv_thr=0.5, cell_factor=2,
                                etot_conv_thr=etot_conv_thr, forc_conv_thr=forc_conv_thr, nstep=nstep,
                                scf_conv_thr=scf_conv_thr, print_forces=True, print_stress=True,
                                restart=False, collect_wf=True, force_ibrav0=False,
                                kpt=kpt, Nkpt=Nkpt, kpt_offset=[0, 0, 0],
                                ppPath=ppPath)


    exitstatus = run_qe_hpc(input_str,dirName,verbose=False,hpc=hpc, node=node,
                    tasks_per_node=tasks,name='{}'.format(it),dry_run=dry_run,
                    cpus_per_tasks=cpus_per_tasks, mem=mem, time=time, debug=debug)

    pbar.update()


pbar.close()