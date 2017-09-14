import os,sys
import numpy as np
import subprocess as sp

os.environ['TMPDIR'] = "/tmp/"

def make_dir(file_path):
    directory = os.path.abspath(file_path)
    suffix = 0
    while os.path.exists(directory+'-{}'.format(suffix)):
        suffix += 1
    path = directory +'-{}/'.format(suffix)
    os.makedirs(path)
    return path

def run_qe_local(input,dirName,verbose=False,
                 path2mpi='/usr/bin/',np=2,path2pw='/home/musil/source/qe-6.1/bin/'):

    path = make_dir(dirName)
    inputName = os.path.abspath(path+'/qe.in')
    outputName = os.path.abspath(path+'/qe.out')
    errName = os.path.abspath(path+'/qe.err')
    with open(inputName,'w') as f:
        f.write(input)
    param = '-in '+ inputName
    if verbose:
        print param
    # Set up the echo command and direct the output to a pipe
    exitState = sp.call('{}mpirun -np {:.0f} {}pw.x {}'.format(path2mpi,np,path2pw,param)
                            , stdout=open(outputName, 'w'),
                            stderr=open(errName, 'w'), shell=True)
    error = 'No error'
    if exitState:
        print dirName
        with open(outputName,'r') as f:
            error = f.read()

    return exitState,error


deneb = {'sbatch':'#SBATCH --constrain=E5v2 ',
         'module':'module load intel/17.0.2  intel-mpi/2017.2.174 intel-mkl/2017.2.174 ' ,
          'p2pw':'/home/musil/source/qe-6.1/bin/pw.x' , }


def make_submit_script(hpc='deneb', input_fn='qe.in', output_fn='qe.out',
                       workdir='/scratch/musil/qmat/run_qe/', node=1, tasks=1,
                       cpus_per_tasks=1, mem=63000, time='00:10:00', debug=False):
    if hpc == 'deneb':
        config = deneb
    else:
        print 'hpc not recognized'
        return None

    ndl = '\n'
    sbatch = '#!/bin/bash {ndl}\
#SBATCH --workdir {workdir} {ndl}\
#SBATCH --nodes {node} {ndl}\
#SBATCH --ntasks {tasks} {ndl}\
#SBATCH --cpus-per-task {cpus} {ndl}\
#SBATCH --mem {mem} {ndl}\
#SBATCH --time {time}  {ndl}'.format(
        workdir=workdir, node=str(node),
        tasks=str(tasks), cpus=str(cpus_per_tasks),
        mem=str(mem), time=time, ndl=ndl)

    sbatch += config['sbatch'] + ndl
    if debug:
        sbatch += '#SBATCH --partition=debug '+ ndl
    if cpus_per_tasks > 1:
        sbatch += 'export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK '+ ndl
    sbatch += ndl

    module = 'module purge ' + ndl
    module += config['module'] + ndl
    module += ndl

    cmd = 'srun '
    cmd += config['p2pw'] + ' ' + '-in ' + input_fn +' > '+ output_fn +' '+ ndl

    return sbatch + module + cmd

def run_qe_hpc(input_str,dirName,verbose=False,hpc='deneb', node=1, tasks=1,
                cpus_per_tasks=1, mem=63000, time='00:10:00', debug=False):
    path = make_dir(dirName)
    inputName = os.path.abspath(path+'/qe.in')
    submit_scriptName = os.path.abspath(path+'/qe.sh')
    outputName = os.path.abspath(path+'/qe.out')
    jobName = os.path.abspath(path+'/job.id')
    errName = os.path.abspath(path+'/job.err')

    submit_script = make_submit_script(hpc=hpc, input_fn=inputName, output_fn=outputName,
                                       workdir=path, node=node, tasks=tasks,
                                       cpus_per_tasks=cpus_per_tasks, mem=mem, time=time, debug=debug)

    with open(inputName,'w') as f:
        f.write(input_str)
    with open(submit_scriptName,'w') as f:
        f.write(submit_script)

    #  Set up the echo command and direct the output to a pipe
    exitState = sp.call('sbatch {}'.format(submit_scriptName),
                        stdout=open(jobName, 'a'),stderr=open(errName, 'w'),shell=True)

    with open(jobName,'r') as f:
        lines = f.readlines()
        print lines[-1]

    return exitState


