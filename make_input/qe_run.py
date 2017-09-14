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