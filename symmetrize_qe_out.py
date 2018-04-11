import numpy as np
from read_output.qe_reader import read_qe
from make_input.qe_input import makeQEInput_new
from make_input.SSSP_acc_PBE_info import wfccutoffs,rhocutoffs
import argparse
from ase.spacegroup import crystal
import spglib as spg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Creates a QE input file from the last geometry of a pw output file. 
    typical use: python symmetrize_qe_out.py -out qe.in -sfn qe.out -pp /home/musil/git/run_qe/pseudo/SSSP_acc_PBE/ -asn 14 -c vc-relax -nk 3000 -sm 1e-3 -ethr 1e-4 -fthr 1e-4 -sthr 1e-6 -nstep 100""")

    parser.add_argument('-asn',"--asymmetric-sites-number", type=str, default="",
                        help="Comma-separated list of atom Z (e.g. qmat-1 Si would be 14 and qmat-2 Si,Ge would be 14,32)")
    parser.add_argument("-sfn", "--structure-fn", type=str, default="",
                        help="Name of the QE output file to take the last structure from.")
    parser.add_argument("-pp", "--pseudo-path", type=str, default="",
                        help="Path to the folder containing the pseudopotentials.")
    parser.add_argument("-out", "--out-file-name", type=str, default="./qe.in",
                        help="Name of the QE input file to write into.")
    parser.add_argument("-c", "--calculation-type", type=str, default='vc-relax',
                        help="type of pw calculation.")
    parser.add_argument("-nk","--Nkpt", type=int, default=3000, help="Number of kpoints")
    parser.add_argument("-sm", "--smearing", type=float, default=1e-3, help="Smearing strenght")
    parser.add_argument("-ethr", "--etot-conv-thr", type=float, default=1e-4, help="Total energy convergence threshold")
    parser.add_argument("-fthr", "--forc-conv-thr", type=float, default=1e-4, help="Total force convergence threshold")
    parser.add_argument("-nstep", "--n-scf-step", type=int, default=100, help="Maximal number of scf step")
    parser.add_argument("-sthr", "--scf_conv_thr", type=float, default=1e-6, help="SCF convergence threshold")


    args = parser.parse_args()

    #calculation_type = '"vc-relax"'
    calculation_type = '"' + args.calculation_type + '"'
    Nkpt = args.Nkpt
    smearing = args.smearing
    etot_conv_thr = args.etot_conv_thr
    forc_conv_thr = args.forc_conv_thr
    nstep = args.n_scf_step
    scf_conv_thr = args.scf_conv_thr


    fn_qe_out = args.structure_fn
    fn_qe_in = args.out_file_name
    ppPath = '"' + args.pseudo_path + '"'
    if args.asymmetric_sites_number == "":
        raise ValueError('must provide --asymmetric-sites-number arg ')
    else:
        sites_z = map(int, args.asymmetric_sites_number.split(','))


    rhocutoff, wfccutoff = [], []
    for zatom in sites_z:
        rhocutoff.append(rhocutoffs[zatom])
        wfccutoff.append(wfccutoffs[zatom])

    rhocutoff = np.max(rhocutoff)
    wfccutoff = np.max(wfccutoff)


    symprec = 1e-5

    crystal = read_qe(fn_qe_out)[-1]

    sym_data = spg.get_symmetry_dataset(crystal)

    if len(np.unique(sym_data['equivalent_atoms'])) > len(sites_z):
        raise ValueError('The input structure has more sites than expected: {}>{}'.format(
            len(np.unique(sym_data['equivalent_atoms'])), len(sites_z)
        ))

    symbols = []
    asym_positions = []
    # cell.

    cc = crystal(symbols=symbols, basis=asym_positions, spacegroup=sg,
                 cellpar=cell, symprec=1e-7, pbc=True, primitive_cell=False)


    input_str = makeQEInput_new(crystal, sites_z, symprec=symprec,
                                rhocutoff=rhocutoff, wfccutoff=wfccutoff,
                                calculation_type=calculation_type, smearing=smearing,
                                pressure=0, press_conv_thr=0.5, cell_factor=2,
                                etot_conv_thr=etot_conv_thr, forc_conv_thr=forc_conv_thr, nstep=nstep,
                                scf_conv_thr=scf_conv_thr, print_forces=True, print_stress=True,
                                restart=False, collect_wf=True, force_ibrav0=False,
                                Nkpt=Nkpt, kpt_offset=[0, 0, 0],
                                ppPath=ppPath)


    with open(fn_qe_in,'w') as f:
        f.write(input_str)