import MDAnalysis as mda
import numpy as np
import pandas as pd
import scipy
import nglview as nv
import matplotlib.pyplot as plt
from MDAnalysis.analysis.rms import RMSD
from MDAnalysis.analysis.rms import RMSF
from MDAnalysis.analysis import pca, align
import os



def mda_selection_from_list(resnum_list):
    '''
    Provide a list of residue numbers to include in an mdanalysis selection
    Parameters
    ----------
    resnum_list: list
        list of all the residue numbers you want to include in the selection
        if tuple, the range from tuple[0] to tuple[1] inclusive will be generated

    Returns
    -------
    String selecting each residue number individually
    '''

    mda_selection = ""

    if type(resnum_list) == tuple:
        resnum_list = list(range(resnum_list[0],resnum_list[1]+1))

    for j,resnum in enumerate(resnum_list):
        if j != len(array)-1:
            mda_selection += f"resnum {resnum} or "
        else:
            mda_selection += f"resnum {resnum}"

    return mda_selection


def write_ca_traj(structure, traj, output, resnums=None, frame_start=0, frame_end=None, align_ref=None,
                  split_subunits=False):
    '''
    Save a trajectory as only CA coordinates

    Parameters
    ----------
    structure: string
        path to structure file
    
    traj: string
        path to trajectory

    output: string
        path to the output CA trajectory in xtc format
    
    resnums: list or tuple
        list of residue numbers (1 indexed) corresponding to the residue numbers in the structure file
    
    frame_start: int
        frame number to begin writing the trajectory

    frame_end: int
        frame number to stop writing the trajectory
    
    align_ref: string
        path to reference structure (CA only) that has the same number of atoms as the 
        CA trajectory will have 

    split_subunits: bool
        TODO add function to separate the trajectory into individual subunits and then concatenate them.
    '''
    
    u = mda.Universe(structure, traj)

    # deal with residue number selections
    if resnums:
        selections = mda_selection_from_list(resnums)
        CAs = u.select_atoms(f'name CA and ({selections})')
    else:
        CAs = u.select_atoms('name CA')

    # check that output is xtc format
    if not output.endswith('.xtc'):
        print('output file extension needs to be xtc', flush=True)
        return
    if frame_end==None:
        frame_end = len(u.trajectory)

    if align_ref:
        # should be using a CA only structure here the works for all systems in the combined PCA
        ref = mda.Universe(align_ref)   
        alignment = align.AlignTraj(u, ref, filename=output)
        alignment.run()
    else:
        # write trajectory
        with mda.Writer(f'{output}', CAs.n_atoms) as W:
            for ts in u.trajectory[frame_start:frame_end]:
                W.write(CAs)

def write_ca_structure(structure, output, resnums=None):
    '''
    write a structure file of just the CA selection

    '''

    u = mda.Universe(structure)

    # check to see if resnums are already in mda format 
    if resnums:
        if type(resnums) != str:
            resnums = mda_selection_from_list(resnums)

        ca_selection = u.select_atoms('name CA and ({resnums})')
    else:

        ca_selection = u.select_atoms('name CA')
    if not output.endswith('.pdb'):
        print('CA structure file needs to be pdb format')
    ca_selection.write(output)

def cartesian_pca(trajs, structure, selection=None, selection_align=False, output_folder=None, output_prepend=None):
    '''
    PCA on the cartesian coordinates of a trajectory

    Parameters
    ----------
    trajs: list
        list of CA trajectories. All trajectories must already be aligned to the same reference structure.
        trajectory file names will be used to name corresponding outputs.
    
    structure: string
        path to input CA structure
    
    selection: string 
        mdanalysis format selection string for the region of the protein you want to perform PCA on.
        if None, PCA will be done on all CA atoms.
    
    selection_align: bool
        If True, the trajectory will be aligned by the selection rather than just doing PCA on the coordinates
        that the input trajectories were originally aligned on.

    output_prepend: string
        prefix to name output folder and files containing data for all inputs

    '''
    
    if output_prepend and output_folder == None:
        output_folder = f'{output_prepend}_cartesian_pca_output'
    elif output_folder == None:
        output_folder = f'cartesian_pca_output'
    
    os.makedirs(output_folder)

    # get the file names from all the input trajectories to use as 
    # record names for the analyses
    ordered_systems = []
    for traj in trajs:
        ordered_systems.append(traj.split("/")[-1].split('.')[0])

    u = mda.Universe(structure, trajs)
    
    # if you're choosing to do PCA on a selection and want the additional alignment based on the selection
    # that happens here 
    if selection and selection_align=True:
        ref = mda.Universe(structure)   
        alignment = align.AlignTraj(u, ref, select=selection, filename=f'{output_folder}/selection_aligned_traj.xtc')
        alignment.run()
        # continue based on the new aligned trajectory
        u = mda.Universe(structure, f'{output_folder}/selection_aligned_traj.xtc')
    
    # trajectories are expected to be only CA atoms
    # ca is used for the transformation
    if selection:
        selection = f'name CA and ({selection})'
        ca = u.select_atoms(f'name CA and ({selection})')
    else:
        selection = 'name CA'
        ca = u.select_atoms('name CA')
    # trajectory was either just aligned or individual trajectories are expected to already be aligned 
    # to same reference structure so here align=False
    pc = pca.PCA(u, select=selection,
             align=False, mean=None,
             n_components=None).run()


    