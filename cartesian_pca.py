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




def get_traj_names(trajs):
    # get the file names from all the input trajectories to use as 
    # record names for the analyses
    # could be in the form of a dictionary of system name: path to traj
    ordered_systems = {}

    for traj in trajs:
        ordered_systems[(traj.split("/")[-1].split('.')[0])] = traj
    return ordered_systems

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
    
    # right now nothing is going here if you're not outputting a new aligned traj
    os.makedirs(output_folder)

    ordered_systems = get_traj_names(trajs)

    u = mda.Universe(structure, trajs)
    
    # if you're choosing to do PCA on a selection and want the additional alignment based on the selection
    # that happens here 
    if selection and selection_align==True:
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
    
    # return the pca object and the ca selection
    # TODO return ca structure or can just generate it separately

    # use this outside of the function to transformed = pc.transform(ca)
    return pc, ca

def get_traj_lengths(structure, trajs):
    '''
    find the shortest trajectory to determine the length to make each traj in the concatenated trajectory
    Parameters
    ----------
    structure: string
        path to C-alpha structure file that corresponds to the number of atoms in all the trajectories in trajs

    trajs: list
        list of paths to trajectories
    '''
    traj_names = get_traj_names(trajs)

    traj_lengths = {}
    for traj in traj_names:
        u = mda.Universe(structure, traj_names[traj])
        traj_lengths[traj] = len(u)

    return traj_lengths



 # get the trajectory that has the minimum length 
 # min(d, key=d.get)


class Combined_PCA:
    '''
    Should make this into a class
    transform the pc on given components
    calculate average projections
    average rmsf
    rmsf per residue
    return min and max structures 
    return rmsf color per residue pymol selections
    etc.

    
    '''
    def __init__(self,pc_object, ca_selection, structure, trajectories, 
                 original_structure=None,original_trajectories=None,
                 start_offsets=None):
        self.pc = pc_object
        self.ca = ca_selection
        self.transformed = pc_object.transform(ca_selection)
        self.structure = structure
        self.trajectories = trajectories
        self.original_structure = original_structure
        self.original_trajectories = original_trajectories
        self.system_traj_paths = get_traj_names(trajectories)
        self.ordered_systems = list(self.system_traj_paths.keys())


    ### Track the frame numbers corresponding to the length of the combined trajectory
    #### and the corresponding frame numbers (to get matching CA conformation) in original all-atom traj
    ### This is so you can do combined PCA on edited (shorter) trajectories but still go back and find
    ### The frame in the original traj

        self.u = mda.Universe(self.structure, self.trajectories)

        # number of frames per system
        self.n_frames = int(len(self.u.trajectory)/len(self.ordered_systems))
        self.system_frames = {self.ordered_systems[i]:(i*self.n_frames,i*(self.n_frames)+self.n_frames) for i in range(len(self.ordered_systems)) }
        #TODO
        # add offset list -input the frame number of the original traj that corresponds to the first frame of the ca traj
        # add additional stride option if the ca traj is produced with a stride so you can find the correct frames
        # in original traj
    #principal_components = [i for i in range(10)]
    #mean_rmsfs = {name:[] for name in ordered_systems}
    #system_average_projections = {name:[] for name in ordered_systems}
    #rmsf = []
    #averaged_projection = []




    def get_projection(self,component):
        '''
        get a projection 
        component is the canonical principal component ID rather than 0 based index
        i.e. enter 1 if you want PC1 (0 will not return anything)
        
        '''
        component = component-1

        pcn = self.pc.p_components[:, component]
        # Take the ca selection transform = #n frames
        trans = self.transformed[:, component]
        # project the frames onto the principal component (mean centers the motion)
        projected = np.outer(trans, pcn) + self.pc.mean.flatten()
        # reshape the projection so that you have n_frames X n_atoms X 3 dimensions
        coordinates = projected.reshape(len(trans), -1, 3)
        return coordinates

    def get_average_projection(self,component,system):
        '''
        Find the mean of the transformed column components that correspond to the frames of one of the 
        system's trajectories
        
        '''
        # should do n_components and automatically do for all systems

        component = component -1

        # This should get saved to an attribute
        return self.transformed[self.system_frames[system][0]:self.system_frames[system][1],component].mean()
    
    def get_average_rmsf(self,component,system):

        # n_components

        component = component-1


        return np.abs(self.transformed[self.system_frames[system][0]:self.system_frames[system][1],component]-
                                   self.system_average_projections[system][component]).mean()
        
       
       
    def get_rmsf_per_residue(self,component,system):

        #component = component-1

        # Take the first principal component
       # principal_component = self.pc.p_components[:, component]
        # Take the ca selection transform = n_frames
        #transformation = self.transformed[:, component]
        #projection = np.outer(transformation, principal_component) + self.pc.mean.flatten()
        # reshape the projection so that you have n_frames X n_atoms X 3 dimensions
        #coordinates = projection.reshape(len(transformation), -1, 3)

        rmsfs = {}
        # add option to do it for all systems

        coordinates = self.get_projection(component)

        for system in self.system_frames:
            start,end = self.system_frames[system][0], self.system_frames[system][1]

            # create a projected universe
            pu = mda.Merge(self.ca)
            # load frames for just the system of interest
            pu.load_new(coordinates[start:end,:,:])
            #rmsf
            calphas = pu.select_atoms('name CA')
            rmsfer = RMSF(calphas, verbose=True).run()
            
            rmsfs[system] = rmsfer.rmsf
        
        return rmsfs



            