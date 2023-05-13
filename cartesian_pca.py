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
        CA trajectory will have.  If None, the structure will be used as align_ref.  
        align_ref is necessary if the starting structures for different combined pca systems are different or do not 
        perfectly align. In this case use write_ca_structure from one of the systems and supply that as align_ref for 
        all the combined pca systems.

    split_subunits: bool
        TODO add function to separate the trajectory into individual subunits and then concatenate them.
    '''
    
    u = mda.Universe(structure, traj)

    # deal with residue number selections
    if resnums:
        selections = mda_selection_from_list(resnums)
        selection = f'name CA and ({selections})'
        CAs = u.select_atoms(selection)
    else:
        selection = 'name CA'
        CAs = u.select_atoms(selection)

    # check that output is xtc format
    if not output.endswith('.xtc'):
        print('output file extension needs to be xtc', flush=True)
        return
    if frame_end==None:
        frame_end = len(u.trajectory)

    if align_ref:
        # should be using a CA only structure here that works for all systems in the combined PCA
        ref = mda.Universe(align_ref)   
        alignment = align.AlignTraj(u, ref, select=selection)
        alignment.run()
    else:
        align_ref = structure
        alignment = align.AlignTraj(u, ref, select=selection)
        alignment.run()
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
    # if you're providing a name, this will error out if you've already produced output with it
    if output_prepend and output_folder == None:
        output_folder = f'{output_prepend}_cartesian_pca_output'

    # if there is already the default path name, append an id
    elif os.path.exists('cartesian_pca_output-1'):
        existing_folder_ids = []
        for item in os.listdir('./'):
            split_item = item.split('-')
            if len(split_item) > 1 and split_item[0] == 'cartesian_pca_output':
                existing_folder_ids.append(int(split_item[1]))
        last_output_id = max(existing_folder_ids)
        output_folder = f'cartesian_pca_output-{last_output_id+1}'
    elif output_folder == None:
        output_folder = f'cartesian_pca_output-1'
    
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

    # write the selection to a file and save it in the output directory so you know what the pca was done on
    with open(f'{output_folder}/pca_atom_selection.txt','w') as f:
        f.write(selection)
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

    pc_object: MDAnalysis PCA object
        Output from MDAnalysis.analysis.pca.PCA
        https://docs.mdanalysis.org/stable/documentation_pages/analysis/pca.html

    ca_sel: MDAnalysis atom group
        The selection (Assumed to be Calphas) corresponding to the atoms that pca was done on
    
    structure: string
        Path to structure file corresponding to the CA trajectory 
    
    trajectories: list
        List of paths to trajectory files used in the (combined) pca. Must be in the same order that 
        the combined trajectory was in.

    original_structure: list
        Path to the all atom structures. 

    original_trajectories: list
        Path to all atom trajectories.

    start_offsets: list of int
        List of the frame numbers of the all atom trajectories that the corresponding CA trajectories start at
        (assumed to be the first frame). If supplied, an integer value is required for each trajectory.

    original_traj_strid: list of int
        If your CA trajectories were taken at intervals (might be done if the trajectories are really large), supply the strides.
        If supplied, an integer value is required for each trajectory.
    
    '''
    def __init__(self,pc_object, ca_selection, structure, trajectories, 
                 original_structure=None,original_trajectories=None,
                 start_offsets=None,original_traj_stride=None,
                 ):
        self.pc = pc_object
        self.ca = ca_selection
        self.transformed = pc_object.transform(ca_selection)
        self.structure = structure
        self.trajectories = trajectories
        self.original_structure = original_structure
        self.original_trajectories = original_trajectories
        self.system_traj_paths = get_traj_names(trajectories)
        self.ordered_systems = list(self.system_traj_paths.keys())
        self.average_projections = None
        self.average_rmsf = None


    ### Track the frame numbers corresponding to the length of the combined trajectory
    #### and the corresponding frame numbers (to get matching CA conformation) in original all-atom traj
    ### This is so you can do combined PCA on edited (shorter) trajectories but still go back and find
    ### The frame in the original traj

        self.u = mda.Universe(self.structure, self.trajectories)

        # number of frames per system
        self.n_frames = int(len(self.u.trajectory)/len(self.ordered_systems))
        self.system_frames = {self.ordered_systems[i]:(i*self.n_frames,i*(self.n_frames)+self.n_frames) for i in range(len(self.ordered_systems)) }
        
        if start_offsets = None:
            self.start_offsets = {self.ordered_systems[i]: 0 for i in range(len(self.ordered_systems))}
        else:
            self.start_offsets = {self.ordered_systems[i]: start_offsets[i] for i in start_offsets}

        #TODO Add a function to align the CA residues (assuming resids match) and if it's beyond a tolerance, return a warning
        if original_traj_stride = None:
            self.original_traj_stride = {self.ordered_systems[i]: 1 for i in range(len(self.ordered_systems))}
        else:
            self.original_traj_stride = {self.ordered_systems[i]: i for i in original_traj_stride}
    
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
        Get the coordinates projected onto a principal component.
        component is the canonical principal component ID rather than 0 based index
        i.e. enter 1 if you want PC1 (0 will not return anything)

        Returns
        -------
        array (n_frames, n_atoms, 3)
        
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

    def get_average_projections(self,n_components=10):
        '''
        Find the mean of the transformed column components that correspond to the frames of each 
        system's trajectories
        
        n_components: int
            Number of components to calculate the mean projection for.
            
        Returns
        -------
        Dictionary of system names and corresponding means
        '''

        #TODO check if self.average projections exists and if n_components is less than len(list(self.proj.keys())) then
        # don't run this elif n_compontns is greater do range(len(keys),n_components)

        if self.average_projections.keys() == None:

            self.average_projections = {self.ordered_systems[i]:[] for i in self.ordered_systems}
            start = 0
        
        else:
             # check current length of a dictionary entry to see where to start
            check = list(self.average_projections.keys())[0]
            start = len(self.average_projections[check])

        if self.pc.p_components.shape[0] < n_components:
            n_components = self.pc.p_components.shape[0]
            if n_components <= start:
                print('Too few components.')
                return

        for component in range(start,n_components):
            for system in self.average_projections.keys():

            self.average_projections[system].append(
                    self.transformed[self.system_frames[system][0]:self.system_frames[system][1],component].mean()
                )
        
    
    def get_average_rmsf(self,n_components):

        '''
        Get the average rmsf on n_components for each system.
        This is the mean deviation from the mean projection.  If get_average_projections must be called first.
        
        n_components: int
            Number of components to calculate the mean rmsf for.
        '''

        # n_components

        if self.average_rmsf.keys() == None:

            self.average_rmsf = {self.ordered_systems[i]:[] for i in self.ordered_systems}
            start = 0

        else:
             # check current length of a dictionary entry to see where to start
            check = list(self.average_rmsf.keys())[0]
            start = len(self.average_rmsf[check])

        if self.pc.p_components.shape[0] < n_components:
            n_components = self.pc.p_components.shape[0]
            if n_components <= start:
                print('Too few components.')
                return


        for component in range(start,n_components):
            for system in self.average_rmsf.keys():


                self.average_rmsf[system].append(
                        np.abs(self.transformed[self.system_frames[system][0]:self.system_frames[system][1],component]-
                                   self.average_projections[system][component]).mean()
                     )
            
       
    def get_rmsf_per_residue(self,component):

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

def get_extreme_projections(self, n_components, output_path):
    '''
    Find the minimum and maximum projection on each of n_components.
    This will be the most apparent description of the dynamic captured by the PC.
    
    '''

 
    for component in range(n_components):
        projection_min = np.where(self.transformed[:,component] == self.transformed[:,component].min())
        projection_max = np.where(self.transformed[:,component] == self.transformed[:,component].max())

        # get_projection is expecting the 1 indexed pc id
        coordinates = self.get_projection(component+1)

        pu = mda.Merge(self.ca)
        pu.load_new(coordinates[projection_min])
        proj_calpha = pu.select_atoms('name CA')
        proj_calpha.write(f'{output_path}/pc_{component+1}_min_projected_structure.pdb')

        pu = mda.Merge(self.ca)
        pu.load_new(coordinates[projection_max])
        proj_calpha = pu.select_atoms('name CA')
        proj_calpha.write(f'{output_path}/pc_{component+1}_max_projected_structure.pdb')


def get_original_extremes(component,output_path):
    '''
    Find the frames in the original trajectories that correspond to the extreme structures for each system

    '''

    # get frames of concatenated trajectory where each construct is at its min or max region of the eigenvector
    
    min_frames = {system:None for system in self.ordered_systems}
    max_frames = {system:None for system in self.ordered_systems}

    component = component-1
        
    for system in self.system_frames:
        start = self.system_frames[system][0]
        end = self.system_frames[system][1]

        projection_min = np.where(self.transformed[:,component] == self.transformed[start:end,component].min())[0][0]
        min_frames[system] = projection_min
        projection_max = np.where(self.transformed[:,component] == self.transformed[start:end,component].max())[0][0]
        max_frames[system] = projection_max

    #local = '../from_pronto/trajectories_3'
    #output_dir = f'../cartesian_pca/results/structures/extreme_structures/{subset_ids}'
    #for system, concatenated_frame_indices in system_frames.items():

    #TODO add checks to make sure number of items in input lists (original trajs/ca trajs) match
    # might change to dictionary inputs expecting standard names for the keys
    for i in range(len(self.ordered_systems)):

        
        system = self.ordered_systems[i]
        structure = self.original_structures[i]
        traj = self.original_trajectories[i]
        offset = self.start_offsets[i]
        stride = self.original_traj_stride[i]
                
        ou = mda.Universe(structure, traj)

        # minimum
        # frame index in concatenated trajectory minus the frame number where the given system begins in the concatenated traj
        adjusted_frame = min_frames[system] - self.system_frames[system][0]
        # plus the frame number where the ca traj start in relation to the beginning of the original traj (0 by default)
        # this is useful if you didn't include early frames in the original traj due to them not being equilibrated
        adjusted_frame = adjusted_frame + offset
        # multiplied by a stride interval (1 by default)
        adjusted_frame = adjusted_frame * stride
        #TODO print the frame numbers
        ou.trajectory[adjusted_frame]
        selection = u.select_atoms('all')

        output = f'{output_path}/{system}/{component+1}'
        if os.path.exists(output):
            pass
        else:
            os.makedirs(output)
        with mda.Writer(f'{output}/{system}_{subsel_name}_pc{component+1}_min.pdb',selection.n_atoms) as w:
            w.write(selection)

        # maximum
        adjusted_frame = max_frames[system] - self.system_frames[system][0]
        adjusted_frame = adjusted_frame + offset
        adjusted_frame = adjusted_frame * stride
        ou.trajectory[adjusted_frame]
        selection = u.select_atoms('all')
        with mda.Writer(f'{output}/{system}_{subsel_name}_pc{component+1}_max.pdb',selection.n_atoms) as w:
            w.write(selection)


