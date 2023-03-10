from Bio.PDB import PDBParser
import MDAnalysis.analysis.encore as encore
import MDAnalysis as mda

#@markdown Set maximum number of clusters to generate.
n_clusters = 5 #@param {type:"number"}

pdb   = 'template.pdb'
model = PDBParser().get_structure('structure',pdb)

# hold chain IDs and residue object lists
# use to compare the subunits and figure out if they're identical or not
chains = {}
for chain in model.get_chains():
  chains[chain.id] = chain.get_unpacked_list()

trajectory = 'movie.dcd'
structure = 'template.pdb'
u = mda.Universe(structure, trajectory)

#############################

# make a dictionary of chain/subunit keys with atom selection values
### Need to decide how to handle homo-multimers vs multimers
selections = {}
for chain in chains.keys():
  selections[chain] = u.select_atoms('segid '+chain)

# make a directory to hold the seperated subunits
out_dir = 'seperated_trajectories/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
# clean the directory if it already exists
for f in os.listdir(out_dir):
    os.remove(out_dir+f)

# Instead of writing the seperated trajectories to files, 
# can probably use u.merge() - but people might want to download trajectories...
for chain, selection in selections.items():
    with mda.Writer(out_dir+'chain_'+chain+'.dcd', selection.n_atoms) as W:
        for ts in u.trajectory:
            W.write(selection)
# write one subunit to a pdb
# can probably just use the selection instead of a saved pdb
# assuming we're just working with EI dimer for this
selections[list(selections.keys())[0]].write('one_subunit.pdb')

# align
from MDAnalysis.analysis import align
seperated_trajs = [out_dir+traj for traj in os.listdir(out_dir) if traj.endswith('dcd')]
new_pdb = 'one_subunit.pdb'
ref = mda.Universe(new_pdb)
sep_u = mda.Universe(new_pdb, seperated_trajs)
align.AlignTraj(sep_u, ref,select='name CA',filename='aligned_seperated_subunits.dcd').run()

for cluster_iteration in range(n_clusters):
    structure = 'one_subunit.pdb'
    trajectory = 'aligned_seperated_subunits.dcd'
    u = mda.Universe(structure, trajectory)
    # add n_jobs argument so that all n_init are run in parallel (default is 10)
    ensemble = encore.cluster(u, method=encore.clustering.ClusteringMethod.KMeans(n_clusters=cluster_iteration+1))
    #####################################
    # change this selection from 'name CA' to 'protein' when using structure file with 
    # correct residue names
    #####################################
    selection = u.select_atoms('protein')
    ######################################
    clusters = []
    for i, cluster in enumerate(ensemble.clusters):
        u.trajectory[cluster.centroid]
        if not os.path.exists(f'cluster_{cluster_iteration+1}/'):
          os.makedirs(f'cluster_{cluster_iteration+1}/')
        selection.write(f'cluster_{cluster_iteration+1}/centroid_{i+1}.pdb')
        clusters.append(int(cluster.centroid))