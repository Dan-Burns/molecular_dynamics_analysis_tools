import mdtraj as md
import numpy as np
import MDAnalysis as mda
import pandas as pd
import sys
import os
import json

# provide the naming scheme for the files
name = sys.argv[1]

structure = f'{name}.pdb'
#trajs = [f'../run_output/{file}' for file in os.listdir('../run_output') if file.endswith('.xtc')]
#trajs.sort()

u = mda.Universe(structure)
# dictionary of residues and atom indices composing those residues
residues = [str(res.resnum) for res in list(u.residues)]
with open(f'{name}_resnums.json','w') as f:
    json.dump(residues,f)

# gotta use chunks with mdtraj if files are big
for i, traj in enumerate(md.iterload(f'{name}.xtc', top=structure, chunk=10000)):
   
    sasa = md.shrake_rupley(traj,mode='residue')

    np.save(f"{name}_sasa_{i}.npy", sasa)


# after all the arrays are saved, stack them and put them into a dataframe to save the final version
files = [file for file in os.listdir('./') if file.endswith('.npy')]
files.sort()
files_to_stack = [np.load(i) for i in files]
stack = np.vstack(files_to_stack)


with open(f'{name}_resnums.json','r') as f:
    resnums = json.load(f)

df = pd.DataFrame(stack,columns=resnums)
df.to_csv(f'{name}_sasa.csv')
# could delete the .npy files after this