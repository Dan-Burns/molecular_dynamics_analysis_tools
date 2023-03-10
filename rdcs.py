#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 18:53:59 2022

@author: dburns
"""
from Bio import PDB
from Bio.PDB import PDBParser
import os
import re
import pandas as pd
import numpy as np
from paramagpy import protein, fit, dataparse, metal

# name of the reformatted rdc file that will be used by a couple functions
formatted_rdc_file = 'formatted_rdcs.txt'

def determine_rdc_file_format(rdc_file):
    # Pandas based function to determine file format
    # make a dataframe with column ids 0,1,2....
    df = pd.read_table(rdc_file, header=None, sep='\s+')
    # create a dictionary of expected datatype categories 
    categories = {'int64':[], 'float64':[], 'object':[]}
    # append the column id to the corresponding datatype
    for column in df.columns:
        for key in categories:
            if df[column].dtype == key:
                categories[key].append(column)
    
    # This will contain the column ids to pass to format_rdc_file
    file_format = {'res':None,'rdc_val':None, 'error':None}
    
    # check that only one column corresponds to integers which will be resid
    try:
        if len(categories['int64']) == 1:
            file_format['res'] = categories['int64'][0]
    except:
        print("Can't identify residue id columns. Make sure only one column\
              contains only integer residue id values.")
    
    # Can try to test and see if any column containing integers 
    # is the same as all the other integer-only columns and assume they're
    # duplicate res_id columns
    '''
    elif len(categories['int64']) > 1:
        if df[column_id] != df[categories['int64'][0]]:
            print('Not able to identify the residue ID column. \n \
                  Make sure the file only has one column containing \
                      residue ID numbers as integer values.')
                          else:
   format['res'] = categories['int64'][0]
   '''

    # If there are two columns containing floats, determine which one has the 
    # larger standard deviation. The larger standard deviation will correspond
    # to the RDC column and the other one should be the error column.
    # If there is only one column containing floats, that will be the RDCs.
    if len(categories['float64']) == 1:
        file_format['rdc_val'] = categories['float64'][0]
    else:
        file_format['rdc_val'] = df[categories['float64']].std().idxmax()
        file_format['error'] = df[categories['float64']].std().idxmin()
                      
    return file_format    
                
        


def format_rdc_file(rdc_file, file_format=None, output_file=formatted_rdc_file
                    ):
    # Open the RDC file and convert it the paramagpy input format
    # 
    with open(rdc_file, 'r') as f:
        with open(output_file, 'w') as g:
            if file_format['error'] == None:
                for line in f.readlines():
                    res, rdc_val = np.array(line.split())[[file_format['res'],
                                                        file_format['rdc_val']]
                                                          ]
                    g.write(f'{res:>7} {"N" : >3} {res :>7} {"H" : >3}\
                        {rdc_val :>10} {"0" :>10}\n')
            else:
                for line in f.readlines():
                    res, rdc_val, error = np.array(line.split())[
                                                        [file_format['res'],
                                                        file_format['rdc_val'],
                                                        file_format['error']]
                                                          ]
                    g.write(f'{res:>7} {"N" : >3} {res :>7} {"H" : >3}\
                        {rdc_val :>10} {error:>10}\n')

def get_pdb_files(folder, extension, file_numbers):
    # make a list of the files and use re split to find the files corresponding
    # to the itertools tuple combo of numbers/ ids
    files = [folder+file for file in os.listdir(folder)\
             if file.endswith(f'{extension}')]
    
    selected_files = []
    for file in files:
        if int(re.split('_|\.',file)[-2]) in file_numbers:
            selected_files.append(file)
    
    return selected_files


def make_multi_pdb_file(output_file, files=None):
    # biopython make multi pdb from list of pdbs
    # files must be a list of integers corresponding to the files 
    # that are to be made into a multi-pdb
    # might be faster to use pdb_tools'  pdb_mkensemble

  
    pdb_io = PDB.PDBIO()
    ms = PDB.Structure.Structure('master')
    
    structures = [PDB.PDBParser().get_structure(str(i),file) for i, \
                  file in enumerate(files) \
                      if file.endswith('pdb')]
    for i, structure in enumerate(structures):
        for model in list(structure):
            new_model=model.copy()
            new_model.id=i
            new_model.serial_num=i+1
            ms.add(new_model)
    
    pdb_io = PDB.PDBIO()
    pdb_io.set_structure(ms)
    pdb_io.save(output_file)


def fit_rdc(formatted_rdc_file, pdb_file, field_strength=18.8, temp=300):
    # Run the rdc fitting calculation with paramagpy
    #https://henryorton.github.io/paramagpy/build/html/examples/rdc_fit.html
    # Load the PDB file
    prot = protein.load_pdb(pdb_file)
    
    # Load the RDC data
    rawdata = dataparse.read_rdc(formatted_rdc_file)
    
    # Associate RDC data with atoms of the PDB
    parsedData = prot.parse(rawdata)

    # define initial tensor
    mStart1 = metal.Metal(B0=field_strength, temperature=temp)
    
    [sol1], [data1] = fit.svd_fit_metal_from_rdc([mStart1], [parsedData], 
                                                 ensembleAverage=True)
    
    return sol1, data1, fit.qfactor(data1, ensembleAverage=True)

#############################################
###### Convert PDB to Matlab format #########
def matlab_pdb_format(input_pdb, output_file):
    parser = PDBParser()
    structure = parser.get_structure('centroid',input_pdb)
    
    with open(output_file,'w') as f:
        for res in structure.get_residues():
        
                
            index = res.get_id()[1]
            atoms = [atom.name for atom in res.get_unpacked_list()]
            if 'N' not in atoms or 'H' not in atoms:
                continue
            for atom in res.get_atoms():
                if atom.name == 'N': 
                    Nx, Ny, Nz = np.around(atom.coord[[0,1,2]],3)
                if atom.name == 'H':
                    Hx, Hy, Hz = np.around(atom.coord[[0,1,2]],3)
            f.write(f'{index} {Nx} {Ny} {Nz}\n{index} {Hx} {Hy} {Hz}\n')


# Take an mdanalysis ClusterCollections object and determine which other
# frames are shared with the best fitting centroids cluster 
# Take those frames and check the euclidean distance between the worst fitting 
# residues and then perform a rdc fit with that structure replacing the 
# centroid structure (after making a larger cluster set and limiting the 
# number of structures to search)
# experimental and predicted values are in the data1 array

'''
# plotting stuff from paramagpy
from matplotlib import pyplot as plt
fig = plt.figure(figsize=(5,10))
ax1 = fig.add_subplot(211)

ax1.set_title('ei')


for sol, ax, data in zip([sol1], [ax1], [data1]):

    # Calculate ensemble averages
    dataEAv = fit.ensemble_average(data)

    # Calculate the Q-factor
    qfac = fit.qfactor(data, ensembleAverage=True)

    # Plot all models
    ax.plot(data['exp'], data['cal'], marker='o', lw=0, ms=2, c='b', 
        alpha=0.5, label="All models: Q = {:5.4f}".format(qfac))

    # Plot the ensemble average
    ax.plot(dataEAv['exp'], dataEAv['cal'], marker='o', lw=0, ms=2, c='r', 
        label="Ensemble Average: Q = {:5.4f}".format(qfac))

    # Plot a diagonal
    l, h = ax.get_xlim()
    ax.plot([l,h],[l,h],'-k',zorder=0)
    ax.set_xlim(l,h)
    ax.set_ylim(l,h)

    # Make axis labels and save figure
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Calculated")
    ax.legend()

fig.tight_layout()

'''

