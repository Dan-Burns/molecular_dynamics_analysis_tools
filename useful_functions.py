#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 13:08:11 2022

@author: dburns
"""
import pandas as pd
import numpy as np
import Bio
import copy
from parmed import gromacs
import parmed as pmd
from parmed.gromacs import GromacsTopologyFile


def identical_subunits(chains):
  '''
  take dictionary of biopython unpacked chain lists and see if they're
  composed of an identical number of identical residues
  Produce the chain dictionary :
  pdb   = 'template.pdb'
  model = PDBParser().get_structure('structure',pdb)
  chains = {chain.id: chain.get_unpacked_list() for chain in model.get_chains()}
  '''
  # Get just the resnames (not the whole biopython res object)
  res_dict = {chain: [res.resname for res in chains[chain]] 
                      for chain in chains.keys()}
  # index the chain ids (can't index dictionary keys directly)
  chain_ids = list(chains.keys())
  # if it's only one chain, we're not going to break the trajectory up
  if len(chain_ids) == 1:
    return(False)
  # if it's multiple chains, check if they are all identical
  else:
    test = [res_dict[chain_ids[i]] == res_dict[chain_ids[i+1]]
            for i in range(len(chain_ids)-1)]
  # will have one boolean if everything is True or anything is False
  return test[0]

def rename_df_indices(df, append_string):
    # rename indices for a dataframe
    
    indices = df.index
    new_indices = {}
    for index in indices:
        new_indices[index]=(f'{index}_{append_string}')
    return df.rename(mapper=new_indices)

# Should actually take the min records, turn to np array, boolean > 1 = 1, else = 0 and then make df column of that
def make_binary_labels(data, cutoff, zero_one=True, pos_neg=False, nan_cutoff=None, cutoff_sign='negative'):
    '''
    Makes an np array and converts things below the cutoff to 0 and above to 1
    or -1 and 1 if pos_neg=True.
    if you want -1 turned to NaN then nan_cutoff=True and cutoff_sign='negative'
    '''
  
    ############
    #######NEEDS WORK
    ##############
    a = np.array(data)
    b = np.where((a<cutoff) & (a>0))
    if pos_neg==True:
        a[b] = -1
    else:    
        a[b] = 0

    b = np.where(a > cutoff)
    # turn > 1 into label 1 
    a[b] = 1

    # can't deal with another cutoff sign right now
    # probably just move this part to the beginning and then deal with
    # remaining values.
    if nan_cutoff and cutoff_sign=='negative':
    # indices for negative radius convert to np.nan
        b = np.where(nan_cutoff > a)
        a[b] = np.nan
        return a
    else:
        return a
        
def sort_dictionary_values(dictionary):
    return dict(sorted(dictionary.items(), key=lambda item: item[1]))


def fromdeg(d):
    '''
    Used to convert angles for ml input so you have max positive at 180 deg
    And everything else is in between.
    np.linalg.norm(fromdeg(0) - fromdeg(angle))
    Note for torsions: Convert negative torsions to positive by adding 360 to df[df<0] = df+360 
    '''
    r = np.radians(d)
    return np.array([np.cos(r), np.sin(r)])

def deg_to_norm(d):
    '''
    get the norm of a degree so at 180 = 2 and everything else is in between.
    '''
    
    return np.linalg.norm(fromdeg(0) - fromdeg(d))

def convert_rms_to_rgb(rmsf, color='red'):
    '''
    Take a normalized rmsf value and create a color code along a spectrum of white to red for the corresponding 
    residue to depict the regions of most motion.
    '''
    adjustment = 255 - (int(rmsf*255))

    if color == 'blue':
        r = str(hex(adjustment))[2:]
        if len(r) == 1:
            r='0'+r
        g = str(hex(adjustment))[2:]
        if len(g) == 1:
            g='0'+g
        b = str(hex(255))[2:]
    elif color == 'red':
        r = str(hex(255))[2:]
        g = str(hex(adjustment))[2:]
        if len(g) == 1:
            g='0'+g
        b = str(hex(adjustment))[2:]
        if len(b) == 1:
            b='0'+b
    elif color == 'green':
        # 0, 1, 0
        # forest green 0.2, 0.6, 0.2
        r = str(hex(int((0.2*255)+(adjustment*0.8))))[2:]
        if len(r) == 1:
            r='0'+r
        g = str(hex(int((255*0.6)+adjustment*0.4)))[2:]
        b = str(hex(int((0.2*255)+(adjustment*0.8))))[2:]
        if len(b) == 1:
            b='0'+b
    elif color == 'orange':
        # 1, 0.5, 0
        r = str(hex(255))[2:] 
        g = str(hex(int(127+adjustment/2)))[2:]
        b = str(hex(adjustment))[2:]
        if len(b) == 1:
            b='0'+b

    return '0x'+ r+g+b

def parmed_underscore_topology(gromacs_processed_top, atom_indices, output_top):
    '''
    Add underscores to atom types of selected atoms.
    This is useful if using the plumed_scaled_topologies script 
    for hremd system modification.
    With this, you still need to open the new topology file and delete the 
    underscores from the beginning of the file [atomtypes]
    or else plumed will look for atoms with 2 underscores to apply lambda to.
    '''
    top = GromacsTopologyFile(gromacs_processed_top)

    for atom in top.view[atom_indices].atoms:
        atom.type = f"{atom.type}_"
        if atom.atom_type is not pmd.UnassignedAtomType:
            atom.atom_type = copy.deepcopy(atom.atom_type)
            atom.atom_type.name = f"{atom.atom_type.name}_"


    top.save(output_top)


def combine_selections(selections,start_with_select=True):
    '''
    Given a list of MDAnalysis selections, this will combine them into a long string of selections
    If start_with_select=False, return the selection string without the first word as "start"
    This is useful if the selection goes into an MDA function that doesn't want "select" in it, just the items to be
    selected.
    '''

    if start_with_select == True:
        combined_selection = "select "
    else:
        combined_selection = ""
    for selection in selections:
        if selection != selections[-1]:
            # remove the "selection" word from beginning of string and put back into string format
            edit = " ".join(selection.split()[1:])
            combined_selection += f"({edit}) or "
    # add the final selection without the "or" at the end
    edit = " ".join(selections[-1].split()[1:])
    combined_selection += f"({edit})"
    return combined_selection