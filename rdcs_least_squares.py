#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 18:53:59 2022
Port of Tim Egner's matlab script for fitting RDCs from Venditti Lab ISU
Contributors:
Divyanshu Shukla 
Dan Burns 

"""
import math
import numpy as np
import os
from scipy.optimize import least_squares
import biopandas
from biopandas.pdb import PandasPdb
from scipy.optimize import least_squares



def pdbs_to_df(pdb_folder):
    # dictionary of dictionarys.  Outer dictionary is index for each conformer,
    # inner dictionary for each pdb, keys are resids, values are lists of Nx,Ny,Nz,Hx,Hy,Hz
    # This will be converted to a dictionary of dataframes (or possibly multiindex dataframe later)
    ensemble_coordinates={}
    txt_files = [f'{pdb_folder}/{f}' for f in os.listdir(pdb_folder) if f.endswith('.pdb')]
    for pdb_num, pdb_file in enumerate(txt_files):
        # This won't deal with HETATM entries atm, you have to separately read the 'HETATM' record...
        # Needs to be addressed by checking contents of 'HETATM' or opening the RDC file first and seeing
        # what resids you need
        ppdb = PandasPdb().read_pdb(pdb_file)
        ensemble_coordinates[pdb_num] = ppdb.df['ATOM'].loc[(ppdb.df['ATOM']['atom_name'] == 'N') | (ppdb.df['ATOM']['atom_name'] == 'H')][
                                    ['residue_number','atom_name','x_coord','y_coord','z_coord']]
        
    return ensemble_coordinates
    

def get_rdc_data(rdc_file):
    '''
    this is only dealing with two column format
    with no error.  Can replace with functions from
    rdcs.py
    '''
    rdc_values=[]
    residue_ids=[]
    with open(rdc_file,'r') as f:
        lines=f.readlines()
        for i in lines:
            # Need to deal with error column
            # can replace this with the tools for paramagpy
            temp=i.split("\t")
            residue_id,rdc=int(temp[0]),float(temp[1])
            rdc_values.append(rdc)
            residue_ids.append(residue_id)
    return np.array(residue_ids), np.array(rdc_values)


def combine_resids_coordinates(residue_ids, ensemble_coordinates):
    '''
    Returns 2 dictionaries of dictionaries, one for
    N residue coordinates and on for H residues.
    Outer dictionary is id of pdb structure,
    inner is resid keys and values of xyz coordinate np arrays.
    Only resids that have RDCs will be here.
    '''
    N_coord_dict = {i:{}for i in range(len(ensemble_coordinates))}
    H_coord_dict = {i:{}for i in range(len(ensemble_coordinates))}

    # Go through each pdb's data and get just the N and H lines corresponding to the residues with RDCs
    #will have an np array of shape n_structures X n_rdcs X n_coordinates for both N and H's.
    for s in range(len(ensemble_coordinates)):
        for i, rdc_resid in enumerate(residue_ids):
        
            N_coord_dict[s][rdc_resid] = ensemble_coordinates[s][['x_coord','y_coord','z_coord']].loc[(ensemble_coordinates[s]['atom_name']=='N') & 
                                                                        (ensemble_coordinates[s]['residue_number']==rdc_resid)].to_numpy()[0]
        
            H_coord_dict[s][rdc_resid] = ensemble_coordinates[s][['x_coord','y_coord','z_coord']].loc[(ensemble_coordinates[s]['atom_name']=='H') & 
                                                                        (ensemble_coordinates[s]['residue_number']==rdc_resid)].to_numpy()[0]
    return N_coord_dict, H_coord_dict



def rdc_residuals(par0):
    par0 = np.reshape(par0,(-1,5))
    rdc_calc_results = {i:[] for i in range(par0.shape[0])}

    for j in range(par0.shape[0]):
        a=par0[j][0]
        b=par0[j][1]
        c=par0[j][2]
        azz=par0[j][3]
        r=par0[j][4]
        dmax = 22000
        Da = 3*dmax*azz/4
        x = np.array([np.cos(a)*np.cos(b),np.cos(a)*np.sin(b)*np.sin(c)-np.sin(a)*np.cos(c),np.cos(a)*np.sin(b)*np.cos(c)+np.sin(a)*np.sin(c)
                            ])
        y = np.array([np.sin(a)*np.cos(b),np.sin(a)*np.sin(b)*np.sin(c)+np.cos(a)*np.cos(c),np.sin(a)*np.sin(b)*np.cos(c)-np.cos(a)*np.sin(c)
                        ])
        z = np.array([-np.sin(b),np.cos(b)*np.sin(c),np.cos(b)*np.cos(c)])
        for resid in rdc_resids:
            N_coords = N_coord_dict[j][resid]

            H_coords = H_coord_dict[j][resid]

            NH=np.subtract(H_coords,N_coords)
            theta=np.arccos(np.dot(z,NH)/np.sqrt(np.sum(np.square(z))*np.sum(np.square(NH))))
            NHxy=np.subtract(NH,[z[0]*NH[0],z[1]*NH[1],z[2]*NH[2]])
            phi=np.arccos(np.dot(x,NHxy)/np.sqrt(np.sum(np.square(x))*np.sum(np.square(NHxy))))
            rdc_calc_results[j].append(Da*((3*np.square(np.cos(theta))-1)+3*r*(np.square(np.sin(theta)))*np.cos(2*phi)/2))

    rdc_sums = np.sum(np.array([np.array(rdc_calc_results[i]) for i in rdc_calc_results]),axis=0)
    return rdc_sums - actual_rdcs

def rdc_calc(par0, N_coord_dict, H_coord_dict, rdc_resids, actual_rdcs):
    par0 = np.reshape(par0,(-1,5))
    rdc_calc_results = {i:[] for i in range(par0.shape[0])}

    for j in range(par0.shape[0]):
        a=par0[j][0]
        b=par0[j][1]
        c=par0[j][2]
        azz=par0[j][3]
        r=par0[j][4]
        dmax = 22000
        Da = 3*dmax*azz/4
        x = np.array([np.cos(a)*np.cos(b),np.cos(a)*np.sin(b)*np.sin(c)-np.sin(a)*np.cos(c),np.cos(a)*np.sin(b)*np.cos(c)+np.sin(a)*np.sin(c)
                            ])
        y = np.array([np.sin(a)*np.cos(b),np.sin(a)*np.sin(b)*np.sin(c)+np.cos(a)*np.cos(c),np.sin(a)*np.sin(b)*np.cos(c)-np.cos(a)*np.sin(c)
                        ])
        z = np.array([-np.sin(b),np.cos(b)*np.sin(c),np.cos(b)*np.cos(c)])
        for resid in rdc_resids:
            N_coords = N_coord_dict[j][resid]

            H_coords = H_coord_dict[j][resid]

            NH=np.subtract(H_coords,N_coords)
            theta=np.arccos(np.dot(z,NH)/np.sqrt(np.sum(np.square(z))*np.sum(np.square(NH))))
            NHxy=np.subtract(NH,[z[0]*NH[0],z[1]*NH[1],z[2]*NH[2]])
            phi=np.arccos(np.dot(x,NHxy)/np.sqrt(np.sum(np.square(x))*np.sum(np.square(NHxy))))
            rdc_calc_results[j].append(Da*((3*np.square(np.cos(theta))-1)+3*r*(np.square(np.sin(theta)))*np.cos(2*phi)/2))

    rdc_sums = np.sum(np.array([np.array(rdc_calc_results[i]) for i in rdc_calc_results]),axis=0)
    return rdc_sums





def rdc_residuals(par0, N_coord_dict, H_coord_dict, rdc_resids, actual_rdcs):
    par0 = np.reshape(par0,(-1,5))
    rdc_calc_results = {i:[] for i in range(par0.shape[0])}

    for j in range(par0.shape[0]):
        a=par0[j][0]
        b=par0[j][1]
        c=par0[j][2]
        azz=par0[j][3]
        r=par0[j][4]
        dmax = 22000
        Da = 3*dmax*azz/4
        x = np.array([np.cos(a)*np.cos(b),np.cos(a)*np.sin(b)*np.sin(c)-np.sin(a)*np.cos(c),np.cos(a)*np.sin(b)*np.cos(c)+np.sin(a)*np.sin(c)
                            ])
        y = np.array([np.sin(a)*np.cos(b),np.sin(a)*np.sin(b)*np.sin(c)+np.cos(a)*np.cos(c),np.sin(a)*np.sin(b)*np.cos(c)-np.cos(a)*np.sin(c)
                        ])
        z = np.array([-np.sin(b),np.cos(b)*np.sin(c),np.cos(b)*np.cos(c)])
        for resid in rdc_resids:
            N_coords = N_coord_dict[j][resid]

            H_coords = H_coord_dict[j][resid]

            NH=np.subtract(H_coords,N_coords)
            theta=np.arccos(np.dot(z,NH)/np.sqrt(np.sum(np.square(z))*np.sum(np.square(NH))))
            NHxy=np.subtract(NH,[z[0]*NH[0],z[1]*NH[1],z[2]*NH[2]])
            phi=np.arccos(np.dot(x,NHxy)/np.sqrt(np.sum(np.square(x))*np.sum(np.square(NHxy))))
            rdc_calc_results[j].append(Da*((3*np.square(np.cos(theta))-1)+3*r*(np.square(np.sin(theta)))*np.cos(2*phi)/2))

    rdc_sums = np.sum(np.array([np.array(rdc_calc_results[i]) for i in rdc_calc_results]),axis=0)
    return rdc_sums - actual_rdcs

class FitRDCs:
    '''
    Class takes a directory to pdbs and an rdc file and perfors
    a least squares fit.
    '''
    def __init__(self,
                 pdb_folder,
                 rdc_file,

            ):
        self.ensemble_coordinates = pdbs_to_df(pdb_folder)
        self.residue_ids, self.rdc_values = get_rdc_data(rdc_file)
        self.N_coordinate_dict, self.H_coordinate_dict = combine_resids_coordinates(
                                        self.residue_ids, self.ensemble_coordinates)
        self.par0 = np.array([math.pi,  math.pi, math.pi,   0.01, 0.1])
        self.lb = np.array([  0,    0,   0,   0, 0])
        self.ub = np.array([4*math.pi, 4*math.pi, 4*math.pi, np.inf, 0.66667])
        self.nIter = 10000;       
        self.tolerance = 10**-9

        for x in range(1, len(self.ensemble_coordinates)):
            typ = np.array([math.pi,  math.pi,  math.pi,   0.01, 0.1])
            typL = np.array([  0,    0,   0,   0, 0])
            typU = np.array([2*math.pi, 2*math.pi, 2*math.pi, np.inf, 0.66667])
            self.par0=np.vstack((self.par0,typ))
            self.lb=np.vstack((self.lb,typL))
            self.ub=np.vstack((self.ub,typU))

        self.rdc_calc_args = (self.N_coordinate_dict,self.H_coordinate_dict,self.residue_ids,self.rdc_values)
    
    def fit_rdcs(self):

        self.jac0=least_squares(rdc_residuals,self.par0.flatten(),bounds=(self.lb.flatten(),self.ub.flatten()),
                ftol=10**-5,xtol=10**-7,x_scale='jac', args=self.rdc_calc_args)
        self.par=self.jac0.x
        self.jacobian = self.jac0.jac

        self.jac=least_squares(rdc_residuals,self.par,bounds=(self.lb.flatten(),self.ub.flatten()),
                ftol=tolerance,x_scale='jac',
                max_nfev=self.nIter,xtol=sel.tolerance,jac_sparsity=self.jacobian,args=self.rdc_calc_args)
    
        self.fitted_rdcs = rdc_calc(self.jac.x, self.N_coord_dict, self.H_coord_dict, self.rdc_resids, self.rdc_values)

        self.residuals =  self.fitted_rdcs - self.rdc_values

        self.Rfactor = np.sqrt(np.sum(np.square(self.residuals))/(
                        2*np.sum(np.square(self.rdc_values))))