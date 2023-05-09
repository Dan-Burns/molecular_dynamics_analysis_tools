
from openff.toolkit.topology import *
import openmmtools as omt
from openmmtools.integrators import *
from openforcefields import *

import MDAnalysis as mda
from MDAnalysis.analysis.distances import dist
import numpy as np


def minimize_morph(structure, reference, output):
    '''
    force a structure into the backbone conformation of another reference structure.
    Useful for homologs where you might have an experimental structure and then a modeled structure with a different sequence.
    If you think the modelled structure is more likely to take the conformation of the experimental structure,
    you can make the morph the modelled structure into the backbone conformation of the experimental structure.
    
    '''

    ### NOT TESTED, Just ported it over from a notebook 

    structure = mda.Universe(structure)

    reference = mda.Universe(structure)

    # identical number of residues?
    n_residues = int(structure.residues.n_residues)
    reference.residues.n_residues == structure.residues.n_residues

    # make a ndarray to populate with the distances
    distance_array = np.zeros((n_residues, n_residues))

    # How far do you want to search for neighboring C alphas?
    cutoff = 10
    # Go through all the atoms and find the CAs and the neighboring CAs
    for atom in u_aws.atoms:
        if atom.name == 'CA':
            neighbors = structure.select_atoms(f'around {cutoff} index {atom.ix} and name CA')
            for neighbor_atom in neighbors.atoms:
                
                if neighbor_atom.name == 'CA' and neighbor_atom.ix != atom.ix:
                
                    distance_array[atom.residue.ix][neighbor_atom.residue.ix] = np.linalg.norm(
                                                                atom.position-neighbor_atom.position)
    
    # get the equilibrium distances for the restraint
    r0_distances = distance_array[distance_array>4]

    # get the residues to apply potential to
    residue_indices = np.argwhere(distance_array>4)

    ###############################################################################
    ## Simulation Set Up - can use the minimize sidechains function above
    # use implicit solvent since we're just minimizing and forcing into conformation
    pdb = PDBFile(structure)
    # have to adjust this for implicit solvent 
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    modeller = Modeller(pdb.topology, pdb.positions)
   
    modeller.addSolvent(forcefield, model='tip3p', padding=1*nanometer, ionicStrength=0.1*molar )
    temperature = 303.15*kelvin
    integrator = LangevinMiddleIntegrator(temperature, 2/picosecond, 0.002*picoseconds)
    pressure = 1 * bar
    system = forcefield.createSystem(modeller.topology,nonbondedMethod=PME, nonbondedCutoff=1*nanometer,
            constraints=HBonds)
    system.addForce(MonteCarloBarostat(pressure, temperature))
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    #################################################################################

    ########### Restraints ################
    # map the residue indices to their CA atom indices
    CA_res_atom_indices = {}
    for atom in list(modeller.topology.atoms()):
        if atom.name == 'CA':
            CA_res_atom_indices[atom.residue.index] = atom.index

    # restraint_force[0].setBondParameter

    restraint_weight = 10000 * openmm.unit.kilocalories_per_mole / openmm.unit.angstrom ** 2
    restraint_force = CustomBondForce("0.5*K*(r-r0)^2")
    # Add the restraint weight as a global parameter in kcal/mol/nm^2
    restraint_force.addGlobalParameter("K", restraint_weight)
    # Define the target xyz coords for the restraint as per-atom (per-particle) parameters
    restraint_force.addPerBondParameter("r0")
    custom_forces ={}
    for i, pair in enumerate(residue_indices):
        
        restraint_force.addBond(CA_res_atom_indices[pair[0]],CA_res_atom_indices[pair[1]], [r0_distances[i]])
    custom_forces['distance_restraints']=system.addForce(restraint_force)
    # necessary?
    simulation.context.reinitialize(preserveState=True)


    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()
    state = simulation.context.getState(getPositions=True)

    with open(output, 'w') as f:
        PDBFile.writeFile(modeller.topology, state.positions, f)