
from MDAnalysis.analysis.rms import rmsd

def get_rmsd(universe, stride=1, selections=None):

    # Should just use mda 2's RMSD.run() function

    u = universe

    if selections is None:

    selections['protein'] = u.select_atoms('protein')
 

    u.trajectory[0]   # rewind trajectory
    xref0 = dict((name, g.positions - g.center_of_mass()) for name, g in selections.items())

    nframes = len(u.trajectory[::stride])
    results = dict((name, np.zeros((nframes, 2), dtype=np.float64)) for name in selections)

    for iframe, ts in enumerate(u.trajectory[::stride]):
        for name, g in selections.items():
            results[name][iframe, :] = (u.trajectory.time,
                                        rmsd(g.positions, xref0[name],
                                            center=True, superposition=True))
   
   return results
        
   