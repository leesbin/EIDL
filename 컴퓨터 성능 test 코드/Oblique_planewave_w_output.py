import math
import numpy as np
import meep as mp
import matplotlib.pyplot as plt
import os
import h5py

def run_sim(rot_angle=0) :

    resolution = 100
    Lx=10.0 * 4
    Ly=5.0 * 4

    cell_size=mp.Vector3(Lx,Ly)
    dpml = 0.5  # PML thickness

    Air = mp.Medium(index=1.0)
    pml_layers = [mp.PML(thickness=dpml, direction=mp.Y)]

    fsrc = 1.0  # frequency of planewave (wavelength = 1/fsrc)

    k_point = mp.Vector3(fsrc).rotate(mp.Vector3(z=1), rot_angle)
    #k_point = mp.Vector3(1,0,0)
    sources = [
            mp.Source(
            mp.GaussianSource(fsrc, fwidth=1.0),
        center=mp.Vector3(0,-2.6,0),
        size=mp.Vector3(x=10),
        component=mp.Ez
        )
 
        ]

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=pml_layers,
        k_point=k_point,
        default_material=Air,
        sources=sources,
    )
  #  sim.run(
  #      until_after_sources=mp.stop_when_dft_decayed
  #  )
    Output=sim.add_dft_fields(
        [mp.Ez],
        fsrc,
        0,
        1,
        center=mp.Vector3(0,0,0),
        size=mp.Vector3(Lx,Ly),
    )

#    sim.run(
#                mp.dft_ldos(1.0,1.0,1,1.0),
#                #*self.step_funcs,
#                until_after_sources=mp.stop_when_dft_decayed
#    )
    sim.run(until=100)
 #   plt.figure(dpi=100)
 #   sim.plot2D(fields=mp.Ez)
 #   plt.show()

    sim.output_dft(Output, "Ez_field")

for rot_angle in np.radians([0]):
    run_sim(rot_angle)
