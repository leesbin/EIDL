'''
SPDX-FileCopyrightText: © Sangbin Lee (kairus0320@gmail.com)
SPDX-License-Identifier: Proprietary License
    - This code was developed by the EIDL at Hanyang University and is subject to the following conditions.
    - Plano_Convex_lens.py code is only limited to KAIST EE Jang's Lab and also it is only limited to co-research use.
    - Prohibit unauthorized third-party sharing.
    - Prohibit other research that reworks the code.
'''
'''
# ----------------------------------------------------------------------------
# -                        EIDL: sites.google.com/view/eidl                  -
# ----------------------------------------------------------------------------
# Author: Sangbin Lee (EIDL, Hanyang University)
# Date: 2024-02-22
​
###------------- Code description -------------###
2D Design of the Plano convex lens
​

'''

import meep as mp
import meep.adjoint as mpa
import numpy as np
import nlopt
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
from matplotlib import pyplot as plt
import os
import math

mp.verbosity(1)

SiO2 = mp.Medium(index=1.5*1.5)
resolution = 100

diameter_values = [10.0, 5.0, 2.5, 2.0, 1.5, 1.0, 0.5]

for diameter_val in diameter_values:
    # Update diameter in the simulation parameters
    diameter = diameter_val # enter the diameter

    # Constants
    PI = np.pi

    # Constants
    a = 1.0
    scale_down = 1

    # Parameters of the Plano convex lens
    d = (diameter / (a * scale_down)) / 2.0 # (diameter / scale_down) / 2 -> half of the diameter // fixed
    focal = d * math.sqrt(21.0) / 2.0 # d * NA 
    r_cur = focal / (2 * a * scale_down)  # focal / (2.0 * a * scale_down); // radius of curvature(5 fixed) -> f = R/(n-1) -> n = 1.5(fixed) -> f = 2R -> f = 10, R = 5

    # Geometry properties
    Lpml = 0.25 * d
    Lx = diameter / (a * scale_down) + Lpml * 2
    Ly = Lpml + 0.25 * d + focal + r_cur - math.sqrt(r_cur * r_cur - d * d) + 0.5 * d + Lpml 

    pml_layers = [mp.PML(thickness=Lpml,)]

    # Position information
    src_ypos =  - Ly / 2 + Lpml + 0.25 * d
    transmission_ypos = Lpml + 1.0
    lens_ypos = - Ly / 2 + Lpml + 0.5 * d + math.sqrt(r_cur * r_cur - d * d)
    x_center = 0
    y_center = lens_ypos + math.sqrt(r_cur * r_cur - d * d)

    # Source settings
    width = 0.1

    fcen_red = 1/(0.65)
    fwidth_red = fcen_red * width

    fcen_green = 1/(0.55)
    fwidth_green = fcen_green * width

    fcen_blue = 1/(0.45)
    fwidth_blue = fcen_blue * width

    src_0 = mp.GaussianSource(frequency=fcen_red, fwidth=fwidth_red, is_integrated=True)
    src_1 = mp.GaussianSource(frequency=fcen_green, fwidth=fwidth_green, is_integrated=True)
    src_2 = mp.GaussianSource(frequency=fcen_blue, fwidth=fwidth_blue, is_integrated=True)

    source_center = [0, src_ypos] # Source position
    source_size = mp.Vector3(Lx)

    source = [mp.Source(src_0, component=mp.Ez, size=source_size, center=source_center,)]

    cell = mp.Vector3(math.ceil(Lx*10)/10, math.ceil(Ly*10)/10)

    geometry = []
    for pos_x in np.arange(-10, 10, 0.01):
        for pos_y in np.arange(-10, 10, 0.01):
            if pos_y <= (lens_ypos):
                if ((pos_x - x_center) ** 2 + (pos_y - y_center) ** 2 <= (r_cur ** 2)):
                    geometry.append(mp.Block(center=mp.Vector3(pos_x, pos_y), size=mp.Vector3(0.5/resolution, 0.5/resolution), material=SiO2))

    sim = mp.Simulation(
        cell_size=cell, 
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=source,
        resolution=resolution,
    )
    sim.plot2D()
    
    tran_total0 = sim.add_dft_fields([mp.Ez], fcen_red, 0 , 1, center=mp.Vector3(0, 0, 0), size=mp.Vector3(Lx - 2 * Lpml ,Ly - 2 * Lpml), yee_grid=True)


    tran_Ex = sim.add_dft_fields([mp.Ex], fcen_red, 0 , 1, center=mp.Vector3(0, 0, 0), size=mp.Vector3(Lx - 2 * Lpml ,Ly - 2 * Lpml), yee_grid=True)
    tran_Ey = sim.add_dft_fields([mp.Ey], fcen_red, 0 , 1, center=mp.Vector3(0, 0, 0), size=mp.Vector3(Lx - 2 * Lpml ,Ly - 2 * Lpml), yee_grid=True)
    tran_Ez = sim.add_dft_fields([mp.Ez], fcen_red, 0 , 1, center=mp.Vector3(0, 0, 0), size=mp.Vector3(Lx - 2 * Lpml ,Ly - 2 * Lpml), yee_grid=True)
    tran_Hx = sim.add_dft_fields([mp.Hx], fcen_red, 0 , 1, center=mp.Vector3(0, 0, 0), size=mp.Vector3(Lx - 2 * Lpml ,Ly - 2 * Lpml), yee_grid=True)
    tran_Hy = sim.add_dft_fields([mp.Hy], fcen_red, 0 , 1, center=mp.Vector3(0, 0, 0), size=mp.Vector3(Lx - 2 * Lpml ,Ly - 2 * Lpml), yee_grid=True)
    tran_Hz = sim.add_dft_fields([mp.Hz], fcen_red, 0 , 1, center=mp.Vector3(0, 0, 0), size=mp.Vector3(Lx - 2 * Lpml ,Ly - 2 * Lpml), yee_grid=True)
    pt = mp.Vector3(0,lens_ypos-focal,0)

    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, pt, 1e-3)) 

    Ex_Namei="Ex_field"
    Ey_Namei="Ey_field"
    Ez_Namei="Ez_field"

    Hx_Namei="Hx_field"
    Hy_Namei="Hy_field"
    Hz_Namei="Hz_field"



    sim.output_dft(tran_Ex,str(diameter)+Ex_Namei)
    sim.output_dft(tran_Ey,str(diameter)+Ey_Namei)
    sim.output_dft(tran_Ez,str(diameter)+Ez_Namei)

    sim.output_dft(tran_Hx,str(diameter)+Hx_Namei)
    sim.output_dft(tran_Hy,str(diameter)+Hy_Namei)
    sim.output_dft(tran_Hz,str(diameter)+Hz_Namei)

    sim.reset_meep()