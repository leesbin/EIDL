# Design CIS 2D color router
# Resoultion 25 

# ## 1. Simulation Environment

import meep as mp
import meep.adjoint as mpa
import numpy as np
import nlopt
import math
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
from matplotlib import pyplot as plt
import os

mp.verbosity(1)

# Constants
PI = np.pi

# User input
diameter = 2.5  # enter the diameter
a = 1.0  # 1a = 500nm(0.5), 1um(1.0)
scale_down = 1.0  # scale down factor
# geometry properties
Lx = diameter / (a * scale_down)  # Lx -> simulation diameter -> 50 fix
Ly = 2.0 * Lx  # 10.0 + 25.0 * 1.25 / (a * scale_down); //
resolution = 100  # 1/resolution = a size of one pixel
Lpml = 0.25  # thickness of the PML

# About position information
src_ypos = Ly - Lpml - 0.25  # position of the source -> y = Ly - 4.0
transmission_ypos = Lpml + 1.0  # position of the transmission -> calculate with NA index -> y = 4.0

# Source properties
pol = 1  # polarization (0 = TE, 1 = TM)
fmin_dft = 1.0  # minimum f0_srcuency
fmax_dft = 1.0  # maximum f0_srcuency
df_src = 1.0  # bandwidth of the source
f0_src = (fmin_dft + fmax_dft) / 2.0  # center f0_srcuency
theta_f0 = 0.0 * PI / 180.0  # incidence angle degree to radian
k_per = 2.0 * PI * f0_src * np.sin(theta_f0)
Nf_dft = 1  # the number of DFT f0_srcuency

# Parameters of the Plano convex lens
d = (diameter / (a * scale_down)) / 2.0  # (diameter / scale_down) / 2 -> half of the diameter // fixed
focal = d * np.sqrt(21.0) / 2.0  # NA = 0.6, 100.0/3.0 = 33.33
r_cur = focal / (2 * a * scale_down)  # focal / (2.0 * a * scale_down); // radius of curvature(5 fixed) -> f = R/(n-1) -> n = 1.5(fixed) -> f = 2R -> f = 10, R = 5
sin_NA = d / np.sqrt(d * d + focal * focal)
k_airy = 2.0 * PI / 1.0
airy_NA = 0.38317 * Lx / 2.0  # 3.8317/(a * scale_down * k_airy * sin_NA);
lens_ypos = src_ypos - 0.5
x_center = Lx / 2.0  # x center position of the circle 
y_center = lens_ypos + np.sqrt(r_cur * r_cur - d * d)  # y center position of the circle -> change with diameter

# f = 12um, diameter = 10um -> NA = nsin(theta) = 0.3846 -> kasin(theta) = 3.8317 -> a*scale_down = 0.8721(green), 1.11(red)
