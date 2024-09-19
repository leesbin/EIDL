# Description: This file contains the functions for the mapping of the design variables to the physical domain.

import meep as mp
import meep.adjoint as mpa
import numpy as np
import nlopt
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
from matplotlib import pyplot as plt

# Function for the mapping of the design variables with minimum feature size constraint
def get_conic_radius(b, eta_e):
    if (eta_e >= 0.5) and (eta_e < 0.75):
        return b / (2 * np.sqrt(eta_e - 0.5))
    elif (eta_e >= 0.75) and (eta_e <= 1):
        return b / (2 - 2 * np.sqrt(1 - eta_e))
    else:
        raise ValueError("(eta_e) must be between 0.5 and 1.")

# Function for the mapping of the design variables with binarization constraint    
def tanh_projection_m(x: np.ndarray, beta: float, eta: float) -> np.ndarray:
    """Sigmoid projection filter.

    Ref: F. Wang, B. S. Lazarov, & O. Sigmund, On projection methods,
    convergence and robust formulations in topology optimization.
    Structural and Multidisciplinary Optimization, 43(6), pp. 767-784 (2011).

    Args:
        x: 2d design weights to be filtered.
        beta: thresholding parameter in the range [0, inf]. Determines the
            degree of binarization of the output.
        eta: threshold point in the range [0, 1].

    Returns:
        The filtered design weights.
    """
    if beta == npa.inf:
        if eta < 0.5:
            return npa.where(x > eta, 1.0, 0.0)
        else:
            return npa.where(x < eta, 0.0, 1.0)            
    else:
        if eta < 0.5: # dilation case -> remove boundary
            x=npa.where(x == eta, eta - 1e-4, x) 
        else:         # erosion & smoothing case-> save boundary 
            x=npa.where(x == eta, eta + 1e-4, x)
        return (npa.tanh(beta * eta) + npa.tanh(beta * (x - eta))) / (
            npa.tanh(beta * eta) + npa.tanh(beta * (1 - eta))
        )
    
# Function for the mapping of the design variables with symmetry constraint
def left_right_symmetric(array):
    rows, cols = array.shape
    mid_col = cols // 2
    left_half = (array[:, :mid_col] + npa.fliplr(array[:, :mid_col])) / 2
    right_half = (array[:, mid_col:] + npa.fliplr(array[:, mid_col:])) / 2
    new_array = npa.hstack((left_half, right_half))
    return new_array

def up_down_symmetric(array):
    rows, cols = array.shape
    mid_row = rows // 2
    top_half = (array[:mid_row, :] + npa.flipud(array[:mid_row, :])) / 2
    bottom_half = (array[mid_row:, :] + npa.flipud(array[mid_row:, :])) / 2
    new_array = npa.vstack((top_half, bottom_half))
    return new_array

# Function for the mapping of the design variables
def multiregion_mapping(
        x,                             # design variables
        eta,                           # threshold point of the linear filter
        beta,                          # binarization parameter
        ind,                           # the index of the design variable
        Nx,                            # number of design variables in x direction
        Ny,                            # number of design variables in y direction
        filter_radius,                 # filter radius
        design_region_width_x,         # design region width in x direction
        design_region_width_y,         # design region width in y direction
        design_region_resolution,      # design region resolution
        pad_value,                     # padding value
        o_grid                         # one grid size
        ):
    x_copy = (x.reshape(Nx * Ny, 5)).transpose()

    # projection
    z = 0
    x2 = []

    while z < 5:
        
        number = z
        x1 = x_copy[int(number)]
        x1 = x1.reshape(Nx, Ny)

        # Symmetry
        # lrfilp = left_right_symmetric(x1) # left-right symmetric
        # upfilp = up_down_symmetric(lrfilp) # up-down symmetric
        diagonalfilp = (x1 + x1.transpose()) / 2 # diagonal symmetric

        #filter padding
        padx = npa.pad(diagonalfilp, ((pad_value,pad_value),(pad_value,pad_value)), 'constant', constant_values=0) 

        #filter
        filtered_field = mpa.conic_filter(
            padx,
            filter_radius,
            design_region_width_x+pad_value*o_grid*2,
            design_region_width_y+pad_value*o_grid*2,
            design_region_resolution,
        )

        #crop
        crop_field = filtered_field[pad_value:-pad_value, pad_value:-pad_value]

        z_slice = crop_field.flatten()

        x2 = npa.concatenate((x2,z_slice),axis=0) 

        z = z + 1

    x2 = ((x2.reshape(5, Nx*Ny)).transpose()).flatten() # reshape

    x2 = mpa.tanh_projection(x2, beta, eta) # projection
    
    x2 = (x2.reshape(Nx * Ny, 5)).transpose()
        
    x_0 = npa.array(x2[0]).flatten()
    x_1 = npa.array(x2[1]).flatten()
    x_2 = npa.array(x2[2]).flatten()
    x_3 = npa.array(x2[3]).flatten()
    x_4 = npa.array(x2[4]).flatten()

    stacked_x2 = npa.concatenate([x_0, x_1, x_2, x_3, x_4], axis=0)

    if ind==0:
        return x_0, x_1, x_2, x_3, x_4
    elif ind==1:
        return stacked_x2.reshape(5, Nx*Ny).transpose().flatten()
#----------------------------------------------------------------------------------------------------------------------------------
    
def mapping(
        x,                             # design variables
        eta,                           # threshold point of the linear filter
        beta,                          # binarization parameter
        Nx,                            # number of design variables in x direction
        Ny,                            # number of design variables in y direction
        Nz,                            # number of design variables in z direction
        filter_radius,                 # filter radius
        design_region_width_x,         # design region width in x direction
        design_region_width_y,         # design region width in y direction
        design_region_resolution,      # design region resolution
        pad_value,                     # padding value
        o_grid                         # one grid size
        ):
    x_copy = (x.reshape(Nx * Ny, Nz)).transpose()

    # projection
    z = 0
    x2 = []

    while z < Nz:
        
        number = z
        x1 = x_copy[int(number)] 
        x1 = x1.reshape(Nx, Ny)
        
        diagonalfilp = (x1 + x1.transpose()) / 2 # diagonal symmetric

        #filter padding
        padx = npa.pad(diagonalfilp, ((pad_value,pad_value),(pad_value,pad_value)), 'constant', constant_values=0) 

        #filter
        filtered_field = mpa.conic_filter(
            padx,
            filter_radius,
            design_region_width_x+pad_value*o_grid*2,
            design_region_width_y+pad_value*o_grid*2,
            design_region_resolution,
        )

        #crop
        crop_field = filtered_field[pad_value:-pad_value, pad_value:-pad_value]

        z_slice = crop_field.flatten()

        x2 = npa.concatenate((x2,z_slice.flatten()),axis=0) 

        z = z + 1

    x2 = ((x2.reshape(Nz,Nx*Ny)).transpose()).flatten()

    x2 = mpa.tanh_projection(x2, beta, eta) # projection

    x = x2

    return x