import sys
import meep as mp
import math
import numpy as np
import h5py
import scipy
from autograd import numpy as npa


def h5_load_fields (FileName1, is_forward_field: bool = True, coo=0):
    # output: target field = adjoint field^* <- in this case, you should set "is_forward_field = False"
    R_coo=['ex_0.r','ey_0.r','ez_0.r']
    I_coo=['ex_0.i','ey_0.i','ez_0.i']
    
    # Load DFT Field
    print(FileName1)
    hf = h5py.File(FileName1, 'r')
    A=hf.get(R_coo[coo])
    R=np.array(A) # Real value

    B=hf.get(I_coo[coo])
    I=np.array(B)*1j # Imaginary value

    Field=R+I # Complex field

    if is_forward_field:
        return Field
    else:
        return np.conjugate(Field)


def load_target_field ():
    Ex_Name=f"Ex_field.h5"
    E_x=h5_load_fields(Ex_Name)

    Ey_Name=f"Ey_field.h5"
    E_y=h5_load_fields(Ey_Name, coo=1)

    Ez_Name=f"Ez_field.h5"
    E_z=h5_load_fields(Ez_Name, coo=2)

    return [E_x, E_y, E_z]

def My_LiNbO3_pso():
    um_scale=1
    mLiNbO3_range = mp.FreqRange(min=um_scale / 4.9, max=um_scale / 0.4) # Possible range: 5 um ~ 400 nm
    # n_o
    LiNbO3_frq1 = 0.051067
    LiNbO3_gam1 = 0
    LiNbO3_sig1 = 10.055543
    LiNbO3_frq2 = 4.082783
    LiNbO3_gam2 = 0
    LiNbO3_sig2 = 0.952295
    LiNbO3_frq3 = 6.354196
    LiNbO3_gam3 = 0
    LiNbO3_sig3 = 2.940970

    mLiNbO3_susc_o = [
        mp.LorentzianSusceptibility(
            frequency=LiNbO3_frq1,
            gamma=LiNbO3_gam1,
            sigma_diag=LiNbO3_sig1 * mp.Vector3(1, 0, 1),
        ),
        mp.LorentzianSusceptibility(
            frequency=LiNbO3_frq2,
            gamma=LiNbO3_gam2,
            sigma_diag=LiNbO3_sig2 * mp.Vector3(1, 0, 1),
        ),
        mp.LorentzianSusceptibility(
            frequency=LiNbO3_frq3,
            gamma=LiNbO3_gam3,
            sigma_diag=LiNbO3_sig3 * mp.Vector3(1, 0, 1),
        ),
    ]
    # n_e
    LiNbO3_frq1 = 0.062152
    LiNbO3_gam1 = 0
    LiNbO3_sig1 = 5.315306
    LiNbO3_frq2 = 5.932484
    LiNbO3_gam2 = 0
    LiNbO3_sig2 = 3.041606
    LiNbO3_frq3 = 5.512815
    LiNbO3_gam3 = 0
    LiNbO3_sig3 = 0.547265

    mLiNbO3_susc_e = [
        mp.LorentzianSusceptibility(
            frequency=LiNbO3_frq1,
            gamma=LiNbO3_gam1,
            sigma_diag=LiNbO3_sig1 * mp.Vector3(0, 1, 0),
        ),
        mp.LorentzianSusceptibility(
            frequency=LiNbO3_frq2,
            gamma=LiNbO3_gam2,
            sigma_diag=LiNbO3_sig2 * mp.Vector3(0, 1, 0),
        ),
        mp.LorentzianSusceptibility(
            frequency=LiNbO3_frq3,
            gamma=LiNbO3_gam3,
            sigma_diag=LiNbO3_sig3 * mp.Vector3(0, 1, 0),
        ),
    ]

    Munseong_LiNbO3 = mp.Medium(
        epsilon=1.0,
        E_susceptibilities=mLiNbO3_susc_o + mLiNbO3_susc_e,
        valid_freq_range=mLiNbO3_range,
    )
    return Munseong_LiNbO3      