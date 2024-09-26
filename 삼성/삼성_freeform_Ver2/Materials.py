# Description: This file contains the definition of the materials used in the simulation
#The materials are defined by their Lorentzian susceptibility
#The values are taken from the data of Hynix
import numpy as np
import meep as mp

# Define the materials ###############################################################
Al2O3_frq1 = 10.45201873779817
Al2O3_gam1 = 0.0
Al2O3_sig1 = 0.7485390758646482
Al2O3_frq2 = 10.520778320800689
Al2O3_gam2 = 0.0
Al2O3_sig2 = 1.321424754906942


Al2O3_susc = [
    mp.LorentzianSusceptibility(frequency=Al2O3_frq1, gamma=Al2O3_gam1, sigma=Al2O3_sig1),
    mp.LorentzianSusceptibility(frequency=Al2O3_frq2, gamma=Al2O3_gam2, sigma=Al2O3_sig2),
]
Al2O3 = mp.Medium(
    epsilon=1, E_susceptibilities=Al2O3_susc)
##########################################################################################

# Define the materials ###############################################################
HfO2_frq1 = 6.951670963192224
HfO2_gam1 = 0.0
HfO2_sig1 = 0.7839356658238223
HfO2_frq2 = 9.138192089889
HfO2_gam2 = 0.0
HfO2_sig2 = 1.7078673883568463


HfO2_susc = [
    mp.LorentzianSusceptibility(frequency=HfO2_frq1, gamma=HfO2_gam1, sigma=HfO2_sig1),
    mp.LorentzianSusceptibility(frequency=HfO2_frq2, gamma=HfO2_gam2, sigma=HfO2_sig2),
]
HfO2 = mp.Medium(
    epsilon=1, E_susceptibilities=HfO2_susc)
##########################################################################################

# Define the materials ###############################################################
Si_frq1 = 3.6636779088975473
Si_gam1 = 0.0
Si_sig1 = 7.3201281378957015
Si_frq2 = 2.7935818128476333
Si_gam2 = 0.17126010693525032
Si_sig2 = 3.454969712267653


Si_susc = [
    mp.LorentzianSusceptibility(frequency=Si_frq1, gamma=Si_gam1, sigma=Si_sig1),
    mp.LorentzianSusceptibility(frequency=Si_frq2, gamma=Si_gam2, sigma=Si_sig2),
]
Si = mp.Medium(
    epsilon=1, E_susceptibilities=Si_susc)
##########################################################################################

# Define the materials ###############################################################
SiN_frq1 = 7.7377859974045755
SiN_gam1 = 0.0
SiN_sig1 = 1.147707903082043
SiN_frq2 = 8.506180133298983
SiN_gam2 = 0.0
SiN_sig2 = 1.3401524904921767


SiN_susc = [
    mp.LorentzianSusceptibility(frequency=SiN_frq1, gamma=SiN_gam1, sigma=SiN_sig1),
    mp.LorentzianSusceptibility(frequency=SiN_frq2, gamma=SiN_gam2, sigma=SiN_sig2),
]
SiN = mp.Medium(
    epsilon=1, E_susceptibilities=SiN_susc)
##########################################################################################

# Define the materials ###############################################################
SiO2_frq1 = 10.469053074021327
SiO2_gam1 = 0.0
SiO2_sig1 = 0.4149782511721946
SiO2_frq2 = 10.943135105413115
SiO2_gam2 = 0.0
SiO2_sig2 = 0.6411115779816661


SiO2_susc = [
    mp.LorentzianSusceptibility(frequency=SiO2_frq1, gamma=SiO2_gam1, sigma=SiO2_sig1),
    mp.LorentzianSusceptibility(frequency=SiO2_frq2, gamma=SiO2_gam2, sigma=SiO2_sig2),
]
SiO2 = mp.Medium(
    epsilon=1, E_susceptibilities=SiO2_susc)
##########################################################################################