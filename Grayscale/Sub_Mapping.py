import scipy
import math
import numpy as np
import meep as mp
import meep.adjoint as mpa
import autograd.numpy as npa

def get_conic_radius(b, eta_e):
    if (eta_e >= 0.5) and (eta_e < 0.75):
        return b / (2 * np.sqrt(eta_e - 0.5))
    elif (eta_e >= 0.75) and (eta_e <= 1):
        return b / (2 - 2 * np.sqrt(1 - eta_e))
    else:
        raise ValueError("(eta_e) must be between 0.5 and 1.")
#------------------------------------------------------------------
"""
2D to 3D Sub mapping moudule for x-cut LN waveguide's slanted side wall (67 deg)

The code was optimized for X: heights, Y: width, Z: length (propagation axis)
"""
def get_reference_layer(#-- Input parameters ---------------------|
    Symmetry_in_Sim,
    Width_symmetry,
    Length_symmetry,  
    #                                                             |
    DR_width, # Disign region width                               |
    N_width,  # Number of DR width grids                          |
    #                                                             |
    DR_length, # Disign region length                             |
    N_length,  # Number of DR length grids                        |
    #                                                             |
    DR_res, # Design region resolution                            |
    #                                                             |
    Min_size_top, # Minimum linwidth of top layer                 |
    Min_gap, # Minimum gap width of ref layer                     |
    #                                                             |
    input_w_top, # Top width of input waveguide                   |
    w_top, # Top width of output waveguide                        |
    #                                                             |
    x, # Input material density array                             |
    #                                                             |
    eta_Ref, # Thresholding point of hyper tangential projection  |
    beta, # Binrization parameter                                 |
):                                                                #-- Impose MFS and MGS on x ------------------------|
    Size_Ref= Min_size_top                                                                   # Top layer -------------|
    Gap_Ref= Min_gap                                                                         #                        |
    Mask_sz_i = (input_w_top)                                                                #                        |
    Mask_sz_o = (w_top)     #-----------------------------------------------------------------------------------------|
    width_grid = np.linspace(-DR_width /2, DR_width /2, N_width)                             # 2D Grid for Mask ------|
    length_grid = np.linspace(-DR_length /2, DR_length /2, N_length)                         #                        |
    W_g, L_g = np.meshgrid(width_grid, length_grid, sparse=True, indexing="ij")              #                        |
    down_wg_mask= ((L_g <=(-DR_length/2)+ 15/DR_res) & (np.abs(W_g) <=Mask_sz_i/2))          # Core mask -------------|
    up_wg_mask= ((L_g >=(DR_length/2)- 15/DR_res) & (np.abs(W_g) <=Mask_sz_o/2))             #                        |
    Li_mask = down_wg_mask | up_wg_mask #-----------------------------------------------------------------------------|
    border_mask = (                                                                          # Cladding mask ---------|
        (W_g >= (DR_width /2)- 15/DR_res)                                                    #                        |
        | (W_g <= (-DR_width /2)+ 15/DR_res)                                                 #                        |
        | (L_g <= (-DR_length /2)+ 15/DR_res)                                                #                        |
        | (L_g >= (DR_length /2)- 15/DR_res)                                                 #                        |
    ) #---------------------------------------------------------------------------------------------------------------|
    Air_mask = border_mask.copy()                                                            # Mask ------------------|
    Air_mask[Li_mask] = False                                                                #                        |
    x_ref = npa.reshape(x,(N_width,N_length))  # Reshape for flip ----------------------------------------------------|
    if Width_symmetry:                                                                       # Width Mirror ----------|
        if Symmetry_in_Sim:                                                                  #   Y Mirror Sim Case    |
            if N_width/2 == int(N_width/2):                                                  #                        |
                x_ref = ((npa.flipud(x_ref)) + x_ref)/2                                      #                        |
                print('ODD pixels & EVEN grids')                                             #                        |  
            else:                                                                            #                        |
                x_ref = ((npa.flipud(x_ref)) + x_ref)/2                                      #                        |
                print('EVEN pixels & ODD grids')                                             #                        |
        else:                                                                                #                        |
            x_ref = (npa.flipud(x_ref) + x_ref)/2                                            #                        |
    if Length_symmetry:                                                                      # Length Mirror ---------|
        x_ref = (npa.fliplr(x_ref) + x_ref)/2                                                #                        |
    x_ref = mpa.tanh_projection(npa.clip(x_ref, 0.0, 1.0), beta, 0.5)                        # Binarization ----------|
    x_ref = x_ref.flatten()                                                                  #                        |
    #-----------------------------------------------------------------------------------------------------------------|
    x_copy = npa.where(Li_mask.flatten(), 1, npa.where(Air_mask.flatten(), 0, x_ref))        # Masking ---------------|
    #-----------------------------------------------------------------------------------------------------------------|
    x_copy = mpa.conic_filter(x_copy, Min_size_top, DR_width, DR_length, DR_res)             #                        |
    x_copy = mpa.tanh_projection(npa.clip(x_copy, 0.0, 1.0), beta, 0.5)                      # Top layer smoothing ---|
    #-----------------------------------------------------------------------------------------------------------------|
    Single_pixel= round(1/DR_res,2)
    N_Div=int(get_conic_radius(Gap_Ref, 1-eta_Ref)/(4*Single_pixel))       #------------------------------------------|
    Total_filter_R= get_conic_radius(Gap_Ref, 1-eta_Ref)
    Unit_filter_R= get_conic_radius(Gap_Ref/(N_Div), 1-eta_Ref) 
    Compensation_weight= Total_filter_R/Single_pixel- int(Unit_filter_R/Single_pixel)*N_Div
    Compensated_R= Unit_filter_R+ Single_pixel*Compensation_weight                                              
    Erosion_index= 0                                          #------- Activate multiple dilation N_Div times --------|
    while Erosion_index < N_Div:                                                             # Dilation: Gap/N_Div ---|
        if Erosion_index == N_Div-1:                           #--------- Final erosion ------------------------------|
            x_copy= mpa.conic_filter(x_copy, Compensated_R, DR_width, DR_length, DR_res)     # Compensated dilation   |
            x_copy = mpa.tanh_projection(npa.clip(x_copy, 0.0, 1.0), beta, eta_Ref)    #-----#                        |
            x_copy= mpa.conic_filter(x_copy, Compensated_R, DR_width, DR_length, DR_res)     # Compensated erosion    |
            x_copy =  mpa.tanh_projection(npa.clip(x_copy, 0.0, 1.0), beta, 1-eta_Ref) #-----#                        |
            Gap_idx = 0                                        #------- Activate multiple erosion N_Div times --------|
            while Gap_idx < N_Div-1:                                                         # Erosion: Gap/N_Div ----|
                x_copy= mpa.conic_filter(x_copy, Unit_filter_R, DR_width, DR_length, DR_res) #                        |
                x_copy =  mpa.tanh_projection(npa.clip(x_copy, 0.0, 1.0), beta, 1-eta_Ref)   # Unit erosion           |
                Gap_idx= Gap_idx +1                                                          #                        |
            break                                                                            # Ensure minimum gap     |
        else:    
            x_copy= mpa.conic_filter(x_copy, Unit_filter_R, DR_width, DR_length, DR_res)     #                        |
            x_copy = mpa.tanh_projection(npa.clip(x_copy, 0.0, 1.0), beta, eta_Ref)          # Unit dilation          |
        Erosion_index= Erosion_index+1 #------------------------------------------------------------------------------|
    #-----------------------------------------------------------------------------------------------------------------|
    x_copy = x_copy.flatten()                                                                #                        |
    return x_copy                                                                            # Reference layer -------|
#---------------------------------------------------------------------------------------------------------------------|    

def eta_z(
    z, 
    eta_top, 
    bottom, 
    top
): #- Height dependent threshold for linear filter -------|<- eta_bot=0.5
    norm_h=(z-bottom)/(top-bottom)                        #
    Size= 2*(norm_h)*(1-np.sqrt(1-1*eta_top))             #<- Size_rad*2
    if Size < 1.0:                                        #
        eta=(Size**2)/4 + 0.5                             #
    else:                                                 #
        eta=1 - ((Size-2)**2)/4                           #
    return eta # 0.5 ~ eta_top                            #
#---------------------------------------------------------|

def Local_dilation(
    Reference_layer, 
    DR_width, 
    DR_length, 
    DR_res, 
    beta, 
    eta_gap, 
    N_local, 
    Local_Size, 
    const
): #----------------------------- Dilation by heights on local region---------| 
    Local_x = []                 #--------------------------------------------|    
    local_h = N_local            # initialize the local height index as a top |
    eta_bot = 0.5-eta_gap            # Maximum dilation size for bottom layer |
    while local_h > 0:      #- h: Top to Bottom, linear dilation by heights ------------------------|
        Local_R = get_conic_radius(Local_Size, 1-eta_bot)                                 #         |
        eta_dh= eta_z(local_h, 1-eta_bot, 1, N_local+const) - eta_gap       # dilation threshold (h)| 
        z_slice = mpa.conic_filter(Reference_layer, Local_R, DR_width, DR_length, DR_res) #         |
        z_slice = mpa.tanh_projection(npa.clip(z_slice, 0.0, 1.0), beta, eta_dh)      # Dilation    |
        z_slice = z_slice.flatten()                                                   #             |
        Local_x = npa.concatenate((z_slice, Local_x), axis=0)                         #             |
        local_h = local_h -1                                                          #             |
    return Local_x, z_slice #- 3D geo, 2D Reference plane for next subsection ----------------------|
#-----------------------------------------------------------------------------| 

def Slant_sidewall(
    Reference_layer,
    DR_width, 
    DR_length, 
    DR_res,
    beta,
    eta_gap,
    Min_gap,
    N_height,
    Number_of_local_region,
):                           #-- Impose Slanted angle. Than, structs 3D density array map --|
    x2 = []                                                        #                        |
    Nl=int(N_height/Number_of_local_region)                        #                        |
    Nl_r=int(N_height-(Nl*Number_of_local_region))                 #                        |
    Local_Size= Min_gap*((Nl)/(N_height))                          #                        |
    Single_pixel= round(1/DR_res,2) 
    LR_temp = 0                                                    #                        |
    Layer0, z_slice= Local_dilation(                               # Main section ----------|
        Reference_layer= Reference_layer,                          #                        |
        DR_width= DR_width,                                        #                        |
        DR_length= DR_length,                                      #                        |
        DR_res= DR_res,                                            #                        |
        beta= beta,                                                #                        |
        eta_gap= eta_gap,                                          #                        |
        N_local= Nl_r + Nl,                                        #                        |
        Local_Size= Min_gap*(Nl_r + Nl)/(N_height) +Single_pixel,  #                        |
        const= 0.001,                                              #                        |
    )                                                              #                        |
    x2= npa.concatenate((Layer0, x2), axis=0)                      #                        |
    while LR_temp < Number_of_local_region-1:                      # Sub sections X NR -----|
        Layer, z_slice= Local_dilation(                            # Dilated layer          |
            Reference_layer=z_slice,                               #                        |
            DR_width= DR_width,                                    #                        |
            DR_length= DR_length,                                  #                        |
            DR_res= DR_res,                                        #                        |
            beta= beta,                                            #                        |
            eta_gap= eta_gap,                                      #                        |
            N_local= Nl,                                           #                        |
            Local_Size= Local_Size +Single_pixel,                  #                        |
            const= 1,                                              #                        |
        )                                                          #                        |
        x2= npa.concatenate((Layer, x2), axis=0)                   #                        |
        LR_temp= LR_temp +1      # ---------------------------------------------------------|
    x= x2.flatten()                                                #                        |
    return x                                                       # 3D slanted geometry----|
#-------------------------------------------------------------------------------------------|









def GrayScale_Grating(
    Width_symmetry,
    Length_symmetry,
    x, 
    N_width, 
    N_length, 
    N_height, 
): #---------------Grating vertical cross section ---------|
    x2 = []
    height= 0
    while height < N_height:     
        z_slice = npa.reshape(x, (N_width, N_length))
        if Width_symmetry:
            z_slice = (npa.flipud(z_slice) + z_slice) / 2 #<---------width symmetry
        if Length_symmetry:
            z_slice = (npa.fliplr(z_slice) + z_slice) / 2 #<---------length symmetry
        z_slice= z_slice.flatten()
        x2 = npa.concatenate((x2,z_slice),axis=0) 
        height = height + 1
    x= x2.flatten()
    return x

def Binarization(
    Width_symmetry,
    Length_symmetry,
    x, 
    N_width, 
    N_length, 
    N_height,
    beta, 
): #---------------Grating vertical cross section ---------|
    x2 = []
    height= 0
    while height < N_height:     
        z_slice = npa.reshape(x, (N_width, N_length))
        if Width_symmetry:
            z_slice = (npa.flipud(z_slice) + z_slice) / 2 #<---------width symmetry
        if Length_symmetry:
            z_slice = (npa.fliplr(z_slice) + z_slice) / 2 #<---------length symmetry
        z_slice = mpa.tanh_projection(npa.clip(z_slice, 0.0, 1.0), beta, 0.5)
        z_slice= z_slice.flatten()
        x2 = npa.concatenate((x2,z_slice),axis=0) 
        height = height + 1
    x= x2.flatten()
    return x

def TwoD_Filtered(
    Width_symmetry,
    Length_symmetry,
    x, 
    DR_width, 
    DR_length,
    DR_res, 
    N_height,
    Cone_rad,
    beta,
): #---------------Grating V cross section, Filterd H cross section ---------|
    x2 = []
    height= 0
    while height < N_height:    
        z_slice = mpa.conic_filter(x, Cone_rad, DR_width, DR_length, DR_res)
        if Width_symmetry:
            z_slice = (npa.flipud(z_slice) + z_slice) / 2 #<---------width symmetry
        if Length_symmetry:
            z_slice = (npa.fliplr(z_slice) + z_slice) / 2 #<---------length symmetry
        z_slice = mpa.tanh_projection(npa.clip(z_slice, 0.0, 1.0), beta, 0.5)
        z_slice= z_slice.flatten()
        x2 = npa.concatenate((x2,z_slice),axis=0) 
        height = height + 1
    x= x2.flatten()
    return x    
