#!/bin/bash
#mpirun -np 6 python 0_Gray_scale.py
#mpirun -np 6 python 1_Penalization.py
#mpirun -np 6 python 2_Slant_penalization_last.py
#mpirun -np 90 python Target_field.py
#mpirun -np 130 python Optimization.py
mpirun -np 6 python Plane_focus.py
mpirun -np 6 python Plane_focus_EH.py
mpirun -np 6 python Plane_focus_gray.py
mpirun -np 6 python Plane_focus_EH_gray.py
mpirun -np 6 python Plane_focus_EH_gray_norm.py
