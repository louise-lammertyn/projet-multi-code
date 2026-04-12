from Main_functions import Potential_extraction
from Data import Data
import os
from bempp_cl.core import opencl_kernels
from Field_calculation import Calculation_field
import gc

#opencl_kernels.show_available_platforms_and_devices()
#opencl_kernels.set_default_cpu_device(0,1)

os.environ['PYOPENCL_COMPILER_OUTPUT']='1'
os.environ['PYOPENCL_NO_CACHE'] = '1'

OUTPUT_DIR = r"C:\Users\zoeno\OneDrive - INSA Toulouse\Documents\INSA\4GP\Projet multi\projet-multi-code\Programs\Files"
#OUTPUT_DIR = r"C:\Users\llamm\OneDrive\Documents\Projet\BEMPP\okayama\projet_multi\projet-multi-code\Quadrupole_model\Files"
"-0.0299087*Va, -0.18808*Va, +0.18808*Va" #tension qui fit

#Dictionnary to do step by step
quad = {
    "aperture1": [0, 0, 1, 0],
    "aperture2": [0, 0, 0, 1],
    "quad13":    [1, 0, 0, 0],
    "quad24":    [0, 1, 0, 0]
}

#Generates the mesh
data_initial = Data(4, 5, 21, 19, 2, 15, 2, 13, 3.4934, 0, 0, 0, 0, 0, 1.5, 0.1, 0.5, 30, OUTPUT_DIR)
fun_initial = Potential_extraction(data_initial, True, "temp.npz")
fun_initial.mesh_without_shield() 
"""
fun_initial.mesh_quad2() 
fun_initial.mesh_quad3() 
fun_initial.mesh_quad4() 
saved_group_ids_quad1 = data_initial.group_id_quad1
saved_group_ids_quad2 = data_initial.group_id_quad2
saved_group_ids_quad3 = data_initial.group_id_quad3
saved_group_ids_quad4 = data_initial.group_id_quad4

systems_config = {
    "quad1": {"mesh": "mesh_quadrupole1.msh", "ids": saved_group_ids_quad1},
    "quad2": {"mesh": "mesh_quadrupole2.msh", "ids": saved_group_ids_quad2},
    "quad3": {"mesh": "mesh_quadrupole3.msh", "ids": saved_group_ids_quad3},
    "quad4": {"mesh": "mesh_quadrupole4.msh", "ids": saved_group_ids_quad4},
}

fun = Potential_extraction(data_initial, True, "mesh_quadrupole1.msh")

fun.potential_extraction() #BEM resolution
fun.potential_visualisation()

fun.graph_potential_axis()
"""
"""
#Element by element
for sys_name, config in systems_config.items():
    current_mesh = config["mesh"]
    current_ids = config["ids"]
    
    print(f"--- Processing System: {sys_name} with {current_mesh} ---")
    for element, tensions in quad.items():
        vq13, vq24, va1, va2 = tensions 

        file_name = f"{sys_name}_{element}.npz" 

        data = Data(4, 5, 21, 19, 2, 15, 2, 13, 3.4934, vq13, vq24, va1, va2, 0, 1.5, 0.9, 2, 30, OUTPUT_DIR)
        data.mesh_name = current_mesh 
        data.group_id = current_ids

        fun = Potential_extraction(data, False, file_name)

        fun.potential_extraction() #BEM resolution
        fun.potential_visualisation()

        fun.graph_potential_axis()
        del fun
        gc.collect()
"""
#data = Data(4, 5, 21, 19, 2, 15, 2, 13, 3.4934,1, -0.18808*Va, +0.18808*Va, 0, Va, 0.9, 3, 20, OUTPUT_DIR)