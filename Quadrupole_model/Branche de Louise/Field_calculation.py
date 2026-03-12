import os
import bempp_cl.api as bempp  
import numpy as np  
from matplotlib import pylab as plt
from Data import Data

class Calculation_field:
    """
    Handles the electrostatic field calculation using the Boundary Element Method (BEM).
    It manages mesh importation, potential settings, matrix inversion, and 
    derivative calculations for a quadrupole geometry.
    """
    def __init__(self, data: Data , file_name : str) -> None:

        """
        Initialize the calculation field with geometry data and BEM spaces.

        Args:
            data (Data): Object containing physical dimensions, potentials, and mesh paths.
        """
        self.data =  data
        self.file_name = file_name

        #Importation of the mesh
        self.mesh_path = os.path.join(self.data.output_dir, "mesh_quadrupole.msh")
        self.grid = bempp.import_grid(self.mesh_path)
        
        # Functions spaces 
        self.dp0_space = bempp.function_space(self.grid, "DP", 0)
        self.p1_space  = bempp.function_space(self.grid, "P", 1)

        # --- operators ---
        self.identity = bempp.operators.boundary.sparse.identity(self.p1_space, self.p1_space, self.dp0_space)

        self.dlp = bempp.operators.boundary.laplace.double_layer(self.p1_space, self.p1_space, self.dp0_space)

        self.slp = bempp.operators.boundary.laplace.single_layer(self.dp0_space, self.p1_space, self.dp0_space)

        #Initial our variable to store the results 

        self.dirichlet_fun = None
        self.neumann_fun = None
        self.u_evaluated = None
        self.points = None

        #Potenital derivative 

        self.E_eval = None
        self.D2_eval = None
        self.D3_eval = None
        self.D4_eval = None


    def mesh_Importation(self) -> None:
        plt.clf()

        
    def potentials_settings(self) -> None:
        """
        Map physical potentials to the corresponding geometric domain indices.
        Defines the Dirichlet boundary conditions for the BEM problem.
        """

        #Settings of the potentials
        pot_apert1 = self.data.pot_apert1
        pot_apert2 = self.data.pot_apert2
        pot_electrode13 = self.data.pot_electrode13
        pot_electrode24 = self.data.pot_electrode24
        pot_shield = self.data.pot_shield  

        apert1=self.data.group_id[0]
        apert2=self.data.group_id[1]
        elec1=self.data.group_id[2]
        elec2=self.data.group_id[3]
        elec3=self.data.group_id[4]
        elec4=self.data.group_id[5]
        shield=self.data.group_id[6]

        @bempp.real_callable
    #Setting of the potential on the different elements of the geometry
    #electrode 1 and 2 are coupled and 3 and 4 are coupled
        def dirichlet_data(x, n, domain_index, result):
            if domain_index == apert1: #potential of the 1st aperture
                result[0]=pot_apert1
                
            elif domain_index == apert2: #potential of the 2nd aperture
                result[0]=pot_apert2
            
            elif domain_index == elec1: #potential of the 1st electrode
                result[0]=pot_electrode13

            elif domain_index == elec2: #potential of the 2nd electrode
                result[0]=pot_electrode13

            elif domain_index == elec3: #potential of the 3rd electrode
                result[0]=pot_electrode24

            elif domain_index == elec4: #potential of the 4th electrode
                result[0]=pot_electrode24

            elif domain_index == shield: #potential of the shield
                result[0]=pot_shield
            
        # Create the GridFunction representing the boundary potential
        self.dirichlet_fun = bempp.GridFunction(self.p1_space, fun=dirichlet_data) 
        # Après avoir créé self.dirichlet_fun
        v_max = np.max(self.dirichlet_fun.coefficients)
        v_min = np.min(self.dirichlet_fun.coefficients)
        print(f"Potentiel Max sur le maillage : {v_max} V")
        print(f"Potentiel Min sur le maillage : {v_min} V")

    def visualisation(self) -> None:
        """Visualize the Dirichlet potential mapped onto the 3D geometry."""
        self.dirichlet_fun.plot()
   
    def matrix_inversion(self) -> None:
    #Sum of the right part of the integral 
        rhs = (-0.5 * self.identity + self.dlp) * self.dirichlet_fun

    #Resolution of the linear system
        self.neumann_fun, _ = bempp.linalg.cg(self.slp, rhs, tol=1e-8) #1e-5 à tester

        #Creation of the tracing of the solution
        n_grid_points = 100
        self.points = np.stack((np.zeros(n_grid_points), np.zeros(n_grid_points), (np.linspace(0, self.data.total_length, n_grid_points)))) #x=y=0 and z changes from 0 to elec_id[7][11]=total_length

        #Green's representation
        slp_pot = bempp.operators.potential.laplace.single_layer(self.dp0_space, self.points) #total matrix
        self.u_evaluated = -slp_pot * self.neumann_fun 


    def derivatives(self) -> None:
        """
        Calculate the 1st to 4th order derivatives of the potential at the evaluation points.
        Uses OpenCL for hardware acceleration.
        """
        #Field 
        E = bempp.operators.potential.laplace.single_layer_gradient(self.dp0_space, self.points, device_interface="opencl")
        self.E_eval = -E * self.neumann_fun

        #Field's 2nd derivative
        D2 = bempp.operators.potential.laplace.single_layer_2nd_deriv(self.dp0_space, self.points, device_interface="opencl")
        self.D2_eval = D2 * self.neumann_fun
    
        #Field's 3rd derivative
        D3 = bempp.operators.potential.laplace.single_layer_3rd_deriv(self.dp0_space, self.points, device_interface="opencl")
        self.D3_eval = D3 * self.neumann_fun

        #Field's 4th derivative
        D4 = bempp.operators.potential.laplace.single_layer_4th_deriv(self.dp0_space, self.points, device_interface="opencl")
        self.D4_eval = D4 * self.neumann_fun


    def potential_exportation(self) -> None: 
        """
        Export all calculated potentials, derivatives, and geometric parameters 
        to a compressed .npz file for post-processing.
        """
        try:
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython is not None:
                ipython.run_line_magic("matplotlib", "inline")
                ipython = True

        except NameError:
            ipython = False

        savefile = os.path.join(self.data.output_dir, self.file_name)
    
        #Creation of a .npz file with the extracted potential
        if "savefile" is not None:
            np.savez_compressed(
        savefile,
        #Points
        points=self.points,

        #Extracted potential
        potential=self.u_evaluated,

        #Geometry identification
        group_id_ap1=self.data.group_id[0],
        group_id_ap2=self.data.group_id[1],
        group_id_cyl1=self.data.group_id[2],
        group_id_cyl2=self.data.group_id[3],
        group_id_cyl3=self.data.group_id[4],
        group_id_cyl4=self.data.group_id[5],
        group_id_shield=self.data.group_id[6],

        #Mesh parameters
        MeshSizeMin = self.data.MeshSizeMin,
        MeshSizeMax = self.data.MeshSizeMax,
        MeshSizeFromCurvature = self.data.MeshSizeFromCurvature,

        #Output directory
        output_dir = self.data.output_dir,

        #Dimensions
        dist_shield_apert = self.data.dist_shield_apert,
        dist_apert_quad = self.data.dist_apert_quad,
        radius_ext_shield = self.data.radius_ext_shield,
        radius_in_shield = self.data.radius_in_shield,
        thickness_shield = self.data.thickness_shield,
        radius_apert = self.data.radius_apert,
        thickness_apert = self.data.thickness_apert,
        length_cylinder = self.data.length_cylinder,
        radius_axis = self.data.radius_axis,
        total_length = self.data.total_length,

        #Coordinates
        coord_cylinder_x_or_y = self.data.coord_cylinder_x_or_y,
        coord_cylinder_z = self.data.coord_cylinder_z,
        coord_apert_z1 = self.data.coord_apert_z1,
        coord_apert_z2 = self.data.coord_apert_z2,
        start_shield1 = self.data.start_shield1,
        end_shield1 = self.data.end_shield1,
        start_apert1 = self.data.start_apert1,
        end_apert1 = self.data.end_apert1,
        start_cyl = self.data.start_cyl,
        end_cyl = self.data.end_cyl,
        start_apert2 = self.data.start_apert2,
        end_apert2 = self.data.end_apert2,
        start_shield2 = self.data.start_shield2,
        end_shield2 = self.data.end_shield2,

        #Potentials
        pot_electrode13 = self.data.pot_electrode13,
        pot_electrode24 = self.data.pot_electrode24,
        pot_apert1 = self.data.pot_apert1,
        pot_apert2 = self.data.pot_apert2,
        pot_shield = self.data.pot_shield, 
        pot_acceleration = self.data.pot_acceleration,

        #Derivatives
        E_eval=self.E_eval,
        D2_eval=self.D2_eval,
        D3_eval=self.D3_eval,
        D4_eval=self.D4_eval,)
                            
        print(f"Saved potential to {savefile}.npz")


    def potential_axis_printing(self) -> None:
        """Plot the calculated potential along the Z-axis."""
        plt.plot(self.points[2], self.u_evaluated[0])
        plt.title("Potentiel du quadrupole d'Okayama le long de l'axe Z")
        plt.xlabel("Position en z (mm)")
        plt.ylabel("Potentiel (V)")
        plt.show()