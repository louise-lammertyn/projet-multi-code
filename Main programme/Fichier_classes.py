#Importation of all the necessary packages
import os
import numpy as np
from matplotlib import pylab as plt
import bempp_cl.api as bempp            
import gmsh
from IPython import get_ipython
import matplotlib
from bempp_cl.core import opencl_kernels
import bempp_cl.api as bempp

opencl_kernels.show_available_platforms_and_devices()
opencl_kernels.set_default_cpu_device(1,0)

#Function used to empty the RAM
os.environ['PYOPENCL_COMPILER_OUTPUT']='1'

#Pour redémarrer le kernel
#Ctrl + Shift + P
#Jupyter: Restart Kernel

OUTPUT_DIR = r"C:\Users\zoeno\OneDrive - INSA Toulouse\Documents\INSA\4GP\Projet multi\projet-multi-code\Main programme"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#Definition of classes to describe the geometry
class Cylinder:
    def __init__(self, length: int, radius: int, coord_x: int, coord_y: int, coord_z: int) -> None:
        self.length = length
        self.radius = radius
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.coord_z = coord_z

        self.cyl_tag = None
        self.cyl_surf = None


    def ajouter(self) -> None:
        self.cyl_tag = gmsh.model.occ.addCylinder(self.coord_x, self.coord_y, self.coord_z, 0, 0, self.length, self.radius)
        self.cyl_surf = gmsh.model.occ.get_surface_loops(self.cyl_tag)[1][0]


class Aperture: 
    def __init__(self, radius_ext: int, radius_in: int, thickness: int, coord_x: int, coord_y: int, coord_z: int) -> None:
        self.radius_ext = radius_ext
        self.radius_in = radius_in
        self.thickness = thickness
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.coord_z = coord_z

        self.apert_tag = None
        self.apert_surf = None


    def ajouter (self) -> None:
        aperture_out = gmsh.model.occ.addCylinder(self.coord_x, self.coord_y, self.coord_z, 0, 0, self.thickness, self.radius_ext)
        aperture_in = gmsh.model.occ.addCylinder(self.coord_x, self.coord_y, self.coord_z, 0, 0, self.thickness, self.radius_in)
        apert_vol , _=  gmsh.model.occ.cut([(3,aperture_out)],[(3,aperture_in)])
        self.apert_tag=apert_vol[0][1]
        self.apert_surf = gmsh.model.occ.get_surface_loops(self.apert_tag)[1][0]

class Shield:
    def __init__(self, length: int, radius_ext: int, radius_in: int, radius_hole: int, thickness: int, coord_x: int, coord_y: int, coord_z: int) -> None:
        self.length = length
        self.radius_ext = radius_ext
        self.radius_in = radius_in
        self.radius_hole = radius_hole
        self.thickness = thickness
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.coord_z = coord_z

        self.shield_tag = None
        self.shield_surf = None


    def ajouter(self) -> None:
        shield_in = gmsh.model.occ.addCylinder(self.coord_x, self.coord_y, self.coord_z + self.thickness, 0, 0, self.length -2*self.thickness, self.radius_in)
        shield_out = gmsh.model.occ.addCylinder(self.coord_x, self.coord_y, self.coord_z, 0, 0, self.length, self.radius_ext)
        shield , _=  gmsh.model.occ.cut([(3,shield_out)],[(3,shield_in)])
        shield_hole1 = gmsh.model.occ.addCylinder(self.coord_x, self.coord_y, self.length - self.thickness, 0, 0, self.thickness, self.radius_hole)
        shield_hole2 = gmsh.model.occ.addCylinder(self.coord_x, self.coord_y, self.coord_z, 0, 0, self.thickness, self.radius_hole)
        shield_vol_1, _=  gmsh.model.occ.cut([(3,shield[0][1])],[(3,shield_hole1)])
        shield_vol_2, _=  gmsh.model.occ.cut([(3,shield_vol_1[0][1])],[(3,shield_hole2)])
        self.shield_tag=shield_vol_2[0][1]
        self.shield_surf = gmsh.model.occ.get_surface_loops(self.shield_tag)[1][0]

    
class Data: 
    """
    #1 - Distance between the sield ant the aperture 
    #2 - Distance between the aperture and the cylinders (d in Okayama's paper)
    #3 - Exterior radius of the shield 
    #4 - Inside radius of the shield 
    #5 - Thickness of the shield
    #6 - Radius of the aperture 
    #7 - Thickness of the aperture 
    #8 - Length of the cylinder (l in Okayama's paper)
    #9 - Radius of the elements around the axis (a in Okayama's paper)
    """
    def __init__(self, 
                dist_shield_apert: int, dist_apert_quad: int,
                radius_ext_shield: int, radius_in_shield: int, thickness_shield: int, 
                radius_apert: int, thickness_apert: int, 
                length_cylinder: int, 
                radius_axis: int,
                pot_electrode: int, pot_apert1: int, pot_apert2: int, pot_shield: int, pot_acceleration: int,
                MeshSizeMin: int, MeshSizeMax: int, MeshSizeFromCurvature: int, 
                output_dir: str) -> None:
        self.dist_shield_apert = dist_shield_apert
        self.dist_apert_quad = dist_apert_quad

        self.radius_ext_shield = radius_ext_shield
        self.radius_in_shield = radius_in_shield
        self.thickness_shield = thickness_shield

        self.radius_apert = radius_apert
        self.thickness_apert = thickness_apert

        self.length_cylinder = length_cylinder

        self.radius_axis = radius_axis

        self.pot_electrode = pot_electrode #quadrupole 
        self.pot_apert1 = pot_apert1 #first aperture (pour le champs rond)
        self.pot_apert2 = pot_apert2 #second aperture 
        self.pot_shield = pot_shield #shield (0)
        self.pot_acceleration = pot_acceleration #ion acceleration potential
    
        self.MeshSizeMin = MeshSizeMin
        self.MeshSizeMax = MeshSizeMax
        self.MeshSizeFromCurvature = MeshSizeFromCurvature

        self.output_dir = output_dir
            
        self.group_id = None

        self.total_length = 2*self.dist_shield_apert + 2*self.thickness_shield + 2*self.dist_apert_quad + 2*self.thickness_apert + self.length_cylinder

        self.coord_cylinder_x_or_y = 2*self.radius_axis
        self.coord_cylinder_z = self.thickness_shield + self.dist_shield_apert + self.thickness_apert + self.dist_apert_quad

        self.coord_apert_z1 = self.thickness_shield + self.dist_shield_apert
        self.coord_apert_z2 = self.thickness_shield + self.dist_shield_apert + self.thickness_apert + 2*self.dist_apert_quad + self.length_cylinder

        self.start_shield1 = 0
        self.end_shield1 = self.start_shield1 + self.thickness_shield
        self.start_apert1 = self.end_shield1 + self.dist_shield_apert
        self.end_apert1 = self.start_apert1 + self.thickness_apert
        self.start_cyl = self.end_apert1 + self.dist_apert_quad
        self.end_cyl = self.start_cyl + self.length_cylinder
        self.start_apert2 = self.end_cyl + self.dist_apert_quad
        self.end_apert2 = self.start_apert2 + self.thickness_apert
        self.start_shield2 = self.end_apert2 + self.dist_shield_apert
        self.end_shield2 = self.start_shield2 + self.thickness_shield

class Mesh_Generation: 
    def __init__(self, data: Data) -> None:
        self.data = data

        self.objects = None

    def Initialisation(self) -> None:

        gmsh.initialize()
        gmsh.clear()
        gmsh.model.add("Quadrupole lens")


    def Geometry(self) -> None:

        cylinder1 = Cylinder(self.data.length_cylinder, self.data.radius_axis, self.data.coord_cylinder_x_or_y, 0, self.data.coord_cylinder_z)
        Cylinder.ajouter(cylinder1)

        cylinder2 = Cylinder(self.data.length_cylinder, self.data.radius_axis, -self.data.coord_cylinder_x_or_y, 0, self.data.coord_cylinder_z)
        Cylinder.ajouter(cylinder2)

        cylinder3 = Cylinder(self.data.length_cylinder, self.data.radius_axis, 0, self.data.coord_cylinder_x_or_y, self.data.coord_cylinder_z)
        Cylinder.ajouter(cylinder3)

        cylinder4 = Cylinder(self.data.length_cylinder, self.data.radius_axis, 0, -self.data.coord_cylinder_x_or_y, self.data.coord_cylinder_z)
        Cylinder.ajouter(cylinder4)


        aperture1 = Aperture(self.data.radius_apert, self.data.radius_axis, self.data.thickness_apert, 0, 0, self.data.coord_apert_z1)
        Aperture.ajouter(aperture1)

        aperture2 = Aperture(self.data.radius_apert, self.data.radius_axis, self.data.thickness_apert, 0, 0, self.data.coord_apert_z2)
        Aperture.ajouter(aperture2)


        shield = Shield(self.data.total_length, self.data.radius_ext_shield, self.data.radius_in_shield, self.data.radius_axis, self.data.thickness_shield, 0, 0, 0)
        Shield.ajouter(shield)

        self.objects = aperture1, aperture2, cylinder1, cylinder2, cylinder3, cylinder4, shield


    def Creation_mesh(self) -> None:
        gmsh.model.occ.synchronize()
    
    def Surfaces(self) -> None:
        aperture1, aperture2, cylinder1, cylinder2, cylinder3, cylinder4, shield = self.objects

    #Outwards orientation of the surfaces' normals
        gmsh.model.mesh.setOutwardOrientation(aperture1.apert_tag)
        gmsh.model.mesh.setOutwardOrientation(aperture2.apert_tag)
        gmsh.model.mesh.setOutwardOrientation(cylinder1.cyl_tag)
        gmsh.model.mesh.setOutwardOrientation(cylinder2.cyl_tag)
        gmsh.model.mesh.setOutwardOrientation(cylinder3.cyl_tag)
        gmsh.model.mesh.setOutwardOrientation(cylinder4.cyl_tag)
        gmsh.model.mesh.setOutwardOrientation(shield.shield_tag)

        #Adds surfaces on top of the volumes created 
        group_id_apert1 = gmsh.model.addPhysicalGroup(2, aperture1.apert_surf)
        group_id_apert2 = gmsh.model.addPhysicalGroup(2, aperture2.apert_surf)
        group_id_cyl1 = gmsh.model.addPhysicalGroup(2, cylinder1.cyl_surf)
        group_id_cyl2 = gmsh.model.addPhysicalGroup(2, cylinder2.cyl_surf)
        group_id_cyl3 = gmsh.model.addPhysicalGroup(2, cylinder3.cyl_surf)
        group_id_cyl4 = gmsh.model.addPhysicalGroup(2, cylinder4.cyl_surf)
        group_id_shield = gmsh.model.addPhysicalGroup(2, shield.shield_surf)

        self.data.group_id = group_id_apert1, group_id_apert2, group_id_cyl1, group_id_cyl2, group_id_cyl3, group_id_cyl4, group_id_shield


    def Mesh(self) -> None:
    #Size of the mesh
    #ATTENTION, MeshSizeMin and MeshSizeMax need to be equal
        gmsh.option.set_number('Mesh.MeshSizeMin', self.data.MeshSizeMin)
        gmsh.option.set_number('Mesh.MeshSizeMax', self.data.MeshSizeMax)
        gmsh.option.set_number('Mesh.MeshSizeFromCurvature', self.data.MeshSizeFromCurvature)

        gmsh.model.mesh.generate(2)

    def Finalize(self) -> None:
    #Creates a file .msh
        mesh_path = os.path.join(self.data.output_dir, "mesh_quadrupole.msh")

        gmsh.write(mesh_path)
        print("Mesh saved to:", mesh_path)

    #Opens a terminal to see the geometry (il faut préalablement installer une extension dans Visual studio code intitulé "STL Viewer")
        gmsh.fltk.run()

        gmsh.finalize()


class Calculation_field:
    def __init__(self, data: Data) -> None:
        self.data =  data

        #Importation of the mesh
        self.mesh_path = os.path.join(self.data.output_dir, "mesh_quadrupole.msh")
        self.grid = bempp.import_grid(self.mesh_path)
        # --- spaces ---
        self.dp0_space = bempp.function_space(self.grid, "DP", 0)
        self.p1_space  = bempp.function_space(self.grid, "P", 1)

        # --- operators ---
        self.identity = bempp.operators.boundary.sparse.identity(self.p1_space, self.p1_space, self.dp0_space)

        self.dlp = bempp.operators.boundary.laplace.double_layer(self.p1_space, self.p1_space, self.dp0_space)

        self.slp = bempp.operators.boundary.laplace.single_layer(self.dp0_space, self.p1_space, self.dp0_space)

        self.dirichlet_fun = None
        self.neumann_fun = None
        self.u_evaluated = None
        self.points = None

        self.E_eval = None
        self.D2_eval = None
        self.D3_eval = None
        self.D4_eval = None


    def Mesh_Importation(self) -> None:
        plt.clf()

        
    def Potentials_settings(self) -> None:
    #Settings of the potentials
        pot_apert1 = self.data.pot_apert1
        pot_apert2 = self.data.pot_apert2
        pot_electrode = self.data.pot_electrode
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
        def dirichlet_data(x, n, domain_index, result):
            if domain_index == apert1: #potential of the 1st aperture
                result[0]=pot_apert1
                
            elif domain_index == apert2: #potential of the 2nd aperture
                result[0]=pot_apert2
            
            elif domain_index == elec1: #potential of the 1st electrode
                result[0]=pot_electrode

            elif domain_index == elec2: #potential of the 2nd electrode
                result[0]=pot_electrode

            elif domain_index == elec3: #potential of the 3rd electrode
                result[0]=-pot_electrode

            elif domain_index == elec4: #potential of the 4th electrode
                result[0]=-pot_electrode

            elif domain_index == shield: #potential of the shield
                result[0]=pot_shield

        self.dirichlet_fun = bempp.GridFunction(self.p1_space, fun=dirichlet_data) 

    
   
    def Matrix_inversion(self) -> None:
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


    def Derivatives(self) -> None:
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


    def Potential_exportation(self) -> None: 
        try:
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython is not None:
                ipython.run_line_magic("matplotlib", "inline")
                ipython = True

        except NameError:
            ipython = False

        savefile = os.path.join(self.data.output_dir, "potentiel_quadrupole_VF.npz")
    
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
        pot_electrode = self.data.pot_electrode,
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


    def Potential_axis_printing(self) -> None:
    #Printing of the solution --> printing of the potential along the z axis ~ 0V
        plt.plot(self.points[2], self.u_evaluated[0])
        plt.title("Potentiel du quadrupole d'Okayama le long de l'axe Z")
        plt.xlabel("Position en z (mm)")
        plt.ylabel("Potentiel (V)")
        plt.show()

#Class to extract all the necessary informations from the files
class Extracted_data:
    def __init__(self, file_path: str) -> None:

        # file load 
        data = np.load(file_path)
        
        # axes data
        self.points = data["points"]
        self.axe_z = data["points"][2] 
        
        # Potential 
        self.potential = data["potential"]    # Le tableau complet [V]

        # Identifiants de groupes (Physical Groups de GMSH)
        self.group_id_ap1 = data["group_id_ap1"]
        self.group_id_ap2 = data["group_id_ap2"]
        self.group_id_cyl1 = data["group_id_cyl1"]
        self.group_id_cyl2 = data["group_id_cyl2"]
        self.group_id_cyl3 = data["group_id_cyl3"]
        self.group_id_cyl4 = data["group_id_cyl4"]
        self.group_id_shield = data["group_id_shield"]

        #Mesh parameters
        self.MeshSizeMin = data["MeshSizeMin"]
        self.MeshSizeMax = data["MeshSizeMax"]
        self.MeshSizeFromCurvature = data["MeshSizeFromCurvature"]

        #Output directory
        self.output_dir = data["output_dir"]

        # Dimensions et géométrie
        self.dist_shield_apert = data["dist_shield_apert"]
        self.dist_apert_quad = data["dist_apert_quad"]
        self.radius_ext_shield = data["radius_ext_shield"]
        self.radius_in_shield = data["radius_in_shield"]
        self.thickness_shield = data["thickness_shield"]
        self.radius_apert = data["radius_apert"]
        self.thickness_apert = data["thickness_apert"]
        self.length_cylinder = data["length_cylinder"]
        self.radius_axis = data["radius_axis"]
        self.total_length = data["total_length"]

        #Coordinates
        self.coord_cylinder_x_or_y = data["coord_cylinder_x_or_y"]
        self.coord_cylinder_z = data["coord_cylinder_z"]
        self.coord_apert_z1 = data["coord_apert_z1"]
        self.coord_apert_z2 = data["coord_apert_z2"]
        self.start_shield1 = data["start_shield1"]
        self.end_shield1 = data["end_shield1"]
        self.start_apert1 = data["start_apert1"]
        self.end_apert1 = data["end_apert1"]
        self.start_cyl = data["start_cyl"]
        self.end_cyl = data["end_cyl"]
        self.start_apert2 = data["start_apert2"]
        self.end_apert2 = data["end_apert2"]
        self.start_shield2 = data["start_shield2"]
        self.end_shield2 = data["end_shield2"]
        
        # Valeurs des tensions appliquées
        self.Vacceleration = data["pot_acceleration"]
        self.Velectrode = data["pot_electrode"]
        self.Vapert1 = data["pot_apert1"]
        self.Vapert2 = data["pot_apert2"]
        self.Vshield = data["pot_shield"]

        #Derivatives
        self.D0 = data["potential"][0]      # Le potentiel sur l'axe
        self.D1 = data["E_eval"]               # Champ électrique [V/mm]
        self.D2 = data["D2_eval"]             # 2ème dérivée [V/mm^2]
        self.D3 = data["D3_eval"]             # 3ème dérivée [V/mm^3
        self.D4 = data["D4_eval"]             # 4ème dérivée [V/mm^4
        

#Creation of a class that describes characteristic values for the fit function
class Fit_constants:
    def __init__(self, a0: int, b0: int, b2: int, a4: int, b41: int, b42: int, z0: int, a: int, data: Extracted_data) -> None:
        # Paramètres pour la composante lentille ronde k0(Z) :
        # k0(Z) = a0 * exp[-(Z/b0)^2]
        self.a0 = a0 
        self.b0 = b0 
        # Paramètres pour la composante quadripolaire k2(Z) :
        self.b2 = b2

        # Paramètres pour la composante octupolaire k4(Z) :
        self.a4 = a4
        self.b41 = b41
        self.b42 = b42
        self.z0 = z0
        # Paramètre géométrique, rayon elecrtode 
        self.a = a

        self.data = data

        self.k0 = None
        self.k2 = None
        self.k4 = None



    def fonction_fit(self) -> None:

        milieu_cyl = 0.5 * (self.data.start_cyl + self.data.end_cyl)

        #on translte tout
        z_ref = self.data.axe_z - milieu_cyl

        # un axe pour chaque k
        z_quad_center = 0                          # Milieu du cylindre = 0
        z_quad_edge   = self.data.length_cylinder / 2        # Bord du cylindre
        z_ap_center   = z_quad_edge + self.data.dist_apert_quad + self.data.thickness_apert / 2 # Milieu ouverture

        Z_k2 = z_ref - z_quad_center     # pour le quadrupôle
        Z_k4 = z_ref - z_quad_edge       # pour l'octupôle
        Z_k0 = z_ref - z_ap_center       # pour la lentille ronde (aperture)

        # k0 (Lentille ronde) 
        a0, b = 0.80751, 5.08
        self.k0 = a0 * np.exp(-(Z_k0**2) / b**2)

        # k2 (Quadrupôle)
        Z0, b2 = 5, 2.54
        self.k2 = np.where(np.abs(Z_k2) <= Z0, 1, np.exp(-(np.abs(Z_k2) - Z0)**2 / b2**2))

        # k4 (Octupôle)
        a4, b1, b2_k4 = 0.03891461, 3.113, 2.015
        self.k4 = np.where(Z_k4 <= 0, a4 / (1 + (Z_k4**2 / b1**2))**2, a4 * np.exp(-(Z_k4**2) / b2_k4**2))


class Decomposition:
    def __init__(self, data: Extracted_data) -> None:
        self.data = data 

        self.Phi0_maj = None
        self.Phi1_maj = None
        self.Phi2_maj = None
        self.Phi3_maj = None
        self.Phi4_maj = None

        self.Phi0_fit = None
        self.Phi2_fit = None
        self.Phi4_fit = None


    def composantes(self) -> None:
        self.Phi0_maj = -self.data.D0  # potentiel monopolaire sur l’axe
        self.Phi1_maj = self.data.D1[0] 
        self.Phi2_maj = (1/4)*(self.data.D2[0] - self.data.D2[3])
        self.Phi3_maj = (1/24) * (self.data.D3[0] - 3*self.data.D3[3])
        self.Phi4_maj = (1/192)* (self.data.D4[0] + self.data.D4[10] - 6*self.data.D4[3])

        self.Phi0_fit = self.Phi0_maj / self.data.Vapert1
        self.Phi2_fit = self.Phi2_maj *((self.data.radius_axis**2)/ self.data.Velectrode)
        self.Phi4_fit = self.Phi4_maj *((self.data.radius_axis**4)/ self.data.Vapert1)
        
        Phi2_max = np.max(np.abs(self.Phi2_fit))

        self.Phi0_fit = self.Phi0_fit/Phi2_max
        self.Phi2_fit = self.Phi2_fit/Phi2_max
        self.Phi4_fit = self.Phi4_fit/Phi2_max

class Affichage:
    def __init__(self, data: Extracted_data, decomposition: Decomposition, fit: Fit_constants) -> None:
        self.data = data
        self.decomposition = decomposition
        self.fit = fit

        self.ax = None

    def trace_geo(self, ax) -> None:

        ax.axvspan(self.data.start_shield1, self.data.end_shield1, color='red', alpha=0.3, label='Shield')
        ax.axvspan(self.data.start_apert1, self.data.end_apert1, color='blue', alpha=0.3, label='Aperture')
        ax.axvspan(self.data.start_cyl, self.data.end_cyl, color='green', alpha=0.3, label='Electrodes')
        ax.axvspan(self.data.start_apert2, self.data.end_apert2, color='blue', alpha=0.3)
        ax.axvspan(self.data.start_shield2, self.data.end_shield2, color='red', alpha=0.3)

    def graphe_composantes(self) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))

        self.trace_geo(ax)

        #Printing of the different composants of the potential
        plt.plot(self.data.axe_z, -self.decomposition.Phi0_maj, label=r'$\Phi_0 $ $[V]$', color='crimson')
        plt.plot(self.data.axe_z, self.decomposition.Phi1_maj, label=r'$\Phi_1$ $[V/mm^1]$', color='darkviolet')
        plt.plot(self.data.axe_z, self.decomposition.Phi2_maj, label=r'$\Phi_2$ $[V/mm^2]$', color='green')
        plt.plot(self.data.axe_z, self.decomposition.Phi3_maj, label=r'$\Phi_3$ $[V/mm^3]$', color='gold')
        plt.plot(self.data.axe_z, 10*self.decomposition.Phi4_maj, label=r'$\Phi_4 *100$ $[V/mm^4]$', color='royalblue')
        plt.xlabel("z (mm)")
        plt.ylabel("Potentiel")
        plt.title("Décomposition multipolaire sur l’axe")
        plt.grid()
        plt.legend()
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
        plt.tight_layout()
        plt.show()

    def graphe_fit(self) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        self.trace_geo(ax)

        #Printing of the different composants of the potential
        plt.plot(self.data.axe_z, self.decomposition.Phi0_fit, label=r'$\Phi_0 $ $[V]$', color='crimson')
        plt.plot(self.data.axe_z, self.decomposition.Phi2_fit, label=r'$\Phi_2$ $[V/mm^2]$', color='green')
        plt.plot(self.data.axe_z, 10*self.decomposition.Phi4_fit, label=r'$\Phi_4 *10$ $[V/mm^4]$', color='royalblue')
        plt.plot(self.data.axe_z, self.fit.k0, label=r'k0', color='crimson', linestyle='dashed')
        plt.plot(self.data.axe_z, self.fit.k2, label=r'k2', color='green', linestyle='dashed')
        plt.plot(self.data.axe_z, 10*self.fit.k4, label=r'k4 *10', color='royalblue', linestyle='dashed')
        plt.xlabel("z (mm)")
        plt.ylabel("Potentiel")
        plt.title("Décomposition multipolaire sur l’axe et fonctions de fit")
        plt.grid()
        plt.legend()
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
        plt.tight_layout()
        plt.show()
