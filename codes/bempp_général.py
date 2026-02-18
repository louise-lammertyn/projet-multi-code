
import os

os.environ['PYOPENCL_COMPILER_OUTPUT']='1'
os.environ['PYOPENCL_NO_CACHE'] = '1'

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
from dataclasses import dataclass
from scipy.interpolate import interp1d

opencl_kernels.show_available_platforms_and_devices()
opencl_kernels.set_default_cpu_device(0,1)


@dataclass
class dimension:
    """
    Docstring pour dimension
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
    dist_shield_apert: float
    dist_apert_quad: float
    radius_ext_shield: float
    radius_in_shield: float
    thickness_shield: float
    radius_apert: float
    thickness_apert: float
    length_cylinder: float
    radius_axis: float



@dataclass
class Potentials:
    """
    Docstring pour Potentials:
    Potentiel electrode, potentiel aperture 1, potentiel aperture2, potentiel shied, pot acceleration en v
    """
    pot_acceleration: float
    pot_shield: float = 0.0
    
    @property
    def pot_electrode(self) -> float:
        return -0.0299087 * self.pot_acceleration

    @property
    def pot_apert1(self) -> float:
        return -0.18808 * self.pot_acceleration

    @property
    def pot_apert2(self) -> float:
        return 0.10918 * self.pot_acceleration
    
# construction géometrie -


class Cylinder:
    def __init__(self, length, radius, coord_x, coord_y, coord_z):
        self.length, self.radius = length, radius
        self.coord_x, self.coord_y, self.coord_z = coord_x, coord_y, coord_z
        self.tag, self.surf = None, None

    def ajouter(self):
        # On utilise OpenCASCADE (occ)
        self.tag = gmsh.model.occ.addCylinder(self.coord_x, self.coord_y, self.coord_z, 0, 0, self.length, self.radius)
        gmsh.model.occ.synchronize() # CRUCIAL : synchronise avant de demander les boundaries
        boundary = gmsh.model.getBoundary([(3, self.tag)], combined=False)
        self.surf = [tag for dim, tag in boundary if dim == 2]

class Aperture: 
    def __init__(self, radius_ext, radius_in, thickness, coord_x, coord_y, coord_z):
        self.radius_ext, self.radius_in, self.thickness = radius_ext, radius_in, thickness
        self.coord_x, self.coord_y, self.coord_z = coord_x, coord_y, coord_z
        self.tag, self.surf = None, None

    def ajouter(self):
        c1 = gmsh.model.occ.addCylinder(self.coord_x, self.coord_y, self.coord_z, 0, 0, self.thickness, self.radius_ext)
        c2 = gmsh.model.occ.addCylinder(self.coord_x, self.coord_y, self.coord_z, 0, 0, self.thickness, self.radius_in)
        vol, _ = gmsh.model.occ.cut([(3, c1)], [(3, c2)])
        self.tag = vol[0][1]
        gmsh.model.occ.synchronize()
        boundary = gmsh.model.getBoundary([(3, self.tag)], combined=False)
        self.surf = [tag for dim, tag in boundary if dim == 2]

class Shield:
    def __init__(self, length, radius_ext, radius_in, radius_hole, thickness, coord_x, coord_y, coord_z):
        self.length, self.radius_ext, self.radius_in, self.radius_hole = length, radius_ext, radius_in, radius_hole
        self.thickness, self.coord_x, self.coord_y, self.coord_z = thickness, coord_x, coord_y, coord_z
        self.tag, self.surf = None, None

    def ajouter(self):
        s_out = gmsh.model.occ.addCylinder(self.coord_x, self.coord_y, self.coord_z, 0, 0, self.length, self.radius_ext)
        s_in = gmsh.model.occ.addCylinder(self.coord_x, self.coord_y, self.coord_z + self.thickness, 0, 0, self.length - 2*self.thickness, self.radius_in)
        sh, _ = gmsh.model.occ.cut([(3, s_out)], [(3, s_in)])
        h1 = gmsh.model.occ.addCylinder(self.coord_x, self.coord_y, self.coord_z, 0, 0, self.thickness, self.radius_hole)
        h2 = gmsh.model.occ.addCylinder(self.coord_x, self.coord_y, self.coord_z + self.length - self.thickness, 0, 0, self.thickness, self.radius_hole)
        res, _ = gmsh.model.occ.cut([(3, sh[0][1])], [(3, h1), (3, h2)])
        self.tag = res[0][1]
        gmsh.model.occ.synchronize()
        boundary = gmsh.model.getBoundary([(3, self.tag)], combined=False)
        self.surf = [tag for dim, tag in boundary if dim == 2]

class Quadrupole:
    def __init__(self, potentials: Potentials, dims: dimension) -> None:
        """
        Docstring pour quadrupole
        Potentials type Potentials et dims type dimension 
        """
        self.p = potentials
        self.d = dims
        self.total_length = (2 * dims.dist_shield_apert + 2 * dims.thickness_shield +  2 * dims.dist_apert_quad + 2 * dims.thickness_apert + dims.length_cylinder)

        #coordonée cylinder 
        
        self.x_or_y = 2*dims.radius_axis
        self.z = dims.thickness_shield + dims.dist_shield_apert + dims.thickness_apert + dims.dist_apert_quad

        #coord_apert

        self.z1 = dims.thickness_shield + dims.dist_shield_apert
        self.z2 = dims.thickness_shield + dims.dist_shield_apert + dims.thickness_apert + 2*dims.dist_apert_quad + dims.length_cylinder

        #coord total
        self.start_shield1 = 0
        self.end_shield1 = self.start_shield1 + dims.thickness_shield
        self.start_apert1 = self.end_shield1 + dims.dist_shield_apert
        self.end_apert1 = self.start_apert1 + dims.thickness_apert

        self.start_cyl = self.end_apert1 + dims.dist_apert_quad
        self.end_cyl = self.start_cyl + dims.length_cylinder

        self.start_apert2 = self.end_cyl + dims.dist_apert_quad
        self.end_apert2 = self.start_apert2 + dims.thickness_apert

        self.start_shield2 = self.end_apert2 + dims.dist_shield_apert
        self.end_shield2 = self.start_shield2 + dims.thickness_shield
    
class QuadrupoleMesh(Quadrupole):
    def __init__(self, potentials: Potentials, dims: dimension):
        #initisitaiton classe parente

        super().__init__(potentials, dims)
        
        # Initialisation 
        self.group_id_apert1 = None
        self.group_id_apert2 = None
        self.group_id_cyl1 = None
        self.group_id_cyl2 = None
        self.group_id_cyl3 = None
        self.group_id_cyl4 = None
        self.group_id_shield = None

    def generate_mesh(self): 
        gmsh.initialize()
        gmsh.model.add("Quadrupole_Lens")

        # --- Cylindres : On remplit les IDs un par un ---
        c1 = Cylinder(self.d.length_cylinder, self.d.radius_axis, self.x_or_y, 0, self.start_cyl)
        c1.ajouter()
        # IMPORTANT : On stocke l'ID dans self.group_id_cyl1
        self.group_id_cyl1 = gmsh.model.addPhysicalGroup(2, c1.surf, name="Cyl1")
        
        c2 = Cylinder(self.d.length_cylinder, self.d.radius_axis, -self.x_or_y, 0, self.start_cyl)
        c2.ajouter()
        self.group_id_cyl2 = gmsh.model.addPhysicalGroup(2, c2.surf, name="Cyl2")
        
        c3 = Cylinder(self.d.length_cylinder, self.d.radius_axis, 0, self.x_or_y, self.start_cyl)
        c3.ajouter()
        self.group_id_cyl3 = gmsh.model.addPhysicalGroup(2, c3.surf, name="Cyl3")
        
        c4 = Cylinder(self.d.length_cylinder, self.d.radius_axis, 0, -self.x_or_y, self.start_cyl)
        c4.ajouter()
        self.group_id_cyl4 = gmsh.model.addPhysicalGroup(2, c4.surf, name="Cyl4")

        # --- Apertures ---
        ap1 = Aperture(self.d.radius_ext_shield, self.d.radius_apert, self.d.thickness_apert, 0, 0, self.z1)
        ap1.ajouter()
        self.group_id_apert1 = gmsh.model.addPhysicalGroup(2, ap1.surf, name="Aperture1")

        ap2 = Aperture(self.d.radius_ext_shield, self.d.radius_apert, self.d.thickness_apert, 0, 0, self.z2)
        ap2.ajouter()
        self.group_id_apert2 = gmsh.model.addPhysicalGroup(2, ap2.surf, name="Aperture2")

        # --- Shield ---
        sh = Shield(self.total_length, self.d.radius_ext_shield, self.d.radius_in_shield, self.d.radius_axis, self.d.thickness_shield, 0, 0, 0)
        sh.ajouter()
        self.group_id_shield = gmsh.model.addPhysicalGroup(2, sh.surf, name="Shield")

        # Reste du code (Orientation, generation, write...)
        for obj in [c1, c2, c3, c4, ap1, ap2, sh]:
            gmsh.model.mesh.setOutwardOrientation(obj.tag)

        gmsh.option.set_number('Mesh.MeshSizeMin',4 )
        gmsh.option.set_number('Mesh.MeshSizeMax', 4)
        gmsh.model.mesh.generate(2)
        gmsh.write("mesh_quadrupole.msh")
        gmsh.fltk.run()
        gmsh.finalize()


   
    #change à partir de ici pour fichier mesh 

class Solver(Quadrupole):
    def __init__(self, mesh_obj: QuadrupoleMesh, filename: str = "mesh_quadrupole.msh"):
        """
        docstring for solver -> mesh_obj(par défaut on utilise le quadrupole au quadrupole), ficher en .msh
        """
        # Initialisation du parent avec les objets stockés dans le mesh
        super().__init__(mesh_obj.p, mesh_obj.d)
        self.mesh_obj = mesh_obj
        
        # Importation de la grille
        self.grid = bempp.import_grid(filename)
        
        # Espaces 
        self.dp0_space = bempp.function_space(self.grid, "DP", 0)
        self.p1_space = bempp.function_space(self.grid, "P", 1)
        
        # Initialisation des variables de stockage
        self.u_evaluated = None
        self.neumann_fun = None
        self.points = None
        self.E_eval = None
        self.D2_eval = None
        self.D3_eval = None
        self.D4_eval = None
        print("initialisation faite")

    
  

    def solve(self, tol=1e-8):
        print("debut calcul")
        
        identity = bempp.operators.boundary.sparse.identity(self.p1_space, self.p1_space, self.dp0_space)
        dlp = bempp.operators.boundary.laplace.double_layer(self.p1_space, self.p1_space, self.dp0_space)
        slp = bempp.operators.boundary.laplace.single_layer(self.dp0_space, self.p1_space, self.dp0_space)


        id_ap1 = self.mesh_obj.group_id_apert1
        id_ap2 = self.mesh_obj.group_id_apert2
        id_c1 = self.mesh_obj.group_id_cyl1
        id_c2 = self.mesh_obj.group_id_cyl2
        id_c3 = self.mesh_obj.group_id_cyl3
        id_c4 = self.mesh_obj.group_id_cyl4
        id_sh = self.mesh_obj.group_id_shield

        # --- POTENTIELS ---
        v_ap1, v_ap2 = self.p.pot_apert1, self.p.pot_apert2
        v_elec, v_sh = self.p.pot_electrode, self.p.pot_shield

        
        @bempp.real_callable
        #Setting of the potential on the different elements of the geometry
        def dirichlet_data(x, n, domain_index, result):
            if domain_index == id_ap1: #potential of the 1st aperture
                result[0]=v_ap1
                
            elif domain_index == id_ap2: #potential of the 2nd aperture
                result[0]=v_ap2

            elif domain_index == id_c1: #potential of the 1st electrode
                result[0]=v_elec

            elif domain_index == id_c2: #potential of the 2nd electrode
                result[0]=v_elec

            elif domain_index == id_c3: #potential of the 3rd electrode
                result[0]=-v_elec

            elif domain_index == id_c3: #potential of the 4th electrode
                result[0]=-v_elec

            elif domain_index == id_sh: #potential of the shield
                result[0]=v_sh
            else: result[0] = 0.0



        os.environ['PYOPENCL_COMPILER_OUTPUT']='1'
        os.environ['PYOPENCL_NO_CACHE'] = '1'
    

        dirichlet_fun = bempp.GridFunction(self.p1_space, fun=dirichlet_data)
        rhs = (-0.5 * identity + dlp) * dirichlet_fun
        self.neumann_fun, _ = bempp.linalg.cg(slp, rhs, tol=tol)
        print("fini")

    def evaluate(self, n_points=100):
        
        #initialsition de notre axe et nos points 
        z_range = np.linspace(0, self.total_length, n_points)
        self.points = np.vstack([np.zeros(n_points), np.zeros(n_points), z_range])

        # Potentiel (U)
        slp_pot = bempp.operators.potential.laplace.single_layer(self.dp0_space, self.points)
        self.u_evaluated = -slp_pot * self.neumann_fun

        # Gradients et Dérivées d'ordre supérieur
        print("Calcul des dérivées")
        E_op = bempp.operators.potential.laplace.single_layer_gradient(self.dp0_space, self.points, device_interface="opencl")
        self.E_eval = -E_op * self.neumann_fun

        D2_op = bempp.operators.potential.laplace.single_layer_2nd_deriv(self.dp0_space, self.points, device_interface="opencl")
        self.D2_eval = D2_op * self.neumann_fun

        D3_op = bempp.operators.potential.laplace.single_layer_3rd_deriv(self.dp0_space, self.points, device_interface="opencl")
        self.D3_eval = D3_op * self.neumann_fun

        D4_op = bempp.operators.potential.laplace.single_layer_4th_deriv(self.dp0_space, self.points, device_interface="opencl")
        self.D4_eval = D4_op * self.neumann_fun

    def save_results(self, savefile="potentiel_quadrupole_VF"):
        """Sauvegarde les données au format .npz ."""
        if self.u_evaluated is None:
            self.evaluate()

        np.savez_compressed(
            savefile,
            points=self.points,
            potential=self.u_evaluated,
            group_id_ap1=self.mesh_obj.group_id_apert1,
            group_id_ap2=self.mesh_obj.group_id_apert2,
            group_id_cyl1=self.mesh_obj.group_id_cyl1,
            group_id_cyl2=self.mesh_obj.group_id_cyl2,
            group_id_cyl3=self.mesh_obj.group_id_cyl3,
            group_id_cyl4=self.mesh_obj.group_id_cyl4,
            group_id_shield=self.mesh_obj.group_id_shield,
            dimensions=np.array((
                self.d.dist_shield_apert, self.d.dist_apert_quad, self.d.radius_ext_shield, 
                self.d.radius_in_shield, self.d.thickness_shield, self.d.radius_apert, 
                self.d.thickness_apert, self.d.length_cylinder, self.d.radius_axis, 
                self.total_length, self.x_or_y, 0, 0, 0 
            )),
            potentials=(
                self.p.pot_electrode, self.p.pot_apert1, self.p.pot_apert2, 
                self.p.pot_shield, self.p.pot_acceleration
            ),
            E_eval=self.E_eval,
            D2_eval=self.D2_eval,
            D3_eval=self.D3_eval,
            D4_eval=self.D4_eval,
        )
        print(f"Saved potential to {savefile}.npz")
