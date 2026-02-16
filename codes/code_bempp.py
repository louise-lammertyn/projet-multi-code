#%%
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
#%%


import os

os.environ['PYOPENCL_COMPILER_OUTPUT']='1'
os.environ['PYOPENCL_NO_CACHE'] = '1'

#Importation of all the necessary packages
import os
import numpy as np
from matplotlib import pylab as plt
from IPython import get_ipython
import matplotlib
from dataclasses import dataclass
from scipy.interpolate import interp1d

#%%
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
        Classe générale du quadrpole 
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
        # On appelle le constructeur de la classe parente
        super().__init__(potentials, dims)
        
        # Initialisation des attributs
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


   
    

class QuadrupoleSolver(Quadrupole):
    def __init__(self, mesh_obj: QuadrupoleMesh, filename: str = "mesh_quadrupole.msh"):
        """
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
        """Résout le système BEM."""
        identity = bempp.operators.boundary.sparse.identity(self.p1_space, self.p1_space, self.dp0_space)
        dlp = bempp.operators.boundary.laplace.double_layer(self.p1_space, self.p1_space, self.dp0_space)
        slp = bempp.operators.boundary.laplace.single_layer(self.dp0_space, self.p1_space, self.dp0_space)

        # --- EXTRACTION DES IDS EN VARIABLES SIMPLES (Crucial pour Numba) ---
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

        print("debut calcul2")
        

        @bempp.real_callable
        def dirichlet_data(x, n, domain_index, result):
            if domain_index == id_ap1: result[0] = v_ap1
            elif domain_index == id_ap2: result[0] = v_ap2
            elif domain_index == id_c1 or domain_index == id_c2: result[0] = v_elec
            elif domain_index == id_c3 or domain_index == id_c4: result[0] = -v_elec
            elif domain_index == id_sh: result[0] = v_sh
            else: result[0] = 0.0

        os.environ['PYOPENCL_COMPILER_OUTPUT']='1'
        os.environ['PYOPENCL_NO_CACHE'] = '1'
        print("cachde")

        dirichlet_fun = bempp.GridFunction(self.p1_space, fun=dirichlet_data)
        rhs = (-0.5 * identity + dlp) * dirichlet_fun
        self.neumann_fun, _ = bempp.linalg.cg(slp, rhs, tol=tol)
        print("jkzhl")

    def evaluate(self, n_points=100):
        """Calcule le potentiel et toutes les dérivées sur l'axe Z."""
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

class Extraction():
    def __init__(self, file_path):
        # On initialise la classe parente
        
        # Chargement des données
        data = np.load(file_path)
        self.points = data["points"]
        self.axe_z = data["points"][2]
        
        # Potentiels et dérivées (OpenCL)
        self.phi0 = data["potential"][0] # Potentiel sur l'axe
        self.E = data["E_eval"]              # [Ex, Ey, Ez]
        self.D2 = data["D2_eval"]            # 6 composantes (xx, xy, xz, yy, yz, zz)
        self.D3 = data["D3_eval"]            # 10 composantes
        self.D4 = data["D4_eval"]            # 15 composantes
        
        # Initialisation des résultats de décomposition
        self.Phi0, self.Phi1, self.Phi2, self.Phi3, self.Phi4 = (None,) * 5
        self.Phi0_fit, self.Phi2_fit, self.Phi4_fit = (None,) * 3
        self.phi0_norm, self.phi2_norm, self.phi4_norm = (None,)*3

        #initialisation fonction de fit d'okayama
        self.k0, self.k2, self.k4 = (None,) *3

        
        # Dimensions et géométrie
        self.dimension = data["dimensions"] #accées au dimension de la géométrie (plus d'info en haut)
        
        # Identifiants de groupes (Physical Groups de GMSH)
        self.group_id_ap1 = data["group_id_ap1"]
        self.group_id_ap2 = data["group_id_ap2"]
        self.group_id_cyl1 = data["group_id_cyl1"]
        self.group_id_cyl2 = data["group_id_cyl2"]
        self.group_id_cyl3 = data["group_id_cyl3"]
        self.group_id_cyl4 = data["group_id_cyl4"]
        self.group_id_shield = data["group_id_shield"]
        
        # Valeurs des tensions appliquées
        self.Va = data["potentials"][4]
        self.Vq2 = data["potentials"][0]
        self.Vapert1 = data["potentials"][1]
        self.Vapert2 = data["potentials"][2]
        self.Vshield = data["potentials"][3]
        
        # Tuple des potentiels (si besoin de la structure originale)
        self.potentials = data["potentials"]

    def derive(self):
            
        self.D2zphi0 = self.D2[5]  #phi_0''
        self.D1zphi0 = self.E[2]   #phi_0'
        
    

    def decompose(self):
        """Calcule les composantes de la décomposition multipolaire."""
        # Phi0 : Monopolaire (Potentiel sur l'axe)
        self.Phi0 = -self.phi0
        
        # Phi1 : Dipolaire (Ez)
        self.Phi1 = self.E[2]
        
        # Phi2 : Quadrupolaire - 
        self.Phi2 = (1/4) * (self.D2[0] - self.D2[3])
        
        # Phi3 : 
        self.Phi3 = (1/24) * (self.D3[0] - 3*self.D3[3])
        
        # Phi4 : Octupolaire
        self.Phi4 = (1/192) * (self.D4[0] + self.D4[10] - 6*self.D4[3])

        
        self.Phi0_fit = self.Phi0 / abs(self.p.pot_apert1)
        self.Phi2_fit = (self.Phi2 / abs(self.p.pot_electrode)) * (self.d.radius_axis**2)
        self.Phi4_fit = (self.Phi4 / abs(self.p.pot_apert1)) * (self.d.radius_axis**4)
        

    def fit_functions(self):
       
        # Calcul du milieu des cylindres pour la translation (z_ref)
        milieu_cyl = self.start_cyl + (self.d.length_cylinder / 2)
        z_ref = self.axe_z - milieu_cyl

        # Paramètres géométriques
        z_quad_center = 0
        z_quad_edge = self.d.length_cylinder / 2
        z_ap_center = z_quad_edge + self.d.dist_apert_quad + (self.d.thickness_apert / 2)

        z_k2 = z_ref - z_quad_center     # pour le quadrupôle
        z_k4 = z_ref - z_quad_edge       # pour l'octupôle
        z_k0 = z_ref - z_ap_center       # pour la lentille ronde (aperture)


        # k0 (Lentille ronde)
        a0, b = 0.80751, 5.08
        self.k0 = a0 * np.exp(-((z_k0)**2) / b**2)

        # k2 (Quadrupôle)
        Z0, b2 = 5.0, 2.54
        self.k2 = np.where(np.abs(z_k2) <= Z0, 1.0, 
                      np.exp(-(np.abs(z_k2) - Z0)**2 / b2**2))

        # k4 (Octupôle)
        a4, b1, b2_4 = 0.03891461, 3.113, 2.015
        self.k4 = np.where(z_k4 <= 0, 
                      a4 / (1 + (z_k4**2 / b1**2))**2,
                      a4 * np.exp(-(z_k4**2) / b2_4**2))
        
        

    def plot_decomposition(self):
        
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Tracé des zones géométriques (via les propriétés héritées de Quadrupole)
        ax.axvspan(self.start_shield1, self.end_shield1, color='red', alpha=0.2, label='Shield')
        ax.axvspan(self.start_apert1, self.end_apert1, color='blue', alpha=0.1, label='Aperture')
        ax.axvspan(self.start_cyl, self.end_cyl, color='green', alpha=0.3, label='Electrode')
        ax.axvspan(self.start_apert2, self.end_apert2, color='blue', alpha=0.3)
        ax.axvspan(self.start_shield2, self.end_shield2, color='red', alpha=0.3)
        
        # Tracé des résultats BEM
        ax.plot(self.axe_z, -self.Phi0_fit, 'r-', label=r'$\Phi_0$ (BEM)')
        ax.plot(self.axe_z, -self.Phi2_fit, 'g-', label=r'$\Phi_2$ (BEM)')
        ax.plot(self.axe_z, -10*self.Phi4_fit, 'b-', label=r'$\Phi_4 \times 10$ (BEM)')
        
        # Tracé des fonctions de FIT
        ax.plot(self.axe_z,self. k0, 'r--', alpha=0.7, label='k0 (Fit)')
        ax.plot(self.axe_z,self. k2, 'g--', alpha=0.7, label='k2 (Fit)')
        ax.plot(self.axe_z,10*self.k4, 'b--', alpha=0.7, label='k4 $\times 10$ (Fit)')
        
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("Amplitude normalisée")
        ax.set_title("Comparaison Décomposition BEM vs Fonctions de Fit")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True, linestyle=':')
        plt.tight_layout()
        plt.show()

    #normalisation pour avoir phi2 à 1


    def normalisation (self):
        phi2_max = np.max(np.abs(self.Phi2_fit))

        self.phi0_norm = self.Phi0_fit/phi2_max
        self.phi2_norm = self.Phi2_fit/phi2_max
        self.phi4_norm = self.Phi4_fit/phi2_max



    def plot_normalisation(self):
        
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Tracé des zones géométriques (via les propriétés héritées de Quadrupole)
        ax.axvspan(self.start_shield1, self.end_shield1, color='red', alpha=0.2, label='Shield')
        ax.axvspan(self.start_apert1, self.end_apert1, color='blue', alpha=0.1, label='Aperture')
        ax.axvspan(self.start_cyl, self.end_cyl, color='green', alpha=0.3, label='Electrode')
        ax.axvspan(self.start_apert2, self.end_apert2, color='blue', alpha=0.3)
        ax.axvspan(self.start_shield2, self.end_shield2, color='red', alpha=0.3)
        
        # Tracé des résultats BEM
        ax.plot(self.axe_z, -self.phi0_norm, 'r-', label=r'$\Phi_0$ (BEM norm)')
        ax.plot(self.axe_z, -self.phi2_norm, 'g-', label=r'$\Phi_2$ (BEM norm)')
        ax.plot(self.axe_z, -10*self.phi4_norm, 'b-', label=r'$\Phi_4 \times 10$ (BEM norù)')
        
        # Tracé des fonctions de FIT
        ax.plot(self.axe_z,self. k0, 'r--', alpha=0.7, label='k0 (Fit)')
        ax.plot(self.axe_z,self. k2, 'g--', alpha=0.7, label='k2 (Fit)')
        ax.plot(self.axe_z,10*self.k4, 'b--', alpha=0.7, label='k4 $\times 10$ (Fit)')
        
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("Amplitude normalisée")
        ax.set_title("Comparaison Décomposition BEM vs Fonctions de Fit")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True, linestyle=':')
        plt.tight_layout()
        plt.show()


class Paraxial():
    def __init__(self):
        # On passe les arguments nécessaires au constructeur de Quadrupole
        self.y_next = None

    def RK4_step(self, f, y, t, h, alpha, beta):
        f1 = f(y, t, alpha, beta)
        f2 = f(y + h*f1/2, t + h/2, alpha, beta)
        f3 = f(y + h*f2/2, t + h/2, alpha, beta)
        f4 = f(y + h*f3, t + h, alpha, beta)
        return y + (h/6)*(f1 + 2*f2 + 2*f3 + f4)

class Ion(Paraxial): 
    def __init__(self, mass, charge, name, x, vx, y, vy, potentials_obj, dims_obj):
        # Utilisation correcte du super() pour remonter la chaîne d'héritage
        super().__init__()
        
        self.mass = mass
        self.charge = charge
        self.name = name
        
        self.state_x = np.array([x, vx], dtype=float)
        self.state_y = np.array([y, vy], dtype=float)
        
        self.history_x = []
        self.history_y = []

    def afficher_detail(self):
        print(f"Ion: {self.name}, Masse: {self.mass}, Charge: {self.charge}")
    
    def save_step(self): 
        self.history_x.append(self.state_x[0])
        self.history_y.append(self.state_y[0])

class Trajectoire(Paraxial):
    def __init__(self, potentials_obj, dims_obj):
        super().__init__()
        pass
    
    def equation(self, y, t, alpha, beta) -> None:
        """
        forme équation second degrès à résoudre
        y, t, alpha (coefficient ordre 1), beta coefféficient ordre 2
        """
        u = y[0] 
        v = y[1] 
        du = v
        dv = -alpha * u - beta * v
        return np.array([du, dv])
    
    def simulation3(self, ion : Ion, data : Extraction )-> None:
        """
        class ion
        class Extraction 
        """
        V_acc = data.p.pot_acceleration 
        print(V_acc)
        
        dz_mm = data.axe_z[1] - data.axe_z[0] 
        ion.save_step()
        
        for i in range(len(data.axe_z) - 1):
            phi_total = V_acc + data.Phi0[i]
            
            if abs(phi_total) < 0.1:
                phi_total = 0.1 if phi_total >= 0 else -0.1

            terme_axial = data.D2zphi0[i] / (4 * phi_total)
            terme_quad = data.Phi2[i] / phi_total
            
            alphax = terme_axial - terme_quad
            alphay = terme_axial + terme_quad
            beta = data.D1zphi0[i] / (2 * phi_total)
            
            # CORRECTION : Ajout de 'self.' car RK4_step est une méthode de la classe parente
            ion.state_x = self.RK4_step(self.equation, ion.state_x, data.axe_z[i], dz_mm, alphax, beta)
            ion.state_y = self.RK4_step(self.equation, ion.state_y, data.axe_z[i], dz_mm, alphay, beta)
            
            ion.save_step()

    def convergence(self, data : Extraction, n : int):
        """
        Data (type extraction)
        d (entier) ordre convergence que l'on veut vérifier 
        """
        n_points= len(data.axe_z)*n
        self.z_conv = np.linspace(data.axe_z[0], data.axe_z[-1], n_points)
        self.dz_conv = self.z_conv[1]-self.z_conv[0]
    


        # création de fonctions continues comme ca on peut réduire le pas 
        self.f_phi0 = interp1d(data.axe_z, data.Phi0, kind = 'cubic')
        self.f_phi2 = interp1d(data.axe_z, data.Phi2, kind = 'cubic')
        self.f_phi4 = interp1d(data.axe_z, data.Phi4, kind = 'cubic')
        self.f_D1zphi0 = interp1d(data.axe_z, data.D1zphi0, kind = 'cubic')
        self.f_D2zphi0 = interp1d(data.axe_z, data.D2zphi0, kind = 'cubic')



    def simulationf(self, ion : Ion, data : Extraction) -> None:
        """
        class ion 
        class extraction 
        """
        V_acc = data.p.pot_acceleration 
        ion.save_step()
        
        for i in self.z_conv[:-1]:
            phi_total = V_acc + self.f_phi0(i)
                
            if abs(phi_total) < 0.1:
                phi_total = 0.1 if phi_total >= 0 else -0.1

            terme_axial = self.f_D2zphi0(i) / (4 * phi_total)
            terme_quad = self.f_phi2(i) / phi_total
                
            alphax = terme_axial - terme_quad
            alphay = terme_axial + terme_quad
            beta = self.f_D1zphi0(i) / (2 * phi_total)
                
            # CORRECTION : Ajout de 'self.' ici aussi
            ion.state_x = self.RK4_step(self.equation, ion.state_x, i, self.dz_conv, alphax, beta)
            ion.state_y = self.RK4_step(self.equation, ion.state_y, i, self.dz_conv, alphay, beta)
                
            ion.save_step()

    def plot_discret(self, principal: Ion, marginal: Ion, data: Extraction)-> None:
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Résultats Simulation Discrète (Pas BEM)", fontsize=14, fontweight='bold')
        
        # On aplatit la liste des axes pour y accéder facilement de 0 à 3
        ax = axs.flatten()

        # 1. Trajectoire X (Principal vs Marginal)
        ax[0].plot(data.axe_z, principal.history_x, 'r-', label="Principal (x)")
        ax[0].plot(data.axe_z, marginal.history_x, 'b-', label="Marginal (x)")
        ax[0].set_title("X-Trajectories (Compare Rays)")
        
        # 2. Trajectoire Y (Principal vs Marginal)
        ax[1].plot(data.axe_z, principal.history_y, 'r-', label="Principal (y)")
        ax[1].plot(data.axe_z, marginal.history_y, 'b-', label="Marginal (y)")
        ax[1].set_title("Y-Trajectories (Compare Rays)")
        
        # 3. Chief Ray seul (X vs Y)
        ax[2].plot(data.axe_z, principal.history_x, 'r-', label="Chief Ray (x)")
        ax[2].plot(data.axe_z, principal.history_y, 'b-', label="Chief Ray (y)")
        ax[2].set_title("Chief Ray: X vs Y")

        # 4. Marginal Ray seul (X vs Y)
        ax[3].plot(data.axe_z, marginal.history_x, 'm-', label="Marginal (x)")
        ax[3].plot(data.axe_z, marginal.history_y, 'c-', label="Marginal (y)")
        ax[3].set_title("Marginal Ray")

        
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_continu(self, principal: Ion, marginal: Ion, data: Extraction) -> None:
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Résultats Simulation Continue ", fontsize=14, fontweight='bold')
        
        ax = axs.flatten()

        # 1. Trajectoire X (Principal vs Marginal)
        ax[0].plot(self.z_conv, principal.history_x, 'm-', label="Principal (x)")
        ax[0].plot(self.z_conv, marginal.history_x, 'c-', label="Marginal (x)")
        ax[0].set_title("X-Trajectories (Interpolated)")

        # 2. Trajectoire Y (Principal vs Marginal)
        ax[1].plot(self.z_conv, principal.history_y, 'm-', label="Principal (y)")
        ax[1].plot(self.z_conv, marginal.history_y, 'c-', label="Marginal (y)")
        ax[1].set_title("Y-Trajectories (Interpolated)")

        # 3. Chief Ray seul (X vs Y)
        ax[2].plot(self.z_conv, principal.history_x, 'r-', label="Chief (x)")
        ax[2].plot(self.z_conv, principal.history_y, 'b-', label="Chief (y)")
        ax[2].set_title("Chief Ray: X vs Y (Fine)")

        # 4. Marginal Ray seul (X vs Y)
        ax[3].plot(self.z_conv, marginal.history_x, 'g-', label="Marginal (x)")
        ax[3].plot(self.z_conv, marginal.history_y, 'y-', label="Marginal (y)")
        ax[3].set_title("Marginal Ray: X vs Y (Fine)")

        
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()







            
        

# %%
