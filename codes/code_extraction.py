import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from dataclasses import dataclass


@dataclass
class dimension:
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
    

class Extraction():
    def __init__(self, file_path, ):
        
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

        p_vals = data["potentials"] # [v_elec, v_ap1, v_ap2, v_sh, v_acc]
        self.p = Potentials(pot_acceleration=p_vals[4], pot_shield=p_vals[3])

        d_vals = data["dimensions"]
        self.d = dimension(*d_vals[:9]) # On prend les 9 premières valeurs

        self._setup_geometry()
        
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

    def _setup_geometry(self):
        # On recrée les variables dont plot_decomposition a besoin
        self.start_shield1 = 0
        self.end_shield1 = self.d.thickness_shield
        self.start_apert1 = self.end_shield1 + self.d.dist_shield_apert
        self.end_apert1 = self.start_apert1 + self.d.thickness_apert
        self.start_cyl = self.end_apert1 + self.d.dist_apert_quad
        self.end_cyl = self.start_cyl + self.d.length_cylinder
        self.start_apert2 = self.end_cyl + self.d.dist_apert_quad
        self.end_apert2 = self.start_apert2 + self.d.thickness_apert
        self.start_shield2 = self.end_apert2 + self.d.dist_shield_apert
        self.end_shield2 = self.start_shield2 + self.d.thickness_shield

    def derive(self):
            
        self.D2zphi0 = self.D2[5]  #phi_0'' pour la trajectoire
        self.D1zphi0 = self.E[2]   #phi_0' pour la trajectoire
        
    

    def decompose(self):
        """Calcule les composantes de la décomposition multipolaire."""
        # Phi0 : Monopolaire (
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
        
        # Tracé quadrupole 
        ax.axvspan(self.start_shield1, self.end_shield1, color='red', alpha=0.2, label='Shield')
        ax.axvspan(self.start_apert1, self.end_apert1, color='blue', alpha=0.1, label='Aperture')
        ax.axvspan(self.start_cyl, self.end_cyl, color='green', alpha=0.3, label='Electrode')
        ax.axvspan(self.start_apert2, self.end_apert2, color='blue', alpha=0.3)
        ax.axvspan(self.start_shield2, self.end_shield2, color='red', alpha=0.3)
        
        # Tracé des résultats BEM
        ax.plot(self.axe_z, -self.phi0_norm, 'r-', label=r'$\Phi_0$')
        ax.plot(self.axe_z, -self.phi2_norm, 'g-', label=r'$\Phi_2$ ')
        ax.plot(self.axe_z, -10*self.phi4_norm, 'b-', label=r'$\Phi_4 \times 10$')
        
        # Tracé des fonctions de FIT
        ax.plot(self.axe_z,self. k0, 'r--', alpha=0.7, label='k0 ')
        ax.plot(self.axe_z,self. k2, 'g--', alpha=0.7, label='k2 ')
        ax.plot(self.axe_z,10*self.k4, 'b--', alpha=0.7, label='k4 $\times 10$ ')
        
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
    def __init__(self, mass, charge, name, x, vx, y, vy):
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
    def __init__(self):
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
            
            #on envoie a RK4 pour résoudre equa diff 
            ion.state_x = self.RK4_step(self.equation, ion.state_x, data.axe_z[i], dz_mm, alphax, beta)
            ion.state_y = self.RK4_step(self.equation, ion.state_y, data.axe_z[i], dz_mm, alphay, beta)
            
            # on sauve la posistion 
            ion.save_step()

    def convergence(self, data : Extraction, n : int):
        """
        Data (type extraction)
        n (entier) ordre convergence que l'on veut vérifier (diminue le pas de l'axe)
        """

        #on redefinie notre axe z et le pas de l'intégral
        n_points= len(data.axe_z)*n
        self.z_conv = np.linspace(data.axe_z[0], data.axe_z[-1], n_points)
        self.dz_conv = self.z_conv[1]-self.z_conv[0]
    


        # création de fonctions continues comme ca on peut réduire le pas 
        self.f_phi0 = interp1d(data.axe_z, data.Phi0, kind = 'cubic')
        self.f_phi2 = interp1d(data.axe_z, data.Phi2, kind = 'cubic')
        self.f_phi4 = interp1d(data.axe_z, data.Phi4, kind = 'cubic')
        self.f_D1zphi0 = interp1d(data.axe_z, data.D1zphi0, kind = 'cubic')
        self.f_D2zphi0 = interp1d(data.axe_z, data.D2zphi0, kind = 'cubic')


    #pareil qu'avant avec nos fonctions continues 
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
                
            #envoie a RK4
            ion.state_x = self.RK4_step(self.equation, ion.state_x, i, self.dz_conv, alphax, beta)
            ion.state_y = self.RK4_step(self.equation, ion.state_y, i, self.dz_conv, alphay, beta)
                
            ion.save_step()


    #plot de la trajectoire avec les fonctions définit sur N points
    def plot_discret(self, principal: Ion, marginal: Ion, data: Extraction)-> None:
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Paraxial", fontsize=14, fontweight='bold')
        
        ax = axs.flatten()

        # 1. Trajectoire X (Principal vs Marginal)
        ax[0].plot(data.axe_z, principal.history_x, 'r-', label="Principal (x)")
        ax[0].plot(data.axe_z, marginal.history_x, 'b-', label="Marginal (x)")
        ax[0].set_title("X-Trajectories ")
        ax[0].legend()
        
        # 2. Trajectoire Y (Principal vs Marginal)
        ax[1].plot(data.axe_z, principal.history_y, 'r-', label="Principal (y)")
        ax[1].plot(data.axe_z, marginal.history_y, 'b-', label="Marginal (y)")
        ax[1].set_title("Y-Trajectories")
        ax[1].legend()
        
        # 3. Chief Ray seul (X vs Y)
        ax[2].plot(data.axe_z, principal.history_x, 'r-', label="Chief Ray (x)")
        ax[2].plot(data.axe_z, principal.history_y, 'b-', label="Chief Ray (y)")
        ax[2].set_title("Chief Ray: X vs Y")
        ax[2].legend()

        # 4. Marginal Ray seul (X vs Y)
        ax[3].plot(data.axe_z, marginal.history_x, 'm-', label="Marginal (x)")
        ax[3].plot(data.axe_z, marginal.history_y, 'c-', label="Marginal (y)")
        ax[3].set_title("Marginal Ray")
        ax[3].legend()

        
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


    # plot pout la convergence 
    def plot_continu(self, principal: Ion, marginal: Ion, data: Extraction) -> None:
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("praxial avec la convergence, ", fontsize=14, fontweight='bold')
        
        ax = axs.flatten()

        # 1. Trajectoire X (Principal vs Marginal)
        ax[0].plot(self.z_conv, principal.history_x, 'm-', label="Principal (x)")
        ax[0].plot(self.z_conv, marginal.history_x, 'c-', label="Marginal (x)")
        ax[0].set_title("X-Trajectories")
        ax[0].legend()

        # 2. Trajectoire Y (Principal vs Marginal)
        ax[1].plot(self.z_conv, principal.history_y, 'm-', label="Principal (y)")
        ax[1].plot(self.z_conv, marginal.history_y, 'c-', label="Marginal (y)")
        ax[1].set_title("Y-Trajectories")
        ax[1].legend()

        # 3. Chief Ray seul (X vs Y)
        ax[2].plot(self.z_conv, principal.history_x, 'r-', label="Chief (x)")
        ax[2].plot(self.z_conv, principal.history_y, 'b-', label="Chief (y)")
        ax[2].set_title("Chief Ray: X vs Y ")
        ax[2].legend()

        # 4. Marginal Ray seul (X vs Y)
        ax[3].plot(self.z_conv, marginal.history_x, 'g-', label="Marginal (x)")
        ax[3].plot(self.z_conv, marginal.history_y, 'y-', label="Marginal (y)")
        ax[3].set_title("Marginal Ray: X vs Y ")
        ax[3].legend()

        
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_faisceau(self , Liste_ion, data : Extraction ):
        plt.figure(1)
        plt.title("Faisceau d'ion")
        for element in Liste_ion :
            plt.plot(data.axe_z, element.history_x,label=f"x0 = {element.history_x[0]:.2f} mm" )
            plt.legend()
        plt.x_axis =("z en mm")
        plt.axis_y = ("axe x en mm")
        plt.show()


class abberation(Paraxial):
    def __init__(self)-> None:
        super().__init__()
        self.C30 = None #coéfficient spérique que l'on veut déterminer 

    def coefficient(self, ion : Ion, data : Extraction ):
        """
        Extraction -> accés au donnée de bemmpp
        ion -> accés à la trajectoire

        Recupère les varible puis on intègre
        """

        z = data.axe_z
        dz = data.axe_z[1] - data.axe_z[0]

        phiA = data.Va + data.phi0
        R = data.phi0/phiA
        Q = data.Phi2/phiA
        O = data.Phi4/phiA


        x = np.array(ion.history_x)
        dx = np.gradient(x, z)
        d2x = np.gradient(dx, z)
        d3x = np.gradient(d2x,z)
        d4x = np.gradient(d3x,z)

        #dérive chaque terme plus fois 

        dphiA = np.gradient(phiA, z)
        ddphiA = np.gradient(dphiA, z)

        dR = np.gradient(R, z)
        d2R = np.gradient(dR, z)
        d3R = np.gradient(d2R, z)
        d4R = np.gradient(d3R, z)

        dQ = np.gradient(Q, z)
        d2Q = np.gradient(dQ, z)
        d3Q = np.gradient(d2Q, z)
        d4Q = np.gradient(d3Q, z)


        dO= np.gradient(O, z)
        d2O = np.gradient(dO, z)
        d3O= np.gradient(d2O, z)
        d4O = np.gradient(d3O, z)

        Tx4 =(-d4R/32 +d2Q/6 + (d2R)**2/16 - ((d2R-Q)/2)+(Q**2)-2*O)*(x**4)
        Tx3x = (-d3R/8 + dQ/2 + (dR*d2R)/8 - (dR*Q)/2)*(x**3)*dx
        Tx2x2 = ((d2R /4) - Q)*(x**2)*(d2x)**2
        Txx3 = (dR/2) *x*((d3x)**3)

        I = np.sqrt(phiA/data.Va)*(Tx4+Tx3x+Tx2x2+Txx3)

        self.C30 = np.trapz(I, z)
        print(self.C30)





        return self.C30


