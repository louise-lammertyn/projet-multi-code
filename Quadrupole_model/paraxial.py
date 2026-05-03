import numpy as np
import matplotlib.pyplot as plt
from Data import Data
from Extraction_data import Extracted_data
from Multipolar_decomposition import Decomposition
from scipy.interpolate import interp1d

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
    
    def simulation3(self, ion : Ion,  data : Extracted_data, decomp : Decomposition )-> None:
        """
        class ion
        class Data 
        """

        ion.history_x = []
        ion.history_y = []
        
        V_acc = data.Vacceleration
        print(V_acc)
        
        dz_mm = data.axe_z[1] - data.axe_z[0] 
        ion.save_step()
        
        
        for i in range(len(data.axe_z) - 1):
            phi_total = V_acc + decomp.Phi0_maj[i]
            
            if abs(phi_total) < 0.1:
                phi_total = 0.1 if phi_total >= 0 else -0.1

            terme_axial = data.D2zphi0[i] / (4 * phi_total)
            terme_quad = decomp.Phi2_maj[i] / phi_total
            
            alphax = terme_axial - terme_quad
            alphay = terme_axial + terme_quad
            beta = data.D1zphi0[i] / (2 * phi_total)
            
            #on envoie a RK4 pour résoudre equa diff 
            ion.state_x = self.RK4_step(self.equation, ion.state_x, data.axe_z[i], dz_mm, alphax, beta)
            ion.state_y = self.RK4_step(self.equation, ion.state_y, data.axe_z[i], dz_mm, alphay, beta)
            
            # on sauve la posistion 
            ion.save_step()

    def convergence(self, data : Extracted_data,decomp : Decomposition, n : int) -> None:
        """
        Data (type extraction)
        n (entier) ordre convergence que l'on veut vérifier (diminue le pas de l'axe)
        """

        #on redefinie notre axe z et le pas de l'intégral
        n_points= len(data.axe_z)*n
        self.z_conv = np.linspace(data.axe_z[0], data.axe_z[-1], n_points)
        self.dz_conv = self.z_conv[1]-self.z_conv[0]
    


        # création de fonctions continues comme ca on peut réduire le pas 
        self.f_phi0 = interp1d(data.axe_z, decomp.Phi0_maj, kind = 'cubic')
        self.f_phi2 = interp1d(data.axe_z, decomp.Phi2_maj, kind = 'cubic')
        self.f_phi4 = interp1d(data.axe_z, decomp.Phi4_maj, kind = 'cubic')
        self.f_D1zphi0 = interp1d(data.axe_z, data.D1zphi0, kind = 'cubic')
        self.f_D2zphi0 = interp1d(data.axe_z, data.D2zphi0, kind = 'cubic')


    #pareil qu'avant avec nos fonctions continues 
    def simulationf(self, ion : Ion, data : Extracted_data ) -> None:
        """
        class ion 
        class extraction 
        """
        V_acc = data.Vacceleration
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
        
  
    def solution_analytique(self, ion : Ion, data : Extracted_data, decomp : Decomposition, z0 = 0) -> None : 
        z_axes = data.axe_z
        n = len(z_axes)
        
        # Initialisation des tableaux
        self.xp = np.zeros(n)
        self.xm = np.zeros(n)
        self.yp = np.zeros(n)
        self.ym = np.zeros(n)

        phi0 = abs(data.Vacceleration)
        print(phi0)

        for i in range(n):
            dist = z_axes[i] - z0
            
            val_phi2 = abs(decomp.Phi2_maj[i])
            print
            
            if val_phi2 < 1e-10:
                # Cas sans champ (k=0) : mouvement rectiligne
                self.xp[i] = 1.0 
                self.xm[i] = dist
                self.yp[i] = 1.0
                self.ym[i] = dist
            else:
                k = np.sqrt(val_phi2 / phi0)
                
                self.xp[i] = np.cos(k * dist)
                self.xm[i] = (1/k) * np.sin(k * dist)
                
                self.yp[i] = np.cosh(k * dist)
                self.ym[i] = (1/k) * np.sinh(k * dist)


    #xp,xm, ym, yp = solution_analytique(Ion,data, 0)  
        
    def solution_analytique2(self, ion : Ion, data : Extracted_data, decomp : Decomposition, z0 = 0) -> None : 
        #pour un champs quad unique -> on met les apertures a 0
        z_axes = data.axe_z
        n = len(z_axes)
        self.xp, self.xm = np.zeros(n), np.zeros(n)
        self.yp, self.ym = np.zeros(n), np.zeros(n)
        
      

        for i in range(len(data.axe_z)):
            z = data.axe_z[i]
            vz = data.Vacceleration + decomp.Phi0_maj[i]
            phi2 = decomp.Phi2_maj[i]
            w0 = np.sqrt(phi2*ion.charge/ion.mass*(vz)**2)
            self.xp[i] = np.cos(w0*(z-z0))
            self.xm[i] = (1/w0)*np.sin(w0*(z-z0))
            self.yp[i] = np.cosh(w0*(z-z0))
            self.ym[i] = (1/w0)*np.sinh(w0*(z-z0))
            print(self.xp)
    
    def plot_theorique(self, data : Extracted_data, z0 = 0) -> None :
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Trajectoire theorique d'un quad seul ", fontsize=14, fontweight='bold')
        
        ax = axs.flatten()

        # 1. Trajectoire X (Principal vs Marginal)
        ax[0].plot(data.axe_z, self.xp, 'r-', label=" Rayon Principal (x)")
        ax[0].plot(data.axe_z, self.xm, 'b-', label="Rayon Marginal (x)")
        ax[0].set_title("Trajectoire selon la direction x ")
        ax[0].legend()
        
        # 2. Trajectoire Y (Principal vs Marginal)
        ax[1].plot(data.axe_z, self.yp, 'r-', label="Rayon Principal (y)")
        ax[1].plot(data.axe_z, self.ym, 'b-', label="Rayon Marginal (y)")
        ax[1].set_title("Trajectoire selon la direction y ")
        ax[1].legend()
        
        # 3. Chief Ray seul (X vs Y)
        ax[2].plot(data.axe_z, self.xp, 'r-', label=" Rayon principal (x)")
        ax[2].plot(data.axe_z, self.yp, 'b-', label=" Rayon principal(y) ")
        ax[2].set_title("rayon princpal ")
        ax[2].legend()

        # 4. Marginal Ray seul (X vs Y)
        ax[3].plot(data.axe_z, self.xm, 'r-', label="rayon marginal (x)")
        ax[3].plot(data.axe_z,self.ym, 'b-', label="rayon marginal (y)")
        ax[3].set_title("rayon marginal")
        ax[3].legend()

     

        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


        



    




    #plot de la trajectoire avec les fonctions définit sur N points
    def plot_discret(self, principal: Ion, marginal: Ion, data : Extracted_data)-> None:
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Trajectoire paraxiale", fontsize=14, fontweight='bold')
        
        ax = axs.flatten()

        # 1. Trajectoire X (Principal vs Marginal)
        ax[0].plot(data.axe_z, principal.history_x, 'r-', label="Principal (x)")
        ax[0].plot(data.axe_z, marginal.history_x, 'b-', label="Marginal (x)")
        ax[0].set_title(" Trajectoire selon la direction x ")
        ax[0].legend()
        
        # 2. Trajectoire Y (Principal vs Marginal)
        ax[1].plot(data.axe_z, principal.history_y, 'r-', label="Principal (y)")
        ax[1].plot(data.axe_z, marginal.history_y, 'b-', label="Marginal (y)")
        ax[1].set_title("Trajectoire selon la direction y ")
        ax[1].legend()
        
        # 3. Chief Ray seul (X vs Y)
        ax[2].plot(data.axe_z, principal.history_x, 'r-', label=" rayon principal ")
        ax[2].plot(data.axe_z, principal.history_y, 'b-', label=" rayon principal ")
        ax[2].set_title("rayon princpal ")
        ax[2].legend()

        # 4. Marginal Ray seul (X vs Y)
        ax[3].plot(data.axe_z, marginal.history_x, 'r-', label="rayon marginal")
        ax[3].plot(data.axe_z, marginal.history_y, 'b-', label="rayon marginal")
        ax[3].set_title("rayon marginal")
        ax[3].legend()

        ax[0].set_xlabel("z position (mm)")
        ax[0].set_ylabel("x position (mm)")

        ax[1].set_xlabel("z position (mm)")
        ax[1].set_ylabel("y position (mm)")

        ax[2].set_xlabel("z position (mm)")
        ax[2].set_ylabel("x-y position (mm)")

        ax[3].set_xlabel("z position (mm)")
        ax[3].set_ylabel("x-y position (mm)")

        

     

        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        plt.figure(2)
        plt.plot(data.axe_z, marginal.history_x, 'r-', label="rayon marginal")
        plt.plot(data.axe_z, marginal.history_y, 'b-', label="rayon marginal")
        plt.xlabel('mm')
        plt.show()


    # plot pout la convergence 
    def plot_continu(self, principal: Ion, marginal: Ion, data : Extracted_data, n) -> None:
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Trajectoire paraxiale,  convergence ordre {n}   ", fontsize=14, fontweight='bold')
        
        ax = axs.flatten()

        # 1. Trajectoire X (Principal vs Marginal)
        ax[0].plot(self.z_conv, principal.history_x, 'r-', label="Principal (x)")
        ax[0].plot(self.z_conv, marginal.history_x, 'b-', label="Marginal (x)")
        ax[0].set_title("X Trajectories")
        ax[0].legend()

        # 2. Trajectoire Y (Principal vs Marginal)
        ax[1].plot(self.z_conv, principal.history_y, 'r-', label="Principal (y)")
        ax[1].plot(self.z_conv, marginal.history_y, 'b-', label="Marginal (y)")
        ax[1].set_title("Y Trajectories")
        ax[1].legend()

        # 3. Chief Ray seul (X vs Y)
        ax[2].plot(self.z_conv, principal.history_x, 'r-', label="Principal (x)")
        ax[2].plot(self.z_conv, principal.history_y, 'r-', label="Principal (y)")
        ax[2].set_title("Chief Ray ")
        ax[2].legend()

        # 4. Marginal Ray seul (X vs Y)
        ax[3].plot(self.z_conv, marginal.history_x, 'b-', label="Marginal (x)")
        ax[3].plot(self.z_conv, marginal.history_y, 'b-', label="Marginal (y)")
        ax[3].set_title("Marginal Ray")
        ax[3].legend()

        
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_faisceau(self , Liste_ion,  data : Extracted_data, ):
        plt.figure(1)
        plt.title("Faisceau d'ion")
        for element in Liste_ion :
            plt.plot(data.axe_z, element.history_x,label=f"x0 = {element.history_x[0]:.2f} mm" )
            plt.legend()
        plt.x_axis =("z en mm")
        plt.y_axis = ("axe x en mm")
        plt.show()