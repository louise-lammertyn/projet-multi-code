from matplotlib import pylab as plt
import numpy as np 
import matplotlib.pyplot
from Extraction_data import Extracted_data
from Multipolar_decomposition import Decomposition
from Fit_functions import Fit_constants

class Graphs:
    """
    A class to handle the visualization of multipolar decomposition data 
    and the quadrupole geometry.
    """
    def __init__(self, data: Extracted_data, decomposition: Decomposition, fit : Fit_constants = None) -> None:
        """
        data (Extracted_data): Object containing axis coordinates and geometry boundaries.
        decomposition (Decomposition): Object containing multipolar component values.
        fit (Fit_constants): Object containing Okayama fitting function constants.
        """
        self.data = data
        self.decomposition = decomposition
        self.fit = fit

        self.ax = None
   
        

    def trace_geo(self, ax) -> None:
            """
            Overlay the physical geometry of the system.
            """
            if self.data.z_off is not None :
                offset = self.data.z_off
                print(offset)
                
                # On utilise un drapeau pour ne mettre les labels qu'une seule fois
                for i, z in enumerate(offset):
                    label_s = 'Shield' if i == 0 else ""
                    label_a = 'Aperture electrode ' if i == 0 else ""
                    label_e = 'quadrupole electrode' if i == 0 else ""

                    ax.axvspan(z + self.data.start_shield1, z + self.data.end_shield1, color='red', alpha=0.15, label=label_s)
                    ax.axvspan(z + self.data.start_apert1, z + self.data.end_apert1, color='blue', alpha=0.15, label=label_a)
                    ax.axvspan(z + self.data.start_cyl, z + self.data.end_cyl, color='yellow', alpha=0.15, label=label_e)
                    ax.axvspan(z + self.data.start_apert2, z + self.data.end_apert2, color='blue', alpha=0.15)
                    ax.axvspan(z + self.data.start_shield2, z + self.data.end_shield2, color='red', alpha=0.15)
            else :
                # Cas d'un seul quadrupole
                ax.axvspan(self.data.start_shield1, self.data.end_shield1, color='red', alpha=0.15, label='Shield')
                ax.axvspan(self.data.start_apert1, self.data.end_apert1, color='blue', alpha=0.15, label='Aperture')
                ax.axvspan(self.data.start_cyl, self.data.end_cyl, color='yellow', alpha=0.15, label='Electrodes')
                ax.axvspan(self.data.start_apert2, self.data.end_apert2, color='blue', alpha=0.15)
                ax.axvspan(self.data.start_shield2, self.data.end_shield2, color='red', alpha=0.15)

##tracer les phi reel 
    def graphe_composantes(self) -> None:
        """
        Plot the multipolar components (Phi0 to Phi4) along the Z-axis 
        overlaid with the system geometry.
        """
        # 1. Initialize figure and axis
        fig, ax = plt.subplots(figsize=(9, 5))

        # 2. Trace the background geometry
        self.trace_geo(ax)
        # Plot Calculated decomposition data and Okayam'a model 

       
        ax.plot(self.data.axe_z, self.decomposition.Phi0_maj, 
                label=r'$\Phi_0 $ $[V]$', color='deepskyblue')
        
        ax.plot(self.data.axe_z, self.decomposition.Phi1_maj, 
                label=r'$\Phi_1$ $[V/mm]$', color='darkviolet')
        
        ax.plot(self.data.axe_z, self.decomposition.Phi2_maj, 
                label=r'$ \Phi_2$ $[V/mm^2]$', color='red')
        
        ax.plot(self.data.axe_z, self.decomposition.Phi3_maj, 
                label=r'$\Phi_3$ [V/mm$^3$]', color='pink')
        
        ax.plot(self.data.axe_z, self.decomposition.Phi4_maj, 
                label=r'$\Phi_4$ [V/mm$^4$]', color='green')
        

        # 4. label and style
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("Potentiel")
        ax.set_title("Multipolaire decomposition ")
        ax.grid(True)
        
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
        
        plt.tight_layout()
        plt.show()
        V = 2000


        print("valeur theroqie de phi2 max Vq/a2 =", V)
        print("valeur de phi2 max",np.max(self.decomposition.Phi2_maj) )

        ##uilisé pour les tracer de l'article 1982
        ## utilisé pour les k

    def graphe_k1982(self) -> None:
                """
                Plot the k(z) along the Z-axis 
                overlaid with the system geometry.
                """
                # 1. Initialize figure and axis
                fig, ax1 = plt.subplots(figsize=(9, 5))

                # 2. Trace the background geometry
                self.trace_geo(ax1)
                # Plot Calculated decomposition data and Okayam'a model 
        
        
                ax1.plot(self.data.axe_z,np.abs(self.decomposition.k0), 
                        label=r'k0(z)', color='deepskyblue')
                
                ax1.plot(self.data.axe_z, self.decomposition.k2, 
                        label=r'k2(z)', color='red')
                ax2 = ax1.twinx()  # Créer un second axe partageant l'axe X

                
                ax2.plot(self.data.axe_z, np.abs(self.decomposition.k4), 
                        label=r'k4(z)', color='green')
                



            # 4. label and style
                ax1.set_xlim(30,60)
            
                ax1.set_xlabel("z (mm)")
                ax1.set_ylabel("k0(z), k2(z)")
                ax2.set_ylabel("k(4)")
                ax1.set_title("potentiel function of the new correction lens k0,k2,K4 ")
                ax1.set_ylim(0, 1)
                ax2.set_ylim(0, 0.05)

                ax1.grid(True)

                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()

                ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
                fig.tight_layout()

                
                
                plt.show()
        
    def graphe_zoom_y(self, ymin = -1, ymax  = 1 ) :
         """
         Allow to zoom in the y axis 
         """
          # 1. Initialize figure and axis
         fig, ax = plt.subplots(figsize=(9, 5))

         # 2. Trace the background geometry
         self.trace_geo(ax)
         # Plot Calculated decomposition data and Okayam'a model 

       
         ax.plot(self.data.axe_z, self.decomposition.Phi0_maj, 
                label=r'$\Phi_0 $ $[V]$', color='deepskyblue')
        
         ax.plot(self.data.axe_z, self.decomposition.Phi1_maj, 
                label=r'$\Phi_1$ $[V/mm]$', color='darkviolet')
        
         ax.plot(self.data.axe_z, self.decomposition.Phi2_maj, 
                label=r'$ \Phi_2$ $[V/mm^2]$', color='red')
        
         ax.plot(self.data.axe_z, self.decomposition.Phi3_maj, 
                label=r'$\Phi_3$ [V/mm$^3$]', color='pink')
        
         ax.plot(self.data.axe_z, self.decomposition.Phi4_maj, 
                label=r'$\Phi_4$ [V/mm$^4$]', color='green')
        

         # 4. label and style
         ax.set_xlabel("z (mm)")
         ax.set_ylabel("Potentiel")
         ax.set_ylim(ymin, ymax)
         ax.set_title("Multipolaire decomposition ")
         ax.grid(True)
        
         ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
        
         plt.tight_layout()
         plt.show()
        
         

        

    def graphe_quad(self) -> None:
        """
        Plot the multipolar components (Phi0 to Phi4) along the Z-axis 
        overlaid with the system geometry.
        """
        # 1. Initialize figure and axis
        fig, ax = plt.subplots(figsize=(9, 5))

        # 2. Trace the background geometry
        self.trace_geo(ax)
        # Plot Calculated decomposition data and Okayam'a model 

       
        ax.plot(self.data.axe_z, self.decomposition.Phi0_maj/50, 
                label=r'$\Phi_0 / 50$ $[V]$', color='deepskyblue')
        
        ax.plot(self.data.axe_z, self.decomposition.Phi1_maj, 
                label=r'$\Phi_1$ $[V/mm]$', color='darkviolet')
        
        ax.plot(self.data.axe_z, 15*self.decomposition.Phi2_maj, 
                label=r'$15 \times \Phi_2$ $[V/mm^2]$', color='red')
        
        ax.plot(self.data.axe_z, self.decomposition.Phi3_maj, 
                label=r'$\Phi_3$ $[V/mm^3]$', color='pink')
        
        ax.plot(self.data.axe_z, 1000*self.decomposition.Phi4_maj, 
                label=r'$1000 \times \Phi_4$ $[V/mm^4]$', color='green')
        

        # 4. label and style
        ax.set_xlabel("z (mm)")
        ax.set_ylim(0, 50)
        ax.set_ylabel("Potentiel")
        ax.set_title("Décomposition multipolaire ")
        ax.grid(True)
        
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
        
        plt.tight_layout()
        plt.show()

    def graphe_zoom(self, z_range=(-5, 20)) -> None:
        """
        
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # 1. Tracer la géométrie en arrière-plan
        self.trace_geo(ax)
        
        # 2. Tracer Phi0 (le champ rond)
       
        ax.plot(self.data.axe_z, self.decomposition.Phi0_maj, 
                label=r'$\Phi_0$ (Champ rond)', color='crimson', lw=2)
        
        ax.axhline(0, color='black', linestyle='-', alpha=0.5)

        ax.set_xlim(z_range)
        
        # Ajustement dynamique de l'échelle Y pour voir les détails près de 0
        mask = (self.data.axe_z >= z_range[0]) & (self.data.axe_z <= z_range[1])
        if any(mask):
            y_data = self.decomposition.Phi0_maj[mask]
            ax.set_ylim(min(y_data) - 5, max(y_data) + 5)

        # 5. Style et labels
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("Potentiel $\Phi_0$ (V)")
        ax.set_title("Zoom")
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    #same but with fitting potential componant  
    def graphe_fit(self, bool_fit = True ) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        self.trace_geo(ax)

        #Printing of the different composants of the potential
       
        ax.plot(self.data.axe_z, -self.decomposition.Phi0_fit, label=r'$\Phi_0 $ $[V]$', color='deepskyblue')
        ax.plot(self.data.axe_z, self.decomposition.Phi2_fit, label=r'$\Phi_2$ $[V/mm^2]$', color='red')
        plt.plot(self.data.axe_z, 10*self.decomposition.Phi4_fit, label=r'$\Phi_4 *10$ $[V/mm^4]$', color='green')
        
        if (bool_fit == True ):
            ax.plot(self.data.axe_z, self.fit.k0, label=r'k0(Z) fitting function', color='skyblue', linestyle='dashed')
            ax.plot(self.data.axe_z, self.fit.k2, label=r'k2(Z) fitting function', color='salmon', linestyle='dashed')
            ax.plot(self.data.axe_z, 10*self.fit.k4, label=r'10 *k4(z) fitting function ', color='darkgreen', linestyle='dashed')
        

        params_text = (
            f"Quadrupole geometry (mm) :\n"
            f"Aperture: r={self.data.radius_apert}, t={self.data.thickness_apert} | "
            f"quadrupole: a ={self.data.radius_axis}, l={self.data.length_cylinder} | "
            f" shield : rshield ={self.data.radius_in_shield}, d={self.data.dist_apert_quad}")

        plt.figtext(0.5, 0.01, params_text, ha="center", fontsize=8, 
                    bbox={"facecolor":"white", "alpha":0.8, "edgecolor":"gray", "pad":3})

        
        plt.subplots_adjust(bottom=0.15, right=0.8) 
        
        ax.set_xlabel("z (mm)")
        ax.set_xlim(0,50)
        ax.set_ylabel("Potentiel (u.a.)")
        ax.set_title("Decomposition multipoalaire")
        ax.grid(True, alpha=0.3)
        
        # On place la légende bien à droite sans qu'elle descende trop bas
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        
        
        plt.show()

