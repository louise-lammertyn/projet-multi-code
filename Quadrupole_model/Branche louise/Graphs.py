from matplotlib import pylab as plt
import matplotlib.pyplot
from Extraction_data import Extracted_data
from Multipolar_decomposition import Decomposition
from Fit_functions import Fit_constants

class Graphs:
    """
    Class to handle the visualization of multipolar decomposition data 
    and the quadrupole's geometry.
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
        Overlays the physical geometry of the system.
        """
        if self.data.z_off is not None :
            offset = self.data.z_off
            print(offset)
            
            #Use of flags to put the labels just one time
            for i, z in enumerate(offset):
                label_s = 'Shield' if i == 0 else ""
                label_a = 'Aperture' if i == 0 else ""
                label_e = 'Electrodes' if i == 0 else ""

                ax.axvspan(z + self.data.start_shield1, z + self.data.end_shield1, color='red', alpha=0.15, label=label_s)
                ax.axvspan(z + self.data.start_apert1, z + self.data.end_apert1, color='blue', alpha=0.15, label=label_a)
                ax.axvspan(z + self.data.start_cyl, z + self.data.end_cyl, color='green', alpha=0.15, label=label_e)
                ax.axvspan(z + self.data.start_apert2, z + self.data.end_apert2, color='blue', alpha=0.15)
                ax.axvspan(z + self.data.start_shield2, z + self.data.end_shield2, color='red', alpha=0.15)

        else :
            #Case of a single quadrupole
            ax.axvspan(self.data.start_shield1, self.data.end_shield1, color='red', alpha=0.15, label='Shield')
            ax.axvspan(self.data.start_apert1, self.data.end_apert1, color='blue', alpha=0.15, label='Aperture')
            ax.axvspan(self.data.start_cyl, self.data.end_cyl, color='green', alpha=0.15, label='Electrodes')
            ax.axvspan(self.data.start_apert2, self.data.end_apert2, color='blue', alpha=0.15)
            ax.axvspan(self.data.start_shield2, self.data.end_shield2, color='red', alpha=0.15)

    def graph_components(self) -> None:
        """
        Plots the multipolar components (Phi0 to Phi4) along the Z-axis 
        overlaid with the system geometry.
        """
        #1.Initializes figures and axis
        fig, ax = plt.subplots(figsize=(9, 5))

        #2.Traces the background geometry
        self.trace_geo(ax)

        # Plots calculated decomposition data and Okayama's model 
        ax.plot(self.data.axe_z, self.decomposition.Phi0_maj, label=r'$\Phi_0 $ $[V]$', color='crimson')
        ax.plot(self.data.axe_z, self.decomposition.Phi1_maj, label=r'$\Phi_1$ $[V/mm^1]$', color='darkviolet')
        ax.plot(self.data.axe_z, self.decomposition.Phi2_maj, label=r'$\Phi_2$ $[V/mm^2]$', color='green')
        ax.plot(self.data.axe_z, self.decomposition.Phi3_maj, label=r'$\Phi_3$ $[V/mm^3]$', color='gold')
        ax.plot(self.data.axe_z, 10*self.decomposition.Phi4_maj, label=r'$\Phi_4 \times 10$ $[V/mm^4]$', color='royalblue')

        # 4.Labels and style
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("Potentiel")
        ax.set_title("Multipolar decomposition")
        ax.grid(True)
        
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
        
        plt.tight_layout()
        plt.show()
    
    #Same graph but with fitting potential component  
    def graph_fit(self, bool_fit = True ) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        self.trace_geo(ax)

        #Printing of the different components of the potential
        ax.plot(self.data.axe_z, self.decomposition.Phi0_fit, label=r'$\Phi_0 $ $[V]$', color='crimson')
        ax.plot(self.data.axe_z, self.decomposition.Phi2_fit, label=r'$\Phi_2$ $[V/mm^2]$', color='green')
        plt.plot(self.data.axe_z, 10*self.decomposition.Phi4_fit, label=r'$\Phi_4 *10$ $[V/mm^4]$', color='royalblue')
        
        if (bool_fit == True ):
            ax.plot(self.data.axe_z, self.fit.k0, label=r'k0', color='crimson', linestyle='dashed')
            ax.plot(self.data.axe_z, self.fit.k2, label=r'k2', color='green', linestyle='dashed')
            ax.plot(self.data.axe_z, 10*self.fit.k4, label=r'k4 *10', color='royalblue', linestyle='dashed')

        ax.set_xlabel("z (mm)")
        ax.set_ylabel("Potentiel")
        ax.set_title("Décomposition multipolaire sur l’axe et fonctions de fit")
        ax.grid()
        ax.legend()
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
        plt.tight_layout()
        plt.show()

