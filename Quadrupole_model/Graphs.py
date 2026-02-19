from matplotlib import pylab as plt
import matplotlib.pyplot
from Extraction_data import Extracted_data
from Multipolar_decomposition import Decomposition
from Fit_functions import Fit_constants

class Graphs:
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
        plt.figure()
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
        plt.figure
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
