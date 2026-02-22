from Geometry import Mesh_Generation
from Field_calculation import Calculation_field
from Data import Data
from Extraction_data import Extracted_data
from Fit_functions import Fit_constants
from Graphs import Graphs
from Multipolar_decomposition import Decomposition
from paraxial import Paraxial, Ion, Trajectoire

class Potential_extraction:
    def __init__(self, data: Data, mesh_visual: bool):
        self.data = data
        self.mesh_visual = mesh_visual

        self.mesh_generation=Mesh_Generation(self.data, self.mesh_visual)
        self.calculation_field=Calculation_field(self.data)

    def mesh(self):
        self.mesh_generation.initialisation()
        self.mesh_generation.geometry()
        self.mesh_generation.creation_mesh()
        self.mesh_generation.surfaces()
        self.mesh_generation.mesh()
        self.mesh_generation.finalize()

    def potential_extraction(self):
        self.calculation_field.mesh_Importation()
        self.calculation_field.potentials_settings()
        self.calculation_field.matrix_inversion()
        self.calculation_field.derivatives()
        self.calculation_field.potential_exportation()

    def potential_visualisation(self):
        self.calculation_field.visualisation()

    def graph_potential_axis(self):
        self.calculation_field.potential_axis_printing()


class Data_exploitation:
    def __init__(self, extracted_data: Extracted_data, fit_constants: Fit_constants):
        self.extracted_data = extracted_data
        self.fit_constants = fit_constants

        self.decomposition=Decomposition(self.extracted_data)

        self.graphs=Graphs(self.extracted_data, self.decomposition, self.fit_constants)

    def decomposition_calculation(self):
        self.decomposition.composantes()

    def decomposition_graph(self):
        self.graphs.graphe_composantes()

    def fit_calculation(self):
        self.fit_constants.fonction_fit()
    
    def fit_graph(self):
        self.graphs.graphe_fit()


class SimulationParaxiale:

    def __init__(self, extracted: Extracted_data, decomp: Decomposition):
        
    
        self.extracted = extracted
        self.decomp = decomp
        
        self.traj = Trajectoire()

    def run_discret(self, ion_principal: Ion, ion_marginal: Ion):
        """
        
        """
        #trajectoire
        self.traj.simulation3(ion_principal, self.extracted, self.decomp)
        self.traj.simulation3(ion_marginal, self.extracted, self.decomp)

        # plot
        self.traj.plot_discret(ion_principal, ion_marginal, self.extracted)


    def run_convergence(self, ion_principal: Ion, ion_marginal: Ion, n: int):
        """
        
        """

        # calcul convergence
        self.traj.convergence(self.extracted, self.decomp, n)

        # simulation contnue
        self.traj.simulationf(ion_principal, self.extracted)
        self.traj.simulationf(ion_marginal, self.extracted)

        # plot
        self.traj.plot_continu(ion_principal, ion_marginal, self.extracted, n)


    def run_faisceau(self, liste_ions: list):
        """
        Simulation pour un faisceau d'ions
        """

        for ion in liste_ions:
            ion.history_x = []
            ion.history_y = []
            self.traj.simulation3(ion, self.extracted, self.decomp)

        self.traj.plot_faisceau(liste_ions, self.extracted)

        