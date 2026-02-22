from Geometry import Mesh_Generation
from Field_calculation import Calculation_field
from Data import Data
from Extraction_data import Extracted_data
from Fit_functions import Fit_constants
from Graphs import Graphs
from Multipolar_decomposition import Decomposition

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