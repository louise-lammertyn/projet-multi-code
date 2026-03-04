from Geometry import Mesh_Generation
from Field_calculation import Calculation_field
from Data import Data
from Extraction_data import Extracted_data
from Fit_functions import Fit_constants
from Graphs import Graphs
from Multipolar_decomposition import Decomposition
from paraxial import Paraxial, Ion, Trajectoire
from Field_calculation import Calculation_field
from reconstruction import Reconstruction

class Potential_extraction:
    """
    Orchestrates the first stage of the workflow: 
    Quadrupole generation, meshing, and solving the electrostatic BEM problem.
    """
    def __init__(self, data: Data, mesh_visual: bool, file_name : str):

        """
            data (Data): Configuration object containing geometry and mesh parameters.
            mesh_visual (bool): If True, opens the GMSH GUI during the process.
        """
        self.data = data
        self.mesh_visual = mesh_visual
        self.file_name = file_name

        self.mesh_generation=Mesh_Generation(self.data, self.mesh_visual)
        self.calculation_field=Calculation_field(self.data, self.file_name)

    def mesh(self):
        """Execute ull GMSH """
        self.mesh_generation.initialisation()
        self.mesh_generation.geometry()
        self.mesh_generation.creation_mesh()
        self.mesh_generation.surfaces()
        self.mesh_generation.mesh()
        self.mesh_generation.finalize()

    def potential_extraction(self):
        """Solve the Laplace equation on the mesh and extract axial potential/derivatives."""
        self.calculation_field.mesh_Importation()
        self.calculation_field.potentials_settings()
        self.calculation_field.matrix_inversion()
        self.calculation_field.derivatives()
        self.calculation_field.potential_exportation()

    def potential_visualisation(self):
        """Render the 3D potential distribution on the electrode surfaces."""
        self.calculation_field.visualisation()

    def graph_potential_axis(self):
        """Plot the calculated potential along the central Z-axis."""
        self.calculation_field.potential_axis_printing()

    def generation_quad(self, tension_dico, file_name = "quad_reconstuire.npz"):
        reconstr = Reconstruction(self,self.data.output_dir, tension_dico )
        reconstr.derivative()
        reconstr.save(file_name)
        





class Data_exploitation:
    """
    the second stage: 
    Decomposing the BEM results into multipolar components and comparing with okayama' models.
    """
    def __init__(self, extracted_data: Extracted_data, fit_constants: Fit_constants):
        """
            extracted_data (Extracted_data): Data loaded from the .npz solver output.
            fit_constants (Fit_constants): The Okayama model parameters for comparison.
        """
        self.extracted_data = extracted_data
        self.fit_constants = fit_constants

        self.decomposition=Decomposition(self.extracted_data)

        self.graphs=Graphs(self.extracted_data, self.decomposition, self.fit_constants)

    def decomposition_calculation(self):
        """Compute the multipolar expansion (Phi0, Phi2, Phi4) ."""
        self.decomposition.composantes()

    def decomposition_graph(self):
        """Visualize the calculated multipolar components overlaid with geometry."""
        self.graphs.graphe_composantes()

    def fit_calculation(self):
        """Generate the theoretical Okayama fit curves."""
        self.fit_constants.fonction_fit()
    
    def fit_graph(self):
        """Compare the BEM decomposition with the theoretical fitting functions."""
        self.graphs.graphe_fit()


class SimulationParaxiale:
    """
    the third stage: 
    Simulating ion trajectories through the quadrupole using paraxial approximations.
    """

    def __init__(self, extracted: Extracted_data, decomp: Decomposition):
        """"
            extracted (Extracted_data): Geometry and axis data.
            decomp (Decomposition): Multipolar components .
        """
    
        self.extracted = extracted
        self.decomp = decomp
        
        self.traj = Trajectoire()

    def run_discret(self, ion_principal: Ion, ion_marginal: Ion):
        """
        Run a discrete step simulation for two specific ions.
        
        Args:
            ion_principal (Ion)
            ion_marginal (Ion)
        """
        ## Run RK4 integration
        self.traj.simulation3(ion_principal, self.extracted, self.decomp)
        self.traj.simulation3(ion_marginal, self.extracted, self.decomp)

        # plot
        self.traj.plot_discret(ion_principal, ion_marginal, self.extracted)


    def run_convergence(self, ion_principal: Ion, ion_marginal: Ion, n: int):
        """
        convergence .
        
        Args:
            n (int): Convergence factor.
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
        Simulate a full beam  of ions.

        Args:
            liste_ions (list[Ion]): A list of Ion objects with varying initial conditions.
        """

        for ion in liste_ions:
            ion.history_x = []
            ion.history_y = []
            self.traj.simulation3(ion, self.extracted, self.decomp)

        self.traj.plot_faisceau(liste_ions, self.extracted)

        