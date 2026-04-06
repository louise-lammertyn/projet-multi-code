from Geometry import Mesh_Generation
from Field_calculation import Calculation_field
from Data import Data
from Extraction_data import Extracted_data
from Fit_functions import Fit_constants
from Graphs import Graphs
from Multipolar_decomposition import Decomposition
from Paraxial import Paraxial_trajectories, Ion, Trajectory
from Field_calculation import Calculation_field
from Reconstruction import Reconstruction

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

    def mesh_system(self):
        """Execute ull GMSH """
        self.mesh_generation.initialisation()
        self.mesh_generation.geometry_system()
        self.mesh_generation.creation_mesh()
        self.mesh_generation.surfaces_system()
        self.mesh_generation.mesh()
        self.mesh_generation.finalize_system()

    def mesh_quad1(self):
        """Execute ull GMSH """
        self.mesh_generation.initialisation()
        self.mesh_generation.geometry_quad1()
        self.mesh_generation.creation_mesh()
        self.mesh_generation.surfaces_quad1()
        self.mesh_generation.mesh()
        self.mesh_generation.finalize_quad1()

    def mesh_quad2(self):
        """Execute ull GMSH """
        self.mesh_generation.initialisation()
        self.mesh_generation.geometry_quad2()
        self.mesh_generation.creation_mesh()
        self.mesh_generation.surfaces_quad2()
        self.mesh_generation.mesh()
        self.mesh_generation.finalize_quad2()

    def mesh_quad3(self):
        """Execute ull GMSH """
        self.mesh_generation.initialisation()
        self.mesh_generation.geometry_quad3()
        self.mesh_generation.creation_mesh()
        self.mesh_generation.surfaces_quad3()
        self.mesh_generation.mesh()
        self.mesh_generation.finalize_quad3()

    def mesh_quad4(self):
        """Execute ull GMSH """
        self.mesh_generation.initialisation()
        self.mesh_generation.geometry_quad4()
        self.mesh_generation.creation_mesh()
        self.mesh_generation.surfaces_quad4()
        self.mesh_generation.mesh()
        self.mesh_generation.finalize_quad4()

    def mesh_doublet(self):
        """Execute ull GMSH """
        self.mesh_generation.initialisation()
        self.mesh_generation.geometry_doublet()
        self.mesh_generation.creation_mesh()
        self.mesh_generation.surfaces_doublet()
        self.mesh_generation.mesh()
        self.mesh_generation.finalize_doublet()   

    def potential_extraction(self):
        """Solve the Laplace equation on the mesh and extract axial potential/derivatives."""
        self.calculation_field.mesh_importation()
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


class Generation_quad():
    """
    Class that reconstructs the global potential by summing unitary potentials
    """
    def __init__(self, tension_dico : dict,output_dir : str) -> None:
        self.tension_dico = tension_dico
        self.output_dir = output_dir

        self.sum = None

    def reconstruction(self, file_name = "quad_reconstuit.npz") ->  str:
        self.sum = Reconstruction(self.output_dir, self.tension_dico )
        self.sum.derivative()
        file_path = self.sum.save(file_name)
        return file_path


class Data_exploitation:
    """ 
    Decomposing the BEM results into multipolar components and comparing with okayama' model
    """
    def __init__(self, extracted_data: Extracted_data, fit_constants: Fit_constants):
        """
        extracted_data (Extracted_data): Data loaded from the .npz solver output
        fit_constants (Fit_constants): Okayama model parameters for comparison
        """
        self.extracted_data = extracted_data
        self.fit_constants = fit_constants

        self.decomposition=Decomposition(self.extracted_data)

        self.graphs=Graphs(self.extracted_data, self.decomposition, self.fit_constants)

    def decomposition_calculation(self):
        """Compute the multipolar expansion (Phi0, Phi2, Phi4) ."""
        self.decomposition.components()

    def decomposition_graph(self):
        """Visualize the calculated multipolar components overlaid with geometry."""
        self.graphs.graph_components()

    def fit_calculation(self):
        """Generate the theoretical Okayama fit curves."""
        self.fit_constants.fit_function()
    
    def fit_graph(self):
        """Compare the BEM decomposition with the theoretical fitting functions."""
        self.graphs.graph_fit()

class Data_exploitation_whitoutfit:
    """
    Decomposing the BEM results into multipolar components 
    """
    def __init__(self, extracted_data: Extracted_data):
        """
        extracted_data (Extracted_data): Data loaded from the .npz solver output
        """
        self.extracted_data = extracted_data
        self.decomposition=Decomposition(self.extracted_data)

        self.graphs=Graphs(self.extracted_data, self.decomposition)

    def decomposition_calculation(self):
        """Compute the multipolar expansion (Phi0, Phi2, Phi4) ."""
        self.decomposition.components()

    def decomposition_graph(self):
        """Visualize the calculated multipolar components overlaid with geometry."""
        self.graphs.graph_components()

    
    def fit_graph(self):
        """Compare the BEM decomposition with the theoretical fitting functions."""
        self.graphs.graph_fit(False)


class ParaxialSimulation:
    """
    Simulating ion trajectories through quadrupole(s) using paraxial approximations
    """

    def __init__(self, extracted: Extracted_data, decomp: Decomposition):
        """"
        extracted (Extracted_data): Geometry and axis data
        decomp (Decomposition): Multipolar components 
        """
    
        self.extracted = extracted
        self.decomp = decomp
        
        self.traj = Trajectory()

    def run_discrete(self, ion_chief: Ion, ion_marginal: Ion):
        """
        Run a discrete step simulation for two specific ions
        Args:
            ion_chief (Ion)
            ion_marginal (Ion)
        """
        #Run RK4 integration
        self.traj.simulation_discrete(ion_chief, self.extracted, self.decomp)
        self.traj.simulation_discrete(ion_marginal, self.extracted, self.decomp)

        #Plot
        self.traj.plot_discrete(ion_chief, ion_marginal, self.extracted)


    def run_convergence(self, ion_chief: Ion, ion_marginal: Ion, n: int):
        """
        Verify the convergence of RK4 method
        
        Args:
            n (int): Convergence factor
        """

        #Convergence calculus
        self.traj.convergence(self.extracted, self.decomp, n)

        #Continuus simulation
        self.traj.simulation_continuus(ion_chief, self.extracted)
        self.traj.simulation_continuus(ion_marginal, self.extracted)

        #Plot
        self.traj.plot_continuus(ion_chief, ion_marginal, self.extracted, n)


    def run_beam(self, ion_list: list):
        """
        Simulates a full beam  of ions

        Args:
            ion_list (list[Ion]): A list of Ion objects with varying initial conditions
        """

        for ion in ion_list:
            ion.history_x = []
            ion.history_y = []
            self.traj.simulation_discrete(ion, self.extracted, self.decomp)

        self.traj.plot_ray(ion_list, self.extracted)

    def plot_analytical_solution(self, data: Extracted_data):
        self.traj.analytical_solution(data)