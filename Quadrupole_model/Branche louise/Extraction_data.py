import numpy as np

#Class to extract all the necessary informations from the files
class Extracted_data:
    """
        Class to extract the data of the files.
        Z_off is used only if we use more than one quadrupole, it represents the offset of space between quadrupoles.
        """
    
    def __init__(self, file_path: str, z_off = None) -> None:

        #Files load 
        data = np.load(file_path)

        
        #Axes data
        self.points = data["points"]
        self.axe_z = data["points"][2] 
        self.z_off = z_off
        
        #Potential data
        self.potential = data["potential"] #Full matrix [V]

        #Identifiers of each groups (Physical Groups from GMSH)
        self.group_id_ap1 = data["group_id_ap1"]
        self.group_id_ap2 = data["group_id_ap2"]
        self.group_id_cyl1 = data["group_id_cyl1"]
        self.group_id_cyl2 = data["group_id_cyl2"]
        self.group_id_cyl3 = data["group_id_cyl3"]
        self.group_id_cyl4 = data["group_id_cyl4"]
        self.group_id_shield = data["group_id_shield"]

        #Mesh parameters
        self.MeshSizeMin = data["MeshSizeMin"]
        self.MeshSizeMax = data["MeshSizeMax"]
        self.MeshSizeFromCurvature = data["MeshSizeFromCurvature"]

        #Output directory
        self.output_dir = data["output_dir"]

        #Dimensions and geometry
        self.dist_shield_apert = data["dist_shield_apert"]
        self.dist_apert_quad = data["dist_apert_quad"]
        self.radius_ext_shield = data["radius_ext_shield"]
        self.radius_in_shield = data["radius_in_shield"]
        self.thickness_shield = data["thickness_shield"]
        self.radius_apert = data["radius_apert"]
        self.thickness_apert = data["thickness_apert"]
        self.length_cylinder = data["length_cylinder"]
        self.radius_axis = data["radius_axis"]
        self.total_length = data["total_length"]

        #Coordinates
        self.coord_cylinder_x_or_y = data["coord_cylinder_x_or_y"]
        self.coord_cylinder_z = data["coord_cylinder_z"]
        self.coord_apert_z1 = data["coord_apert_z1"]
        self.coord_apert_z2 = data["coord_apert_z2"]
        self.start_shield1 = data["start_shield1"]
        self.end_shield1 = data["end_shield1"]
        self.start_apert1 = data["start_apert1"]
        self.end_apert1 = data["end_apert1"]
        self.start_cyl = data["start_cyl"]
        self.end_cyl = data["end_cyl"]
        self.start_apert2 = data["start_apert2"]
        self.end_apert2 = data["end_apert2"]
        self.start_shield2 = data["start_shield2"]
        self.end_shield2 = data["end_shield2"]
        
        # Values of applied potentials
        self.Vacceleration = data["pot_acceleration"]
        self.Velectrode13 = data["pot_electrode13"]
        self.Velectrode24 = data["pot_electrode24"]
        self.Vapert1 = data["pot_apert1"]
        self.Vapert2 = data["pot_apert2"]
        self.Vshield = data["pot_shield"]

        #Derivatives
        self.D0 = data["potential"][0] #Potential along the axis
        self.D1 = data["E_eval"] #Electric field [V/mm]
        self.D2 = data["D2_eval"] #Second derivative [V/mm^2]
        self.D3 = data["D3_eval"] #Third derivative [V/mm^3
        self.D4 = data["D4_eval"] #Fourth derivative [V/mm^4
        

    def derivative(self)-> None:
        """
        Computation of the derivatives of phi0 to use in paraxial equation
        """  
        self.D2zphi0 = self.D2[5] #phi_0'' for trajectory
        self.D1zphi0 = self.D1[2] #phi_0' for trajectory

    
    def position_quad(self) :
        """Function useful if simulation of multiple quadrupoles."""
        if self.z_off is None:
            print("Only one quadrupole")
            return
          
        self.pos_ap1 = [z + self.coord_apert_z1 for z in self.z_off]
        self.pos_ap2 = [z + self.coord_apert_z2 for z in self.z_off]
        self.pos_cyl_start = [z + self.start_cyl for z in self.z_off]
        self.pos_cyl_end = [z + self.end_cyl for z in self.z_off]