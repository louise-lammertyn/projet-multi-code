class Data: 
    """
    #1 - Distance between the sield ant the aperture 
    #2 - Distance between the aperture and the cylinders (d in Okayama's paper)
    #3 - Exterior radius of the shield 
    #4 - Inside radius of the shield 
    #5 - Thickness of the shield
    #6 - Radius of the aperture 
    #7 - Thickness of the aperture 
    #8 - Length of the cylinder (l in Okayama's paper)
    #9 - Radius of the elements around the axis (a in Okayama's paper)
    """
    def __init__(self, 
                dist_shield_apert: int, dist_apert_quad: int,
                radius_ext_shield: int, radius_in_shield: int, thickness_shield: int, 
                radius_apert: int, thickness_apert: int, 
                length_cylinder: int, 
                radius_axis: int,
                pot_electrode13: int,pot_electrode24 :int,  pot_apert1: int, pot_apert2: int, pot_shield: int, pot_acceleration: int,
                MeshSizeMin: int, MeshSizeMax: int, MeshSizeFromCurvature: int, 
                output_dir: str) -> None:
        self.dist_shield_apert = dist_shield_apert
        self.dist_apert_quad = dist_apert_quad

        self.radius_ext_shield = radius_ext_shield
        self.radius_in_shield = radius_in_shield
        self.thickness_shield = thickness_shield

        self.radius_apert = radius_apert
        self.thickness_apert = thickness_apert

        self.length_cylinder = length_cylinder

        self.radius_axis = radius_axis

        self.pot_electrode13 = pot_electrode13 #Potential of cylinders 1&3
        self.pot_electrode24 = pot_electrode24 #Potential of cylinders 2&4
        self.pot_apert1 = pot_apert1 #Potential of the first aperture (round field)
        self.pot_apert2 = pot_apert2 #Potential of the second aperture
        self.pot_shield = pot_shield #Potential of the shield (0V)
        self.pot_acceleration = pot_acceleration #Ion acceleration potential
    
        self.MeshSizeMin = MeshSizeMin
        self.MeshSizeMax = MeshSizeMax
        self.MeshSizeFromCurvature = MeshSizeFromCurvature

        self.output_dir = output_dir
            
        self.group_id = None

        self.total_length = 2*self.dist_shield_apert + 2*self.thickness_shield + 2*self.dist_apert_quad + 2*self.thickness_apert + self.length_cylinder

        self.coord_cylinder_x_or_y = 2*self.radius_axis
        self.coord_cylinder_z = self.thickness_shield + self.dist_shield_apert + self.thickness_apert + self.dist_apert_quad

        self.coord_apert_z1 = self.thickness_shield + self.dist_shield_apert
        self.coord_apert_z2 = self.thickness_shield + self.dist_shield_apert + self.thickness_apert + 2*self.dist_apert_quad + self.length_cylinder

        self.start_shield1 = 0
        self.end_shield1 = self.start_shield1 + self.thickness_shield
        self.start_apert1 = self.end_shield1 + self.dist_shield_apert
        self.end_apert1 = self.start_apert1 + self.thickness_apert
        self.start_cyl = self.end_apert1 + self.dist_apert_quad
        self.end_cyl = self.start_cyl + self.length_cylinder
        self.start_apert2 = self.end_cyl + self.dist_apert_quad
        self.end_apert2 = self.start_apert2 + self.thickness_apert
        self.start_shield2 = self.end_apert2 + self.dist_shield_apert
        self.end_shield2 = self.start_shield2 + self.thickness_shield