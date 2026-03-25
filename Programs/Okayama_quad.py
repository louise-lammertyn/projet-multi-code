from Reconstruction import Reconstruction
from Main_functions import Generation_quad
import os
import numpy as np 

#This code creates a serie of n quadrupoles
class Okayama_quad():
    def __init__(self, output_dir, zquad : list, tension_list : list):
        """
        zquad = list of quadrupoles' offsets 
        List of the tensions of n quadrupoles
        """
        self.output_dir = output_dir
        self.tension_list = tension_list
        self.zquad = zquad #position of the different quadrupoles

        self.quad1 = Reconstruction(output_dir, tension_list[0]) #to access parameters
        self.zquad = [z - self.quad1.start_apert1 for z in zquad]

        #Creation of total Z axis
        dz = self.quad1.axe_z[1] - self.quad1.axe_z[0]

        #Definition of a long end of line
        z_max = max(zquad) + self.quad1.total_length + 20 
        self.axe_zt = np.arange(0, z_max, dz)
        lenz = len(self.axe_zt)


        #Matrix to store sum of potentials
        self.potential_total = np.zeros((self.quad1.potential.shape[0],lenz))
        self.D1_total = np.zeros((self.quad1.D1.shape[0],lenz))
        self.D2_total = np.zeros((self.quad1.D2.shape[0],lenz))
        self.D3_total = np.zeros((self.quad1.D3.shape[0],lenz))
        self.D4_total = np.zeros((self.quad1.D4.shape[0],lenz))
        
        #Sum of quads
        for i, z_off in enumerate(zquad):
            quad_i = self.quad1 = Reconstruction(output_dir, tension_list[i])
            self.add_matrix(self.potential_total, quad_i.potential,quad_i.axe_z, z_off)
            self.add_matrix(self.D1_total, quad_i.D1,quad_i.axe_z, z_off)
            self.add_matrix(self.D2_total, quad_i.D2,quad_i.axe_z, z_off)
            self.add_matrix(self.D3_total, quad_i.D3,quad_i.axe_z, z_off)
            self.add_matrix(self.D4_total, quad_i.D4,quad_i.axe_z, z_off)

    #Interpolation of every quadrupole's potential to obtain the global potential because 
    #we cannot sum discrte functions because the steps need to be aligned
    def add_matrix(self, total_array, local_array, local_z, offset):
        for j in range(total_array.shape[0]):
            interpolation = np.interp(self.axe_zt, local_z + offset, local_array[j, :], left=0, right=0)
            total_array[j, :] += interpolation

    def save(self, filename="okayama_quad_total.npz"):
        save_path = os.path.join(self.output_dir, filename)
        m = self.quad1
        
        pts = np.zeros((3, len(self.axe_zt)))
        pts[2] = self.axe_zt

        np.savez_compressed(
            save_path,
            # --- Necessity for Extracted_data ---
            points=pts,
            potential=self.potential_total,
            E_eval=self.D1_total,
            D2_eval=self.D2_total,
            D3_eval=self.D3_total,
            D4_eval=self.D4_total,
            
            # --- Tensions ---
            pot_acceleration=m.Vacceleration,
            pot_electrode13=m.Velectrode13,
            pot_electrode24=m.Velectrode24,
            pot_apert1=m.Vapert1,
            pot_apert2=m.Vapert2,
            pot_shield=0.0,
            
            # --- Geometry & Mesh ---
            z_offsets = np.array(self.zquad),
            output_dir=self.output_dir,
            total_length=float(self.axe_zt[-1]),
            radius_axis=m.radius_axis,
            dist_shield_apert=m.dist_shield_apert,
            dist_apert_quad=m.dist_apert_quad,
            radius_ext_shield=m.radius_ext_shield,
            radius_in_shield=m.radius_in_shield,
            thickness_shield=m.thickness_shield,
            radius_apert=m.radius_apert,
            thickness_apert=m.thickness_apert,
            length_cylinder=m.length_cylinder,
            
            # --- Coordinates ---
            coord_cylinder_x_or_y=m.coord_cylinder_x_or_y,
            coord_cylinder_z=m.coord_cylinder_z,
            coord_apert_z1=m.coord_apert_z1,
            coord_apert_z2=m.coord_apert_z2,
            start_cyl=m.start_cyl,
            end_cyl=m.end_cyl,
            start_shield1=m.start_shield1, end_shield1=m.end_shield1,
            start_apert1=m.start_apert1, end_apert1=m.end_apert1,
            start_apert2=m.start_apert2, end_apert2=m.end_apert2,
            start_shield2=m.start_shield2, end_shield2=m.end_shield2,

            # --- IDs ---
            group_id_ap1=m.group_id_ap1, group_id_ap2=m.group_id_ap2,
            group_id_cyl1=m.group_id_cyl1, group_id_cyl2=m.group_id_cyl2,
            group_id_cyl3=m.group_id_cyl3, group_id_cyl4=m.group_id_cyl4,
            group_id_shield=m.group_id_shield,
            MeshSizeMin=m.MeshSizeMin, MeshSizeMax=m.MeshSizeMax,
            MeshSizeFromCurvature=m.MeshSizeFromCurvature
        )
        print(f" File saved for Extracted_data : {save_path}")
        return save_path