from reconstruction import Reconstruction
from Main_functions import Generation_quad
import os
import numpy as np 

class Okayama_quad():
    def __init__(self, output_dir, zquad : list, liste_tension : list):
        """
        zquad = liste des offsets des quads
        Liste tensions des n quads
        """
        self.output_dir = output_dir
        self.liste_tension = liste_tension
        self.zquad = zquad

       


        self.quad1 = Reconstruction(output_dir, liste_tension[0])

        #creation axe z total 
        dz = self.quad1.axe_z[1] - self.quad1.axe_z[0]
        # On définit une fin de ligne assez longue
        z_max = max(zquad) + self.quad1.total_length + 20 
        self.axe_zt = np.arange(0, z_max, dz)
        lenz = len(self.axe_zt)
        lenz = len(self.axe_zt)

        #creation des tableaux pour y mettre la somme des potentiel 

        self.potentiel_total = np.zeros((self.quad1.potential.shape[0],lenz))
        self.D1_total = np.zeros((self.quad1.D1.shape[0],lenz))
        self.D2_total = np.zeros((self.quad1.D2.shape[0],lenz))
        self.D3_total = np.zeros((self.quad1.D3.shape[0],lenz))
        self.D4_total = np.zeros((self.quad1.D4.shape[0],lenz))

        #ajout au tableau global  pour chaque quad 

        
        #somme des quads 

        for i, z_off in enumerate(zquad):
            quad_i = self.quad1 = Reconstruction(output_dir, liste_tension[i])
            self.ajout_tab(self.potentiel_total, quad_i.potential,quad_i.axe_z, z_off)
            self.ajout_tab(self.D1_total, quad_i.D1,quad_i.axe_z, z_off)
            self.ajout_tab(self.D2_total, quad_i.D2,quad_i.axe_z, z_off)
            self.ajout_tab(self.D3_total, quad_i.D3,quad_i.axe_z, z_off)
            self.ajout_tab(self.D4_total, quad_i.D4,quad_i.axe_z, z_off)



    def ajout_tab(self, total_array, local_array, local_z, offset):
        for j in range(total_array.shape[0]):
            valeurs_interpolees = np.interp(
                self.axe_zt, 
                local_z + offset, 
                local_array[j, :], 
                left=0, 
                right=0
            )
            total_array[j, :] += valeurs_interpolees

    def save(self, filename="okayama_quad_total.npz"):
        save_path = os.path.join(self.output_dir, filename)
        m = self.quad1
        
        # Création du tableau points (3, N) attendu par Extracted_data
        pts = np.zeros((3, len(self.axe_zt)))
        pts[2] = self.axe_zt

        np.savez_compressed(
            save_path,
            # --- Indispensables pour Extracted_data ---
            points=pts,
            potential=self.potentiel_total,
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
            
            # --- Géométrie & Mesh ---
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
            
            # --- Coordonnées pour les fonctions de position ---
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
        print(f"✅ Fichier sauvegardé pour Extracted_data : {save_path}")
        return save_path









