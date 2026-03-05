import numpy as np
import os
from Extraction_data import Extracted_data

class Reconstruction:
    def __init__(self, output_dir, tensions_reelles: dict):
        """
        Reconstitue le potentiel complet 
        tensions_reelles : dict {'va1':int, 'va2':int, 'vq13':int, 'vq24':int , vaAc:int}
        """
        self.v = tensions_reelles
        self.output_dir = output_dir

        # 1. Chemins des fichiers de base (unitaires)
        ap1_path = os.path.join(output_dir, "aperture1.npz")
        ap2_path = os.path.join(output_dir, "aperture2.npz")
        quad13_path = os.path.join(output_dir, "quad13.npz")
        quad24_path = os.path.join(output_dir, "quad24.npz")

        # 2. Chargement des données via la classe Extracted_data
        d_ap1 = Extracted_data(ap1_path)
        d_ap2 = Extracted_data(ap2_path)
        d_q13 = Extracted_data(quad13_path)
        d_q24 = Extracted_data(quad24_path)

        # 3. Paramètres géométriques et fixes (extraits de la base 1)
        self.points = d_ap1.points
        self.axe_z = d_ap1.axe_z
        self.radius_axis = d_ap1.radius_axis
        
        # Identifiants de groupes
        self.group_id_ap1 = d_ap1.group_id_ap1
        self.group_id_ap2 = d_ap1.group_id_ap2
        self.group_id_cyl1 = d_ap1.group_id_cyl1
        self.group_id_cyl2 = d_ap1.group_id_cyl2
        self.group_id_cyl3 = d_ap1.group_id_cyl3
        self.group_id_cyl4 = d_ap1.group_id_cyl4
        self.group_id_shield = d_ap1.group_id_shield

        # Paramètres de maillage
        self.MeshSizeMin = d_ap1.MeshSizeMin
        self.MeshSizeMax = d_ap1.MeshSizeMax
        self.MeshSizeFromCurvature = d_ap1.MeshSizeFromCurvature

        # Dimensions
        self.dist_shield_apert = d_ap1.dist_shield_apert
        self.dist_apert_quad = d_ap1.dist_apert_quad
        self.radius_ext_shield = d_ap1.radius_ext_shield
        self.radius_in_shield = d_ap1.radius_in_shield
        self.thickness_shield = d_ap1.thickness_shield
        self.radius_apert = d_ap1.radius_apert
        self.thickness_apert = d_ap1.thickness_apert
        self.length_cylinder = d_ap1.length_cylinder
        self.total_length = d_ap1.total_length

        # Coordonnées
        self.coord_cylinder_x_or_y = d_ap1.coord_cylinder_x_or_y
        self.coord_cylinder_z = d_ap1.coord_cylinder_z
        self.coord_apert_z1 = d_ap1.coord_apert_z1
        self.coord_apert_z2 = d_ap1.coord_apert_z2
        self.start_shield1 = d_ap1.start_shield1
        self.end_shield1 = d_ap1.end_shield1
        self.start_apert1 = d_ap1.start_apert1
        self.end_apert1 = d_ap1.end_apert1
        self.start_cyl = d_ap1.start_cyl
        self.end_cyl = d_ap1.end_cyl
        self.start_apert2 = d_ap1.start_apert2
        self.end_apert2 = d_ap1.end_apert2
        self.start_shield2 = d_ap1.start_shield2
        self.end_shield2 = d_ap1.end_shield2

        #somme pot unitaire / Vrelle que l'on veut appliquer
        v = self.v
        
        self.potential = (v["va1"] * d_ap1.potential + v["va2"] * d_ap2.potential + 
                          v["vq13"] * d_q13.potential + v["vq24"] * d_q24.potential)

        self.D1 = (v["va1"] * d_ap1.D1 + v["va2"] * d_ap2.D1 + 
                   v["vq13"] * d_q13.D1 + v["vq24"] * d_q24.D1)
        
        self.D2 = (v["va1"] * d_ap1.D2 + v["va2"] * d_ap2.D2 + 
                   v["vq13"] * d_q13.D2 + v["vq24"] * d_q24.D2)

        self.D3 = (v["va1"] * d_ap1.D3 + v["va2"] * d_ap2.D3 + 
                   v["vq13"] * d_q13.D3 + v["vq24"] * d_q24.D3)
        
        self.D4 = (v["va1"] * d_ap1.D4 + v["va2"] * d_ap2.D4 + 
                   v["vq13"] * d_q13.D4 + v["vq24"] * d_q24.D4)

        # Potentiel sur l'axe (D0)
        self.D0 = self.potential[0]

        # Tensions finales
        self.Vapert1 = v["va1"]
        self.Vapert2 = v["va2"]
        self.Velectrode13 = v["vq13"]
        self.Velectrode24 = v["vq24"]
        self.Vacceleration = v['vaAc']

    def derivative(self) -> None:
        """ Calcul des dérivées spécifiques pour les trajectoires paraxiales """
        self.D2zphi0 = self.D2[5]  # phi_0''
        self.D1zphi0 = self.D1[2]  # phi_0'

    def save(self, filename="quad_reconstuire.npz"):
        
        save_path = os.path.join(self.output_dir, filename)
        np.savez_compressed(
            save_path,
            points=self.points,
            potential=self.potential,
            E_eval=self.D1,
            D2_eval=self.D2,
            D3_eval=self.D3,
            D4_eval=self.D4,


            pot_apert1=self.Vapert1,
            pot_apert2=self.Vapert2,
            pot_electrode13=self.Velectrode13,
            pot_electrode24=self.Velectrode24,
            pot_acceleration=self.Vacceleration,
            pot_shield=0,


            radius_axis=self.radius_axis,
            total_length=self.total_length,
            output_dir=self.output_dir,
            group_id_ap1=self.group_id_ap1,
            group_id_ap2=self.group_id_ap2,
            group_id_cyl1=self.group_id_cyl1,
            group_id_cyl2=self.group_id_cyl2,
            group_id_cyl3=self.group_id_cyl3,
            group_id_cyl4=self.group_id_cyl4,
            group_id_shield=self.group_id_shield,


            MeshSizeMin=self.MeshSizeMin,
            MeshSizeMax=self.MeshSizeMax,
            MeshSizeFromCurvature=self.MeshSizeFromCurvature,

            
            dist_shield_apert=self.dist_shield_apert,
            dist_apert_quad=self.dist_apert_quad,
            radius_ext_shield=self.radius_ext_shield,
            radius_in_shield=self.radius_in_shield,
            thickness_shield=self.thickness_shield,
            radius_apert=self.radius_apert,
            thickness_apert=self.thickness_apert,
            length_cylinder=self.length_cylinder,
            coord_cylinder_x_or_y=self.coord_cylinder_x_or_y,
            coord_cylinder_z=self.coord_cylinder_z,
            coord_apert_z1=self.coord_apert_z1,
            coord_apert_z2=self.coord_apert_z2,
            start_shield1=self.start_shield1,
            end_shield1=self.end_shield1,
            start_apert1=self.start_apert1,
            end_apert1=self.end_apert1,
            start_cyl=self.start_cyl,
            end_cyl=self.end_cyl,
            start_apert2=self.start_apert2,
            end_apert2=self.end_apert2,
            start_shield2=self.start_shield2,
            end_shield2=self.end_shield2
        )
        print(f"Fichier reconstruit sauvegardé : {save_path}")

        return save_path

