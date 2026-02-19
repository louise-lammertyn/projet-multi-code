import numpy as np
from Extraction_data import Extracted_data

#Creation of a class that describes characteristic values for the fit function
class Fit_constants:
    def __init__(self, a0: int, b0: int, b2: int, a4: int, b41: int, b42: int, z0: int, a: int, data: Extracted_data) -> None:
        # Paramètres pour la composante lentille ronde k0(Z) :
        # k0(Z) = a0 * exp[-(Z/b0)^2]
        self.a0 = a0 
        self.b0 = b0 
        # Paramètres pour la composante quadripolaire k2(Z) :
        self.b2 = b2

        # Paramètres pour la composante octupolaire k4(Z) :
        self.a4 = a4
        self.b41 = b41
        self.b42 = b42
        self.z0 = z0
        # Paramètre géométrique, rayon elecrtode 
        self.a = a

        self.data = data

        self.k0 = None
        self.k2 = None
        self.k4 = None



    def fonction_fit(self) -> None:

        milieu_cyl = 0.5 * (self.data.start_cyl + self.data.end_cyl)

        #on translte tout
        z_ref = self.data.axe_z - milieu_cyl

        # un axe pour chaque k
        z_quad_center = 0                          # Milieu du cylindre = 0
        z_quad_edge   = self.data.length_cylinder / 2        # Bord du cylindre
        z_ap_center   = z_quad_edge + self.data.dist_apert_quad + self.data.thickness_apert / 2 # Milieu ouverture

        Z_k2 = z_ref - z_quad_center     # pour le quadrupôle
        Z_k4 = z_ref - z_quad_edge       # pour l'octupôle
        Z_k0 = z_ref - z_ap_center       # pour la lentille ronde (aperture)

        # k0 (Lentille ronde) 
        a0, b = 0.80751, 5.08
        self.k0 = a0 * np.exp(-(Z_k0**2) / b**2)

        # k2 (Quadrupôle)
        Z0, b2 = 5, 2.54
        self.k2 = np.where(np.abs(Z_k2) <= Z0, 1, np.exp(-(np.abs(Z_k2) - Z0)**2 / b2**2))

        # k4 (Octupôle)
        a4, b1, b2_k4 = 0.03891461, 3.113, 2.015
        self.k4 = np.where(Z_k4 <= 0, a4 / (1 + (Z_k4**2 / b1**2))**2, a4 * np.exp(-(Z_k4**2) / b2_k4**2))
