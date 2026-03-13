import numpy as np
from Extraction_data import Extracted_data

#Creation of a class that describes characteristic values for the fit functions.
class Fit_constants:
    """
    Defines and calculates the characteristic fitting functions (k0, k2, k4) 
    for multipolar components based on Okayama's model.
    """
    def __init__(self, a0: int, b0: int, b2: int, a4: int, b41: int, b42: int, z0: int, a: int, data: Extracted_data) -> None:
        """
        Initializes fit parameters for round lens, quadrupole, and octupole components.

        Args:
            a0, b0: Round lens component (k0).
            b2:  Quadrupole component (k2).
            a4, b41, b42:  Octupole component (k4).
            z0: Half-length of the plateau for the quadrupole field.
            a: Electrode radius.
            data (Extracted_data): Object containing axis and geometry information.
        """
        
        # Round lens (k0) parameters
        # k0(Z) = a0 * exp[-(Z/b0)^2]
        self.a0 = a0 
        self.b0 = b0 

        # Quadrupole (k2) parameters
        self.b2 = b2

        # Octupole (k4) parameters
        self.a4 = a4
        self.b41 = b41
        self.b42 = b42
        self.z0 = z0

        # Geometry
        self.a = a

        self.data = data

        # Resulting function arrays
        self.k0 = None
        self.k2 = None
        self.k4 = None


    def fit_function(self) -> None:
        """
        Calculates the axial distributions for k0, k2, and k4 by shifting the 
        coordinate system to the relevant geometric centers.
        """

        middle_cyl = 0.5 * (self.data.start_cyl + self.data.end_cyl) #middle of the cylinder

        #Translates the Z-axis relative to the cylinder center
        z_ref = self.data.axe_z - middle_cyl

        #Defines reference points for each component
        z_quad_center = 0 #Center of cylinder
        z_quad_edge   = self.data.length_cylinder / 2 #Edge of cylinder
        z_ap_center   = z_quad_edge + self.data.dist_apert_quad + self.data.thickness_apert / 2 #Middle of the aperture

        Z_k2 = z_ref - z_quad_center #For the quadrupole
        Z_k4 = z_ref - z_quad_edge #For the octupole
        Z_k0 = z_ref - z_ap_center #For the round lens (aperture)

        # k0 (Round lens) 
        a0, b = 0.80751, 5.08
        self.k0 = a0 * np.exp(-(Z_k0**2) / b**2)

        # k2 (Quadrupole)
        Z0, b2 = 5, 2.54
        self.k2 = np.where(np.abs(Z_k2) <= Z0, 1, np.exp(-(np.abs(Z_k2) - Z0)**2 / b2**2))

        # k4 (Octupole)
        a4, b1, b2_k4 = 0.03891461, 3.113, 2.015
        self.k4 = np.where(Z_k4 <= 0, a4 / (1 + (Z_k4**2 / b1**2))**2, a4 * np.exp(-(Z_k4**2) / b2_k4**2))