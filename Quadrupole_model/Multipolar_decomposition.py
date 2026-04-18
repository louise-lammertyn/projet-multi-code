import numpy as np
from Extraction_data import Extracted_data

class Decomposition:
    def __init__(self, data: Extracted_data) -> None:
        self.data = data 
        self.Vacceleration= data.Vacceleration
        self.axe_z = data.axe_z

        self.Phi0_maj = None
        self.Phi1_maj = None
        self.Phi2_maj = None
        self.Phi3_maj = None
        self.Phi4_maj = None

        self.Phi0_fit = None
        self.Phi2_fit = None
        self.Phi4_fit = None


    def composantes(self) -> None:
        # Force le potentiel à 0 loin de l'entrée (z_min)

        self.Phi0_maj= self.data.D0  # potentiel monopolaire sur l’axe
        self.Phi1_maj = self.data.D1[0] # D1 correspond au champs electrostique -> on met un moins
        self.Phi2_maj = (1/4)*(self.data.D2[0] - self.data.D2[3])
        self.Phi3_maj = (1/24) * (self.data.D3[0] - 3*self.data.D3[3])
        self.Phi4_maj = (1/192)* (self.data.D4[0] + self.data.D4[10] - 6*self.data.D4[3])

     
        self.Phi0_fit = self.Phi0_maj / self.data.Vapert1
        self.Phi2_fit = self.Phi2_maj *((self.data.radius_axis**2)/ self.data.Velectrode13)
        self.Phi4_fit = self.Phi4_maj *((self.data.radius_axis**4)/ self.data.Vapert1)
        
        Phi2_max = np.max(np.abs(self.Phi2_fit))

        self.Phi0_fit = self.Phi0_fit/Phi2_max
        self.Phi2_fit = self.Phi2_fit/Phi2_max
        self.Phi4_fit = self.Phi4_fit/Phi2_max
    
   