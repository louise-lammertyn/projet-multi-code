import numpy as np
import matplotlib.pyplot as plt
from Data import Data
from Extraction_data import Extracted_data
from Multipolar_decomposition import Decomposition
from paraxial import Paraxial,Ion

#a faire avec le rayon marginal

class Aberration():
    def __init__(self)-> None:
        self.C30 = None #coéfficient spérique que l'on veut déterminer 

    def coefficient(self,ion : Ion, data : Decomposition):
        """
        Extraction -> accés au donnée de bemmpp
        ion -> accés à la trajectoire

        Recupère les varible puis on intègre
        """

        z = data.axe_z
        dz = data.axe_z[1] - data.axe_z[0]

        phiA = data.Vacceleration + data.P
        R = data.Phi0_maj/phiA
        Q = data.Phi0_maj/phiA
        O = data.Phi0_maj/phiA


        x = np.array(ion.history_x)
        dx = np.gradient(x, z)
        d2x = np.gradient(dx, z)
        d3x = np.gradient(d2x,z)
        d4x = np.gradient(d3x,z)

        #dérive chaque terme plus fois 

        dphiA = np.gradient(phiA, z)
        ddphiA = np.gradient(dphiA, z)

        dR = np.gradient(R, z)
        d2R = np.gradient(dR, z)
        d3R = np.gradient(d2R, z)
        d4R = np.gradient(d3R, z)

        dQ = np.gradient(Q, z)
        d2Q = np.gradient(dQ, z)
        d3Q = np.gradient(d2Q, z)
        d4Q = np.gradient(d3Q, z)


        dO= np.gradient(O, z)
        d2O = np.gradient(dO, z)
        d3O= np.gradient(d2O, z)
        d4O = np.gradient(d3O, z)

        Tx4 =(-d4R/32 +d2Q/6 + (d2R)**2/16 - ((d2R-Q)/2)+(Q**2)-2*O)*(x**4)
        Tx3x = (-d3R/8 + dQ/2 + (dR*d2R)/8 - (dR*Q)/2)*(x**3)*dx
        Tx2x2 = ((d2R /4) - Q)*(x**2)*(d2x)**2
        Txx3 = (dR/2) *x*((d3x)**3)

        I = np.sqrt(phiA/data.Vacceleration)*(Tx4+Tx3x+Tx2x2+Txx3)

        self.C30 = np.trapezoid(I, z)
        print(self.C30)



        return self.C30


