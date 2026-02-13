#Importation of all the necessary packages
import os
import numpy as np
from matplotlib import pylab as plt
#import bempp_cl.api as bempp            
#import gmsh
from IPython import get_ipython
import matplotlib
#from bempp_cl.core import opencl_kernels
#import bempp_cl.api as bempp
from dataclasses import dataclass
# da

@dataclass
class dimension:
    """
    Docstring pour dimension
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
    dist_shield_apert: float
    dist_apert_quad: float
    radius_ext_shield: float
    radius_in_shield: float
    thickness_shield: float
    radius_apert: float
    thickness_apert: float
    length_cylinder: float
    radius_axis: float

@dataclass
class Potentials:
    """
    Docstring pour Potentials:
    Potentiel electrode, potentiel aperture 1, potentiel aperture2, potentiel shied, pot acceleration en v
    """
    pot_acceleration: float
    pot_shield: float = 0.0
    
    # On peut utiliser des propriétés calculées pour les dépendances
    @property
    def pot_electrode(self) -> float:
        return -0.0299087 * self.pot_acceleration

    @property
    def pot_apert1(self) -> float:
        return -0.18808 * self.pot_acceleration

    @property
    def pot_apert2(self) -> float:
        return 0.10918 * self.pot_acceleration


class quadrupole:
    """
    Docstring pour quadrupole
    Classe générale du quadrpole 
    """

    
    def __init__(self, va):
        self.va = va
    
    class modélisation():
    



