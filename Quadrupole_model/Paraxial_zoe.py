import numpy as np
import matplotlib.pyplot as plt
from Data import Data
from Extraction_data import Extracted_data
from Multipolar_decomposition import Decomposition
from scipy.interpolate import interp1d

class Paraxial:
    def __init__ (self, extracted_data):
        self.extracted_data = extracted_data

        self.result = None

    def RK4(self, f, y, h, t, alpha, beta):
        k1 = f(y, t, alpha, beta)
        k2 = f(y + h*k1/2, t + h/2, alpha, beta)
        k3 = f(y + h*k2/2, t + h/2, alpha, beta)
        k4 = f(y + h*k3, t + h, alpha, beta)
        self. result = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
