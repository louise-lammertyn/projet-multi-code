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
        self.z_image = None
    

    def coefficient(self, ion_principal: Ion, ion_marginal: Ion, data: Decomposition):
        """
        C_A03 selon Hawkes/Okayama — plan convergent Y.

        ion_principal : y(0)=0, y'(0)=1  → rayon axial (diverge dans Y)
        ion_marginal  : y(0)=1, y'(0)=0  → rayon marginal (converge dans Y)
        """

        self.data = data
        self.C30  = None

        z     = np.array(data.axe_z)
        phi_a = data.Vacceleration
        phiA  = phi_a + data.Phi0_maj
        a     = data.a

        # =====================================================
        # POTENTIELS RÉDUITS (sans unité)
        # =====================================================
        R = data.Phi0_maj        / phiA
        Q = data.Phi2_maj * a**2 / phiA
        O = data.Phi4_maj * a**4 / phiA

        print(f"R max = {np.max(np.abs(R)):.6e}")
        print(f"Q max = {np.max(np.abs(Q)):.6e}")
        print(f"O max = {np.max(np.abs(O)):.6e}")

        # =====================================================
        # RAYON MARGINAL → plan image + Mx
        # =====================================================
        y_m  = np.array(ion_marginal.history_y)
        dy_m = np.gradient(y_m, z)

        # Recherche du passage par zéro (plan image gaussien)
        z_image     = None
        idx_image   = None

        for i in range(len(y_m) - 1):
            if y_m[i] * y_m[i+1] < 0:
                # Interpolation linéaire pour z_image précis
                t       = y_m[i] / (y_m[i] - y_m[i+1])
                z_image = z[i] + t * (z[i+1] - z[i])
                idx_image = i
                break

        if z_image is None:
            # Extrapolation si le rayon ne coupe pas l'axe dans le domaine
            z_image = z[-1] - y_m[-1] / dy_m[-1]
            idx_image = len(z) - 1
            print(f"⚠️  Plan image extrapolé (hors domaine) : z_image = {z_image:.4f} mm")
        else:
            print(f"✅ Plan image trouvé : z_image = {z_image:.4f} mm  (entre z={z[idx_image]:.2f} et z={z[idx_image+1]:.2f})")

        self.z_image = z_image

        # Mx = y_marginal(z_image) / y_marginal(z_objet)
        # Au plan image y_m = 0 par définition → Mx = 0  (ce n'est pas ça qu'on veut)
        #
        # La bonne définition de Mx dans Hawkes est via le Wronskien :
        # W = y_p * y'_m - y_m * y'_p = constante
        # Mx = W / y'_p(z_image)
        #
        # Comme W est constant, on l'évalue au début :

        y_p  = np.array(ion_principal.history_y)
        dy_p = np.gradient(y_p, z)

        W = y_p * dy_m - y_m * dy_p
        print(f"\nWronskien début  = {W[0]:.6e}")
        print(f"Wronskien milieu = {W[len(W)//2]:.6e}")
        print(f"Wronskien fin    = {W[-1]:.6e}")

        # y'_p au plan image (interpolation linéaire)
        if idx_image < len(z) - 1:
            t         = (z_image - z[idx_image]) / (z[idx_image+1] - z[idx_image])
            dyp_image = dy_p[idx_image] + t * (dy_p[idx_image+1] - dy_p[idx_image])
        else:
            dyp_image = dy_p[-1]

        Mx = W[0] / dyp_image
        print(f"\nMx = W / y'_p(z_image) = {W[0]:.6e} / {dyp_image:.6e} = {Mx:.6f}")

        # =====================================================
        # RAYON DE RÉFÉRENCE POUR L'INTÉGRALE
        # =====================================================
        # Convention Okayama : rayon marginal normalisé à y(z_objet)=1
        # → y_m[0] = 1 déjà, donc x = y_m directement

        x  = y_m / y_m[0]    # normalisé : x(z_objet) = 1, x(z_image) = 0
        dx = np.gradient(x, z)

        print(f"\nx[0]   = {x[0]:.6f}  (doit être 1)")
        print(f"x[-1]  = {x[-1]:.6f}  (doit être ~0 si image dans domaine)")
        print(f"x max  = {np.max(np.abs(x)):.6f}")

       # Après avoir trouvé idx_image et z_image :

        # Tronquer tous les tableaux jusqu'au plan image
        z_cut   = z[:idx_image+1]
        x_cut   = x[:idx_image+1]
        dx_cut  = dx[:idx_image+1]
        phiA_cut = phiA[:idx_image+1]
        R_cut   = R[:idx_image+1]

        dR_cut  = np.gradient(R_cut,  z_cut)
        d2R_cut = np.gradient(dR_cut, z_cut)
        d3R_cut = np.gradient(d2R_cut,z_cut)
        d4R_cut = np.gradient(d3R_cut,z_cut)

        Q_cut   = Q[:idx_image+1]
        dQ_cut  = np.gradient(Q_cut,  z_cut)
        d2Q_cut = np.gradient(dQ_cut, z_cut)

        O_cut   = O[:idx_image+1]

        # Termes C_A03 sur le domaine tronqué
        T1 = (-d4R_cut/32 - d2Q_cut/6 + d2R_cut**2/16 
            + d2R_cut*Q_cut/2 + Q_cut**2 + 2*O_cut) * x_cut**4

        T2 = (-d3R_cut/8 - dQ_cut/2 
            + dR_cut*d2R_cut/8 + dR_cut*Q_cut/2) * x_cut**3 * dx_cut

        T3 = (d2R_cut/4 + Q_cut) * x_cut**2 * dx_cut**2

        T4 = (dR_cut/2) * x_cut * dx_cut**3

        I = np.sqrt(phiA_cut / phi_a) * (T1 + T2 + T3 + T4)

        integral = np.trapezoid(I, z_cut)
        self.C30 = Mx**4 * integral

        y_p  = np.array(ion_principal.history_y)
        dy_p = np.gradient(y_p, z)

        # Focale depuis rayon axial
        fy = -y_p[-1] / dy_p[-1]
        print(f"fy = {fy:.4f} mm")

        # Comparer avec valeur théorique Okayama
        # fy_theo ≈ phi_a * a² / (Phi2_max * a² * L)
        L = z[-1] - z[0]
        fy_theo = phi_a / (np.max(Q) * (1/a**2) * L)
        print(f"fy_theo ≈ {fy_theo:.2f} mm")

        # Tension quadrupôle effective
        print(f"Vq13  = {data.Velectrode13:.2f} V")
        print(f"phi_a = {phi_a:.2f} V")
        print(f"ratio = {data.Velectrode13/phi_a:.4f}")
        # Okayama : phi2/phi_a = 651.8/20000 = 0.0326

        print(f"z_cut max  = {z_cut[-1]:.4f} mm  (doit être juste avant {z_image:.4f} mm)")
        print(f"x_cut[-1]  = {x_cut[-1]:.6f}   (doit être ~0)")
        print(f"Intégrale  = {integral:.6e} mm")
        print(f"C_A03      = {self.C30:.4f} mm")

        return self.C30
    

    

   


