
def solution_analytique(ion,potentiel,quad1, z0): #champ quadrupolaire  uniquement 
        
        xp = np.zeros(len(potentiel.axe_z)-1)
        xm = np.zeros(len(potentiel.axe_z)-1)
        yp = np.zeros(len(potentiel.axe_z)-1)
        ym = np.zeros(len(potentiel.axe_z)-1)

        for i in range(len(potentiel.axe_z)-1):
            z = potentiel.axe_z[i]
            w0 = np.sqrt(2*ion.charge*potentiel.VQ/ion.masse*(Dimension.radius)**2)
            xp[i] = np.cos(w0*(z-z0))
            xm[i] = (1/w0)*np.sin(w0*(z-z0))
            yp[i] = np.cosh(w0*(z-z0))
            ym[i] = (1/w0)*np.sinh(w0*(z-z0))
        return xp,xm, ym, yp



#xp,xm, ym, yp = solution_analytique(Ion,data, 0)  

'''