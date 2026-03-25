import numpy as np
import matplotlib.pyplot as plt
from Data import Data
from Extraction_data import Extracted_data
from Multipolar_decomposition import Decomposition
from scipy.interpolate import interp1d


#Definition of a class to calculate ion's trajectories using RK4
class Paraxial_trajectories():
    def __init__(self):

        self.y_next = None

    #Runge-Kutta 4 function
    def RK4_step(self, f, y, t, h, alpha, beta):
        """
        alpha and beta are the coefficients of the differential equation: x'' + alpha x' + beta x= 0
        """
        f1 = f(y, t, alpha, beta)
        f2 = f(y + h*f1/2, t + h/2, alpha, beta)
        f3 = f(y + h*f2/2, t + h/2, alpha, beta)
        f4 = f(y + h*f3, t + h, alpha, beta)
        return y + (h/6)*(f1 + 2*f2 + 2*f3 + f4) 


#Definition of a class to determine ion's properties
class Ion(Paraxial_trajectories): 
    def __init__(self, mass, charge, name, x, vx, y, vy):
        #Use of super() function to give access methods and properties of a parent or sibling class
        super().__init__()
        
        self.mass = mass
        self.charge = charge
        self.name = name
        
        self.state_x = np.array([x, vx], dtype=float)
        self.state_y = np.array([y, vy], dtype=float)
        
        #Variable to memorize past states
        self.history_x = []
        self.history_y = []

    #Definition of a printing function
    def print_details(self):
        print(f"Ion: {self.name}, Masse: {self.mass}, Charge: {self.charge}")
    
    #Function to memorize past states
    def save_step(self): 
        self.history_x.append(self.state_x[0])
        self.history_y.append(self.state_y[0])


#Definition of a class to simulate trajectories
class Trajectoire(Paraxial_trajectories):
    def __init__(self):
        #Use of super() function to give access methods and properties of a parent or sibling class
        super().__init__()
        pass

    
    #Function of a second degree equation
    def equation(self, y, t, alpha, beta) -> None:
        """
        Second order equation to solve
        y, t, alpha (first order coefficient), beta (second order coefficient)
        """
        u = y[0] #initial position
        v = y[1] #initial velocity
        du = v #derivative of initial position = initial velocity
        dv = -alpha * u - beta * v #x'' = -alpha*x' - beta*x = - alpha*v - beta*u
        return np.array([du, dv])
    
    #Function that simulates ions' trajectories
    def simulation(self, ion : Ion,  data : Extracted_data, decomp : Decomposition )-> None:
        """
        class Ion
        class Data 
        class Decomposition
        """
        V_acc = data.Vacceleration
        print(V_acc)
        

        dz_mm = data.axe_z[1] - data.axe_z[0] #z step
        ion.save_step()
        
        #------Second order differential equation-------------
        #Form : x'' + alpha x' + beta x = 0
        #x''+ (phi0'/2phi0)x'+ ((phi0''/4phi0)-(phi2/phi0))x=0
        #Here we have:
        #---> alpha = (phi0'/2phi0)
        #---> beta = (phi0''/4phi0)-(phi2/phi0)

        for i in range(len(data.axe_z) - 1):
            phi_total = V_acc + decomp.Phi0_maj[i] #Comment connaître la jauge? 
            
            #To make sure coefficient are not too big
            #if abs(phi_total) < 0.1:
                #phi_total = 0.1 if phi_total >= 0 else -0.1

            terme_axial = data.D2zphi0[i] / (4 * phi_total) #(phi0''/4phi0)
            terme_quad = decomp.Phi2_maj[i] / phi_total #(phi2/phi0)
            
            alphax = terme_axial - terme_quad #(phi0''/4phi0)-(phi2/phi0)
            alphay = terme_axial + terme_quad #(phi0''/4phi0)+(phi2/phi0) 
            beta = data.D1zphi0[i] / (2 * phi_total) #(phi0'/2phi0)
            
            #We send this equation to RK4 to be solved  
            ion.state_x = self.RK4_step(self.equation, ion.state_x, data.axe_z[i], dz_mm, alphax, beta)
            ion.state_y = self.RK4_step(self.equation, ion.state_y, data.axe_z[i], dz_mm, alphay, beta)
            
            #Save the position
            ion.save_step()

    #To verify that RK4 method converges
    def convergence(self, data : Extracted_data,decomp : Decomposition, n : int) -> None:
        """
        Data (type extraction)
        n (int) convergence order that needs to be verified 
        """

        #Redefinition of the step and the z axis
        n_points= len(data.axe_z)*n
        self.z_conv = np.linspace(data.axe_z[0], data.axe_z[-1], n_points)
        self.dz_conv = self.z_conv[1]-self.z_conv[0]

        #Creation of continuus fonctions to verify the convergence of our discrete functions
        self.f_phi0 = interp1d(data.axe_z, decomp.Phi0_maj, kind = 'cubic')
        self.f_phi2 = interp1d(data.axe_z, decomp.Phi2_maj, kind = 'cubic')
        self.f_phi4 = interp1d(data.axe_z, decomp.Phi4_maj, kind = 'cubic')
        self.f_D1zphi0 = interp1d(data.axe_z, data.D1zphi0, kind = 'cubic')
        self.f_D2zphi0 = interp1d(data.axe_z, data.D2zphi0, kind = 'cubic')


    #To verify that the RK4 method functions and simulates the same way as continuus functions
    def simulationf(self, ion : Ion, data : Extracted_data ) -> None:
        """
        class ion 
        class extraction 
        """
        V_acc = data.Vacceleration
        ion.save_step()
        
        for i in self.z_conv[:-1]:
            phi_total = V_acc + self.f_phi0(i)
                
            if abs(phi_total) < 0.1:
                phi_total = 0.1 if phi_total >= 0 else -0.1

            terme_axial = self.f_D2zphi0(i) / (4 * phi_total)
            terme_quad = self.f_phi2(i) / phi_total
                
            alphax = terme_axial - terme_quad
            alphay = terme_axial + terme_quad
            beta = self.f_D1zphi0(i) / (2 * phi_total)
                
            #envoie a RK4
            ion.state_x = self.RK4_step(self.equation, ion.state_x, i, self.dz_conv, alphax, beta)
            ion.state_y = self.RK4_step(self.equation, ion.state_y, i, self.dz_conv, alphay, beta)
                
            ion.save_step()


    #Trajectory plot of the functions 
    def plot_discret(self, chief_ray: Ion, marginal: Ion, data : Extracted_data)-> None:
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Trajectoire paraxiale", fontsize=14, fontweight='bold')
        
        ax = axs.flatten()

        # 1. X trajectory (Chief ray vs Marginal ray)
        ax[0].plot(data.axe_z, chief_ray.history_x, 'r-', label="Chief (x)")
        ax[0].plot(data.axe_z, marginal.history_x, 'b-', label="Marginal (x)")
        ax[0].set_title("X Trajectories ")
        ax[0].legend()
        
        # 2. Y trajectory (Chief ray vs Marginal ray)
        ax[1].plot(data.axe_z, chief_ray.history_y, 'r-', label="Principal (y)")
        ax[1].plot(data.axe_z, marginal.history_y, 'b-', label="Marginal (y)")
        ax[1].set_title("Y Trajectories")
        ax[1].legend()
        
        # 3. Chief Ray only (X vs Y)
        ax[2].plot(data.axe_z, chief_ray.history_x, 'r-', label=" rayon chief_ray ")
        ax[2].plot(data.axe_z, chief_ray.history_y, 'b-', label=" rayon chief_ray ")
        ax[2].set_title("rayon princpal ")
        ax[2].legend()

        # 4. Marginal Ray only (X vs Y)
        ax[3].plot(data.axe_z, marginal.history_x, 'r-', label="rayon marginal")
        ax[3].plot(data.axe_z, marginal.history_y, 'b-', label="rayon marginal")
        ax[3].set_title("rayon marginal")
        ax[3].legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        plt.figure(2)
        plt.plot(data.axe_z, marginal.history_x, 'r-', label="rayon marginal")
        plt.plot(data.axe_z, marginal.history_y, 'b-', label="rayon marginal")
        plt.xlabel('mm')
        plt.show()


    # plot pout la convergence 
    def plot_continu(self, chief_ray: Ion, marginal: Ion, data : Extracted_data, n) -> None:
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Paraxial trajectory,  convergence order {n}   ", fontsize=14, fontweight='bold')
        
        ax = axs.flatten()

        # 1. Trajectoire X (Principal vs Marginal)
        ax[0].plot(self.z_conv, chief_ray.history_x, 'r-', label="Chief ray (x)")
        ax[0].plot(self.z_conv, marginal.history_x, 'b-', label="Marginal ray (x)")
        ax[0].set_title("X Trajectories")
        ax[0].legend()

        # 2. Trajectoire Y (Principal vs Marginal)
        ax[1].plot(self.z_conv, chief_ray.history_y, 'r-', label="Chief ray (y)")
        ax[1].plot(self.z_conv, marginal.history_y, 'b-', label="Marginal ray (y)")
        ax[1].set_title("Y Trajectories")
        ax[1].legend()

        # 3. Chief Ray only (X vs Y)
        ax[2].plot(self.z_conv, chief_ray.history_x, 'r-', label="Chief ray (x)")
        ax[2].plot(self.z_conv, chief_ray.history_y, 'r-', label="Chief ray (y)")
        ax[2].set_title("Chief Ray")
        ax[2].legend()

        # 4. Marginal Ray only (X vs Y)
        ax[3].plot(self.z_conv, marginal.history_x, 'b-', label="Marginal ray (x)")
        ax[3].plot(self.z_conv, marginal.history_y, 'b-', label="Marginal ray (y)")
        ax[3].set_title("Marginal Ray")
        ax[3].legend()
  
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_ray(self , ion_list,  data : Extracted_data, ):
        plt.figure(1)
        plt.title("Ion beam")
        for element in ion_list :
            plt.plot(data.axe_z, element.history_x,label=f"x0 = {element.history_x[0]:.2f} mm" )
            plt.legend()
        plt.x_axis =("z (mm)")
        plt.y_axis = ("x axis (mm)")
        plt.show()