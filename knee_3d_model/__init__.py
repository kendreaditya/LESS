import numpy as np
from numpy import sin, cos, pi
from scipy.integrate import odeint
from sympy import symbols, Function, Matrix, simplify, lambdify
from sympy.physics.mechanics import (ReferenceFrame, Point, RigidBody, 
                                     inertia, KanesMethod, dynamicsymbols)

class Knee3DModel:
    def __init__(self):
        # Declare dynamical symbols
        self.knee_angle, self.knee_angle_vel = dynamicsymbols('theta omega')
        t = dynamicsymbols._t
        
        # Constants
        self.femur_length, self.tibia_length = symbols('l_femur, l_tibia')
        self.femur_mass, self.tibia_mass = symbols('m_femur, m_tibia')
        self.g = symbols('g')
        
        # Reference frames
        self.N = ReferenceFrame('N')  # Inertial frame
        self.F = self.N.orientnew('F', 'Axis', [self.knee_angle, self.N.z])  # Femur frame
        
        # Define points
        self.O = Point('O')  # Origin
        self.Fc = self.O.locatenew('Fc', self.femur_length/2 * self.F.y)  # Femur center of mass
        self.K = self.O.locatenew('K', self.femur_length * self.F.y)  # Knee joint
        self.Tc = self.K.locatenew('Tc', -self.tibia_length/2 * self.F.y)  # Tibia center of mass
        
        # Define velocities
        self.O.set_vel(self.N, 0)
        self.K.v2pt_theory(self.O, self.N, self.F)
        self.Fc.v2pt_theory(self.O, self.N, self.F)
        self.Tc.v2pt_theory(self.K, self.N, self.F)
        
        # Define rigid bodies
        self.femur_inertia = inertia(self.F, self.femur_mass/12 * self.femur_length**2, 0, 
                                     self.femur_mass/12 * self.femur_length**2)
        self.tibia_inertia = inertia(self.F, self.tibia_mass/12 * self.tibia_length**2, 0, 
                                     self.tibia_mass/12 * self.tibia_length**2)
        
        self.femur = RigidBody('Femur', self.Fc, self.F, self.femur_mass, (self.femur_inertia, self.Fc))
        self.tibia = RigidBody('Tibia', self.Tc, self.F, self.tibia_mass, (self.tibia_inertia, self.Tc))
        
        # Generalized coordinates and speeds
        self.q = Matrix([self.knee_angle])
        self.u = Matrix([self.knee_angle_vel])
        
        # Kinematical differential equations
        self.kd = [self.knee_angle.diff(t) - self.knee_angle_vel]
        
        # Forces
        self.gravity = (self.Fc, -self.femur_mass*self.g*self.N.y)
        self.gravity_t = (self.Tc, -self.tibia_mass*self.g*self.N.y)
        
        # Kane's method
        self.KM = KanesMethod(self.N, q_ind=[self.knee_angle], u_ind=[self.knee_angle_vel], 
                              kd_eqs=self.kd)
        
        self.fr, self.frstar = self.KM.kanes_equations([self.femur, self.tibia], 
                                                       loads=[self.gravity, self.gravity_t])
        
        # Generate equations of motion
        self.mass_matrix = self.KM.mass_matrix_full
        self.forcing_vector = self.KM.forcing_full
        
        # Debug: Print matrices
        print("Mass Matrix:")
        print(simplify(self.mass_matrix))
        print("\nForcing Vector:")
        print(simplify(self.forcing_vector))
        
        self.parameters = [self.femur_length, self.tibia_length, self.femur_mass, 
                           self.tibia_mass, self.g]
        self.parameter_vals = [0.4, 0.4, 5.0, 4.0, 9.81]  # Example values

        # Create lambda functions
        self.mass_matrix_func = lambdify([self.knee_angle] + self.parameters, self.mass_matrix)
        self.forcing_vector_func = lambdify([self.knee_angle, self.knee_angle_vel] + self.parameters, self.forcing_vector)
                # Add ligament parameters
        self.k_acl = 5000  # N/m, ACL stiffness
        self.k_pcl = 6000  # N/m, PCL stiffness
        self.k_mcl = 4000  # N/m, MCL stiffness
        
        self.l0_acl = 0.038  # m, ACL resting length
        self.l0_pcl = 0.038  # m, PCL resting length
        self.l0_mcl = 0.094  # m, MCL resting length

    def derivatives(self, state, t):
        theta, omega = state
        params = self.parameter_vals
        
        M = self.mass_matrix_func(theta, *params)
        F = self.forcing_vector_func(theta, omega, *params)
        
        # Solve M * x = F for x
        x = np.linalg.solve(M, F)
        
        return [omega, x[1,0]]

    def simulate(self, initial_conditions, t):
        # Simulate the system
        y = odeint(self.derivatives, initial_conditions, t)
        return y

    def calculate_ligament_forces(self, state):
        theta, omega = state

        # Calculate ligament lengths based on knee angle
        l_acl = self.l0_acl + 0.01 * sin(theta)  # Simplified model
        l_pcl = self.l0_pcl - 0.01 * sin(theta)  # Simplified model
        l_mcl = self.l0_mcl + 0.005 * (1 - cos(theta))  # Simplified model

        # Calculate ligament strains
        strain_acl = max(0, (l_acl - self.l0_acl) / self.l0_acl)
        strain_pcl = max(0, (l_pcl - self.l0_pcl) / self.l0_pcl)
        strain_mcl = max(0, (l_mcl - self.l0_mcl) / self.l0_mcl)

        # Calculate ligament forces using Hooke's law
        acl_force = self.k_acl * strain_acl
        pcl_force = self.k_pcl * strain_pcl
        mcl_force = self.k_mcl * strain_mcl

        return acl_force, pcl_force, mcl_force

    def simulate(self, initial_conditions, t):
        # Simulate the system
        y = odeint(self.derivatives, initial_conditions, t)
        return y

if __name__ == '__main__':
    # Example usage
    knee_model = Knee3DModel()
    t = np.linspace(0, 10, 1000)
    initial_conditions = [0, 0]  # Initial angle and angular velocity
    simulation_results = knee_model.simulate(initial_conditions, t)

    # Calculate ligament forces for the last time step
    final_state = simulation_results[-1]
    acl_force, pcl_force, mcl_force = knee_model.calculate_ligament_forces(final_state)

    print(f"Final ACL force: {acl_force:.2f} N")
    print(f"Final PCL force: {pcl_force:.2f} N")
    print(f"Final MCL force: {mcl_force:.2f} N")

    # Plot results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(t, simulation_results[:, 0], label='Knee Angle')
    plt.plot(t, simulation_results[:, 1], label='Angular Velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad) / Angular Velocity (rad/s)')
    plt.title('Knee Motion Simulation')
    plt.legend()
    plt.grid(True)
    plt.show()