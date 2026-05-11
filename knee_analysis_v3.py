import numpy as np
import sympy as sp
from sympy import symbols, sin, cos
from sympy.physics.mechanics import (
    dynamicsymbols, ReferenceFrame, Point, RigidBody, Particle, KanesMethod
)
from pydy.system import System
import matplotlib.pyplot as plt

# Time symbol
t = symbols('t')

# Generalized coordinates (joint angles)
q_knee = dynamicsymbols('q_knee')       # Knee flexion angle
q_hip = dynamicsymbols('q_hip')         # Hip flexion angle
q_valgus = dynamicsymbols('q_valgus')   # Knee valgus angle
q_adduction = dynamicsymbols('q_adduction')  # Hip adduction angle

# Generalized speeds (first derivatives)
qd_knee = dynamicsymbols('q_knee', 1)
qd_hip = dynamicsymbols('q_hip', 1)
qd_valgus = dynamicsymbols('q_valgus', 1)
qd_adduction = dynamicsymbols('q_adduction', 1)

# Inertial reference frame
N = ReferenceFrame('N')

# Thigh reference frame
Thigh = N.orientnew('Thigh', 'Body', [q_hip, q_adduction, 0], 'XYZ')

# Shank reference frame
Shank = Thigh.orientnew('Shank', 'Body', [q_knee, q_valgus, 0], 'XYZ')

# Constants for lengths
l_thigh, l_shank = symbols('l_thigh l_shank')

# Hip joint point
O = Point('O')
O.set_vel(N, 0)

# Knee joint point
K = O.locatenew('K', -l_thigh * Thigh.y)
K.set_vel(N, K.pos_from(O).dt(N))

# Ankle joint point
A = K.locatenew('A', -l_shank * Shank.y)
A.set_vel(N, A.pos_from(O).dt(N))

# Masses and inertias
m_thigh, m_shank = symbols('m_thigh m_shank')
I_thigh = (0, 0, 0)  # Simplified inertia
I_shank = (0, 0, 0)

# Create rigid bodies
Thigh_RB = RigidBody('Thigh', K, Thigh, m_thigh, (I_thigh, K))
Shank_RB = RigidBody('Shank', A, Shank, m_shank, (I_shank, A))

# Ligament attachment points on thigh and shank
# Positions relative to respective frames
P_thigh = K.locatenew('P_thigh', r_thigh * Thigh.x + s_thigh * Thigh.z)
P_shank = K.locatenew('P_shank', r_shank * Shank.x + s_shank * Shank.z)

# Velocities
P_thigh.set_vel(N, P_thigh.pos_from(O).dt(N))
P_shank.set_vel(N, P_shank.pos_from(O).dt(N))

# Ligament stretch calculation
ligament_vector = P_shank.pos_from(P_thigh)
ligament_length = ligament_vector.magnitude()
ligament_rest_length = symbols('l_rest')

# Ligament properties
k_ligament, c_ligament = symbols('k_ligament c_ligament')

# Ligament stretch and strain rate
ligament_stretch = ligament_length - ligament_rest_length
ligament_strain_rate = ligament_vector.dt(N).dot(ligament_vector.normalize())

# Ligament force magnitude (nonlinear spring-damper model)
f_ligament = k_ligament * ligament_stretch + c_ligament * ligament_strain_rate

# Force vector
ligament_force_vector = f_ligament * ligament_vector.normalize()

# Apply forces to the attachment points
force_P_thigh = (P_thigh, -ligament_force_vector)
force_P_shank = (P_shank, ligament_force_vector)

# List of generalized coordinates and speeds
coordinates = [q_knee, q_hip, q_valgus, q_adduction]
speeds = [qd_knee, qd_hip, qd_valgus, qd_adduction]
kinematic_differential_equations = [q.diff(t) - qd for q, qd in zip(coordinates, speeds)]

# External forces (gravity, ligament forces)
g = symbols('g')  # Gravity
forces = [
    (A, m_shank * g * N.y),    # Gravity on shank
    (K, m_thigh * g * N.y),    # Gravity on thigh
    force_P_thigh,
    force_P_shank
]

# Bodies
bodies = [Thigh_RB, Shank_RB]

# Kane's Method
kane = KanesMethod(
    frame=N,
    q_ind=coordinates,
    u_ind=speeds,
    kd_eqs=kinematic_differential_equations
)

(fr, frstar) = kane.kanes_equations(bodies, forces)

# System parameters
parameters = {
    l_thigh: 0.5,    # meters
    l_shank: 0.5,
    m_thigh: 7.0,    # kg
    m_shank: 5.0,
    k_ligament: 10000.0,  # N/m
    c_ligament: 50.0,     # Ns/m
    ligament_rest_length: 0.05,
    g: 9.81
}

# Initial conditions
initial_conditions = {
    q_knee: 0.0,
    q_hip: 0.0,
    q_valgus: 0.0,
    q_adduction: 0.0,
    qd_knee: 0.0,
    qd_hip: 0.0,
    qd_valgus: 0.0,
    qd_adduction: 0.0
}

# Time vector
time = np.linspace(0, 2, 100)  # 2 seconds, 100 steps

# Suppose you have angle data arrays
time_data = np.array([...])         # Time stamps from your video frames
knee_angle_data = np.array([...])   # Knee flexion angles
hip_angle_data = np.array([...])    # Hip flexion angles
valgus_angle_data = np.array([...]) # Knee valgus angles
adduction_angle_data = np.array([...])  # Hip adduction angles

# Interpolate the data
from scipy.interpolate import interp1d

knee_angle_func = interp1d(time_data, knee_angle_data, fill_value="extrapolate")
hip_angle_func = interp1d(time_data, hip_angle_data, fill_value="extrapolate")
valgus_angle_func = interp1d(time_data, valgus_angle_data, fill_value="extrapolate")
adduction_angle_func = interp1d(time_data, adduction_angle_data, fill_value="extrapolate")

# Update kinematic differential equations
kinematic_differential_equations = []

# Speeds are derivatives of prescribed angles
kinematic_differential_equations.append(qd_knee - sp.diff(knee_angle_func(t), t))
kinematic_differential_equations.append(qd_hip - sp.diff(hip_angle_func(t), t))
kinematic_differential_equations.append(qd_valgus - sp.diff(valgus_angle_func(t), t))
kinematic_differential_equations.append(qd_adduction - sp.diff(adduction_angle_func(t), t))

# No generalized coordinates since they are prescribed
coordinates = []

# Update Kane's Method
kane = KanesMethod(
    frame=N,
    q_ind=coordinates,
    u_ind=speeds,
    kd_eqs=kinematic_differential_equations
)

(fr, frstar) = kane.kanes_equations(bodies, forces)

# Parameters for numerical evaluation
params = parameters.copy()
params.update({t: 0})

# Substitute numerical values into symbolic expressions
fr_num = fr.subs(params)
frstar_num = frstar.subs(params)

from scipy.integrate import odeint

def derivatives(y, t):
    # Update parameters with current time
    params[t] = t
    
    # Update joint angles and speeds from pose data
    q_knee_val = float(knee_angle_func(t))
    q_hip_val = float(hip_angle_func(t))
    q_valgus_val = float(valgus_angle_func(t))
    q_adduction_val = float(adduction_angle_func(t))
    
    # Update speeds
    qd_knee_val, qd_hip_val, qd_valgus_val, qd_adduction_val = y

    # Update symbols
    subs = {
        q_knee: q_knee_val,
        q_hip: q_hip_val,
        q_valgus: q_valgus_val,
        q_adduction: q_adduction_val,
        qd_knee: qd_knee_val,
        qd_hip: qd_hip_val,
        qd_valgus: qd_valgus_val,
        qd_adduction: qd_adduction_val
    }
    subs.update(params)
    
    # Evaluate fr and frstar numerically
    fr_evaluated = fr_num.subs(subs)
    frstar_evaluated = frstar_num.subs(subs)
    
    # Solve for accelerations
    # Since there are no coordinates, accelerations are zero
    dydt = [0, 0, 0, 0]  # No accelerations in prescribed motion
    
    return dydt

# Compute velocities from pose data derivatives
qd_knee_data = np.gradient(knee_angle_data, time_data)
qd_hip_data = np.gradient(hip_angle_data, time_data)
qd_valgus_data = np.gradient(valgus_angle_data, time_data)
qd_adduction_data = np.gradient(adduction_angle_data, time_data)

ligament_forces = []

for idx, t_i in enumerate(time_data):
    # Get joint angles at time t_i
    q_knee_val = knee_angle_data[idx]
    q_hip_val = hip_angle_data[idx]
    q_valgus_val = valgus_angle_data[idx]
    q_adduction_val = adduction_angle_data[idx]
    
    # Update symbols
    subs = {
        q_knee: q_knee_val,
        q_hip: q_hip_val,
        q_valgus: q_valgus_val,
        q_adduction: q_adduction_val
    }
    subs.update(params)
    
    # Evaluate ligament stretch
    ligament_length_val = ligament_length.subs(subs)
    ligament_stretch_val = ligament_length_val - params[ligament_rest_length]
    
    # Evaluate ligament force
    f_ligament_val = params[k_ligament] * ligament_stretch_val
    ligament_forces.append(f_ligament_val)

plt.figure(figsize=(10, 6))
plt.plot(time_data, ligament_forces, label='Ligament Force')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('Ligament Force Over Time')
plt.legend()
plt.grid(True)
plt.show()
