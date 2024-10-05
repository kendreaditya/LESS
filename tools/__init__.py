import numpy as np
from scipy.signal import savgol_filter

def calculate_knee_flexion_angle(hip, knee, ankle):
    """Calculate the knee flexion angle."""
    # Vectors
    thigh = np.array([hip.x - knee.x, hip.y - knee.y, hip.z - knee.z])
    shank = np.array([ankle.x - knee.x, ankle.y - knee.y, ankle.z - knee.z])
    # Angle between thigh and shank
    cosine_angle = np.dot(thigh, shank) / (np.linalg.norm(thigh) * np.linalg.norm(shank))
    # Handle numerical issues
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def calculate_hip_flexion_angle(shoulder, hip, knee):
    """Calculate the hip flexion angle."""
    # Vectors
    trunk = np.array([shoulder.x - hip.x, shoulder.y - hip.y, shoulder.z - hip.z])
    thigh = np.array([knee.x - hip.x, knee.y - hip.y, knee.z - hip.z])
    # Angle between trunk and thigh
    cosine_angle = np.dot(trunk, thigh) / (np.linalg.norm(trunk) * np.linalg.norm(thigh))
    # Handle numerical issues
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def calculate_knee_valgus_angle(hip, knee, ankle):
    """Calculate the knee valgus/varus angle in the frontal plane."""
    # Vectors in frontal plane (X, Z)
    femur = np.array([hip.x - knee.x, hip.z - knee.z])
    tibia = np.array([ankle.x - knee.x, ankle.z - knee.z])
    # Angle between femur and tibia
    cosine_angle = np.dot(femur, tibia) / (np.linalg.norm(femur) * np.linalg.norm(tibia))
    # Handle numerical issues
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    # Convert to degrees
    angle_deg = np.degrees(angle)
    return angle_deg

def calculate_hip_adduction_angle(hip, knee):
    """Calculate the hip adduction angle."""
    # Vectors in frontal plane
    vertical = np.array([0, 1])  # Positive Y-axis
    thigh = np.array([knee.x - hip.x, knee.y - hip.y])
    # Angle between vertical and thigh
    cosine_angle = np.dot(thigh, vertical) / (np.linalg.norm(thigh) * np.linalg.norm(vertical))
    # Handle numerical issues
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle) - 90  # Adjust relative to vertical

def calculate_pose_angles(landmarks, mp_pose):
    """Calculate the required angles from pose landmarks."""
    # Extract landmarks
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

    angles = {
        # Knee Flexion Angles
        "Left Knee Flexion": calculate_knee_flexion_angle(left_hip, left_knee, left_ankle),
        "Right Knee Flexion": calculate_knee_flexion_angle(right_hip, right_knee, right_ankle),

        # Hip Flexion Angles
        "Left Hip Flexion": calculate_hip_flexion_angle(left_shoulder, left_hip, left_knee),
        "Right Hip Flexion": calculate_hip_flexion_angle(right_shoulder, right_hip, right_knee),

        # Knee Valgus Angles
        "Left Knee Valgus": calculate_knee_valgus_angle(left_hip, left_knee, left_ankle),
        "Right Knee Valgus": calculate_knee_valgus_angle(right_hip, right_knee, right_ankle),

        # Hip Adduction Angles
        "Left Hip Adduction": calculate_hip_adduction_angle(left_hip, left_knee),
        "Right Hip Adduction": calculate_hip_adduction_angle(right_hip, right_knee)
    }

    return angles

def calculate_velocities(angle_series, fps):
    """Calculate angular velocities from angle series."""
    velocities = {}
    dt = 1 / fps
    window_length = 31  # Should be an odd integer
    poly_order = 2
    for joint, angles in angle_series.items():
        if len(angles) >= window_length:
            adjusted_window_length = min(window_length, len(angles))
            if adjusted_window_length % 2 == 0:
                adjusted_window_length -= 1
            velocity = savgol_filter(angles, adjusted_window_length, poly_order, deriv=1, delta=dt)
            velocities[joint] = velocity
    return velocities

def calculate_accelerations(angle_series, fps):
    """Calculate accelerations from angle series."""
    accelerations = {}
    dt = 1 / fps
    window_length = 31  # Should be an odd integer
    poly_order = 2
    for joint, angles in angle_series.items():
        if len(angles) >= window_length:
            adjusted_window_length = min(window_length, len(angles))
            if adjusted_window_length % 2 == 0:
                adjusted_window_length -= 1
            acceleration = savgol_filter(angles, adjusted_window_length, poly_order, deriv=2, delta=dt)
            accelerations[joint] = acceleration
    return accelerations

def calculate_jerks(accelerations, fps):
    """Calculate jerks from acceleration series."""
    jerks = {}
    dt = 1 / fps
    window_length = 31  # Should be an odd integer
    poly_order = 2
    for joint, acceleration in accelerations.items():
        if len(acceleration) >= window_length:
            adjusted_window_length = min(window_length, len(acceleration))
            if adjusted_window_length % 2 == 0:
                adjusted_window_length -= 1
            jerk = savgol_filter(acceleration, adjusted_window_length, poly_order, deriv=1, delta=dt)
            jerks[joint] = jerk
    return jerks
