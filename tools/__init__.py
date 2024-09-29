import numpy as np
from scipy.signal import savgol_filter

def calculate_angle(a, b, c):
    """Calculate the angle between three points in 3D space."""
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def calculate_angle_frontal_plane(a, b, c):
    """Calculate the angle between three points projected onto the frontal plane (X-Y plane)."""
    ba = np.array([a.x - b.x, a.y - b.y])  # Ignore Z (depth)
    bc = np.array([c.x - b.x, c.y - b.y])  # Ignore Z (depth)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip to handle numerical errors
    return np.degrees(angle)

def calculate_angle_with_vertical(a, b):
    """Calculate the angle between the vector a-b and the vertical axis in the frontal plane."""
    ab = np.array([b.x - a.x, b.y - a.y])  # Ignore Z (depth)
    vertical = np.array([0, 1])  # Vertical axis in Y-direction
    cosine_angle = np.dot(ab, vertical) / (np.linalg.norm(ab) * np.linalg.norm(vertical))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip to handle numerical errors
    return np.degrees(angle)

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
        "Left Knee Flexion": 180 - calculate_angle(left_hip, left_knee, left_ankle),
        "Right Knee Flexion": 180 - calculate_angle(right_hip, right_knee, right_ankle),

        # Hip Flexion Angles
        "Left Hip Flexion": 180 - calculate_angle(left_shoulder, left_hip, left_knee),
        "Right Hip Flexion": 180 - calculate_angle(right_shoulder, right_hip, right_knee),

        # Knee Valgus Angles
        "Left Knee Valgus": 180 - calculate_angle_frontal_plane(left_hip, left_knee, left_ankle),
        "Right Knee Valgus": 180 - calculate_angle_frontal_plane(right_hip, right_knee, right_ankle),

        # Hip Adduction Angles
        "Left Hip Adduction": calculate_angle_with_vertical(left_hip, left_knee) - 90,
        "Right Hip Adduction": calculate_angle_with_vertical(right_hip, right_knee) - 90,
    }

    return angles

def calculate_accelerations(angle_series, fps):
    """Calculate accelerations from angle series."""
    accelerations = {}
    for joint, angles in angle_series.items():
        if len(angles) >= 30:
            window_length = min(31, len(angles))  # Ensure window_length is odd and <= len(angles)
            poly_order = 2
            dt = 1 / fps
            velocity = savgol_filter(angles, window_length, poly_order, deriv=1, delta=dt)
            acceleration = savgol_filter(angles, window_length, poly_order, deriv=2, delta=dt)
            accelerations[joint] = acceleration
    return accelerations

def calculate_jerks(accelerations, fps):
    """Calculate jerks from acceleration series."""
    jerks = {}
    for joint, acceleration in accelerations.items():
        if len(acceleration) >= 30:
            window_length = min(31, len(acceleration))  # Ensure window_length is odd and <= len(acceleration)
            poly_order = 2
            dt = 1 / fps
            jerk = savgol_filter(acceleration, window_length, poly_order, deriv=1, delta=dt)
            jerks[joint] = jerk
    return jerks