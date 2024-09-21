import numpy as np

from LESS import (
    knee_flexion_angle_at_initial_contact,
    hip_flexion_angle_at_initial_contact,
    trunk_flexion_angle_at_initial_contact,
    ankle_plantar_flexion_angle_at_initial_contact,
    knee_valgus_angle_at_initial_contact,
    lateral_trunk_flexion_angle_at_initial_contact,
    stance_width_wide,
    stance_width_narrow,
    foot_position_toe_in,
    foot_position_toe_out,
    symmetric_initial_foot_contact,
    knee_flexion_displacement,
    hip_flexion_at_max_knee_flexion,
    trunk_flexion_at_max_knee_flexion,
    knee_valgus_displacement,
    joint_displacement,
    overall_impression,
    calculate_less_score,
    interpret_less_score,
)


# Define the 3D coordinates of the joints
joints = {
    'left_ankle': np.array([0.5, 0.2, 0.1]),  # x, y, z coordinates
    'left_knee': np.array([0.4, 0.3, 0.2]),
    'left_hip': np.array([0.3, 0.4, 0.3]),
    'right_ankle': np.array([0.6, 0.2, 0.1]),
    'right_knee': np.array([0.7, 0.3, 0.2]),
    'right_hip': np.array([0.8, 0.4, 0.3]),
    'left_shoulder': np.array([0.2, 0.5, 0.4]),
    'right_shoulder': np.array([0.9, 0.5, 0.4]),
    'left_trunk': np.array([0.1, 0.6, 0.5]),
    'right_trunk': np.array([1.0, 0.6, 0.5]),
}

# Define the pose estimation data
pose_estimation_data = {
    'joints': joints,
    'stance_width': 0.8,  # width of the stance
    'foot_rotation': 15,  # rotation of the foot in degrees
    'symmetric_foot_contact': True,  # whether the foot contact is symmetric
}

# Define a function to calculate the LESS scores from the pose estimation data
def calculate_less_scores(pose_estimation_data):
    # Calculate the knee flexion angle
    knee_flexion_angle = np.arccos(np.dot(pose_estimation_data['joints']['left_knee'] - pose_estimation_data['joints']['left_hip'], pose_estimation_data['joints']['left_ankle'] - pose_estimation_data['joints']['left_knee']) / (np.linalg.norm(pose_estimation_data['joints']['left_knee'] - pose_estimation_data['joints']['left_hip']) * np.linalg.norm(pose_estimation_data['joints']['left_ankle'] - pose_estimation_data['joints']['left_knee'])))

    # Calculate the hip flexion angle
    hip_flexion_angle = np.arccos(np.dot(pose_estimation_data['joints']['left_hip'] - pose_estimation_data['joints']['left_trunk'], pose_estimation_data['joints']['left_knee'] - pose_estimation_data['joints']['left_hip']) / (np.linalg.norm(pose_estimation_data['joints']['left_hip'] - pose_estimation_data['joints']['left_trunk']) * np.linalg.norm(pose_estimation_data['joints']['left_knee'] - pose_estimation_data['joints']['left_hip'])))

    # Calculate the trunk flexion angle
    trunk_flexion_angle = np.arccos(np.dot(pose_estimation_data['joints']['left_trunk'] - pose_estimation_data['joints']['left_shoulder'], pose_estimation_data['joints']['left_hip'] - pose_estimation_data['joints']['left_trunk']) / (np.linalg.norm(pose_estimation_data['joints']['left_trunk'] - pose_estimation_data['joints']['left_shoulder']) * np.linalg.norm(pose_estimation_data['joints']['left_hip'] - pose_estimation_data['joints']['left_trunk'])))

    # Calculate the ankle plantarflexion angle
    ankle_plantarflexion_angle = np.arccos(np.dot(pose_estimation_data['joints']['left_ankle'] - pose_estimation_data['joints']['left_knee'], np.array([0, 0, 1])) / np.linalg.norm(pose_estimation_data['joints']['left_ankle'] - pose_estimation_data['joints']['left_knee']))

    # Calculate the knee valgus angle
    knee_valgus_angle = np.arccos(np.dot(pose_estimation_data['joints']['left_knee'] - pose_estimation_data['joints']['left_hip'], pose_estimation_data['joints']['right_knee'] - pose_estimation_data['joints']['right_hip']) / (np.linalg.norm(pose_estimation_data['joints']['left_knee'] - pose_estimation_data['joints']['left_hip']) * np.linalg.norm(pose_estimation_data['joints']['right_knee'] - pose_estimation_data['joints']['right_hip'])))

    # Calculate the LESS scores
    scores = [
        knee_flexion_angle_at_initial_contact(knee_flexion_angle),
        hip_flexion_angle_at_initial_contact(hip_flexion_angle),
        trunk_flexion_angle_at_initial_contact(trunk_flexion_angle),
        ankle_plantar_flexion_angle_at_initial_contact(ankle_plantarflexion_angle),
        knee_valgus_angle_at_initial_contact(knee_valgus_angle),
        lateral_trunk_flexion_angle_at_initial_contact(False),  # assume no lateral trunk flexion
        stance_width_wide(pose_estimation_data['stance_width'], 0.5),  # assume average shoulder width
        stance_width_narrow(pose_estimation_data['stance_width'], 0.5),  # assume average shoulder width
        foot_position_toe_in(pose_estimation_data['foot_rotation']),
        foot_position_toe_out(pose_estimation_data['foot_rotation']),
        symmetric_initial_foot_contact(pose_estimation_data['symmetric_foot_contact']),
        knee_flexion_displacement(knee_flexion_angle, knee_flexion_angle + 0.1),  # assume small displacement
        hip_flexion_at_max_knee_flexion(hip_flexion_angle, hip_flexion_angle - 0.1),  # assume small displacement
        trunk_flexion_at_max_knee_flexion(trunk_flexion_angle, trunk_flexion_angle + 0.1),  # assume small displacement
        knee_valgus_displacement(knee_valgus_angle),
        joint_displacement('average'),  # assume average joint displacement
        overall_impression('good'),  # assume good overall impression
    ]

    return scores

# Calculate the LESS scores
scores = calculate_less_scores(pose_estimation_data)

# Print the LESS scores
print("LESS Scores:")
for i, score in enumerate(scores):
    print(f"Item {i+1}: {score}")

# Calculate the total LESS score
total_score = calculate_less_score(scores)

# Print the total LESS score
print(f"Total LESS Score: {total_score}")

# Interpret the total LESS score
interpretation = interpret_less_score(total_score)

# Print the interpretation
print(f"Interpretation: {interpretation}")