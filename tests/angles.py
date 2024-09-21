import numpy as np
from LESS.angles import analyze_all_angles

# Define 3D coordinates for pose estimation (example values)
pose_coordinates = {
    "Initial Contact": {
        "Knee": np.array([0.5, 0.2, 0.1]),  # x, y, z coordinates
        "Ankle": np.array([0.3, 0.1, 0.0]),
        "Hip": np.array([0.7, 0.4, 0.2]),
        "Knee_medial": np.array([0.4, 0.2, 0.1]),
        "Knee_lateral": np.array([0.6, 0.2, 0.1]),
        "Ankle_medial": np.array([0.2, 0.1, 0.0]),
        "Ankle_lateral": np.array([0.4, 0.1, 0.0]),
        "Hip_medial": np.array([0.6, 0.4, 0.2]),
        "Hip_lateral": np.array([0.8, 0.4, 0.2])
    },
    "Peak Angle": {
        "Knee": np.array([0.6, 0.3, 0.2]),
        "Ankle": np.array([0.4, 0.2, 0.1]),
        "Hip": np.array([0.8, 0.5, 0.3]),
        "Knee_medial": np.array([0.5, 0.3, 0.2]),
        "Knee_lateral": np.array([0.7, 0.3, 0.2]),
        "Ankle_medial": np.array([0.3, 0.2, 0.1]),
        "Ankle_lateral": np.array([0.5, 0.2, 0.1]),
        "Hip_medial": np.array([0.7, 0.5, 0.3]),
        "Hip_lateral": np.array([0.9, 0.5, 0.3])
    },
    "Displacement": {
        "Knee": np.array([0.7, 0.4, 0.3]),
        "Ankle": np.array([0.5, 0.3, 0.2]),
        "Hip": np.array([0.9, 0.6, 0.4]),
        "Knee_medial": np.array([0.6, 0.4, 0.3]),
        "Knee_lateral": np.array([0.8, 0.4, 0.3]),
        "Ankle_medial": np.array([0.4, 0.3, 0.2]),
        "Ankle_lateral": np.array([0.6, 0.3, 0.2]),
        "Hip_medial": np.array([0.8, 0.6, 0.4]),
        "Hip_lateral": np.array([1.0, 0.6, 0.4])
    }
}

# Define a function to calculate joint angles from 3D coordinates
def calculate_joint_angles(coordinates):
    joint_angles = {}
    for phase, joints in coordinates.items():
        joint_angles[phase] = {}
        for joint, coords in joints.items():
            if joint.endswith("_medial") or joint.endswith("_lateral"):
                continue
            if joint == "Knee":
                knee_medial = joints[joint + "_medial"]
                knee_lateral = joints[joint + "_lateral"]
                joint_angles[phase]["Knee Flexion"] = np.arccos(np.dot(coords - knee_medial, knee_lateral - knee_medial) / (np.linalg.norm(coords - knee_medial) * np.linalg.norm(knee_lateral - knee_medial)))
                joint_angles[phase]["Knee Valgus"] = np.arccos(np.dot(coords - knee_lateral, knee_medial - knee_lateral) / (np.linalg.norm(coords - knee_lateral) * np.linalg.norm(knee_medial - knee_lateral)))
            elif joint == "Ankle":
                ankle_medial = joints[joint + "_medial"]
                ankle_lateral = joints[joint + "_lateral"]
                joint_angles[phase]["Tibial Rotation"] = np.arccos(np.dot(coords - ankle_medial, ankle_lateral - ankle_medial) / (np.linalg.norm(coords - ankle_medial) * np.linalg.norm(ankle_lateral - ankle_medial)))
            elif joint == "Hip":
                hip_medial = joints[joint + "_medial"]
                hip_lateral = joints[joint + "_lateral"]
                joint_angles[phase]["Hip Flexion"] = np.arccos(np.dot(coords - hip_medial, hip_lateral - hip_medial) / (np.linalg.norm(coords - hip_medial) * np.linalg.norm(hip_lateral - hip_medial)))
                joint_angles[phase]["Hip Adduction"] = np.arccos(np.dot(coords - hip_lateral, hip_medial - hip_lateral) / (np.linalg.norm(coords - hip_lateral) * np.linalg.norm(hip_medial - hip_lateral)))
                joint_angles[phase]["Hip Rotation"] = np.arccos(np.dot(coords - hip_medial, hip_lateral - hip_medial) / (np.linalg.norm(coords - hip_medial) * np.linalg.norm(hip_lateral - hip_medial)))
    return joint_angles

# Calculate joint angles from 3D coordinates
joint_angles = calculate_joint_angles(pose_coordinates)

# Score joint angles using the provided module
results = analyze_all_angles(joint_angles)

# Print results
for phase, measurements in results.items():
    print(f"\n{phase}:")
    for measurement, result in measurements.items():
        print(f"  {measurement}:")
        print(f"    Angle: {joint_angles[phase][measurement]:.2f}")
        print(f"    Z-Score: {result['zscore']:.2f}")
        print(f"    Category: {result['category']}")