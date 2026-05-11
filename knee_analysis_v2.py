# Import necessary libraries
import numpy as np

# Existing imports
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse

# Import existing functions (assuming they are in the same module or adjust the import paths accordingly)
from tools import calculate_pose_angles, calculate_velocities, calculate_accelerations, calculate_jerks
from biomechanical import score_accelerations

def get_local_knee_coordinate_system_coords(hip, knee, ankle):
    """Compute the local coordinate system at the knee joint using physical coordinates."""
    # Create vectors
    thigh = hip - knee
    shank = ankle - knee
    # Normalize vectors
    thigh /= np.linalg.norm(thigh)
    shank /= np.linalg.norm(shank)
    # Knee flexion axis (approximate)
    flexion_axis = np.cross(thigh, shank)
    flexion_axis /= np.linalg.norm(flexion_axis)
    # Create coordinate system
    y_axis = thigh
    z_axis = flexion_axis
    x_axis = np.cross(y_axis, z_axis)
    return x_axis, y_axis, z_axis

def get_ligament_attachment_points_coords(knee, x_axis, y_axis, z_axis):
    """Compute the approximate positions of ligament attachment points using physical coordinates."""
    # Offsets in millimeters
    # Adjust these values based on anatomical data
    # ACL
    acl_proximal_offset = np.array([5, 0, 5])  # mm
    acl_distal_offset = np.array([-5, 0, -5])  # mm
    # PCL
    pcl_proximal_offset = np.array([-5, 0, 5])  # mm
    pcl_distal_offset = np.array([5, 0, -5])    # mm
    # MCL (medial side)
    mcl_proximal_offset = np.array([0, 5, 0])   # mm
    mcl_distal_offset = np.array([0, 5, 0])     # mm

    # Compute attachment points
    acl_proximal = knee + acl_proximal_offset[0] * x_axis + acl_proximal_offset[1] * y_axis + acl_proximal_offset[2] * z_axis
    acl_distal = knee + acl_distal_offset[0] * x_axis + acl_distal_offset[1] * y_axis + acl_distal_offset[2] * z_axis

    pcl_proximal = knee + pcl_proximal_offset[0] * x_axis + pcl_proximal_offset[1] * y_axis + pcl_proximal_offset[2] * z_axis
    pcl_distal = knee + pcl_distal_offset[0] * x_axis + pcl_distal_offset[1] * y_axis + pcl_distal_offset[2] * z_axis

    mcl_proximal = knee + mcl_proximal_offset[0] * x_axis + mcl_proximal_offset[1] * y_axis + mcl_proximal_offset[2] * z_axis
    mcl_distal = knee + mcl_distal_offset[0] * x_axis + mcl_distal_offset[1] * y_axis + mcl_distal_offset[2] * z_axis

    attachment_points = {
        'ACL': {'proximal': acl_proximal, 'distal': acl_distal},
        'PCL': {'proximal': pcl_proximal, 'distal': pcl_distal},
        'MCL': {'proximal': mcl_proximal, 'distal': mcl_distal}
    }
    return attachment_points

def calculate_ligament_lengths_coords(attachment_points):
    """Calculate the lengths of the ligaments using physical coordinates."""
    ligament_lengths = {}
    for ligament, points in attachment_points.items():
        proximal = points['proximal']
        distal = points['distal']
        length = np.linalg.norm(proximal - distal)  # mm
        ligament_lengths[ligament] = length
    return ligament_lengths


# Define ligament properties and functions
# Ligament stiffness values (N/mm)
ligament_properties = {
    'ACL': {'stiffness': 242},  # Approximate stiffness in N/mm
    'PCL': {'stiffness': 331},
    'MCL': {'stiffness': 354}
}

def get_local_knee_coordinate_system(hip, knee, ankle):
    """Compute the local coordinate system at the knee joint."""
    # Create vectors
    thigh = np.array([hip.x - knee.x, hip.y - knee.y, hip.z - knee.z])
    shank = np.array([ankle.x - knee.x, ankle.y - knee.y, ankle.z - knee.z])
    # Normalize vectors
    thigh /= np.linalg.norm(thigh)
    shank /= np.linalg.norm(shank)
    # Knee flexion axis (approximate)
    flexion_axis = np.cross(thigh, shank)
    flexion_axis /= np.linalg.norm(flexion_axis)
    # Create coordinate system
    y_axis = thigh
    z_axis = flexion_axis
    x_axis = np.cross(y_axis, z_axis)
    return x_axis, y_axis, z_axis

def get_ligament_attachment_points(knee, x_axis, y_axis, z_axis):
    """Compute the approximate positions of ligament attachment points."""
    # Offsets in millimeters (simplified and scaled)
    # These values are approximations and may need refinement
    # Convert to the same units as the landmark positions (usually normalized to image size)
    # Adjust scaling as necessary
    scale_factor = 0.1  # Adjust based on actual units and scaling
    # ACL
    acl_proximal_offset = scale_factor * np.array([5, 0, 5])  # Anterior on tibia
    acl_distal_offset = scale_factor * np.array([-5, 0, -5])  # Posterior on femur
    # PCL
    pcl_proximal_offset = scale_factor * np.array([-5, 0, 5])  # Posterior on tibia
    pcl_distal_offset = scale_factor * np.array([5, 0, -5])    # Anterior on femur
    # MCL (medial side)
    mcl_proximal_offset = scale_factor * np.array([0, 5, 0])   # Medial on tibia
    mcl_distal_offset = scale_factor * np.array([0, 5, 0])     # Medial on femur

    # Compute attachment points
    acl_proximal = np.array([knee.x, knee.y, knee.z]) + acl_proximal_offset[0] * x_axis + acl_proximal_offset[1] * y_axis + acl_proximal_offset[2] * z_axis
    acl_distal = np.array([knee.x, knee.y, knee.z]) + acl_distal_offset[0] * x_axis + acl_distal_offset[1] * y_axis + acl_distal_offset[2] * z_axis

    pcl_proximal = np.array([knee.x, knee.y, knee.z]) + pcl_proximal_offset[0] * x_axis + pcl_proximal_offset[1] * y_axis + pcl_proximal_offset[2] * z_axis
    pcl_distal = np.array([knee.x, knee.y, knee.z]) + pcl_distal_offset[0] * x_axis + pcl_distal_offset[1] * y_axis + pcl_distal_offset[2] * z_axis

    mcl_proximal = np.array([knee.x, knee.y, knee.z]) + mcl_proximal_offset[0] * x_axis + mcl_proximal_offset[1] * y_axis + mcl_proximal_offset[2] * z_axis
    mcl_distal = np.array([knee.x, knee.y, knee.z]) + mcl_distal_offset[0] * x_axis + mcl_distal_offset[1] * y_axis + mcl_distal_offset[2] * z_axis

    attachment_points = {
        'ACL': {'proximal': acl_proximal, 'distal': acl_distal},
        'PCL': {'proximal': pcl_proximal, 'distal': pcl_distal},
        'MCL': {'proximal': mcl_proximal, 'distal': mcl_distal}
    }
    return attachment_points

def calculate_ligament_lengths(attachment_points):
    """Calculate the lengths of the ligaments."""
    ligament_lengths = {}
    for ligament, points in attachment_points.items():
        proximal = points['proximal']
        distal = points['distal']
        length = np.linalg.norm(proximal - distal)
        ligament_lengths[ligament] = length
    return ligament_lengths

def calculate_ligament_forces(ligament_lengths, reference_lengths):
    """Calculate the forces in the ligaments using a linear elastic model."""
    ligament_forces = {}
    for ligament, length in ligament_lengths.items():
        delta_L = length - reference_lengths[ligament]
        stiffness = ligament_properties[ligament]['stiffness']  # N/mm
        force = stiffness * delta_L * 1000  # Convert from meters to millimeters
        force = max(force, 0)  # Ligaments can only resist tension, not compression
        ligament_forces[ligament] = force
    return ligament_forces

# Update the process_video function to include ligament force calculations
# Update process_video function
def process_video(video_path, show_windows=True):
    def calculate_scale_factor(landmarks, mp_pose):
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        # Pixel coordinates
        left_hip_coords = np.array([left_hip.x * image_width, left_hip.y * image_height])
        right_hip_coords = np.array([right_hip.x * image_width, right_hip.y * image_height])
        pixel_distance = np.linalg.norm(left_hip_coords - right_hip_coords)
        # Average hip width in millimeters
        real_world_distance = 300  # mm
        scale_factor = real_world_distance / pixel_distance  # mm per pixel
        return scale_factor

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        raise ValueError("Unable to determine FPS of the video.")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    image_width = width
    image_height = height

    # Generate output file path
    base_name = os.path.basename(video_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(os.path.dirname(video_path), f"{name}_output_with_ligament_forces{ext}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    angle_series = {
        "Left Knee Flexion": [], "Right Knee Flexion": [],
        "Left Hip Flexion": [], "Right Hip Flexion": [],
        "Left Knee Valgus": [], "Right Knee Valgus": [],
        "Left Hip Adduction": [], "Right Hip Adduction": []
    }
    ligament_force_series = {
        'ACL': [], 'PCL': [], 'MCL': []
    }
    frame_count = 0

    if show_windows:
        plt.ion()
        fig, axs = plt.subplots(5, 1, figsize=(10, 20))

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for _ in tqdm(range(total_frames), desc="Processing video"):
            ret, frame = cap.read()
            if not ret:
                break

            image = frame.copy()
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = pose.process(image_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )

                landmarks = results.pose_landmarks.landmark

                # Estimate scale factor
                scale_factor = calculate_scale_factor(landmarks, mp_pose)

                # Convert landmarks to pixel coordinates and scale to millimeters
                def get_landmark_coords(landmark):
                    x = landmark.x * image_width * scale_factor  # mm
                    y = landmark.y * image_height * scale_factor  # mm
                    z = landmark.z * scale_factor  # Assuming z is normalized similar to x and y
                    return np.array([x, y, z])

                # Calculate joint angles
                current_angles = calculate_pose_angles(landmarks, mp_pose)
                for joint, angle in current_angles.items():
                    angle_series[joint].append(angle)

                # Calculate ligament forces
                # For simplicity, we'll consider only the left leg in this example
                left_hip = get_landmark_coords(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
                left_knee = get_landmark_coords(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
                left_ankle = get_landmark_coords(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

                # Compute local coordinate system at the knee
                x_axis, y_axis, z_axis = get_local_knee_coordinate_system_coords(left_hip, left_knee, left_ankle)

                # Compute ligament attachment points
                attachment_points = get_ligament_attachment_points_coords(left_knee, x_axis, y_axis, z_axis)

                # Calculate ligament lengths
                ligament_lengths = calculate_ligament_lengths_coords(attachment_points)

                # Initialize reference lengths at the first frame
                if frame_count == 0:
                    reference_lengths = ligament_lengths.copy()

                # Calculate ligament forces
                ligament_forces = calculate_ligament_forces(ligament_lengths, reference_lengths)
                for ligament, force in ligament_forces.items():
                    ligament_force_series[ligament].append(force)

                # Display angles and ligament forces on the frame
                y_pos = 30
                for joint in angle_series.keys():
                    angle = current_angles[joint]
                    angle_text = f"{joint}: Angle={angle:.2f}"
                    angle_color = (0, 0, 0)
                    cv2.putText(image, angle_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, angle_color, 1)
                    y_pos += 20

                for ligament, force in ligament_forces.items():
                    force_text = f"{ligament} Force: {force:.2f} N"
                    force_color = (0, 0, 255) if force > 0 else (0, 255, 0)
                    cv2.putText(image, force_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, force_color, 1)
                    y_pos += 20

                if show_windows:
                    # Update plots
                    axs[0].cla()
                    axs[1].cla()
                    # Plot joint angles
                    for joint in angle_series.keys():
                        axs[0].plot(angle_series[joint], label=f"{joint} Angle")
                    axs[0].legend(loc='upper right')
                    axs[0].set_title('Joint Angles')
                    # Plot ligament forces
                    for ligament in ligament_force_series.keys():
                        axs[1].plot(ligament_force_series[ligament], label=f"{ligament} Force")
                    axs[1].legend(loc='upper right')
                    axs[1].set_title('Ligament Forces')
                    plt.pause(0.001)

            out.write(image)
            if show_windows:
                cv2.imshow('Pose Estimation with Ligament Forces', image)
            frame_count += 1

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    if show_windows:
        plt.ioff()

        # Save the final plots as images
        if not os.path.exists('./outputs'):
            os.makedirs('./outputs')
        fig.savefig('./outputs/angles_and_ligament_forces.png')
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video for pose estimation with ligament force estimation.")
    parser.add_argument('video_path', type=str, help="Path to the input video file.")
    parser.add_argument('--show-windows', action='store_true', help="Flag to show the windows for plt and cv2.")
    
    args = parser.parse_args()
    
    process_video(args.video_path, show_windows=args.show_windows)