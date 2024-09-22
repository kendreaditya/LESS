import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from LESS import angles

def calculate_angle(a, b, c):
    """Calculate the angle between three points in 3D space."""
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def calculate_pose_angles(landmarks):
    """Calculate the required angles from pose landmarks."""
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    left_angles = {
        "Knee Flexion": 180 - calculate_angle(left_hip, left_knee, left_ankle),
        "Hip Flexion": calculate_angle(left_shoulder, left_hip, left_knee) - 90,
        "Knee Valgus": calculate_angle(left_hip, left_knee, left_ankle) - calculate_angle(right_hip, right_knee, right_ankle),
        "Hip Adduction": calculate_angle(left_shoulder, left_hip, left_knee) - calculate_angle(right_shoulder, right_hip, right_knee),
        "Tibial Rotation": calculate_angle(left_hip, left_knee, left_ankle) - 90,
        "Hip Rotation": calculate_angle(left_shoulder, left_hip, left_knee) - calculate_angle(right_shoulder, right_hip, right_knee)
    }
    right_angles = {
        "Knee Flexion": 180 - calculate_angle(right_hip, right_knee, right_ankle),
        "Hip Flexion": calculate_angle(right_shoulder, right_hip, right_knee) - 90,
        "Knee Valgus": calculate_angle(right_hip, right_knee, right_ankle) - calculate_angle(left_hip, left_knee, left_ankle),
        "Hip Adduction": calculate_angle(right_shoulder, right_hip, right_knee) - calculate_angle(left_shoulder, left_hip, left_knee),
        "Tibial Rotation": calculate_angle(right_hip, right_knee, right_ankle) - 90,
        "Hip Rotation": calculate_angle(right_shoulder, right_hip, right_knee) - calculate_angle(left_shoulder, left_hip, left_knee)
    }
    angles = {
        "left": left_angles,
        "right": right_angles
    }
    return angles

def plot_3d_landmarks(landmarks, ax):
    """Plot the 3D landmarks in X, Y, Z coordinates."""
    ax.cla()  # Clear the plot
    ax.set_title('3D Pose Landmarks')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    xs = [landmark.x for landmark in landmarks]
    ys = [landmark.y for landmark in landmarks]
    zs = [landmark.z for landmark in landmarks]

    ax.scatter(xs, ys, zs, c='r', marker='o')
    ax.plot(xs, ys, zs, c='b')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    plt.pause(0.001)

def process_video(video_path):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(video_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_with_3d_view.mp4', fourcc, fps, (width, height))

    # Create a figure for the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    frame_count = 0
    initial_angles = None
    peak_angles = {k: -float('inf') for k in ["Knee Flexion", "Knee Valgus", "Tibial Rotation", "Hip Flexion", "Hip Adduction", "Hip Rotation"]}

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )

                current_angles = calculate_pose_angles(results.pose_landmarks.landmark)

                # Plot the 3D landmarks
                plot_3d_landmarks(results.pose_landmarks.landmark, ax)

                # Display scores and angles as in the previous code

            out.write(image)
            cv2.imshow('Pose Estimation with 3D View', image)
            frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Global variable for MediaPipe Pose
mp_pose = mp.solutions.pose

process_video('pose.mov')
