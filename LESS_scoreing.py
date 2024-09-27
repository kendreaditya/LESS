"""
Evalute pose with angles distribution from LESS for each frame of a video.
"""

import cv2
import mediapipe as mp
import numpy as np
from LESS import angledist
from tools import calculate_pose_angles

def process_video(video_path):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(video_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./outputs/LESS_scoring.mp4', fourcc, fps, (width, height))

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

                current_angles = calculate_pose_angles(results.pose_landmarks.landmark, mp_pose)

                angles_for_scoring = {
                    "Initial Contact": current_angles,
                    "Peak Angle": current_angles,
                    "Displacement": current_angles,
                }

                scores = angledist.analyze_all_angles(angles_for_scoring)

                y = 30
                for phase, measurements in scores.items():
                    cv2.putText(image, f"{phase}:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    y += 20
                    for measurement, result in measurements.items():
                        text = f"{measurement}: {result['category']} (Z: {result['zscore']:.2f}, Angle: {current_angles[measurement]:.2f})"
                        cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        y += 20

            out.write(image)
            cv2.imshow('Pose Estimation', image)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Global variable for MediaPipe Pose
mp_pose = mp.solutions.pose

process_video('./outputs/pose.mov')