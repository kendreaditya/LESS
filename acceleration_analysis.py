import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from tools import calculate_pose_angles, calculate_accelerations

mp_pose = mp.solutions.pose

def process_video(video_path):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_with_acceleration.mp4', fourcc, fps, (width, height))

    angle_series = {
        "Left Knee Flexion": [], "Right Knee Flexion": [],
        "Left Hip Flexion": [], "Right Hip Flexion": [],
        "Left Knee Valgus": [], "Right Knee Valgus": [],
        "Left Hip Adduction": [], "Right Hip Adduction": []
    }
    frame_count = 0

    plt.ion()
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

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
                for joint, angle in current_angles.items():
                    angle_series[joint].append(angle)

                # Calculate accelerations if we have enough frames
                if frame_count >= 30:  # Adjust this value based on your needs
                    accelerations = calculate_accelerations(angle_series, fps)

                    # Display angles and accelerations on the frame
                    y = 30
                    for joint in angle_series.keys():
                        angle = current_angles[joint]
                        accel = accelerations[joint][-1] if joint in accelerations else 0
                        text = f"{joint}: Angle={angle:.2f}, Accel={accel:.2f}"
                        cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        y += 20

                    # Update plots
                    axs[0].cla()
                    axs[1].cla()
                    for joint in angle_series.keys():
                        axs[0].plot(angle_series[joint], label=f"{joint} Angle")
                        if joint in accelerations:
                            axs[1].plot(accelerations[joint], label=f"{joint} Acceleration")
                    axs[0].legend(loc='upper right')
                    axs[1].legend(loc='upper right')
                    axs[0].set_title('Joint Angles')
                    axs[1].set_title('Joint Accelerations')
                    plt.pause(0.001)

            out.write(image)
            cv2.imshow('Pose Estimation with Accelerations', image)
            frame_count += 1

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    plt.ioff()

    # Save the final plots as images
    fig.savefig('angles_and_accelerations.png')
    plt.show()

if __name__ == "__main__":
    process_video('pose.mov')  # Replace with your video file path