import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from tools import calculate_pose_angles, calculate_accelerations, calculate_jerks
from biomechanical import score_accelerations

mp_pose = mp.solutions.pose

def process_video(video_path):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        raise ValueError("Unable to determine FPS of the video, this tool requires a valid FPS value to calculate accelerations and jerks.")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./outputs/output_with_acceleration.mp4', fourcc, fps, (width, height))

    angle_series = {
        "Left Knee Flexion": [], "Right Knee Flexion": [],
        "Left Hip Flexion": [], "Right Hip Flexion": [],
        "Left Knee Valgus": [], "Right Knee Valgus": [],
        "Left Hip Adduction": [], "Right Hip Adduction": []
    }
    frame_count = 0

    plt.ion()
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

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
                
                # Calculate accelerations and jerks if we have enough frames
                if frame_count >= 30:  # Adjust this value based on your needs
                    accelerations = calculate_accelerations(angle_series, fps)
                    acceleration_scores = score_accelerations(accelerations)
                    jerks = calculate_jerks(accelerations, fps)

                    # Display angles, accelerations, and jerks on the frame
                    y = 30

                    for joint in angle_series.keys():
                        angle = current_angles[joint]
                        accel = accelerations[joint][-1] if joint in accelerations else 0
                        jerk = jerks[joint][-1] if joint in jerks else 0

                        angle_text = f"{joint}: Angle={angle:.2f}"
                        angle_color = (0, 0, 0)
                        cv2.putText(image, angle_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, angle_color, 1)
                        y += 20

                        accel_text = f"{joint}: Accl={accel:.2f}, Score={acceleration_scores[joint]['risk_score']:.2f}, Category={acceleration_scores[joint]['risk_category']}"
                        risk_score = acceleration_scores[joint]['risk_score']
                        green_component = int(225 - (risk_score * 2.25))
                        red_component = int(risk_score * 2.55)
                        accel_color = (0, green_component, red_component)
                        cv2.putText(image, accel_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, accel_color, 1)
                        y += 20

                    # Update plots
                    axs[0].cla()
                    axs[1].cla()
                    axs[2].cla()
                    for joint in angle_series.keys():
                        axs[0].plot(angle_series[joint], label=f"{joint} Angle")
                        if joint in accelerations:
                            axs[1].plot(accelerations[joint], label=f"{joint} Acceleration")
                        if joint in jerks:
                            axs[2].plot(jerks[joint], label=f"{joint} Jerk")
                    axs[0].legend(loc='upper right')
                    axs[1].legend(loc='upper right')
                    axs[2].legend(loc='upper right')
                    axs[0].set_title('Joint Angles')
                    axs[1].set_title('Joint Accelerations')
                    axs[2].set_title('Joint Jerks')
                    plt.pause(0.001)

            out.write(image)
            cv2.imshow('Pose Estimation with Accelerations and Jerks', image)
            frame_count += 1

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    plt.ioff()

    # Save the final plots as images
    fig.savefig('./outputs/angles_accelerations_jerks.png')
    plt.show()

if __name__ == "__main__":
    process_video('./outputs/pose.mov')  # Replace with your video file path