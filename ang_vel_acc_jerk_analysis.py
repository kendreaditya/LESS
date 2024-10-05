import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tools import calculate_pose_angles, calculate_velocities, calculate_accelerations, calculate_jerks
from biomechanical import score_accelerations
import os
import argparse

mp_pose = mp.solutions.pose

def process_video(video_path, show_windows=True):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        raise ValueError("Unable to determine FPS of the video, this tool requires a valid FPS value to calculate accelerations and jerks.")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Generate output file path
    base_name = os.path.basename(video_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(os.path.dirname(video_path), f"{name}_output_with_ang_vel_acc_jerk{ext}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    angle_series = {
        "Left Knee Flexion": [], "Right Knee Flexion": [],
        "Left Hip Flexion": [], "Right Hip Flexion": [],
        "Left Knee Valgus": [], "Right Knee Valgus": [],
        "Left Hip Adduction": [], "Right Hip Adduction": []
    }
    frame_count = 0

    if show_windows:
        plt.ion()
        fig, axs = plt.subplots(4, 1, figsize=(10, 16))

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for _ in tqdm(range(total_frames), desc="Processing video"):
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
                
                # Calculate velocities, accelerations, and jerks if we have enough frames
                window_length = 31  # Should be an odd integer
                min_frames = window_length
                if frame_count >= min_frames:
                    velocities = calculate_velocities(angle_series, fps)
                    accelerations = calculate_accelerations(angle_series, fps)
                    acceleration_scores = score_accelerations(accelerations)
                    jerks = calculate_jerks(accelerations, fps)

                    # Display angles, velocities, accelerations, and jerks on the frame
                    y = 30

                    for joint in angle_series.keys():
                        angle = current_angles[joint]
                        velocity = velocities[joint][-1] if joint in velocities else 0
                        accel = accelerations[joint][-1] if joint in accelerations else 0
                        jerk = jerks[joint][-1] if joint in jerks else 0
                        if joint in acceleration_scores:
                            accel_score = acceleration_scores[joint]['risk_score']
                            accel_category = acceleration_scores[joint]['risk_category']
                        else:
                            accel_score = 0
                            accel_category = 'Unknown'

                        angle_text = f"{joint}: Angle={angle:.2f}"
                        angle_color = (0, 0, 0)
                        cv2.putText(image, angle_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, angle_color, 1)
                        y += 20

                        velocity_text = f"{joint}: Vel={velocity:.2f}"
                        velocity_color = (0, 0, 0)
                        cv2.putText(image, velocity_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, velocity_color, 1)
                        y += 20

                        accel_text = f"{joint}: Accl={accel:.2f}, Score={accel_score:.2f}, Category={accel_category}"
                        # Determine color based on risk score
                        risk_score = accel_score
                        green_component = int(225 - (risk_score * 2.25))
                        green_component = min(max(green_component, 0), 255)
                        red_component = int(risk_score * 2.55)
                        red_component = min(max(red_component, 0), 255)
                        accel_color = (0, green_component, red_component)
                        cv2.putText(image, accel_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, accel_color, 1)
                        y += 20

                    if show_windows:
                        # Update plots
                        axs[0].cla()
                        axs[1].cla()
                        axs[2].cla()
                        axs[3].cla()
                        for joint in angle_series.keys():
                            axs[0].plot(angle_series[joint], label=f"{joint} Angle")
                            if joint in velocities:
                                axs[1].plot(velocities[joint], label=f"{joint} Velocity")
                            if joint in accelerations:
                                axs[2].plot(accelerations[joint], label=f"{joint} Acceleration")
                            if joint in jerks:
                                axs[3].plot(jerks[joint], label=f"{joint} Jerk")
                        axs[0].legend(loc='upper right')
                        axs[1].legend(loc='upper right')
                        axs[2].legend(loc='upper right')
                        axs[3].legend(loc='upper right')
                        axs[0].set_title('Joint Angles')
                        axs[1].set_title('Joint Velocities')
                        axs[2].set_title('Joint Accelerations')
                        axs[3].set_title('Joint Jerks')
                        plt.pause(0.001)

            out.write(image)
            if show_windows:
                cv2.imshow('Pose Estimation with Velocities, Accelerations, and Jerks', image)
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
        fig.savefig('./outputs/angles_velocities_accelerations_jerks.png')
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video for pose estimation with velocities, accelerations, and jerks.")
    parser.add_argument('video_path', type=str, help="Path to the input video file.")
    parser.add_argument('--show-windows', action='store_true', help="Flag to show the windows for plt and cv2.")
    
    args = parser.parse_args()
    
    process_video(args.video_path, show_windows=args.show_windows)