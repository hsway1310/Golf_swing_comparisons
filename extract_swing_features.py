import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
import os
import argparse
import csv

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Swing event names
EVENT_NAMES = {
    0: "Address",
    1: "Toe-up",
    2: "Mid-backswing (arm parallel)",
    3: "Top",
    4: "Mid-downswing (arm parallel)",
    5: "Impact",
    6: "Mid-follow-through (shaft parallel)",
    7: "Finish",
}


# Helper function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)


# Function to extract swing features from frames
def extract_swing_features(frame_path_folder, frame_timestamps, fps):

    frame_paths = [
        jpg for jpg in os.listdir(frame_path_folder) if jpg.split(".")[-1] == "jpg"
    ]

    features = {}
    event_times = {}

    for i, frame_path in enumerate(frame_paths):
        print(f"Calculating swing features for: {frame_path}")
        event_name = EVENT_NAMES[i]

        # Read frame
        image = cv2.imread(f"{frame_path_folder}/{frame_path}")
        if image is None:
            print(f"Error loading frame: {frame_path}")
            continue

        # Convert to RGB for Mediapipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            print(f"No pose detected for {event_name}")
            continue

        landmarks = results.pose_landmarks.landmark

        # Get key landmark coordinates
        left_shoulder = (
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
        )
        right_shoulder = (
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
        )
        left_elbow = (
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y,
        )
        right_elbow = (
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
        )
        left_wrist = (
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y,
        )
        right_wrist = (
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y,
        )
        left_knuckle = (
            landmarks[mp_pose.PoseLandmark.LEFT_INDEX].x,
            landmarks[mp_pose.PoseLandmark.LEFT_INDEX].y,
        )  # Knuckle
        right_knuckle = (
            landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].y,
        )  # Knuckle
        left_hip = (
            landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP].y,
        )
        right_hip = (
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y,
        )

        # Calculate joint angles
        sholder_tilt = calculate_angle(
            left_shoulder, right_shoulder, right_hip
        )  # Torso tilt
        left_elbow_angle = calculate_angle(
            left_shoulder, left_elbow, left_wrist
        )  # Lead arm angle
        right_elbow_angle = calculate_angle(
            right_shoulder, right_elbow, right_wrist
        )  # Trail arm angle
        left_wrist_angle = calculate_angle(
            left_elbow, left_wrist, left_knuckle
        )  # left wrist hinge
        right_wrist_angle = calculate_angle(
            right_elbow, right_wrist, right_knuckle
        )  # right wrist hinge
        hip_rotation = calculate_angle(left_hip, right_hip, right_shoulder)  # Hip turn

        # Convert frame timestamp to seconds
        event_times[event_name] = (
            frame_timestamps[i] / fps
        )  # Convert frame number to seconds

        # Store features
        features[event_name] = {
            "sholder_tilt": sholder_tilt,
            "left_elbow_angle": left_elbow_angle,
            "right_elbow_angle": right_elbow_angle,
            "left_wrist_angle": left_wrist_angle,
            "right_wrist_angle": right_wrist_angle,
            "hip_rotation": hip_rotation,
            "frame_number": frame_timestamps[i],
            "time_seconds": event_times[event_name],
        }

    # Calculate Tempo Ratio
    if (
        "Address" in event_times
        and "Top" in event_times
        and "Impact" in event_times
    ):
        backswing_time = event_times["Top"] - event_times["Address"]
        downswing_time = event_times["Impact"] - event_times["Top"]

        if downswing_time > 0:
            tempo = round(backswing_time / downswing_time, 2)  # Tempo Ratio
        else:
            tempo = None  # Avoid division by zero

        features["Tempo"] = {
            "backswing_time": round(backswing_time, 2),
            "downswing_time": round(downswing_time, 2),
            "tempo": tempo,
        }

    return features


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        help="Path to video that you want to test",
        default="test_video.mp4",
        required=True,
    )
    # Replace with paths to extracted swing frames
    args = parser.parse_args()
    swing_folder = args.path
    frame_path_folder = f"{swing_folder}/frames"

    frame_timestamps_path = f"{frame_path_folder}/event_frames.csv"
    with open(frame_timestamps_path, "r") as file:
        reader = csv.reader(file)
        frame_timestamps = [int(row[0]) for row in reader]
        print(frame_timestamps)


    swing_features = extract_swing_features(
        frame_path_folder=frame_path_folder, frame_timestamps=frame_timestamps, fps=60
    )

    # Save features to JSON file
    with open(f"{frame_path_folder}/swing_features.json", "w") as f:
        json.dump(swing_features, f, indent=4)

    print("Swing features extracted and saved to swing_features.json")
