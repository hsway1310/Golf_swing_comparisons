import cv2
import mediapipe as mp
import numpy as np
import json
import os

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
def extract_swing_features(frame_path_folder):

    frame_paths = os.listdir(frame_path_folder)

    features = {}

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
        left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
        right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
        left_elbow = (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y)
        right_elbow = (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y)
        left_wrist = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y)
        right_wrist = (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y)
        left_knuckle = (landmarks[mp_pose.PoseLandmark.LEFT_INDEX].x, landmarks[mp_pose.PoseLandmark.LEFT_INDEX].y)  # Knuckle
        right_knuckle = (landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].x, landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].y)  # Knuckle
        left_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y)
        right_hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y)

        # Calculate joint angles
        sholder_tilt = calculate_angle(left_shoulder, right_shoulder, right_hip)  # Torso tilt
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)  # Lead arm angle
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)  # Lead arm angle
        left_wrist_angle = calculate_angle(left_elbow, left_wrist, left_knuckle)  # left wrist hinge
        right_wrist_angle = calculate_angle(right_elbow, right_wrist, right_knuckle)  # right wrist hinge
        hip_rotation = calculate_angle(left_hip, right_hip, right_shoulder)  # Hip turn

        # Store features
        features[event_name] = {
            "sholder_tilt": sholder_tilt,
            "left_elbow_angle": left_elbow_angle,
            "right_elbow_angle": right_elbow_angle,
            "left_wrist_angle": left_wrist_angle,
            "right_wrist_angle": right_wrist_angle,
            "hip_rotation": hip_rotation,
        }

    return features

# Example usage
if __name__ == "__main__":
    # Replace with paths to extracted swing frames
    frame_path_folder = "HS_bali"

    swing_features = extract_swing_features(f"{frame_path_folder}/frames")

    # Save features to JSON file
    with open(f"{frame_path_folder}/swing_features.json", "w") as f:
        json.dump(swing_features, f, indent=4)

    print("Swing features extracted and saved to swing_features.json")