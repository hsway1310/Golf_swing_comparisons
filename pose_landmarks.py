import cv2
import mediapipe as mp
import argparse
import os


def process_video(video_file):
    # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    # Open input video
    input_video_path = f"amateur_swings/{video_file}"
    swing_id = os.path.splitext(os.path.basename(input_video_path))[0]
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define output video writer
    output_dir = f"processed_swings/{swing_id}"
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = f"{output_dir}/{swing_id}_overlayed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB (MediaPipe requires RGB format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with MediaPipe Pose
        results = pose.process(rgb_frame)

        # Draw landmarks if detected
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        # Write frame to output video
        out.write(frame)

        # Display the frame (optional)
        cv2.imshow("Golf Swing Pose Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a golf swing video for pose detection."
    )
    parser.add_argument(
        "video_file",
        type=str,
        help="Path to the input video file (relative to 'amateur_swings/')",
    )
    args = parser.parse_args()

    process_video(args.video_file)
