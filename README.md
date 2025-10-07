# Golf Swing Analyser - Haaris sohoye

Full demo of Golf Swing Comparison Application Demoed on Foundry AIP: [Feel vs Real](https://drive.google.com/file/d/1WnacQs7NpUQBKby2cN7FtH61IjmvbAuA/view?usp=sharing)

## Overview

Golf Swing Analyser is a tool that processes a video of a golf swing, identifies key swing positions, and compares them to professional golf swings. The tool overlays MediaPipe pose landmarks to assist with comparisons and analysis.

The full database of professional golf swings can be seen [here](https://drive.google.com/file/d/1uBwRxFxW04EqG87VCoX3l6vXeV5T5JYJ/view). Download, unzip and add the `videos_160` folder to `data/`.

```
@InProceedings{McNally_2019_CVPR_Workshops,
author = {McNally, William and Vats, Kanav and Pinto, Tyler and Dulhanty, Chris and McPhee, John and Wong, Alexander},
title = {GolfDB: A Video Database for Golf Swing Sequencing},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2019}
}
```

## Features

### 1. Frame Identification: 
Detects and extracts frames corresponding to key positions in a golf swing:

1. Address
2. Toe-up
3. Mid-backswing (arm parallel) 
4. Top
5. Mid-downswing (arm parallel)
6. Impact
7. Mid-follow-through (shaft parallel)
8. Finish

#### How to Use Frame Identification

- First, download the model weights from this Google Drive link: [model weights](https://drive.google.com/file/d/1MBIDwHSM8OKRbxS8YfyRLnUBAdt0nupW/view)

- Place the downloaded weights in the `models/` folder (if it doesn't already exist).

- Upload your swing video (croped and trimmed to show only your swing) to the `amateur_swings/`  folder.

- Run the following command (reference your own .mp4 file): `python3 test_video.py -p swings/HS_bali.mp4 -f overlayed` (`-f overlayed` to save the overlayed poselandmarks frames or `-f raw` to save the raw swing frames)

This will output 8 frames from the video, showing the different stages of the golf swing. Press any key to show the next frame.

After running the above, you can now run `extract_swing_features.py` to return the calculated angles at each stage of your swing:

1. sholder tilt (Torso tilt)
2. left elbow angle (Lead arm angle)
3. right elbow angle (Trail arm angle)
4. left wrist angle (Left wrist hinge)
5. right wrist angle (Right wrist hinge)
6. hip rotation (Hip turn)
7. Tempo (ratio between backswing time and downswing time -- a typical tour pro has a tempo ~3)

### 2. Pose Landmark Overlay: 
This script processes a golf swing video using MediaPipe's Pose detection model, overlays the detected pose landmarks, and saves the processed video.

#### Usage

Run the script from the terminal by providing the video filename as an argument:

`python pose_detection.py <VIDEO_FILE>`

#### Example

`python pose_detection.py AM_sawgrass.mp4`

#### Input & Output

Input Video: The script looks for the input video inside the `amateur_swings/` directory.

Output Video: The processed video with pose overlays is saved inside `processed_swings/<swing_id>/`.




### 3. Professional Swing Comparison (coming soon): 
Matches each extracted frame to the closest corresponding frame from professional golf swings.


## Technologies Used

[GolfDB](https://github.com/wmcnally/golfdb): A database of professional golf swings for benchmarking and comparison.

[MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python): A framework for real-time landmark detection and pose estimation.

[OpenCV](https://opencv.org/): For video processing and frame extraction.

## Installation

### Prerequisites

Ensure you have Python installed (>= 3.7). Then install dependencies:

`pip install -r requirements.txt`
