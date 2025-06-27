# Blur faces in MP4 videos

## Summary

The [blur.py](blur.py) Python script outlines the use of [OpenCV](https://opencv.org/) and the [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) face detection model to blur faces in MP4 videos. It is intended to be run as a standalone script to support privacy preservation. The script is inspired by several excellent tutorials available on the web [[1](https://www.geeksforgeeks.org/python/faces-blur-in-videos-using-opencv-in-python/), [2](https://pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/), [3](https://colab.research.google.com/github/AsadiAhmad/Deep-Face-Blurring/blob/main/Code/Deep_Face_Blurring.ipynb)].

## Prerequisites

* A local installation of [Python](https://www.python.org/downloads/)

## Setup

1. Create a local working directory (e.g. `project`). You will set up the following structure:
    ```
    project
    ├── blur.py
    ├── face_detection_yunet_2023mar.onnx
    ├── videos
    └── output
    ```

2. Download the [blur.py](./blur.py) script to the working directory and your MP4 videos to the `videos` sub-directory.

3. Download the [YuNet face detection model](https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) to the working directory. The model is described [here](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).

4. Run the commands below in a terminal from within your working directory:

    Create a virtual Python environment
    ```bash
    python -m venv .venv
    ```

    Activate the virtual environment
    ```bash
    source .venv/bin activate           # Linux/macOS
    
    .venv\Scripts\activate               # Windows
    ```

    Install the OpenCV library in your virtual environment:
    ```bash
    pip install opencv-python~=4.11.0
    ```

## Usage

With your virtual environment activated, run the script:

```bash
python blur.py
```

This will process all MP4 files in the `videos` sub-directoy and save the MP4 files with blurred faces under the same names in the `output` sub-directory. You can customize these directories and the video codec used, run `python blur.py --help` for options.
