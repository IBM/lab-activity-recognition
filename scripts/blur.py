"""
Blur faces in MP4 files recursively in a directory.

For help:
python blur.py --help
"""

# Import dependencies
import argparse
import cv2
from pathlib import Path

# Parse user arguments
parser = argparse.ArgumentParser(
    prog="blur.py",
    description="Blur faces in MP4 files recursively in a directory.",
    epilog="""
        By default, the script will look for MP4 files in the 'video' sub-directory and save the outputs to the 'output' sub-directory.
        If not explicitly provided, the codec used is 'avc1'. If this gives problems, you can try other options such as 'mp4v'.
        Press 'Q' to interrupt the script.
        """
)
parser.add_argument("-i", "--input_directory", type=str, default="videos")
parser.add_argument("-o", "--output_directory", type=str, default="output")
parser.add_argument("-c", "--codec", type=str, default="avc1")
args = parser.parse_args()

# Specify directories
input_directory = Path(args.input_directory)
output_directory = Path(args.output_directory)

# Get list of MP4 files in directory
mp4_files = [file for file in input_directory.glob("*.mp4")]
if mp4_files == []:
    print(f"Sub-directory {input_directory} does not contain any MP4 files. Exiting...")
    quit()

# Create the output directory if it doesn't exist
output_directory.mkdir(exist_ok=True)

# Loop through MP4 files
for mp4_file in mp4_files:

    # Print filename
    print(f"Processing {mp4_file}...")

    # Instantiate a VideoCapture object with the MP4 file
    video_capture = cv2.VideoCapture(mp4_file)

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize YuNet face detection model
    detector = cv2.FaceDetectorYN.create(
        model="face_detection_yunet_2023mar.onnx",
        config="",
        input_size=(frame_width, frame_height),
        score_threshold=0.8,
        nms_threshold=0.3,
        top_k=5000
    )
    detector.setInputSize((frame_width, frame_height))

    # Define the codec and create a VideoWriter object for the output
    fourcc = cv2.VideoWriter_fourcc(*f"{args.codec}")
    out = cv2.VideoWriter(Path(output_directory, mp4_file.name), fourcc=fourcc, fps=fps, frameSize=(frame_width, frame_height))

    # Read video until end
    while(video_capture.isOpened()):
        
        # Capture frame
        ret, frame = video_capture.read()
        if ret:
            result = detector.detect(frame)
            faces = []
            if result[1] is not None:
                for idx, face in enumerate(result[1]):
                    coords = face[:-1].astype(int)
                    x, y, w, h = coords[:4]
                    if x < 0:
                        x = 0
                    if y < 0:
                        y = 0
                    faces.append([x, y, w, h])  
                    frame[y:y+h, x:x+w] = cv2.medianBlur(frame[y:y+h, x:x+w], 95)
        
            # Show the processed video frame
            cv2.imshow("Processed video", frame)
            key = cv2.waitKey(1)
        
            # Press 'Q' to interrupt
            if key == ord('q'):
                break

            # Write the frame into the output file
            out.write(frame)

        else:
            break

    # Release all objects
    video_capture.release()
    out.release()
    
    # Close all frames
    cv2.destroyAllWindows()
