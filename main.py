import argparse
from ultralytics import YOLO
import cv2

from sort.sort import *
from util1 import get_car, read_license_plate, write_csv
import datetime
import numpy as np
import os

def main(video_path, output_csv, license_plate_model_path, frame_skip=2):
    results = {}
    mot_tracker = Sort()

    # Load models
    try:
        coco_model = YOLO('yolov8n.pt')
        license_plate_detector = YOLO(license_plate_model_path)
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # Load video
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
    except Exception as e:
        print(f"Error opening video: {e}")
        return

    # Get video properties
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    vehicles = [2, 3, 5, 7]  # Vehicle classes in COCO dataset

    # Read frames
    frame_nmr = -1
    
    ret = True
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()

        # Skip frames if necessary
        if frame_nmr % 2 != 0:
            continue

        if ret:
            results[frame_nmr] = {}
            # Calculate timestamp
            timestamp = datetime.timedelta(seconds=frame_nmr / frame_rate)
            timestamp_str = str(timestamp)

            # Detect vehicles
            try:
                detections = coco_model(frame)[0]
                detections_ = []
                for detection in detections.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = detection
                    if int(class_id) in vehicles:
                        detections_.append([x1, y1, x2, y2, score])

                # Track vehicles
                track_ids = mot_tracker.update(np.asarray(detections_))

                # Detect license plates
                license_plates = license_plate_detector(frame)[0]
                for license_plate in license_plates.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = license_plate

                    # Assign license plate to car
                    xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                    if car_id != -1:
                        # Crop license plate
                        license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                        # Process license plate
                        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                        # Read license plate number
                        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                        if license_plate_text is not None:
                            # Store timestamp for this car ID at this frame
                            if car_id not in results[frame_nmr]:
                                results[frame_nmr][car_id] = {
                                    'timestamp': timestamp_str,
                                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                    'license_plate': {
                                        'bbox': [x1, y1, x2, y2],
                                        'text': license_plate_text,
                                        'bbox_score': score,
                                        'text_score': license_plate_text_score
                                    }
                                }
                            else:
                                # If the car ID already exists in this frame, update the timestamp
                                results[frame_nmr][car_id]['timestamp'] = timestamp_str
            except Exception as e:
                print(f"Error processing frame {frame_nmr}: {e}")

    # Write results to CSV
    try:
        write_csv(results, output_csv, frame_rate)
        print(f"Results saved to {output_csv}")
    except Exception as e:
        print(f"Error writing CSV: {e}")

    # Release video capture
    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a video for license plate detection.')
    parser.add_argument('--video', type=str, required=True, help='Path to the video file.')
    parser.add_argument('--output_csv', type=str, default='./test.csv', help='Path to the output CSV file.')
    parser.add_argument('--license_plate_model', type=str, default='./models/license_plate_detector.pt', help='Path to the license plate detection model.')
    parser.add_argument('--frame_skip', type=int, default=10, help='Process every Nth frame (default: 1).')
    args = parser.parse_args()

    # Check if video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file {args.video} does not exist.")
    else:
        main(args.video, args.output_csv, args.license_plate_model, args.frame_skip)