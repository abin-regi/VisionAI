import cv2
import numpy as np
import datetime
import tkinter as tk
from tkinter import filedialog
import os
import urllib.request

class HighAccuracyFaceTracker:
    def __init__(self):
        # Initialize face detection and recognition models
        self.detector = self.initialize_face_detector()
        self.recognizer = self.initialize_face_recognizer()
        self.known_embeddings = []
        self.tracking_history = []
        self.current_track = None
        self.similarity_threshold = 0.20  # Lower is more strict
        self.frame_processing_rate = 1      # Process 1 frame per second in search mode
        # ADDED: Frame processing rate for tracking mode
        self.tracking_frame_rate = 5       # Process 5 frames per second in tracking mode

        # Mode management: "search" (detection/recognition) or "tracking" (optical flow)
        self.mode = "search"
        self.person_present = False
        self.last_presence_change = 0

        # Variables for optical flow–based tracking
        self.tracking_points = None  # Points to track (Nx1x2 np.array)
        self.tracking_bbox = None    # Current bounding box (x, y, w, h)
        self.prev_gray = None        # Previous frame in grayscale
        
        # ADDED: Tracking motion metrics
        self.last_tracking_update = 0  # Last time tracking was updated
        self.total_frames_processed = 0
        self.total_frames_skipped = 0

        # Snapshot directory
        self.snapshot_directory = "static/person_snapshots"
        os.makedirs(self.snapshot_directory, exist_ok=True)

    def initialize_face_detector(self):
        """Initialize YuNet face detector"""
        model_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        model_path = "face_detection_yunet_2023mar.onnx"
        if not os.path.exists(model_path):
            print("Downloading YuNet face detection model...")
            try:
                urllib.request.urlretrieve(model_url, model_path)
            except Exception as e:
                print(f"Failed to download detector: {str(e)}")
                exit(1)
        return cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.8
        )

    def initialize_face_recognizer(self):
        """Initialize SFace recognizer"""
        model_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
        model_path = "face_recognition_sface_2021dec.onnx"
        if not os.path.exists(model_path):
            print("Downloading SFace recognition model...")
            try:
                urllib.request.urlretrieve(model_url, model_path)
            except Exception as e:
                print(f"Failed to download recognizer: {str(e)}")
                exit(1)
        return cv2.FaceRecognizerSF.create(
            model=model_path,
            config=""
        )

    def load_reference_images(self, image_paths):
        """Process reference images to create face embeddings"""
        print("\nProcessing reference images:")
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"! Failed to load image: {os.path.basename(path)}")
                continue
            faces = self.detect_faces(img)
            if len(faces) == 0:
                print(f"! No face found in: {os.path.basename(path)}")
                continue
            # Use the first detected face
            aligned_face = self.recognizer.alignCrop(img, faces[0])
            embedding = self.recognizer.feature(aligned_face)
            self.known_embeddings.append(embedding)
            print(f"- Processed: {os.path.basename(path)}")

    def detect_faces(self, frame):
        """Detect faces using YuNet"""
        self.detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self.detector.detect(frame)
        return faces if faces is not None else []

    def capture_person_snapshot(self, frame, bbox, timestamp):
        """
        Capture and save a full frame snapshot of the detected person
        
        Args:
            frame (numpy.ndarray): The current video frame
            bbox (tuple): Bounding box of the detected person (x, y, w, h)
            timestamp (float): Timestamp of the detection
        
        Returns:
            str: Path to the saved snapshot image
        """
        # Create a copy of the frame to annotate
        annotated_frame = frame.copy()
        
        # Draw a rectangle around the detected face
        x, y, w, h = bbox
        cv2.rectangle(annotated_frame, 
                      (x, y), 
                      (x+w, y+h), 
                      (0, 255, 0),  # Green color
                      2)  # 2-pixel thickness
        
        # Add timestamp text to the frame
        timestamp_str = self.format_time(timestamp)
        cv2.putText(annotated_frame, 
                    f"Detected at: {timestamp_str}", 
                    (10, 30),  # Position of text
                    cv2.FONT_HERSHEY_SIMPLEX,  # Font
                    1,  # Font scale
                    (0, 255, 0),  # Text color (green)
                    2)  # Text thickness
        
        # Generate a unique filename based on timestamp
        filename = "snapshot.jpg"
        filepath = os.path.join(self.snapshot_directory, filename)

        # Save the annotated full frame snapshot
        try:
            cv2.imwrite(filepath, annotated_frame)
            print(f"✓ Full frame snapshot saved: {filename}")
            return filepath
        except Exception as e:
            print(f"! Failed to save snapshot: {str(e)}")
            return None

    def process_video(self, video_path):
        """Main processing pipeline running in background"""
        print("\nStarting video analysis (background processing)...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize default values
        appearance_info = "No person detected."
        total_duration = 0

        # Calculate frame intervals for both modes
        frame_interval = int(fps / self.frame_processing_rate)  # For search mode
        tracking_frame_interval = int(fps / self.tracking_frame_rate)  # For tracking mode

        print(f"Video Info:")
        print(f"  - FPS: {fps:.1f}")
        print(f"  - Total Frames: {total_frames}")
        print(f"  - Search mode processing at: {self.frame_processing_rate} FPS (every {frame_interval} frames)")
        print(f"  - Tracking mode processing at: {self.tracking_frame_rate} FPS (every {tracking_frame_interval} frames)\n")

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_count / fps
            time_str = self.format_time(timestamp)
            frame_count += 1
            self.total_frames_processed += 1

            if self.mode == "search":
                # Process only at the specified rate (1 frame per second)
                if frame_count % frame_interval == 0:
                    person_detected, bbox = self.process_frame_search(frame, timestamp)
                    if person_detected:
                        print(f"=== PERSON FOUND at {time_str} ===")
                        self.person_present = True
                        self.last_presence_change = timestamp
                        self.tracking_bbox = bbox
                        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        x, y, w, h = bbox
                        roi = self.prev_gray[y:y+h, x:x+w]
                        points = cv2.goodFeaturesToTrack(roi, maxCorners=100, qualityLevel=0.01, minDistance=5)
                        if points is not None:
                            points[:, 0, 0] += x
                            points[:, 0, 1] += y
                            self.tracking_points = points
                        else:
                            self.tracking_points = None
                        self.mode = "tracking"
                        self.last_tracking_update = timestamp
                        self.update_tracking(timestamp)
                    else:
                        print(f"Timestamp {time_str} - No matching face found.")
                        self.handle_tracking_status(False, timestamp)
                else:
                    self.total_frames_skipped += 1
            elif self.mode == "tracking":
                if frame_count % tracking_frame_interval == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if self.tracking_points is not None:
                        new_points, status, error = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.tracking_points, None)
                        good_new = new_points[status.flatten() == 1]
                        if len(good_new) < 5:
                            print(f"=== TRACKING LOST at {time_str}. Reverting to search mode. ===")
                            self.handle_tracking_status(False, timestamp)
                            self.mode = "search"
                            self.tracking_points = None
                            self.tracking_bbox = None
                        else:
                            x, y, w, h = cv2.boundingRect(good_new.reshape(-1, 1, 2))
                            self.tracking_bbox = (x, y, w, h)
                            self.update_tracking(timestamp)
                            self.tracking_points = good_new.reshape(-1, 1, 2)
                            self.prev_gray = gray.copy()
                            if timestamp - self.last_tracking_update >= 1.0:
                                print(f"Tracking at {time_str} ...")
                                self.last_tracking_update = timestamp
                    else:
                        print(f"=== NO TRACKING POINTS at {time_str}. Reverting to search mode. ===")
                        self.handle_tracking_status(False, timestamp)
                        self.mode = "search"
                else:
                    self.total_frames_skipped += 1

        cap.release()
        self.finalize_tracking()
        self.generate_report()

        # Format appearance_info and total_duration
        if self.tracking_history:
            appearance_info = ""
            for idx, track in enumerate(self.tracking_history, 1):
                for appearance in track['appearances']:
                    start_time = self.format_time(appearance['start'])
                    end_time = self.format_time(appearance['end'])
                    duration = self.format_time(appearance['end'] - appearance['start'])
                    appearance_info += f"Appearance {idx}:\n  {start_time} - {end_time} ({duration})\n"
            
            total_duration = sum(appearance['end'] - appearance['start'] for track in self.tracking_history for appearance in track['appearances'])
            total_duration_str = self.format_time(total_duration)
        else:
            appearance_info = "No person detected."
            total_duration_str = "0:00:00"

        return appearance_info, total_duration_str

    def process_frame_search(self, frame, timestamp):
        """
        In search mode, run face detection/recognition.
        Returns (person_detected, bbox) where bbox is (x, y, w, h)
        """
        time_str = self.format_time(timestamp)
        faces = self.detect_faces(frame)
        print(f"\nTimestamp {time_str} - Detected {len(faces)} face(s)")
        best_match = None
        best_similarity = -1
        
        for face in faces:
            aligned_face = self.recognizer.alignCrop(frame, face)
            embedding = self.recognizer.feature(aligned_face)
            for ref_emb in self.known_embeddings:
                similarity = self.recognizer.match(ref_emb, embedding)
                print(f"  Similarity score: {similarity:.2f} at {time_str}")
                
                # Keep track of the best match
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = (int(face[0]), int(face[1]), int(face[2]), int(face[3]))
        
        if best_match:
            print(f"  ✓ Person identified at {time_str} with best similarity {best_similarity:.2f}")
            
            # Capture and save full frame snapshot
            self.capture_person_snapshot(frame, best_match, timestamp)
            
            return True, best_match
        
        return False, None

    def update_tracking(self, timestamp):
        """Update tracking history"""
        if not self.current_track:
            self.current_track = {
                'start': timestamp,
                'last_seen': timestamp,
                'appearances': []
            }
        else:
            self.current_track['last_seen'] = timestamp

        if (not self.current_track['appearances'] or 
            (timestamp - self.current_track['appearances'][-1]['end'] > 2)):
            self.current_track['appearances'].append({
                'start': timestamp,
                'end': timestamp
            })
        else:
            self.current_track['appearances'][-1]['end'] = timestamp

    def handle_tracking_status(self, detected, timestamp):
        """Record the end of a tracking interval if the person is lost"""
        if not detected and self.current_track:
            self.tracking_history.append(self.current_track.copy())
            self.current_track = None

    def finalize_tracking(self):
        """Finalize any ongoing tracking interval"""
        if self.current_track:
            self.tracking_history.append(self.current_track)
            self.current_track = None

    def generate_report(self):
        """Print a summary report of the detection/tracking events"""
        print("\n=== FINAL DETECTION REPORT ===")
        if not self.tracking_history:
            print("No matching person detected.")
            return

        total_duration = 0
        for idx, track in enumerate(self.tracking_history, 1):
            print(f"\nAppearance {idx}:")
            for appearance in track['appearances']:
                duration = appearance['end'] - appearance['start']
                total_duration += duration
                print(f"  {self.format_time(appearance['start'])} - {self.format_time(appearance['end'])} " +
                      f"({self.format_time(duration)})")
        print(f"\nTotal tracked duration: {self.format_time(total_duration)}")
        
        # ADDED: Performance statistics
        skip_percentage = (self.total_frames_skipped / (self.total_frames_processed + self.total_frames_skipped)) * 100
        print(f"\nPerformance Statistics:")
        print(f"  - Total frames processed: {self.total_frames_processed}")
        print(f"  - Total frames skipped: {self.total_frames_skipped}")
        print(f"  - Frame skip rate: {skip_percentage:.1f}%")
        
        # Snapshot directory information
        print(f"\nSnapshot Directory: {os.path.abspath(self.snapshot_directory)}")

    @staticmethod
    def format_time(seconds):
        """Format seconds to HH:MM:SS"""
        return str(datetime.timedelta(seconds=seconds)).split('.')[0]
def main():
    print("=== High Accuracy Face Tracking (Background Processing) ===")
    root = tk.Tk()
    root.withdraw()

    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )
    image_paths = filedialog.askopenfilenames(
        title="Select Reference Images",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not video_path or not image_paths:
        print("File selection cancelled")
        return

    tracker = HighAccuracyFaceTracker()
    tracker.load_reference_images(image_paths)
    if not tracker.known_embeddings:
        print("No valid reference images loaded")
        return

    tracker.process_video(video_path)

if __name__ == "__main__":
    main()