import os
import cv2
import numpy as np
import smtplib
import glob
import time
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import insightface
from insightface.app import FaceAnalysis
from flask import Blueprint, render_template, Response, request, jsonify

# Create the Blueprint
home_security_bp = Blueprint('home_security', __name__)

class HomeSecuritySystem:
    def __init__(self):
        self.face_analyzer = self.initialize_face_analyzer()
        self.known_faces = []
        self.known_names = []
        self.recognition_threshold = 0.5
        self.tracking_threshold = 0.3
        self.unknown_buffer_timeout = 3.0
        self.track_timeout = 1.0
        self.alert_cooldown = 60
        self.person_fingerprints = {}
        self.has_new_alert = False 
        self.email_sent = False
        
        # Create directories
        for dir_path in ["known_faces", "alerts", "static/uploads"]:
            os.makedirs(dir_path, exist_ok=True)
        
        self.faces_dir = "known_faces"
        self.alert_dir = "alerts"
        
        # Camera variables
        self.camera = None
        self.camera_type = "webcam"
        self.ip_address = None
        self.is_running = False
        self.last_frame = None
        self.output_frame = None
        self.lock = threading.Lock()
        
        # Load existing faces
        self.load_existing_faces()

    def initialize_face_analyzer(self):
        try:
            face_analyzer = FaceAnalysis(name="buffalo_l", root=".")
            face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            return face_analyzer
        except Exception as e:
            print(f"Failed to initialize InsightFace model: {str(e)}")
            print("Make sure you've installed InsightFace: pip install insightface onnxruntime")
            exit(1)
    
    def load_existing_faces(self):
        person_dirs = glob.glob(os.path.join(self.faces_dir, "*"))
        for person_dir in person_dirs:
            name = os.path.basename(person_dir)
            for face_file in glob.glob(os.path.join(person_dir, "*.npz")):
                try:
                    face_data = np.load(face_file)
                    self.known_faces.append(face_data['embedding'])
                    self.known_names.append(name)
                except Exception as e:
                    print(f"Error loading face: {str(e)}")
        
        print(f"Loaded {len(self.known_faces)} faces for {len(set(self.known_names))} people.")

    def add_face_from_images(self, name, image_files):
        person_dir = os.path.join(self.faces_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        
        if not image_files:
            return {"status": "error", "message": "No images provided"}, 0
        
        processed_count = 0
        for image_file in image_files:
            # Save the uploaded file temporarily
            temp_path = os.path.join("static/uploads", image_file.filename)
            image_file.save(temp_path)
            
            img = cv2.imread(temp_path)
            if img is None:
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.face_analyzer.get(img_rgb)
            
            if not faces:
                continue
                
            for face_idx, face in enumerate(faces):
                # Extract face and embedding
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = [max(0, bbox[0]), max(0, bbox[1]), 
                                  min(bbox[2], img.shape[1]), min(bbox[3], img.shape[0])]
                
                base_filename = os.path.splitext(os.path.basename(image_file.filename))[0]
                
                # Save embedding
                embedding_path = os.path.join(person_dir, f"{base_filename}_face{face_idx}.npz")
                np.savez(embedding_path, embedding=face.normed_embedding)
                
                # Save face image for reference
                cv2.imwrite(os.path.join(person_dir, f"{base_filename}_face{face_idx}.jpg"), 
                           img[y1:y2, x1:x2])
                
                self.known_faces.append(face.normed_embedding)
                self.known_names.append(name)
                processed_count += 1
        
        message = f"Added {processed_count} faces for {name}." if processed_count else "No faces detected."
        return {"status": "success", "message": message}, processed_count

    def calculate_similarity(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2)

    def find_best_match(self, face_embedding, threshold):
        if not self.known_faces:
            return None, 0, "Unknown"
            
        similarities = [self.calculate_similarity(face_embedding, known_face) 
                        for known_face in self.known_faces]
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity > threshold:
            return best_idx, best_similarity, self.known_names[best_idx]
        return None, best_similarity, "Unknown"

    def get_iou(self, bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        tx1, ty1, tx2, ty2 = bbox2
        
        # Calculate intersection
        ix1, iy1 = max(x1, tx1), max(y1, ty1)
        ix2, iy2 = min(x2, tx2), min(y2, ty2)
        
        if ix1 < ix2 and iy1 < iy2:
            intersection = (ix2 - ix1) * (iy2 - iy1)
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (tx2 - tx1) * (ty2 - ty1)
            union = area1 + area2 - intersection
            return intersection / union
        return 0

    def is_same_person(self, fingerprint1, fingerprint2, threshold=0.85):
        embedding_similarity = np.dot(fingerprint1[:20], fingerprint2[:20])
        size1, size2 = fingerprint1[20:], fingerprint2[20:]
        width_ratio = min(size1[0], size2[0]) / max(size1[0], size2[0])
        height_ratio = min(size1[1], size2[1]) / max(size1[1], size2[1])
        
        combined_score = embedding_similarity * 0.8 + (width_ratio * height_ratio) * 0.2
        return combined_score > threshold

    def setup_camera(self, camera_type, ip_address=None):
        if self.camera is not None:
            self.camera.release()
            
        if camera_type == "ipcam" and ip_address:
            print(f"Connecting to IP camera at: {ip_address}")
            cap = cv2.VideoCapture(ip_address)
            
            # Improved IP camera settings
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)
            
            if ip_address.startswith('rtsp'):
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
            
            self.camera_type = "ipcam"
            self.ip_address = ip_address
        else:
            cap = cv2.VideoCapture(0)
            self.camera_type = "webcam"
            self.ip_address = None
            
        self.camera = cap
        return cap.isOpened()

    def process_frames(self):
        # For tracking faces
        unknown_face_tracker = {}
        known_face_tracker = {}
        next_track_id = 0
        
        # Performance variables
        frame_count = 0
        last_status_time = time.time()
        
        # Frame grab optimization for IP camera - fix flickering
        last_processed_frame = None
        process_this_frame = True
        frame_interval = 3 if self.camera_type == "ipcam" else 1
        frame_counter = 0
        
        while self.is_running:
            if self.camera is None or not self.camera.isOpened():
                if self.setup_camera(self.camera_type, self.ip_address):
                    print("Camera reconnected")
                else:
                    print("Failed to connect to camera, retrying...")
                    time.sleep(1)
                    continue
            
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to capture frame. Retrying...")
                if self.camera_type == "ipcam" and self.ip_address:
                    self.camera.release()
                    time.sleep(1)
                    self.camera = cv2.VideoCapture(self.ip_address)
                continue
            
            current_time = time.time()
            
            # Maintain a consistent processing rate
            frame_counter = (frame_counter + 1) % frame_interval
            process_this_frame = (frame_counter == 0)
            
            # Always have a display frame
            display_frame = frame.copy()
            
            # Show status every 10 seconds
            if current_time - last_status_time > 10:
                fps = frame_count / (current_time - last_status_time)
                print(f"FPS: {fps:.1f}, Frames processed: {frame_count}")
                frame_count = 0
                last_status_time = current_time
            
            # Only process some frames for performance
            if process_this_frame:
                frame_count += 1
                
                # Resize large frames for better performance
                height, width = frame.shape[:2]
                if width > 640:
                    scale = 640 / width
                    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                    display_frame = frame.copy()
                
                # Process faces
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    faces = self.face_analyzer.get(frame_rgb)
                except Exception as e:
                    print(f"Error detecting faces: {e}")
                    faces = []
                
                # Reset matched flags
                for trackers in [known_face_tracker, unknown_face_tracker]:
                    for face_id in trackers:
                        trackers[face_id]['matched_this_frame'] = False
                
                # Process detected faces
                known_faces_in_frame = []
                
                for face in faces:
                    bbox = face.bbox.astype(int)
                    face_embedding = face.normed_embedding
                    
                    best_match, best_similarity, name = self.find_best_match(face_embedding, self.recognition_threshold)
                    
                    if best_match is not None:
                        # Known person
                        known_faces_in_frame.append(name)
                        
                        # Update or create tracker
                        tracked = False
                        for face_id, data in known_face_tracker.items():
                            if data['name'] == name and self.get_iou(bbox, data['bbox']) > 0.3:
                                known_face_tracker[face_id].update({
                                    'last_seen': current_time,
                                    'bbox': tuple(bbox),
                                    'embedding': face_embedding,
                                    'matched_this_frame': True
                                })
                                tracked = True
                                break
                        
                        if not tracked:
                            # Start tracking this known face
                            track_id = f"known_{next_track_id}"
                            next_track_id += 1
                            
                            known_face_tracker[track_id] = {
                                'name': name,
                                'first_seen': current_time,
                                'last_seen': current_time,
                                'bbox': tuple(bbox),
                                'embedding': face_embedding,
                                'matched_this_frame': True
                            }
                        
                        # Display
                        cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        cv2.putText(display_frame, f"{name} ({best_similarity:.2f})", 
                                    (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        # Remove from unknown if needed
                        for face_id in list(unknown_face_tracker.keys()):
                            if self.get_iou(bbox, unknown_face_tracker[face_id]['bbox']) > 0.5:
                                del unknown_face_tracker[face_id]
                    else:
                        # Unknown person - look for existing tracker first
                        tracked = False
                        for face_id, data in unknown_face_tracker.items():
                            if self.get_iou(bbox, data['bbox']) > 0.3:
                                unknown_face_tracker[face_id].update({
                                    'last_seen': current_time,
                                    'frame': frame.copy(),
                                    'bbox': tuple(bbox),
                                    'embedding': face_embedding,
                                    'matched_this_frame': True
                                })
                                tracked = True
                                
                                # Generate fingerprint if needed
                                if 'fingerprint' not in data:
                                    unknown_face_tracker[face_id]['fingerprint'] = \
                                        np.concatenate([face_embedding[:20], np.array([bbox[2]-bbox[0], bbox[3]-bbox[1]])])
                                break
                        
                        if not tracked:
                            # New unknown face
                            track_id = f"unknown_{next_track_id}"
                            next_track_id += 1
                            
                            unknown_face_tracker[track_id] = {
                                'first_seen': current_time,
                                'last_seen': current_time,
                                'last_alert': 0,
                                'frame': frame.copy(),
                                'bbox': tuple(bbox),
                                'embedding': face_embedding,
                                'fingerprint': np.concatenate([face_embedding[:20], np.array([bbox[2]-bbox[0], bbox[3]-bbox[1]])]),
                                'matched_this_frame': True
                            }
                        
                        # Display
                        cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                        cv2.putText(display_frame, f"Unknown ({best_similarity:.2f})", 
                                    (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                # Clean up old trackers
                for tracker_dict, timeout in [(known_face_tracker, self.track_timeout), 
                                            (unknown_face_tracker, self.track_timeout)]:
                    for face_id in list(tracker_dict.keys()):
                        if current_time - tracker_dict[face_id]['last_seen'] > timeout:
                            del tracker_dict[face_id]
                
                # Process alerts for unknown faces (if no known people present)
                if not known_faces_in_frame and unknown_face_tracker:
                    for face_id, data in unknown_face_tracker.items():
                        if not data.get('matched_this_frame', False):
                            continue
                        
                        time_visible = current_time - data['first_seen']
                        last_alert_time = data.get('last_alert', 0)
                        
                        if (time_visible > self.unknown_buffer_timeout and 
                            current_time - last_alert_time > self.alert_cooldown):
                            
                            # Check for duplicate alerts using fingerprints
                            is_duplicate = False
                            current_fingerprint = data['fingerprint']
                            
                            # Clean old fingerprints (older than 1 hour)
                            for fp_id in list(self.person_fingerprints.keys()):
                                if current_time - self.person_fingerprints[fp_id]['time'] > 3600:
                                    del self.person_fingerprints[fp_id]
                            
                            # Check against stored fingerprints
                            for fp_data in self.person_fingerprints.values():
                                if self.is_same_person(current_fingerprint, fp_data['fingerprint']):
                                    fp_data['time'] = current_time
                                    is_duplicate = True
                                    break
                            
                            if not is_duplicate:
                                print(f"Alert: Unknown face visible for {time_visible:.1f} seconds")
                                self.send_alert_email(data['frame'], data['bbox'], self.camera_type)
                                unknown_face_tracker[face_id]['last_alert'] = current_time
                                
                                # Store fingerprint
                                self.person_fingerprints[f"fp_{next_track_id}"] = {
                                    'fingerprint': current_fingerprint,
                                    'time': current_time
                                }
                                next_track_id += 1
                
                # Display info
                cv2.putText(display_frame, f"Known: {len(known_face_tracker)} | Unknown: {len(unknown_face_tracker)}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Save the processed frame for consistent display
                last_processed_frame = display_frame.copy()
            elif last_processed_frame is not None and self.camera_type == "ipcam":
                # If not processing this frame but we have a previous processed frame, use it
                display_frame = last_processed_frame.copy()
            
            # Update the output frame
            with self.lock:
                self.output_frame = display_frame.copy()
                
            # Small delay to reduce CPU usage
            time.sleep(0.01)
        
        # Clean up when stopped
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def send_alert_email(self, frame, bbox, camera_type="webcam"):
        print("Preparing alert email...")
        email = "projectmini969@gmail.com"  # Replace with your email
        password = "dqjy kezc bvum ucut"    # Replace with your password
        recipient = "sivaneeitheathjilla@gmail.com"  # Replace with recipient

        try:
            # Highlight face and add timestamp
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, time.strftime("%Y-%m-%d %H:%M:%S"), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Save image
            timestamp_str = time.strftime("%Y%m%d-%H%M%S")
            alert_path = os.path.join(self.alert_dir, f"unknown_{timestamp_str}.jpg")

            if frame is None or frame.size == 0:
                print("Error: Invalid frame for alert")
                return False

            cv2.imwrite(alert_path, frame)

            # Create email
            msg = MIMEMultipart()
            msg['From'] = email
            msg['To'] = recipient
            msg['Subject'] = f"🚨 Security Alert - {time.strftime('%I:%M %p')}"

            body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6;">
                <div style="padding: 20px; background-color: #f8f8f8; border-left: 4px solid #ff0000;">
                    <h2 style="color: #cc0000;">⚠️ Home Security Alert</h2>
                    <p><strong>Alert:</strong> Unidentified person detected</p>
                    <p><strong>Time:</strong> {time.strftime('%I:%M %p, %A, %B %d, %Y')}</p>
                    <p><strong>Camera:</strong> {camera_type.upper()}</p>
                </div>
            </body>
            </html>
            """

            msg.attach(MIMEText(body, 'html'))

            # Attach image
            with open(alert_path, "rb") as attachment:
                part = MIMEImage(attachment.read(), name=os.path.basename(alert_path))
                part['Content-Disposition'] = f'attachment; filename="{os.path.basename(alert_path)}"'
                msg.attach(part)

            # Send email
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(email, password)
                server.sendmail(email, recipient, msg.as_string())

            print("Alert email sent successfully.")
             # Set the flag to indicate a new alert
            self.email_sent = True  # Set the flag to indicate email was sent
            return True
        except Exception as e:
            print(f"Failed to send email: {str(e)}")
            self.email_sent = False  # Set the flag to indicate email failed
            return False

    def clear_faces(self):
        self.known_faces = []
        self.known_names = []
        print("All known faces cleared from memory.")
        
        import shutil
        if os.path.exists(self.faces_dir):
            shutil.rmtree(self.faces_dir)
            os.makedirs(self.faces_dir)
        print("All face images deleted from disk.")
        return True
        
    def start_security(self):
        """Start the security system in a separate thread"""
        if self.is_running:
            return False
            
        if not self.known_faces:
            print("No faces in the database. Please add faces first.")
            return False
            
        self.is_running = True
        threading.Thread(target=self.process_frames, daemon=True).start()
        return True
        
    def stop_security(self):
        """Stop the security system"""
        self.is_running = False
        # Wait for cleanup
        time.sleep(0.5)
        return True
    
    def generate_frames(self):
        """Generate frames for streaming"""
        while True:
            # Wait until we have a frame
            if self.output_frame is None:
                time.sleep(0.1)
                continue
                
            # Encode the frame in JPEG format
            with self.lock:
                if self.output_frame is None:
                    continue
                    
                (flag, encodedImage) = cv2.imencode(".jpg", self.output_frame)
                
                if not flag:
                    continue
                    
            # Yield the output frame in the byte format
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                 bytearray(encodedImage) + b'\r\n')
            
            # Brief pause
            time.sleep(0.05)

# Create an instance of the security system
security_system = HomeSecuritySystem()

@home_security_bp.route('/home_security')
def index():
    # Get list of known people
    known_people = set(security_system.known_names)
    
    # Get list of alerts
    alerts = []
    for alert_file in sorted(glob.glob(os.path.join(security_system.alert_dir, "*.jpg")), reverse=True):
        alerts.append({
            'filename': os.path.basename(alert_file),
            'timestamp': os.path.basename(alert_file).split("_")[1].split(".")[0],
            'path': '/alerts/' + os.path.basename(alert_file)
        })
    
    return render_template('index.html', is_running=security_system.is_running,
                          known_people=known_people, alerts=alerts[:10],
                          camera_type=security_system.camera_type,
                          ip_address=security_system.ip_address)

@home_security_bp.route('/home_security/video_feed')
def video_feed():
    if not security_system.is_running:
        # Return a placeholder image when not running
        with open('static/placeholder.jpg', 'rb') as f:
            placeholder = f.read()
        return Response(placeholder, mimetype='image/jpeg')
    
    # Return the video stream
    return Response(security_system.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@home_security_bp.route('/home_security/alerts/<path:filename>')
def serve_alert(filename):
    return Response(open(os.path.join(security_system.alert_dir, filename), 'rb').read(), 
                   mimetype='image/jpeg')

@home_security_bp.route('/home_security/start_security', methods=['POST'])
def start_security():
    camera_type = request.form.get('camera_type', 'webcam')
    ip_address = request.form.get('ip_address', '') if camera_type == 'ipcam' else None
    
    # Setup camera
    if not security_system.setup_camera(camera_type, ip_address):
        return jsonify({"status": "error", "message": "Failed to connect to camera"})
    
    # Start security system
    if security_system.start_security():
        return jsonify({"status": "success", "message": "Security system started"})
    else:
        return jsonify({"status": "error", "message": "Failed to start security system"})

@home_security_bp.route('/home_security/stop_security', methods=['POST'])
def stop_security():
    if security_system.stop_security():
        return jsonify({"status": "success", "message": "Security system stopped"})
    return jsonify({"status": "error", "message": "Failed to stop security system"})

@home_security_bp.route('/home_security/add_face', methods=['POST'])
def add_face():
    name = request.form.get('name', '')
    files = request.files.getlist('images')
    
    if not name or not files:
        return jsonify({"status": "error", "message": "Name and image files are required"})
    
    result, count = security_system.add_face_from_images(name, files)
    return jsonify(result)

@home_security_bp.route('/home_security/clear_faces', methods=['POST'])
def clear_faces():
    if security_system.clear_faces():
        return jsonify({"status": "success", "message": "All faces cleared"})
    return jsonify({"status": "error", "message": "Failed to clear faces"})

@home_security_bp.route('/home_security/get_status', methods=['GET'])
def get_status():
    return jsonify({
        "is_running": security_system.is_running,
        "camera_type": security_system.camera_type,
        "ip_address": security_system.ip_address,
        "known_faces_count": len(security_system.known_faces),
        "known_people_count": len(set(security_system.known_names)) if security_system.known_names else 0,
        "has_new_alert": security_system.has_new_alert,  # Include the new alert flag
        "email_sent": security_system.email_sent  # Include the email sending status
    })

@home_security_bp.route('/home_security/reset_email_sent', methods=['POST'])
def reset_email_sent():
    security_system.email_sent = False
    return jsonify({"status": "success", "message": "Email sent flag reset"})