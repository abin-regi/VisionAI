from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, Blueprint
from flask_login import login_required, current_user
import cv2
import numpy as np
from sort.sort import Sort
from ultralytics import YOLO
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import threading
import queue

# Create a Blueprint
crowd_bp = Blueprint('crowd_bp', __name__)

# Global variables
frame_queue = queue.Queue(maxsize=10)
processing_data = {
    'count': 0,
    'limit_exceeded': False,
    'fps': 0,
    'alert_sent': False
}
video_running = False
config = {
    'video_source': None,
    'people_limit': 1,
    'line_start': (0, 0),
    'line_end': (0, 0),
    'inside_area_above': True,
    'receiver_email': None  # Add receiver_email to config
}

# Load YOLOv8 model
try:
    model = YOLO("yolov8n.pt")  # Load YOLOv8 Nano model
    print("YOLOv8 model loaded successfully")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    exit(1)

# Initialize SORT tracker
tracker = Sort(max_age=30, min_hits=3)

# Email configuration
def send_alert_email(count, limit, receiver_email, location="Monitored Area"):
    try:
        # Email credentials
        sender_email = "projectmini969@gmail.com"
        password = "dqjy kezc bvum ucut"  # App password
        
        # Create message
        msg = MIMEMultipart("alternative")
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = f"🚨 ALERT: Crowd Limit Exceeded at {location}"
        
        # Current time
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Calculate severity level (for color coding)
        if count <= limit + 2:
            severity = "Low"
            color = "#FFA500"  # Orange
        elif count <= limit + 5:
            severity = "Medium"
            color = "#FF6347"  # Tomato
        else:
            severity = "High"
            color = "#FF0000"  # Red
        
        # Plain text version (fallback)
        text = f"""
        CROWD DENSITY ALERT
        
        Location: {location}
        Time: {current_time}
        Current Count: {count}
        Maximum Allowed: {limit}
        Severity: {severity}
        
        This is an automated message from your Crowd Control System.
        """
        
        # HTML version
        html = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                }}
                .container {{
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background-color: #303f9f;
                    color: white;
                    padding: 15px;
                    text-align: center;
                    font-size: 24px;
                    border-radius: 5px 5px 0 0;
                }}
                .content {{
                    background-color: #f5f5f5;
                    padding: 20px;
                    border-left: 1px solid #ddd;
                    border-right: 1px solid #ddd;
                }}
                .footer {{
                    background-color: #e0e0e0;
                    padding: 15px;
                    text-align: center;
                    font-size: 12px;
                    border-radius: 0 0 5px 5px;
                    border: 1px solid #ddd;
                }}
                .alert-box {{
                    background-color: {color};
                    color: white;
                    padding: 15px;
                    margin: 15px 0;
                    border-radius: 5px;
                    text-align: center;
                    font-weight: bold;
                    font-size: 18px;
                }}
                .info-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                }}
                .info-table td {{
                    padding: 10px;
                    border-bottom: 1px solid #ddd;
                }}
                .info-table td:first-child {{
                    font-weight: bold;
                    width: 40%;
                }}
                .count {{
                    font-size: 24px;
                    font-weight: bold;
                    color: {color};
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    🚨 CROWD DENSITY ALERT
                </div>
                <div class="content">
                    <div class="alert-box">
                        Crowd Limit Exceeded! Severity: {severity}
                    </div>
                    
                    <table class="info-table">
                        <tr>
                            <td>Location:</td>
                            <td>{location}</td>
                        </tr>
                        <tr>
                            <td>Time:</td>
                            <td>{current_time}</td>
                        </tr>
                        <tr>
                            <td>Current Count:</td>
                            <td class="count">{count}</td>
                        </tr>
                        <tr>
                            <td>Maximum Allowed:</td>
                            <td>{limit}</td>
                        </tr>
                        <tr>
                            <td>Overflow:</td>
                            <td>+{count - limit} people</td>
                        </tr>
                    </table>
                    
                    <p>This alert requires immediate attention. Please check the monitored area and take appropriate action.</p>
                </div>
                <div class="footer">
                    This is an automated message from your Crowd Control System.<br>
                    Generated on {current_time}
                </div>
            </div>
        </body>
        </html>
        """
        
        # Attach both plain text and HTML versions
        part1 = MIMEText(text, "plain")
        part2 = MIMEText(html, "html")
        
        msg.attach(part1)
        msg.attach(part2)
        
        # Connect to Gmail SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        
        # Send email
        server.send_message(msg)
        server.quit()
        
        print(f"Alert email sent to {receiver_email}")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

def process_video_thread():
    global video_running, config, processing_data
    
    # Initialize variables
    count = 0
    people_in_room = set()
    previous_positions = {}
    alert_sent = False
    alert_cooldown = 300  # Cooldown time in seconds before sending another alert
    last_alert_time = 0
    
    # Resize dimensions for processing
    resize_width = 640
    resize_height = 480
    
    # Open video capture
    cap = cv2.VideoCapture(config['video_source'])
    if not cap.isOpened():
        print("Error: Could not access camera or video file.")
        video_running = False
        return
    
    # Get original frame dimensions
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        video_running = False
        cap.release()
        return
    
    original_height, original_width = first_frame.shape[:2]
    
    # Calculate scaling factors
    scale_x = resize_width / original_width
    scale_y = resize_height / original_height
    
    # Scale the line coordinates to match the resized frame
    line_start = (int(config['line_start'][0] * scale_x), int(config['line_start'][1] * scale_y))
    line_end = (int(config['line_end'][0] * scale_x), int(config['line_end'][1] * scale_y))
    
    print("Starting video processing...")
    
    # Frame processing settings
    frame_count = 0
    process_every_n_frames = 1  # Process every frame (adjust for performance)
    
    while video_running:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break
            
        # Resize for performance
        frame = cv2.resize(frame, (resize_width, resize_height))
        
        # Only process every n frames
        frame_count += 1
        if frame_count % process_every_n_frames != 0:
            # Still display the frame with basic info when skipping processing
            cv2.line(frame, line_start, line_end, (0, 0, 255), 2)
            cv2.putText(frame, f"People: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Convert to JPEG for streaming
            _, jpeg = cv2.imencode('.jpg', frame)
            
            # Put in queue for streaming
            try:
                if not frame_queue.full():
                    frame_queue.put(jpeg.tobytes())
            except:
                pass
                
            continue
            
        # Start timing for FPS calculation
        start_time = time.time()
        
        # Run object detection
        try:
            results = model(frame, verbose=False)  # Disable verbose output
            
            # Extract person detections
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if int(box.cls[0]) == 0:  # Class 0 is person
                        if float(box.conf[0]) > 0.4:  # Confidence threshold
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0])
                            detections.append([x1, y1, x2, y2, conf])
            
            # Convert to numpy array for tracker
            detections = np.array(detections, dtype=np.float32) if detections else np.empty((0, 5))
            
            # Update tracker
            tracks = tracker.update(detections)
            
            # Process each tracked person
            for track in tracks:
                x1, y1, x2, y2, track_id = map(int, track)
                    
                # Calculate center point
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                    
                # Draw bounding box and ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                # Check if person crossed the line
                if track_id in previous_positions:
                    prev_x, prev_y = previous_positions[track_id]
                        
                    # Check if the line is vertical
                    if line_end[0] == line_start[0]:
                        # Handle vertical line case
                        prev_side = prev_x > line_start[0]
                        curr_side = center_x > line_start[0]
                    else:
                        # Calculate if point crossed the line
                        slope = (line_end[1] - line_start[1]) / (line_end[0] - line_start[0])
                        intercept = line_start[1] - slope * line_start[0]
                            
                        # Determine the side of the line
                        prev_side = (prev_y - (slope * prev_x + intercept)) >= 0
                        curr_side = (center_y - (slope * center_x + intercept)) >= 0
                        
                    if prev_side != curr_side:  # Person crossed the line
                        if config['inside_area_above']:
                            if not curr_side:  # Entered (inside area is above the line)
                                count += 1  # Increase count
                                print(f"Someone entered. Count: {count}")
                            else:  # Exited
                                count = max(0, count - 1)  # Decrease count
                                print(f"Someone exited. Count: {count}")
                        else:
                            if curr_side:  # Entered (inside area is below the line)
                                count += 1  # Increase count
                                print(f"Someone entered. Count: {count}")
                            else:  # Exited
                                count = max(0, count - 1)  # Decrease count
                                print(f"Someone exited. Count: {count}")
                            
                # Update previous position
                previous_positions[track_id] = (center_x, center_y)
                    
        except Exception as e:
            print(f"Error in detection/tracking: {e}")
            
        # Draw line
        cv2.line(frame, line_start, line_end, (0, 0, 255), 2)
            
        # Display people count
        cv2.putText(frame, f"People: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        # Check if limit exceeded
        current_time = time.time()
        limit_exceeded = count > config['people_limit']
        if limit_exceeded:
            cv2.putText(frame, "WARNING: Limit exceeded!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            # Send email alert (with cooldown)
            if not alert_sent or (current_time - last_alert_time > alert_cooldown):
                if send_alert_email(count, config['people_limit'], config['receiver_email']):
                    alert_sent = True
                    last_alert_time = current_time
                    cv2.putText(frame, "Email Alert Sent!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            # Reset alert flag when count goes back below limit
            alert_sent = False
        
        # Calculate and display FPS
        end_time = time.time()
        time_diff = end_time - start_time
        fps = 1 / time_diff if time_diff > 0 else 0  # Avoid division by zero
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Update processing data for the web interface
        processing_data['count'] = count
        processing_data['limit_exceeded'] = limit_exceeded
        processing_data['fps'] = int(fps)
        processing_data['alert_sent'] = alert_sent
        
        # Convert to JPEG for streaming
        _, jpeg = cv2.imencode('.jpg', frame)
        
        # Put in queue for streaming
        try:
            if not frame_queue.full():
                frame_queue.put(jpeg.tobytes())
        except:
            pass
            
    # Release resources
    cap.release()
    print("Video processing terminated")

def generate_frames():
    while True:
        if not video_running:
            # Return a blank frame with a message when not running
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "Video Feed Stopped", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, jpeg = cv2.imencode('.jpg', blank_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        else:
            try:
                # Get frame from queue with timeout
                frame_bytes = frame_queue.get(timeout=1)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except queue.Empty:
                # Return a blank frame if queue is empty
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Processing...", (240, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, jpeg = cv2.imencode('.jpg', blank_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@crowd_bp.route('/crowd')
@login_required
def crowd_index():
    return render_template('crowd_control.html')

@crowd_bp.route('/crowd/setup')
def crowd_setup():
    return render_template('setup.html')

@crowd_bp.route('/crowd/video_feed')
def crowd_video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@crowd_bp.route('/crowd/start', methods=['POST'])
@login_required
def crowd_start():
    global video_running, config
    
    if video_running:
        return jsonify({'status': 'error', 'message': 'Video processing is already running'})
    
    # Get setup parameters
    source_choice = request.form.get('source_choice')
    if source_choice == '1':
        config['video_source'] = 0  # Webcam
    elif source_choice == '2':
        config['video_source'] = request.form.get('video_file')
    else:
        return jsonify({'status': 'error', 'message': 'Invalid source choice'})
    
    # Get people limit
    try:
        config['people_limit'] = int(request.form.get('people_limit', 1))
    except ValueError:
        config['people_limit'] = 1
    
    # Get line parameters
    try:
        x1 = int(request.form.get('line_x1', 0))
        y1 = int(request.form.get('line_y1', 0)) 
        x2 = int(request.form.get('line_x2', 640))
        y2 = int(request.form.get('line_y2', 0))
        config['line_start'] = (x1, y1)
        config['line_end'] = (x2, y2)
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Invalid line coordinates'})
    
    # Get inside area choice
    inside_choice = request.form.get('inside_choice')
    if inside_choice == '1':
        config['inside_area_above'] = True
    elif inside_choice == '2':
        config['inside_area_above'] = False
    else:
        return jsonify({'status': 'error', 'message': 'Invalid inside area choice'})
    
    # Set receiver email from current user
    config['receiver_email'] = current_user.email
    
    # Start video processing thread
    video_running = True
    threading.Thread(target=process_video_thread, daemon=True).start()
    
    return jsonify({'status': 'success', 'message': 'Video processing started'})

@crowd_bp.route('/crowd/stop', methods=['POST'])
def crowd_stop():
    global video_running
    if video_running:
        video_running = False
        return jsonify({'status': 'success', 'message': 'Video processing stopped'})
    else:
        return jsonify({'status': 'error', 'message': 'Video processing is not running'})

@crowd_bp.route('/crowd/status')
def crowd_status():
    return jsonify({
        'running': video_running,
        'count': processing_data['count'],
        'limit': config['people_limit'],
        'limit_exceeded': processing_data['limit_exceeded'],
        'fps': processing_data['fps'],
        'alert_sent': processing_data['alert_sent']
    })

@crowd_bp.route('/crowd/reset_count', methods=['POST'])
def crowd_reset_count():
    processing_data['count'] = 0
    return jsonify({'status': 'success', 'message': 'Count reset to 0'})