import csv
import datetime
import cv2

def parse_timestamp(timestamp):
    """
    Parse a timestamp string into a datetime object.
    """
    try:
        if '.' in timestamp:
            hours, remainder = timestamp.split(':')[0], timestamp.split(':')[1:]
            minutes, seconds = remainder[0], remainder[1]
            seconds, microseconds = seconds.split('.')
            return datetime.datetime.strptime(f"{hours}:{minutes}:{seconds}.{microseconds}", "%H:%M:%S.%f")
        else:
            return datetime.datetime.strptime(timestamp, "%H:%M:%S")
    except ValueError:
        formats = ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%H:%M:%S"]
        for fmt in formats:
            try:
                return datetime.datetime.strptime(timestamp, fmt)
            except ValueError:
                continue
    raise ValueError(f"Timestamp format not recognized: {timestamp}")

def extract_snapshot(video_file, frame_nmr, bbox, output_path):
    """
    Extract a snapshot from the video at the specified frame number and bounding box.
    """
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_file}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nmr)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read frame {frame_nmr} from video.")

    x1, y1, x2, y2 = map(int, bbox)
    snapshot = frame[y1:y2, x1:x2]
    cv2.imwrite(output_path, snapshot)
    return output_path

def find_vehicle_details(input_plate, csv_file, video_file):
    """
    Find details of a vehicle with the specified license plate in the CSV file and video.
    """
    try:
        # Load CSV data
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            data = list(reader)

        if not data:
            return None

        # Filter rows that match the input license plate
        matching_rows = [row for row in data if row['license_number'].upper() == input_plate.upper()]

        if not matching_rows:
            return None

        # Group rows by car_id
        car_id_to_rows = {}
        for row in matching_rows:
            car_id = row['car_id']
            if car_id not in car_id_to_rows:
                car_id_to_rows[car_id] = []
            car_id_to_rows[car_id].append(row)

        # Process each car ID
        results = []
        for car_id, rows in car_id_to_rows.items():
            # Sort rows by frame number
            rows.sort(key=lambda x: int(x['frame_nmr']))

            # Get the first and last appearance timestamps
            first_timestamp = rows[0]['timestamp']
            last_timestamp = rows[-1]['timestamp']

            # Parse timestamps
            first_time = parse_timestamp(first_timestamp)
            last_time = parse_timestamp(last_timestamp)

            # Calculate duration
            duration = last_time - first_time

            # Extract snapshot from the first frame
            snapshot_frame_nmr = int(rows[0]['frame_nmr'])
            snapshot_bbox = list(map(float, rows[0]['car_bbox'][1:-1].split()))
            snapshot_path = f"static/snapshots/snapshot_{input_plate}_car_{car_id}.jpg"
            snapshot_url = extract_snapshot(video_file, snapshot_frame_nmr, snapshot_bbox, snapshot_path)

            # Prepare results
            results.append({
                'license_plate': input_plate,
                'car_id': car_id,
                'start_timestamp': first_timestamp,
                'end_timestamp': last_timestamp,
                'duration': str(duration),
                'snapshot_url': f'/static/snapshots/snapshot_{input_plate}_car_{car_id}.jpg'
            })

        return results
    except Exception as e:
        print(f"Error in find_vehicle_details: {e}")
        return None