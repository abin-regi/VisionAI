import string
import easyocr
import time
import numpy as np

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

def write_csv(results, output_path, fps):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
        fps (int): Frame rate of the video.
    """
    with open(output_path, 'w') as f:
        # Write header
        f.write('frame_nmr,car_id,timestamp,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score\n')
        
        # Write data
        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                row = results[frame_nmr][car_id]
                f.write(
                    f"{frame_nmr},{car_id},{row['timestamp']},"
                    f"[{row['car']['bbox'][0]} {row['car']['bbox'][1]} {row['car']['bbox'][2]} {row['car']['bbox'][3]}],"
                    f"[{row['license_plate']['bbox'][0]} {row['license_plate']['bbox'][1]} {row['license_plate']['bbox'][2]} {row['license_plate']['bbox'][3]}],"
                    f"{row['license_plate']['bbox_score']},{row['license_plate']['text']},{row['license_plate']['text_score']}\n"
                )
def license_complies_format(text):
    """Check if the license plate text complies with the required format."""
    if len(text) != 7:
        return False
    return all(
        (text[i] in string.ascii_uppercase or text[i] in dict_int_to_char.keys()) if i in [0, 1, 4, 5, 6] else
        (text[i] in '0123456789' or text[i] in dict_char_to_int.keys()) for i in range(7)
    )

def format_license(text):
    """Format the license plate text by converting characters using the mapping dictionaries."""
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    return ''.join(mapping[i].get(text[i], text[i]) for i in range(7))

def read_license_plate(license_plate_crop):
    """Read the license plate text from the given cropped image."""
    detections = reader.readtext(license_plate_crop)
    for bbox, text, score in detections:
        text = text.upper().replace(' ', '')
        if license_complies_format(text):
            return format_license(text), score
    return None, None

def get_car(license_plate, vehicle_track_ids):
    """Retrieve the vehicle coordinates and ID based on the license plate coordinates."""
    x1, y1, x2, y2, _, _ = license_plate
    for xcar1, ycar1, xcar2, ycar2, car_id in vehicle_track_ids:
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id
    return -1, -1, -1, -1, -1