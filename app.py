from flask import Flask, request, jsonify, render_template, Response, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
from werkzeug.utils import secure_filename
import subprocess
import final
import time
import sjcetsurveillance 
from app1 import crowd_bp
from home_secuflask import home_security_bp

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'  # Change this to a secure key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # SQLite database
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable modification tracking
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'}

# Initialize database
db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager(app)
login_manager.login_view = 'login'

app.register_blueprint(crowd_bp, url_prefix='/crowd')
app.register_blueprint(home_security_bp, url_prefix='/home_security')

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Load user for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Routes
@app.route('/')
def index():
    return render_template('indexfinal.html')

@app.route('/upload_video_and_images', methods=['POST'])
@login_required
def upload_video_and_images():
    if 'video' not in request.files or 'images' not in request.files:
        return jsonify({'error': 'No video or image files uploaded'}), 400

    video_file = request.files['video']
    image_files = request.files.getlist('images')

    if video_file.filename == '':
        return jsonify({'error': 'No selected video file'}), 400

    if not image_files or any(image.filename == '' for image in image_files):
        return jsonify({'error': 'No selected image files'}), 400

    if allowed_file(video_file.filename):
        video_filename = secure_filename(video_file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        video_file.save(video_path)

        # Save images
        image_paths = []
        for image_file in image_files:
            if allowed_file(image_file.filename):
                image_filename = secure_filename(image_file.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
                image_file.save(image_path)
                image_paths.append(image_path)

    try:
        # Process the video and images using sjcetsurveillance
        tracker = sjcetsurveillance.HighAccuracyFaceTracker()
        tracker.load_reference_images(image_paths)
        
        # Process the video and get appearance info and total duration
        appearance_info, total_duration = tracker.process_video(video_path)
        appearance_info = appearance_info or "No appearance info available"
        total_duration = total_duration or "Unknown duration"

        # Hardcode the snapshot path for testing
        snapshot_path = 'static/person_snapshots/snapshot.jpg'  # Use your sample image

        # Prepare data to pass to the results page
        result_data = {
            'message': 'Video processed successfully',
            'video_path': video_path,
            'images': image_paths,
            'snapshot': snapshot_path,  # Add this line for the single snapshot
            'appearance_info': appearance_info,  # Add this line
            'total_duration': total_duration  # Add this line
        }
        return jsonify(result_data)  # Return as JSON
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid video file type'}), 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/vehicle')
@login_required  # Protect this route
def vehicle_identification():
    return render_template('vehicle.html')

@app.route('/person')
@login_required  # Protect this route
def person_identification():
    return render_template('person.html')

@app.route('/home')
@login_required  # Protect this route
def home_security():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        print("Login form submitted!")  # Debug statement
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
    return render_template('loginnew.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')

        # Debugging: Print form data
        print(f"Email: {email}, Username: {username}, Password: {password}")

        # Check if username already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('register'))

        # Check if email already exists
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
            return redirect(url_for('register'))

        # Create new user
        new_user = User(
            email=email,
            username=username
        )
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('registernew.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# Other routes (upload, search, progress) remain the same...

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    video_file = request.files['video']

    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if video_file and allowed_file(video_file.filename):
        filename = secure_filename(video_file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(video_path)

        try:
            # Process the video
            result = subprocess.run(
                ['python', 'main.py', '--video', video_path],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print(f"Error in main.py: {result.stderr}")
                return jsonify({'error': 'Failed to process video. Check server logs.'}), 500

            # Return success message and video path
            return jsonify({'message': 'Video processed successfully', 'video_path': video_path}), 200
        except Exception as e:
            print(f"Error: {e}")
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/search', methods=['POST'])
def search_license_plate():
    data = request.get_json()
    license_plate = data.get('license_plate')
    video_path = data.get('video_path')

    if not license_plate or not video_path:
        return jsonify({'error': 'License plate and video path are required'}), 400

    try:
        # Search for the license plate in the processed video
        results = final.find_vehicle_details(license_plate, 'test.csv', video_path)

        if not results:
            return jsonify({'error': 'No matching license plate found'}), 404

        return jsonify(results), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/progress')
def progress():
    def generate():
        progress = 0
        while progress <= 100:
            yield f"data: {progress}\n\n"
            time.sleep(1)
            progress += 10
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)