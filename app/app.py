from flask import Flask, render_template, redirect, url_for, flash, request, session, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import cv2
import numpy as np
from ultralytics import YOLO
import json
import requests
from datetime import datetime
import os

# ZeptoEmail Configuration
def get_email_config():
    config = EmailConfig.query.first()
    if config:
        return {
            'api_key': config.api_key,
            'from_email': config.from_email,
            'to_email': config.to_email
        }
    return None

def send_alert_email(people_count, max_allowed, frame_data=None):
    config = get_email_config()
    if not config:
        print('Email configuration not found')
        return
        
    url = 'https://api.zeptomail.in/v1.1/email'  # Updated to .in domain
    headers = {
        'accept': 'application/json',
        'content-type': 'application/json',
        'authorization': f'Zoho-enczapikey {config["api_key"]}'
    }
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Convert frame data to base64 if provided
    attachments = []
    if frame_data is not None:
        # Convert numpy array to PIL Image
        img = Image.fromarray(cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB))
        # Save to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        # Encode to base64
        import base64
        img_base64 = base64.b64encode(img_byte_arr).decode()
        
        attachments = [{
            'name': f'crowd_alert_{current_time.replace(" ", "_")}.jpg',
            'content': img_base64,
            'mime_type': 'image/jpeg'
        }]
    
    data = {
        'from': {
            'address': config['from_email']
        },
        'to': [{
            'email_address': {
                'address': config['to_email'],
                'name': 'Alert Recipient'
            }
        }],
        'subject': 'Crowd Density Alert!',
        'htmlbody': f'''
        <div><b>⚠️ Crowd Density Alert</b>
        <p>The current number of people ({people_count}) has exceeded the maximum allowed limit ({max_allowed}).</p>
        <p>Time: {current_time}</p>
        <p>Please take necessary action.</p>
        <p>See attached image for the current crowd situation.</p></div>
        ''',
        'attachments': attachments
    }
    
    try:
        response = requests.request('POST', url, data=json.dumps(data), headers=headers)
        print(f'ZeptoMail Response: {response.text}')
        if response.status_code == 200:
            print(f'Alert email sent successfully at {current_time}')
        else:
            print(f'Failed to send alert email: {response.text}')
    except Exception as e:
        print(f'Error sending alert email: {str(e)}')

# Initialize Flask App
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class CrowdLimit(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    max_people = db.Column(db.Integer, nullable=False, default=10)

class EmailConfig(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    api_key = db.Column(db.String(200), nullable=False)
    from_email = db.Column(db.String(100), nullable=False)
    to_email = db.Column(db.String(100), nullable=False)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

# Create tables
with app.app_context():
    db.create_all()

# Home Route
@app.route("/")
def home():
    return render_template("home.html")

# Signup Route
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash("Account created! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html", form={})  # 👈 Fix: Passing an empty form dictionary


# Login Route
@app.route("/login", methods=["GET", "POST"])
def login():
    # Clear any existing flash messages
    session.pop('_flashes', None)
    
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, password):
            session["user_id"] = user.id
            session["username"] = user.username
            flash("Logged in successfully!", "success")
            return redirect(url_for("admin"))
        else:
            flash("Login failed. Check email & password.", "danger")

    return render_template("login.html", form={})  # 👈 Fix: Passing an empty form dictionary


# Logout Route
@app.route("/logout")
def logout():
    # Clear any existing flash messages
    session.pop('_flashes', None)
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))

# Admin Page to Set Max Allowed People
@app.route("/admin", methods=["GET", "POST"])
def admin():
    if "user_id" not in session:
        flash("You must be logged in to access this page!", "danger")
        return redirect(url_for("login"))

    limit_entry = CrowdLimit.query.first()
    
    if request.method == "POST":
        max_people = int(request.form["max_people"])
        
        if limit_entry:
            limit_entry.max_people = max_people
        else:
            db.session.add(CrowdLimit(max_people=max_people))

        db.session.commit()
        flash(f"Max allowed persons set to {max_people}", "success")

    return render_template("admin.html", limit=limit_entry, form={})  # 👈 Fix: Passing an empty form dictionary

# Live Camera Feed Processing
import io
from PIL import Image
from datetime import datetime, timedelta

# Initialize YOLO model globally for better performance
model = YOLO("yolov8n.pt")

# Variable to track last alert time
last_alert_time = None

# Cooldown period for alerts (1 minute)
ALERT_COOLDOWN = timedelta(minutes=1)

@app.route('/process_frame', methods=['POST'])
def process_frame():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400
        
    # Get the frame from the request
    frame_file = request.files['frame']
    
    # Convert to PIL Image
    img_stream = io.BytesIO(frame_file.read())
    img = Image.open(img_stream)
    
    # Convert PIL image to numpy array for YOLO
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Run YOLOv8 inference on the frame
    results = model(frame, conf=0.3, iou=0.4)  # Using the same confidence and IOU thresholds
    
    people_count = 0
    annotated_frame = frame.copy()
    
    # Process detections
    for r in results:
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            class_id = int(cls.item())
            
            if class_id == 0:  # Class 0 is 'person'
                people_count += 1
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Store count in session
    session['people_count'] = people_count
    
    # Get crowd limit
    with app.app_context():
        limit = CrowdLimit.query.first()
        max_allowed = limit.max_people if limit else 10
        
        # Send email alert if people count exceeds limit
        global last_alert_time
        current_time = datetime.now()
        
        if people_count > max_allowed:
            # Check if enough time has passed since the last alert
            if last_alert_time is None or (current_time - last_alert_time) >= ALERT_COOLDOWN:
                # Pass the current frame to the email function
                send_alert_email(people_count, max_allowed, frame)
                last_alert_time = current_time
                print(f'Alert sent at {current_time}. Next alert possible after {(current_time + ALERT_COOLDOWN).strftime("%Y-%m-%d %H:%M:%S")}')
    
    return jsonify({
        'people_count': people_count,
        'max_allowed': max_allowed
    })

@app.route('/email_settings', methods=['GET', 'POST'])
def email_settings():
    if "user_id" not in session:
        flash("You must be logged in to access this page!", "danger")
        return redirect(url_for("login"))
    
    user = User.query.get(session["user_id"])
    config = EmailConfig.query.first()
    
    if request.method == 'POST':
        api_key = request.form['api_key']
        from_email = request.form['from_email']
        
        if config:
            config.api_key = api_key
            config.from_email = from_email
            config.to_email = user.email  # Use logged-in user's email
            config.last_updated = datetime.utcnow()
        else:
            config = EmailConfig(
                api_key=api_key,
                from_email=from_email,
                to_email=user.email  # Use logged-in user's email
            )
            db.session.add(config)
        
        try:
            db.session.commit()
            flash('Email settings updated successfully!', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating settings: {str(e)}', 'danger')
    
    return render_template('email_settings.html', config=config, user=user)

@app.route('/test_email', methods=['POST'])
def test_email():
    if "user_id" not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    try:
        # Send a test email with dummy data
        send_alert_email(5, 3)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/camera')
def camera():
    if "user_id" not in session:
        flash("You must be logged in to access this page!", "danger")
        return redirect(url_for("login"))
    
    with app.app_context():
        limit = CrowdLimit.query.first()
    return render_template("camera.html", limit=limit)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=3000)
