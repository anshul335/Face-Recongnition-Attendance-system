import os
import json
import csv
import base64
import numpy as np
from datetime import datetime
from io import BytesIO

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image

import torch
import torchvision.transforms as transforms
from resnet_face import get_face_resnet, myResnet20

# load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_face_resnet(embedding_dim=128, device=device)

# transforms for face image
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

app = Flask(__name__)
CORS(app)

# paths
data_dir = 'data'
known_faces_dir = 'known_faces'
students_file = os.path.join(data_dir, 'students.json')
attendance_file = os.path.join(data_dir, 'attendance.csv')

os.makedirs(data_dir, exist_ok=True)
os.makedirs(known_faces_dir, exist_ok=True)

# create files if not exists
if not os.path.exists(students_file):
    with open(students_file, 'w') as f:
        json.dump([], f)

if not os.path.exists(attendance_file):
    with open(attendance_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Roll Number', 'Date', 'Time'])

known_encodings = []
known_students = []

# load registered students into memory
def load_known_faces():
    global known_encodings, known_students
    known_encodings = []
    known_students = []

    try:
        with open(students_file, 'r') as f:
            students = json.load(f)
    except:
        students = []

    for s in students:
        path = os.path.join(known_faces_dir, f"{s['roll_number']}.npy")
        if os.path.exists(path):
            try:
                emb = np.load(path)
                known_encodings.append(emb)
                known_students.append(s)
            except:
                pass

load_known_faces()

# get embedding from face crop using our resnet
def get_face_embedding(face_img):
    x = transform(face_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    return out.cpu().squeeze(0).numpy()

# detect face region and extract embedding
def detect_and_encode(image_bytes):
    try:
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        w, h = img.size
        arr = np.array(img, dtype=np.float32)

        # skin color thresholding in YCbCr
        R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
        Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B

        skin = (Cb >= 75) & (Cb <= 130) & (Cr >= 130) & (Cr <= 180) & (Y >= 35)
        skin_count = np.sum(skin)

        # check if face is present
        if skin_count < (w * h * 0.015):
            return [], []

        ys, xs = np.where(skin)
        min_x, max_x = max(0, int(np.min(xs) - 15)), min(w, int(np.max(xs) + 15))
        min_y, max_y = max(0, int(np.min(ys) - 15)), min(h, int(np.max(ys) + 15))

        if (max_x - min_x) < 30 or (max_y - min_y) < 30:
            return [], []

        cropped = img.crop((min_x, min_y, max_x, max_y))
        emb = get_face_embedding(cropped)
        return [emb], [(min_y, max_x, max_y, min_x)]
    except Exception as e:
        print("Detection error:", e)
        return [], []

# compare embedding with registered students
def match_face(candidate, threshold=0.30):
    if len(known_encodings) == 0:
        return -1, 999.0
    dists = [float(np.linalg.norm(k - candidate)) for k in known_encodings]
    min_idx = int(np.argmin(dists))
    min_dist = dists[min_idx]
    if min_dist <= threshold:
        return min_idx, min_dist
    return -1, min_dist


# routes
@app.route('/')
def index():
    return render_template('index.html')

# register new student
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        if not data or not all(k in data for k in ('name', 'roll_number', 'image')):
            return jsonify({'success': False, 'message': 'Please fill all fields'}), 400

        name = data['name'].strip()
        roll_number = data['roll_number'].strip()
        img_b64 = data['image']

        if ',' in img_b64:
            img_b64 = img_b64.split(',', 1)[1]

        image_bytes = base64.b64decode(img_b64)
        encodings, _ = detect_and_encode(image_bytes)

        if len(encodings) == 0:
            return jsonify({'success': False, 'message': 'No face detected. Please face the camera properly.'}), 400

        # check duplicate roll number
        with open(students_file, 'r') as f:
            students = json.load(f)

        for s in students:
            if s['roll_number'] == roll_number:
                return jsonify({'success': False, 'message': f'Roll {roll_number} already registered.'}), 409

        # save embedding and image
        np.save(os.path.join(known_faces_dir, f"{roll_number}.npy"), encodings[0])
        pil_img = Image.open(BytesIO(image_bytes))
        pil_img.save(os.path.join(known_faces_dir, f"{roll_number}.jpg"), 'JPEG')

        students.append({'name': name, 'roll_number': roll_number})
        with open(students_file, 'w') as f:
            json.dump(students, f, indent=2)

        load_known_faces()
        return jsonify({'success': True, 'message': f'{name} (Roll: {roll_number}) registered successfully!'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# mark attendance via recognition
@app.route('/api/recognize', methods=['POST'])
def recognize():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'No image provided'}), 400

        img_b64 = data['image']
        if ',' in img_b64:
            img_b64 = img_b64.split(',', 1)[1]

        image_bytes = base64.b64decode(img_b64)
        encodings, _ = detect_and_encode(image_bytes)

        if len(encodings) == 0:
            return jsonify({'success': True, 'recognized': []})

        recognized = []
        for emb in encodings:
            idx, dist = match_face(emb)
            if idx != -1:
                student = known_students[idx]
                name = student['name']
                roll_number = student['roll_number']

                now = datetime.now()
                date_str = now.strftime('%Y-%m-%d')
                time_str = now.strftime('%H:%M:%S')

                # check if already marked today
                already_marked = False
                try:
                    with open(attendance_file, 'r', newline='') as f:
                        reader = csv.reader(f)
                        next(reader, None)
                        for row in reader:
                            if len(row) >= 3 and row[1] == roll_number and row[2] == date_str:
                                already_marked = True
                                break
                except:
                    pass

                # update spreadsheet
                if not already_marked:
                    with open(attendance_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([name, roll_number, date_str, time_str])

                recognized.append({
                    'name': name,
                    'roll_number': roll_number,
                    'time': time_str,
                    'already_marked': already_marked
                })
            else:
                recognized.append({'name': 'Unknown', 'roll_number': 'N/A'})

        return jsonify({'success': True, 'recognized': recognized})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# get attendance records
@app.route('/api/attendance', methods=['GET'])
def get_attendance():
    try:
        date_filter = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
        records = []
        try:
            with open(attendance_file, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 4 and row[2] == date_filter:
                        records.append({
                            'name': row[0],
                            'roll_number': row[1],
                            'date': row[2],
                            'time': row[3]
                        })
        except:
            pass

        return jsonify({'success': True, 'records': records, 'date': date_filter})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# download attendance csv
@app.route('/api/attendance/download', methods=['GET'])
def download_attendance():
    try:
        if os.path.exists(attendance_file):
            return send_file(attendance_file, mimetype='text/csv', as_attachment=True, download_name='attendance.csv')
        return jsonify({'success': False, 'message': 'File not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# get all students
@app.route('/api/students', methods=['GET'])
def get_students():
    try:
        with open(students_file, 'r') as f:
            students = json.load(f)
        return jsonify({'success': True, 'students': students, 'count': len(students)})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# delete student
@app.route('/api/students/<roll_number>', methods=['DELETE'])
def delete_student(roll_number):
    try:
        with open(students_file, 'r') as f:
            students = json.load(f)

        updated = [s for s in students if s['roll_number'] != roll_number]
        if len(updated) == len(students):
            return jsonify({'success': False, 'message': 'Student not found'}), 404

        with open(students_file, 'w') as f:
            json.dump(updated, f, indent=2)

        npy_path = os.path.join(known_faces_dir, f"{roll_number}.npy")
        jpg_path = os.path.join(known_faces_dir, f"{roll_number}.jpg")
        if os.path.exists(npy_path):
            os.remove(npy_path)
        if os.path.exists(jpg_path):
            os.remove(jpg_path)

        load_known_faces()
        return jsonify({'success': True, 'message': f'Student {roll_number} deleted'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


if __name__ == '__main__':
    print(f"Server starting on {device}...")
    app.run(debug=True, host='0.0.0.0', port=5000)
