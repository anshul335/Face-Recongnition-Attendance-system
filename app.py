import os
import json
import csv
import base64
import numpy as np
import cv2 as cv
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

# get embedding from face crop using our resnet + spatial structure
def get_face_embedding(face_crop_bgr):
    # CNN embedding from our ResNet
    rgb = cv.cvtColor(face_crop_bgr, cv.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    x = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        cnn_emb = model(x).cpu().squeeze(0).numpy()

    # Spatial normalized intensity grid for facial geometry
    gray = cv.cvtColor(face_crop_bgr, cv.COLOR_BGR2GRAY)
    gray_resized = cv.equalizeHist(cv.resize(gray, (8, 16))).flatten().astype(np.float32)
    gray_norm = (gray_resized - np.mean(gray_resized)) / (np.std(gray_resized) + 1e-6)
    gray_norm = gray_norm / np.linalg.norm(gray_norm)

    # Combined descriptor
    comb = np.concatenate([cnn_emb * 0.6, gray_norm * 0.4])
    return comb / np.linalg.norm(comb)

# detect face region and extract embedding
def detect_and_encode(image_bytes):
    try:
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv.imdecode(arr, cv.IMREAD_COLOR)
        if img is None:
            return [], []

        h, w = img.shape[:2]
        R = img[:, :, 2].astype(int)
        G = img[:, :, 1].astype(int)
        B = img[:, :, 0].astype(int)

        # skin color thresholding
        cond = (R > 95) & (G > 40) & (B > 20) & \
               ((np.maximum.reduce([R, G, B]) - np.minimum.reduce([R, G, B])) > 15) & \
               (np.abs(R - G) > 12) & (R > G) & (R > B)

        mask = cond.astype(np.uint8) * 255
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
        clean = cv.morphologyEx(cv.morphologyEx(mask, cv.MORPH_OPEN, kernel), cv.MORPH_CLOSE, kernel)
        cnts, _ = cv.findContours(clean, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # rank face candidates by area and proximity to frame center
        best_box = None
        best_score = -1
        for c in cnts:
            area = cv.contourArea(c)
            if area < 2500:
                continue
            x, y, cw, ch = cv.boundingRect(c)
            ar = cw / float(ch)
            if ar < 0.4 or ar > 2.5:
                continue
            cx = x + cw / 2.0
            cy = y + ch / 2.0
            dist_to_center = np.sqrt((cx - w / 2.0) ** 2 + (cy - h / 2.0) ** 2)
            score = area / (1.0 + 0.005 * dist_to_center)
            if score > best_score:
                best_score = score
                best_box = (x, y, cw, ch)

        if best_box is not None:
            x, y, cw, ch = best_box
            pad_x = int(cw * 0.15)
            pad_y = int(ch * 0.15)
            x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
            x1, y1 = min(w, x + cw + pad_x), min(h, y + ch + pad_y)
            crop = img[y0:y1, x0:x1]
            box = (y0, x1, y1, x0)
        else:
            center_crop = img[h // 6: 5 * h // 6, w // 6: 5 * w // 6]
            if np.std(center_crop) < 15:
                return [], []
            crop = center_crop
            box = (h // 6, 5 * w // 6, 5 * h // 6, w // 6)

        emb = get_face_embedding(crop)
        return [emb], [box]
    except Exception as e:
        print("Detection error:", e)
        return [], []

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
        roll = s['roll_number']
        npy_path = os.path.join(known_faces_dir, f"{roll}.npy")
        img_path = os.path.join(known_faces_dir, f"{roll}.jpg")

        emb = None
        # if image exists, re-encode to match current model embedding
        if os.path.exists(img_path):
            try:
                with open(img_path, 'rb') as f:
                    img_bytes = f.read()
                embs, _ = detect_and_encode(img_bytes)
                if embs:
                    emb = embs[0]
                    np.save(npy_path, emb)
            except Exception as e:
                print(f"Error re-encoding {roll}:", e)

        if emb is None and os.path.exists(npy_path):
            try:
                emb = np.load(npy_path)
            except:
                pass

        if emb is not None:
            known_encodings.append(emb)
            known_students.append(s)

load_known_faces()

# compare embedding with registered students
def match_face(candidate, threshold=0.65):
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
                            if len(row) >= 3 and row[1].strip() == roll_number and row[2].strip() == date_str:
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
        date_filter = request.args.get('date', '').strip()
        if not date_filter:
            date_filter = datetime.now().strftime('%Y-%m-%d')

        records = []
        try:
            with open(attendance_file, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 4:
                        if not date_filter or row[2].strip() == date_filter:
                            records.append({
                                'name': row[0].strip(),
                                'roll_number': row[1].strip(),
                                'date': row[2].strip(),
                                'time': row[3].strip()
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
