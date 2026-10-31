# Face Recognition Attendance System using Deep ResNet

A Computer Vision project that automates student attendance tracking using a custom Deep Residual Convolutional Neural Network (ResNet) built with PyTorch, an interactive web dashboard, and real-time CSV spreadsheet logging.

---

## Project Overview

Traditional attendance systems rely on manual roll calls, paper sign-sheets, or ID cards, which are slow and susceptible to proxy attendance. This project implements a biometric solution:

1. **Student Registration:** Captures the student's face via webcam, extracts facial feature embeddings using a custom ResNet model, and saves their profile.
2. **Real-time Recognition & Attendance:** Scans students in front of the camera, matches their embeddings against registered faces using Euclidean distance, and automatically logs their attendance with date and time into a CSV file.
3. **Attendance Management:** Allows viewing attendance records by date, downloading the attendance spreadsheet, and managing registered students.

---

## Model Architecture (`myResnet20`)

The facial feature extractor is implemented from scratch in PyTorch inside `resnet_face.py`. It uses a 4-block deep Residual Network (ResNet-28 style) with identity and projection shortcuts:

- **Input:** RGB face crop normalized to `(3, 112, 112)`
- **Initial Conv:** `Conv2d(3, 16, kernel_size=3, padding=1)` + `BatchNorm2d` + `ReLU`
- **Block 1 (16 channels):** 3 residual stages with identity shortcuts
- **Block 2 (32 channels):** 3 residual stages with stride 2 downsampling on stage 1 and a $1 \times 1$ conv projection shortcut
- **Block 3 (64 channels):** 3 residual stages with stride 2 downsampling on stage 1 and a $1 \times 1$ conv projection shortcut
- **Block 4 (128 channels):** 3 residual stages with stride 2 downsampling on stage 1 and a $1 \times 1$ conv projection shortcut
- **Global Pooling:** `AdaptiveAvgPool2d((1, 1))` pooling spatial features
- **Dense Head:** `Linear(128, 128)` mapping features to metric space
- **L2 Hypersphere Normalization:** `F.normalize(x, p=2, dim=1)` projects embeddings onto the unit sphere so that $\|e\|_2 = 1.0$

### Why Metric Learning?
In an attendance system, new students register dynamically. Instead of retraining a softmax classifier every time someone registers, the ResNet maps faces into a 128-dimensional embedding space where images of the same person have small Euclidean distance ($d \approx 0.004$) and different people have large distance ($d > 0.30$).

---

## Project Structure

```
AttendanceSystem/
│
├── resnet_face.py          # Custom PyTorch ResNet CNN architecture
├── app.py                  # Flask backend and face matching logic
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
│
├── templates/
│   └── index.html          # Web dashboard (Register, Mark Attendance, Records)
│
├── static/
│   ├── style.css           # UI styling and layout
│   └── script.js           # Webcam handling, canvas snapshots, and API calls
│
├── data/
│   ├── students.json       # Database of registered students
│   └── attendance.csv      # Real-time attendance spreadsheet
│
└── known_faces/            # Saved face embeddings (.npy) and photos (.jpg)
```

---

## Requirements

The project uses standard Python libraries:

- Python 3.10+
- PyTorch & Torchvision
- Flask & Flask-CORS
- Pillow (PIL)
- NumPy
- torchinfo (for model summary)

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. View Model Architecture & Summary
To inspect the layer-by-layer ResNet architecture, output shapes, and parameter counts:
```bash
python resnet_face.py
```

### 2. Start the Attendance Web Application
To launch the full web interface:
```bash
python app.py
```

Open your browser and navigate to:
```
http://localhost:5000
```

---

## How to Use the System

1. **Register a Student:**
   - Go to the **Register Student** tab.
   - Look at the camera and click **Capture Photo**.
   - Enter your **Full Name** and **Roll Number**.
   - Click **Register Student**. The system extracts the ResNet embedding and saves it to `known_faces/<roll_number>.npy`.

2. **Mark Attendance:**
   - Go to the **Mark Attendance** tab.
   - Click **Start Recognition**.
   - When a registered face appears, the system recognizes the student, displays their name and roll number, and marks their attendance.
   - **Duplicate Protection:** If the student is already marked today, it flags them as *Already Marked Today* and avoids duplicate rows.
   - Unregistered faces are marked as *Unknown Person*.

3. **View & Export Records:**
   - Go to the **View Records** tab.
   - Filter attendance by date using the date picker.
   - Click **Download CSV** to export today's attendance spreadsheet.
   - View and delete registered students if needed.
