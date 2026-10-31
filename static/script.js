// global variables
let registerStream = null;
let attendanceStream = null;
let recognitionInterval = null;
let isRecognizing = false;
let photoCaptured = false;

// DOM elements
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

let els = {};

function cacheDOMElements() {
    els = {
        tabBtns: $$('.tab-btn'),
        registerSection: $('#register-section'),
        attendanceSection: $('#attendance-section'),
        recordsSection: $('#records-section'),
        registerVideo: $('#register-video'),
        registerCanvas: $('#register-canvas'),
        registerOverlay: $('#register-overlay'),
        captureBtn: $('#capture-btn'),
        capturedPreview: $('#captured-preview'),
        retakeBtn: $('#retake-btn'),
        studentName: $('#student-name'),
        rollNumber: $('#roll-number'),
        registerBtn: $('#register-btn'),
        registerStatus: $('#register-status'),
        attendanceVideo: $('#attendance-video'),
        attendanceCanvas: $('#attendance-canvas'),
        attendanceOverlay: $('#attendance-overlay'),
        startRecBtn: $('#start-recognition-btn'),
        stopRecBtn: $('#stop-recognition-btn'),
        scanningIndicator: $('#scanning-indicator'),
        recognitionResults: $('#recognition-results'),
        datePicker: $('#date-picker'),
        attendanceTbody: $('#attendance-tbody'),
        noRecords: $('#no-records'),
        attendanceTable: $('#attendance-table'),
        totalStudents: $('#total-students'),
        attendanceCount: $('#attendance-count'),
        studentsGrid: $('#students-grid'),
        noStudents: $('#no-students'),
        toastContainer: $('#toast-container'),
    };
}

// initialization
document.addEventListener('DOMContentLoaded', () => {
    cacheDOMElements();
    initTabs();
    initRegisterCamera();
    initDatePicker();
    initEventListeners();
    loadAttendanceRecords();
    loadStudents();
});

function initEventListeners() {
    els.captureBtn.addEventListener('click', capturePhoto);
    els.retakeBtn.addEventListener('click', retakePhoto);

    els.studentName.addEventListener('input', validateRegisterForm);
    els.rollNumber.addEventListener('input', validateRegisterForm);

    els.registerBtn.addEventListener('click', registerStudent);
    els.startRecBtn.addEventListener('click', startRecognition);
    els.stopRecBtn.addEventListener('click', stopRecognition);

    els.datePicker.addEventListener('change', loadAttendanceRecords);
}

// tab navigation
function initTabs() {
    els.tabBtns.forEach((btn) => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });
}

window.switchTab = switchTab;
window.deleteStudent = deleteStudent;
window.downloadCSV = downloadCSV;
window.loadAttendanceRecords = loadAttendanceRecords;

function switchTab(tabName) {
    els.tabBtns.forEach((btn) => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });

    const sections = { register: els.registerSection, attendance: els.attendanceSection, records: els.recordsSection };
    Object.values(sections).forEach((s) => {
        s.style.display = 'none';
        s.classList.remove('active');
    });

    const active = sections[tabName];
    if (active) {
        active.style.display = '';
        active.classList.add('active');
    }

    if (tabName === 'register') {
        initRegisterCamera();
        stopAttendanceCamera();
    } else if (tabName === 'attendance') {
        startAttendanceCamera();
        stopRegisterCamera();
    } else {
        stopRegisterCamera();
        stopAttendanceCamera();
        if (isRecognizing) stopRecognition();
        loadAttendanceRecords();
        loadStudents();
    }
}

// camera controls
function initRegisterCamera() {
    if (registerStream) return;
    navigator.mediaDevices
        .getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false })
        .then((stream) => {
            registerStream = stream;
            els.registerVideo.srcObject = stream;
            els.registerVideo.onloadedmetadata = () => {
                els.registerOverlay.classList.add('hidden');
            };
        })
        .catch((err) => {
            console.error('Camera error:', err);
            showToast('Camera access denied', 'error');
        });
}

function stopRegisterCamera() {
    if (registerStream) {
        registerStream.getTracks().forEach((t) => t.stop());
        registerStream = null;
        els.registerVideo.srcObject = null;
        els.registerOverlay.classList.remove('hidden');
    }
}

function startAttendanceCamera() {
    if (attendanceStream) return;
    navigator.mediaDevices
        .getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false })
        .then((stream) => {
            attendanceStream = stream;
            els.attendanceVideo.srcObject = stream;
            els.attendanceVideo.onloadedmetadata = () => {
                els.attendanceOverlay.classList.add('hidden');
            };
        })
        .catch((err) => {
            console.error('Camera error:', err);
            showToast('Camera access denied', 'error');
        });
}

function stopAttendanceCamera() {
    if (isRecognizing) stopRecognition();
    if (attendanceStream) {
        attendanceStream.getTracks().forEach((t) => t.stop());
        attendanceStream = null;
        els.attendanceVideo.srcObject = null;
        els.attendanceOverlay.classList.remove('hidden');
    }
}

// capture and retake
function capturePhoto() {
    const video = els.registerVideo;
    const canvas = els.registerCanvas;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const dataURL = canvas.toDataURL('image/jpeg', 0.9);

    els.capturedPreview.src = dataURL;
    els.capturedPreview.style.display = 'block';
    els.registerVideo.parentElement.style.display = 'none';
    els.captureBtn.parentElement.style.display = 'none';
    els.retakeBtn.style.display = '';
    photoCaptured = true;
    validateRegisterForm();
}

function retakePhoto() {
    els.capturedPreview.style.display = 'none';
    els.capturedPreview.src = '';
    els.registerVideo.parentElement.style.display = '';
    els.captureBtn.parentElement.style.display = '';
    els.retakeBtn.style.display = 'none';
    photoCaptured = false;
    validateRegisterForm();
}

function validateRegisterForm() {
    const nameOk = els.studentName.value.trim().length > 0;
    const rollOk = els.rollNumber.value.trim().length > 0;
    els.registerBtn.disabled = !(nameOk && rollOk && photoCaptured);
}

// register student
async function registerStudent() {
    const name = els.studentName.value.trim();
    const roll_number = els.rollNumber.value.trim();
    const image = els.capturedPreview.src;

    if (!name || !roll_number || !image) {
        showToast('Please fill all fields and capture photo', 'warning');
        return;
    }

    els.registerBtn.disabled = true;
    const origText = els.registerBtn.innerHTML;
    els.registerBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Registering...';

    try {
        const res = await fetch('/api/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, roll_number, image }),
        });
        const data = await res.json();

        if (data.success) {
            showToast(data.message || 'Student registered!', 'success');
            els.studentName.value = '';
            els.rollNumber.value = '';
            retakePhoto();
        } else {
            showToast(data.message || 'Registration failed', 'error');
        }
    } catch (err) {
        console.error(err);
        showToast('Server error, please try again', 'error');
    } finally {
        els.registerBtn.innerHTML = origText;
        els.registerBtn.disabled = true;
    }
}

// recognition loop
function startRecognition() {
    if (!attendanceStream) {
        startAttendanceCamera();
        setTimeout(() => _beginRecognitionLoop(), 1000);
    } else {
        _beginRecognitionLoop();
    }
}

function _beginRecognitionLoop() {
    isRecognizing = true;
    els.startRecBtn.style.display = 'none';
    els.stopRecBtn.style.display = '';
    els.scanningIndicator.style.display = '';
    els.recognitionResults.innerHTML = '';
    recognitionInterval = setInterval(recognizeFrame, 2000);
    recognizeFrame();
}

function stopRecognition() {
    if (recognitionInterval) {
        clearInterval(recognitionInterval);
        recognitionInterval = null;
    }
    isRecognizing = false;
    els.startRecBtn.style.display = '';
    els.stopRecBtn.style.display = 'none';
    els.scanningIndicator.style.display = 'none';
}

async function recognizeFrame() {
    if (!isRecognizing || !attendanceStream) return;

    const video = els.attendanceVideo;
    const canvas = els.attendanceCanvas;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const dataURL = canvas.toDataURL('image/jpeg', 0.85);

    try {
        const res = await fetch('/api/recognize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataURL }),
        });
        const data = await res.json();

        if (data.success && data.recognized && data.recognized.length > 0) {
            displayRecognitionResult(data.recognized);
        } else if (data.success) {
            els.recognitionResults.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-face-meh"></i>
                    <p>No face detected</p>
                </div>`;
        } else {
            els.recognitionResults.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-exclamation-circle"></i>
                    <p>${data.message || 'Recognition error'}</p>
                </div>`;
        }
    } catch (err) {
        console.error(err);
        els.recognitionResults.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-exclamation-triangle"></i>
                <p>Connection error. Retrying...</p>
            </div>`;
    }
}

function displayRecognitionResult(recognized) {
    let html = '';
    recognized.forEach((person) => {
        if (person.name === 'Unknown' || !person.name) {
            html += `
                <div class="result-card unknown">
                    <div class="result-info">
                        <div class="result-name">Unknown Person</div>
                        <div class="result-roll">Face not recognized</div>
                    </div>
                    <span class="badge badge-danger">Not Registered</span>
                </div>`;
        } else {
            const isAlready = person.already_marked;
            const cardClass = isAlready ? 'already' : '';
            const badgeClass = isAlready ? 'badge-warning' : 'badge-success';
            const badgeText = isAlready ? 'Already Marked Today' : '✓ Attendance Marked';
            html += `
                <div class="result-card ${cardClass}">
                    <div class="result-info">
                        <div class="result-name">${escapeHtml(person.name)}</div>
                        <div class="result-roll">Roll: ${escapeHtml(person.roll_number)}</div>
                        <div class="result-time">Time: ${escapeHtml(person.time || '')}</div>
                    </div>
                    <span class="badge ${badgeClass}">${badgeText}</span>
                </div>`;
        }
    });
    els.recognitionResults.innerHTML = html;
}

// records and students
function initDatePicker() {
    const today = new Date().toISOString().slice(0, 10);
    els.datePicker.value = today;
}

async function loadAttendanceRecords() {
    const date = els.datePicker.value;
    if (!date) return;

    try {
        const res = await fetch(`/api/attendance?date=${date}`);
        const data = await res.json();

        if (data.success && data.records && data.records.length > 0) {
            els.noRecords.style.display = 'none';
            els.attendanceTable.style.display = '';
            els.attendanceCount.textContent = data.records.length;

            let rows = '';
            data.records.forEach((rec, i) => {
                const name = rec.name || rec.Name || '';
                const roll = rec.roll_number || rec['Roll Number'] || '';
                const recDate = rec.date || rec.Date || date;
                const recTime = rec.time || rec.Time || '';
                rows += `<tr>
                    <td>${i + 1}</td>
                    <td>${escapeHtml(name)}</td>
                    <td>${escapeHtml(roll)}</td>
                    <td>${escapeHtml(recDate)}</td>
                    <td>${escapeHtml(recTime)}</td>
                </tr>`;
            });
            els.attendanceTbody.innerHTML = rows;
        } else {
            els.attendanceTable.style.display = 'none';
            els.noRecords.style.display = '';
            els.attendanceTbody.innerHTML = '';
            els.attendanceCount.textContent = '0';
        }
    } catch (err) {
        console.error(err);
        showToast('Failed to load attendance records', 'error');
    }
}

async function loadStudents() {
    try {
        const res = await fetch('/api/students');
        const data = await res.json();

        if (data.success && data.students && data.students.length > 0) {
            els.totalStudents.textContent = data.count || data.students.length;
            let html = '';
            data.students.forEach((s) => {
                html += `
                    <div class="student-card">
                        <button class="delete-student-btn" title="Delete student" onclick="deleteStudent('${escapeAttr(s.roll_number)}', '${escapeAttr(s.name)}')">
                            <i class="fas fa-trash"></i>
                        </button>
                        <div class="student-name">${escapeHtml(s.name)}</div>
                        <div class="student-roll">Roll: ${escapeHtml(s.roll_number)}</div>
                    </div>`;
            });
            els.studentsGrid.innerHTML = html;
        } else {
            els.totalStudents.textContent = '0';
            els.studentsGrid.innerHTML = `
                <div class="empty-state" style="grid-column: 1/-1;">
                    <i class="fas fa-user-slash"></i>
                    <p>No students registered yet</p>
                </div>`;
        }
    } catch (err) {
        console.error(err);
        showToast('Failed to load students', 'error');
    }
}

async function deleteStudent(rollNumber, name) {
    if (!confirm(`Are you sure you want to delete ${name} (${rollNumber})?`)) return;

    try {
        const res = await fetch(`/api/students/${encodeURIComponent(rollNumber)}`, { method: 'DELETE' });
        const data = await res.json();

        if (data.success) {
            showToast(data.message || 'Student deleted', 'success');
            loadStudents();
        } else {
            showToast(data.message || 'Failed to delete student', 'error');
        }
    } catch (err) {
        console.error(err);
        showToast('Server error, please try again', 'error');
    }
}

function downloadCSV() {
    const rows = els.attendanceTbody.querySelectorAll('tr');
    if (!rows.length) {
        showToast('No records to download', 'warning');
        return;
    }

    let csv = 'S.No,Name,Roll Number,Date,Time\n';
    rows.forEach((row) => {
        const cells = row.querySelectorAll('td');
        const line = Array.from(cells)
            .map((c) => `"${c.textContent.replace(/"/g, '""')}"`)
            .join(',');
        csv += line + '\n';
    });

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `attendance_${els.datePicker.value}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast('CSV downloaded', 'success');
}

// toast message
function showToast(message, type = 'info') {
    const icons = {
        success: 'fa-circle-check',
        error: 'fa-circle-xmark',
        warning: 'fa-triangle-exclamation',
        info: 'fa-circle-info',
    };

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `<i class="fas ${icons[type] || icons.info}"></i> <span>${escapeHtml(message)}</span>`;
    els.toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('removing');
        toast.addEventListener('animationend', () => toast.remove());
    }, 4000);
}

// helpers
function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function escapeAttr(str) {
    if (!str) return '';
    return str.replace(/'/g, "\\'").replace(/"/g, '&quot;');
}
