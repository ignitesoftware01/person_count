import os
import time
import sqlite3
import threading
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for, session, jsonify, flash
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO

# ---------------- Config ----------------
APP_SECRET = os.environ.get("FLASK_SECRET", "change_this_secret")
DB_PATH = "users.db"
MODEL_PATH = os.environ.get("YOLO_MODEL", "yolov8n.pt")
IMG_SIZE = int(os.environ.get("IMG_SIZE", "640"))
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 5000))
# ----------------------------------------

app = Flask(__name__)
app.secret_key = APP_SECRET

# Shared state
latest_frame_jpg = None
latest_person_count = 0
frame_lock = threading.Lock()
running = True

# ---------------- Database helpers & migration ----------------
def init_db_and_migrate():
    need_create = False
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    if cur.fetchone() is None:
        need_create = True
    else:
        cur.execute("PRAGMA table_info(users)")
        cols = [r[1] for r in cur.fetchall()]
        if "email" not in cols:
            if "username" in cols:
                try:
                    conn.execute("ALTER TABLE users RENAME TO old_users")
                    conn.execute("""
                        CREATE TABLE users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            email TEXT UNIQUE NOT NULL,
                            password TEXT NOT NULL
                        )
                    """)
                    conn.execute("INSERT OR IGNORE INTO users (email, password) SELECT username, password FROM old_users")
                    conn.commit()
                    print("Migrated old users.username -> users.email")
                except Exception as e:
                    print("Migration failed:", e)
                    conn.execute("DROP TABLE IF EXISTS old_users")
                    need_create = True
            else:
                need_create = True

    if need_create:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)
        conn.commit()

    conn.close()

# Initialize DB
init_db_and_migrate()

# ---------------- YOLO model load ----------------
print("Loading YOLO model:", MODEL_PATH)
model = YOLO(MODEL_PATH)

# ---------------- Camera auto-detect ----------------
def find_working_camera_index(max_index=5):
    backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)
    backends.append(cv2.CAP_ANY)

    for i in range(max_index):
        for b in backends:
            cap = cv2.VideoCapture(i, b)
            if not cap.isOpened():
                cap.release()
                continue
            ret, frame = cap.read()
            cap.release()
            if ret:
                return i, b
    return None, None

camera_index, camera_backend = find_working_camera_index(5)
if camera_index is None:
    raise RuntimeError("No working camera found.")
print(f"Using camera index {camera_index} backend {camera_backend}")

# ---------------- PERSON-ONLY SAFE INFERENCE ----------------
def safe_infer_and_annotate(frame):
    try:
        # 👇 FORCE YOLO TO DETECT ONLY PERSON (class 0)
        results = model(frame, imgsz=IMG_SIZE, classes=[0])
        r = results[0]
    except:
        return None, 0

    boxes = r.boxes
    if boxes is None:
        return frame, 0

    cls_vals = boxes.cls.cpu().numpy().astype(int)
    xyxy_vals = boxes.xyxy.cpu().numpy()
    person_count = 0
    annotated = frame.copy()

    # Draw ONLY PERSONS
    for box, cls in zip(xyxy_vals, cls_vals):
        if cls == 0:
            x1, y1, x2, y2 = box.astype(int)
            person_count += 1
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, "Person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(annotated, f"Persons: {person_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    return annotated, person_count

# ---------------- Camera capture thread ----------------
def capture_loop():
    global latest_frame_jpg, latest_person_count, running
    cap = cv2.VideoCapture(camera_index, camera_backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        annotated_bgr, cnt = safe_infer_and_annotate(frame)
        if annotated_bgr is None:
            continue

        success, jpeg = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not success:
            continue

        with frame_lock:
            latest_frame_jpg = jpeg.tobytes()
            latest_person_count = int(cnt)

        time.sleep(0.02)

    cap.release()

t = threading.Thread(target=capture_loop, daemon=True)
t.start()

# ---------------- Helpers ----------------
def init_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def login_required(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_email" not in session:
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return wrapper

# ---------------- Routes ----------------
@app.route("/")
@login_required
def dashboard():
    return render_template("dashboard.html")

@app.route("/video")
@login_required
def video_feed():
    def generator():
        while True:
            with frame_lock:
                frame = latest_frame_jpg
            if frame is None:
                time.sleep(0.05)
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.02)

    return Response(generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/count")
@login_required
def count():
    with frame_lock:
        cnt = latest_person_count
    return jsonify({"persons": cnt, "timestamp": int(time.time())})

@app.route("/snapshot")
@login_required
def snapshot():
    with frame_lock:
        frame = latest_frame_jpg
    if frame is None:
        return ("", 503)
    return Response(frame, mimetype="image/jpeg")

# ------------- Auth routes -------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email", "").lower()
        pwd = request.form.get("password", "")
        if not email or not pwd:
            flash("Email and password required.", "error")
            return redirect(url_for("register"))

        conn = init_db_connection()
        try:
            conn.execute("INSERT INTO users (email, password) VALUES (?, ?)",
                         (email, generate_password_hash(pwd)))
            conn.commit()
            flash("Registered — log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already used.", "error")
        finally:
            conn.close()

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").lower()
        pwd = request.form.get("password", "")
        conn = init_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email=?", (email,))
        row = cur.fetchone()
        conn.close()

        if row and check_password_hash(row["password"], pwd):
            session["user_email"] = email
            return redirect(url_for("dashboard"))

        flash("Invalid email or password.", "error")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

def shutdown():
    global running
    running = False
    t.join(timeout=2)

if __name__ == "__main__":
    try:
        print(f"Running on http://{HOST}:{PORT}")
        app.run(host=HOST, port=PORT, threaded=True)
    except KeyboardInterrupt:
        shutdown()
