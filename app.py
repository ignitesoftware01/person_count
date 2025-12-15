# app.py
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
HISTORY_DB_PATH = "history.db" # 🔥 NEW: Separate DB for track history
MODEL_PATH = os.environ.get("YOLO_MODEL", "yolov8n.pt")
IMG_SIZE = int(os.environ.get("IMG_SIZE", "640"))
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 5000))

TRACKER_MAX_AGE = float(os.environ.get("TRACKER_MAX_AGE", 5.0))
ARROW_SCALE = float(os.environ.get("ARROW_SCALE", 0.08))
HISTORY_PRUNE_TIME = float(os.environ.get("HISTORY_PRUNE_TIME", 5.0))

ANOMALY_LOITERING_SPEED = float(os.environ.get("ANOMALY_LOITERING_SPEED", 5.0))
ANOMALY_LOITERING_DURATION = float(os.environ.get("ANOMALY_LOITERING_DURATION", 3.0))
ANOMALY_STAMPEDE_SPEED = float(os.environ.get("ANOMALY_STAMPEDE_SPEED", 150.0))
ANOMALY_CROWD_COUNT = int(os.environ.get("ANOMALY_CROWD_COUNT", 6))
ANOMALY_CONFLICT_RATIO = float(os.environ.get("ANOMALY_CONFLICT_RATIO", 0.25))
ANOMALY_DENSITY_THRESHOLD = int(os.environ.get("ANOMALY_DENSITY_THRESHOLD", 10))

# ---------------- App & shared state ----------------
app = Flask(__name__)
app.secret_key = APP_SECRET

latest_frame_jpg = None
latest_person_count = 0
latest_people_report = []
latest_crowd_conflict = False
frame_lock = threading.Lock()
running = True

_tracks = {}
_next_track_id = 1
_tracks_lock = threading.Lock()

# ---------------- Database ----------------
def init_db_and_migrate():
    # 1. Users DB Initialization (users.db)
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

    # 2. 🔥 History DB Initialization (history.db)
    h_conn = sqlite3.connect(HISTORY_DB_PATH)
    h_cur = h_conn.cursor()
    h_cur.execute("""
        CREATE TABLE IF NOT EXISTS tracks_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER NOT NULL,
            timestamp REAL NOT NULL,
            centroid_x INTEGER NOT NULL,
            centroid_y INTEGER NOT NULL,
            speed_px_s REAL,
            direction TEXT,
            anomaly_reason TEXT
        )
    """)
    h_conn.commit()
    h_conn.close()

init_db_and_migrate()

# 🔥 NEW: History Logging Function
def _log_track_data(track_id, ts, centroid_x, centroid_y, speed, direction, anomaly_reason):
    """Logs the track data into the history database."""
    try:
        conn = sqlite3.connect(HISTORY_DB_PATH)
        conn.execute("""
            INSERT INTO tracks_history (track_id, timestamp, centroid_x, centroid_y, speed_px_s, direction, anomaly_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            track_id, ts, centroid_x, centroid_y, speed, direction, anomaly_reason
        ))
        conn.commit()
    except Exception as e:
        print(f"Error logging track data for ID {track_id}: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()


# ---------------- Load YOLO ----------------
print("Loading YOLO model:", MODEL_PATH)
model = YOLO(MODEL_PATH)

# ---------------- Camera ----------------
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

# ---------------- Tracking helpers ----------------
def _next_id():
    global _next_track_id
    with _tracks_lock:
        tid = _next_track_id
        _next_track_id += 1
    return tid

def _add_track(track_id, pos, ts):
    with _tracks_lock:
        _tracks[track_id] = {"pos": pos, "last_seen": ts, "history": [(ts, pos)], "anomaly_reason": None}

def _update_track(track_id, pos, ts):
    global HISTORY_PRUNE_TIME
    with _tracks_lock:
        tr = _tracks.get(track_id)
        if tr is None:
            _tracks[track_id] = {"pos": pos, "last_seen": ts, "history": [(ts, pos)], "anomaly_reason": None}
            return
        tr["pos"] = pos
        tr["last_seen"] = ts
        tr["history"].append((ts, pos))
        cutoff = ts - HISTORY_PRUNE_TIME
        tr["history"] = [(t,p) for (t,p) in tr["history"] if t >= cutoff]

def _age_and_cleanup(max_age=TRACKER_MAX_AGE):
    now = time.time()
    remove = []
    with _tracks_lock:
        for tid, tr in list(_tracks.items()):
            if now - tr["last_seen"] > max_age:
                remove.append(tid)
        for tid in remove:
            del _tracks[tid]

def _match_and_assign_ids(detected_centroids):
    matches = []
    unmatched = []
    used_tids = set()
    if not detected_centroids:
        return matches, []

    with _tracks_lock:
        track_items = [(tid, tr["pos"]) for tid, tr in _tracks.items()]
    pairs = []
    for tid, tpos in track_items:
        for dpos in detected_centroids:
            dist = ((tpos[0]-dpos[0])**2 + (tpos[1]-dpos[1])**2)**0.5
            pairs.append((dist, tid, dpos))
    pairs.sort(key=lambda x: x[0])
    used_d = set()
    for dist, tid, dpos in pairs:
        if dist > 150:
            continue
        if tid in used_tids:
            continue
        if dpos in used_d:
            continue
        matches.append((tid, dpos))
        used_tids.add(tid)
        used_d.add(dpos)
    unmatched = [d for d in detected_centroids if d not in used_d]
    new_assigned = []
    for d in unmatched:
        nid = _next_id()
        _add_track(nid, d, time.time())
        new_assigned.append((nid, d))
    return matches + new_assigned, []

def _compute_speed_and_direction(history):
    if len(history) < 2:
        return 0.0, 0.0, 0.0, "STILL", []
    t0, p0 = history[0]
    t1, p1 = history[-1]
    dt = t1 - t0
    if dt <= 0.0001:
        return 0.0, 0.0, 0.0, "STILL", []

    vx = (p1[0] - p0[0]) / dt
    vy = (p1[1] - p0[1]) / dt
    speed = (vx**2 + vy**2)**0.5
    if abs(vx) < 2 and abs(vy) < 2:
        d = "STILL"
    else:
        if abs(vx) >= abs(vy):
            d = "RIGHT" if vx > 0 else "LEFT"
        else:
            d = "DOWN" if vy > 0 else "UP"

    recent_speeds = []
    for i in range(1, len(history)):
        t_prev, p_prev = history[i-1]
        t_curr, p_curr = history[i]
        dt_segment = t_curr - t_prev
        if dt_segment > 0.0001:
            dx = p_curr[0] - p_prev[0]
            dy = p_curr[1] - p_prev[1]
            recent_speeds.append((dx**2+dy**2)**0.5 / dt_segment)

    return vx, vy, speed, d, recent_speeds

# ---------------- Anomaly detection improvements ----------------
def _check_for_anomalies(track_id, speed, direction, history, people_report_count):
    anomaly_reason = None
    now = time.time()
    # Loitering detection
    if speed < ANOMALY_LOITERING_SPEED and len(history) > 1:
        start_time = history[0][0]
        if (now - start_time) >= ANOMALY_LOITERING_DURATION:
            anomaly_reason = "LOITERING"

    # Extreme movement in high-density crowd
    if people_report_count >= ANOMALY_CROWD_COUNT and speed > ANOMALY_STAMPEDE_SPEED:
        if anomaly_reason:
            anomaly_reason += "/EXTREME_SPEED"
        else:
            anomaly_reason = "EXTREME_SPEED"

    # Very high crowd density anomaly
    if people_report_count >= ANOMALY_DENSITY_THRESHOLD:
        if anomaly_reason:
            anomaly_reason += "/HIGH_DENSITY"
        else:
            anomaly_reason = "HIGH_DENSITY"

    return anomaly_reason is not None, anomaly_reason

# ---------------- Inference & annotate ----------------
def safe_infer_and_annotate(frame):
    global _tracks, latest_people_report, latest_crowd_conflict
    try:
        results = model.track(frame, imgsz=IMG_SIZE, classes=[0], tracker="bytetrack.yaml")
        r = results[0]
    except Exception:
        try:
            results = model(frame, imgsz=IMG_SIZE, classes=[0])
            r = results[0]
        except Exception:
            with frame_lock:
                latest_crowd_conflict = False
            return frame, 0, [], False

    boxes = getattr(r, "boxes", None)
    if boxes is None:
        with frame_lock:
            latest_crowd_conflict = False
        return frame, 0, [], False

    try:
        xyxy = boxes.xyxy.cpu().numpy()
    except Exception:
        xyxy = []

    ids_arr = None
    try:
        ids_arr = boxes.id.cpu().numpy()
    except Exception:
        ids_arr = None

    annotated = frame.copy()
    detected_centroids = []
    raw_items = []

    for i, box in enumerate(xyxy):
        try:
            x1, y1, x2, y2 = [int(v) for v in box]
        except Exception:
            continue
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        tid = None
        if ids_arr is not None and i < len(ids_arr):
            try:
                val = ids_arr[i]
                if np.isnan(val):
                    tid = None
                else:
                    tid = int(val)
            except Exception:
                tid = None
        detected_centroids.append((cx, cy))
        raw_items.append(((cx, cy), tid, (x1, y1, x2, y2)))

    people_report = []
    now = time.time()
    centroids_no_id = []
    for (cx, cy), tid_in, bbox in raw_items:
        if tid_in is not None:
            _update_track(tid_in, (cx, cy), now)
        else:
            centroids_no_id.append((cx, cy))
    if centroids_no_id:
        matches, _ = _match_and_assign_ids(centroids_no_id)
        for tid, centroid in matches:
            _update_track(tid, centroid, now)

    with _tracks_lock:
        tracks_snapshot = dict(_tracks)

    vx_sum, vy_sum = 0.0, 0.0
    person_count = 0
    directions_map = {"UP":0,"DOWN":0,"LEFT":0,"RIGHT":0,"STILL":0}

    for (cx, cy), tid_in, bbox in raw_items:
        x1, y1, x2, y2 = bbox
        chosen_tid = tid_in
        if chosen_tid is None:
            nearest_tid = None
            nearest_dist = None
            for tid, tr in tracks_snapshot.items():
                tx, ty = tr["pos"]
                dist = ((tx - cx)**2 + (ty - cy)**2)**0.5
                if nearest_dist is None or dist < nearest_dist:
                    nearest_dist = dist
                    nearest_tid = tid
            if nearest_tid is not None and nearest_dist <= 150:
                chosen_tid = nearest_tid
                _update_track(chosen_tid, (cx, cy), now)
            else:
                chosen_tid = _next_id()
                _add_track(chosen_tid, (cx, cy), now)

        with _tracks_lock:
            tr = _tracks.get(chosen_tid)
            history = tr["history"] if tr else [(now, (cx, cy))]

        vx, vy, speed_px_s, direction, _ = _compute_speed_and_direction(history)
        is_anomaly, anomaly_reason = _check_for_anomalies(chosen_tid, speed_px_s, direction, history, len(raw_items))

        with _tracks_lock:
            if tr: tr["anomaly_reason"] = anomaly_reason

        # 🔥 NEW: Log track data to database
        _log_track_data(
            track_id=chosen_tid,
            ts=now,
            centroid_x=cx,
            centroid_y=cy,
            speed=round(float(speed_px_s), 2),
            direction=direction,
            anomaly_reason=anomaly_reason if is_anomaly else None
        )

        if direction in directions_map:
            directions_map[direction] += 1
        vx_sum += vx
        vy_sum += vy
        person_count += 1

        color = (0,200,0) if not is_anomaly else (0,0,255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, (cx, cy), 6, color, -1)
        ax = int(cx + vx*ARROW_SCALE)
        ay = int(cy + vy*ARROW_SCALE)
        cv2.arrowedLine(annotated, (cx, cy), (ax, ay), color, 2, tipLength=0.3)
        label = f"ID:{chosen_tid} {speed_px_s:.1f}px/s {direction}"
        if anomaly_reason: label += f" | ANOMALY:{anomaly_reason}"
        cv2.putText(annotated, label, (x1, max(10, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        people_report.append({
            "id": int(chosen_tid),
            "centroid": [int(cx), int(cy)],
            "speed_px_s": round(float(speed_px_s),2),
            "direction": direction,
            "anomaly": is_anomaly,
            "anomaly_reason": anomaly_reason
        })

    # ---------------- Enhanced crowd conflict detection ----------------
    crowd_conflict = False
    if person_count >= ANOMALY_CROWD_COUNT:
        active_directions = {k:v for k,v in directions_map.items() if k!="STILL"}
        total_active = sum(active_directions.values())
        if total_active > 1:
            dominant_direction = max(active_directions, key=active_directions.get)
            dominant_count = active_directions[dominant_direction]
            opposite_count = 0
            if dominant_direction=="LEFT": opposite_count=active_directions.get("RIGHT",0)
            elif dominant_direction=="RIGHT": opposite_count=active_directions.get("LEFT",0)
            elif dominant_direction=="UP": opposite_count=active_directions.get("DOWN",0)
            elif dominant_direction=="DOWN": opposite_count=active_directions.get("UP",0)
            if dominant_count>0 and (opposite_count/dominant_count)>=ANOMALY_CONFLICT_RATIO:
                crowd_conflict=True
        # Additional density-based alert
        if person_count >= ANOMALY_DENSITY_THRESHOLD:
            crowd_conflict=True

    _age_and_cleanup(TRACKER_MAX_AGE)
    person_count=len(people_report)
    count_color=(0,255,0) if not crowd_conflict else (0,0,255)
    cv2.putText(annotated, f"Persons: {person_count}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, count_color, 2)
    if crowd_conflict:
        cv2.putText(annotated, "ALERT: CROWD CONFLICT/BACKFLOW/HIGH DENSITY", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    with frame_lock:
        latest_people_report = people_report.copy()
        latest_crowd_conflict = crowd_conflict

    return annotated, person_count, people_report, crowd_conflict

# ---------------- Camera thread ----------------
def capture_loop():
    global latest_frame_jpg, latest_person_count, running, latest_people_report, latest_crowd_conflict
    cap = cv2.VideoCapture(camera_index, camera_backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Camera capture loop started.")
    while running:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue
        annotated_bgr, cnt, people_report_data, conflict = safe_infer_and_annotate(frame)
        if annotated_bgr is None:
            time.sleep(0.01)
            continue
        success, jpeg = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not success:
            time.sleep(0.01)
            continue
        with frame_lock:
            latest_frame_jpg = jpeg.tobytes()
            latest_person_count = int(cnt)
            latest_people_report = people_report_data.copy()
            latest_crowd_conflict = conflict
        time.sleep(0.02)
    cap.release()
    print("Camera capture loop stopped.")

t = threading.Thread(target=capture_loop, daemon=True)
t.start()

# ---------------- Helpers ----------------
def init_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_history_db_connection(): # 🔥 NEW: History DB connection helper
    conn = sqlite3.connect(HISTORY_DB_PATH)
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
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+frame+b"\r\n")
            time.sleep(0.02)
    return Response(generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/count")
@login_required
def count():
    with frame_lock:
        cnt = latest_person_count
        people = latest_people_report.copy() if isinstance(latest_people_report, list) else []
        crowd_conflict_state = latest_crowd_conflict

    alert_flag = cnt > ANOMALY_CROWD_COUNT or crowd_conflict_state
    if not alert_flag:
        for p in people:
            if p.get("anomaly"):
                alert_flag = True
                break
    return jsonify({
        "persons": cnt,
        "alert": alert_flag,
        "crowd_conflict": crowd_conflict_state,
        "timestamp": int(time.time()),
        "people": people
    })

@app.route("/snapshot")
@login_required
def snapshot():
    with frame_lock:
        frame = latest_frame_jpg
    if frame is None:
        return ("", 503)
    return Response(frame, mimetype="image/jpeg")

# 🔥 NEW: History Page Route
@app.route("/history")
@login_required
def history_page():
    # Fetch initial history data (e.g., last 24 hours of data)
    one_day_ago = time.time() - (24 * 60 * 60)
    conn = init_history_db_connection()
    # Select track history, ordered by timestamp descending
    cur = conn.execute("""
        SELECT * FROM tracks_history WHERE timestamp > ? ORDER BY timestamp DESC
    """, (one_day_ago,))
    history_rows = cur.fetchall()
    conn.close()

    history_data = []
    for row in history_rows:
        history_data.append({
            "track_id": row["track_id"],
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(row["timestamp"])),
            "centroid": f'({row["centroid_x"]}, {row["centroid_y"]})',
            "speed": f'{round(row["speed_px_s"], 2)} px/s',
            "direction": row["direction"],
            "anomaly": row["anomaly_reason"] if row["anomaly_reason"] else "No"
        })

    return render_template("history.html", history_data=history_data)

# 🔥 NEW: History Data API Endpoint (Optional, but good practice)
@app.route("/history/data")
@login_required
def history_data_api():
    # You can implement filtering here (e.g., by track_id, time range)
    limit = request.args.get('limit', 100, type=int)
    conn = init_history_db_connection()
    cur = conn.execute("""
        SELECT * FROM tracks_history ORDER BY timestamp DESC LIMIT ?
    """, (limit,))
    history_rows = [dict(row) for row in cur.fetchall()]
    conn.close()

    return jsonify(history_rows)


# ---------------- Auth ----------------
@app.route("/register", methods=["GET","POST"])
def register():
    if request.method=="POST":
        email = request.form.get("email","").lower()
        pwd = request.form.get("password","")
        if not email or not pwd:
            flash("Email and password required.","error")
            return redirect(url_for("register"))
        conn = init_db_connection()
        try:
            conn.execute("INSERT INTO users (email,password) VALUES (?,?)", (email, generate_password_hash(pwd)))
            conn.commit()
            flash("Registered — log in.","success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already used.","error")
        finally:
            conn.close()
    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method=="POST":
        email = request.form.get("email","").lower()
        pwd = request.form.get("password","")
        conn = init_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email=?",(email,))
        row = cur.fetchone()
        conn.close()
        if row and check_password_hash(row["password"], pwd):
            session["user_email"] = email
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.","error")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

def shutdown():
    global running
    running = False
    t.join(timeout=2)

if __name__=="__main__":
    try:
        print(f"Running on http://{HOST}:{PORT}")
        app.run(host=HOST, port=PORT, threaded=True)
    except KeyboardInterrupt:
        shutdown()