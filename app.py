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

# Tracking config
TRACKER_MAX_AGE = float(os.environ.get("TRACKER_MAX_AGE", 5.0))   # seconds before deleting track
ARROW_SCALE = float(os.environ.get("ARROW_SCALE", 0.08))         # visual scale for arrows

# ----------------------------------------

app = Flask(__name__)
app.secret_key = APP_SECRET

# Shared state
latest_frame_jpg = None
latest_person_count = 0
latest_people_report = []
frame_lock = threading.Lock()
running = True

# Tracking state: track_id -> {"pos":(x,y), "last_seen":ts, "history":[(t,(x,y)), ...]}
_tracks = {}
_next_track_id = 1
_tracks_lock = threading.Lock()

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

# ---------------- Load YOLO ----------------
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

# ---------------- Tracking helpers ----------------
def _next_id():
    global _next_track_id
    with _tracks_lock:
        tid = _next_track_id
        _next_track_id += 1
    return tid

def _add_track(track_id, pos, ts):
    with _tracks_lock:
        _tracks[track_id] = {"pos": pos, "last_seen": ts, "history": [(ts, pos)]}

def _update_track(track_id, pos, ts):
    with _tracks_lock:
        tr = _tracks.get(track_id)
        if tr is None:
            _tracks[track_id] = {"pos": pos, "last_seen": ts, "history": [(ts, pos)]}
            return
        tr["pos"] = pos
        tr["last_seen"] = ts
        tr["history"].append((ts, pos))
        # prune history to recent 1 second for stable speed
        cutoff = ts - 1.0
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
    """
    Simple greedy matching: match detected centroids to existing tracks by nearest neighbor.
    Returns list of (track_id, centroid) matched, and list of unmatched centroids.
    If a detection doesn't match any track within threshold, assign a new ID.
    """
    matches = []
    unmatched = []
    used_tids = set()
    if not detected_centroids:
        return matches, []

    with _tracks_lock:
        # build list of (tid, pos)
        track_items = [(tid, tr["pos"]) for tid, tr in _tracks.items()]
    # compute all distances
    pairs = []
    for tid, tpos in track_items:
        for dpos in detected_centroids:
            dist = ((tpos[0]-dpos[0])**2 + (tpos[1]-dpos[1])**2)**0.5
            pairs.append((dist, tid, dpos))
    pairs.sort(key=lambda x: x[0])
    used_d = set()
    for dist, tid, dpos in pairs:
        # threshold = half diagonal? set 150 px typical
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
    # create new ids for unmatched
    new_assigned = []
    for d in unmatched:
        nid = _next_id()
        _add_track(nid, d, time.time())
        new_assigned.append((nid, d))
    return matches + new_assigned, []

def _compute_speed_and_direction(history):
    """
    history: list of (t,(x,y)) with recent entries (we prune to ~1s)
    returns vx, vy (px/s), speed (px/s), direction string
    """
    if len(history) < 2:
        return 0.0, 0.0, 0.0, "STILL"
    t0, p0 = history[0]
    t1, p1 = history[-1]
    dt = t1 - t0
    if dt <= 0.0001:
        return 0.0, 0.0, 0.0, "STILL"
    vx = (p1[0] - p0[0]) / dt
    vy = (p1[1] - p0[1]) / dt
    speed = (vx**2 + vy**2)**0.5
    # direction by dominant axis
    if abs(vx) < 2 and abs(vy) < 2:
        d = "STILL"
    else:
        if abs(vx) >= abs(vy):
            d = "RIGHT" if vx > 0 else "LEFT"
        else:
            d = "DOWN" if vy > 0 else "UP"
    return vx, vy, speed, d

# ---------------- Model inference + annotate (person-only) ----------------
def safe_infer_and_annotate(frame):
    """
    Runs YOLO tracker, obtains bboxes and (if available) track ids, computes speed/direction
    Draws bbox, centroid circle, arrow for heading, and label with ID+speed+direction.
    Returns annotated_bgr (BGR uint8), person_count (int), people_report (list of dicts).
    """
    global _tracks
    try:
        # Use tracker. This uses ultralytics' track API; some versions accept model.track()
        # If your ultralytics doesn't support .track, this may raise — fallback detection-only below.
        results = model.track(frame, imgsz=IMG_SIZE, classes=[0], tracker="bytetrack.yaml")
        r = results[0]
    except Exception:
        try:
            # fallback: detection-only (no persistent IDs)
            results = model(frame, imgsz=IMG_SIZE, classes=[0])
            r = results[0]
        except Exception:
            return None, 0, []

    boxes = getattr(r, "boxes", None)
    if boxes is None:
        return frame, 0, []

    # get xyxy
    try:
        xyxy = boxes.xyxy.cpu().numpy()
    except Exception:
        try:
            xyxy = np.array(boxes.xyxy)
        except Exception:
            xyxy = []

    # try to fetch ids (ByteTrack sets boxes.id)
    ids_arr = None
    try:
        # prefer boxes.id if present
        ids_arr = boxes.id.cpu().numpy()
    except Exception:
        try:
            ids_arr = np.array(boxes.id)
        except Exception:
            ids_arr = None

    annotated = frame.copy()
    detected_centroids = []
    raw_items = []  # list of tuples (centroid, tid_or_none, bbox)

    # build detection list
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

    # If tracker provided no ids (detection-only), we'll match to existing tracks greedily and create new IDs
    people_report = []
    now = time.time()

    # First, if any detections have tid, update/create tracks for those
    # We'll collect centroids that do not have tid and match them
    centroids_no_id = []
    for (cx, cy), tid, bbox in raw_items:
        if tid is not None:
            # update or create track with this tid
            _update_track(tid, (cx, cy), now)
        else:
            centroids_no_id.append((cx, cy))

    # greedy matching for centroids without id
    if centroids_no_id:
        # match these centroids to existing tracks
        matches, _ = _match_and_assign_ids(centroids_no_id)
        # matches is list of (tid, centroid) where new tids were also created
        for tid, centroid in matches:
            _update_track(tid, centroid, now)

    # now prepare drawing & report by iterating current tracks and marking those near detections
    # Build a quick list of tracks snapshot
    with _tracks_lock:
        tracks_snapshot = dict(_tracks)

    # For each detection (raw_items), find nearest track id (within threshold) to display
    for (cx, cy), tid_in, bbox in raw_items:
        x1, y1, x2, y2 = bbox
        # find nearest track
        nearest_tid = None
        nearest_dist = None
        for tid, tr in tracks_snapshot.items():
            tx, ty = tr["pos"]
            dist = ((tx - cx)**2 + (ty - cy)**2)**0.5
            if nearest_dist is None or dist < nearest_dist:
                nearest_dist = dist
                nearest_tid = tid
        # if nearest distance is reasonably close (150 px threshold), use that id; else use provided tid_in if exists
        chosen_tid = None
        if tid_in is not None:
            chosen_tid = tid_in
            # ensure track exists; if not, add
            with _tracks_lock:
                if chosen_tid not in _tracks:
                    _add_track(chosen_tid, (cx, cy), now)
        elif nearest_tid is not None and nearest_dist is not None and nearest_dist <= 150:
            chosen_tid = nearest_tid
            # update that track with current detection to keep it fresh
            _update_track(chosen_tid, (cx, cy), now)
        else:
            # create a new track id
            chosen_tid = _next_id()
            _add_track(chosen_tid, (cx, cy), now)

        # compute speed & direction from track history
        with _tracks_lock:
            tr = _tracks.get(chosen_tid)
            history = tr["history"] if tr is not None else [(now, (cx, cy))]
        vx, vy, speed_px_s, direction = _compute_speed_and_direction(history)

        # draw bounding box (keep green)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        # draw a filled circle at centroid
        cv2.circle(annotated, (cx, cy), 6, (0, 255, 0), -1)
        # draw heading arrow
        # arrow end is centroid + (vx, vy) scaled down for visibility
        ax = int(cx + vx * ARROW_SCALE)
        ay = int(cy + vy * ARROW_SCALE)
        cv2.arrowedLine(annotated, (cx, cy), (ax, ay), (0, 255, 0), 2, tipLength=0.3)
        # label: ID and speed and direction
        label = f"ID:{chosen_tid} {speed_px_s:.1f}px/s {direction}"
        # place text avoiding overflow
        tx = x1
        ty = max(10, y1 - 10)
        cv2.putText(annotated, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        # append to report
        people_report.append({
            "id": int(chosen_tid),
            "centroid": [int(cx), int(cy)],
            "speed_px_s": round(float(speed_px_s), 2),
            "direction": direction
        })

    # cleanup old tracks
    _age_and_cleanup(TRACKER_MAX_AGE)

    # final overlay persons count
    person_count = len(people_report)
    cv2.putText(annotated, f"Persons: {person_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # store latest report thread-safely
    with frame_lock:
        # update global latest_people_report
        global latest_people_report
        latest_people_report = people_report.copy()

    return annotated, person_count, people_report

# ---------------- Camera capture thread ----------------
def capture_loop():
    global latest_frame_jpg, latest_person_count, running, latest_people_report
    cap = cv2.VideoCapture(camera_index, camera_backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)

    # set resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Camera capture loop started.")
    while running:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        annotated_bgr, cnt, people = safe_infer_and_annotate(frame)
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
            latest_people_report = people.copy()

        # avoid saturating CPU
        time.sleep(0.02)

    cap.release()
    print("Camera capture loop stopped.")

# Start capture thread
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
        people = latest_people_report.copy() if isinstance(latest_people_report, list) else []
    alert_flag = True if cnt > 5 else False
    return jsonify({
        "persons": cnt,
        "alert": alert_flag,
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
