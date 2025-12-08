from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Capture from webcam
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # detect PERSON only
        results = model(frame, classes=[0])

        # Count persons + draw boxes
        person_count = 0
        for r in results:
            for box in r.boxes:
                person_count += 1
                x1, y1, x2, y2 = box.xyxy[0]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.putText(frame, "Person", (int(x1), int(y1)-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # add count to screen
        cv2.putText(frame, f"Person Count: {person_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
