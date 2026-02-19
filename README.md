# Smart Crowd Monitor & Anomaly Detection

A real-time crowd monitoring system built with **Flask**, **YOLOv8**, and **OpenCV**. This application detects and tracks people in a video feed, providing analytics on crowd density, movement anomalies, and potential safety risks.

## Features

- **Real-time People Tracking**: Uses YOLOv8 and ByteTrack to detect and track individuals.
- **Anomaly Detection**:
  - **Loitering**: Detects individuals staying in one area for too long.
  - **Stampede / Extreme Speed**: Identifies unusually fast movements when the crowd count is high.
  - **Crowd Conflict**: Detects opposing flows of movement (e.g., people running against the main flow).
  - **High Density**: Alerts when the number of people exceeds a safety threshold.
- **Density Analytics**: Calculates crowd density levels (Low, Medium, High, Critical) based on configured capacity.
- **Web Dashboard**: Use the browser to view the live video feed with overlay annotations, real-time alerts, and statistics.
- **Historical Data**: Logs tracking data and anomalies to a SQLite database for review.
- **Authentication**: Secure login and registration system to protect access.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python 3.10
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download YOLOv8 Model:**
    The application defaults to `yolov8n.pt`. It will automatically download on the first run, or you can place your own model file in the project root.

## Usage

1.  **Run the application:**
    ```bash
    conda activate yolo
    python app.py
    ```

2.  **Access the Dashboard:**
    Open your web browser and go to `http://localhost:5000`.

3.  **Register / Login:**
    - Create a new account on the `/register` page.
    - Log in to access the dashboard and video feed.

## Configuration

You can configure the application using Environment Variables. The default values are tuned for general use.

| Variable | Default | Description |
| :--- | :--- | :--- |
| `FLASK_SECRET` | `change_this_secret` | Secret key for Flask sessions. **Change this in production.** |
| `YOLO_MODEL` | `yolov8n.pt` | Path to the YOLO model file. |
| `IMG_SIZE` | `640` | Inference image size for YOLO. |
| `HOST` | `0.0.0.0` | Host interface to bind to. |
| `PORT` | `5000` | Port to run the server on. |
| `MAX_CAPACITY` | `30` | Maximum expected person count for 100% density calculation. |
| `ANOMALY_CROWD_COUNT` | `6` | Minimum count to trigger crowd-specific alerts (e.g., stampede). |
| `ANOMALY_LOITERING_DURATION` | `3.0` | Seconds a person must act stationary to trigger loitering alert. |
| `ANOMALY_LOITERING_SPEED` | `5.0` | Speed threshold (px/s) below which a person is considered stationary. |
| `ANOMALY_STAMPEDE_SPEED` | `150.0` | Speed threshold (px/s) to trigger stampede alert. |

## Project Structure

- **`app.py`**: Main application logic, including Flask routes, YOLO video processing, and anomaly detection algorithms.
- **`requirements.txt`**: Python dependencies.
- **`templates/`**: HTML templates for the dashboard, login, and history pages.
- **`users.db`**: SQLite database for user credentials.
- **`history.db`**: SQLite database for storing tracking logs and anomaly events.
