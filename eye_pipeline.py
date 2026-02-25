import cv2
import mediapipe
from math import sqrt
import numpy as np
import time, psutil
import subprocess  # Force-release any lingering handle on the device

# ---------------------------------------------
# CAM SETUP
# ---------------------------------------------
WARMUP = 5
MEASURE = 30
WIDTH = 320
HEIGHT = 240

subprocess.run(["fuser", "-k", "/dev/video0"], capture_output=True)
time.sleep(1)

# Lock exposure for consistent 30 FPS regardless of lighting
subprocess.run(
    ["v4l2-ctl", "--device=/dev/video0", "--set-ctrl=exposure_dynamic_framerate=0"]
)
subprocess.run(["v4l2-ctl", "--device=/dev/video0", "--set-ctrl=auto_exposure=1"])
subprocess.run(
    ["v4l2-ctl", "--device=/dev/video0", "--set-ctrl=gain=120"]
)  # Amplifies the signal (Add brightness, grains/noise)
subprocess.run(
    ["v4l2-ctl", "--device=/dev/video0", "--set-ctrl=brightness=160"]
)  # Shifts pixel values up/down

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
# Force MJPEG (important)
if not cap.isOpened():
    print("ERROR: Camera not opened")
    exit()
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

# Verify settings were applied
actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera configured: {actual_w}x{actual_h} @ {actual_fps} FPS")

# Drain the buffer (stale frames can skew first-run latency)
for _ in range(10):
    cap.read()

# ---------------------------------------------
# CONFIG
# ---------------------------------------------

# MediaPipe Face Mesh landmark indices
LEFT_EYE_INDICES = [
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
]
RIGHT_EYE_INDICES = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
]

# Iris landmarks (MediaPipe Face Mesh with refine_landmarks=True)
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

# EAR landmarks
# p1=inner corner, p2=upper-inner, p3=upper-outer, p4=outer corner, p5=lower-outer, p6=lower-inner
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]

# Head pose: 6-point 3D model (nose, chin, left/right eye corners, left/right mouth corners)
FACE_3D_MODEL = np.array(
    [
        [0.0, 0.0, 0.0],  # Nose tip                    - 1
        [0.0, -330.0, -65.0],  # Chin                   - 152
        [-225.0, 170.0, -135.0],  # Left eye corner     - 33
        [225.0, 170.0, -135.0],  # Right eye corner     - 263
        [-150.0, -150.0, -125.0],  # Left mouth corner  - 61
        [150.0, -150.0, -125.0],  # Right mouth corner  - 291
    ],
    dtype=np.float64,
)

FACE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

BLINK_THRESHOLD = 0.21  # EAR below this = blink
BLINK_CONSEC_FRAMES = 2

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------
# Euclidean distance to calculate the distance between the two points
def euclideanDistance(p1, p2):
    x, y = p1
    x1, y1 = p2
    distance = sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance


# ---------------------------------------------
# MODULES
# ---------------------------------------------
def eye_aspect_ratio(landmarks, ear_indices, img_w, img_h):
    """
    Calculates Eye Aspect Ratio (EAR) for blink detection.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    pts = np.array(
        [[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in ear_indices],
        dtype=np.float64,
    )
    A = np.linalg.norm(pts[1] - pts[5])  # upper-inner to lower-inner
    B = np.linalg.norm(pts[2] - pts[4])  # upper-outer to lower-outer
    C = np.linalg.norm(pts[0] - pts[3])  # inner corner to outer corner
    return (A + B) / (2.0 * C) if C > 0 else 0.0


def detect_blink(landmarks, img_w, img_h, blink_counter, blink_total):
    """
    Detects blink using EAR on both eyes.
    Returns (left_ear, right_ear, avg_ear, is_blinking, updated_counter, updated_total).
    """
    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_EAR, img_w, img_h)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_EAR, img_w, img_h)
    avg_ear = (left_ear + right_ear) / 2.0

    if avg_ear < BLINK_THRESHOLD:
        blink_counter += 1
    else:
        if blink_counter >= BLINK_CONSEC_FRAMES:
            blink_total += 1
        blink_counter = 0

    is_blinking = avg_ear < BLINK_THRESHOLD
    return left_ear, right_ear, avg_ear, is_blinking, blink_counter, blink_total


def estimate_gaze(landmarks, img_w, img_h):
    """
    Estimates gaze direction from iris center relative to eye corner midpoint.
    Returns (left_gaze_x, left_gaze_y, right_gaze_x, right_gaze_y) ï¿½ normalised offsets.
    """

    def iris_offset(iris_ids, eye_ids):
        iris_center = np.mean(
            [[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in iris_ids], axis=0
        )
        # eye_ids[0] = inner_corner, eye_ids[8] = outer corner
        eye_inner = np.array(
            [landmarks[eye_ids[0]].x * img_w, landmarks[eye_ids[0]].y * img_h]
        )
        eye_outer = np.array(
            [landmarks[eye_ids[8]].x * img_w, landmarks[eye_ids[8]].y * img_h]
        )
        eye_width = np.linalg.norm(eye_outer - eye_inner) + 1e-6
        eye_mid = (eye_inner + eye_outer) / 2
        offset = (iris_center - eye_mid) / eye_width
        return offset[0], offset[1]

    lx, ly = iris_offset(LEFT_IRIS_INDICES, LEFT_EYE_INDICES)
    rx, ry = iris_offset(RIGHT_IRIS_INDICES, RIGHT_EYE_INDICES)
    return lx, ly, rx, ry


def estimate_head_pose(landmarks, img_w, img_h, cam_matrix, dist_coeffs):
    """
    Estimates head pose using solvePnP.
    Returns (success, pitch, yaw, roll) in degrees.
    """
    face_2d = np.array(
        [[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in FACE_LANDMARK_IDS],
        dtype=np.float64,
    )

    success, rot_vec, _ = cv2.solvePnP(
        FACE_3D_MODEL, face_2d, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return False, 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch = angles[0] * 360
    yaw = angles[1] * 360
    roll = angles[2] * 360
    return True, pitch, yaw, roll


# ---------------------------------------------
# MEDIAPIPE FACEMESH
# ---------------------------------------------
mediapipe_face_mesh = mediapipe.solutions.face_mesh
face_mesh = mediapipe_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Iris Landmarks
    min_detection_confidence=0.6,
    min_tracking_confidence=0.7,
)

# ---------------------------------------------
# METRIC LIST
# ---------------------------------------------
fps_list = []
lat_total_list = []  # full pipeline latency per frame
lat_capture_list = []  # camera read only
lat_facemesh_list = []  # MediaPipe FaceMesh only
lat_blink_list = []  # blink ratio calculation only
lat_gaze_list = []  # gaze direction calculation only
lat_pose_list = []
cpu_list = []
ram_list = []

BLINK_COUNTER = 0
TOTAL_BLINKS = 0
FONT = cv2.FONT_HERSHEY_SIMPLEX

t_start = time.perf_counter()
t_measure_start = None
t_prev = time.perf_counter()

# ---------------------------------------------
# MAIN LOOP
# ---------------------------------------------
while True:
    t_frame_start = time.perf_counter()

    # --- 1. Camera Capture ---
    t0 = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        break
    t1 = time.perf_counter()
    capture_ms = (t1 - t0) * 1000

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Camera matrix: estimated from frame size (no calibration file needed)
    # Focal length approximated as image width; principal point at image center
    cam_matrix = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # --- 2. FaceMesh Inference ---
    t0 = time.perf_counter()
    results = face_mesh.process(rgb)
    t1 = time.perf_counter()
    facemesh_ms = (t1 - t0) * 1000

    #  Only when face detected ---
    blink_ms = 0.0
    gaze_ms = 0.0
    pose_ms = 0.0

    avg_ear = 0.0
    pitch = yaw = roll = 0.0
    lx = ly = rx = ry = 0.0
    pose_ok = False

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark  # 1 FACE ONLY

        # -- 3. Blink Detection --
        t0 = time.perf_counter()
        left_ear, right_ear, avg_ear, is_blinking, BLINK_COUNTER, TOTAL_BLINKS = (
            detect_blink(lm, w, h, BLINK_COUNTER, TOTAL_BLINKS)
        )
        t1 = time.perf_counter()
        blink_ms = (t1 - t0) * 1000

        # --- 4. Gaze Estimation
        t0 = time.perf_counter()
        lx, ly, rx, ry = estimate_gaze(lm, w, h)
        t1 = time.perf_counter()
        gaze_ms = (t1 - t0) * 1000

        # --- 5. Head Pose Estimation
        t0 = time.perf_counter()
        pose_ok, pitch, yaw, roll = estimate_head_pose(
            lm, w, h, cam_matrix, dist_coeffs
        )
        t1 = time.perf_counter()
        pose_ms = (t1 - t0) * 1000

        # Overlay on frame
        status = "BLINK" if is_blinking else "OPEN"
        cv2.putText(
            frame,
            f"EAR: {avg_ear:.2f}  [{status}]",
            (10, 30),
            FONT,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame, f"Blinks: {TOTAL_BLINKS}", (10, 55), FONT, 0.6, (0, 255, 0), 2
        )
        cv2.putText(
            frame,
            f"Gaze L: ({lx:.2f}, {ly:.2f})",
            (10, 80),
            FONT,
            0.55,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Gaze R: ({rx:.2f}, {ry:.2f})",
            (10, 100),
            FONT,
            0.55,
            (0, 255, 255),
            2,
        )
        if pose_ok:
            cv2.putText(
                frame,
                f"P:{pitch:.1f} Y:{yaw:.1f} R:{roll:.1f}",
                (10, 125),
                FONT,
                0.55,
                (255, 200, 0),
                2,
            )

    # -------- Total pipeline latency & FPS
    t_frame_end = time.perf_counter()
    total_ms = (t_frame_end - t_frame_start) * 1000
    fps = 1.0 / (t_frame_end - t_prev)
    t_prev = t_frame_end
    elapsed = t_frame_end - t_start

    # Metrics are collected after warm up
    if elapsed >= WARMUP:
        if t_measure_start is None:
            t_measure_start = t_frame_end

        fps_list.append(fps)
        lat_total_list.append(total_ms)
        lat_capture_list.append(capture_ms)
        lat_facemesh_list.append(facemesh_ms)
        lat_blink_list.append(blink_ms)
        lat_gaze_list.append(gaze_ms)
        lat_pose_list.append(pose_ms)
        cpu_list.append(psutil.cpu_percent(interval=None))
        ram_list.append(psutil.virtual_memory().percent)

        if (t_frame_end - t_measure_start) >= MEASURE:
            break

    cv2.imshow("Benchmark: Combined", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

backend = cap.getBackendName()
cap.release()
cv2.destroyAllWindows()


# ---------------------------------------------
# BENCHMARK SUMMARY
# ---------------------------------------------
print("\n===== BENCHMARK SUMMARY =====")
print(f"Backend    : {backend}")
print(
    f"Resolution : {int(actual_w)}x{int(actual_h)} @ {int(actual_fps)} FPS (configured)"
)
print(f"Frames     : {len(fps_list)}")

if fps_list:
    print("\n--- System Performance ---")
    print(
        f"Avg FPS : {np.mean(fps_list):.2f}  |  Min: {np.min(fps_list):.2f}  |  Max: {np.max(fps_list):.2f}"
    )
    print(f"Avg CPU : {np.mean(cpu_list):.1f}%")
    print(f"Avg RAM : {np.mean(ram_list):.1f}%")

    print("\n--- Per-Module Latency (ms) ---")
    header = f"{'Module':<24} {'Avg':>7} {'P50':>7} {'P95':>7} {'Max':>7}"
    print(header)
    print("-" * len(header))

    modules = [
        ("Total (pipeline)", lat_total_list),
        ("  Camera capture", lat_capture_list),
        ("  FaceMesh infer", lat_facemesh_list),
        ("  Blink EAR", lat_blink_list),
        ("  Gaze estimation", lat_gaze_list),
        ("  Head pose PnP", lat_pose_list),
    ]
    for name, data in modules:
        arr = np.array(data)
        print(
            f"{name:<24} {np.mean(arr):>7.2f} {np.median(arr):>7.2f} "
            f"{np.percentile(arr,95):>7.2f} {np.max(arr):>7.2f}"
        )

    print(f"\nTotal blinks detected : {TOTAL_BLINKS}")
else:
    print("No frames captured during measurement window.")
