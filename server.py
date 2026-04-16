from flask import (Flask, request, jsonify,
                   send_from_directory, Response)
import cv2, json, math
import os, tempfile, threading, uuid, time
import subprocess
import numpy as np
from math import sqrt, acos, atan2, degrees
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO
import boto3
from botocore.client import Config

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
REACT_BUILD_DIR = os.path.join(BASE_DIR, 'static_react')

# Serve React build if it exists, otherwise fall back to old static folder
_static_folder = REACT_BUILD_DIR if os.path.isdir(REACT_BUILD_DIR) else os.path.join(BASE_DIR, 'static')

app = Flask(__name__, static_folder=_static_folder, static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

JOINT_IDS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
MAX_FILE_MB = 500
jobs = {}           # job_id → job dict
jobs_lock = threading.Lock()

# ── Skeleton drawing constants ─────────────────────────────────────────────────
LEAD_COLOR   = (255, 79, 74)     # BGR for #4a8eff blue
TRAIL_COLOR  = (14, 65, 183)     # BGR for #b7410e rust orange
JOINT_COLOR  = (255, 255, 255)   # white
JOINT_RADIUS  = 6
LINE_THICKNESS = 2

LEAD_LANDMARKS  = [11, 13, 15, 23, 25, 27]
TRAIL_LANDMARKS = [12, 14, 16, 24, 26, 28]

CONNECTIONS = [
    (11, 12, LEAD_COLOR),    # shoulders
    (11, 13, LEAD_COLOR),    # lead upper arm
    (13, 15, LEAD_COLOR),    # lead lower arm
    (12, 14, TRAIL_COLOR),   # trail upper arm
    (14, 16, TRAIL_COLOR),   # trail lower arm
    (23, 24, LEAD_COLOR),    # hips
    (11, 23, LEAD_COLOR),    # lead torso
    (12, 24, TRAIL_COLOR),   # trail torso
    (23, 25, LEAD_COLOR),    # lead upper leg
    (25, 27, LEAD_COLOR),    # lead lower leg
    (24, 26, TRAIL_COLOR),   # trail upper leg
    (26, 28, TRAIL_COLOR),   # trail lower leg
]


# ── Rotation detection ─────────────────────────────────────────────────────────
def detect_rotation(video_path):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream_tags=rotate',
             '-of', 'default=noprint_wrappers=1:nokey=1',
             video_path],
            capture_output=True, text=True, timeout=10
        )
        val = result.stdout.strip()
        if val in ('90', '180', '270'):
            return int(val)
        return 0
    except Exception as e:
        print(f'[ROTATION] ffprobe failed: {e}', flush=True)
        return 0


# ── Frame rotation correction ──────────────────────────────────────────────────
def rotate_frame(frame, rotation):
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


# ── Skeleton drawing ───────────────────────────────────────────────────────────
def draw_skeleton(frame, landmarks_2d, frame_w, frame_h):
    """landmarks_2d: { '11': {'x': float, 'y': float}, ... }  (normalized 0-1)"""
    pts = {}
    for idx_str, lm in landmarks_2d.items():
        px = int(lm['x'] * frame_w)
        py = int(lm['y'] * frame_h)
        pts[int(idx_str)] = (px, py)

    for a, b, color in CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], color, LINE_THICKNESS)

    for idx, pt in pts.items():
        color = LEAD_COLOR if idx in LEAD_LANDMARKS else TRAIL_COLOR
        cv2.circle(frame, pt, JOINT_RADIUS, JOINT_COLOR, -1)
        cv2.circle(frame, pt, JOINT_RADIUS, color, 1)

    return frame


def draw_yolo(frame, yolo_detections, frame_w, frame_h):
    """yolo_detections: dict with keys club_head, club_handle, golf_ball."""
    YOLO_COLOR = (255, 0, 255)   # magenta
    LABELS = {
        'club_head':   'Club Head',
        'club_handle': 'Handle',
        'golf_ball':   'Ball',
    }
    for key, label in LABELS.items():
        det = yolo_detections.get(key)
        if not det:
            continue
        x1 = int(det['x1'] * frame_w)
        y1 = int(det['y1'] * frame_h)
        x2 = int(det['x2'] * frame_w)
        y2 = int(det['y2'] * frame_h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), YOLO_COLOR, 1)
        cv2.putText(frame, label, (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, YOLO_COLOR, 1,
                    cv2.LINE_AA)
    return frame


# ── R2 upload ──────────────────────────────────────────────────────────────────
def upload_to_r2(file_path, key):
    try:
        s3 = boto3.client(
            's3',
            endpoint_url=os.environ.get('CLOUDFLARE_R2_ENDPOINT'),
            aws_access_key_id=os.environ.get('CLOUDFLARE_R2_ACCESS_KEY'),
            aws_secret_access_key=os.environ.get('CLOUDFLARE_R2_SECRET_KEY'),
            config=Config(signature_version='s3v4'),
            region_name='auto',
        )
        bucket = os.environ.get('CLOUDFLARE_R2_BUCKET', 'user-swing-archives')
        s3.upload_file(file_path, bucket, key,
                       ExtraArgs={'ContentType': 'video/mp4'})
        public_url = os.environ.get('CLOUDFLARE_R2_PUBLIC_URL', '')
        video_url  = f'{public_url}/{key}'
        print(f'[R2] uploaded: {video_url}', flush=True)
        return video_url
    except Exception as e:
        print(f'[R2] upload failed: {e}', flush=True)
        return None


# ── YOLO model ─────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'golf_driver_v2_best.pt')
_YOLO_DOWNLOAD_URL = (
    'https://github.com/codwithjin/chip-render/raw/master/'
    'models/golf_driver_v2_best.pt'
)

def _ensure_yolo_model():
    """Download YOLO model if missing (handles Railway build-cache stripping models/)."""
    if os.path.exists(YOLO_MODEL_PATH):
        return
    import urllib.request
    os.makedirs(os.path.dirname(YOLO_MODEL_PATH), exist_ok=True)
    print(f'[YOLO] Model missing — downloading from GitHub...', flush=True)
    try:
        urllib.request.urlretrieve(_YOLO_DOWNLOAD_URL, YOLO_MODEL_PATH)
        print(f'[YOLO] Download complete: {YOLO_MODEL_PATH}', flush=True)
    except Exception as e:
        print(f'[YOLO] Download failed: {e}', flush=True)

_ensure_yolo_model()

MEDIAPIPE_MODEL_PATH = os.path.join(BASE_DIR, 'pose_landmarker_heavy.task')
_MEDIAPIPE_DOWNLOAD_URL = (
    'https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
    'pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task'
)

def _ensure_mediapipe_model():
    """Download MediaPipe pose landmarker if missing (gitignored, not in Railway image)."""
    if os.path.exists(MEDIAPIPE_MODEL_PATH):
        return
    import urllib.request
    print('[MediaPipe] Model missing — downloading from Google Storage...', flush=True)
    try:
        urllib.request.urlretrieve(_MEDIAPIPE_DOWNLOAD_URL, MEDIAPIPE_MODEL_PATH)
        print(f'[MediaPipe] Download complete: {MEDIAPIPE_MODEL_PATH}', flush=True)
    except Exception as e:
        print(f'[MediaPipe] Download failed: {e}', flush=True)

_ensure_mediapipe_model()

yolo_model = None
if os.path.exists(YOLO_MODEL_PATH):
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print(f"[YOLO] Club head model loaded: {YOLO_MODEL_PATH}")
else:
    print(f"[YOLO] Model not found at {YOLO_MODEL_PATH}")


# ── Job cleanup ───────────────────────────────────────────────────────────────
def cleanup_old_jobs():
    while True:
        time.sleep(300)
        cutoff = time.time() - 1800
        with jobs_lock:
            stale = [jid for jid, j in jobs.items()
                     if j.get('created_at', 0) < cutoff]
            for jid in stale:
                del jobs[jid]

cleaner = threading.Thread(target=cleanup_old_jobs, daemon=True)
cleaner.start()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(os.path.join(BASE_DIR, 'static'), 'index.html')


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/ping')
def ping():
    return 'pong', 200


@app.route('/process', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400

    f = request.files['video']

    # Size check
    f.seek(0, 2)
    size_mb = f.tell() / (1024 * 1024)
    f.seek(0)
    if size_mb > MAX_FILE_MB:
        return jsonify({
            'error': f'File too large ({size_mb:.0f}MB). Max {MAX_FILE_MB}MB.'
        }), 400

    job_id = str(uuid.uuid4())[:8]
    suffix = os.path.splitext(f.filename)[1] or '.mp4'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.save(tmp.name)
    tmp.close()

    with jobs_lock:
        jobs[job_id] = {
            'status':     'processing',
            'progress':   0,
            'total':      0,
            'result':     None,
            'error':      None,
            'created_at': time.time(),
            'filename':   f.filename,
        }

    start_ms = float(request.form.get('start_ms', 0))
    end_ms   = float(request.form.get('end_ms', 9999999))

    t = threading.Thread(target=run_mediapipe,
                         args=(tmp.name, job_id, start_ms, end_ms),
                         daemon=True)
    t.start()
    return jsonify({'job_id': job_id})


def run_mediapipe(video_path, job_id, start_ms=0, end_ms=9999999):
    raw_path       = None
    annotated_path = None
    try:
        print(f"[MediaPipe] Starting job {job_id}", flush=True)
        print(f"[MediaPipe] Video: {video_path}", flush=True)

        # Detect rotation from file metadata before opening cap
        rotation = detect_rotation(video_path)
        print(f'[ROTATION] detected: {rotation}°', flush=True)

        model_path   = MEDIAPIPE_MODEL_PATH
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options      = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            num_poses=1,
        )
        landmarker = mp_vision.PoseLandmarker.create_from_options(options)

        cap   = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Skip frames so we process at most ~30fps effective rate.
        # iPhone slo-mo at 120/240fps would otherwise process 4-8x more
        # frames than needed for phase detection.
        skip = max(1, round(fps / 30.0))
        frames_to_process = max(1, total // skip)
        print(f"[MediaPipe] FPS:{fps} Frames:{total} Size:{src_w}x{src_h} "
              f"rotation:{rotation}° skip:{skip} effective:{frames_to_process}",
              flush=True)

        with jobs_lock:
            jobs[job_id]['total'] = frames_to_process

        frames_out       = []
        annotated_frames = []
        frame_num        = 0
        processed        = 0
        out_w            = None
        out_h            = None

        while cap.isOpened():
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            ret, frame = cap.read()
            if not ret:
                break
            if pos_ms < start_ms:
                frame_num += 1
                continue
            if pos_ms > end_ms:
                break

            if frame_num % skip != 0:
                frame_num += 1
                continue

            # Rotate frame — MediaPipe and YOLO both see the corrected orientation
            frame     = rotate_frame(frame, rotation)
            h, w      = frame.shape[:2]
            if out_w is None:
                out_w, out_h = w, h

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = landmarker.detect(mp_image)

            fd = {
                'frame':        frame_num,
                'timestamp_ms': int(frame_num / fps * 1000),
                'poses':        [],
            }

            landmarks_2d_for_draw = {}

            if result.pose_landmarks and result.pose_world_landmarks:
                entry = {
                    'pose_index':   0,
                    'landmarks_2d': {},
                    'landmarks_3d': {},
                }
                for jid in JOINT_IDS:
                    lm2 = result.pose_landmarks[0][jid]
                    lm3 = result.pose_world_landmarks[0][jid]
                    # Frame is already rotated — MediaPipe returns coords in
                    # the rotated frame's space. No coordinate transform needed.
                    entry['landmarks_2d'][str(jid)] = {
                        'x':          round(lm2.x, 6),
                        'y':          round(lm2.y, 6),
                        'z':          round(lm2.z, 6),
                        'visibility': round(lm2.visibility, 4),
                    }
                    entry['landmarks_3d'][str(jid)] = {
                        'x':          round(lm3.x, 6),
                        'y':          round(lm3.y, 6),
                        'z':          round(lm3.z, 6),
                        'visibility': round(lm3.visibility, 4),
                    }
                fd['poses'].append(entry)
                landmarks_2d_for_draw = entry['landmarks_2d']

            # YOLO detection — club head, handle, golf ball
            club_head   = None
            club_handle = None
            golf_ball   = None

            if yolo_model is not None:
                try:
                    yolo_results = yolo_model(frame, verbose=False, conf=0.25)
                    if yolo_results:
                        for box in yolo_results[0].boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            cx   = (x1 + x2) / 2
                            cy   = (y1 + y2) / 2
                            conf = float(box.conf[0])
                            cls  = int(box.cls[0])
                            names = yolo_results[0].names
                            label = names[cls].lower()
                            obj = {
                                'x':    cx / w,
                                'y':    cy / h,
                                'x1':   x1 / w,
                                'y1':   y1 / h,
                                'x2':   x2 / w,
                                'y2':   y2 / h,
                                'x_px': cx,
                                'y_px': cy,
                                'conf': conf,
                            }
                            if 'head' in label:
                                club_head = obj
                            elif 'handle' in label:
                                club_handle = obj
                            elif 'ball' in label:
                                golf_ball = obj
                except Exception:
                    pass

            fd['club_head']   = club_head
            fd['club_handle'] = club_handle
            fd['golf_ball']   = golf_ball

            # Draw skeleton and YOLO boxes onto annotated frame
            ann = frame.copy()
            if landmarks_2d_for_draw:
                ann = draw_skeleton(ann, landmarks_2d_for_draw, w, h)
            ann = draw_yolo(ann,
                            {'club_head':   club_head,
                             'club_handle': club_handle,
                             'golf_ball':   golf_ball},
                            w, h)
            annotated_frames.append(ann)

            frames_out.append(fd)
            processed  += 1
            frame_num  += 1

            with jobs_lock:
                jobs[job_id]['progress'] = processed

        # Fallback if no frames were processed
        if out_w is None:
            out_w, out_h = src_w, src_h

        pose_frame_count = sum(1 for f in frames_out if f['poses'])
        print(f"[MediaPipe] Done. Pose frames: {pose_frame_count}/{processed} "
              f"(skip={skip}, source={total}) out:{out_w}x{out_h}", flush=True)

        # ── Encode annotated video ─────────────────────────────────────────────
        video_url = None
        if annotated_frames:
            raw_path = tempfile.NamedTemporaryFile(delete=False, suffix='.raw').name
            with open(raw_path, 'wb') as rf:
                for af in annotated_frames:
                    rf.write(af.tobytes())

            annotated_path = tempfile.NamedTemporaryFile(
                delete=False, suffix='.mp4').name
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{out_w}x{out_h}',
                '-pix_fmt', 'bgr24',
                '-r', '30',
                '-i', raw_path,
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'fast',
                annotated_path,
            ]
            ffmpeg_result = subprocess.run(
                ffmpeg_cmd, capture_output=True, text=True, timeout=120)

            # Delete raw file immediately after encode attempt
            if raw_path and os.path.exists(raw_path):
                try:
                    os.unlink(raw_path)
                except Exception:
                    pass
                raw_path = None

            if ffmpeg_result.returncode != 0:
                print(f'[ANNOTATE] ffmpeg failed: {ffmpeg_result.stderr}',
                      flush=True)
                annotated_path = None
            else:
                print(f'[ANNOTATE] encoded: {annotated_path}', flush=True)
                r2_key    = f'swings/{job_id}.mp4'
                video_url = upload_to_r2(annotated_path, r2_key)

        result = {
            'fps':          fps,
            'total_frames': total,
            'width':        out_w,
            'height':       out_h,
            'video_url':    video_url,
            'joints':       JOINT_IDS,
            'frames':       frames_out,
        }

        with jobs_lock:
            jobs[job_id]['status'] = 'done'
            jobs[job_id]['result'] = result

    except Exception as e:
        import traceback
        print(f"[MediaPipe] ERROR: {e}", flush=True)
        traceback.print_exc()
        with jobs_lock:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['error']  = str(e)
    finally:
        for path in [video_path, raw_path, annotated_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass
        try:
            cap.release()
            landmarker.close()
        except Exception:
            pass


@app.route('/progress/<job_id>')
def get_progress(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify({
        'status':   job['status'],
        'progress': job['progress'],
        'total':    job['total'],
        'error':    job.get('error'),
        'filename': job.get('filename', ''),
    })


@app.route('/result/<job_id>')
def get_result(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    print(f"[Result] job_id={job_id} status={job.get('status') if job else 'NOT_FOUND'} "
          f"result_is_none={job['result'] is None if job else 'N/A'}", flush=True)
    if not job or job['status'] != 'done':
        return jsonify({'error': 'Not ready'}), 404

    result = job['result']

    def generate():
        yield json.dumps(result)

    with jobs_lock:
        jobs[job_id]['result'] = None   # free memory after streaming

    return Response(generate(), mimetype='application/json')


def compute_swing_plane(positions):
    """
    Compute swing plane normal vector via SVD on
    a list of 3D wrist positions (dicts with x,y,z).
    Returns unit normal vector as np.array [x,y,z].
    The swing plane is the best-fit plane minimizing
    orthogonal distances from all positions.
    Scientific basis: wrist traces a circle on this
    plane (Nature/Sci Reports 2024, golf IMU study).
    """
    if len(positions) < 3:
        return np.array([0.0, 0.0, 1.0])
    pts = np.array([[p['x'], p['y'], p['z']]
                    for p in positions
                    if p is not None])
    if len(pts) < 3:
        return np.array([0.0, 0.0, 1.0])
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    try:
        _, _, Vt = np.linalg.svd(centered)
        n_hat = Vt[-1]
        norm = np.linalg.norm(n_hat)
        if norm < 1e-9:
            return np.array([0.0, 0.0, 1.0])
        return n_hat / norm
    except Exception:
        return np.array([0.0, 0.0, 1.0])


def compute_spine_axis(lm_shoulder_mid, lm_hip_mid):
    """
    Compute spine axis unit vector pointing from
    hip midpoint to shoulder midpoint.
    Both args are dicts with x, y, z keys.
    Returns np.array [x, y, z].
    """
    vec = np.array([
        lm_shoulder_mid['x'] - lm_hip_mid['x'],
        lm_shoulder_mid['y'] - lm_hip_mid['y'],
        lm_shoulder_mid['z'] - lm_hip_mid['z']
    ])
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return np.array([0.0, 1.0, 0.0])
    return vec / norm


def inplane_speed(pos_curr, pos_prev, n_hat):
    """
    Compute wrist speed projected onto swing plane.
    pos_curr, pos_prev: dicts with x, y, z.
    n_hat: swing plane normal (np.array).
    Returns scalar in-plane speed (float).
    Removes out-of-plane component (Z arc reversal
    artifact) so only true arc motion contributes.
    """
    def project(p):
        v = np.array([p['x'], p['y'], p['z']])
        return v - np.dot(v, n_hat) * n_hat

    p1 = project(pos_curr)
    p0 = project(pos_prev)
    return float(np.linalg.norm(p1 - p0))


def segment_rotation(lm_a, lm_b, ref_vec, s_hat):
    """
    Compute rotation of segment a->b around spine axis
    relative to a reference vector ref_vec.
    All args are np.arrays [x, y, z].
    Returns angle in degrees (unsigned, 0-180).
    Projects both vectors onto plane perpendicular
    to spine axis before measuring angle — removes
    spine tilt noise from rotation measurement.
    """
    def project_perp(v):
        return v - np.dot(v, s_hat) * s_hat

    seg = np.array([lm_b['x'] - lm_a['x'],
                    lm_b['y'] - lm_a['y'],
                    lm_b['z'] - lm_a['z']])
    seg_perp = project_perp(seg)
    ref_perp = project_perp(ref_vec)
    seg_norm = np.linalg.norm(seg_perp)
    ref_norm = np.linalg.norm(ref_perp)
    if seg_norm < 1e-9 or ref_norm < 1e-9:
        return 0.0
    cos_a = np.dot(seg_perp, ref_perp) / (seg_norm * ref_norm)
    cos_a = float(np.clip(cos_a, -1.0, 1.0))
    return degrees(acos(cos_a))


@app.route('/phases', methods=['POST'])
def detect_phases():
    data   = request.json
    frames = data.get('frames', [])
    fps    = data.get('fps', 30.0)

    pose_frames = [(f['frame'], f) for f in frames if f.get('poses')]
    if not pose_frames:
        return jsonify({'error': 'No pose frames'}), 400

    club_lookup = {}
    for f in frames:
        if f.get('club_head'):
            club_lookup[f['frame']] = f['club_head']

    print(f"[PHASES] Club head detected in {len(club_lookup)}/{len(frames)} frames")

    def lm3(frame, jid):
        try:
            return frame['poses'][0]['landmarks_3d'][str(jid)]
        except Exception:
            return None

    def normalize(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    def proj_plane(p, n_hat):
        p_arr = np.array([p['x'], p['y'], p['z']])
        return p_arr - np.dot(p_arr, n_hat) * n_hat

    def angle_3pt(a, b, c):
        v1 = np.array([a['x']-b['x'], a['y']-b['y'], a['z']-b['z']])
        v2 = np.array([c['x']-b['x'], c['y']-b['y'], c['z']-b['z']])
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        return degrees(acos(np.clip(cos_a, -1, 1)))

    def gaussian_smooth(arr, sigma):
        ks = sigma * 2 + 1
        xk = np.arange(ks) - ks // 2
        g  = np.exp(-xk**2 / (2 * sigma**2))
        g /= g.sum()
        if len(arr) >= ks:
            return np.convolve(arr, g, mode='valid'), ks // 2
        return np.array(arr), 0

    # Trail wrist Y series (joint 16) — used for P1 detection
    rw = [
        (fi, f['poses'][0]['landmarks_3d'].get('16', {}).get('y', 0), f)
        for fi, f in pose_frames
        if f['poses'] and f['poses'][0]['landmarks_3d'].get('16')
    ]

    if not rw:
        return jsonify({'error': 'No wrist data'}), 400

    # P1 — Address: first stable 10-frame window (trail wrist lm16 world-Y)
    p1_frame = rw[0][2]
    p1_idx   = rw[0][0]
    for i in range(len(rw) - 10):
        window = [rw[i + j][1] for j in range(10)]
        if max(window) - min(window) < 0.005:
            p1_frame = rw[i][2]
            p1_idx   = rw[i][0]
            break

    # Foundational math — compute once before phase detection
    # SVD swing plane from all lm15 world positions
    lm15_positions = []
    for fi, f in pose_frames:
        lm15 = lm3(f, 15)
        if lm15:
            lm15_positions.append([lm15['x'], lm15['y'], lm15['z']])

    if len(lm15_positions) >= 3:
        lm15_arr = np.array(lm15_positions)
        centroid = lm15_arr.mean(axis=0)
        _, _, Vt = np.linalg.svd(lm15_arr - centroid)
        n_hat    = Vt[-1]
        norm     = np.linalg.norm(n_hat)
        n_hat    = n_hat / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
    else:
        n_hat = np.array([0.0, 0.0, 1.0])

    # Canonicalize: flip n_hat if mean early-backswing omega is negative
    early_bs_pts = []
    for fi, f in pose_frames:
        if fi < p1_idx:
            continue
        lm15 = lm3(f, 15)
        if lm15:
            early_bs_pts.append(np.array([lm15['x'], lm15['y'], lm15['z']]))
        if len(early_bs_pts) >= 20:
            break

    if len(early_bs_pts) >= 3:
        omegas = []
        for i in range(1, len(early_bs_pts) - 1):
            v1 = early_bs_pts[i]   - early_bs_pts[i-1]
            v2 = early_bs_pts[i+1] - early_bs_pts[i]
            omega_vec = np.cross(v1, v2)
            omegas.append(np.dot(omega_vec, n_hat))
        if omegas and np.mean(omegas) < 0:
            n_hat = -n_hat

    # Spine axis s_hat = normalize(shoulder_mid - hip_mid)
    lm_ls_p1 = lm3(p1_frame, 11)
    lm_rs_p1 = lm3(p1_frame, 12)
    lh1      = lm3(p1_frame, 23)
    rh1      = lm3(p1_frame, 24)

    if lm_ls_p1 and lm_rs_p1 and lh1 and rh1:
        sh_mid_p1 = np.array([(lm_ls_p1['x']+lm_rs_p1['x'])/2,
                               (lm_ls_p1['y']+lm_rs_p1['y'])/2,
                               (lm_ls_p1['z']+lm_rs_p1['z'])/2])
        hp_mid_p1 = np.array([(lh1['x']+rh1['x'])/2,
                               (lh1['y']+rh1['y'])/2,
                               (lh1['z']+rh1['z'])/2])
        s_hat = normalize(sh_mid_p1 - hp_mid_p1)
    else:
        s_hat = np.array([0.0, 1.0, 0.0])

    # Reference vectors at P1
    hip_ref_vec = None
    sh_ref_vec  = None
    if lh1 and rh1:
        hip_ref_vec = np.array([rh1['x']-lh1['x'],
                                 rh1['y']-lh1['y'],
                                 rh1['z']-lh1['z']])
    if lm_ls_p1 and lm_rs_p1:
        sh_ref_vec  = np.array([lm_rs_p1['x']-lm_ls_p1['x'],
                                 lm_rs_p1['y']-lm_ls_p1['y'],
                                 lm_rs_p1['z']-lm_ls_p1['z']])

    print(f"[PHASES] n_hat={n_hat.round(3).tolist()} s_hat={s_hat.round(3).tolist()}", flush=True)

    post_p1 = [(fi, wy, fr) for fi, wy, fr in rw if fi > p1_idx]

    # P3 — Halfway Back
    # wrist_rise >= 0.15 and elbow_fold >= 15 degrees
    p3_idx, p3_frame = None, None
    lm16_p1 = lm3(p1_frame, 16)
    lm15_p1 = lm3(p1_frame, 15)
    lm13_p1 = lm3(p1_frame, 13)
    lm11_p1 = lm3(p1_frame, 11)

    p1_wrist_y   = lm16_p1['y'] if lm16_p1 else None
    p1_elbow_ang = (angle_3pt(lm15_p1, lm13_p1, lm11_p1)
                    if (lm15_p1 and lm13_p1 and lm11_p1) else None)

    if p1_wrist_y is not None and p1_elbow_ang is not None:
        for fi, wy, fr in post_p1:
            lm16 = lm3(fr, 16)
            lm15 = lm3(fr, 15)
            lm13 = lm3(fr, 13)
            lm11 = lm3(fr, 11)
            if not (lm16 and lm15 and lm13 and lm11):
                continue
            wrist_rise = (p1_wrist_y - lm16['y']) / (abs(p1_wrist_y) + 1e-9)
            elbow_fold = p1_elbow_ang - angle_3pt(lm15, lm13, lm11)
            if wrist_rise >= 0.15 and elbow_fold >= 15:
                p3_frame = fr
                p3_idx   = fi
                print(f"[PHASES] P3 halfway back: frame={p3_idx} "
                      f"wrist_rise={wrist_rise:.3f} elbow_fold={elbow_fold:.1f}",
                      flush=True)
                break

    if p3_idx is None:
        print("[PHASES] P3 not detected", flush=True)

    # P4 — Top of Backswing (Approach 12)
    # Lead wrist (lm15) in-plane speed argmin, smoothed sigma=5
    p4_idx, p4_frame = None, None
    p3_start      = p3_idx if p3_idx is not None else p1_idx
    post_p3_poses = [(fi, fr) for fi, fr in pose_frames if fi > p3_start]

    p4_speeds  = []
    p4_ents    = []
    lm15_prev  = None
    proj_prev  = None

    for fi, fr in post_p3_poses:
        lm15_curr = lm3(fr, 15)
        if lm15_curr is None:
            lm15_prev = None
            proj_prev = None
            continue
        if lm15_prev is not None and proj_prev is not None:
            p_curr = proj_plane(lm15_curr, n_hat)
            v_x    = p_curr[0] - proj_prev[0]
            v_y    = lm15_curr['y'] - lm15_prev['y']
            v_z    = p_curr[2] - proj_prev[2]
            p4_speeds.append(sqrt(v_x**2 + v_y**2 + v_z**2))
            p4_ents.append((fi, fr))
        lm15_prev = lm15_curr
        proj_prev = proj_plane(lm15_curr, n_hat)

    if len(p4_speeds) >= 3:
        sp_arr        = np.array(p4_speeds)
        sp_sm, sp_off = gaussian_smooth(sp_arr, sigma=5)
        p4_raw_i      = int(np.argmin(sp_sm))
        p4_ent_i      = min(p4_raw_i + sp_off, len(p4_ents) - 1)
        p4_idx        = p4_ents[p4_ent_i][0]
        p4_frame      = p4_ents[p4_ent_i][1]
        print(f"[PHASES] P4 top of backswing: frame={p4_idx} "
              f"speed={sp_sm[p4_raw_i]:.5f}", flush=True)
    else:
        print("[PHASES] P4 not detected — insufficient frames", flush=True)

    # P7 — Impact (YOLO — intentional, critical)
    # Primary: argmin(club_head ↔ golf_ball distance) over frames post-P4
    p7_idx, p7_frame, p7_method = None, None, None

    if len(club_lookup) > 5:
        min_dist = float('inf')
        for f in frames:
            if not f.get('club_head') or not f.get('golf_ball'):
                continue
            frame_num = f['frame']
            if p4_idx and frame_num <= p4_idx:
                continue
            ch   = f['club_head']
            gb   = f['golf_ball']
            dist = sqrt((ch['x'] - gb['x'])**2 + (ch['y'] - gb['y'])**2)
            if dist < min_dist:
                min_dist = dist
                p7_frame = f
                for fi, wy, fr in rw:
                    if fr['frame'] == frame_num:
                        p7_idx = fi
                        break
        if p7_frame:
            p7_method = 'yolo_proximity'
            print(f"[PHASES] P7 from ball proximity: "
                  f"frame {p7_frame['frame']} dist={min_dist:.4f}", flush=True)

    # Fallback: max wrist velocity post-P4
    if p7_idx is None:
        post_p4_rw = [(fi, wy, fr) for fi, wy, fr in rw if fi > (p4_idx or 0)]
        if len(post_p4_rw) > 1:
            velocities = [
                (abs(post_p4_rw[i][1] - post_p4_rw[i-1][1]),
                 post_p4_rw[i][0], post_p4_rw[i][2])
                for i in range(1, len(post_p4_rw))
            ]
            velocities.sort(reverse=True)
            p7_idx    = velocities[0][1]
            p7_frame  = velocities[0][2]
            p7_method = 'wrist_velocity_fallback'
            print(f"[PHASES] P7 fallback wrist velocity: "
                  f"frame {p7_frame['frame']}", flush=True)

    # P2 — Takeaway (estimated after P4 confirmed)
    p2_idx, p2_frame = None, None
    if p4_idx is not None:
        p2_idx   = p1_idx + round((p4_idx - p1_idx) * 0.25)
        p2_frame = next((fr for fi, fr in pose_frames if fi >= p2_idx), None)
        print(f"[PHASES] P2 estimated: frame={p2_idx}", flush=True)

    # P5 — Transition: hip rotation peak
    # First frame where d_hip_angle (smoothed sigma=3) goes from >= 0 to < 0
    p5_idx, p5_frame = None, None
    post_p4_poses    = [(fi, fr) for fi, fr in pose_frames if fi > (p4_idx or 0)]
    hip_angles       = []
    hip_ents         = []

    for fi, fr in post_p4_poses:
        lm23 = lm3(fr, 23)
        lm24 = lm3(fr, 24)
        if not (lm23 and lm24) or hip_ref_vec is None:
            continue
        seg      = np.array([lm24['x']-lm23['x'],
                              lm24['y']-lm23['y'],
                              lm24['z']-lm23['z']])
        seg_perp = seg - np.dot(seg, s_hat) * s_hat
        ref_perp = hip_ref_vec - np.dot(hip_ref_vec, s_hat) * s_hat
        cos_a    = np.dot(seg_perp, ref_perp) / (
                       np.linalg.norm(seg_perp) * np.linalg.norm(ref_perp) + 1e-9)
        hip_angles.append(degrees(acos(np.clip(cos_a, -1, 1))))
        hip_ents.append((fi, fr))

    if len(hip_angles) >= 3:
        ha_arr        = np.array(hip_angles)
        ha_sm, ha_off = gaussian_smooth(ha_arr, sigma=3)
        d_ha          = np.diff(ha_sm)
        for i in range(1, len(d_ha)):
            if d_ha[i-1] >= 0 and d_ha[i] < 0:
                raw_i    = min(i + ha_off, len(hip_ents) - 1)
                p5_idx   = hip_ents[raw_i][0]
                p5_frame = hip_ents[raw_i][1]
                print(f"[PHASES] P5 transition: frame={p5_idx}", flush=True)
                break

    if p5_idx is None:
        print("[PHASES] P5 not detected — no hip angle reversal after P4", flush=True)

    # P6 — Pre-Impact (estimated after P7 confirmed)
    p6_idx, p6_frame = None, None
    if p4_idx is not None and p7_idx is not None:
        p6_idx   = p4_idx + round((p7_idx - p4_idx) * 0.75)
        p6_frame = next((fr for fi, fr in pose_frames if fi >= p6_idx), None)
        print(f"[PHASES] P6 estimated: frame={p6_idx}", flush=True)

    print(f"[PHASES] P1={p1_idx} P2={p2_idx} P3={p3_idx} P4={p4_idx} "
          f"P5={p5_idx} P6={p6_idx} P7={p7_idx} | "
          f"n_hat={n_hat.round(3).tolist()}", flush=True)

    def pr(label, fr, fi, detected=True):
        if fr is None:
            return {'label': label, 'frame': None, 'time_s': None, 'detected': False}
        return {'label': label, 'frame': fi,
                'time_s': round(fi / fps, 3), 'detected': detected}

    # ── Per-phase metric computation ───────────────────────────────────────────
    def _metrics(fr, phase=None):
        m = {}
        if fr is None:
            return m

        ls  = lm3(fr, 11); rs  = lm3(fr, 12)
        lh  = lm3(fr, 23); rh  = lm3(fr, 24)
        le  = lm3(fr, 13); te  = lm3(fr, 14)
        lw  = lm3(fr, 15); tw  = lm3(fr, 16)
        lk  = lm3(fr, 25); rk  = lm3(fr, 26)
        la  = lm3(fr, 27); ra  = lm3(fr, 28)

        if ls and rs and sh_ref_vec is not None:
            m['shoulder_turn'] = round(segment_rotation(ls, rs, sh_ref_vec, s_hat), 1)
        if lh and rh and hip_ref_vec is not None:
            m['hip_turn'] = round(segment_rotation(lh, rh, hip_ref_vec, s_hat), 1)
        if 'shoulder_turn' in m and 'hip_turn' in m:
            m['x_factor'] = round(m['shoulder_turn'] - m['hip_turn'], 1)
        if rs and te and tw:
            v = angle_3pt(rs, te, tw)
            if v is not None: m['trail_elbow'] = round(v, 1)
        if ls and le and lw:
            v = angle_3pt(ls, le, lw)
            if v is not None: m['lead_elbow'] = round(v, 1)
        if lh and lk and la:
            v = angle_3pt(lh, lk, la)
            if v is not None: m['lead_knee'] = round(v, 1)
        if rh and rk and ra:
            v = angle_3pt(rh, rk, ra)
            if v is not None: m['trail_knee'] = round(v, 1)

        if phase == 'P1' and rh and rk and ra:
            v = angle_3pt(rh, rk, ra)
            if v is not None:
                flex = round(180.0 - v, 1)
                m['trail_knee_flex_p1'] = flex
                m['trail_knee_status'] = 'pass' if 12.0 <= flex <= 35.0 else 'fail'

        if ls and rs and lh and rh:
            sh_mid = np.array([(ls['x']+rs['x'])/2,
                               (ls['y']+rs['y'])/2,
                               (ls['z']+rs['z'])/2])
            hp_mid = np.array([(lh['x']+rh['x'])/2,
                               (lh['y']+rh['y'])/2,
                               (lh['z']+rh['z'])/2])
            sv = sh_mid - hp_mid
            n  = float(np.linalg.norm(sv))
            if n > 1e-9:
                sv /= n
                m['forward_bend'] = round(
                    math.degrees(math.atan2(abs(float(sv[0])), abs(float(sv[1])))), 1
                )
                m['spine_angle_estimated'] = m['forward_bend']
                m['spine_angle_confidence'] = 'placeholder'

        ch_det  = fr.get('club_head')
        hdl_det = fr.get('club_handle')
        if ch_det and hdl_det:
            dx = abs(ch_det['x'] - hdl_det['x'])
            dy = abs(ch_det['y'] - hdl_det['y'])
            if dx > 0.005 or dy > 0.005:
                m['shaft_angle'] = round(math.degrees(math.atan2(dy, dx)), 1)
                if phase == 'P1':
                    sa = abs(m['shaft_angle'])
                    m['shaft_angle_p1'] = sa
                    m['shaft_angle_status'] = 'pass' if 2.0 <= sa <= 10.0 else 'fail'

        return m

    phase_metrics = {
        'P1': _metrics(p1_frame, phase='P1'),
        'P2': _metrics(p2_frame, phase='P2'),
        'P3': _metrics(p3_frame, phase='P3'),
        'P4': _metrics(p4_frame, phase='P4'),
        'P5': _metrics(p5_frame, phase='P5'),
        'P6': _metrics(p6_frame, phase='P6'),
        'P7': _metrics(p7_frame, phase='P7'),
    }
    print(f"[METRICS] P1={phase_metrics['P1']} P4={phase_metrics['P4']}", flush=True)

    freeze_frames = [
        fi for label, fi in [
            ('P1', p1_idx), ('P3', p3_idx),
            ('P4', p4_idx), ('P5', p5_idx),
            ('P7', p7_idx),
        ] if fi is not None
    ]

    p7_result          = pr('Impact', p7_frame, p7_idx)
    p7_result['method'] = p7_method

    return jsonify({
        'P1': pr('Address',    p1_frame, p1_idx),
        'P2': pr('Takeaway',   p2_frame, p2_idx, detected=False),
        'P3': pr('Halfway',    p3_frame, p3_idx),
        'P4': pr('Top',        p4_frame, p4_idx),
        'P5': pr('Transition', p5_frame, p5_idx),
        'P6': pr('Pre-Impact', p6_frame, p6_idx, detected=False),
        'P7': p7_result,
        'freeze_frames': freeze_frames,
        'metrics': phase_metrics,
    })


# ── React SPA fallback ────────────────────────────────────────────────────────
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    full = os.path.join(_static_folder, path)
    if path and os.path.exists(full):
        return send_from_directory(_static_folder, path)
    return send_from_directory(_static_folder, 'index.html')


@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    if not DATABASE_URL:
        return jsonify([])
    conn = get_db(); cur = conn.cursor()
    cur.execute("""
        SELECT id, golfer_name, video_filename, created_at
        FROM swing_sessions ORDER BY created_at DESC LIMIT 50
    """)
    rows = cur.fetchall()
    cur.close(); conn.close()
    return jsonify([{
        'id': str(r[0]), 'golfer_name': r[1],
        'video_filename': r[2], 'created_at': str(r[3]),
    } for r in rows])


# ── Notes API ─────────────────────────────────────────────────────────────────
@app.route('/api/notes', methods=['POST'])
def save_note():
    if not DATABASE_URL:
        return jsonify({'error': 'No database configured'}), 503
    data = request.json
    conn = get_db(); cur = conn.cursor()
    cur.execute("""
        INSERT INTO swing_notes
        (session_id, phase_key, note_text, note_type, screenshot_url)
        VALUES (%s,%s,%s,%s,%s) RETURNING id, created_at
    """, (
        data['session_id'], data.get('phase_key'),
        data.get('note_text', ''), data.get('note_type', 'text'),
        data.get('screenshot_url'),
    ))
    row = cur.fetchone()
    conn.commit(); cur.close(); conn.close()
    return jsonify({**data, 'id': str(row[0]), 'created_at': str(row[1])})


@app.route('/api/notes/<session_id>', methods=['GET'])
def get_notes(session_id):
    if not DATABASE_URL:
        return jsonify([])
    conn = get_db(); cur = conn.cursor()
    cur.execute("""
        SELECT id, phase_key, note_text, note_type, screenshot_url, created_at
        FROM swing_notes WHERE session_id=%s ORDER BY created_at
    """, (session_id,))
    rows = cur.fetchall()
    cur.close(); conn.close()
    return jsonify([{
        'id': str(r[0]), 'phase_key': r[1], 'note_text': r[2],
        'note_type': r[3], 'screenshot_url': r[4], 'created_at': str(r[5]),
    } for r in rows])


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
