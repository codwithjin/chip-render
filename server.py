from flask import (Flask, request, jsonify,
                   send_from_directory, Response)
import cv2, json, math
import os, tempfile, threading, uuid, time
import numpy as np
from math import sqrt, acos, atan2, degrees
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO

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

    t = threading.Thread(target=run_mediapipe,
                         args=(tmp.name, job_id),
                         daemon=True)
    t.start()
    return jsonify({'job_id': job_id})


def run_mediapipe(video_path, job_id):
    try:
        print(f"[MediaPipe] Starting job {job_id}", flush=True)
        print(f"[MediaPipe] Video: {video_path}", flush=True)

        model_path = MEDIAPIPE_MODEL_PATH
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            num_poses=1,
        )
        landmarker = mp_vision.PoseLandmarker.create_from_options(options)

        cap    = cv2.VideoCapture(video_path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Skip frames so we process at most ~30fps effective rate.
        # iPhone slo-mo at 120/240fps would otherwise process 4-8x more
        # frames than needed for phase detection.
        skip = max(1, round(fps / 30.0))
        frames_to_process = max(1, total // skip)
        print(f"[MediaPipe] FPS:{fps} Frames:{total} Size:{width}x{height} skip:{skip} effective:{frames_to_process}", flush=True)

        with jobs_lock:
            jobs[job_id]['total'] = frames_to_process

        frames_out = []
        frame_num  = 0
        processed  = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % skip != 0:
                frame_num += 1
                continue

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = landmarker.detect(mp_image)

            fd = {
                'frame':        frame_num,
                'timestamp_ms': int(frame_num / fps * 1000),
                'poses':        [],
            }

            if result.pose_landmarks and result.pose_world_landmarks:
                entry = {
                    'pose_index':   0,
                    'landmarks_2d': {},
                    'landmarks_3d': {},
                }
                for jid in JOINT_IDS:
                    lm2 = result.pose_landmarks[0][jid]
                    lm3 = result.pose_world_landmarks[0][jid]
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

            # YOLO detection — club head, handle, golf ball
            club_head = None
            club_handle = None
            golf_ball = None

            if yolo_model is not None:
                try:
                    yolo_results = yolo_model(
                        frame, verbose=False, conf=0.25)
                    if yolo_results:
                        for box in yolo_results[0].boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            cx = (x1 + x2) / 2
                            cy = (y1 + y2) / 2
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            names = yolo_results[0].names
                            label = names[cls].lower()
                            obj = {
                                'x': cx / width,
                                'y': cy / height,
                                'x2': x2 / width,
                                'y2': y2 / height,
                                'x_px': cx,
                                'y_px': cy,
                                'conf': conf
                            }
                            if 'head' in label:
                                club_head = obj
                            elif 'handle' in label:
                                club_handle = obj
                            elif 'ball' in label:
                                golf_ball = obj
                except Exception as e:
                    pass

            fd['club_head'] = club_head
            fd['club_handle'] = club_handle
            fd['golf_ball'] = golf_ball

            frames_out.append(fd)
            processed  += 1
            frame_num  += 1

            with jobs_lock:
                jobs[job_id]['progress'] = processed

        cap.release()
        landmarker.close()

        result = {
            'fps':          fps,
            'total_frames': total,
            'width':        width,
            'height':       height,
            'joints':       JOINT_IDS,
            'frames':       frames_out,
        }

        pose_frame_count = sum(1 for f in frames_out if f['poses'])
        print(f"[MediaPipe] Done. Pose frames: {pose_frame_count}/{processed} (skip={skip}, source={total})", flush=True)

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
        try:
            os.unlink(video_path)
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
    print(f"[Result] job_id={job_id} status={job.get('status') if job else 'NOT_FOUND'} result_is_none={job['result'] is None if job else 'N/A'}", flush=True)
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

    # Build club head position lookup by frame number
    club_head_by_frame = club_lookup

    print(f"[PHASES] Club head detected in "
          f"{len(club_lookup)}/{len(frames)} frames")

    def lm3(frame, jid):
        try:
            return frame['poses'][0]['landmarks_3d'][str(jid)]
        except Exception:
            return None

    def rot_xz(a, b):
        return math.degrees(math.atan2(b['z'] - a['z'], b['x'] - a['x']))

    def midpt(a, b):
        return {k: (a[k] + b[k]) / 2 for k in ['x', 'y', 'z']}

    def dist3d(a, b):
        return math.sqrt(sum((a[k] - b[k]) ** 2 for k in ['x', 'y', 'z']))

    def angle_3pt(ja, jb, jc):
        ba = {k: ja[k] - jb[k] for k in ['x', 'y', 'z']}
        bc = {k: jc[k] - jb[k] for k in ['x', 'y', 'z']}
        dot    = sum(ba[k] * bc[k] for k in ['x', 'y', 'z'])
        mag_ba = math.sqrt(sum(ba[k] ** 2 for k in ['x', 'y', 'z']))
        mag_bc = math.sqrt(sum(bc[k] ** 2 for k in ['x', 'y', 'z']))
        if mag_ba * mag_bc == 0:
            return None
        return math.degrees(math.acos(max(-1, min(1, dot / (mag_ba * mag_bc)))))

    def angle_3pt_lm(frame, a, b, c):
        try:
            lm3d = frame['poses'][0]['landmarks_3d']
            ja = lm3d.get(str(a)); jb = lm3d.get(str(b)); jc = lm3d.get(str(c))
            if not all([ja, jb, jc]):
                return None
            return angle_3pt(ja, jb, jc)
        except Exception:
            return None

    # Trail wrist Y series (joint 16)
    rw = [
        (fi, f['poses'][0]['landmarks_3d'].get('16', {}).get('y', 0), f)
        for fi, f in pose_frames
        if f['poses'] and f['poses'][0]['landmarks_3d'].get('16')
    ]

    if not rw:
        return jsonify({'error': 'No wrist data'}), 400

    # P1 — first stable 10-frame window
    p1_frame = rw[0][2]
    p1_idx   = rw[0][0]
    for i in range(len(rw) - 10):
        window = [rw[i + j][1] for j in range(10)]
        if max(window) - min(window) < 0.005:
            p1_frame = rw[i][2]
            p1_idx   = rw[i][0]
            break

    lh1        = lm3(p1_frame, 23)
    rh1        = lm3(p1_frame, 24)
    hip_rot_p1 = rot_xz(lh1, rh1) if lh1 and rh1 else 0

    lm_ls_p1      = lm3(p1_frame, 11)
    lm_rs_p1      = lm3(p1_frame, 12)
    shoulder_rot_p1 = rot_xz(lm_ls_p1, lm_rs_p1) if lm_ls_p1 and lm_rs_p1 else 0

    p1_rw_y = next((wy for fi, wy, fr in rw if fi == p1_idx), rw[0][1])
    rw_y_p1 = rw[0][1]
    post_p1 = [(fi, wy, fr) for fi, wy, fr in rw if fi > p1_idx]

    # Swing plane normal + spine axis from P1 landmarks
    post_p1_wrist_pos = [lm3(fr, 16) for fi, wy, fr in rw if fi >= p1_idx]
    n_hat = compute_swing_plane([p for p in post_p1_wrist_pos if p is not None])

    if all([lm_ls_p1, lm_rs_p1, lh1, rh1]):
        sh_mid_p1 = {k: (lm_ls_p1[k] + lm_rs_p1[k]) / 2 for k in ['x', 'y', 'z']}
        hp_mid_p1 = {k: (lh1[k] + rh1[k]) / 2 for k in ['x', 'y', 'z']}
        s_hat = compute_spine_axis(sh_mid_p1, hp_mid_p1)
    else:
        s_hat = np.array([0.0, 1.0, 0.0])

    hip_ref_vec = None; sh_ref_vec = None
    if lh1 and rh1:
        hip_ref_vec = np.array([rh1['x'] - lh1['x'],
                                rh1['y'] - lh1['y'],
                                rh1['z'] - lh1['z']])
    if lm_ls_p1 and lm_rs_p1:
        sh_ref_vec  = np.array([lm_rs_p1['x'] - lm_ls_p1['x'],
                                lm_rs_p1['y'] - lm_ls_p1['y'],
                                lm_rs_p1['z'] - lm_ls_p1['z']])
    print(f"[PHASES] n_hat={n_hat.round(3).tolist()} s_hat={s_hat.round(3).tolist()}", flush=True)

    # ── Gaussian smooth helper (shared by P2/P3/P5/P6) ──────────────
    def _sm(arr, sigma):
        ks  = sigma * 2 + 1
        xk  = np.arange(ks) - ks // 2
        g   = np.exp(-xk**2 / (2 * sigma**2))
        g  /= g.sum()
        if len(arr) >= ks:
            return np.convolve(arr, g, mode='valid'), ks // 2
        return np.array(arr), 0

    # P2 — Club shaft parallel to floor (takeaway)
    # Signal: |club_head.y − club_handle.y| in normalised 2D image coords.
    # Y increases downward in image; shaft horizontal = both at same Y.
    # First local minimum of smoothed |Δy| = shaft most horizontal.
    # Fallback: first frame hip rotation > 5° (takeaway started).
    p2_frame = p2_idx = None
    shaft_b_diffs, shaft_b_ents = [], []
    for fi, wy, fr in post_p1:
        ch  = fr.get('club_head')
        hdl = fr.get('club_handle')
        if not ch or not hdl:
            continue
        shaft_b_diffs.append(abs(ch['y'] - hdl['y']))
        shaft_b_ents.append((fi, fr))

    if len(shaft_b_diffs) >= 11:
        sb_sm, sb_off = _sm(np.array(shaft_b_diffs), sigma=5)
        dsb = np.diff(sb_sm)
        for i in range(1, len(dsb)):
            if dsb[i-1] < 0 and dsb[i] >= 0:
                raw_i    = min(i + sb_off, len(shaft_b_ents) - 1)
                p2_idx   = shaft_b_ents[raw_i][0]
                p2_frame = shaft_b_ents[raw_i][1]
                print(f"[PHASES] P2 club horizontal: frame={p2_idx}", flush=True)
                break

    # Fallback: hip rotation > 5°
    if p2_idx is None:
        for fi, wy, fr in post_p1:
            lh = lm3(fr, 23); rh = lm3(fr, 24)
            if lh and rh and hip_ref_vec is not None and \
               segment_rotation(lh, rh, hip_ref_vec, s_hat) > 5:
                p2_frame = fr; p2_idx = fi
                print(f"[PHASES] P2 fallback hip rotation: frame={p2_idx}", flush=True)
                break

    # P3 — Lead arm parallel to floor (backswing)
    # Signal: lm15_3d.y − lm11_3d.y  (world coords, Y increases downward)
    #   > 0  : wrist below shoulder  (address, early backswing)
    #   = 0  : arm horizontal        ← P3
    #   < 0  : wrist above shoulder  (arm rising past horizontal toward top)
    # Detect: first zero-crossing positive → negative.
    p3_idx, p3_frame = None, None
    arm_b_diffs, arm_b_ents = [], []
    for fi, wy, fr in post_p1:
        lm15 = lm3(fr, 15); lm11 = lm3(fr, 11)
        if not lm15 or not lm11:
            continue
        arm_b_diffs.append(lm15['y'] - lm11['y'])
        arm_b_ents.append((fi, fr))

    if len(arm_b_diffs) >= 7:
        ab_sm, ab_off = _sm(np.array(arm_b_diffs), sigma=3)
        for i in range(1, len(ab_sm)):
            if ab_sm[i-1] >= 0 and ab_sm[i] < 0:
                raw_i    = min(i + ab_off, len(arm_b_ents) - 1)
                p3_idx   = arm_b_ents[raw_i][0]
                p3_frame = arm_b_ents[raw_i][1]
                print(f"[PHASES] P3 arm horizontal: frame={p3_idx}", flush=True)
                break

    if p3_idx is None:
        print("[PHASES] P3 not detected — no arm zero-crossing", flush=True)

    # P4 — Top of backswing
    # Trail wrist (lm16) world-Y first local minimum after P3.
    # Smoothed with Gaussian σ=9. First valley = wrist peaks = top.
    # No window bound needed — first local min occurs before follow-through.
    # Validated: frame 222 on SPEITH_DRIVER.MP4 (ground truth 7.4s).

    p4_idx, p4_frame = None, None

    def _gauss_sm(arr, sigma):
        ks  = sigma * 2 + 1
        xk  = np.arange(ks) - ks // 2
        g   = np.exp(-xk**2 / (2 * sigma**2))
        g  /= g.sum()
        if len(arr) >= ks:
            return np.convolve(arr, g, mode='valid'), ks // 2
        return np.array(arr), 0

    # Build trail wrist Y series from P3 onward (rw already tracks lm16 Y)
    search_start = p3_idx or p1_idx
    post_p3_rw   = [(fi, wy, fr) for fi, wy, fr in rw if fi > search_start]

    if len(post_p3_rw) < 20:
        print("[P4] insufficient frames after P3", flush=True)
    else:
        ly_vals = np.array([wy for fi, wy, fr in post_p3_rw])
        ly_ents = [(fi, fr) for fi, wy, fr in post_p3_rw]

        ly_sm, ly_off = _gauss_sm(ly_vals, sigma=9)
        dy = np.diff(ly_sm)

        for i in range(1, len(dy)):
            if dy[i-1] < 0 and dy[i] >= 0:
                raw_i    = i + ly_off
                p4_idx   = ly_ents[min(raw_i, len(ly_ents)-1)][0]
                p4_frame = ly_ents[min(raw_i, len(ly_ents)-1)][1]
                print(f"[PHASES] P4 confirmed: frame={p4_idx} "
                      f"lm16_y={ly_sm[i]:.5f}",
                      flush=True)
                break

        if p4_idx is None:
            print("[PHASES] P4 not detected — no local Y minimum found",
                  flush=True)

    max_sh_rot = 0.0

    # P5 — Lead arm parallel to floor (downswing)
    # Signal: lm15_3d.y - lm11_3d.y
    # After P4 arm is above horizontal (signal < 0); at P5 it crosses 0 going positive
    p5_idx, p5_frame = None, None
    post_p4_poses = [(fi, wy, fr) for fi, wy, fr in rw if fi > (p4_idx or 0)]
    arm_d_diffs, arm_d_ents = [], []
    for fi, wy, fr in post_p4_poses:
        lm15 = lm3(fr, 15); lm11 = lm3(fr, 11)
        if not lm15 or not lm11: continue
        arm_d_diffs.append(lm15['y'] - lm11['y'])
        arm_d_ents.append((fi, fr))

    if len(arm_d_diffs) >= 7:
        ad_sm, ad_off = _sm(np.array(arm_d_diffs), sigma=3)
        for i in range(1, len(ad_sm)):
            if ad_sm[i-1] <= 0 and ad_sm[i] > 0:
                raw_i    = min(i + ad_off, len(arm_d_ents) - 1)
                p5_idx   = arm_d_ents[raw_i][0]
                p5_frame = arm_d_ents[raw_i][1]
                print(f"[PHASES] P5 arm horizontal: frame={p5_idx}", flush=True)
                break

    if p5_idx is None:
        print("[PHASES] P5 not detected — no arm zero-crossing after P4", flush=True)

    print(f"[PHASES] P5 frame: {p5_idx}", flush=True)

    # P7 — Impact: frame of minimum distance between club head and golf ball
    p7_idx, p7_frame = None, None

    if len(club_lookup) > 5:
        min_dist = float('inf')
        for f in frames:
            if not f.get('club_head') or not f.get('golf_ball'):
                continue
            frame_num = f['frame']
            if p4_idx and frame_num <= p4_idx:
                continue
            ch = f['club_head']
            gb = f['golf_ball']
            dist = ((ch['x'] - gb['x'])**2 +
                    (ch['y'] - gb['y'])**2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                p7_frame = f
                for fi, wy, fr in rw:
                    if fr['frame'] == frame_num:
                        p7_idx = fi
                        break

        if p7_frame:
            print(f"[PHASES] P7 from ball proximity: "
                  f"frame {p7_frame['frame']} dist={min_dist:.4f}", flush=True)

    # Fallback to wrist velocity if no ball detected
    if p7_idx is None:
        post_p4 = [(fi, wy, fr) for fi, wy, fr in rw if fi > (p4_idx or 0)]
        if len(post_p4) > 1:
            velocities = [
                (abs(post_p4[i][1] - post_p4[i-1][1]),
                 post_p4[i][0], post_p4[i][2])
                for i in range(1, len(post_p4))
            ]
            velocities.sort(reverse=True)
            p7_idx = velocities[0][1]
            p7_frame = velocities[0][2]
            print(f"[PHASES] P7 fallback wrist velocity: "
                  f"frame {p7_frame['frame']}", flush=True)

    # P6 — Club shaft parallel to floor (downswing)
    # Signal: |club_head.y - club_handle.y| YOLO 2D; argmin in (P5, P7) window.
    # argmin (not first local min) avoids edge-of-window noise artefacts.
    p6_idx, p6_frame = None, None
    search_start_p6 = p5_idx or p4_idx or 0
    search_end_p6   = p7_idx if p7_idx is not None else float('inf')
    shaft_d_diffs, shaft_d_ents = [], []
    for f in sorted(frames, key=lambda x: x['frame']):
        fn = f['frame']
        if fn <= search_start_p6: continue
        if fn >= search_end_p6:   break
        ch  = f.get('club_head')
        hdl = f.get('club_handle')
        if not ch or not hdl: continue
        shaft_d_diffs.append(abs(ch['y'] - hdl['y']))
        shaft_d_ents.append((fn, f))

    if len(shaft_d_diffs) >= 3:
        raw_i    = int(np.argmin(shaft_d_diffs))
        fn6      = shaft_d_ents[raw_i][0]
        p6_frame = shaft_d_ents[raw_i][1]
        for fi, wy, fr in rw:
            if fr['frame'] == fn6:
                p6_idx = fi; break
        print(f"[PHASES] P6 club horizontal (argmin): frame={p6_idx}", flush=True)

    # Fallback: wrist returns to shoulder Y level on downswing (P5 → P7)
    if p6_idx is None:
        fb_start = p5_idx or p4_idx or 0
        for fi, wy, fr in rw:
            if fi <= fb_start: continue
            if p7_idx is not None and fi >= p7_idx: break
            ls = lm3(fr, 11); rs = lm3(fr, 12)
            if ls and rs:
                smy = (ls['y'] + rs['y']) / 2
                if wy > smy:
                    p6_frame = fr; p6_idx = fi; break
        if p6_idx:
            print(f"[PHASES] P6 fallback wrist-shoulder: frame={p6_idx}", flush=True)

    if p6_idx is None:
        print("[PHASES] P6 not detected", flush=True)

    print(f"[PHASES] P6 frame: {p6_idx}", flush=True)

    print(f"[PHASES] P1={p1_idx} P2={p2_idx} P3={p3_idx} P4={p4_idx} P5={p5_idx} P6={p6_idx} P7={p7_idx} | n_hat={n_hat.round(3).tolist()}", flush=True)

    def pr(label, fr, fi):
        if fr is None:
            return {'label': label, 'frame': None, 'time_s': None, 'detected': False}
        return {'label': label, 'frame': fi,
                'time_s': round(fi / fps, 3), 'detected': True}

    # ── Per-phase metric computation ───────────────────────────────────────────
    def _metrics(fr, phase=None):
        """Compute measured biomechanical metrics from a phase frame dict."""
        m = {}
        if fr is None:
            return m

        ls  = lm3(fr, 11); rs  = lm3(fr, 12)   # lead/trail shoulder
        lh  = lm3(fr, 23); rh  = lm3(fr, 24)   # lead/trail hip
        le  = lm3(fr, 13); te  = lm3(fr, 14)   # lead/trail elbow
        lw  = lm3(fr, 15); tw  = lm3(fr, 16)   # lead/trail wrist
        lk  = lm3(fr, 25); rk  = lm3(fr, 26)   # lead/trail knee
        la  = lm3(fr, 27); ra  = lm3(fr, 28)   # lead/trail ankle

        # Shoulder turn (degrees from P1 reference, projected onto transverse plane)
        if ls and rs and sh_ref_vec is not None:
            m['shoulder_turn'] = round(segment_rotation(ls, rs, sh_ref_vec, s_hat), 1)

        # Hip turn
        if lh and rh and hip_ref_vec is not None:
            m['hip_turn'] = round(segment_rotation(lh, rh, hip_ref_vec, s_hat), 1)

        # X-Factor (shoulder turn minus hip turn)
        if 'shoulder_turn' in m and 'hip_turn' in m:
            m['x_factor'] = round(m['shoulder_turn'] - m['hip_turn'], 1)

        # Trail elbow angle (trail shoulder – trail elbow – trail wrist)
        if rs and te and tw:
            v = angle_3pt(rs, te, tw)
            if v is not None: m['trail_elbow'] = round(v, 1)

        # Lead elbow angle (lead shoulder – lead elbow – lead wrist)
        if ls and le and lw:
            v = angle_3pt(ls, le, lw)
            if v is not None: m['lead_elbow'] = round(v, 1)

        # Lead knee angle (lead hip – lead knee – lead ankle)
        if lh and lk and la:
            v = angle_3pt(lh, lk, la)
            if v is not None: m['lead_knee'] = round(v, 1)

        # Trail knee angle (trail hip – trail knee – trail ankle)
        if rh and rk and ra:
            v = angle_3pt(rh, rk, ra)
            if v is not None: m['trail_knee'] = round(v, 1)

        # P1 only — trail knee flex and status (DTL; trail = right side for RH golfer)
        # Flex = 180° − joint angle (0° = straight leg, 25° = flexed).
        # Internal pass: 12–35°  |  Display target: 15–30°  |  Fail: <12° or >35°
        if phase == 'P1' and rh and rk and ra:
            v = angle_3pt(rh, rk, ra)
            if v is not None:
                flex = round(180.0 - v, 1)
                m['trail_knee_flex_p1'] = flex
                m['trail_knee_status'] = 'pass' if 12.0 <= flex <= 35.0 else 'fail'

        # Forward bend — sagittal-plane forward tilt of spine from vertical.
        # Uses Z component of normalised hip→shoulder vector in MediaPipe world coords.
        # Z = depth axis (toward/away from camera); Y = vertical (negative = up).
        # forward_bend = atan2(|sv_z|, |sv_y|)
        # Coaching benchmark (down-the-line, 2D): 32-45°.
        # Note: face-on camera underestimates this value vs down-the-line measurement.
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

        # Shaft angle — angle of club shaft from horizontal (face-on view).
        # Driver avg ~55°, irons ~60-65°. Benchmark 55-65°.
        # Computed from YOLO 2D: atan2(|dy|, |dx|).
        # P1 only: shaft_angle_p1 = abs value; pass 2–10°, fail <2° or >12°.
        # All other phases: magnitude-only behaviour unchanged.
        ch_det  = fr.get('club_head')
        hdl_det = fr.get('club_handle')
        if ch_det and hdl_det:
            dx = abs(ch_det['x'] - hdl_det['x'])
            dy = abs(ch_det['y'] - hdl_det['y'])
            if dx > 0.005 or dy > 0.005:
                m['shaft_angle'] = round(math.degrees(math.atan2(dy, dx)), 1)
                # P1 only — absolute shaft angle with pass/fail status.
                # Pass: 2–10°  |  Fail: <2° or >12°
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
        fi for _, fi in [
            ('P1', p1_idx), ('P3', p3_idx),
            ('P4', p4_idx), ('P5', p5_idx),
            ('P6', p6_idx), ('P7', p7_idx),
        ] if fi is not None
    ]

    return jsonify({
        'P1': pr('Address',    p1_frame, p1_idx),
        'P2': pr('Takeaway',   p2_frame, p2_idx),
        'P3': pr('Halfway',    p3_frame, p3_idx),
        'P4': pr('Top',        p4_frame, p4_idx),
        'P5': pr('Transition', p5_frame, p5_idx),
        'P6': pr('Pre-Impact', p6_frame, p6_idx),
        'P7': pr('Impact',     p7_frame, p7_idx),
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
