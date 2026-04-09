from flask import (Flask, request, jsonify,
                   send_from_directory, Response)
import cv2, json, math
import os, tempfile, threading, uuid, time
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

DATABASE_URL = os.environ.get('DATABASE_URL')

def get_db():
    import psycopg2
    return psycopg2.connect(DATABASE_URL)

JOINT_IDS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
MAX_FILE_MB = 500
jobs = {}           # job_id → job dict
jobs_lock = threading.Lock()

YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'golf_driver_v2_best.pt')
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

        model_path = os.path.join(BASE_DIR, 'pose_landmarker_heavy.task')
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

        print(f"[MediaPipe] FPS:{fps} Frames:{total} Size:{width}x{height}", flush=True)

        with jobs_lock:
            jobs[job_id]['total'] = total

        frames_out = []
        frame_num  = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

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
            frame_num += 1

            with jobs_lock:
                jobs[job_id]['progress'] = frame_num

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
        print(f"[MediaPipe] Done. Pose frames: {pose_frame_count}/{frame_num}", flush=True)

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

    # P2 — hip rotation > 5°
    p2_frame = p2_idx = None
    for fi, wy, fr in post_p1:
        lh = lm3(fr, 23)
        rh = lm3(fr, 24)
        if lh and rh and abs(rot_xz(lh, rh) - hip_rot_p1) > 5:
            p2_frame = fr
            p2_idx   = fi
            break

    # P3 — wrist risen 15%+ from P1 AND trail elbow folded 15°+
    p3_idx, p3_frame = None, None
    p1_re_angle = angle_3pt_lm(p1_frame, 12, 14, 16)

    for fi, wy, fr in post_p1:
        if fi <= p1_idx:
            continue
        wrist_rise = p1_rw_y - wy   # positive = risen (Y inverted in 3D)
        if wrist_rise < 0.15:
            continue
        re_angle = angle_3pt_lm(fr, 12, 14, 16)
        if re_angle is None:
            continue
        if p1_re_angle is not None and (p1_re_angle - re_angle) >= 15:
            p3_idx = fi; p3_frame = fr; break

    # Fallback: wrist rise threshold only
    if p3_idx is None:
        for fi, wy, fr in post_p1:
            if fi <= p1_idx:
                continue
            if (p1_rw_y - wy) >= 0.20:
                p3_idx = fi; p3_frame = fr; break

    # P4 — frame of maximum trail shoulder rotation from P1
    # Search only frames after P3 (or post_p1 if P3 not found)
    p4_idx, p4_frame = None, None
    max_shoulder_rot = 0
    p4_search_start = p3_idx if p3_idx else p1_idx
    post_p3 = [(fi, wy, fr) for fi, wy, fr in rw if fi > p4_search_start]

    # P4 using club head (more accurate than shoulder rotation)
    if len(club_lookup) > 10:
        # Find frame with minimum club_head Y after P3
        # (highest point = minimum Y in image coords)
        post_p3_club = [
            (fi, club_lookup[fr['frame']]['y'], fr)
            for fi, wy, fr in rw
            if fi > (p3_idx or p1_idx)
            and fr['frame'] in club_lookup
        ]
        if len(post_p3_club) > 5:
            best_y = 1.0
            frames_since_peak = 0
            for i, (fi, cy, fr) in enumerate(post_p3_club):
                if cy < best_y:
                    best_y = cy
                    p4_idx = fi
                    p4_frame = fr
                    frames_since_peak = 0
                else:
                    frames_since_peak += 1
                    if frames_since_peak > 10:
                        break
            print(f"[PHASES] P4 from club head: "
                  f"frame {p4_frame['frame'] if p4_frame else None}, y={best_y:.3f}")

    # Body-landmark P4: maximum trail shoulder rotation (fallback)
    if p4_idx is None:
        for fi, wy, fr in post_p3:
            lm_ls = lm3(fr, 11)
            lm_rs = lm3(fr, 12)
            if not lm_ls or not lm_rs:
                continue
            shoulder_rot = abs(rot_xz(lm_ls, lm_rs) - shoulder_rot_p1)
            if shoulder_rot > max_shoulder_rot:
                max_shoulder_rot = shoulder_rot
                p4_idx   = fi
                p4_frame = fr
            elif p4_idx and (fi - p4_idx) > 10:
                break  # 10+ frame sustained decrease = past the top

    # Fallback: wrist Y minimum after P3
    if p4_idx is None:
        fallback = [(fi, wy, fr) for fi, wy, fr in post_p3]
        if fallback:
            p4_idx, _, p4_frame = min(fallback, key=lambda x: x[1])

    # hr_p4 needed by P5
    if p4_frame:
        lh4   = lm3(p4_frame, 23)
        rh4   = lm3(p4_frame, 24)
        hr_p4 = abs(rot_xz(lh4, rh4) - hip_rot_p1) if lh4 and rh4 else 0
    else:
        hr_p4 = 0

    # P5 — first frame after P4 where hip rotation RATE exceeds shoulder rotation RATE
    # Hips accelerating faster than shoulders = transition has started
    p5_idx, p5_frame = None, None
    if p4_idx:
        post_p4       = [(fi, wy, fr) for fi, wy, fr in rw if fi > p4_idx]
        prev_hip_rot  = hr_p4
        prev_sh_rot   = max_shoulder_rot

        for i in range(1, len(post_p4)):
            fi, wy, fr         = post_p4[i]
            lm_lh = lm3(fr, 23); lm_rh = lm3(fr, 24)
            lm_ls = lm3(fr, 11); lm_rs = lm3(fr, 12)
            if not all([lm_lh, lm_rh, lm_ls, lm_rs]):
                continue
            curr_hip_rot = abs(rot_xz(lm_lh, lm_rh) - hip_rot_p1)
            curr_sh_rot  = abs(rot_xz(lm_ls, lm_rs) - shoulder_rot_p1)
            hip_rate = curr_hip_rot - prev_hip_rot
            sh_rate  = curr_sh_rot  - prev_sh_rot
            # Hips moving toward target while shoulders decelerate
            if hip_rate > 0 and sh_rate < 0:
                p5_idx = fi; p5_frame = fr; break
            prev_hip_rot = curr_hip_rot
            prev_sh_rot  = curr_sh_rot

        # Fallback: first frame hips lead shoulders after P4
        if p5_idx is None:
            for fi, wy, fr in post_p4[5:]:
                lm_lh = lm3(fr, 23); lm_rh = lm3(fr, 24)
                lm_ls = lm3(fr, 11); lm_rs = lm3(fr, 12)
                if not all([lm_lh, lm_rh, lm_ls, lm_rs]):
                    continue
                hr = abs(rot_xz(lm_lh, lm_rh) - hip_rot_p1)
                sr = abs(rot_xz(lm_ls, lm_rs) - shoulder_rot_p1)
                if hr > sr:
                    p5_idx = fi; p5_frame = fr; break

    # P6 — wrist returns to shoulder Y level going down
    p6_frame = p6_idx = None
    if p4_idx:
        for fi, wy, fr in rw:
            if fi <= p4_idx:
                continue
            ls = lm3(fr, 11)
            rs = lm3(fr, 12)
            if ls and rs:
                smy = (ls['y'] + rs['y']) / 2
                if wy > smy:
                    p6_frame = fr
                    p6_idx   = fi
                    break

    # P7 — frame of maximum trail wrist Y velocity on downswing
    p7_frame = p7_idx = None
    if p4_idx:
        post_p4 = [(fi, wy, fr) for fi, wy, fr in rw if fi > p4_idx]
        if len(post_p4) > 1:
            velocities = []
            for i in range(1, len(post_p4)):
                v = abs(post_p4[i][1] - post_p4[i-1][1])
                velocities.append((v, post_p4[i][0], post_p4[i][2]))
            velocities.sort(reverse=True)
            p7_idx, p7_frame = velocities[0][1], velocities[0][2]

    print(f"[PHASES] P1={p1_idx} P3={p3_idx} P4={p4_idx} P5={p5_idx} P7={p7_idx}", flush=True)

    def pr(label, fr, fi):
        if fr is None:
            return {'label': label, 'frame': None, 'time_s': None, 'detected': False}
        return {'label': label, 'frame': fi,
                'time_s': round(fi / fps, 3), 'detected': True}

    freeze_frames = [
        fi for _, fi in [
            ('P1', p1_idx), ('P3', p3_idx),
            ('P4', p4_idx), ('P5', p5_idx),
            ('P7', p7_idx),
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
    })


# ── React SPA fallback ────────────────────────────────────────────────────────
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    full = os.path.join(_static_folder, path)
    if path and os.path.exists(full):
        return send_from_directory(_static_folder, path)
    return send_from_directory(_static_folder, 'index.html')


# ── Sessions API ──────────────────────────────────────────────────────────────
@app.route('/api/sessions', methods=['POST'])
def create_session():
    if not DATABASE_URL:
        return jsonify({'error': 'No database configured'}), 503
    data = request.json
    conn = get_db(); cur = conn.cursor()
    cur.execute("""
        INSERT INTO swing_sessions
        (golfer_name, video_filename, fps, total_frames, phases, metrics)
        VALUES (%s,%s,%s,%s,%s,%s) RETURNING id
    """, (
        data.get('golfer_name', ''),
        data.get('video_filename', ''),
        data.get('fps', 30),
        data.get('total_frames', 0),
        json.dumps(data.get('phases', {})),
        json.dumps(data.get('metrics', {})),
    ))
    session_id = cur.fetchone()[0]
    conn.commit(); cur.close(); conn.close()
    return jsonify({'id': str(session_id)})


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
