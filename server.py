from flask import (Flask, request, jsonify,
                   send_from_directory, Response)
import cv2, mediapipe as mp, json, math
import os, tempfile, threading, uuid, time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__,
            static_folder=os.path.join(BASE_DIR, 'static'),
            static_url_path='/static')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

JOINT_IDS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
MAX_FILE_MB = 500
jobs = {}           # job_id → job dict
jobs_lock = threading.Lock()


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

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

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

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            fd = {
                'frame':        frame_num,
                'timestamp_ms': int(frame_num / fps * 1000),
                'poses':        [],
            }

            if results.pose_landmarks and results.pose_world_landmarks:
                entry = {
                    'pose_index':   0,
                    'landmarks_2d': {},
                    'landmarks_3d': {},
                }
                for jid in JOINT_IDS:
                    lm2 = results.pose_landmarks.landmark[jid]
                    lm3 = results.pose_world_landmarks.landmark[jid]
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

            frames_out.append(fd)
            frame_num += 1

            with jobs_lock:
                jobs[job_id]['progress'] = frame_num

        cap.release()
        pose.close()

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

    lh1       = lm3(p1_frame, 23)
    rh1       = lm3(p1_frame, 24)
    hip_rot_p1 = rot_xz(lh1, rh1) if lh1 and rh1 else 0
    rw_y_p1   = rw[0][1]
    post_p1   = [(fi, wy, fr) for fi, wy, fr in rw if fi > p1_idx]

    # P2 — hip rotation > 5°
    p2_frame = p2_idx = None
    for fi, wy, fr in post_p1:
        lh = lm3(fr, 23)
        rh = lm3(fr, 24)
        if lh and rh and abs(rot_xz(lh, rh) - hip_rot_p1) > 5:
            p2_frame = fr
            p2_idx   = fi
            break

    # P3 — wrist crosses hip Y upward
    p3_frame = p3_idx = None
    for fi, wy, fr in post_p1:
        lh = lm3(fr, 23)
        rh = lm3(fr, 24)
        if lh and rh:
            hmy = (lh['y'] + rh['y']) / 2
            if wy < hmy:
                p3_frame = fr
                p3_idx   = fi
                break

    # P4 — wrist Y minimum
    search = [(fi, wy, fr) for fi, wy, fr in post_p1
              if fi < p1_idx + int(fps * 30)]
    p4_idx = p4_frame = None
    if search:
        p4_idx, _, p4_frame = min(search, key=lambda x: x[1])

    # P5 — hip rotation changes ≥ 2° from P4 value
    p5_frame = p5_idx = None
    if p4_frame and p4_idx:
        lh4   = lm3(p4_frame, 23)
        rh4   = lm3(p4_frame, 24)
        hr_p4 = abs(rot_xz(lh4, rh4) - hip_rot_p1) if lh4 and rh4 else 0
        for fi, wy, fr in rw:
            if fi <= p4_idx + 10:
                continue
            lh = lm3(fr, 23)
            rh = lm3(fr, 24)
            if lh and rh:
                if abs(abs(rot_xz(lh, rh) - hip_rot_p1) - hr_p4) >= 2.0:
                    p5_frame = fr
                    p5_idx   = fi
                    break
            if fi > p4_idx + 80:
                break

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

    # P7 — wrist closest to P1 Y after P4
    p7_frame = p7_idx = None
    if p4_idx:
        post = [(fi, wy, fr) for fi, wy, fr in rw if fi > p4_idx]
        if post:
            p7_idx, _, p7_frame = min(post, key=lambda x: abs(x[1] - rw_y_p1))

    def pr(label, fr, fi):
        if fr is None:
            return {'label': label, 'frame': None, 'time_s': None, 'detected': False}
        return {'label': label, 'frame': fi,
                'time_s': round(fi / fps, 3), 'detected': True}

    return jsonify({
        'P1': pr('Address',    p1_frame, p1_idx),
        'P2': pr('Takeaway',   p2_frame, p2_idx),
        'P3': pr('Halfway',    p3_frame, p3_idx),
        'P4': pr('Top',        p4_frame, p4_idx),
        'P5': pr('Transition', p5_frame, p5_idx),
        'P6': pr('Pre-Impact', p6_frame, p6_idx),
        'P7': pr('Impact',     p7_frame, p7_idx),
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
