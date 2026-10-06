import argparse
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MARKER_SIZE = 0.135  #metres, measured black square side 
MARKER_ID = 0
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = Path("models/hand_landmarker.task")

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def camera_matrix(w, h, fov_deg):
    """Pinhole intrinsics from the field of view along the LONG image side
    (works for portrait and landscape; no calibration needed)."""
    f = (max(w, h) / 2) / np.tan(np.radians(fov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def open_video(path):
    """Open a video and return (capture, rotate_fn). Applies the phone's rotation
    metadata ourselves so results are identical on every OS / OpenCV version."""
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    rot = int(round(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)) % 360
    code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(rot)
    return cap, (lambda f: cv2.rotate(f, code)) if code is not None else (lambda f: f)


def marker_pose(gray, detector, K):
    """(R, t) of the marker in camera coordinates, or None. Marker frame: origin at the
    centre, x right / y up along the printed square, z out of the table."""
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or MARKER_ID not in ids.flatten():
        return None
    c = corners[list(ids.flatten()).index(MARKER_ID)].reshape(4, 2)
    s = MARKER_SIZE / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, c, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return (cv2.Rodrigues(rvec)[0], tvec.reshape(3)) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--fov", type=float, default=69.0,
                    help="field of view along the long image side (deg); iPhone 1x is ~69")
    ap.add_argument("--no-overlay", action="store_true")
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(exist_ok=True)
        print("Downloading MediaPipe hand model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    out_dir = Path("data/processed") / Path(args.video).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    cap, rotate = open_video(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not read {args.video}")
    frame = rotate(frame)
    h, w = frame.shape[:2]
    K = camera_matrix(w, h, args.fov)
    print(f"{args.video}: {w}x{h} @ {fps:.1f} fps")

    aruco = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                    cv2.aruco.DetectorParameters())
    landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH),
                                          delegate=mp_tasks.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO, num_hands=1,
        min_hand_detection_confidence=0.5, min_tracking_confidence=0.5))

    px_all, world_all, poses = [], [], []
    marker_t, marker_r = [], []  #per-frame marker pose (NaN when not visible)
    n = 0
    while ok:
        pose = marker_pose(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), aruco, K)
        if pose is not None:
            poses.append(pose)
        marker_t.append(pose[1] if pose is not None else np.full(3, np.nan))
        marker_r.append(cv2.Rodrigues(pose[0])[0].ravel() if pose is not None else np.full(3, np.nan))
        res = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
            int(n * 1000 / fps))
        if res.hand_landmarks:
            px_all.append([[l.x * w, l.y * h] for l in res.hand_landmarks[0]])
            world_all.append([[l.x, l.y, l.z] for l in res.hand_world_landmarks[0]])
        else:
            px_all.append(np.full((21, 2), np.nan))
            world_all.append(np.full((21, 3), np.nan))
        n += 1
        if n % 60 == 0:
            print(f"  frame {n}")
        ok, frame = cap.read()
        if ok:
            frame = rotate(frame)
    cap.release()
    landmarker.close()

    if not poses:
        raise SystemExit("Marker never detected. Check lighting / that the marker is in view.")
    #Robust single pose (used for the overlay); stage 2 uses the smoothed per-frame poses because a phone on a makeshift mount can sag a few cm over a long recording.
    t_med = np.median([p[1] for p in poses], axis=0)
    R_cm = min(poses, key=lambda p: np.linalg.norm(p[1] - t_med))[0]
    jitter_mm = float(np.linalg.norm(np.std([p[1] for p in poses], axis=0)) * 1000)

    px_all, world_all = np.array(px_all), np.array(world_all)
    np.savez(out_dir / "raw.npz", px=px_all, world=world_all, K=K, R_marker_cam=R_cm,
             t_marker_cam=t_med, fps=fps, size=np.array([w, h]),
             marker_t=np.array(marker_t), marker_rvec=np.array(marker_r))

    stats = {
        "frames": n,
        "marker_detected_pct": round(100 * len(poses) / n, 1),
        "hand_detected_pct": round(100 * float((~np.isnan(px_all[:, 0, 0])).mean()), 1),
        "camera_to_marker_distance_m": round(float(np.linalg.norm(t_med)), 3),
        "camera_shake_mm": round(jitter_mm, 1),
    }
    (out_dir / "detect_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    if not args.no_overlay:
        cap, rotate = open_video(args.video)
        vw = cv2.VideoWriter(str(out_dir / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        rvec = cv2.Rodrigues(R_cm)[0]
        for i in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            frame = rotate(frame)
            cv2.drawFrameAxes(frame, K, None, rvec, t_med, MARKER_SIZE)
            if not np.isnan(px_all[i, 0, 0]):
                p = px_all[i].astype(int)
                for a, b in HAND_EDGES:
                    cv2.line(frame, tuple(p[a]), tuple(p[b]), (0, 255, 0), 3)
            vw.write(frame)
        vw.release()
    print(f"Saved to {out_dir}/  ->  next: python extract_demos.py {out_dir}")


if __name__ == "__main__":
    main()