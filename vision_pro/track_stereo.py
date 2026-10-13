import argparse
import json
import struct
from pathlib import Path

import av
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from track_hand import MARKER_ID, MARKER_SIZE, MODEL_PATH

SCALE = 0.5
HAND_ROI = (0.45, 1.0, 0.35, 1.0)


def spatial_metadata(path):
    data = Path(path).read_bytes()
    hfov = struct.unpack(">I", data[data.find(b"hfov") + 4 :][:4])[0] / 1000
    blin = struct.unpack(">I", data[data.find(b"blin") + 8 :][:4])[0] / 1e6
    return hfov, blin


def stereo_frames(path):
    c = av.open(str(path))
    s = c.streams.video[0]
    s.codec_context.options = {"view_ids": "-1"}
    pair = []
    for f in c.decode(s):
        if f.pts is None:
            continue
        pair.append(f.to_ndarray(format="rgb24"))
        if len(pair) == 2:
            yield pair
            pair = []
    c.close()


def marker_pose(gray, detector, K):
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
    args = ap.parse_args()

    hfov, baseline = spatial_metadata(args.video)
    out_dir = Path("data/processed") / Path(args.video).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = float(av.open(args.video).streams.video[0].average_rate)

    aruco = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters()
    )
    landmarkers = [
        vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(
                    model_asset_path=str(MODEL_PATH), delegate=mp_tasks.BaseOptions.Delegate.CPU
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            )
        )
        for _ in range(2)
    ]

    K = None
    px, world, R_left, t_left, from_eye = [], [], [], [], []
    for n, frames in enumerate(stereo_frames(args.video)):
        frames = [cv2.resize(f, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA) for f in frames]
        h, w = frames[0].shape[:2]
        if K is None:
            f = (w / 2) / np.tan(np.radians(hfov) / 2)
            K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
            print(
                f"{args.video}: {w}x{h} per eye @ {fps:.0f} fps, FOV {hfov:.2f} deg, baseline {baseline * 1000:.1f} mm"
            )

        poses = [marker_pose(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), aruco, K) for im in frames]
        if poses[0] is not None:
            R, t = poses[0]
            from_eye.append(0)
        elif poses[1] is not None:
            R, t = poses[1][0], poses[1][1] + np.array([baseline, 0, 0])
            from_eye.append(1)
        else:
            R, t = np.full((3, 3), np.nan), np.full(3, np.nan)
            from_eye.append(-1)
        R_left.append(R)
        t_left.append(t)

        y0, y1, x0, x1 = (int(a * b) for a, b in zip(HAND_ROI, (h, h, w, w)))
        p2, wl = np.full((2, 21, 2), np.nan), np.full((21, 3), np.nan)
        for eye, (im, lm) in enumerate(zip(frames, landmarkers)):
            roi = np.ascontiguousarray(im[y0:y1, x0:x1])
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=roi), int(n * 1000 / fps))
            if res.hand_landmarks:
                p2[eye] = [[x0 + l.x * (x1 - x0), y0 + l.y * (y1 - y0)] for l in res.hand_landmarks[0]]
                if eye == 0:
                    wl = [[l.x, l.y, l.z] for l in res.hand_world_landmarks[0]]
        px.append(p2)
        world.append(wl)
        if (n + 1) % 150 == 0:
            print(f"  frame {n + 1}")

    px, world = np.array(px), np.array(world)
    R_left, t_left, from_eye = np.array(R_left), np.array(t_left), np.array(from_eye)

    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    disparity = px[:, 0, :, 0] - px[:, 1, :, 0]
    z = f * baseline / disparity
    x = (px[:, 0, :, 0] - cx) * z / f
    y = ((px[:, 0, :, 1] + px[:, 1, :, 1]) / 2 - cy) * z / f
    cam = np.stack([x, y, z], -1)
    marker = np.einsum("nji,nkj->nki", R_left, cam - t_left[:, None])

    both_hands = ~np.isnan(px[:, :, 0, 0]).any(1)
    found = from_eye >= 0
    np.savez(
        out_dir / "stereo.npz",
        px=px,
        world=world,
        cam=cam,
        marker=marker,
        R_left=R_left,
        t_left=t_left,
        K=K,
        baseline=baseline,
        fps=fps,
    )
    stats = {
        "frames": len(px),
        "fov_deg": hfov,
        "baseline_mm": round(baseline * 1000, 2),
        "marker_detected_pct": round(100 * found.mean(), 1),
        "hand_in_both_eyes_pct": round(100 * both_hands.mean(), 1),
        "eye_to_marker_m": round(float(np.nanmedian(np.linalg.norm(t_left, axis=1))), 3),
        "disparity_px_median": round(float(np.nanmedian(disparity[:, [4, 8]])), 1),
    }
    (out_dir / "stereo_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
