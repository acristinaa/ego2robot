import argparse
import glob
import json
from pathlib import Path

import av
import numpy as np
import scipy.signal as sg

THUMB_TIP, INDEX_TIP = 4, 8
RATE, ENV_RATE = 8000, 200


def audio_envelope(path):
    c = av.open(str(path))
    res = av.AudioResampler(format="s16", layout="mono", rate=RATE)
    x = np.concatenate([f.to_ndarray().ravel() for fr in c.decode(c.streams.audio[0]) for f in res.resample(fr)])
    c.close()
    x = sg.sosfilt(sg.butter(4, [300, 3500], "bandpass", fs=RATE, output="sos"), x.astype(float))
    e = sg.decimate(np.abs(x), RATE // ENV_RATE)
    return (e - e.mean()) / e.std()


def audio_offset(phone_video, vp_video):
    a, b = audio_envelope(phone_video), audio_envelope(vp_video)
    c = sg.correlate(a, b, "full")
    lag = sg.correlation_lags(len(a), len(b))[np.argmax(c)]
    return lag / ENV_RATE, float(c.max() / min(len(a), len(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phone", default="session_02")
    ap.add_argument("--vp", default="vp_02")
    ap.add_argument("--out", default="outputs/vision_pro")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    phone_dir, vp_dir = Path("data/processed") / args.phone, Path("data/processed") / args.vp
    offset, match = audio_offset(
        next(Path("data/raw_videos").glob(args.phone + ".*")), next(Path("data/raw_videos").glob(args.vp + ".*"))
    )
    print(f"audio sync: phone time = Vision Pro time + {offset:.2f} s (match {match:.2f})")

    raw = np.load(phone_dir / "raw.npz")
    fps_p = float(raw["fps"])
    cam_pos = -raw["R_marker_cam"].T @ raw["t_marker_cam"]

    vp = np.load(vp_dir / "stereo.npz")
    fps_v = float(vp["fps"])
    pinch_v = (vp["marker"][:, THUMB_TIP] + vp["marker"][:, INDEX_TIP]) / 2
    pinch_v[(pinch_v[:, 2] < -0.05) | (pinch_v[:, 2] > 0.6)] = np.nan

    rows = []
    for f in sorted(glob.glob(str(phone_dir / "demo_*.npz"))):
        d = np.load(f)
        R_task = d["R_task_from_marker"]
        pos_m = d["pos"] @ R_task
        for k, p, closed in zip(d["frames"], pos_m, d["gripper_closed"]):
            kv = int(round((k / fps_p - offset) * fps_v))
            if 0 <= kv < len(pinch_v) and not np.isnan(pinch_v[kv, 0]):
                rows.append((k / fps_p, int(Path(f).stem[-2:]), bool(closed), *p, *pinch_v[kv]))
    rows = np.array(rows)
    t, demo, closed = rows[:, 0], rows[:, 1].astype(int), rows[:, 2].astype(bool)
    phone, ref = rows[:, 3:6], rows[:, 6:9]

    err = phone - ref
    ray = phone - cam_pos
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    along = np.einsum("ni,ni->n", err, ray)
    across = np.linalg.norm(err - along[:, None] * ray, axis=1)
    e3 = np.linalg.norm(err, axis=1)

    def stats(x):
        x = np.abs(x)
        return {"median_cm": round(100 * float(np.median(x)), 1), "p90_cm": round(100 * float(np.percentile(x, 90)), 1)}

    rests = json.loads((phone_dir / "demos.json").read_text())["rests_s"]
    rest_h = np.concatenate(
        [pinch_v[max(int((a - offset) * fps_v), 0) : int((b - offset) * fps_v), 2] for a, b in rests]
    )
    rest_h = rest_h[~np.isnan(rest_h)]

    result = {
        "audio_offset_s": round(offset, 3),
        "vision_pro_height_hand_flat_cm": {
            "median": round(100 * float(np.median(rest_h)), 1),
            "p10": round(100 * float(np.percentile(rest_h, 10)), 1),
            "p90": round(100 * float(np.percentile(rest_h, 90)), 1),
        },
        "frames_compared": int(len(rows)),
        "demos_covered": int(len(set(demo))),
        "error_3d": stats(e3),
        "error_height": stats(err[:, 2]),
        "error_along_phone_ray": stats(along),
        "error_across_phone_ray": stats(across),
        "error_3d_while_holding": stats(e3[closed]),
        "error_3d_while_moving_freely": stats(e3[~closed]),
    }
    (out / "depth_check.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    a1.plot(100 * ref[~closed, 2], 100 * phone[~closed, 2], ".", ms=3, label="hand free")
    a1.plot(100 * ref[closed, 2], 100 * phone[closed, 2], ".", ms=3, label="holding the bowl")
    lim = [0, 100 * max(ref[:, 2].max(), phone[:, 2].max())]
    a1.plot(lim, lim, "k--", lw=1)
    a1.set_xlabel("Vision Pro stereo height (cm)")
    a1.set_ylabel("iPhone anchored-depth height (cm)")
    a1.set_aspect("equal")
    a1.legend(markerscale=4)
    bins = np.linspace(-25, 25, 51)
    a2.hist(100 * along, bins=bins, alpha=0.6, label="along the phone's view (depth)")
    a2.hist(100 * (err - along[:, None] * ray)[:, 0], bins=bins, alpha=0.6, label="sideways (x)")
    a2.set_xlabel("iPhone - Vision Pro (cm)")
    a2.set_ylabel("frames")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "depth_check.png", dpi=110)
    print(f"saved {out}/depth_check.png")


if __name__ == "__main__":
    main()
