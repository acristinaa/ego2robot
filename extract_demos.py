import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

H_REST = 0.015   #m, pinch point height when the hand lies flat on the table
H_GRASP = 0.065  #m, where you pinch the bowl: rim is 7 cm, fingertips sit ~0.5 cm below it
PALM = [0, 1, 5, 9, 13, 17]  #wrist, thumb base, finger bases: rigid part of the hand
THUMB_TIP, INDEX_TIP = 4, 8
STILL_SPEED = 0.6   #palm-widths per second below which the hand counts as still
MIN_REST_S = 0.5
MIN_GRASP_S = 1.0


def smooth(x, fps, win_s=0.3):
    win = max(5, int(win_s * fps) | 1)
    return savgol_filter(median_filter(x, size=(5,) + (1,) * (x.ndim - 1), mode="nearest"),
                         win, 2, axis=0, mode="interp")


def runs(mask):
    """[(start, end_exclusive)] of consecutive True values."""
    m = np.concatenate([[0], mask.astype(int), [0]])
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def fill_nans(a):
    a = a.copy()
    idx = np.arange(len(a))
    good = ~np.isnan(a.reshape(len(a), -1)[:, 0])
    for j in range(a.reshape(len(a), -1).shape[1]):
        col = a.reshape(len(a), -1)[:, j]
        col[~good] = np.interp(idx[~good], idx[good], col[good])
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--grasp-height", type=float, default=H_GRASP)
    args = ap.parse_args()
    folder = Path(args.folder)
    d = np.load(folder / "raw.npz")
    px, world, K = d["px"], d["world"], d["K"]
    R, tm, fps = d["R_marker_cam"], d["t_marker_cam"], float(d["fps"])
    N = len(px)
    t = np.arange(N) / fps
    px, world = fill_nans(px), fill_nans(world)
    Kinv = np.linalg.inv(K)

    #1. relative 3D hand from palm PnP (fixed nominal palm size)
    def palm_size(p):
        return np.linalg.norm(p - p.mean(0), axis=1).mean()

    NOMINAL = 0.04
    cam = np.zeros((N, 21, 3))
    for k in range(N):
        model = world[k] * (NOMINAL / palm_size(world[k][PALM]))
        _, rv, tv = cv2.solvePnP(model[PALM], px[k][PALM], K, None, flags=cv2.SOLVEPNP_SQPNP)
        _, rv, tv = cv2.solvePnP(model[PALM], px[k][PALM], K, None, rv, tv, True,
                                 flags=cv2.SOLVEPNP_ITERATIVE)
        cam[k] = (cv2.Rodrigues(rv)[0] @ model.T).T + tv.ravel()

    pinch_px = (px[:, THUMB_TIP] + px[:, INDEX_TIP]) / 2
    depth_rel = smooth((cam[:, THUMB_TIP, 2] + cam[:, INDEX_TIP, 2]) / 2, fps, 0.5)

    def depth_at_height(pix, hgt):
        """Camera depth at which the pixel ray through `pix` is `hgt` above the table."""
        ray = Kinv @ np.array([pix[0], pix[1], 1.0])
        return (hgt - (R.T @ -tm)[2]) / (R.T @ ray)[2]

    #2. grasp state (scale-free thumb-index gap) + stillness
    gap_n = np.linalg.norm(world[:, THUMB_TIP] - world[:, INDEX_TIP], axis=1) / \
        np.array([palm_size(w[PALM]) for w in world])
    gap_n = smooth(gap_n, fps, 0.4)
    lo, hi = np.percentile(gap_n, [10, 90])
    thr_close, thr_open = lo + 0.25 * (hi - lo), lo + 0.45 * (hi - lo)
    closed = np.zeros(N, bool)
    state = gap_n[0] < thr_close
    for k in range(N):  # hysteresis so the gripper does not chatter
        state = gap_n[k] < (thr_open if state else thr_close)
        closed[k] = state

    palm_px = np.array([palm_size(p[PALM]) for p in px])
    speed = np.linalg.norm(np.gradient(smooth(px[:, 0], fps, 0.3), axis=0), axis=1) * fps / palm_px
    still = smooth(speed, fps, 0.3) < STILL_SPEED
    rests = [(a, b) for a, b in runs(still & ~closed) if (b - a) / fps >= MIN_REST_S]
    # a real grasp lasts > MIN_GRASP_S (shorter dips are the fingers pre-shaping while reaching)
    grasps = [(a, b) for a, b in runs(closed) if (b - a) / fps >= MIN_GRASP_S]
    closed = np.zeros(N, bool)
    for a, b in grasps:
        closed[a:b] = True

    #3. anchors -> depth scale over time
    anchors = []  # (frame, true_depth / relative_depth)
    for a, b in rests:
        for k in range(a, b, max(1, (b - a) // 5)):
            anchors.append((k, depth_at_height(pinch_px[k], H_REST) / depth_rel[k]))
    for a, b in grasps:
        for k in (a, b - 1):
            anchors.append((k, depth_at_height(pinch_px[k], args.grasp_height) / depth_rel[k]))
    anchors.sort()
    ak = np.array([a for a, _ in anchors])
    ar = np.array([r for _, r in anchors])
    scale = np.exp(np.interp(np.arange(N), ak, np.log(ar)))
    depth = depth_rel * scale

    rays = (Kinv @ np.c_[pinch_px, np.ones(N)].T).T
    pos = (R.T @ (rays * depth[:, None] - tm).T).T  # marker frame
    # Task frame: same origin and z (up) as the marker, but rotated so that +y points
    # "up the image" (away from you, since the phone films from behind/above you) and +x
    # to your right. This makes results independent of how the marker was taped down.
    up_img = R.T @ np.array([0.0, -1.0, 0.0])  # camera's image-up direction in marker frame
    yaw = np.arctan2(up_img[1], up_img[0]) - np.pi / 2
    c, s_ = np.cos(-yaw), np.sin(-yaw)
    R_task = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
    pos = pos @ R_task.T
    pos = smooth(pos, fps, 0.4)
    pos[:, 2] = np.maximum(pos[:, 2], 0.0)
    gap_m = gap_n * NOMINAL * np.median(scale)  # thumb-index gap in metres

    #4. cut into demos: rest -> (grasp ...) -> rest
    demos = []
    for (a0, b0), (a1, b1) in zip(rests[:-1], rests[1:]):
        if any(b0 <= g0 < a1 for g0, _ in grasps):
            s = max(a0, b0 - int(0.5 * fps))
            e = min(b1, a1 + int(0.5 * fps))
            demos.append((s, e))

    summary = []
    for i, (s, e) in enumerate(demos):
        np.savez(folder / f"demo_{i:02d}.npz", t=t[s:e] - t[s], pos=pos[s:e],
                 gripper_closed=closed[s:e], gap=gap_m[s:e], fps=fps, frames=np.arange(s, e),
                 R_task_from_marker=R_task)
        summary.append({"demo": i, "start_s": round(t[s], 2), "end_s": round(t[e - 1], 2),
                        "z_max_cm": round(100 * float(pos[s:e, 2].max()), 1),
                        "path_len_cm": round(100 * float(np.linalg.norm(np.diff(pos[s:e], axis=0), axis=1).sum()), 1)})
    info = {"rests_s": [[round(t[a], 2), round(t[b - 1], 2)] for a, b in rests],
            "grasps_s": [[round(t[a], 2), round(t[b - 1], 2)] for a, b in grasps],
            "depth_scale_range_at_anchors": [round(float(ar.min()), 3), round(float(ar.max()), 3)],
            "demos": summary}
    (folder / "demos.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))

    #plotting
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(2, 2, (1, 2))
    raw_z = (R.T @ (rays * depth_rel[:, None] * np.median(ar) - tm).T).T[:, 2]
    picks = [s0 for s0, _ in grasps]
    places = [e0 - 1 for _, e0 in grasps]
    ax1.plot(t, raw_z * 100, color="0.75", lw=1, label="z, size-based depth only")
    for k, (name, c) in enumerate(zip("xyz", ["C0", "C1", "C2"])):
        ax1.plot(t, pos[:, k] * 100, c, label=f"{name} (anchored)")
    for a, b in grasps:
        ax1.axvspan(t[a], t[b - 1], color="k", alpha=0.08)
    for a, b in rests:
        ax1.axvspan(t[a], t[b - 1], color="C2", alpha=0.08)
    ax1.plot(t[ak], [100 * (H_REST if any(a <= k < b for a, b in rests) else args.grasp_height) for k in ak],
             "kx", label="height anchors")
    ax1.set_ylabel("pinch point, table frame (cm)")
    ax1.set_xlabel("time (s)   grey = gripper closed, green = resting")
    ax1.legend(ncol=5, fontsize=8, loc="upper left")
    ax2 = fig.add_subplot(2, 2, 3)
    ax2.plot(t, gap_m * 100, "k")
    ax2.set_ylabel("thumb-index gap (cm)")
    ax2.set_xlabel("time (s)")
    ax3 = fig.add_subplot(2, 2, 4)
    for i, (s, e) in enumerate(demos):
        ax3.plot(pos[s:e, 0] * 100, pos[s:e, 1] * 100, label=f"demo {i}")
        g = closed[s:e]
        ax3.plot(pos[s:e][g, 0] * 100, pos[s:e][g, 1] * 100, "k.", ms=2)
    sq = 0.135 / 2
    corners = np.array([[-sq, -sq, 0], [sq, -sq, 0], [sq, sq, 0], [-sq, sq, 0], [-sq, -sq, 0]]) @ R_task.T
    ax3.plot(corners[:, 0] * 100, corners[:, 1] * 100, "k-", lw=1)
    ax3.set_aspect("equal")
    ax3.plot(pos[picks, 0] * 100, pos[picks, 1] * 100, "g^", ms=9, label="grasp")
    ax3.plot(pos[places, 0] * 100, pos[places, 1] * 100, "rv", ms=9, label="release")
    ax3.legend(fontsize=8)
    ax3.set_title("top view (square = marker, black = holding)", fontsize=9)
    ax3.set_xlabel("x: to your right (cm)")
    ax3.set_ylabel("y: away from you (cm)")
    fig.tight_layout()
    fig.savefig(folder / "demos.png", dpi=110)
    print(f"Saved {len(demos)} demo(s) + demos.png to {folder}/")


if __name__ == "__main__":
    main()