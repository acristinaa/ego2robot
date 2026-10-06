import argparse
from pathlib import Path

import cv2
import imageio
import numpy as np

from sim import retarget as rt
from sim.bowl_plate_env import make_env
from sim.replay_demos import rollout

H = 480


def open_video(path):
    #apply the phone's rotation metadata ourselves
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    rot = int(round(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)) % 360
    code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(rot)
    return cap, (lambda f: cv2.rotate(f, code)) if code is not None else (lambda f: f)


def read_frames(video, indices):
    cap, rotate = open_video(video)
    wanted, out, k = set(indices.tolist()), {}, 0
    while k <= indices.max():
        ok, f = cap.read()
        if not ok:
            break
        if k in wanted:
            out[k] = rotate(f)[:, :, ::-1]
        k += 1
    return [out[i] for i in indices if i in out]


def fit(img, h=H):
    w = int(img.shape[1] * h / img.shape[0]) // 2 * 2
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True, help="e.g. session_01/demo_00")
    ap.add_argument("--video", required=True, help="the phone video this demo came from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    demo_file = Path("data/processed") / f"{args.demo}.npz"
    d = np.load(demo_file)
    demo = rt.load_demo(demo_file)
    human = read_frames(args.video, d["frames"])

    env = make_env(render=True, size=H)
    ok, _, actions, observations, info = rollout(env, demo, args.seed, record=True)
    robot = [o["agentview_image"][::-1] for o in observations]

    # piecewise-linear time alignment: start / grasp / release / end
    grip = np.array(actions)[:, 6]
    r_close = int(np.argmax(grip > 0))
    r_open = r_close + int(np.argmax(grip[r_close:] < 0))
    h_keys = [0, info["pick_idx"], info["place_idx"], len(human) - 1]
    r_keys = [0, r_close, r_open, len(robot) - 1]
    h_idx = np.interp(np.arange(len(robot)), r_keys, h_keys).round().astype(int)

    frames = []
    for k, r in enumerate(robot):
        left = label(fit(human[min(h_idx[k], len(human) - 1)]), "my hand (iPhone)")
        right = label(fit(r), "Panda in sim (retargeted)")
        frames.append(np.concatenate([left, np.zeros((H, 8, 3), np.uint8), right], 1))

    out = args.out or f"outputs/showcase_{args.demo.replace('/', '_')}_seed{args.seed}.mp4"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=20, quality=8, macro_block_size=1)
    print(("success" if ok else "FAILED (robot did not finish the task)") + f" -> {out}")


if __name__ == "__main__":
    main()