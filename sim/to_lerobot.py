import argparse
import glob
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

STATE_NAMES = [
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_ax",
    "eef_ay",
    "eef_az",
    "finger_l",
    "finger_r",
]
ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]


def read_video(path):
    import imageio

    return np.array(imageio.mimread(path, memtest=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument(
        "--root", default=None, help="local output folder (default: HF cache)"
    )
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.folder) / "ep_*.npz")))
    h, w = read_video(files[0].replace(".npz", "_camera1.mp4"))[0].shape[:2]
    features = {
        f"observation.images.camera{i}": {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
        for i in (1, 2, 3)
    }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (8,),
        "names": STATE_NAMES,
    }
    features["action"] = {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES}

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=20,
        features=features,
        root=args.root,
        robot_type="panda",
        use_videos=True,
    )
    for f in files:
        ep = np.load(f)
        task = str(ep["task"])
        cams = {i: read_video(f.replace(".npz", f"_camera{i}.mp4")) for i in (1, 2, 3)}
        for t in range(len(ep["action"])):
            ds.add_frame(
                {
                    **{f"observation.images.camera{i}": cams[i][t] for i in (1, 2, 3)},
                    "observation.state": ep["state"][t],
                    "action": ep["action"][t],
                    "task": task,
                }
            )
        ds.save_episode()
    ds.finalize()
    print(f"{len(files)} episodes -> {ds.root}")
    if args.push:
        ds.push_to_hub()


if __name__ == "__main__":
    main()
