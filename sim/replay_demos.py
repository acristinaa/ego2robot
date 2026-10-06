"""Replay every human demo in N random simulated scenes and report the success rate"""
import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

from sim import retarget as rt
from sim.bowl_plate_env import CAMERAS, make_env

TASK = "put the bowl on the plate"


def robot_state(o):
    """8-D state, same layout as the LIBERO LeRobot datasets: eef xyz, eef axis-angle, 2 finger joints."""
    import robosuite.utils.transform_utils as T
    return np.concatenate([o["robot0_eef_pos"], T.quat2axisangle(o["robot0_eef_quat"]),
                           o["robot0_gripper_qpos"]]).astype(np.float32)


def save_episode(stem, observations, actions, **meta):
    """Cameras as near-lossless MP4 (~1 MB each), state/actions as a small .npz."""
    import imageio
    stem.parent.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(CAMERAS):
        imageio.mimsave(f"{stem}_camera{i + 1}.mp4", [o[f"{c}_image"][::-1] for o in observations],
                        fps=20, quality=9, macro_block_size=1)
    np.savez(f"{stem}.npz", state=np.array([robot_state(o) for o in observations]),
             action=np.array(actions, dtype=np.float32), task=TASK, **meta)


def rollout(env, demo, seed, record):
    np.random.seed(seed)
    obs = env.reset()
    for _ in range(10): #objects settle
        obs, *_ = env.step(np.zeros(7))
    targets, grip, info = rt.plan(demo, env.bowl_pos, env.plate_pos, obs["robot0_eef_pos"])
    ok, frames, actions, observations = rt.run(env, targets, grip, record=record)
    return ok, frames, actions, observations, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="data/processed/*/demo_*.npz")
    ap.add_argument("--scenes", type=int, default=10)
    ap.add_argument("--save", default=None, help="folder to save successful rollouts")
    ap.add_argument("--video", type=int, default=None, help="seed for a single rendered video")
    ap.add_argument("--demo", default=None, help="e.g. session_01a/demo_00 (with --video)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.demos))
    if args.demo:
        files = [f for f in files if args.demo in f]
    record = args.save is not None or args.video is not None
    env = make_env(render=record)

    if args.video is not None:
        import imageio
        demo = rt.load_demo(files[0])
        ok, frames, *_ = rollout(env, demo, args.video, True)
        out = f"outputs/replay_{Path(files[0]).parent.name}_{Path(files[0]).stem}_seed{args.video}.mp4"
        os.makedirs("outputs", exist_ok=True)
        imageio.mimsave(out, frames, fps=20)
        print("success" if ok else "FAILED", "->", out)
        return

    results, n_saved = {}, 0
    t0 = time.time()
    for f in files:
        name = f"{Path(f).parent.name}/{Path(f).stem}"
        demo = rt.load_demo(f)
        wins = []
        for s in range(args.scenes):
            seed = 1000 * len(results) + s
            ok, frames, actions, observations, info = rollout(env, demo, seed, record)
            wins.append(bool(ok))
            if ok and args.save:
                save_episode(Path(args.save) / f"ep_{n_saved:04d}", observations, actions,
                             source_demo=name, seed=seed)
                n_saved += 1
        results[name] = {"success": f"{sum(wins)}/{len(wins)}", "rate": sum(wins) / len(wins), **info}
        print(f"{name:28s} {sum(wins):3d}/{len(wins)}  lift {info['human_lift_cm']:.0f} cm  "
              f"({time.time() - t0:.0f}s)", flush=True)

    total = np.mean([r["rate"] for r in results.values()])
    print(f"\nOverall replay success: {100 * total:.1f}%  over {len(files)} human demos x {args.scenes} scenes")
    if args.save:
        print(f"Saved {n_saved} successful robot episodes to {args.save}")
    os.makedirs("outputs", exist_ok=True)
    Path("outputs/replay_results.json").write_text(json.dumps(
        {"overall_success": total, "scenes_per_demo": args.scenes, "per_demo": results}, indent=2))


if __name__ == "__main__":
    main()