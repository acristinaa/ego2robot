import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.factory import get_policy_class
from sim.bowl_plate_env import CAMERAS, make_env
from sim.replay_demos import TASK, robot_state

EVAL_SEED0 = 900_000


def to_batch(obs):
    batch = {f"observation.images.camera{i + 1}":
             torch.from_numpy(obs[f"{c}_image"][::-1].copy()).permute(2, 0, 1).float().unsqueeze(0) / 255
             for i, c in enumerate(CAMERAS)}
    batch["observation.state"] = torch.from_numpy(robot_state(obs)).unsqueeze(0)
    batch["task"] = [TASK]
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="folder containing config.json + model.safetensors")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--videos", type=int, default=6, help="save this many episodes as mp4")
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--device", default=None, help="cuda / mps (Apple GPU) / cpu; default: best available")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(args.checkpoint, config=cfg).to(device).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=args.checkpoint,
                                         preprocessor_overrides={"device_processor": {"device": device}})
    env = make_env(render=True)
    os.makedirs(args.out, exist_ok=True)

    results, t0 = [], time.time()
    for ep in range(args.episodes):
        np.random.seed(EVAL_SEED0 + ep)
        obs = env.reset()
        for _ in range(10):
            obs, *_ = env.step(np.zeros(7))
        policy.reset()
        frames, success, steps = [], False, 0
        for steps in range(1, args.max_steps + 1):
            with torch.inference_mode():
                action = post(policy.select_action(pre(to_batch(obs))))
            action = action.squeeze(0).float().cpu().numpy()
            if ep < args.videos:
                frames.append(np.concatenate([obs[f"{c}_image"][::-1] for c in CAMERAS], 1))
            obs, *_ = env.step(action)
            if env._check_success():
                success = True
                break
        results.append({"episode": ep, "seed": EVAL_SEED0 + ep, "success": success, "steps": steps})
        if frames:
            import imageio
            imageio.mimsave(f"{args.out}/ep{ep:02d}_{'success' if success else 'fail'}.mp4", frames, fps=20)
        rate = np.mean([r["success"] for r in results])
        print(f"episode {ep:3d}: {'SUCCESS' if success else 'fail   '} in {steps:3d} steps | "
              f"running success {100 * rate:.0f}% ({time.time() - t0:.0f}s)", flush=True)

    summary = {"checkpoint": str(args.checkpoint), "episodes": args.episodes,
               "success_rate": float(np.mean([r["success"] for r in results])), "per_episode": results}
    Path(f"{args.out}/results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSuccess rate: {100 * summary['success_rate']:.1f}% over {args.episodes} unseen scenes")


if __name__ == "__main__":
    main()