import numpy as np
from scipy.ndimage import uniform_filter1d

import robosuite.utils.transform_utils as T
from sim.bowl_plate_env import BOWL_HEIGHT, BOWL_RADIUS, BOWL_WALL, CAMERAS, PLATE_HALF_THICK

HUMAN_TO_ROBOT = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
CONTROL_HZ = 20
GRASP_DEPTH = 0.018     #fingertips this far below the rim top
CLEAR = 0.06            #stay above the grasp point until over it
CLOSE_STEPS, OPEN_STEPS = 12, 8
MAX_STEP = 0.015
MAX_SIDE = 0.05


def load_demo(path):
    d = np.load(path)
    pos = d["pos"] @ HUMAN_TO_ROBOT.T
    closed = d["gripper_closed"].astype(bool)
    return pos, closed, float(d["fps"])


def keypoints(pos, closed, fps):
    """Pick = stillest moment early in the grasp, place = stillest moment late in it."""
    idx = np.where(closed)[0]
    g0, g1 = idx[0], idx[-1]
    speed = np.linalg.norm(np.gradient(uniform_filter1d(pos, 5, axis=0), axis=0), axis=1) * fps
    n = g1 - g0
    first = slice(g0, g0 + max(2, int(0.45 * n)))
    last = slice(g1 - max(2, int(0.45 * n)), g1 + 1)
    pick = first.start + int(np.argmin(speed[first]))
    place = last.start + int(np.argmin(speed[last]))
    return pick, place


def resample(seg, fps, speed=1.0):
    """Resample a (T,3) segment recorded at `fps` to the robot's control rate."""
    n_out = max(2, int(round(len(seg) / fps * CONTROL_HZ / speed)))
    src = np.linspace(0, len(seg) - 1, n_out)
    return np.stack([np.interp(src, np.arange(len(seg)), seg[:, k]) for k in range(3)], 1)


def stretch_time(path, grip, max_step):
    """Insert intermediate steps wherever the target would move more than `max_step` per
    control step, so the arm can follow. Time is stretched; the gripper timeline stays aligned."""
    P, G = [path[0]], [grip[0]]
    for k in range(1, len(path)):
        n = int(np.ceil(np.linalg.norm(path[k] - path[k - 1]) / max_step))
        for j in range(1, max(n, 1) + 1):
            P.append(path[k - 1] + (path[k] - path[k - 1]) * j / max(n, 1))
            G.append(grip[k])
    return np.array(P), np.array(G)


def smooth_path(x, fps, win_s):
    return uniform_filter1d(x, max(1, int(win_s * fps)), axis=0, mode="nearest")


def smoothstep(n):
    x = np.linspace(0, 1, n)
    return x * x * (3 - 2 * x)


def plan(demo, bowl_pos, plate_pos, eef_start, speed=1.0):
    """Returns (targets (N,3), gripper (N,) in {-1 open, +1 closed}, info dict)."""
    pos, closed, fps = demo
    pick, place = keypoints(pos, closed, fps)
    H_pick, H_place = pos[pick], pos[place]

    # Grasp the bowl wall on the side facing the robot, fingers closing radially.
    u = np.array([-1.0, 0.0, 0.0])
    rim_top = bowl_pos[2] + BOWL_HEIGHT / 2
    G = bowl_pos + u * (BOWL_RADIUS - BOWL_WALL / 2)
    G[2] = rim_top - GRASP_DEPTH
    lift_to_plate = 2 * PLATE_HALF_THICK + 0.012          # bowl ends on top of the plate (+ small drop)
    P = plate_pos + u * (BOWL_RADIUS - BOWL_WALL / 2)
    P[2] = G[2] + lift_to_plate

    #approach
    A = resample(pos[: pick + 1] - H_pick, fps, speed) + G
    A += (1 - smoothstep(len(A)))[:, None] * (eef_start - A[0])
    dxy = np.linalg.norm(A[:, :2] - G[:2], axis=1)
    # never below the grasp height, and above the rim until right over the wall
    A[:, 2] = np.maximum(A[:, 2], G[2] + CLEAR * np.clip(dxy / 0.015, 0, 1))
    A = np.vstack([A, np.repeat(G[None], 4, 0)])

    #carry: human pick->place path mapped onto sim bowl->plate
    seg = pos[pick: place + 1] - H_pick
    Dh, Ds = (H_place - H_pick)[:2], (P - G)[:2]
    Lh, Ls = max(np.linalg.norm(Dh), 1e-3), max(np.linalg.norm(Ds), 1e-3)
    uh, us = Dh / Lh, Ds / Ls
    nh, ns = np.array([-uh[1], uh[0]]), np.array([-us[1], us[0]])
    tau = np.maximum.accumulate(np.clip(seg[:, :2] @ uh / Lh, 0, 1))
    tau[-1] = 1.0
    side = np.clip(seg[:, :2] @ nh, -MAX_SIDE, MAX_SIDE) * np.sin(np.pi * tau)  # 0 at both ends
    human_lift = seg[:, 2] - tau * (H_place - H_pick)[2]
    C = np.zeros_like(seg)
    C[:, :2] = tau[:, None] * Ds + side[:, None] * ns
    C[:, 2] = tau * (P - G)[2] + np.maximum(human_lift, 0)
    C = smooth_path(C, fps, 0.5)
    theta, s = np.arctan2(us[1], us[0]) - np.arctan2(uh[1], uh[0]), Ls / Lh
    C = resample(C, fps, speed) + G
    C[:, 2] = np.maximum(C[:, 2], np.minimum(G[2], P[2]))
    # keep the bowl above the plate rim until it is over the plate
    dxy = np.linalg.norm(C[:, :2] - P[:2], axis=1)
    C[:, 2] = np.maximum(C[:, 2], P[2] + 0.04 * np.clip(dxy / 0.04, 0, 1) * (np.arange(len(C)) > 3))
    C = np.vstack([C, np.repeat(P[None], 6, 0)])

    #retreat: open, then follow the human's retreat (re-anchored), lifting first
    Rt = resample(pos[place:] - H_place, fps, speed) + P
    Rt[:, 2] = np.maximum(Rt[:, 2], P[2] + 0.08 * smoothstep(len(Rt)))

    targets = np.vstack([A, np.repeat(G[None], CLOSE_STEPS, 0), C,
                         np.repeat(P[None], OPEN_STEPS, 0), Rt])
    grip = np.concatenate([-np.ones(len(A)), np.ones(CLOSE_STEPS), np.ones(len(C)),
                           -np.ones(OPEN_STEPS), -np.ones(len(Rt))])
    targets, grip = stretch_time(targets, grip, MAX_STEP)
    info = {"pick_idx": int(pick), "place_idx": int(place), "carry_rotation_deg": float(np.degrees(theta)),
            "carry_scale": float(s), "human_lift_cm": float(100 * human_lift.max())}
    return targets, grip, info


def run(env, targets, grip, record=False, settle=25):
    obs = env._get_observations()
    R_goal = T.quat2mat(T.axisangle2quat(np.array([0, 0, np.pi / 2]))) @ T.quat2mat(obs["robot0_eef_quat"])
    frames, actions, observations = [], [], []
    k, waited, n = 0, 0, len(targets)
    while k < n + settle:
        i = min(k, n - 1)
        tgt, g = targets[i].copy(), grip[i]
        eef = obs["robot0_eef_pos"]
        switching = 0 < k < n and grip[i] != grip[i - 1]
        if switching and np.linalg.norm(tgt - eef) > 0.01 and waited < 40:
            g, waited = grip[i - 1], waited + 1
        else:
            k, waited = k + 1, 0
        if tgt[2] < eef[2] and np.linalg.norm(tgt[:2] - eef[:2]) > 0.02:
            tgt[2] = eef[2]
        a = np.zeros(7)
        a[:3] = np.clip((tgt - eef) / 0.05, -1, 1)
        a[3:6] = np.clip(T.quat2axisangle(T.mat2quat(R_goal @ T.quat2mat(obs["robot0_eef_quat"]).T)) / 0.5, -1, 1)
        a[6] = g
        if record:
            observations.append(dict(obs))
            actions.append(a.copy())
            frames.append(np.concatenate([obs[f"{c}_image"][::-1] for c in CAMERAS], 1))
        obs, _, _, _ = env.step(a)
    return env._check_success(), frames, actions, observations