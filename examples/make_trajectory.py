#!/usr/bin/env python
# Copyright 2026 The LOCI Authors.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Write a camera trajectory file for scripts/generate.py.

Format (JSON):
  {
    "x_fov_deg": 90.0,           # horizontal field of view used by the camera conditioning
    "xi": 0.0,                   # unified-camera-model xi (0 = pinhole)
    "intrinsics_norm": [fx/W, fy/H, cx/W, cy/H],   # optional, only used to select history views
    "c2w": [ [[r00,r01,r02,tx],[r10,r11,r12,ty],[r20,r21,r22,tz]], ... ]
  }
One camera-to-world pose per *latent* frame (1 latent frame = 4 video frames at 16 fps), in
OpenCV axes (x right, y down, z forward), translation in metres, first pose = identity.
The number of poses must be 1 + 5k (k generated chunks).

Presets (all but walk_turn end exactly on the first view, so the last frames revisit it):
  look_around  : yaw right to +ANGLE, sweep to -ANGLE, return to 0              (default ANGLE 70 deg)
  forward_back : walk forward DISTANCE metres, then walk backwards to the start  (default 4 m)
  orbit_return : orbit to +ANGLE around a point DISTANCE metres ahead while looking at it, then
                 orbit back to the start                                         (default 60 deg, 4 m)
  walk_turn    : walk forward DISTANCE metres, turn 180 deg, walk back           (default 3 m)

The model was trained on walking-speed motion (about 1 m/s, turns of about 30 deg/s); with 0.25 s
per latent frame, keep steps below roughly 0.25 m / 7.5 deg per pose (a warning is printed otherwise).
"""
import argparse
import json
import math

import numpy as np


def yaw(deg):
    a = math.radians(deg)
    # rotation about the camera/world y axis (y points down); positive = turn right
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def look_around(n, angle=None, distance=None):
    angle = 70.0 if angle is None else angle
    keys = np.array([0, angle, -angle, 0], dtype=float)
    s = np.linspace(0, len(keys) - 1, n)
    angles = np.interp(s, np.arange(len(keys)), keys)
    return [np.concatenate([yaw(a), np.zeros((3, 1))], 1) for a in angles]


def forward_back(n, angle=None, distance=None):
    distance = 4.0 if distance is None else distance
    z = np.interp(np.linspace(0, 2, n), [0, 1, 2], [0, distance, 0])
    return [np.concatenate([np.eye(3), np.array([[0], [0], [v]])], 1) for v in z]


def orbit_return(n, angle=None, distance=None):
    angle = 60.0 if angle is None else angle
    distance = 4.0 if distance is None else distance
    centre = np.array([0, 0, distance])
    out = []
    for a in np.interp(np.linspace(0, 2, n), [0, 1, 2], [0, angle, 0]):
        rot = yaw(-a)                                   # keep looking at the centre while moving right
        pos = centre - rot @ np.array([0, 0, distance])
        out.append(np.concatenate([rot, pos[:, None]], 1))
    return out


def walk_turn(n, angle=None, distance=None):
    distance = 3.0 if distance is None else distance
    out, pos, heading = [], np.zeros(3), 0.0
    third = n // 3
    for i in range(n):
        if i and i <= third:
            pos = pos + yaw(heading) @ np.array([0, 0, distance / third])
        elif third < i <= 2 * third:
            heading += 180.0 / third
        elif i > 2 * third:
            pos = pos + yaw(heading) @ np.array([0, 0, distance / max(1, n - 1 - 2 * third)])
        out.append(np.concatenate([yaw(heading), pos[:, None]], 1))
    return out


PRESETS = {"look_around": look_around, "forward_back": forward_back, "orbit_return": orbit_return,
           "walk_turn": walk_turn}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=tuple(PRESETS), default="look_around")
    p.add_argument("--chunks", type=int, default=16,
                   help="generated chunks; 1 chunk = 5 latent frames = 20 video frames = 1.25 s (16 chunks = 20 s)")
    p.add_argument("--angle", type=float, default=None, help="rotation amplitude in degrees (preset default if omitted)")
    p.add_argument("--distance", type=float, default=None, help="distance in metres (preset default if omitted)")
    p.add_argument("--fov", type=float, default=90.0, help="horizontal field of view of the first frame, degrees")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    n = 1 + 5 * args.chunks
    if args.chunks < 1:
        p.error("--chunks must be >= 1")
    poses = PRESETS[args.preset](n, angle=args.angle, distance=args.distance)
    step_m = max(np.linalg.norm(b[:, 3] - a[:, 3]) for a, b in zip(poses, poses[1:]))
    step_deg = max(np.degrees(np.arccos(np.clip((np.trace(a[:, :3].T @ b[:, :3]) - 1) / 2, -1, 1)))
                   for a, b in zip(poses, poses[1:]))
    if step_m > 0.26 or step_deg > 7.6:
        print(f"warning: fastest step is {step_m:.2f} m / {step_deg:.1f} deg per latent frame "
              f"({step_m * 4:.1f} m/s, {step_deg * 4:.0f} deg/s), faster than the training motion; use more --chunks")
    spec = {"x_fov_deg": args.fov, "xi": 0.0,
            "c2w": [np.round(m, 6).tolist() for m in poses]}
    with open(args.output, "w") as f:
        json.dump(spec, f)
    print(f"wrote {n} poses ({args.chunks * 1.25:.1f} s) to {args.output}")


if __name__ == "__main__":
    main()
