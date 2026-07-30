"""pi0.5 record writer (VLA pipeline step 7). Turns a validated trajectory log into a LeRobot/openpi-style
episode record. Gated by the step-4 consistency validator (only PASS trajectories become data; REJECTs get a
provenance stub). Codex's top de-risk = the ACTION REPRESENTATION, so we log it UNAMBIGUOUSLY in multiple
explicit forms and name the primary:
  observation.state       (T,8)  proprioception = [arm_qpos(7), gripper(1)]
  observation.ee_pose     (T,7)  world EEF [p3, q4(wxyz)]
  observation.images.base (T,H,W,3) uint8 RGB
  action.joint_abs        (T,8)  next-step absolute [arm_qpos_target(7), gripper(1)]  <- pi0.5-Franka primary
  action.ee_delta         (T,7)  next-step EEF delta in the CURRENT ee frame [dp3, dq4] <- cross-embodiment
  action.ee_abs           (T,7)  next-step absolute world EEF target
Actions are next-step targets (action[t] moves state[t]->state[t+1]); the last frame holds. LeRobot chunks
actions at load time (action_horizon) via delta_timestamps, so we store per-frame, not pre-chunked.

    python pi0_writer.py [figures/vla_trajectory.npz] [out_dir]
"""
import sys, os, json, numpy as np
import consistency

PATH = sys.argv[1] if len(sys.argv) > 1 else "figures/vla_trajectory.npz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures/vla_records/episode_000"
os.makedirs(OUT, exist_ok=True)
FPS = 10

d = np.load(PATH, allow_pickle=True)
phase = d["phase"].astype(str); qpos = d["qpos"]; ee = d["ee"]; grip = d["grip"]; rgb = d["rgb"]
N = len(phase)

# ---- gate on the consistency validator ----
findings = consistency.validate(d); v, nh, nw = consistency.verdict(findings)
obj_id = str(d["obj_id"]) if "obj_id" in d.files else "006_mustard_bottle"
obj_name = " ".join(w for w in obj_id.split("_") if not w[0].isdigit()).replace("-", " ").strip() or obj_id
grasp_source = str(d["grasp_source"]) if "grasp_source" in d.files else "scripted_topdown"
grasp_id = int(d["grasp_id"]) if "grasp_id" in d.files else None

def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])
def qconj(q): return np.array([q[0], -q[1], -q[2], -q[3]])
def qR(q):
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

# ---- proprioception + action tensors ----
state = np.concatenate([qpos[:, :7], grip[:, None]], axis=1).astype(np.float32)          # (T,8)
nxt = np.concatenate([state[1:], state[-1:]], axis=0)                                     # shift; last holds
action_joint_abs = nxt.astype(np.float32)                                                # (T,8)
ee_abs_nxt = np.concatenate([ee[1:], ee[-1:]], axis=0)
action_ee_abs = ee_abs_nxt.astype(np.float32)                                            # (T,7)
ee_delta = np.zeros((N, 7), np.float32)                                                   # (T,7) local-frame delta
for t in range(N):
    Rt = qR(ee[t][3:])
    ee_delta[t, :3] = Rt.T @ (ee_abs_nxt[t][:3] - ee[t][:3])
    ee_delta[t, 3:] = qmul(qconj(ee[t][3:]), ee_abs_nxt[t][3:])
gripper_action = nxt[:, 7].astype(np.float32)

# ---- language (placeholder until step 8) ----
side = "away from the robot" if ee[-1][0] > ee[0][0] + 0.05 else "nearby"
task = f"pick up the {obj_name} and place it down"

# ---- phase spans + provenance ----
spans = []
s = 0
for i in range(1, N + 1):
    if i == N or phase[i] != phase[s]:
        spans.append({"phase": phase[s], "start": int(s), "end": int(i - 1)}); s = i

meta = {
    "episode_id": os.path.basename(OUT), "schema": "lerobot/openpi-style v0",
    "robot": "fr3_robotiq", "object_id": obj_id, "object_name": obj_name,
    "scene_id": "CDWMScene-v0", "seed": 0, "fps": FPS, "num_frames": int(N),
    "task": task, "task_source": "templated_placeholder (step-8 language-gen pending)",
    "grasp_source": grasp_source, "grasp_id": grasp_id,
    "grasp_net_tilt": float(d["grasp_net_tilt"]) if "grasp_net_tilt" in d.files else None,
    "wm_grasp": "grasp_mtl (in-gripper settle)", "wm_drop": "roll_corpus (min-reorient place regime)",
    "action_space_primary": "joint_abs (pi0.5-Franka); ee_delta provided for cross-embodiment",
    "action_dim": {"joint_abs": 8, "ee_delta": 7, "ee_abs": 7},
    "state_dim": 8, "image_res": list(rgb.shape[1:3]), "cameras": ["base"],
    "phases": spans,
    "validator": {"verdict": v, "hard": nh, "warn": nw,
                  "findings": [{"sev": f[0], "inv": f[1], "frame": int(f[2]), "phase": f[3], "detail": f[4]} for f in findings]},
    "provenance": {"trajectory_log": os.path.abspath(PATH), "pipeline": "CDWM grasp WM + drop WM kinematic replay"},
}

if v == "REJECT":
    meta["rejected"] = True
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)
    print(f"REJECTED trajectory ({nh} hard-invalid) -> wrote provenance stub only, no data.", flush=True)
    sys.exit(0)

# ---- write episode ----
timestamp = (np.arange(N) / FPS).astype(np.float32)
np.savez(os.path.join(OUT, "episode.npz"),
         **{"observation.state": state, "observation.ee_pose": ee.astype(np.float32),
            "observation.images.base": rgb.astype(np.uint8),
            "action.joint_abs": action_joint_abs, "action.ee_delta": ee_delta, "action.ee_abs": action_ee_abs,
            "action.gripper": gripper_action, "phase": phase, "timestamp": timestamp,
            "frame_index": np.arange(N), "episode_index": np.zeros(N, int)})
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)

# ---- loadability / roundtrip check ----
e = np.load(os.path.join(OUT, "episode.npz"), allow_pickle=True)
m = json.load(open(os.path.join(OUT, "meta.json")))
lengths = {k: e[k].shape[0] for k in e.files}
ok = len(set(lengths.values())) == 1 and list(lengths.values())[0] == N
print(f"=== pi0.5 writer: {OUT} ===", flush=True)
print(f"validator: {v} (hard {nh}, warn {nw}) | task: \"{m['task']}\"", flush=True)
print(f"frames {N} @ {FPS}fps | state{state.shape} | action.joint_abs{action_joint_abs.shape} "
      f"ee_delta{ee_delta.shape} | images {rgb.shape}", flush=True)
print(f"phases: {[ (s['phase'], s['start'], s['end']) for s in spans ]}", flush=True)
print(f"per-key frame counts aligned: {ok} ({sorted(set(lengths.values()))})", flush=True)
print(f"keys: {e.files}", flush=True)
print("=== done ===", flush=True)
