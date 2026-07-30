"""Per-frame physical-consistency validator (VLA pipeline step 4). PHASE-AWARE corpus QA gate over a
trajectory log (`figures/vla_trajectory.npz` from render_demo). Splits findings into HARD-INVALID (drop the
trajectory) vs WARNING (diagnostic). Tolerances derive from the trajectory's own distributions + object/
scene scale, not fixed constants. Pure numpy on the log (no sim needed). Top invariants (Codex-prioritized):
  1. frame/action consistency  — during attached phases the object stays in the jaws (obj center in the ee
     frame near the TCP): catches wrong mult order / stale frame / bad TCP pivot / sign errors.
  2. contact/attachment        — during a rigid hold (transport) the gripper->object transform is ~constant;
     during grasp_close/lift the WM settle may change it: catches teleport / wrong-frame settle / mixed deltas.
  3. table/support             — object lowest rotated-corner never below the table; supported (not floating)
     at rest: geometry-derived from the rotated box, not a hand-picked height.
  4. feasibility replay        — arm qpos within joint limits every frame; no outlier joint jumps (robust MAD).

Reusable: validate(d) -> list of findings. CLI: python consistency.py [figures/vla_trajectory.npz]
"""
import sys, numpy as np


def _qR(q):
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
def _geodesic(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def validate(d):
    """d: dict-like with phase/attached/qpos/ee/obj/obj_half/P_ee/qlimits. Returns findings
    [(severity, invariant, frame, phase, detail)] — 'HARD' entries mean REJECT the trajectory."""
    phase = d["phase"].astype(str); attached = d["attached"]; qpos = d["qpos"]
    ee = d["ee"]; obj = d["obj"]; obj_half = d["obj_half"]; P_ee = d["P_ee"]; qlim = d["qlimits"]
    N = len(phase)
    CORNERS = obj_half * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    findings = []
    def add(sev, inv, i, detail): findings.append((sev, inv, i, phase[i], detail))
    def inv7(p):
        R = _qR(p[3:]); return R.T, -R.T @ p[:3]
    def grip_obj(i):
        Ri, ti = inv7(ee[i]); Ro = _qR(obj[i][3:]); to = obj[i][:3]
        return Ri @ to + ti, Ri @ Ro
    def lowest_corner_z(i):
        return float(np.min((_qR(obj[i][3:]) @ CORNERS.T)[2]) + obj[i][2])

    diag = float(np.linalg.norm(obj_half)); JAW_TOL = diag; Z_PEN = 0.01

    # 1. frame/action consistency: object stays in the jaws during attached phases
    for i in range(N):
        if not attached[i]: continue
        off = np.linalg.norm(grip_obj(i)[0] - P_ee)
        if off > JAW_TOL * 1.5:
            add("HARD", "frame/action", i, f"object {off*100:.1f}cm from TCP in ee frame (>{JAW_TOL*1.5*100:.1f})")
        elif off > JAW_TOL:
            add("WARN", "frame/action", i, f"object {off*100:.1f}cm from TCP (grasp offset large)")

    # 2. contact/attachment: rigid hold during 'transport' (settle done)
    hold = np.where(phase == "transport")[0]
    if len(hold) > 1:
        t0, R0 = grip_obj(hold[0])
        dt = np.array([np.linalg.norm(grip_obj(i)[0] - t0) for i in hold[1:]])
        dr = np.array([_geodesic(R0, grip_obj(i)[1]) for i in hold[1:]])
        tol_t = max(0.005, np.median(dt) + 5 * (np.median(np.abs(dt - np.median(dt))) + 1e-6))
        for j, i in enumerate(hold[1:]):
            if dt[j] > tol_t or dr[j] > 5.0:
                sev = "HARD" if (dt[j] > 0.03 or dr[j] > 15) else "WARN"
                add(sev, "attachment", i, f"grip->obj drift {dt[j]*100:.1f}cm / {dr[j]:.1f}deg during rigid hold")

    # 3. table/support
    for i in range(N):
        lz = lowest_corner_z(i)
        if lz < -Z_PEN:
            add("HARD", "table/support", i, f"object penetrates table (lowest corner z={lz*100:.1f}cm)")
    lz = lowest_corner_z(N - 1)
    if lz > 0.02:
        add("WARN", "table/support", N - 1, f"object floating at rest (lowest corner z={lz*100:.1f}cm above table)")
    elif lz < -Z_PEN:
        add("HARD", "table/support", N - 1, f"object sunk into table at rest (z={lz*100:.1f}cm)")

    # 4. feasibility replay
    arm = qpos[:, :7]; lo = qlim[:7, 0]; hi = qlim[:7, 1]
    for i in range(N):
        viol = np.where((arm[i] < lo - 1e-3) | (arm[i] > hi + 1e-3))[0]
        if len(viol): add("HARD", "feasibility", i, f"joint-limit violation j{viol.tolist()}")
    steps = np.linalg.norm(np.diff(arm, axis=0), axis=1)
    med = np.median(steps); mad = np.median(np.abs(steps - med)) + 1e-6
    for i in range(1, N):
        if steps[i - 1] > med + 8 * mad and steps[i - 1] > 0.15:
            add("WARN", "feasibility", i, f"joint jump {steps[i-1]:.3f} rad (>{med+8*mad:.3f}) at phase boundary")
    return findings


def verdict(findings):
    hard = [f for f in findings if f[0] == "HARD"]; warn = [f for f in findings if f[0] == "WARN"]
    return ("REJECT" if hard else "PASS"), len(hard), len(warn)


if __name__ == "__main__":
    PATH = sys.argv[1] if len(sys.argv) > 1 else "figures/vla_trajectory.npz"
    d = np.load(PATH, allow_pickle=True)
    findings = validate(d); v, nh, nw = verdict(findings)
    print(f"=== consistency validator: {len(d['phase'])} frames, phases {sorted(set(d['phase'].astype(str)))} ===", flush=True)
    for sev, inv, i, ph, detail in findings:
        print(f"  [{sev}] {inv:14s} frame {i:3d} ({ph:12s}): {detail}", flush=True)
    print(f"\nHARD-INVALID: {nh} | WARNINGS: {nw}\nVERDICT: {v}" + (" (hard-invalid present)" if nh else " (no hard-invalid)"), flush=True)
