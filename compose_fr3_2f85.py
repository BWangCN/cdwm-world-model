"""Compose FR3 + Robotiq 2F-85 into one MJCF for ManiSkill. Uses mujoco MjSpec to attach the 2F-85 gripper spec to the
FR3's flange `attachment_site`, then compiles + exports the combined model. Reports joints/actuators so we can wire the
ManiSkill agent (arm joints + gripper driver joint). -> assets/fr3_2f85.xml
"""
import os, mujoco
HOME = os.path.expanduser("~/.cache/robot_descriptions/mujoco_menagerie")
FR3 = f"{HOME}/franka_fr3/fr3.xml"
GRIP = f"{HOME}/robotiq_2f85/2f85.xml"
OUT = "/misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model/assets/fr3_2f85.xml"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
print("mujoco", mujoco.__version__, flush=True)

arm = mujoco.MjSpec.from_file(FR3)
grip = mujoco.MjSpec.from_file(GRIP)


def absolutize(spec, base):
    """Rewrite mesh/texture file refs to absolute paths (using the spec's own meshdir/texturedir) so the merged model
    resolves assets from BOTH source dirs after export."""
    md = spec.meshdir or ""; td = getattr(spec, "texturedir", "") or ""
    for mesh in spec.meshes:
        if mesh.file and not os.path.isabs(mesh.file): mesh.file = os.path.normpath(os.path.join(base, md, mesh.file))
    for tex in getattr(spec, "textures", []):
        if getattr(tex, "file", "") and not os.path.isabs(tex.file): tex.file = os.path.normpath(os.path.join(base, td, tex.file))
    spec.meshdir = ""
    try: spec.texturedir = ""
    except Exception: pass


absolutize(arm, os.path.dirname(FR3)); absolutize(grip, os.path.dirname(GRIP))
print("FR3 sites:", [s.name for s in arm.sites], flush=True)
print("FR3 bodies (last few):", [b.name for b in arm.bodies][-4:], flush=True)
print("GRIP top bodies:", [b.name for b in grip.bodies][:4], flush=True)

# attach the gripper to the FR3 attachment_site. Try the site.attach_body API (mujoco>=3.2), fall back to spec.attach.
site = next(s for s in arm.sites if s.name == "attachment_site")
attached = False
for attempt in ("site.attach_body", "spec.attach"):
    try:
        if attempt == "site.attach_body":
            gbase = next(b for b in grip.bodies if b.name in ("base_mount", "base"))
            site.attach_body(gbase, "2f85_", "")
        else:
            arm.attach(grip, prefix="2f85_", site=site)
        attached = True; print("attached via", attempt, flush=True); break
    except Exception as e:
        print(f"  {attempt} failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
assert attached, "both attach APIs failed"

m = arm.compile()
print(f"COMPILED: nq={m.nq} nu={m.nu} nbody={m.nbody}", flush=True)
print("actuators:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)], flush=True)
print("joints:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)], flush=True)
with open(OUT, "w") as f: f.write(arm.to_xml())
print("wrote", OUT, flush=True)
