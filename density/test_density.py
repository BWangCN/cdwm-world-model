"""T3 self-check: MuJoCo density -> CoM mechanics, the load-bearing fact for the whole CoG-diversity plan.

Proves, with runnable asserts, on a 2-hull body (the minimal analogue of an object's CoACD decomposition):
  (1) UNIFORM density -> body CoM = geometric centroid, INDEPENDENT of the density value
      (doubling density leaves CoM put) -> "update per-category uniform density" gives ZERO CoG diversity.
  (2) NON-UNIFORM per-geom density -> CoM shifts to the analytic mass-weighted centroid
      -> per-CoACD-hull density is the real lever (matches `notes/drop_density_todo.md`).

    python test_density.py     # prints CoM per case, asserts all three claims

ponytail: two boxes stand in for CoACD hulls; the CoM algebra is identical for real hulls (MuJoCo
mass-weights per-geom volumetric centroids), so this validates the mechanism without needing mesh assets.
"""
import numpy as np, mujoco

# two equal 0.1 m cubes, centers at x = -0.5 and +0.5 (body frame); no <inertial> -> CoM from geoms.
XML = """
<mujoco model="two_hull">
  <worldbody>
    <body name="obj">
      <freejoint/>
      <geom name="A" type="box" size="0.05 0.05 0.05" pos="-0.5 0 0" density="{dA}"/>
      <geom name="B" type="box" size="0.05 0.05 0.05" pos="+0.5 0 0" density="{dB}"/>
    </body>
  </worldbody>
</mujoco>
"""

def com_x(dA, dB):
    m = mujoco.MjModel.from_xml_string(XML.format(dA=dA, dB=dB))
    b = m.body("obj")
    return float(b.ipos[0]), float(b.mass[0])          # ipos = CoM in body frame; mass = total

def analytic_com_x(dA, dB):                            # equal volumes V cancel: mass-weighted centroid
    return (dA*(-0.5) + dB*(+0.5)) / (dA + dB)

def main():
    # (1) uniform density, two different VALUES -> same CoM at the centroid (x=0)
    com_lo, m_lo = com_x(1000, 1000)
    com_hi, m_hi = com_x(5000, 5000)
    print(f"[uniform 1000] CoM_x={com_lo:+.4f}  mass={m_lo:.3f}")
    print(f"[uniform 5000] CoM_x={com_hi:+.4f}  mass={m_hi:.3f}  (5x denser)")
    assert abs(com_lo) < 1e-9, f"uniform CoM should be at centroid 0, got {com_lo}"
    assert abs(com_hi - com_lo) < 1e-9, "CoM moved when only the uniform density VALUE changed"
    assert m_hi > 4.9 * m_lo, "mass should scale ~5x with density"

    # (2) non-uniform per-geom density -> CoM shifts to the analytic mass-weighted centroid
    for dA, dB in [(8000, 1000), (1000, 8000), (2700, 1000)]:   # steel|plastic, plastic|steel, alu|plastic
        com, _ = com_x(dA, dB)
        exp = analytic_com_x(dA, dB)
        print(f"[hetero {dA}/{dB}] CoM_x={com:+.4f}  expected {exp:+.4f}")
        assert abs(com - exp) < 1e-6, f"CoM {com} != analytic {exp}"
        assert abs(com) > 0.05, "heterogeneous density should move CoM off the centroid"

    print("\nOK: uniform density -> CoM fixed at centroid (value-invariant); "
          "per-hull density -> CoM shifts as mass-weighted centroid.")

if __name__ == "__main__":
    main()
