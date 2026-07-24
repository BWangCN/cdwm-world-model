"""Rotation / SE(3) numpy utilities for the WM (targets + eval). Quaternions are wxyz (MuJoCo)."""
import numpy as np


def quat_to_R(q):                      # q (...,4) wxyz -> (...,3,3)
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                  2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                  2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)
    return R.reshape(q.shape[:-1] + (3, 3))


def R_to_6d(R):                        # (...,3,3) -> (...,6): first two columns
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def sixd_to_R(d):                      # (...,6) -> (...,3,3): Gram-Schmidt
    a, b = d[..., :3], d[..., 3:]
    x = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b - (x * b).sum(-1, keepdims=True) * x
    y = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=-1)


def geodesic_deg(R1, R2):              # angle between rotations (...,)
    Rr = np.matmul(R1, np.swapaxes(R2, -1, -2))
    tr = np.clip((Rr[..., 0, 0] + Rr[..., 1, 1] + Rr[..., 2, 2] - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


def rot_angle_axis(R):                 # (...,3,3) -> angle(deg), unit axis (...,3)
    tr = np.clip((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2, -1, 1)
    ang = np.degrees(np.arccos(tr))
    ax = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    ax = ax / (np.linalg.norm(ax, axis=-1, keepdims=True) + 1e-12)
    return ang, ax


def axis_err_deg(R1, R2):              # angle between net-rotation axes (sign-agnostic)
    _, a1 = rot_angle_axis(R1); _, a2 = rot_angle_axis(R2)
    return np.degrees(np.arccos(np.clip(np.abs((a1 * a2).sum(-1)), 0, 1)))
