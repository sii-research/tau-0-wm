import torch
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    SupportsIndex,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
    runtime_checkable,
)

# ---------------------------------------------------------------------------
# Rotation 6D helper functions (used by TransformActionAbs2Delta and TransformStateAbs2Delta)
# ---------------------------------------------------------------------------

def quaternion_to_matrix(quat: torch.Tensor, quat_order: str = "xyzw", eps: float = 1e-8) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix.

    Args:
        quat: (..., 4) tensor
        quat_order: "xyzw" or "wxyz"
        eps: small value for numerical stability

    Returns:
        (..., 3, 3) rotation matrix
    """
    assert quat.shape[-1] == 4, f"Expected (..., 4), got {quat.shape}"

    quat = quat / (quat.norm(dim=-1, keepdim=True) + eps)

    if quat_order == "xyzw":
        x, y, z, w = quat.unbind(dim=-1)
    elif quat_order == "wxyz":
        w, x, y, z = quat.unbind(dim=-1)
    else:
        raise ValueError(f"Unsupported quat_order: {quat_order}")

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    r00 = 1 - 2 * (yy + zz)
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)

    r10 = 2 * (xy + wz)
    r11 = 1 - 2 * (xx + zz)
    r12 = 2 * (yz - wx)

    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = 1 - 2 * (xx + yy)

    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )  # (..., 3, 3)
    
    
def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix.

    Args:
        rotation_6d: (..., 6) tensor — first two columns of rotation matrix, flattened.

    Returns:
        (..., 3, 3) rotation matrix.
    """
    a1 = rotation_6d[..., :3]
    a2 = rotation_6d[..., 3:6]

    # Gram-Schmidt orthogonalization
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + 1e-8)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = a2 - dot * b1
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3)


def compute_ee6d_delta(current_6d: torch.Tensor, reference_6d: torch.Tensor) -> torch.Tensor:
    """Compute rotation delta in 6D representation: R_delta = R_ref^T @ R_cur.

    Args:
        current_6d: (..., 6) current rotation in 6D.
        reference_6d: (..., 6) reference rotation in 6D.

    Returns:
        (..., 6) delta rotation in 6D.
    """
    R_cur = rotation_6d_to_matrix(current_6d)
    R_ref = rotation_6d_to_matrix(reference_6d)
    R_delta = R_ref.transpose(-1, -2) @ R_cur
    return matrix_to_rotation_6d(R_delta)


def apply_ee6d_delta(reference_6d: torch.Tensor, delta_6d: torch.Tensor) -> torch.Tensor:
    """Apply rotation delta to reference: R_cur = R_ref @ R_delta.

    Args:
        reference_6d: (..., 6) reference rotation in 6D.
        delta_6d: (..., 6) delta rotation in 6D.

    Returns:
        (..., 6) resulting rotation in 6D.
    """
    R_ref = rotation_6d_to_matrix(reference_6d)
    R_delta = rotation_6d_to_matrix(delta_6d)
    R_cur = R_ref @ R_delta
    return matrix_to_rotation_6d(R_cur)


def euler_xyz_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    """Convert XYZ extrinsic euler angles to rotation matrix: R = Rz @ Ry @ Rx.

    Args:
        euler: (..., 3) tensor of (x_angle, y_angle, z_angle).

    Returns:
        (..., 3, 3) rotation matrix.
    """
    x, y, z = euler[..., 0], euler[..., 1], euler[..., 2]
    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)

    # R = Rz(z) @ Ry(y) @ Rx(x)
    r00 = cy * cz
    r01 = sx * sy * cz - cx * sz
    r02 = cx * sy * cz + sx * sz
    r10 = cy * sz
    r11 = sx * sy * sz + cx * cz
    r12 = cx * sy * sz - sx * cz
    r20 = -sy
    r21 = sx * cy
    r22 = cx * cy

    return torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)  # (..., 3, 3)


def matrix_to_rotation_6d(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """Convert 3x3 rotation matrix to 6D representation (first two columns).

    Args:
        rotation_matrix: (..., 3, 3) tensor.

    Returns:
        (..., 6) tensor — col0(3) + col1(3).
    """
    # Stack column 0 and column 1 (not flatten, which would interleave)
    return torch.cat([rotation_matrix[..., :, 0], rotation_matrix[..., :, 1]], dim=-1)


def matrix_to_euler_xyz(R: torch.Tensor, threshold: float = 1e-6) -> torch.Tensor:
    """
    R: (..., 3, 3)
    return: (..., 3)  # (roll, pitch, yaw)

    Inverse convention of:
        euler_xyz_to_matrix(euler): R = Rz(z) @ Ry(y) @ Rx(x)
    """
    sy = -R[..., 2, 0]
    mask = sy.abs() < (1 - threshold)

    roll = torch.zeros_like(sy)
    pitch = torch.zeros_like(sy)
    yaw = torch.zeros_like(sy)

    # normal case
    roll_normal  = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    pitch_normal = torch.asin(sy)
    yaw_normal   = torch.atan2(R[..., 1, 0], R[..., 0, 0])

    # gimbal lock: fix roll = 0
    roll_gimbal  = torch.zeros_like(sy)
    pitch_gimbal = (torch.pi / 2) * torch.sign(sy)
    yaw_gimbal   = torch.atan2(-R[..., 0, 1], R[..., 1, 1])

    roll  = torch.where(mask, roll_normal, roll_gimbal)
    pitch = torch.where(mask, pitch_normal, pitch_gimbal)
    yaw   = torch.where(mask, yaw_normal, yaw_gimbal)

    return torch.stack([roll, pitch, yaw], dim=-1)


def quaternion_to_euler_xyz(quat: torch.Tensor, quat_order: str = "xyzw") -> torch.Tensor:
    """
    Convert quaternion to Euler XYZ angles.

    Args:
        quat: (..., 4)
        quat_order: "xyzw" or "wxyz"

    Returns:
        (..., 3) tensor of (x, y, z)
    """
    R = quaternion_to_matrix(quat, quat_order=quat_order)
    return matrix_to_euler_xyz(R)


def quaternion_to_rotation_6d(quat: torch.Tensor, quat_order: str = "xyzw") -> torch.Tensor:
    """
    Convert quaternion to rotation 6D.

    Args:
        quat: (..., 4)
        quat_order: "xyzw" or "wxyz"

    Returns:
        (..., 6)
    """
    R = quaternion_to_matrix(quat, quat_order=quat_order)
    return matrix_to_rotation_6d(R)


def _unwrap_angles_delta(delta: torch.Tensor) -> torch.Tensor:
    return torch.remainder(delta + torch.pi, 2 * torch.pi) - torch.pi
    

def rela_eef_to_abs(action, state):
    """
    action: (T, C), relative action
    state:  (1, C) or (C,), absolute reference state

    C == 18:
        left(xyz + rot6d) + right(xyz + rot6d)

    C == 12:
        left(xyz + rpy) + right(xyz + rpy)

    return:
        abs action: (T, C)
    """
    if not isinstance(action, torch.Tensor):
        action = torch.from_numpy(action)

    if not isinstance(state, torch.Tensor):
        state = torch.from_numpy(state)

    state = state.to(device=action.device, dtype=action.dtype)

    if state.ndim == 1:
        state = state.unsqueeze(0)

    assert action.ndim == 2, f"action should be (T, C), got {action.shape}"
    assert state.ndim == 2 and state.shape[0] == 1, f"state should be (1, C) or (C,), got {state.shape}"

    T, D = action.shape
    assert D in (12, 18), f"Expected D in (12, 18), got {D}"
    assert state.shape[1] == D, f"state dim mismatch: state.shape={state.shape}, action.shape={action.shape}"

    ref_val = state
    abs_action = torch.zeros_like(action)

    if D == 18:
        # layout:
        # left:  xyz(0:3) + rot6d(3:9)
        # right: xyz(9:12) + rot6d(12:18)

        left_pos = [0, 1, 2]
        right_pos = [9, 10, 11]
        left_rot_indices = list(range(3, 9))
        right_rot_indices = list(range(12, 18))

        # reference rotation
        R_left_ref = rotation_6d_to_matrix(ref_val[:, left_rot_indices])      # (1, 3, 3)
        R_right_ref = rotation_6d_to_matrix(ref_val[:, right_rot_indices])    # (1, 3, 3)

        # Position inverse:
        # p_rela = R_ref^T @ (p_abs - p_ref)
        # => p_abs = p_ref + R_ref @ p_rela
        left_p_rela = action[:, left_pos].unsqueeze(-1)       # (T, 3, 1)
        right_p_rela = action[:, right_pos].unsqueeze(-1)     # (T, 3, 1)

        abs_action[:, left_pos] = (
            ref_val[:, left_pos] + (R_left_ref @ left_p_rela).squeeze(-1)
        )
        abs_action[:, right_pos] = (
            ref_val[:, right_pos] + (R_right_ref @ right_p_rela).squeeze(-1)
        )

        # Rotation inverse:
        # R_delta = R_ref^T @ R_abs
        # => R_abs = R_ref @ R_delta
        abs_action[:, left_rot_indices] = apply_ee6d_delta(
            ref_val[:, left_rot_indices].expand(T, -1),
            action[:, left_rot_indices],
        )
        abs_action[:, right_rot_indices] = apply_ee6d_delta(
            ref_val[:, right_rot_indices].expand(T, -1),
            action[:, right_rot_indices],
        )

    else:
        # layout:
        # left:  xyz(0:3) + rpy(3:6)
        # right: xyz(6:9) + rpy(9:12)

        left_pos = [0, 1, 2]
        right_pos = [6, 7, 8]
        left_rpy = [3, 4, 5]
        right_rpy = [9, 10, 11]

        # reference rotation
        R_left_ref = euler_xyz_to_matrix(ref_val[:, left_rpy])      # (1, 3, 3)
        R_right_ref = euler_xyz_to_matrix(ref_val[:, right_rpy])    # (1, 3, 3)

        # Position inverse:
        # p_rela = R_ref^T @ (p_abs - p_ref)
        # => p_abs = p_ref + R_ref @ p_rela
        left_p_rela = action[:, left_pos].unsqueeze(-1)       # (T, 3, 1)
        right_p_rela = action[:, right_pos].unsqueeze(-1)     # (T, 3, 1)

        abs_action[:, left_pos] = (
            ref_val[:, left_pos] + (R_left_ref @ left_p_rela).squeeze(-1)
        )
        abs_action[:, right_pos] = (
            ref_val[:, right_pos] + (R_right_ref @ right_p_rela).squeeze(-1)
        )

        # Rotation inverse:
        # R_delta = R_ref^T @ R_abs
        # => R_abs = R_ref @ R_delta
        R_left_delta = euler_xyz_to_matrix(action[:, left_rpy])      # (T, 3, 3)
        R_right_delta = euler_xyz_to_matrix(action[:, right_rpy])    # (T, 3, 3)

        R_left_abs = R_left_ref @ R_left_delta                       # (T, 3, 3)
        R_right_abs = R_right_ref @ R_right_delta                   # (T, 3, 3)

        abs_action[:, left_rpy] = matrix_to_euler_xyz(R_left_abs)
        abs_action[:, right_rpy] = matrix_to_euler_xyz(R_right_abs)

        abs_action[:, left_rpy] = _unwrap_angles_delta(abs_action[:, left_rpy])
        abs_action[:, right_rpy] = _unwrap_angles_delta(abs_action[:, right_rpy])

    return abs_action
    

def abs_eef_to_rela(action, state):
    if not isinstance(action, torch.Tensor):
        action = torch.from_numpy(action)
    if not isinstance(state, torch.Tensor):
        state = torch.from_numpy(state)
    T, D = action.shape

    # Determine gripper indices based on action dimension
    assert (D==12 or D==18)
    assert (len(state.shape) == 2)
    assert (state.shape[-1] == D)

    # For non-gripper dimensions: compute delta
    ref_val = state

    if D == 18:

        delta_raw = torch.zeros_like(action)
        left_pos = [0, 1, 2]
        right_pos = [9, 10, 11]
        left_rot_indices = list(range(3, 9))
        right_rot_indices = list(range(12, 18))

        # Body-frame position delta: R_ref^T @ (p_cur - p_ref)
        R_left_ref = rotation_6d_to_matrix(ref_val[:,left_rot_indices])    # (1, 3, 3)
        R_right_ref = rotation_6d_to_matrix(ref_val[:,right_rot_indices])  # (1, 3, 3)
        left_p_diff = (action[:, left_pos] - ref_val[:, left_pos]).unsqueeze(-1)   # (T, 3, 1)
        right_p_diff = (action[:, right_pos] - ref_val[:, right_pos]).unsqueeze(-1)
        delta_raw[:, left_pos] = (R_left_ref.transpose(-1, -2) @ left_p_diff).squeeze(-1)  # (T, 3)
        delta_raw[:, right_pos] = (R_right_ref.transpose(-1, -2) @ right_p_diff).squeeze(-1)

        # Rotation delta: R_ref^T @ R_cur
        delta_raw[:, left_rot_indices] = compute_ee6d_delta(
            action[:, left_rot_indices], ref_val[:, left_rot_indices].expand(T, -1)
        )
        delta_raw[:, right_rot_indices] = compute_ee6d_delta(
            action[:, right_rot_indices], ref_val[:, right_rot_indices].expand(T, -1)
        )

    else:
        delta_raw = action - ref_val

        # Body-frame position delta: R_ref^T @ (p_cur - p_ref)
        # Non-gripper layout (12 dims): left(pos3 + rpy3) + right(pos3 + rpy3)
        left_pos = [0, 1, 2]
        right_pos = [6, 7, 8]
        left_rpy = [3, 4, 5]
        right_rpy = [9, 10, 11]

        R_left_ref = euler_xyz_to_matrix(ref_val[:,left_rpy])    # (1, 3, 3)
        R_right_ref = euler_xyz_to_matrix(ref_val[:,right_rpy])  # (1, 3, 3)
        left_p_diff = delta_raw[:, left_pos].unsqueeze(-1)     # (T, 3, 1)
        right_p_diff = delta_raw[:, right_pos].unsqueeze(-1)
        delta_raw[:, left_pos] = (R_left_ref.transpose(-1, -2) @ left_p_diff).squeeze(-1)  # (T, 3)
        delta_raw[:, right_pos] = (R_right_ref.transpose(-1, -2) @ right_p_diff).squeeze(-1)
        
        R_left_action = euler_xyz_to_matrix(action[:,left_rpy])  # (T,3,3)
        R_right_action = euler_xyz_to_matrix(action[:,right_rpy])
        
        R_rela_left = (R_left_ref.transpose(-1, -2) @ R_left_action)  # (T, 3, 3)
        R_rela_right = (R_right_ref.transpose(-1, -2) @ R_right_action)  # (T, 3, 3)
        
        delta_raw[:, left_rpy] = matrix_to_euler_xyz(R_rela_left)
        delta_raw[:, right_rpy] = matrix_to_euler_xyz(R_rela_right)
        
        # Unwrap RPY angles
        rpy_indices_in_mask = torch.cat(
            [
                torch.arange(3, 6),   # Left RPY
                torch.arange(9, 12),  # Right RPY
            ]
        ).long()
        delta_raw[:, rpy_indices_in_mask] = _unwrap_angles_delta(delta_raw[:, rpy_indices_in_mask])

    return delta_raw


def abs_eef_to_delta(action, state):
    """
    action: (T, C), absolute action chunk
    state:  (1, C) or (C,), absolute reference state

    Delta definition:
        delta[0] = action[0] relative to state
        delta[t] = action[t] relative to action[t - 1], for t > 0

    C == 18:
        left(xyz + rot6d) + right(xyz + rot6d)

    C == 12:
        left(xyz + rpy) + right(xyz + rpy)

    return:
        delta action: (T, C)
    """
    if not isinstance(action, torch.Tensor):
        action = torch.from_numpy(action)

    if not isinstance(state, torch.Tensor):
        state = torch.from_numpy(state)

    state = state.to(device=action.device, dtype=action.dtype)

    if state.ndim == 1:
        state = state.unsqueeze(0)

    assert action.ndim == 2, f"action should be (T, C), got {action.shape}"
    assert state.ndim == 2 and state.shape[0] == 1, f"state should be (1, C) or (C,), got {state.shape}"

    T, D = action.shape
    assert D in (12, 18), f"Expected D in (12, 18), got {D}"
    assert state.shape[1] == D, f"state dim mismatch: state.shape={state.shape}, action.shape={action.shape}"

    # frame-wise reference:
    # ref[0] = state
    # ref[t] = action[t - 1]
    if T > 1:
        ref_val = torch.cat([state, action[:-1]], dim=0)  # (T, D)
    else:
        ref_val = state                                  # (1, D)

    delta_raw = torch.zeros_like(action)

    if D == 18:
        # layout:
        # left:  xyz(0:3) + rot6d(3:9)
        # right: xyz(9:12) + rot6d(12:18)

        left_pos = [0, 1, 2]
        right_pos = [9, 10, 11]
        left_rot_indices = list(range(3, 9))
        right_rot_indices = list(range(12, 18))

        # reference rotation
        R_left_ref = rotation_6d_to_matrix(ref_val[:, left_rot_indices])      # (T, 3, 3)
        R_right_ref = rotation_6d_to_matrix(ref_val[:, right_rot_indices])    # (T, 3, 3)

        # Body-frame position delta:
        # p_delta = R_ref^T @ (p_cur - p_ref)
        left_p_diff = (action[:, left_pos] - ref_val[:, left_pos]).unsqueeze(-1)    # (T, 3, 1)
        right_p_diff = (action[:, right_pos] - ref_val[:, right_pos]).unsqueeze(-1)

        delta_raw[:, left_pos] = (
            R_left_ref.transpose(-1, -2) @ left_p_diff
        ).squeeze(-1)
        delta_raw[:, right_pos] = (
            R_right_ref.transpose(-1, -2) @ right_p_diff
        ).squeeze(-1)

        # Rotation delta:
        # R_delta = R_ref^T @ R_cur
        delta_raw[:, left_rot_indices] = compute_ee6d_delta(
            action[:, left_rot_indices],
            ref_val[:, left_rot_indices],
        )
        delta_raw[:, right_rot_indices] = compute_ee6d_delta(
            action[:, right_rot_indices],
            ref_val[:, right_rot_indices],
        )

    else:
        # layout:
        # left:  xyz(0:3) + rpy(3:6)
        # right: xyz(6:9) + rpy(9:12)

        left_pos = [0, 1, 2]
        right_pos = [6, 7, 8]
        left_rpy = [3, 4, 5]
        right_rpy = [9, 10, 11]

        # reference rotation
        R_left_ref = euler_xyz_to_matrix(ref_val[:, left_rpy])      # (T, 3, 3)
        R_right_ref = euler_xyz_to_matrix(ref_val[:, right_rpy])    # (T, 3, 3)

        # Body-frame position delta:
        # p_delta = R_ref^T @ (p_cur - p_ref)
        left_p_diff = (action[:, left_pos] - ref_val[:, left_pos]).unsqueeze(-1)    # (T, 3, 1)
        right_p_diff = (action[:, right_pos] - ref_val[:, right_pos]).unsqueeze(-1)

        delta_raw[:, left_pos] = (
            R_left_ref.transpose(-1, -2) @ left_p_diff
        ).squeeze(-1)
        delta_raw[:, right_pos] = (
            R_right_ref.transpose(-1, -2) @ right_p_diff
        ).squeeze(-1)

        # Rotation delta:
        # R_delta = R_ref^T @ R_cur
        R_left_action = euler_xyz_to_matrix(action[:, left_rpy])    # (T, 3, 3)
        R_right_action = euler_xyz_to_matrix(action[:, right_rpy])  # (T, 3, 3)

        R_left_delta = R_left_ref.transpose(-1, -2) @ R_left_action
        R_right_delta = R_right_ref.transpose(-1, -2) @ R_right_action

        delta_raw[:, left_rpy] = matrix_to_euler_xyz(R_left_delta)
        delta_raw[:, right_rpy] = matrix_to_euler_xyz(R_right_delta)

        # Unwrap RPY delta angles
        rpy_indices_in_mask = torch.cat(
            [
                torch.arange(3, 6, device=action.device),
                torch.arange(9, 12, device=action.device),
            ]
        ).long()
        delta_raw[:, rpy_indices_in_mask] = _unwrap_angles_delta(
            delta_raw[:, rpy_indices_in_mask]
        )

    return delta_raw
