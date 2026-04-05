from os.path import join as pjoin

from ..common.skeleton import Skeleton
import numpy as np
import os
from ..common.quaternion import *
from ..utils.paramUtil import *

import torch
from tqdm import tqdm
l_idx1, l_idx2 = 5, 8
# Right/Left foot
fid_r, fid_l = [8, 11], [7, 10]
# Face direction, r_hip, l_hip, sdr_r, sdr_l
face_joint_indx = [2, 1, 17, 16]
# l_hip, r_hip
r_hip, l_hip = 2, 1
joints_num = 22 
n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
kinematic_chain = t2m_kinematic_chain
# positions (batch, joint_num, 3)
def uniform_skeleton(positions, target_offset):
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset#.numpy()
    # print(src_offset)
    # print(tgt_offset)
    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    # print(scale_rt)
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    # print(quat_params.shape)

    '''Forward Kinematics'''
    src_skel.set_offset(torch.tensor(tgt_offset))
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos.reshape(-1,3))
    return new_joints


def extract_features(positions, feet_thre, n_raw_offsets, kinematic_chain, face_joint_indx, fid_r, fid_l):
    global_positions = positions.copy()
    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float64)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float64)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float64)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float64)
        return feet_l, feet_r

    #
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    #     trejec = np.cumsum(np.concatenate([np.array([[0, 0, 0]]), velocity], axis=0), axis=0)
    #     r_rotations, r_pos = recover_ric_glo_np(r_velocity, velocity[:, [0, 2]])

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(ground_positions[:, 0, 0], ground_positions[:, 0, 2], marker='o', color='r')
    # plt.plot(trejec[:, 0], trejec[:, 2], marker='^', color='g')
    # plt.plot(r_pos[:, 0], r_pos[:, 2], marker='s', color='y')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    #     print(dataset.shape, local_vel.shape)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data


def _matrix_to_quaternion_np(matrix):
    """Convert rotation matrices (..., 3, 3) to quaternions (..., 4), real-first."""
    m = np.asarray(matrix)
    if m.shape[-2:] != (3, 3):
        raise ValueError(f"matrix must have shape (...,3,3), got {m.shape}")

    q = np.zeros(m.shape[:-2] + (4,), dtype=m.dtype)
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]

    mask = trace > 0.0
    if np.any(mask):
        s = np.sqrt(trace[mask] + 1.0) * 2.0
        q[mask, 0] = 0.25 * s
        q[mask, 1] = (m[mask, 2, 1] - m[mask, 1, 2]) / s
        q[mask, 2] = (m[mask, 0, 2] - m[mask, 2, 0]) / s
        q[mask, 3] = (m[mask, 1, 0] - m[mask, 0, 1]) / s

    mask_x = (~mask) & (m[..., 0, 0] > m[..., 1, 1]) & (m[..., 0, 0] > m[..., 2, 2])
    if np.any(mask_x):
        s = np.sqrt(1.0 + m[mask_x, 0, 0] - m[mask_x, 1, 1] - m[mask_x, 2, 2]) * 2.0
        q[mask_x, 0] = (m[mask_x, 2, 1] - m[mask_x, 1, 2]) / s
        q[mask_x, 1] = 0.25 * s
        q[mask_x, 2] = (m[mask_x, 0, 1] + m[mask_x, 1, 0]) / s
        q[mask_x, 3] = (m[mask_x, 0, 2] + m[mask_x, 2, 0]) / s

    mask_y = (~mask) & (~mask_x) & (m[..., 1, 1] > m[..., 2, 2])
    if np.any(mask_y):
        s = np.sqrt(1.0 + m[mask_y, 1, 1] - m[mask_y, 0, 0] - m[mask_y, 2, 2]) * 2.0
        q[mask_y, 0] = (m[mask_y, 0, 2] - m[mask_y, 2, 0]) / s
        q[mask_y, 1] = (m[mask_y, 0, 1] + m[mask_y, 1, 0]) / s
        q[mask_y, 2] = 0.25 * s
        q[mask_y, 3] = (m[mask_y, 1, 2] + m[mask_y, 2, 1]) / s

    mask_z = (~mask) & (~mask_x) & (~mask_y)
    if np.any(mask_z):
        s = np.sqrt(1.0 + m[mask_z, 2, 2] - m[mask_z, 0, 0] - m[mask_z, 1, 1]) * 2.0
        q[mask_z, 0] = (m[mask_z, 1, 0] - m[mask_z, 0, 1]) / s
        q[mask_z, 1] = (m[mask_z, 0, 2] + m[mask_z, 2, 0]) / s
        q[mask_z, 2] = (m[mask_z, 1, 2] + m[mask_z, 2, 1]) / s
        q[mask_z, 3] = 0.25 * s

    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-8)
    return q


def process_file_from_rot(cont6d_params, root_positions, align_canonical=True):
    """
    Build root_data from rotation 6D + root trajectory.

    Args:
        cont6d_params: (seq_len, joints_num, 6), joint 0 is global root rotation.
            Supports numpy.ndarray or torch.Tensor.
        root_positions: (seq_len, 3), global root trajectory.
            Supports numpy.ndarray or torch.Tensor.
        align_canonical: if True, apply process_file-like canonicalization.

    Returns:
        root_data: (seq_len - 1, 4), [yaw_vel, vel_x, vel_z, root_y].
    """
    input_is_torch = torch.is_tensor(cont6d_params) or torch.is_tensor(root_positions)
    out_device = None
    out_dtype = None

    if torch.is_tensor(root_positions):
        out_device = root_positions.device
        out_dtype = root_positions.dtype
    elif torch.is_tensor(cont6d_params):
        out_device = cont6d_params.device
        out_dtype = cont6d_params.dtype

    if torch.is_tensor(cont6d_params):
        cont6d_params = cont6d_params.detach().cpu().numpy()
    else:
        cont6d_params = np.asarray(cont6d_params)

    if torch.is_tensor(root_positions):
        root_positions = root_positions.detach().cpu().numpy()
    else:
        root_positions = np.asarray(root_positions)

    if cont6d_params.ndim != 3 or cont6d_params.shape[-1] != 6:
        raise ValueError(f"cont6d_params must have shape (T, J, 6), got {cont6d_params.shape}")
    if root_positions.ndim != 2 or root_positions.shape[-1] != 3:
        raise ValueError(f"root_positions must have shape (T, 3), got {root_positions.shape}")
    if cont6d_params.shape[0] != root_positions.shape[0]:
        raise ValueError("cont6d_params and root_positions must have the same sequence length")
    if cont6d_params.shape[1] < 1:
        raise ValueError("cont6d_params must contain at least the root joint")

    valid_mask = np.isfinite(cont6d_params).all(axis=(1, 2)) & np.isfinite(root_positions).all(axis=1)
    cont6d_params = cont6d_params[valid_mask]
    root_positions = root_positions[valid_mask]
    root_positions, pre_mask = filter_invalid_frames(root_positions, "before uniform_skeleton", return_mask=True)
    cont6d_params = cont6d_params[pre_mask]

    root_rot_mats = cont6d_to_matrix_np(cont6d_params[:, 0])
    root_quat = _matrix_to_quaternion_np(root_rot_mats)
    root_quat = qfix(root_quat[:, None, :])[:, 0, :]

    root_positions = root_positions.copy()
    if align_canonical:
        floor_height = root_positions[:, 1].min()
        root_positions[:, 1] -= floor_height

        root_pose_init_xz = root_positions[0] * np.array([1.0, 0.0, 1.0])
        root_positions = root_positions - root_pose_init_xz

        forward_init = qrot_np(root_quat[0:1], np.array([[0.0, 0.0, 1.0]]))
        forward_init[:, 1] = 0.0
        forward_init = forward_init / np.linalg.norm(forward_init, axis=-1, keepdims=True).clip(min=1e-8)

        target = np.array([[0.0, 0.0, 1.0]])
        root_quat_init = qbetween_np(forward_init, target)

        root_quat_init_for_all = np.repeat(root_quat_init, len(root_positions), axis=0)
        root_positions = qrot_np(root_quat_init_for_all, root_positions)
        root_quat = qmul_np(root_quat_init_for_all, root_quat)

    root_positions, post_mask = filter_invalid_frames(root_positions, "after uniform_skeleton", return_mask=True)
    root_quat = root_quat[post_mask]

    velocity = (root_positions[1:] - root_positions[:-1]).copy()
    velocity = qrot_np(root_quat[1:], velocity)

    r_velocity = qmul_np(root_quat[1:], qinv_np(root_quat[:-1]))
    yaw_velocity = np.arcsin(np.clip(r_velocity[:, 2:3], -1.0, 1.0))
    l_velocity = velocity[:, [0, 2]]
    root_y = root_positions[:, 1:2]

    root_data = np.concatenate([yaw_velocity, l_velocity, root_y[:-1]], axis=-1)
    if not np.isfinite(root_data).all():
        #raise ValueError(f"root_data contains NaN/Inf, shape={root_data.shape}")
        root_data = np.nan_to_num(root_data, nan=0.0, posinf=0.0, neginf=0.0)

    if input_is_torch:
        root_data_t = torch.from_numpy(root_data)
        if out_dtype is not None and torch.is_floating_point(torch.empty((), dtype=out_dtype)):
            root_data_t = root_data_t.to(dtype=out_dtype)
        if out_device is not None:
            root_data_t = root_data_t.to(device=out_device)
        return root_data_t

    return root_data


def process_file_from_root_and_hips(root_positions, r_hip_positions, l_hip_positions, align_canonical=True):
    """
    Build root_data from positions only (no rotation input required).

    Args:
        root_positions: (seq_len, 3), global root trajectory.
        r_hip_positions: (seq_len, 3), right hip trajectory.
        l_hip_positions: (seq_len, 3), left hip trajectory.
        align_canonical: if True, apply process_file-like canonicalization.

    Returns:
        root_data: (seq_len - 1, 4), [yaw_vel, vel_x, vel_z, root_y].
    """
    input_is_torch = (
        torch.is_tensor(root_positions)
        or torch.is_tensor(r_hip_positions)
        or torch.is_tensor(l_hip_positions)
    )
    out_device = None
    out_dtype = None

    for x in (root_positions, r_hip_positions, l_hip_positions):
        if torch.is_tensor(x):
            out_device = x.device
            out_dtype = x.dtype
            break

    def _to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    root_positions = _to_numpy(root_positions)
    r_hip_positions = _to_numpy(r_hip_positions)
    l_hip_positions = _to_numpy(l_hip_positions)

    if root_positions.ndim != 2 or root_positions.shape[-1] != 3:
        raise ValueError(f"root_positions must have shape (T, 3), got {root_positions.shape}")
    if r_hip_positions.ndim != 2 or r_hip_positions.shape[-1] != 3:
        raise ValueError(f"r_hip_positions must have shape (T, 3), got {r_hip_positions.shape}")
    if l_hip_positions.ndim != 2 or l_hip_positions.shape[-1] != 3:
        raise ValueError(f"l_hip_positions must have shape (T, 3), got {l_hip_positions.shape}")
    if not (len(root_positions) == len(r_hip_positions) == len(l_hip_positions)):
        raise ValueError("root/hip positions must have the same sequence length")

    # Joint-like packing, so we can reuse the same frame-cleaning routine as process_file.
    packed = np.stack([root_positions, r_hip_positions, l_hip_positions], axis=1)
    packed, pre_mask = filter_invalid_frames(packed, "before uniform_skeleton", return_mask=True)

    root_positions = packed[:, 0].copy()
    r_hip_positions = packed[:, 1].copy()
    l_hip_positions = packed[:, 2].copy()

    across = r_hip_positions - l_hip_positions
    across = across / np.linalg.norm(across, axis=-1, keepdims=True).clip(min=1e-8)

    y_axis = np.zeros_like(across)
    y_axis[:, 1] = 1.0
    forward = np.cross(y_axis, across)
    forward = forward / np.linalg.norm(forward, axis=-1, keepdims=True).clip(min=1e-8)

    # Build yaw-only quaternion that rotates each forward vector to +Z.
    yaw = np.arctan2(forward[:, 0], forward[:, 2])
    half = -0.5 * yaw
    root_quat = np.stack([
        np.cos(half),
        np.zeros_like(half),
        np.sin(half),
        np.zeros_like(half),
    ], axis=-1)
    root_quat = qfix(root_quat[:, None, :])[:, 0, :]

    if align_canonical:
        floor_height = root_positions[:, 1].min()
        root_positions[:, 1] -= floor_height
        r_hip_positions[:, 1] -= floor_height
        l_hip_positions[:, 1] -= floor_height

        root_pose_init_xz = root_positions[0] * np.array([1.0, 0.0, 1.0])
        root_positions = root_positions - root_pose_init_xz
        r_hip_positions = r_hip_positions - root_pose_init_xz
        l_hip_positions = l_hip_positions - root_pose_init_xz

        forward_init = forward[0:1].copy()
        forward_init[:, 1] = 0.0
        forward_init = forward_init / np.linalg.norm(forward_init, axis=-1, keepdims=True).clip(min=1e-8)
        init_yaw = np.arctan2(forward_init[:, 0], forward_init[:, 2])
        init_half = -0.5 * init_yaw
        root_quat_init = np.stack([
            np.cos(init_half),
            np.zeros_like(init_half),
            np.sin(init_half),
            np.zeros_like(init_half),
        ], axis=-1)

        root_quat_init_for_all = np.repeat(root_quat_init, len(root_positions), axis=0)
        root_positions = qrot_np(root_quat_init_for_all, root_positions)
        r_hip_positions = qrot_np(root_quat_init_for_all, r_hip_positions)
        l_hip_positions = qrot_np(root_quat_init_for_all, l_hip_positions)
        root_quat = qmul_np(root_quat_init_for_all, root_quat)

    packed_post = np.stack([root_positions, r_hip_positions, l_hip_positions], axis=1)
    packed_post, post_mask = filter_invalid_frames(packed_post, "after uniform_skeleton", return_mask=True)
    root_positions = packed_post[:, 0]
    root_quat = root_quat[post_mask]

    velocity = (root_positions[1:] - root_positions[:-1]).copy()
    velocity = qrot_np(root_quat[1:], velocity)

    r_velocity = qmul_np(root_quat[1:], qinv_np(root_quat[:-1]))
    yaw_velocity = np.arcsin(np.clip(r_velocity[:, 2:3], -1.0, 1.0))
    l_velocity = velocity[:, [0, 2]]
    root_y = root_positions[:, 1:2]

    root_data = np.concatenate([yaw_velocity, l_velocity, root_y[:-1]], axis=-1)
    if not np.isfinite(root_data).all():
        root_data = np.nan_to_num(root_data, nan=0.0, posinf=0.0, neginf=0.0)

    if input_is_torch:
        root_data_t = torch.from_numpy(root_data)
        if out_dtype is not None and torch.is_floating_point(torch.empty((), dtype=out_dtype)):
            root_data_t = root_data_t.to(dtype=out_dtype)
        if out_device is not None:
            root_data_t = root_data_t.to(device=out_device)
        return root_data_t

    return root_data


def get_tgt():
    tgt = np.load('experiments/offset.npy')
    return tgt


def filter_invalid_frames(positions, stage_name, return_mask=False):
    if positions.ndim == 3:
        valid_mask = np.isfinite(positions).all(axis=(1, 2))
    elif positions.ndim == 2:
        valid_mask = np.isfinite(positions).all(axis=1)
    else:
        raise ValueError(f"{stage_name}: unsupported ndim={positions.ndim}, expected 2 or 3")

    positions = positions[valid_mask]

    if len(positions) < 2:
        raise ValueError(f"{stage_name}: not enough valid frames after filtering NaN/Inf")

    if return_mask:
        return positions, valid_mask

    return positions


def process_file(positions, feet_thre):
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    '''Uniform Skeleton'''

    tgt_offsets = get_tgt()

    positions = filter_invalid_frames(positions, "before uniform_skeleton")
    positions = uniform_skeleton(positions, tgt_offsets)
    positions = filter_invalid_frames(positions, "after uniform_skeleton")

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height
    #     print(floor_height)

    #     plot_3d_motion("./positions_1.mp4", kinematic_chain, positions, 'title', fps=20)

    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # '''Move the first pose to origin '''
    # root_pos_init = positions[0]
    # positions = positions - root_pos_init[0]

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    #     print(forward_init)

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions_b = positions.copy()

    positions = qrot_np(root_quat_init, positions)

    #     plot_3d_motion("./positions_2.mp4", kinematic_chain, positions, 'title', fps=20)

    '''New ground truth positions'''
    global_positions = positions.copy()

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(positions[:, 0, 0], positions[:, 0, 2], marker='o', color='r')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float64)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float64)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float64)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float64)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    #     trejec = np.cumsum(np.concatenate([np.array([[0, 0, 0]]), velocity], axis=0), axis=0)
    #     r_rotations, r_pos = recover_ric_glo_np(r_velocity, velocity[:, [0, 2]])

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(ground_positions[:, 0, 0], ground_positions[:, 0, 2], marker='o', color='r')
    # plt.plot(trejec[:, 0], trejec[:, 2], marker='^', color='g')
    # plt.plot(r_pos[:, 0], r_pos[:, 2], marker='s', color='y')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    #     print(dataset.shape, local_vel.shape)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)
    assert not np.isnan(data).any(), f"data contains NaN, shape={data.shape}"

    return data, global_positions, positions, l_velocity


# Recover global angle and positions for rotation dataset
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions

def recover_rot(data):
    # dataset [bs, seqlen, 263/251] HumanML/KIT
    joints_num = 22 if data.shape[-1] == 263 else 21
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    r_pos_pad = torch.cat([r_pos, torch.zeros_like(r_pos)], dim=-1).unsqueeze(-2)
    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)
    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)
    cont6d_params = torch.cat([cont6d_params, r_pos_pad], dim=-2)
    return cont6d_params


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions
'''
For Text2Motion Dataset
'''
'''
if __name__ == "__main__":
    example_id = "000021"
    # Lower legs
    l_idx1, l_idx2 = 5, 8
    # Right/Left foot
    fid_r, fid_l = [8, 11], [7, 10]
    # Face direction, r_hip, l_hip, sdr_r, sdr_l
    face_joint_indx = [2, 1, 17, 16]
    # l_hip, r_hip
    r_hip, l_hip = 2, 1
    joints_num = 22
    # ds_num = 8
    data_dir = '../dataset/pose_data_raw/joints/'
    save_dir1 = '../dataset/pose_data_raw/new_joints/'
    save_dir2 = '../dataset/pose_data_raw/new_joint_vecs/'

    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_kinematic_chain

    # Get offsets of target skeleton
    example_data = np.load(os.path.join(data_dir, example_id + '.npy'))
    example_data = example_data.reshape(len(example_data), -1, 3)
    example_data = torch.from_numpy(example_data)
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    # (joints_num, 3)
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
    # print(tgt_offsets)

    source_list = os.listdir(data_dir)
    frame_num = 0
    for source_file in tqdm(source_list):
        source_data = np.load(os.path.join(data_dir, source_file))[:, :joints_num]
        try:
            dataset, ground_positions, positions, l_velocity = process_file(source_data, 0.002)
            rec_ric_data = recover_from_ric(torch.from_numpy(dataset).unsqueeze(0).float(), joints_num)
            np.save(pjoin(save_dir1, source_file), rec_ric_data.squeeze().numpy())
            np.save(pjoin(save_dir2, source_file), dataset)
            frame_num += dataset.shape[0]
        except Exception as e:
            print(source_file)
            print(e)

    print('Total clips: %d, Frames: %d, Duration: %fm' %
          (len(source_list), frame_num, frame_num / 20 / 60))
'''

if __name__ == "__main__":
    example_id = "03950_gt"
    # Lower legs
    l_idx1, l_idx2 = 17, 18
    # Right/Left foot
    fid_r, fid_l = [14, 15], [19, 20]
    # Face direction, r_hip, l_hip, sdr_r, sdr_l
    face_joint_indx = [11, 16, 5, 8]
    # l_hip, r_hip
    r_hip, l_hip = 11, 16
    joints_num = 21
    # ds_num = 8
    data_dir = '../dataset/kit_mocap_dataset/joints/'
    save_dir1 = '../dataset/kit_mocap_dataset/new_joints/'
    save_dir2 = '../dataset/kit_mocap_dataset/new_joint_vecs/'

    n_raw_offsets = torch.from_numpy(kit_raw_offsets)
    kinematic_chain = kit_kinematic_chain

    '''Get offsets of target skeleton'''
    example_data = np.load(os.path.join(data_dir, example_id + '.npy'))
    example_data = example_data.reshape(len(example_data), -1, 3)
    example_data = torch.from_numpy(example_data)
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    # (joints_num, 3)
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
    # print(tgt_offsets)

    source_list = os.listdir(data_dir)
    frame_num = 0
    '''Read source dataset'''
    for source_file in tqdm(source_list):
        source_data = np.load(os.path.join(data_dir, source_file))[:, :joints_num]
        try:
            name = ''.join(source_file[:-7].split('_')) + '.npy'
            data, ground_positions, positions, l_velocity = process_file(source_data, 0.05)
            rec_ric_data = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), joints_num)
            if np.isnan(rec_ric_data.numpy()).any():
                print(source_file)
                continue
            np.save(pjoin(save_dir1, name), rec_ric_data.squeeze().numpy())
            np.save(pjoin(save_dir2, name), data)
            frame_num += data.shape[0]
        except Exception as e:
            print(source_file)
            print(e)

    print('Total clips: %d, Frames: %d, Duration: %fm' %
          (len(source_list), frame_num, frame_num / 12.5 / 60))
