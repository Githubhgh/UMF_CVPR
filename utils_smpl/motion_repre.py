from utils.quaternion import qrot_np
import numpy as np
import torch

def get_rifke(positions, r_rot):
    '''Local pose'''
    positions[..., 0] -= positions[:, 0:1, 0]
    positions[..., 2] -= positions[:, 0:1, 2]
    '''All pose face Z+'''
    positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
    return positions


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



import numpy as np

# Define a kinematic tree for the skeletal struture
kit_kinematic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]]

kit_raw_offsets = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [-1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, 1]
    ]
)

t2m_raw_offsets = np.array([[0,0,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,0,1],
                           [0,0,1],
                           [0,1,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,0,1],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0]])

t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
t2m_left_hand_chain = [[20, 22, 23, 24], [20, 34, 35, 36], [20, 25, 26, 27], [20, 31, 32, 33], [20, 28, 29, 30]]
t2m_right_hand_chain = [[21, 43, 44, 45], [21, 46, 47, 48], [21, 40, 41, 42], [21, 37, 38, 39], [21, 49, 50, 51]]


### define Global variable
n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
kinematic_chain = t2m_kinematic_chain
face_joint_indx = [2, 1, 17, 16]


from common.quaternion import *
import scipy.ndimage.filters as filters

class Skeleton(object):
    def __init__(self, offset, kinematic_tree, device):
        self.device = device
        self._raw_offset_np = offset.numpy()
        self._raw_offset = offset.clone().detach().to(device).float()
        self._kinematic_tree = kinematic_tree
        self._offset = None
        self._parents = [0] * len(self._raw_offset)
        self._parents[0] = -1
        for chain in self._kinematic_tree:
            for j in range(1, len(chain)):
                self._parents[chain[j]] = chain[j-1]

    def njoints(self):
        return len(self._raw_offset)

    def offset(self):
        return self._offset

    def set_offset(self, offsets):
        self._offset = offsets.clone().detach().to(self.device).float()

    def kinematic_tree(self):
        return self._kinematic_tree

    def parents(self):
        return self._parents

    # joints (batch_size, joints_num, 3)
    def get_offsets_joints_batch(self, joints):
        assert len(joints.shape) == 3
        _offsets = self._raw_offset.expand(joints.shape[0], -1, -1).clone()
        for i in range(1, self._raw_offset.shape[0]):
            _offsets[:, i] = torch.norm(joints[:, i] - joints[:, self._parents[i]], p=2, dim=1)[:, None] * _offsets[:, i]

        self._offset = _offsets.detach()
        return _offsets

    # joints (joints_num, 3)
    def get_offsets_joints(self, joints):
        assert len(joints.shape) == 2
        _offsets = self._raw_offset.clone()
        for i in range(1, self._raw_offset.shape[0]):
            # print(joints.shape)
            _offsets[i] = torch.norm(joints[i] - joints[self._parents[i]], p=2, dim=0) * _offsets[i]

        self._offset = _offsets.detach()
        return _offsets

    # face_joint_idx should follow the order of right hip, left hip, right shoulder, left shoulder
    # joints (batch_size, joints_num, 3)
    def inverse_kinematics_np(self, joints, face_joint_idx, smooth_forward=False):
        assert len(face_joint_idx) == 4
        '''Get Forward Direction'''
        l_hip, r_hip, sdr_r, sdr_l = face_joint_idx
        across1 = joints[:, r_hip] - joints[:, l_hip]
        across2 = joints[:, sdr_r] - joints[:, sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across**2).sum(axis=-1))[:, np.newaxis]
        # print(across1.shape, across2.shape)

        # forward (batch_size, 3)
        forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        if smooth_forward:
            forward = filters.gaussian_filter1d(forward, 20, axis=0, mode='nearest')
            # forward (batch_size, 3)
        forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

        '''Get Root Rotation'''
        target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
        root_quat = qbetween_np(forward, target)

        '''Inverse Kinematics'''
        # quat_params (batch_size, joints_num, 4)
        # print(joints.shape[:-1])
        quat_params = np.zeros(joints.shape[:-1] + (4,))
        # print(quat_params.shape)
        root_quat[0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        quat_params[:, 0] = root_quat
        # quat_params[0, 0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        for chain in self._kinematic_tree:
            R = root_quat
            for j in range(len(chain) - 1):
                # (batch, 3)
                u = self._raw_offset_np[chain[j+1]][np.newaxis,...].repeat(len(joints), axis=0)
                # print(u.shape)
                # (batch, 3)
                v = joints[:, chain[j+1]] - joints[:, chain[j]]
                v = v / np.sqrt((v**2).sum(axis=-1))[:, np.newaxis]
                # print(u.shape, v.shape)
                rot_u_v = qbetween_np(u, v)

                R_loc = qmul_np(qinv_np(R), rot_u_v)

                quat_params[:,chain[j + 1], :] = R_loc
                R = qmul_np(R, R_loc)

        return quat_params

    # Be sure root joint is at the beginning of kinematic chains
    def forward_kinematics(self, quat_params, root_pos, skel_joints=None, do_root_R=True):
        # quat_params (batch_size, joints_num, 4)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(quat_params.shape[0], -1, -1)
        joints = torch.zeros(quat_params.shape[:-1] + (3,)).to(self.device)
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                R = quat_params[:, 0]
            else:
                R = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(len(quat_params), -1).detach().to(self.device)
            for i in range(1, len(chain)):
                R = qmul(R, quat_params[:, chain[i]])
                offset_vec = offsets[:, chain[i]]
                joints[:, chain[i]] = qrot(R, offset_vec) + joints[:, chain[i-1]]
        return joints

    # Be sure root joint is at the beginning of kinematic chains
    def forward_kinematics_np(self, quat_params, root_pos, skel_joints=None, do_root_R=True):
        # quat_params (batch_size, joints_num, 4)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(quat_params.shape[0], -1, -1)
        offsets = offsets.numpy()
        joints = np.zeros(quat_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                R = quat_params[:, 0]
            else:
                R = np.array([[1.0, 0.0, 0.0, 0.0]]).repeat(len(quat_params), axis=0)
            for i in range(1, len(chain)):
                R = qmul_np(R, quat_params[:, chain[i]])
                offset_vec = offsets[:, chain[i]]
                joints[:, chain[i]] = qrot_np(R, offset_vec) + joints[:, chain[i - 1]]
        return joints

    def forward_kinematics_cont6d_np(self, cont6d_params, root_pos, skel_joints=None, do_root_R=True):
        # cont6d_params (batch_size, joints_num, 6)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)
        offsets = offsets.numpy()
        joints = np.zeros(cont6d_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                matR = cont6d_to_matrix_np(cont6d_params[:, 0])
            else:
                matR = np.eye(3)[np.newaxis, :].repeat(len(cont6d_params), axis=0)
            for i in range(1, len(chain)):
                matR = np.matmul(matR, cont6d_to_matrix_np(cont6d_params[:, chain[i]]))
                offset_vec = offsets[:, chain[i]][..., np.newaxis]
                # print(matR.shape, offset_vec.shape)
                joints[:, chain[i]] = np.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i-1]]
        return joints

    def forward_kinematics_cont6d(self, cont6d_params, root_pos, skel_joints=None, do_root_R=True):
        # cont6d_params (batch_size, joints_num, 6)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            # skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)
        joints = torch.zeros(cont6d_params.shape[:-1] + (3,)).to(cont6d_params.device)
        joints[..., 0, :] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                matR = cont6d_to_matrix(cont6d_params[:, 0])
            else:
                matR = torch.eye(3).expand((len(cont6d_params), -1, -1)).detach().to(cont6d_params.device)
            for i in range(1, len(chain)):
                matR = torch.matmul(matR, cont6d_to_matrix(cont6d_params[:, chain[i]]))
                offset_vec = offsets[:, chain[i]].unsqueeze(-1)
                # print(matR.shape, offset_vec.shape)
                joints[:, chain[i]] = torch.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i-1]]
        return joints


# def tranform_humanml3d_to_interhuman(hu_motion):

#     assert in_motion.shape[-1] == 262
#     return in_motion




# def tranform__interhuman_to_humanml3d(in_motion):
#     assert in_motion.shape[-1] == 262
#     ### hu_motion should be [root_data:4, ric_data (local position):21*3, rot_data:21*6, local_vel:22*3, ground contact: 4] called Canonical Representation
#     ### in_motion should be [joint position (global position): 21*3, joint velocity (global velocity): 22*3, rot_data:21*6, ground contact: 4] called Non-Canonical Representation



#     assert hu_motion.shape[-1] == 263
#     return hu_motion


def recover_root_rot_pos_full_lengths(data):
    data = torch.tensor(data)
    if data.shape[0] != 1:
        data = data.unsqueeze(0)

    rot_vel = data[..., 0] # [1, 124]
    r_rot_ang = torch.zeros([rot_vel.shape[0], rot_vel.shape[1]+1]).to(data.device) # [1, 125]
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(r_rot_ang.shape + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang) # [1, 125, 4]

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device) # [1, 124, 3]
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat[:,:-1,:]), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos





# Input: hu_motion: [len-1, 263]
# Output: position: [len, 22, 3]
def restore_global_position_from_humanml3d(hu_motion, mode='numpy'):
    if isinstance(hu_motion, torch.Tensor):
        data = hu_motion
    else:
        if hu_motion.shape[0]!=1:
            hu_motion = torch.tensor(hu_motion).unsqueeze(0)
            data = torch.tensor(hu_motion)



    joints_num = 22
    
    r_rot_quat_full_length, r_pos = recover_root_rot_pos_full_lengths(data) #  (1, 124, 263) -> r_pos: [1, 125, 3] r_rot_quat: [1, 125, 4]
    positions = data[..., 4:(joints_num - 1) * 3 + 4] # local position
    positions = positions.view(positions.shape[:-1] + (-1, 3)) # [1,124,21,3]

    velocity = data[..., 4 + 21*3 + 21*6 : 4 + 21*3 + 21*6 + 22*3] # local velocity

    velocity = velocity.reshape(1, positions.shape[1], 22, 3) # [1, 124, 22, 3]

    non_root_velocity = velocity[..., 1:, :]
    root_velocity = velocity[..., 0, :]


    last_pos = (positions[:, -1, :, :] + non_root_velocity[:, -1, :, :]).unsqueeze(1)
    last_root_pos = (r_pos[:, -1, :] + root_velocity[:, -1, :]).unsqueeze(1)
    positions = torch.concatenate([positions, last_pos], 1) # [1, 125, 21, 3]
    r_pos = torch.concatenate([r_pos, last_root_pos], 1) # [1, 125, 3]
    


    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat_full_length[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)



    return np.array(positions.squeeze().cpu()) # [1, 125, 22, 3]


#Input: in_motion: [len-1, 262]
#Output: postions: [len, 22, 3]


def restore_global_position_from_interhuman(in_motion):
    positions = in_motion[:, :22*3]
    posistion_velocity = in_motion[:,22*3:22*3+22*3]

    last_last_pos = positions[-1, :]
    last_vel = posistion_velocity[-1, :]
    last_pos = (last_last_pos+last_vel)
    positions = np.vstack((positions, np.expand_dims(last_pos, axis=0)))

    return positions.reshape(-1, 22, 3)


# Input: Positions: [len, 22, 3]
# Output: data: [len-1, 263]
def generate_humanml3d_from_global_position(positions, rotations=None, feet_thre=0.001, global_flag=True):

    
    
    l_idx1, l_idx2 = 5, 8
    # Right/Left foot
    fid_r, fid_l = [8, 11], [7, 10]
    # Face direction, r_hip, l_hip, sdr_r, sdr_l
    face_joint_indx = [2, 1, 17, 16]
    # l_hip, r_hip
    r_hip, l_hip = 2, 1
    joints_num = 22



    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]


    if not global_flag:
        '''Uniform Skeleton'''
        #positions = uniform_skeleton(positions, tgt_offsets)

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
    else:
        postions = positions





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
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.001)

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

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions) # [len, 22, 6] , [len-1, 1], [len-1, 3], [len, 4]
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
    local_vel = qrot_np(
        np.repeat(r_rot[:-1, None],  # (len-1, 1, 4)
                global_positions.shape[1], # 22
                axis=1), # final obtain (len-1, 22, 4)
        global_positions[1:] - global_positions[:-1]) # (len-1, 22, 3)
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data # root_data [len, 4] -> 4
    data = np.concatenate([data, ric_data[:-1]], axis=-1) # ric_data [len, 63] -> 21*3

    if rotations is not None:
        assert rotations.shape[-1] == 126 # 21 * 6
        data = np.concatenate([data, rotations[:-1]], axis=-1) # rot_data [len, 126] -> 21*6
    else:
        data = np.concatenate([data, rot_data[:-1]], axis=-1) # rot_data [len, 126] -> 21*6
    #     print(data.shape, local_vel.shape)
    data = np.concatenate([data, local_vel], axis=-1) # local_vel [len-1, 66] -> 22*3
    data = np.concatenate([data, feet_l, feet_r], axis=-1) # feet_l [len-1, 2], feet_r [len-1, 2] -> 4

    #return data, global_positions, positions, l_velocity
    return data




# Input: Positions: [len, 22, 3] Rotations: [len, 22*6]
# Output: data: [len-1, 262]
def generate_interhuman_from_global_position(positions, rotations, feet_thre=0.001, global_flag=True):
    n_joints = 22
    prev_frames = 0
    face_joint_indx = [2,1,17,16]
    fid_l = [7,10]
    fid_r = [8,11]
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    if not global_flag:
        '''Uniform Skeleton'''
        # positions = uniform_skeleton(positions, tgt_offsets)

        positions = positions.reshape(-1, n_joints, 3)

        trans_matrix = torch.Tensor([[1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [0.0, -1.0, 0.0]])
        positions = np.einsum("mn, tjn->tjm", trans_matrix, positions) #change the yz, but why?

        '''Put on Floor'''
        floor_height = positions.min(axis=0).min(axis=0)[1] # the value of y
        positions[:, :, 1] -= floor_height


        '''XZ at origin'''
        root_pos_init = positions[prev_frames]
        root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1]) # Get xz position, and keep y constant
        positions = positions - root_pose_init_xz

        '''All initially face Z+'''
        r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
        across = root_pos_init[r_hip] - root_pos_init[l_hip]
        across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

        # forward (3,), rotate around y-axis
        forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        # forward (3,)
        forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis] # (1,3)

        target = np.array([[0, 0, 1]]) # targeted direction which face z+, screen out
        root_quat_init = qbetween_np(forward_init, target) # (1,4)
        root_quat_init_for_all = np.ones(positions.shape[:-1] + (4,)) * root_quat_init #[276,22,4]


        positions = qrot_np(root_quat_init_for_all, positions)


    else:
        postions = positions


    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([0.12, 0.05])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l_h = positions[:-1,fid_l,1]
        feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r_h = positions[:-1,fid_r,1]
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)


    '''Get Joint Rotation Representation'''
    rot_data = rotations

    '''Get Joint Rotation Invariant Position Represention'''
    joint_positions = positions.reshape(len(positions), -1)
    joint_vels = positions[1:] - positions[:-1]
    joint_vels = joint_vels.reshape(len(joint_vels), -1)

    data = joint_positions[:-1] # [n, 66] -> 22*3 -> joint position
    data = np.concatenate([data, joint_vels], axis=-1) # [n ,66] -> 22*3 -> joint velocity
    data = np.concatenate([data, rot_data[:-1]], axis=-1) # [n, 126] -> 21*6 -> 6D rotation 
    data = np.concatenate([data, feet_l, feet_r], axis=-1) # [n, 2] (feet_l) -> feet contact flag (True/False)

    #from utils_smpl.motion_repre import tranform__interhuman_to_humanml3d, tranform_humanml3d_to_interhuman
    #hu_motion = tranform__interhuman_to_humanml3d(data)
    #in_motion = tranform_humanml3d_to_interhuman(torch.tensor(hu_motion[None])) # (14, 263)
    return data

def generate_interhuman_from_global_position_wo_rotation(positions, feet_thre=0.001, global_flag=True):
    n_joints = 22
    prev_frames = 0
    face_joint_indx = [2,1,17,16]
    fid_l = [7,10]
    fid_r = [8,11]
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    if not global_flag:
        '''Uniform Skeleton'''
        # positions = uniform_skeleton(positions, tgt_offsets)

        positions = positions.reshape(-1, n_joints, 3)

        trans_matrix = torch.Tensor([[1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [0.0, -1.0, 0.0]])
        positions = np.einsum("mn, tjn->tjm", trans_matrix, positions) #change the yz, but why?

        '''Put on Floor'''
        floor_height = positions.min(axis=0).min(axis=0)[1] # the value of y
        positions[:, :, 1] -= floor_height


        '''XZ at origin'''
        root_pos_init = positions[prev_frames]
        root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1]) # Get xz position, and keep y constant
        positions = positions - root_pose_init_xz

        '''All initially face Z+'''
        r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
        across = root_pos_init[r_hip] - root_pos_init[l_hip]
        across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

        # forward (3,), rotate around y-axis
        forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        # forward (3,)
        forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis] # (1,3)

        target = np.array([[0, 0, 1]]) # targeted direction which face z+, screen out
        root_quat_init = qbetween_np(forward_init, target) # (1,4)
        root_quat_init_for_all = np.ones(positions.shape[:-1] + (4,)) * root_quat_init #[276,22,4]


        positions = qrot_np(root_quat_init_for_all, positions)


    else:
        postions = positions


    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([0.12, 0.05])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l_h = positions[:-1,fid_l,1]
        feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r_h = positions[:-1,fid_r,1]
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)


    '''Get Joint Rotation Representation'''
    #ot_data = rotations

    '''Get Joint Rotation Invariant Position Represention'''
    joint_positions = positions.reshape(len(positions), -1)
    joint_vels = positions[1:] - positions[:-1]
    joint_vels = joint_vels.reshape(len(joint_vels), -1)

    data = joint_positions[:-1] # [n, 66] -> 22*3 -> joint position
    data = np.concatenate([data, joint_vels], axis=-1) # [n ,66] -> 22*3 -> joint velocity
    #data = np.concatenate([data, rot_data[:-1]], axis=-1) # [n, 126] -> 21*6 -> 6D rotation 
    data = np.concatenate([data, feet_l, feet_r], axis=-1) # [n, 2] (feet_l) -> feet contact flag (True/False)

    #from utils_smpl.motion_repre import tranform__interhuman_to_humanml3d, tranform_humanml3d_to_interhuman
    #hu_motion = tranform__interhuman_to_humanml3d(data)
    #in_motion = tranform_humanml3d_to_interhuman(torch.tensor(hu_motion[None])) # (14, 263)
    return data



def tranform_humanml3d_to_interhuman(hu_motion, nort = 'numpy'): # [46, 263]
    if nort == 'torch':
        device = hu_motion.device

    hu_motion = np.array(hu_motion.cpu())
    assert hu_motion.shape[-1] ==263
    glo = restore_global_position_from_humanml3d(hu_motion)

    in_motion_wo_rotation =  generate_interhuman_from_global_position_wo_rotation(glo)
    part1 = in_motion_wo_rotation[:,:22*3+22*3]
    part2 = in_motion_wo_rotation[:, 22*3+22*3:]
    assert part2.shape[-1] == 4

    in_motion = np.concatenate((part1, hu_motion[:, 21*3:21*3+21*6], part2), -1)
    in_motion[:,-4:] = hu_motion[:,-4:]

    #inmotion = generate_interhuman_from_global_position(glo)
    assert in_motion.shape[-1] == 262

    if nort == 'numpy':
        return np.array(in_motion)
    elif nort == 'torch':
        return torch.tensor(in_motion).to(device)
    else:
        raise Exception



def tranform_interhuman_to_humanml3d(in_motion, nort = 'numpy'):   
    if nort == 'torch':
        device = hu_motion.device


    assert in_motion.shape[-1] == 262
    ### hu_motion should be [root_data:4, ric_data (local position):21*3, rot_data:21*6, local_vel:22*3, ground contact: 4] called Canonical Representation
    ### in_motion should be [joint position (global position): 22*3, joint velocity (global velocity): 22*3, rot_data:21*6, ground contact: 4] called Non-Canonical Representation
    glo = restore_global_position_from_interhuman(in_motion)
    hu_motion = generate_humanml3d_from_global_position(glo, rotations=None, feet_thre=0.001, global_flag=True)
    hu_motion[:,4+21*3:4+21*3+21*6] = in_motion[:, 21*3+22*3:21*3+22*3+21*6]
    hu_motion[-4:] = in_motion[-4:]

    assert hu_motion.shape[-1] == 263


    if nort == 'numpy':
        return np.array(hu_motion)
    elif nort == 'torch':
        return torch.tensor(hu_motion).to(device)
    else:
        raise Exception

