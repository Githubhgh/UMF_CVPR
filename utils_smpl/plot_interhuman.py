import numpy as np
import torch
from utils_smpl.SMPL import SMPL_layer
#from utils.preprocess import load_motion
from data_loaders.interhuman.utils.utils import *
# #from utils.plot_script import *
# from utils.preprocess import *
import random

from utils_smpl.rotation_conv import *
from data_loaders.interhuman.utils.preprocess import load_motion





import os
import pyrender
import trimesh


# Load SMPL model
# Set SMPL_BODY_MODEL_DIR env var to point to your SMPL body model directory.
# Expected files: J_regressor_extra.npy, SMPL_NEUTRAL.pkl
from smplx import SMPLLayer
SMPL_BODY_MODEL_DIR = os.environ.get("SMPL_BODY_MODEL_DIR", "./body_models/smpl")
h36m_jregressor = np.load(os.path.join(SMPL_BODY_MODEL_DIR, 'J_regressor_extra.npy'))
smpl = SMPL_layer(
    os.path.join(SMPL_BODY_MODEL_DIR, 'SMPL_NEUTRAL.pkl'),
    h36m_jregressor=h36m_jregressor,
    dtype=torch.float32
)


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"目录 '{directory}' 已创建。")
    else:
        print(f"目录 '{directory}' 已经存在。")

def batch_generate(file_path1, file_path2, return_split = True):
    max_cond_length = 1
    min_cond_length = 1
    max_gt_length = 100
    min_gt_length = 15

    max_length = max_cond_length + max_gt_length -1
    min_length = min_cond_length + min_gt_length -1

    motion1, motion1_swap = load_motion(file_path1, min_length, swap=False)
    motion2, motion2_swap = load_motion(file_path2, min_length, swap=False)

    full_motion1 = motion1
    full_motion2 = motion2

    length = full_motion1.shape[0]
    if length > max_length:
        idx = random.choice(list(range(0, length - max_gt_length, 1)))
        gt_length = max_gt_length
        motion1 = full_motion1[idx:idx + gt_length]
        motion2 = full_motion2[idx:idx + gt_length]

    else:
        idx = 0
        gt_length = min(length - idx, max_gt_length )
        motion1 = full_motion1[idx:idx + gt_length]
        motion2 = full_motion2[idx:idx + gt_length]

    # if np.random.rand() > 0.5:
    #     motion1, motion2 = motion2, motion1
    # motion1: [bs, 262] 
    # root_quat_init1: [1,4] 
    # root_pos_init1: [1,3]
    motion1, root_quat_init1, root_pos_init1 = process_motion_np(motion1, 0.001, 0, n_joints=22) 
    motion2, root_quat_init2, root_pos_init2 = process_motion_np(motion2, 0.001, 0, n_joints=22)
    r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
    angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])

    xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:, [0, 2]]
    relative = np.concatenate([angle, xz], axis=-1)[0]
    motion2 = rigid_transform(relative, motion2)


    gt_motion1 = motion1
    gt_motion2 = motion2

    gt_length = len(gt_motion1)
    if gt_length < max_gt_length:
        padding_len = max_gt_length - gt_length
        D = gt_motion1.shape[1]
        padding_zeros = np.zeros((padding_len, D))
        gt_motion1 = np.concatenate((gt_motion1, padding_zeros), axis=0)
        gt_motion2 = np.concatenate((gt_motion2, padding_zeros), axis=0)


    assert len(gt_motion1) == max_gt_length
    assert len(gt_motion2) == max_gt_length

    # if np.random.rand() > 0.5:
    #     gt_motion1, gt_motion2 = gt_motion2, gt_motion1


    if return_split:
        return gt_motion1[...,:22*3].astype(np.float32), gt_motion1[:, 22*6:22*6 + 21 * 6].astype(np.float32) , gt_motion2[...,:22*3].astype(np.float32) , gt_motion2[:, 22*6:22*6 + 21 * 6].astype(np.float32) 
    return  gt_motion1, gt_motion2, gt_length 

def get_orientation(skeleton):
    r_hip = 2
    l_hip = 1
    # skeleton: [bn, 300, 2, 22, 3]  across: [bn, 300, 2, 3]
    across = skeleton[..., r_hip, :] - skeleton[..., l_hip, :] 
    across = across / across.norm(dim=-1, keepdim=True)


    # y_axis: [bn, 300, 2, 3]
    y_axis = torch.zeros_like(across) 
    y_axis[..., 1] = 1


    # forward: [bn, 300, 2, 3]
    forward = torch.cross(y_axis, across, axis=-1)
    forward = forward / forward.norm(dim=-1, keepdim=True)
    from utils_smpl.quaternion import qbetween
    from utils_smpl.rotation_conv import quaternion_to_axis_angle
    q1 = qbetween(forward[..., 0, :], torch.tensor([0,0,1], dtype=torch.float32))
    q2 = qbetween(forward[..., 1, :], torch.tensor([0,0,1], dtype=torch.float32))
    a1 = quaternion_to_axis_angle(q1)
    a2 = quaternion_to_axis_angle(q2)

    return -a1.unsqueeze(0), -a2.unsqueeze(0)

def read_id(id):
    # Define file paths
    person1_dir = 'data/InterHuman/motions_processed/person1/' + str(id) + '.npy'
    person2_dir = 'data/InterHuman/motions_processed/person2/' + str(id) + '.npy'
    text_path = person1_dir.replace("motions_processed", "annots").replace("person1", "").replace("npy", "txt")


    # Batch generate key points and relative 6D coordinates
    kp, r6d_1_21, kp_2, r6d_2_21 = batch_generate(person1_dir, person2_dir)
    r6d_1_21 = torch.tensor(r6d_1_21)
    r6d_2_21 = torch.tensor(r6d_2_21)



    # Read texts from annotation file
    texts = [item.replace("\n", "") for item in open(text_path, "r").readlines()]

    # Extract traj2_listectories from key points
    traj2_list = np.zeros([kp.shape[0], 3])
    traj2_list_2 = np.zeros([kp.shape[0], 3])
    traj2_list[:, [0, 2]] = kp.reshape(-1, 22, 3)[:, 0, [0, 2]]
    traj2_list_2[:, [0, 2]] = kp_2.reshape(-1, 22, 3)[:, 0, [0, 2]]



    # Prepare keypoints for skeleton
    kp_copy = torch.tensor(kp).unsqueeze(1)
    kp_2_copy = torch.tensor(kp_2).unsqueeze(1)
    skeleton = torch.concat([kp_copy, kp_2_copy], 1).view(-1, 2, 22, 3)
    #if data_require=='r6d':
        #return r6d, r6d_2, traj2_list, traj2_list_2, skeleton, texts

    glo_1_list = []
    glo_2_list = []

    for i in range(len(skeleton)):
        g_1, g_2 = get_orientation(skeleton[i])
        glo_1_list.append(g_1)
        glo_2_list.append(g_2)


    
    # 将列表转换为Numpy数组
    glo_1 = torch.tensor(np.concatenate(glo_1_list))
    glo_2 = torch.tensor(np.concatenate(glo_2_list))
    glo_1_mat = axis_angle_to_matrix(glo_1).unsqueeze(1)
    glo_2_mat = axis_angle_to_matrix(glo_2).unsqueeze(1) #torch.Size([100, 1, 3, 3])

    #print(glo_1.shape, axis_angle_to_matrix(glo_1).shape) # [100,3], [100,3,3]

    glo_1_r6d = matrix_to_rotation_6d(axis_angle_to_matrix(glo_1)) 
    glo_2_r6d = matrix_to_rotation_6d(axis_angle_to_matrix(glo_1)) 


    #print(glo_1_r6d.shape)


    rots1_wo_glo = rotation_6d_to_matrix(r6d_1_21.reshape(-1, 21, 6)) # [100, 21, 3, 3]
    rots2_wo_glo = rotation_6d_to_matrix(r6d_2_21.reshape(-1, 21, 6))

    rots1 = torch.concat([glo_1_mat, rots1_wo_glo], 1 ) #torch.Size([100, 22, 3, 3])
    rots2 = torch.concat([glo_2_mat, rots2_wo_glo], 1 )

    trans1 = torch.tensor(traj2_list)
    trans2 = torch.tensor(traj2_list_2)

    #r6d_1 = torch.concat([glo_1_r6d, r6d_1_21], 1)
    #r6d_2 = torch.concat([glo_2_r6d, r6d_2_21], 1)
    #return  r6d, r6d_2, rots1, rots2, trans1, trans2, skeleton, texts, glo_0, glo_1
        # Create a dictionary for the outputs
    output_dict = {
        #'r6d1': torch.nan_to_num(r6d_1),
        #'r6d2': torch.nan_to_num(r6d_2),
        'rots1': torch.nan_to_num(rots1),
        'rots2': torch.nan_to_num(rots2),
        'trans1': trans1,
        'trans2': trans2,
        'skeleton': skeleton,
        'texts': texts,
        'orien_1': glo_1,
        'orien_2': glo_2
    }

    return output_dict

def read_gene_single(motion1, motion2, texts):

    motion_1 = motion1.clone().detach()
    motion_2 = motion2.clone().detach()
    # Batch generate key points and relative 6D coordinates

    r6d_1_21, r6d_2_21 = motion_1[..., 22*6:22*6 + 21 * 6], motion_2[..., 22*6:22*6 + 21 * 6]
    kp, kp_2 = motion_1[..., :22*3], motion_2[..., :22*3]




    # Extract traj2_listectories from key points
    traj2_list = np.zeros([kp.shape[0], 3])
    traj2_list_2 = np.zeros([kp.shape[0], 3])
    traj2_list[:, [0, 2]] = kp.reshape(-1, 22, 3)[:, 0, [0, 2]]
    traj2_list_2[:, [0, 2]] = kp_2.reshape(-1, 22, 3)[:, 0, [0, 2]]



    # Prepare keypoints for skeleton
    kp_copy = torch.tensor(kp).unsqueeze(1)
    kp_2_copy = torch.tensor(kp_2).unsqueeze(1)
    skeleton = torch.concat([kp_copy, kp_2_copy], 1).view(-1, 2, 22, 3)
    #if data_require=='r6d':
        #return r6d, r6d_2, traj2_list, traj2_list_2, skeleton, texts

    glo_1_list = []
    glo_2_list = []

    for i in range(len(skeleton)):
        g_1, g_2 = get_orientation(skeleton[i])
        glo_1_list.append(g_1)
        glo_2_list.append(g_2)


    
    # 将列表转换为Numpy数组
    glo_1 = torch.tensor(np.concatenate(glo_1_list))
    glo_2 = torch.tensor(np.concatenate(glo_2_list))
    glo_1_mat = axis_angle_to_matrix(glo_1).unsqueeze(1)
    glo_2_mat = axis_angle_to_matrix(glo_2).unsqueeze(1) #torch.Size([100, 1, 3, 3])

    #print(glo_1.shape, axis_angle_to_matrix(glo_1).shape) # [100,3], [100,3,3]

    glo_1_r6d = matrix_to_rotation_6d(axis_angle_to_matrix(glo_1)) 
    glo_2_r6d = matrix_to_rotation_6d(axis_angle_to_matrix(glo_1)) 


    #print(glo_1_r6d.shape)


    rots1_wo_glo = rotation_6d_to_matrix(r6d_1_21.reshape(-1, 21, 6)) # [100, 21, 3, 3]
    rots2_wo_glo = rotation_6d_to_matrix(r6d_2_21.reshape(-1, 21, 6))

    rots1 = torch.concat([glo_1_mat, rots1_wo_glo], 1 ) #torch.Size([100, 22, 3, 3])
    rots2 = torch.concat([glo_2_mat, rots2_wo_glo], 1 )

    trans1 = torch.tensor(traj2_list)
    trans2 = torch.tensor(traj2_list_2)

    #r6d_1 = torch.concat([glo_1_r6d, r6d_1_21], 1)
    #r6d_2 = torch.concat([glo_2_r6d, r6d_2_21], 1)
    #return  r6d, r6d_2, rots1, rots2, trans1, trans2, skeleton, texts, glo_0, glo_1
        # Create a dictionary for the outputs
    output_dict = {
        #'r6d1': torch.nan_to_num(r6d_1),
        #'r6d2': torch.nan_to_num(r6d_2),
        'rots1': torch.nan_to_num(rots1),
        'rots2': torch.nan_to_num(rots2),
        'trans1': trans1,
        'trans2': trans2,
        'skeleton': skeleton,
        'texts': texts,
        'orien_1': glo_1,
        'orien_2': glo_2
    }

    return output_dict



import imageio
def plot_naive_rot(r6d, smpl, id, another=None, traj=None, tt=None, return_mat = True, mul_view = False, save_dir = './tmp_results/'):
    mul_view_file = ''

    def ge_smpl(r6d, traj_c):

        if not isinstance(r6d, torch.Tensor):
            pose_ax_ang = matrix_to_axis_angle(torch.tensor(r6d).view(-1,3,3)).view(-1,22,3)
        else:
            pose_ax_ang = matrix_to_axis_angle(r6d.view(-1,3,3)).view(-1,22,3)
        beta_0 = torch.tensor(np.array([0.0001]*10))
        # 使用 unsqueeze 扩展维度
        beta_0 = torch.unsqueeze(beta_0, 0).to(torch.float)  # 在第一维度前插入一个维度
        # 将 tensor_b 重复 220 次
        beta_0 = beta_0.repeat(1, 1)

        
        # 创建形状为 [220, 3, 24] 的目标张量，使用 torch.zeros 进行零填充
        pose_full = torch.zeros(1, 24, 3)
        # 将原始张量的数据复制到目标张量的前 22 个维度
        pose_full[:, :22, :] = pose_ax_ang[0]

        pose_output = smpl(
            pose_axis_angle=pose_full, betas=beta_0, global_orient=None, transl = traj_c
        )
        if return_mat:
            pose_full_24 = pose_full
            return pose_output, pose_full_24
        return pose_output
    

    if traj is not None:
        traj[0] = torch.tensor(traj[0])
        traj[1] = torch.tensor(traj[1])

    if another is not None:
        pose_output_another, pose_mat_24_another = ge_smpl(another, traj[1])
    
    pose_output, pose_mat_24 = ge_smpl(r6d, traj[0])

    #print(pose_output)
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    # 创建pyrender场景
    scene = pyrender.Scene()

    # 创建相机
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, 6])  # 设置相机位置
    #


    if mul_view:
        mul_view_file = 'mulview_'
        # 创建围绕y轴旋转90度的旋转矩阵
        angle = np.pi / 2  # 90度转换为弧度
        rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle), 0],
            [0, 1, 0, 0],
            [-np.sin(angle), 0, np.cos(angle), 0],
            [0, 0, 0, 1]
        ])

        # 更新相机姿态矩阵，将旋转矩阵乘以原始姿态矩阵
        camera_pose = rotation_matrix @ camera_pose

        # 添加相机到场景中（假设已经有一个名为scene的场景对象）
        scene.add(camera, pose=camera_pose)
    else:
        scene.add(camera, pose=camera_pose)

    # 创建灯光
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=10.0)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([1, 1, 1])  # 设置灯光位置
    scene.add(light, pose=light_pose)



    
    # 将SMPL模型添加到场景中
    #smpl_mesh = trimesh.Trimesh(pose_output[0].vertices[0], smpl.faces, process=False)
    smpl_mesh = trimesh.Trimesh(pose_output[0][0], smpl.faces, process=False)
    #smpl_mesh.visual.face_colors = [200, 200, 250, 100]
    #if traj is not None:
        #smpl_mesh.apply_translation(0.01*traj[0])

    smpl_node = scene.add(pyrender.Mesh.from_trimesh(smpl_mesh))
    
    if another is not None:
        smpl_mesh2 = trimesh.Trimesh(pose_output_another[0][0], smpl.faces, process=False)
        #if traj is not None:
            #smpl_mesh2.apply_translation(0.01*traj[1])
        smpl_node2 = scene.add(pyrender.Mesh.from_trimesh(smpl_mesh2)) 



    # 创建渲染器
    renderer = pyrender.OffscreenRenderer(800,800)
    color = None
    # 渲染场景
    color, _ = renderer.render(scene)

    #pyrender.Viewer(scene, use_raymond_lighting=True)

    from PIL import Image
    # 可以保存渲染结果
    image = Image.fromarray(color)

    # 保存为 PNG 文件
    if tt is not None:
        results_dir = mul_view_file+'results_smpl/'+str(tt)+'/'
    else:
        results_dir = mul_view_file+'results_smpl/Interhuman/'

    results_dir = save_dir + results_dir

    name = results_dir+str(id)+'.png'

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    image.save(name)

    if return_mat:
        if another is None:
            return pose_mat_24
        else:
            return pose_mat_24, pose_mat_24_another



def generate_gif_interhuman(rots, rots_2, trans1, trans2, texts, id_gif, mul_view = False, save_dir = 'tmp'):

    def non_zero(rots2):
        rots2 = torch.tensor(rots2)
        index = -1  # 初始化index
        for i in range(len(rots2)):
            x = rots2[i]
            # 检查x是否全为零
            if torch.all(x == 0):
                index = i  # 更新index
                return index  # 找到第一个全零矩阵，直接返回索引

        # 如果所有矩阵都非零，返回Tensor的长度
        return len(rots2)
    index1 = non_zero(rots)
    index2 = non_zero(rots_2)

    index = min(index1, index2)

    rots, rots_2, trans1, trans2 = rots[:index], rots_2[:index], trans1[:index], trans2[:index]


    for i in range(len(trans2)):    
        #plot_naive(rots_com.reshape(-1,21,6), smpl, 'multiperson'+str(i))
        plot_naive_rot(rots.reshape(-1,22,3,3)[i], smpl, str(i), 
                rots_2.reshape(-1,22,3,3)[i],
                    [trans1[i], trans2[i]], texts, save_dir = save_dir
                )
        
        if mul_view:
                plot_naive_rot(rots.reshape(-1,22,3,3)[i], smpl, str(i), 
                rots_2.reshape(-1,22,3,3)[i],
                    [trans1[i], trans2[i]], texts, mul_view=mul_view, save_dir = save_dir
                )

    #create_directory_if_not_exists(f'results_smpl/'+str(texts))
    #print('results_smpl/'+str(texts))
    image_names = ['tmp' + '/results_smpl/'+str(texts)+f'/{i}.png' for i in range(len(rots))]
    print(texts,'\n',texts[1])
    gif_name = save_dir+'/'+str(id_gif)+texts+'.gif'
    with imageio.get_writer(gif_name, mode='I') as writer:
        for image_name in image_names:
            image = imageio.imread(image_name)
            writer.append_data(image)

    if mul_view:
            image_names = ['tmp'+ '/mulview_results_smpl/'+str(texts)+f'/{i}.png' for i in range(len(rots))]
            print(texts[0],'\n',texts[1])
            gif_name = save_dir+'/'+str(id_gif)+texts+'_mulview.gif'
            with imageio.get_writer(gif_name, mode='I') as writer:
                for image_name in image_names:
                    image = imageio.imread(image_name)
                    writer.append_data(image)



def generate_gif_interhuman_fail_case(rots, rots_2, trans1, trans2, texts, id_gif, mul_view = False, save_dir = './tmp_results/'):

    def non_zero(rots2):
        rots2 = torch.tensor(rots2)
        index = -1  # 初始化index
        for i in range(len(rots2)):
            x = rots2[i]
            # 检查x是否全为零
            if torch.all(x == 0):
                index = i  # 更新index
                return index  # 找到第一个全零矩阵，直接返回索引

        # 如果所有矩阵都非零，返回Tensor的长度
        return len(rots2)
    index1 = non_zero(rots)
    index2 = non_zero(rots_2)

    index = min(index1, index2)

    rots, rots_2, trans1, trans2 = rots[:index], rots_2[:index], trans1[:index], trans2[:index]


    for i in range(len(trans2)):    
        #plot_naive(rots_com.reshape(-1,21,6), smpl, 'multiperson'+str(i))
        plot_naive_rot(rots.reshape(-1,22,3,3)[i], smpl, str(i), 
                rots_2.reshape(-1,22,3,3)[i],
                    [trans1[i], trans2[i]], texts, save_dir = 'tmp/'
                )
        
        if mul_view:
                plot_naive_rot(rots.reshape(-1,22,3,3)[i], smpl, str(i), 
                rots_2.reshape(-1,22,3,3)[i],
                    [trans1[i], trans2[i]], texts, mul_view=mul_view, save_dir = 'tmp/'
                )

    
    #print('results_smpl/'+str(texts))
    image_names = ['tmp/' + 'results_smpl/'+str(texts)+f'/{i}.png' for i in range(len(rots))]
    print(texts[0],'\n',texts[1])
    create_directory_if_not_exists(save_dir)
    gif_name = save_dir+'/'+str(id_gif)+texts+'.gif'
    with imageio.get_writer(gif_name, mode='I') as writer:
        for image_name in image_names:
            image = imageio.imread(image_name)
            writer.append_data(image)

    if mul_view:
            image_names = ['tmp/' + 'mulview_results_smpl/'+str(texts)+f'/{i}.png' for i in range(len(rots))]
            print(texts[0],'\n',texts[1])
            gif_name = save_dir+'/'+str(id_gif)+texts+'_mulview.gif'
            with imageio.get_writer(gif_name, mode='I') as writer:
                for image_name in image_names:
                    image = imageio.imread(image_name)
                    writer.append_data(image)





def plot_motion_id(motion_id):
    try:
        data_ih_1 = read_id(motion_id) 
    except Exception as e:
        print('Error happens in ', motion_id, e)

    #print(data_ih_1['texts'])
    generate_gif_interhuman(data_ih_1['rots1'], data_ih_1['rots2'], 
                        data_ih_1['trans1'], data_ih_1['trans2'], 
                        str(motion_id)+'_', #data_ih_1['texts'][1][0:100], 
                        motion_id, mul_view=True)


#def plot_gene_motion(motion1, motion2, texts, motion_id): # [bn, 300, 262]
def plot_gene_motion(batch, fail_case, name): # [bn, 300, 262]   
    
    for i, id_score in enumerate(fail_case):
        if i>=1:
            continue
        motion_id = id_score[0]
        i = batch[0].index(motion_id)

        motion1, motion2, texts = torch.tensor(torch.tensor(batch[2][i]).double(), dtype = torch.float32), torch.tensor(torch.tensor(batch[3][i]).double(), dtype = torch.float32), batch[1][i]

        
        #try:
        data_ih_1 = read_gene_single(motion1, motion2, texts) 
        #except Exception as e:
            #print('Error happens in ', motion_id, e)

        #print(data_ih_1['texts'])
        generate_gif_interhuman(data_ih_1['rots1'], data_ih_1['rots2'], 
                            data_ih_1['trans1'], data_ih_1['trans2'], 
                            str(motion_id)+'_', #data_ih_1['texts'][1][0:100], 
                            motion_id, mul_view=True)

def plot_gene_motion_v2(batch, save_dir): # [bn, 300, 262]   
    
    for fail_sample in enumerate(batch):
        fail_sample = fail_sample[1]
        motion_id = fail_sample[0]
        motion1, motion2, texts = torch.tensor(torch.tensor(fail_sample[2]).double(), dtype = torch.float32), torch.tensor(torch.tensor(fail_sample[3]).double(), dtype = torch.float32), fail_sample[1]
        data_ih_1 = read_gene_single(motion1, motion2, texts) 
        #print(data_ih_1['texts'])
        generate_gif_interhuman_fail_case(data_ih_1['rots1'], data_ih_1['rots2'], 
                            data_ih_1['trans1'], data_ih_1['trans2'], 
                            str(motion_id)+'_', #data_ih_1['texts'][1][0:100], 
                            motion_id, mul_view=True, save_dir = save_dir)




