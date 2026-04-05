import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3

# import pytorch3d
# import pytorch3d.renderer
import torch
from scipy.spatial.transform import Rotation
import cv2
import os
import PIL.Image as pil_img
from data_loaders.interhuman.utils.rotation_conversions import rotation_6d_to_axis_angle
# from utils.smpl.SMPL import SMPL_layer
# from utils.renderer import SMPLRenderer

import trimesh
import pyrender
from smplx import SMPL


#from utils.renderer_pt3d import Renderer
# import cv2

def vis_2d(image, bbox, pts):

    x1, y1, x2, y2 = bbox

    image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (154, 201, 219), 5)

    for pt in pts:
        x, y = pt
        image = cv2.circle(image, (int(x), int(y)), 3, (255, 136, 132), 3)
    image = pil_img.fromarray(image[:, :, :3].astype(np.uint8))

    return np.asarray(image)

def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(save_path, kinematic_tree, mp_joints, title, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        ax.set_xlim3d([-radius / 4, radius / 4])
        ax.set_ylim3d([0, radius / 2])
        ax.set_zlim3d([0, radius / 2])
        # print(title)
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()

    mp_data = []
    frame_number = min([data.shape[0] for data in mp_joints])
    print(frame_number)

    # colors = ['red', 'blue', 'black', 'red', 'blue',
    #           'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
    #           'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    #
    colors = ['red', 'green', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']

    mp_offset = list(range(-len(mp_joints)//2, len(mp_joints)//2, 1))
    mp_colors = [[colors[i]] * 15 for i in range(len(mp_offset))]

    for i,joints in enumerate(mp_joints): # 2, 210, 22, 3

        # (seq_len, joints_num, 3)
        data = joints.copy().reshape(len(joints), -1, 3)

        MINS = data.min(axis=0).min(axis=0)
        MAXS = data.max(axis=0).max(axis=0)


        #     print(data.shape)

        height_offset = MINS[1]
        data[:, :, 1] -= height_offset
        trajec = data[:, 0, [0, 2]]

        # data[:, :, 0] -= data[0:1, 0:1, 0]
        # data[:, :, 0] += mp_offset[i]
        #
        # data[:, :, 2] -= data[0:1, 0:1, 2]
        mp_data.append({"joints":data,
                        "MINS":MINS,
                        "MAXS":MAXS,
                        "trajec":trajec, })

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 15#7.5
        #         ax =
        plot_xzPlane(-3, 3, 0, -3, 3)
        for pid,data in enumerate(mp_data): # data: 210, 22, 3
            for i, (chain, color) in enumerate(zip(kinematic_tree, mp_colors[pid])):
                #             print(color)
                if i < 5:
                    linewidth = 2.0
                else:
                    linewidth = 1.0
                ax.plot3D(data["joints"][index, chain, 0], data["joints"][index, chain, 1], data["joints"][index, chain, 2], linewidth=linewidth,
                          color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    plt.close()

def render_mesh(vertices, faces, translation, focal_length, height, width, device=None):
    ''' Render the mesh under camera coordinates
    vertices: (N_v, 3), vertices of mesh
    faces: (N_f, 3), faces of mesh
    translation: (3, ), translations of mesh or camera
    focal_length: float, focal length of camera
    height: int, height of image
    width: int, width of image
    device: "cpu"/"cuda:0", device of torch
    :return: the rgba rendered image
    '''
    if device is None:
        device = vertices.device

    bs = vertices.shape[0]

    # add the translation
    vertices = vertices + translation[:, None, :]

    # upside down the mesh
    # rot = Rotation.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix().astype(np.float32)
    rot = Rotation.from_euler('z', 180, degrees=True).as_matrix().astype(np.float32)
    rot = torch.from_numpy(rot).to(device).expand(bs, 3, 3)
    faces = faces.expand(bs, *faces.shape).to(device)

    vertices = torch.matmul(rot, vertices.transpose(1, 2)).transpose(1, 2)

    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(vertices)  # (B, V, 3)
    textures = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb)
    mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures)

    # Initialize a camera.
    cameras = pytorch3d.renderer.PerspectiveCameras(
        focal_length=((2 * focal_length / min(height, width), 2 * focal_length / min(height, width)),),
        device=device,
    )

    # Define the settings for rasterization and shading.
    raster_settings = pytorch3d.renderer.RasterizationSettings(
        image_size=(height, width),   # (H, W)
        # image_size=height,   # (H, W)
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )

    # Define the material
    materials = pytorch3d.renderer.Materials(
        ambient_color=((1, 1, 1),),
        diffuse_color=((1, 1, 1),),
        specular_color=((1, 1, 1),),
        shininess=64,
        device=device
    )

    # Place a directional light in front of the object.
    lights = pytorch3d.renderer.DirectionalLights(device=device, direction=((0, 0, -1),))

    # Create a phong renderer by composing a rasterizer and a shader.
    renderer = pytorch3d.renderer.MeshRenderer(
        rasterizer=pytorch3d.renderer.MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=pytorch3d.renderer.SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            materials=materials
        )
    )

    # Do rendering
    imgs = renderer(mesh)
    return imgs


def render_mesh_single_frame(vertices, faces, translation, focal_length, height, width, device=None):
    ''' Render the mesh under camera coordinates
    vertices: (N_v, 3), vertices of mesh
    faces: (N_f, 3), faces of mesh
    translation: (3, ), translations of mesh or camera
    focal_length: float, focal length of camera
    height: int, height of image
    width: int, width of image
    device: "cpu"/"cuda:0", device of torch
    :return: the rgba rendered image
    '''
    if device is None:
        device = vertices.device

    assert vertices.shape[0] == 1
    vertices = vertices[0]
    translation = translation[0]

    # add the translation
    vertices = vertices + translation

    # upside down the mesh
    # rot = Rotation.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix().astype(np.float32)
    rot = Rotation.from_euler('z', 180, degrees=True).as_matrix().astype(np.float32)
    rot = torch.from_numpy(rot).to(device)
    faces = faces.to(device)

    vertices = torch.matmul(rot, vertices.T).T

    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(vertices)[None]  # (B, V, 3)
    textures = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb)
    mesh = pytorch3d.structures.Meshes(
        verts=[vertices], faces=[faces], textures=textures)

    # Initialize a camera.
    cameras = pytorch3d.renderer.PerspectiveCameras(
        focal_length=((2 * focal_length / min(height, width),
                      2 * focal_length / min(height, width)),),
        device=device,
    )

    # Define the settings for rasterization and shading.
    raster_settings = pytorch3d.renderer.RasterizationSettings(
        # image_size=(height, width),   # (H, W)
        image_size=height,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Define the material
    materials = pytorch3d.renderer.Materials(
        ambient_color=((1, 1, 1),),
        diffuse_color=((1, 1, 1),),
        specular_color=((1, 1, 1),),
        shininess=64,
        device=device
    )

    # Place a directional light in front of the object.
    lights = pytorch3d.renderer.DirectionalLights(
        device=device, direction=((0, 0, -1),))

    # Create a phong renderer by composing a rasterizer and a shader.
    renderer = pytorch3d.renderer.MeshRenderer(
        rasterizer=pytorch3d.renderer.MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=pytorch3d.renderer.SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            materials=materials
        )
    )

    # Do rendering
    imgs = renderer(mesh)
    return imgs[0]


def plot_smpl(pose_output,face):
    #uv_29 = pose_output.pred_uvd_jts.reshape(29, 3)[:, :2]
    transl = pose_output.transl.detach()

    ###生成空白图片
    width, height = 224, 224
    input_image = np.ones((height, width, 3), dtype=np.uint8) * 255

    face = face.astype(np.int32)

    smpl_faces = torch.from_numpy(face) # [13776, 3]

    
    # Visualization
    image = input_image.copy()
    focal = 1000.0
    #bbox_xywh = xyxy2xywh(bbox)
    transl_camsys = transl.clone()
    #transl_camsys = transl_camsys * 256 / bbox_xywh[2]

    #focal = focal / 256 * bbox_xywh[2]

    vertices = pose_output.vertices.detach()

    verts_batch = vertices # [1,6890,3]
    transl_batch = transl

    color_batch = render_mesh(
        vertices=verts_batch, faces=smpl_faces,
        translation=transl_batch,
        focal_length=focal, height=image.shape[0], width=image.shape[1])

    valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
    image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
    image_vis_batch = (image_vis_batch * 255).cpu().numpy()

    color = image_vis_batch[0]
    valid_mask = valid_mask_batch[0].cpu().numpy()
    input_img = image
    alpha = 0.9
    image_vis = alpha * color[:, :, :3] * valid_mask + (
        1 - alpha) * input_img * valid_mask + (1 - valid_mask) * input_img

    image_vis = image_vis.astype(np.uint8)
    image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)

    if True:
        idx += 1
        res_path = os.path.join('results', 'res_images', f'image-{idx:06d}.jpg')
        cv2.imwrite(res_path, image_vis)
    #write_stream.write(image_vis)

    # vis 2d
    # pts = uv_29 * bbox_xywh[2]
    # pts[:, 0] = pts[:, 0] + bbox_xywh[0]
    # pts[:, 1] = pts[:, 1] + bbox_xywh[1]
    # image = input_image.copy()
    #bbox_img = vis_2d(image, tight_bbox, pts)
    #bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR)
    #write2d_stream.write(bbox_img)

    #if opt.save_img:
    # if True:
    #     res_path = os.path.join(
    #         'results', 'res_2d_images', f'image-{idx:06d}.jpg')
    #     cv2.imwrite(res_path, bbox_img)


# def plot_hmr(vertices):
#     ###生成空白图片
#     width, height = 224, 224
#     input_image = np.ones((height, width, 3), dtype=np.uint8) * 255
#     renderer = SMPLRenderer()
#     rend_img = renderer(vertices, img=input_image, color_id=None)
#     plt.imsave(os.path.join('results/', '3D_mesh.png'), rend_img)

# def plot_naive(r6d, smpl):

#     pose_ax_ang = rotation_6d_to_axis_angle(r6d.view(-1,6)).view(210,21,3)

#     beta_0 = torch.tensor(np.array([0.0001]*10))
#     # 使用 unsqueeze 扩展维度
#     beta_0 = torch.unsqueeze(beta_0, 0).to(torch.float)  # 在第一维度前插入一个维度
#     # 将 tensor_b 重复 210 次
#     beta_0 = beta_0.repeat(r6d.shape[0], 1)

    
#     # 创建形状为 [210, 3, 24] 的目标张量，使用 torch.zeros 进行零填充
#     pose_full = torch.zeros(210, 24, 3)
#     # 将原始张量的数据复制到目标张量的前 21 个维度
#     pose_full[:, -21:, :] = pose_ax_ang[0]
    

#     pose_output = smpl(
#         pose_axis_angle=pose_full, betas=beta_0, global_orient=None
#     )

#     os.environ['PYOPENGL_PLATFORM'] = 'egl'
#     # 创建pyrender场景
#     scene = pyrender.Scene()

#     # 创建相机
#     camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
#     camera_pose = np.eye(4)
#     camera_pose[:3, 3] = np.array([0, 0, 4])  # 设置相机位置
#     scene.add(camera, pose=camera_pose)

#     # 创建灯光
#     light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=10.0)
#     light_pose = np.eye(4)
#     light_pose[:3, 3] = np.array([1, 1, 1])  # 设置灯光位置
#     scene.add(light, pose=light_pose)



#     # 将SMPL模型添加到场景中
#     smpl_mesh = trimesh.Trimesh(pose_output.vertices, smpl.faces, process=False)
#     smpl_node = scene.add(pyrender.Mesh.from_trimesh(smpl_mesh), pose=np.eye(4))

#     # 创建渲染器
#     renderer = pyrender.OffscreenRenderer(800, 800)

#     # 渲染场景
#     color, depth = renderer.render(scene)

#     # 显示渲染结果
#     #pyrender.Viewer(scene, use_raymond_lighting=True)
#     from PIL import Image
#     # 可以保存渲染结果
#     image = Image.fromarray(color)

#     # 保存为 PNG 文件
#     image.save("rendered_image.png")
#     #pyrender.io.write_image("rendered_image.png", color)



def plot_naive(r6d, smpl, id, another=None, traj=None):

    def ge_smpl(r6d):
        # if type(r6d)=='numpy.ndarray':   
        r6d = torch.from_numpy(r6d)
        pose_ax_ang = rotation_6d_to_axis_angle(r6d.view(-1,6)).view(-1,21,3)

        beta_0 = torch.tensor(np.array([0.0001]*10))
        # 使用 unsqueeze 扩展维度
        beta_0 = torch.unsqueeze(beta_0, 0).to(torch.float)  # 在第一维度前插入一个维度
        # 将 tensor_b 重复 210 次
        beta_0 = beta_0.repeat(1, 1)

        
        # 创建形状为 [210, 3, 24] 的目标张量，使用 torch.zeros 进行零填充
        pose_full = torch.zeros(1, 24, 3)
        # 将原始张量的数据复制到目标张量的前 21 个维度
        pose_full[:, 1:22, :] = pose_ax_ang[0]
        

        pose_output = smpl(
            pose_axis_angle=pose_full, betas=beta_0, global_orient=None
        )
        return pose_output
    
    if another is not None:
        pose_output_another = ge_smpl(another)
    
    pose_output = ge_smpl(r6d)

    #print(pose_output)
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    # 创建pyrender场景
    scene = pyrender.Scene()

    # 创建相机
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0.5, 0, 2])  # 设置相机位置
    scene.add(camera, pose=camera_pose)

    # 创建灯光
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=10.0)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([1, 1, 1])  # 设置灯光位置
    scene.add(light, pose=light_pose)



    # 将SMPL模型添加到场景中
    #smpl_mesh = trimesh.Trimesh(pose_output[0].vertices[0], smpl.faces, process=False)
    smpl_mesh = trimesh.Trimesh(pose_output[0][0], smpl.faces, process=False)
    # Define relative rotation and translation for person 2
    #relative_rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0])[:3, :3]
    #relative_translation = np.array([1.0, 0.0, 0.0])

    # # Create a 4x4 transformation matrix
    # transformation_matrix = np.eye(4)
    # transformation_matrix[:3, :3] = relative_rotation_matrix
    # #transformation_matrix[:3, 3] = relative_translation

    # # Apply relative transformation to person 2
    # smpl_mesh.apply_transform(transformation_matrix)
    # if traj is not None:
    #     smpl_mesh.apply_translation(traj[0])
    #smpl_mesh.apply_translation([1,0,0]) # [y,z,x]

    smpl_node = scene.add(pyrender.Mesh.from_trimesh(smpl_mesh))
    # 调整第一个人的节点位置和旋转
    pose_person1 = np.eye(4)
    #pose_person1[:3, 3] = [0.0, 0.0, 0.0]  # 调整平移
    #pose_person1[:3, :3] = trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0])[:3, :3]  # 调整旋转
    #scene.set_pose(smpl_node, pose_person1)

    
    if another is not None:
        # #smpl_mesh = trimesh.Trimesh(pose_output[0].vertices[0], smpl.faces, process=False)
        smpl_mesh2 = trimesh.Trimesh(pose_output_another[0][0], smpl.faces, process=False)
        
        # 调整第二个人的节点位置和旋转
        pose_person2 = np.eye(4)
        # Define relative rotation and translation for person 2
        relative_rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(-45), [0, 1, 0])[:3, :3]
        relative_translation = np.array([1.0, 0.0, 0.0])

        # Create a 4x4 transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = relative_rotation_matrix
        transformation_matrix[:3, 3] = relative_translation

        # Apply relative transformation to person 2
        smpl_mesh2.apply_transform(transformation_matrix)
        # if traj is not None:
        #     smpl_mesh2.apply_translation(traj[1])

        #pose_person2[:3, 3] = [1.0, 0.0, 0.0]  # 调整平移
        # 旋转第二个人的 Mesh 180 度
        #pose_person2[:3, :3] = trimesh.transformations.rotation_matrix(np.radians(180), [0, 1, 0])[:3, :3]
        smpl_node2 = scene.add(pyrender.Mesh.from_trimesh(smpl_mesh2)) 
        #scene.set_pose(smpl_node2, pose_person2)


    # 创建渲染器
    renderer = pyrender.OffscreenRenderer(800, 800)

    # 渲染场景
    color, depth = renderer.render(scene)

    # 显示渲染结果
    #pyrender.Viewer(scene, use_raymond_lighting=True)
    from PIL import Image
    # 可以保存渲染结果
    image = Image.fromarray(color)

    # 保存为 PNG 文件
    name = 'results/rendertest'+str(id)+'.png'
    image.save(name) 
    #pyrender.io.write_image("rendered_image.png", color)



#     #for key, item in outputs.items():
#     #    print(key, item.shape)
#     return 

