#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image as PILImage 
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from scene.colmap_loader import Camera, Image # Define(Import) the collections from dataset_readers.py
from gs_slam_msgs.msg import visual_merged_msg
from cv_bridge import CvBridge, CvBridgeError
import cv2
import rospy

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key] # Get the correct camera ID
        intr = cam_intrinsics[extr.camera_id]  # Get the correct Camera ID and Pose 
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = PILImage.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    print(path) # use output path 
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    # print(cam_extrinsics)
    print(cam_extrinsics)
    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    # quit()

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = PILImage.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = PILImage.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

######### ROS Part Implementation #######
def imgmsg_to_pli(msg, filename, path):
    try:
        # Extract image properties
        width = msg.width
        height = msg.height
        encoding = msg.encoding  # e.g., 'rgb8', 'bgr8', 'mono8'
        is_bigendian = msg.is_bigendian
        step = msg.step

        # Convert raw data to NumPy array
        data = np.frombuffer(msg.data, dtype=np.uint8)

        # Handle different encodings
        if encoding == 'rgb8':
            image = data.reshape((height, width, 3))
        elif encoding == 'bgr8':
            image = data.reshape((height, width, 3))[:, :, ::-1]  # Convert BGR to RGB
        elif encoding == 'mono8':
            image = data.reshape((height, width))
        else:
            rospy.logerr(f"Unsupported encoding: {encoding}")
            return

        # Convert NumPy array to PIL Image
        pil_image = PILImage.fromarray(image)
        # Save the image as JPEG
        pil_image.save(path+filename)
        rospy.loginfo(f"Image saved as {path+filename}")

    except Exception as e:
        rospy.logerr(f"Failed to process image: {e}")
        return PILImage

def initCameraIntrinsics(rosmsg_list):
    cameras = {}
    for msg in rosmsg_list:
        camera_info = msg.CameraInfo
        K = camera_info.K
        camera_id = 1
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]
        cameras[camera_id] = Camera(id=camera_id, model="PINHOLE",
                                    width=camera_info.width, height=camera_info.height,
                                    params= np.array([fx, fy, cx, cy]))
    print(cameras)
    return cameras

def initCameraExtrinsics(rosmsg_list):
    images = {}
    # if (not rospy.has_param('images_idx')):
    rospy.set_param('images_idx',0)
    for msg in rosmsg_list:
        pose = msg.CameraPose
        image_index = rospy.get_param('images_idx',default=0)
        q = pose.transform.rotation
        t = pose.transform.translation
        qvec = np.array([q.w, q.x, q.y, q.z])
        tvec = np.array([t.x, t.y, t.z])
        camera_id = 1
        image_name = str(image_index) + ".jpg"
        xys = -1
        point3D_ids = -1
        images[image_index] = Image(
            image_index,qvec,tvec,camera_id,image_name,xys,point3D_ids)
        rospy.set_param('images_idx',image_index+1)
    print(images)
    return images

def initSceneInfo(cam_extrinsics, cam_intrinsics, rosmsg_list, path, model_path, eval, llffhold=8):
    bridge = CvBridge()
    cam_infos = []
    output_folder = model_path + "/sparse/0/"
    images_folder = model_path + "/ros_images/"
    os.makedirs(images_folder ,exist_ok = True)
    os.makedirs(output_folder ,exist_ok = True)
    print(f"output_folder = {output_folder}")

    for idx, key in enumerate(cam_extrinsics):  
        img_msg = rosmsg_list[idx].Image
        extr = cam_extrinsics[key] # Get the correct camera ID
        intr = cam_intrinsics[extr.camera_id]  # Get the correct Camera ID and Pose 
        height = intr.height
        width = intr.width
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        
        # Save the image first
        
        # cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
        # cv2.imwrite('output_image.jpg', cv_image)
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        imgmsg_to_pli(img_msg, extr.name, images_folder)

        image = PILImage.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    ply_path = output_folder + "points3D.ply"

    xyz = np.array([])
    shs = np.array([])
    print("[Info] RAIN-GS Method Selected.")
    num_pts = 100
    cam_pos = []
    for k in cam_extrinsics.keys():
        cam_pos.append(cam_extrinsics[k].tvec)
    cam_pos = np.array(cam_pos)
    min_cam_pos = np.min(cam_pos)
    max_cam_pos = np.max(cam_pos)
    mean_cam_pos = (min_cam_pos + max_cam_pos) / 2.0
    cube_mean = (max_cam_pos - min_cam_pos) * 1.5
    # Method 1 to intialization
    # This initialize the pointcloud using random points according to nerf normalization.
        # print("[Info] Select Random number initialization.")
        # xyz = np.random.random((num_pts, 3)) * nerf_normalization["radius"] * 3 - nerf_normalization["radius"] * 1.5
        # xyz = xyz + nerf_normalization["translate"]
        # print(f"Generating random point cloud ({num_pts})...")

    # Method 2 to intialization
    # If we get the training data from reconstructed point cloud, and use more more sparse pointcloud, starts here.
    xyz = np.random.random((num_pts, 3)) * (max_cam_pos - min_cam_pos) * 3 - (cube_mean - mean_cam_pos)
    print(f"[Info] Generating OUR point cloud ({num_pts})...")
    shs = np.random.random((num_pts, 3)) # Color
    pcd = BasicPointCloud(points=xyz, colors=shs, normals=np.zeros((num_pts, 3)))
    print(f"[INFO] 1st/2nd xyz: {xyz[0]},{xyz[1]}.")
    print(f"[INFO] 1st/2nd color: {shs[0]},{shs[1]}.")

    # Store the random color and points here.
    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    print("[INFO] Save the random PLY pointcloud. The following reading pointcloud will be random!")
    
    try:
        print(f"[DEBUG] point_cloud = {ply_path}")
        pcd = fetchPly(ply_path)
    except:
        print("[DEBUG] point_cloud = None")
        pcd = None

    nerf_normalization = getNerfppNorm(train_cam_infos)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info
# def read
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
    # "Live" : readROSSceneInfo
}