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
import random
import json
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, SceneInfo, initCameraIntrinsics, initCameraExtrinsics, initSceneInfo
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import numpy as np
from gs_slam_msgs.msg import visual_merged_msg

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.live = args.live

        self.loaded_iter = None
        self.gaussians = gaussians
        self.eval = args.eval
        if load_iteration: 
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if(not self.live):
            if os.path.exists(os.path.join(args.source_path, "sparse")):
                self.scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
            elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
                print("Found transforms_train.json file, assuming Blender data set!")
                self.scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
            else:
                assert False, "Could not recognize scene type!"
        else:
            self.scene_info = SceneInfo(point_cloud=BasicPointCloud(np.array([]),np.array([]),np.array([])),
                                   train_cameras=[],
                                   test_cameras=[],
                                   nerf_normalization={},
                                   ply_path='')
            return
        # print(self.scene_info)
        self.write_scene_ply_camera()
        self.shuffle_Camera(shuffle=shuffle)
        self.cameras_extent = self.scene_info.nerf_normalization["radius"]
        self.resoultion_scale(resolution_scales=resolution_scales, args=args)
        self.load_init_pointcloud()




    def load_init_pointcloud(self):
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(self.scene_info.point_cloud, self.cameras_extent)

    def resoultion_scale(self, resolution_scales, args):
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.test_cameras, resolution_scale, args)

    def shuffle_Camera(self, shuffle):
        if shuffle:
            random.shuffle(self.scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(self.scene_info.test_cameras)  # Multi-res consistent random shuffling
    
    def write_scene_ply_camera(self):
        # Write the scene_info to a ply file, and dump all cameras into a json file.
        if not self.loaded_iter:
            with open(self.scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if self.scene_info.test_cameras:
                camlist.extend(self.scene_info.test_cameras)
            if self.scene_info.train_cameras:
                camlist.extend(self.scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]
    
    def getScene_Train_Cameras(self):
        return SceneInfo.train_cameras

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def initROSCameras(self, dataset, visual_merged_list, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        print(dataset.model_path)
        camera_intrinsic = initCameraIntrinsics(visual_merged_list)
        camera_extrinsic = initCameraExtrinsics(visual_merged_list) #Only deal with tvec and rvec. no xys and point3did

        # In this part, we artifically set the ply file, but actually, this file does not exist. Consider create a random instead.
        self.scene_info = initSceneInfo(camera_extrinsic,camera_intrinsic,visual_merged_list,dataset.source_path, dataset.model_path, self.eval)
        # self.write_scene_ply_camera()
        self.shuffle_Camera(shuffle=shuffle)
        self.cameras_extent = self.scene_info.nerf_normalization["radius"]
        self.resoultion_scale(resolution_scales=resolution_scales, args=dataset)
        self.load_init_pointcloud()


        # print(cam_infos)
        
        

        