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

import rospy
from sensor_msgs.msg import Image
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, cameras
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from gs_slam_msgs.msg import visual_merged_msg
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False



def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    print(f"Live Mode = {dataset.live}")
    rospy.init_node("gaussian_splattingg_slam",anonymous=False)
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians) # [Head]
        # pass through the dataset.live to Scene,
        # Write a code to wait the ROS message, and merge that to our database
    # scene.
    # Waiting For message ex, up to 100 frames, including registered pointcloud, Pose and image.
    
    # Initialization Part
    # 
    # T=array([-0.84267078, -0.09141566,  1.53474622]), FovY=0.881621552836618, FovX=1.3986958653070056, 
    # image=<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=979x546 at 0x7FA9F0043050>, image_path='/home/wolftech/lxiang3.lab/Desktop/sdu6/codes/gaussian-splatting/dataset/tandt/truck/images/000145.jpg', image_name='000145', width=1957, height=1091), CameraInfo(uid=1, R=array([[ 0.418596  , -0.40569684,  0.81251921],
    #    [ 0.28220317,  0.90849087,  0.30822995],
    #    [-0.8632142 ,  0.10027167,  0.49477958]]), 
    merged_msg_cache:list[visual_merged_msg] = list()
    if(dataset.live):
        while(len(merged_msg_cache)< 500 and not rospy.is_shutdown()):
            try:
                merged_msg:visual_merged_msg = rospy.wait_for_message(topic="/Visual_Merged",topic_type=visual_merged_msg, timeout=0.2)
                rospy.loginfo(merged_msg.CameraPose.transform.translation.x)
                merged_msg_cache.append(merged_msg)
                rospy.loginfo(f"len(merged_msg_cache)={len(merged_msg_cache)}")
                rospy.Rate(100).sleep()
            except:
                rospy.logwarn("Still Waitinng For Message.")
                rospy.Rate(100).sleep()
    # After that, update the scene object (4 Parts)
    # Update scene info, cameras_extent, make the resolution, and update the gaussians points
    # scene.
    # For scene info, we have to update the (or write a code inside the dataset_Readers) to add images
        # detaily, update the camera intrinsics and extrinsics and use these data to create an object called cam_infos
        # Then, separate them based on probability
    scene.initROSCameras(dataset, merged_msg_cache)


    # /home/wolftech/lxiang3.lab/anaconda3/envs/gaussian_splatting_sdu6/bin/python /home/wolftech/lxiang3.lab/Desktop/sdu6/gaussian-splatting-slam/train_sdu6.py --eval --live





    
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    # ema_loss_for_log = 0.0
    # progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()


        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        # print(viewpoint_stack)
        # print(type(viewpoint_stack))
        # IMPORTANT: Pop One Camera to train.
        viewpoint_cam:cameras.Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        print(f"Iteration = {iteration}, viewpoint_stack size ={len(viewpoint_stack)}, picked camera ID: {viewpoint_cam.uid}, GS-Pts={gaussians._xyz.size()}")

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        

        # Use RAIN GS method
        # Coarse-to-fine
        # c2f = True
        # c2f_every_step = 500
        # c2f_max_lowpass = 300
        # if c2f == True:
        #     # c2f_every_step default = 500
        #     # OptimizationParams -> opt
        #     #  opt.densify_until_iter default value is 15_000 (15000)
        #     if iteration == 1 or (iteration % c2f_every_step == 0 and iteration < opt.densify_until_iter) :
        #         H = viewpoint_cam.image_height
        #         W = viewpoint_cam.image_width
        #         N = gaussians.get_xyz.shape[0]
        #         low_pass = max (H * W / N / (9 * np.pi), 0.3) #0.3 is the minimum value of lowpass filter value
        #         if c2f_max_lowpass > 0:
        #             low_pass = min(low_pass, c2f_max_lowpass)
        #         print(f"[ITER {iteration}] Low pass filter : {low_pass}")
        # else:
        #     low_pass = 0.3

        # gaussian_renderer, in this part, we use the package we just built from cuda files.
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)

        rendered_image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # print(f"Types: rendered_image {type(rendered_image)}, viewspace_point_tensor -> {(viewspace_point_tensor)}, visibility_filter -> {type(visibility_filter)}, radii -> {radii}")

        # print(f"Image size = {rendered_image.shape}")
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(rendered_image, gt_image) #abs((network_output - gt)).mean()
        # lambda_dssim = 0.2 in default.
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(rendered_image, gt_image))
        # print(type(loss)) #class 'torch.Tensor'
        
        # Learn the environment, BUT not add points, just change the position, rgb and opacity here.
        loss.backward()

        iter_end.record()
 
        with torch.no_grad():
            # Progress bar
            # ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            # if iteration % 10 == 0:
            #     progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
            #     progress_bar.update(10)
            # if iteration == opt.iterations:
            #     progress_bar.close()

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

                # print(gaussians._xyz.shape, iteration)
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print(f"Live Mode = {lp.live}")
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    print("Before training.")
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
