#!/usr/bin/env python

import ctypes
import rospy, rosbag
from gs_slam_msgs.msg import visual_merged_msg  # Replace with your message type
from geometry_msgs.msg import TransformStamped, Quaternion
from sensor_msgs.msg import Image, PointCloud2, PointField
from PIL import Image as pilImage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D scatter plots
import os, argparse, struct
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import math
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
from scipy.spatial.transform import Rotation as R

pc2_pub = rospy.Publisher("/pointcloud", PointCloud2,queue_size=1)

MOVING_FORWARD = 0x00
MOVING_BACKWARD = 0x01
class visual_message_extractor():
    def __init__(self):
        self.vm_topic = '/Visual_Merged'
def process_and_save_image(msg, timestamp, output_dir, img_name):
    try:
        # msg:Image = msg.Image 
        # Get image properties
        width = msg.width
        height = msg.height
        encoding = msg.encoding
        data = msg.data

        # Map encoding to PIL mode and raw mode
        if encoding == 'rgb8':
            mode = 'RGB'
            raw_mode = 'RGB'
        elif encoding == 'rgba8':
            mode = 'RGBA'
            raw_mode = 'RGBA'
        elif encoding == 'mono8':
            mode = 'L'
            raw_mode = 'L'
        elif encoding == 'mono16':
            mode = 'I;16'
            raw_mode = 'I;16'
        elif encoding == 'bgr8':
            mode = 'RGB'
            raw_mode = 'BGR'
        elif encoding == 'bgra8':
            mode = 'RGBA'
            raw_mode = 'BGRA'
        else:
            print(f"Unsupported encoding: {encoding}")
            return

        # Create PIL Image from data
        image = pilImage.frombytes(mode, (width, height), data, 'raw', raw_mode, msg.step)

        # Save image
        img_path = os.path.join(output_dir, img_name)
        image.save(img_path)
        # print(f"Saved image {img_name}")

    except Exception as e:
        print(f"Could not process image at time {timestamp}: {e}")
def extract_camera_info(msg, camera_info_path):
    try:
        CAMERA_ID = 1  # Assuming a single camera with ID 1
        MODEL = 'PINHOLE'
        width = msg.width
        height = msg.height
        fx = msg.K[0]
        fy = msg.K[4]
        cx = msg.K[2]
        cy = msg.K[5]

        with open(camera_info_path, 'w') as f:
            f.write(f"{CAMERA_ID} {MODEL} {width} {height} {fx} {fy} {cx} {cy}\n")
        print(f"Saved camera info to {camera_info_path}")

    except Exception as e:
        print(f"Could not extract camera info: {e}")

def extract_camera_pose_image(msg_list, poses_path, images_output_dir, max_workers):
    CAMERA_ID = 1  # Assuming a single camera ID
    scale_index = 0
    with open(poses_path, 'w') as poses_file:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, (pose_msg, image_msg) in enumerate(msg_list, start=1):
                if(idx%1 == 0):
                    # print("img", type(image_msg))
                    # print("pose", type(pose_msg))
                    # img_msg = image_msgs[img_time]
                    img_name = f"frame_{scale_index:06d}.jpg"

                    # Submit image processing to the thread pool
                    future = executor.submit(process_and_save_image, image_msg, image_msg.header.stamp.to_sec(), images_output_dir, img_name)
                    futures.append(future)

                    # pose_msg:TransformStamped = msg
                    qw = pose_msg.transform.rotation.w
                    qx = pose_msg.transform.rotation.x
                    qy = pose_msg.transform.rotation.y
                    qz = pose_msg.transform.rotation.z
                    tx = pose_msg.transform.translation.x
                    ty = pose_msg.transform.translation.y
                    tz = pose_msg.transform.translation.z

                    poses_file.write(f"{scale_index} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {CAMERA_ID} {img_name}\n\n")
                    scale_index += 1

def process_pointcloud(pc, xyz_offset, quaternion_realsense, distance_threshold=10.0):
    """
    Processes a PointCloud2 message by transforming coordinates, filtering by distance,
    rotating around the z-axis, downsampling, and saving the point cloud with normals.

    Parameters:
        pc: The input PointCloud2 message.
        initial_heading_degree: The rotation angle in degrees.
        pointcloud_path: The file path to save the processed point cloud (should end with .ply).
        distance_threshold: The maximum allowed Euclidean distance for points (default is 8.0 meters).
    """
    field_names = ('x', 'y', 'z', 'rgb')  # Fields to extract
    # Use pc2.read_points to read the point cloud data
    points_gen = pc2.read_points(pc, field_names=field_names, skip_nans=True)
    points = list(points_gen)  # Convert generator to list if needed
    rotation_mtx = np.eye(4)
    rotation_mtx[:3,:3] = R.from_quat(np.array([
                                        quaternion_realsense.x, 
                                        quaternion_realsense.y, 
                                        quaternion_realsense.z, 
                                        quaternion_realsense.w])).as_matrix()
    rotation_mtx[:3, 3] = np.array([xyz_offset[0], xyz_offset[1], xyz_offset[2]])
    # z_component = R.from_quat(np.array([
    #                                     quaternion_realsense.x, 
    #                                     quaternion_realsense.y, 
    #                                     quaternion_realsense.z, 
    #                                     quaternion_realsense.w])).as_euler('zyx', degrees=True)[0]
    
    # rotation_mtx = rot(-z_component,'y')
    print(rotation_mtx)
    transformed_points = []

    for point in points:
        x_rs, y_rs, z_rs, rgb = point  # Realsense frame coordinates
        if(y_rs<-0.1) or (np.linalg.norm([x_rs, y_rs, z_rs]))>distance_threshold:
            # Discard if distance exceeds threshold
            continue
        
        rotated_vec = rotation_mtx@np.array([x_rs, y_rs, z_rs,1])

        x_rs = rotated_vec[0]
        y_rs = rotated_vec[1]
        z_rs = rotated_vec[2]
        xyz_transformed =  np.array([x_rs,y_rs,z_rs,1])

        # cast float32 to int so that bitwise operations are possible
        s = struct.pack('>f' ,rgb)
        i = struct.unpack('>l',s)[0]
        pack = ctypes.c_uint32(i).value
        r = (pack & 0x00FF0000)>> 16
        g = (pack & 0x0000FF00)>> 8
        b = (pack & 0x000000FF)
        transformed_points.append((xyz_transformed[0], xyz_transformed[1], xyz_transformed[2], r, g, b))

    # Extract XYZ coordinates and colors
    xyz = np.array([[p[0], p[1], p[2]] for p in transformed_points])
    rgb = np.array([[p[3], p[4], p[5]] for p in transformed_points], dtype=np.uint32)
    rgb = rgb / 255.0  # Normalize to [0, 1]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    pcd = pcd.voxel_down_sample(voxel_size=0.05)  # Adjust voxel_size as needed

    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    return pcd

def pointcloud_registeration(source, target):
    voxel_size = 0.05  # Adjust voxel size as needed
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)

    # Estimate normals
    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    # Compute FPFH features
    # source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    #     source_down,
    #     o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    # target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    #     target_down,
    #     o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))

    # RANSAC registration
    # distance_threshold = voxel_size * 1.5
    # result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    #     source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
    #         o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
    #         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
    #     ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))

    # # Apply the transformation to the original source point cloud
    # source.transform(result_ransac.transformation)

    # Visualize the alignment
    # o3d.visualization.draw_geometries([source, target])
    threshold = 0.5  # Adjust based on your data
    # result_icp = o3d.pipelines.registration.registration_icp(
    #     source, target, threshold, result_ransac.transformation,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    # 设置初始变换矩阵为单位矩阵
    initial_transformation = np.identity(4)

    # 使用 ICP 进行配准
    threshold = 0.5  # 根据您的数据调整
    result_icp = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    # Apply the transformation
    source.transform(result_icp.transformation)
    print(f"result_icp.transformation = \n{result_icp.transformation}")

    # Visualize the refined alignment
    o3d.visualization.draw_geometries([source, target])

    # 合并配准后的源点云和目标点云
    merged_pcd = source + target

    # （可选）对合并后的点云进行下采样或去除重复点
    merged_pcd = merged_pcd.voxel_down_sample(voxel_size)
    merged_pcd.remove_duplicated_points()

    # 返回合并后的点云
    return merged_pcd

def read_pointcloud(cloud, field_names=None, skip_nans=False):
    '''Read points from a PointCloud2 message, including RGB color.

    Parameters:
        cloud: The PointCloud2 message.
        field_names: List of field names to read. If None, read all fields.
        skip_nans: If True, skip points that have NaN values.

    Yields:
        A generator of tuples of field values for each point.
    '''
    import numpy as np
    import struct
    from sensor_msgs.msg import PointField

    # Mapping from datatype to struct format
    _DATATYPES = {
        PointField.INT8:    'b',
        PointField.UINT8:   'B',
        PointField.INT16:   'h',
        PointField.UINT16:  'H',
        PointField.INT32:   'i',
        PointField.UINT32:  'I',
        PointField.FLOAT32: 'f',
        PointField.FLOAT64: 'd'
    }

    if field_names is None:
        field_names = [field.name for field in cloud.fields]

    # Build a dict mapping field names to their properties
    fields_dict = {field.name: field for field in cloud.fields}
    fields_to_read = []
    for field_name in field_names:
        if field_name not in fields_dict:
            raise ValueError(f"Field '{field_name}' does not exist in PointCloud2 message.")
        field = fields_dict[field_name]
        offset = field.offset
        datatype = field.datatype
        count = field.count
        datatype_fmt = _DATATYPES[datatype]
        fmt = datatype_fmt
        if count != 1:
            fmt = f"{count}{datatype_fmt}"
        fields_to_read.append((offset, fmt))

    # Sort fields by offset
    fields_to_read.sort(key=lambda x: x[0])

    # Build the struct format string
    fmt = '>' if cloud.is_bigendian else '<'
    last_offset = 0
    for offset, field_fmt in fields_to_read:
        gap = offset - last_offset
        if gap > 0:
            fmt += 'x' * gap  # Pad bytes
        fmt += field_fmt
        last_offset = offset + struct.calcsize(field_fmt)
    point_step = cloud.point_step
    unpack_from = struct.Struct(fmt).unpack_from

    # Read the data
    data = cloud.data
    row_step = cloud.row_step
    width = cloud.width
    height = cloud.height

    for v in range(height):
        offset = v * row_step
        for u in range(width):
            point_data = data[offset:offset+point_step]
            point = unpack_from(point_data)
            offset += point_step
            if skip_nans and any(np.isnan(p) for p in point):
                continue
            yield point

def save_pointcloud(pcd, pointcloud_path, name):
    '''Save the transformed points with normals to a PLY file using Open3D.

    Parameters:
        points: List of tuples containing (x, y, z, rgb).
        pointcloud_path: File path to save the point cloud (should end with .ply).
    '''
    pointcloud_path = pointcloud_path + name
    # Extract XYZ coordinates and colors
    # xyz = np.array([[p[0], p[1], p[2]] for p in points])
    # rgb = np.array([[p[3], p[4], p[5]] for p in points], dtype=np.uint32)

    # rgb = rgb / 255.0  # Normalize to [0, 1]

    # Create Open3D point cloud
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz)
    # pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd = pcd.voxel_down_sample(voxel_size=0.05)  # Adjust voxel_size as needed
    
    # Save the point cloud in PLY format
    o3d.io.write_point_cloud(pointcloud_path, pcd, write_ascii=True)
    print(f"Point cloud saved to {pointcloud_path} with {len(pcd.points)} points.")
    return pcd

def visualize(pointcloud,xyz_np, rgb_np, intrinsic, extrinsics):
    # print(type(pointcloud.points))
    # xyz = np.array([[p[0], p[1], p[2]] for p in xyz_list])
    # rgb = np.zeros_like(xyz)
    # rgb[1,:] = 1
    # Create the camera visualization geometry
    camera = o3d.geometry.LineSet.create_camera_visualization(
        view_width_px=640,
        view_height_px=480,
        intrinsic=intrinsic,
        extrinsic=extrinsics,
        scale=0.5  # Scale of the camera visualization for better visibility
        )


    pointcloud.points.extend(o3d.utility.Vector3dVector(xyz_np))
    pointcloud.colors.extend(o3d.utility.Vector3dVector(rgb_np))

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()

    # for geometry in geometries:
    viewer.add_geometry(pointcloud)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0, 0, 0])
    viewer.add_geometry(coordinate_frame)
    viewer.add_geometry(camera)

    opt = viewer.get_render_option()
    opt.background_color = np.asarray([0.5, 0.5, 0.5])
    viewer.run()
    viewer.destroy_window()

    return pointcloud
    # o3d.visualization.draw_geometries([pointcloud])

def pointcloud_registration_gpu(source, target):
    # 转换为 Open3D Tensor 点云
    device = o3d.core.Device("CUDA:0")

    source_t = o3d.t.geometry.PointCloud.from_legacy(source, device=device)
    target_t = o3d.t.geometry.PointCloud.from_legacy(target, device=device)

    # 体素下采样
    voxel_size = 0.05
    source_down = source_t.voxel_down_sample(voxel_size)
    target_down = target_t.voxel_down_sample(voxel_size)

    # 估计法线
    source_down.estimate_normals(radius=voxel_size * 2, max_nn=30)
    target_down.estimate_normals(radius=voxel_size * 2, max_nn=30)

    # 设置初始变换
    init_trans = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32, device)

    # 执行 GPU 加速的 ICP
    max_correspondence_distance = voxel_size * 5.0
    criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                   relative_rmse=1e-6,
                                                                   max_iteration=50)

    result_icp = o3d.t.pipelines.registration.icp(
        source_down, target_down, max_correspondence_distance, init_trans,
        estimation_method=o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=criteria)

    # 将变换应用于源点云
    source_t.transform(result_icp.transformation)

    # 转换回经典点云以进行可视化或保存
    source_transformed = source_t.to_legacy()

    # 合并点云
    merged_pcd = source_transformed + target

    return merged_pcd


def rot(degree, axis='x'):
    rad = np.deg2rad(degree)
    if axis == 'x':
        return np.array([
            [1, 0, 0,0],
            [0, np.cos(rad), -np.sin(rad),0],
            [0, np.sin(rad), np.cos(rad),0],
            [0,0,0,1]])
    elif axis == 'y':
        return np.array([
            [np.cos(rad), 0, np.sin(rad),0],
            [0, 1, 0, 0 ],
            [-np.sin(rad), 0, np.cos(rad),0],
            [0,0,0,1]
            ])
    elif axis == 'z':
        return np.array([
        [np.cos(rad), -np.sin(rad), 0,0],
        [np.sin(rad), np.cos(rad), 0,0],
        [0, 0, 1,0],
        [0,0,0,1]
        ])
    
def main(bag_file, output_dir, max_workers):
    moving_direction = MOVING_FORWARD
    rospy.init_node("node_pc_pub")
    sparse0_dir = os.path.join(output_dir, 'sparse/0/')
    # sparse0_dir = os.path.join(sparse_dir, '0')
    if not os.path.exists(sparse0_dir):
        os.makedirs(sparse0_dir)
    camera_info_path = os.path.join(sparse0_dir, 'cameras.txt')
    poses_path = os.path.join(sparse0_dir, 'images.txt')
    pointcloud_path = os.path.join(sparse0_dir, 'points3D.ply')

    images_output_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(images_output_dir):
        os.makedirs(images_output_dir)
    
    camera_pose_image = []
    image_list = []
    x_data, y_data, z_data = [], [], []
    quaternion_list = []
    iteration_pc = 30 
    index = 0
    pc_list = []
    pointcloud_offset = []
    pointcloud_rotation = []
    with rosbag.Bag(bag_file, 'r') as bag:
        # Iterate over all messages in the 'Visual_merged' topic        
        for topic, msg, t in bag.read_messages(topics=['/Visual_Merged']):
            # Extract XYZ from CameraPose
            msg:visual_merged_msg = msg
            # camera_pose_image.append((msg.CameraPose, msg.Image))
            image_list.append(msg.Image)
            camera_info = msg.CameraInfo
            x_data.append(msg.CameraPose.transform.translation.x)
            y_data.append(msg.CameraPose.transform.translation.y)
            z_data.append(msg.CameraPose.transform.translation.z)
            quaternion_list.append(msg.CameraPose.transform.rotation)
            # print(msg.Local_Map.header.stamp)
            # pc2_pub.publish(msg.Local_Map)
            # rospy.Rate(5).sleep()
            if(index%iteration_pc == 0):
                pc_list.append(msg.Local_Map)
                pointcloud_offset.append([msg.CameraPose.transform.translation.x,
                                    msg.CameraPose.transform.translation.y,
                                    msg.CameraPose.transform.translation.z])
                pointcloud_rotation.append(msg.CameraPose.transform.rotation)
            index += 1

    delta_x = x_data[100] - x_data[0]
    delta_y = y_data[100] - y_data[0]
    angle_rad = math.atan2(delta_y,delta_x)
    initial_heading_degree = math.degrees(angle_rad)
    print(f"Init Angle = {initial_heading_degree}")

    if(moving_direction == MOVING_FORWARD):
        if(initial_heading_degree < 0 and initial_heading_degree >= -90):
            initial_heading_degree = 90 - initial_heading_degree
            assert initial_heading_degree >= 90
        elif(initial_heading_degree<-90 and initial_heading_degree> -180):
            initial_heading_degree = -90 + initial_heading_degree
            assert initial_heading_degree<-180
        elif(initial_heading_degree>=0 and initial_heading_degree < 90):
            initial_heading_degree = initial_heading_degree
            assert initial_heading_degree > 0
        elif(initial_heading_degree>90 and initial_heading_degree < 180):
            initial_heading_degree = -(initial_heading_degree - 90)
            assert initial_heading_degree<0

    # if(initial_heading_degree < -90):
    #     initial_heading_degree = 90 + initial_heading_degree
    # elif(initial_heading_degree > 90):
    #     initial_heading_degree = initial_heading_degree - 90
    print(f"Init Angle = {initial_heading_degree}")
    
    xyz_list = []
    rgb_list = []
    for i in range(len(x_data)):
        xyz_list.append([x_data[i], y_data[i],z_data[i]])
        rgb_list.append([1,0,0])
    x_offset = xyz_list[0][0]
    y_offset = xyz_list[0][1]
    z_offset = xyz_list[0][2]

    for i in range(len(pointcloud_offset)):
        cloud_gps_pt = pointcloud_offset[i]
        pt_shifted = np.array([cloud_gps_pt[0] - x_offset, cloud_gps_pt[1] - y_offset, cloud_gps_pt[2] - z_offset, 1])
        pt_rotated_init = rot(initial_heading_degree,'z')@pt_shifted
        pt_rotated_x = rot(90,'x')@pt_rotated_init
        pointcloud_offset[i] = [pt_rotated_x[0], pt_rotated_x[1], pt_rotated_x[2]]
        print(pointcloud_offset[i])
    # quit()
    from tqdm import tqdm
    merged_cloud = process_pointcloud(pc_list[0], pointcloud_offset[0], pointcloud_rotation[0]) #Also Target
    for i in tqdm(range(len(pc_list)-1)):
        source = process_pointcloud(pc_list[i+1], pointcloud_offset[i+1], pointcloud_rotation[i+1])
        merged_cloud = pointcloud_registration_gpu(source ,merged_cloud)
        save_pointcloud(merged_cloud, sparse0_dir,f"registered_{i}.ply")
        save_pointcloud(merged_cloud, sparse0_dir,f"registered_{i}.pcd")
        merged_cloud = merged_cloud + source
        # if(i%5 == 0):
        #     o3d.visualization.draw_geometries([merged_cloud])
    save_pointcloud(merged_cloud, sparse0_dir,f"points3D.ply")



    intrinsic = np.array([
        [500, 0, 320],  # fx, 0, cx
        [0, 500, 240],  # 0, fy, cy
        [0, 0, 1]       # 0, 0, 1
    ])

    extrinsics = np.eye(4)
    T_x = rot(-90,'x')
    T_y = rot(-1*(90 - initial_heading_degree),'z')
    T_y @ T_x
    # Optionally set a translation (e.g., place the camera at (1, 0, 0))
    extrinsics[:3, 3] = [x_data[0],y_data[0],z_data[0]]  # Example translation
    print(extrinsics)
    q = R.from_matrix(extrinsics[:3, :3]).as_quat()

    extrinsics = np.linalg.inv(extrinsics)
    T = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
    r = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    rotation = R.from_matrix(r)
    quaternion = rotation.as_quat()

    # 应用转换
    R_colmap = T @ r
    t_colmap = T @ t

    rotation_colmap = R.from_matrix(R_colmap)
    quaternion_colmap = rotation_colmap.as_quat()
    qx_colmap, qy_colmap, qz_colmap, qw_colmap = quaternion_colmap


    transformed_xyz = []
    for i in range(len(quaternion_list)):
        ts = TransformStamped()
        q:Quaternion = quaternion_list[i]
        pt_shifted = np.array([xyz_list[i][0] - x_offset, xyz_list[i][1] - y_offset, xyz_list[i][2] - z_offset, 1])
        pt_rotated_init = rot(initial_heading_degree,'z')@pt_shifted
        pt_rotated_x = rot(90,'x')@pt_rotated_init
        print(pt_shifted, pt_rotated_x)
        transformed_xyz.append([pt_rotated_x[0],pt_rotated_x[1],pt_rotated_x[2]])


        extrinsic = np.eye(4)
        extrinsic[:3,:3] = R.from_quat(np.array([q.x, q.y, q.z, q.w])).as_matrix()
        extrinsic[:3,3] = np.array([pt_rotated_x[0], pt_rotated_x[1], pt_rotated_x[2]])
        inv_extrinsic = np.linalg.inv(extrinsic)
        print(inv_extrinsic)
        inv_q = R.from_matrix(inv_extrinsic[:3, :3]).as_quat() #xyzw
        inv_p = inv_extrinsic[:3, 3]

        ts.transform.translation.x = inv_p[0]
        ts.transform.translation.y = inv_p[1]
        ts.transform.translation.z = inv_p[2]
        
        # Put the data here.
        ts.transform.rotation.w = inv_q[3]
        ts.transform.rotation.x = inv_q[0]
        ts.transform.rotation.y = inv_q[1]
        ts.transform.rotation.z = inv_q[2]

        camera_pose_image.append((ts,image_list[i]))
        # print(ts.transform.translation)


    extract_camera_info(msg.CameraInfo, camera_info_path)
    extract_camera_pose_image(camera_pose_image, poses_path, images_output_dir, max_workers)
    pcd = visualize(merged_cloud,np.array(transformed_xyz), np.array(rgb_list),intrinsic,extrinsics)
    o3d.io.write_point_cloud(pointcloud_path +"_1.ply", pcd, write_ascii=True)
    o3d.io.write_point_cloud(pointcloud_path + "_1.pcd", pcd, write_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract data from a ROS bag file.")
    parser.add_argument('bag_file', help="Path to the input ROS bag file.")
    parser.add_argument('output_dir', help="Directory to save extracted data.")
    parser.add_argument('--visual_merged_msg', default='/Visual_Merged', help="ROS topic for point cloud.")
    parser.add_argument('--max_workers', type=int, default=20, help="Maximum number of worker threads.")

    args = parser.parse_args()

    main(args.bag_file, args.output_dir, args.max_workers)
