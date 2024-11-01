#!/usr/bin/env python

import ctypes
import rospy, rosbag
from gs_slam_msgs.msg import visual_merged_msg  # Replace with your message type
from geometry_msgs.msg import TransformStamped
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
        print(f"Saved image {img_name}")

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

def process_pointcloud(pc, initial_heading_degree, translation, pointcloud_path, distance_threshold=8.0):
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
    
    transformed_points = []
    theta = (np.pi - np.deg2rad(initial_heading_degree))  # Negative rotation angle in radians
    # theta = 0.2
    # Rotation matrix around z-axis
    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

    # Frame transformation matrix from Realsense frame to your frame
    # Realsense Frame: X (right), Y (down), Z (front)
    # Your Frame:      X (right), Y (front), Z (up)
    T = np.array([
        [1,  0,  0],  # X remains the same
        [0,  0,  1],  # Y becomes Z
        [0, -1,  0]   # Z becomes -Y
    ])

    for point in points:
        x_rs, y_rs, z_rs, rgb = point  # Realsense frame coordinates
        if(y_rs<0.1):
            continue
        # Discard if distance exceeds threshold
        distance = np.linalg.norm([x_rs, y_rs, z_rs])
        if distance > distance_threshold:
            continue
        # Convert to numpy array

        xyz_rotated = T @ np.array([x_rs,y_rs,z_rs])
        xyz_rotated = R_z @ xyz_rotated
        xyz_transformed = xyz_rotated +  np.array([translation.x, translation.y, translation.z])
            # Transform to your frame
        # xyz_custom = np.linalg.inv(T) @ xyz_rs
            # Rotate around the z-axis
        # xyz_rotated = R_z @ xyz_rs
            # Compute Euclidean distance


        # Keep the point
        # cast float32 to int so that bitwise operations are possible

        s = struct.pack('>f' ,rgb)
        i = struct.unpack('>l',s)[0]
        # you can get back the float value by the inverse operations
        pack = ctypes.c_uint32(i).value
        r = (pack & 0x00FF0000)>> 16
        g = (pack & 0x0000FF00)>> 8
        b = (pack & 0x000000FF)

        transformed_points.append((xyz_transformed[0], xyz_transformed[1], xyz_transformed[2], r, g, b))
        # transformed_points.append((xyz_rs[0], xyz_rs[1], xyz_rs[2], r, g, b))
    # print(transformed_points[0])
    # Save the transformed point cloud with normals and downsampling
    return save_pointcloud(transformed_points, pointcloud_path)

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

def save_pointcloud(points, pointcloud_path):
    '''Save the transformed points with normals to a PLY file using Open3D.

    Parameters:
        points: List of tuples containing (x, y, z, rgb).
        pointcloud_path: File path to save the point cloud (should end with .ply).
    '''

    # Extract XYZ coordinates and colors
    xyz = np.array([[p[0], p[1], p[2]] for p in points])
    rgb = np.array([[p[3], p[4], p[5]] for p in points], dtype=np.uint32)

    rgb = rgb / 255.0  # Normalize to [0, 1]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    pcd = pcd.voxel_down_sample(voxel_size=0.05)  # Adjust voxel_size as needed
    
    print(pcd.points[0])
    print(pcd.colors[0])
    print(pcd.normals[0])
    # Optionally, you can re-orient the normals if needed
    # pcd.orient_normals_consistent_tangent_plane(k=10)
    # with open(pointcloud_path, 'w') as f:
    #     # Write PLY header
    #     f.write("ply\n")
    #     f.write("format ascii 1.0\n")
    #     f.write(f"element vertex {num_points}\n")
    #     f.write("property float x\n")
    #     f.write("property float y\n")
    #     f.write("property float z\n")
    #     f.write("property float nx\n")
    #     f.write("property float ny\n")
    #     f.write("property float nz\n")
    #     f.write("property uchar red\n")
    #     f.write("property uchar green\n")
    #     f.write("property uchar blue\n")
    #     f.write("end_header\n")
    #     # Write point data with black color (0,0,0)
    #     for point in points:
    #         f.write(f"{point[0]} {point[1]} {point[2]} 0 0 1 1 1 1\n")
    #     print(f"Saved point cloud to {pointcloud_path}")
    # except Exception as e:
    #     print(f"Could not extract point cloud: {e}")

    
    # Save the point cloud in PLY format
    o3d.io.write_point_cloud(pointcloud_path, pcd, write_ascii=True)
    o3d.io.write_point_cloud(pointcloud_path + "_.pcd", pcd, write_ascii=False)

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
    # o3d.visualization.draw_geometries([pointcloud])

    
def main(bag_file, output_dir, max_workers):
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
    camera_info = None
    translation = None
    x_data, y_data, z_data = [], [], []
    quaternion_list = []
    has_pointcloud = False
    with rosbag.Bag(bag_file, 'r') as bag:
        # Iterate over all messages in the 'Visual_merged' topic        
        for topic, msg, t in bag.read_messages(topics=['/Visual_Merged']):
            # Extract XYZ from CameraPose
            msg:visual_merged_msg = msg
            camera_pose_image.append((msg.CameraPose, msg.Image))
            camera_info = msg.CameraInfo
            x_data.append(msg.CameraPose.transform.translation.x)
            y_data.append(msg.CameraPose.transform.translation.y)
            z_data.append(msg.CameraPose.transform.translation.z)
            quaternion_list.append(msg.CameraPose.transform.rotation)
            # print(msg.Local_Map.header.stamp)
            # pc2_pub.publish(msg.Local_Map)
            # rospy.Rate(5).sleep()
            if(not has_pointcloud):
                pc = msg.Local_Map
                translation = msg.CameraPose.transform.translation
                has_pointcloud = True
                # print(translation)
    
    
    extract_camera_info(msg.CameraInfo, camera_info_path)
    extract_camera_pose_image(camera_pose_image, poses_path, images_output_dir, max_workers)
    print (f' ======  Open3D=={o3d.__version__}  ======= ')

    delta_x = x_data[-1] - x_data[0]
    delta_y = y_data[-1] - y_data[0]
    angle_rad = math.atan2(delta_y,delta_x)
    initial_heading_degree = math.degrees(angle_rad)
    
    print(f"Start {x_data[0]}, {y_data[0]}")
    print(f"End {x_data[-1]}, {y_data[-1]}")
    print(initial_heading_degree)
    # if(initial_heading_degree < 0):

    pcd = process_pointcloud(pc,initial_heading_degree, translation, pointcloud_path)
    xyz_np = np.array([])
    xyz_list = []
    rgb_list = []
    for i in range(len(x_data)):
        xyz_list.append([x_data[i], y_data[i],z_data[i]])
        rgb_list.append([1,0,0])
        # print([x_data[i], y_data[i],z_data[i]])
        # xyz = np.array([x_data[i], y_data[i],z_data[i]])
    # xyz_list.append([0,0,0])
    # rgb_list.append([0,0,1])
    # Define the quaternion (w, x, y, z) - example for 90-degree rotation around the z-axis
    T = np.array([
        [1,  0,  0],  # X remains the same
        [0,  0,  -1],  # Y becomes Z
        [0, 1,  0]   # Z becomes -Y
    ])

    q = quaternion_list[0]
    quaternion = [q.x, q.y, q.z, q.w]  # w, x, y, z format

    # Convert the quaternion to a 3x3 rotation matrix
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    rotation_matrix = T@rotation_matrix
    theta = (np.pi/2 + np.deg2rad(initial_heading_degree))  # Negative rotation angle in radians
    print(theta)
    # theta = 0.2
    # Rotation matrix around z-axis
    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    rotation_matrix = R_z@rotation_matrix

    print(rotation_matrix, quaternion)
    # Initialize a 4x4 extrinsics matrix
    extrinsics = np.eye(4)

    # Set the top-left 3x3 part of the extrinsics matrix to the rotation matrix
    extrinsics[:3, :3] = rotation_matrix

    # Optionally set a translation (e.g., place the camera at (1, 0, 0))
    extrinsics[:3, 3] = [-x_data[0],y_data[0], z_data[0]]  # Example translation
    print(x_data[0],y_data[0], z_data[0])
    intrinsic = np.array([
        [500, 0, 320],  # fx, 0, cx
        [0, 500, 240],  # 0, fy, cy
        [0, 0, 1]       # 0, 0, 1
    ])
    visualize(pcd,np.array(xyz_list), np.array(rgb_list),intrinsic,extrinsics)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract data from a ROS bag file.")
    parser.add_argument('bag_file', help="Path to the input ROS bag file.")
    parser.add_argument('output_dir', help="Directory to save extracted data.")
    parser.add_argument('--visual_merged_msg', default='/Visual_Merged', help="ROS topic for point cloud.")
    parser.add_argument('--max_workers', type=int, default=20, help="Maximum number of worker threads.")

    args = parser.parse_args()

    main(args.bag_file, args.output_dir, args.max_workers)
