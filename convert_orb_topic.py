#!/usr/bin/env python

import os
import argparse
import rosbag
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sensor_msgs.msg import CameraInfo, Image as ImageMsg, PointCloud2
from geometry_msgs.msg import PoseStamped
import struct
from plyfile import PlyData, PlyElement

def process_and_save_image(msg, timestamp, output_dir, img_name):
    try:
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
        image = Image.frombytes(mode, (width, height), data, 'raw', raw_mode, msg.step)

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

def extract_camera_poses_and_images(bag, output_dir, image_topic, camera_pose_topic, max_workers=4):
    # Dictionaries to hold messages keyed by header timestamp
    image_msgs = {}
    pose_msgs = {}

    # Read all messages from the bag
    print("Reading messages from bag...")
    for topic, msg, t in bag.read_messages(topics=[image_topic, camera_pose_topic]):
        if not hasattr(msg, 'header') or not hasattr(msg.header, 'stamp'):
            print(f"Message on topic {topic} does not have a valid header.")
            continue
        timestamp = msg.header.stamp.to_sec()
        if topic == image_topic:
            image_msgs[timestamp] = msg
        elif topic == camera_pose_topic:
            pose_msgs[timestamp] = msg

    # Find common timestamps
    image_times = sorted(image_msgs.keys())
    pose_times = sorted(pose_msgs.keys())

    # Synchronize messages based on header timestamps within a threshold
    threshold = 0.033  # e.g., 33 milliseconds for 30 Hz data
    matched_data = []

    pose_idx = 0
    for img_time in image_times:
        # Find the closest pose time
        while pose_idx < len(pose_times) and pose_times[pose_idx] < img_time - threshold:
            pose_idx += 1
        if pose_idx >= len(pose_times):
            break
        time_diff = abs(pose_times[pose_idx] - img_time)
        if time_diff <= threshold:
            matched_data.append((img_time, pose_times[pose_idx]))
            pose_idx += 1  # Move to next pose time to prevent duplicate matches

    print(f"Found {len(matched_data)} synchronized messages.")

    # Process images and poses
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images_output_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(images_output_dir):
        os.makedirs(images_output_dir)
    scale_index = 0
    sparse0_dir = os.path.join(output_dir, 'sparse/0/')
    poses_path = os.path.join(sparse0_dir, 'images.txt')
    with open(poses_path, 'w') as poses_file:
        CAMERA_ID = 1  # Assuming a single camera ID
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, (img_time, pose_time) in enumerate(matched_data, start=1):
                if(idx%4 == 0):
                    img_msg = image_msgs[img_time]
                    pose_msg = pose_msgs[pose_time]

                    img_name = f"frame_{scale_index:06d}.jpg"
                    
                    # Submit image processing to the thread pool
                    future = executor.submit(process_and_save_image, img_msg, img_time, images_output_dir, img_name)
                    futures.append(future)

                    # Write pose information
                    qw = pose_msg.pose.orientation.w
                    qx = pose_msg.pose.orientation.x
                    qy = pose_msg.pose.orientation.y
                    qz = pose_msg.pose.orientation.z
                    tx = pose_msg.pose.position.x
                    ty = pose_msg.pose.position.y
                    tz = pose_msg.pose.position.z

                    poses_file.write(f"{scale_index} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {CAMERA_ID} {img_name}\n\n")
                    scale_index += 1
            # Wait for all image processing tasks to complete
            for future in as_completed(futures):
                pass  # Handle exceptions if needed

    print(f"Saved {len(matched_data)} images and poses.")

def extract_pointcloud(bag, pointcloud_topic, output_dir):
    try:
        sparse0_dir = os.path.join(output_dir, 'sparse/0/')
        pointcloud_path = os.path.join(sparse0_dir, 'points3D.ply')
        last_msg = None
        # Read the first point cloud message
        for topic, msg, t in bag.read_messages(topics=[pointcloud_topic]):
            if(topic == pointcloud_topic):
                # Convert PointCloud2 message to numpy array
                last_msg = msg
                # break  # Only process the first point cloud message
        
        points_list = []
        for point in read_points(last_msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = point
            points_list.append([x, y, z])

        print(f"points_list size = {len(points_list)}")
        
        points = np.array(points_list)
        print(points)
        # Write to PLY file
        num_points = points.shape[0]
        with open(pointcloud_path, 'w') as f:
            # Write PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {num_points}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            # Write point data with black color (0,0,0)
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]} 0 0 1 1 1 1\n")
        print(f"Saved point cloud to {pointcloud_path}")
    except Exception as e:
        print(f"Could not extract point cloud: {e}")

def read_points(cloud, field_names=None, skip_nans=False):
    # Helper function to read PointCloud2 data
    fmt = 'fff'  # Assuming point cloud has x, y, z as float32
    width = cloud.width
    height = cloud.height
    point_step = cloud.point_step
    row_step = cloud.row_step
    data = cloud.data

    unpacker = struct.Struct(fmt)
    is_bigendian = cloud.is_bigendian

    for v in range(height):
        for u in range(width):
            offset = v * row_step + u * point_step
            point_data = data[offset:offset+point_step]
            if is_bigendian:
                point = unpacker.unpack(point_data)
            else:
                point = unpacker.unpack_from(point_data)
            if skip_nans and any(np.isnan(p) for p in point):
                continue
            yield point

def main(bag_file, output_dir, image_topic, camera_info_topic, camera_pose_topic, pointcloud_topic, max_workers=4):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sparse0_dir = os.path.join(output_dir, 'sparse/0/')
    # sparse0_dir = os.path.join(sparse_dir, '0')
    if not os.path.exists(sparse0_dir):
        os.makedirs(sparse0_dir)

    print(f"Opening bag file: {bag_file}")
    bag = rosbag.Bag(bag_file, 'r')

    # Extract camera info
    camera_info_saved = False
    for topic, msg, t in bag.read_messages(topics=[camera_info_topic]):
        camera_info_path = os.path.join(sparse0_dir, 'cameras.txt')
        extract_camera_info(msg, camera_info_path)
        camera_info_saved = True
        break  # Only need the first camera info message

    if not camera_info_saved:
        print("No camera info message found.")

    # Extract images and poses
    extract_camera_poses_and_images(bag, output_dir, image_topic, camera_pose_topic, max_workers)

    # Extract point cloud
    extract_pointcloud(bag, pointcloud_topic, output_dir)

    bag.close()
    print("Extraction complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract data from a ROS bag file.")
    parser.add_argument('bag_file', help="Path to the input ROS bag file.")
    parser.add_argument('output_dir', help="Directory to save extracted data.")
    parser.add_argument('--image_topic', default='/camera/color/image_raw', help="ROS topic for image data.")
    parser.add_argument('--camera_info_topic', default='/camera/color/camera_info', help="ROS topic for camera info.")
    parser.add_argument('--camera_pose_topic', default='/orb_slam3/camera_pose', help="ROS topic for camera poses.")
    parser.add_argument('--pointcloud_topic', default='/orb_slam3/all_points', help="ROS topic for point cloud.")
    parser.add_argument('--max_workers', type=int, default=4, help="Maximum number of worker threads.")

    args = parser.parse_args()

    main(args.bag_file, args.output_dir, args.image_topic, args.camera_info_topic,
         args.camera_pose_topic, args.pointcloud_topic, args.max_workers)
