import os
import argparse
import rosbag, rospy
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sensor_msgs.msg import CameraInfo, Image as ImageMsg, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
import struct
from plyfile import PlyData, PlyElement
import open3d as o3d

def read_points_with_color(cloud, field_names=None, skip_nans=False):
    '''Read points from a PointCloud2 message, including RGB color.

    Parameters:
        cloud: The PointCloud2 message.
        field_names: List of field names to read. If None, read all fields.
        skip_nans: If True, skip points that have NaN values.

    Yields:
        A generator of tuples of field values for each point.
    '''

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


def extract_pointcloud(msg:PointCloud2):
    try:
        pointcloud_path = './points3D.ply'

        points_list = []
        field_names = ['x', 'y', 'z', 'rgb']  # Fields you want to extract
        
        points_list = []
        for point in read_points_with_color(msg, field_names=['x', 'y', 'z', 'rgb']):
            x, y, z, rgb = point
            # Extract RGB values from packed float
            if isinstance(rgb, float):
                # For float RGB values
                rgb_int = int(rgb)
            else:
                # For packed integer RGB values
                rgb_int = rgb
            r = (rgb_int >> 16) & 0x0000ff
            g = (rgb_int >> 8) & 0x0000ff
            b = rgb_int & 0x0000ff
            points_list.append([x, y, z, r, g, b])
        points = np.array(points_list)

        # Process the point cloud
        processed_pcd = process_point_cloud(points, voxel_size=0.05, nb_neighbors=20, std_ratio=2.0)

        # Save or visualize the processed point cloud
        o3d.io.write_point_cloud("./processed_point_cloud.ply", processed_pcd)
        # for point in points:
        #     x, y, z, rgb = point

        print(f"points_list size = {len(points)}")
        
        # points = np.array(points_list)
        # print(points)
        # Write to PLY file
        # num_points = points.shape[0]
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
        # print(f"Saved point cloud to {pointcloud_path}")
    except Exception as e:
        print(f"Could not extract point cloud: {e}")

def create_open3d_point_cloud(points):
    '''Create an Open3D point cloud from numpy array of points.

    Parameters:
        points: Nx6 numpy array containing x, y, z, r, g, b values.

    Returns:
        Open3D PointCloud object.
    '''
    xyz = points[:, :3]
    rgb = points[:, 3:6] / 255.0  # Normalize RGB values to [0, 1]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd

def downsample_point_cloud(pcd, voxel_size):
    '''Downsample the point cloud using voxel grid filtering.

    Parameters:
        pcd: Open3D PointCloud object.
        voxel_size: Voxel size for downsampling.

    Returns:
        Downsampled Open3D PointCloud object.
    '''
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    return downsampled_pcd

def remove_statistical_outliers(pcd, nb_neighbors, std_ratio):
    '''Remove outliers using statistical outlier removal.

    Parameters:
        pcd: Open3D PointCloud object.
        nb_neighbors: Number of neighbors to analyze for each point.
        std_ratio: Standard deviation ratio threshold.

    Returns:
        Filtered Open3D PointCloud object.
    '''
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                             std_ratio=std_ratio)
    filtered_pcd = pcd.select_by_index(ind)
    return filtered_pcd

def process_point_cloud(points, voxel_size=0.05, nb_neighbors=20, std_ratio=2.0):
    '''Process the point cloud: downsample and remove outliers.

    Parameters:
        points: Nx6 numpy array containing x, y, z, r, g, b values.
        voxel_size: Voxel size for downsampling.
        nb_neighbors: Number of neighbors for outlier removal.
        std_ratio: Standard deviation ratio for outlier removal.

    Returns:
        Processed Open3D PointCloud object.
    '''
    pcd = create_open3d_point_cloud(points)
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud("./original.ply", pcd)
    pcd_downsampled = downsample_point_cloud(pcd, voxel_size)
    pcd_filtered = remove_statistical_outliers(pcd_downsampled, nb_neighbors, std_ratio)
    return pcd_filtered

def main():

    rospy.init_node("pc_listener")
    rospy.Subscriber('/camera/depth/color/points', PointCloud2,extract_pointcloud)
    rospy.spin()
    return

if __name__ == '__main__':


    main()
