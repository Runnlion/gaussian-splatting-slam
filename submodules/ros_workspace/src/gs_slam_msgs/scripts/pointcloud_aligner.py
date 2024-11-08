import rospy
from gs_slam_msgs.msg import visual_merged_msg
from sensor_msgs.msg import PointCloud2, PointField, Image
import sensor_msgs.point_cloud2 as pc2
import numpy as np

vsm_cache = []
pc2_cache = []

rospy.init_node('pointcloud_aligner', anonymous=False)

pub = rospy.Publisher('image', Image, queue_size=1)
def transform_point_cloud(point_cloud, transformation_matrix):
    """
    Transforms a PointCloud2 message using a 4x4 transformation matrix.
    
    Args:
        point_cloud (PointCloud2): Input point cloud message.
        transformation_matrix (numpy.ndarray): 4x4 transformation matrix.
    
    Returns:
        PointCloud2: Transformed point cloud message.
    """
    # Extract points from PointCloud2 message
    points = pc2.read_points(point_cloud, skip_nans=True, field_names=("x", "y", "z"))
    transformed_points = []

    # Apply the transformation matrix to each point
    for point in points:
        # Create a 4x1 vector for the point (homogeneous coordinates)
        p = np.array([point[0], point[1], point[2], 1.0]).reshape(4, 1)
        
        # Apply transformation
        p_transformed = np.dot(transformation_matrix, p)
        
        # Convert back to 3D point and store
        transformed_points.append([p_transformed[0, 0], p_transformed[1, 0], p_transformed[2, 0]])

    # Create a new PointCloud2 message with transformed points
    header = point_cloud.header
    fields = point_cloud.fields
    transformed_point_cloud = pc2.create_cloud(header, fields, transformed_points)

    return transformed_point_cloud

def visual_merged_callback(data:visual_merged_msg):
    # rospy.loginfo("Visual_Merged Received")
    pub.publish(data.Image)
    vsm_cache.append(data)

def pointcloud_callback(pc:PointCloud2):
    # rospy.loginfo("Pointcloud Received, Pair with VSM")

    for i in range(len(vsm_cache)):
        vsm = vsm_cache.pop() 
        # time_diff_img_pc = abs((visual_merged_msg.Image.header.stamp - pc.header.stamp).to_sec())
        time_diff_pose_pc = abs((vsm.CameraPose.header.stamp - pc.header.stamp).to_sec())
        if (time_diff_pose_pc) <= 0.2:
            
            rospy.loginfo(f"Paired, Time Diff = {time_diff_pose_pc}, VSM Size = {len(vsm_cache)}, PointCloud Size = {pc.width}")
            transformation_matrix = np.array([
                [1, 0, 0, 1],  # Translate x by 1
                [0, 1, 0, 2],  # Translate y by 2
                [0, 0, 1, 3],  # Translate z by 3
                [0, 0, 0, 1]
            ])
            # transform the point to X + 90 Degree
            # transformed_cloud = transform_point_cloud(pc, transformation_matrix)

            # convert the pointclud in train program
            
            vsm_cache.clear()
            break
    pass
            

rospy.Subscriber('/Visual_Merged', visual_merged_msg, visual_merged_callback, queue_size=1)
rospy.Subscriber('/camera/depth/color/points', PointCloud2, pointcloud_callback, queue_size=1)


rospy.spin()