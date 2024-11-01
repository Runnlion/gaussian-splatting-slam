import rospy, tf_conversions
from geometry_msgs.msg import PointStamped, QuaternionStamped, Quaternion, PoseStamped, TransformStamped
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, Imu
from gs_slam_msgs.msg import visual_merged_msg
from scipy.spatial.transform import Rotation as R
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sensor_msgs.point_cloud2 as pc2
import sensor_msgs.point_cloud2

class CameraPosefuser():

    def __init__(self) -> None:

        self.times = 0
        self.color_camera_info = CameraInfo()
        self.previous = [0,0,0]
        self.diff_stored = []
        self.visual_merged_msg = visual_merged_msg()
        self.visual_merged_msg.Local_Map = PointCloud2()

        self.imu_cache:list[QuaternionStamped] = list()
        self.pc2_cache:list[PointCloud2] = list()
        self.gps_cache:list[PointStamped] = list()

        # Expiry time in seconds
        self.expiry_time = 0.05
        # self.curr_quaternion = QuaternionStamped() 
        self.gps_topic = "/rtk_gps_pos"
        self.imu_topic = "/imu/data"
        self.map_initialized = False
        rospy.init_node("camera_pose_listener",anonymous=False)
        self.visual_merged_pub = rospy.Publisher('/Visual_Merged',visual_merged_msg,queue_size=1)
        self.pose_pub = rospy.Publisher('/Camera_Pose', PoseStamped, queue_size=1)
        self.quaternion_pub = rospy.Publisher('/camera_quaternion',Quaternion,queue_size=1)

        rospy.Subscriber(self.gps_topic,PointStamped, self.gps_position_callback)
        rospy.Subscriber(self.imu_topic, Imu, self.imu_callback)
        rospy.loginfo(f"Start listening {self.gps_topic}.")
        rospy.loginfo(f"Start listening {self.imu_topic}.")

        # Subscriber for CameraInfo
        self.image_info_sub = rospy.Subscriber(
            '/camera/color/camera_info',
            CameraInfo,
            self.camera_info_listener,
            queue_size=1
        )

        # Subscribers for Image and TF topics
        self.image_sub = rospy.Subscriber(
            '/camera/color/image_raw',
            Image,
            self.image_callback,
            queue_size=1
        )

        self.pointcloud_sub = rospy.Subscriber(
            '/camera/depth/color/points',
            PointCloud2,
            self.pointcloud_callback,
            queue_size=1
        )

        # Initialize ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=1)

        # Initial heading value (in degrees), set via ROS parameter
        self.initial_heading = float(rospy.get_param('~initial_heading', 0.0))

        # Store the latest IMU orientation
        self.latest_imu_quaternion = None

        # Flag to check if transformation is in progress
        self.transforming = False

    def pointcloud_callback(self, pc2_msg:PointCloud2):
        # Save point cloud to cache
        rospy.loginfo("PointCloud")
        pc2_msg.header.stamp = rospy.Time.now()
        self.pc2_cache.append(pc2_msg)

    def image_callback(self, img_msg: Image):
        # Pair image with nearest data
        img_msg.header.stamp = rospy.Time.now() #Update the timestamp, because the IMU speed is faster than Image
        image_timestamp_offset = rospy.Time.now() - img_msg.header.stamp
        self.pair_image_Pose(img_msg)

    def gps_position_callback(self, gps_position_msg:PointStamped):
        self.gps_cache.append(gps_position_msg)

    def find_nearest_message(self, cache, target_time, max_time_diff):
        nearest_msg = None
        min_time_diff = max_time_diff
        m = Imu()
        
        for msg in cache:
            time_diff = abs((msg.header.stamp - target_time).to_sec())
            # print(target_time, time_diff, msg.header.stamp)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                nearest_msg = msg
        return nearest_msg

    def pair_image_Pose(self, img_msg:Image)->bool:
        current_time = img_msg.header.stamp
        self.pc2_cache = [pc for pc in self.pc2_cache if abs((current_time - pc.header.stamp).to_sec()) <= self.expiry_time]
        self.imu_cache = [imu for imu in self.imu_cache if abs((current_time - imu.header.stamp).to_sec()) <= self.expiry_time]
        self.gps_cache = [gps for gps in self.gps_cache if abs((current_time - gps.header.stamp).to_sec()) <= self.expiry_time]

        imu_msg:QuaternionStamped = self.find_nearest_message(self.imu_cache, current_time, self.expiry_time)
        gps_msg:PointStamped = self.find_nearest_message(self.gps_cache, current_time, self.expiry_time)
        pc_msg:PointCloud2 = self.find_nearest_message(self.pc2_cache, current_time, self.expiry_time)
        
        if imu_msg is None or gps_msg is None or pc_msg is None:
            imu_status = "IMU" if imu_msg is None else ""
            gps_status = "GPS" if gps_msg is None else ""
            pc_status = "PointCloud" if pc_msg is None else ""

            reason = f"NO [{imu_status} {gps_status} {pc_status}]"
            rospy.loginfo(f"Could not find matching messages for the image. Reason: {reason}")
            return False
        rospy.loginfo("Processing.")
        # Create PoseStamped message using GPS and IMU data
        ps = PoseStamped()
        ps.header = img_msg.header
        ps.pose.position = gps_msg.point
        ps.pose.orientation = imu_msg.quaternion

        # Prepare TransformStamped
        ts = TransformStamped()
        ts.header = img_msg.header
        ts.child_frame_id = "base"
        ts.transform.rotation = ps.pose.orientation
        ts.transform.translation = ps.pose.position

        self.visual_merged_msg.CameraPose = ts
        self.visual_merged_msg.CameraInfo = self.color_camera_info
        self.visual_merged_msg.Image = img_msg

        # Submit the point cloud transformation task
        # if not self.transforming:
        #     self.transforming = True
        # if(not self.map_initialized):
        #     self.executor.submit(self.transform_pointcloud, pc_msg, imu_msg, gps_msg)
        # else:
            # publish here
        self.visual_merged_msg.Local_Map = pc_msg
        # Publish the VisualMergedMsg
        self.visual_merged_pub.publish(self.visual_merged_msg)
        rospy.loginfo("Published Without Pointcloud")
        # else:
        #     rospy.loginfo("Transformation already in progress.")

        return True

    def transform_pointcloud(self, pc_msg, imu_msg, gps_msg):
        try:
            # Convert initial heading and current IMU reading to rotations
            initial_rotation = R.from_euler('z', np.deg2rad(self.initial_heading))
            imu_quaternion = [imu_msg.quaternion.x, imu_msg.quaternion.y, imu_msg.quaternion.z, imu_msg.quaternion.w]
            imu_rotation = R.from_quat(imu_quaternion)

            # Compute the relative rotation
            relative_rotation = imu_rotation * initial_rotation.inv()

            # Build transformation matrix
            translation = np.array([gps_msg.point.x, gps_msg.point.y, gps_msg.point.z])
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = relative_rotation.as_matrix()
            transformation_matrix[:3, 3] = translation

            # Transform the point cloud
            transformed_points = []
            for point in pc2.read_points(pc_msg, skip_nans=True):
                point_xyz = np.array([point[0], point[1], point[2], 1.0])
                transformed_point = transformation_matrix @ point_xyz
                transformed_points.append((transformed_point[0], transformed_point[1], transformed_point[2], point[3]))

            # Create a new PointCloud2 message
            header = pc_msg.header
            header.stamp = rospy.Time.now()
            transformed_pc2 = pc2.create_cloud(header, pc_msg.fields, transformed_points)

            self.visual_merged_msg.Local_Map = transformed_pc2

            print("finished")
            self.map_initialized = True
            # Publish the VisualMergedMsg
            self.visual_merged_pub.publish(self.visual_merged_msg)
            
        except Exception as e:
            rospy.logerr(f"Error in transforming point cloud: {e}")
        finally:
            self.transforming = False

    def imu_callback(self, data:Imu):
        q = data.orientation
        imu_quaternion = [q.x, q.y, q.z, q.w]

        # Convert IMU quaternion to rotation object
        imu_rotation = R.from_quat(imu_quaternion)

        # Apply -90 degree rotation along the Z-axis (yaw)
        z_rotation = R.from_euler('z', -90, degrees=True)  # -90 degree around Z-axis
        
        transformed_rotation = z_rotation * imu_rotation  # Compose the rotations

        # Convert the transformed rotation to Euler angles (roll, pitch, yaw)
        transformed_euler_xyz = transformed_rotation.as_euler('xyz', degrees=False)
        transformed_euler_xyz[2] = transformed_euler_xyz[2] - (0.00000441403) * self.times
        self.times = self.times + 1

        curr_quaternion = QuaternionStamped() 
        curr_quaternion.header = data.header

        curr_quaternion.quaternion = Quaternion(
            *tf_conversions.transformations.quaternion_from_euler(
                transformed_euler_xyz[0],
                transformed_euler_xyz[1],
                transformed_euler_xyz[2]))
        self.imu_cache.append(curr_quaternion)
        # print(len(self.imu_cache))
        self.latest_imu_quaternion = curr_quaternion  # Update latest IMU orientation
        self.quaternion_pub.publish(curr_quaternion.quaternion)

        # Clean up old IMU data
        if len(self.imu_cache) > 100:
            current_time = curr_quaternion.header.stamp
            self.imu_cache = [
                imu for imu in self.imu_cache
                if (current_time - imu.header.stamp).to_sec() <= self.expiry_time
            ]
            rospy.loginfo("Cleaned IMU cache")
            # for i in range(50):
            #     self.imu_cache.pop()

        # print((rospy.Time.now() - data.header.stamp).to_sec())
    
    def camera_info_listener(self, data: CameraInfo):
        rospy.loginfo("Camera Info Received.")
        self.color_camera_info = data
        self.visual_merged_msg.CameraInfo = data
        self.image_info_sub.unregister()
        
    def run(self, rate = 50):

        while not rospy.is_shutdown():
            rospy.Rate(rate).sleep()
        else:
            rospy.loginfo("Shutting Down Camera Logger.")
            quit()

def main():
    CPF = CameraPosefuser()
    CPF.run()

if __name__ == "__main__":
    main()
    # . /mnt/Data/gaussian-splatting-slam/submodules/ros_workspace/devel/setup.bash 
    # rosrun gs_slam_msgs camera_info_real_env_optimized.py _initial_heading:=30.0
    # rostopic hz /imu/data /rtk_gps_pos /camera/color/image_raw /camera/depth/color/points 

    #             topic               rate   min_delta   max_delta   std_dev    window
    # ==============================================================================
    # /imu/data                    99.98   1.502e-05   0.03001     0.004789   3107  
    # /rtk_gps_pos                 10.0    0.002208    0.1975      0.03117    3107  
    # /camera/color/image_raw      29.12   0.001945    0.09617     0.01767    311   
    # /camera/depth/color/points   22.41   0.02743     0.182       0.021      311   


