import rospy, tf_conversions
from geometry_msgs.msg import PointStamped, QuaternionStamped, Quaternion, PoseStamped, TransformStamped
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, Imu
from gs_slam_msgs.msg import visual_merged_msg
from scipy.spatial.transform import Rotation as R
import numpy as np

class CameraPosefuser():

    def __init__(self) -> None:

        self.times = 0
        self.color_camera_info = CameraInfo()
        self.previous = [0,0,0]
        self.diff_stored = []
        self.visual_merged_msg = visual_merged_msg()
        self.visual_merged_msg.Local_Map = PointCloud2()

        self.camera_pose_cache:list[PoseStamped] = list()
        self.imu_cache:list[QuaternionStamped] = list()
        self.pc2_cache:list[PointCloud2] = list()

        # Expiry time in seconds
        self.expiry_time = 0.05
        self.curr_quaternion = QuaternionStamped() 
        self.gps_topic = "/rtk_gps_pos"
        self.imu_topic = "/imu/data"

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
            queue_size=10
        )

        self.pointcloud_sub = rospy.Subscriber(
            '/camera/depth/points',
            PointCloud2,
            self.pointcloud_callback,
            queue_size=1
        )

    def pointcloud_callback(self, pc2:PointCloud2):
        # Save frist and then match.
        self.pc2_cache.append(pc2)

    def image_callback(self, img_msg: Image):
        # rospy.loginfo("Image Received.")
        self.pair_image_Pose(img_msg)

    def gps_position_callback(self,gps_position_msg:PointStamped):
        self.pair_gps_imu(gps_position_msg)

    def pair_image_Pose(self, img_msg:Image)->bool:
        if(len(self.pc2_cache)==0):
            rospy.loginfo("No Pointcloud Received.")
        current_time = img_msg.header.stamp
        self.camera_pose_cache = [
            pose for pose in self.camera_pose_cache
            if (current_time - pose.header.stamp).to_sec() <= 0.2
        ]
        self.pc2_cache = [
            pc for pc in self.pc2_cache
            if (current_time - pc.header.stamp).to_sec() <= 0.2
        ]
        image_paired = False
        # Attempt to find matching pairs
        for i in range(len(self.camera_pose_cache)):
            pose = self.camera_pose_cache.pop() 
            time_diff = abs((img_msg.header.stamp - pose.header.stamp).to_sec())
            print(time_diff)

            if time_diff <= 0.2:
                rospy.loginfo(f"[Image + Pose] Pair found with time difference: {time_diff}s")

                # Align Timestamp and publish
                ts = TransformStamped()
                # ts.header = img_msg.header
                ts.child_frame_id = "base"
                ts.transform.rotation = pose.pose.orientation
                ts.transform.translation = pose.pose.position
                self.visual_merged_msg.CameraPose = ts
                self.visual_merged_msg.CameraInfo = self.color_camera_info
                self.visual_merged_msg.Image = img_msg
                image_paired = True
                self.camera_pose_cache.clear()
                break
        pointcloud_paired = False
        if(image_paired == True):
            for i in range(len(self.pc2_cache)):
                pc = self.pc2_cache.pop() 
                time_diff_img_pc = abs((self.visual_merged_msg.Image.header.stamp - pc.header.stamp).to_sec())
                time_diff_pose_pc = abs((self.visual_merged_msg.CameraPose.header.stamp - pc.header.stamp).to_sec())
                print(time_diff)

                if (time_diff_pose_pc) <= 0.2 or (time_diff_img_pc) <= 0.2:
                    
                    rospy.loginfo(f"{"Point Cloud Close to Pose Stamp" if (time_diff_pose_pc) <= 0.2 else "Point Cloud Close to Image Stamp"}")
                    # convert the pointclud in train program
                    self.visual_merged_msg.Local_Map = pc
                    self.visual_merged_pub.publish(self.visual_merged_msg)
                    self.pc2_cache.clear()
                    pointcloud_paired = True
                    break
        return (image_paired and pointcloud_paired)
    
    def pair_gps_imu(self, point:PointStamped):
        current_time = point.header.stamp
        self.imu_cache = [
            imu for imu in self.imu_cache
            if (current_time - imu.header.stamp).to_sec() <= self.expiry_time
        ]

        # Attempt to find matching pairs
        for i in range(len(self.imu_cache)):
            imu = self.imu_cache.pop() 
            time_diff = abs((point.header.stamp - imu.header.stamp).to_sec())
            if time_diff <= self.expiry_time:
                ps = PoseStamped()
                ps.header = imu.header
                ps.pose.position = point.point
                ps.pose.orientation = imu.quaternion
                self.pose_pub.publish(ps)
                self.camera_pose_cache.append(ps)                    
                break  # Proceed to next imu

        # If the frist tf is not latest, clear all.
        self.imu_cache.clear()
    
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

        # diff = transformed_euler_xyz[2] - self.previous[2]
        self.curr_quaternion.header = data.header

        self.curr_quaternion.quaternion = Quaternion(
            *tf_conversions.transformations.quaternion_from_euler(
                transformed_euler_xyz[0],
                transformed_euler_xyz[1],
                transformed_euler_xyz[2]))
        self.imu_cache.append(self.curr_quaternion)
        # self.diff_stored.append(diff)
        # self.previous = transformed_euler_xyz
        # rospy.logdebug_throttle(1,"Receiving IMU data")
        # print(len(self.imu_cache))
        # print(f"rad = {transformed_euler_xyz}")
        self.quaternion_pub.publish(self.curr_quaternion.quaternion)

        if(len(self.imu_cache) > 100):
            current_time = self.curr_quaternion.header.stamp
            self.imu_cache = [
                imu for imu in self.imu_cache
                if (current_time - imu.header.stamp).to_sec() <= self.expiry_time
            ]
            print("Cleaned IMU cache")
            self.imu_cache.clear()

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
            # print(sum(self.diff_stored))
            # print(len(self.diff_stored))
            # print(f"Mean error = {sum(self.diff_stored)/len(self.diff_stored)}")
            quit()

def main():
    CPF = CameraPosefuser()
    CPF.run()

    pass

if __name__ == "__main__":
    main()