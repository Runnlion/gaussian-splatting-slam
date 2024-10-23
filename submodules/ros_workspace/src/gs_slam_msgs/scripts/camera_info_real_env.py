import rospy, tf_conversions
from geometry_msgs.msg import PointStamped, QuaternionStamped, Quaternion
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, Imu

from scipy.spatial.transform import Rotation as R
import numpy as np

class CameraPosefuser():

    def __init__(self) -> None:

        self.times = 0
        self.color_camera_info = CameraInfo()
        self.previous = [0,0,0]
        self.diff_stored = []

        self.imu_cache:list[QuaternionStamped] = list()
        # Expiry time in seconds
        self.expiry_time = 0.05
        self.curr_quaternion = QuaternionStamped() 
        self.gps_topic = "/rtk_gps_pos"
        self.imu_topic = "/imu/data"
        rospy.init_node("camera_pose_listener",anonymous=False)
        rospy.Subscriber(self.gps_topic,PointStamped, self.gps_position_callback)
        rospy.Subscriber(self.imu_topic, Imu, self.imu_callback)
        rospy.loginfo(f"Start listening {self.gps_topic}.")
        rospy.loginfo(f"Start listening {self.imu_topic}.")
        # Subscriber for CameraInfo
        self.image_info_sub = rospy.Subscriber(
            'camera/rgb/camera_info',
            CameraInfo,
            self.camera_info_listener,
            queue_size=1
        )
        # Subscribers for Image and TF topics
        self.image_sub = rospy.Subscriber(
            '/camera/rgb/image_color',
            Image,
            self.image_listener,
            queue_size=10
        )


    def image_listener(self, img_msg: Image):
        self.pair_image_Pose(img_msg)

    def gps_position_callback(self,data:PointStamped):
        rospy.loginfo("GPS Received")
        self.pair_gps_imu(data)

    def pair_image_Pose(self, image:Image):

        pass
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
                rospy.loginfo(f"Pair found with time difference: {time_diff}s")
                # self.visual_merged_msg.CameraPose = tf
                # self.visual_merged_msg.Image = img
                # self.visual_merged_pub.publish(self.visual_merged_msg)
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
        transformed_euler_xyz = transformed_rotation.as_euler('xyz', degrees=True)
        transformed_euler_xyz[2] = transformed_euler_xyz[2] - (0.00025292437722087153) * self.times
        self.times = self.times + 1

        # diff = transformed_euler_xyz[2] - self.previous[2]
        # print(transformed_euler_xyz)
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
        # self.visual_merged_msg.CameraInfo = data
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