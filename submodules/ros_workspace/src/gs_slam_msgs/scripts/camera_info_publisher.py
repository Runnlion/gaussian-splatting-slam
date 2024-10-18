import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import TransformStamped
from tf.msg import tfMessage
import tf2_ros
import time
from gs_slam_msgs.msg import visual_merged_msg
class Visual_Data_Merger():
    def __init__(self, node_name: str):
        rospy.init_node(node_name, anonymous=False)
        rospy.loginfo("ROS Node Established.")
        self.visual_merged_pub = rospy.Publisher('/Visual_Merged',visual_merged_msg,queue_size=1)
        self.visual_merged_msg = visual_merged_msg()
        self.visual_merged_msg.Local_Map = PointCloud2()

        # Subscriber for CameraInfo
        self.image_info_sub = rospy.Subscriber(
            'camera/rgb/camera_info',
            CameraInfo,
            self.camera_info_listener,
            queue_size=1
        )
        self.color_camera_info = CameraInfo()

        # Subscribers for Image and TF topics
        self.image_sub = rospy.Subscriber(
            '/camera/rgb/image_color',
            Image,
            self.image_listener,
            queue_size=10
        )

        self.tf_sub = rospy.Subscriber(
            '/tf',
            tfMessage,
            self.tf_listener_callback,
            queue_size=1
        )
        # TF listener setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Caches for images and transforms
        self.tf_cache : list[TransformStamped] = list()

        # Expiry time in seconds
        self.expiry_time = 0.05

    def camera_info_listener(self, data: CameraInfo):
        rospy.loginfo("Camera Info Received.")
        self.color_camera_info = data
        self.visual_merged_msg.CameraInfo = data
        self.image_info_sub.unregister()

    def image_listener(self, img_msg: Image):
        self.pair_messages(img_msg)

    def tf_listener_callback(self,data:tfMessage):
        # This part is only written for TUM dataset
        if(len(data.transforms)==2 and data.transforms[1].header.frame_id == '/world' and data.transforms[1].child_frame_id == '/kinect'):
            self.tf_cache.append(data.transforms[1])
            # rospy.loginfo("TF message received")
        

    def pair_messages(self, img:Image):
        current_time = img.header.stamp
        self.tf_cache = [
            tf for tf in self.tf_cache
            if (current_time - tf.header.stamp).to_sec() <= self.expiry_time
        ]
        # Attempt to find matching pairs
        for i in range(len(self.tf_cache)):
            tf = self.tf_cache.pop() 
            time_diff = abs((img.header.stamp - tf.header.stamp).to_sec())
            if time_diff <= self.expiry_time:
                rospy.loginfo(f"Pair found with time difference: {time_diff}s")
                self.visual_merged_msg.CameraPose = tf
                self.visual_merged_msg.Image = img
                self.visual_merged_pub.publish(self.visual_merged_msg)
                break  # Proceed to next image
            # If the frist tf is not latest, clear all.
        self.tf_cache.clear()
        

    def run(self):
        rate = rospy.Rate(100)  # 20 Hz
        while not rospy.is_shutdown():
            rate.sleep()

def main():
    vdm = Visual_Data_Merger("visual_data_merger")
    vdm.run()

if __name__ == "__main__":
    main()
