import rospy
from sensor_msgs.msg import Image


rospy.init_node("gaussian_splattingg_slam",anonymous=False)
print(rospy.wait_for_message(topic="/camera/rgb/image_color",topic_type=Image, timeout=1))