#!/usr/bin/env python

import rospy
import tf
from geometry_msgs.msg import PoseStamped

def pose_callback(msg:PoseStamped):
    # Initialize the TF broadcaster
    br = tf.TransformBroadcaster()

    # Extract position and orientation from PoseStamped message
    position = msg.pose.position
    orientation = msg.pose.orientation

    # Publish transform from parent frame ",ap" to the child frame (from msg.header.frame_id)
    br.sendTransform(
        (position.x, position.y, position.z),  # Translation (x, y, z)
        (orientation.x, orientation.y, orientation.z, orientation.w),  # Rotation (quaternion)
        msg.header.stamp,  # Timestamp
        msg.header.frame_id,  # Child frame (from the message)
        "map"  # Parent frame
    )

if __name__ == '__main__':
    # Initialize ROS node
    rospy.init_node('pose_to_tf_broadcaster')

    # Subscribe to the PoseStamped topic
    rospy.Subscriber('/Camera_Pose', PoseStamped, pose_callback)

    # Keep the node alive
    rospy.spin()
