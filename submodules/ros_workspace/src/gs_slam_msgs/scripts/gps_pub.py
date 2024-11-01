import csv
import datetime
from sbp.client.drivers.pyserial_driver import PySerialDriver
from sbp.client import Handler, Framer
from sbp.navigation import SBP_MSG_BASELINE_NED, SBP_MSG_VEL_NED, MsgVelNED, MsgBaselineHeadingDepA,MsgBaselineNED, SBP_MSG_BASELINE_HEADING_DEP_A
import argparse
import rospy
from geometry_msgs.msg import PointStamped

def main():
    parser = argparse.ArgumentParser(
        description="Swift Navigation SBP NED.")
    parser.add_argument(
        "-p",
        "--port",
        default=['/dev/ttyUSB0'],
        nargs=1,
        help="specify the serial port to use.")
    args = parser.parse_args()
    rospy.init_node('camera_gps_pose',anonymous=False)

    ps = PointStamped()
    ps.header.frame_id = "gps_antenna"
    ps.header.seq = -1
    gps_pos_pub = rospy.Publisher('/rtk_gps_pos', PointStamped, queue_size=1)
    with open('baseline_ned.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        csv_writer.writerow(['TS', 'X', 'Y', 'Z'])

        # Open a connection to Piksi using the default baud rate (1Mbaud)
        with PySerialDriver(args.port[0], baud=115200) as driver:
            with Handler(Framer(driver.read, None, verbose=True)) as source:
                try:
                    # print(source)
                    # Think, add the tf here or directly convert?
                    # directly convert for fast implementation.
                    for msg, metadata in source.filter([SBP_MSG_BASELINE_NED,SBP_MSG_VEL_NED]):
                        if(type(msg) == MsgBaselineNED):
                            ps.header.stamp = rospy.Time.now()
                            ps.point.x = msg.e * 1e-3
                            ps.point.y = msg.n * 1e-3
                            ps.point.z = -1.0 * msg.d * 1e-3
                            ps.header.seq = ps.header.seq + 1
                            gps_pos_pub.publish(ps)
                            rospy.loginfo(f"position X: {ps.point.x}, Y: {ps.point.y}, Z:{ps.point.z}")
                            # start position X: -9.304, Y: 1.219, Z:0.483
                            # end position X: -11.695, Y: -3.091, Z:0.434
                            # angle = 119 Degree
                        # csv_writer.writerow([ps.header.stamp.to_nsec(), ps.point.x, ps.point.y, ps.point.z])
                    
                except KeyboardInterrupt:
                    rospy.on_shutdown(exit())
                    pass

if __name__ == "__main__":
    main()