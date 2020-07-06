#!/usr/bin/env python

import cv2
import rospy
from cv_bridge import CvBridge
from std_msgs.msg import String, Int32
from sensor_msgs.msg import Image

def callback(data):
    bridge = CvBridge()
    image = bridge.imgmsg_to_cv2(data, "mono8")
    rospy.loginfo(image)
    # resized = cv2.resize(image, None, fx=0.9, fy=0.9)
    # rospy.loginfo(rospy.get_caller_id() + " Reciving -> %s", resized.shape)
    image = bridge.cv2_to_imgmsg(image)
    processed_image_talker(image)
    
def listener():
    rospy.init_node('processor', anonymous=True)

    rospy.Subscriber("/camera/image", Image, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

def processed_image_talker(image):
    processed_image_pub = rospy.Publisher('/processed/image', Image, queue_size=1024)
    rate = rospy.Rate(0.5)

    while not rospy.is_shutdown():
        # rospy.loginfo(" Processed image -> %s", image)
        processed_image_pub.publish(image)
        rate.sleep()

if __name__ == '__main__':
    listener()
