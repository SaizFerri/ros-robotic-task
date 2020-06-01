#!/usr/bin/env python

import os
import cv2
import rospy
import random
import numpy as np
from std_msgs.msg import String, Int32, Header
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from smart_camera.msg import IntWithHeader

def talker():
    img = cv2.imread(os.path.dirname(os.path.abspath(__file__)) + "/0.png")
    bridge = CvBridge()
    img_msg = bridge.cv2_to_imgmsg(img, encoding="rgb8")
    image_pub = rospy.Publisher('/camera/image', Image, queue_size=5)
    int_pub = rospy.Publisher('/camera/class', IntWithHeader, queue_size=5)
    rospy.init_node('cam')
    rate = rospy.Rate(0.5)

    while not rospy.is_shutdown():
        h = Header()
        cam_str = "cam %s" % rospy.get_time()
        # int_value = random.randint(1, 20)
        # rospy.loginfo(cam_str)
        # rospy.loginfo(int_value)
        image_pub.publish(img_msg)
        int_pub.publish(h, 0)
        rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
