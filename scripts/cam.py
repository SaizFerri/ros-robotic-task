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
    rospy.init_node('cam', anonymous=True)
    bridge = CvBridge()
    image_pub = rospy.Publisher('/camera/image', Image, queue_size=1)
    int_pub = rospy.Publisher('/camera/class', IntWithHeader, queue_size=1)
    rate = rospy.Rate(0.5)

    while not rospy.is_shutdown():
        h = Header()
        number = random.randint(0, 9)

        img = read_image(number)
        img_msg = bridge.cv2_to_imgmsg(img, encoding="rgb8")
        image_pub.publish(img_msg)

        int_pub.publish(h, number)
        
        rate.sleep()

def read_image(num):
    return cv2.imread(os.path.dirname(os.path.abspath(__file__)) + "/%i.png"%num)

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
