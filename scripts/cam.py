#!/usr/bin/env python

import rospy
import random
import numpy as np
from std_msgs.msg import String, Int32

def talker():
    image_pub = rospy.Publisher('image', String, queue_size=5)
    int_pub = rospy.Publisher('number', Int32, queue_size=5)
    rospy.init_node('cam', anonymous=True)
    rate = rospy.Rate(0.5)

    while not rospy.is_shutdown():
        cam_str = "cam %s" % rospy.get_time()
        int_value = random.randint(1, 20)
        rospy.loginfo(cam_str)
        rospy.loginfo(int_value)
        image_pub.publish(cam_str)
        int_pub.publish(int_value)
        rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
