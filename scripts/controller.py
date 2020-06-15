#!/usr/bin/env python

from __future__ import print_function

import sys

import cv2
import rospy
from cv_bridge import CvBridge
import message_filters
from smart_camera.srv import *
from smart_camera.msg import IntWithHeader
from sensor_msgs.msg import Image

values = []

def callback(image, _class):
    bridge = CvBridge()
    _image = bridge.imgmsg_to_cv2(image)

    values.append({ _class.data: _image })

    rospy.wait_for_service('ai')
    try:
        classify = rospy.ServiceProxy('ai', Ai)
        response = classify(image)
        rospy.loginfo("Service response: %s"%response.result)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

    rospy.loginfo(rospy.get_caller_id() + " Reciving -> %s", { _class.data: _image })
    
def listener():
    rospy.init_node('controller')

    image_sub = message_filters.Subscriber("/processed/image", Image)
    class_sub = message_filters.Subscriber("/camera/class", IntWithHeader)

    ts = message_filters.TimeSynchronizer([image_sub, class_sub], 10)
    ts.registerCallback(callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
