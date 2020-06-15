#!/usr/bin/env python

from __future__ import print_function

import sys
import rospy
from smart_camera.srv import *

def ai_client(image):
    rospy.wait_for_service('ai')
    try:
        classify = rospy.ServiceProxy('ai', Ai)
        resp1 = classify(image)
        return resp1.result
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def usage():
    return "Send a sensor_msgs/Image"

if __name__ == "__main__":
    ai_client(image)