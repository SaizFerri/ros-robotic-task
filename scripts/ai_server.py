#!/usr/bin/env python

from __future__ import print_function

from smart_camera.srv import Ai, AiResponse
import rospy

def handle_request(req):
    response = 0
    print("Returning class %s"%response)
    return AiResponse(response)

def ai_server():
    rospy.init_node('ai_server')
    s = rospy.Service('ai', Ai, handle_request)
    print("Ready to classify an image.")
    rospy.spin()

if __name__ == "__main__":
    ai_server()