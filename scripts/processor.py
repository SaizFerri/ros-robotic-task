#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Int32

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + " Reciving -> %s", data.data)
    processed_image_talker(data.data)
    
def listener():
    rospy.init_node('processor')

    rospy.Subscriber("image", String, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

def processed_image_talker(image):
    processed_image_pub = rospy.Publisher('processed_image', String, queue_size=5)
    rate = rospy.Rate(0.5)

    while not rospy.is_shutdown():
        rospy.loginfo(" Processed image -> %s", image)
        processed_image_pub.publish(image)
        rate.sleep()

if __name__ == '__main__':
    listener()
