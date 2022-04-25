#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Illuminance
from std_msgs.msg import Float64
from AIC import AIC
import cv2

def ain_control(gradient):
    rospy.init_node("AIN_Controller")

    controller = AIC(gradient)

    count = 0
    rate = rospy.Rate(1000)
    while not rospy.is_shutdown():
        # Skip the first cycle so that we only move once we have sensory data, and only cycle once we have data
        if (count != 0) & (controller.data_received) & (controller.g_data_received) :
            # print("beep")
            # rospy.spin()
            controller.minimise_f()
            # input()

        else:
            count += 1

        rate.sleep()

if __name__ == '__main__':
    try:
        print("loading gradient")
        g = np.load(open("/home/joseph/catkin_ws/src/fep_light_tracker/src/gradient.npy", "rb"))
        print("starting controller")
        ain_control(g)
    except rospy.ROSInterruptException:
        pass