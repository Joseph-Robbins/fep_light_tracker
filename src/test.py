#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Illuminance
from std_msgs.msg import Float64

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + f"Illuminance = {data.illuminance}")

def move_head():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber('/light_sensor_plugin/lightSensor', Illuminance, callback)
    pub = rospy.Publisher('/joint1_controller/command', Float64)

    angle = 0

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        angle = round(angle + 0.1, 1)
        if angle == 3.1:
            angle = -3.1
        rospy.loginfo(f"angle = {angle}")
        pub.publish(angle)

        rate.sleep()

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    move_head()