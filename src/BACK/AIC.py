import rospy
import numpy  as np
from sensor_msgs.msg import Illuminance
from std_msgs.msg import Float64

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + f"Illuminance = {data.illuminance}")

class AIC():
    rospy.Subscriber('/light_sensor_plugin/lightSensor', Illuminance, callback)
    pub = rospy.Publisher('/joint1_position_controller/command', Float64)

    # Support variable
    dataReceived = 0

    # Variances associated with the beliefs and the sensory inputs
    var_mu = 5.0
    var_muprime = 10.0
    var_q = 1
    var_qdot = 1

    # Learning rates for the gradient descent (found that a ratio of 60 works good)
    k_mu = 11.67
    k_a = 700

    # Precision matrices
    SigmaP_yq0 = 1 / var_q
    SigmaP_yq1 = 1 / var_qdot
    SigmaP_mu = 1 / var_mu
    SigmaP_muprime = 1 / var_muprime

    # Initialize control actions
    u = 0.0

    # Initialize prior beliefs about the second ordet derivatives of the states of the robot
    mu_pp = 0.0

    # Integration step
    h = 0.001

