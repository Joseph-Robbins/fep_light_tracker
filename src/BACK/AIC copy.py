import rospy
import numpy  as np
from numpy.random import default_rng
from sensor_msgs.msg import Illuminance, JointState
from std_msgs.msg import Float64

class AIC():
    def __init__(self):
        # rospy.Subscriber('/light_sensor_plugin/lightSensor', Illuminance, self.sensor_callback)
        rospy.Subscriber('/joint_states', JointState, self.joint_sensor_callback)
        self.pub = rospy.Publisher('/joint1_controller/command', Float64, queue_size=1)
        
        self.rng = default_rng()

        self.data_received = False

        self.dt = 0.1 # Timestep
        self.T = 10  # Length of simulation
        self.k_mu = 1 #1.167 # Learning rate of the gradient descent on the belief about our state (mu)
        self.k_u = 0.001 #70 # Learning rate of action
        
        self.sig_z = 1 # Variance in state estimation noise
        self.sig_w = 1 # Variance in sensor estimation noise

        self.x = 0 # State - joint pos
        self.mu_x = 0 # Belief of position
        self.mu_v = 0 # Prior belief of position
        self.y = 0 # Sensor reading
        self.mu_y = 0 # Belief about sensor reading
        self.ypred = 0 # Predicted sensor reading
        self.u = 0 # Control - joint torque
        # x = np.array([0, 0, 0]).T # State - jointpos, jointvel, jointtorque

    def sensor_callback(self, msg):
        rospy.loginfo(f"/Illuminance = {msg.illuminance}")
        self.y = msg.illuminance
        self.data_received = True

    def joint_sensor_callback(self, msg):
        self.y = msg.position[0]
        # print(f"jointpos = {self.y}")
        self.data_received = True

    def minimise_f(self):
        # Make a prediction about the sensor reading, to compare with the real sensor reading in the next step
        self.ypred = self.g(self.mu_x)
        # Predict where the robot will be next, given the current state and the control command
        self.xpred = self.f(self.mu_x, self.u)

        # Calcuate sensory and motion prediction error
        SPE = self.y - self.ypred
        print(f"predicted: {self.ypred}  value: {self.y}")
        MPE = self.mu_x - self.xpred

        # Calcuate free energy
        F = SPE**2/(2*self.sig_z**2) + MPE**2/(2*self.sig_w**2) + np.log(self.sig_z**2 * self.sig_w**2)/2 #+ (self.mu_x - 0)**2
        print(f"F = {F}, SPE = {SPE}, MPE = {MPE}")
        
        # Calculate the derivative of free energy with respect mu_x
        dF = MPE/self.sig_w**2 + SPE/self.sig_z**2
    
        # Use gradient descent to find mu for which F is lowest, over time
        dmu_x = 0 - self.k_mu * dF
        self.mu_x = self.mu_x + dmu_x * self.dt
        # self.mu_x

        # Act
        self.du = -self.k_u * SPE * self.sig_z**-1
        self.u = self.u + self.du * self.dt

        # if dF > 0:
        #     self.u = 1
        # else:
        #     self.u = -1

        print(f"u = {self.u}")
        self.pub.publish(self.u)
        

    #--Generative Model--#
    def f(self, x, u): # Function of motion (motion model) - p(x)
        # noise = self.rng.standard_normal(0, self.sig_w, np.int(np.round(self.T/self.dt)))
        noise = self.sig_w * self.rng.standard_normal()
        # angvel = u * self.dt
        # xpred = x + angvel * self.dt
        xpred = 1 - x
        return xpred + noise

    def g(self, x): # Function of sensory mapping (measurement model) - p(y | x)
        # noise = self.rng.standard_normal(0, self.sig_z, np.int(np.round(self.T/self.dt)))
        noise = self.sig_z * self.rng.standard_normal()
        ypred = x
        return ypred + noise
        # return self.y + noise

    def df(self, x, u):
        return 1

    def dg(self, x): # Derivative of g
        return 1 # Derivative of x + c = 1
