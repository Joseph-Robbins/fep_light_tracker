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
        self.pub_f = rospy.Publisher('/joebot/free_energy', Float64, queue_size=1)
        
        self.rng = default_rng()

        self.data_received = False

        self.dt = 0.001 # Timestep
        self.T = 10  # Length of simulation
        self.k_mu = 1 #1.167 # Learning rate of the gradient descent on the belief about our state (mu)
        self.k_u = 70 #70 # Learning rate of action
        
        self.sig_z = 1 # Variance in state estimation noise
        self.sig_w = 1 # Variance in sensor estimation noise

        self.x = 0 # State - joint pos

        self.mu_x = 0    # Belief of position
        self.mu_xp = 0   # Derivative of mu_x (belief about velocity)
        self.mu_xpp = 0  # Derivative of mu_xp (belief about acceleration)
        self.dmu_x = 0   # Gradient of belief about position (mu_x)
        self.dmu_xp = 0  # Gradient of mu_xp
        self.dmu_xpp = 0 # Gradient of mu_xpp

        self.y = 0 # Sensor reading
        self.yp = 0
        self.ypp = 0
        self.y_old = 0 # Past sensor reading

        self.mu_v = 0 # Prior belief of position
        self.mu_y = 0 # Belief about sensor reading
        self.ypred = 0 # Predicted sensor reading
        self.u = 0 # Control - joint torque

    def sensor_callback(self, msg):
        rospy.loginfo(f"/Illuminance = {msg.illuminance}")
        self.y = msg.illuminance
        self.data_received = True

    def joint_sensor_callback(self, msg):
        self.y_old = self.y
        self.y = msg.position[0]
        # self.yp = msg.velocity[0]
        self.yp = (self.y - self.y_old)/self.dt
        # print(f"jointpos = {self.y}")
        self.data_received = True

    def minimise_f(self):
        # Make a prediction about the sensor reading, to compare with the real sensor reading in the next step
        self.ypred = self.g(self.mu_x)
        # Predict where the robot will be next, given the current state and the control command
        self.xpred = self.f(self.mu_x)

        eps_y = self.y - self.g(self.mu_x)
        eps_mu = self.mu_xp - self.f(self.mu_x)

        # Calcuate precision-wighted squared sensory and motion prediction errors
        SPE_0 = eps_y**2*self.sig_z**-1
        MPE_0 = eps_mu*2*self.sig_w**-1

        # Calcuate free energy
        F = 1/2 * (SPE_0 + MPE_0)
        print(f"F = {F}, SPE = {SPE_0}, MPE = {MPE_0}")
        
        # Calculate the derivative of free energy with respect mu_x
        # dF = MPE/self.sig_w**2 + SPE/self.sig_z**2
    
        # Calculate gradients for mu_x, mu_xp and mu_xpp by calculating the gradient of F w.r.t. mu
        dmu_x = self.mu_xp - self.k_mu*(-self.sig_z**-1 * eps_y + self.sig_w**-1 * eps_mu)

        # Descend one step down the gradient
        self.mu_x = self.mu_x + dmu_x * self.dt

        # Use gradient descent to act
        self.du = -self.k_u * (self.sig_z**-1 * eps_y)
        self.u = self.u + self.du * self.dt
        # self.u = np.clip(self.u, -0.5, 0.5)

        # print(f"u = {self.u}")
        self.pub.publish(self.u)
        self.pub_f.publish(F)
        

    #--Generative Model--#
    def f(self, x): # Function of motion (motion model) - p(x)
        # noise = self.rng.standard_normal(0, self.sig_w, np.int(np.round(self.T/self.dt)))
        noise = self.sig_w * self.rng.standard_normal()
        # angvel = u * self.dt
        # xpred = x + angvel * self.dt
        xpred = (np.pi) - x
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
