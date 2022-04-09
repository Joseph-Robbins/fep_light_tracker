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

        self.dt = 0.001   # Timestep
        self.k_mu = 1.67  # Learning rate of the gradient descent on the belief about our state (mu)
        self.k_u = 70     # Learning rate of action
        self.mu_d = 0 # Desired position (prior belief about position)
        
        self.sig_z  = 0.1 # Variance in state estimation noise
        self.sig_zp = 0.1 # Variance in state' estimation noise
        self.sig_w  = 0.1 # Variance in sensor estimation noise
        self.sig_wp = 0.1 # Variance in sensor' estimation noise

        self.PI_z = 1/self.sig_z # Precision in state estimation noise
        self.PI_w = 1/self.sig_w # Precision in sensor estimation noise

        self.mu_x = 0    # Belief of position
        self.mu_xp = 0   # Derivative of mu_x (belief about velocity)
        self.mu_xpp = 0  # Derivative of mu_xp (belief about acceleration)

        self.y = 0     # Sensor reading
        self.yp = 0    # Derivative of sensor reading
        self.ypp = 0   # Second derivative of sensor reading
        self.y_old = 0 # Past sensor reading

        self.u = 0    # Control - joint torque

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
        eps_y = self.y - self.g(self.mu_x)
        eps_yp = self.yp - self.dg(self.mu_xp)
        eps_mu = self.mu_xp - self.f(self.mu_x)
        eps_mup = self.mu_xpp - self.df(self.mu_xp)

        # Calculate precision-weighted prediction errors
        zet_y = self.PI_z * eps_y
        zet_yp = self.PI_z * eps_yp
        zet_mu = self.PI_w * eps_mu
        zet_mup = self.PI_w * eps_mup

        # Calcuate free energy using precision-weighted squared sensory and motion prediction errors
        F = 1/2 * (eps_y**2*self.PI_z + eps_yp**2*self.PI_z + eps_mu*2*self.PI_w + eps_mup**2*self.PI_w)
        print(f"F = {F}, SPE = {eps_y}, MPE = {eps_mu}")
        
        # Calculate the derivative of free energy with respect mu_x
        # dF = MPE/self.sig_w**2 + SPE/self.sig_z**2
    
        # Calculate gradients for mu_x, mu_xp and mu_xpp by calculating the gradient of F w.r.t. mu
        dmu_x = self.mu_xp - self.k_mu*(-zet_y + zet_mu)                # Gradient of belief about position (mu_x)
        dmu_xp = self.mu_xpp - self.k_mu * (-zet_yp + zet_mu + zet_mup) # Gradient of mu_xp
        dmu_xpp = -self.k_mu * (zet_mup)                                # Gradient of mu_xpp

        # Descend one step down the gradient
        self.mu_x = self.mu_x + dmu_x * self.dt
        self.mu_xp = self.mu_xp + dmu_xp * self.dt
        self.mu_xpp = self.mu_xpp + dmu_xpp * self.dt

        # Use gradient descent to act
        self.du = -self.k_u * (zet_yp + zet_y)
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
        xpred = (self.mu_d) - x
        return xpred #+ noise

    def g(self, x): # Function of sensory mapping (measurement model) - p(y | x)
        # noise = self.rng.standard_normal(0, self.sig_z, np.int(np.round(self.T/self.dt)))
        noise = self.sig_z * self.rng.standard_normal()
        ypred = x
        return ypred #+ noise
        # return self.y + noise

    def df(self, xp):
        return -xp

    def dg(self, xp): # Derivative of g
        return xp # Derivative of x + c = 1