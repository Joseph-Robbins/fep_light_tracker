from dataclasses import dataclass
import rospy
import numpy  as np
from numpy.random import default_rng
from sensor_msgs.msg import Illuminance, JointState
from std_msgs.msg import Float64

@dataclass
class state:
    theta = 0
    direction = 0

class AIC():
    def __init__(self, m):
        rospy.Subscriber('/light_sensor_plugin/lightSensor', Illuminance, self.sensor_callback)
        rospy.Subscriber('/joint_states', JointState, self.joint_sensor_callback)
        self.pub = rospy.Publisher('/joint1_controller/command', Float64, queue_size=1)
        self.pub_f = rospy.Publisher('/joebot/free_energy', Float64, queue_size=1)
        
        self.rng = default_rng()

        self.data_received = False
        self.g_data_received = False

        self.dt = 0.001 # Timestep
        self.k_mu = 1 #1.167 # Learning rate of the gradient descent on the belief about our state (mu)
        self.k_u = 70 #70 # Learning rate of action
        self.m = m # Initialise map

        ind = np.where(self.m == self.m.max())[0][0]
        self.v = (ind - self.m.size/2) * (np.pi / (self.m.size/2)) # Prior belief of position
        # self.v = np.pi
        # self.v = np.interp(self.v, np.linspace(0, 2*np.pi, self.m.size), range(255))

        self.sig_z = 1 # Variance in sensor estimation noise
        self.sig_w = 0.1 # Variance in state estimation noise

        self.PI_z = 1/self.sig_z # Precision in state estimation noise
        self.PI_w = 1/self.sig_w # Precision in sensor estimation noise

        self.mu = state()
        self.mu_p = state()    

        self.y = 0 # Sensor reading
        self.y_old = 0 # Past sensor reading

        self.u = 0 # Control - joint torque

    def sensor_callback(self, msg):
        # rospy.loginfo(f"/Illuminance = {msg.illuminance}")
        self.y_old = self.y
        self.y = msg.illuminance
        # self.yp = (self.y - self.y_old) / self.dt
        self.data_received = True

    def joint_sensor_callback(self, msg):
        # print(f"jointpos = {self.y}")
        self.g_truth_x = msg.position[0]
        self.g_truth_xp = msg.velocity[0]

        # self.y_old = self.y
        # self.y = self.g_truth_x
        # # self.yp = self.g_truth_xp
        # self.yp = (self.y - self.y_old) / self.dt
        self.g_data_received = True

    def minimise_f(self):
        # eps_y = self.y - self.g_gt(self.mu_x)
        # eps_mu = self.mu_xp - self.f_old(self.mu_x)
        eps_y = self.y - self.g(self.mu)
        # eps_y = (eps_y + np.pi) / 2*np.pi # Normalise eps_y
        # eps_mu = self.mu_p.theta - self.f(self.mu)
        eps_mu = self.v - self.mu_p.theta

        # Calculate precision-weighted prediction errors
        zet_y = self.PI_z * eps_y
        zet_mu = self.PI_w * eps_mu

        # Calculate free energy using precision-weighted squared sensory and motion prediction errors
        F = 1/2 * (eps_y**2*self.PI_z + eps_mu*2*self.PI_w)
        # print(f"F = {F}, eps_y = {eps_y}, eps_mu = {eps_mu}")
        
        # Calculate gradient for mu_x by calculating the gradient of F w.r.t. mu
        dmu_theta = -self.k_mu*(-zet_y + zet_mu)
        
        # Descend one step down the gradient
        self.mu.theta = self.mu.theta + dmu_theta * self.dt
        self.mu.direction = np.sign(dmu_theta)
        print(f"mu_x = {self.mu.theta:.2f}, dmu_x = {dmu_theta:.2f}, zet_y = {zet_y:.2f}, zet_mu = {zet_mu:.2f}")

        # Estimate mu_xp
        self.mu_theta_old = self.mu.theta
        self.mu_xp = (self.mu.theta - self.mu_theta_old) / self.dt
        # print(f"mu_x = {self.mu_x}, dmu_x = {dmu_x}, y = {self.y}")

        # Use gradient descent to act
        # self.du = -self.k_u * (zet_y)
        # self.u = self.u + self.du * self.dt
        # Use velocity controller not torque - g is what will allow us to choose the correct action
        if (self.y > 245) & (self.y < 265): # If light spot found
            print("finished")
            self.u = 0
        else:
            self.u = np.sign(eps_y) # self.u * np.sign(eps_y)

        # self.u = 0.25
        # self.u = np.clip(self.u, -0.5, 0.5)

        # print(f"u = {self.u}")
        # print(f"mu_x = {round(self.mu_x, 2)} x = {round(self.g_truth_x, 2)}")
        self.pub.publish(self.u)
        self.pub_f.publish(F)
        
    #--Generative Model--#
    # f gives how much the next state should increment by
    def f(self, x): # Function of motion (motion model) - p(x)
        noise = self.sig_w * self.rng.standard_normal()

        return noise

    def g(self, x): # Function of sensory mapping (measurement model) - p(y | x)
        noise = self.sig_z * self.rng.standard_normal()

        ypred = self.m[round(np.interp(x.theta, np.linspace(-np.pi, np.pi, self.m.size), range(self.m.size)))]
        
        # return self.y + noise
        # print(f"ypred = {ypred}, y = {self.y}")
        return x.theta #ypred + noise

    def f_old(self, x): # Function of motion (motion model) - p(x)
        # noise = self.rng.standard_normal(0, self.sig_w, np.int(np.round(self.T/self.dt)))
        noise = self.sig_w * self.rng.standard_normal()
        # angvel = u * self.dt
        # xpred = x + angvel * self.dt
        # x = self.y

        x = self.joint2map(x)
        if x > 0:
            xpred = self.v - x
        else:
            xpred = x + self.v

        # print(f"distance to goal = {xpred}, mu_x = {self.mu_x}")
        # xpred = self.v - x
        return xpred #+ noise

    def g_gt(self, x): # Sensory mapping to be used with joint sensor values
        # noise = self.rng.standard_normal(0, self.sig_z, np.int(np.round(self.T/self.dt)))
        noise = self.sig_z * self.rng.standard_normal()
        ypred = x
        return ypred #+ noise
        # return self.y + noise

    def joint2map(self, x):
        if x == 0:
            return 0
        elif x > 0:
            x = x % (2*np.pi)
            if (x > 0) & (x < np.pi):
                x = x
            elif (x > np.pi) & (x < 2*np.pi):
                x = x % -np.pi
        else:
            x = x % (-2*np.pi)
            if (x < 0) & (x > -np.pi):
                x = x
            elif (x < -np.pi) & (x > -2*np.pi):
                x = x % np.pi

        return x