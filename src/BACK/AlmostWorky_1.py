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
        self.k_mu = 0.1 #0.05 #1.167 # Learning rate of the gradient descent on the belief about our state (mu)
        self.k_u = 10 #70 # Learning rate of action
        self.m = m # Initialise map

        ind = np.where(self.m == self.m.max())[0][0]
        self.v = (ind - self.m.size/2) * (np.pi / (self.m.size/2)) # Prior belief of position
        self.v = -np.pi/2 #2*np.pi - 0.1

        self.sig_z   = 1 # Variance in state estimation noise
        self.sig_z_p = 1 # Variance in state estimation noise
        self.sig_w   = 1 # Variance in sensor estimation noise
        self.sig_w_p = 1 # Variance in sensor estimation noise

        self.Pi_z = 1/self.sig_z # Precision in state estimation noise
        self.Pi_z_p = 1/self.sig_z # Precision in sensor estimation noise
        self.Pi_w = 1/self.sig_w
        self.Pi_w_p = 1/self.sig_w

        self.mu = 0 #state()
        self.mu_p = 0 #state()    
        self.mu_pp = 0 #state()    
        self.g_truth_x = 0 #state()

        self.y = 0 # Sensor reading
        self.y_p = 0 # Sensor reading
        self.y_pp = 0 # Sensor reading
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
        self.g_truth_xpp = msg.effort[0]

        # self.y_old = self.y
        # self.y = self.g_truth_x
        # # self.yp = self.g_truth_xp
        # self.yp = (self.y - self.y_old) / self.dt
        self.g_data_received = True

    def minimise_f(self):
        # eps_y = self.y - self.g_gt(self.mu_x)
        # eps_mu = self.mu_xp - self.f_old(self.mu_x)
        # eps_y = 255 - self.y #self.g(self.mu)
        eps_y = self.y - self.g(self.mu)
        # print(f"y = {self.y:.2f}, g(mu) = {self.g(self.mu):.2f}, mu = {self.mu:.2f}")
        eps_y_p = self.y_p - self.g_p(self.mu_p)
        # print(f"y = {self.y:.2f} ypred = {self.g(self.mu):.2f}, ypredx = {self.g(self.g_truth_x):.2f}")
        # eps_y = (eps_y + np.pi) / 2*np.pi # Normalise eps_y
        eps_mu = self.mu_p - self.f(self.mu)
        eps_mu_p = self.mu_pp - self.f_p(self.mu_p)
        print(f"eps_y = {eps_y:.2f}, eps_y_p = {eps_y_p:.2f}, eps_mu = {eps_mu:.2f}, eps_mu_p = {eps_mu_p:.2f}")
        # print(f"eps_mu = {eps_mu:.2f}, mu_p = {self.mu_p:.2f}, mu = {self.mu:.2f}")

        # Calculate precision-weighted prediction errors
        zet_y = self.Pi_z * eps_y
        zet_y_p = eps_y_p * self.Pi_z_p
        zet_mu = self.Pi_w * eps_mu
        zet_mu_p = eps_mu_p * self.Pi_w_p

        # Calculate free energy using precision-weighted squared sensory and motion prediction errors
        SPE_0 = eps_y**2*self.Pi_z
        SPE_1 = eps_y_p**2*self.Pi_z
        MPE_0 = eps_mu*2*self.Pi_w
        MPE_1 = eps_mu_p**2*self.Pi_w
        F = 1/2 * (SPE_0 + SPE_1 + MPE_0 + MPE_1)
        # print(f"F = {F:.2e}, eps_y = {eps_y:.2e}, eps_mu = {eps_mu:.2e}")
        # print(f"F = {F:.2f}, eps_y = {eps_y:.2f}, eps_mu = {eps_mu:.2f}")
        
        # Calculate gradient for mu_x by calculating the gradient of F w.r.t. mu
        dmu = self.mu_p - self.k_mu*(zet_mu + zet_y)
        dmu_p = self.mu_pp - self.k_mu * (-zet_y_p + zet_mu + zet_mu_p)
        dmu_pp = -self.k_mu * (zet_mu_p)
        
        # Descend one step down the gradient
        self.mu    = self.mu + dmu * self.dt
        self.mu_p  = self.mu_p + dmu_p * self.dt
        self.mu_pp = self.mu_pp + dmu_pp * self.dt
        # self.mu =    self.v - self.g_truth_x
        # self.mu_p =  self.g_truth_xp
        # self.mu_pp = self.g_truth_xpp
        # self.mu.direction = np.sign(dmu_theta)
        # print(f"mu_x = {self.mu.theta:.2f}, dmu_x = {dmu_theta:.2f}, zet_y = {zet_y:.2f}, zet_mu = {zet_mu:.2f}")

        # Estimate mu_xp
        # self.mu_old = self.mu
        # self.mu_xp = (self.mu - self.mu_old) / self.dt
        # print(f"mu_x = {self.mu}, dmu_x = {dmu}, y = {self.y}")

        # Use gradient descent to act
        # self.du = -self.k_u * (zet_y)
        # self.u = self.u + self.du * self.dt
        # Use velocity controller not torque - g is what will allow us to choose the correct action
        # if (self.y > np.max(self.m) - 10) & (self.y < np.max(self.m) + 10): # If light spot found
        #     print("Got EM")
        #     self.u = 0
        # else:
        # self.u = np.sign(eps_y) # self.u * np.sign(eps_y)

        self.du = np.sign(eps_y) * self.k_u * (zet_y + zet_y_p)
        self.u = self.u + self.du * self.dt
        self.u = np.clip(self.u, -1, 1)

        # self.u = 0.25

        # print(f"u = {self.u}")
        # print(f"mu_x = {self.mu:.2f} x = {self.g_truth_x:.2f} y = {self.y:.2f} ypred = {self.g(self.mu):.2f} gypred = {self.g(self.g_truth_x):.2f}")
        self.pub.publish(self.u)
        self.pub_f.publish(F)
        
    #--Generative Model--#
    # f gives how much the next state should increment/decrement by
    def f(self, x): # Function of motion (motion model) - p(x)
        noise = self.sig_w * self.rng.standard_normal()

        return self.v - x

    def f_p(self, x):
        return 0 #-x #self.f(x)

    def g(self, x): # Function of sensory mapping (measurement model) - p(y | x)
        noise = self.sig_z * self.rng.standard_normal()
        
        x = self.joint2map(x)
        ypred = self.m[round(np.interp(x, np.linspace(-np.pi, np.pi, self.m.size), range(self.m.size)))]
        # ypred = (255/np.pi) * x
        
        # return self.y + noise
        # print(f"ypred = {ypred}, y = {self.y}")
        # return (np.max(self.m) - ypred) + noise
        return ypred #+ noise

    def g_p(self, x):
        return self.g(x) #(255/np.pi) * x

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