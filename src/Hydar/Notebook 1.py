# Import the dependencies
import numpy as np
from scipy.linalg import toeplitz, cholesky, sqrtm, inv
# import scipy.linalg as la
from scipy import signal
from scipy.integrate import odeint
import time
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

def g_gp(x,v):
    """
    Generative process, equation of sensory mapping: g_pp(x) at point x    
   
    INPUTS:
        x       - Hidden state, depth position in centimetres
        v       - Hidden causal state, in this example not used
        
    OUTPUT:
        y       - Temperature in degrees celsius
        
    """
    #t0= 20
    #y=(t0-8)/(0.2*x**2+1)+8
    t0=25
    return t0 -16 / (1 + np.exp(5-x/5))

def dg_gp(x):
    """
    Partial derivative of generative process towards x, equation of sensory mapping: g'_gp(x) at point x    
   
    INPUTS:
        x       - Position in centimetres    
        
    OUTPUT:
        y       - Temperature in degrees celsius
        
    """
    #t0= 20
    #y=-2*0.2*x*(t0-8)/(0.2*x**2+1)**2
    
    return -16/5* np.exp(5-x/5) / (np.exp(5-x/5)+1)**2

# Show the temperature curve
x_show = np.arange (-0,50,0.01)
y_show = g_gp(x_show,0)
dy_show = dg_gp(x_show)
plt.plot(y_show, x_show)
#plt.plot(dy_show, x_show)
plt.ylabel('Depth (centimeters)')
plt.xlabel('Temperature (° C)')
plt.gca().invert_yaxis()
plt.vlines(17, 50, 25, colors='r', linestyles='dashed')
plt.hlines(25, 10,17, colors='r', linestyles='dashed')
plt.text(17.3,27,"Optimal temparature 17° C")
# plt.show()

print('Temperature at 25 centimetres is: ', g_gp(25,0), ' degrees celsius')

# a very simple example to showcase forward Euler method in action
# The derivative of x^2 = 2x
# Taking small steps along the gradient (= derivative) should show x^2

def f(x):
    return x**2

def df(x):
    return 2*x

_dt=0.001 # The small timestep
_T=2
_x = np.arange (-_T,_T,_dt)
_N=np.int(2*_T/_dt) # amount of data points

_I = np.zeros(_N)
_I[0]=4
for _i in np.arange(1,_N):
    _I[_i]= _I[_i-1] + _dt*df(_x[_i-1])

plt.figure()
plt.plot(_x,_I, "-");
plt.plot(_x,f(_x), '--');

# Setting up the time data:
dt = 0.005; # integration step, average neuron resets 200 times per second
T = 5+dt; # maximum time considered
t = np.arange(0,T,dt)
N= t.size #Amount of data points
print ('Amount of data points: ', N)
print ('Starting with', t[0:5])
print ('Ending with', t[N-5:N])
print ('Data elements', np.size(t))

def makeNoise(C,s2,t):
    """
    Generate coloured noise 
    Code by Sherin Grimbergen (V1 2019) and Charel van Hoof (V2 2020)
    
    INPUTS:
        C       - variance of the required coloured noise expressed as desired covariance matrix
        s2      - temporal smoothness of the required coloured noise, expressed as variance of the filter
        t       - timeline 
        
    OUTPUT:
        ws      - coloured noise, noise sequence with temporal smoothness
    """
    
    if np.size(C)== 1:
        n = 1
    else:
        n = C.shape[1]  # dimension of noise
        
    # Create the white noise with correct covariance
    N = np.size(t)      # number of elements
    L =cholesky(C, lower=True)  #Cholesky method
    w = np.dot(L,np.random.randn(n,N))
    
    if s2 <= 1e-5: # return white noise
        return w
    else: 
        # Create the noise with temporal smoothness
        P = toeplitz(np.exp(-t**2/(2*s2)))
        F = np.diag(1./np.sqrt(np.diag(np.dot(P.T,P))))
        K = np.dot(P,F)
        ws = np.dot(w,K)
        return ws

class ai_capsule():
    """
        Class that constructs a group of neurons that perform Active Inference for one hidden state, one sensory input, one prior
        In neurology it could eg represent a (micro) column
        
        Version 0.1
    """
    def __init__(self,dt, mu_v, Sigma_w, Sigma_z, a_mu):   
        self.dt = dt    # integration step
        self.mu_x = mu_v   # initializing the best guess of hidden state by the hierarchical prior
        self.F = 0      # Free Energy
        self.eps_x = 0  # epsilon_x, prediction error on hidden state
        self.eps_y = 0  # epsilon_y, prediction error on sensory measurement
        self.Sigma_w = Sigma_w #Estimated variance of the hidden state 
        self.Sigma_z = Sigma_z # Estimated variance of the sensory observation 
        self.alpha_mu = a_mu # Learning rate of the gradient descent mu (hidden state)
    
    def g(self,x,v):
        """
            equation of sensory mapping of the generative model: g(x) at point x 
            Given as input for this example equal to the true generative process g_gp(x)
        """
        return g_gp(x,v)
    
    def dg(self, x):
        """
            Partial derivative of the equation of sensory mapping of the generative model towards x: g'(x) at point x 
            Given as input for this example equal to the true derivative of generative process dg_gp(x)
        """
        return dg_gp(x)
    
    def f(self,x,v):
        """
            equation of motion of the generative model: f(x) at point x 
            Given as input for this example equal to the prior belief v
        """
        return v
    
    # def df(self,x): Derivative of the equation of motion of the generative model: f'(x) at point x
    # not needed in this example 

        
    def inference_step (self, i, mu_v, y):
        """
        Perceptual inference    

        INPUTS:
            i       - tic, timestamp
            mu_v    - Hierarchical prior input signal (mean) at timestamp
            y       - sensory input signal at timestamp

        INTERNAL:
            mu      - Belief or hidden state estimation

        """

        # Calculate prediction errors
        self.eps_x = self.mu_x - self.f(self.mu_x, mu_v)  # prediction error hidden state
        self.eps_y = y - self.g(self.mu_x, mu_v) #prediction error sensory observation

        # Free energy gradient
        dFdmu_x = self.eps_x/self.Sigma_w - self.dg(self.mu_x) * self.eps_y/self.Sigma_z

        # Perception dynamics
        dmu_x   = 0 - self.alpha_mu*dFdmu_x  # Note that this is an example without generalised coordinates of motion hence mu'=0
        
        # motion of mu_x 
        self.mu_x = self.mu_x + self.dt * dmu_x
        
        # Calculate Free Energy to report out
        self.F = 0.5 * (self.eps_x**2 / self.Sigma_w + self.eps_y**2 / self.Sigma_z + np.log(self.Sigma_w * self.Sigma_z))
        
        return self.F, self.mu_x , self.g(self.mu_x,0)

def simulation (v, mu_v, Sigma_w, Sigma_z, noise, a_mu):
    """
    Basic simplist example perceptual inference    
   
    INPUTS:
        v        - Hydars actual depth, used in generative model, since it is a stationary example hidden state x = v + random fluctuation
        mu_v     - Hydar prior belief/hypotheses of the hidden state
        Sigma_w  - Estimated variance of the hidden state 
        Sigma_z  - Estimated variance of the sensory observation  
        noise    - white, smooth or none
        a_mu     - Learning rate for mu
        
    """

    # Init tracking
    mu_x = np.zeros(N) # Belief or estimation of hidden state 
    F = np.zeros(N) # Free Energy of AI neuron
    mu_y = np.zeros(N) # Belief or prediction of sensory signal
    x = np.zeros(N) # True hidden state
    y = np.zeros(N) # Sensory signal as input to AI neuron

    # Create active inference neuron
    capsule = ai_capsule(dt, mu_v, Sigma_w, Sigma_z, a_mu)  

    # Construct noise signals with emporal smoothness:
    np.random.seed(42)
    sigma = 1/2000 # smoothness of the noise parameter, variance of the filter
    w = makeNoise(Sigma_w,sigma,t)
    z = makeNoise(Sigma_z,sigma,t)

    ssim = time.time() # start sim
    
    # Simulation
    for i in np.arange(1,N):
        # Generative process
        if noise == 'white':
            x[i] = v + np.random.randn(1)* Sigma_w
            y[i] = g_gp(x[i],v) + np.random.randn(1)* Sigma_z
        elif noise == 'smooth':
            x[i]= v + w[0,i]
            y[i] = g_gp(x[i],v) + z[0,i]
        else: #no noise
            x[i]= v 
            y[i] = g_gp(x[i],v)
        #Active inference
        F[i], mu_x[i], mu_y[i] = capsule.inference_step(i,mu_v,y[i])

    # Print the results
    tsim = time.time() - ssim
    #print('Simulation time: ' + "%.2f" % tsim + ' sec' )

    return F, mu_x, mu_y, x, y

# Test case
v = 30 # actual depth Hydar
mu_v = 25 # Hydars belief of the depth
F1, mu_x1, mu_y1, x1, y1 = simulation(v,mu_v,1,1,'no noise',1) # prior and observation balanced, both variance of 1

# Plot results:
fig, axes = plt.subplots(3, 1, sharex='col');
fig.suptitle('Basic Active Inference, inference part');
axes[0].plot(t[1:],mu_x1[1:],label='Belief');
axes[0].plot(t[1:],x1[1:],label='Generative process');
axes[0].hlines(mu_v, 0,T, label='Prior belief')
axes[0].set_ylabel('$\mu_x$ position');
fig.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
axes[0].grid(1);
axes[1].plot(t[1:],mu_y1[1:],label='Belief');
axes[1].plot(t[1:],y1[1:],label='Generative process');
axes[1].hlines(g_gp(mu_v,0), 0,T, label='Prior belief')
axes[1].set_ylabel('$\mu_y$ temperature');
axes[1].grid(1);
axes[2].semilogy(t[1:],F1[1:],label='Belief');
axes[2].set_xlabel('time [s]');
axes[2].set_ylabel('Free energy');
axes[2].grid(1);

plt.show()