%  A Simple Bayesian Thermostat
%  The free energy principle for action and perception: A mathematical review, Journal of Mathematical Psychology
%  Christopher L. Buckley, Chang Sub Kim, Simon M. McGregor and Anil K. Seth
clear;
rng(6);

%  simulation parameters
simTime = 100; dt = 0.005; time = 0:dt:simTime;
N = length(time);
action = true;

% Generative Model Parameters
Td = 4;% desired temperature

% The time that action onsets
actionTime = simTime/4;

% initialise sensors
rho_0(1) = 0;
rho_1(1) = 0;

%  sensory variances
Omega_z0 = 0.1;
Omega_z1 = 0.1;

% hidden state variances
Omega_w0 = .1;
Omega_w1 = .1;

% Params for generative process
T0 = 100; % temperature at x = 0

% Initialise brain state variables
mu_0(1) = 0;
mu_1(1) = 0;
mu_2(1) = 0;

% Sensory noise in the generative process
zgp_0 = randn(1,N)*.1;
zgp_1 = randn(1,N)*.1;

% Initialise the action variable
a(1) = 0;

% Initialise generative process
x_dot(1) = a(1);
x(1) = 2;
T(1) = T0 / (x(1)^2 + 1);
Tx(1) = -2 * T0 * x(1) * (x(1)^2 + 1)^-2;
T_dot(1) = Tx(1) * (x_dot(1));

% Initialise sensory input
rho_0(1) = T(1);
rho_1(1) = T_dot(1);

% Initialise error terms
epsilon_z_0 = rho_0(1) - mu_0(1);
epsilon_z_1 = rho_1(1) - mu_1(1);
epsilon_w_0 = mu_1(1)+  mu_0(1) - Td;
epsilon_w_1 = mu_2(1) + mu_1(1);

% Initialise Variational Energy
VFE(1) = 1/2 * (Omega_z0^-1*epsilon_z_0^2 + Omega_z1^-1*epsilon_z_1^2 + Omega_w0^-1*epsilon_w_0^2 + Omega_w1^-1* epsilon_w_1^2)

% Gradient descent learning parameters
k = .1; % for inference
ka = .01; % for learning

for i = 2:N
    % The generative process (i.e. the real world)
    x_dot(i) = a(i-1); % action
    x(i) = x(i-1) + dt * (x_dot(i));
    T(i) = T0 / (x(i)^2 + 1);
    Tx(i) = -2 * T0 * x(i) * (x(i)^2 + 1)^-2;
    T_dot(i) = Tx(i) * (x_dot(i));

    rho_0(i) =  T(i) + zgp_0(i); % calclaute sensory input
    rho_1(i) =  T_dot(i) + zgp_1(i);

    % The generative model (i.e. the agents brain)
    epsilon_z_0 = (rho_0(i-1) - mu_0(i-1)); % error terms
    epsilon_z_1 = (rho_1(i-1) - mu_1(i-1));

    epsilon_w_0 = (mu_1(i-1) + mu_0(i-1) - Td);
    epsilon_w_1 = (mu_2(i-1) + mu_1(i-1));

    VFE(i) = 1/2 * (Omega_z0^-1 * epsilon_z_0^2 + Omega_z1^-1 * epsilon_z_1^2 + Omega_w0^-1 * epsilon_w_0^2 + Omega_w1^-1 * epsilon_w_1^2);

    mu_0(i) = mu_0(i-1) + dt * (mu_1(i-1) - k * (-epsilon_z_0 / Omega_z0 + epsilon_w_0 / Omega_w0));

    mu_1(i) = mu_1(i-1) + dt * (mu_2(i-1) - k * (-epsilon_z_1 / Omega_z1 + epsilon_w_0 / Omega_w0 + epsilon_w_1 / Omega_w1));

    mu_2(i) = mu_2(i-1) + dt * (-k * (epsilon_w_1 / Omega_w1));

    if (time(i) > 25)
        a(i) = a(i-1) + dt * (-ka * Tx(i) * epsilon_z_1 / Omega_z1); % active inference
    else
        a(i) = 0;
    end
end

figure(1); clf;

subplot(5, 1, 1)
plot(time, T); hold on;
plot(time, x); hold on;
legend('T', 'x')

subplot(5, 1, 2)
plot(time, mu_0, 'k'); hold on;
plot(time, mu_1, 'm'); hold on;
plot(time, mu_2, 'b'); hold on;
legend('\mu','\mu''','\mu''''');

subplot(5, 1, 3)
plot(time, rho_0, 'k'); hold on;
plot(time, rho_1, 'm'); hold on;
legend('\rho','\rho''');

subplot(5, 1, 4)
plot(time, a, 'k');
ylabel('a')

subplot(5, 1, 5)
plot(time, VFE, 'k'); xlabel('time'); hold on;
ylabel('VFE')