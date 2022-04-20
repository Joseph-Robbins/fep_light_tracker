for i = 2:N
    % The generative process (i.e. the real world)
    x_dot(i) = a(i-1); % action
    x(i) = x(i-1) + dt * (x_dot(i));
    T(i) = T0 / (x(i)^2 + 1);
    Tx(i) = -2 * T0 * x(i) * (x(i)^2 + 1)^ - 2;
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