%%%%%%%%%%%%%% Falling Ball Simulation %%%%%%%%%%%%
%% run simulatin // Hard-collision
% initial state
r0 = [0.0;2.2];
v0 = [5.5;0.0];

x0 = [r0;v0];

Nx = length(x0);
h = 0.001;
Tf = 10.0;
Nt = floor(Tf / h) + 1;
thist = linspace(0.0, Tf,Nt);
xhist = zeros(Nx, Nt);
xhist(:,1) = x0;
for k=1:(Nt-1)
    xhist(:,k+1) = Fight_Dynamics_rk4(xhist(:,k), h);
    if guard(xhist(:,k+1)) <= 0
        % for hard collision
%         xhist(:,k+1) = jump_map(xhist(:,k));
        % for soft collision
        xhist(:,k+1) = Stance_Dynamics_rk4(xhist(:,k), h);
    end
end

%% PLot
figure(1);
subplot(3,2,1);
plot(thist, xhist(1,:),'k', 'LineWidth', 2.0);
grid on;
title('$x$','Interpreter','latex', 'FontSize', 15, 'FontAngle', 'italic');
xlabel('Time[s]','Interpreter','latex', 'FontSize', 10, 'FontAngle', 'italic');
subplot(3,2,2);
plot(thist, xhist(2,:),'k', 'LineWidth', 2.0);
grid on;
title('$y$','Interpreter','latex', 'FontSize', 15, 'FontAngle', 'italic');
xlabel('Time[s]','Interpreter','latex', 'FontSize', 10, 'FontAngle', 'italic');
subplot(3,2,3);
plot(thist, xhist(3,:),'k', 'LineWidth', 2.0);
grid on;
title('$v_x$','Interpreter','latex', 'FontSize', 15, 'FontAngle', 'italic');
xlabel('Time[s]','Interpreter','latex', 'FontSize', 10, 'FontAngle', 'italic');
subplot(3,2,4);
plot(thist, xhist(4,:),'k', 'LineWidth', 2.0);
grid on;
title('$v_y$','Interpreter','latex', 'FontSize', 15, 'FontAngle', 'italic');
xlabel('Time[s]','Interpreter','latex', 'FontSize', 10, 'FontAngle', 'italic');
grid on;
hold off;
%% animation
subplot(3,2,5:6);
run_animation(xhist);hold on; 
plot(xhist(1,:), 0.*xhist(2,:),'g-','LineWidth',5.0);hold on;
plot(xhist(1,:), xhist(2,:),'k-.','LineWidth',2.0);
xlabel('$x$','Interpreter','latex', 'FontSize', 10, 'FontAngle', 'italic');
ylabel('$y$','Interpreter','latex', 'FontSize', 10, 'FontAngle', 'italic');
legend('$Ball$', '$Ground$', '$Trajectory$','Interpreter','latex', 'FontSize', 10, 'FontAngle', 'italic');
%% Functions
% Dynamics
function xdot = Flight_Dynamics(x)
    g = 9.81;
    r = x(1:2);
    v = x(3:4);
    
    a =  [0; -g];
    
    xdot = [v; a];
end

function xdot = Stance_Dynamics(x)
    m = 1.0;
    g = 9.81;
    mu = 0.01;
    
    r = x(1:2);
    v = x(3:4);
    K_g = 2000;
    K_d = 1.2;

    lambda_y = -K_g * r(2) - K_d * v(2);
    ay = (lambda_y - m*g) / m;
    
    lambda_x = - mu * abs(lambda_y) * sign(v(1));
    ax = lambda_x / m;
    a = [ax;ay];
    xdot = [v;a];
end

function x_next = Fight_Dynamics_rk4(x, dt)
    k1 = Flight_Dynamics(x);
    k2 = Flight_Dynamics(x + 0.5 * dt * k1);
    k3 = Flight_Dynamics(x + 0.5 * dt * k2);
    k4 = Flight_Dynamics(x +       dt * k3);
    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4);
end

function x_next = Stance_Dynamics_rk4(x, dt)
    k1 = Stance_Dynamics(x);
    k2 = Stance_Dynamics(x + 0.5 * dt * k1);
    k3 = Stance_Dynamics(x + 0.5 * dt * k2);
    k4 = Stance_Dynamics(x +       dt * k3);
    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4);
end

% gurad function
function l = guard(x)
    l = x(2);
end

% jump map function
function xn = jump_map(x)
    gamma = 0.9;
    xn = [x(1);0.0;x(3);-gamma*x(4)];
end

% run animation
function run_animation(x)
    [~, N] = size(x);
    for k=1:50:N 
        plot(x(1,k), x(2,k), 'o', 'MarkerSize', 12, 'MarkerFaceColor', 'c');
        grid on;
        xlim([x(1,1), x(1, end)]);
        ylim([-1, x(2,1)]);
        pause(0.0001);
    end
end

