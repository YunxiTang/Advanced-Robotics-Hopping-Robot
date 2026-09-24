%%%%%%%%%%%%%% Falling Ball Simulation %%%%%%%%%%%%
%% run simulatin // Hard collision
% initial state
r0 = [0.0;2.2];
v0 = [1.5;0.0];

x0 = [r0;v0];

Nx = length(x0);
h = 0.01;
Tf = 5.0;
Nt = floor(Tf / h) + 1;
thist = linspace(0.0, Tf,Nt);
xhist = zeros(Nx, Nt);
xhist(:,1) = x0;
for k=1:(Nt-1)
    xhist(:,k+1) = Dynamics_rk4(xhist(:,k), h);
    if guard(xhist(:,k+1)) <= 0
        xhist(:,k+1) = jump_map(xhist(:,k));
    end
end

%% PLot
figure(1);
subplot(2,1,1);
plot(thist, xhist(1:2,:), 'LineWidth', 2.0);
grid on;
legend('$x$','$y$','Interpreter','latex');
subplot(2,1,2);
plot(thist, xhist(3:4,:), 'LineWidth', 2.0);
legend('$v_x$','$v_y$','Interpreter','latex');
xlabel('Time[s]');
grid on;
%% animation
run_animation(xhist);
%% Functions
% Dynamics
function xdot = Dynamics(x)
    g = 9.81;
    r = x(1:2);
    v = x(3:4);
    
    a =  [0; -g];
    
    xdot = [v; a];
end

function x_next = Dynamics_rk4(x, dt)
    k1 = Dynamics(x);
    k2 = Dynamics(x + 0.5 * dt * k1);
    k3 = Dynamics(x + 0.5 * dt * k2);
    k4 = Dynamics(x +       dt * k3);
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
    figure(2);
    for k=1:1:N 
        plot(x(1,k), x(2,k), 'o', 'MarkerSize', 12, 'MarkerFaceColor', 'b');
        xlim([x(1,1), x(1, end)]);
        ylim([0, x(2,1)]);
        pause(0.005);
    end
end

