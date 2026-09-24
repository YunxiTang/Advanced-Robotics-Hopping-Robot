%%% Simulation of stancephase hopping robot 

clear all;
clc;
close all;

%% set parameters of hopping robot
model = set_model();

%% simulation time span
t_span = [0 10];

%% simulation
% initial state x0 = [l;theta;dl;dtheta]
l0 = 0.2;theta0 = deg2rad(30);
dl0 = 0;dtheta0 = deg2rad(0);

x0 = [l0;theta0;dl0;dtheta0];

%% run flight phase simulation
% t is time array; x is the state matrix;te is the final time; xe is the
% final state
[t,x,te,xe] = run_Stance_simulation(t_span,model,x0);
figure(1);
plot(t,x(1,:));
grid on;
figure(2);
plot(t,x(2,:));
grid on;

figure(3)
scatter(x(1,:).*sin(-x(2,:)),x(1,:).*cos(x(2,:)),'filled','c','blue');
hold on;
scatter(0,0,'filled','c','red');
xlim([-0.2 0.2]);
ylim([0 0.4]);
axis equal;

