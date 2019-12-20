%%% Simulation of fight phase hopping robot 

clear all;
clc;
close all;


global THETA 
THETA = deg2rad(60);
%% set parameters of hopping robot
model = set_model();

%% simulation time span
t_span = [0 10];

%% simulation
% initial state x0 = [xc;zc;dxc;dzc;theta;l;dtheta;dl]
xc0 = 0.0;zc0 = 0.5;
dxc0 = 0.1;dzc0 = 0.0;

x0 = [xc0;zc0;dxc0;dzc0];

%% run flight phase simulation
% t is time array; x is the state matrix;te is the stop time; xe is the
% stop state
[t,x,te,xe] = run_Fight_simulation(t_span,model,x0);
plot(x(1,:),x(2,:));

