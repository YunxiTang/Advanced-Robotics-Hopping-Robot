function [M,C,G] = Flight_EoM(model,x)
%FLIGHT_EOM Equation of motion in flight phase
%   input: model --> robot model
%          x --> current state x = [xc;zc;dxc;dzc]
%   ouput: M --> inertia matrix
%          C --> C term
%          G --> gravity term

g = model.g;
M = [1 0;0 1];
C = [0 0;0 0];
G = [0;g];
end

