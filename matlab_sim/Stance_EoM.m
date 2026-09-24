function [M,C,G,B] = Stance_EoM(model,x)
%STANCE_EOM Equation of motion in stance phase
%   input: model --> robot model
%          x --> current state x = [l;theta;dl;dtheta]
%   ouput: M --> inertia matrix
%          C --> C term
%          G --> gravity term
%          B --> damping term
g = model.g;
m = model.m;
k = model.k;
L0 = model.L0;
b = model.b;

l = x(1);
theta = x(2);
dl = x(3);
dtheta = x(4);

M = [m 0;
     0 m*l^2];
C = [0           -m*l*dtheta;
     m*l*dtheta  m*l*dl];
G =[k*(l-L0)+m*g*cos(theta);
    -m*g*l*sin(theta)];
B = [-b*dl;
     0];

end

