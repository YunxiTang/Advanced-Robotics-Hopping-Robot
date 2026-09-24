function [x_p,pos_touch,t_p] = Cart2Planar(x_c,t_c,model)
global THETA
%COORDINATE_TRANSFER From Cartesian Space to Planar coordinate
%   input: x_c --> Cartesian space state vector x_c = [xc;zc;dxc;dzc]
%          t_c --> time of last flight phase
%          model --> robot system
%   output: x_p --> Planar space vector x_p = [l;theta;dl;dtheta]
%           t_p --> planar start time
xc = x_c(1);
zc = x_c(2);

dxc = x_c(3);
dzc = x_c(4);

l = model.L0;
theta = THETA;

dl = dzc*cos(THETA) - dxc*sin(THETA);
dtheta = -(dxc*cos(THETA)+dzc*sin(THETA))/l;

x_p = [l;
       theta;
       dl;
       dtheta];

t_p = t_c;
pos_touch = [xc + model.L0*sin(THETA);
             zc - model.L0*cos(THETA)];
end

