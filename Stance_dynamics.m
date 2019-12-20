function [dx] = Stance_dynamics(t,model,x,u)
%STANCE_DYNAMICS Summary of Stance phase dynamics
%   input: t --> time step
%          x --> state vector in stance phase  x = [l;theta;dl;dtheta]
%          u --> control input                 u = [f_leg, t_hip]
%   output:dx --> first order 
%          dx = [dl;dtheta;ddl;ddtheta]
l = x(1);
theta = x(2);
dl = x(3);
dtheta = x(4);

[M,C,G,B] = Stance_EoM(model,x);
ddq = M \ (u + B - G - C*[dl;dtheta]);
dx = [dl;dtheta;ddq(1);ddq(2)];
end

