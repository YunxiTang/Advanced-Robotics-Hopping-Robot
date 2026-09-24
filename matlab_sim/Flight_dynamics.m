function [dx] = Flight_dynamics(t,model,x,u)
%FLIGHT_DYNAMICS Summary of dynamics of flight phase
%   input: t --> time step
%          x --> state vector in fligh phase
%          x = [xc;zc;dxc;dzc]
%   output:dx --> first order 
%          dx = [dxc;dzc;ddxc;ddzc]
xc = x(1);
zc = x(2);
dxc = x(3);
dzc = x(4);

[M,C,G] = Flight_EoM(model,x);
ddq = M\(u - C * [dxc;dzc] - G);
dx = [dxc;dzc;ddq(1);ddq(2)];
end

