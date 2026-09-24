function [x_c,t_c] = Planar2Cart(x_p,t_p,pos_last_touch,model)
%PLANAR2CART From Planar coordinate to Cartesian Space
%   input: x_p --> Planar space vector x_p = [l;theta;dl;dtheta]
%          t_p --> time of last planar phase
%   output: x_c --> Cartesian space state vector x_c = [xc;zc;dxc;dzc]
%           t_c --> Cartesian start time

l = x_p(1);
theta = x_p(2);
dl = x_p(3);
dtheta = x_p(4);

L0 = model.L0;

if abs((l - L0)) >= 0.01
    disp('The robot has fall down');
    dxc = 0;
    dzc = 0;
    xc = 0;
    zc = 0;
else
    x_pos_lasttouch = pos_last_touch(1);
    z_pos_lasttouch = pos_last_touch(2);
    
    dxc = dl*sin(-theta) - dtheta*L0*cos(-theta);
    dzc = dl*cos(-theta) + dtheta*L0*sin(-theta);
%     disp(dxc);
%     disp(dzc);
    
    xc = x_pos_lasttouch + L0*sin(2*pi-theta);
    zc = z_pos_lasttouch + L0*cos(2*pi-theta);
end
x_c = [xc;
       zc;
       dxc;
       dzc];
t_c = t_p;
end

