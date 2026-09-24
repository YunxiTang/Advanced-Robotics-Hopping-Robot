function [value,isterminal, direction] = StanceEvent(t,x)
%STANCEEVENT determine the stop condition of simulation when robot stand on
%ground
%   input: t --> current time
%          x --> current state x=[l;theta;dl;dtheta]
%   output: value = 0 --> stop condition

% if the mass hit the ground, terminate the simulation
l = x(1);
theta = x(2);

value(1) = l*cos(theta);
value(2) = (l-0.2);

direction(1) = 0;
direction(2) = 1;

isterminal(1) = true;
isterminal(2) = true;
end

