function [value,isterminal, direction] = FlightEvent(t,x)
%FLIGHTEVENT determine the stop condition of simulation when robot falldown
%   input: t --> current time
%          x --> current state x=[xc;zc;dxc;dzc]
%   output: value = 0 --> stop condition
global THETA
isterminal = true;
direction = -1;
flag = 1;
if (x(2) - 0.2*cos(THETA)) < 0.0001
    flag = 0;
end
value = flag;
end

