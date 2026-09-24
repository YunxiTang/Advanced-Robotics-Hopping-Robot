function [tau] = Stance_Controller(t,model,x)
%STANCE_CONTROLLER Design the controller in stance phase
%   Input: t --> time 
%          x --> current state
%   output: tau --> torques tau = [f_leg;0]
global Hc_max
global Hs
global bounce_count
global Ed Vc_cors

l  = x(1);
theta = x(2);
dl = x(3);
dtheta = x(4);

En_lastf = model.m*model.g*Hc_max + 0.5*model.m*Vc_cors^2;

error = En_lastf - Ed;

if bounce_count <= 10
    f_leg = 0;
else
    if dl <0
        f_leg = 2.0 * error;% + 0.3*(Hc_max-Hs);
    else
        f_leg = -2.0 * error;% - 0.3*(Hc_max-Hs);
    end
end

% if bounce_count == 1
%     f_leg = 0;
% else
%     if bounce_count <= 5
%         f_leg = 0;
%     else
%         if dl < 0
%             f_leg = 35*(Hc_max-Hs);
%         else
%             f_leg = -35*(Hc_max-Hs);
%         end
%     end
% end

tau = [f_leg;
       0];
end

