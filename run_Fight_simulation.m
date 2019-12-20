function [t,x,te,xe] = run_Fight_simulation(t_span,model,x0)
%FIGHT_SIMULATION run flight simulation
%   input: model --> robot model params
%          x0 --> initial state vector x0 = xc0;zc0;dxc0;dzc0
%          t_span --> simulation time span
%   output: t --> time sequence
%           x --> state matrix (4xn)
%           te --> stop time
%           xe --> stop state
u = [0;0];
%% ODE45 solver
option = model.option1;

sol = ode45(@(t,x)(Flight_dynamics(t,model,x,u)), t_span, x0, option);
% disp(sol);

% te = sol.xe;
% xe = sol.ye;

te = sol.xe;
xe = sol.ye;
x = sol.y;
t = sol.x;
% t = linspace(t_span(1),te,(te-t_span(1))*1000);
% x = deval(sol,t);
end

