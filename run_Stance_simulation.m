function [t,x,te,xe] = run_Stance_simulation(t_span,model,x0)
%RUN_STANCE_SIMULATION run stance simulation
%   input: model --> robot model params
%          x0 --> initial state vector x0 = [l0;theta0;dl0;dtheta0]
%          t_span --> simulation time span
%   output: t --> time sequence
%           x --> state matrix (4xn)
%           te --> stop time
%           xe --> stop state  

%% controller 
% u = [70;0];
u = @(t,x)Stance_Controller(t,model,x);
%% ODE45 solver
option = model.option2;

sol = ode45(@(t,x)(Stance_dynamics(t,model,x,u(t,x))), t_span, x0, option);

te = sol.xe;
xe = sol.ye;
x = sol.y;
t = sol.x;
% t = linspace(t_span(1),te,(te-t_span(1))*1000);
% x = deval(sol,t);
end

