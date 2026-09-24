function [model] = set_model()
%SET_MODEL set model parameters of the system
%   m --> body mass
%   g --> gravity
%   k --> stiffness of spring
%   b --> damping coefffecient of spring
%   L0 --> oringinal length of spring

model.m = 1;
model.g = 10;
model.k = 3000;
model.b = .0;
model.L0 = 0.2;

%% ode45 option struct
model.option1 = odeset('RelTol',1e-10,...
                       'AbsTol', 1e-10,...
                       'Events', @FlightEvent,...
                       'Vectorized',true,...
                       'MaxStep',0.0001);
                   
model.option2 = odeset('RelTol',1e-10,...
                       'AbsTol', 1e-10,...
                       'Events', @StanceEvent,...
                       'Vectorized',true,...
                       'MaxStep',0.0001);
end

