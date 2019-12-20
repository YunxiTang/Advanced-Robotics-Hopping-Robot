%%% to connect two dynamic system together
clc;
clear all;
close all;

%% set parameters of hopping robot
model = set_model();
m = model.m;
g = model.g;
%% touch down angle
global THETA 
global Hs
global Vs
global Hc_max
global Ed
global bounce_count 
global Vc_cors
bounce_count = 0;

THETA = deg2rad(0);
% Hs: Desired Height(m)
Hs = 0.5;
% Vs: Desired Horizontal Velocity(m/s)
Vs = 2.5;
Vc_cors = Vs;

Ed = 0.5*model.m*Vs^2 + model.m*model.g*Hs;
%% simulation time span
t_start = 0;
t_end = 400;
t_span = [t_start t_end];

%% initial state x0 = [xc;zc;dxc;dzc]
xc0 = 0.0;zc0 = 0.5;
dxc0 = 0.1;dzc0 = .0;
angle_array = [];
Xc0 = [xc0;zc0;dxc0;dzc0];
%% Record all the data
Data  = {};

for i=1:100
    angle_array(i)=THETA;
    %% START FROM FLIGHT PHASE(Cartesian Space)
    % Tc: flight time array.         Xc: flight state matrix
    % Tce: fight state end time      Xce: flight state end state vector
    [Tc,Xc,Tce,Xce] = run_Fight_simulation(t_span,model,Xc0);
    [Hc_max,index] = max(Xc(2,:));
    Vc_cors = Xc(3,index);
    
    %% TOUCH DOWN: transfer the Cartesian state to planar state
    % Xps: The start state of planar space(stance phase)
    % pos_touch: The landing point: (xc,zc)
    [Xps, pos_touch, Tps] = Cart2Planar(Xce,Tce,model);
    
    %% PREPARE TIME SPAN FOR NEXT STANCE STAGE
    % record the end time and state
    % t_start: stance phase start time
    % t_span: stance phase span time
    t_start = Tps;
    t_span = [t_start t_end];
    
    %% SIMULATION of STANCE phase
    % Tp: stance time array           Xp: stance state matrix
    % Tpe: stance phase end time      Xpe: stance phase end state vector
    [Tp,Xp,Tpe,Xpe] = run_Stance_simulation(t_span,model,Xps);
%     En_c = m*g*Xp(1,:).*cos(Xp(2,:))+1/2*m*(Xp(3,:).^2+(Xp(4,:).*Xp(1,:)).^2)+1/2*(model.k)*(Xp(1,:)-model.L0).^2;
    
    %% TAKE OFF: transfer the planar coordinate to Cartesian coordinate
    % Xcs: The start state vector of Cartesian Space(flight phase)
    % Tcs: The start time of flight phase
    [Xc0,Tc0] = Planar2Cart(Xpe,Tpe,pos_touch,model);
     
    %% PREPARE TIME FOR NEXT FLIGHT PHASE
    t_start = Tc0;
    t_span = [t_start t_end];
    %% update bounce_count
    bounce_count = i;
    %% update THETA
    if abs(Xc0(3)-Vs)>0.4
        error = sign((Xc0(3)-Vs))*0.4;
    else
        error = Xc0(3)-Vs;
    end
    THETA = asin((Xc0(3)*(Tp(end)-Tp(1))/2)/model.L0)+0.12*(error); 
    
    disp('Bounce Count:');disp(bounce_count);
    disp('Zc_max:');disp(Hc_max);
    Data(i) = {{Tc Xc pos_touch Tp Xp angle_array(i)}};
end
[tt,xxc,zzc,angle_total,l_total,En,dzzc,dxxc]=Data_arrange(Data,model);
Data_draw(tt,xxc,zzc,angle_total,l_total,En,dzzc,dxxc);
% animation(tt,xxc,zzc,angle_total,l_total,dxxc);

