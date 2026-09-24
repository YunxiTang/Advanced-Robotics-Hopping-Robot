function [] = Data_draw(tt,xxc,zzc,angle_total,l_total,En,dzzc,dxxc)
%DATA_DRAW draw nessesary datas
%   Detailed explanation goes here

%% 
figure(2);
subplot(2,2,1);
plot(tt,zzc,'linewidth',2);
grid on;
xlabel('Time / [s]');
ylabel('Height / [m]');
title('Height of Body Mass');

subplot(2,2,2);
plot(tt,dxxc,'linewidth',2);
grid on;
xlabel('Time / [s]');
ylabel('Forward Velocity / [m/s]');
title('Forward Velocity');
 

subplot(2,2,3);
plot(tt,angle_total,'linewidth',2);
grid on;
xlabel('Time / [s]');
ylabel('\Theta / [rad]');
title('\Theta');

subplot(2,2,4);
plot(xxc,zzc,'linewidth',2);
grid on;
xlabel('x / [m]');
ylabel('z / [m]');
title('Body Trajectory');

figure(3);
subplot(2,1,1);
plot(zzc,dzzc,'linewidth',2);
grid on;
xlabel('z / [m/s]');
ylabel('z^{‘} / [m]');
title('Limit Cycle');

subplot(2,1,2);
plot(tt,En,'linewidth',2);
grid on;
xlabel('Time / [s]');
ylabel('Total Energy / [J]');
title('Total Energy');
end

