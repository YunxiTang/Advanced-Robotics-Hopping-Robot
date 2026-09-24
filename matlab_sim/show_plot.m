function [] = show_plot(tc,xc,tp,xp,pos_touch,flight_theta)
%SHOW_PLOT Summary of this function goes here
%   Detailed explanation goes here
% global THETA

figure(1);
plot(xc(1,:),xc(2,:),'linewidth',2);
hold on;

plot(pos_touch(1) + xp(1,:).*sin(-xp(2,:)),pos_touch(2) + xp(1,:).*cos(xp(2,:)),'linewidth',2);
hold on;

scatter(pos_touch(1),pos_touch(2),'filled','c','b');
title('Trajectory of Mass');
grid on;
% axis 'auto y';
% axis 'auto x';
axis equal;
% 
% figure(2)
% plot(xp(1,:).*sin(-xp(2,:)),xp(1,:).*cos(xp(2,:)),'linewidth',2);
% hold on;
% grid on;
% title('Trajectory of Mass (only Stance Phase)');
% % axis 'auto y';
% % axis 'auto x';
% axis equal;

% figure(3);
% subplot(2,1,1);
% plot(tc,0.2*tc./tc,tp,xp(1,:),'linewidth',2);
% hold on;
% title('Length of Spring');
% grid on;

% subplot(2,1,2);
% plot(tc,flight_theta*tc./tc,tp,xp(2,:),'linewidth',2);
% hold on;
% title('\Theta');
% grid on;

figure(4);
plot(tc,xc(3,:),'linewidth',2);
hold on;
plot(tp,0*tp./tp,'linewidth',2);
hold on;
title('dx_c');


end

