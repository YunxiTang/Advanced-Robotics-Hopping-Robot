function [] = draw_Leg(tt,xc,zc,leg,angle,i,dxxc)
%DRAW_LEG To draw the robot leg
%   input: x_mass --> the position of body mass
%          leg --> length of leg
%          angle --> theta
global Hs
global Vs
x_tip = xc + leg * sin(angle);
z_tip = zc - leg * cos(angle);

scatter(xc,zc,'filled','MarkerEdgeColor',[0 0.4470 0.7410],...
              'MarkerFaceColor',[0 0.4470 0.7410],...
              'LineWidth',16);
          
plot([xc x_tip],[zc z_tip],'linewidth',6,'color','black');
hold on;

hold on;
scatter([xc x_tip],[zc z_tip],'filled');
hold on;
text(xc-0.50,1.0,strcat('Simulation Time: ',num2str(tt(i))),'Color','red','FontSize',12);
text(xc-0.50,0.95,strcat('Desired Height: ',num2str(Hs)),'Color','red','FontSize',12);
text(xc+0.1,0.95,strcat('Hopping Height: ',num2str(zc)),'Color','b','FontSize',12);
text(xc+0.1,0.90,strcat('Foward Velocity: ',num2str(dxxc(i))),'Color','b','FontSize',12);
text(xc-0.5,0.90,strcat('Desired Velocity: ',num2str(Vs)),'Color','red','FontSize',12);



grid on;
ylim([0,1.2]);
% axis 'auto x';
axis equal;
end

