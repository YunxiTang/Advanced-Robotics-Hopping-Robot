function [tt,xxc,zzc,theta,l_total,En,dzzc,dxxc] = Data_arrange(Data,model)
%HOPPINGMOVIE Data rearrangement
%   Input: Data is a cell :{Tc Xc Postouch Tp Xp}
n = length(Data);
m = model.m;
g = model.g;

tt = [];
xxc = [];
zzc = [];
dzzc = [];
dxxc = [];
theta = [];
l_total = [];
En = [];


% for i=1:n
%     PerCycle = Data{i};
%     
%     Tc = PerCycle{1};
%     Xc = PerCycle{2};
%     Pos_touch = PerCycle{3};
%     Tp = PerCycle{4};
%     Xp = PerCycle{5};
%     flight_theta = PerCycle{6};
%     
%     show_plot(Tc,Xc,Tp,Xp,Pos_touch,flight_theta);
%     
%     figure(5);
%     plot(Tc,m*g*Xc(2,:)+0.5*m*(Xc(3,:).^2+Xc(4,:).^2),'linewidth',2.0);
%     hold on;
%     plot(Tp,m*g*Xp(1,:).*cos(Xp(2,:))+0.5*m*(Xp(3,:).^2+(Xp(4,:).*Xp(1,:)).^2)+0.0009,'linewidth',2.0);
%     hold on;
%     title('Total Energy');
% end

for j=1:n
    PerCycle = Data{j};
    
    Tc = PerCycle{1};
    Xc = PerCycle{2};
    
    xc = Xc(1,:);
    yc = Xc(2,:);
    lc = model.L0*xc./xc;
    
    Pos_touch = PerCycle{3};
    Tp = PerCycle{4};
    Xp = PerCycle{5};
    flight_theta = linspace(Xp(2,end),PerCycle{6},length(xc));
    
    lp = Xp(1,:);
    thp = Xp(2,:);
    xp = Pos_touch(1) - lp.*sin(thp);
    yp = Pos_touch(2) + lp.*cos(thp);
    theta_to = [flight_theta thp];
    
    l_total = [l_total lc lp];
    tt = [tt Tc Tp];
    xxc = [xxc xc xp];
    zzc = [zzc yc yp];
    theta = [theta theta_to];
    
    %% dzzc
    dzzc_f = Xc(4,:);
    dzzc_s = Xp(3,:).*cos(Xp(2,:)) - (-Xp(4,:)).*Xp(1,:).*sin(-Xp(2,:));%dl*cos(-theta) + dtheta*L0*sin(-theta)
    
    dzzc = [dzzc dzzc_f dzzc_s];
    
    %% dxxc
    dxxc_f = Xc(3,:);
    dxxc_s = Xp(3,:).*sin(-Xp(2,:)) - (Xp(4,:)).*Xp(1,:).*cos(-Xp(2,:));%dl*sin(-theta) - dtheta*L0*cos(-theta)
    dxxc = [dxxc dxxc_f dxxc_s];
    
    %% Energy
    % flight phase
    En_f = m*g*Xc(2,:) + 0.5*m*(Xc(3,:).^2+Xc(4,:).^2);

    % stance phase
    En_c = m*g*Xp(1,:).*cos(Xp(2,:))+1/2*m*(Xp(3,:).^2+(Xp(4,:).*Xp(1,:)).^2)+1/2*(model.k)*(Xp(1,:)-model.L0).^2;
    En = [En En_f En_c];
end

end

